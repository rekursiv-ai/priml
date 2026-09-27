"""Replay frozen evaluation trajectories through the evaluation runners.

The goldens in ``testdata/`` were recorded from the reference implementation
this module was ported from; this module imports none of it. Every case runs
one real runner end to end at tiny size on CPU -- no released checkpoint
anywhere: generators and verifiers are trained inside
the case, so the chain is trainer -> harvest -> verifier fits -> HPS /
agreement lock / sieve, and the pipeline cases run it all segmented, screened,
and triggered as exp012-exp014 do. Each records every array its runner writes,
its numeric metrics, and the trained weights, compared with ``torch.equal``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import importlib
import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import trainer, trm
from priml.baselines.sudoku.eval import Reproduction
from priml.baselines.sudoku.trainer_test import read_golden, stored
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from numpy.lib.npyio import NpzFile

    from priml.baselines.sudoku.eval import (
        AgreementLockEval,
        Harvest,
        HpsEval,
        HpsSearch,
        Member,
        SieveEval,
        SudokuVerifier,
        VerifierAcceptor,
        VerifierFit,
        View,
    )
    from priml.baselines.sudoku.puzzle_data import PuzzleDataset


_CWD: Final = Path(__file__).resolve().parent

CASES: Final = (
    "hps",
    "agreement_lock",
    "harvest",
    "node_corpus",
    "verifier_fit",
    "sieve",
    "exp012",
    "exp013",
    "exp014",
)
"""One golden per runner: HpsEval, AgreementLockEval, Harvest (corpus and node
corpus), VerifierFit, SieveEval, and the from-scratch pipeline in each of its
three configurations (exp012, exp013, exp014)."""


class Stack(Protocol):
    """The trainer and evaluation modules an implementation supplies."""

    @property
    def trainer(self) -> ModuleType:
        """Return the trainer module."""
        ...

    @property
    def eval(self) -> ModuleType:
        """Return the eval module."""
        ...


def write_dataset(root: Path) -> Path:
    """Write a solvable-shaped fixture: 16 train groups x 3 views, 6 test rows.

    Inputs keep ~35% of the solution as givens and blank the rest, and cell 0 is
    always blank, so harvest unstuck starts and search pins have cells to act
    on.

    Args:
      root: Dataset root; ``train/`` and ``test/`` are written below.

    Returns:
      root: The dataset root.

    """
    rng = np.random.default_rng(0)
    for split, rows, groups in (
        ("train", 48, np.arange(0, 49, 3)),
        ("test", 6, np.arange(7)),
    ):
        labels = rng.integers(2, 11, (rows, 81))
        inputs = np.where(rng.random((rows, 81)) < 0.35, labels, 1)
        inputs[:, 0] = 1
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "all__inputs.npy", inputs.astype(np.uint8))
        np.save(directory / "all__labels.npy", labels.astype(np.uint8))
        np.save(directory / "all__group_indices.npy", groups.astype(np.int32))
        (directory / "dataset.json").write_text('{"vocab_size": 11, "seq_len": 81}')
    return root


def run_case(stack: Stack, case: str, scratch: Path) -> dict[str, Tensor]:
    """Build the fixture, run ``case`` under host-agnostic numerics, record it.

    Args:
      stack: The port under test.
      case: One of :data:`CASES`.
      scratch: Empty directory owned by this call.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    write_dataset(scratch / "data")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        return {
            k: stored(v) for k, v in _RECORDERS[case](_Harness(stack, scratch)).items()
        }


def mismatches(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> list[str]:
    """Every key whose presence, dtype, shape, or bits differ.

    Args:
      expected: Reference record.
      actual: Candidate record.

    Returns:
      report: One line per mismatch, all of them.

    """
    report = [f"missing {k}" for k in sorted(expected.keys() - actual.keys())]
    report += [f"unexpected {k}" for k in sorted(actual.keys() - expected.keys())]
    for key in sorted(expected.keys() & actual.keys()):
        want, got = expected[key], actual[key]
        if want.dtype != got.dtype or want.shape != got.shape:
            report.append(
                f"{key}: {got.dtype}{list(got.shape)} vs {want.dtype}{list(want.shape)}",
            )
        elif not torch.equal(want, got):
            report.append(f"{key}: {(want != got).sum().item()}/{want.numel()} differ")
    return report


def golden_path(case: str) -> Path:
    """Where the golden for ``case`` lives."""
    return _CWD / "testdata" / f"{case}.pt"


def load_golden(case: str) -> dict[str, Tensor]:
    """Load a frozen golden."""
    return read_golden(golden_path(case))


class PrimlStack:
    """The priml port."""

    trainer = trainer
    eval = importlib.import_module("priml.baselines.sudoku.eval")


@pytest.mark.parametrize("case", CASES)
def test_golden_replays_bit_for_bit(case: str, tmp_path: Path) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    report = mismatches(load_golden(case), run_case(PrimlStack(), case, tmp_path))
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


class _Harness:
    """Tiny, size-only configs of one port's runners, rooted at ``scratch``."""

    def __init__(self, stack: Stack, scratch: Path) -> None:
        self.trainer = cast("_TrainerModule", stack.trainer)
        self.eval = cast("_EvalModule", stack.eval)
        self.scratch = scratch
        self.data = scratch / "data"

    def model(self) -> trm.TRM.Config:
        """Return the generator architecture, shrunk by size."""
        cfg = self.eval.TRM.Config()
        cfg.channels_in = 4
        cfg.num_heads = 1
        cfg.num_layers = 1
        cfg.slow_cycles = 1
        cfg.fast_cycles = 1
        cfg.puzzle_emb_len = 2
        cfg.compile = False
        cfg.dtype = None
        return cfg

    def search(self) -> HpsSearch.Config:
        """Return the search policy, shrunk by tree size and ACT depth."""
        cfg = self.eval.HpsSearch.Config()
        cfg.max_act_steps = 3
        cfg.acceptance_checkpoints = (2, 3)
        cfg.dtype_autocast = None
        cfg.search_depth = 2
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 8
        cfg.search_max_rows = 64
        return cfg

    def verifier_model(self) -> SudokuVerifier.Config:
        """Return the verifier architecture, shrunk by size."""
        cfg = self.eval.SudokuVerifier.Config()
        cfg.width = 4
        cfg.depth = 1
        cfg.heads = 1
        return cfg

    def generator(self, name: str, seed: int) -> str:
        """Train a two-step generator; return its logical checkpoint path."""
        cfg = self.trainer.Trainer.Config()
        cfg.experiment_name = name
        cfg.base_dir = self.scratch
        cfg.seed = seed
        cfg.runtime.device = "cpu"
        cfg.model = self.model()
        cfg.dataset.working_dir = self.data
        cfg.dataset.batch_size = 4
        cfg.max_steps = 2
        cfg.max_act_steps = 3
        cfg.num_steps_eval = float("inf")
        cfg.eval_warmup_batches = 0
        cfg.dtype_autocast = None
        cfg.make().run()
        return f"/runs/{name}/checkpoints/step_00000002.pt"

    def harvest(self, checkpoint: str) -> Path:
        """Harvest a tiny corpus from ``checkpoint``; return its directory."""
        cfg = self.eval.Harvest.Config()
        cfg.base_dir = self.scratch
        cfg.harvest_source_checkpoint = checkpoint
        cfg.model = self.model()
        cfg.working_dir = "/data"
        cfg.group_count = 16
        cfg.views_per_group = 1
        cfg.max_act_steps = 2
        cfg.checkpoints = (1, 2)
        cfg.search_depth = 1
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 2
        cfg.random_corruption_strengths = (1, 2)
        cfg.random_starts_per_strength = 1
        cfg.dtype_autocast = None
        cfg.device = "cpu"
        return cfg.make().run()

    def verifier(self, seed: int) -> Path:
        """Fit one tiny committee member on the harvest; return its checkpoint."""
        cfg = self.eval.VerifierFit.Config()
        cfg.experiment_name = f"verifier_s{seed}"
        cfg.seed = seed
        cfg.base_dir = self.scratch
        cfg.model = self.verifier_model()
        cfg.dataset.working_dir = "/data"
        cfg.dataset.harvest_dir = "/runs/harvest/harvest"
        cfg.dataset.node_corpus = "/runs/harvest/harvest/node_corpus.npz"
        cfg.dataset.batch_size = 20
        cfg.dataset.steps_per_epoch = 2
        cfg.dataset.train_group_end = 10
        cfg.dataset.calibration_group_end = 13
        cfg.dataset.holdout_group_end = 16
        cfg.dataset.dev_puzzles = 6
        cfg.device = "cpu"
        cfg.dtype = None
        cfg.max_steps = 4
        return cfg.make().run()

    def dumps(self, name: str) -> dict[str, Tensor]:
        """Every array in every npz under a run.

        A packed search ``rows`` array is split at its grid columns: the
        leading score columns stay float32 while the token grids narrow to
        bytes, which is most of a golden's size.
        """
        out: dict[str, Tensor] = {}
        root = self.scratch / "runs" / name
        for path in sorted(root.rglob("*.npz")):
            with cast("NpzFile", np.load(path)) as archive:
                for key in archive.files:
                    array = torch.from_numpy(np.array(archive[key]))
                    prefix = f"{path.relative_to(root)}/{key}"
                    if key == "rows":
                        grids = 2 * 81
                        out[f"{prefix}/scores"] = array[:, :-grids]
                        out[f"{prefix}/grids"] = array[:, -grids:]
                    else:
                        out[prefix] = array
        return out


class _TrainerModule(Protocol):
    Trainer: type[trainer.Trainer]


class _EvalModule(Protocol):
    TRM: type[trm.TRM]
    HpsSearch: type[HpsSearch]
    SudokuVerifier: type[SudokuVerifier]
    Harvest: type[Harvest]
    VerifierFit: type[VerifierFit]
    VerifierAcceptor: type[VerifierAcceptor]
    HpsEval: type[HpsEval]
    AgreementLockEval: type[AgreementLockEval]
    SieveEval: type[SieveEval]
    NINE_VIEWS: tuple[View, ...]
    Member: type[Member]
    Reproduction: type[Reproduction]

    def nine_view_members(self, checkpoint: str) -> tuple[Member, ...]: ...


class _ProtocolConfig(Protocol):
    model: trm.TRM.Config
    dataset: PuzzleDataset.Config
    search: HpsSearch.Config
    evaluation_count: int


_PipelineConfig = Reproduction.Config


# A trained weight is the product of every step before it, so its bits already pin
# the whole trajectory; a handful of elements per tensor catches any divergence
# without storing the model.
def _weights(prefix: str, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return the first 4 elements of every trained tensor."""
    return {f"{prefix}/{k}": v.flatten()[:4].clone() for k, v in state.items()}


def _metrics(values: Mapping[str, object]) -> dict[str, Tensor]:
    """Numeric metrics as float64 tensors; timings and labels are dropped."""
    return {
        f"metrics/{key}": torch.tensor(float(value), dtype=torch.float64)
        for key, value in values.items()
        if isinstance(value, int | float) and "seconds" not in key
    }


def _record_hps(h: _Harness) -> dict[str, Tensor]:
    checkpoint = h.generator("gen_a", seed=0)
    cfg = h.eval.HpsEval.Config()
    cfg.experiment_name = "hps"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.checkpoint_path = checkpoint
    cfg.model = h.model()
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 4
    cfg.search = h.search()
    cfg.evaluation_count = 6
    # A two-step generator's halt logits sit near its -5 init, far below the
    # recipe's 7.875. A probe pass that accepts every root reads them; the
    # thresholds are then placed among them -- the root one at the median, the
    # node one a quartile lower -- so the golden records root accepts, search
    # accepts, and search failures at any model size.
    probe = cfg.copy_tree()
    probe.experiment_name = "hps_probe"
    probe.search.acceptance_threshold = -1e9
    probe.make().run()
    roots = h.dumps("hps_probe")["dumps/hps_eval_predictions.npz/rows/scores"][:, 0]
    cfg.search.root_acceptance_threshold = float(roots.double().quantile(0.5))
    cfg.search.acceptance_threshold = float(roots.double().quantile(0.25))
    return {**_metrics(cfg.make().run()), **h.dumps("hps")}


def _record_lock(h: _Harness) -> dict[str, Tensor]:
    gen_a = h.generator("gen_a", seed=0)
    gen_b = h.generator("gen_b", seed=1)
    cfg = h.eval.AgreementLockEval.Config()
    cfg.experiment_name = "lock"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.model = h.model()
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 4
    cfg.search = h.search()
    cfg.evaluation_count = 6
    views = h.eval.NINE_VIEWS
    cfg.members = (
        h.eval.Member(gen_a, views[0]),
        h.eval.Member(gen_b, views[0], name="gen_b"),
        h.eval.Member(gen_a, views[1]),
    )
    return {**_metrics(cfg.make().run()), **h.dumps("lock")}


def _record_harvest(h: _Harness) -> dict[str, Tensor]:
    h.harvest(h.generator("gen_a", seed=0))
    return h.dumps("harvest")


def _record_verifier(h: _Harness) -> dict[str, Tensor]:
    h.harvest(h.generator("gen_a", seed=0))
    state = cast(
        "dict[str, dict[str, dict[str, Tensor]]]",
        torch.load(h.verifier(seed=0), weights_only=True),
    )
    return _weights("model", state["step"]["model"])


def _record_sieve(h: _Harness) -> dict[str, Tensor]:
    checkpoint = h.generator("gen_a", seed=0)
    h.harvest(checkpoint)
    members = tuple(h.verifier(seed) for seed in range(3))
    cfg = h.eval.SieveEval.Config()
    cfg.experiment_name = "sieve"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.checkpoint_path = checkpoint
    cfg.model = h.model()
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 4
    cfg.search = h.search()
    cfg.views = h.eval.NINE_VIEWS[:2]
    cfg.tail_search = (2, 2, 1, 8)
    cfg.evaluation_count = 6
    cfg.verifier_checkpoints = members
    acceptor = h.eval.VerifierAcceptor.Config()
    acceptor.model = h.verifier_model()
    # Four-step verifiers score every grid near zero, so the recipe's 0 locks
    # nothing. The threshold is the median committee score on the test
    # puzzles' givens-only boards, so each round both locks and defers.
    acceptor.threshold = _committee_median(h, acceptor, members)
    cfg.acceptor = acceptor
    metrics = cfg.make().run()
    rounds = cast("list[dict[str, object]]", metrics["eval/rounds"])
    per_round = {
        f"rounds/{i}/{key}": torch.tensor(cast(int, stat[key]), dtype=torch.int64)
        for i, stat in enumerate(rounds)
        for key in ("entering", "locked", "deferred")
    }
    return {**_metrics(metrics), **per_round, **h.dumps("sieve")}


def _committee_median(
    h: _Harness,
    acceptor: VerifierAcceptor.Config,
    members: tuple[Path, ...],
) -> float:
    """Median unanimity score of the committee over the test puzzles' labels."""
    probe = acceptor.copy_tree()
    probe.checkpoint_paths = members
    probe.device = "cpu"
    labels = torch.from_numpy(np.load(h.data / "test" / "all__labels.npy")).long()
    media = torch.from_numpy(np.load(h.data / "test" / "all__inputs.npy")).long()
    scores = probe.make().scores(labels, media)
    return float(scores.double().quantile(0.5))


def _record_node_corpus(h: _Harness) -> dict[str, Tensor]:
    checkpoint = h.generator("gen_a", seed=0)
    cfg = h.eval.Harvest.Config()
    cfg.base_dir = h.scratch
    cfg.harvest_source_checkpoint = checkpoint
    cfg.model = h.model()
    cfg.working_dir = "/data"
    cfg.group_count = 16
    cfg.max_act_steps = 2
    cfg.checkpoints = (1, 2)
    cfg.search_max_rows = 64
    cfg.dtype_autocast = None
    cfg.device = "cpu"
    # Every root enters the frontier: a tiny model's q never clears 0.
    cfg.make().regenerate_node_corpus(
        dev_puzzles=6,
        batch_size=4,
        search_depth=2,
        search_candidates=2,
        search_cell_attempts=1,
        search_budget=6,
    )
    return h.dumps("harvest")


def _pipeline(h: _Harness, name: str) -> object:
    """Return a tiny from-scratch pipeline: 4 steps in 2-step segments."""
    cfg = h.eval.Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = name
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    generator = h.trainer.Trainer.Config()
    generator.model = h.model()
    generator.dataset.working_dir = h.data
    generator.dataset.batch_size = 4
    generator.dataset.augment = True  # The recipe's; draws from the segment seed.
    generator.max_steps = 4
    generator.max_act_steps = 3
    generator.dtype_autocast = None
    generator.eval_warmup_batches = 0
    cfg.generator = generator
    cfg.generator_names = (f"{name}_generator",)
    cfg.eval_every_steps = 2
    cfg.dev_screen_count = 4
    # A two-step model never reaches the recipe's 0.98 gate; opening it makes
    # every boundary run its dev screen.
    cfg.screen_gate_det_accuracy = 0.0
    return cfg


def _full_eval_defaults(h: _Harness, full_eval: object) -> None:
    """Shrink a pipeline's full-set protocol by size."""
    cast("_ProtocolConfig", full_eval).model = h.model()
    dataset = cast("_ProtocolConfig", full_eval).dataset
    dataset.working_dir = h.data
    dataset.batch_size = 4
    cast("_ProtocolConfig", full_eval).search = h.search()
    cast("_ProtocolConfig", full_eval).evaluation_count = 4


def _run_pipeline(
    h: _Harness,
    cfg: object,
    names: tuple[str, ...],
) -> dict[str, Tensor]:
    """Run a pipeline; record its dumps, metrics, and final generator weights."""
    config = cast("_PipelineConfig", cfg)
    config.make().run()
    out = h.dumps(config.experiment_name)
    metrics = cast(
        "dict[str, object]",
        json.loads(
            (h.scratch / "runs" / config.experiment_name / "metrics.json").read_text(),
        ),
    )
    out |= _metrics(metrics)
    for generator in names:
        state = cast(
            "dict[str, dict[str, dict[str, Tensor]]]",
            torch.load(
                h.scratch / "runs" / generator / "checkpoints" / "step_00000004.pt",
                weights_only=True,
            ),
        )
        out |= _weights(generator, state["step"]["model"])
    return out


def _record_repro_lock(h: _Harness) -> dict[str, Tensor]:
    cfg = cast("_PipelineConfig", _pipeline(h, "repro_lock"))
    lock = h.eval.AgreementLockEval.Config()
    _full_eval_defaults(h, lock)
    lock.members = h.eval.nine_view_members("template.pt")[:2]
    cfg.full_eval = lock
    return _run_pipeline(h, cfg, cfg.generator_names)


def _record_repro_sieve(h: _Harness) -> dict[str, Tensor]:
    source = h.generator("source", seed=0)
    cfg = cast("_PipelineConfig", _pipeline(h, "repro_sieve"))
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = source
    cfg.harvest = h.eval.Harvest.Config()
    cfg.harvest.model = h.model()
    cfg.harvest.working_dir = "/data"
    cfg.harvest.group_count = 16
    cfg.harvest.views_per_group = 1
    cfg.harvest.max_act_steps = 2
    cfg.harvest.checkpoints = (1, 2)
    cfg.harvest.search_depth = 1
    cfg.harvest.search_candidates = 2
    cfg.harvest.search_cell_attempts = 1
    cfg.harvest.search_budget = 2
    cfg.harvest.random_corruption_strengths = (1, 2)
    cfg.harvest.random_starts_per_strength = 1
    cfg.harvest.dtype_autocast = None
    cfg.verifier_names = tuple(f"repro_verifier_s{s}" for s in range(3))
    cfg.verifier.model = h.verifier_model()
    cfg.verifier.dataset.working_dir = "/data"
    cfg.verifier.dataset.batch_size = 20
    cfg.verifier.dataset.steps_per_epoch = 2
    cfg.verifier.dataset.train_group_end = 10
    cfg.verifier.dataset.calibration_group_end = 13
    cfg.verifier.dataset.holdout_group_end = 16
    cfg.verifier.dataset.dev_puzzles = 6
    cfg.verifier.dtype = None
    cfg.verifier.max_steps = 4
    sieve = h.eval.SieveEval.Config()
    _full_eval_defaults(h, sieve)
    sieve.views = h.eval.NINE_VIEWS[:2]
    sieve.tail_search = (2, 2, 1, 8)
    acceptor = cast("VerifierAcceptor.Config", sieve.acceptor)
    acceptor.model = h.verifier_model()
    # The recipe's 0 locks nothing on a four-step committee (see
    # _record_sieve), which _record_sieve already covers; locking everything
    # instead drives the pipeline's committee screens and the sieve's lock.
    acceptor.threshold = -1e9
    cfg.full_eval = sieve
    out = _run_pipeline(h, cfg, cfg.generator_names)
    for name in cfg.verifier_names:
        state = cast(
            "dict[str, dict[str, dict[str, Tensor]]]",
            torch.load(
                h.scratch / "runs" / name / "checkpoints" / "step_00000004.pt",
                weights_only=True,
            ),
        )
        out |= _weights(name, state["step"]["model"])
    return out


def _record_repro_seeds(h: _Harness) -> dict[str, Tensor]:
    cfg = cast("_PipelineConfig", _pipeline(h, "repro_seeds"))
    cfg.screen = "single_view"
    cfg.trigger = "final_only"
    cfg.generator_seeds = (44, 45, 46)
    cfg.generator_names = tuple(f"repro_s{seed}" for seed in cfg.generator_seeds)
    lock = h.eval.AgreementLockEval.Config()
    _full_eval_defaults(h, lock)
    cfg.full_eval = lock
    return _run_pipeline(h, cfg, cfg.generator_names)


_RECORDERS: Final[dict[str, Callable[[_Harness], dict[str, Tensor]]]] = {
    "hps": _record_hps,
    "agreement_lock": _record_lock,
    "harvest": _record_harvest,
    "node_corpus": _record_node_corpus,
    "verifier_fit": _record_verifier,
    "sieve": _record_sieve,
    "exp012": _record_repro_lock,
    "exp013": _record_repro_sieve,
    "exp014": _record_repro_seeds,
}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
