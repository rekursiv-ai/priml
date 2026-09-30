"""Replay frozen evaluation trajectories through the evaluation runners.

The goldens in ``testdata/`` were recorded from the reference implementation
this module was ported from; this module imports none of it. Every case runs
one real runner end to end at tiny size on CPU -- no released checkpoint
anywhere: generators and verifiers are trained inside the case, so the chain is
trainer -> harvest -> verifier fits -> HPS / agreement lock / sieve, and the
pipeline cases run it all segmented, screened, and triggered as exp012-exp014
do. Each records every array its runner writes,
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

from priml.baselines.sudoku import (
    trainer,
    trm,
)
from priml.baselines.sudoku.eval import (
    Reproduction,
)
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors, stored

import priml.baselines.sudoku.eval


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
        VerifierData,
        VerifierFit,
        View,
    )
    from priml.baselines.sudoku.puzzle_data import PuzzleDataset
    from priml.model.transformer.block import TransformerBlock


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

    @property
    def model(self) -> ModuleType:
        """Return the module defining the TRM and its ``recipe_block``."""
        ...


def write_dataset(root: Path) -> Path:
    """Write a solvable-shaped fixture: 3 train groups x 3 views, 2 test rows.

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
        ("train", 9, np.arange(0, 10, 3)),
        ("test", 2, np.arange(3)),
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


def golden_path(case: str) -> Path:
    """Where the golden for ``case`` lives."""
    return _CWD / "testdata" / f"{case}.pt"


def load_golden(case: str) -> dict[str, Tensor]:
    """Load a frozen golden."""
    return read_tensors(golden_path(case))


class PrimlStack:
    """The priml port."""

    trainer = trainer
    eval = importlib.import_module("priml.baselines.sudoku.eval")
    model = trm


@pytest.mark.parametrize("case", CASES)
def test_golden_replays_bit_for_bit(case: str, tmp_path: Path) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    report = mismatches(load_golden(case), run_case(PrimlStack(), case, tmp_path))
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


class _Harness:
    """Configurations of one port's runners, rooted at ``scratch``."""

    def __init__(self, stack: Stack, scratch: Path) -> None:
        self.trainer = cast("_TrainerModule", stack.trainer)
        self.eval = cast("_EvalModule", stack.eval)
        self.recipe_block = cast("_ModelModule", stack.model).recipe_block
        self.scratch = scratch
        self.data = scratch / "data"

    def model(self) -> trm.TRM.Config:
        """Return the generator architecture."""
        cfg = self.eval.TRM.Config()
        cfg.vocab_size = 11
        cfg.puzzle_grid_shape = (81,)
        cfg.pos2d_grid_shape = (9, 9)
        cfg.pos2d_box_shape = (3, 3)
        cfg.channels_in = 4
        cfg.num_heads = 2
        cfg.num_layers = 1
        cfg.slow_cycles = 1
        cfg.fast_cycles = 1
        cfg.puzzle_emb_len = 2
        cfg.compile = False
        cfg.dtype = None
        cfg.block = self.recipe_block()
        assert isinstance(cfg.block.ffn, SwiGLU.Config)
        cfg.block.ffn.expansion = 1
        cfg.block.ffn.round_to = 1
        return cfg

    def search(self) -> HpsSearch.Config:
        """Return the search policy."""
        cfg = self.eval.HpsSearch.Config()
        cfg.max_act_steps = 2
        cfg.acceptance_checkpoints = (1,)
        cfg.dtype_autocast = None
        cfg.search_depth = 1
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 2
        cfg.search_max_rows = 16
        return cfg

    def verifier_model(self) -> SudokuVerifier.Config:
        """Return the verifier architecture."""
        cfg = self.eval.SudokuVerifier.Config()
        cfg.width = 4
        cfg.depth = 1
        cfg.heads = 2
        return cfg

    def generator(self, name: str, seed: int) -> str:
        """Train a three-step generator; return its logical checkpoint path."""
        cfg = self.trainer.Trainer.Config()
        cfg.experiment_name = name
        cfg.base_dir = self.scratch
        cfg.seed = seed
        cfg.runtime.device = "cpu"
        cfg.model = self.model()
        cfg.dataset.working_dir = self.data
        cfg.dataset.batch_size = 2
        cfg.max_steps = 3
        cfg.max_act_steps = 2
        cfg.num_steps_eval = float("inf")
        cfg.eval_warmup_batches = 0
        cfg.dtype_autocast = None
        cfg.make().run()
        return f"/runs/{name}/checkpoints/step_00000003.pt"

    def harvest(self, checkpoint: str) -> Path:
        """Harvest a tiny corpus from ``checkpoint``; return its directory."""
        cfg = self.eval.Harvest.Config()
        cfg.base_dir = self.scratch
        cfg.harvest_source_checkpoint = checkpoint
        cfg.model = self.model()
        cfg.working_dir = "/data"
        cfg.group_count = 3
        cfg.views_per_group = 1
        cfg.max_act_steps = 2
        cfg.checkpoints = (1,)
        cfg.search_depth = 1
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 2
        cfg.random_corruption_strengths = (1,)
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
        cfg.dataset.batch_size = 2
        cfg.dataset.steps_per_epoch = 1
        cfg.dataset.train_group_end = 1
        cfg.dataset.calibration_group_end = 2
        cfg.dataset.holdout_group_end = 3
        cfg.dataset.dev_puzzles = 2
        cfg.device = "cpu"
        cfg.dtype = None
        cfg.max_steps = 3
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


class _ModelModule(Protocol):
    def recipe_block(self) -> TransformerBlock.Config: ...


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


def _weights(prefix: str, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return every trained tensor under ``prefix``, whole."""
    return {f"{prefix}/{k}": v.detach().clone() for k, v in state.items()}


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
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.evaluation_count = 2
    # A three-step generator's halt logits sit near its -5 init, far below the
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
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.evaluation_count = 2
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
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.views = h.eval.NINE_VIEWS[:1]
    cfg.tail_search = (1, 1, 1, 1)
    cfg.evaluation_count = 2
    cfg.verifier_checkpoints = members
    acceptor = h.eval.VerifierAcceptor.Config()
    acceptor.model = h.verifier_model()
    # Three-step verifiers score every grid near zero, so the recipe's 0 locks
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
    cfg.group_count = 3
    cfg.max_act_steps = 2
    cfg.checkpoints = (1,)
    cfg.search_max_rows = 16
    cfg.dtype_autocast = None
    cfg.device = "cpu"
    # Every root enters the frontier: a tiny model's q never clears 0.
    cfg.make().regenerate_node_corpus(
        dev_puzzles=2,
        batch_size=2,
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=1,
        search_budget=2,
    )
    return h.dumps("harvest")


def _pipeline(h: _Harness, name: str) -> _PipelineConfig:
    """Return a 4-step pipeline with 2-step evaluation segments."""
    cfg = h.eval.Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = name
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    generator = h.trainer.Trainer.Config()
    generator.model = h.model()
    generator.dataset.working_dir = h.data
    generator.dataset.batch_size = 2
    generator.dataset.augment = True  # The recipe's; draws from the segment seed.
    generator.max_steps = 4
    generator.max_act_steps = 3
    generator.dtype_autocast = None
    generator.eval_warmup_batches = 0
    cfg.generator = generator
    cfg.generator_names = (f"{name}_generator",)
    cfg.eval_every_steps = 2
    cfg.dev_screen_count = 2
    # A two-step model never reaches the recipe's 0.98 gate; opening it makes
    # every boundary run its dev screen.
    cfg.screen_gate_det_accuracy = 0.0
    return cfg


def _full_eval_defaults(h: _Harness, full_eval: _ProtocolConfig) -> None:
    """Configure a pipeline's full-set protocol for the shrunk fixture."""
    full_eval.model = h.model()
    full_eval.dataset.working_dir = h.data
    full_eval.dataset.batch_size = 2
    full_eval.search = h.search()
    full_eval.evaluation_count = 2


def _run_pipeline(
    h: _Harness,
    config: _PipelineConfig,
    names: tuple[str, ...],
) -> dict[str, Tensor]:
    """Run a pipeline; record its dumps, metrics, and final generator weights."""
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
    cfg = _pipeline(h, "repro_lock")
    lock = h.eval.AgreementLockEval.Config()
    _full_eval_defaults(h, lock)
    lock.members = h.eval.nine_view_members("template.pt")[:2]
    cfg.full_eval = lock
    return _run_pipeline(h, cfg, cfg.generator_names)


def _record_repro_sieve(h: _Harness) -> dict[str, Tensor]:
    source = h.generator("source", seed=0)
    cfg = _pipeline(h, "repro_sieve")
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = source
    cfg.harvest = h.eval.Harvest.Config()
    cfg.harvest.model = h.model()
    cfg.harvest.working_dir = "/data"
    cfg.harvest.group_count = 3
    cfg.harvest.views_per_group = 1
    cfg.harvest.max_act_steps = 2
    cfg.harvest.checkpoints = (1,)
    cfg.harvest.search_depth = 1
    cfg.harvest.search_candidates = 2
    cfg.harvest.search_cell_attempts = 1
    cfg.harvest.search_budget = 2
    cfg.harvest.random_corruption_strengths = (1,)
    cfg.harvest.random_starts_per_strength = 1
    cfg.harvest.dtype_autocast = None
    cfg.verifier_names = tuple(f"repro_verifier_s{s}" for s in range(3))
    cfg.verifier.model = h.verifier_model()
    cfg.verifier.dataset.working_dir = "/data"
    cfg.verifier.dataset.batch_size = 2
    cfg.verifier.dataset.steps_per_epoch = 1
    cfg.verifier.dataset.train_group_end = 1
    cfg.verifier.dataset.calibration_group_end = 2
    cfg.verifier.dataset.holdout_group_end = 3
    cfg.verifier.dataset.dev_puzzles = 2
    cfg.verifier.dtype = None
    cfg.verifier.max_steps = 3
    sieve = h.eval.SieveEval.Config()
    _full_eval_defaults(h, sieve)
    sieve.views = h.eval.NINE_VIEWS[:1]
    sieve.tail_search = (1, 1, 1, 1)
    acceptor = cast("VerifierAcceptor.Config", sieve.acceptor)
    acceptor.model = h.verifier_model()
    # The recipe's 0 locks nothing on a three-step committee (see
    # _record_sieve), which _record_sieve already covers; locking everything
    # instead drives the pipeline's committee screens and the sieve's lock.
    acceptor.threshold = -1e9
    cfg.full_eval = sieve
    out = _run_pipeline(h, cfg, cfg.generator_names)
    for name in cfg.verifier_names:
        state = cast(
            "dict[str, dict[str, dict[str, Tensor]]]",
            torch.load(
                h.scratch / "runs" / name / "checkpoints" / "step_00000003.pt",
                weights_only=True,
            ),
        )
        out |= _weights(name, state["step"]["model"])
    return out


def _record_repro_seeds(h: _Harness) -> dict[str, Tensor]:
    cfg = _pipeline(h, "repro_seeds")
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


def _tiny_trm() -> trm.TRM:
    """Build a real CPU TRM at the smallest size the eval runners accept."""
    model = _tiny_config().make()
    assert isinstance(model, trm.TRM)
    return model.eval()


def _tiny_config() -> trm.TRM.Config:
    """Return the smallest TRM architecture the eval runners accept."""
    cfg = trm.TRM.Config()
    cfg.vocab_size = 11
    cfg.puzzle_grid_shape = (81,)
    cfg.pos2d_grid_shape = (9, 9)
    cfg.pos2d_box_shape = (3, 3)
    cfg.channels_in = 4
    cfg.num_heads = 2
    cfg.num_layers = 1
    cfg.slow_cycles = 1
    cfg.fast_cycles = 1
    cfg.num_puzzle_identifiers = 0
    cfg.compile = False
    cfg.dtype = None
    cfg.block = trm.recipe_block()
    assert isinstance(cfg.block.ffn, SwiGLU.Config)
    cfg.block.ffn.expansion = 1
    cfg.block.ffn.round_to = 1
    return cfg


def test_eval_pure_helpers_and_validation(tmp_path: Path) -> None:
    """Exercise label-free helpers, dump schemas, and validation branches."""
    ev = priml.baselines.sudoku.eval
    grid = torch.arange(2 * 81).reshape(2, 81) % 9 + 2
    for view in ev.NINE_VIEWS:
        assert torch.equal(view.invert(view.apply(grid)), grid)
    with pytest.raises(ValueError, match="permutation"):
        ev.validate_grid_permutation("rows", (0, 0, 1, 2, 3, 4, 5, 6, 7))
    with pytest.raises(ValueError, match="band"):
        ev.validate_grid_permutation("rows", (0, 1, 3, 2, 4, 5, 6, 7, 8))
    with pytest.raises(ValueError, match="at least two"):
        ev.AgreementLockEval.Config(
            experiment_name="x",
            members=(ev.Member("x", ev.NINE_VIEWS[0]),),
        ).make()
    assert ev.fixed_hps_node_count(depth=2, candidates=3, cell_attempts=2) == 24
    assert (
        ev.rows_per_view(
            search_depth=2,
            search_candidates=3,
            search_cell_attempts=2,
            random_corruption_strengths=(2, 4),
            random_starts_per_strength=2,
        )
        == 29
    )
    with pytest.raises(ValueError, match="positive"):
        ev.fixed_hps_node_count(depth=0, candidates=2, cell_attempts=2)
    with pytest.raises(ValueError, match="nonempty"):
        ev.rows_per_view(
            search_depth=2,
            search_candidates=2,
            search_cell_attempts=2,
            random_corruption_strengths=(),
            random_starts_per_strength=2,
        )
    selected = ev.select_harvest_views(
        torch.tensor([0, 2, 4, 6]),
        group_count=2,
        views_per_group=2,
        seed=3,
    )
    assert selected.flat_view_id.shape == (4,)
    originals = grid[:2].clone()
    originals[:, :4] = 1
    starts = ev.make_random_unstuck_starts(
        originals,
        grid[:2],
        base_group_ids=torch.tensor([0, 2]),
        view_ids=torch.tensor([1, 0]),
        seed=4,
        corruption_strengths=(2,),
        starts_per_strength=2,
    )
    assert starts.current_state.shape == (4, 81)
    assert torch.equal(
        starts.current_state[:, 4:],
        grid[:2, 4:].repeat_interleave(2, dim=0),
    )
    with pytest.raises(ValueError, match="no blank"):
        ev.make_random_unstuck_starts(
            torch.full((2, 81), 2),
            grid[:2],
            base_group_ids=torch.tensor([0, 1]),
            view_ids=torch.tensor([0, 0]),
            seed=0,
            corruption_strengths=(2,),
            starts_per_strength=2,
        )
    assert ev.modal_grid_predictions(
        torch.stack((grid[:2], grid[:2].flip(0), grid[:2])),
    )[1].tolist() == [0, 0]
    assert ev.escalated_search_config().search_candidates == 7
    assert (
        ev._fill_template(
            "/runs/{experiment_name}/x-{index}",
            base_dir=tmp_path,
            experiment_name="e",
            index=2,
        ).name
        == "x-2"
    )


def test_eval_search_helpers_and_dump_roundtrip(tmp_path: Path) -> None:
    """Cover search acceptance, persistence, packing, and npz round trips."""
    ev = priml.baselines.sudoku.eval
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    logits = torch.zeros(2, 81, 11)
    logits[..., 2] = 4
    cells, _ = ev.select_pin_candidates(logits, media, n_cells=2, n_digits=3)
    assert cells.shape == (2, 2)
    groups = ev.sudoku_groups()
    preds = torch.full((2, 81), 2, dtype=torch.long)
    assert not bool(ev.accepted_grids(preds, media, groups).any())
    assert torch.equal(ev._violated_group_counts(preds, groups), torch.full((2,), 27.0))
    result = ev.SearchResult(
        accepted=torch.tensor([True, False]),
        root_predictions=preds,
        final_predictions=preds,
        scores=torch.tensor([2.0, -1.0]),
        root_scores=torch.tensor([2.0, -1.0]),
        nodes=torch.tensor([0, 3]),
        depth=torch.tensor([0, -1]),
        scored=torch.tensor([True, True]),
        visited_predictions=[],
        visited_puzzles=[],
    )
    rows = ev.pack_search_rows(result)
    assert rows.shape == (2, ev.learned_hps_output_width(81))
    path = tmp_path / "member.npz"
    ev.write_member_dump(path, rows, media=media, label=preds)
    loaded = ev.read_member_dump(path)
    assert torch.equal(loaded.final_predictions, preds.to(torch.uint8))
    assert ev.summarize_search(rows, preds)["accepted"] == 1.0
    with pytest.raises(ValueError, match="packed width"):
        ev.write_member_dump(tmp_path / "bad.npz", rows[:, :-1], media=media)
    with pytest.raises(ValueError, match="at least one"):
        ev.learned_checkpoint_rollout_rows(
            _tiny_trm(),
            {},
            2,
            media,
            torch.arange(2),
            checkpoints=(),
        )
    rollout = ev.LearnedCheckpointRollout(
        logits=logits,
        q_scores=torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]),
        predictions=preds.unsqueeze(1).expand(-1, 3, -1),
    )
    assert torch.equal(
        ev.learned_persistence_scores(rollout, require_prediction_stability=True),
        torch.tensor([2.0, 2.0]),
    )
    unstable = ev.LearnedCheckpointRollout(
        logits=rollout.logits,
        q_scores=rollout.q_scores,
        predictions=rollout.predictions.clone(),
    )
    unstable.predictions[0, 0, 0] = 3
    assert torch.isneginf(
        ev.learned_persistence_scores(unstable, require_prediction_stability=True)[0],
    )


def test_eval_rollout_and_pin_search_engines() -> None:
    """Exercise segmented ACT rollouts and candidate-parallel search engines."""
    ev = priml.baselines.sudoku.eval
    typed_model = _tiny_trm()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    kwargs: dict[str, Tensor] = {}
    logits, scores = ev.segmented_rollout_rows(
        typed_model,
        kwargs,
        10,
        boards,
        torch.tensor([0, 1]),
        continue_threshold=1.0,
        early_exit_at_q8=True,
    )
    assert logits.shape == (2, 81, 11)
    assert scores.shape == (2,)
    checkpoint = ev.learned_checkpoint_rollout_rows(
        typed_model,
        kwargs,
        3,
        boards,
        torch.tensor([0, 1]),
        checkpoints=(2, 3),
    )
    assert checkpoint.q_scores.shape == (2, 2)
    assert ev.learned_persistence_scores(
        checkpoint,
        require_prediction_stability=False,
    ).shape == (2,)
    root_logits = torch.zeros(2, 81, 11)
    root_logits[..., 2] = 1
    active = torch.tensor([True, False])

    def rollout(candidate: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        out = torch.zeros(candidate.shape[0], 81, 11)
        out[..., 2] = 1
        return out, torch.full((candidate.shape[0],), 3.0)

    learned = ev.run_learned_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        acceptance_threshold=4.0,
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=8,
        max_rows=3,
    )
    assert learned.accepted.shape == (2,)

    def accepts(predictions: Tensor, media: Tensor) -> Tensor:
        del media
        return torch.zeros(predictions.shape[0], dtype=torch.bool)

    found, grids, nodes, depth = ev.run_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        groups=ev.sudoku_groups(),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=8,
        max_rows=3,
        accept_fn=accepts,
    )
    assert found.shape == grids.shape[:1] == nodes.shape == depth.shape


def test_eval_search_run_branches_and_errors() -> None:
    ev = priml.baselines.sudoku.eval
    model = _tiny_trm()
    config = ev.HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(1, 2),
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
        acceptance_threshold=0.0,
    )
    batch = {"media": torch.full((2, 81), 1, dtype=torch.long), "valid_count": 2}
    learned = config.make().run(model, batch)
    assert learned.accepted.shape == (2,)
    empty = config.make().run(model, {"media": batch["media"], "valid_count": 0})
    assert not bool(empty.scored.any())
    predicate = config.make().run(
        model,
        batch,
        accept_fn=lambda preds, _media: torch.zeros(preds.shape[0], dtype=torch.bool),
    )
    assert predicate.nodes.shape == (2,)


def test_eval_config_error_branches() -> None:
    """Exercise constructor guards for search, verifier, and harvest jobs."""
    ev = priml.baselines.sudoku.eval
    with pytest.raises(ValueError, match="max_act_steps"):
        ev.HpsSearch.Config(max_act_steps=0).make()
    with pytest.raises(ValueError, match="finite"):
        ev.HpsSearch.Config(acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="root_acceptance"):
        ev.HpsSearch.Config(root_acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="within"):
        ev.HpsSearch.Config(max_act_steps=2, acceptance_checkpoints=(3,)).make()
    with pytest.raises(ValueError, match="needs"):
        ev.HpsSearch.Config(
            acceptance_checkpoints=(),
            require_prediction_stability=True,
        ).make()
    with pytest.raises(ValueError, match="search_budget"):
        ev.HpsSearch.Config(search_budget=2, search_candidates=3).make()
    with pytest.raises(ValueError, match="strictly increasing"):
        ev.HpsSearch.Config(acceptance_checkpoints=(2, 2)).make()
    with pytest.raises(ValueError, match="early_exit"):
        ev.HpsSearch.Config(early_exit_at_q8=True, acceptance_checkpoints=(2,)).make()
    with pytest.raises(ValueError, match="width"):
        ev.SudokuVerifier.Config(width=5, heads=2).make()
    with pytest.raises(ValueError, match="max_steps"):
        ev.VerifierFit.Config(max_steps=0).make()
    with pytest.raises(ValueError, match="max_rows"):
        ev.VerifierAcceptor.Config(max_rows=0).make()
    with pytest.raises(ValueError, match="checkpoint"):
        ev.Harvest.Config().make()
    with pytest.raises(ValueError, match="experiment_name"):
        ev.HpsEval.Config().make()


def _reproduction_config(tmp_path: Path) -> Reproduction.Config:
    """Return a constructible pipeline config that never trains."""
    cfg = Reproduction.Config()
    cfg.experiment_name = "repro"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.generator.max_steps = 4
    cfg.generator_names = ("gen",)
    return cfg


def test_reproduction_rejects_an_invalid_config(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    cfg = _reproduction_config(tmp_path)
    cfg.experiment_name = ""
    with pytest.raises(ValueError, match="experiment_name is required"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.eval_every_steps = 0
    with pytest.raises(ValueError, match="eval_every_steps"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator_seeds = (2, 3)
    with pytest.raises(ValueError, match="exactly one training seed"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator.max_steps = float("inf")
    with pytest.raises(ValueError, match="finite max_steps"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator_names = ("gen", "gen")
    cfg.generator_seeds = (2, 3)
    with pytest.raises(ValueError, match="unique"):
        cfg.make()
    cfg.generator_names = ("gen", "other")
    with pytest.raises(ValueError, match="ONE generator"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.trigger = "final_only"
    cfg.full_eval = ev.SieveEval.Config()
    with pytest.raises(ValueError, match="agreement lock"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.full_eval = ev.SieveEval.Config()
    with pytest.raises(ValueError, match="nine_view"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.screen = "committee"
    with pytest.raises(ValueError, match="harvest_source_checkpoint"):
        cfg.make()
    cfg.harvest_source_checkpoint = "/runs/source/step.pt"
    cfg.verifier_names = ("a", "b")
    with pytest.raises(ValueError, match="three verifier"):
        cfg.make()
    cfg.verifier_names = ("a", "b", "c")
    with pytest.raises(ValueError, match="SieveEval"):
        cfg.make()


def test_reproduction_restores_stage_seconds_from_progress_or_metrics(
    tmp_path: Path,
) -> None:
    repro = _reproduction_config(tmp_path).make()
    assert repro._load_stage_seconds() == {}
    metrics = repro._metrics_path()
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("[]")
    assert repro._load_stage_seconds() == {}
    metrics.write_text('{"eval/stages": 3}')
    assert repro._load_stage_seconds() == {}
    rows = [{"stage": "train_to_2", "seconds": 2.5}, {"stage": 7, "seconds": 3}, "x"]
    metrics.write_text(json.dumps({"eval/stages": rows}))
    assert repro._load_stage_seconds() == {"train_to_2": 2.5}
    restored = repro._record_stage_seconds(
        "train_to_2",
        9.0,
        completed=False,
        stage_seconds={"train_to_2": 2.5},
    )
    assert restored == 2.5
    recorded = repro._record_stage_seconds(
        "train_to_4",
        4.0,
        completed=True,
        stage_seconds={},
    )
    assert recorded == 4.0
    assert repro._load_stage_seconds() == {"train_to_4": 4.0}
    progress = repro._progress_path()
    for payload in ("[]", '{"schema_version": 2}', '{"schema_version": 1}'):
        progress.write_text(payload)
        with pytest.raises(ValueError, match="invalid reproduction progress"):
            repro._load_stage_seconds()


def test_reproduction_reuses_only_a_harvest_from_its_own_source(
    tmp_path: Path,
) -> None:
    cfg = _reproduction_config(tmp_path)
    source = tmp_path / "source.pt"
    cfg.harvest_source_checkpoint = source
    repro = cfg.make()
    out_dir = tmp_path / "runs" / cfg.harvest.experiment_name / "harvest"
    out_dir.mkdir(parents=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"source_checkpoint": str(source)}))
    assert repro._run_harvest() == (out_dir, 0.0)
    manifest.write_text(json.dumps({"source_checkpoint": "/elsewhere.pt"}))
    with pytest.raises(ValueError, match="rolled out from"):
        repro._run_harvest()


def test_reproduction_skips_a_trained_verifier(tmp_path: Path) -> None:
    cfg = _reproduction_config(tmp_path)
    cfg.verifier_names = ("member_a", "member_b", "member_c")
    repro = cfg.make()
    checkpoint = repro._verifier_checkpoint_path(1)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    assert repro._verifier_fit(1, tmp_path) == 0.0
    fit = repro._verifier_fit_config(2, tmp_path)
    assert (fit.experiment_name, fit.seed) == ("member_c", 2)
    suffix = f"/checkpoints/step_{cfg.verifier.max_steps:08d}.pt"
    assert repro._verifier_checkpoints()[0].endswith(suffix)


class _RecordingTracker:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int, str]] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        self.logged.append((dict(metrics), step, prefix))

    def close(self) -> None:
        self.closed = True


def test_outer_run_clamps_steps_and_forwards_segments(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    tracker = _RecordingTracker()
    outer = ev._OuterRun(tracker)
    outer.log({"a": 1}, 5, prefix="x/")
    outer.log({"b": 2}, 3)
    segment = tmp_path / "metrics.json"
    outer.forward_segment(segment, 7)
    segment.write_text('{"eval/loss": 0.5}')
    outer.forward_segment(segment, 7, prefix="gen/")
    outer.close()
    assert tracker.logged == [
        ({"a": 1}, 5, "x/"),
        ({"b": 2}, 5, ""),
        ({"eval/loss": 0.5}, 7, "gen/"),
    ]
    assert tracker.closed
    disabled = ev._outer_run(None, name="n", notes="")
    disabled.log({"a": 1}, 2)
    disabled.forward_segment(segment, 2)
    disabled.close()


def test_run_training_skips_or_refuses_existing_segment_checkpoints(
    tmp_path: Path,
) -> None:
    ev = priml.baselines.sudoku.eval
    cfg = trainer.Trainer.Config()
    cfg.experiment_name = "segment"
    cfg.base_dir = tmp_path
    cfg.max_steps = 4
    checkpoints = tmp_path / "runs" / "segment" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_00000006.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="cannot safely resume"):
        ev._run_training(cfg)
    (checkpoints / "step_00000004.pt").write_bytes(b"")
    assert ev._run_training(cfg) == (0.0, False)


def test_segmented_rollout_continues_confident_rows_past_q8() -> None:
    ev = priml.baselines.sudoku.eval
    model = _tiny_trm()
    boards = torch.full((3, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    rows = torch.arange(3)
    with pytest.raises(ValueError, match="max_steps"):
        ev.segmented_rollout_rows(
            model,
            {},
            0,
            boards,
            rows,
            continue_threshold=0.0,
            early_exit_at_q8=False,
        )
    logits, scores = ev.segmented_rollout_rows(
        model,
        {},
        10,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )
    assert logits.shape == (3, 81, 11)
    assert scores.shape == (3,)


def _write_harvest_shard(directory: Path, name: str, *, groups: list[int]) -> str:
    """Write one harvest shard and return its sha256."""
    rows = len(groups)
    grid = np.full((rows, 81), 2, dtype=np.uint8)
    np.savez(
        directory / name,
        base_group_id=np.array(groups, dtype=np.int64),
        original=grid,
        candidate=grid,
        flat_view_id=np.arange(rows, dtype=np.int64),
    )
    return priml.baselines.sudoku.eval._sha256(directory / name)


def _write_manifest(
    manifest: Path,
    shards: list[dict[str, object]],
    **overrides: object,
) -> None:
    fields: dict[str, object] = {
        "schema_version": 1,
        "groups_per_shard": 2,
        "shards": shards,
        "shard_count": 2,
        "row_count": 5,
    }
    manifest.write_text(json.dumps(fields | overrides))


def test_load_harvest_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    cpu = torch.device("cpu")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        ev._load_harvest(tmp_path, device=cpu)
    cases: tuple[tuple[str, type[Exception], str], ...] = (
        ("{", ValueError, "invalid harvest manifest"),
        ("[]", TypeError, "JSON object"),
        ('{"schema_version": 2}', ValueError, "schema_version 1"),
        ('{"schema_version": 1, "shards": {}}', TypeError, "invalid shard metadata"),
        (
            '{"schema_version": 1, "shards": [3], "groups_per_shard": 2}',
            TypeError,
            "invalid entry",
        ),
        (
            '{"schema_version": 1, "shards": [{"file": "a/b"}], "groups_per_shard": 2}',
            ValueError,
            "invalid shard name",
        ),
    )
    for text, error, message in cases:
        manifest.write_text(text)
        with pytest.raises(error, match=message):
            ev._load_harvest(tmp_path, device=cpu)


def test_load_harvest_binds_shards_to_the_manifest(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    cpu = torch.device("cpu")
    manifest = tmp_path / "manifest.json"
    first = _write_harvest_shard(tmp_path, "shard-00000.npz", groups=[0, 1])
    second = _write_harvest_shard(tmp_path, "shard-00001.npz", groups=[1, 0, 1])
    good: list[dict[str, object]] = [
        {"file": "shard-00000.npz", "sha256": first, "rows": 2},
        {"file": "shard-00001.npz", "sha256": second, "rows": 3},
    ]
    broken: tuple[tuple[list[dict[str, object]], dict[str, object], str], ...] = (
        (good[:1], {}, "shard set mismatch"),
        (good[::-1], {}, "canonical shard order"),
        ([{**good[0], "sha256": "0"}, good[1]], {}, "digest mismatch"),
        ([{**good[0], "rows": 4}, good[1]], {}, "row count mismatch"),
        (good, {"row_count": 6}, "aggregate count mismatch"),
        (good, {"groups_per_shard": 1}, "base_group_id must lie"),
    )
    for shards, overrides, message in broken:
        _write_manifest(manifest, shards, **overrides)
        with pytest.raises(ValueError, match=message):
            ev._load_harvest(tmp_path, device=cpu)
    _write_manifest(manifest, good)
    originals, candidates, views, groups = ev._load_harvest(tmp_path, device=cpu)
    assert originals.shape == candidates.shape == (5, 81)
    assert views.tolist() == [0, 1, 0, 1, 2]
    assert groups.tolist() == [0, 1, 3, 2, 3]


def _load_array(path: Path) -> np.ndarray:
    """Load one ``.npy`` fixture array; ``np.load`` itself returns ``Any``."""
    return cast("np.ndarray", np.load(path))


def _write_shard(
    shard: Path,
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    flat_views: list[int],
    groups: list[int],
) -> None:
    np.savez(
        shard,
        base_group_id=np.array(groups, dtype=np.int64),
        original=original,
        candidate=candidate,
        flat_view_id=np.array(flat_views, dtype=np.int64),
    )


def _bound_harvest(
    data: Path,
    harvest: Path,
    *,
    flat_views: list[int],
    groups: list[int],
) -> Path:
    """Write a one-shard harvest whose originals are the named train rows."""
    inputs = _load_array(data / "train" / "all__inputs.npy")
    labels = _load_array(data / "train" / "all__labels.npy")
    harvest.mkdir(parents=True)
    shard = harvest / "shard-00000.npz"
    _write_shard(
        shard,
        original=inputs[flat_views],
        candidate=labels[flat_views],
        flat_views=flat_views,
        groups=groups,
    )
    digest = priml.baselines.sudoku.eval._sha256(shard)
    entry = {"file": shard.name, "sha256": digest, "rows": len(groups)}
    manifest = {
        "schema_version": 1,
        "groups_per_shard": max(groups) + 1,
        "shards": [entry],
        "shard_count": 1,
        "row_count": len(groups),
    }
    (harvest / "manifest.json").write_text(json.dumps(manifest))
    return harvest


def _verifier_data_config(root: Path, harvest: Path) -> VerifierData.Config:
    cfg = priml.baselines.sudoku.eval.VerifierData.Config()
    cfg.working_dir = root / "data"
    cfg.harvest_dir = harvest
    cfg.node_corpus = root / "node_corpus.npz"
    cfg.device = "cpu"
    cfg.batch_size = 6
    cfg.contradiction_fraction = 0.2
    cfg.train_group_end = 2
    cfg.calibration_group_end = 3
    cfg.holdout_group_end = 4
    cfg.dev_puzzles = 2
    return cfg


def test_verifier_data_rejects_a_broken_harvest_binding(tmp_path: Path) -> None:
    write_dataset(tmp_path / "data")
    harvest = _bound_harvest(
        tmp_path / "data",
        tmp_path / "harvest",
        flat_views=[0, 1, 2, 3],
        groups=[0, 1, 2, 3],
    )
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.real_fraction = 0.9
    with pytest.raises(ValueError, match="batch fractions"):
        cfg.make()
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.calibration_group_end = 5
    with pytest.raises(ValueError, match="group ends"):
        cfg.make()
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.holdout_group_end = 6
    cfg.calibration_group_end = 5
    with pytest.raises(ValueError, match="exceeds the harvest corpus"):
        cfg.make()
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.train_group_end = 0
    with pytest.raises(ValueError, match="no harvest rows"):
        cfg.make()
    inputs = _load_array(tmp_path / "data" / "train" / "all__inputs.npy")
    shard = harvest / "shard-00000.npz"
    for flat_views, message in (
        ([0, 1, 2, 9], "flat_view_id exceeds"),
        ([3, 1, 2, 0], "pairing contract"),
    ):
        _write_shard(
            shard,
            original=inputs[[0, 1, 2, 3]],
            candidate=inputs[[0, 1, 2, 3]],
            flat_views=flat_views,
            groups=[0, 1, 2, 3],
        )
        _rebind_digest(harvest)
        with pytest.raises(ValueError, match=message):
            _verifier_data_config(tmp_path, harvest).make()


def _rebind_digest(harvest: Path) -> None:
    manifest = harvest / "manifest.json"
    fields = cast(
        "dict[str, list[dict[str, object]]]",
        json.loads(manifest.read_text()),
    )
    fields["shards"][0]["sha256"] = priml.baselines.sudoku.eval._sha256(
        harvest / "shard-00000.npz",
    )
    manifest.write_text(json.dumps(fields))


def test_verifier_data_mixes_contradictions_and_node_candidates(tmp_path: Path) -> None:
    write_dataset(tmp_path / "data")
    harvest = _bound_harvest(
        tmp_path / "data",
        tmp_path / "harvest",
        flat_views=[0, 1, 2, 3],
        groups=[0, 1, 2, 3],
    )
    test_inputs = _load_array(tmp_path / "data" / "test" / "all__inputs.npy")
    node = tmp_path / "node_corpus.npz"
    np.savez(
        node,
        media=test_inputs[[1, 0]],
        final_prediction=test_inputs[[1, 0]],
        global_index=np.array([1, 0], dtype=np.int64),
    )
    data = _verifier_data_config(tmp_path, harvest).make()
    batch = data.generate_batch(torch.Generator().manual_seed(0))
    assert batch["puzzle"].shape == batch["candidate"].shape == (6, 81)
    assert data.state_dict() == {"train_epochs": 0}
    data.load_state_dict({"train_epochs": 3})
    assert data.state_dict() == {"train_epochs": 3}
    data.load_state_dict({})
    assert data.state_dict() == {"train_epochs": 3}
    strata = torch.cat([block["stratum"] for block in data.eval_dataloader()])
    assert 2 in strata.tolist()
    broken: tuple[tuple[dict[str, np.ndarray], str], ...] = (
        (
            {
                "media": test_inputs[:0],
                "final_prediction": test_inputs[:0],
                "global_index": np.zeros(0, dtype=np.int64),
            },
            "holds no rows",
        ),
        (
            {
                "media": test_inputs[:1],
                "final_prediction": test_inputs[:1],
                "global_index": np.array([7], dtype=np.int64),
            },
            "exceeds the dev slice",
        ),
        (
            {
                "media": test_inputs[[1]],
                "final_prediction": test_inputs[[1]],
                "global_index": np.array([0], dtype=np.int64),
            },
            "node-corpus/dataset binding",
        ),
    )
    for arrays, message in broken:
        np.savez(
            node,
            media=arrays["media"],
            final_prediction=arrays["final_prediction"],
            global_index=arrays["global_index"],
        )
        with pytest.raises(ValueError, match=message):
            _verifier_data_config(tmp_path, harvest).make()


def test_harvest_rejects_an_invalid_plan_before_loading() -> None:
    ev = priml.baselines.sudoku.eval
    plans: tuple[tuple[dict[str, object], str], ...] = (
        ({"group_count": 0}, "group_count"),
        ({"search_max_rows": 0}, "search_max_rows"),
        ({"checkpoints": ()}, "strictly increasing"),
        ({"search_budget": 1}, "cannot exhaust"),
    )
    for overrides, message in plans:
        cfg = ev.Harvest.Config()
        cfg.harvest_source_checkpoint = "/runs/source.pt"
        for name, value in overrides.items():
            setattr(cfg, name, value)
        with pytest.raises(ValueError, match=message):
            cfg.make()


def test_search_without_checkpoints_rolls_the_family_forward() -> None:
    ev = priml.baselines.sudoku.eval
    config = ev.HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(),
        require_prediction_stability=False,
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
        acceptance_threshold=1e9,
    )
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    result = config.make().run(_tiny_trm(), {"media": media, "valid_count": 2})
    assert result.accepted.shape == (2,)
    assert not bool(result.accepted.any())


def test_dump_and_summary_guards_reject_misaligned_rows(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    width = ev.learned_hps_output_width(81)
    rows = torch.zeros(2, width)
    media = torch.full((2, 81), 2, dtype=torch.long)
    with pytest.raises(ValueError, match="packed width"):
        ev.summarize_search(rows[:, :-1], media)
    with pytest.raises(ValueError, match="disagree on N"):
        ev.write_member_dump(tmp_path / "n.npz", rows[:1], media=media)
    with pytest.raises(ValueError, match="disagree on shape"):
        ev.write_member_dump(tmp_path / "s.npz", rows, media=media, label=media[:, :3])
    unlabeled = tmp_path / "unlabeled.npz"
    ev.write_member_dump(unlabeled, rows, media=media)
    assert ev.read_member_dump(unlabeled).label is None
    narrow = tmp_path / "narrow.npz"
    np.savez(
        narrow,
        rows=np.zeros((2, 3), np.float32),
        media=np.zeros((2, 81), np.uint8),
    )
    with pytest.raises(ValueError, match="packed width"):
        ev.read_member_dump(narrow)


def test_checkpoint_rollout_rejects_disordered_or_out_of_range_steps() -> None:
    ev = priml.baselines.sudoku.eval
    model = _tiny_trm()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    for checkpoints, message in (((3, 2), "strictly increasing"), ((2, 5), "within")):
        with pytest.raises(ValueError, match=message):
            ev.learned_checkpoint_rollout_rows(
                model,
                {},
                3,
                boards,
                torch.arange(2),
                checkpoints=checkpoints,
            )
    rollout = ev.LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2, 3),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="align by row and step"):
        ev.learned_persistence_scores(rollout, require_prediction_stability=False)
    flat = ev.LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="ranks 2 and 3"):
        ev.learned_persistence_scores(flat, require_prediction_stability=False)


def test_committee_and_view_validation_rejects_empty_inputs() -> None:
    ev = priml.baselines.sudoku.eval
    with pytest.raises(ValueError, match=">= 1 checkpoint"):
        ev.VerifierAcceptor.Config(checkpoint_paths=()).make()
    with pytest.raises(ValueError, match="at least one checkpoint"):
        ev.seed_ensemble_members(())
    with pytest.raises(ValueError, match=r"\[members, puzzles, cells\]"):
        ev.modal_grid_predictions(torch.zeros(2, 81))
    bad = ev.View("bad", (1, 1, 2, 3, 4, 5, 6, 7, 8), False)
    with pytest.raises(ValueError, match="digit_permutation"):
        ev._validate_views((bad,))
    literal = Path("/abs/metrics.json")
    assert (
        ev._fill_template(literal, base_dir=None, experiment_name="unused") is literal
    )


def test_modal_tail_votes_over_every_round_a_survivor_reached() -> None:
    ev = priml.baselines.sudoku.eval
    first = torch.full((3, 81), 2, dtype=torch.long)
    second = torch.full((2, 81), 3, dtype=torch.long)
    third = torch.full((2, 81), 3, dtype=torch.long)
    collected = [
        (torch.tensor([0, 1, 2]), first),
        (torch.tensor([1, 2]), second),
        (torch.tensor([2, 1]), third),
    ]
    tail = ev._modal_tail(collected, torch.tensor([2, 1]))
    assert torch.equal(tail, torch.full((2, 81), 3, dtype=torch.long))


def test_sieve_tail_search_escalates_the_view_policy(tmp_path: Path) -> None:
    ev = priml.baselines.sudoku.eval
    cfg = ev.SieveEval.Config()
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.tail_search = (5, 4, 3, 7)
    sieve = cfg.make()
    tail = sieve._tail_search_config()
    assert (
        tail.search_candidates,
        tail.search_depth,
        tail.search_cell_attempts,
        tail.search_budget,
    ) == (5, 4, 3, 7)
    assert tail.max_act_steps == cfg.search.max_act_steps
    assert sieve._round_names()[0] == "det_cond_halt"
    assert sieve._round_names()[-1] == "tail_escalated"
    cfg.evaluation_count = 0
    with pytest.raises(ValueError, match="evaluation_count"):
        cfg.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
