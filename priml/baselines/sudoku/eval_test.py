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
    eval as sudoku_eval,
    puzzle_data,
    trainer,
    trm,
)
from priml.baselines.sudoku.eval import (
    NINE_VIEWS,
    AgreementLockEval,
    Harvest,
    HpsEval,
    HpsSearch,
    LearnedCheckpointRollout,
    Member,
    Reproduction,
    SearchResult,
    SieveEval,
    SudokuVerifier,
    VerifierAcceptor,
    VerifierData,
    VerifierFit,
    View,
    _fill_template,
    _load_harvest,
    _outer_run,
    _OuterRun,
    _run_training,
    _sha256,
    _validate_views,
    _violated_group_counts,
    accepted_grids,
    fixed_hps_node_count,
    learned_checkpoint_rollout_rows,
    learned_hps_output_width,
    learned_persistence_scores,
    make_random_unstuck_starts,
    modal_grid_predictions,
    pack_search_rows,
    read_member_dump,
    rows_per_view,
    run_learned_pin_search_fast,
    run_pin_search_fast,
    seed_ensemble_members,
    segmented_rollout_rows,
    select_harvest_views,
    select_pin_candidates,
    sudoku_groups,
    summarize_search,
    validate_grid_permutation,
    write_member_dump,
)
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors, stored


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from numpy.lib.npyio import NpzFile

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
    puzzle_data._build_dihedral_indices.cache_clear()
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
def test_golden_replays_bit_for_bit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    puzzle_data._build_dihedral_indices.cache_clear()
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
    """Return a 3-step pipeline with 2-step evaluation segments."""
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
    generator.max_steps = 3
    generator.max_act_steps = 2
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
                h.scratch
                / "runs"
                / generator
                / "checkpoints"
                / f"step_{int(config.generator.max_steps):08d}.pt",
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
    cfg.eval_every_steps = 3  # One training segment per seed.
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
    return _tiny_config().make().eval()


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
    grid = torch.arange(2 * 81).reshape(2, 81) % 9 + 2
    for view in NINE_VIEWS:
        assert torch.equal(view.invert(view.apply(grid)), grid)
    with pytest.raises(ValueError, match="permutation"):
        validate_grid_permutation("rows", (0, 0, 1, 2, 3, 4, 5, 6, 7))
    with pytest.raises(ValueError, match="band"):
        validate_grid_permutation("rows", (0, 1, 3, 2, 4, 5, 6, 7, 8))
    with pytest.raises(ValueError, match="at least two"):
        AgreementLockEval.Config(
            experiment_name="x",
            members=(Member("x", NINE_VIEWS[0]),),
        ).make()
    assert fixed_hps_node_count(depth=2, candidates=3, cell_attempts=2) == 24
    assert (
        rows_per_view(
            search_depth=2,
            search_candidates=3,
            search_cell_attempts=2,
            random_corruption_strengths=(2, 4),
            random_starts_per_strength=2,
        )
        == 29
    )
    with pytest.raises(ValueError, match="positive"):
        fixed_hps_node_count(depth=0, candidates=2, cell_attempts=2)
    with pytest.raises(ValueError, match="nonempty"):
        rows_per_view(
            search_depth=2,
            search_candidates=2,
            search_cell_attempts=2,
            random_corruption_strengths=(),
            random_starts_per_strength=2,
        )
    selected = select_harvest_views(
        torch.tensor([0, 2, 4, 6]),
        group_count=2,
        views_per_group=2,
        seed=3,
    )
    assert selected.flat_view_id.shape == (4,)
    originals = grid[:2].clone()
    originals[:, :4] = 1
    starts = make_random_unstuck_starts(
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
        make_random_unstuck_starts(
            torch.full((2, 81), 2),
            grid[:2],
            base_group_ids=torch.tensor([0, 1]),
            view_ids=torch.tensor([0, 0]),
            seed=0,
            corruption_strengths=(2,),
            starts_per_strength=2,
        )
    assert modal_grid_predictions(
        torch.stack((grid[:2], grid[:2].flip(0), grid[:2])),
    )[1].tolist() == [0, 0]
    escalated = sudoku_eval.escalated_search_config()
    assert (
        escalated.search_candidates,
        escalated.search_depth,
        escalated.search_cell_attempts,
        escalated.search_budget,
    ) == (7, 4, 3, 8_400)
    assert (
        _fill_template(
            "/runs/{experiment_name}/x-{index}",
            base_dir=tmp_path,
            experiment_name="e",
            index=2,
        ).name
        == "x-2"
    )


def test_eval_search_helpers_and_dump_roundtrip(tmp_path: Path) -> None:
    """Cover search acceptance, persistence, packing, and npz round trips."""
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    logits = torch.zeros(2, 81, 11)
    logits[..., 2] = 4
    cells, _ = select_pin_candidates(logits, media, n_cells=2, n_digits=3)
    assert cells.shape == (2, 2)
    groups = sudoku_groups()
    preds = torch.full((2, 81), 2, dtype=torch.long)
    assert not bool(accepted_grids(preds, media, groups).any())
    assert torch.equal(_violated_group_counts(preds, groups), torch.full((2,), 27.0))
    result = SearchResult(
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
    rows = pack_search_rows(result)
    assert rows.shape == (2, learned_hps_output_width(81))
    path = tmp_path / "member.npz"
    write_member_dump(path, rows, media=media, label=preds)
    loaded = read_member_dump(path)
    assert torch.equal(loaded.final_predictions, preds.to(torch.uint8))
    assert summarize_search(rows, preds)["accepted"] == 1.0
    with pytest.raises(ValueError, match="packed width"):
        write_member_dump(tmp_path / "bad.npz", rows[:, :-1], media=media)
    with pytest.raises(ValueError, match="at least one"):
        learned_checkpoint_rollout_rows(
            _tiny_trm(),
            {},
            2,
            media,
            torch.arange(2),
            checkpoints=(),
        )
    rollout = LearnedCheckpointRollout(
        logits=logits,
        q_scores=torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]),
        predictions=preds.unsqueeze(1).expand(-1, 3, -1),
    )
    assert torch.equal(
        learned_persistence_scores(rollout, require_prediction_stability=True),
        torch.tensor([2.0, 2.0]),
    )
    unstable = LearnedCheckpointRollout(
        logits=rollout.logits,
        q_scores=rollout.q_scores,
        predictions=rollout.predictions.clone(),
    )
    unstable.predictions[0, 0, 0] = 3
    assert torch.isneginf(
        learned_persistence_scores(unstable, require_prediction_stability=True)[0],
    )


def test_eval_rollout_and_pin_search_engines() -> None:
    """Exercise segmented ACT rollouts and candidate-parallel search engines."""
    typed_model = _tiny_trm()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    kwargs: dict[str, Tensor] = {}
    logits, scores = segmented_rollout_rows(
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
    checkpoint = learned_checkpoint_rollout_rows(
        typed_model,
        kwargs,
        3,
        boards,
        torch.tensor([0, 1]),
        checkpoints=(2, 3),
    )
    assert checkpoint.q_scores.shape == (2, 2)
    assert learned_persistence_scores(
        checkpoint,
        require_prediction_stability=False,
    ).shape == (2,)
    root_logits = torch.zeros(2, 81, 11)
    root_logits[..., 2] = 1
    active = torch.tensor([True, False])
    rollout_batches: list[tuple[int, tuple[int, ...]]] = []
    rollout_boards: list[Tensor] = []

    def rollout(candidate: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        rollout_batches.append((len(candidate), tuple(int(row) for row in rows)))
        rollout_boards.append(candidate.clone())
        out = torch.zeros(candidate.shape[0], 81, 11)
        out[..., 2] = 1
        return out, torch.full((candidate.shape[0],), 3.0)

    learned = run_learned_pin_search_fast(
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
    assert torch.equal(learned.accepted, torch.tensor([False, False]))
    assert torch.equal(learned.grids, boards)
    assert torch.equal(learned.scores, torch.full((2,), float("-inf")))
    assert torch.equal(learned.nodes, torch.tensor([8, 0]))
    assert torch.equal(learned.depth, torch.tensor([-1, -1]))
    assert [len(rows) for rows in learned.visited_predictions] == [2, 4, 2]
    assert [len(rows) for rows in learned.visited_puzzles] == [2, 4, 2]
    expected_pins = boards[0].repeat(2, 1)
    expected_pins[:, 1] = torch.tensor([2, 3])
    assert torch.equal(rollout_boards[0], expected_pins)

    def accepts(predictions: Tensor, media: Tensor) -> Tensor:
        assert media.shape == (predictions.shape[0], 81)
        return torch.zeros(predictions.shape[0], dtype=torch.bool)

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        groups=sudoku_eval.sudoku_groups(),
        depth=3,
        candidates=2,
        cell_attempts=2,
        budget=12,
        max_rows=3,
        accept_fn=accepts,
    )
    assert torch.equal(found, torch.tensor([False, False]))
    assert torch.equal(grids, boards)
    assert torch.equal(nodes, torch.tensor([12, 0]))
    assert torch.equal(depth, torch.tensor([-1, -1]))
    default_found, default_grids, default_nodes, default_depth = (
        sudoku_eval.run_pin_search_fast(
            rollout,
            media=boards,
            base_logits=root_logits,
            active=active,
            groups=sudoku_eval.sudoku_groups(),
            depth=3,
            candidates=2,
            cell_attempts=2,
            budget=12,
            max_rows=3,
        )
    )
    assert torch.equal(default_found, torch.tensor([False, False]))
    assert torch.equal(default_grids, boards)
    assert torch.equal(default_nodes, torch.tensor([12, 0]))
    assert torch.equal(default_depth, torch.tensor([-1, -1]))
    assert rollout_batches
    assert all(size <= 3 for size, _ in rollout_batches)
    assert all(set(rows) <= {0} for _, rows in rollout_batches)


def test_eval_search_run_branches_and_errors() -> None:
    model = _tiny_trm()
    config = HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(1, 2),
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
        acceptance_threshold=0.0,
    )
    media = torch.full((2, 81), 1, dtype=torch.long)
    batch = {"media": media, "valid_count": 2}
    learned = config.make().run(model, batch)
    assert learned.accepted.shape == (2,)
    empty = config.make().run(model, {"media": media, "valid_count": 0})
    assert torch.equal(empty.scored, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(empty.accepted, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(empty.root_predictions, media)
    assert torch.equal(empty.final_predictions, media)
    assert torch.equal(empty.scores, torch.zeros(2))
    assert torch.equal(empty.root_scores, torch.zeros(2))
    assert torch.equal(empty.nodes, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(empty.depth, torch.full((2,), -1, dtype=torch.int64))
    assert empty.visited_predictions == []
    assert empty.visited_puzzles == []
    meta_empty = sudoku_eval._empty_result(media.to("meta"))
    assert all(
        tensor.device.type == "meta"
        for tensor in (
            meta_empty.accepted,
            meta_empty.root_predictions,
            meta_empty.final_predictions,
            meta_empty.scores,
            meta_empty.root_scores,
            meta_empty.nodes,
            meta_empty.depth,
            meta_empty.scored,
        )
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        typed_empty = sudoku_eval._empty_result(media)
        assert typed_empty.scores.dtype == torch.float32
        assert typed_empty.root_scores.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous_dtype)
    predicate = config.make().run(
        model,
        batch,
        accept_fn=lambda preds, _media: torch.zeros(preds.shape[0], dtype=torch.bool),
    )
    assert predicate.nodes.shape == (2,)


def test_eval_config_error_branches() -> None:
    """Exercise constructor guards for search, verifier, and harvest jobs."""
    with pytest.raises(ValueError, match="max_act_steps"):
        HpsSearch.Config(max_act_steps=0).make()
    with pytest.raises(ValueError, match="finite"):
        HpsSearch.Config(acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="root_acceptance"):
        HpsSearch.Config(root_acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="within"):
        HpsSearch.Config(max_act_steps=2, acceptance_checkpoints=(3,)).make()
    with pytest.raises(ValueError, match="needs"):
        HpsSearch.Config(
            acceptance_checkpoints=(),
            require_prediction_stability=True,
        ).make()
    with pytest.raises(ValueError, match="search_budget"):
        HpsSearch.Config(search_budget=2, search_candidates=3).make()
    with pytest.raises(ValueError, match="strictly increasing"):
        HpsSearch.Config(acceptance_checkpoints=(2, 2)).make()
    with pytest.raises(ValueError, match="early_exit"):
        HpsSearch.Config(early_exit_at_q8=True, acceptance_checkpoints=(2,)).make()
    with pytest.raises(ValueError, match="width"):
        SudokuVerifier.Config(width=5, heads=2).make()
    with pytest.raises(ValueError, match="max_steps"):
        VerifierFit.Config(max_steps=0).make()
    with pytest.raises(ValueError, match="max_rows"):
        VerifierAcceptor.Config(max_rows=0).make()
    with pytest.raises(ValueError, match="checkpoint"):
        Harvest.Config().make()
    with pytest.raises(ValueError, match="experiment_name"):
        HpsEval.Config().make()


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
    cfg.full_eval = SieveEval.Config()
    with pytest.raises(ValueError, match="agreement lock"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.full_eval = SieveEval.Config()
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
    tracker = _RecordingTracker()
    outer = _OuterRun(tracker)
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
    disabled = _outer_run(None, name="n", notes="")
    disabled.log({"a": 1}, 2)
    disabled.forward_segment(segment, 2)
    disabled.close()


def test_run_training_skips_or_refuses_existing_segment_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = trainer.Trainer.Config()
    cfg.experiment_name = "segment"
    cfg.base_dir = tmp_path
    cfg.max_steps = 4
    checkpoints = tmp_path / "runs" / "segment" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_00000006.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="cannot safely resume"):
        _run_training(cfg)
    (checkpoints / "step_00000004.pt").write_bytes(b"")
    assert _run_training(cfg) == (0.0, False)


def test_segmented_rollout_continues_confident_rows_past_q8() -> None:
    model = _tiny_trm()
    boards = torch.full((3, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    rows = torch.arange(3)
    with pytest.raises(ValueError, match="max_steps"):
        segmented_rollout_rows(
            model,
            {},
            0,
            boards,
            rows,
            continue_threshold=0.0,
            early_exit_at_q8=False,
        )
    one_step_logits, one_step_scores = sudoku_eval.segmented_rollout_rows(
        model,
        {},
        1,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )
    assert one_step_logits.shape == (3, 81, 11)
    assert one_step_scores.shape == (3,)
    logits, scores = sudoku_eval.segmented_rollout_rows(
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


def test_segmented_rollout_without_q8_gate_continues_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    boards = torch.tensor([[2, 1], [1, 3]])
    arange = cast("Callable[..., Tensor]", torch.arange)
    devices: list[object] = []

    def record_arange(*args: object, **kwargs: object) -> Tensor:
        devices.append(kwargs.get("device"))
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", record_arange)

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            del puzzle_identifiers, feedback_ids
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros(len(input_ids), 2, 4)
            logits[..., 2] = 1
            return {
                "logits": logits,
                "q_halt": torch.tensor([-1.0, -2.0])[: len(input_ids)],
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = ev.segmented_rollout_rows(
        cast("trm.TRM", model),
        {},
        10,
        boards,
        torch.tensor([1, 0]),
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )

    assert model.batch_sizes == [2] * 10
    assert devices == [boards.device]
    assert logits.device == scores.device == boards.device
    assert scores.tolist() == [-1.0, -2.0]


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
    return _sha256(directory / name)


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
    cpu = torch.device("cpu")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        _load_harvest(tmp_path, device=cpu)
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
            _load_harvest(tmp_path, device=cpu)


def test_load_harvest_binds_shards_to_the_manifest(tmp_path: Path) -> None:
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
            _load_harvest(tmp_path, device=cpu)
    _write_manifest(manifest, good)
    originals, candidates, views, groups = _load_harvest(tmp_path, device=cpu)
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
    digest = _sha256(shard)
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
    cfg = VerifierData.Config()
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
    fields["shards"][0]["sha256"] = _sha256(
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
    plans: tuple[tuple[dict[str, object], str], ...] = (
        ({"group_count": 0}, "group_count"),
        ({"search_max_rows": 0}, "search_max_rows"),
        ({"checkpoints": ()}, "strictly increasing"),
        ({"search_budget": 1}, "cannot exhaust"),
    )
    for overrides, message in plans:
        cfg = Harvest.Config()
        cfg.harvest_source_checkpoint = "/runs/source.pt"
        for name, value in overrides.items():
            setattr(cfg, name, value)
        with pytest.raises(ValueError, match=message):
            cfg.make()


def test_search_without_checkpoints_rolls_the_family_forward() -> None:
    config = HpsSearch.Config(
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
    width = learned_hps_output_width(81)
    rows = torch.zeros(2, width)
    media = torch.full((2, 81), 2, dtype=torch.long)
    with pytest.raises(ValueError, match="packed width"):
        summarize_search(rows[:, :-1], media)
    with pytest.raises(ValueError, match="disagree on N"):
        write_member_dump(tmp_path / "n.npz", rows[:1], media=media)
    with pytest.raises(ValueError, match="disagree on shape"):
        write_member_dump(tmp_path / "s.npz", rows, media=media, label=media[:, :3])
    unlabeled = tmp_path / "unlabeled.npz"
    write_member_dump(unlabeled, rows, media=media)
    assert read_member_dump(unlabeled).label is None
    narrow = tmp_path / "narrow.npz"
    np.savez(
        narrow,
        rows=np.zeros((2, 3), np.float32),
        media=np.zeros((2, 81), np.uint8),
    )
    with pytest.raises(ValueError, match="packed width"):
        read_member_dump(narrow)


def test_read_member_dump_decodes_each_packed_field(tmp_path: Path) -> None:
    ev = sudoku_eval
    width = ev.learned_hps_output_width(81)
    rows = np.zeros((2, width), dtype=np.float64)
    rows[0, :7] = (0.75, 0.8, 3.9, -1.2, 4.8, 0.9, 7.0)
    rows[1, :7] = (-0.5, 0.2, 8.1, 2.9, 5.1, 0.1, 9.0)
    roots = np.arange(2 * 81).reshape(2, 81) + 2
    finals = roots + 1
    rows[:, 7 : 7 + 81] = roots
    rows[:, -81:] = finals
    media = np.arange(2 * 81).reshape(2, 81).astype(np.int16) + 1
    label = media + 1
    path = tmp_path / "member.npz"
    np.savez(path, rows=rows, media=media, label=label)
    dump = ev.read_member_dump(path)
    assert dump.rows.dtype == torch.float32
    assert dump.media.dtype == torch.uint8
    assert dump.label is not None
    assert dump.label.dtype == torch.uint8
    assert dump.score.tolist() == [0.75, -0.5]
    assert dump.accepted.tolist() == [True, False]
    assert dump.nodes.tolist() == [3, 8]
    assert dump.depth.tolist() == [-1, 2]
    assert dump.candidate_count.tolist() == [4, 5]
    assert dump.solution_visited.tolist() == [True, False]
    assert dump.root_predictions.dtype == torch.uint8
    assert dump.final_predictions.dtype == torch.uint8
    assert torch.equal(dump.root_predictions, torch.from_numpy(roots.astype(np.uint8)))
    assert torch.equal(
        dump.final_predictions,
        torch.from_numpy(finals.astype(np.uint8)),
    )
    assert torch.equal(dump.media, torch.from_numpy(media.astype(np.uint8)))
    assert torch.equal(dump.label, torch.from_numpy(label.astype(np.uint8)))


def test_checkpoint_rollout_rejects_disordered_or_out_of_range_steps() -> None:
    model = _tiny_trm()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    for checkpoints, message in (((3, 2), "strictly increasing"), ((2, 5), "within")):
        with pytest.raises(ValueError, match=message):
            learned_checkpoint_rollout_rows(
                model,
                {},
                3,
                boards,
                torch.arange(2),
                checkpoints=checkpoints,
            )
    rollout = LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2, 3),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="align by row and step"):
        learned_persistence_scores(rollout, require_prediction_stability=False)
    flat = LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="ranks 2 and 3"):
        learned_persistence_scores(flat, require_prediction_stability=False)


def test_committee_and_view_validation_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match=">= 1 checkpoint"):
        VerifierAcceptor.Config(checkpoint_paths=()).make()
    with pytest.raises(ValueError, match="at least one checkpoint"):
        seed_ensemble_members(())
    with pytest.raises(ValueError, match=r"\[members, puzzles, cells\]"):
        modal_grid_predictions(torch.zeros(2, 81))
    bad = View("bad", (1, 1, 2, 3, 4, 5, 6, 7, 8), False)
    with pytest.raises(ValueError, match="digit_permutation"):
        _validate_views((bad,))
    literal = Path("/abs/metrics.json")
    assert _fill_template(literal, base_dir=None, experiment_name="unused") is literal


def test_grid_acceptance_and_group_counts_pin_validity_rules() -> None:
    ev = sudoku_eval
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    valid = solution.repeat(4, 1)
    duplicate = valid[1].clone()
    duplicate[0] = duplicate[1]
    negative = valid[2].clone()
    negative[0] = -1
    too_large = valid[3].clone()
    too_large[0] = 11
    predictions = torch.stack((valid[0], duplicate, negative, too_large))
    media = torch.zeros_like(predictions)
    media[:, 0] = valid[:, 0]
    groups = ev.sudoku_groups()
    assert ev.accepted_grids(predictions, media, groups).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert ev._violated_group_counts(predictions, groups).tolist() == [
        0.0,
        3.0,
        3.0,
        3.0,
    ]


def test_modal_tail_votes_over_every_round_a_survivor_reached() -> None:
    ev = sudoku_eval
    first = torch.full((3, 81), 2, dtype=torch.float32)
    second = torch.full((2, 81), 3, dtype=torch.float32)
    third = torch.full((2, 81), 3, dtype=torch.float32)
    collected = [
        (torch.tensor([0, 1, 2]), first),
        (torch.tensor([1, 2]), second),
        (torch.tensor([2, 1]), third),
    ]
    tail = ev._modal_tail(collected, torch.tensor([2, 1, 4]))
    assert tail.dtype == torch.float32
    assert torch.equal(tail[:2], torch.full((2, 81), 3, dtype=torch.float32))
    assert torch.equal(tail[2], torch.full((81,), 255, dtype=torch.float32))


def test_sieve_hps_round_scopes_search_to_the_survivors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    cfg = ev.SieveEval.Config()
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.model = _tiny_config()
    sieve = cfg.make()
    seen: dict[str, object] = {}
    model = object()
    released: list[bool] = []

    def eval_model(
        model_config: object,
        checkpoint_path: Path,
        device: torch.device,
    ) -> object:
        seen["eval_model"] = (model_config, checkpoint_path, device)
        return model

    monkeypatch.setattr(ev, "_eval_model", eval_model)
    monkeypatch.setattr(ev, "_release", lambda: released.append(True))

    def search_pass(**kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        dataset = cast("PuzzleDataset", kwargs["dataset"])
        seen["indices"] = dataset.config.eval_instance_indices
        seen["eval_num_instances"] = dataset.config.eval_num_instances
        # _search_pass returns batch x grid rows x grid columns for this stub.
        media = torch.ones(2, 2, 3, dtype=torch.int64)
        labels = torch.full((2, 2, 3), 2, dtype=torch.int64)
        rows = torch.zeros(len(media), ev.learned_hps_output_width(3))
        rows[:, -3:] = 7
        seen["view"] = kwargs["view"]
        seen["search"] = kwargs["search"]
        seen["deadline"] = kwargs["deadline_seconds"]
        seen["label"] = kwargs["label"]
        seen["model"] = kwargs["model"]
        seen["device"] = kwargs["device"]
        return rows, media, labels

    monkeypatch.setattr(ev, "_search_pass", search_pass)
    view = ev.NINE_VIEWS[0]
    search = cfg.search.copy_tree()
    grids, media, labels = sieve._hps_round(view, search, torch.tensor([0, 1]))
    model_config, checkpoint_path, device = cast(
        "tuple[object, Path, torch.device]",
        seen.pop("eval_model"),
    )
    assert seen == {
        "indices": (0, 1),
        "eval_num_instances": None,
        "view": view,
        "search": search,
        "deadline": cfg.max_round_eval_seconds,
        "label": f"round {view.name}",
        "model": model,
        "device": sieve.device,
    }
    assert model_config is sieve.config.model
    assert checkpoint_path == sieve._path(cfg.checkpoint_path)
    assert device == sieve.device
    assert grids.shape == (2, 3)
    assert media.shape == labels.shape == (2, 2, 3)
    assert grids.dtype == torch.int64
    assert torch.equal(grids, torch.full((2, 3), 7, dtype=torch.int64))
    assert released == [True]
    sieve._hps_round(view, search, torch.tensor([7.0, 2.0]))
    assert seen["indices"] == ()


def test_sieve_hps_round_releases_model_after_search_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    cfg = ev.SieveEval.Config()
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.model = _tiny_config()
    sieve = cfg.make()
    released: list[bool] = []

    def eval_model(
        config: object,
        checkpoint_path: Path,
        device: torch.device,
    ) -> object:
        del config, checkpoint_path, device
        return object()

    monkeypatch.setattr(ev, "_eval_model", eval_model)

    def fail_search(**_kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        raise RuntimeError("search failed")

    monkeypatch.setattr(ev, "_search_pass", fail_search)
    monkeypatch.setattr(ev, "_release", lambda: released.append(True))

    with pytest.raises(RuntimeError, match="search failed"):
        sieve._hps_round(
            ev.NINE_VIEWS[0],
            cfg.search,
            torch.tensor([4]),
        )

    assert released == [True]


def test_sieve_tail_search_escalates_the_view_policy(tmp_path: Path) -> None:
    cfg = SieveEval.Config()
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


def test_gelu_cost_counts_exact_forward_and_backward_work() -> None:
    ev = sudoku_eval
    result = ev._gelu_cost(channels=3, rows=5, dtype=torch.float32)
    assert result["flops", "primal", "elementwise", torch.float32] == 120
    assert result["flops", "adjoint", "elementwise", torch.float32] == 120
    assert result["bytes", "primal", "elementwise", torch.float32] == 120
    assert result["bytes", "adjoint", "elementwise", torch.float32] == 180
    float64_result = ev._gelu_cost(channels=3, rows=5, dtype=torch.float64)
    assert float64_result["flops", "primal", "elementwise", torch.float64] == 120
    assert float64_result["bytes", "primal", "elementwise", torch.float64] == 240


def test_verifier_eval_iterator_length_counts_ragged_batches() -> None:
    ev = sudoku_eval
    blocks = [
        {"label": torch.zeros(3), "media": torch.zeros(3, 4)},
        {"label": torch.zeros(2), "media": torch.zeros(2, 4)},
    ]
    batches = ev._VerifierEvalIterator(blocks, batch_size=2)
    assert len(batches) == 3
    assert [len(batch["label"]) for batch in batches] == [2, 1, 2]


def test_pin_search_acceptance_records_exact_winning_path() -> None:
    ev = sudoku_eval
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    media = torch.full((2, 81), 1, dtype=torch.int64)
    base_logits = torch.zeros(2, 81, 11)
    seen_boards: list[Tensor] = []
    seen_rows: list[Tensor] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        seen_boards.append(boards.clone())
        seen_rows.append(rows.clone())
        logits = torch.zeros(len(boards), 81, 11)
        logits.scatter_(2, solution.expand(len(boards), -1).unsqueeze(-1), 1)
        return logits, torch.full((len(boards),), -99.0)

    found, grids, nodes, depth = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=ev.sudoku_groups(),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=8,
        max_rows=3,
        accept_fn=lambda predictions, _: torch.ones(len(predictions), dtype=torch.bool),
    )
    assert found.tolist() == [True, False]
    assert torch.equal(grids[0], solution)
    assert torch.equal(grids[1], media[1])
    assert nodes.tolist() == [2, 0]
    assert depth.tolist() == [1, -1]
    assert [batch.shape for batch in seen_boards] == [(2, 81)]
    assert seen_rows[0].tolist() == [0, 0]
    assert torch.equal(seen_boards[0][:, 0], torch.tensor([2, 3]))
    assert torch.equal(seen_boards[0][:, 1:], torch.ones(2, 80, dtype=torch.int64))


def test_grid_scoring_and_acceptance_pin_token_boundaries() -> None:
    ev = sudoku_eval
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    predictions = solution.repeat(5, 1)
    predictions[1, 0] = 0
    predictions[2, 0] = 1
    predictions[3] = 0
    predictions[4] = 1
    assert ev._violated_group_counts(
        predictions,
        ev.sudoku_groups(),
    ).tolist() == [0.0, 3.0, 3.0, 27.0, 27.0]

    media = torch.zeros(2, 81, dtype=torch.int64)
    media[0, 0] = media[1, 0] = 2
    media[0, 8] = media[1, 8] = 10
    media[1, 1] = 1
    media[1, 2] = 0
    assert ev.accepted_grids(
        solution.expand(2, -1),
        media,
        ev.sudoku_groups(),
    ).tolist() == [True, True]
    media[1, 8] = 9
    assert ev.accepted_grids(
        solution.expand(2, -1),
        media,
        ev.sudoku_groups(),
    ).tolist() == [True, False]


def test_segmented_rollout_returns_q8_for_early_rows_and_continues_selected_rows() -> (
    None
):
    ev = sudoku_eval
    boards = torch.tensor([[2, 1, 1], [1, 1, 3]])
    rows = torch.tensor([1, 0])
    puzzle_ids = torch.tensor([17, 29, 41])

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.steps = 0
            self.calls: list[tuple[Tensor, Tensor, Tensor]] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 3), torch.zeros(batch_size, 4)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            assert puzzle_identifiers is not None
            self.steps += 1
            self.calls.append(
                (input_ids.clone(), feedback_ids.clone(), puzzle_identifiers.clone()),
            )
            logits = torch.zeros(len(input_ids), 3, 5)
            logits[..., self.steps % 5] = 1
            return {
                "logits": logits,
                "q_halt": torch.tensor([0.0, -1.0])[: len(input_ids)],
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = ev.segmented_rollout_rows(
        cast("trm.TRM", model),
        {"puzzle_identifiers": puzzle_ids},
        10,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )
    assert model.steps == 10
    assert [call[0].shape[0] for call in model.calls] == [2] * 8 + [1] * 2
    assert model.calls[0][2].tolist() == [29, 17]
    assert model.calls[8][2].tolist() == [29]
    assert torch.equal(model.calls[0][1], boards)
    assert torch.equal(model.calls[8][1], torch.tensor([[2, 3, 3]]))
    assert logits.argmax(dim=-1).tolist() == [[0, 0, 0], [3, 3, 3]]
    assert scores.tolist() == [0.0, -1.0]


def test_run_pin_search_fast_obeys_budget_and_skips_inactive_rows() -> None:
    ev = sudoku_eval
    media = torch.ones(2, 4, dtype=torch.long)
    logits = torch.zeros(2, 4, 11)
    logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        out = torch.zeros(len(boards), 4, 11)
        out[..., 2] = 1
        return out, torch.zeros(len(boards))

    found, grids, nodes, depth = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=3,
        candidates=2,
        cell_attempts=1,
        budget=6,
        max_rows=3,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )
    assert not found.any()
    assert torch.equal(grids, media)
    assert nodes.tolist() == [6, 0]
    assert depth.tolist() == [-1, -1]
    assert [len(rows) for _, rows in calls] == [2, 3, 1]
    assert [rows.tolist() for _, rows in calls] == [[0, 0], [0, 0, 0], [0]]

    calls.clear()
    _, _, depth_limited_nodes, _ = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=1,
        candidates=2,
        cell_attempts=1,
        budget=100,
        max_rows=100,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )
    assert depth_limited_nodes.tolist() == [2, 0]
    assert [len(rows) for _, rows in calls] == [2]


def test_member_dump_decodes_fields_for_a_small_grid(tmp_path: Path) -> None:
    ev = sudoku_eval
    grid_len = 2
    rows = np.array(
        [
            [0.25, 0.75, 3.9, -1.2, 4.8, 0.9, 42, 2, 3, 4, 5],
            [-0.5, 0.2, 8.1, 2.9, 5.1, 0.1, 43, 6, 7, 8, 9],
        ],
    )
    assert rows.shape[1] == ev.learned_hps_output_width(grid_len)
    path = tmp_path / "member.npz"
    np.savez(
        path,
        rows=rows,
        media=np.array([[7, 8], [10, 11]], dtype=np.int16),
        label=np.array([[9, 10], [12, 13]], dtype=np.int16),
    )
    dump = ev.read_member_dump(path)
    assert dump.rows.dtype == torch.float32
    assert dump.score.tolist() == [0.25, -0.5]
    assert dump.accepted.tolist() == [True, False]
    assert dump.nodes.tolist() == [3, 8]
    assert dump.depth.tolist() == [-1, 2]
    assert dump.candidate_count.tolist() == [4, 5]
    assert dump.solution_visited.tolist() == [True, False]
    assert dump.root_predictions.tolist() == [[2, 3], [6, 7]]
    assert dump.final_predictions.tolist() == [[4, 5], [8, 9]]
    assert dump.media.tolist() == [[7, 8], [10, 11]]
    assert dump.label is not None
    assert dump.label.tolist() == [[9, 10], [12, 13]]


def test_sieve_tail_search_preserves_non_escalated_policy() -> None:
    ev = sudoku_eval
    config = ev.SieveEval.Config()
    config.search.max_act_steps = 3
    config.search.acceptance_threshold = 2.5
    config.tail_search = (5, 4, 3, 7)
    sieve = object.__new__(ev.SieveEval)
    sieve.config = config
    tail = sieve._tail_search_config()
    assert (
        tail.search_candidates,
        tail.search_depth,
        tail.search_cell_attempts,
        tail.search_budget,
    ) == (5, 4, 3, 7)
    assert tail.max_act_steps == 3
    assert tail.acceptance_threshold == 2.5
    assert tail.search_max_rows == config.search.search_max_rows


def test_contradiction_rows_follow_two_seeded_draws_and_align_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    data = object.__new__(ev.VerifierData)
    data.device = torch.device("meta")
    data.train_inputs = torch.tensor([[2, 1], [3, 1], [4, 1]])
    data.train_labels = torch.tensor([[5, 6], [7, 8], [9, 10]])
    generator = torch.Generator().manual_seed(91)
    expected_generator = torch.Generator().manual_seed(91)
    row_indices = torch.randint(3, (4,), generator=expected_generator)
    sibling_indices = torch.randint(3, (4,), generator=expected_generator)
    original_randint = torch.randint
    devices: list[object] = []

    def record_randint(*args: object, **kwargs: object) -> Tensor:
        devices.append(kwargs.get("device"))
        return original_randint(
            cast(int, args[0]),
            cast(tuple[int, ...], args[1]),
            device="cpu",
            generator=cast(torch.Generator, kwargs["generator"]),
        )

    monkeypatch.setattr(torch, "randint", record_randint)
    puzzle, candidate, solution = data._contradiction_rows(4, generator=generator)
    assert devices == [data.device, data.device]
    assert torch.equal(puzzle, data.train_inputs[row_indices].long())
    assert torch.equal(candidate, data.train_labels[sibling_indices].long())
    assert torch.equal(solution, data.train_labels[row_indices].long())


def test_verifier_eval_iterator_yields_exact_ragged_chunks() -> None:
    ev = sudoku_eval
    blocks = [
        {"label": torch.arange(5), "media": torch.arange(10).reshape(5, 2)},
        {
            "label": torch.arange(2) + 10,
            "media": torch.arange(6).reshape(2, 3) + 10,
        },
    ]
    batches = list(ev._VerifierEvalIterator(blocks, batch_size=2))
    assert [batch["label"].tolist() for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
        [10, 11],
    ]
    assert [batch["media"].shape for batch in batches] == [
        (2, 2),
        (2, 2),
        (1, 2),
        (2, 3),
    ]
    assert [batch["media"].tolist() for batch in batches] == [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9]],
        [[10, 11, 12], [13, 14, 15]],
    ]


def test_unchecked_eval_helpers_have_exact_cost_and_round_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    cost = ev._gelu_cost(channels=2, rows=3, dtype=torch.float32)
    assert cost["flops", "primal", "elementwise", torch.float32] == 48
    assert cost["flops", "adjoint", "elementwise", torch.float32] == 48

    blocks = [
        {"label": torch.arange(5), "media": torch.arange(10).reshape(5, 2)},
        {"label": torch.arange(2) + 10, "media": torch.arange(6).reshape(2, 3)},
    ]
    batches = ev._VerifierEvalIterator(blocks, batch_size=2)
    assert len(batches) == 4
    assert [batch["label"].tolist() for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
        [10, 11],
    ]

    config = ev.SieveEval.Config()
    config.experiment_name = "sieve"
    config.base_dir = tmp_path
    config.runtime.device = "cpu"
    config.tail_search = (3, 2, 4, 30)
    sieve = config.make()
    observed: dict[str, object] = {}

    def eval_model(*_: object) -> object:
        return object()

    def search_pass(**kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        dataset = cast("PuzzleDataset", kwargs["dataset"])
        observed["indices"] = dataset.config.eval_instance_indices
        observed["search"] = kwargs["search"]
        return (
            torch.zeros(2, ev.learned_hps_output_width(4)),
            torch.ones(2, 4),
            torch.ones(2, 4),
        )

    monkeypatch.setattr(ev, "_eval_model", eval_model)
    monkeypatch.setattr(ev, "_search_pass", search_pass)
    grids, media, labels = sieve._hps_round(
        ev.NINE_VIEWS[0],
        config.search,
        torch.tensor([1, 3]),
    )
    assert observed == {"indices": (1, 3), "search": config.search}
    assert grids.shape == media.shape == labels.shape == (2, 4)
    assert grids.dtype == torch.int64


def test_accepted_grids_rejects_each_invalid_grid_and_preserves_valid_solution() -> (
    None
):
    solution = _solution()
    duplicate = solution.clone()
    duplicate[0] = duplicate[1]
    blank = solution.clone()
    blank[0] = 1
    media = torch.zeros(4, 81, dtype=torch.long)
    media[:, 0] = solution[0]
    media[3, 1] = solution[1]
    media[3, 2] = 1
    mismatched_given = solution.clone()
    mismatched_given[0] = 3
    predictions = torch.stack((solution, duplicate, blank, mismatched_given))

    accepted = sudoku_eval.accepted_grids(
        predictions,
        media,
        sudoku_eval.sudoku_groups(),
    )

    assert accepted.tolist() == [True, False, False, False]


def test_run_pin_search_fast_uses_default_acceptance_and_preserves_outputs() -> None:
    ev = sudoku_eval
    solution = _solution()
    media = torch.ones(2, 81, dtype=torch.long)
    base_logits = torch.zeros(2, 81, 11)
    base_logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        logits = torch.zeros(len(boards), 81, 11)
        logits.scatter_(2, solution.expand(len(boards), -1).unsqueeze(-1), 1)
        return logits, torch.zeros(len(boards))

    found, grids, nodes, depth = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=ev.sudoku_groups(),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=6,
        max_rows=3,
    )

    assert found.tolist() == [True, False]
    assert torch.equal(grids[0], solution)
    assert torch.equal(grids[1], media[1])
    assert nodes.tolist() == [2, 0]
    assert depth.tolist() == [1, -1]
    assert [len(rows) for _, rows in calls] == [2]
    assert calls[0][1].tolist() == [0, 0]


def test_run_pin_search_fast_expands_breadth_first_with_exact_budget_and_chunks() -> (
    None
):
    ev = sudoku_eval
    media = torch.ones(2, 5, dtype=torch.long)
    base_logits = torch.zeros(2, 5, 11)
    base_logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        logits = torch.zeros(len(boards), 5, 11)
        logits[..., 2] = 1
        return logits, torch.zeros(len(boards))

    found, grids, nodes, depth = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=3,
        candidates=2,
        cell_attempts=1,
        budget=6,
        max_rows=3,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )

    assert not found.any()
    assert torch.equal(grids, media)
    assert nodes.tolist() == [6, 0]
    assert depth.tolist() == [-1, -1]
    assert [len(rows) for _, rows in calls] == [2, 3, 1]
    assert [rows.tolist() for _, rows in calls] == [[0, 0], [0, 0, 0], [0]]


def test_run_pin_search_fast_uses_root_and_child_selector_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    original = ev.select_pin_candidates
    widths: list[tuple[int, int]] = []

    def record_widths(
        logits: Tensor,
        boards: Tensor,
        *,
        n_cells: int,
        n_digits: int,
    ) -> tuple[Tensor, Tensor]:
        widths.append((n_cells, n_digits))
        return original(logits, boards, n_cells=n_cells, n_digits=n_digits)

    monkeypatch.setattr(ev, "select_pin_candidates", record_widths)
    # Pin-search inputs use the production batch x token-grid contract.
    media = torch.ones((1, 5), dtype=torch.int64)
    base_logits = torch.zeros((1, 5, 11))
    base_logits[..., 2] = 1

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        return (
            base_logits.expand(len(boards), -1, -1),
            torch.zeros(len(boards)),
        )

    ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(1, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=100,
        max_rows=100,
        accept_fn=lambda predictions, _: torch.zeros(
            len(predictions),
            dtype=torch.bool,
        ),
    )

    assert widths == [(2, 2), (1, 2), (1, 2)]


def test_run_pin_search_fast_preserves_device_dtypes_and_conversions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = sudoku_eval
    missing = object()
    zero = cast("Callable[..., Tensor]", torch.zeros)
    full = cast("Callable[..., Tensor]", torch.full)
    arange = cast("Callable[..., Tensor]", torch.arange)
    tensor_to = cast("Callable[..., Tensor]", Tensor.to)
    zero_calls: list[tuple[object, object]] = []
    full_calls: list[tuple[object, object]] = []
    arange_calls: list[object] = []
    to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_zeros(*args: object, **kwargs: object) -> Tensor:
        zero_calls.append((kwargs.get("dtype", missing), kwargs.get("device", missing)))
        return zero(*args, **kwargs)

    def record_full(*args: object, **kwargs: object) -> Tensor:
        full_calls.append((kwargs.get("dtype", missing), kwargs.get("device", missing)))
        return full(*args, **kwargs)

    def record_arange(*args: object, **kwargs: object) -> Tensor:
        arange_calls.append(kwargs.get("device", missing))
        return arange(*args, **kwargs)

    def record_to(
        self: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        to_calls.append((args, kwargs))
        return tensor_to(self, *args, **kwargs)

    media = torch.ones((2, 5), dtype=torch.int64)
    media[0, 0] = 2
    media[1, 0] = 3
    base_logits = torch.zeros((2, 5, 11))
    base_logits[..., 2] = 1
    monkeypatch.setattr(torch, "zeros", record_zeros)
    monkeypatch.setattr(torch, "full", record_full)
    monkeypatch.setattr(torch, "arange", record_arange)
    monkeypatch.setattr(Tensor, "to", record_to)
    acceptance_calls = 0

    def accept_one_row(predictions: Tensor, boards: Tensor) -> Tensor:
        nonlocal acceptance_calls
        acceptance_calls += 1
        if acceptance_calls == 1:
            return boards[:, 0] == 2
        return torch.ones(predictions.shape[0], dtype=torch.bool)

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        return base_logits[:1].expand(len(boards), -1, -1), boards[:, 0]

    found, _, _, depth_used = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(2, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=2,
        candidates=2,
        cell_attempts=1,
        budget=10,
        max_rows=100,
        accept_fn=accept_one_row,
    )

    assert found.tolist() == [True, True]
    assert depth_used.tolist() == [1, 2]
    assert zero_calls == [
        (torch.int64, media.device),
        (torch.bool, media.device),
    ]
    assert full_calls == [(torch.int64, media.device)]
    assert arange_calls == [media.device, media.device]
    assert to_calls == [((torch.int64,), {})] * 4


def test_run_pin_search_fast_gathers_every_active_puzzle() -> None:
    ev = sudoku_eval
    media = torch.ones((2, 5), dtype=torch.int64)
    base_logits = torch.zeros((2, 5, 11))
    base_logits[..., 2] = 1

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        logits = base_logits[:1].expand(len(boards), -1, -1)
        return logits, torch.zeros(len(boards))

    _, _, nodes, _ = ev.run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(2, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=1,
        candidates=2,
        cell_attempts=1,
        budget=2,
        max_rows=100,
        accept_fn=lambda predictions, _: torch.zeros(
            len(predictions),
            dtype=torch.bool,
        ),
    )

    assert nodes.tolist() == [2, 2]


def test_empty_result_preserves_batch_device_and_field_contract() -> None:
    ev = sudoku_eval
    media = torch.tensor([[2, 1, 3], [1, 4, 1]])

    class ModelStub:
        puzzle_emb = None

    result = (
        ev.HpsSearch.Config()
        .make()
        .run(
            cast("trm.TRM", ModelStub()),
            {"media": media, "valid_count": 0},
        )
    )

    assert result.depth.tolist() == [-1, -1]
    assert result.accepted.shape == result.scored.shape == (2,)
    assert result.accepted.dtype == result.scored.dtype == torch.bool
    assert result.root_predictions.shape == result.final_predictions.shape == (2, 3)
    assert (
        result.root_predictions.dtype == result.final_predictions.dtype == torch.int64
    )
    assert result.scores.shape == result.root_scores.shape == (2,)
    assert result.scores.dtype == result.root_scores.dtype == torch.float32
    assert result.nodes.shape == result.depth.shape == (2,)
    assert result.nodes.dtype == result.depth.dtype == torch.int64
    assert result.visited_predictions == []
    assert result.visited_puzzles == []

    meta_result = (
        ev.HpsSearch.Config()
        .make()
        .run(
            cast("trm.TRM", ModelStub()),
            {"media": media.to("meta"), "valid_count": 0},
        )
    )
    assert all(
        tensor.device.type == "meta"
        for tensor in (
            meta_result.accepted,
            meta_result.root_predictions,
            meta_result.final_predictions,
            meta_result.scores,
            meta_result.root_scores,
            meta_result.nodes,
            meta_result.depth,
            meta_result.scored,
        )
    )


def test_segmented_rollout_keeps_q8_boundary_and_gathers_live_rows() -> None:
    ev = sudoku_eval
    boards = torch.tensor([[10, 1, 1], [11, 1, 3], [1, 4, 1]])
    puzzle_ids = torch.tensor([10, 20, 30, 40])

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls: list[tuple[Tensor, Tensor, Tensor]] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            assert puzzle_identifiers is not None
            self.calls.append(
                (input_ids.clone(), feedback_ids.clone(), puzzle_identifiers.clone()),
            )
            step = int(z_slow[0, 0]) + 1
            logits = torch.zeros(len(input_ids), 3, 5)
            logits[..., step % 5] = 1
            q_by_id = {10: -1.0, 20: -1.0, 30: 0.0, 40: 1.0}
            q_halt = torch.tensor(
                [q_by_id[int(identifier)] for identifier in puzzle_identifiers],
            )
            return {
                "logits": logits,
                "q_halt": q_halt,
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = ev.segmented_rollout_rows(
        cast("trm.TRM", model),
        {"puzzle_identifiers": puzzle_ids},
        10,
        boards,
        torch.tensor([2, 3, 0]),
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )

    assert len(model.calls) == 10
    assert [call[0].shape[0] for call in model.calls] == [3] * 8 + [2] * 2
    assert model.calls[0][2].tolist() == [30, 40, 10]
    assert model.calls[8][2].tolist() == [30, 40]
    assert torch.equal(model.calls[0][1], boards)
    assert torch.equal(model.calls[8][1], torch.tensor([[10, 3, 3], [3, 3, 3]]))
    assert logits.argmax(dim=-1).tolist() == [[0, 0, 0], [0, 0, 0], [3, 3, 3]]
    assert scores.tolist() == [0.0, 1.0, -1.0]


def test_segmented_rollout_feedback_uses_per_cell_predictions() -> None:
    ev = sudoku_eval
    boards = torch.ones((2, 3), dtype=torch.int64)

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.feedback: list[Tensor] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            del puzzle_identifiers
            self.feedback.append(feedback_ids.clone())
            logits = torch.zeros(len(input_ids), 3, 4)
            logits[:, 0, 1] = 1
            logits[:, 1, 2] = 2
            logits[:, 2, 3] = 3
            return {
                "logits": logits,
                "q_halt": torch.ones(len(input_ids)),
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    ev.segmented_rollout_rows(
        cast("trm.TRM", model),
        {},
        10,
        boards,
        torch.tensor([0, 1]),
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )

    assert len(model.feedback) == 10
    assert torch.equal(model.feedback[9], torch.tensor([[1, 2, 3], [1, 2, 3]]))


def test_read_member_dump_uses_strict_half_threshold_for_flags(
    tmp_path: Path,
) -> None:
    grid_len = 2
    rows = np.zeros(
        (3, sudoku_eval.learned_hps_output_width(grid_len)),
        dtype=np.float32,
    )
    rows[:, 1] = (0.49, 0.5, 0.51)
    rows[:, 2] = (3.1, 4.2, 5.3)
    rows[:, 5] = (0.49, 0.5, 0.51)
    path = tmp_path / "member.npz"
    np.savez(path, rows=rows, media=np.ones((3, grid_len), dtype=np.uint8))

    dump = sudoku_eval.read_member_dump(path)

    assert dump.accepted.tolist() == [False, False, True]
    assert dump.nodes.tolist() == [3, 4, 5]
    assert dump.nodes.dtype == torch.int64
    assert dump.solution_visited.tolist() == [False, False, True]


def test_read_member_dump_reports_actual_packed_width(tmp_path: Path) -> None:
    rows = np.zeros((2, 3), dtype=np.float32)
    # Packed-member fixtures use two rows and three token columns.
    media = np.zeros((2, 3), dtype=np.uint8)
    path = tmp_path / "wrong_width.npz"
    np.savez(path, rows=rows, media=media)

    with pytest.raises(
        ValueError,
        match="packed width 3 does not match learned-HPS width 13\\.",
    ) as error:
        sudoku_eval.read_member_dump(path)

    assert str(error.value) == "packed width 3 does not match learned-HPS width 13."


def test_modal_tail_preserves_dtype_sentinel_and_survivor_order() -> None:
    ev = sudoku_eval
    first = torch.full((3, 5), 2, dtype=torch.float32)
    second = torch.full((2, 5), 3, dtype=torch.float32)
    collected = [
        (torch.tensor([0, 1, 2]), first),
        (torch.tensor([1, 2]), second),
    ]
    result = ev._modal_tail(collected, torch.tensor([2, 1, 3]))

    assert result.dtype == torch.float32
    assert torch.equal(result[0], torch.full((5,), 2.0))
    assert torch.equal(result[1], torch.full((5,), 2.0))
    assert torch.equal(result[2], torch.full((5,), 255.0))


def test_group_violations_token_normalization_matches_clamp() -> None:
    ev = sudoku_eval
    predictions = torch.tensor([[-3, -1, 0, 1, 2, 10, 11, 12, 13]])
    # Sudoku group indexing is a single nine-cell row by contract.
    groups = torch.arange(9).reshape(1, 9)
    old_tokens = predictions.clamp(min=0, max=10)[:, groups]
    expected_counts = torch.nn.functional.one_hot(
        old_tokens,
        num_classes=len(ev.IDENTITY_DIGITS) + 2,
    ).sum(dim=2)
    expected_violations = (
        (
            (expected_counts[..., 2:] > 1).any(dim=2)
            | (expected_counts[..., :2] > 0).any(dim=2)
        )
        .sum(dim=1)
        .float()
    )

    assert torch.equal(
        ev._violated_group_counts(predictions, groups),
        expected_violations,
    )


def test_group_violations_and_acceptance_enforce_sudoku_token_rules() -> None:
    ev = sudoku_eval
    solution = _solution()
    duplicate = solution.clone()
    duplicate[0] = duplicate[1]
    blank = solution.clone()
    blank[0] = 1
    negative = solution.clone()
    negative[0] = -1
    too_large = solution.clone()
    too_large[0] = 11
    predictions = torch.stack((solution, duplicate, blank, negative, too_large))
    groups = ev.sudoku_groups()

    assert ev._violated_group_counts(predictions, groups).tolist() == [
        0.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ]
    media = torch.zeros_like(predictions)
    media[:, 0] = solution[0]
    assert ev.accepted_grids(predictions, media, groups).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    mismatched_given = media[:1].clone()
    mismatched_given[0, 0] = 3
    assert ev.accepted_grids(
        solution.unsqueeze(0),
        mismatched_given,
        groups,
    ).tolist() == [False]
    ten_token = torch.where(solution == 10)[0][0]
    invalid_digit = solution.clone()
    invalid_digit[ten_token] = 11
    assert ev.accepted_grids(
        invalid_digit.unsqueeze(0),
        torch.zeros_like(solution).unsqueeze(0),
        groups,
    ).tolist() == [False]
    two_token = torch.where(solution == 2)[0][0]
    alternate_two = torch.where(
        solution == 2,
        3,
        torch.where(solution == 3, 2, solution),
    )
    given_two = torch.zeros_like(solution).unsqueeze(0)
    given_two[0, two_token] = 2
    assert ev.accepted_grids(
        alternate_two.unsqueeze(0),
        given_two,
        groups,
    ).tolist() == [
        False,
    ]
    invalid_given = torch.zeros_like(solution).unsqueeze(0)
    invalid_given[0, two_token] = 11
    assert ev.accepted_grids(solution.unsqueeze(0), invalid_given, groups).tolist() == [
        True,
    ]
    alternate = torch.where(
        solution == 10,
        9,
        torch.where(solution == 9, 10, solution),
    )
    ten_givens = torch.zeros_like(solution).unsqueeze(0)
    ten_givens[0, solution == 10] = 10
    assert ev.accepted_grids(alternate.unsqueeze(0), ten_givens, groups).tolist() == [
        False,
    ]


def _solution() -> Tensor:
    return torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
