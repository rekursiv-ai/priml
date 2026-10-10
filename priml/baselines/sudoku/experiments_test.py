"""Tests for the sudoku experiment chain.

Each fork test asserts the DELTA the fork applies, which is what enforces one
change per experiment: a fork that quietly moved a second knob fails here
rather than producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
chain stays checkable on any machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import pytest

from priml.baselines.sudoku import experiments
from priml.baselines.sudoku.agreement_lock import AgreementLockEval
from priml.baselines.sudoku.hps_eval import HpsEval
from priml.baselines.sudoku.hypotheses import (
    HypothesisRecursion,
    WinnerRecompute,
    WinnerTakeAll,
)
from priml.baselines.sudoku.latent_recursion import LatentRecursion, recipe_block
from priml.baselines.sudoku.reproduction import Reproduction
from priml.baselines.sudoku.sieve import FittedSieve, SieveEval, StaircaseSieveEval
from priml.baselines.sudoku.trainer import Trainer
from priml.baselines.sudoku.verifier import (
    DenoisingVerifier,
    VerifierAcceptor,
    row_column_positions,
)
from priml.baselines.sudoku.verifier_fit import DenoisingLoss
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.model.attention.attention import Attention
from priml.paths import resolve_working_dir
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable


type _Tree = dict[str, "_Tree"] | list["_Tree"] | str | int | float | bool | None
"""A serialized config tree (jsonpickle-style ``py/*`` dict/list/primitives)."""


type _Runner = HpsEval.Config | Reproduction.Config | StaircaseSieveEval.Config
"""A rung that evaluates or runs a pipeline, under its own process runtime."""

type _Rung = Trainer.Config | _Runner | FittedSieve.Config
"""Any rung's config."""


RUNGS: Final[list[tuple[str, Callable[[], _Rung]]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp003", experiments.exp003),
    ("exp004", experiments.exp004),
    ("exp005", experiments.exp005),
    ("exp006", experiments.exp006),
    ("exp007", experiments.exp007),
    ("exp008", experiments.exp008),
    ("exp009", experiments.exp009),
    ("exp010", experiments.exp010),
    ("exp011", experiments.exp011),
    ("exp012", experiments.exp012),
    ("exp013", experiments.exp013),
    ("exp_smoke", experiments.exp_smoke),
]
"""Every factory, in chain order."""

TRAINED: Final = (
    experiments.exp000,
    experiments.exp001,
    experiments.exp002,
    experiments.exp003,
    experiments.exp004,
    experiments.exp005,
    experiments.exp006,
    experiments.exp009,
    experiments.exp010,
)
"""The rungs that train a generator in one run."""

EXP006_CHECKPOINT: Final = "/runs/exp006/checkpoints/step_00019500.pt"
EXP010_CHECKPOINT: Final = "/runs/exp010/checkpoints/step_00010000.pt"


def test_experiment_chain_has_fourteen_rungs() -> None:
    """Expose exactly the fourteen numbered experiment factories."""
    assert {f"exp{index:03d}" for index in range(14)} == {
        name
        for name in vars(experiments)
        if name.startswith("exp") and name[3:].isdigit()
    }


@pytest.mark.parametrize(("name", "factory"), RUNGS, ids=[n for n, _ in RUNGS])
def test_every_rung_finalizes_under_its_own_name(
    name: str,
    factory: Callable[[], _Rung],
) -> None:
    """A config must build without a dataset or a GPU."""
    config = factory()
    assert config.experiment_name == name
    config.copy_tree().finalize()


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("exp000", experiments.exp000),
        ("exp001", experiments.exp001),
        ("exp002", experiments.exp002),
        ("exp003", experiments.exp003),
        ("exp004", experiments.exp004),
        ("exp005", experiments.exp005),
        ("exp006", experiments.exp006),
        ("exp007", experiments.exp007),
        ("exp008", experiments.exp008),
        ("exp009", experiments.exp009),
        ("exp010", experiments.exp010),
        ("exp011", experiments.exp011),
        ("exp012", experiments.exp012),
        ("exp013", experiments.exp013),
        ("exp_smoke", experiments.exp_smoke),
    ],
)
def test_experiment_matches_its_full_config_golden(
    name: str,
    factory: Callable[[], _Rung],
) -> None:
    assert_pprint_golden(test_file=__file__, name=name, config=factory())


@pytest.mark.parametrize(
    "factory",
    [experiments.exp007, experiments.exp008, experiments.exp011, experiments.exp012],
)
def test_evaluation_runner_makes_without_data(
    factory: Callable[[], _Runner],
) -> None:
    """Runners defer data, weights, and models to ``run()``; ``make()`` is cheap."""
    config = factory()
    config.runtime.device = "cpu"
    config.make()


@pytest.mark.parametrize("factory", TRAINED)
def test_schedule_horizon_matches_the_step_budget(
    factory: Callable[[], Trainer.Config],
) -> None:
    """A schedule annealing past the end of training wastes the last steps."""
    config = factory()
    assert config.max_steps == config.total_train_steps


def test_exp000_mirrors_the_annealed_baseline() -> None:
    cfg = experiments.exp000().copy_tree().finalize()
    assert cfg.seed == 0
    assert (cfg.max_steps, cfg.total_train_steps) == (62_000, 62_000)
    assert (cfg.max_act_steps, cfg.label_smoothing) == (16, 0.2)
    assert (cfg.ema_decay, cfg.ema_warmup_steps) == (0.9, 5_000)
    assert (cfg.csp_loss_weight, cfg.feedback) == (0.0, False)
    assert cfg.model.pos2d_grid_shape is None
    assert (cfg.model.slow_cycles, cfg.model.fast_cycles) == (3, 4)
    assert cfg.model.block is not None
    attn = cfg.model.block.attn
    assert isinstance(attn, Attention.Config)
    assert attn.norm_qk is None
    assert cfg.dataset.augment_digits_only is False


def test_exp001_adds_only_the_2d_position_tables() -> None:
    assert_rung_delta(
        experiments.exp000(),
        experiments.exp001(),
        {"experiment_name", "model.pos2d_grid_shape"},
    )


def test_exp002_adds_only_qk_norm_and_the_convergence_horizon() -> None:
    assert_rung_delta(
        experiments.exp001(),
        experiments.exp002(),
        {
            "experiment_name",
            "model.block.attn.norm_qk",
            "model.block.attn.norm_qk.channels_in",
            "model.block.attn.norm_qk.device",
            "model.block.attn.norm_qk.dtype",
            "model.block.attn.norm_qk.elementwise_affine",
            "model.block.attn.norm_qk.eps",
            "max_steps",
            "total_train_steps",
            "num_steps_eval",
        },
    )


def test_exp003_adds_only_the_consolidated_recipe_stack() -> None:
    assert_rung_delta(
        experiments.exp002(),
        experiments.exp003(),
        {
            "experiment_name",
            "max_act_steps",
            "label_smoothing",
            "ema_decay",
            "ema_warmup_steps",
            "csp_loss_weight",
            "dataset.augment_digits_only",
        },
    )


def test_exp004_adds_only_the_feedback_repair_channel() -> None:
    assert_rung_delta(
        experiments.exp003(),
        experiments.exp004(),
        {"experiment_name", "feedback"},
    )


def test_exp005_adds_only_depth_and_the_recipe_seed() -> None:
    assert_rung_delta(
        experiments.exp004(),
        experiments.exp005(),
        {"experiment_name", "model.slow_cycles", "model.fast_cycles", "seed"},
    )


def test_exp006_adds_only_the_longer_horizon() -> None:
    assert_rung_delta(
        experiments.exp005(),
        experiments.exp006(),
        {"experiment_name", "max_steps", "total_train_steps"},
    )


def test_exp006_is_the_flat_recipe() -> None:
    """The trainer defaults ARE the recipe: exp006 adds only identity and data."""
    expected = Trainer.Config()
    expected.study_name = "sudoku"
    expected.experiment_name = "exp006"
    expected.seed = 44
    expected.max_steps = 19_500
    expected.num_steps_eval = 500
    expected.model.block = recipe_block()
    expected.dataset.batch_size = 384
    expected.dataset.seed = 0
    expected.dataset.eval_num_instances = 2_000
    expected.dataset.augment = True
    expected.dataset.augment_digits_only = True
    assert leaf_delta(expected, experiments.exp006()) == []


def test_exp007_scores_exp006_with_the_recipe_search() -> None:
    cfg = experiments.exp007().copy_tree().finalize()
    assert str(cfg.checkpoint_path) == EXP006_CHECKPOINT
    parent = experiments.exp006().copy_tree().finalize()
    assert leaf_delta(parent.model, cfg.model) == []
    assert cfg.evaluation_count == 422_786
    search = cfg.search
    assert (
        search.search_candidates,
        search.search_depth,
        search.search_cell_attempts,
        search.search_budget,
    ) == (5, 3, 2, 512)
    assert search.acceptance_threshold == 7.875
    assert search.acceptance_checkpoints == (24, 28, 32)


def test_exp008_trains_exp006_then_sieves_under_a_fresh_committee() -> None:
    cfg = experiments.exp008()
    assert leaf_delta(experiments.exp006(), cfg.generator) == []
    assert str(cfg.harvest_source_checkpoint) == EXP006_CHECKPOINT
    assert cfg.generator_name == "exp008_generator"
    assert cfg.verifier_names == tuple(f"exp008_verifier_s{s}" for s in range(3))
    assert cfg.screen == "committee"
    sieve = cfg.full_eval
    assert isinstance(sieve, SieveEval.Config)
    assert isinstance(sieve.acceptor, VerifierAcceptor.Config)
    assert sieve.acceptor.threshold == 0.0
    assert sieve.views == NINE_VIEWS
    assert sieve.tail_search == (7, 4, 3, 8_400)


@pytest.mark.parametrize("factory", [experiments.exp008, experiments.exp012])
def test_pipeline_stages_build_the_generators_architecture(
    factory: Callable[[], Reproduction.Config],
) -> None:
    """Every stage that loads a generator checkpoint builds its architecture.

    A stage's ``model`` is otherwise a bare ``LatentRecursion.Config``, which has no
    vocabulary or grid shape and cannot build at all.
    """
    cfg = factory().copy_tree().finalize()
    model = cfg.generator.model
    assert leaf_delta(model, cfg.full_eval.model) == []
    if cfg.screen == "committee":
        assert leaf_delta(model, cfg.harvest.model) == []
    assert cfg.full_eval.model.vocab_size == 11
    assert cfg.full_eval.model.puzzle_grid_shape == (81,)


def test_exp008_full_evaluation_locks_with_its_own_committee() -> None:
    """The full-set sieve offers grids to the verifiers this pipeline trained."""
    cfg = experiments.exp008()
    cfg.runtime.device = "cpu"
    pipeline = cfg.make()
    acceptor = pipeline._sieve_run_config(19_500).make().config.acceptor
    assert isinstance(acceptor, VerifierAcceptor.Config)
    assert acceptor.checkpoint_paths == tuple(
        f"/runs/exp008_verifier_s{seed}/checkpoints/step_00004000.pt"
        for seed in range(3)
    )


def test_exp009_trains_two_hypotheses_winner_take_all() -> None:
    parent, cfg = experiments.exp006(), experiments.exp009()
    assert isinstance(cfg.model, HypothesisRecursion.Config)
    assert (cfg.model.num_hypotheses, cfg.model.hypothesis_seed) == (2, 44)
    assert isinstance(cfg.routing, WinnerTakeAll.Config)
    plain = LatentRecursion.Config().update(cfg.model, skip_missing=True)
    assert leaf_delta(parent.model, plain) == []
    cfg.model, cfg.routing = parent.model, parent.routing
    assert_rung_delta(
        parent,
        cfg,
        {"experiment_name", "max_steps", "total_train_steps"},
    )
    assert (cfg.max_steps, cfg.total_train_steps) == (22_000, 22_000)


def test_exp010_recomputes_the_winner_of_three_at_batch_1152() -> None:
    parent, cfg = experiments.exp009(), experiments.exp010()
    assert isinstance(cfg.routing, WinnerRecompute.Config)
    assert isinstance(cfg.model, HypothesisRecursion.Config)
    assert (cfg.model.num_hypotheses, cfg.dataset.batch_size) == (3, 1_152)
    assert_rung_delta(
        parent,
        cfg,
        {
            "experiment_name",
            "routing",
            "model.num_hypotheses",
            "dataset.batch_size",
            "max_steps",
            "total_train_steps",
        },
    )


def test_exp011_stairs_exp010_ahead_of_exp008s_sieve() -> None:
    cfg = experiments.exp011().copy_tree().finalize()
    hypothesis_parent = experiments.exp010().copy_tree().finalize()
    # Finalization bookkeeping differs for a slot finalized twice; values do not.
    assert [
        delta
        for delta in leaf_delta(hypothesis_parent.model, cfg.hypothesis_model)
        if delta[0] != "_finalized"
    ] == []
    assert str(cfg.hypothesis_checkpoint_path) == EXP010_CHECKPOINT
    ladder_parent = experiments.exp006().copy_tree().finalize()
    assert leaf_delta(ladder_parent.model, cfg.model) == []
    assert str(cfg.checkpoint_path) == EXP006_CHECKPOINT
    assert (cfg.qsel_hypotheses, cfg.qsel_seed) == (5, 8_902)
    committee = tuple(
        f"/runs/exp008_verifier_s{seed}/checkpoints/step_00004000.pt"
        for seed in range(3)
    )
    assert cfg.verifier_checkpoints == committee
    assert isinstance(cfg.acceptor, VerifierAcceptor.Config)
    assert cfg.acceptor.checkpoint_paths == committee
    reference = SieveEval.Config().copy_tree().finalize()
    assert (cfg.views, cfg.tail_search, cfg.det_halt_threshold) == (
        reference.views,
        reference.tail_search,
        reference.det_halt_threshold,
    )
    assert leaf_delta(reference.search, cfg.search) == []


def test_exp011_runs_the_staircase_then_the_view_ladder() -> None:
    cfg = experiments.exp011()
    cfg.runtime.device = "cpu"
    sieve = cfg.make()
    assert sieve._round_names() == [
        "hypothesis_det_cond_halt",
        "hypothesis_qsel",
        "hypothesis_canonical_hps",
        *(view.name for view in NINE_VIEWS),
        "tail_escalated",
    ]


def test_exp012_trains_exp006_then_locks_nine_views() -> None:
    cfg = experiments.exp012()
    assert leaf_delta(experiments.exp006(), cfg.generator) == []
    assert cfg.generator_name == "exp012_generator"
    assert cfg.screen == "nine_view"
    lock = cfg.full_eval
    assert isinstance(lock, AgreementLockEval.Config)
    assert tuple(member.view for member in lock.members) == NINE_VIEWS


def test_exp013_fits_a_prior_free_committee_then_runs_exp011s_sieve() -> None:
    cfg = experiments.exp013()
    assert [member.seed for member in cfg.members] == [0, 1, 2]
    for seed, member in enumerate(cfg.members):
        assert member.experiment_name == f"exp013_verifier_s{seed}"
        assert member.max_steps == 16_000
        assert isinstance(member.model, DenoisingVerifier.Config)
        assert member.model.attention_mask is None
        assert member.model.position_index is row_column_positions
        assert isinstance(member.loss, DenoisingLoss.Config)
    sieve = cfg.sieve
    acceptor = sieve.acceptor
    assert isinstance(acceptor, VerifierAcceptor.Config)
    assert leaf_delta(cfg.members[0].model, acceptor.model) == []
    acceptor.model = VerifierAcceptor.Config().model
    assert leaf_delta(experiments.exp011(), sieve) == []


def test_exp013_locks_with_the_members_it_fits() -> None:
    cfg = experiments.exp013().copy_tree().finalize()
    assert isinstance(cfg, FittedSieve.Config)
    committee = tuple(
        f"/runs/exp013_verifier_s{seed}/checkpoints/step_00016000.pt"
        for seed in range(3)
    )
    assert cfg.sieve.experiment_name == "exp013"
    assert cfg.sieve.verifier_checkpoints == committee
    assert isinstance(cfg.sieve.acceptor, VerifierAcceptor.Config)
    assert cfg.sieve.acceptor.checkpoint_paths == committee


def _final_checkpoint(rung: Trainer.Config, base: Path) -> Path:
    """Return where a training rung's own checkpointer writes its last step."""
    cfg = rung.copy_tree()
    cfg.base_dir = base
    checkpointer = cfg.finalize().checkpointer
    assert checkpointer is not None
    step = int(cfg.max_steps)
    return Path(checkpointer.working_dir) / checkpointer.filename.format(step=step)


def test_every_consumer_reads_the_checkpoint_its_producer_writes(
    tmp_path: Path,
) -> None:
    """Each frozen-checkpoint rung derives the path its parent run writes."""
    exp006 = _final_checkpoint(experiments.exp006(), tmp_path)
    exp010 = _final_checkpoint(experiments.exp010(), tmp_path)
    staircase = experiments.exp011()
    reads = {
        "exp007": experiments.exp007().checkpoint_path,
        "exp008": experiments.exp008().harvest_source_checkpoint,
        "exp011": staircase.checkpoint_path,
    }
    for rung, path in reads.items():
        assert resolve_working_dir(tmp_path, str(path)) == exp006, rung
    assert resolve_working_dir(tmp_path, str(staircase.hypothesis_checkpoint_path)) == (
        exp010
    )
    pipeline = experiments.exp008()
    pipeline.base_dir = tmp_path
    pipeline.runtime.device = "cpu"
    committee = pipeline.make()
    assert tuple(
        resolve_working_dir(tmp_path, str(path))
        for path in staircase.verifier_checkpoints
    ) == tuple(committee._verifier_checkpoint_path(seed) for seed in range(3))


def test_exp013_fits_on_exp008s_harvest_corpus(tmp_path: Path) -> None:
    pipeline = experiments.exp008()
    pipeline.base_dir = tmp_path
    harvest = pipeline.copy_tree().finalize().harvest
    corpus = resolve_working_dir(
        tmp_path,
        str(harvest.run_dir).format(experiment_name=harvest.experiment_name),
    )
    cfg = experiments.exp013()
    cfg.base_dir = tmp_path
    for member in cfg.copy_tree().finalize().members:
        assert Path(member.dataset.harvest_dir) == corpus


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke, base = experiments.exp_smoke(), experiments.exp000()
    assert smoke.max_steps < base.max_steps
    assert smoke.model.channels_in < base.model.channels_in
    assert smoke.model.num_layers < base.model.num_layers
    assert smoke.dataset.batch_size < base.dataset.batch_size
    assert (
        smoke.model.channels_in,
        smoke.model.num_layers,
        smoke.model.slow_cycles,
        smoke.model.fast_cycles,
        smoke.max_act_steps,
        smoke.dataset.batch_size,
        smoke.dataset.eval_num_instances,
        smoke.max_steps,
        smoke.total_train_steps,
        smoke.num_steps_eval,
    ) == (32, 1, 1, 1, 2, 8, 4, 4, 4, 2)
    assert smoke.checkpointer is None


class _Serializable(Protocol):
    def serialize(self) -> object:
        """Return the config as a jsonpickle-style tree."""
        ...


def leaf_delta(
    base: _Serializable,
    variant: _Serializable,
) -> list[tuple[str, object, object]]:
    """Differing serialized leaves as sorted (path, base, variant) tuples."""
    base_leaves = _flatten(cast(_Tree, base.serialize()))
    variant_leaves = _flatten(cast(_Tree, variant.serialize()))
    return [
        (path, base_leaves.get(path, "<absent>"), variant_leaves.get(path, "<absent>"))
        for path in sorted(base_leaves.keys() | variant_leaves.keys())
        if base_leaves.get(path, "<absent>") != variant_leaves.get(path, "<absent>")
    ]


def assert_rung_delta(
    base: _Serializable,
    variant: _Serializable,
    allowed: set[str],
) -> None:
    """Assert the fork changes EXACTLY the allowed config fields, no more, no less."""
    changed = {_field_path(path) for path, _, _ in leaf_delta(base, variant)}
    unexpected = {
        path
        for path in changed
        if not any(path == a or path.startswith(f"{a}.") for a in allowed)
    }
    assert not unexpected, f"fork changes undeclared fields: {sorted(unexpected)}"
    dead = {
        a
        for a in allowed
        if not any(path == a or path.startswith(f"{a}.") for path in changed)
    }
    assert not dead, f"fork declares fields that did not change: {sorted(dead)}"


def _field_path(path: str) -> str:
    """Collapse a serialized leaf path to its config field path."""
    return path.split(".py/", maxsplit=1)[0]


def _flatten(tree: _Tree, prefix: str = "") -> dict[str, object]:
    """Flatten a serialized dict/list tree into ``{dotted_path: leaf}``."""
    if isinstance(tree, dict):
        out: dict[str, object] = {}
        for key, value in tree.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(tree, list):
        out = {}
        for index, value in enumerate(tree):
            out.update(_flatten(value, f"{prefix}[{index}]"))
        return out
    return {prefix: tree}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
