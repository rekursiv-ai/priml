"""Tests for the sudoku experiment ladder.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
ladder stays checkable on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, Self, cast

import pytest

from priml.baselines.sudoku import experiments
from priml.baselines.sudoku.act import (
    HaltTraining,
    LearnedStart,
    SampledMinimum,
    StreamingPool,
    ZeroStart,
)
from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridEmbedding,
    PredictionFeedback,
)
from priml.baselines.sudoku.eval import (
    NINE_VIEWS,
    AgreementLockEval,
    SieveEval,
    VerifierAcceptor,
)
from priml.baselines.sudoku.trainer import Trainer
from priml.baselines.sudoku.trm import recipe_block
from priml.model.attention.attention import Attention
from priml.model.init import kaiming_uniform
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from priml.baselines.sudoku.experiments import SudokuTrainLoop


LADDER: Final[list[tuple[str, Callable[[], SudokuTrainLoop]]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp003", experiments.exp003),
    ("exp015", experiments.exp015),
    ("exp_smoke", experiments.exp_smoke),
]


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[[], SudokuTrainLoop],
) -> None:
    """A config must build without a dataset or a GPU."""
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "sudoku"


def test_the_lattice_is_two_independent_axes() -> None:
    """Architecture and recurrence vary separately, spanning all four corners."""
    corners = {
        (
            type(config.step.model.block).__qualname__.split(".")[0],
            config.step.model.recurrence is not None,
        )
        for config in (
            experiments.exp000(),
            experiments.exp001(),
            experiments.exp002(),
            experiments.exp003(),
        )
    }
    assert corners == {
        ("TransformerBlock", False),
        ("MLPMixerBlock", False),
        ("TransformerBlock", True),
        ("MLPMixerBlock", True),
    }


def test_solver_geometry_follows_the_dataset_spec() -> None:
    default = experiments.exp000().finalize()
    default_embedding = default.step.model.embedding
    assert isinstance(default_embedding, GridEmbedding.Config)
    assert default_embedding.grid_shape == (81,)
    assert default.step.model.vocab_size == 11

    config = experiments.exp000()
    config.dataset.spec.grid_shape = (4, 4)
    config.dataset.spec.box_shape = (2, 2)
    config.dataset.spec.vocab_size = 6
    finalized = config.finalize()
    model = finalized.step.model
    assert model.vocab_size == 6
    embedding = model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_shape == (16,)
    positions = embedding.channels[0]
    assert isinstance(positions, FactoredPositions.Config)
    assert positions.grid_shape == (4, 4)
    assert positions.box_shape == (2, 2)


def test_exp001_changes_only_the_block() -> None:
    base, fork = experiments.exp000(), experiments.exp001()
    assert isinstance(base.step.model.block, TransformerBlock.Config)
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.model.recurrence is base.step.model.recurrence is None
    assert fork.step.pool is base.step.pool is None
    assert fork.max_steps == base.max_steps


def test_exp002_adds_recurrence_and_its_feedback_channel() -> None:
    """Recurrence and prediction feedback move together, and say why.

    A recurrence that re-reads only the original puzzle carries its belief
    solely in the latent; the feedback channel is what lets it refine its own
    answer, so the two are one change rather than two.
    """
    base, fork = experiments.exp000(), experiments.exp002()
    assert base.step.model.recurrence is None
    assert fork.step.model.recurrence is not None
    assert fork.step.pool is not None
    assert type(fork.step.model.block) is type(base.step.model.block)

    base_embedding = base.step.model.embedding
    fork_embedding = fork.step.model.embedding
    assert isinstance(base_embedding, GridEmbedding.Config)
    assert isinstance(fork_embedding, GridEmbedding.Config)
    assert [type(c) for c in base_embedding.channels] == [FactoredPositions.Config]
    assert [type(c) for c in fork_embedding.channels] == [
        FactoredPositions.Config,
        PredictionFeedback.Config,
    ]


@pytest.mark.parametrize("factory", [experiments.exp002, experiments.exp003])
def test_the_recurrent_rungs_replay_as_trained(
    factory: Callable[[], SudokuTrainLoop],
) -> None:
    """They trained with no fed-back grid and zero-seeded slots; keep it so."""
    pool = factory().step.pool
    assert pool is not None
    assert pool.feedback is None
    assert isinstance(pool.start, ZeroStart.Config)


def test_exp002_pool_uses_its_trained_halting_and_batch_settings() -> None:
    pool = experiments.exp002().step.pool
    assert pool is not None
    assert pool.batch_size == experiments.exp002().dataset.batch_size
    assert pool.max_steps == 32
    assert isinstance(pool.halting, HaltTraining.Config)
    assert isinstance(pool.halting.exploration, SampledMinimum.Config)


def test_exp015_changes_only_the_pool() -> None:
    """Feedback and the learned start move together; nothing else does."""
    base, fork = experiments.exp002(), experiments.exp015()
    assert base.step.pool is not None
    assert fork.step.pool is not None
    assert fork.step.pool.feedback is not None
    assert isinstance(fork.step.pool.start, LearnedStart.Config)
    fork.step.pool.feedback = base.step.pool.feedback
    fork.step.pool.start = base.step.pool.start
    fork.experiment_name = base.experiment_name
    assert fork.pformat() == base.pformat()


def test_exp015_rejects_a_parent_without_an_atomic_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = experiments.exp002()
    non_atomic_pool = StreamingPool.Config()
    config.step.pool = non_atomic_pool
    monkeypatch.setattr(experiments, "exp002", lambda: config)

    with pytest.raises(TypeError) as error:
        experiments.exp015()
    assert str(error.value) == (
        f"exp002's pool is atomic; got {type(non_atomic_pool)}."
    )


def test_the_clue_range_follows_the_vocabulary() -> None:
    """Clues are every digit token, so a resized vocabulary moves the range."""
    config = experiments.exp015()
    config.dataset.spec.vocab_size = 6
    finalized = config.copy_tree().finalize()
    assert finalized.step.pool is not None
    assert finalized.step.pool.feedback is not None
    assert finalized.step.pool.feedback.givens == (2, 5)


def test_exp003_is_exp002_with_the_other_block() -> None:
    base, fork = experiments.exp002(), experiments.exp003()
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.pool is not None
    assert base.step.pool is not None
    assert fork.step.pool.max_steps == base.step.pool.max_steps


def test_mixer_block_pins_both_mixing_paths() -> None:
    block = experiments._mixer_block()
    assert block.seq_len == -1
    assert block.prenorm is False
    assert isinstance(block.token_mixer, SwiGLU.Config)
    assert isinstance(block.channel_mixer, SwiGLU.Config)
    for mixer in (block.token_mixer, block.channel_mixer):
        assert isinstance(mixer.norm, RMSNorm.Config)
        assert mixer.init_weight is kaiming_uniform
        assert mixer.init_weight_out is kaiming_uniform


def test_the_pool_is_built_to_the_models_shape() -> None:
    """A pool sized independently of the model would fail only at runtime."""
    config = experiments.exp002().copy_tree().finalize()
    act = config.step.pool
    model = config.step.model
    assert act is not None
    assert act.grid_len == model.grid_len
    assert act.seq_len == model.total_seq_len
    assert act.channels_hidden == model.channels_in


def test_schedule_horizon_matches_the_step_budget() -> None:
    """A schedule annealing past the end of training wastes the last steps."""
    for name, factory in LADDER:
        config = factory()
        assert config.max_steps == config.step.total_train_steps, name


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke, base = experiments.exp_smoke(), experiments.exp000()
    assert smoke.max_steps < base.max_steps
    assert smoke.step.model.channels_in < base.step.model.channels_in
    assert smoke.step.model.num_layers <= base.step.model.num_layers
    assert smoke.dataset.batch_size < base.dataset.batch_size
    assert smoke.dataset.num_train_puzzles is not None
    assert (
        smoke.step.model.channels_in,
        smoke.step.model.num_layers,
        smoke.dataset.batch_size,
        smoke.dataset.num_train_puzzles,
        smoke.dataset.num_eval_puzzles,
        smoke.max_steps,
        smoke.step.total_train_steps,
        smoke.num_steps_eval,
    ) == (32, 1, 8, 4, 4, 4, 4, 2)


def test_exp000_matches_its_golden_config() -> None:
    """Pin the WHOLE finalized ``exp000``: the control every fork is measured against."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=experiments.exp000())


TRM_LADDER: Final = (
    experiments.exp004,
    experiments.exp005,
    experiments.exp006,
    experiments.exp007,
    experiments.exp008,
    experiments.exp009,
    experiments.exp010,
)
"""The TRM training ladder; each forks the previous one."""

EVALUATIONS: Final = (
    experiments.exp011,
    experiments.exp012,
    experiments.exp013,
    experiments.exp014,
)
"""Forks of exp010 that change how an answer is produced and accepted."""

EXP010_CHECKPOINT: Final = "/runs/exp010/checkpoints/step_00019500.pt"


type _Tree = dict[str, "_Tree"] | list["_Tree"] | str | int | float | bool | None
"""A serialized config tree (jsonpickle-style ``py/*`` dict/list/primitives)."""


class _Runtime(Protocol):
    device: torch.device | str | None


class _Runner(Protocol):
    """What every TRM experiment config shares: identity, runtime, lifecycle."""

    experiment_name: str

    @property
    def runtime(self) -> _Runtime:
        """Return the runtime config."""
        ...

    def copy_tree(self) -> Self:
        """Return a deep copy."""
        ...

    def finalize(self) -> Self:
        """Apply derived defaults."""
        ...

    def make(self) -> object:
        """Build the runner."""
        ...


@pytest.mark.parametrize(
    ("name", "factory"),
    [(f.__name__, f) for f in (*TRM_LADDER, *EVALUATIONS)],
)
def test_trm_experiment_constructs_and_finalizes(
    name: str,
    factory: Callable[[], _Runner],
) -> None:
    """Every rung builds without a dataset or a GPU, under its own name."""
    cfg = factory()
    assert cfg.experiment_name == name
    cfg.copy_tree().finalize()


@pytest.mark.parametrize("factory", EVALUATIONS)
def test_evaluation_runner_makes(factory: Callable[[], _Runner]) -> None:
    """Runners defer data, weights, and models to ``run()``; ``make()`` is cheap."""
    config = factory()
    config.runtime.device = "cpu"
    config.make()


def test_exp005_adds_only_the_2d_position_tables() -> None:
    assert_rung_delta(
        experiments.exp004(),
        experiments.exp005(),
        {"experiment_name", "model.pos2d_grid_shape"},
    )


def test_exp006_adds_only_qk_norm_and_the_convergence_horizon() -> None:
    assert_rung_delta(
        experiments.exp005(),
        experiments.exp006(),
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


def test_exp006_pins_its_convergence_horizon() -> None:
    config = experiments.exp006()
    assert (config.max_steps, config.total_train_steps, config.num_steps_eval) == (
        12_000,
        12_000,
        500,
    )


def test_exp007_adds_only_the_consolidated_recipe_stack() -> None:
    assert_rung_delta(
        experiments.exp006(),
        experiments.exp007(),
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


def test_exp008_adds_only_the_feedback_repair_channel() -> None:
    assert_rung_delta(
        experiments.exp007(),
        experiments.exp008(),
        {"experiment_name", "feedback"},
    )


def test_exp009_adds_only_depth_and_the_recipe_seed() -> None:
    assert_rung_delta(
        experiments.exp008(),
        experiments.exp009(),
        {"experiment_name", "model.slow_cycles", "model.fast_cycles", "seed"},
    )


def test_exp010_adds_only_the_capmax_horizon() -> None:
    assert_rung_delta(
        experiments.exp009(),
        experiments.exp010(),
        {"experiment_name", "max_steps", "total_train_steps"},
    )


def test_exp010_is_the_flat_recipe() -> None:
    """The trainer defaults ARE the recipe: exp010 adds only identity and data."""
    expected = Trainer.Config()
    expected.study_name = "sudoku"
    expected.experiment_name = "exp010"
    expected.seed = 44
    expected.max_steps = 19_500
    expected.num_steps_eval = 500
    expected.model.block = recipe_block()
    expected.dataset.batch_size = 384
    expected.dataset.seed = 0
    expected.dataset.eval_num_instances = 2_000
    expected.dataset.augment = True
    expected.dataset.augment_digits_only = True
    assert leaf_delta(expected, experiments.exp010()) == []


def test_exp004_mirrors_the_annealed_baseline() -> None:
    cfg = experiments.exp004().copy_tree().finalize()
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


def test_exp011_scores_exp010_with_the_recipe_search() -> None:
    cfg = experiments.exp011().copy_tree().finalize()
    assert str(cfg.checkpoint_path) == EXP010_CHECKPOINT
    parent = experiments.exp010().copy_tree().finalize()
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


def test_exp012_trains_exp010_then_locks_nine_views() -> None:
    cfg = experiments.exp012()
    assert leaf_delta(experiments.exp010(), cfg.generator) == []
    assert (cfg.generator_names, cfg.generator_seeds) == (
        ("exp012_generator",),
        (44,),
    )
    assert (cfg.screen, cfg.trigger) == ("nine_view", "dev_perfect")
    lock = cfg.full_eval
    assert isinstance(lock, AgreementLockEval.Config)
    assert tuple(member.view for member in lock.members) == NINE_VIEWS


def test_exp013_trains_exp010_then_sieves_under_a_fresh_committee() -> None:
    cfg = experiments.exp013()
    assert leaf_delta(experiments.exp010(), cfg.generator) == []
    assert str(cfg.harvest_source_checkpoint) == EXP010_CHECKPOINT
    assert cfg.verifier_names == tuple(f"exp013_verifier_s{s}" for s in range(3))
    assert (cfg.screen, cfg.trigger) == ("committee", "dev_perfect")
    sieve = cfg.full_eval
    assert isinstance(sieve, SieveEval.Config)
    assert isinstance(sieve.acceptor, VerifierAcceptor.Config)
    assert sieve.acceptor.threshold == 0.0
    assert sieve.views == NINE_VIEWS
    assert sieve.tail_search == (7, 4, 3, 8_400)


def test_exp013_full_evaluation_locks_with_its_own_committee() -> None:
    """The full-set sieve offers grids to the verifiers this pipeline trained."""
    cfg = experiments.exp013()
    cfg.runtime.device = "cpu"
    sieve = cfg.make()._sieve_run_config(19_500).make()
    acceptor = sieve.config.acceptor
    assert isinstance(acceptor, VerifierAcceptor.Config)
    assert acceptor.checkpoint_paths == tuple(
        f"/runs/exp013_verifier_s{seed}/checkpoints/step_00004000.pt"
        for seed in range(3)
    )


def test_evaluation_forks_pin_their_run_identity_and_seeds() -> None:
    exp012 = experiments.exp012()
    exp013 = experiments.exp013()
    exp014 = experiments.exp014()
    assert (exp012.study_name, exp012.generator_names) == (
        "sudoku",
        ("exp012_generator",),
    )
    assert (exp013.study_name, exp013.generator_names) == (
        "sudoku",
        ("exp013_generator",),
    )
    assert (exp014.screen, exp014.trigger, exp014.generator_seeds) == (
        "single_view",
        "final_only",
        (44, 45, 46),
    )


def test_exp014_swaps_nine_views_for_three_seeds() -> None:
    assert_rung_delta(
        experiments.exp012(),
        experiments.exp014(),
        {
            "experiment_name",
            "screen",
            "trigger",
            "generator_seeds",
            "generator_names",
        },
    )


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
