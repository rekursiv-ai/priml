"""Exercise ARC checkpoint imports through the existing evaluation step."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pickle

from configgle.cli_override import apply_overrides
from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.checkpoint import (
    CheckpointActTrainStep,
    read_step_state,
)
from priml.baselines.arcagi.train_step import PooledActTrainStep
from priml.baselines.arcagi.train_step_test import (
    batches,
    canonical_name,
    port_config,
)
from priml.model.ae.checkpoint import LocalFile
from priml.model.attention.attention import Attention
from priml.model.conditioning import CategorySlotFiLM
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import SingleLatentGridSolver
from priml.model.prefix import ComposedSparsePrefix, RegisterTokens
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers.parameter_filter import excluding, trainable
from priml.testing.golden import read_tensors
from priml.train.ema import EMA, NoEMA
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Literal

    from torch import Tensor

    from priml.model.grid_solver import GridSolver


@pytest.mark.parametrize("register_prefix", [False, True])
def test_native_and_historical_ema_import(
    tmp_path: Path,
    register_prefix: bool,
) -> None:
    """The historical flat EMA and current native EMA select the same weights."""
    config = port_config("exp006").model
    if register_prefix:
        config.prefix = RegisterTokens.Config(num_tokens=2)
    model = config.make()
    live = {name: value.clone() for name, value in model.state_dict().items()}
    shadow = {
        name: torch.full_like(value, 0.375) for name, value in model.named_parameters()
    }
    payloads = [
        {"step": {"model": live, "ema": {"shadow_params": shadow}}},
        {
            "step": {
                "model": {canonical_name(name): value for name, value in live.items()},
                "ema": {canonical_name(name): value for name, value in shadow.items()},
            },
        },
    ]
    for index, payload in enumerate(payloads):
        path = tmp_path / f"checkpoint-{index}.pt"
        torch.save(payload, path)
        actual = _read_eval_state(path, model=model)
        assert actual.keys() == live.keys()
        for name, value in live.items():
            assert torch.equal(actual[name], shadow.get(name, value)), name


def test_model_only_and_uninitialized_native_ema(tmp_path: Path) -> None:
    """Model-only exports and checkpoints before EMA begins retain live weights."""
    model = port_config("exp006").model.make()
    for index, payload in enumerate(
        [
            {"model": model.state_dict()},
            {"step": {"model": model.state_dict(), "ema": {"global_step": 0}}},
        ],
    ):
        path = tmp_path / f"checkpoint-{index}.pt"
        torch.save(payload, path)
        actual = _read_eval_state(path, model=model)
        _assert_state_equal(actual, expected=model.state_dict())


def test_historical_ema_restores_native_module_shadow(tmp_path: Path) -> None:
    """Adapt historical flat shadows to the injected module EMA representation."""
    model = port_config("exp006").model.make()
    live = model.state_dict()
    shadow = {
        name: torch.full_like(value, 0.25) for name, value in model.named_parameters()
    }
    path = tmp_path / "historical.pt"
    torch.save({"model": live, "ema": shadow}, path)
    ema = EMA.Config().make()
    state = read_step_state(path, model=model, ema=ema)
    ema.load_state_dict(from_plain(state["ema"], dict[str, object]))
    with ema.apply_to(model):
        _assert_state_equal(
            model.state_dict(),
            expected={name: shadow.get(name, value) for name, value in live.items()},
        )


def test_native_module_ema_overlays_parameters_and_retains_live_buffers(
    tmp_path: Path,
) -> None:
    """Module EMA saves buffers too, but native eval swaps parameters only."""
    model = port_config("exp006").model.make()
    live = model.state_dict()
    shadow = {name: torch.full_like(value, 0.625) for name, value in live.items()}
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"step": {"model": live, "ema": {"shadow_model": shadow, "global_step": 1}}},
        path,
    )
    parameters = dict(model.named_parameters())
    actual = _read_eval_state(path, model=model)
    _assert_state_equal(
        actual,
        expected={
            name: shadow[name] if name in parameters else value
            for name, value in live.items()
        },
    )


def test_checkpoint_step_loads_before_eval_without_restoring_training(
    tmp_path: Path,
) -> None:
    """Restore evaluation progress and EMA while leaving the optimizer fresh."""
    source = port_config("exp006").make()
    live = source.model.state_dict()
    shadow = {
        name: torch.full_like(value, 0.25)
        for name, value in source.model.named_parameters()
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": {
                "model": live,
                "ema": {"shadow_params": shadow},
                "optimizer": {"must_not_be_loaded": True},
                "global_step": 280_000,
            },
        },
        path,
    )
    config = CheckpointActTrainStep.Config().update(port_config("exp006"))
    config.checkpoint = LocalFile.Config(path=path)
    loaded = config.make()
    assert loaded.global_step == 280_000
    assert isinstance(loaded.ema, EMA)
    expected = {name: shadow.get(name, value) for name, value in live.items()}
    with loaded.ema.apply_to(loaded.model):
        _assert_state_equal(loaded.model.state_dict(), expected=expected)
    assert loaded.optimizer.state_dict() == source.optimizer.state_dict()


@pytest.mark.parametrize("shadow_kind", ["param_dict", "module"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_native_checkpoint_preserves_evaluation_policy(
    tmp_path: Path,
    shadow_kind: Literal["param_dict", "module"],
    selected: bool,
    steps: int,
) -> None:
    """Round-trip native evaluation before, at, and after EMA warmup.

    Args:
      tmp_path: Isolated checkpoint directory.
      shadow_kind: Native EMA representation.
      selected: Exclude the heads from averaging when true.
      steps: Updates completed before saving.

    """
    config = port_config("exp006")
    assert isinstance(config.ema, EMA.Config)
    config.ema.shadow_kind = shadow_kind
    if selected:
        config.ema.select = excluding(trainable, "head")
    source = config.make()
    optimizer_fresh = source.optimizer.state_dict()
    for batch in batches()[:steps]:
        source.train_step(**batch)
    path = tmp_path / "native.pt"
    torch.save({"step": source.state_dict()}, path)
    imported = CheckpointActTrainStep.Config().update(config)
    imported.checkpoint = LocalFile.Config(path=path)
    loaded = imported.make()
    assert loaded.global_step == source.global_step
    with source._eval_weights(), loaded._eval_weights():
        _assert_state_equal(
            loaded.model.state_dict(),
            expected=source.model.state_dict(),
        )
    assert loaded.optimizer.state_dict() == optimizer_fresh


def test_real_module_ema_checkpoint_is_safe_to_read(tmp_path: Path) -> None:
    """Read the actual serialized EMA mapping with Torch's weights-only loader."""
    config = port_config("exp006")
    assert isinstance(config.ema, EMA.Config)
    config.ema.shadow_kind = "module"
    config.ema.update_after_step = 0
    source = config.make()
    for batch in batches()[:2]:
        source.train_step(**batch)
    path = tmp_path / "module.pt"
    torch.save({"step": source.state_dict()}, path)
    actual = _read_eval_state(path, model=source.net)
    with source._eval_weights():
        _assert_state_equal(actual, expected=source.model.state_dict())


@pytest.mark.parametrize("edit", ["missing", "unexpected", "shape", "duplicate"])
def test_checkpoint_rejects_incompatible_state(tmp_path: Path, edit: str) -> None:
    """Missing, extra, wrong-shaped and aliased weights fail before evaluation."""
    model = port_config("exp006").model.make()
    state = dict(model.state_dict())
    key = "embedding.embed_tokens.weight"
    if edit == "missing":
        del state[key]
    elif edit == "unexpected":
        state["model.inner.unknown.weight"] = torch.zeros(2, 3)
    elif edit == "shape":
        state[key] = torch.zeros(2, 3)
    else:
        state["embed_tokens.weight"] = state[key]
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match=edit):
        _read_eval_state(path, model=model)


def test_checkpoint_rejects_arbitrary_pickle_globals(tmp_path: Path) -> None:
    """Importing weights never enables unrestricted pickle loading."""
    model = port_config("exp006").model.make()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "untrusted": Path("forbidden")}, path)
    before = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError):
        _read_eval_state(path, model=model)
    assert set(torch.serialization.get_safe_globals()) == before


def test_checkpoint_step_supports_model_only_with_no_ema(tmp_path: Path) -> None:
    """A model-only checkpoint retains live weights with averaging disabled."""
    config = CheckpointActTrainStep.Config().update(port_config("exp006"))
    config.ema = NoEMA.Config()
    path = tmp_path / "checkpoint.pt"
    expected = config.model.make().state_dict()
    torch.save({"model": expected}, path)
    config.checkpoint = LocalFile.Config(path=path)
    loaded = config.make()
    with loaded._eval_weights():
        _assert_state_equal(loaded.model.state_dict(), expected=expected)


def test_checkpoint_step_requires_a_path_before_building_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an omitted checkpoint before allocating the model and optimizer."""
    config = CheckpointActTrainStep.Config().update(port_config("exp006"))
    monkeypatch.setattr(PooledActTrainStep, "__init__", _fail_model_build)
    with pytest.raises(ValueError, match="LocalFile needs a path"):
        config.make()


@pytest.mark.parametrize(
    ("parent_factory", "factory"),
    [
        (experiments.exp006, experiments.eval006),
        (experiments.exp009, experiments.eval009),
        (experiments.exp010, experiments.eval010),
    ],
)
def test_evaluation_recipes_reuse_training_architecture(
    parent_factory: Callable[
        [],
        experiments.Arc1TrainLoop | experiments.CharmTrainLoop.Config,
    ],
    factory: Callable[
        [],
        experiments.Arc1TrainLoop | experiments.CharmTrainLoop.Config,
    ],
) -> None:
    """Checkpoint evaluation preserves each rung's model, loader, and metric tree."""
    parent = parent_factory().copy_tree().finalize()
    config = factory().copy_tree().finalize()
    assert config.eval_only
    assert config.checkpointer is None
    assert isinstance(config.step, CheckpointActTrainStep.Config)
    assert config.step.ema == parent.step.ema
    assert config.step.ema_initialization is None
    assert config.step.model == parent.step.model
    assert config.dataset == parent.dataset
    assert config.metrics_eval == parent.metrics_eval
    checkpoint = config.step.checkpoint
    assert isinstance(checkpoint, LocalFile.Config)
    assert checkpoint.path == ""
    assert checkpoint.sha256 == ""


@pytest.mark.parametrize(
    "factory",
    [experiments.eval006, experiments.eval009, experiments.eval010],
)
def test_evaluation_recipes_accept_checkpoint_cli_overrides(
    factory: Callable[
        [],
        experiments.Arc1TrainLoop | experiments.CharmTrainLoop.Config,
    ],
    tmp_path: Path,
) -> None:
    """Apply checkpoint CLI overrides before finalizing an evaluation recipe.

    Args:
      factory: Native evaluation recipe under test.
      tmp_path: Isolated directory for the caller's checkpoint path.

    """
    config = factory()
    path = tmp_path / "archive.pt"
    digest = "a" * 64

    apply_overrides(
        config,
        overrides=[f"step.checkpoint.path={path}", f"step.checkpoint.sha256={digest}"],
    )

    config = config.copy_tree().finalize()
    assert isinstance(config.step, CheckpointActTrainStep.Config)
    checkpoint = config.step.checkpoint
    assert isinstance(checkpoint, LocalFile.Config)
    assert checkpoint.path == str(path)
    assert checkpoint.sha256 == digest


@pytest.mark.parametrize("preallowed", [False, True])
def test_checkpoint_accepts_observed_numpy_rng_without_global_allowlist(
    tmp_path: Path,
    preallowed: bool,
) -> None:
    """Released training envelopes carry NumPy RNG that is ignored after safe reading."""
    model = port_config("exp006").model.make()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "numpy_rng_state": (
                "MT19937",
                np.arange(624, dtype=np.uint32),
                624,
                0,
                0.0,
            ),
        },
        path,
    )
    with torch.serialization.safe_globals([np.ndarray] if preallowed else []):
        before = set(torch.serialization.get_safe_globals())
        actual = _read_eval_state(path, model=model)
        _assert_state_equal(actual, expected=model.state_dict())
        assert set(torch.serialization.get_safe_globals()) == before


@pytest.mark.parametrize("compiled", [False, True])
def test_released_checkpoint_maps_source_minted_tensors(
    tmp_path: Path,
    compiled: bool,
) -> None:
    """Frozen source weights load completely under both released wrapper names."""
    golden = read_tensors(_CWD / "testdata/checkpoint.pt")
    live = {
        name.removeprefix("live/"): value
        for name, value in golden.items()
        if name.startswith("live/")
    }
    shadow = {
        name.removeprefix("ema/"): value
        for name, value in golden.items()
        if name.startswith("ema/")
    }
    if not compiled:
        live = {name.removeprefix("_orig_mod."): value for name, value in live.items()}
        shadow = {
            name.removeprefix("_orig_mod."): value for name, value in shadow.items()
        }
    path = tmp_path / "release.pt"
    torch.save({"step": 3, "model_state_dict": live, "ema_state": shadow}, path)
    model = _released_model_config().make()
    actual = _read_eval_state(path, model=model)
    model.load_state_dict(actual, strict=True)
    names = {
        "embed_tokens.embedding_weight": "embedding.embed_tokens.weight",
        "lm_head.weight": "head.weight",
        "q_head.weight": "halt_head.weight",
        "q_head.bias": "halt_head.bias",
        "init_hidden": "slow_init",
        "layers.0.self_attn.qkv_proj.weight": "reasoning.0.attn.proj_qkv.weight",
        "layers.0.self_attn.o_proj.weight": "reasoning.0.attn.proj_out.weight",
        "layers.0.mlp.gate_up_proj.weight": "reasoning.0.ffn.up_proj.weight",
        "layers.0.mlp.dwconv.weight": "reasoning.0.ffn.middle.conv.weight",
        "layers.0.mlp.dwconv.bias": "reasoning.0.ffn.middle.conv.bias",
        "layers.0.mlp.down_proj.weight": "reasoning.0.ffn.down_proj.weight",
        "puzzle_emb.puzzle_embed.weight": "prefix.composition.category_embed.weight",
        "puzzle_emb.dih_embed.weight": "prefix.composition.augmentation_embed.weight",
        "puzzle_emb.colorperm_base": "prefix.composition.slot_base",
        "puzzle_emb.colorperm_slots": "prefix.composition.slot_embeddings",
        "puzzle_emb.film_gamma.weight": "prefix.composition.film_gamma.weight",
        "puzzle_emb.film_gamma.bias": "prefix.composition.film_gamma.bias",
        "puzzle_emb.film_beta.weight": "prefix.composition.film_beta.weight",
        "puzzle_emb.film_beta.bias": "prefix.composition.film_beta.bias",
        "puzzle_emb.instance_residual_embed.weights": "prefix.residual.weights",
        "puzzle_emb.instance_residual_up_proj.weight": "prefix.residual_projection.weight",
        "puzzle_emb.interaction_gate.weight": "prefix.residual_gate.linear.weight",
        "puzzle_emb.interaction_gate.bias": "prefix.residual_gate.linear.bias",
    }
    wrapper = "_orig_mod.model.inner." if compiled else "model.inner."
    assert live.keys() == {wrapper + name for name in names}
    assert actual.keys() == set(names.values()) == model.state_dict().keys()
    for source, target in names.items():
        key = wrapper + source
        assert torch.equal(actual[target], shadow.get(key, live[key])), target
    config = CheckpointActTrainStep.Config().update(experiments.eval009().step)
    config.parallelism = NoParallel.Config(device="cpu")
    config.model = _released_model_config()
    config.pool.batch_size = 2
    config.checkpoint = LocalFile.Config(path=path)
    loaded = config.make()
    assert loaded.global_step == 3
    with loaded._eval_weights():
        _assert_state_equal(loaded.model.state_dict(), expected=actual)


def _fail_model_build(
    self: PooledActTrainStep,
    config: PooledActTrainStep.Config,
) -> None:
    del self, config
    raise AssertionError("Checkpoint validation must precede model construction.")


def _read_eval_state(path: Path, *, model: GridSolver) -> dict[str, Tensor]:
    ema = EMA.Config(shadow_kind="param_dict").make()
    state = read_step_state(path, model=model, ema=ema)
    model.load_state_dict(from_plain(state["model"], dict[str, torch.Tensor]))
    if "ema" in state:
        shadow = from_plain(state["ema"], dict[str, object])
        if "shadow_model" in shadow:
            ema = EMA.Config().make()
        ema.load_state_dict(shadow)
    with ema.apply_to(model):
        return {name: value.clone() for name, value in model.state_dict().items()}


def _released_model_config() -> SingleLatentGridSolver.Config:
    # Geometry is fixed by the immutable source-minted checkpoint fixture.
    config = SingleLatentGridSolver.Config()
    config.channels_in = 12
    config.num_layers = 1
    config.vocab_size = 7
    assert isinstance(config.embedding, GridEmbedding.Config)
    config.embedding.grid_shape = (2, 3)
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, Attention.Config)
    assert isinstance(config.block.ffn, SwiGLU.Config)
    config.block.attn.num_heads = 3
    config.block.attn.channels_head = 4
    config.block.ffn.channels_hidden = 9
    prefix = config.prefix
    assert isinstance(prefix, ComposedSparsePrefix.Config)
    prefix.channels_out = 12
    prefix.num_tokens = 16
    assert isinstance(prefix.composition, CategorySlotFiLM.Config)
    prefix.composition.num_categories = 3
    prefix.composition.num_augmentations = 8
    prefix.composition.num_slots = 9
    prefix.residual.num_puzzles = 5
    prefix.residual.num_tokens = 1
    prefix.residual.channels_in = 4
    prefix.residual.channels_out = 4
    prefix.residual.batch_size = 2
    prefix.residual.dtype = torch.bfloat16
    prefix.residual.dtype_scale = torch.bfloat16
    prefix.residual.embed_scale = 1.0
    prefix.residual_projection.channels_in = 4
    prefix.residual_projection.channels_out = 12
    return config


def _assert_state_equal(
    actual: Mapping[str, Tensor],
    *,
    expected: Mapping[str, Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
