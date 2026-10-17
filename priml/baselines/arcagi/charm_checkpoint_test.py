"""Strict source-state conversion tests for CHARM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from priml.baselines.arcagi.charm_checkpoint import remap_charm_checkpoint
from priml.model.attention.attention import Attention
from priml.model.conditioning import CategorySlotFiLM
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import SingleLatentGridSolver
from priml.model.prefix import ComposedSparsePrefix
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Mapping


_CWD: Final = Path(__file__).resolve().parent
_SOURCE_TO_NATIVE: Final = {
    "_orig_mod.model.inner.embed_tokens.embedding_weight": "embedding.embed_tokens.weight",
    "_orig_mod.model.inner.lm_head.weight": "head.weight",
    "_orig_mod.model.inner.q_head.weight": "halt_head.weight",
    "_orig_mod.model.inner.q_head.bias": "halt_head.bias",
    "_orig_mod.model.inner.init_hidden": "slow_init",
    "_orig_mod.model.inner.layers.0.self_attn.qkv_proj.weight": (
        "reasoning.0.attn.proj_qkv.weight"
    ),
    "_orig_mod.model.inner.layers.0.self_attn.o_proj.weight": (
        "reasoning.0.attn.proj_out.weight"
    ),
    "_orig_mod.model.inner.layers.0.mlp.gate_up_proj.weight": (
        "reasoning.0.ffn.up_proj.weight"
    ),
    "_orig_mod.model.inner.layers.0.mlp.dwconv.weight": "reasoning.0.ffn.middle.conv.weight",
    "_orig_mod.model.inner.layers.0.mlp.dwconv.bias": "reasoning.0.ffn.middle.conv.bias",
    "_orig_mod.model.inner.layers.0.mlp.down_proj.weight": (
        "reasoning.0.ffn.down_proj.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.puzzle_embed.weight": (
        "prefix.composition.category_embed.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.dih_embed.weight": (
        "prefix.composition.augmentation_embed.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.colorperm_base": ("prefix.composition.slot_base"),
    "_orig_mod.model.inner.puzzle_emb.colorperm_slots": (
        "prefix.composition.slot_embeddings"
    ),
    "_orig_mod.model.inner.puzzle_emb.film_gamma.weight": (
        "prefix.composition.film_gamma.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.film_gamma.bias": (
        "prefix.composition.film_gamma.bias"
    ),
    "_orig_mod.model.inner.puzzle_emb.film_beta.weight": (
        "prefix.composition.film_beta.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.film_beta.bias": (
        "prefix.composition.film_beta.bias"
    ),
    "_orig_mod.model.inner.puzzle_emb.instance_residual_embed.weights": (
        "prefix.residual.weights"
    ),
    "_orig_mod.model.inner.puzzle_emb.instance_residual_up_proj.weight": (
        "prefix.residual_projection.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.interaction_gate.weight": (
        "prefix.residual_gate.linear.weight"
    ),
    "_orig_mod.model.inner.puzzle_emb.interaction_gate.bias": (
        "prefix.residual_gate.linear.bias"
    ),
}


def test_source_minted_compiled_state_and_ema_map_every_tensor() -> None:
    """A source-minted compiled wrapper pins every live and EMA tensor name."""
    golden = read_tensors(_CWD / "testdata/charm_checkpoint.pt")
    live = {
        key.removeprefix("live/"): value
        for key, value in golden.items()
        if key.startswith("live/")
    }
    ema = {
        key.removeprefix("ema/"): value
        for key, value in golden.items()
        if key.startswith("ema/")
    }
    model = _model(with_prefix=True)

    assert live.keys() == _SOURCE_TO_NATIVE.keys()
    assert ema.keys() == _SOURCE_TO_NATIVE.keys() - {
        "_orig_mod.model.inner.init_hidden",
        "_orig_mod.model.inner.puzzle_emb.instance_residual_embed.weights",
    }
    mapped = remap_charm_checkpoint(live, model=model, ema_state=ema)

    assert mapped.keys() == model.state_dict().keys()
    assert set(_SOURCE_TO_NATIVE.values()) == set(model.state_dict())
    for source_key, native_key in _SOURCE_TO_NATIVE.items():
        expected = ema[source_key] if source_key in ema else live[source_key]
        assert torch.equal(mapped[native_key], expected), source_key
    assert mapped["reasoning.0.attn.proj_qkv.weight"].ndim == 2
    assert mapped["prefix.composition.slot_base"].shape == (1, 3)
    assert mapped["prefix.composition.slot_embeddings"].shape == (1, 9, 1)


def test_source_wrappers_remap_every_tensor_strictly() -> None:
    """Released _orig_mod/model/inner prefixes map with exact shapes."""
    model = _model()
    source = _source_state(model)

    actual = remap_charm_checkpoint(source, model=model)

    assert actual.keys() == model.state_dict().keys()
    for name, expected in model.state_dict().items():
        assert torch.equal(actual[name], expected)


def test_source_cose_parameters_and_live_sparse_table_map_completely() -> None:
    """CoSE source names and sparse buffers survive the compiled wrapper map."""
    model = _model(with_prefix=True)
    source = _source_state(model)

    actual = remap_charm_checkpoint(source, model=model)

    assert actual.keys() == model.state_dict().keys()
    assert actual["prefix.composition.slot_base"].shape[0] == 1
    assert actual["prefix.composition.slot_embeddings"].shape[0] == 1
    assert torch.equal(
        actual["prefix.residual.weights"],
        model.state_dict()["prefix.residual.weights"],
    )


def test_duplicate_source_aliases_are_rejected() -> None:
    """Two wrapper aliases cannot silently overwrite one native tensor."""
    model = _model()
    source = _source_state(model)
    source["model.inner.embed_tokens.embedding_weight"] = source[
        "_orig_mod.model.inner.embed_tokens.embedding_weight"
    ]

    with pytest.raises(ValueError, match="duplicate"):
        remap_charm_checkpoint(source, model=model)


def test_unrecognized_source_wrapper_is_rejected() -> None:
    """Only the observed compiled and source wrappers are accepted."""
    model = _model()
    source = _source_state(model)
    first_key = next(iter(source))
    source["unexpected." + first_key.removeprefix("_orig_mod.model.inner.")] = (
        source.pop(first_key)
    )

    with pytest.raises(ValueError, match="unexpected source model key"):
        remap_charm_checkpoint(source, model=model)


@pytest.mark.parametrize(
    "key",
    [
        "_orig_mod.model.inner.layers.99.self_attn.qkv_proj.weight",
        "_orig_mod.model.inner.layers.0.norm1.weight",
        "_orig_mod.model.inner.puzzle_emb.unknown.weight",
        "_orig_mod.model.inner.puzzle_emb.instance_residual_up_proj.bias",
    ],
)
def test_known_wrapper_with_unknown_source_tensor_is_rejected(key: str) -> None:
    """A valid source wrapper cannot smuggle unknown model keys into state."""
    model = _model(with_prefix=key.endswith("unknown.weight"))
    source = _source_state(model)
    source[key] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="unexpected"):
        remap_charm_checkpoint(source, model=model)


@pytest.mark.parametrize(
    ("child", "known_suffix"),
    [("self_attn", "qkv_proj.weight"), ("mlp", "down_proj.weight")],
)
def test_unknown_block_projection_is_rejected_at_mapping_boundary(
    child: str,
    known_suffix: str,
) -> None:
    """Block children accept only their declared projection suffixes."""
    model = _model()
    source = _source_state(model)
    known_key = f"_orig_mod.model.inner.layers.0.{child}.{known_suffix}"
    unknown_key = f"_orig_mod.model.inner.layers.0.{child}.unrecognized.weight"
    source[unknown_key] = source.pop(known_key)

    with pytest.raises(
        ValueError,
        match=rf"^unexpected source model key .*.{child}\.unrecognized\.weight",
    ):
        remap_charm_checkpoint(source, model=model)


def test_mlp_down_projection_key_passes_through_unchanged() -> None:
    """The released MLP down projection already uses the native suffix."""
    model = _model()
    source = _source_state(model)
    source_key = "_orig_mod.model.inner.layers.0.mlp.down_proj.weight"

    mapped = remap_charm_checkpoint(source, model=model)

    assert torch.equal(
        mapped["reasoning.0.ffn.down_proj.weight"],
        source[source_key],
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("shape", "shape"),
    ],
)
def test_source_state_rejects_missing_extra_or_wrong_shape(
    edit: str,
    message: str,
) -> None:
    """No strict=False behavior hides malformed checkpoint tensors."""
    model = _model()
    source = _source_state(model)
    key = next(iter(source))
    if edit == "missing":
        del source[key]
    elif edit == "unexpected":
        source["model.inner.unused.weight"] = torch.zeros(2, 3)
    else:
        source[key] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match=message):
        remap_charm_checkpoint(source, model=model)


def test_ema_replaces_dense_parameters_and_keeps_live_sparse_buffer() -> None:
    """EMA source excludes sparse buffers, which remain from the live state."""
    model = _model(with_prefix=True)
    live = _source_state(model)
    parameters: Mapping[str, torch.Tensor] = dict(model.named_parameters())
    ema = {
        source_name: torch.full_like(value, 3)
        for name, value in parameters.items()
        for source_name in [_source_key(name)]
    }

    actual = remap_charm_checkpoint(live, model=model, ema_state=ema)

    for name, value in parameters.items():
        assert torch.equal(actual[name], torch.full_like(value, 3))
    assert torch.equal(actual["slow_init"], model.state_dict()["slow_init"])
    assert torch.equal(
        actual["prefix.residual.weights"],
        model.state_dict()["prefix.residual.weights"],
    )


def test_absent_ema_overlay_differs_from_incomplete_empty_ema() -> None:
    """No EMA overlay is valid, while an explicitly empty EMA is malformed."""
    model = _model()
    live = _source_state(model)

    actual = remap_charm_checkpoint(live, model=model, ema_state=None)

    assert actual.keys() == model.state_dict().keys()
    with pytest.raises(ValueError, match="EMA checkpoint is missing keys"):
        remap_charm_checkpoint(live, model=model, ema_state={})


def _model(*, with_prefix: bool = False) -> SingleLatentGridSolver:
    config = SingleLatentGridSolver.Config()
    config.channels_in = 12
    config.num_layers = 1
    config.vocab_size = 7
    embedding = cast(GridEmbedding.Config, config.embedding)
    embedding.grid_shape = (2, 3)
    block = cast(TransformerBlock.Config, config.block)
    attention = cast(Attention.Config, block.attn)
    ffn = cast(SwiGLU.Config, block.ffn)
    attention.num_heads = 3
    attention.channels_head = 4
    ffn.channels_hidden = 9
    if not with_prefix:
        config.prefix = None
    else:
        prefix = cast(ComposedSparsePrefix.Config, config.prefix)
        prefix.channels_out = 12
        prefix.num_tokens = 16
        composition = cast(CategorySlotFiLM.Config, prefix.composition)
        composition.num_categories = 3
        composition.num_augmentations = 8
        composition.num_slots = 9
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
    return config.make()


def _source_state(model: SingleLatentGridSolver) -> dict[str, torch.Tensor]:
    return {
        _source_key(name): value.clone() for name, value in model.state_dict().items()
    }


def _source_key(target_key: str) -> str:
    if target_key == "embedding.embed_tokens.weight":
        key = "embed_tokens.embedding_weight"
    elif target_key == "head.weight":
        key = "lm_head.weight"
    elif target_key.startswith("halt_head."):
        key = "q_head." + target_key.removeprefix("halt_head.")
    elif target_key == "slow_init":
        key = "init_hidden"
    elif target_key.startswith("prefix.composition."):
        key = target_key.removeprefix("prefix.composition.")
        key = "puzzle_emb." + key
    elif target_key == "prefix.residual.weights":
        key = "puzzle_emb.instance_residual_embed.weights"
    elif target_key.startswith("prefix.residual_projection."):
        key = target_key.removeprefix("prefix.residual_projection.")
        key = "puzzle_emb.instance_residual_up_proj." + key
    elif target_key.startswith("prefix.residual_gate.linear."):
        key = target_key.removeprefix("prefix.residual_gate.linear.")
        key = "puzzle_emb.interaction_gate." + key
    elif target_key.startswith("reasoning."):
        key = target_key.removeprefix("reasoning.")
        key = "layers." + key
        key = key.replace(".attn.", ".self_attn.")
        key = key.replace(".ffn.", ".mlp.")
        key = key.replace(".self_attn.proj_qkv.", ".self_attn.qkv_proj.")
        key = key.replace(".self_attn.proj_out.", ".self_attn.o_proj.")
        key = key.replace(".mlp.up_proj.", ".mlp.gate_up_proj.")
        key = key.replace(".mlp.middle.conv.", ".mlp.dwconv.")
    else:
        raise AssertionError(target_key)
    return "_orig_mod.model.inner." + key


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
