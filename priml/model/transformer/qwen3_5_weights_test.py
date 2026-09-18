"""Strict text checkpoint mapping and local loading tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, override
from unittest.mock import patch

import json

from configgle import Fig, Makes

import pytest
import torch

from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import has_weight
from priml.model.linear import Linear
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35
from priml.model.transformer.qwen3_5_weights import remap_hf_state_dict
from priml.testing.qwen3_5 import hf_config


pytest.importorskip("transformers")


if TYPE_CHECKING:
    from pathlib import Path

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
else:
    from wrapt import lazy_import

    Qwen3_5TextConfig = lazy_import(
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "Qwen3_5TextConfig",
    )
    Qwen3_5ForCausalLM = lazy_import(
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5ForCausalLM",
    )


def test_reference_weights_load_strictly() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    native = Qwen35.Config.from_hf(config)
    mapped = remap_hf_state_dict(reference.state_dict(), native)
    model = native.make()
    model.load_state_dict(mapped, strict=True)
    assert has_weight(model.proj_out)
    assert torch.equal(model.proj_out.weight, reference.lm_head.weight)


def test_missing_head_is_not_discarded() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    del state["lm_head.weight"]
    with pytest.raises(ValueError, match="lm_head"):
        remap_hf_state_dict(state, Qwen35.Config.from_hf(config), non_text="discard")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "Missing"),
        ("extra", "Unexpected"),
        ("shape", "has shape"),
        ("integer", "floating-point"),
    ],
)
def test_rejects_incomplete_or_malformed_text_weights(
    mutation: str,
    match: str,
) -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    if mutation == "missing":
        del state[name]
    elif mutation == "extra":
        state["model.unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[name] = state[name][1:]
    else:
        state[name] = state[name].to(torch.int32)
    with pytest.raises(ValueError, match=match):
        remap_hf_state_dict(state, Qwen35.Config.from_hf(config))


def test_conditional_namespace_and_explicit_nontext_policy() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = {
        name.replace("model.", "model.language_model.", 1): tensor
        for name, tensor in reference.state_dict().items()
    }
    state["model.visual.patch_embed.proj.weight"] = torch.zeros(2, 2)
    native_config = Qwen35.Config.from_hf(config)
    with pytest.raises(ValueError, match="Unexpected"):
        remap_hf_state_dict(state, native_config)
    mapped = remap_hf_state_dict(state, native_config, non_text="discard")
    assert torch.equal(mapped["proj_out.weight"], state["lm_head.weight"])
    state["model.language_model.extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="extra"):
        remap_hf_state_dict(state, native_config, non_text="discard")


def test_local_text_loading_preserves_parameters_and_logits(tmp_path: Path) -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(reference.state_dict(), tmp_path / "pytorch_model.bin")
    native_config = Qwen35.Config.from_hf(config)
    expected = native_config.make()
    expected.load_state_dict(remap_hf_state_dict(reference.state_dict(), native_config))
    loaded = Qwen35.load(tmp_path)
    torch.testing.assert_close(
        loaded.state_dict(),
        expected.state_dict(),
        rtol=0,
        atol=0,
    )
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.equal(loaded(tokens), expected(tokens))
    assert torch.equal(loaded.hidden_states(tokens), expected.hidden_states(tokens))


def test_rejects_an_unknown_non_text_policy() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    with pytest.raises(ValueError, match="Unsupported non_text policy"):
        remap_hf_state_dict(
            reference.state_dict(),
            Qwen35.Config.from_hf(config),
            non_text="keep",
        )


@pytest.mark.parametrize("mutation", ["none", "both"])
def test_rejects_zero_or_two_text_embedding_namespaces(mutation: str) -> None:
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    if mutation == "none":
        del state["model.embed_tokens.weight"]
    else:
        state["embed_tokens.weight"] = state["model.embed_tokens.weight"]
    with pytest.raises(ValueError, match="exactly one text embedding namespace"):
        remap_hf_state_dict(state, Qwen35.Config.from_hf(config))


def test_rejects_a_config_that_builds_no_module() -> None:
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    with (
        patch.object(Qwen35.Config, "make", return_value=object()),
        pytest.raises(TypeError, match=r"must build an nn\.Module"),
    ):
        remap_hf_state_dict(state, native_config)


def test_rejects_an_injected_block_that_is_not_a_transformer_block() -> None:
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    assert isinstance(native_config.block, list)
    native_config.block[0] = _CustomBlock.Config()
    with pytest.raises(TypeError, match="native TransformerBlock"):
        remap_hf_state_dict(state, native_config)


def test_rejects_a_top_level_parameter_without_an_hf_counterpart() -> None:
    """A head bias exists in no Qwen3.5 checkpoint, so the mapping refuses it."""
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    assert isinstance(native_config.proj_out, Linear.Config)
    native_config.proj_out.bias = True
    with pytest.raises(
        ValueError,
        match=r"Unsupported native checkpoint parameter: proj_out\.bias",
    ):
        remap_hf_state_dict(state, native_config)


def test_rejects_an_injected_attention_without_an_hf_counterpart() -> None:
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    assert isinstance(native_config.block, list)
    block = native_config.block[1]
    assert isinstance(block, TransformerBlock.Config)
    attention = SelfAttention.Config()
    attention.num_heads = 2
    attention.num_heads_kv = 1
    attention.channels_head = 8
    block.attn = attention
    with pytest.raises(
        ValueError,
        match=r"Unsupported native checkpoint parameter: blocks\.1\.attn",
    ):
        remap_hf_state_dict(state, native_config)


def test_an_attention_parameter_outside_the_projections_maps_by_name() -> None:
    """A gated-attention parameter with no projection prefix keeps its own name."""
    config = hf_config()
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    assert isinstance(native_config.block, list)
    block = native_config.block[1]
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, GatedSelfAttention.Config)
    extra = _ExtraGatedAttention.Config()
    extra.update(block.attn)
    block.attn = extra
    with pytest.raises(
        ValueError,
        match=r"Missing checkpoint weight: model\.layers\.1\.self_attn\.extra",
    ):
        remap_hf_state_dict(state, native_config)
    state["model.layers.1.self_attn.extra"] = torch.full((1,), 7.0)
    mapped = remap_hf_state_dict(state, native_config)
    assert torch.equal(mapped["blocks.1.attn.extra"], torch.full((1,), 7.0))


class _ExtraGatedAttention(GatedSelfAttention):
    """Gated attention carrying one parameter outside the projection map."""

    class Config(Makes["_ExtraGatedAttention"], GatedSelfAttention.Config):
        """Same fields as the parent; only the built module differs."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.extra = torch.nn.Parameter(torch.zeros(1))


def test_tied_head_on_a_different_device_is_rejected() -> None:
    config = hf_config()
    config["tie_word_embeddings"] = True
    state = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config)).state_dict()
    native_config = Qwen35.Config.from_hf(config)
    with patch("torch.equal", side_effect=AssertionError("compared across devices")):
        head = state["lm_head.weight"]
        state["lm_head.weight"] = _OtherDevice(head)
        with pytest.raises(ValueError, match="device"):
            remap_hf_state_dict(state, native_config)


class _OtherDevice(torch.Tensor):
    """A CPU tensor reporting a different device, for the tied-head check."""

    device = torch.device("meta")


class _CustomBlock(torch.nn.Module):
    """A Configgle-injectable block with a parameter but no HF layout."""

    class Config(Fig["_CustomBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.ffn = torch.nn.Linear(16, 16, bias=False)

    def reset_parameters(self) -> None:
        """Leave the test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return self.ffn(x)


def test_tied_head_alias_must_match_the_embedding() -> None:
    config = hf_config()
    config["tie_word_embeddings"] = True
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    native_config = Qwen35.Config.from_hf(config)
    mapped = remap_hf_state_dict(state, native_config)
    native_config.make().load_state_dict(mapped, strict=True)
    assert "proj_out.weight" not in mapped
    state["lm_head.weight"] = state["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="Tied"):
        remap_hf_state_dict(state, native_config)
    state["lm_head.weight"] = state["model.embed_tokens.weight"].double()
    with pytest.raises(ValueError, match="dtype"):
        remap_hf_state_dict(state, native_config)
    del state["lm_head.weight"]
    native_config.make().load_state_dict(remap_hf_state_dict(state, native_config))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
