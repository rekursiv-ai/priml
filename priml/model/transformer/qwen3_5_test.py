"""CPU-tiny tests for the native Qwen3.5 hybrid text language model."""

from __future__ import annotations

from typing import TYPE_CHECKING, override
from unittest.mock import patch

import dataclasses
import json
import re

from configgle import Fig

import pytest
import torch

from priml.lib.custom_json import ReadError
from priml.model.attention.attention import Attention
from priml.model.attention.gated_attention import GatedAttention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, YarnScaling
from priml.model.custom_types import DepthIndex, LayerCache
from priml.model.generate import generate
from priml.model.norm import CenteredRMSNorm
from priml.model.special import TiedLinear
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import (
    Qwen35,
    _full_attention_mask,
    _layer_types,
)
from priml.model.transformer.qwen3_5_weights import _sources
from priml.testing.golden import assert_pprint_golden
from priml.testing.qwen3_5 import hf_config


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


def _prepared_causal_mask(*, batch: int, queries: int, keys: int) -> torch.Tensor:
    """Return a text-only additive mask, one per attention head, hiding future keys."""
    causal = torch.arange(keys) <= (
        torch.arange(queries).unsqueeze(-1) + keys - queries
    )
    heads = hf_config()["num_attention_heads"]
    assert isinstance(heads, int)
    return torch.zeros(batch, heads, queries, keys).masked_fill(
        ~causal,
        torch.finfo(torch.float32).min,
    )


def test_qwen35_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="qwen3_5",
        config=Qwen35.Config.from_hf(hf_config()),
    )


def test_hybrid_config_preserves_injected_blocks() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    assert len(config.block) == 2
    assert isinstance(config.block[0], TransformerBlock.Config)
    config.block[0].checkpoint = True
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.block, list)
    assert isinstance(finalized.block[0], TransformerBlock.Config)
    assert finalized.block[0].checkpoint
    model = config.make()
    assert model(torch.arange(15).reshape(3, 5)).shape == (3, 5, 32)
    assert model.hidden_states(torch.arange(15).reshape(3, 5)).shape == (3, 5, 16)


def test_projecting_hidden_states_reproduces_forward() -> None:
    """``hidden_states`` is pre-norm; ``project_to_logits`` norms it exactly once."""
    model = Qwen35.Config.from_hf(hf_config()).make().eval()
    assert isinstance(model.norm, CenteredRMSNorm)
    with torch.no_grad():
        # Zero-centered weight: a non-identity scale makes a doubled norm visible.
        model.norm.weight.fill_(1.0)
    tokens = torch.arange(15).reshape(3, 5)
    with torch.no_grad():
        hidden = model.hidden_states(tokens)
        assert torch.equal(model.project_to_logits(hidden), model(tokens))
        assert not torch.equal(model.norm(hidden), hidden)


def test_hidden_states_rejects_token_ids_without_input_embedding() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    config.proj_in = None
    model = config.make()

    with pytest.raises(
        ValueError,
        match=re.escape("Token IDs require an input embedding."),
    ) as error:
        model.hidden_states(torch.arange(15).reshape(3, 5))
    assert str(error.value) == "Token IDs require an input embedding."


def test_hidden_states_raises_on_a_cache_missing_a_layer_slot() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(KeyError):
        model.hidden_states(torch.arange(15).reshape(3, 5), cache={})


def test_hidden_states_rejects_positions_and_position_ids_together() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(
        ValueError,
        match=re.escape("Pass either positions or position_ids, not both."),
    ) as error:
        model(
            torch.arange(15).reshape(3, 5),
            positions=torch.arange(5),
            position_ids=torch.arange(15).reshape(3, 5),
        )
    assert str(error.value) == "Pass either positions or position_ids, not both."


def test_forward_rejects_a_cache_that_is_not_a_layer_cache() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(
        TypeError,
        match=re.escape("cache must satisfy LayerCache or be None."),
    ):
        model(torch.arange(15).reshape(3, 5), cache=7)


def test_alloc_cache_keys_one_slot_per_cached_attention_by_depth_index() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    cache = model.alloc_cache(batch=3, max_seq=5)

    assert sorted(cache) == [((0, 2),), ((1, 2),)]


def test_hidden_states_rejects_an_attention_mask_that_is_neither_2d_nor_4d() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(
        ValueError,
        match=re.escape("attention_mask must be a 2-D padding mask or 4-D mask."),
    ) as error:
        model(torch.arange(15).reshape(3, 5), attention_mask=torch.zeros(3, 5, 4))
    assert str(error.value) == "attention_mask must be a 2-D padding mask or 4-D mask."


def test_final_norm_width_inference_respects_explicit_configuration() -> None:
    """Final norms infer the model width without rewriting explicit configurations."""
    inferred = Qwen35.Config.from_hf(hf_config()).copy_tree().finalize()
    assert isinstance(inferred.norm, CenteredRMSNorm.Config)
    assert inferred.norm.channels_in == 16

    matching = Qwen35.Config.from_hf(hf_config())
    matching_norm = CenteredRMSNorm.Config()
    matching_norm.channels_in = 16
    matching.norm = matching_norm
    finalized = matching.copy_tree().finalize()
    assert isinstance(finalized.norm, CenteredRMSNorm.Config)
    assert finalized.norm.channels_in == 16

    incompatible = Qwen35.Config.from_hf(hf_config())
    incompatible_norm = CenteredRMSNorm.Config()
    incompatible_norm.channels_in = 8
    incompatible.norm = incompatible_norm
    with pytest.raises(
        ValueError,
        match=r"norm\.channels_in=8 must equal channels_in=16",
    ):
        incompatible.make()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "model_type",
            "qwen3_5_moe",
            "Expected a dense Qwen3.5 text config, got 'qwen3_5_moe'.",
        ),
        (
            "hidden_act",
            "gelu",
            "Only the SwiGLU/silu Qwen3.5 architecture is supported.",
        ),
        (
            "layer_types",
            ["linear_attention"],
            "layer_types must name one supported attention type per layer.",
        ),
        (
            "layer_types",
            ["linear_attention", "sliding"],
            "layer_types must name one supported attention type per layer.",
        ),
        ("num_attention_heads", 0, "num_attention_heads must be positive."),
        (
            "initializer_range",
            -0.02,
            "initializer_range must be finite and positive.",
        ),
        ("rms_norm_eps", 0.0, "rms_norm_eps must be finite and positive."),
        (
            "num_key_value_heads",
            3,
            "num_attention_heads must be divisible by num_key_value_heads.",
        ),
        (
            "quantization_config",
            {"bits": 4},
            "Quantized checkpoint configurations are unsupported.",
        ),
        ("sliding_window", 128, "Sliding-window attention is unsupported."),
        (
            "attention_dropout",
            -0.1,
            "attention_dropout must be finite and in [0, 1).",
        ),
        (
            "attention_dropout",
            1.0,
            "attention_dropout must be finite and in [0, 1).",
        ),
        (
            "attention_dropout",
            float("nan"),
            "attention_dropout must be finite and in [0, 1).",
        ),
        (
            "rope_scaling",
            {"type": "yarn", "factor": 4.0},
            "Only default text rotary frequencies are supported.",
        ),
        (
            "full_attention_interval",
            0,
            "full_attention_interval must be positive.",
        ),
    ],
)
def test_unsupported_config_is_rejected(
    key: str,
    value: object,
    message: str,
) -> None:
    config = hf_config()
    if key == "full_attention_interval":
        del config["layer_types"]
    if key == "rope_scaling":
        del config["rope_parameters"]
    config[key] = value
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        Qwen35.Config.from_hf(config)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("num_attention_heads", []),
        ("attention_bias", {}),
        ("attention_dropout", True),
        ("partial_rotary_factor", True),
        ("rope_theta", []),
        ("rope_parameters", []),
        ("layer_types", "bad"),
        # A non-string entry is a type error, not a silently dropped layer.
        ("layer_types", ["full_attention", 1]),
        ("full_attention_interval", None),
    ],
)
def test_wrongly_typed_hf_fields_are_rejected(
    key: str,
    value: object,
) -> None:
    config = hf_config()
    if key in {"partial_rotary_factor", "rope_theta"}:
        rope = config["rope_parameters"]
        assert isinstance(rope, dict)
        rope[key] = value
    else:
        if key == "full_attention_interval":
            del config["layer_types"]
        config[key] = value
    with pytest.raises(ReadError):
        Qwen35.Config.from_hf(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "rope_type",
            "yarn",
            "Only default text rotary frequencies are supported.",
        ),
        (
            "partial_rotary_factor",
            -0.1,
            "partial_rotary_factor must be finite and in (0, 1].",
        ),
        (
            "partial_rotary_factor",
            0.0,
            "partial_rotary_factor must be finite and in (0, 1].",
        ),
        (
            "partial_rotary_factor",
            1.5,
            "partial_rotary_factor must be finite and in (0, 1].",
        ),
        (
            "partial_rotary_factor",
            0.1,
            "The rotary prefix must have a positive even width.",
        ),
        (
            "partial_rotary_factor",
            float("nan"),
            "partial_rotary_factor must be finite and in (0, 1].",
        ),
    ],
)
def test_unsupported_rope_parameters_are_rejected(
    key: str,
    value: object,
    message: str,
) -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope[key] = value
    if key == "rope_type":
        rope["factor"] = 4.0
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        Qwen35.Config.from_hf(config)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rope_scaling", {"type": "yarn", "factor": 4.0}),
        ("rope_parameters", {"rope_type": "linear", "factor": 2.0}),
        ("rope_parameters", {"rope_type": "yarn"}),
        ("rope_parameters", {"factor": 4.0}),
    ],
)
def test_rotary_scaling_is_read_as_yarn_reads_it(key: str, value: object) -> None:
    """A scaled rotary is refused exactly when :class:`YarnScaling` finds one."""
    config = hf_config()
    config.pop("rope_parameters")
    config[key] = value
    expected: Exception = ValueError(
        "Only default text rotary frequencies are supported.",
    )
    try:
        _ = YarnScaling.Config.from_hf(config)
    except (ValueError, ReadError) as error:
        expected = error
    with pytest.raises(type(expected), match=re.escape(str(expected))):
        Qwen35.Config.from_hf(config)


@pytest.mark.parametrize(
    "value",
    [{"rope_type": "default", "rope_theta": 1e6}, {"rope_theta": 1e6}],
)
def test_an_unscaled_rotary_is_accepted(value: dict[str, object]) -> None:
    config = hf_config()
    config["rope_parameters"] = value
    assert YarnScaling.Config.from_hf(config) is None
    Qwen35.Config.from_hf(config)


def test_layer_types_default_to_a_full_attention_interval() -> None:
    config = hf_config()
    del config["layer_types"]
    config["num_hidden_layers"] = 5
    config["full_attention_interval"] = 3
    assert _layer_types(config, count=5) == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    ]
    native = Qwen35.Config.from_hf(config)
    assert isinstance(native.block, list)
    attentions = [
        type(block.attn).__qualname__
        for block in native.block
        if isinstance(block, TransformerBlock.Config)
    ]
    assert attentions == [
        "Qwen35GatedDeltaNet.Config",
        "Qwen35GatedDeltaNet.Config",
        "GatedAttention.Config",
        "Qwen35GatedDeltaNet.Config",
        "Qwen35GatedDeltaNet.Config",
    ]


def test_layer_types_use_the_default_interval_when_unspecified() -> None:
    config = hf_config()
    del config["layer_types"]
    config["num_hidden_layers"] = 5

    assert _layer_types(config, count=5) == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    every_layer = {**config, "full_attention_interval": 1}
    assert _layer_types(every_layer, count=5) == ["full_attention"] * 5

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    attentions = [
        type(block.attn).__qualname__
        for block in native.block
        if isinstance(block, TransformerBlock.Config)
    ]
    assert attentions == [
        "Qwen35GatedDeltaNet.Config",
        "Qwen35GatedDeltaNet.Config",
        "Qwen35GatedDeltaNet.Config",
        "GatedAttention.Config",
        "Qwen35GatedDeltaNet.Config",
    ]


def test_conditional_generation_rejects_a_non_object_text_config() -> None:
    with pytest.raises(ReadError):
        Qwen35.Config.from_hf({"model_type": "qwen3_5", "text_config": []})


def test_conditional_generation_text_defaults_and_nested_quantization() -> None:
    nested = hf_config()
    nested.pop("model_type")
    nested.pop("hidden_act")
    native = Qwen35.Config.from_hf({"model_type": "qwen3_5", "text_config": nested})
    assert native.channels_in == 16

    nested["quantization_config"] = {"bits": 4}
    with pytest.raises(
        ValueError,
        match=re.escape("Quantized checkpoint configurations are unsupported."),
    ):
        Qwen35.Config.from_hf({"model_type": "qwen3_5", "text_config": nested})

    outer_quantized = {"model_type": "qwen3_5", "text_config": hf_config()}
    outer_quantized["quantization_config"] = {"bits": 4}
    with pytest.raises(
        ValueError,
        match=re.escape("Quantized checkpoint configurations are unsupported."),
    ):
        Qwen35.Config.from_hf(outer_quantized)


def test_conditional_generation_config_unwraps_its_text_config() -> None:
    """The multimodal wrapper nests the text config and owns the tie flag."""
    wrapped: dict[str, object] = {
        "model_type": "qwen3_5",
        "text_config": hf_config(),
        "tie_word_embeddings": True,
    }
    native = Qwen35.Config.from_hf(wrapped)
    assert isinstance(native.proj_out, TiedLinear.Config)
    assert native.proj_out.tied == "proj_in"
    assert native.channels_out == 32

    inner = hf_config()
    inner["model_type"] = "qwen3_5_text_moe"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The nested text config must have model_type='qwen3_5_text'.",
        ),
    ) as error:
        Qwen35.Config.from_hf({"model_type": "qwen3_5", "text_config": inner})
    assert str(error.value) == (
        "The nested text config must have model_type='qwen3_5_text'."
    )


def test_reset_parameters_reinitializes_the_final_norm_and_the_backbone() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    assert isinstance(model.norm, CenteredRMSNorm)
    with torch.no_grad():
        model.norm.weight.fill_(3.0)
        for parameter in model.blocks.parameters():
            parameter.fill_(3.0)

    model.reset_parameters()

    assert torch.equal(model.norm.weight, torch.zeros_like(model.norm.weight))
    assert all(
        not torch.equal(p, torch.full_like(p, 3.0)) for p in model.blocks.parameters()
    )


def test_hidden_states_rejects_a_non_tensor_message_field() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    with pytest.raises(TypeError, match="positions must be a Tensor or None"):
        model(torch.arange(15).reshape(3, 5), positions=[0, 1, 2, 3, 4])


def test_hidden_states_rejects_an_integer_4d_attention_mask() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    with pytest.raises(
        TypeError,
        match=re.escape("attention_mask must be floating additive when it is 4-D."),
    ) as error:
        model(
            torch.arange(15).reshape(3, 5),
            attention_mask=torch.zeros(3, 2, 5, 4, dtype=torch.long),
        )
    assert (
        str(error.value) == "attention_mask must be floating additive when it is 4-D."
    )


def test_load_rejects_a_non_object_checkpoint_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("[]")

    with pytest.raises(ReadError):
        Qwen35.load(tmp_path)


def test_load_reads_a_local_checkpoint_onto_the_requested_dtype(
    tmp_path: Path,
) -> None:
    pytest.importorskip("transformers")
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(reference.state_dict(), tmp_path / "pytorch_model.bin")

    loaded = Qwen35.load(tmp_path, dtype=torch.bfloat16)

    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert loaded(torch.arange(15).reshape(3, 5)).shape == (3, 5, 32)


@pytest.mark.parametrize(
    ("source_dtype", "requested_dtype", "expected_dtype"),
    [
        (torch.bfloat16, None, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.bfloat16),
    ],
)
def test_local_load_preserves_source_dtype_and_non_text_policy(
    tmp_path: Path,
    source_dtype: torch.dtype,
    requested_dtype: torch.dtype | None,
    expected_dtype: torch.dtype,
) -> None:
    config, state, native_state = _local_checkpoint(source_dtype=source_dtype)
    state["model.visual.extra.weight"] = torch.zeros(2, 3)
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(state, tmp_path / "pytorch_model.bin")

    with pytest.raises(ValueError, match="Unexpected checkpoint weights"):
        Qwen35.load(tmp_path)
    loaded = Qwen35.load(
        tmp_path,
        dtype=requested_dtype,
        non_text="discard",
    )

    assert all(parameter.dtype == expected_dtype for parameter in loaded.parameters())
    assert all(
        torch.equal(loaded.state_dict()[name], value.to(dtype=expected_dtype))
        for name, value in native_state.items()
    )


def test_local_load_keeps_strict_state_dict_coverage(tmp_path: Path) -> None:
    config, state, native_state = _local_checkpoint(source_dtype=torch.float32)
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(state, tmp_path / "pytorch_model.bin")
    mapped = dict(native_state)
    del mapped["proj_out.weight"]

    with (
        patch(
            "priml.model.transformer.qwen3_5.remap_hf_state_dict",
            return_value=mapped,
        ),
        pytest.raises(RuntimeError, match="Missing key"),
    ):
        Qwen35.load(tmp_path)


def test_local_load_applies_requested_device(tmp_path: Path) -> None:
    config, state, _ = _local_checkpoint(source_dtype=torch.float32)
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(state, tmp_path / "pytorch_model.bin")

    with pytest.warns(UserWarning, match="copying from a non-meta parameter"):
        loaded = Qwen35.load(tmp_path, device="meta")

    assert all(parameter.device.type == "meta" for parameter in loaded.parameters())


def _local_checkpoint(
    *,
    source_dtype: torch.dtype,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build a tiny local HF-named checkpoint without the transformers package."""
    config = hf_config()
    native_config = Qwen35.Config.from_hf(config)
    assert isinstance(native_config.block, list)
    blocks: list[TransformerBlock.Config] = []
    for block in native_config.block:
        assert isinstance(block, TransformerBlock.Config)
        blocks.append(block)
    native = native_config.make().to(dtype=source_dtype)
    native_state = native.state_dict()
    state: dict[str, torch.Tensor] = {}
    for target, value in native_state.items():
        sources = _sources(target, prefix="model.", blocks=blocks)
        if len(sources) == 2:
            first, second = value.chunk(2, dim=0)
            state[sources[0]] = first
            state[sources[1]] = second
        else:
            state[sources[0]] = value
    return config, state, native_state


def test_unsupported_delta_head_ratio_is_rejected() -> None:
    config = hf_config()
    config["linear_num_key_heads"] = 2
    config["linear_num_value_heads"] = 3

    with pytest.raises(ValueError, match="linear_num_value_heads"):
        Qwen35.Config.from_hf(config)


def test_positive_fractional_rotary_base_is_accepted() -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope["rope_theta"] = 0.5

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    full_attention = native.block[1]
    assert isinstance(full_attention, TransformerBlock.Config)
    assert isinstance(full_attention.attn, GatedAttention.Config)
    assert isinstance(full_attention.attn.rope, RoPE.Config)
    assert isinstance(
        full_attention.attn.rope.frequencies,
        HuggingFaceFrequencies.Config,
    )
    assert full_attention.attn.rope.frequencies.base == 0.5


@pytest.mark.parametrize("theta", [-1.0, 0.0])
def test_unsupported_rotary_base_is_rejected(theta: float) -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope["rope_theta"] = theta

    with pytest.raises(
        ValueError,
        match=re.escape("rope_theta must be finite and positive."),
    ) as error:
        Qwen35.Config.from_hf(config)
    assert str(error.value) == "rope_theta must be finite and positive."


def test_rotary_defaults_and_top_level_fallbacks_are_applied() -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope.clear()

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    attention = native.block[1]
    assert isinstance(attention, TransformerBlock.Config)
    assert isinstance(attention.attn, GatedAttention.Config)
    assert attention.attn.channels_head == 8
    assert isinstance(attention.attn.rope, RoPE.Config)
    assert attention.attn.rope.channels_head == 2
    assert isinstance(attention.attn.rope.frequencies, HuggingFaceFrequencies.Config)
    assert attention.attn.rope.frequencies.base == 10_000_000.0


def test_optional_hf_fields_use_documented_defaults() -> None:
    config = hf_config()
    for key in (
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "partial_rotary_factor",
        "rope_parameters",
        "rope_theta",
        "rms_norm_eps",
        "tie_word_embeddings",
    ):
        config.pop(key, None)

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.norm, CenteredRMSNorm.Config)
    assert native.norm.eps == 1e-6
    assert not isinstance(native.proj_out, TiedLinear.Config)
    assert isinstance(native.block, list)
    attention_block = native.block[1]
    assert isinstance(attention_block, TransformerBlock.Config)
    assert isinstance(attention_block.attn, GatedAttention.Config)
    assert attention_block.attn.bias is False
    assert attention_block.attn.dropout == 0.0
    assert isinstance(attention_block.attn.rope, RoPE.Config)
    assert attention_block.attn.rope.channels_head == 2
    assert isinstance(
        attention_block.attn.rope.frequencies,
        HuggingFaceFrequencies.Config,
    )
    assert attention_block.attn.rope.frequencies.base == 10_000_000.0


def test_full_attention_fields_are_parsed_and_defaulted() -> None:
    config = hf_config()
    config.pop("attention_bias")
    config.pop("attention_dropout")

    defaulted = Qwen35.Config.from_hf(config)
    assert isinstance(defaulted.block, list)
    default_attention = defaulted.block[1]
    assert isinstance(default_attention, TransformerBlock.Config)
    assert isinstance(default_attention.attn, GatedAttention.Config)
    assert default_attention.attn.bias is False
    assert default_attention.attn.dropout == 0.0

    config["attention_bias"] = True
    config["attention_dropout"] = 0.25
    configured = Qwen35.Config.from_hf(config)
    assert isinstance(configured.block, list)
    custom_attention = configured.block[1]
    assert isinstance(custom_attention, TransformerBlock.Config)
    assert isinstance(custom_attention.attn, GatedAttention.Config)
    assert custom_attention.attn.bias is True
    assert custom_attention.attn.dropout == 0.25


def test_full_rotary_fraction_is_supported() -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope["partial_rotary_factor"] = 1.0

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    attention = native.block[1]
    assert isinstance(attention, TransformerBlock.Config)
    assert isinstance(attention.attn, GatedAttention.Config)
    assert isinstance(attention.attn.rope, RoPE.Config)
    assert attention.attn.rope.channels_head == attention.attn.channels_head


def test_rotary_parameters_prefer_nested_values_over_top_level_fallbacks() -> None:
    config = hf_config()
    config["partial_rotary_factor"] = 1.0
    config["rope_theta"] = 20_000.0
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope["partial_rotary_factor"] = 0.5
    rope["rope_theta"] = 30_000.0

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    attention = native.block[1]
    assert isinstance(attention, TransformerBlock.Config)
    assert isinstance(attention.attn, GatedAttention.Config)
    assert isinstance(attention.attn.rope, RoPE.Config)
    assert attention.attn.rope.channels_head == 4
    assert isinstance(attention.attn.rope.frequencies, HuggingFaceFrequencies.Config)
    assert attention.attn.rope.frequencies.base == 30_000.0


def test_full_attention_uses_top_level_rotary_fallbacks() -> None:
    config = hf_config()
    config["partial_rotary_factor"] = 0.5
    config["rope_theta"] = 30_000.0
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope.clear()

    native = Qwen35.Config.from_hf(config)

    assert isinstance(native.block, list)
    attention_block = native.block[1]
    assert isinstance(attention_block, TransformerBlock.Config)
    assert isinstance(attention_block.attn, GatedAttention.Config)
    assert isinstance(attention_block.attn.rope, RoPE.Config)
    assert attention_block.attn.rope.channels_head == 4
    frequencies = attention_block.attn.rope.frequencies
    assert isinstance(frequencies, HuggingFaceFrequencies.Config)
    assert frequencies.base == 30_000.0


def test_hybrid_forwards_messages_to_injected_attention() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorders: list[_MessageRecorder] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        recorder = _MessageRecorder()
        block.attn = recorder
        recorders.append(recorder)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    position_ids = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])

    model(
        torch.arange(15).reshape(3, 5),
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    expected_positions = position_ids.unsqueeze(-1)
    for recorder in recorders:
        assert len(recorder.messages) == 1
        observed_mask, observed_ids, observed_positions = recorder.messages[0]
        assert torch.equal(observed_mask, attention_mask)
        assert torch.equal(observed_ids, position_ids)
        assert torch.equal(observed_positions, expected_positions)


def test_hybrid_forwards_messages_to_injected_output_head_and_cache() -> None:
    """Preserve output-head messages across direct and cached Qwen calls."""
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorder = _HeadMessageRecorder()
    model.proj_out = recorder
    message = torch.tensor([7])

    tokens = torch.arange(15).reshape(3, 5)
    model(tokens, marker=message)
    cache = model.alloc_cache(batch=3, max_seq=5)
    model(tokens, cache=cache, marker=message)
    model.project_to_logits(torch.ones(3, 5, 16), marker=message)

    assert len(recorder.messages) == 3
    assert recorder.messages[0] is message
    assert recorder.messages[1] is message
    assert recorder.messages[2] is message


def test_prepared_causal_mask_reaches_injected_self_attention_prefill_and_cache() -> (
    None
):
    """A prepared causal 4-D text mask is forwarded through an injected attention."""
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    block = config.block[1]
    assert isinstance(block, TransformerBlock.Config)
    attention = Attention.Config()
    attention.num_heads = 2
    attention.num_heads_kv = 1
    attention.channels_head = 8
    attention.attn_kernel = SdpaNaive.Config()
    block.attn = attention
    model = config.make().eval()

    prompt = torch.arange(15).reshape(3, 5)
    prompt_mask = _prepared_causal_mask(batch=3, queries=5, keys=5)
    assert not torch.equal(
        model(prompt),
        model(prompt, attention_mask=prompt_mask),
    )

    plain_cache = model.alloc_cache(batch=3, max_seq=5)
    masked_cache = model.alloc_cache(batch=3, max_seq=5)
    prefix = prompt[:, :4]
    prefix_mask = _prepared_causal_mask(batch=3, queries=4, keys=4)
    plain_prefix = model(prefix, cache=plain_cache)
    masked_prefix = model(
        prefix,
        cache=masked_cache,
        attention_mask=prefix_mask,
    )
    assert not torch.equal(plain_prefix, masked_prefix)

    continuation_mask = _prepared_causal_mask(batch=3, queries=1, keys=5)
    continuation_mask[..., 0] = torch.finfo(continuation_mask.dtype).min
    continuation = prompt[:, 4:]
    plain_continuation = model(
        continuation,
        cache=plain_cache,
    )
    masked_continuation = model(
        continuation,
        cache=masked_cache,
        attention_mask=continuation_mask,
    )
    assert not torch.equal(plain_continuation, masked_continuation)


def test_padding_mask_builds_a_causal_additive_mask_for_cached_queries() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorders: list[_MaskMessageRecorder] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        recorder = _MaskMessageRecorder()
        block.attn = recorder
        recorders.append(recorder)
    padding = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]])
    tokens = torch.tensor([[1, 2], [3, 4], [5, 6]])

    model.hidden_states(tokens, attention_mask=padding)

    # _full_attention_mask emits [B, 1, Q, K] for broadcasting.
    expected = torch.zeros(3, 1, 2, 4)
    fill = torch.finfo(expected.dtype).min
    expected[0, 0, :, 1] = fill
    expected[1, 0, :, 0] = fill
    expected[1, 0, :, 2] = fill
    expected[2, 0, 1, 3] = fill
    for recorder in recorders:
        assert len(recorder.messages) == 1
        observed_padding, observed_additive = recorder.messages[0]
        assert torch.equal(observed_padding, padding)
        assert torch.equal(observed_additive, expected)


def test_full_attention_mask_uses_input_dtype_and_device() -> None:
    x = torch.zeros(3, 2, 5, dtype=torch.float64)
    padding = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]])

    mask = _full_attention_mask(padding, x=x)

    assert mask is not None
    assert mask.shape == (3, 1, 2, 4)
    assert mask.dtype == torch.float64
    # _full_attention_mask emits [B, 1, Q, K] for broadcasting.
    expected = torch.zeros(3, 1, 2, 4, dtype=torch.float64)
    fill = torch.finfo(torch.float64).min
    expected[0, 0, :, 1] = fill
    expected[1, 0, :, 0] = fill
    expected[1, 0, :, 2] = fill
    expected[2, 0, 1, 3] = fill
    assert torch.equal(mask, expected)

    meta_x = torch.empty(3, 2, 5, device="meta")
    meta_padding = torch.empty(3, 4, dtype=torch.bool)
    meta_mask = _full_attention_mask(meta_padding, x=meta_x)
    assert meta_mask is not None
    assert meta_mask.device.type == "meta"


def test_hybrid_preserves_caller_attn_mask_precedence() -> None:
    """An explicit additive mask wins over a 2-D padding mask derived mask."""
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorders: list[_MaskMessageRecorder] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        recorder = _MaskMessageRecorder()
        block.attn = recorder
        recorders.append(recorder)
    padding = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    # Qwen35.forward passes this full-sequence mask to self-attention.
    explicit = torch.full((3, 2, 5, 5), -7.0)

    model(torch.arange(15).reshape(3, 5), attention_mask=padding, attn_mask=explicit)

    for recorder in recorders:
        assert recorder.messages == [(padding, explicit)]


def test_native_generation_uses_hybrid_cache() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make().eval()
    prompt = torch.arange(15).reshape(3, 5)
    output = generate(model, prompt, max_new_tokens=2, temperature=0, max_seq_len=8)
    expected = prompt
    with torch.no_grad():
        for _ in range(2):
            token = model(expected)[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat([expected, token], dim=-1)
    assert torch.equal(output, expected)


def test_hybrid_cached_dispatch_accepts_injected_cached_block() -> None:
    """An injected block can opt into cache handling structurally."""
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block = [_CachedIdentityBlock.Config(), _CachedIdentityBlock.Config()]
    model = config.make().eval()
    tokens = torch.arange(15).reshape(3, 5)

    cache = model.alloc_cache(
        batch=3,
        max_seq=tokens.shape[-1],
        device="meta",
        dtype=torch.float64,
    )
    actual = model(tokens, cache=cache)

    assert actual.shape == (3, 5, 32)
    assert cache == {
        ((index, 2),): _Slot(
            batch=3,
            max_seq=5,
            device="meta",
            dtype=torch.float64,
            reads=1,
        )
        for index in range(2)
    }


def test_forward_rejects_a_block_without_cache_argument() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _AttnOnlyBlock.Config()
    model = config.make().eval()
    cache = model.alloc_cache(batch=3, max_seq=5)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cache",
        ),
    ) as error:
        model(torch.arange(15).reshape(3, 5), cache=cache)
    assert "cache" in str(error.value)


def test_forward_returns_hidden_states_without_an_output_head() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    config.proj_out = None
    model = config.make().eval()
    tokens = torch.arange(15).reshape(3, 5)
    cache = model.alloc_cache(batch=3, max_seq=5)

    output = model(tokens, cache=cache)

    assert output.shape == (3, 5, 16)


@pytest.mark.parametrize("block", ["uncached", "non_cacheable"])
def test_alloc_cache_allocates_nothing_for_a_block_with_no_cached_attention(
    block: str,
) -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = (
        _UncachedIdentityBlock.Config()
        if block == "uncached"
        else _NonCacheableAttentionBlock.Config()
    )
    model = config.make()

    assert sorted(model.alloc_cache(batch=3, max_seq=5)) == [((1, 2),)]


def test_alloc_cache_rejects_a_cached_attention_without_a_depth_index() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _CachedIdentityBlock.Config()
    model = config.make()
    block = model.blocks[0]
    assert isinstance(block, _CachedIdentityBlock)
    del block.attn.depth_index

    with pytest.raises(TypeError, match="has no depth_index"):
        model.alloc_cache(batch=3, max_seq=5)


@dataclasses.dataclass(slots=True, kw_only=True)
class _Slot:
    """What one injected attention allocated, and how often it was read."""

    batch: int | tuple[int, ...]
    max_seq: int
    device: torch.device | str | None
    dtype: torch.dtype | None
    reads: int = 0


class _CachedIdentityAttention(torch.nn.Module):
    """Allocate a cache for an injected identity block, and count its reads."""

    def __init__(self, depth_index: DepthIndex) -> None:
        super().__init__()
        self.depth_index = depth_index

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> _Slot:
        return _Slot(batch=batch, max_seq=max_seq, device=device, dtype=dtype)

    @override
    def forward(
        self,
        x: torch.Tensor,
        /,
        *,
        cache: LayerCache,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        state = cache[self.depth_index]
        assert isinstance(state, _Slot)
        state.reads += 1
        return x


class _CachedIdentityBlock(torch.nn.Module):
    """A Configgle-injectable cached block preserving hidden states."""

    class Config(Fig["_CachedIdentityBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

        depth_index: DepthIndex = ()
        """Stack position, stamped by the transformer."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn = _CachedIdentityAttention(config.depth_index)

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(
        self,
        x: torch.Tensor,
        /,
        *,
        cache: LayerCache,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.attn(x, cache=cache, **kwargs)


class _UncachedIdentityBlock(torch.nn.Module):
    """A Configgle-injectable block missing both attn and cache handling."""

    class Config(Fig["_UncachedIdentityBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _AttnOnlyBlock(torch.nn.Module):
    """A Configgle-injectable block with attn but no cache argument."""

    class Config(Fig["_AttnOnlyBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

        depth_index: DepthIndex = ()
        """Stack position, stamped by the transformer."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn = _CachedIdentityAttention(config.depth_index)

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NonCacheableAttention(torch.nn.Module):
    """An attn submodule missing alloc_kv_cache."""

    def reset_parameters(self) -> None:
        """Leave the parameter-free test attention unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _NonCacheableAttentionBlock(torch.nn.Module):
    """A Configgle-injectable block whose attn lacks alloc_kv_cache."""

    class Config(Fig["_NonCacheableAttentionBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.attn = _NonCacheableAttention()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _MessageRecorder(torch.nn.Module):
    """Record public attention messages while preserving the residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        attention_mask = kwargs["attention_mask"]
        position_ids = kwargs["position_ids"]
        positions = kwargs["positions"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(position_ids, torch.Tensor)
        assert isinstance(positions, torch.Tensor)
        self.messages.append((attention_mask, position_ids, positions))
        return x


class _HeadMessageRecorder(torch.nn.Module):
    """Record output-head messages while preserving normalized hidden states."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[object | None] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.messages.append(kwargs.get("marker"))
        return x


class _MaskMessageRecorder(torch.nn.Module):
    """Record the two public mask messages while preserving the residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        attention_mask = kwargs["attention_mask"]
        attn_mask = kwargs["attn_mask"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(attn_mask, torch.Tensor)
        self.messages.append((attention_mask, attn_mask))
        return x


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
