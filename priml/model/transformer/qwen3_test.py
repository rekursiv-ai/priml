"""Tests for priml.model.transformer.qwen3."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import json
import os
import sys

from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml import hub
from priml.lib.custom_json import IntCodec
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import (
    GeometricFrequencies,
    HuggingFaceFrequencies,
    RoPE,
)
from priml.model.embedding import Embedding
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer import qwen3
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3 import Qwen3, remap_hf_state_dict
from priml.model.transformer.transformer import Transformer, head_is_tied
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from torch import LongTensor
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


_CWD: Final = Path(__file__).resolve().parent


def hf_config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_type": "qwen3",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    base.update(overrides)
    return base


def canonical_config() -> Qwen3.Config:
    return Qwen3.Config.from_hf(
        hf_config(
            vocab_size=9,
            hidden_size=8,
            intermediate_size=7,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
        ),
    )


def test_qwen3_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="qwen3",
        config=canonical_config(),
    )


def test_qwen3_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="qwen3",
        build_module=canonical_config().make,
        build_input=lambda: torch.tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 0], [8, 6, 4, 2, 1]],
        ),
        seed=0,
    )


@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_qwen3_cost_matches_torch(tie_embeddings: bool) -> None:
    """Every projection, the GQA products, and the head are torch's whole count.

    The fused CPU SDPA op is not in ``FlopCounterMode``'s registry, so the
    naive kernel makes the same two products countable. A tied head borrows
    the table, so it adds a matmul and no parameters.
    """
    config = Qwen3.Config.from_hf(
        hf_config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=tie_embeddings,
        ),
    )
    config.channels_out = 32
    attn(config).attn_kernel = SdpaNaive.Config()
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 32, (3, 5)),
        seq_len=5,
        batch_size=3,
        dtype=None,
    )
    assert analytical.bytes_state == 4 * 2 * 2 * 8  # Two KV heads cached per layer.


def synth_hf_state_dict(cfg: Qwen3.Config) -> dict[str, Tensor]:
    """Build a random-weight state_dict in HF Qwen3 layout."""
    h = cfg.channels_in
    inter = ffn(cfg).channels_hidden
    attention = attn(cfg)
    n_q = attention.num_heads
    n_kv = attention.num_heads_kv
    d = attention.channels_head
    sd: dict[str, Tensor] = {
        "model.embed_tokens.weight": torch.randn(cfg.channels_out, h),
        "model.norm.weight": torch.randn(h),
    }
    if not head_is_tied(cfg):
        sd["lm_head.weight"] = torch.randn(cfg.channels_out, h)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(h)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(n_q * d, h)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(n_kv * d, h)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(n_kv * d, h)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, n_q * d)
        sd[f"{p}.self_attn.q_norm.weight"] = torch.randn(d)
        sd[f"{p}.self_attn.k_norm.weight"] = torch.randn(d)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, h)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, h)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(h, inter)
    return sd


# Accepts a template or a finalized per-layer list, so a caller need not know which side
# of ``finalize`` it is on.
def attn(cfg: Qwen3.Config, layer: int = 0) -> Attention.Config:
    """One layer's attention -- where the head geometry lives now."""
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    return attn


def ffn(cfg: Qwen3.Config, layer: int = 0) -> SwiGLU.Config:
    """One layer's FFN -- where the hidden width lives now."""
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    ffn = block.ffn
    assert isinstance(ffn, SwiGLU.Config)
    return ffn


def _block(cfg: Qwen3.Config, layer: int = 0) -> TransformerBlock.Config:
    """One layer's block, template or finalized list alike."""
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    return block


def _final_norm(cfg: Transformer.Config) -> RMSNorm.Config:
    """Return the head's norm -- HF's ``model.norm``, first element of ``proj_out``."""
    assert isinstance(cfg.proj_out, Sequential.Config)
    elements = cfg.proj_out.elements
    assert isinstance(elements, list)
    norm = elements[0]
    assert isinstance(norm, RMSNorm.Config)
    return norm


class TestConfig:
    def test_parse_basic(self):
        cfg = Qwen3.Config.from_hf(hf_config())
        assert cfg.channels_out == 128
        assert cfg.channels_in == 64
        assert attn(cfg).channels_head == 16
        assert attn(cfg).num_heads_kv == 2
        rope = attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 1_000_000

    def test_wrong_model_type_rejected(self):
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(hf_config(model_type="qwen2"))
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(hf_config(model_type="qwen3_moe"))

    def test_head_dim_inferred_when_missing(self):
        cfg = hf_config()
        cfg.pop("head_dim")
        parsed = Qwen3.Config.from_hf(cfg)
        assert attn(parsed).channels_head == (
            IntCodec.coerce(cfg["hidden_size"])
            // IntCodec.coerce(cfg["num_attention_heads"])
        )

    def test_num_key_value_heads_inferred_when_missing(self):
        cfg = hf_config()
        cfg.pop("num_key_value_heads")
        parsed = Qwen3.Config.from_hf(cfg)
        assert attn(parsed).num_heads_kv == IntCodec.coerce(cfg["num_attention_heads"])

    @pytest.mark.parametrize("field_name", ["num_key_value_heads", "head_dim"])
    def test_explicit_zero_head_geometry_rejected(self, field_name: str):
        with pytest.raises(
            ValueError,
            match=rf"{field_name} must be > 0, got 0\.",
        ):
            Qwen3.Config.from_hf(hf_config(**{field_name: 0}))

    def test_nested_rope_parameters_accept_numeric_text(self):
        cfg = hf_config(rope_theta=None, rope_parameters={"rope_theta": "25000"})
        parsed = Qwen3.Config.from_hf(cfg)
        rope = attn(parsed).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 25_000.0

    def test_malformed_nested_rope_parameters_use_default(self):
        cfg = hf_config(rope_theta=None, rope_parameters=["not", "an", "object"])
        parsed = Qwen3.Config.from_hf(cfg)
        rope = attn(parsed).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 1_000_000.0

    @pytest.mark.parametrize(
        "overrides",
        [{"hidden_size": 0}, {"num_attention_heads": 0}],
    )
    def test_invalid_architecture_still_prints(
        self,
        overrides: dict[str, int],
    ) -> None:
        """A degenerate width renders; building it is torch's to refuse."""
        config = Qwen3.Config.from_hf(hf_config(**overrides))

        assert "Qwen3.Config" in config.pformat(hide_default_values=False)

    def test_explicit_head_dim_not_equal_hidden(self):
        """Qwen3 with hidden != num_heads*head_dim builds, forwards, and loads.

        Regression for MODEL-008: hidden=32, num_heads=4, head_dim=16
        (4*16=64 != 32). The attention inner width
        differs from the residual width.
        """
        cfg = Qwen3.Config.from_hf(
            hf_config(hidden_size=32, num_attention_heads=4, head_dim=16),
        ).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(synth_hf_state_dict(cfg), cfg))
        toks = torch.randint(0, cfg.channels_out, (3, 5))
        assert model(toks).shape == (3, 5, cfg.channels_out)

    def test_make_returns_qwen3_instance(self):
        """Makes[Qwen3] re-narrows .make() to Qwen3, not Transformer."""
        Qwen3.Config.from_hf(hf_config()).make()

    def test_attn_of_defaults_to_layer_zero(self):
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        assert isinstance(cfg.block, list)
        second = cfg.block[1]
        assert isinstance(second, TransformerBlock.Config)
        assert isinstance(second.attn, Attention.Config)
        second.attn.channels_head = 12

        assert qwen3._attn_of(cfg).channels_head == 16

    def test_attn_of_broadcasts_a_single_block_to_any_layer(self):
        cfg = Qwen3.Config.from_hf(hf_config())
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        cfg.block = [block]

        assert qwen3._attn_of(cfg, layer=1) is block.attn

    def test_attn_of_selects_the_requested_layer(self):
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        assert isinstance(cfg.block, list)
        second = cfg.block[1]
        assert isinstance(second, TransformerBlock.Config)
        assert isinstance(second.attn, Attention.Config)
        second.attn.channels_head = 12

        assert qwen3._attn_of(cfg, layer=1) is second.attn
        assert qwen3._attn_of(cfg, layer=1).channels_head == 12


class TestSlots:
    """The parent holds slots, not copies of its children's vocabulary."""

    def test_a_rope_edit_survives_finalize(self):
        """Editing the rope slot must reach the built attention.

        The parent used to redeclare ``rope_theta`` and ``frequencies`` and
        rebuild the child in ``finalize``, so this edit was discarded.
        """
        cfg = Qwen3.Config.from_hf(hf_config())
        template_rope = attn(cfg).rope
        assert isinstance(template_rope, RoPE.Config)
        template_rope.frequencies = GeometricFrequencies.Config(base=12_345.0)
        cfg = cfg.copy_tree().finalize()
        rope = attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, GeometricFrequencies.Config)
        assert rope.frequencies.base == 12_345.0
        # The width still comes from the attention it rotates.
        assert rope.channels_head == attn(cfg).channels_head

    def test_a_norm_edit_reaches_every_norm(self):
        """One template, so an epsilon set once applies throughout."""
        cfg = Qwen3.Config.from_hf(hf_config())
        template = _block(cfg)
        assert isinstance(template.norm1, RMSNorm.Config)
        template.norm1.eps = 1e-3
        _final_norm(cfg).eps = 1e-3
        cfg = cfg.copy_tree().finalize()
        assert isinstance(_block(cfg).norm1, RMSNorm.Config)
        norm1 = _block(cfg).norm1
        assert isinstance(norm1, RMSNorm.Config)
        assert norm1.eps == 1e-3
        assert _final_norm(cfg).eps == 1e-3

    def test_each_norm_is_its_own_object(self):
        """Templates are copied, so one consumer cannot edit another's."""
        cfg = Qwen3.Config.from_hf(hf_config()).copy_tree().finalize()
        block = _block(cfg)
        assert block.norm1 is not block.norm2
        assert block.norm1 is not _final_norm(cfg)
        assert block.norm1 is not _block(cfg, 1).norm1

    def test_architecture_specific_sizing_skips_other_blocks(self):
        cfg = Qwen3.Config.from_hf(hf_config())
        block = RMSNorm.Config()
        cfg._size_block(block)
        assert block.channels_in == cfg.channels_in


class TestLoad:
    def test_load_defaults_missing_checkpoint_dtype_to_bfloat16(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        config_dict.pop("torch_dtype")
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        monkeypatch.setattr(
            hub,
            "load_hf_checkpoint",
            Mock(return_value=(config_dict, synth_hf_state_dict(cfg))),
        )
        resolve_dtype = Mock(return_value=torch.bfloat16)
        monkeypatch.setattr(hub, "resolve_hf_dtype", resolve_dtype)

        model = Qwen3.load("Qwen/tiny-qwen")

        resolve_dtype.assert_called_once_with("bfloat16")
        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == torch.bfloat16

    def test_load_uses_checkpoint_dtype_when_no_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1, torch_dtype="float32")
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        monkeypatch.setattr(
            hub,
            "load_hf_checkpoint",
            Mock(return_value=(config_dict, synth_hf_state_dict(cfg))),
        )
        resolve_dtype = Mock(return_value=torch.float64)
        monkeypatch.setattr(hub, "resolve_hf_dtype", resolve_dtype)

        model = Qwen3.load("Qwen/tiny-qwen")

        resolve_dtype.assert_called_once_with("float32")
        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == torch.float64

    def test_local_load_reads_config_and_local_weights(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        (tmp_path / "config.json").write_text(json.dumps(config_dict))
        load_local_state_dict = Mock(return_value=synth_hf_state_dict(cfg))
        monkeypatch.setattr(hub, "load_local_state_dict", load_local_state_dict)

        model = Qwen3.load(tmp_path, device="cpu", dtype=torch.float32)

        assert isinstance(model.proj_in, Embedding)
        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == torch.float32
        assert model.num_layers == 1
        assert model.proj_in.weight.shape == (cfg.channels_out, cfg.channels_in)
        load_local_state_dict.assert_called_once_with(tmp_path)

    def test_explicit_load_dtype_and_device_are_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        monkeypatch.setattr(
            hub,
            "load_hf_checkpoint",
            Mock(return_value=(config_dict, synth_hf_state_dict(cfg))),
        )

        model = Qwen3.load("Qwen/tiny-qwen", dtype=torch.float64, device="meta")

        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == torch.float64
        assert model.proj_in.weight.device.type == "meta"

    def test_load_requires_complete_state_dict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        state_dict = synth_hf_state_dict(cfg)
        loop_state_dict = remap_hf_state_dict(state_dict, cfg)
        loop_state_dict["unexpected.weight"] = torch.empty(2, 3)
        monkeypatch.setattr(
            hub,
            "load_hf_checkpoint",
            Mock(return_value=(config_dict, state_dict)),
        )
        monkeypatch.setattr(
            qwen3,
            "remap_hf_state_dict",
            Mock(return_value=loop_state_dict),
        )

        with pytest.raises(RuntimeError, match=r"unexpected\.weight"):
            Qwen3.load("Qwen/tiny-qwen")

    def test_remote_load_uses_hf_model_config_and_weights(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(config_dict).finalize()
        hf_model = Mock()
        hf_model.config.to_dict.return_value = config_dict
        hf_model.state_dict.return_value = synth_hf_state_dict(cfg)
        load_transformers_model = Mock(return_value=hf_model)
        monkeypatch.setattr(
            hub,
            "load_transformers_model",
            load_transformers_model,
        )

        Qwen3.load("Qwen/tiny-qwen", dtype=torch.float32)

        load_transformers_model.assert_called_once_with(
            "Qwen/tiny-qwen",
            "AutoModelForCausalLM",
            dtype=torch.float32,
            trust_remote_code=False,
        )


class TestRemap:
    def test_end_to_end_load(self):
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        model = cfg.make()
        hf_sd = synth_hf_state_dict(cfg)
        loop_sd = remap_hf_state_dict(hf_sd, cfg)
        model.load_state_dict(loop_sd, strict=True)

    def test_forward_after_load(self):
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(synth_hf_state_dict(cfg), cfg))
        toks = torch.randint(0, cfg.channels_out, (3, 5))
        logits = model(toks)
        assert logits.shape == (3, 5, cfg.channels_out)

    def test_qkv_preserves_rows(self):
        """Per-head rows from HF Q/K/V land in the expected ensemble slots."""
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        h = cfg.channels_in
        attention = attn(cfg)
        d = attention.channels_head
        n_q, n_kv = attention.num_heads, attention.num_heads_kv
        hf_sd = synth_hf_state_dict(cfg)
        q = hf_sd["model.layers.0.self_attn.q_proj.weight"].view(n_q, d, h)
        k = hf_sd["model.layers.0.self_attn.k_proj.weight"].view(n_kv, d, h)
        v = hf_sd["model.layers.0.self_attn.v_proj.weight"].view(n_kv, d, h)
        remapped = remap_hf_state_dict(hf_sd, cfg)
        qkv = remapped["blocks.0.attn.proj_qkv.weight"]
        assert qkv.shape == (n_q + 2 * n_kv, d, h)
        assert torch.equal(qkv[:n_q], q)
        assert torch.equal(qkv[n_q : n_q + n_kv], k)
        assert torch.equal(qkv[n_q + n_kv :], v)

    def test_swiglu_gate_up_order(self):
        """Loop's chunk(2) yields (gate, x); cat must match."""
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        hf_sd = synth_hf_state_dict(cfg)
        gate = hf_sd["model.layers.0.mlp.gate_proj.weight"]
        up = hf_sd["model.layers.0.mlp.up_proj.weight"]
        remapped = remap_hf_state_dict(hf_sd, cfg)
        fused = remapped["blocks.0.ffn.up_proj.weight"]
        assert fused.shape == (2 * ffn(cfg).channels_hidden, cfg.channels_in)
        assert torch.equal(fused[: ffn(cfg).channels_hidden], gate)
        assert torch.equal(fused[ffn(cfg).channels_hidden :], up)

    def test_tied_embeddings(self):
        cfg = Qwen3.Config.from_hf(hf_config(tie_word_embeddings=True)).finalize()
        hf_sd = synth_hf_state_dict(cfg)
        remapped = remap_hf_state_dict(hf_sd, cfg)
        assert "proj_out.1.weight" not in remapped
        model = cfg.make()
        model.load_state_dict(remapped, strict=True)
        assert isinstance(model.proj_out, Sequential)
        assert isinstance(model.proj_out[1], TiedLinear)
        tokens = torch.arange(15).reshape(3, 5)
        assert model(tokens).shape == (3, 5, cfg.channels_out)

    def test_independent_qk_norms(self):
        """q_norm and k_norm weights must be independent after load."""
        cfg = Qwen3.Config.from_hf(hf_config()).finalize()
        hf_sd = synth_hf_state_dict(cfg)
        hf_sd["model.layers.0.self_attn.q_norm.weight"].fill_(2.0)
        hf_sd["model.layers.0.self_attn.k_norm.weight"].fill_(3.0)
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(hf_sd, cfg))
        block = model.blocks[0]
        assert isinstance(block, TransformerBlock)
        attn = block.attn
        assert isinstance(attn, Attention)
        q_norm = attn.norm_q
        k_norm = attn.norm_k
        assert isinstance(q_norm, RMSNorm)
        assert isinstance(k_norm, RMSNorm)
        assert q_norm is not k_norm
        q_weight = q_norm.weight
        k_weight = k_norm.weight
        assert isinstance(q_weight, Tensor)
        assert isinstance(k_weight, Tensor)
        assert torch.equal(q_weight, torch.full_like(q_weight, 2.0))
        assert torch.equal(k_weight, torch.full_like(k_weight, 3.0))

    @pytest.mark.parametrize("bad_part", ["block", "attention"])
    def test_remap_rejects_incompatible_layer_configs(self, bad_part: str):
        cfg = Qwen3.Config.from_hf(hf_config(num_hidden_layers=1))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        if bad_part == "block":
            cfg.block = RMSNorm.Config()
            match = "layer 0 is RMSNorm.Config, not a transformer."
        else:
            block.attn = RMSNorm.Config()
            match = "layer 0 attention is RMSNorm.Config, not self-attention."
        with pytest.raises(TypeError) as error:
            qwen3.remap_hf_state_dict({}, cfg)
        assert str(error.value) == match


@pytest.mark.compute_torch_compile
@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_qwen3_matches_hf(tie_embeddings: bool) -> None:
    """Our Qwen3 output must match HF's Qwen3ForCausalLM bit-for-bit."""
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    rng_state = torch.get_rng_state()
    try:
        torch.use_deterministic_algorithms(True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        torch.set_rng_state(generator.get_state())
        hf_out, loop_out = _qwen3_parity_outputs(tie_embeddings)
        assert torch.equal(hf_out, loop_out)
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        torch.set_rng_state(rng_state)


def test_qwen3_parity_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules[__name__],
        "_qwen3_parity_outputs",
        Mock(return_value=(torch.tensor([0.0]), torch.tensor([1.0]))),
    )

    with pytest.raises((AssertionError, pytest.skip.Exception)) as exc_info:
        test_qwen3_matches_hf(False)

    assert isinstance(exc_info.value, AssertionError)


@pytest.mark.parametrize(
    ("algorithms_enabled", "warn_only_enabled"),
    [(False, False), (True, True)],
)
def test_importing_parity_module_preserves_global_determinism(
    algorithms_enabled: bool,
    warn_only_enabled: bool,
) -> None:
    original_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        spec = util.spec_from_file_location(
            "_qwen3_hf_import_probe",
            __file__,
        )
        assert spec is not None
        assert spec.loader is not None
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert torch.are_deterministic_algorithms_enabled() == algorithms_enabled
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled() == warn_only_enabled
        )
    finally:
        torch.use_deterministic_algorithms(
            original_algorithms_enabled,
            warn_only=original_warn_only_enabled,
        )


def test_parity_test_restores_process_state_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    flash_sdp_enabled = torch.backends.cuda.flash_sdp_enabled()
    memory_efficient_sdp_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    rng_state = torch.get_rng_state()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(983)
        torch.set_rng_state(generator.get_state())
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        cudnn_available = Mock(return_value=True)
        cuda_available = Mock(return_value=True)
        enable_flash_sdp = Mock(wraps=torch.backends.cuda.enable_flash_sdp)
        enable_mem_efficient_sdp = Mock(
            wraps=torch.backends.cuda.enable_mem_efficient_sdp,
        )
        monkeypatch.setattr(torch.backends.cudnn, "is_available", cudnn_available)
        monkeypatch.setattr(torch.cuda, "is_available", cuda_available)
        monkeypatch.setattr(
            torch.backends.cuda,
            "enable_flash_sdp",
            enable_flash_sdp,
        )
        monkeypatch.setattr(
            torch.backends.cuda,
            "enable_mem_efficient_sdp",
            enable_mem_efficient_sdp,
        )
        expected_rng_state = torch.get_rng_state()
        monkeypatch.setattr(
            sys.modules[__name__],
            "_build_qwen3_hf_model",
            Mock(side_effect=RuntimeError("setup failed")),
        )

        with pytest.raises(RuntimeError, match="setup failed"):
            test_qwen3_matches_hf(False)

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        cudnn_available.assert_not_called()
        cuda_available.assert_not_called()
        enable_flash_sdp.assert_not_called()
        enable_mem_efficient_sdp.assert_not_called()
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.enable_flash_sdp(flash_sdp_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(memory_efficient_sdp_enabled)
        torch.set_rng_state(rng_state)


def _build_qwen3_hf_model(cfg_dict: dict[str, object]) -> Qwen3ForCausalLM:
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        Qwen3Config,
    )
    from transformers.models.qwen3.modeling_qwen3 import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        Qwen3ForCausalLM,
    )

    config = Qwen3Config(**cfg_dict, attn_implementation="eager")
    return Qwen3ForCausalLM(config).eval().to(dtype=torch.float32)


def _qwen3_parity_outputs(tie_embeddings: bool) -> tuple[Tensor, Tensor]:
    """Build HF and loop Qwen3 models with shared weights and inputs."""
    cfg_dict: dict[str, object] = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tie_embeddings,
        "attention_bias": False,
        "rope_theta": 1_000_000.0,
    }

    hf_model = _build_qwen3_hf_model(cfg_dict)
    config = Qwen3.Config.from_hf(
        {"model_type": "qwen3", "torch_dtype": "float32", **cfg_dict},
    ).finalize()
    assert isinstance(config.block, list)
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.attn, Attention.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        block.attn.split_qkv_projection = True
        block.ffn.split_gate_projection = True
        block.attn.attn_kernel = SdpaNaive.Config()
    loop_model = config.make()
    loop_model.load_state_dict(
        remap_hf_state_dict(
            {key: value.detach().cpu() for key, value in hf_model.state_dict().items()},
            config,
        ),
    )
    loop_model.eval().to(dtype=torch.float32)

    tokens = torch.randint(0, IntCodec.coerce(cfg_dict["vocab_size"]), (3, 5))
    with torch.no_grad():
        # ``forward`` rather than ``__call__``: the stub's ``__call__`` cannot
        # bind a ``forward`` taking ``**kwargs: Unpack[...]``, and an eval-mode
        # parity model registers no hooks to skip.
        hf_out = hf_model.forward(input_ids=cast("LongTensor", tokens)).logits
        loop_out = loop_model(tokens)
    assert hf_out is not None
    return hf_out, loop_out


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
