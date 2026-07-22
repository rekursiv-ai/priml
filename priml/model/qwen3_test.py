"""Tests for priml.model.qwen3."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from priml.model.causal_lm import CausalLM
from priml.model.qwen3 import Qwen3, remap_hf_state_dict


def _hf_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": "qwen3",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
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


def _synth_hf_state_dict(cfg: Qwen3.Config) -> dict[str, torch.Tensor]:
    """Build a random-weight state_dict in HF Qwen3 layout."""
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, h),
        "model.norm.weight": torch.randn(h),
    }
    if not cfg.tie_embeddings:
        sd["lm_head.weight"] = torch.randn(cfg.vocab_size, h)
    for i in range(cfg.num_hidden_layers):
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


class TestConfig:
    def test_parse_basic(self):
        cfg = Qwen3.Config.from_hf(_hf_config())
        assert cfg.vocab_size == 128
        assert cfg.hidden_size == 64
        assert cfg.head_dim == 16
        assert cfg.num_key_value_heads == 2
        assert cfg.rope_theta == 1_000_000

    def test_wrong_model_type_rejected(self):
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(_hf_config(model_type="qwen2"))
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(_hf_config(model_type="qwen3_moe"))

    def test_head_dim_inferred_when_missing(self):
        cfg = _hf_config()
        cfg.pop("head_dim")
        parsed = Qwen3.Config.from_hf(cfg)
        assert parsed.head_dim == cfg["hidden_size"] // cfg["num_attention_heads"]

    def test_explicit_head_dim_not_equal_hidden(self):
        """Qwen3 with hidden != heads*head_dim builds, forwards, and loads.

        Regression for MODEL-008: e.g. hidden=1024, heads=16,
        head_dim=128 (16*128=2048 != 1024). The attention inner width
        differs from the residual width.
        """
        cfg = Qwen3.Config.from_hf(
            _hf_config(hidden_size=1024, num_attention_heads=16, head_dim=128),
        ).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf_state_dict(cfg), cfg))
        toks = torch.randint(0, cfg.vocab_size, (2, 5))
        assert model(toks).shape == (2, 5, cfg.vocab_size)

    def test_make_returns_qwen3_instance(self):
        """Makes[Qwen3] re-narrows .make() to Qwen3, not CausalLM."""
        model = Qwen3.Config.from_hf(_hf_config()).make()
        assert isinstance(model, Qwen3)
        assert isinstance(model, CausalLM)


class TestRemap:
    def test_end_to_end_load(self):
        cfg = Qwen3.Config.from_hf(_hf_config()).finalize()
        model = cfg.make()
        hf_sd = _synth_hf_state_dict(cfg)
        loop_sd = remap_hf_state_dict(hf_sd, cfg)
        model.load_state_dict(loop_sd, strict=True)

    def test_forward_after_load(self):
        cfg = Qwen3.Config.from_hf(_hf_config()).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf_state_dict(cfg), cfg))
        toks = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(toks)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_qkv_preserves_rows(self):
        """Per-head rows from HF Q/K/V land in the expected ensemble slots."""
        cfg = Qwen3.Config.from_hf(_hf_config()).finalize()
        h = cfg.hidden_size
        d = cfg.head_dim
        n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        hf_sd = _synth_hf_state_dict(cfg)
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
        cfg = Qwen3.Config.from_hf(_hf_config()).finalize()
        hf_sd = _synth_hf_state_dict(cfg)
        gate = hf_sd["model.layers.0.mlp.gate_proj.weight"]
        up = hf_sd["model.layers.0.mlp.up_proj.weight"]
        remapped = remap_hf_state_dict(hf_sd, cfg)
        fused = remapped["blocks.0.ffn.up_proj.weight"]
        assert fused.shape == (2 * cfg.intermediate_size, cfg.hidden_size)
        assert torch.equal(fused[: cfg.intermediate_size], gate)
        assert torch.equal(fused[cfg.intermediate_size :], up)

    def test_tied_embeddings(self):
        cfg = Qwen3.Config.from_hf(_hf_config(tie_word_embeddings=True)).finalize()
        hf_sd = _synth_hf_state_dict(cfg)
        remapped = remap_hf_state_dict(hf_sd, cfg)
        assert "lm_head.weight" not in remapped
        model = cfg.make()
        model.load_state_dict(remapped, strict=True)
        assert model.lm_head is None

    def test_independent_qk_norms(self):
        """q_norm and k_norm weights must be independent after load."""
        cfg = Qwen3.Config.from_hf(_hf_config()).finalize()
        hf_sd = _synth_hf_state_dict(cfg)
        hf_sd["model.layers.0.self_attn.q_norm.weight"].fill_(2.0)
        hf_sd["model.layers.0.self_attn.k_norm.weight"].fill_(3.0)
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(hf_sd, cfg))
        q_norm = model.blocks[0].attn.norm_q
        k_norm = model.blocks[0].attn.norm_k
        assert q_norm is not k_norm
        assert torch.all(q_norm.weight == 2.0)
        assert torch.all(k_norm.weight == 3.0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
