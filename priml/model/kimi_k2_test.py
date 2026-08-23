"""Tests for priml.model.kimi_k2."""

from __future__ import annotations

from typing import Any

from torch import Tensor

import pytest
import torch

from priml.model.causal_lm import CausalLM
from priml.model.kimi_k2 import KimiK2, remap_hf_state_dict


def _hf_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": "kimi_k2",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "q_lora_rank": None,
        "kv_lora_rank": 32,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 50_000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    base.update(overrides)
    return base


def _synth_hf(cfg: KimiK2.Config) -> dict[str, Tensor]:
    h = cfg.hidden_size
    n = cfg.num_attention_heads
    qkn, qkr, vd = (
        cfg.channels_qk_nope_head,
        cfg.channels_qk_rope_head,
        cfg.channels_v_head,
    )
    lr = cfg.kv_lora_rank
    sd: dict[str, Tensor] = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, h),
        "model.norm.weight": torch.randn(h),
    }
    if not cfg.tie_embeddings:
        sd["lm_head.weight"] = torch.randn(cfg.vocab_size, h)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(h)
        if cfg.q_lora_rank is None:
            sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(n * (qkn + qkr), h)
        else:
            sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(cfg.q_lora_rank, h)
            sd[f"{p}.self_attn.q_a_layernorm.weight"] = torch.randn(cfg.q_lora_rank)
            sd[f"{p}.self_attn.q_b_proj.weight"] = torch.randn(
                n * (qkn + qkr),
                cfg.q_lora_rank,
            )
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(lr + qkr, h)
        sd[f"{p}.self_attn.kv_a_layernorm.weight"] = torch.randn(lr)
        sd[f"{p}.self_attn.kv_b_proj.weight"] = torch.randn(n * (qkn + vd), lr)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, n * vd)
        if i < cfg.first_k_dense_replace:
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(cfg.intermediate_size, h)
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(cfg.intermediate_size, h)
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(h, cfg.intermediate_size)
        else:
            sd[f"{p}.mlp.gate.weight"] = torch.randn(cfg.n_routed_experts, h)
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(
                cfg.n_routed_experts,
            )
            for e in range(cfg.n_routed_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                    cfg.moe_intermediate_size,
                    h,
                )
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                    cfg.moe_intermediate_size,
                    h,
                )
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                    h,
                    cfg.moe_intermediate_size,
                )
            sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
                cfg.moe_intermediate_size,
                h,
            )
            sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(
                cfg.moe_intermediate_size,
                h,
            )
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(
                h,
                cfg.moe_intermediate_size,
            )
    return sd


class TestConfig:
    def test_parse_kimi_k2(self):
        cfg = KimiK2.Config.from_hf(_hf_config())
        assert cfg.kv_lora_rank == 32
        assert cfg.q_lora_rank is None
        assert cfg.first_k_dense_replace == 1
        assert cfg.channels_qk_nope_head == 16
        assert cfg.channels_qk_rope_head == 8
        assert cfg.channels_v_head == 16

    def test_parse_deepseek_v3(self):
        cfg = KimiK2.Config.from_hf(
            _hf_config(model_type="deepseek_v3", q_lora_rank=16),
        )
        assert cfg.q_lora_rank == 16

    def test_wrong_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            KimiK2.Config.from_hf(_hf_config(model_type="qwen3"))

    def test_yarn_scaling_wired(self):
        """YaRN params propagate from rope_scaling into the Config.yarn field."""
        cfg = KimiK2.Config.from_hf(
            _hf_config(
                rope_scaling={
                    "type": "yarn",
                    "factor": 32.0,
                    "original_max_position_embeddings": 4096,
                    "beta_fast": 1.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                },
            ),
        )
        assert cfg.yarn is not None
        assert cfg.yarn.factor == 32.0
        assert cfg.yarn.original_max_position_embeddings == 4096

    def test_non_yarn_scaling_rejected(self):
        with pytest.raises(ValueError, match="only yarn"):
            KimiK2.Config.from_hf(
                _hf_config(rope_scaling={"type": "linear", "factor": 2.0}),
            )

    def test_make_returns_kimik2_instance(self):
        model = KimiK2.Config.from_hf(_hf_config()).make()
        assert isinstance(model, KimiK2)
        assert isinstance(model, CausalLM)


class TestRemap:
    def test_end_to_end_no_q_lora(self):
        cfg = KimiK2.Config.from_hf(_hf_config()).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf(cfg), cfg), strict=True)
        logits = model(torch.randint(0, cfg.vocab_size, (1, 4)))
        assert logits.shape == (1, 4, cfg.vocab_size)

    def test_end_to_end_with_q_lora(self):
        cfg = KimiK2.Config.from_hf(_hf_config(q_lora_rank=24)).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf(cfg), cfg), strict=True)
        logits = model(torch.randint(0, cfg.vocab_size, (1, 3)))
        assert logits.shape == (1, 3, cfg.vocab_size)

    def test_dense_then_moe_layers(self):
        cfg = KimiK2.Config.from_hf(_hf_config(first_k_dense_replace=2)).finalize()
        model = cfg.make()
        remapped = remap_hf_state_dict(_synth_hf(cfg), cfg)
        model.load_state_dict(remapped, strict=True)
        # Layers 0, 1 are dense (no routing gate); layer 2 is MoE.
        assert "blocks.0.ffn.up_proj.weight" in remapped
        assert "blocks.0.ffn.router.gate.weight" not in remapped
        assert "blocks.2.ffn.router.gate.weight" in remapped
        assert "blocks.2.ffn.router.e_score_correction_bias" in remapped

    def test_missing_bias_defaults_zero(self):
        """HF checkpoints may omit e_score_correction_bias; remap defaults it."""
        cfg = KimiK2.Config.from_hf(_hf_config()).finalize()
        sd = _synth_hf(cfg)
        for i in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            sd.pop(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
        remapped = remap_hf_state_dict(sd, cfg)
        for i in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            key = f"blocks.{i}.ffn.router.e_score_correction_bias"
            assert key in remapped
            assert torch.all(remapped[key] == 0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
