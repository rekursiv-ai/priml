"""Tests for priml.model.transformer.kimi_k2."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import sys
import warnings

from torch import Tensor

import pytest
import torch

from priml import hub
from priml.lib.absent import ABSENT
from priml.lib.codec import from_plain
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, YarnScaling
from priml.model.embedding import Embedding
from priml.model.moe import MoE, Router, SigmoidRouter, SoftmaxRouter
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer import kimi_k2, qwen3_test
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.kimi_k2 import KimiK2, remap_hf_state_dict
from priml.model.transformer.transformer import head_is_tied
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Generator

    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.modeling_utils import PreTrainedModel


_CWD: Final = Path(__file__).resolve().parent


def hf_config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_type": "kimi_k2",
        "vocab_size": 2,
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "q_lora_rank": None,
        "kv_lora_rank": 2,
        "intermediate_size": 4,
        "moe_intermediate_size": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
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


def canonical_config() -> KimiK2.Config:
    return KimiK2.Config.from_hf(
        hf_config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            kv_lora_rank=8,
            intermediate_size=32,
            moe_intermediate_size=16,
            n_routed_experts=2,
            num_experts_per_tok=1,
        ),
    )


@pytest.mark.compute_large_fixture
def test_kimi_k2_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="kimi_k2",
        config=canonical_config(),
    )


def test_kimi_k2_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="kimi_k2",
        build_module=lambda: KimiK2.Config.from_hf(hf_config()).make(),
        build_input=lambda: torch.tensor([[0, 1]]),
        seed=0,
    )


@pytest.mark.parametrize("q_lora_rank", [None, 6])
def test_kimi_k2_cost_requires_realized_routing(q_lora_rank: int | None) -> None:
    """Exact MoE bytes require the realized per-layer routing occupancy.

    The whole-invocation harness rejects this model without that geometry;
    top-k alone does not identify which routed experts executed.
    """
    config = KimiK2.Config.from_hf(
        hf_config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=8,
            intermediate_size=32,
            moe_intermediate_size=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
        ),
    )
    with pytest.raises(ValueError, match="expert_rows"):
        assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randint(0, 32, (2, 5)),
            seq_len=5,
            batch_size=2,
            dtype=None,
        )


def _router(cfg: KimiK2.Config, layer: int = -1) -> Router.Config:
    """Return the routing config -- where the expert COUNT lives now."""
    blocks = cfg.block if isinstance(cfg.block, list) else [cfg.block]
    block = blocks[0] if len(blocks) == 1 else blocks[layer]
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, MoE.Config)
    assert isinstance(block.ffn.router, Router.Config)
    return block.ffn.router


def _synth_hf(cfg: KimiK2.Config) -> dict[str, Tensor]:
    h = cfg.channels_in
    attn = _attn(cfg)
    n = attn.num_heads
    qkn, qkr, vd = (
        attn.channels_qk_nope_head,
        attn.channels_qk_rope_head,
        attn.channels_v_head,
    )
    lr = attn.kv_lora_rank
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
        if attn.q_lora_rank is None:
            sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(n * (qkn + qkr), h)
        else:
            sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(attn.q_lora_rank or 0, h)
            sd[f"{p}.self_attn.q_a_layernorm.weight"] = torch.randn(
                attn.q_lora_rank or 0,
            )
            sd[f"{p}.self_attn.q_b_proj.weight"] = torch.randn(
                n * (qkn + qkr),
                attn.q_lora_rank or 0,
            )
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(lr + qkr, h)
        sd[f"{p}.self_attn.kv_a_layernorm.weight"] = torch.randn(lr)
        sd[f"{p}.self_attn.kv_b_proj.weight"] = torch.randn(n * (qkn + vd), lr)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, n * vd)
        if i < cfg.first_k_dense_replace:
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(cfg.channels_hidden_dense, h)
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(cfg.channels_hidden_dense, h)
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(h, cfg.channels_hidden_dense)
        else:
            sd[f"{p}.mlp.gate.weight"] = torch.randn(_router(cfg).num_experts, h)
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(
                _router(cfg).num_experts,
            )
            for e in range(_router(cfg).num_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                    cfg.channels_hidden_expert,
                    h,
                )
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                    cfg.channels_hidden_expert,
                    h,
                )
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                    h,
                    cfg.channels_hidden_expert,
                )
            sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
                cfg.channels_hidden_expert,
                h,
            )
            sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(
                cfg.channels_hidden_expert,
                h,
            )
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(
                h,
                cfg.channels_hidden_expert,
            )
    return sd


# Accepts a template or a finalized per-layer list, so a caller need not know which side
# of ``finalize`` it is on.
def _attn(cfg: KimiK2.Config, layer: int = 0) -> MultiHeadLatentAttention.Config:
    """One layer's attention -- where the head geometry lives now."""
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, MultiHeadLatentAttention.Config)
    return attn


class TestConfig:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_type", []),
            ("norm_topk_prob", "false"),
            ("tie_word_embeddings", "false"),
            ("rope_scaling", {"type": "yarn"}),
            ("rope_scaling", False),
            ("rope_scaling", []),
            ("rope_scaling", {"type": [], "factor": 2.0}),
            ("scoring_func", []),
            ("rope_theta", None),
            ("num_experts_per_tok", None),
            ("n_routed_experts", None),
        ],
    )
    def test_wrongly_typed_hf_fields_rejected(self, field: str, value: object) -> None:
        with pytest.raises(TypeError):
            KimiK2.Config.from_hf(hf_config(**{field: value}))

    def test_nonpositive_dense_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="intermediate_size"):
            KimiK2.Config.from_hf(hf_config(intermediate_size=0))

    def test_yarn_keeps_base_and_scales_attention(self) -> None:
        cfg = KimiK2.Config.from_hf(
            hf_config(
                rope_scaling={"type": "yarn", "factor": 32.0, "mscale_all_dim": 1.0},
            ),
        )
        attn = _attn(cfg)
        assert isinstance(attn.rope, RoPE.Config)
        assert isinstance(attn.rope.frequencies, YarnScaling.Config)
        assert isinstance(attn.rope.frequencies.inner, HuggingFaceFrequencies.Config)
        assert attn.rope.frequencies.inner.base == 50_000.0
        assert attn.softmax_scale == pytest.approx(
            attn.channels_qk_head**-0.5
            * (1 + 0.1 * torch.log(torch.tensor(32.0)).item()) ** 2,
        )

    def test_rope_parameters_alias_preserves_yarn(self) -> None:
        config = hf_config(
            rope_parameters={"type": "yarn", "factor": 32.0, "mscale_all_dim": 1.0},
        )
        attn = _attn(KimiK2.Config.from_hf(config))
        assert isinstance(attn.rope, RoPE.Config)
        assert isinstance(attn.rope.frequencies, YarnScaling.Config)
        assert attn.rope.frequencies.factor == 32.0

    def test_quantized_checkpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantization"):
            KimiK2.Config.from_hf(
                hf_config(quantization_config={"quant_method": "fp8"}),
            )

    def test_default_router_is_buildable_sigmoid(self) -> None:
        router = _router(KimiK2.Config())
        assert isinstance(router, SigmoidRouter.Config)
        router.channels_in = 4
        assert isinstance(router.make(), SigmoidRouter)

    @pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
    def test_parse_router_variant(self, scoring_func: str) -> None:
        cfg = KimiK2.Config.from_hf(
            hf_config(scoring_func=scoring_func, norm_topk_prob=False),
        ).finalize()
        router = _router(cfg)
        assert router.norm_topk_prob is False
        assert router.channels_in == cfg.channels_in
        assert router.num_experts == 2
        assert router.top_k == 1
        if scoring_func == "sigmoid":
            assert isinstance(router, SigmoidRouter.Config)
            assert router.routed_scaling_factor == 2.0
        else:
            assert isinstance(router, SoftmaxRouter.Config)
        assert router.make().scoring_func == scoring_func

    def test_parse_kimi_k2(self):
        cfg = KimiK2.Config.from_hf(hf_config())
        attn = _attn(cfg)
        assert attn.kv_lora_rank == 2
        assert attn.q_lora_rank is None
        assert cfg.first_k_dense_replace == 1
        assert attn.channels_qk_nope_head == 2
        assert attn.channels_qk_rope_head == 2
        assert attn.channels_v_head == 2

    def test_parse_deepseek_v3(self):
        cfg = KimiK2.Config.from_hf(
            hf_config(model_type="deepseek_v3", q_lora_rank=16),
        )
        assert _attn(cfg).q_lora_rank == 16

    def test_wrong_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            KimiK2.Config.from_hf(hf_config(model_type="qwen3"))

    def test_wrong_scoring_function(self):
        with pytest.raises(ValueError, match="scoring_func"):
            KimiK2.Config.from_hf(hf_config(scoring_func="linear"))

    def test_rope_scaling_without_type_is_rejected(self):
        with pytest.raises(TypeError):
            KimiK2.Config.from_hf(hf_config(rope_scaling={"factor": 2.0}))

    def test_rope_type_alias_selects_yarn(self):
        cfg = KimiK2.Config.from_hf(
            hf_config(
                rope_scaling={
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 4096,
                },
            ),
        )
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, YarnScaling.Config)
        assert rope.frequencies.factor == 2.0

    @pytest.mark.parametrize("moe_intermediate_size", [0, -64])
    def test_nonpositive_moe_width_rejected(self, moe_intermediate_size: int):
        """An explicit width names its own field instead of silently defaulting."""
        with pytest.raises(ValueError, match="moe_intermediate_size must be > 0"):
            KimiK2.Config.from_hf(
                hf_config(moe_intermediate_size=moe_intermediate_size),
            )

    def test_absent_moe_width_falls_back_to_dense_width(self):
        """Only a MISSING key defaults; the fallback itself must survive."""
        config = hf_config()
        del config["moe_intermediate_size"]
        cfg = KimiK2.Config.from_hf(config)
        assert cfg.channels_hidden_expert == cfg.channels_hidden_dense == 4

    @pytest.mark.compute_large_fixture
    def test_nonpositive_channels_still_print(self) -> None:
        """The degenerate config is the one worth rendering; torch rejects it."""
        config = KimiK2.Config.from_hf(hf_config(hidden_size=0))

        assert "KimiK2.Config" in config.pformat(hide_default_values=False)

    def test_yarn_scaling_wired(self):
        """YaRN params land on the rope slot's frequency builder."""
        cfg = KimiK2.Config.from_hf(
            hf_config(
                rope_scaling={
                    "type": "yarn",
                    "factor": 32.0,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 1.0,
                    "beta_slow": 3.0,
                    "mscale": 2.5,
                    "mscale_all_dim": 4.5,
                },
            ),
        )
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        yarn = rope.frequencies
        assert isinstance(yarn, YarnScaling.Config)
        assert yarn.factor == 32.0
        assert yarn.original_max_position_embeddings == 8192
        assert yarn.beta_fast == 1.0
        assert yarn.beta_slow == 3.0
        assert yarn.mscale == 2.5
        assert yarn.mscale_all_dim == 4.5

    def test_yarn_scaling_defaults(self):
        cfg = KimiK2.Config.from_hf(
            hf_config(
                rope_scaling={"type": "yarn", "factor": 2.0},
            ),
        )
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        yarn = rope.frequencies
        assert isinstance(yarn, YarnScaling.Config)
        assert yarn.original_max_position_embeddings == 4_096
        assert yarn.beta_fast == 32.0
        assert yarn.beta_slow == 1.0
        assert yarn.mscale == 1.0
        assert yarn.mscale_all_dim == 0.0


class TestSlots:
    """The parent holds slots, not copies of its children's vocabulary."""

    def test_layer_accessors_broadcast_one_template(self):
        cfg = KimiK2.Config.from_hf(hf_config())
        assert isinstance(cfg.block, TransformerBlock.Config)
        assert kimi_k2._attn_of(cfg, 1) is cfg.block.attn
        assert kimi_k2._moe_of(cfg, 1) is cfg.block.ffn

    def test_layer_accessors_use_each_layers_config(self):
        cfg = KimiK2.Config.from_hf(hf_config()).finalize()
        assert isinstance(cfg.block, list)
        second = cfg.block[1]
        assert isinstance(second, TransformerBlock.Config)
        assert isinstance(second.attn, MultiHeadLatentAttention.Config)
        assert isinstance(second.ffn, MoE.Config)
        assert isinstance(second.ffn.router, Router.Config)
        second.attn.q_lora_rank = 6
        second.ffn.router.top_k = 2

        assert kimi_k2._attn_of(cfg, 1).q_lora_rank == 6
        assert kimi_k2._moe_of(cfg, 1).router.top_k == 2

    def test_a_router_edit_survives_finalize(self):
        """Editing the router slot must reach the built MoE layers.

        The parent used to redeclare Router's fields and rebuild the child in
        ``finalize``, so this edit was silently discarded.
        """
        cfg = KimiK2.Config.from_hf(hf_config())
        template = cfg.block
        assert isinstance(template, TransformerBlock.Config)
        assert isinstance(template.ffn, MoE.Config)
        assert isinstance(template.ffn.router, SigmoidRouter.Config)
        template.ffn.router.routed_scaling_factor = 2.5
        cfg = cfg.copy_tree().finalize()
        assert isinstance(cfg.block, list)
        last = cfg.block[-1]
        assert isinstance(last, TransformerBlock.Config)
        assert isinstance(last.ffn, MoE.Config)
        assert isinstance(last.ffn.router, SigmoidRouter.Config)
        assert last.ffn.router.routed_scaling_factor == 2.5

    def test_a_norm_edit_reaches_every_norm(self):
        """One template, so an epsilon set once applies throughout."""
        cfg = KimiK2.Config.from_hf(hf_config())
        template = cfg.block
        assert isinstance(template, TransformerBlock.Config)
        assert isinstance(template.norm1, RMSNorm.Config)
        template.norm1.eps = 1e-3
        qwen3_test._final_norm(cfg).eps = 1e-3
        cfg = cfg.copy_tree().finalize()
        assert isinstance(cfg.block, list)
        block = cfg.block[0]
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.norm1, RMSNorm.Config)
        assert block.norm1.eps == 1e-3
        assert qwen3_test._final_norm(cfg).eps == 1e-3

    def test_each_layer_gets_its_own_norm_object(self):
        """Templates are copied, so one layer's finalize cannot edit another."""
        cfg = KimiK2.Config.from_hf(hf_config()).copy_tree().finalize()
        assert isinstance(cfg.block, list)
        first, second = cfg.block[0], cfg.block[1]
        assert isinstance(first, TransformerBlock.Config)
        assert isinstance(second, TransformerBlock.Config)
        assert first.norm1 is not second.norm1
        assert first.norm1 is not first.norm2

    def test_non_yarn_scaling_rejected(self):
        with pytest.raises(ValueError, match="only yarn"):
            KimiK2.Config.from_hf(
                hf_config(rope_scaling={"type": "linear", "factor": 2.0}),
            )

    def test_make_returns_kimik2_instance(self):
        KimiK2.Config.from_hf(hf_config()).make()

    def test_architecture_specific_sizing_skips_other_blocks(self):
        cfg = KimiK2.Config.from_hf(hf_config())
        block = RMSNorm.Config()
        cfg._size_block(block, 0)
        assert block.channels_in == cfg.channels_in

    def test_custom_nondense_ffn_survives_sizing(self):
        cfg = KimiK2.Config.from_hf(hf_config(first_k_dense_replace=0))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        block.ffn = SwiGLU.Config(channels_hidden=17)
        cfg._size_block(block, 0)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert block.ffn.channels_hidden == 17


class TestLoad:
    @pytest.mark.parametrize(
        ("torch_dtype", "expected_dtype"),
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            (None, torch.bfloat16),
        ],
    )
    def test_load_uses_checkpoint_dtype_without_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        torch_dtype: str | None,
        expected_dtype: torch.dtype,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        if torch_dtype is None:
            del config_dict["torch_dtype"]
        else:
            config_dict["torch_dtype"] = torch_dtype
        config = KimiK2.Config.from_hf(config_dict).finalize()
        monkeypatch.setattr(
            hub,
            "load_hf_checkpoint",
            Mock(return_value=(config_dict, _synth_hf(config))),
        )

        model = KimiK2.load("moonshotai/tiny-kimi", dtype=None)

        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == expected_dtype

    def test_remote_load_uses_hf_config_weights_dtype_and_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dict = hf_config(num_hidden_layers=1)
        cfg = KimiK2.Config.from_hf(config_dict).finalize()
        hf_model = Mock()
        hf_model.config.to_dict.return_value = config_dict
        hf_model.state_dict.return_value = _synth_hf(cfg)
        load_transformers_model = Mock(return_value=hf_model)
        monkeypatch.setattr(
            hub,
            "load_transformers_model",
            load_transformers_model,
        )

        model = KimiK2.load("moonshotai/tiny-kimi", device="meta", dtype=torch.bfloat16)

        assert isinstance(model.proj_in, Embedding)
        assert model.proj_in.weight.dtype == torch.bfloat16
        assert model.proj_in.weight.device.type == "meta"
        load_transformers_model.assert_called_once_with(
            "moonshotai/tiny-kimi",
            "AutoModelForCausalLM",
            dtype=torch.bfloat16,
            trust_remote_code=False,
        )


class TestRemap:
    @pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("correction_bias", [False, True])
    def test_remap_router_buffers_match_variant(
        self,
        scoring_func: str,
        correction_bias: bool,
    ) -> None:
        cfg = KimiK2.Config.from_hf(
            hf_config(scoring_func=scoring_func),
        )
        router = _router(cfg)
        if isinstance(router, SigmoidRouter.Config):
            router.use_correction_bias = correction_bias
        cfg.finalize()
        hf_state = _synth_hf(cfg)
        remapped = remap_hf_state_dict(hf_state, cfg)
        bias_key = "blocks.1.ffn.router.e_score_correction_bias"
        assert (bias_key in remapped) == (scoring_func == "sigmoid" and correction_bias)
        if bias_key in remapped:
            assert torch.equal(
                remapped[bias_key],
                hf_state["model.layers.1.mlp.gate.e_score_correction_bias"],
            )
        model = cfg.make()
        model.load_state_dict(remapped, strict=True)
        assert model(torch.tensor([[0, 1]])).shape == (1, 2, cfg.channels_out)

    def test_end_to_end_no_q_lora(self):
        cfg = KimiK2.Config.from_hf(hf_config()).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf(cfg), cfg), strict=True)
        logits = model(torch.randint(0, cfg.channels_out, (2, 4)))
        assert logits.shape == (2, 4, cfg.channels_out)

    def test_end_to_end_with_q_lora(self):
        cfg = KimiK2.Config.from_hf(hf_config(q_lora_rank=24)).finalize()
        model = cfg.make()
        model.load_state_dict(remap_hf_state_dict(_synth_hf(cfg), cfg), strict=True)
        logits = model(torch.randint(0, cfg.channels_out, (2, 3)))
        assert logits.shape == (2, 3, cfg.channels_out)

    def test_dense_then_moe_layers(self):
        cfg = KimiK2.Config.from_hf(
            hf_config(first_k_dense_replace=2, num_hidden_layers=3),
        ).finalize()
        model = cfg.make()
        remapped = remap_hf_state_dict(_synth_hf(cfg), cfg)
        model.load_state_dict(remapped, strict=True)
        # Layers 0, 1 are dense (no routing gate); layer 2 is MoE.
        assert "blocks.0.ffn.up_proj.weight" in remapped
        assert "blocks.0.ffn.router.gate.weight" not in remapped
        assert "blocks.2.ffn.router.gate.weight" in remapped
        assert "blocks.2.ffn.router.e_score_correction_bias" in remapped

    def test_missing_correction_bias_rejected(self) -> None:
        cfg = KimiK2.Config.from_hf(hf_config()).finalize()
        sd = _synth_hf(cfg)
        del sd["model.layers.1.mlp.gate.e_score_correction_bias"]
        with pytest.raises(KeyError, match="e_score_correction_bias"):
            remap_hf_state_dict(sd, cfg)

    def test_multiple_shared_experts_split_fused_hf_weights(self) -> None:
        cfg = KimiK2.Config.from_hf(hf_config(n_shared_experts=2)).finalize()
        sd = _synth_hf(cfg)
        prefix = "model.layers.1.mlp.shared_experts"
        width = cfg.channels_hidden_expert
        gate = torch.randn(2 * width, cfg.channels_in)
        up = torch.randn(2 * width, cfg.channels_in)
        down = torch.randn(cfg.channels_in, 2 * width)
        sd[f"{prefix}.gate_proj.weight"] = gate
        sd[f"{prefix}.up_proj.weight"] = up
        sd[f"{prefix}.down_proj.weight"] = down
        remapped = remap_hf_state_dict(sd, cfg)
        model = cfg.make()
        model.load_state_dict(remapped, strict=True)
        for expert in range(2):
            start = expert * width
            key = f"blocks.1.ffn.shared_experts.{expert}"
            assert torch.equal(
                remapped[f"{key}.up_proj.weight"],
                torch.cat([gate[start : start + width], up[start : start + width]]),
            )
            assert torch.equal(
                remapped[f"{key}.down_proj.weight"],
                down[:, start : start + width],
            )

    @pytest.mark.parametrize("bad_part", ["block", "attention", "ffn"])
    def test_remap_rejects_incompatible_layer_configs(self, bad_part: str):
        cfg = KimiK2.Config.from_hf(hf_config(num_hidden_layers=1))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        if bad_part == "block":
            cfg.block = RMSNorm.Config()
            match = "not a transformer"
        elif bad_part == "attention":
            block.attn = RMSNorm.Config()
            match = "not MLA"
        else:
            block.ffn = SwiGLU.Config()
            match = "not MoE"
        if bad_part in ("block", "ffn"):
            with pytest.raises(TypeError, match=match) as exc_info:
                kimi_k2._moe_of(cfg, 0)
            expected_message = (
                "layer 0 is RMSNorm.Config, not a transformer."
                if bad_part == "block"
                else "layer 0 FFN is SwiGLU.Config, not MoE."
            )
            assert str(exc_info.value) == expected_message
        if bad_part in ("block", "attention"):
            with pytest.raises(TypeError, match=match) as exc_info:
                kimi_k2._attn_of(cfg, 0)
            expected_message = (
                "layer 0 is RMSNorm.Config, not a transformer."
                if bad_part == "block"
                else "layer 0 attention is RMSNorm.Config, not MLA."
            )
            assert str(exc_info.value) == expected_message


@pytest.mark.network_huggingface
@pytest.mark.parametrize("q_lora_rank", [None, 16])
@pytest.mark.parametrize("yarn", [False, True])
@pytest.mark.parametrize("shared_experts", [1, 2])
def test_kimi_k2_matches_hf_deepseek_v3(
    q_lora_rank: int | None,
    yarn: bool,
    shared_experts: int,
):
    """KimiK2 logits must match HF's DeepseekV3ForCausalLM."""
    torch.manual_seed(0)
    # The shims stay installed across the HF forward pass, not just
    # construction: the remote DeepSeek-V3 code reads both symbols at call
    # time, so exiting the block earlier would raise inside ``hf_model(...)``.
    with _install_transformers_compat_shims():
        hf_model = _build_hf_model(
            q_lora_rank,
            yarn=yarn,
            shared_experts=shared_experts,
        )
        config = _our_config_from_hf(hf_model, q_lora_rank)
        loop_sd = remap_hf_state_dict(
            _hf_state_dict_with_bias_fill(hf_model, config),
            config,
        )
        loop_model = config.make()
        loop_model.load_state_dict(loop_sd, strict=True)
        loop_model = loop_model.to(torch.float32).eval()

        if yarn:
            reference_rope = hf_model.get_submodule(
                "model.layers.0.self_attn.rotary_emb",
            )
            rope_config = _attn(config).rope
            assert isinstance(rope_config, RoPE.Config)
            our_rope = rope_config.make()
            reference_freq = reference_rope.get_buffer("inv_freq")
            our_freq = our_rope._inv_freqs[0].squeeze(0)
            assert torch.equal(reference_freq, our_freq), (
                f"inv_freq max abs diff: {(reference_freq - our_freq).abs().max().item():.9g}; "
                f"HF={reference_freq.tolist()}, ours={our_freq.tolist()}"
            )
            positions = torch.arange(5)
            cos, sin = our_rope(positions)
            for name, ours in (("cos_cached", cos), ("sin_cached", sin)):
                reference = reference_rope.get_buffer(name)[:5, :4]
                assert torch.equal(reference, ours.squeeze(-2)), (
                    f"{name} max abs diff: {(reference - ours.squeeze(-2)).abs().max().item():.9g}"
                )
        tokens = torch.randint(0, config.channels_out, (2, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with host_agnostic_numerics(), torch.no_grad():
                # ``forward`` rather than ``__call__``: the stub's ``__call__``
                # cannot bind a ``forward`` taking ``**kwargs: Unpack[...]``, and
                # an eval-mode parity model registers no hooks to skip. The
                # remote-code class has no stub, so its output is narrowed here.
                hf_result = cast(
                    "CausalLMOutputWithPast",
                    hf_model.forward(input_ids=tokens, use_cache=False),
                )
                hf_out = hf_result.logits
                assert hf_out is not None
                loop_out = loop_model(tokens)
    diff = (hf_out - loop_out).abs().max().item()
    assert torch.allclose(hf_out, loop_out, atol=5e-5, rtol=1e-4), (
        f"max abs diff: {diff:.3e}"
    )


@pytest.mark.parametrize("fx_present", [False, True])
@pytest.mark.parametrize("legacy_present", [False, True])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_transformers_compat_shims_restore_module_state(
    monkeypatch: pytest.MonkeyPatch,
    fx_present: bool,
    legacy_present: bool,
    raise_inside: bool,
) -> None:
    """Shims apply inside the block and leave the modules exactly as found."""
    # The real HF parity test above exercises Transformers itself. Here the
    # version-dependent owners cover both existing and removed symbols.
    transformers = ModuleType("transformers")
    utils = ModuleType("transformers.utils")
    import_utils = ModuleType("transformers.utils.import_utils")

    class DynamicCache:
        @classmethod
        def from_legacy_cache(cls, pkv: object) -> object:
            del cls
            return pkv

    if fx_present:
        monkeypatch.setattr(
            import_utils,
            "is_torch_fx_available",
            Mock(return_value=True),
            raising=False,
        )
    if not legacy_present:
        monkeypatch.delattr(DynamicCache, "from_legacy_cache")
    vars(transformers)["DynamicCache"] = DynamicCache
    vars(transformers)["utils"] = utils
    vars(utils)["import_utils"] = import_utils
    for module in (transformers, utils, import_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    # Inspect owner namespaces directly so inherited attributes cannot hide leaks.
    fx_before: object = vars(import_utils).get("is_torch_fx_available", ABSENT)  # pyright: ignore[reportAny] -- vars() exposes dynamic module state as Any.
    legacy_before: object = vars(DynamicCache).get("from_legacy_cache", ABSENT)  # pyright: ignore[reportAny] -- vars() exposes dynamic class state as Any.

    error = (
        pytest.raises(RuntimeError, match="inside shims")
        if raise_inside
        else nullcontext()
    )
    with error, _install_transformers_compat_shims():
        assert callable(cast(object, vars(import_utils)["is_torch_fx_available"]))
        assert callable(DynamicCache.from_legacy_cache)
        if raise_inside:
            raise RuntimeError("inside shims")

    assert vars(import_utils).get("is_torch_fx_available", ABSENT) is fx_before
    assert vars(DynamicCache).get("from_legacy_cache", ABSENT) is legacy_before


@contextmanager
def _install_transformers_compat_shims() -> Generator[None]:
    """Backfill transformers 4.x symbols removed in transformers 5.x."""
    pytest.importorskip("transformers")
    from transformers import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        DynamicCache,
    )
    from transformers.utils import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        import_utils,
    )

    def unavailable() -> bool:
        return False

    def passthrough_cache(cls: type[DynamicCache], pkv: object) -> object:
        del cls
        return pkv

    # Names come from this tuple rather than literals so the shims install and
    # uninstall through one pair of dynamic accesses: a literal
    # ``DynamicCache.from_legacy_cache = ...`` is an attribute the stub does not
    # declare, and unwinding it would need a second, separately-maintained list.
    shims: tuple[tuple[object, str, object], ...] = (
        (import_utils, "is_torch_fx_available", unavailable),
        (DynamicCache, "from_legacy_cache", classmethod(passthrough_cache)),
    )
    installed = [shim for shim in shims if not hasattr(shim[0], shim[1])]
    for owner, name, value in installed:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, _ in installed:
            delattr(owner, name)


# ``PreTrainedModel``, not ``DeepseekV3ForCausalLM``: ``trust_remote_code`` loads
# the checkpoint's own module class, which shares the name but not the identity.
def _build_hf_model(
    q_lora_rank: int | None,
    *,
    yarn: bool,
    shared_experts: int,
) -> PreTrainedModel:
    """Instantiate HF's real ``DeepseekV3ForCausalLM`` at tiny size."""
    pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        AutoConfig,
    )
    from transformers.models.auto.modeling_auto import (  # noqa: PLC0415 -- The optional Transformers dependency is loaded only in these tests.
        AutoModelForCausalLM,
    )

    config = AutoConfig.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        trust_remote_code=True,
    )
    config.hidden_size = 32
    config.num_hidden_layers = 3
    config.num_attention_heads = 4
    config.qk_nope_head_dim = 8
    config.qk_rope_head_dim = 8
    config.v_head_dim = 8
    config.kv_lora_rank = 16
    config.intermediate_size = 64
    config.moe_intermediate_size = 32
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.n_shared_experts = shared_experts
    config.first_k_dense_replace = 1
    config.vocab_size = 64
    config.n_group = 1
    config.topk_group = 1
    config.rope_theta = 50_000.0
    config.max_position_embeddings = 64
    config.rope_scaling = (
        {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
        if yarn
        else None
    )
    config.quantization_config = None
    config.q_lora_rank = q_lora_rank
    config.torch_dtype = "float32"
    config.use_cache = False
    config.tie_word_embeddings = False
    config.scoring_func = "sigmoid"
    config.norm_topk_prob = True
    config.routed_scaling_factor = 1.0
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    return model.to(torch.float32).eval()


def _our_config_from_hf(
    hf_model: PreTrainedModel,
    q_lora_rank: int | None,
) -> KimiK2.Config:
    """Mirror an HF model's config into a ``KimiK2.Config``."""
    hf_cfg = from_plain(hf_model.config.to_dict(), dict[str, object])
    hf_cfg.setdefault("model_type", "deepseek_v3")
    hf_cfg["q_lora_rank"] = q_lora_rank
    hf_cfg["tie_word_embeddings"] = False
    return KimiK2.Config.from_hf(hf_cfg).finalize()


def _hf_state_dict_with_bias_fill(
    hf_model: PreTrainedModel,
    config: KimiK2.Config,
) -> dict[str, Tensor]:
    """Extract HF weights and backfill absent router correction biases."""
    raw: dict[str, Tensor] = {
        key: value.detach().cpu() for key, value in hf_model.state_dict().items()
    }
    for i in range(config.first_k_dense_replace, config.num_layers):
        key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if key not in raw:
            raw[key] = torch.zeros(_router(config).num_experts)
    return raw


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
