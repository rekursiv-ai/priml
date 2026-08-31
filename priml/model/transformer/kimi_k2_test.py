"""Tests for priml.model.transformer.kimi_k2."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import warnings

from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml import hub
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, YarnScaling
from priml.model.custom_types import TensorBlockConfig
from priml.model.moe import MoE, Router
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer import kimi_k2
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.causal_lm import CausalLM
from priml.model.transformer.kimi_k2 import KimiK2, remap_hf_state_dict
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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


def _canonical_config() -> KimiK2.Config:
    return KimiK2.Config.from_hf(
        _hf_config(
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


def test_kimi_k2_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="kimi_k2",
        config=_canonical_config(),
    )


def test_kimi_k2_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="kimi_k2",
        build_module=lambda: _canonical_config().make(),
        build_input=lambda: torch.tensor([[0, 1, 2]]),
        seed=0,
    )


def _router(cfg: KimiK2.Config, layer: int = -1) -> Router.Config:
    """The routing config -- where the expert COUNT lives now."""
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
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, h),
        "model.norm.weight": torch.randn(h),
    }
    if not cfg.tie_embeddings:
        sd["lm_head.weight"] = torch.randn(cfg.vocab_size, h)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(h)
        if attn.q_lora_rank is None:
            sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(n * (qkn + qkr), h)
        else:
            sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(attn.q_lora_rank or 0, h)
            sd[f"{p}.self_attn.q_a_layernorm.weight"] = torch.randn(
                attn.q_lora_rank or 0
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


def _attn(cfg: KimiK2.Config, layer: int = 0) -> MultiHeadLatentAttention.Config:
    """One layer's attention -- where the head geometry lives now.

    Accepts a template or a finalized per-layer list, so a caller need not
    know which side of ``finalize`` it is on.
    """
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, MultiHeadLatentAttention.Config)
    return attn


class TestConfig:
    def test_parse_kimi_k2(self):
        cfg = KimiK2.Config.from_hf(_hf_config())
        attn = _attn(cfg)
        assert attn.kv_lora_rank == 32
        assert attn.q_lora_rank is None
        assert cfg.first_k_dense_replace == 1
        assert attn.channels_qk_nope_head == 16
        assert attn.channels_qk_rope_head == 8
        assert attn.channels_v_head == 16

    def test_parse_deepseek_v3(self):
        cfg = KimiK2.Config.from_hf(
            _hf_config(model_type="deepseek_v3", q_lora_rank=16),
        )
        assert _attn(cfg).q_lora_rank == 16

    def test_wrong_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            KimiK2.Config.from_hf(_hf_config(model_type="qwen3"))

    def test_wrong_scoring_function(self):
        with pytest.raises(ValueError, match="scoring_func"):
            KimiK2.Config.from_hf(_hf_config(scoring_func="linear"))

    def test_rope_scaling_without_type_uses_base_frequencies(self):
        cfg = KimiK2.Config.from_hf(_hf_config(rope_scaling={"factor": 2.0}))
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 50_000.0

    @pytest.mark.parametrize("moe_intermediate_size", [0, -64])
    def test_nonpositive_moe_width_rejected(self, moe_intermediate_size: int):
        """An explicit width names its own field instead of silently defaulting."""
        with pytest.raises(ValueError, match="moe_intermediate_size must be > 0"):
            KimiK2.Config.from_hf(
                _hf_config(moe_intermediate_size=moe_intermediate_size),
            )

    def test_absent_moe_width_falls_back_to_dense_width(self):
        """Only a MISSING key defaults; the fallback itself must survive."""
        config = _hf_config()
        del config["moe_intermediate_size"]
        cfg = KimiK2.Config.from_hf(config)
        assert cfg.channels_hidden_expert == cfg.channels_hidden_dense == 128

    def test_nonpositive_channels_still_print(self) -> None:
        """The degenerate config is the one worth rendering; torch rejects it."""
        config = KimiK2.Config.from_hf(_hf_config(hidden_size=0))

        assert "KimiK2.Config" in config.pformat(hide_default_values=False)

    def test_yarn_scaling_wired(self):
        """YaRN params land on the rope slot's frequency builder."""
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
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        yarn = rope.frequencies
        assert isinstance(yarn, YarnScaling.Config)
        assert yarn.factor == 32.0
        assert yarn.original_max_position_embeddings == 4096


class TestSlots:
    """The parent holds slots, not copies of its children's vocabulary."""

    def test_a_router_edit_survives_finalize(self):
        """Editing the router slot must reach the built MoE layers.

        The parent used to redeclare Router's fields and rebuild the child in
        ``finalize``, so this edit was silently discarded.
        """
        cfg = KimiK2.Config.from_hf(_hf_config())
        template = cfg.block
        assert isinstance(template, TransformerBlock.Config)
        assert isinstance(template.ffn, MoE.Config)
        assert isinstance(template.ffn.router, Router.Config)
        template.ffn.router.routed_scaling_factor = 2.5
        cfg = cfg.copy_tree().finalize()
        assert isinstance(cfg.block, list)
        last = cfg.block[-1]
        assert isinstance(last, TransformerBlock.Config)
        assert isinstance(last.ffn, MoE.Config)
        assert isinstance(last.ffn.router, Router.Config)
        assert last.ffn.router.routed_scaling_factor == 2.5

    def test_a_norm_edit_reaches_every_norm(self):
        """One template, so an epsilon set once applies throughout."""
        cfg = KimiK2.Config.from_hf(_hf_config())
        template = cfg.block
        assert isinstance(template, TransformerBlock.Config)
        assert isinstance(template.norm1, RMSNorm.Config)
        template.norm1.eps = 1e-3
        assert isinstance(cfg.final_norm, RMSNorm.Config)
        cfg.final_norm.eps = 1e-3
        cfg = cfg.copy_tree().finalize()
        assert isinstance(cfg.block, list)
        block = cfg.block[0]
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.norm1, RMSNorm.Config)
        assert block.norm1.eps == 1e-3
        assert isinstance(cfg.final_norm, RMSNorm.Config)
        assert cfg.final_norm.eps == 1e-3

    def test_each_layer_gets_its_own_norm_object(self):
        """Templates are copied, so one layer's finalize cannot edit another."""
        cfg = KimiK2.Config.from_hf(_hf_config()).copy_tree().finalize()
        assert isinstance(cfg.block, list)
        first, second = cfg.block[0], cfg.block[1]
        assert isinstance(first, TransformerBlock.Config)
        assert isinstance(second, TransformerBlock.Config)
        assert first.norm1 is not second.norm1
        assert first.norm1 is not first.norm2

    def test_non_yarn_scaling_rejected(self):
        with pytest.raises(ValueError, match="only yarn"):
            KimiK2.Config.from_hf(
                _hf_config(rope_scaling={"type": "linear", "factor": 2.0}),
            )

    def test_make_returns_kimik2_instance(self):
        model = KimiK2.Config.from_hf(_hf_config()).make()
        assert isinstance(model, KimiK2)
        assert isinstance(model, CausalLM)

    def test_architecture_specific_sizing_skips_other_blocks(self):
        cfg = KimiK2.Config.from_hf(_hf_config())
        block = cast(TensorBlockConfig, RMSNorm.Config())
        cfg._size_block(block, 0)
        assert block.channels_in == cfg.channels_in

    def test_custom_nondense_ffn_survives_sizing(self):
        cfg = KimiK2.Config.from_hf(_hf_config(first_k_dense_replace=0))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        block.ffn = SwiGLU.Config(channels_hidden=17)
        cfg._size_block(block, 0)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert block.ffn.channels_hidden == 17


class TestLoad:
    def test_remote_load_uses_hf_config_weights_dtype_and_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hf_config = _hf_config(num_hidden_layers=1)
        cfg = KimiK2.Config.from_hf(hf_config).finalize()
        hf_model = Mock()
        hf_model.config.to_dict.return_value = hf_config
        hf_model.state_dict.return_value = _synth_hf(cfg)
        load_transformers_model = Mock(return_value=hf_model)
        monkeypatch.setattr(
            hub,
            "load_transformers_model",
            load_transformers_model,
        )

        model = KimiK2.load("moonshotai/tiny-kimi", device="cpu", dtype=torch.float32)

        assert isinstance(model, KimiK2)
        assert model.embed.weight.dtype == torch.float32
        assert model.embed.weight.device.type == "cpu"
        load_transformers_model.assert_called_once_with(
            "moonshotai/tiny-kimi",
            "AutoModelForCausalLM",
            dtype=torch.float32,
            trust_remote_code=True,
        )


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
        for i in range(cfg.first_k_dense_replace, cfg.num_layers):
            sd.pop(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
        remapped = remap_hf_state_dict(sd, cfg)
        for i in range(cfg.first_k_dense_replace, cfg.num_layers):
            key = f"blocks.{i}.ffn.router.e_score_correction_bias"
            assert key in remapped
            assert torch.all(remapped[key] == 0)

    def test_multiple_shared_experts_keep_their_indices(self):
        cfg = KimiK2.Config.from_hf(_hf_config(n_shared_experts=2)).finalize()
        sd = _synth_hf(cfg)
        expected: dict[str, Tensor] = {}
        for layer in range(cfg.first_k_dense_replace, cfg.num_layers):
            prefix = f"model.layers.{layer}.mlp.shared_experts"
            gate = sd.pop(f"{prefix}.gate_proj.weight")
            up = sd.pop(f"{prefix}.up_proj.weight")
            down = sd.pop(f"{prefix}.down_proj.weight")
            for expert in range(2):
                sd[f"{prefix}.{expert}.gate_proj.weight"] = gate + expert
                sd[f"{prefix}.{expert}.up_proj.weight"] = up + expert
                sd[f"{prefix}.{expert}.down_proj.weight"] = down + expert
                key = f"blocks.{layer}.ffn.shared_experts.{expert}.up_proj.weight"
                expected[key] = torch.cat([gate + expert, up + expert], dim=0)

        remapped = remap_hf_state_dict(sd, cfg)

        for key, value in expected.items():
            assert torch.equal(remapped[key], value)

    @pytest.mark.parametrize("bad_part", ["block", "attention", "ffn"])
    def test_remap_rejects_incompatible_layer_configs(self, bad_part: str):
        cfg = KimiK2.Config.from_hf(_hf_config(num_hidden_layers=1))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        if bad_part == "block":
            cfg.block = cast(TensorBlockConfig, RMSNorm.Config())
            match = "not a transformer"
        elif bad_part == "attention":
            block.attn = RMSNorm.Config()
            match = "not MLA"
        else:
            block.ffn = SwiGLU.Config()
            match = "not MoE"
        if bad_part == "ffn":
            with pytest.raises(TypeError, match=match):
                kimi_k2._moe_of(cfg, 0)
        else:
            with pytest.raises(TypeError, match=match):
                kimi_k2._attn_of(cfg, 0)


@pytest.mark.network_huggingface
@pytest.mark.parametrize("q_lora_rank", [None, 16])
def test_kimi_k2_matches_hf_deepseek_v3(q_lora_rank: int | None):
    """KimiK2 logits must match HF's DeepseekV3ForCausalLM."""
    torch.manual_seed(0)
    # The shims stay installed across the HF forward pass, not just
    # construction: the remote DeepSeek-V3 code reads both symbols at call
    # time, so exiting the block earlier would raise inside ``hf_model(...)``.
    with _install_transformers_compat_shims():
        hf_model = _build_hf_model(q_lora_rank)
        config = _our_config_from_hf(hf_model, q_lora_rank)
        loop_sd = remap_hf_state_dict(
            _hf_state_dict_with_bias_fill(hf_model, config), config
        )
        loop_model = config.make()
        loop_model.load_state_dict(loop_sd, strict=True)
        loop_model = loop_model.to(torch.float32).eval()

        tokens = torch.randint(0, config.vocab_size, (2, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with host_agnostic_numerics(), torch.no_grad():
                hf_out = hf_model(input_ids=tokens, use_cache=False).logits
                loop_out = loop_model(tokens)
    diff = (hf_out - loop_out).abs().max().item()
    assert torch.allclose(hf_out, loop_out, atol=5e-5, rtol=1e-4), (
        f"max abs diff: {diff:.3e}"
    )


def test_transformers_compat_shims_restore_module_state() -> None:
    """Shims apply inside the block and leave the modules exactly as found."""
    pytest.importorskip("transformers")
    from transformers import DynamicCache  # noqa: PLC0415
    from transformers.utils import import_utils  # noqa: PLC0415

    # ``vars``, not ``getattr``: ``DynamicCache`` inherits from ``Cache``, so a
    # deleted shim would still resolve through the base class and hide a leak.
    absent = object()
    fx_before: object = vars(import_utils).get("is_torch_fx_available", absent)
    legacy_before: object = vars(DynamicCache).get("from_legacy_cache", absent)

    with _install_transformers_compat_shims():
        assert callable(import_utils.is_torch_fx_available)
        assert callable(DynamicCache.from_legacy_cache)

    assert vars(import_utils).get("is_torch_fx_available", absent) is fx_before
    assert vars(DynamicCache).get("from_legacy_cache", absent) is legacy_before


@contextmanager
def _install_transformers_compat_shims() -> Generator[None]:
    """Backfill transformers 4.x symbols removed in transformers 5.x."""
    pytest.importorskip("transformers")
    from transformers import DynamicCache  # noqa: PLC0415
    from transformers.utils import import_utils  # noqa: PLC0415

    def unavailable() -> bool:
        return False

    def passthrough_cache(_cls: type[DynamicCache], pkv: Any) -> Any:
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


def _build_hf_model(q_lora_rank: int | None) -> Any:
    """Instantiate HF's real ``DeepseekV3ForCausalLM`` at tiny size."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
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
    config.n_shared_experts = 1
    config.first_k_dense_replace = 1
    config.vocab_size = 64
    config.n_group = 1
    config.topk_group = 1
    config.rope_theta = 50_000.0
    config.max_position_embeddings = 64
    config.rope_scaling = None
    config.q_lora_rank = q_lora_rank
    config.torch_dtype = "float32"
    config.use_cache = False
    config.tie_word_embeddings = False
    config.scoring_func = "sigmoid"
    config.norm_topk_prob = True
    config.routed_scaling_factor = 1.0
    config._attn_implementation = "eager"
    model = transformers.AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    return model.to(torch.float32).eval()


def _our_config_from_hf(hf_model: Any, q_lora_rank: int | None) -> KimiK2.Config:
    """Mirror an HF model's config into a ``KimiK2.Config``."""
    hf_cfg = hf_model.config.to_dict()
    hf_cfg.setdefault("model_type", "deepseek_v3")
    hf_cfg["q_lora_rank"] = q_lora_rank
    hf_cfg["rope_scaling"] = None
    hf_cfg["tie_word_embeddings"] = False
    return KimiK2.Config.from_hf(hf_cfg).finalize()


def _hf_state_dict_with_bias_fill(
    hf_model: Any,
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
