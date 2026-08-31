"""Tests for priml.model.transformer.qwen3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import importlib.util
import os
import sys

from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml import hub
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import (
    GeometricFrequencies,
    HuggingFaceFrequencies,
    RoPE,
)
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import TensorBlockConfig
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer import qwen3
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.causal_lm import CausalLM
from priml.model.transformer.qwen3 import Qwen3, remap_hf_state_dict
from priml.testing.bfb import assert_bfb_against_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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


def _canonical_config() -> Qwen3.Config:
    return Qwen3.Config.from_hf(
        _hf_config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        ),
    )


def test_qwen3_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="qwen3",
        config=_canonical_config(),
    )


def test_qwen3_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="qwen3",
        build_module=lambda: _canonical_config().make(),
        build_input=lambda: torch.tensor([[0, 1, 2]]),
        seed=0,
    )


def _synth_hf_state_dict(cfg: Qwen3.Config) -> dict[str, Tensor]:
    """Build a random-weight state_dict in HF Qwen3 layout."""
    h = cfg.channels_in
    inter = _ffn(cfg).channels_hidden
    attn = _attn(cfg)
    n_q = attn.num_heads
    n_kv = attn.num_heads_kv
    d = attn.channels_head
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


def _attn(cfg: Qwen3.Config, layer: int = 0) -> SelfAttention.Config:
    """One layer's attention -- where the head geometry lives now.

    Accepts a template or a finalized per-layer list, so a caller need not
    know which side of ``finalize`` it is on.
    """
    block = cfg.block[layer] if isinstance(cfg.block, list) else cfg.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, SelfAttention.Config)
    return attn


def _ffn(cfg: Qwen3.Config, layer: int = 0) -> SwiGLU.Config:
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


class TestConfig:
    def test_parse_basic(self):
        cfg = Qwen3.Config.from_hf(_hf_config())
        assert cfg.vocab_size == 128
        assert cfg.channels_in == 64
        assert _attn(cfg).channels_head == 16
        assert _attn(cfg).num_heads_kv == 2
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 1_000_000

    def test_wrong_model_type_rejected(self):
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(_hf_config(model_type="qwen2"))
        with pytest.raises(ValueError, match="qwen3"):
            Qwen3.Config.from_hf(_hf_config(model_type="qwen3_moe"))

    def test_head_dim_inferred_when_missing(self):
        cfg = _hf_config()
        cfg.pop("head_dim")
        parsed = Qwen3.Config.from_hf(cfg)
        assert _attn(parsed).channels_head == (
            cfg["hidden_size"] // cfg["num_attention_heads"]
        )

    def test_num_key_value_heads_inferred_when_missing(self):
        cfg = _hf_config()
        cfg.pop("num_key_value_heads")
        parsed = Qwen3.Config.from_hf(cfg)
        assert _attn(parsed).num_heads_kv == cfg["num_attention_heads"]

    @pytest.mark.parametrize("field_name", ["num_key_value_heads", "head_dim"])
    def test_explicit_zero_head_geometry_rejected(self, field_name: str):
        with pytest.raises(
            ValueError,
            match=rf"{field_name} must be > 0, got 0\.",
        ):
            Qwen3.Config.from_hf(_hf_config(**{field_name: 0}))

    def test_nested_rope_parameters_accept_numeric_text(self):
        cfg = _hf_config(rope_theta=None, rope_parameters={"rope_theta": "25000"})
        parsed = Qwen3.Config.from_hf(cfg)
        rope = _attn(parsed).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, HuggingFaceFrequencies.Config)
        assert rope.frequencies.base == 25_000.0

    def test_malformed_nested_rope_parameters_use_default(self):
        cfg = _hf_config(rope_theta=None, rope_parameters=["not", "an", "object"])
        parsed = Qwen3.Config.from_hf(cfg)
        rope = _attn(parsed).rope
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
        config = Qwen3.Config.from_hf(_hf_config(**overrides))

        assert "Qwen3.Config" in config.pformat(hide_default_values=False)

    def test_explicit_head_dim_not_equal_hidden(self):
        """Qwen3 with hidden != num_heads*head_dim builds, forwards, and loads.

        Regression for MODEL-008: hidden=32, num_heads=4, head_dim=16
        (4*16=64 != 32). The attention inner width
        differs from the residual width.
        """
        cfg = Qwen3.Config.from_hf(
            _hf_config(hidden_size=32, num_attention_heads=4, head_dim=16),
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


class TestSlots:
    """The parent holds slots, not copies of its children's vocabulary."""

    def test_a_rope_edit_survives_finalize(self):
        """Editing the rope slot must reach the built attention.

        The parent used to redeclare ``rope_theta`` and ``frequencies`` and
        rebuild the child in ``finalize``, so this edit was discarded.
        """
        cfg = Qwen3.Config.from_hf(_hf_config())
        template_rope = _attn(cfg).rope
        assert isinstance(template_rope, RoPE.Config)
        template_rope.frequencies = GeometricFrequencies.Config(base=12_345.0)
        cfg = cfg.copy_tree().finalize()
        rope = _attn(cfg).rope
        assert isinstance(rope, RoPE.Config)
        assert isinstance(rope.frequencies, GeometricFrequencies.Config)
        assert rope.frequencies.base == 12_345.0
        # The width still comes from the attention it rotates.
        assert rope.channels_head == _attn(cfg).channels_head

    def test_a_norm_edit_reaches_every_norm(self):
        """One template, so an epsilon set once applies throughout."""
        cfg = Qwen3.Config.from_hf(_hf_config())
        template = _block(cfg)
        assert isinstance(template.norm1, RMSNorm.Config)
        template.norm1.eps = 1e-3
        assert isinstance(cfg.final_norm, RMSNorm.Config)
        cfg.final_norm.eps = 1e-3
        cfg = cfg.copy_tree().finalize()
        assert isinstance(_block(cfg).norm1, RMSNorm.Config)
        norm1 = _block(cfg).norm1
        assert isinstance(norm1, RMSNorm.Config)
        assert norm1.eps == 1e-3
        assert isinstance(cfg.final_norm, RMSNorm.Config)
        assert cfg.final_norm.eps == 1e-3

    def test_each_norm_is_its_own_object(self):
        """Templates are copied, so one consumer cannot edit another's."""
        cfg = Qwen3.Config.from_hf(_hf_config()).copy_tree().finalize()
        block = _block(cfg)
        assert block.norm1 is not block.norm2
        assert block.norm1 is not cfg.final_norm
        assert block.norm1 is not _block(cfg, 1).norm1

    def test_architecture_specific_sizing_skips_other_blocks(self):
        cfg = Qwen3.Config.from_hf(_hf_config())
        block = cast(TensorBlockConfig, RMSNorm.Config())
        cfg._size_block(block)
        assert block.channels_in == cfg.channels_in


class TestLoad:
    def test_local_load_reads_config_and_local_weights(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hf_config = _hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(hf_config).finalize()
        (tmp_path / "config.json").write_text("{}")
        decode = Mock(return_value=hf_config)
        load_local_state_dict = Mock(return_value=_synth_hf_state_dict(cfg))
        monkeypatch.setattr(qwen3, "decode", decode)
        monkeypatch.setattr(hub, "load_local_state_dict", load_local_state_dict)

        model = Qwen3.load(tmp_path, device="cpu", dtype=torch.float32)

        assert isinstance(model, Qwen3)
        assert model.embed.weight.dtype == torch.float32
        decode.assert_called_once_with("object", "{}")
        load_local_state_dict.assert_called_once_with(tmp_path)

    def test_remote_load_uses_hf_model_config_and_weights(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hf_config = _hf_config(num_hidden_layers=1)
        cfg = Qwen3.Config.from_hf(hf_config).finalize()
        hf_model = Mock()
        hf_model.config.to_dict.return_value = hf_config
        hf_model.state_dict.return_value = _synth_hf_state_dict(cfg)
        load_transformers_model = Mock(return_value=hf_model)
        monkeypatch.setattr(
            hub,
            "load_transformers_model",
            load_transformers_model,
        )

        model = Qwen3.load("Qwen/tiny-qwen", dtype=torch.float32)

        assert isinstance(model, Qwen3)
        load_transformers_model.assert_called_once_with(
            "Qwen/tiny-qwen",
            "AutoModelForCausalLM",
            dtype=torch.float32,
        )


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
        h = cfg.channels_in
        attn = _attn(cfg)
        d = attn.channels_head
        n_q, n_kv = attn.num_heads, attn.num_heads_kv
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
        assert fused.shape == (2 * _ffn(cfg).channels_hidden, cfg.channels_in)
        assert torch.equal(fused[: _ffn(cfg).channels_hidden], gate)
        assert torch.equal(fused[_ffn(cfg).channels_hidden :], up)

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

    @pytest.mark.parametrize("bad_part", ["block", "attention"])
    def test_remap_rejects_incompatible_layer_configs(self, bad_part: str):
        cfg = Qwen3.Config.from_hf(_hf_config(num_hidden_layers=1))
        block = cfg.block
        assert isinstance(block, TransformerBlock.Config)
        if bad_part == "block":
            cfg.block = cast(TensorBlockConfig, RMSNorm.Config())
            match = "not a transformer"
        else:
            block.attn = RMSNorm.Config()
            match = "not self-attention"
        with pytest.raises(TypeError, match=match):
            qwen3._attn_of(cfg)


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
        spec = importlib.util.spec_from_file_location(
            "_qwen3_hf_import_probe", __file__
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
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
            wraps=torch.backends.cuda.enable_mem_efficient_sdp
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


def _tiny_hf_config() -> dict[str, Any]:
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "rope_theta": 1_000_000.0,
    }


def _build_qwen3_hf_model(cfg_dict: dict[str, Any]) -> Any:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(**cfg_dict, attn_implementation="eager")
    model = transformers.Qwen3ForCausalLM(config)
    model.eval()
    return model.to(dtype=torch.float32)


def _hf_state_dict_to_loop_format(
    hf_model: Any,
    config: Qwen3.Config,
) -> dict[str, Tensor]:
    raw = {key: value.detach().cpu() for key, value in hf_model.state_dict().items()}
    return remap_hf_state_dict(raw, config)


def _qwen3_parity_outputs(tie_embeddings: bool) -> tuple[Tensor, Tensor]:
    """Build HF and loop Qwen3 models with shared weights and inputs."""
    cfg_dict = _tiny_hf_config()
    cfg_dict["tie_word_embeddings"] = tie_embeddings

    hf_model = _build_qwen3_hf_model(cfg_dict)
    config = Qwen3.Config.from_hf(
        {"model_type": "qwen3", "torch_dtype": "float32", **cfg_dict},
    ).finalize()
    assert isinstance(config.block, list)
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.attn, SelfAttention.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        block.attn.split_qkv_projection = True
        block.ffn.split_gate_projection = True
        block.attn.attn_kernel = SdpaNaive.Config()
    loop_model = config.make()
    loop_model.load_state_dict(_hf_state_dict_to_loop_format(hf_model, config))
    loop_model.eval().to(dtype=torch.float32)

    tokens = torch.randint(0, cfg_dict["vocab_size"], (2, 5))
    with torch.no_grad():
        hf_out = cast(Tensor, hf_model(input_ids=tokens).logits)
        loop_out = loop_model(tokens)
    return hf_out, loop_out


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
