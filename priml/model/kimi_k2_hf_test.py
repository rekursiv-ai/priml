"""Numerical parity: KimiK2 vs HF's ``DeepseekV3ForCausalLM``.

Loads the HF ``modeling_deepseek.py`` via ``trust_remote_code``
(from ``deepseek-ai/DeepSeek-V3``), instantiates it with a tiny
config, remaps its ``state_dict`` through our loader, loads into our
fused :class:`KimiK2`, and asserts ``torch.allclose`` on the logits.

This is the **independent** reference — HF authors' implementation,
not a reference we wrote.

Two transformers 5.x compat shims are needed (the uploaded
``modeling_deepseek.py`` targets transformers 4.x):
  - ``is_torch_fx_available`` (removed in 5.x)
  - ``DynamicCache.from_legacy_cache`` (removed in 5.x)

Integration-marked.
"""

from __future__ import annotations

from typing import Any

import warnings

import pytest
import torch

from priml.model.kimi_k2 import KimiK2, remap_hf_state_dict


pytestmark = [pytest.mark.integration]


def _install_transformers_compat_shims() -> None:
    """Backfill symbols removed in transformers 5.x that DSV3's
    ``trust_remote_code`` modeling file still imports.
    """
    import transformers.utils.import_utils as _iu  # noqa: PLC0415

    def _false() -> bool:
        return False

    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = _false
    from transformers import DynamicCache  # noqa: PLC0415

    def _passthrough_cache(_cls: type[DynamicCache], pkv: Any) -> Any:
        return pkv

    if not hasattr(DynamicCache, "from_legacy_cache"):
        DynamicCache.from_legacy_cache = classmethod(_passthrough_cache)  # pyright: ignore[reportAttributeAccessIssue]


def _build_hf_model(q_lora_rank: int | None) -> Any:
    """Instantiate HF's real ``DeepseekV3ForCausalLM`` at tiny size."""
    transformers = pytest.importorskip("transformers")
    _install_transformers_compat_shims()
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
    config.rope_scaling = None  # YaRN off for baseline parity
    config.q_lora_rank = q_lora_rank
    config.torch_dtype = "float32"
    config.use_cache = False
    config.tie_word_embeddings = False
    config.scoring_func = "sigmoid"
    config.norm_topk_prob = True
    config.routed_scaling_factor = 1.0
    model = transformers.AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    return model.to(torch.float32).eval()


def _our_config_from_hf(hf_model: Any, q_lora_rank: int | None) -> KimiK2.Config:
    """Mirror hf_model's config into a ``KimiK2.Config``."""
    hf_cfg = hf_model.config.to_dict()
    # model_type from trust_remote_code config is ``deepseek_v3``.
    hf_cfg.setdefault("model_type", "deepseek_v3")
    hf_cfg["q_lora_rank"] = q_lora_rank
    hf_cfg["rope_scaling"] = None
    hf_cfg["tie_word_embeddings"] = False
    return KimiK2.Config.from_hf(hf_cfg).finalize()


def _hf_state_dict_with_bias_fill(
    hf_model: Any,
    config: KimiK2.Config,
) -> dict[str, torch.Tensor]:
    """Extract HF state_dict; backfill per-router ``e_score_correction_bias``.

    HF's DSV3 model stores the bias at ``model.layers.N.mlp.gate.e_score_correction_bias``
    only when it's been initialized; in freshly-constructed models it's absent.
    Our remap falls back to zeros in that case — mirror that here so our
    forward uses the same bias values as HF's.
    """
    raw = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
    for i in range(config.first_k_dense_replace, config.num_hidden_layers):
        key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if key not in raw:
            raw[key] = torch.zeros(config.n_routed_experts)
    return raw


@pytest.mark.parametrize("q_lora_rank", [None, 16])
def test_kimi_k2_matches_hf_deepseek_v3(q_lora_rank: int | None):
    """KimiK2 logits must match HF's DeepseekV3ForCausalLM bit-for-bit
    (within float32 SDPA tolerance) on shared weights.
    """
    torch.manual_seed(0)
    hf_model = _build_hf_model(q_lora_rank)
    config = _our_config_from_hf(hf_model, q_lora_rank)

    # HF → loop weight remap, then load strict.
    hf_sd = _hf_state_dict_with_bias_fill(hf_model, config)
    loop_sd = remap_hf_state_dict(hf_sd, config)
    loop_model = config.make()
    loop_model.load_state_dict(loop_sd, strict=True)
    loop_model = loop_model.to(torch.float32).eval()

    tokens = torch.randint(0, config.vocab_size, (2, 5))
    # The bundled DSV3 modeling file uses a deprecated attention-mask
    # API that raises FutureWarning under transformers 5.x. The pytest
    # config turns warnings into errors; silence this one since it's
    # cosmetic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with torch.no_grad():
            hf_out = hf_model(input_ids=tokens, use_cache=False).logits
            loop_out = loop_model(tokens)
    diff = (hf_out - loop_out).abs().max().item()
    assert torch.allclose(hf_out, loop_out, atol=5e-5, rtol=1e-4), (
        f"max abs diff: {diff:.3e}"
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
