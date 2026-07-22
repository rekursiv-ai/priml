"""Numerical parity: priml.model.qwen3 vs transformers.Qwen3ForCausalLM.

Builds a tiny Qwen3 via HuggingFace transformers, runs a forward pass,
then remaps the same weights through our pipeline and compares outputs
bit-for-bit. Any drift flags a real regression in: RoPE convention,
QK-norm layout, GQA splitting, SwiGLU gate/up fusion, tied embeddings,
or dtype handling.

Integration-marked so CI can opt out; developers running
``uv run pytest -m integration`` exercise it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from priml.math.seed import enable_determinism
from priml.model.attention import SdpaNaive, SelfAttention
from priml.model.qwen3 import Qwen3, remap_hf_state_dict
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        torch.backends.mps.is_available(),
        reason="CPU matmul ordering is not bit-exact on MPS-capable Macs",
    ),
]


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


def _build_hf_model(cfg_dict: dict[str, Any]) -> Any:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(**cfg_dict)
    model = transformers.Qwen3ForCausalLM(config)
    model.eval()
    return model.to(dtype=torch.float32)


def _hf_state_dict_to_loop_format(
    hf_model: Any,
    config: Qwen3.Config,
) -> dict[str, torch.Tensor]:
    raw = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
    return remap_hf_state_dict(raw, config)


@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_qwen3_matches_hf(tie_embeddings: bool):
    """Our Qwen3 output must match HF's Qwen3ForCausalLM bit-for-bit."""
    enable_determinism(sdpa=True)
    torch.manual_seed(0)
    cfg_dict = _tiny_hf_config()
    cfg_dict["tie_word_embeddings"] = tie_embeddings

    hf_model = _build_hf_model(cfg_dict)
    config = Qwen3.Config.from_hf(
        {"model_type": "qwen3", "torch_dtype": "float32", **cfg_dict},
    )
    # HF uses separate q/k/v and gate/up projections. Our runtime defaults to
    # fused projections, so split here to make this test isolate weight remap
    # and architecture parity instead of harmless matmul-order drift.
    config.hf_split_projections = True
    config = config.finalize()
    # Inject SdpaNaive to match HF's eager codepath exactly.
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    assert isinstance(config.block.ffn, SwiGLU.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()
    loop_model = config.make()
    loop_model.load_state_dict(_hf_state_dict_to_loop_format(hf_model, config))
    loop_model.eval().to(dtype=torch.float32)

    tokens = torch.randint(0, cfg_dict["vocab_size"], (2, 5))
    with torch.no_grad():
        hf_out = hf_model(input_ids=tokens).logits
        loop_out = loop_model(tokens)

    assert torch.equal(hf_out, loop_out), (
        f"not bit-for-bit: max abs diff {(hf_out - loop_out).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
