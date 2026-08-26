"""Model building blocks."""

from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.model.attention.kvcache import KVCache
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.rope import RoPE, RoPEMixed
from priml.model.attention.self_attention import SelfAttention
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.conv import Conv1d, Conv2d, Conv3d
from priml.model.embedding import Embedding
from priml.model.generate import generate
from priml.model.init import (
    InitFn,
    call_init,
    kaiming_normal,
    kaiming_uniform,
    mup_output,
    normal,
    truncated_normal,
    unit_fan_in_uniform,
    xavier_normal,
    xavier_uniform,
)
from priml.model.linear import EnsembleLinear, Linear
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.moe import MoE, Router
from priml.model.norm import (
    BatchNorm,
    BatchNorm2d,
    GroupNorm,
    LayerNorm,
    NormConfigProtocol,
    NormProtocol,
    RMSNorm,
)
from priml.model.patchify import Patchify, Unpatchify
from priml.model.sequential import Sequential
from priml.model.special import Identity, Skip
from priml.model.swiglu import SwiGLU, SwiGLUReluSquared
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.causal_lm import CausalLM
from priml.model.transformer.kimi_k2 import KimiK2
from priml.model.transformer.mmdit import AdaLNZero, MMDiTBlock
from priml.model.transformer.qwen3 import Qwen3


__all__ = [
    "AdaLNZero",
    "BatchNorm",
    "BatchNorm2d",
    "CausalLM",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EnsembleLinear",
    "GroupNorm",
    "Identity",
    "InitFn",
    "KVCache",
    "KimiK2",
    "LayerNorm",
    "Linear",
    "MLPMixerBlock",
    "MMDiTBlock",
    "MoE",
    "MultiHeadLatentAttention",
    "MultiStreamAttention",
    "NormConfigProtocol",
    "NormProtocol",
    "Patchify",
    "Qwen3",
    "RMSNorm",
    "RoPE",
    "RoPEMixed",
    "Router",
    "SdpaFused",
    "SdpaNaive",
    "SelfAttention",
    "Sequential",
    "Skip",
    "SwiGLU",
    "SwiGLUReluSquared",
    "TransformerBlock",
    "Unpatchify",
    "ValueGatedAttention",
    "call_init",
    "generate",
    "kaiming_normal",
    "kaiming_uniform",
    "mup_output",
    "normal",
    "truncated_normal",
    "unit_fan_in_uniform",
    "xavier_normal",
    "xavier_uniform",
]
