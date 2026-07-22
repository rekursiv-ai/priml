"""Model building blocks."""

from priml.model.attention import (
    MultiStreamAttention,
    SdpaFused,
    SdpaNaive,
    SelfAttention,
)
from priml.model.causal_lm import CausalLM
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
    xavier_normal,
    xavier_uniform,
)
from priml.model.kimi_k2 import KimiK2
from priml.model.kvcache import KVCache
from priml.model.linear import EnsembleLinear, Linear
from priml.model.mla import MultiHeadLatentAttention
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.mmdit import AdaLNZero, MMDiTBlock
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
from priml.model.qwen3 import Qwen3
from priml.model.rope import RoPE, RoPEMixed
from priml.model.sequential import Sequential
from priml.model.special import Identity, Skip
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock


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
    "TransformerBlock",
    "Unpatchify",
    "call_init",
    "generate",
    "kaiming_normal",
    "kaiming_uniform",
    "mup_output",
    "normal",
    "truncated_normal",
    "xavier_normal",
    "xavier_uniform",
]
