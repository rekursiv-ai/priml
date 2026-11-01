"""FP4 Speculative Decoding baselines for priml.

Experiments:
- exp004: FP4 fused Triton + Speculative Decoding (full search space)
- exp004_smoke: Minimal config for CI testing
"""

from .experiments import exp004, exp004_smoke, SpeculativeFP4Config
from priml.inference.speculative import (
    EagleDrafter,
    MedusaDrafter,
    TokenTreeVerifier,
    FP4Quantizer,
    speculative_decode_step,
    SpeculativeDecodeEngine,
)

__all__ = [
    "exp004",
    "exp004_smoke",
    "SpeculativeFP4Config",
    "EagleDrafter",
    "MedusaDrafter",
    "TokenTreeVerifier",
    "FP4Quantizer",
    "speculative_decode_step",
    "SpeculativeDecodeEngine",
]