"""FP4 Speculative Decoding baselines for priml.

Experiments:
- exp004: FP4 fused Triton + Speculative Decoding (full search space)
- exp004_smoke: Minimal config for CI testing
"""

from priml.inference.speculative import (
    EagleDrafter,
    FP4Quantizer,
    MedusaDrafter,
    SpeculativeDecodeEngine,
    TokenTreeVerifier,
    speculative_decode_step,
)

from .experiments import SpeculativeFP4Config, exp004, exp004_smoke


__all__ = [
    "EagleDrafter",
    "FP4Quantizer",
    "MedusaDrafter",
    "SpeculativeDecodeEngine",
    "SpeculativeFP4Config",
    "TokenTreeVerifier",
    "exp004",
    "exp004_smoke",
    "speculative_decode_step",
]
