"""FP4 Speculative Decoding experiment (exp004).

Builds on exp003 (fused Triton FP4 kernel) + adds speculative decoding search space.

Research gap: No prior work measures FP4 quantization interaction with speculative decoding.
Survey (arXiv:2401.07851) shows FP16→FP32 loses 27-61% speedup — FP4 is unexplored.

Search space: 480 configs for agent swarm campaign.
"""

from dataclasses import dataclass
from typing import Literal, Optional
from configgle import Fig, field


class SpeculativeFP4Config(Fig["SpeculativeFP4Config"]):
    """Configgle search space for FP4 + Speculative Decoding.

    Research-backed design:
    - EAGLE: best general-purpose (2.39× FP16 on A100), uses token tree + KD
    - Medusa: self-drafting FFN heads, no extra model weights
    - Token tree: increases acceptance rate at cost of verification compute
    - FP4 quantization effect on acceptance: UNKNOWN (this experiment measures it)
    """

    # === FIXED: Your fused FP4 kernel from exp003 ===
    kernel: Literal["triton_fp4"] = "triton_fp4"
    fp4_format: Literal["e2m1", "nvfp4_block16"] = field(
        default="e2m1",
        description="FP4 format: e2m1 (per-tensor) or nvfp4_block16 (block size 16, Blackwell native)",
    )

    # === SEARCH SPACE: Agents explore this ===

    # Drafter architecture
    drafter_type: Literal["eagle", "medusa", "tiny_llama"] = field(
        default="eagle",
        description=(
            "EAGLE: AR head + token tree (best general, 2.39× FP16). "
            "Medusa: FFN heads (self-drafting, 2.42× FP16 on 7B). "
            "Tiny Llama: external small model drafter."
        ),
    )

    # Token tree (EAGLE only) - key for acceptance rate
    tree_depth: int = field(
        default=3, ge=1, le=5,
        description="Token tree depth (EAGLE). 3=~7 nodes, 4=~15, 5=~31 (branching=2)"
    )
    tree_branching: int = field(
        default=2, ge=2, le=4,
        description="Branching factor per tree node"
    )

    # Acceptance criteria
    acceptance_threshold: float = field(
        default=0.9, ge=0.5, le=1.0,
        description="Min target prob to accept draft token"
    )
    max_spec_tokens: int = field(
        default=8, ge=1, le=16,
        description="Max speculative tokens per step"
    )

    # Training/calibration (for Medusa/EAGLE heads)
    calibration_data: Literal["c4", "wikitext", "sharegpt", "ultrachat"] = field(
        default="sharegpt",
        description="Calibration data for drafter knowledge distillation"
    )
    kd_temperature: float = field(
        default=0.5, ge=0.1, le=2.0,
        description="KD temperature for drafter-target alignment"
    )
    kd_loss_weight: float = field(
        default=0.5, ge=0.1, le=1.0,
        description="KD loss weight vs LM loss for drafter training"
    )

    # Verification mode
    verify_mode: Literal["token_tree", "n_gram", "standard"] = field(
        default="token_tree",
        description="Token tree (EAGLE), n-gram (code), standard (single token)"
    )

    # FP4-specific: does quantization hurt acceptance?
    quantize_drafter: bool = field(
        default=True,
        description="Also quantize drafter to FP4 (vs FP32 drafter)"
    )
    drafter_fp4_format: Literal["e2m1", "nvfp4_block16"] = field(
        default="e2m1",
        description="FP4 format for drafter weights"
    )

    # Workload
    batch_size: int = field(default=1, ge=1, le=32)
    input_len: int = field(default=2048, ge=128, le=8192)
    output_len: int = field(default=256, ge=32, le=2048)
    prompt_type: Literal["chat", "code", "summary", "long_context"] = field(
        default="chat",
        description="Prompt type affects speculative effectiveness"
    )

    # Model config (for drafter sizing)
    target_model: Literal["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"] = field(
        default="llama-3.2-1b",
        description="Target model for speculative decoding"
    )
    drafter_model: Optional[str] = field(
        default=None,
        description="Drafter model path (for tiny_llama type)"
    )

    # Reproducibility
    seed: int = field(default=42, description="Random seed for deterministic runs")


@dataclass
class Exp004Result:
    """Result metrics for exp004."""
    tokens_per_sec: float
    acceptance_rate: float
    spec_tokens_per_step: float
    quality_delta_perplexity: float
    quality_delta_accuracy: float
    fp4_vs_fp16_speedup: float
    total_target_calls: int
    total_accepted: int


def exp004() -> SpeculativeFP4Config:
    """FP4 Fused Triton + Speculative Decoding.

    Search space designed for agent swarm:
    - 3 drafter_types × 5 tree_depth × 4 acceptance × 4 calibration × 2 drafter_quant = 480 configs
    - Agents find Pareto frontier: speedup × quality preservation

    Primary metric: tokens_per_sec (throughput)
    Quality guardrails: perplexity delta, accuracy delta
    """
    return SpeculativeFP4Config()


def exp004_smoke() -> SpeculativeFP4Config:
    """Smoke test: tiny model, minimal search, fast CPU run."""
    return SpeculativeFP4Config(
        drafter_type="eagle",
        tree_depth=2,
        tree_branching=2,
        acceptance_threshold=0.9,
        max_spec_tokens=4,
        calibration_data="c4",
        quantize_drafter=True,
        batch_size=1,
        input_len=64,
        output_len=32,
        target_model="llama-3.2-1b",
    )