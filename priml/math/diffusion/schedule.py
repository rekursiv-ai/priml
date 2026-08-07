from __future__ import annotations

from torch import Tensor, nn

import torch

from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.numeric import (
    log1mexp,
    log_arctan_exp,
    log_tan_exp,
    logsubexp,
    safe_log,
)
from priml.math.probability import (
    log_cdf_truncated_normal,
    quantile_truncated_normal,
)


__all__ = [
    "compute_log_alpha",
    "input_conditioning_identity",
    "input_conditioning_rectified_flow",
    "log_sigma_from_log_snr_per_rectified_flow",
    "log_sigma_from_log_snr_per_variance_preserving",
    "log_snr_from_log_sigma_per_rectified_flow",
    "log_snr_from_log_sigma_per_variance_preserving",
    "log_snr_from_log_time_per_logit",
    "log_snr_from_log_time_per_logtan",
    "log_snr_from_log_time_per_truncnormicdf",
    "log_time_from_log_snr_per_logit",
    "log_time_from_log_snr_per_logtan",
    "log_time_from_log_snr_per_truncnormicdf",
]


def compute_log_alpha(log_snr: Tensor, log_sigma: Tensor) -> Tensor:
    """Compute log α from log SNR and log σ.

    Args:
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.

    Returns:
      log_alpha: 0.5 * log_snr + log_sigma.

    Derivation:

        snr ≝ α²/σ², so log α = ½ log(snr) + log σ.

    """
    return 0.5 * log_snr + log_sigma


def log_sigma_from_log_snr_per_variance_preserving(log_snr: Tensorable) -> Tensor:
    """Variance-preserving log σ from log SNR: α² + σ² = 1.

    Args:
      log_snr: Log signal-to-noise ratio.

    Returns:
      log_sigma: ½ logsigmoid(-log_snr).

    References:
      https://arxiv.org/abs/2107.00630
        Kingma et al. 2021, "Variational Diffusion Models."

    Derivation:

        α² + σ² = 1,  snr ≝ α²/σ².

      Substituting α² = snr σ² into α² + σ² = 1,

           1 = σ² (1 + snr)
        ⇔
          σ² = 1/(1 + snr)
             = 1/(1 + exp(logsnr))
             = sigmoid(-logsnr)

          α² = 1 - σ²
             = 1 - sigmoid(-logsnr)
             = sigmoid(logsnr)

      Taking ½ log of each,

          log σ = ½ logsigmoid(-logsnr)
          log α = ½ logsigmoid(+logsnr)

    """
    log_snr = convert_to_tensor(log_snr)
    return 0.5 * nn.functional.logsigmoid(-log_snr)


def log_snr_from_log_sigma_per_variance_preserving(
    *,
    sigma: Tensorable | None = None,
    log_sigma: Tensorable | None = None,
) -> Tensor:
    """Variance-preserving log SNR from σ.

    Args:
      sigma: Noise coefficient σ.
      log_sigma: Log noise coefficient.

    Returns:
      log_snr: Log signal-to-noise ratio.

    Raises:
      ValueError: If both or neither of sigma and log_sigma are
        specified.

    References:
      https://arxiv.org/abs/2107.00630
        Kingma et al. 2021, "Variational Diffusion Models."

    Derivation:
      From log_sigma_from_log_snr_per_variance_preserving,

          σ² = sigmoid(-logsnr)

      Taking -logit of both sides,

          logsnr = -logit(σ²)
                 = -log(σ²) + log(1 - σ²)
                 = -2 log σ + log(1 - exp(2 log σ))
                 = -2 log σ + log1mexp(2 log σ)

    """
    if (sigma is None) == (log_sigma is None):
        raise ValueError("Exactly one of sigma, log_sigma must be provided.")
    if log_sigma is None:
        assert sigma is not None
        sigma = convert_to_tensor(sigma)
        log_sigma = safe_log(sigma)
    else:
        log_sigma = convert_to_tensor(log_sigma)
    return -2 * log_sigma + log1mexp(2 * log_sigma)


def log_sigma_from_log_snr_per_rectified_flow(log_snr: Tensorable) -> Tensor:
    """Rectified flow log σ from log SNR: α + σ = 1.

    Args:
      log_snr: Log signal-to-noise ratio.

    Returns:
      log_sigma: logsigmoid(-½ log_snr).

    References:
      https://arxiv.org/abs/2209.03003
        Liu et al. 2022, "Flow Straight and Fast."

    Derivation:

        α + σ = 1,  snr ≝ α²/σ² = (α/σ)².

        α/σ = exp(½ logsnr)

      Substituting α = 1 - σ and solving for σ,

           σ = 1/(1 + exp(½ logsnr))
             = sigmoid(-½ logsnr)

           α = 1 - σ
             = 1 - sigmoid(-½ logsnr)
             = sigmoid(+½ logsnr)

      Taking logs completes the proof.

    """
    log_snr = convert_to_tensor(log_snr)
    return nn.functional.logsigmoid(-0.5 * log_snr)


def log_snr_from_log_sigma_per_rectified_flow(
    *,
    sigma: Tensorable | None = None,
    log_sigma: Tensorable | None = None,
) -> Tensor:
    """Rectified flow log SNR from σ.

    Args:
      sigma: Noise coefficient σ.
      log_sigma: Log noise coefficient.

    Returns:
      log_snr: Log signal-to-noise ratio.

    Raises:
      ValueError: If both or neither of sigma and log_sigma are specified.

    References:
      https://arxiv.org/abs/2209.03003
        Liu et al. 2022, "Flow Straight and Fast."

    Derivation:
      From log_sigma_from_log_snr_per_rectified_flow,

             log σ = logsigmoid(-½ logsnr)
        ⇔
          logit(σ) = -½ logsnr
        ⇔
            logsnr = -2 logit(σ)

      In log-space,

          logit(σ) = log σ - log(1 - σ)
                   = log σ - log1mexp(log σ)

    """
    if (sigma is None) == (log_sigma is None):
        raise ValueError("Exactly one of sigma, log_sigma must be provided.")
    if log_sigma is None:
        assert sigma is not None
        sigma = convert_to_tensor(sigma)
        logit = torch.logit(sigma)
    else:
        log_sigma = convert_to_tensor(log_sigma)
        logit = log_sigma - log1mexp(log_sigma)
    return -2 * logit


# Schedule naming convention: the prefix names the function applied
# to log-time that gives (proportional to) log_snr, e.g.,
# log_snr_from_log_time_per_logit means
# log_snr ∝ logit(exp(log_t)).


def log_snr_from_log_time_per_logit(
    log_t: Tensorable,
    *,
    low: Tensorable = -20,
    high: Tensorable = +20,
) -> Tensor:
    """log_snr ∝ logit(exp(log_t)). Equivalently, σ = t under RF.

    Args:
      log_t: Log timestep (≤ 0). log_t=-∞ → high, log_t=0 → low.
      low: Minimum log_snr (clamp).
      high: Maximum log_snr (clamp).

    Returns:
      log_snr: Log signal-to-noise ratio.

    Derivation:
      Under RF (α + σ = 1), σ = t, α = 1 - t, so

         logsnr = -2 logit(t)
                = -2 (log t - log(1 - t))
                = -2 (log_t - log1mexp(log_t))

    """
    log_t, low, high = convert_to_tensor(log_t, low, high)
    log_snr = log_snr_from_log_sigma_per_rectified_flow(log_sigma=log_t)
    return torch.clamp(log_snr, low, high)


def log_time_from_log_snr_per_logit(log_snr: Tensorable) -> Tensor:
    """Inverse of log_snr_from_log_time_per_logit.

    Args:
      log_snr: Log signal-to-noise ratio.

    Returns:
      log_t: Log timestep (≤ 0).

    Derivation:
      logsnr = -2 logit(t), so t = sigmoid(-½ logsnr) and

          log_t = logsigmoid(-½ logsnr)

    """
    log_snr = convert_to_tensor(log_snr)
    return nn.functional.logsigmoid(-0.5 * log_snr)


def log_snr_from_log_time_per_logtan(
    log_t: Tensorable,
    *,
    shift: Tensorable = 0,
    low: Tensorable = -20,
    high: Tensorable = +20,
) -> Tensor:
    """log_snr ∝ -log(tan(t)). The "cosine schedule."

    Regarding shift: rule of thumb for images ≥ 64×64:
        shift = log((64 × 64) / (H × W)).

    Args:
      log_t: Log timestep (≤ 0). log_t=-∞ → high, log_t=0 → low.
      shift: Shift parameter.
      low: Minimum log_snr (clamp).
      high: Maximum log_snr (clamp).

    Returns:
      log_snr: Log signal-to-noise ratio.

    References:
      https://arxiv.org/abs/2102.09672
        Nichol & Dhariwal 2021, "Improved DDPM."
      https://arxiv.org/abs/2301.11093
        Hoogeboom et al. 2023, "Simple Diffusion."

    Derivation:

        logsnr(t) = -2 log(tan(a t + b)) + shift

      where b = arctan(exp(-½(high - shift)))  ≈ 0,
            a = arctan(exp(-½(low  - shift))) - b  ≈ π/2.

      In log-space, angle = a*t + b = exp(logaddexp(
      log(a) + log_t, log(b))), so

        logsnr = -2 log_tan_exp(log_angle) + shift

    """
    log_t, shift, low, high = convert_to_tensor(log_t, shift, low, high)
    shift = torch.clamp(shift, low, high)
    log_b = log_arctan_exp(-0.5 * (high - shift))  # ≈ -∞
    log_a = logsubexp(log_arctan_exp(-0.5 * (low - shift)), log_b)  # ≈ log(π/2)
    log_angle = torch.logaddexp(log_a + log_t, log_b)
    log_snr = -2 * log_tan_exp(log_angle) + shift
    return torch.clamp(log_snr, low, high)


def log_time_from_log_snr_per_logtan(
    log_snr: Tensorable,
    *,
    shift: Tensorable = 0,
    low: Tensorable = -20,
    high: Tensorable = +20,
) -> Tensor:
    """Inverse of log_snr_from_log_time_per_logtan.

    Args:
      log_snr: Log signal-to-noise ratio.
      shift: Shift parameter (must match forward).
      low: Minimum log_snr (must match forward).
      high: Maximum log_snr (must match forward).

    Returns:
      log_t: Log timestep (≤ 0).

    Derivation:
      From the forward,

        angle = arctan(exp(-½(logsnr - shift)))

      so log_angle = log_arctan_exp(-½(logsnr - shift)), then

        log_t = logsubexp(log_angle, log(b)) - log(a)

    """
    log_snr, shift, low, high = convert_to_tensor(log_snr, shift, low, high)
    shift = torch.clamp(shift, low, high)
    log_b = log_arctan_exp(-0.5 * (high - shift))
    log_a = logsubexp(log_arctan_exp(-0.5 * (low - shift)), log_b)
    log_angle = log_arctan_exp(-0.5 * (log_snr - shift))
    return logsubexp(log_angle, log_b) - log_a


def log_snr_from_log_time_per_truncnormicdf(
    log_t: Tensorable,
    *,
    shift: Tensorable = 0,
    low: Tensorable = -20,
    high: Tensorable = +20,
    scale: Tensorable = 2,
) -> Tensor:
    """log_snr ∝ Φ_trunc⁻¹(t). The "logistic normal schedule."

    Regarding shift: rule of thumb for images ≥ 64×64:
        shift = log((64 × 64) / (H × W)).

    Args:
      log_t: Log timestep (≤ 0). log_t=-∞ → high, log_t=0 → low.
      shift: Location parameter of the truncated normal.
      low: Minimum log_snr (clamp).
      high: Maximum log_snr (clamp).
      scale: Scale of the truncated normal.
        Default 2 per Esser et al. 2024.

    Returns:
      log_snr: Log signal-to-noise ratio.

    References:
      https://arxiv.org/abs/2403.03206
        Esser et al. 2024, "Scaling Rectified Flow Transformers."
      https://arxiv.org/abs/2301.11093
        Hoogeboom et al. 2023, "Simple Diffusion."

    Derivation:

        logsnr(t) = Φ_trunc⁻¹(1 - t; loc=shift, scale, low, high)

      One exp is required: t = exp(log_t), which is safe
      since log_t ≤ 0 guarantees t ∈ [0, 1].

    """
    log_t, shift, low, high, scale = convert_to_tensor(
        log_t,
        shift,
        low,
        high,
        scale,
    )
    shift = torch.clamp(shift, low, high)
    t = log_t.exp()  # ndtri needs p, not log_p; safe since log_t ≤ 0.
    # Swap low/high to reverse direction: t=0 → high, t=1 → low.
    log_snr = quantile_truncated_normal(
        t,
        loc=shift,
        scale=scale,
        low=high,
        high=low,
    )
    return torch.clamp(log_snr, low, high)


def log_time_from_log_snr_per_truncnormicdf(
    log_snr: Tensorable,
    *,
    shift: Tensorable = 0,
    low: Tensorable = -20,
    high: Tensorable = +20,
    scale: Tensorable = 2,
) -> Tensor:
    """Inverse of log_snr_from_log_time_per_truncnormicdf.

    Args:
      log_snr: Log signal-to-noise ratio.
      shift: Location parameter of the truncated normal.
      low: Minimum log_snr.
      high: Maximum log_snr.
      scale: Scale of the truncated normal. Default 2 per
        Esser et al. 2024.

    Returns:
      log_t: Log timestep (≤ 0).

    Derivation:
      From the forward,

        logsnr = Φ_trunc⁻¹(t; loc=shift, scale, low=high, high=low)

      Inverting with the log truncated-normal CDF,

        log_t = log Φ_trunc(logsnr; loc=shift, scale, low=high,
                            high=low)

    """
    log_snr, shift, low, high, scale = convert_to_tensor(
        log_snr,
        shift,
        low,
        high,
        scale,
    )
    shift = torch.clamp(shift, low, high)
    return log_cdf_truncated_normal(
        log_snr,
        loc=shift,
        scale=scale,
        low=high,
        high=low,
    )


def input_conditioning_rectified_flow(x: Tensorable, log_snr: Tensorable) -> Tensor:
    """Input conditioning for rectified flow.

    Normalizes x_noisy to unit variance, analogous to Karras EDM's
    c_in = 1/√(σ² + σ_data²) generalized to the (α, σ) forward
    process with σ_data = 1.

    Args:
      x: Input tensor (typically x_noisy).
      log_snr: Log signal-to-noise ratio.

    Returns:
      y: x / (α² + σ²)^½.

    References:
      https://arxiv.org/abs/2206.00364
        Karras et al. 2022, "Elucidating the Design Space."

    Derivation:

        Var(x_noisy) = α² Var(x) + σ² Var(ε) = α² + σ²

      Under RF corruption (α + σ = 1), α² + σ² ≤ 1, so this
      normalizes x_noisy to unit variance.

      In log-space,

        log(α² + σ²) = logaddexp(2 log α, 2 log σ)

    """
    x, log_snr = convert_to_tensor(x, log_snr)
    log_sigma = log_sigma_from_log_snr_per_rectified_flow(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    log_norm = 0.5 * torch.logaddexp(2 * log_alpha, 2 * log_sigma)
    return x * torch.exp(-log_norm)


def input_conditioning_identity(
    x: Tensorable,
    log_snr: Tensorable | None = None,
) -> Tensor:
    """Identity input conditioning (no-op).

    Args:
      x: Input tensor.
      log_snr: Log signal-to-noise ratio (unused).

    Returns:
      x: Input unchanged.

    """
    del log_snr
    x = convert_to_tensor(x)
    return x
