"""Target parameterizations for diffusion model training.

All target functions share the forward-process assumption:

    x_noisy = α x_original + σ ε_original

Each function returns a TargetResult(target, predict, x_clean, eps_clean):
  - target: Ground-truth label (requires x_original and eps_original).
  - predict: Model output (or transformation thereof) in the same space.
  - x_clean: Reconstruction of x_original from model output.
  - eps_clean: Reconstruction of eps_original from model output.

The training loss is typically MSE(target, predict).

Example usage:

```python
log_snr = logsnr_fn(t)
log_sigma = corrupt_fn(log_snr)
log_alpha = compute_log_alpha(log_snr, log_sigma)
α, σ = log_alpha.exp(), log_sigma.exp()
x_noisy = α * x0 + σ * eps0
y = denoiser_fn(x_noisy, σ)
target, predict, _, _ = target_fn(
    model=y, x_noisy=x_noisy,
    log_snr=log_snr, log_sigma=log_sigma,
    x_original=x0, eps_original=eps0,
)
loss = F.mse_loss(target, predict)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

from torch import Tensor

import torch

from priml.cost import Cost, set_cost, traffic
from priml.math.diffusion.schedule import compute_log_alpha


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "TargetFn",
    "TargetResult",
    "target_cost",
    "target_eps",
    "target_rectified_flow",
    "target_v",
    "target_v_eps",
    "target_v_x",
    "target_x",
]


class TargetResult(NamedTuple):
    """Result tuple from a target parameterization function."""

    target: Tensor | None
    predict: Tensor | None
    x_clean: Tensor
    eps_clean: Tensor


class TargetFn(Protocol):
    """Protocol for target parameterization functions."""

    def __call__(
        self,
        model: Tensor,
        x_noisy: Tensor,
        log_snr: Tensor,
        log_sigma: Tensor,
        *,
        x_original: Tensor | None = None,
        eps_original: Tensor | None = None,
    ) -> TargetResult:
        """Apply to the input."""
        ...


def target_cost(
    *,
    primal: int,
    adjoint: int,
    vector_elements: int,
    scalar_flops: int,
    scalar_elements: int,
) -> Callable[..., Cost]:
    """Build the cost of a target parameterization per element of ``x0``.

    ``vector`` counts are per element: operations and operand I/O of the
    target, the prediction, and both reconstructions (every branch builds
    ``x_clean`` and ``eps_clean`` even though the loss reads neither).
    ``scalar`` counts are per sample -- the coefficient preparation from
    ``log_snr`` and ``log_sigma`` -- and are spread over the ``rows`` one
    sample holds. ``adjoint`` is the per-element gradient work through
    ``predict``: zero when the model output passes straight through, one
    multiply when ``predict`` is a scaled combination of it.

    Args:
      primal: Operations per element of ``x0``.
      adjoint: Operations per element in the gradient through ``predict``.
      vector_elements: Operands moved per element of ``x0``.
      scalar_flops: Operations per sample preparing the coefficients.
      scalar_elements: Operands moved per sample preparing the coefficients.

    Returns:
      cost_fn: ``cost_fn(*, dtype, rows)``, for :func:`set_cost`.

    """

    def per_element(*, dtype: torch.dtype | None, rows: float) -> Cost:
        return traffic(
            "primal",
            "elementwise",
            elements=vector_elements + scalar_elements / rows,
            flops=primal + scalar_flops / rows,
            dtype=dtype,
        ) + traffic(
            "adjoint",
            "elementwise",
            elements=adjoint * (2 + 1 / rows),
            flops=adjoint,
            dtype=dtype,
        )

    return per_element


# ``target_x``/``target_eps`` pass ``model`` through as ``predict`` and build one
# reconstruction from it (multiply, subtract, multiply); the other is ``model``.
# Seven vector operands; coefficients are two exponentials over a log_alpha of
# two ops: eight scalar ops, thirteen scalar operands.
@set_cost(
    target_cost(
        primal=3,
        adjoint=0,
        vector_elements=7,
        scalar_flops=8,
        scalar_elements=13,
    ),
)
def target_x(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """x-prediction: model directly predicts x_original.

    Args:
      model: Denoiser output.
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (for target).
      eps_original: Clean noise (unused).

    Returns:
      target: x_original.
      predict: model.
      x_clean: model.
      eps_clean: (x_noisy - α model) / σ.

    """
    del eps_original
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    return TargetResult(
        target=x_original,
        predict=model,
        x_clean=model,
        eps_clean=(x_noisy - model * torch.exp(log_alpha)) * torch.exp(-log_sigma),
    )


@set_cost(
    target_cost(
        primal=3,
        adjoint=0,
        vector_elements=7,
        scalar_flops=8,
        scalar_elements=13,
    ),
)
def target_eps(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """ε-prediction: model directly predicts eps_original.

    Args:
      model: Denoiser output.
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (unused).
      eps_original: Clean noise (for target).

    Returns:
      target: eps_original.
      predict: model.
      x_clean: (x_noisy - σ model) / α.
      eps_clean: model.

    References:
      https://arxiv.org/abs/2006.11239
        Ho et al. 2020, "Denoising Diffusion Probabilistic Models."

    """
    del x_original
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    return TargetResult(
        target=eps_original,
        predict=model,
        x_clean=(x_noisy - model * torch.exp(log_sigma)) * torch.exp(-log_alpha),
        eps_clean=model,
    )


# The target ``α ε - σ x`` is three ops, each reconstruction three.
@set_cost(
    target_cost(
        primal=9,
        adjoint=0,
        vector_elements=21,
        scalar_flops=8,
        scalar_elements=32,
    ),
)
def target_v(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """v-prediction: model predicts v ≝ α ε - σ x.

    Recommended for variance-preserving corruption (α² + σ² = 1),
    where the denominator below vanishes.

    Args:
      model: Denoiser output (predicts v).
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (for target).
      eps_original: Clean noise (for target).

    Returns:
      target: α ε - σ x (None if originals not provided).
      predict: model.
      x_clean: (α xₜ - σ v) / (α² + σ²).
      eps_clean: (σ xₜ + α v) / (α² + σ²).

    References:
      https://arxiv.org/abs/2202.00512
        Salimans & Ho 2022, "Progressive Distillation."

    Derivation:

        Notation,

            xₜ = α x + σ ε       (forward process)
            v  = α ε - σ x       (v-prediction target)

        x_clean,

            From v = α ε - σ x,

                ε = (v + σ x) / α

            Substituting into xₜ = α x + σ ε,

                        xₜ = α x + σ (v + σ x) / α
            ⇔         α xₜ = α² x + σ² x + σ v
            ⇔  (α² + σ²) x = α xₜ - σ v
            ⇔            x = (α xₜ - σ v) / (α² + σ²)

        eps_clean,

            From v = α ε - σ x,

                x = (α ε - v) / σ

            Substituting into xₜ = α x + σ ε,

                        xₜ = α (α ε - v) / σ + σ ε
            ⇔         σ xₜ = α² ε - α v + σ² ε
            ⇔  (α² + σ²) ε = σ xₜ + α v
            ⇔            ε = (σ xₜ + α v) / (α² + σ²)

        For VP corruption (α² + σ² = 1) the denominator is
        unity,

            x = α xₜ - σ v
            ε = σ xₜ + α v

    """
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    if eps_original is None or x_original is None:
        target = None
    else:
        a = torch.exp(log_alpha)
        s = torch.exp(log_sigma)
        target = a * eps_original - s * x_original
    del x_original, eps_original
    log_normalizer = torch.logaddexp(2 * log_alpha, 2 * log_sigma)
    norm_alpha = torch.exp(log_alpha - log_normalizer)
    norm_sigma = torch.exp(log_sigma - log_normalizer)
    return TargetResult(
        target=target,
        predict=model,
        x_clean=norm_alpha * x_noisy - norm_sigma * model,
        eps_clean=norm_sigma * x_noisy + norm_alpha * model,
    )


# The target is one subtract, each reconstruction three ops.
@set_cost(
    target_cost(
        primal=7,
        adjoint=0,
        vector_elements=17,
        scalar_flops=8,
        scalar_elements=26,
    ),
)
def target_rectified_flow(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """Rectified flow: model predicts velocity v ≝ ε - x.

    Recommended for rectified flow corruption (α + σ = 1),
    where the denominator below vanishes.

    Args:
      model: Denoiser output (predicts v = ε - x).
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (for target).
      eps_original: Clean noise (for target).

    Returns:
      target: ε - x (None if originals not provided).
      predict: model.
      x_clean: (xₜ - σ v) / (α + σ).
      eps_clean: (xₜ + α v) / (α + σ).

    References:
      https://arxiv.org/abs/2209.03003
        Liu et al. 2022, "Flow Straight and Fast."

    Derivation:

        Notation,

            xₜ = α x + σ ε       (forward process)
            v  = ε - x           (RF velocity target)

        x_clean,

            From v = ε - x,

                ε = x + v

            Substituting into xₜ = α x + σ ε,

               xₜ = α x + σ (x + v)
            ⇔  xₜ = (α + σ) x + σ v
            ⇔   x = (xₜ - σ v) / (α + σ)

        eps_clean,

            From v = ε - x,

                x = ε - v

            Substituting into xₜ = α x + σ ε,

               xₜ = α (ε - v) + σ ε
            ⇔  xₜ = (α + σ) ε - α v
            ⇔   ε = (xₜ + α v) / (α + σ)

        For RF corruption (α + σ = 1) the denominator is
        unity,

            x = xₜ - σ v
            ε = xₜ + α v

    """
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    if eps_original is None or x_original is None:
        target = None
    else:
        target = eps_original - x_original
    del x_original, eps_original
    log_normalizer = torch.logaddexp(log_alpha, log_sigma)
    norm_alpha = torch.exp(log_alpha - log_normalizer)
    norm_sigma = torch.exp(log_sigma - log_normalizer)
    norm_scale = torch.exp(-log_normalizer)
    return TargetResult(
        target=target,
        predict=model,
        x_clean=norm_scale * x_noisy - norm_sigma * model,
        eps_clean=norm_scale * x_noisy + norm_alpha * model,
    )


# ``predict`` is a two-multiply-one-add combination of ``model``, so the adjoint
# through it is one multiply by the saved coefficient.
@set_cost(
    target_cost(
        primal=6,
        adjoint=1,
        vector_elements=14,
        scalar_flops=8,
        scalar_elements=26,
    ),
)
def target_v_x(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """v-prediction with loss in x-space.

    The model predicts v = α ε - σ x, but the loss targets x_original.

    Args:
      model: Denoiser output (predicts v).
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (for target).
      eps_original: Clean noise (unused).

    Returns:
      target: x_original.
      predict: (α xₜ - σ model) / (α² + σ²).
      x_clean: same as predict.
      eps_clean: (σ xₜ + α model) / (α² + σ²).

    References:
      https://arxiv.org/abs/2202.00512
        Salimans & Ho 2022, "Progressive Distillation."

    Derivation:

        From target_v,

            x = (α xₜ - σ v) / (α² + σ²)
            ε = (σ xₜ + α v) / (α² + σ²)

        So predict = x_clean = (α xₜ - σ model) / (α² + σ²).

        For VP corruption (α² + σ² = 1) the denominator is unity.

    """
    del eps_original
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    log_normalizer = torch.logaddexp(2 * log_alpha, 2 * log_sigma)
    norm_alpha = torch.exp(log_alpha - log_normalizer)
    norm_sigma = torch.exp(log_sigma - log_normalizer)
    x_clean = norm_alpha * x_noisy - norm_sigma * model
    return TargetResult(
        target=x_original,
        predict=x_clean,
        x_clean=x_clean,
        eps_clean=norm_sigma * x_noisy + norm_alpha * model,
    )


@set_cost(
    target_cost(
        primal=6,
        adjoint=1,
        vector_elements=14,
        scalar_flops=8,
        scalar_elements=26,
    ),
)
def target_v_eps(
    model: Tensor,
    x_noisy: Tensor,
    log_snr: Tensor,
    log_sigma: Tensor,
    *,
    x_original: Tensor | None = None,
    eps_original: Tensor | None = None,
) -> TargetResult:
    """v-prediction with loss in ε-space.

    The model predicts v = α ε - σ x, but the loss targets eps_original.

    Args:
      model: Denoiser output (predicts v).
      x_noisy: Noisy input, α x + σ ε.
      log_snr: Log signal-to-noise ratio.
      log_sigma: Log noise coefficient.
      x_original: Clean data (unused).
      eps_original: Clean noise (for target).

    Returns:
      target: eps_original.
      predict: (σ xₜ + α model) / (α² + σ²).
      x_clean: (α xₜ - σ model) / (α² + σ²).
      eps_clean: same as predict.

    References:
      https://arxiv.org/abs/2202.00512
        Salimans & Ho 2022, "Progressive Distillation."

    Derivation:

        From target_v,

            ε = (σ xₜ + α v) / (α² + σ²)
            x = (α xₜ - σ v) / (α² + σ²)

        So predict = eps_clean = (σ xₜ + α model) / (α² + σ²).

        For VP corruption (α² + σ² = 1) the denominator is unity.

    """
    del x_original
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    log_normalizer = torch.logaddexp(2 * log_alpha, 2 * log_sigma)
    norm_alpha = torch.exp(log_alpha - log_normalizer)
    norm_sigma = torch.exp(log_sigma - log_normalizer)
    eps_clean = norm_sigma * x_noisy + norm_alpha * model
    return TargetResult(
        target=eps_original,
        predict=eps_clean,
        x_clean=norm_alpha * x_noisy - norm_sigma * model,
        eps_clean=eps_clean,
    )
