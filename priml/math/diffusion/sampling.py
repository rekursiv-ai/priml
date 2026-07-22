from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Final, NamedTuple, Protocol

import functools
import math

from torch import Tensor

import torch

from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.diffusion.schedule import (
    compute_log_alpha,
    log_sigma_from_log_snr_per_rectified_flow,
)
from priml.math.diffusion.target import TargetFn, target_rectified_flow
from priml.math.numeric import log1mexp


__all__ = [
    "SampleModelFn",
    "SampleOneStepFn",
    "SampleOneStepResult",
    "SampleResult",
    "ddim",
    "ddpm",
    "ddpm_ddim",
    "ddpm_ddim_step",
    "rescale_classifier_free_guidance",
    "sample",
]


class SampleOneStepResult(NamedTuple):
    x_clean: Tensor
    mean: Tensor
    log_std: Tensor


class SampleResult(NamedTuple):
    x_curr: Tensor
    x_clean: Tensor
    mean: Tensor
    log_std: Tensor
    model: Tensor


class SampleOneStepFn(Protocol):
    def __call__(
        self,
        model: Tensor,
        x_curr: Tensorable,
        log_snr_curr: Tensorable,
        log_snr_next: Tensorable,
    ) -> SampleOneStepResult: ...


class SampleModelFn(Protocol):
    def __call__(
        self,
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor: ...


def ddpm_ddim_step(
    x_clean: Tensor,
    eps_clean: Tensor,
    log_snr_curr: Tensor,
    log_snr_next: Tensor,
    log_sigma_next: Tensor,
    eta: float = 1,
) -> tuple[Tensor, Tensor]:
    """DDPM/DDIM sampling step.

    Constructs x₀ at the target noise level from pre-decomposed
    (x_clean, eps_clean). Assumes log_snr_next > log_snr_curr
    (denoising direction); violating this produces NaN because
    log1mexp(logsnr₁ - logsnr₀) requires logsnr₁ < logsnr₀.

    Args:
      x_clean: Model's estimate of the clean signal.
      eps_clean: Model's estimate of the noise.
      log_snr_curr: Log SNR at current (noisier) step.
      log_snr_next: Log SNR at target (less noisy) step.
      log_sigma_next: Log σ at target step.
      eta: Interpolation η ∈ [0,1]. 0=DDIM, 1=DDPM.

    Returns:
      mean: Mean of the sampling distribution
        (α₀ x_clean + σ_d ε_clean).
      log_std: Log standard deviation of sampling noise
        (log σ_r).

    References:
      https://arxiv.org/abs/2107.00630
        DDPM posterior (Step 1): Kingma et al. 2021, Eq 21-22, 25-26.
      https://arxiv.org/abs/2010.02502
        η interpolation (Step 2): Song et al. 2020, Eq 12.

    Derivation:
      First, define the notation:
        x₁            = current noisy sample (input)
        x₀            = new sample at target noise level
        logsnr₁       = log SNR at current (noisier) step
        logsnr₀       = log SNR at target (less noisy) step
        α₀, σ₀        = corruption params at target step
        α₁, σ₁        = corruption params at current step
        x_clean, ε_clean = model's predictions of clean signal and noise

      The sampler constructs x₀ at the target noise level,

          x₀ = (α₀ x_clean + σ_d ε_clean) + σ_r z,
          z ~ N(0,I)

      where,

        σ_d² + σ_r² ≝ σ₀²
        σ_r ≝ η σ_ddpm
        η ∈ [0,1]
        q(x₀|x₁, x_clean) = Normal(x₀; μ_ddpm, σ_ddpm²I)

      Step 1: Derive σ_ddpm from the DDPM posterior.

        The forward process defines two distributions over x_clean:

          q(x₁|x_clean) ≝ Normal(x₁; α₁ x_clean, σ₁² I)
          q(x₀|x_clean) ≝ Normal(x₀; α₀ x_clean, σ₀² I)

        From these we can derive the conditional:

          q(x₁|x₀) = Normal(x₁; (α₁/α₀) x₀, σ_c² I)

        where σ_c² ≝ σ₁² - (α₁/α₀)² σ₀² is the conditional variance.

        The posterior,

          q(x₀|x₁, x_clean) ∝ q(x₁|x₀) q(x₀|x_clean).

        Both are Gaussian in x₀. Viewing q(x₁|x₀) as a function of x₀, it is

          Normal(x₀; x₁ α₀/α₁, σ_c² α₀²/α₁²),

        so its precision in x₀ is (α₁/α₀)²/σ_c².

        The posterior precision is the sum,

          1/σ_ddpm² = (α₁/α₀)²/σ_c² + 1/σ₀²

        Inverting,

          σ_ddpm² = σ_c² σ₀² / ((α₁/α₀)² σ₀² + σ_c²)

        The denominator simplifies,

          (α₁/α₀)² σ₀² + σ_c² =
            = α₁²σ₀²/α₀² + σ₁² - α₁²σ₀²/α₀²
            = σ₁²

        So,

          σ_ddpm² =
            = σ_c² σ₀² / σ₁²
            = (σ₁² - α₁²σ₀²/α₀²) σ₀² / σ₁²
            = σ₀² (1 - (α₁²/σ₁²)/(α₀²/σ₀²))
            = σ₀² (1 - snr₁/snr₀)

          where,

            exp(logsnr₁) ≝ snr₁ ≝ α₁²/σ₁²
            exp(logsnr₀) ≝ snr₀ ≝ α₀²/σ₀²

        Taking ½ log of the above σ_ddpm² equality,

          log σ_ddpm =
            = log σ₀ + ½ log(1 - snr₁/snr₀)
            = log σ₀ + ½ log(1 - exp(logsnr₁-logsnr₀))
            = log σ₀ + ½ log1mexp(logsnr₁ - logsnr₀)

        Note that σ_ddpm²,σ₀² ≥ 0 implies snr₀ ≥ snr₁
        which is why log1mexp(logsnr₁ - logsnr₀) is well
        defined.

      Step 2: η scales how much of the DDPM posterior
        randomness to use (Song et al. 2020, "Denoising
        Diffusion Implicit Models"):

          σ_r ≝ η σ_ddpm,

        i.e.,

          log σ_r = log η + log σ_ddpm

      Step 3: σ_d from σ_d² + σ_r² ≝ σ₀²:

          σ_d² = σ₀² - σ_r²
               = σ₀² (1 - σ_r²/σ₀²)

        Taking logs,

          2 log σ_d = 2 log σ₀ + log(1 - exp(2 log σ_r - 2 log σ₀))
                    = 2 log σ₀ + log1mexp(2 log σ_r - 2 log σ₀)

            log σ_d = log σ₀ + ½ log1mexp(2 log σ_r - 2 log σ₀)

      η=0 (DDIM): σ_r=0, σ_d=σ₀, deterministic.
      η=1 (DDPM): σ_r=σ_ddpm, matches the true posterior.

    """
    # η=0 (DDIM) and η=1 (DDPM) need no special cases: log(0)=-inf
    # propagates to log_std_random=-inf, then log1mexp(-inf)=0, giving
    # log_std_determ=log_sigma_next (deterministic). log(1)=0 is a no-op.
    # eta is a per-call scalar, so take its log once on the host rather than
    # allocating a device tensor every step; the broadcast add is identical.
    log_eta = math.log(eta) if eta > 0 else -math.inf
    log_std_random = (
        log_sigma_next + 0.5 * log1mexp(log_snr_curr - log_snr_next) + log_eta
    )
    log_std_determ = log_sigma_next + 0.5 * log1mexp(
        2 * log_std_random - 2 * log_sigma_next,
    )
    log_alpha_next = compute_log_alpha(log_snr_next, log_sigma_next)
    mean = torch.exp(log_alpha_next) * x_clean + torch.exp(log_std_determ) * eps_clean
    return mean, log_std_random


def ddpm_ddim(
    model: Tensor,
    x_curr: Tensorable,
    log_snr_curr: Tensorable,
    log_snr_next: Tensorable,
    *,
    corruption_fn: Callable[
        [Tensorable],
        Tensor,
    ] = log_sigma_from_log_snr_per_rectified_flow,
    target_fn: TargetFn = target_rectified_flow,
    eta: float = 1,
) -> SampleOneStepResult:
    """Convenience wrapper: decompose model output, then step.

    Calls corruption_fn → target_fn → ddpm_ddim_step.

    Args:
      model: Raw model output tensor.
      x_curr: Current noisy sample.
      log_snr_curr: Log SNR at current (noisier) step.
      log_snr_next: Log SNR at target (less noisy) step.
      corruption_fn: Maps log_snr to log_sigma.
      target_fn: Extracts x_clean and ε_clean from model output.
      eta: Interpolation parameter η ∈ [0,1]. 0=DDIM, 1=DDPM.

    Returns:
      x_clean: Model's prediction of the clean signal.
      mean: Mean of the sampling distribution.
      log_std: Log standard deviation of the sampling noise.

    """
    model, x_curr, log_snr_curr, log_snr_next = convert_to_tensor(
        model,
        x_curr,
        log_snr_curr,
        log_snr_next,
    )
    log_sigma_curr = corruption_fn(log_snr_curr)
    log_sigma_next = corruption_fn(log_snr_next)
    _, _, x_clean, eps_clean = target_fn(model, x_curr, log_snr_curr, log_sigma_curr)
    mean, log_std = ddpm_ddim_step(
        x_clean,
        eps_clean,
        log_snr_curr,
        log_snr_next,
        log_sigma_next,
        eta,
    )
    return SampleOneStepResult(x_clean, mean, log_std)


ddpm = functools.partial(ddpm_ddim, eta=1)
ddim = functools.partial(ddpm_ddim, eta=0)


def sample(
    log_snr: Tensorable,
    model_fn: SampleModelFn,
    x_init: Tensorable,
    onestep_fn: SampleOneStepFn = ddpm,
    *,
    noise_last_step: bool = False,
    wrap_steps: Callable[
        [Iterable[SampleResult]], Iterable[SampleResult]
    ] = lambda it: it,
) -> SampleResult:
    """Run a diffusion `onestep_fn` to completion and return the final step.

    For step-by-step access use `sample_iter`.

    Args:
      log_snr: A vector of log(SNR) values, typically increasing
        in value, e.g.,
        `-2 * logit(linspace(1, 0, steps=num_steps + 1))`. The
        actual number of diffusion steps is `log_snr.shape[0] - 1`;
        each `log_snr[i]` broadcasts against `x_init`.
      model_fn: Denoising callable with signature
        `(x, log_snr_curr, it, num_steps, log_snr_next) -> Tensor`.
      x_init: Initial state; broadcasts with `log_snr.shape[1:]`.
      onestep_fn: Core diffusion math. Default: DDPM with eta=1.
      noise_last_step: When False (default), the final update is the noise-free
        posterior mean (Ho et al. 2020 Alg. 2: sigma=0 at t=0). When True, the
        final step adds sampling noise like every other step, for advanced
        users sampling an intermediate latent.
      wrap_steps: Optional transform applied to the per-step iterator before it is
        drained, e.g. `wrap_steps=tqdm` for a progress bar over the diffusion steps
        (`sample(..., wrap_steps=tqdm)`) or `wrap_steps=lambda it: tqdm(it, total=n)`.
        Defaults to identity. The final returned step is unaffected.

    Returns:
      x_curr: The final sample; same shape/dtype as `x_init`.
      x_clean: Model's prediction of the clean signal at the last step.
      mean: Mean of the last step's sampling distribution.
      log_std: Log std of the last step's sampling noise.
      model: Raw model output at the last step.

    ## Example: Conditional Sampling with CFG

    ```python
    from torch import Tensor

    import torch
    from priml.math import diffusion

    def denoiser(x: Tensor, sigma: Tensor, prompt: Tensor) -> Tensor:
        ...  # Your model here.

    @torch.compile
    def sample_with_cfg(
        x_init: Tensor,
        text_prompt: Tensor,
        null_prompt: Tensor,
        cfg: float = 5.,
        num_steps: int = 50,
    ) -> Tensor:
        prompt = torch.cat([null_prompt, text_prompt], dim=0)

        def model_fn(
            x: Tensor,
            log_snr: Tensor,
            it: Tensor,
            num_steps: int,
            log_snr_next: Tensor,
        ) -> Tensor:
            del it, num_steps, log_snr_next
            x2 = x.repeat([2] + [1] * (x.ndim - 1))
            log_sigma = diffusion.log_sigma_from_log_snr_per_rectified_flow(log_snr)
            sigma = log_sigma.exp().repeat([2] + [1] * (log_sigma.ndim - 1))
            y = denoiser(x2, sigma, prompt)
            uncond, cond = y.chunk(2, dim=0)
            return torch.lerp(input=uncond, end=cond, weight=cfg)

        log_t = torch.linspace(1, 0, steps=num_steps + 1).log()
        log_snr = diffusion.log_snr_from_log_time_per_logit(log_t)
        return diffusion.sample(log_snr, model_fn, x_init, diffusion.ddpm).x_clean
    ```

    """
    steps = sample_iter(
        log_snr,
        model_fn,
        x_init,
        onestep_fn,
        noise_last_step=noise_last_step,
    )
    # Drain the (optionally wrapped) step stream, keeping only the final step.
    tail = deque(wrap_steps(steps), maxlen=1)
    if not tail:
        raise ValueError("sample produced no steps; check log_snr length.")
    return tail[0]


def sample_iter(
    log_snr: Tensorable,
    model_fn: SampleModelFn,
    x_init: Tensorable,
    onestep_fn: SampleOneStepFn = ddpm,
    *,
    noise_last_step: bool = False,
) -> Generator[SampleResult, None, None]:
    """Yield each diffusion step of a `sample` run.

    Args:
      log_snr: See `sample`.
      model_fn: See `sample`.
      x_init: See `sample`.
      onestep_fn: See `sample`.
      noise_last_step: See `sample`. When False (default), the final yielded
        step carries the noise-free posterior mean.

    Yields:
      step: A `SampleResult` for each diffusion step, in order.

    """
    x_init, log_snr = convert_to_tensor(x_init, log_snr)

    num_steps: Final[int] = log_snr.shape[0] - 1
    if num_steps < 1:
        raise ValueError(
            "Expected log_snr to have leading size of at least 2 "
            f"but saw {num_steps + 1}.",
        )

    x_curr = x_init
    for i in range(num_steps):
        it = torch.tensor(i, dtype=torch.int64, device=x_curr.device)
        log_snr_curr = log_snr[i]
        log_snr_next = log_snr[i + 1]
        model_output = model_fn(x_curr, log_snr_curr, it, num_steps, log_snr_next)
        x_clean, mean, log_std = onestep_fn(
            model_output,
            x_curr,
            log_snr_curr,
            log_snr_next,
        )
        # The Ho et al. 2020 sampler omits noise on the final step, returning
        # the posterior mean; other steps always add their sampling noise.
        if noise_last_step or i < num_steps - 1:
            x_curr = mean + torch.exp(log_std) * torch.randn_like(x_curr)
        else:
            x_curr = mean
        yield SampleResult(x_curr, x_clean, mean, log_std, model_output)


def rescale_classifier_free_guidance(
    guided: Tensor,
    unguided: Tensor,
    *,
    strength: float = 0.0,
    eps: float = 1e-9,
    spatial_dims: Sequence[int],
) -> Tensor:
    """Rescale CFG output to match the std of the unconditional prediction.

    Args:
      guided: The prediction with classifier-free guidance applied.
      unguided: The prediction without guidance.
      strength: Interpolation in [0, 1]. 0 returns guided unchanged;
        1 fully rescales guided to match unguided's std.
      eps: Small constant for numerical stability.
      spatial_dims: Dimensions over which to compute std (excluding batch).

    Returns:
      The rescaled guided prediction.

    References:
      https://arxiv.org/abs/2305.08891
        Lin et al. 2024, "Common Diffusion Noise Schedules and Sample Steps are Flawed,"
        Eq. 6.

    """
    if strength == 0.0:
        return guided
    sigma_ratio = torch.std(unguided, dim=spatial_dims, keepdim=True) / (
        torch.std(guided, dim=spatial_dims, keepdim=True) + eps
    )
    return guided * sigma_ratio.lerp(torch.ones_like(sigma_ratio), 1 - strength)
