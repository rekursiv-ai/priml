"""Token-level DPPO loss from the released TMax trainer.

TMax uses DPPO instead of PPO. PPO limits the probability ratio to a fixed
range above and below the old value. DPPO instead creates a mask for each
token. The mask decides whether that token may affect the update.

The model first generates terminal responses through vLLM. vLLM saves the log
probability it gave every generated token. This is the behavior policy: the
version of the model that produced the response.

The learner scores the same tokens using its current weights. The importance
ratio compares the current probability with the saved generation probability.
TMax requires ``loss_fn=dppo`` to use ``use_vllm_logprobs=True`` for this
reason. This is Takeaway 2 from the paper. Recomputing the old probability with
the current model would change the ratio, mask, and algorithm.

The mask is asymmetric. A positive advantage asks the model to make a token
more likely. A negative advantage asks it to make the token less likely. DPPO
blocks the token only when the models are already too different and the update
would move them farther apart. An update that moves the current model back
toward the generating model remains allowed.

TMax measures the difference using the paper's binary approximation
(Eqs. 13 and 14). It treats the sampled token as one choice and all other
tokens as the second choice. This needs only the saved and current token log
probabilities. It does not need another model pass or the full vocabulary
distribution.

Ported from TMax commit ``6d3d606``:

- ``grpo_utils.py``: ``mask_logprobs``, ``binary_divergence``,
  ``dppo_mask``, and the DPPO branch of ``compute_grpo_loss``.
- ``rl_utils.py``: ``masked_mean``.

``source_parity_test`` compares ``importance_ratio``, ``binary_divergence``,
``dppo_mask``, ``dppo_token_loss``, packing, and advantages with upstream
results. Direct unit tests cover ``mask_logprobs`` and ``masked_mean``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch


if TYPE_CHECKING:
    from torch import Tensor


DivergenceType = Literal["tv", "kl"]
"""How DPPO measures change in a sampled token's probability.

``"tv"`` uses the absolute probability difference. ``"kl"`` uses Bernoulli
KL divergence, which also accounts for the direction of the change. The
released TMax 4B recipe uses ``"tv"``.
"""


INVALID_LOGPROB: Final = 1.0
"""Mark a token that must not affect training.

Real log probabilities are never positive, so ``1.0`` cannot be mistaken for
a valid value. This matches the value used by upstream TMax.
"""

_LOGPROB_FLOOR: Final = -30.0
"""Lowest log probability used by the divergence calculation.

Clamping at ``-30`` avoids numerical problems for extremely unlikely tokens
and matches upstream TMax.
"""

_BERNOULLI_EPS: Final = 1e-9
"""Keep KL probabilities away from exactly 0 and 1.

This prevents infinite or invalid values when KL takes logarithms.
"""


def mask_logprobs(logprobs: Tensor, response_mask: Tensor) -> Tensor:
    """Mark positions that must not take part in training.

    Prompt tokens and tool output were not sampled by the model. Replace their
    values, and any missing values, with :data:`INVALID_LOGPROB`.

    Args:
      logprobs: Saved log probabilities for each token, ``[rows, tokens]``.
      response_mask: True for response tokens the model sampled, same shape.

    Returns:
      masked: Values with excluded positions and NaNs replaced by
        :data:`INVALID_LOGPROB`.

    """
    masked = torch.masked_fill(logprobs, ~response_mask, INVALID_LOGPROB)
    return torch.nan_to_num(masked, nan=INVALID_LOGPROB)


def importance_ratio(new_logprobs: Tensor, behavior_logprobs: Tensor) -> Tensor:
    """Measure how much each sampled token's probability changed.

    A value above 1 means the current model makes the token more likely. A
    value below 1 means it makes the token less likely. The value is not
    clamped because DPPO controls large changes with its token mask.

    Args:
      new_logprobs: Current model's log probabilities.
      behavior_logprobs: Saved log probabilities from response generation.

    Returns:
      ratio: Current probability divided by generation probability per token.

    """
    return torch.exp(new_logprobs - behavior_logprobs)


def binary_divergence(
    *,
    behavior_logprobs: Tensor,
    policy_logprobs: Tensor,
    response_mask: Tensor,
    divergence_type: DivergenceType = "tv",
) -> Tensor:
    """Measure how different the generating and current models are.

    Follow the paper's Eqs. 13 and 14 by treating the sampled token as one
    choice and every other token as the second choice. This avoids another
    model pass and avoids building a full vocabulary distribution.

    Args:
      behavior_logprobs: Saved log probabilities from response generation.
      policy_logprobs: Current model's log probabilities for the same tokens.
      response_mask: True for response tokens the model sampled.
      divergence_type: ``"tv"`` for absolute probability difference or
        ``"kl"`` for the log-based KL measure.

    Returns:
      divergence: Difference per token, with zero at excluded positions.

    Raises:
      ValueError: ``divergence_type`` is neither ``"tv"`` nor ``"kl"``.

    """
    rollout = torch.exp(behavior_logprobs.clamp(min=_LOGPROB_FLOOR, max=0.0))
    policy = torch.exp(policy_logprobs.clamp(min=_LOGPROB_FLOOR, max=0.0))
    if divergence_type == "tv":
        divergence = (rollout - policy).abs()
    elif divergence_type == "kl":
        rollout_clip = rollout.clamp(_BERNOULLI_EPS, 1.0 - _BERNOULLI_EPS)
        policy_clip = policy.clamp(_BERNOULLI_EPS, 1.0 - _BERNOULLI_EPS)
        divergence = rollout_clip * (rollout_clip.log() - policy_clip.log()) + (
            1.0 - rollout_clip
        ) * ((1.0 - rollout_clip).log() - (1.0 - policy_clip).log())
    else:
        raise ValueError(  # pyright: ignore[reportUnreachable]
            f"Unknown DPPO divergence type: {divergence_type!r}.",
        )
    return torch.where(response_mask, divergence, torch.zeros_like(divergence))


def dppo_mask(
    *,
    new_logprobs: Tensor,
    behavior_logprobs: Tensor,
    advantages: Tensor,
    ratio: Tensor,
    response_mask: Tensor,
    divergence_threshold: float,
    divergence_type: DivergenceType = "tv",
) -> tuple[Tensor, Tensor]:
    """Choose which tokens may affect training.

    Following the paper's Eq. 12, block a token only when the models are
    already too different and the update would move them farther apart. A
    positive advantage asks for a higher probability. A negative advantage
    asks for a lower probability. Changes back toward the generating model
    remain allowed.

    Args:
      new_logprobs: Current model's log probabilities.
      behavior_logprobs: Saved log probabilities from response generation.
      advantages: Whether each token should become more or less likely.
      ratio: Output of :func:`importance_ratio`.
      response_mask: True for response tokens the model sampled.
      divergence_threshold: Largest allowed model difference; TMax uses ``0.1``.
      divergence_type: Passed to :func:`binary_divergence`.

    Returns:
      mask: ``1.0`` where a token may train and ``0.0`` where it is blocked.
      divergence: Model difference used to make the mask.

    """
    with torch.no_grad():
        divergence = binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=new_logprobs,
            response_mask=response_mask,
            divergence_type=divergence_type,
        )
        outside = divergence > divergence_threshold
        pushing_out = ((advantages > 0) & (ratio > 1.0)) | (
            (advantages < 0) & (ratio < 1.0)
        )
        mask = (~(outside & pushing_out) & response_mask).to(new_logprobs.dtype)
    return mask, divergence


def dppo_token_loss(
    *,
    advantages: Tensor,
    ratio: Tensor,
    policy_mask: Tensor,
) -> Tensor:
    """Compute the loss for each token DPPO allows.

    This is the paper's Eq. 11. Unlike PPO, DPPO does not add a second clipped
    loss. TMax also sets ``beta=0``, so it adds no separate KL penalty.

    Args:
      advantages: Whether each token should become more or less likely.
      ratio: Current probability divided by generation probability.
      policy_mask: ``1`` for tokens allowed to train and ``0`` otherwise.

    Returns:
      loss: Loss for each token, before averaging.

    """
    return -advantages * ratio * policy_mask


def masked_mean(values: Tensor, mask: Tensor, *, denominator: float) -> Tensor:
    """Average active token values over the full update.

    Divide by the total response-token count for the update, not the count in
    this row. This gives every token equal weight across all rows and GPUs.

    Args:
      values: Per-token values.
      mask: True or nonzero for values included in the sum.
      denominator: Total response-token count; ``<= 0`` returns zero.

    Returns:
      reduced: One scalar average for this part of the update.

    """
    numerator = (values * mask).sum()
    if denominator <= 0:
        return torch.zeros_like(numerator)
    return numerator / denominator
