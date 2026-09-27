"""Clipped policy-gradient objectives for on-policy learning.

The objective compares the probability a policy NOW assigns to an action with
the probability it assigned when the action was taken, and clips that ratio so
one update cannot move the policy arbitrarily far from the data it was fit on.
The value term is clipped the same way and for the same reason.

Two formulations live here:

- :func:`clipped_policy_loss` takes flat tensors -- the caller has already
  decided what a batch is -- standardizes the advantages, and returns mean
  terms for autograd. It holds no state, so it can be checked against a
  hand-computed ratio without an environment or an optimizer.
- :class:`PPO` is a learning rule over ``[rows, horizon]`` minibatches of a
  policy that emits one fused row per step, the logits and then the value. It
  leaves the advantages unstandardized and writes the objective's gradients
  with respect to the logits and the value in closed form, in the same pass
  as the loss. Called, it is one autograd node whose backward returns those
  gradients (the fused-loss pattern), so a learner calls ``backward()`` on its
  total as on any loss. :class:`TorchPPO` is its reference;
  ``policy_gradient_kernel.TritonPPO`` runs it as fused kernels.

:func:`clipped_policy_loss` is the standard form. :class:`PPO` exists to
reproduce PufferLib's learner.

References:
    https://arxiv.org/abs/1707.06347
        Schulman et al. 2017. Proximal policy optimization algorithms.
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/algo.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Protocol, override

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.math.advantage import observation_aligned_advantage


class ClippedPolicyLoss(NamedTuple):
    """The three terms of the objective, plus its optimization diagnostics."""

    policy: Tensor
    """Clipped policy-gradient term; the quantity being minimized."""

    value: Tensor
    """Clipped value-regression term."""

    entropy: Tensor
    """Mean policy entropy; subtracted from the total to reward exploration."""

    approx_kl: Tensor
    """Estimated divergence from the behavior policy, in nats."""

    clip_fraction: Tensor
    """Fraction of samples whose ratio left the trust region."""


def clipped_policy_loss(
    *,
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    advantages: Tensor,
    values: Tensor,
    behavior_values: Tensor,
    targets: Tensor,
    entropy: Tensor,
    clip_epsilon: float,
) -> ClippedPolicyLoss:
    """Compute the clipped policy and value terms over one flat batch.

    Advantages are standardized across the batch, which fixes the gradient
    scale so the learning rate does not have to absorb the reward magnitude.
    The standardization is centered and scaled by the ORIGINAL advantages, not
    the centered ones, matching the reference implementation.

    Args:
      log_probs: Log-probability of each taken action under the current policy.
      behavior_log_probs: The same quantity recorded during the rollout.
      advantages: Advantage estimate per sample.
      values: Current critic estimate per sample.
      behavior_values: Critic estimate recorded during the rollout.
      targets: Value-regression target per sample.
      entropy: Policy entropy per sample.
      clip_epsilon: Half-width of the trust region, for both ratio and value.

    Returns:
      terms: The policy, value, and entropy terms with their diagnostics.

    """
    log_ratio = log_probs - behavior_log_probs
    ratio = log_ratio.exp()
    # Standardize with the raw spread: an all-equal advantage vector has zero
    # spread, and the floor is what keeps that case finite rather than NaN.
    normalized = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + torch.finfo(advantages.dtype).eps
    )
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy = -torch.minimum(ratio * normalized, clipped_ratio * normalized).mean()

    clipped_values = behavior_values + (values - behavior_values).clamp(
        -clip_epsilon,
        clip_epsilon,
    )
    value = (
        0.5
        * torch.maximum(
            (values - targets) ** 2,
            (clipped_values - targets) ** 2,
        ).mean()
    )

    return ClippedPolicyLoss(
        policy=policy,
        value=value,
        entropy=entropy.mean(),
        # The k3 estimator: non-negative and lower variance than -log_ratio,
        # which is what makes it readable as a trust-region alarm.
        approx_kl=((ratio - 1.0) - log_ratio).mean(),
        clip_fraction=((ratio - 1.0).abs() > clip_epsilon).to(ratio.dtype).mean(),
    )


def categorical_entropy(log_probs: Tensor) -> Tensor:
    """Return the entropy of a categorical distribution given its log-probs.

    Computed as ``-sum(p * log p)`` over the last axis. A masked-out action
    carries ``log p = -inf`` and ``p = 0``, whose product is NaN rather than
    the zero the limit gives. Mask before multiplication to keep both entropy
    and its gradient finite for valid masked distributions.

    Args:
      log_probs: Normalized log-probabilities, ``[..., actions]``.

    Returns:
      entropy: Entropy in nats, shape ``[...]``.

    """
    masked = log_probs == float("-inf")
    safe_log_probs = torch.where(masked, torch.zeros_like(log_probs), log_probs)
    terms = safe_log_probs.exp() * safe_log_probs
    return -terms.sum(-1)


@dataclass(frozen=True, slots=True, kw_only=True)
class LogProbs:
    """What :meth:`PPO.log_probs` leaves behind for the advantage and the loss.

    Attributes:
      values: The live value prediction, ``[rows, horizon]``, in ``decoded``'s dtype.
      logps: The masked log-softmax, ``[rows, horizon, num_actions]`` fp32.
      new_lp: The sampled action's log-probability, ``[rows, horizon]`` fp32.

    """

    values: Tensor
    logps: Tensor
    new_lp: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class Loss:
    """The objective's gradients and its eight summed terms.

    Attributes:
      grad_logits: ``[rows, horizon, num_actions]`` fp32.
      grad_values: ``[rows, horizon]`` fp32.
      losses: ``[8]`` fp32, this minibatch's sums in
        :attr:`TorchPPO.Config.LOSS_NAMES` order; a caller accumulates them
        across minibatches.

    """

    grad_logits: Tensor
    grad_values: Tensor
    losses: Tensor


class PPO(Protocol):
    """The clipped objective as a learning rule that returns its gradients.

    Per minibatch of ``[rows, horizon]`` transitions, in order:

    1. :meth:`log_probs`: from the fused ``[logits, value]`` row, the masked
       log-softmax of the logits (an illegal logit is
       :attr:`TorchPPO.Config.MASKED_LOGIT`),
       the sampled action's new log-probability, and the live value.
    2. :meth:`advantage`: :func:`observation_aligned_advantage` along every
       row in fp32, and returns ``V + A``, both rounded to the dtype the
       values and rewards promote to: bf16 from bf16 inputs, fp32 if either
       is fp32.
    3. :meth:`loss`: the clipped policy loss, the clipped value loss and the
       entropy bonus, each with its gradient written directly: ``grad_logits``
       over the logits and ``grad_values`` over the value, both fp32 and both
       scaled by ``1 / (rows * horizon)``, the mean's weight.

    Calling the rule runs the three as one autograd node.

    The six coefficients are the rule's own, set on its config; each is
    rounded to fp32 once, where the rule reads it.
    """

    def __call__(
        self,
        decoded: Tensor,
        *,
        actions: Tensor,
        action_mask: Tensor,
        old_logprobs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the three stages; return the total, differentiable through ``decoded``.

        Args:
          decoded: ``[rows, horizon, num_actions + 1]`` bf16, logits then
            value.
          actions: ``[rows, horizon]`` fp32, the sampled action ids.
          action_mask: ``[rows, horizon, num_actions]`` bf16, zero where
            illegal.
          old_logprobs: ``[rows, horizon]``, from the rollout.
          rewards: ``[rows, horizon]``, already clamped.
          terminals: ``[rows, horizon]`` bf16.
          values: The ROLLOUT's values, ``[rows, horizon]``.

        Returns:
          total: The minibatch's total loss, 0-dim fp32. Its gradient reaches
            ``decoded`` as :meth:`loss` writes it, times the seed, rounded
            once to ``decoded``'s dtype.
          losses: ``[8]`` fp32, the summed terms; not differentiated.

        """
        ...

    def check_horizon(self, horizon: int) -> None:
        """Refuse a rollout length the rule cannot learn from.

        A learner calls this at construction, so a bad geometry fails before
        the first rollout rather than inside the first learner epoch.

        Args:
          horizon: Steps per rollout.

        """
        ...

    def log_probs(
        self,
        decoded: Tensor,
        actions: Tensor,
        action_mask: Tensor,
    ) -> LogProbs:
        """Score the sampled actions under the live policy.

        Args:
          decoded: ``[rows, horizon, num_actions + 1]`` bf16, logits then
            value.
          actions: ``[rows, horizon]`` fp32, the sampled action ids.
          action_mask: ``[rows, horizon, num_actions]`` bf16, zero where
            illegal.

        Returns:
          result: The live values and log-probabilities.

        """
        ...

    def advantage(
        self,
        values: Tensor,
        rewards: Tensor,
        terminals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Estimate every row's advantages, newest step first.

        Args:
          values: The live values, ``[rows, horizon]``.
          rewards: ``[rows, horizon]``, already clamped; step ``t``'s
            arrived with observation ``t``.
          terminals: ``[rows, horizon]``, likewise.

        Returns:
          advantages: ``[rows, horizon]`` in the dtype ``values`` and
            ``rewards`` promote to, zero at the last step.
          returns: ``values + advantages``, likewise.

        """
        ...

    def loss(
        self,
        logprobs: LogProbs,
        *,
        decoded: Tensor,
        actions: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        values: Tensor,
        returns: Tensor,
    ) -> Loss:
        """Evaluate the clipped objective and its gradients.

        Args:
          logprobs: What :meth:`log_probs` returned.
          decoded: As passed to :meth:`log_probs`.
          actions: As passed to :meth:`log_probs`.
          old_logprobs: ``[rows, horizon]``, from the rollout.
          advantages: From :meth:`advantage`.
          values: The ROLLOUT's values, ``[rows, horizon]`` -- the
            value clip's anchor.
          returns: From :meth:`advantage`.

        Returns:
          result: The gradients and the summed loss terms.

        """
        ...


class TorchPPO:
    """The rule in torch, on any device: the reference for its formulas and shapes.

    It uses torch's precise ``exp`` and ``log``, so the fused kernels, which
    use Triton's fast ``exp``, land a few ulp away from it: it is the reference for
    the derivatives, not for the kernels' bits.
    """

    class Config(Fig["TorchPPO"]):
        """The coefficients; the defaults are PufferLib's ``config/default.ini``."""

        LOSS_NAMES: ClassVar[tuple[str, ...]] = (
            "policy_loss",
            "value_loss",
            "entropy",
            "total_loss",
            "old_approx_kl",
            "approx_kl",
            "clipfrac",
            "importance",
        )
        """The terms :meth:`TorchPPO.loss` sums, in order, each over the minibatch
        times ``1 / (rows * horizon)``. Eight: the loss kernel's tree runs over a
        ``[block, 8]`` tile of them. A fact of the rule, not a knob, so it is a
        class constant and stays out of the printed config."""

        MASKED_LOGIT: ClassVar[float] = -1e4
        """What an illegal action's logit becomes before the softmax. The kernels
        write it as a literal, since a kernel reads no Python constant."""

        discount: float = 0.995
        """Reward discount factor, usually written gamma."""

        trace_decay: float = 0.90
        """Eligibility-trace decay, usually written lambda."""

        clip_epsilon: float = 0.2
        """Half-width of the importance ratio's trust region."""

        value_clip_epsilon: float = 0.2
        """Half-width of the value's trust region around the rollout's value."""

        value_coefficient: float = 2.0
        """The value term's weight in the total."""

        entropy_coefficient: float = 0.001
        """The entropy bonus's weight in the total."""

    def __init__(self, config: Config) -> None:
        """Keep the coefficients.

        Args:
          config: The coefficients.

        Raises:
          ValueError: ``discount`` or ``trace_decay`` is outside ``[0, 1]``,
            or another coefficient is negative, infinite or NaN.

        """
        for name, value in (
            ("discount", config.discount),
            ("trace_decay", config.trace_decay),
        ):
            if math.isnan(value) or value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1], not {value}")
        for name, value in (
            ("clip_epsilon", config.clip_epsilon),
            ("value_clip_epsilon", config.value_clip_epsilon),
            ("value_coefficient", config.value_coefficient),
            ("entropy_coefficient", config.entropy_coefficient),
        ):
            if math.isnan(value) or math.isinf(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative, not {value}")
        self.discount = config.discount
        self.trace_decay = config.trace_decay
        self.clip_epsilon = config.clip_epsilon
        self.value_clip_epsilon = config.value_clip_epsilon
        self.value_coefficient = config.value_coefficient
        self.entropy_coefficient = config.entropy_coefficient

    def __call__(
        self,
        decoded: Tensor,
        *,
        actions: Tensor,
        action_mask: Tensor,
        old_logprobs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the three stages; return the total, differentiable through ``decoded``.

        Args:
          decoded: ``[rows, horizon, num_actions + 1]`` bf16, logits then
            value.
          actions: ``[rows, horizon]`` fp32, the sampled action ids.
          action_mask: ``[rows, horizon, num_actions]`` bf16, zero where
            illegal.
          old_logprobs: ``[rows, horizon]``, from the rollout.
          rewards: ``[rows, horizon]``, already clamped.
          terminals: ``[rows, horizon]`` bf16.
          values: The ROLLOUT's values, ``[rows, horizon]``.

        Returns:
          total: The minibatch's total loss, 0-dim fp32, whose gradient is
            :meth:`loss`'s.
          losses: ``[8]`` fp32, the summed terms; not differentiated.

        """
        return _ClosedForm.apply(
            self,
            decoded,
            actions,
            action_mask,
            old_logprobs,
            rewards,
            terminals,
            values,
        )

    def check_horizon(self, horizon: int) -> None:
        """Refuse an empty rollout; the formulas take any other length.

        Args:
          horizon: Steps per rollout.

        Raises:
          ValueError: ``horizon`` is not positive.

        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, not {horizon}")

    def log_probs(
        self,
        decoded: Tensor,
        actions: Tensor,
        action_mask: Tensor,
    ) -> LogProbs:
        """Score the sampled actions under the live policy.

        Args:
          decoded: ``[rows, horizon, num_actions + 1]`` bf16, logits then
            value.
          actions: ``[rows, horizon]`` fp32, the sampled action ids.
          action_mask: ``[rows, horizon, num_actions]`` bf16, zero where
            illegal.

        Returns:
          result: The live values and log-probabilities.

        """
        num_actions = action_mask.shape[-1]
        masked = torch.where(
            action_mask != 0,
            decoded[..., :num_actions].float(),
            TorchPPO.Config.MASKED_LOGIT,
        )
        logps = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
        new_lp = logps.gather(-1, actions.long()[..., None])[..., 0]
        return LogProbs(values=decoded[..., num_actions], logps=logps, new_lp=new_lp)

    def advantage(
        self,
        values: Tensor,
        rewards: Tensor,
        terminals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Estimate every row's advantages, newest step first.

        Args:
          values: The live values, ``[rows, horizon]``.
          rewards: ``[rows, horizon]``, already clamped; step ``t``'s
            arrived with observation ``t``.
          terminals: ``[rows, horizon]``, likewise.

        Returns:
          advantages: ``[rows, horizon]`` in the dtype ``values`` and
            ``rewards`` promote to, computed in fp32; zero at the last step.
          returns: ``values + advantages``, likewise.

        """
        advantages, returns = observation_aligned_advantage(
            rewards=rewards.float(),
            values=values.float(),
            dones=terminals.float(),
            discount=self.discount,
            trace_decay=self.trace_decay,
        )
        dtype = torch.promote_types(values.dtype, rewards.dtype)
        return advantages.to(dtype), returns.to(dtype)

    def loss(
        self,
        logprobs: LogProbs,
        *,
        decoded: Tensor,
        actions: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        values: Tensor,
        returns: Tensor,
    ) -> Loss:
        """Evaluate the clipped objective and its gradients.

        Args:
          logprobs: What :meth:`log_probs` returned.
          decoded: As passed to :meth:`log_probs`.
          actions: As passed to :meth:`log_probs`.
          old_logprobs: ``[rows, horizon]``, from the rollout.
          advantages: From :meth:`advantage`.
          values: The ROLLOUT's values, ``[rows, horizon]`` -- the
            value clip's anchor.
          returns: From :meth:`advantage`.

        Returns:
          result: The gradients and the summed loss terms.

        """
        num_actions = logprobs.logps.shape[-1]
        rows, horizon = actions.shape
        f32 = torch.float32
        device = decoded.device
        clip = torch.tensor(self.clip_epsilon, dtype=f32, device=device)
        value_clip = torch.tensor(self.value_clip_epsilon, dtype=f32, device=device)
        vf_coef = torch.tensor(self.value_coefficient, dtype=f32, device=device)
        ent_coef = torch.tensor(self.entropy_coefficient, dtype=f32, device=device)
        inv_nt = torch.tensor(1 / (rows * horizon), dtype=f32, device=device)

        adv = advantages.float()
        val = values.float()
        ret = returns.float()
        val_pred = decoded[..., num_actions].float()
        logratio = logprobs.new_lp - old_logprobs.float()
        ratio = torch.exp(logratio)

        v_error = val_pred - val
        v_clipped = val + torch.clamp(v_error, -value_clip, value_clip)
        v_loss_unclipped = (val_pred - ret) * (val_pred - ret)
        v_loss_clipped = (v_clipped - ret) * (v_clipped - ret)
        v_loss = 0.5 * torch.maximum(v_loss_unclipped, v_loss_clipped)
        d_val_pred = torch.where(v_loss_clipped > v_loss_unclipped, 0.0, val_pred - ret)
        grad_values = (inv_nt * vf_coef) * d_val_pred

        clip_lo, clip_hi = 1 - clip, 1 + clip
        ratio_clipped = torch.clamp(ratio, clip_lo, clip_hi)
        wa = -adv
        pg_loss1 = wa * ratio
        pg_loss2 = wa * ratio_clipped
        pg_loss = torch.maximum(pg_loss1, pg_loss2)
        clipped = (pg_loss2 > pg_loss1) & ((ratio <= clip_lo) | (ratio >= clip_hi))
        d_ratio = torch.where(clipped, 0.0, wa * inv_nt)
        d_new_logp = d_ratio * ratio

        logps = logprobs.logps
        probabilities = torch.exp(logps)
        entropy = -(probabilities * logps).sum(-1)
        d_entropy_term = inv_nt * (-ent_coef)
        indicator = torch.nn.functional.one_hot(actions.long(), num_actions).to(f32)
        grad_logits = (indicator - probabilities) * d_new_logp[..., None] + (
            d_entropy_term * probabilities
        ) * (-entropy[..., None] - logps)
        thread_loss = (pg_loss + vf_coef * v_loss - ent_coef * entropy) * inv_nt
        terms = torch.stack(
            (
                pg_loss * inv_nt,
                v_loss * inv_nt,
                entropy * inv_nt,
                thread_loss,
                (-logratio) * inv_nt,
                ((ratio - 1) - logratio) * inv_nt,
                ((ratio - 1).abs() > clip).to(f32) * inv_nt,
                ratio * inv_nt,
            ),
        )
        return Loss(
            grad_logits=grad_logits,
            grad_values=grad_values,
            losses=terms.reshape(len(TorchPPO.Config.LOSS_NAMES), -1).sum(-1),
        )


class _ClosedFormContext(Protocol):
    """What :class:`_ClosedForm` keeps between its forward and its backward."""

    dtype: torch.dtype
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


def _closed_form_backward(
    ctx: _ClosedFormContext,
    /,
    *grad_outputs: Tensor | None,
) -> tuple[None, Tensor, None, None, None, None, None, None]:
    """Scale the rule's gradient by the total's and round it to the policy's dtype."""
    grad_total, _ = grad_outputs
    assert isinstance(grad_total, Tensor)
    grad_logits, grad_values = ctx.saved_tensors
    grad = torch.cat((grad_logits, grad_values[..., None]), dim=-1)
    # Scaled and rounded in one pass: the product is fp32, then the store rounds it.
    scaled = torch.mul(
        grad,
        grad_total,
        out=grad.new_empty(grad.shape, dtype=ctx.dtype),
    )
    return None, scaled, None, None, None, None, None, None


# The fused-loss pattern: the rule's kernels write the loss and its gradient in one
# pass, so the backward only scales what the forward kept. Seeded with 1, as
# ``total.backward()`` seeds it, the scaling is exact, and the gradient is the rule's
# fp32 one rounded once to the policy's dtype.
class _ClosedForm(torch.autograd.Function):
    """A rule's three stages as one autograd node, differentiated in closed form."""

    @classmethod
    @override
    def forward(
        cls,
        ctx: _ClosedFormContext,
        /,
        rule: PPO,
        decoded: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        old_logprobs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logprobs = rule.log_probs(decoded, actions, action_mask)
        advantages, returns = rule.advantage(logprobs.values, rewards, terminals)
        loss = rule.loss(
            logprobs,
            decoded=decoded,
            actions=actions,
            old_logprobs=old_logprobs,
            advantages=advantages,
            values=values,
            returns=returns,
        )
        ctx.dtype = decoded.dtype
        ctx.save_for_backward(loss.grad_logits, loss.grad_values)
        ctx.mark_non_differentiable(loss.losses)
        # The terms carry no gradient, so theirs arrives as None, not zeros.
        ctx.set_materialize_grads(False)
        return loss.losses[
            TorchPPO.Config.LOSS_NAMES.index("total_loss")
        ].clone(), loss.losses

    # Torch declares ``backward`` a staticmethod, and an override must stay one.
    backward = staticmethod(_closed_form_backward)
