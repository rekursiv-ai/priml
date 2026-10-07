r"""Teacher-forced response logprobs for a packed TMax training row.

Teacher-forcing feeds the recorded rollout tokens through the policy and
scores the tokens it sampled; the policy does not generate anything. A
packed row is several complete rollouts concatenated into one sequence
instead of padded to equal length.

The update needs :math:`\log\pi_\theta(y_t \mid s_t)` at every token the policy
actually sampled, with gradient. Upstream gets it in one call
(``grpo_utils.forward_for_logprobs``): run the whole packed row through the
model, drop the last logit, and gather the row's own next tokens.

    logits = model(input_ids, position_ids=position_ids)[:, :-1] / temperature
    labels = query_responses[:, 1:]
    logprob = selected_logit - logsumexp(logits)

This module reproduces that arithmetic with ONE structural change, and the
reason is Qwen 3.5's architecture rather than convenience.

Upstream passes ``attention_mask=None`` on purpose so the model derives each
document's internal mask from ``position_ids``: full-attention layers get a
block-diagonal causal mask, and the linear-attention layers get
variable-length (varlen) boundaries that reset their recurrent state at the
start of each trajectory. Three quarters of Qwen 3.5's layers are gated delta
nets, and PriML's implementation has no argument for those boundaries. Feeding
it a packed row would carry recurrent state from one trajectory into the next
and silently change every logprob after the first segment.

So the row is scored ONE SEGMENT PER FORWARD. That is not an approximation of
upstream: a correctly isolated packed row and a per-segment pass compute the
same number at every position the loss can read. The only positions that differ
are the segment boundaries -- the last token of one trajectory "predicting" the
first token of the next -- and those are prompt positions, which the response
mask zeroes in both. :func:`response_logprobs` leaves them NaN so that a
masking mistake shows up as NaN rather than as a plausible ratio.

Ported from TMax commit ``6d3d606``,
``grpo_utils.forward_for_logprobs`` and ``model_utils.log_softmax_and_gather``.
"""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, cast

from torch import Tensor, nn

import torch
import torch.utils.checkpoint


if TYPE_CHECKING:
    from priml.math.custom_types import TensorFn


class LogitModel(Protocol):
    """A causal language model mapping token ids to logits."""

    def __call__(self, tokens: Tensor, /, **kwargs: object) -> Tensor:
        """Return logits for ``tokens``.

        Args:
          tokens: Token ids, ``[rows, tokens]``.
          **kwargs: Forwarded model messages, such as ``positions``.

        Returns:
          logits: ``[rows, tokens, vocabulary]``.

        """
        ...


def log_softmax_and_gather(logits: Tensor, labels: Tensor) -> Tensor:
    """Return the logprob of each label without materializing a logprob tensor.

    ``selected_logit - logsumexp`` is algebraically ``log_softmax`` then
    ``gather``, and upstream uses it for the same reason PriML does: the
    discarded tensor would be ``[rows, tokens, vocabulary]``, which at Qwen
    3.5's vocabulary size dwarfs the model.

    Args:
      logits: ``[rows, tokens, vocabulary]``.
      labels: Token ids to score, ``[rows, tokens]``.

    Returns:
      logprobs: ``[rows, tokens]``.

    """
    selected = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)


def score_hidden_labels(
    head: TensorFn,
    hidden: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    chunk_size: int | None,
    fp32_head: bool,
) -> Tensor:
    """Project normalized hidden states and score labels without patching a module.

    The released 4B launcher bounds peak logits memory by fusing the head into
    a chunked loss (``--use_liger_grpo_loss --liger_grpo_loss_chunk_size 8``):
    logits exist a few positions at a time, never as one
    ``[tokens, vocabulary]`` tensor -- which for one 67k-token packed row of
    Qwen 3.5 would be ~33 GB in bfloat16 and ~67 GB in the fp32 the loss
    reduces in. This is how PriML enforces the same limit, chunked over scored
    positions instead of sequences. The scorer computes
    :func:`log_softmax_and_gather` per chunk under ``torch.utils.checkpoint``,
    so backward recomputes each chunk's logits instead of saving them. The
    registered head is never replaced, so exports, evaluation, and state-dict
    names remain untouched.

    Args:
      head: The model's registered language-model head.
      hidden: Normalized hidden states for one trajectory.
      labels: Next-token ids, one shorter than ``hidden``.
      temperature: Positive rollout sampling temperature.
      chunk_size: Scored positions per chunk; ``None`` projects at once.
      fp32_head: Cast both operands before the projection, as TMax does.

    Returns:
      logprobs: Float32 label log probabilities, ``[rows, tokens - 1]``.

    Raises:
      TypeError: The head has no supported projection weight for fp32 mode.
      ValueError: A numeric setting is invalid.

    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if getattr(head, "bias", None) is not None and fp32_head:
        raise TypeError("The fp32 head projection does not support a biased head.")

    def score(hidden: Tensor, labels: Tensor) -> Tensor:
        """Score one chunk, selecting the configured projection arithmetic."""
        if fp32_head:
            weight = _projection_weight(head)
            if weight is None:
                raise TypeError("The fp32 head has no supported projection weight.")
            # Keep this cast inside the checkpointed chunk. Hoisting the 4B
            # head's 2.37 GiB fp32 copy would retain it through the full
            # backward pass instead of releasing it between recomputations.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                logits = hidden.float() @ weight.float()
        else:
            logits = head(hidden).float()
        if temperature != 1.0:
            logits = logits / temperature
        return log_softmax_and_gather(logits, labels)

    if chunk_size is None:
        return score(hidden[:, :-1], labels)
    pieces: list[Tensor] = []
    for low in range(0, hidden.shape[1] - 1, chunk_size):
        high = min(low + chunk_size, hidden.shape[1] - 1)
        # Separate checkpoints let backward release one logits chunk at a time.
        pieces.append(
            torch.utils.checkpoint.checkpoint(
                score,
                hidden[:, low:high],
                labels[:, low:high],
                use_reentrant=False,
            ),
        )
    return torch.cat(pieces, dim=1)


def _projection_weight(head: object) -> Tensor | None:
    """Return the ``[hidden, vocab]`` weight a head module projects through.

    A plain head owns its ``[vocab, hidden]`` weight; a tied head
    (:class:`priml.model.special.TiedLinear`) borrows the embedding's, read
    through its ``transpose`` flag.
    """
    source = getattr(head, "_source", None)
    if isinstance(source, nn.Module):
        weight = getattr(source, "weight", None)
        if isinstance(weight, Tensor):
            return weight.T if getattr(head, "transpose", False) else weight
    weight = getattr(head, "weight", None)
    if isinstance(weight, Tensor):
        return weight.T
    return None


def response_logprobs(
    model: LogitModel,
    *,
    tokens: Tensor,
    segments: Tensor,
    positions: Tensor,
    pad_token_id: int,
    temperature: float = 1.0,
    head_chunk_size: int | None = None,
    fp32_head: bool = False,
) -> Tensor:
    """Score a packed row's own next tokens, one trajectory per forward pass.

    Args:
      model: The policy being trained.
      tokens: Packed token ids, ``[tokens]``.
      segments: 1-based segment id per token, ``[tokens]``.
      positions: Per-segment position ids, ``[tokens]``.
      pad_token_id: Replaced with 0 before gathering, as upstream does, so a
        padding id cannot index out of the vocabulary.
      temperature: Sampling temperature the rollouts were drawn at; TMax's 4B
        recipe uses ``1.0``, which makes the division a no-op.
      head_chunk_size: When set, score projected positions in bounded chunks.
      fp32_head: Project hidden states and head weights in fp32 before reduction.

    Returns:
      logprobs: ``[1, tokens - 1]`` in shifted coordinates -- entry ``t`` scores
        ``tokens[t + 1]`` -- with NaN at each inter-segment boundary, which no
        response mask selects. The final segment has no boundary entry because
        the shifted output omits the packed row's final token.

    Raises:
      ValueError: ``temperature`` is not positive, ``head_chunk_size`` is not
        positive, or the row has fewer than two tokens, which cannot be
        teacher forced.
      TypeError: ``segments`` is not one-dimensional, or the model does not
        support the injected hidden projection.

    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if head_chunk_size is not None and head_chunk_size <= 0:
        raise ValueError(f"head_chunk_size must be positive, got {head_chunk_size}.")
    if tokens.numel() < 2:
        raise ValueError("A training row needs at least two tokens to be scored.")
    out = torch.full(
        (1, tokens.numel() - 1),
        float("nan"),
        dtype=torch.float32,
        device=tokens.device,
    )
    boundaries = _segment_bounds(segments)
    for start, stop in boundaries:
        if stop - start < 2:
            # A one-token trajectory has no next token of its own to score.
            continue
        labels = tokens[start + 1 : stop].unsqueeze(0).clone()
        labels[labels == pad_token_id] = 0
        if head_chunk_size is None and not fp32_head:
            logits = model(
                tokens[start:stop].unsqueeze(0),
                positions=positions[start:stop].unsqueeze(0),
            )
            # The plain path preserves the native head kernel, while this
            # upcast keeps only the log-softmax reduction in fp32. exp000 uses
            # the explicit fp32-head path below, which casts before the matmul.
            logits = logits[:, :-1].float()
            if temperature != 1.0:
                logits = logits / temperature
            out[:, start : stop - 1] = log_softmax_and_gather(logits, labels)
        else:

            def output_projection(
                head: TensorFn,
                hidden: Tensor,
                labels: Tensor = labels,
            ) -> Tensor:
                """Score this segment through the model's registered head."""
                return score_hidden_labels(
                    head,
                    hidden,
                    labels,
                    temperature=temperature,
                    chunk_size=head_chunk_size,
                    fp32_head=fp32_head,
                )

            scored = model(
                tokens[start:stop].unsqueeze(0),
                positions=positions[start:stop].unsqueeze(0),
                output_projection=output_projection,
            )
            out[:, start : stop - 1] = scored
    return out


def _segment_bounds(segments: Tensor) -> list[tuple[int, int]]:
    """Return each segment's half-open token range.

    Args:
      segments: 1-based segment id per token, ``[tokens]``, non-decreasing
        because packing appends whole trajectories in order.

    Returns:
      bounds: ``(start, stop)`` per segment, in order.

    Raises:
      TypeError: ``segments`` is not one-dimensional.
      ValueError: ``segments`` is not non-decreasing, which means the row was
        not produced by :func:`priml.baselines.tmax.rollouts.pack_rollouts`.

    """
    if segments.ndim != 1:
        raise TypeError("Packed segment ids must be one-dimensional.")
    values = [int(cast(int, value)) for value in segments.tolist()]
    if any(later < earlier for earlier, later in pairwise(values)):
        raise ValueError("Packed segment ids must be non-decreasing.")
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            bounds.append((start, index))
            start = index
    return bounds
