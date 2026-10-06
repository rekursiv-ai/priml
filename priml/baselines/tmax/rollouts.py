r"""Validate and pack rollout data for the PriML TMax baseline.

The baseline splits work between upstream TMax and PriML. Upstream TMax
provides the tasks, Harbor sandboxes, agent loop, verifier, and vLLM rollout
workers. PriML prepares the pinned assets, starts and drives those components,
records their output, trains the native Qwen3.5 model with DPPO and FSDP,
syncs and exports weights, and launches evaluation with the upstream tools.

This module handles the rollout data between collection and training. It reads
JSONL records, validates their fields, computes group advantages, and packs
tokens for the training step. Each record includes prompt and response tokens,
the generation log probability of each response token, a verifier reward, and
a prompt-group identifier.

Upstream training also uses a token-level mask to exclude tool output from the
loss. The released writer does not save this mask. Our live writer adds it,
while older published traces need a matching sidecar. When tool masking is
enabled, :func:`pack_rollouts` fails if the mask is missing instead of silently
training on sandbox output.

The advantage, packing, and position logic is ported from TMax commit
``6d3d606``. Source-parity tests compare it with values produced by the
upstream implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import json
import math

from torch import Tensor

import torch


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

AdvantageNormalization = Literal["standard", "centered", "maxrl"]
"""How a prompt group's rewards become advantages; TMax's 4B recipe centers."""


_REQUIRED_FIELDS = (
    "step",
    "sample_idx",
    "prompt_idx",
    "prompt_tokens",
    "response_tokens",
    "logprobs",
    "reward",
    "finish_reason",
)
"""Fields every released TMax rollout record carries.

Verified against the published trace of ``allenai/tmax-9b`` run fragment
``swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677``: all 1024 records of its
shard carry exactly these plus ``advantage``, ``dataset``, ``ground_truth``,
and ``request_info``, with stable types.

``logprobs`` is required rather than tolerated when absent: DPPO's ratio is
anchored on the rollout policy's own logprobs (upstream refuses ``loss_fn=dppo``
without ``use_vllm_logprobs``), and a zero-filled fallback would invent a
confident behavior policy at every sampled token.
"""

_STANDARD_EPS = 1e-8
"""Upstream's floor on a group's reward standard deviation."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RolloutRecord:
    """One scored terminal trajectory, as the learner needs to see it.

    Mirrors the subset of TMax's ``rl_utils.RolloutRecord`` the DPPO update
    reads, plus :attr:`tool_mask`. The fields it drops -- ``dataset``,
    ``ground_truth``, ``request_info`` -- describe the task and the sandbox,
    which are TMax's business and not an input to the objective.
    """

    step: int
    """The rollout-collection step that produced the sample."""

    sample_idx: int
    """Index within the step's batch; sets the packing order and group layout."""

    prompt_idx: int
    """Which prompt group the sample belongs to, for the group baseline."""

    prompt_tokens: tuple[int, ...]
    """Prompt token ids, excluded from the loss but scored for context."""

    response_tokens: tuple[int, ...]
    """Sampled response token ids, the only positions the loss can touch."""

    logprobs: tuple[float, ...]
    r"""Behavior logprobs :math:`\log\mu_{\theta'}` from vLLM, one per response token.

    DPPO's trust region is anchored on these, not on a recomputed baseline, so
    they are load-bearing rather than diagnostic.
    """

    reward: float
    """The verifier's scalar score for the trajectory."""

    finish_reason: str
    """Why generation stopped; ``"stop"`` means the agent submitted."""

    tool_mask: tuple[int, ...] | None = None
    """Per-response-token flag: truthy where the POLICY sampled the token.

    Falsy entries are sandbox output spliced into the transcript. ``None`` means
    the shard did not record it, which :func:`pack_rollouts` rejects.
    """

    advantage: float | None = None
    """The advantage the producing run recorded, when it recorded one.

    Never consumed by the update: :func:`group_advantages` recomputes it so the
    baseline owns its own arithmetic. Kept so a test can hold the recomputation
    against what a real run actually wrote.
    """


def read_rollout_records(path: Path) -> list[RolloutRecord]:
    """Parse one TMax rollout shard.

    Args:
      path: A ``*_rollouts_*.jsonl`` shard written by TMax.

    Returns:
      records: One entry per non-blank line, in file order.

    Raises:
      ValueError: A record is missing a required field, or its logprob count
        does not match its response length.

    """
    records: list[RolloutRecord] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records.append(
                _record(
                    cast(Mapping[str, object], json.loads(line)),
                    origin=f"{path}:{number}",
                ),
            )
    return records


def _record(payload: Mapping[str, object], *, origin: str) -> RolloutRecord:
    """Validate and narrow one decoded JSON record.

    Args:
      payload: The decoded object.
      origin: File and line, for error messages.

    Returns:
      record: The narrowed record.

    Raises:
      ValueError: A required field is absent, or the logprobs do not line up
        with the response.

    """
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"{origin}: rollout record is missing {missing}.")
    response = tuple(
        parse_json_integer_array(
            payload["response_tokens"],
            origin=origin,
            field="response_tokens",
        ),
    )
    logprobs = tuple(
        parse_json_float(value, origin=origin, field=f"logprobs[{index}]")
        for index, value in enumerate(
            _json_array(payload["logprobs"], origin=origin, field="logprobs"),
        )
    )
    if len(logprobs) != len(response):
        raise ValueError(
            f"{origin}: {len(logprobs)} logprobs for {len(response)} response "
            "tokens; DPPO needs a behavior logprob at every sampled token.",
        )
    raw_mask = payload.get("tool_mask")
    tool_mask = (
        None
        if raw_mask is None
        else tuple(
            parse_json_integer_array(
                raw_mask,
                origin=origin,
                field="tool_mask",
            ),
        )
    )
    if tool_mask is not None and len(tool_mask) != len(response):
        raise ValueError(
            f"{origin}: {len(tool_mask)} tool-mask values for "
            f"{len(response)} response tokens.",
        )
    advantage = payload.get("advantage")
    return RolloutRecord(
        step=parse_json_integer(payload["step"], origin=origin, field="step"),
        sample_idx=parse_json_integer(
            payload["sample_idx"],
            origin=origin,
            field="sample_idx",
        ),
        prompt_idx=parse_json_integer(
            payload["prompt_idx"],
            origin=origin,
            field="prompt_idx",
        ),
        prompt_tokens=tuple(
            parse_json_integer_array(
                payload["prompt_tokens"],
                origin=origin,
                field="prompt_tokens",
            ),
        ),
        response_tokens=response,
        logprobs=logprobs,
        reward=parse_json_float(payload["reward"], origin=origin, field="reward"),
        finish_reason=str(payload["finish_reason"]),
        tool_mask=tool_mask,
        advantage=(
            None
            if advantage is None
            else parse_json_float(advantage, origin=origin, field="advantage")
        ),
    )


def _json_array(value: object, *, origin: str, field: str) -> Iterable[object]:
    """Narrow a decoded JSON value to an array.

    Args:
      value: Decoded JSON.
      origin: Source location included in errors.
      field: Field name included in errors.

    Returns:
      items: The value as an iterable.

    Raises:
      ValueError: The value is not a JSON array.

    """
    if not isinstance(value, list):
        raise ValueError(  # noqa: TRY004 -- JSON validation has one public error type.
            f"{origin}: {field}: expected a JSON array, got {type(value).__name__}.",
        )
    return cast(list[object], value)


def parse_json_integer_array(
    value: object,
    *,
    origin: str,
    field: str,
) -> list[int]:
    """Parse an array of exact JSON integers with field-aware errors.

    Args:
      value: Decoded JSON array.
      origin: Source location included in errors.
      field: Array field name included in errors.

    Returns:
      items: The array's values as ints.

    """
    return [
        parse_json_integer(item, origin=origin, field=f"{field}[{index}]")
        for index, item in enumerate(_json_array(value, origin=origin, field=field))
    ]


def parse_json_integer(value: object, *, origin: str, field: str) -> int:
    """Parse an integer without narrowing a Python int through float.

    Args:
      value: Decoded JSON scalar.
      origin: Source location included in errors.
      field: Scalar field name included in errors.

    Returns:
      number: The exact integer value.

    Raises:
      ValueError: The value is boolean, nonnumeric, nonfinite, or fractional.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 -- JSON validation has one public error type.
            f"{origin}: {field}: expected an integer, got {value!r}.",
        )
    if isinstance(value, int):
        return value
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{origin}: {field}: expected an integer, got {value!r}.")
    return int(value)


def parse_json_float(value: object, *, origin: str, field: str) -> float:
    """Parse a finite JSON floating-point field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 -- JSON validation has one public error type.
            f"{origin}: {field}: expected a finite number, got {value!r}.",
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{origin}: {field}: expected a finite number, got {value!r}.",
        )
    return result


def group_advantages(
    rewards: Sequence[float],
    *,
    num_samples_per_prompt: int,
    normalization: AdvantageNormalization = "centered",
) -> Tensor:
    """Turn per-rollout rewards into per-rollout advantages, group by group.

    DPPO has no value network: the baseline is the mean reward of the other
    samples drawn from the same prompt, which is why ``rewards`` must arrive in
    the consecutive grouped order the rollout batch was built in.

    ``centered`` -- TMax's 4B setting -- subtracts the group mean and does NOT
    divide by the group's spread. The division is what ``standard`` does, and it
    inflates the gradient of a group that nearly all succeeded, which is exactly
    the group carrying the least information.

    Args:
      rewards: One reward per rollout, grouped consecutively by prompt.
      num_samples_per_prompt: Group size; TMax's 4B recipe draws 32.
      normalization: Which of upstream's three baselines to apply.

    Returns:
      advantages: One float32 advantage per rollout, in input order.

    Raises:
      ValueError: The group size is not positive, the reward count is not a
        multiple of it, or ``normalization`` is unknown.

    """
    if num_samples_per_prompt <= 0:
        raise ValueError(
            f"num_samples_per_prompt must be positive, got {num_samples_per_prompt}.",
        )
    if len(rewards) % num_samples_per_prompt != 0:
        raise ValueError(
            f"{len(rewards)} rewards is not a whole number of groups of "
            f"{num_samples_per_prompt}.",
        )
    grouped = torch.tensor(rewards, dtype=torch.float32).reshape(
        -1,
        num_samples_per_prompt,
    )
    mean = grouped.mean(dim=-1, keepdim=True)
    if normalization == "centered":
        advantages = grouped - mean
    elif normalization == "standard":
        spread = grouped.std(dim=-1, keepdim=True, unbiased=False)
        advantages = (grouped - mean) / (spread + _STANDARD_EPS)
    elif normalization == "maxrl":
        advantages = torch.where(
            mean > 0.0,
            (grouped - mean) / torch.where(mean > 0.0, mean, torch.ones_like(mean)),
            torch.zeros_like(grouped),
        )
    else:
        raise ValueError(  # pyright: ignore[reportUnreachable]
            f"Invalid advantage normalization: {normalization!r}.",
        )
    return advantages.reshape(-1)


@dataclass(frozen=True, kw_only=True, slots=True)
class PackedRow:
    """Several rollouts concatenated into one training row.

    TMax packs rather than pads because a terminal trajectory can run to tens of
    thousands of tokens while another ends in a few hundred; a padded batch
    would spend most of its compute on padding. The cost is that every tensor
    here is segment-aware: attention must not cross a segment boundary, which
    is why :func:`priml.baselines.tmax.scoring.response_logprobs` runs one
    segment per forward pass.
    """

    query_responses: Tensor
    """Concatenated prompt and response ids, ``[tokens]``, int64."""

    attention_mask: Tensor
    """1-based segment id per token, ``0`` never appears, ``[tokens]``, int64.

    Upstream calls this ``attention_masks``; it is the intra-document mask's
    source, not a padding mask.
    """

    position_ids: Tensor
    """Per-segment position ids restarting at 0, ``[tokens]``, int64."""

    response_mask: Tensor
    """``rollout_index + 1`` on trainable response tokens, else ``0``, int64.

    Zero therefore covers prompt tokens AND tool-output tokens. Carrying the
    rollout index rather than a bare flag is what lets :func:`pack_rollouts`
    scatter advantages with one gather.
    """

    prompt_mask: Tensor
    """``1`` on prompt tokens, ``0`` on response tokens, ``[tokens]``, int64."""

    rollout_sample_ids: Tensor
    """Owning rollout's ``sample_idx`` per token, ``[tokens]``, int64."""

    vllm_logprobs: Tensor
    r"""vLLM :math:`\log\mu_{\theta'}` per token, NaN on prompts, float32."""

    advantages: Tensor
    """Owning rollout's advantage per token, ``0`` off the response, float32."""

    model_steps: Tensor
    """The rollout collection step per packed token, ``[tokens]``, int64."""

    dones: Tensor
    """1-based input-batch index at each rollout end, else ``0``, int64."""

    num_actions: int
    """Number of response tokens after padding removal in this row."""

    packed_seq_lens: tuple[int, ...]
    """Length of each unbroken prompt-response sequence in this row."""

    original_responses: tuple[tuple[int, ...], ...]
    """Response token ids after padding removal, retained for traceability."""


def segment_positions(segments: Tensor) -> Tensor:
    """Return position ids that restart at zero in every segment.

    Args:
      segments: 1-based segment id per token, ``[tokens]``.

    Returns:
      positions: ``[tokens]``, int64.

    """
    positions = torch.zeros_like(segments, dtype=torch.long)
    for segment in range(1, int(segments.max().item()) + 1):
        selected = segments == segment
        positions[selected] = torch.arange(
            int(selected.sum().item()),
            device=segments.device,
        )
    return positions


def pack_rollouts(
    records: Sequence[RolloutRecord],
    *,
    advantages: Tensor,
    pack_length: int,
    pad_token_id: int,
    mask_tool_use: bool = True,
    min_num_batches: int = 1,
) -> list[PackedRow]:
    """Concatenate scored rollouts into training rows.

    Follows upstream's loop exactly, including the detail that decides the
    geometry: a rollout is never split. A row is flushed BEFORE appending a
    rollout that would overflow ``pack_length``, so a single trajectory longer
    than the budget simply gets a row to itself rather than being truncated.

    Args:
      records: Scored rollouts, in the order the batch was collected.
      advantages: One advantage per record, aligned with ``records``.
      pack_length: Token budget per row.
      pad_token_id: Padding id. Prompts containing it are rejected. Response
        occurrences and their matching masks and logprobs are removed.
      mask_tool_use: Exclude tool-output tokens from the response mask. The
        released TMax default, and the reason ``tool_mask`` is required.
      min_num_batches: Lower bound on rows. TMax reduces the effective pack
        length to the total-token budget divided by this count, when possible,
        so distributed workers do not receive an empty batch.

    Returns:
      rows: Packed rows, in order.

    Raises:
      ValueError: ``advantages`` does not align with ``records``, a record's
        logprobs do not match its response, or ``mask_tool_use`` is set and a
        record carries no ``tool_mask``. Also raised when ``min_num_batches``
        is not positive or a prompt contains the padding id.

    """
    if advantages.numel() != len(records):
        raise ValueError(
            f"{advantages.numel()} advantages for {len(records)} records.",
        )
    if min_num_batches <= 0:
        raise ValueError(f"min_num_batches must be positive, got {min_num_batches}.")
    if mask_tool_use:
        absent = [r.sample_idx for r in records if r.tool_mask is None]
        if absent:
            raise ValueError(
                "mask_tool_use is set but these rollouts carry no tool_mask: "
                f"{absent[:8]}{'...' if len(absent) > 8 else ''}. Training "
                "without it would fit the policy to sandbox output. TMax's "
                "released rollout writer omits the field; see this baseline's "
                "README for the two ways to supply it.",
            )

    # Upstream indexes advantages from 1 so that response-mask value 0 -- prompt
    # and tool positions alike -- reads a zero advantage with the same gather.
    lookup = torch.zeros(len(records) + 1, dtype=torch.float32)
    lookup[1:] = advantages.to(torch.float32)

    total_tokens = sum(
        len(record.prompt_tokens)
        + sum(token != pad_token_id for token in record.response_tokens)
        for record in records
    )
    effective_pack_length = pack_length
    if total_tokens > 0 and min_num_batches > 1:
        effective_pack_length = min(total_tokens // min_num_batches, pack_length)
        effective_pack_length = max(effective_pack_length, 1)

    rows: list[PackedRow] = []
    builder = _RowBuilder()
    for index, record in enumerate(records):
        if any(token == pad_token_id for token in record.prompt_tokens):
            raise ValueError(
                f"sample_idx={record.sample_idx}: prompt contains pad token "
                f"{pad_token_id}; TMax requires prompts to be unpadded.",
            )
        prompt = list(record.prompt_tokens)
        kept = [
            (token, logprob, flag)
            for token, logprob, flag in zip(
                record.response_tokens,
                record.logprobs,
                record.tool_mask or (1,) * len(record.response_tokens),
                strict=True,
            )
            if token != pad_token_id
        ]
        if (
            builder.length
            and builder.length + len(prompt) + len(kept) > effective_pack_length
        ):
            rows.append(builder.finish(lookup))
            builder = _RowBuilder()
        builder.add(
            prompt=prompt,
            response=kept,
            rollout_index=index,
            sample_idx=record.sample_idx,
            model_step=record.step,
            original_response=tuple(token for token, _, _ in kept),
            mask_tool_use=mask_tool_use,
        )
    if builder.length:
        rows.append(builder.finish(lookup))
    return rows


class _RowBuilder:
    """Accumulate rollouts into one packed row's python lists."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.segments: list[int] = []
        self.response_mask: list[int] = []
        self.prompt_mask: list[int] = []
        self.sample_ids: list[int] = []
        self.model_steps: list[int] = []
        self.behavior_logprobs: list[float] = []
        self.dones: list[int] = []
        self.num_actions = 0
        self.sequence_lengths: list[int] = []
        self.original_responses: list[tuple[int, ...]] = []

    @property
    def length(self) -> int:
        """Tokens accumulated so far.

        Returns:
          length: Token count.

        """
        return len(self.tokens)

    def add(
        self,
        *,
        prompt: Sequence[int],
        response: Sequence[tuple[int, float, int]],
        rollout_index: int,
        sample_idx: int,
        model_step: int,
        original_response: tuple[int, ...],
        mask_tool_use: bool,
    ) -> None:
        """Append one rollout's tokens to the row.

        Args:
          prompt: Unpadded prompt token ids.
          response: ``(token, behavior_logprob, tool_flag)`` per response token.
          rollout_index: Position in the batch; becomes the response-mask value.
          sample_idx: The record's own index, carried per token.
          model_step: Collection step that produced this rollout.
          original_response: Response token ids after padding removal.
          mask_tool_use: Zero the response mask where the tool flag is falsy.

        """
        segment = self.segments[-1] + 1 if self.segments else 1
        sequence_length = len(prompt) + len(response)
        self.segments.extend([segment] * sequence_length)
        self.tokens.extend(prompt)
        self.tokens.extend(token for token, _, _ in response)
        self.prompt_mask.extend([1] * len(prompt))
        self.prompt_mask.extend([0] * len(response))
        self.response_mask.extend([0] * len(prompt))
        self.response_mask.extend(
            rollout_index + 1 if flag or not mask_tool_use else 0
            for _, _, flag in response
        )
        self.sample_ids.extend([sample_idx] * (len(prompt) + len(response)))
        self.model_steps.extend([model_step] * sequence_length)
        # A prompt token has no behavior logprob. Upstream writes NaN rather
        # than a number so that a masking bug surfaces as NaN instead of
        # quietly contributing a plausible ratio.
        self.behavior_logprobs.extend([float("nan")] * len(prompt))
        self.behavior_logprobs.extend(logprob for _, logprob, _ in response)
        self.dones.extend([0] * max(sequence_length - 1, 0))
        self.dones.append(rollout_index + 1)
        self.num_actions += len(response)
        self.sequence_lengths.append(sequence_length)
        self.original_responses.append(original_response)

    def finish(self, lookup: Tensor) -> PackedRow:
        """Freeze the row into tensors.

        Args:
          lookup: Advantage indexed from 1 by response-mask value.

        Returns:
          row: The packed row.

        """
        segments = torch.tensor(self.segments, dtype=torch.long)
        response_mask = torch.tensor(self.response_mask, dtype=torch.long)
        return PackedRow(
            query_responses=torch.tensor(self.tokens, dtype=torch.long),
            attention_mask=segments,
            position_ids=segment_positions(segments),
            response_mask=response_mask,
            prompt_mask=torch.tensor(self.prompt_mask, dtype=torch.long),
            rollout_sample_ids=torch.tensor(self.sample_ids, dtype=torch.long),
            vllm_logprobs=torch.tensor(
                self.behavior_logprobs,
                dtype=torch.float32,
            ),
            advantages=lookup[response_mask],
            model_steps=torch.tensor(self.model_steps, dtype=torch.long),
            dones=torch.tensor(self.dones, dtype=torch.long),
            num_actions=self.num_actions,
            packed_seq_lens=tuple(self.sequence_lengths),
            original_responses=tuple(self.original_responses),
        )
