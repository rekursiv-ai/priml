"""Read openai/math rollout JSONL files and yield packed rows per optimizer update.

The mechanics are TMax's, imported rather than copied: prompts and responses
of any length are concatenated into rows instead of padded, group advantages
come from the same centered baseline, and packing, sharding, and the
``(step, sample_idx)`` grouping rules are the borrowed implementations.

What this module names is the part the openai/math substrate changes: a row
holds one whole Lean episode, so the pack length is the episode's own budget.
The group filters keep TMax's defaults (drop a zero-spread group, refill its
slot), because training draws lemma tasks, whose pass rates are mixed, rather
than whole challenges, where nearly every group would be all zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from priml.baselines.openaimath.episode import MAX_PROMPT_TOKENS, RESPONSE_LENGTH
from priml.baselines.tmax.data import TMaxRolloutData
from priml.baselines.tmax.rollouts import (
    RolloutRecord,
    parse_json_float,
    parse_json_integer,
    parse_json_integer_array,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


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
"""Fields every collected transcript must carry before it can be trained on."""


def record_from_payload(payload: Mapping[str, object]) -> RolloutRecord:
    """Build a rollout record from a just-collected transcript.

    The live path harvests records as JSON-shaped mappings and trains on them in
    the same process, while the replay path reads the same fields back off disk
    with the borrowed reader. Both go through the same exact-integer and
    finite-float parsing, so a record the learner can replay is a record it could
    equally have collected, and a malformed transcript fails the update rather
    than reaching the objective as a plausible number.

    Args:
      payload: One collected record.

    Returns:
      record: The validated record.

    Raises:
      ValueError: A required field is absent, a token array is not an array, or
        the logprobs or tool mask do not line up with the response.

    """
    origin = f"step={payload.get('step')} sample_idx={payload.get('sample_idx')}"
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
            _array(payload["logprobs"], origin=origin, field="logprobs"),
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
            parse_json_integer_array(raw_mask, origin=origin, field="tool_mask"),
        )
    )
    if tool_mask is not None and len(tool_mask) != len(response):
        raise ValueError(
            f"{origin}: {len(tool_mask)} tool-mask values for "
            f"{len(response)} response tokens.",
        )
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
    )


def _array(value: object, *, origin: str, field: str) -> list[object]:
    """Narrow a decoded value to a JSON array.

    Args:
      value: A decoded JSON value.
      origin: Source location included in errors.
      field: Field name included in errors.

    Returns:
      items: The value as a list.

    Raises:
      ValueError: The value is not a JSON array.

    """
    if not isinstance(value, list):
        raise ValueError(  # noqa: TRY004 -- JSON validation has one public error type.
            f"{origin}: {field}: expected a JSON array, got {type(value).__name__}.",
        )
    return cast(list[object], value)


class OpenMathRolloutData(TMaxRolloutData):
    """Read openai/math rollout JSONL files and yield packed rows per update.

    One packed row is one Lean episode: the agent's own tokens and the
    sandbox's compile output, concatenated with its prompt, in the order they
    were generated. The learner sees exactly the token stream vLLM produced,
    which is what lets DPPO's ratio be anchored on the behavior logprobs the
    rollout recorded.
    """

    class Config(TMaxRolloutData.Config):
        """openai/math grouping and packing configuration."""

        pack_length: int = MAX_PROMPT_TOKENS + RESPONSE_LENGTH
        """Maximum prompt-plus-response tokens per packed row.

        One whole episode: the episode's prompt budget plus its response budget,
        which is also the rollout server's context bound. Lean work needs more
        prompt room than a terminal task: the task instruction carries the full
        theorem statement, its imports, and the surrounding definitions, and a
        Mathlib context is not compressible the way a shell transcript is. The
        live dataset refuses a runner whose episodes would not fit.
        """
