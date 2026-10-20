"""Filter reasons: how a pipeline stage marks a sample as rejected.

Stages never drop a sample; they append a reason, and the pipeline's sink
decides. Everything that writes or reads ``filter_reasons`` lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import logging

from treekle import from_plain


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.data.custom_types import FilterableSample


__all__ = [
    "add_filter_reason",
    "read_filter_reasons",
]


logger = logging.getLogger(__name__)


# Once an image fails to decode, no further filter reasons are meaningful and
# logging each one floods the logs; this sentinel short-circuits both.
_DECODE_FAILED_REASON: Final = "CropDuringDecodeImage:decode_failed"


def add_filter_reason(sample: FilterableSample, filter: str, reason: str) -> None:
    """Add filter reason to sample.

    Args:
      sample: Sample to mark for filtering; short-circuits on decode failure
        to avoid log spam.
      filter: Stage name that produced the rejection.
      reason: Specific condition causing rejection (appended to stage name).

    """
    filter_reasons = sample.get("filter_reasons", [])
    if _DECODE_FAILED_REASON in filter_reasons:
        return
    full_reason = f"{filter}:{reason}"
    # A new list, not an append: a sibling sample may share the old one.
    sample["filter_reasons"] = [*filter_reasons, full_reason]
    if full_reason != _DECODE_FAILED_REASON:
        logger.debug("Filter: %s", full_reason)


def read_filter_reasons(sample: Mapping[str, object]) -> list[str]:
    """Return the string ``filter_reasons`` of an open pipeline sample.

    Pipeline-level samples are ``dict[str, object]`` -- any stage may write the
    key -- so it is checked here, once, rather than cast at every reader.

    Args:
      sample: A sample or batch of any shape.

    Returns:
      reasons: Its string reasons, or ``[]`` when it has none.

    """
    reasons = from_plain(sample.get("filter_reasons"), list[object], default=[])
    return [reason for reason in reasons if isinstance(reason, str)]
