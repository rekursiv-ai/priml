"""Validate mapped checkpoint tensors before loading any model state."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


def validate_state_dict(
    state: Mapping[str, Tensor],
    *,
    expected: Mapping[str, Tensor],
) -> None:
    """Require complete matching names, shapes and dtypes before copying tensors.

    Checkpoint format decoding and name remapping belong to the caller. This
    validation never converts a dtype or changes either mapping or its tensors.

    Args:
      state: Complete mapped checkpoint state.
      expected: Destination model's complete state dictionary.

    Raises:
      ValueError: A tensor name, shape or dtype differs.

    """
    if state.keys() != expected.keys():
        raise ValueError(
            "Checkpoint tensor names differ: "
            f"missing={sorted(expected.keys() - state.keys())}; "
            f"unexpected={sorted(state.keys() - expected.keys())}.",
        )
    for name, value in state.items():
        destination = expected[name]
        if value.shape != destination.shape or value.dtype != destination.dtype:
            raise ValueError(f"Checkpoint tensor shape or dtype differs: {name}.")
