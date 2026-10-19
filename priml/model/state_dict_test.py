"""Reject incomplete or coercing checkpoint loads before any tensor changes."""

from __future__ import annotations

from typing import Literal

import re

from torch import nn

import pytest
import torch

from priml.model.state_dict import validate_state_dict
from priml.testing.golden import tensor_bits_equal


def test_complete_mapped_state_loads_without_dtype_or_storage_conversion() -> None:
    model = nn.Linear(3, 2)
    state = {
        name: value.detach().clone().add_(7)
        for name, value in model.state_dict().items()
    }
    original = dict(state)
    before = torch.get_rng_state()

    validate_state_dict(state, expected=model.state_dict())
    model.load_state_dict(state, strict=True)

    assert torch.equal(torch.get_rng_state(), before)
    assert state.keys() == original.keys()
    for name, value in model.state_dict().items():
        assert state[name] is original[name]
        assert tensor_bits_equal(value, state[name])


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "Checkpoint tensor names differ: missing=['bias']; unexpected=[]."),
        (
            "unexpected",
            "Checkpoint tensor names differ: missing=[]; unexpected=['extra'].",
        ),
        ("shape", "Checkpoint tensor shape or dtype differs: bias."),
        ("dtype", "Checkpoint tensor shape or dtype differs: bias."),
    ],
)
def test_rejected_state_leaves_every_model_tensor_unchanged(
    defect: Literal["missing", "unexpected", "shape", "dtype"],
    message: str,
) -> None:
    model = nn.Linear(3, 2)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    state = {name: value.clone().add_(7) for name, value in before.items()}
    # The earlier weight would be copied before torch discovers a bad bias.
    if defect == "missing":
        del state["bias"]
    elif defect == "unexpected":
        state["extra"] = torch.ones(4)
    elif defect == "shape":
        state["bias"] = torch.ones(4)
    else:
        state["bias"] = state["bias"].double()

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        validate_state_dict(state, expected=model.state_dict())

    for name, value in model.state_dict().items():
        assert tensor_bits_equal(value, before[name])


def test_empty_state_dictionaries_are_valid() -> None:
    validate_state_dict({}, expected={})


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
