"""Tests for legacy state-dict key absorption."""

from __future__ import annotations

from torch import nn

import torch

from priml.model.legacy_keys import absorb_legacy_keys


def test_absorb_legacy_keys() -> None:
    module = nn.Module()
    projection = nn.Linear(2, 2)
    module.proj_x = projection
    weight, bias = projection.weight, projection.bias
    assert weight is not None
    assert bias is not None
    absorb_legacy_keys(module, {"old_x": "proj_x"})
    expected = {
        "old_x.weight": torch.randn_like(weight),
        "old_x.bias": torch.randn_like(bias),
    }

    module.load_state_dict(expected, strict=True)

    assert torch.equal(weight, expected["old_x.weight"])
    assert torch.equal(bias, expected["old_x.bias"])
    assert set(module.state_dict()) == {"proj_x.weight", "proj_x.bias"}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
