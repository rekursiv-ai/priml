"""Tests for spatial position tables."""

from __future__ import annotations

import pytest
import torch

from priml.math.position_embedding import (
    image_token_positions,
    sincos_position_table,
)


def test_image_token_positions_put_cls_before_row_major_grid() -> None:
    positions = image_token_positions(2, torch.device("cpu"))
    assert positions.shape == (1, 5, 2)
    assert torch.equal(
        positions,
        torch.tensor([[[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]]]),
    )


def test_sincos_table_has_zero_lead_and_float_output() -> None:
    table = sincos_position_table(4, 2, lead=2, compute_dtype=torch.float64)
    assert table.shape == (6, 4)
    assert table.dtype == torch.float32
    assert torch.equal(table[:2], torch.zeros(2, 4))
    assert table[2, 0] == 0
    assert table[2, 1] == 1
    assert table[2, 2] == 0
    assert table[2, 3] == 1


def test_sincos_table_rejects_non_quarter_width() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        sincos_position_table(6, 2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
