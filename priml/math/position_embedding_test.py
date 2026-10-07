"""Tests for spatial position tables."""

from __future__ import annotations

from unittest.mock import Mock

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
    assert image_token_positions(2, torch.device("meta")).device == torch.device("meta")


def test_sincos_table_has_default_lead_and_float_output() -> None:
    table = sincos_position_table(4, 2, compute_dtype=torch.float64)
    assert table.shape == (5, 4)
    assert table.dtype == torch.float32
    # The position table reserves one leading token row by contract.
    assert torch.equal(table[:1], torch.zeros(1, 4))
    assert table[1, 0] == 0
    assert table[1, 1] == 1
    assert table[1, 2] == 0
    assert table[1, 3] == 1


def test_sincos_table_passes_compute_dtype_to_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arange = Mock(wraps=torch.arange)
    monkeypatch.setattr(torch, "arange", arange)

    sincos_position_table(8, 2, compute_dtype=torch.float64)

    assert [call.kwargs["dtype"] for call in arange.call_args_list] == [
        torch.float64,
        torch.float64,
    ]


def test_sincos_table_encodes_columns_then_rows_at_each_frequency() -> None:
    table = sincos_position_table(8, 2, lead=2, compute_dtype=torch.float64)
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0.84147096, 0.00999983, 0.5403023, 0.99995, 0, 0, 1, 1],
            [0, 0, 1, 1, 0.84147096, 0.00999983, 0.5403023, 0.99995],
            [
                0.84147096,
                0.00999983,
                0.5403023,
                0.99995,
                0.84147096,
                0.00999983,
                0.5403023,
                0.99995,
            ],
        ],
    )
    torch.testing.assert_close(table, expected, rtol=0, atol=1e-7)


def test_sincos_table_rejects_non_quarter_width() -> None:
    with pytest.raises(
        ValueError,
        match=r"^position channels must be divisible by four$",
    ):
        sincos_position_table(6, 2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
