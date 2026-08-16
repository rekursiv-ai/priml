"""Bit-for-bit golden test for ``MMDiTBlock``.

Regenerate (after an intentional numeric change)::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/mmdit_golden_test.py

Run through ``pytest``: the priml ``conftest.py`` caps the math threads before
torch imports, and the harness computes in float64 and rounds once to float32,
which is what makes the golden reproduce across hosts. Minting from a bare
``python`` process skips that setup.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention import MultiStreamAttention
from priml.model.mmdit import MMDiTBlock
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    move_to_device,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_GOLDENS = Path(__file__).parent.resolve() / "goldens"


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_mmdit_block_bfb(device: str) -> None:
    def build() -> nn.Module:
        cfg = MMDiTBlock.Config(channels_in=16, num_streams=2)
        cfg.attn = MultiStreamAttention.Config(heads=2)
        return cfg.make().to(device)

    def build_input() -> list[Tensor]:
        return move_to_device([torch.randn(2, 4, 16), torch.randn(2, 6, 16)], device)

    assert_bfb_against_golden(
        golden_dir=_GOLDENS,
        golden_name=f"mmdit_block_min_{device}",
        build_module=build,
        build_input=build_input,
        seed=0,
        run=lambda m, xs: _first_tensor(m(xs)),
    )


def _first_tensor(result: Tensor | tuple[object, ...] | list[object]) -> Tensor:
    """Extract the primary output tensor from a module's return value."""
    if isinstance(result, (tuple, list)):
        head = result[0]
        assert isinstance(head, Tensor)
        return head
    assert isinstance(result, Tensor)
    return result


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
