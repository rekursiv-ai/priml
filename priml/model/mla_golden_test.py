"""Bit-for-bit golden test for ``MultiHeadLatentAttention``.

Regenerate (after an intentional numeric change)::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/mla_golden_test.py

Run through ``pytest``: the priml ``conftest.py`` caps the math threads before
torch imports, and the harness computes in float64 and rounds once to float32,
which is what makes the golden reproduce across hosts. Minting from a bare
``python`` process skips that setup.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

import pytest
import torch

from priml.model.mla import MultiHeadLatentAttention
from priml.model.rope import RoPE
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
def test_mla_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_GOLDENS,
        golden_name=f"mla_min_{device}",
        build_module=lambda: (
            MultiHeadLatentAttention.Config(
                channels_in=16,
                heads=2,
                channels_qk_nope_head=8,
                channels_qk_rope_head=4,
                channels_v_head=8,
                kv_lora_rank=8,
                rope=RoPE.Config(channels_head=4, base=10_000),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(2, 4, 16), device),
        seed=0,
        run=lambda m, x: _first_tensor(m(x)),
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
