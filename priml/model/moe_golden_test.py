"""Bit-for-bit golden test for ``MoE``.

Regenerate (after an intentional numeric change)::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/moe_golden_test.py

Run through ``pytest``: the priml ``conftest.py`` sets ``MKL_CBWR`` and caps
the math threads before torch imports, both preconditions for the golden to
reproduce. Minting from a bare ``python`` process skips that setup and pins the
golden to the mint host.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

import pytest
import torch

from priml.model.moe import MoE, Router
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
def test_moe_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_GOLDENS,
        golden_name=f"moe_min_{device}",
        build_module=lambda: (
            MoE.Config(
                channels_in=16,
                router=Router.Config(num_experts=4, top_k=2),
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
