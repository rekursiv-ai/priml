"""Bit-for-bit golden test for ``TransformerBlock``.

Regenerate (after an intentional numeric change)::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer_golden_test.py

Run through ``pytest``: the repo-root ``conftest.py`` sets
``MKL_CBWR=COMPATIBLE`` before torch imports, the precondition for the
golden to reproduce across x86 hosts. Minting from a bare ``python``
process pins it to the mint host and fails elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priml.model.attention import SelfAttention
from priml.model.transformer import TransformerBlock
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
def test_transformer_block_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_GOLDENS,
        golden_name=f"transformer_block_min_{device}",
        build_module=lambda: (
            TransformerBlock.Config(
                channels_in=16,
                attn=SelfAttention.Config(heads=2, channels_head=8),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(2, 4, 16), device),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
