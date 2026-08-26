"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import override

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.model.attention.kernel import SdpaFused
from priml.model.attention.window import (
    causal_chunk_mask,
    layer_window,
    window_mask,
    window_sizes,
)
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


class _Window(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return _window_contract(x)


def _window_contract(x: Tensor) -> Tensor:
    local = window_mask(x, x, window=2)
    chunk = causal_chunk_mask(x[..., -2:, :, :], x)
    assert local is not None
    assert chunk is not None
    return torch.cat([local.flatten(), chunk.flatten()])


def test_window_sizes_always_end_long() -> None:
    """The last layer predicts the next token, so it must see everything."""
    assert window_sizes(num_layers=5, max_seq_len=64, pattern="SSSL") == [
        32,
        32,
        32,
        64,
        64,
    ]


def test_layer_window_flattens_nested_depth_index() -> None:
    assert (
        layer_window(
            depth_index=((1, 2), (1, 2)),
            max_seq_len=64,
            pattern="SSSL",
        )
        == 64
    )


def test_layer_window_rejects_unspecified_depth_index() -> None:
    with pytest.raises(ValueError, match="depth_index"):
        layer_window(depth_index=(), max_seq_len=64, pattern="SSSL")


def test_window_sizes_rejects_an_unknown_symbol() -> None:
    """A typo would otherwise cycle silently into a KeyError per layer."""
    with pytest.raises(ValueError, match="only S and L"):
        window_sizes(num_layers=2, max_seq_len=8, pattern="SX")


def test_a_window_admits_its_own_position_and_w_before_it() -> None:
    """``<=``, not ``<``.

    A fused kernel's ``window_size=(w, 0)`` admits w keys of history IN
    ADDITION to the query's own, so the exclusive form attends to one key fewer
    per row -- a different model rather than a rounding difference.
    """
    q = k = torch.zeros(1, 8, 1, 4)
    mask = window_mask(q, k, window=2)
    assert mask is not None
    admitted = torch.isfinite(mask)
    assert admitted[5].tolist() == [False, False, False, True, True, True, False, False]


def test_a_window_reaching_the_context_needs_no_mask() -> None:
    """Masking there costs a kernel dispatch and admits exactly the same keys."""
    q = k = torch.zeros(1, 8, 1, 4)
    assert window_mask(q, k, window=8) is None
    assert window_mask(q, k, window=-1) is None


def test_a_window_and_is_causal_together_are_accepted() -> None:
    """SDPA refuses a mask beside ``is_causal``, and the window IS a mask.

    Every windowed caller passes both -- the window is the recipe, the causal
    flag is the model -- so a kernel forwarding them unchanged raises
    ``Explicit attn_mask should not be set when is_causal=True``.

    Pinned to the MATH backend, which is the only one that refuses: flash
    accepts both and silently ignores the flag, so a test at free dispatch
    passes on a kernel that is broken for every caller pinning MATH -- the bfb
    harness among them.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 8, 2, 4) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        windowed = SdpaFused()(q, k, v, is_causal=True, window=3)
        # The mask is causal by construction, so the flag adds nothing.
        torch.testing.assert_close(
            windowed,
            SdpaFused()(q, k, v, is_causal=False, window=3),
            rtol=0,
            atol=0,
        )


def test_window_text(request: pytest.FixtureRequest) -> None:
    output = _window_contract(torch.zeros(1, 4, 1, 2))
    assert_text_golden(
        request,
        test_file=__file__,
        name="window",
        rendered=repr(output.tolist()),
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_window_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="window",
        build_module=lambda: _Window().to(device),
        build_input=lambda: torch.randn(1, 4, 1, 2),
        seed=0,
    )
