from __future__ import annotations

import pytest
import scipy
import torch

from priml.math.frequency import dct1d, dctnd, idct1d, idctnd


@pytest.mark.parametrize(
    "axes",
    [
        [-1],
        [-3],
        [-1, -2],
        [-3, -2],
        [0, -2],
    ],
)
def test_dctnd(axes: list[int]):
    x = torch.randn(4, 5, 3, 6)
    expected = torch.as_tensor(scipy.fft.dctn(x, axes=axes), dtype=x.dtype)
    actual = dctnd(x, axis=axes)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_dctnd_roundtrip():
    x = torch.randn(4, 5, 3, 6)
    torch.testing.assert_close(
        x,
        idctnd(dctnd(x)),
        atol=1e-5,
        rtol=1e-4,
    )


def test_dctnd_normalized():
    """Test DCT with normalization (ortho mode)."""
    x = torch.randn(2, 3, 7, 5)
    torch.testing.assert_close(
        torch.as_tensor(
            scipy.fft.dctn(x, axes=[-1], norm="ortho"),
            dtype=x.dtype,
        ),
        dctnd(x, axis=[-1], normalize=True),
        atol=1e-5,
        rtol=1e-4,
    )
    # Verify inverse works with normalization
    torch.testing.assert_close(
        x,
        idctnd(dctnd(x, normalize=True), normalize=True),
        atol=1e-5,
        rtol=1e-4,
    )


def test_dct1d_preserves_leading_dims():
    """``dct1d`` must keep the input's leading dims, not the flattened shape.

    Regression for FREQ (Issue#335): ``dct1d`` rebound ``x`` to a 2-D
    ``(-1, n)`` view before computing ``y.view(*x.shape)``, collapsing
    all leading dims into one. A rank-3 input must return a rank-3 output.
    """
    x = torch.randn(2, 3, 4)
    assert dct1d(x).shape == (2, 3, 4)
    assert idct1d(x).shape == (2, 3, 4)
    # Per-row equivalence with the flattened computation (the leading-dim
    # collapse must be a pure reshape, not a reordering).
    flat = dct1d(x.reshape(-1, 4)).reshape(2, 3, 4)
    torch.testing.assert_close(dct1d(x), flat, atol=1e-5, rtol=1e-4)


def test_dctnd_invalid_axis():
    """Duplicate axes raise, and the message echoes the user's original input."""
    x = torch.randn(2, 3, 7, 5)
    with pytest.raises(ValueError, match="Duplicate axes"):
        dctnd(x, axis=[1, 1])  # Duplicate axis.
    # Aliased axes (e.g. 1 and -3 for rank 4) must also report the raw input.
    with pytest.raises(ValueError, match=r"axis=\[1, -3\]"):
        dctnd(x, axis=[1, -3])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
