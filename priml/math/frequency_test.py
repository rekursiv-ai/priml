from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import scipy
import torch

from priml.math.frequency import dct1d, dctnd, idct1d, idctnd


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.math.custom_types import TensorableFn


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


def test_dct1d_default_matches_scipy() -> None:
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0], [0.5, 2.0, -1.0, 3.0]])
    expected_dct = torch.as_tensor(scipy.fft.dct(x, axis=-1), dtype=x.dtype)
    expected_idct = torch.as_tensor(scipy.fft.idct(x, axis=-1), dtype=x.dtype)
    torch.testing.assert_close(dct1d(x), expected_dct, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(idct1d(x), expected_idct, atol=1e-5, rtol=1e-5)


def test_dctnd_accepts_a_single_integer_axis() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expected = torch.as_tensor(scipy.fft.dctn(x, axes=[1]), dtype=x.dtype)
    torch.testing.assert_close(dctnd(x, axis=1), expected, atol=1e-5, rtol=1e-5)


def test_dct1d_arange_preserves_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_arange = torch.arange
    calls: list[dict[str, object]] = []

    def recording_arange(
        end: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        calls.append({"dtype": dtype, "device": device})
        return real_arange(end, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "arange", recording_arange)
    x = torch.ones((2, 3), dtype=torch.float64)
    dct1d(x)
    assert calls == [{"dtype": x.dtype, "device": x.device}]


def test_idctnd_uses_requested_axes() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    axes = [0, -2]
    expected = torch.as_tensor(scipy.fft.idctn(x, axes=axes), dtype=x.dtype)
    torch.testing.assert_close(idctnd(x, axis=axes), expected, atol=1e-5, rtol=1e-5)


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
    # Verify inverse works with normalization.
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
    flat_inverse = idct1d(x.reshape(-1, 4)).reshape(2, 3, 4)
    torch.testing.assert_close(idct1d(x), flat_inverse, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("function", [dct1d, idct1d])
def test_dct1d_rejects_integer_dtype(function: TensorableFn) -> None:
    x = torch.tensor([[1, 2, 3], [3, 1, -2]], dtype=torch.int64)
    with pytest.raises(TypeError, match="floating"):
        function(x)


def test_idct1d_arange_preserves_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_arange = torch.arange
    calls: list[dict[str, object]] = []

    def recording_arange(
        end: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        calls.append({"dtype": dtype, "device": device})
        return real_arange(end, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "arange", recording_arange)
    x = torch.ones((2, 3), dtype=torch.float64)
    idct1d(x)
    assert calls == [{"dtype": x.dtype, "device": x.device}]


def test_dctnd_invalid_axis():
    """Duplicate axes raise, and the message echoes the user's original input."""
    x = torch.randn(2, 3, 7, 5)
    with pytest.raises(ValueError, match="Duplicate axes"):
        dctnd(x, axis=[1, 1])  # Duplicate axis.
    # Aliased axes (e.g. 1 and -3 for rank 4) must also report the raw input.
    with pytest.raises(ValueError, match=r"axis=\[1, -3\]"):
        dctnd(x, axis=[1, -3])


@pytest.mark.parametrize("axis", [3, -4])
@pytest.mark.parametrize("function", [dctnd, idctnd])
def test_dctnd_rejects_out_of_rank_axis(
    axis: int,
    function: Callable[..., Tensor],
) -> None:
    with pytest.raises(IndexError, match="axis"):
        function(torch.ones(2, 3), axis=axis)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
