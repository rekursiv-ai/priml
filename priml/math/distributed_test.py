"""Tests for the distributed log-reduction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import math

from torch import Tensor

import pytest
import torch
import torch.distributed as dist

from priml.math.distributed import (
    _logsumexp_all_to_all,
    collective_device,
    logmeanexp_all_to_all,
    logsumexp_all_to_all,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def test_logsumexp_all_to_all():
    """Test logsumexp_all_to_all without distributed setup."""
    x = torch.randn(2, 3, 4)
    result = logsumexp_all_to_all(x, dim=-1)
    expected = torch.logsumexp(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsumexp_all_to_all_empty_reduction():
    """An empty reduced result must not raise ZeroDivisionError."""
    x = torch.zeros(0, 5)
    result = logsumexp_all_to_all(x, dim=-1)
    assert result.shape == (0,)
    # ``logmeanexp`` shares the divide path; it must also survive the empty case.
    assert logmeanexp_all_to_all(x, dim=-1).shape == (0,)


def test_logsumexp_all_to_all_keepdim():
    """Test logsumexp_all_to_all with keepdim=True."""
    x = torch.randn(2, 3, 4)
    result = logsumexp_all_to_all(x, dim=-1, keepdim=True)
    expected = torch.logsumexp(x, dim=-1, keepdim=True)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsumexp_all_to_all_multiple_dims():
    """Test logsumexp_all_to_all with multiple dimensions."""
    x = torch.randn(2, 3, 4, 5)
    dim = [1, 2]
    result = logsumexp_all_to_all(x, dim=dim)
    expected = torch.logsumexp(x, dim=dim)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all():
    x = torch.randn(4, 3, 6)
    dim = [0, 2]
    result = logmeanexp_all_to_all(x, dim=dim)
    # Reference: compute mean in linear space, then take log.
    linear_mean = x.exp().mean(dim=dim)
    expected = linear_mean.log()
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_logmeanexp_all_to_all_keepdim():
    """Test logmeanexp_all_to_all with keepdim=True."""
    x = torch.randn(2, 3, 4)
    dim = 1
    result = logmeanexp_all_to_all(x, dim=dim, keepdim=True)
    expected = torch.logsumexp(x, dim=dim, keepdim=True) - math.log(x.shape[dim])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all_single_dim():
    """Test logmeanexp_all_to_all with a single dimension."""
    x = torch.randn(10, 20)
    dim = 0
    result = logmeanexp_all_to_all(x, dim=dim)
    expected = torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsumexp_all_to_all_with_world_size_rank():
    """Test logsumexp_all_to_all with explicit world_size and rank (lines 109-118)."""
    # Test non-distributed case with explicit world_size and rank
    # When torch.distributed is not initialized, should still work.
    x = torch.randn(2, 3, 4)
    result = logsumexp_all_to_all(x, dim=-1, world_size=1)
    expected = torch.logsumexp(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all_with_world_size_rank():
    """Test logmeanexp_all_to_all with explicit world_size and rank (lines 109-118)."""
    # Test non-distributed case with explicit world_size and rank.
    x = torch.randn(2, 3, 4)
    dim = -1
    result = logmeanexp_all_to_all(x, dim=dim, world_size=1)
    expected = torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsumexp_all_to_all_multiple_dims_sequence():
    """Test logsumexp_all_to_all with sequence of dimensions."""
    x = torch.randn(2, 3, 4, 5)
    dim = (1, 3)
    result = logsumexp_all_to_all(x, dim=dim)
    expected = torch.logsumexp(x, dim=dim)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all_multiple_dims_sequence():
    """Test logmeanexp_all_to_all with sequence of dimensions."""
    x = torch.randn(2, 3, 4, 5)
    dim = (0, 2)
    result = logmeanexp_all_to_all(x, dim=dim)
    expected = torch.logsumexp(x, dim=dim) - sum(math.log(x.shape[i]) for i in dim)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsumexp_all_to_all_keepdim_false():
    """Test logsumexp_all_to_all with keepdim=False."""
    x = torch.randn(3, 4, 5)
    result = logsumexp_all_to_all(x, dim=1, keepdim=False)
    expected = torch.logsumexp(x, dim=1, keepdim=False)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all_keepdim_false():
    """Test logmeanexp_all_to_all with keepdim=False."""
    x = torch.randn(3, 4, 5)
    dim = 1
    result = logmeanexp_all_to_all(x, dim=dim, keepdim=False)
    expected = torch.logsumexp(x, dim=dim, keepdim=False) - math.log(x.shape[dim])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_internal_logsumexp_defaults_to_keepdim_false():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    result = _logsumexp_all_to_all(x)
    expected = torch.logsumexp(x, dim=-1, keepdim=False)

    assert result.shape == (2, 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_logsumexp_all_to_all_distributed():
    """Test logsumexp_all_to_all with mocked distributed (all_gather path)."""
    x = torch.randn(2, 3, 4)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather") as mock_all_gather,
    ):

        def mock_all_gather_impl(
            gathered: list[Tensor],
            tensor: Tensor,
        ) -> None:
            for t in gathered:
                t.copy_(tensor)

        mock_all_gather.side_effect = mock_all_gather_impl
        result = logsumexp_all_to_all(x, dim=-1)
        expected = torch.logsumexp(x, dim=-1) + math.log(2)
        mock_all_gather.assert_called_once()
        assert torch.equal(result, expected)


def test_logmeanexp_all_to_all_distributed():
    """Test logmeanexp_all_to_all with mocked distributed (all_gather path)."""
    x = torch.randn(2, 3, 4)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather") as mock_all_gather,
    ):

        def mock_all_gather_impl(
            gathered: list[Tensor],
            tensor: Tensor,
        ) -> None:
            for t in gathered:
                t.copy_(tensor)

        mock_all_gather.side_effect = mock_all_gather_impl
        result = logmeanexp_all_to_all(x, dim=-1)
        expected = torch.logsumexp(x, dim=-1) - math.log(4)
        mock_all_gather.assert_called_once()
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)


def test_explicit_world_size_controls_gather_and_mean():
    """Explicit world size determines both gathered ranks and averaging."""
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather") as mock_all_gather,
    ):

        def mock_all_gather_impl(
            gathered: list[Tensor],
            tensor: Tensor,
        ) -> None:
            for rank, output in enumerate(gathered):
                output.copy_(tensor + rank)

        mock_all_gather.side_effect = mock_all_gather_impl
        summed = logsumexp_all_to_all(x, dim=-1, world_size=3)
        expected_sum = torch.logsumexp(
            torch.stack([torch.logsumexp(x, dim=-1) + rank for rank in range(3)]),
            dim=0,
        )
        torch.testing.assert_close(summed, expected_sum)

        averaged = logmeanexp_all_to_all(x, dim=-1, world_size=3)
        expected_mean = expected_sum - math.log(x.shape[-1] * 3)
        torch.testing.assert_close(averaged, expected_mean)


def test_collective_device_follows_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """NCCL reduces on this rank's CURRENT CUDA device; gloo on the CPU."""
    group = cast(dist.ProcessGroup, object())
    backends = {"gloo": torch.device("cpu"), "nccl": torch.device("cuda", 2)}
    for backend, expected in backends.items():
        with monkeypatch.context() as patched:
            patched.setattr(dist, "get_backend", {group: backend}.__getitem__)
            patched.setattr(torch.cuda, "is_available", lambda: True)
            patched.setattr(torch.cuda, "current_device", lambda: 2)
            assert collective_device(group) == expected


def test_collective_device_rejects_nccl_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist, "get_backend", {None: "nccl"}.__getitem__)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=r"^NCCL backend declared but no CUDA"):
        collective_device()


@pytest.mark.parametrize("reduce", [logsumexp_all_to_all, logmeanexp_all_to_all])
def test_log_reductions_reject_an_empty_dim(
    reduce: Callable[..., Tensor],
) -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        reduce(torch.zeros(2, 3), dim=())


def test_log_reductions_reduce_every_axis_for_none() -> None:
    x = torch.arange(6.0).reshape(2, 3)
    torch.testing.assert_close(
        logsumexp_all_to_all(x, dim=None),
        torch.logsumexp(x.flatten(), dim=0),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
