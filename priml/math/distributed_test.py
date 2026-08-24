"""Tests for the distributed log-reduction helpers."""

from __future__ import annotations

from unittest.mock import patch

import math

from torch import Tensor

import torch

from priml.math.distributed import (
    logmeanexp_all_to_all,
    logsumexp_all_to_all,
)


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
    # logmeanexp shares the divide path; it must also survive the empty case.
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
    # When torch.distributed is not initialized, should still work
    x = torch.randn(2, 3, 4)
    result = logsumexp_all_to_all(x, dim=-1, world_size=1)
    expected = torch.logsumexp(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logmeanexp_all_to_all_with_world_size_rank():
    """Test logmeanexp_all_to_all with explicit world_size and rank (lines 109-118)."""
    # Test non-distributed case with explicit world_size and rank
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
        assert mock_all_gather.called
        assert result.shape == (2, 3)


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
        assert mock_all_gather.called
        assert result.shape == (2, 3)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
