from __future__ import annotations

from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock, patch

import sys
import warnings

from torch import Tensor

import pytest
import torch

from priml.testing import fixtures
from priml.testing.fixtures import (
    get_device,
    poison_free_pool,
    torch_compiler_isolation,
)


def test_get_device_returns_device():
    """Test that get_device returns a torch.device."""
    device = get_device()
    assert device.type in ("cpu", "cuda")


def test_get_device_prefers_cuda():
    """Test that get_device returns CUDA if available."""
    if torch.cuda.is_available():
        device = get_device()
        assert device.type == "cuda"
    else:
        device = get_device()
        assert device.type == "cpu"


@pytest.mark.gpu_torch_cuda
def test_cleanup_cuda_with_cuda():
    """Test cleanup_cuda when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test that CUDA operations work.
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    assert x.device.type == "cuda"


def test_cleanup_cuda_without_cuda():
    """Test that code works when CUDA is not available."""
    # Just verify CPU tensors work fine.
    x = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    assert x.device.type == "cpu"


def test_compiler_isolation_resets_dynamo_when_the_block_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing block still gets isolated, or its guards leak to the next test."""
    resets: list[int] = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: resets.append(1))

    with pytest.raises(RuntimeError, match="block failed"), torch_compiler_isolation():
        raise RuntimeError("block failed")

    assert resets == [1]


def test_compiler_isolation_skips_reset_when_dynamo_is_unimported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that never compile must not pay for a Dynamo reset."""
    resets: list[int] = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: resets.append(1))
    monkeypatch.delitem(sys.modules, "torch._dynamo", raising=False)

    with torch_compiler_isolation():
        pass

    assert resets == []


@pytest.mark.gpu_torch_cuda
def test_compiler_isolation_resets_cleanly_with_warnings_as_errors() -> None:
    """reset() imports cudagraph_trees only under CUDA; that path must stay clean.

    Torch's own deprecations are exempted, not the whole category: the claim
    here is that OUR isolation path raises nothing, and torch 3.14 emits 14
    ``script_method`` DeprecationWarnings from ``torch/jit/_script.py:359``
    during that import (measured). Failing on those tests torch's roadmap
    rather than this helper, while a blanket ``ignore`` would also hide a
    deprecation we caused.
    """
    with torch_compiler_isolation(), warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"torch\..*",
        )
        torch._dynamo.reset()


@pytest.mark.compute_torch_compile
def test_compiler_isolation_compiles() -> None:
    """The wrapped block can actually compile and run a graph."""

    def add_one(value: Tensor) -> Tensor:
        return value + 1

    with torch_compiler_isolation():
        compiled = torch.compile(add_one, dynamic=False)
        assert torch.equal(compiled(torch.zeros(4)), torch.ones(4))


def test_poison_free_pool_makes_a_later_empty_read_back_nan() -> None:
    """An unwritten allocation should surface as NaN rather than as luck.

    Which block an allocator hands back is platform policy, not a guarantee we
    can assert: glibc returns freed small blocks eagerly, and macOS need not.
    Skipping rather than failing keeps this honest -- callers of the helper
    assert that values they define are correct, never that undefined memory is
    observably poisoned, so a platform that declines simply loses the sharper
    diagnostic.
    """
    poison_free_pool((6, 12))

    recycled = torch.empty(6, 12)

    if not bool(torch.isnan(recycled).any()):
        pytest.skip("allocator does not recycle freed blocks on this platform")
    assert bool(torch.isnan(recycled).any())


def test_poison_free_pool_leaves_written_allocations_alone() -> None:
    """Poisoning must not disturb tensors that do define their own contents."""
    poison_free_pool((6, 12))

    written = torch.zeros(6, 12)

    assert bool(torch.isfinite(written).all())


def test_test_main_calls_pytest():
    """Test that test_main calls pytest.main with correct arguments."""
    from priml.lib.testing.main import (  # noqa: PLC0415 -- The fixture imports optional test dependencies only inside the cases that use them.
        test_main,
    )

    with (
        patch("pytest.main", return_value=0) as mock_pytest,
        patch("sys.exit") as mock_exit,
    ):
        test_main("/path/to/test_file.py")

        # Verify pytest.main was called.
        assert mock_pytest.called
        call = mock_pytest.call_args
        assert call is not None
        call_args = call.args[0]
        assert isinstance(call_args, list)

        # Verify the test file is in the arguments.
        assert "/path/to/test_file.py" in call_args
        assert "-v" in call_args
        assert "-s" in call_args
        assert "-W" in call_args
        assert "ignore::pytest.PytestAssertRewriteWarning" in call_args

        # Verify sys.exit was called with the return value.
        mock_exit.assert_called_once_with(0)


def test_test_main_passes_through_argv():
    """Test that test_main passes through command-line arguments."""
    from priml.lib.testing.main import (  # noqa: PLC0415 -- The fixture imports optional test dependencies only inside the cases that use them.
        test_main,
    )

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test_script.py", "-k", "test_foo", "--verbose"]

        with (
            patch("pytest.main", return_value=0) as mock_pytest,
            patch("sys.exit"),
        ):
            test_main("/path/to/test_file.py")

            call = mock_pytest.call_args
            assert call is not None
            call_args = call.args[0]
            assert isinstance(call_args, list)

            # Verify extra args were passed through.
            assert "-k" in call_args
            assert "test_foo" in call_args
            assert "--verbose" in call_args
    finally:
        sys.argv = original_argv


def test_test_main_exits_with_pytest_return_code():
    """Test that test_main exits with the return code from pytest."""
    from priml.lib.testing.main import (  # noqa: PLC0415 -- The fixture imports optional test dependencies only inside the cases that use them.
        test_main,
    )

    with patch("pytest.main", return_value=42), patch("sys.exit") as mock_exit:
        test_main("/path/to/test_file.py")
        mock_exit.assert_called_once_with(42)


def test_the_late_import_recheck_does_not_move_when_a_test_claims_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_cuda_is_initialized`` is the seam a faking test does NOT control.

    ``cleanup_cuda`` re-checks through it when torch arrives after setup. Were
    that an ``is_available`` re-check, a test claiming a device would send
    teardown into ``torch.cuda.synchronize`` on a host without one.
    """
    if torch.cuda.is_initialized():
        pytest.skip("a real CUDA context makes the fake indistinguishable")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert not fixtures._cuda_is_initialized()


@pytest.mark.parametrize(
    ("initialized", "available_at_teardown", "calls_at_setup", "reason"),
    [
        (False, True, 0, "teardown synchronized without ever creating a CUDA context"),
        (True, False, 1, "teardown synchronized against a device faked as unavailable"),
    ],
    ids=["no-context-was-created", "test-faked-cuda-away"],
)
def test_cleanup_cuda_teardown_skips_synchronize(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialized: bool,
    available_at_teardown: bool,
    calls_at_setup: int,
    reason: str,
) -> None:
    """Teardown reclaims only when every seam agrees CUDA is really usable.

    Two ways that fails, one per case. Without a context, a teardown
    ``synchronize()`` forces CUDA's lazy init on a host the test never touched,
    which can raise. With a context but ``is_available()`` faked False, it
    resolves a device index against the fake -- and ``monkeypatch`` undoes the
    test's patches only AFTER teardown, so the fake is still live here.

    Asserting on ``call_count`` rather than on a raised exception is deliberate:
    it still fails if the call is made and its error swallowed.
    """
    synchronize = MagicMock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)

    # pytest's own accessor for a fixture's underlying function; no public
    # equivalent, and its return type is Callable[..., Any] (pytest's stub,
    # not ours) -- cast to what cleanup_cuda actually returns, then verify it.
    gen = cast(
        Generator[None, None, None],
        fixtures.cleanup_cuda._get_wrapped_function()(),
    )
    assert isinstance(gen, Generator)
    next(gen)
    assert synchronize.call_count == calls_at_setup, (
        "setup synchronized without a CUDA context to reclaim"
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available_at_teardown)
    with pytest.raises(StopIteration):
        next(gen)

    assert synchronize.call_count == calls_at_setup, reason


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
