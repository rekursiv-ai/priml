from __future__ import annotations

from unittest.mock import patch

import sys
import warnings

import pytest
import torch

from priml.testing.fixtures import get_device, torch_compiler_isolation


def test_get_device_returns_device():
    """Test that get_device returns a torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_get_device_prefers_cuda():
    """Test that get_device returns CUDA if available."""
    if torch.cuda.is_available():
        device = get_device()
        assert device.type == "cuda"
    else:
        device = get_device()
        assert device.type == "cpu"


@pytest.mark.cuda
def test_cleanup_cuda_with_cuda():
    """Test cleanup_cuda when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test that CUDA operations work
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    assert x.device.type == "cuda"


def test_cleanup_cuda_without_cuda():
    """Test that code works when CUDA is not available."""
    # Just verify CPU tensors work fine
    x = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    assert x.device.type == "cpu"


def test_compiler_isolation_warms_the_inductor_mkldnn_import() -> None:
    """Entering the block imports the module Inductor would otherwise re-import."""
    with torch_compiler_isolation():
        pass
    assert "torch.utils.mkldnn" in sys.modules


def test_compiler_isolation_leaves_no_warning_filter_behind() -> None:
    """The block prevents the deprecation; it does not mute it for the caller."""
    with torch_compiler_isolation():
        pass
    assert not [
        entry
        for entry in warnings.filters
        if entry[0] == "ignore" and "script_method" in getattr(entry[1], "pattern", "")
    ]


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


@pytest.mark.cuda
def test_compiler_isolation_resets_cleanly_with_warnings_as_errors() -> None:
    """reset() imports cudagraph_trees only under CUDA; that path must stay clean."""
    with torch_compiler_isolation(), warnings.catch_warnings():
        warnings.simplefilter("error")
        torch._dynamo.reset()


@pytest.mark.slow
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_compiler_isolation_compiles_with_deprecations_fatal() -> None:
    """Inductor's pre-grad pass reaches mkldnn on every host, CUDA or not."""

    def add_one(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    with torch_compiler_isolation():
        compiled = torch.compile(add_one, dynamic=False)
        assert torch.equal(compiled(torch.zeros(4)), torch.ones(4))


def test_test_main_calls_pytest():
    """Test that test_main calls pytest.main with correct arguments."""
    from priml.lib.testing.main import test_main  # noqa: PLC0415

    with (
        patch("pytest.main", return_value=0) as mock_pytest,
        patch("sys.exit") as mock_exit,
    ):
        test_main("/path/to/test_file.py")

        # Verify pytest.main was called
        assert mock_pytest.called
        call_args = mock_pytest.call_args[0][0]

        # Verify the test file is in the arguments
        assert "/path/to/test_file.py" in call_args
        assert "-v" in call_args
        assert "-s" in call_args
        assert "-W" in call_args
        assert "ignore::pytest.PytestAssertRewriteWarning" in call_args

        # Verify sys.exit was called with the return value
        mock_exit.assert_called_once_with(0)


def test_test_main_passes_through_argv():
    """Test that test_main passes through command-line arguments."""
    from priml.lib.testing.main import test_main  # noqa: PLC0415

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test_script.py", "-k", "test_foo", "--verbose"]

        with (
            patch("pytest.main", return_value=0) as mock_pytest,
            patch("sys.exit"),
        ):
            test_main("/path/to/test_file.py")

            call_args = mock_pytest.call_args[0][0]

            # Verify extra args were passed through
            assert "-k" in call_args
            assert "test_foo" in call_args
            assert "--verbose" in call_args
    finally:
        sys.argv = original_argv


def test_test_main_exits_with_pytest_return_code():
    """Test that test_main exits with the return code from pytest."""
    from priml.lib.testing.main import test_main  # noqa: PLC0415

    with patch("pytest.main", return_value=42), patch("sys.exit") as mock_exit:
        test_main("/path/to/test_file.py")
        mock_exit.assert_called_once_with(42)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
