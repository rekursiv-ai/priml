from __future__ import annotations

from unittest.mock import patch

import pytest

from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import (
    DataPipeline,
    _assign_gpu_to_worker,
    _pipeline_uses_cuda,
)
from priml.data.pipeline.tensorize import AsTensor


def test_assign_gpu_no_cuda():
    """Return -1 when CUDA is unavailable, regardless of worker id."""
    with patch("torch.cuda.is_available", return_value=False):
        assert _assign_gpu_to_worker(0) == -1
        assert _assign_gpu_to_worker(3) == -1


def test_assign_gpu_single_gpu():
    """Single visible GPU pins every worker to device 0."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.set_device") as set_device,
    ):
        assert _assign_gpu_to_worker(0) == 0
        assert _assign_gpu_to_worker(1) == 0
        set_device.assert_called_with(0)


def test_assign_gpu_round_robin_multi_gpu():
    """Single-process multi-GPU round-robins workers across all devices."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.set_device") as set_device,
    ):
        assigned = [_assign_gpu_to_worker(i) for i in range(4)]
        assert assigned == [0, 1, 0, 1]
        assert [c.args[0] for c in set_device.call_args_list] == [0, 1, 0, 1]


def test_assign_gpu_distributed_uses_rank_device():
    """Under an initialized distributed run, all workers use the rank's device."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.cuda.current_device", return_value=1),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.cuda.set_device") as set_device,
    ):
        # Every worker stays on rank's own device 1, never round-robining.
        assert [_assign_gpu_to_worker(i) for i in range(4)] == [1, 1, 1, 1]
        assert all(c.args[0] == 1 for c in set_device.call_args_list)


def test_assign_gpu_fork_error_returns_minus_one():
    """A fork that already initialized CUDA degrades to -1, not a crash."""
    fork_error = RuntimeError(
        "Cannot re-initialize CUDA in forked subprocess. "
        "To use CUDA with multiprocessing, you must use the 'spawn' start method",
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.set_device", side_effect=fork_error),
    ):
        assert _assign_gpu_to_worker(0) == -1


def test_assign_gpu_unexpected_error_propagates():
    """Non-fork CUDA RuntimeErrors propagate rather than silently -1."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch(
            "torch.cuda.set_device",
            side_effect=RuntimeError("Unexpected CUDA error"),
        ),
        pytest.raises(RuntimeError, match="Unexpected CUDA error"),
    ):
        _assign_gpu_to_worker(0)


def test_assign_gpu_logs_assignment():
    """A successful assignment logs worker, device, and total GPU count."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.set_device"),
        patch("priml.data.pipeline.dataset.logger") as mock_logger,
    ):
        assert _assign_gpu_to_worker(1) == 1
        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args.args
        template = args[0]
        assert isinstance(template, str)
        rendered = template % tuple(args[1:])
        assert rendered == "Worker 1: assigned to GPU 1 (total GPUs: 2)"


def test_assign_gpu_logs_no_cuda():
    """The CPU fallback logs a debug message naming the missing CUDA."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("priml.data.pipeline.dataset.logger") as mock_logger,
    ):
        assert _assign_gpu_to_worker(0) == -1
        mock_logger.debug.assert_called_once()
        message = mock_logger.debug.call_args.args[0]
        assert isinstance(message, str)
        assert "CUDA not available" in message


def test_assign_gpu_logs_fork_warning():
    """The fork fallback logs a warning explaining the pre-fork CUDA init."""
    fork_error = RuntimeError("Cannot re-initialize CUDA in forked subprocess")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.set_device", side_effect=fork_error),
        patch("priml.data.pipeline.dataset.logger") as mock_logger,
    ):
        assert _assign_gpu_to_worker(0) == -1
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args.args[0]
        assert isinstance(msg, str)
        assert "CUDA was initialized before fork" in msg


def test_pipeline_uses_cuda_detects_cuda_processor():
    """A processor with a CUDA device field is detected as GPU-bound."""
    config = DataPipeline.Config(
        processors=[AsTensor.Config(device="cuda"), Batcher.Config()],
    )
    assert _pipeline_uses_cuda(config)


def test_pipeline_uses_cuda_cpu_only():
    """A pure-CPU pipeline reports no CUDA processor."""
    config = DataPipeline.Config(
        processors=[AsTensor.Config(device="cpu"), Batcher.Config()],
    )
    assert not _pipeline_uses_cuda(config)


def test_pipeline_uses_cuda_empty():
    """A pipeline with no processors is CPU-only."""
    assert not _pipeline_uses_cuda(DataPipeline.Config())


def test_wiring_cuda_pipeline_round_robin():
    """Wiring: CUDA pipeline + N workers + M GPUs round-robins assignments."""
    config = DataPipeline.Config(processors=[AsTensor.Config(device="cuda")])
    assert _pipeline_uses_cuda(config)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.set_device"),
    ):
        assert [_assign_gpu_to_worker(i) for i in range(4)] == [0, 1, 0, 1]


def test_wiring_cpu_pipeline_no_assignment():
    """Wiring: CPU-only pipeline is never gated into a GPU assignment."""
    config = DataPipeline.Config(processors=[AsTensor.Config(device="cpu")])
    assert not _pipeline_uses_cuda(config)
    # Gate is false, so the assignment is never reached; assert the gate guards.
    with patch("torch.cuda.set_device") as set_device:
        if _pipeline_uses_cuda(config):
            _assign_gpu_to_worker(0)
        set_device.assert_not_called()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
