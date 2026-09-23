"""Tests for rank-zero distributed build helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from priml.data.distributed_build import run_rank_zero_build


if TYPE_CHECKING:
    from torch import Tensor

    import torch


class _FakeDist:
    """Tiny torch.distributed stand-in for rank-zero build propagation tests."""

    class ReduceOp:
        MIN = object()

    def __init__(
        self,
        *,
        rank: int,
        status: int,
        message: str | None,
        backend: str = "gloo",
    ) -> None:
        self.rank = rank
        self.status = status
        self.message = message
        self.backend = backend
        self.barrier_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return self.rank

    def get_backend(self) -> str:
        return self.backend

    def all_reduce(self, tensor: Tensor, op: object) -> None:
        del op
        tensor.fill_(self.status)

    def broadcast_object_list(self, objects: list[str | None], *, src: int) -> None:
        if self.rank != src:
            objects[0] = self.message

    def barrier(self) -> None:
        self.barrier_calls += 1


def test_rank_zero_build_reraises_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dist = _FakeDist(rank=0, status=0, message=None)
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)

    with pytest.raises(ValueError, match="boom"):
        run_rank_zero_build(
            name="test build",
            build=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )

    assert fake_dist.barrier_calls == 0


def test_nonzero_rank_fails_fast_with_rank_zero_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dist = _FakeDist(rank=1, status=0, message="ValueError: boom")
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)

    with pytest.raises(RuntimeError, match="test build failed on rank 0: ValueError"):
        run_rank_zero_build(name="test build", build=lambda: None)

    assert fake_dist.barrier_calls == 0


def test_rank_zero_error_survives_broadcast_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the post-failure broadcast itself fails, rank 0 still raises its error.

    Guards the swallow where ``broadcast_object_list`` raising left ``raise
    error`` unreachable, losing the real rank-0 build error.
    """

    class _BroadcastBoomDist(_FakeDist):
        @override
        def broadcast_object_list(self, objects: list[str | None], *, src: int) -> None:
            del objects, src
            raise RuntimeError("NCCL broadcast aborted")

    fake_dist = _BroadcastBoomDist(rank=0, status=0, message="ValueError: boom")
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)

    with pytest.raises(ValueError, match="boom"):
        run_rank_zero_build(
            name="t",
            build=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )


def test_successful_rank_zero_build_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dist = _FakeDist(rank=0, status=1, message=None)
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)
    calls = 0

    def build() -> None:
        nonlocal calls
        calls += 1

    run_rank_zero_build(name="test build", build=build)

    assert calls == 1
    assert fake_dist.barrier_calls == 0


@pytest.mark.parametrize(
    ("backend", "expected_device_type"),
    [("nccl", "cuda"), ("gloo", "cpu")],
)
def test_success_flag_device_tracks_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected_device_type: str,
) -> None:
    """The all_reduce success flag must live on the backend's required device.

    NCCL rejects CPU tensors (cf. seed.py / train_loop.py): a CPU flag here
    would raise an NCCL error on the GPU cluster and defeat this helper's
    fail-fast purpose, so the NCCL branch must select a ``cuda`` device. gloo
    accepts CPU. CI has no GPU, so capture the ``device=`` the helper passes to
    ``torch.tensor`` (and stub ``torch.cuda.current_device``) rather than
    allocating a real CUDA tensor.
    """
    import torch  # noqa: PLC0415 -- import only for this device-specific test.

    fake_dist = _FakeDist(rank=0, status=1, message=None, backend=backend)
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    captured: dict[str, torch.device] = {}
    real_tensor = torch.tensor

    def _capturing_tensor(
        data: list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        captured["device"] = device
        # Allocate on CPU regardless so CI (no CUDA) can still run; the device
        # under test is the captured one, not where the fake all_reduce runs.
        return real_tensor(data, dtype=dtype, device=torch.device("cpu"))

    monkeypatch.setattr(torch, "tensor", _capturing_tensor)

    run_rank_zero_build(name="test build", build=lambda: None)

    assert captured["device"].type == expected_device_type


def test_build_runs_locally_when_no_process_group_is_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UninitializedDist(_FakeDist):
        @override
        def is_initialized(self) -> bool:
            return False

        @override
        def all_reduce(self, tensor: Tensor, op: object) -> None:
            del tensor, op
            raise AssertionError("no collective may run without a process group")

    # Rank 1 with a failed status: had the guard not short-circuited, this
    # rank would skip the build and hit the raising all_reduce.
    fake_dist = _UninitializedDist(rank=1, status=0, message="ValueError: boom")
    monkeypatch.setattr("priml.data.distributed_build.dist", fake_dist)
    calls = 0

    def build() -> None:
        nonlocal calls
        calls += 1

    run_rank_zero_build(name="test build", build=build)

    assert calls == 1


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
