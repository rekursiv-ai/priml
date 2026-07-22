"""Tests for runtime strategies."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import os
import subprocess
import sys

import pytest
import torch

from priml import runtime
from priml.runtime import (
    MultiProcess,
    SingleProcess,
    get_device,
    initialize_global_device_mesh,
    is_rank_zero,
    runtime_output_path,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_get_device_explicit() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_runtime_all_exports_public_helpers() -> None:
    assert {"is_rank_zero", "runtime_output_path"} <= set(runtime.__all__)


def test_is_rank_zero_true_when_not_distributed() -> None:
    # No process group is initialized in the unit-test process; the helper must
    # treat a non-distributed process as rank 0 (the once-per-job side-effect
    # holder) rather than raising on get_rank().
    assert not torch.distributed.is_initialized()
    assert is_rank_zero() is True


def test_is_rank_zero_reflects_get_rank_when_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    assert is_rank_zero() is True
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    assert is_rank_zero() is False


def test_single_process_resolves_device() -> None:
    runtime = SingleProcess.Config(device="cpu").make()
    assert runtime.device == torch.device("cpu")


def test_multiprocess_backend_defaults_to_gloo_on_cpu() -> None:
    """T-055: backend resolution must yield gloo for a CPU device."""
    runtime = MultiProcess.Config(device="cpu").make()
    assert runtime.backend == "gloo"


def test_multiprocess_backend_respects_explicit() -> None:
    runtime = MultiProcess.Config(device="cpu", backend="gloo").make()
    assert runtime.backend == "gloo"


def test_single_process_initialize_sets_float32_matmul_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SingleProcess applies the matmul precision override during initialize()."""
    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)

    process = SingleProcess.Config(
        device="cpu",
        float32_matmul_precision="high",
    ).make()
    process.initialize()

    assert calls == ["high"]
    process.destroy()


def test_multiprocess_backend_resolved_once_by_init() -> None:
    """T-055: __init__ is the single source of backend resolution.

    ``finalize`` must not also compute the backend (duplicated logic that
    silently drifts). The finalized config keeps ``backend`` unset; ``__init__``
    resolves it.
    """
    config = MultiProcess.Config(device="cpu").finalize()
    assert config.backend is None, "finalize must not pre-compute backend"
    assert MultiProcess(config).backend == "gloo"


def test_userdirs_import_and_resolve_do_not_load_torch(tmp_path: Path) -> None:
    """Pure path resolution must stay independent of Torch.

    ``userdirs`` is imported by torch-free processes; a module-top torch import
    would add multi-second startup cost to every one of them.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "assert 'torch' not in sys.modules; "
                "from priml.lib.userdirs import resolve_working_dir; "
                "resolve_working_dir('/scratch', '/datasets/probe'); "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "LOOP_SCRATCH_DIR": str(tmp_path / "scratch")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_runtime_output_path_allows_absolute_path_outside_checkout(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "scratch" / "artifacts" / "profile.json"

    assert runtime_output_path(output_path) == output_path


def test_runtime_output_path_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="runtime output path must not be root"):
        runtime_output_path(Path("/"))


def test_runtime_output_path_rejects_explicit_symlink_into_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(checkout, target_is_directory=True)

    with pytest.raises(ValueError, match="outside Git checkout"):
        runtime_output_path(output_link / "artifacts")


def test_runtime_output_path_rejects_non_normalized_scratch_escape() -> None:
    with pytest.raises(ValueError, match="runtime output path must be normalized"):
        runtime_output_path("/scratch/../repo/output.json")


def test_runtime_output_path_trusts_logical_scratch_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    logical_path = Path("/scratch/runs/job/artifacts")
    resolve = Path.resolve

    def resolve_scratch(path: Path, *, strict: bool = False) -> Path:
        if path == logical_path:
            return checkout / "artifacts"
        return resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_scratch)

    assert runtime_output_path(logical_path) == logical_path


def _patch_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
) -> dict[str, object]:
    """Stub the distributed/CUDA surface so the cuda branch runs off-GPU.

    Records ``set_device`` and ``init_process_group`` arguments for assertion.
    """
    record: dict[str, object] = {}

    def fake_set_device(device: torch.device) -> None:
        record["set_device"] = device

    def fake_init_process_group(**kwargs: object) -> None:
        record["init_kwargs"] = kwargs

    def fake_init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        record["mesh_device_type"] = device_type
        del mesh_shape, mesh_dim_names
        return object()

    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        fake_init_process_group,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(runtime, "init_device_mesh", fake_init_device_mesh)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_device_mesh", None)
    return record


def test_multiprocess_initialize_sets_float32_matmul_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MultiProcess applies the matmul precision override before CUDA binding."""
    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)
    record = _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        float32_matmul_precision="high",
    )

    assert calls == ["high"]
    assert record["mesh_device_type"] == "cpu"


def test_cuda_branch_binds_local_rank_before_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cuda branch binds the LOCAL_RANK GPU and passes it as device_id.

    Without binding, all torchrun ranks default to cuda:0 and NCCL raises
    "Duplicate GPU detected" (Issue#368).
    """
    monkeypatch.setenv("LOCAL_RANK", "3")
    record = _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cuda"),
        backend="nccl",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    assert record["set_device"] == torch.device("cuda", 3)
    init_kwargs = cast("dict[str, object]", record["init_kwargs"])
    assert init_kwargs["device_id"] == torch.device("cuda", 3)
    # init_device_mesh forbids a device index; the type alone is passed.
    assert record["mesh_device_type"] == "cuda"


def test_cuda_branch_falls_back_when_local_rank_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset LOCAL_RANK binds device 0 (single-process / non-torchrun)."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    record = _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cuda"),
        backend="nccl",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    assert record["set_device"] == torch.device("cuda", 0)
    init_kwargs = cast("dict[str, object]", record["init_kwargs"])
    assert init_kwargs["device_id"] == torch.device("cuda", 0)


def test_cpu_branch_does_not_bind_or_pass_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gloo/CPU path never calls set_device and passes device_id=None."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    record = _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    assert "set_device" not in record
    init_kwargs = cast("dict[str, object]", record["init_kwargs"])
    assert init_kwargs["device_id"] is None
    assert record["mesh_device_type"] == "cpu"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
