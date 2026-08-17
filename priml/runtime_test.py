"""Tests for runtime strategies."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from priml import runtime
from priml.runtime import (
    MultiProcess,
    SingleProcess,
    get_device,
    initialize_global_device_mesh,
    is_rank_zero,
)


def test_get_device_explicit() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_runtime_all_exports_public_helpers() -> None:
    assert {"is_rank_zero", "get_device"} <= set(runtime.__all__)


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


def test_single_process_initialize_is_idempotent_for_same_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One process legitimately builds several single-process runtimes.

    An eval sweep constructs a fresh trainer per round; raising on the second
    initialize would kill every construction after the first.
    """
    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    SingleProcess.Config(
        device="cpu", float32_matmul_precision="high"
    ).make().initialize()
    SingleProcess.Config(
        device="cpu", float32_matmul_precision="high"
    ).make().initialize()

    # The second initialize is a no-op, not a re-application.
    assert calls == ["high"]


def test_single_process_initialize_rejects_conflicting_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings are process-global, so a differing second config would not take."""
    precisions: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precisions.append)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    SingleProcess.Config(
        device="cpu", float32_matmul_precision="high"
    ).make().initialize()

    with pytest.raises(RuntimeError, match="different settings"):
        SingleProcess.Config(
            device="cpu",
            float32_matmul_precision="highest",
        ).make().initialize()


def test_single_process_initialize_rejects_multiprocess_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live device mesh is not a single-process runtime to piggyback on."""
    monkeypatch.setattr(runtime, "_runtime_initialized", True)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    with pytest.raises(RuntimeError, match="multi-process strategy"):
        SingleProcess.Config(device="cpu").make().initialize()


def test_single_process_destroy_clears_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After destroy, a differing config may initialize without conflict."""
    precisions: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precisions.append)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    process = SingleProcess.Config(device="cpu", deterministic=False).make()
    process.initialize()
    process.destroy()

    assert not runtime.runtime_initialized()
    # A conflicting config now initializes cleanly rather than raising.
    SingleProcess.Config(device="cpu", deterministic=True).make().initialize()


def test_multiprocess_backend_resolved_once_by_init() -> None:
    """T-055: __init__ is the single source of backend resolution.

    ``finalize`` must not also compute the backend (duplicated logic that
    silently drifts). The finalized config keeps ``backend`` unset; ``__init__``
    resolves it.
    """
    config = MultiProcess.Config(device="cpu").finalize()
    assert config.backend is None, "finalize must not pre-compute backend"
    assert MultiProcess(config).backend == "gloo"


def test_initialize_validates_topology_before_acquiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad topology must not leave an initialized process group behind.

    Validation ran after init_process_group, so a config error acquired a
    distributed resource and then raised, with _runtime_initialized still
    False -- nothing owns the cleanup.
    """
    record = _patch_distributed(monkeypatch, world_size=1)

    with pytest.raises(ValueError, match="mesh_topology cannot be empty"):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={},
        )

    assert "init_kwargs" not in record


def test_initialize_validates_negative_dims_before_acquiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two auto dimensions are unresolvable; reject before acquiring."""
    record = _patch_distributed(monkeypatch, world_size=4)

    with pytest.raises(ValueError, match="At most one mesh dimension"):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": -1, "pp": -1, "tp": 1},
        )

    assert "init_kwargs" not in record


def _patch_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    preinitialized: bool = False,
    destroy_failure: BaseException | None = None,
) -> dict[str, object]:
    """Stub the distributed/CUDA surface so the cuda branch runs off-GPU.

    Records ``set_device`` and ``init_process_group`` arguments for assertion.
    """
    record: dict[str, object] = {}
    initialized = preinitialized
    destroy_calls = 0

    def fake_set_device(device: torch.device) -> None:
        record["set_device"] = device

    def fake_init_process_group(**kwargs: object) -> None:
        nonlocal initialized
        record["init_kwargs"] = kwargs
        initialized = True

    def fake_destroy_process_group() -> None:
        nonlocal destroy_calls, initialized
        destroy_calls += 1
        record["destroy_calls"] = destroy_calls
        if destroy_failure is not None:
            raise destroy_failure
        initialized = False

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
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        fake_init_process_group,
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        fake_destroy_process_group,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(runtime, "init_device_mesh", fake_init_device_mesh)
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_device_mesh", None)
    monkeypatch.setattr(runtime, "_process_group_owned", False)
    return record


def test_world_size_failure_rolls_back_acquired_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=2)

    with pytest.raises(RuntimeError, match="World size 2"):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert record["destroy_calls"] == 1
    assert not torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


def test_mesh_construction_failure_rolls_back_acquired_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1)

    def fail_mesh_construction(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        del device_type, mesh_shape, mesh_dim_names
        raise RuntimeError("mesh construction failed")

    monkeypatch.setattr(runtime, "init_device_mesh", fail_mesh_construction)

    with pytest.raises(RuntimeError, match="mesh construction failed"):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert record["destroy_calls"] == 1
    assert not torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_mesh_baseexception_rolls_back_acquired_process_group(
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("mesh construction interrupted")
    record = _patch_distributed(monkeypatch, world_size=1)

    def fail_mesh_construction(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        del device_type, mesh_shape, mesh_dim_names
        raise primary

    monkeypatch.setattr(runtime, "init_device_mesh", fail_mesh_construction)

    with pytest.raises(exception_type) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert raised.value is primary
    assert record["destroy_calls"] == 1
    assert not torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_mesh_baseexception_preserves_borrowed_process_group(
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("mesh construction interrupted")
    record = _patch_distributed(
        monkeypatch,
        world_size=1,
        preinitialized=True,
    )

    def fail_mesh_construction(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        del device_type, mesh_shape, mesh_dim_names
        raise primary

    monkeypatch.setattr(runtime, "init_device_mesh", fail_mesh_construction)

    with pytest.raises(exception_type) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert raised.value is primary
    assert "init_kwargs" not in record
    assert "destroy_calls" not in record
    assert torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_mesh_baseexception_and_rollback_failure_preserve_identity_order(
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("mesh construction interrupted")
    cleanup = RuntimeError("process group cleanup failed")
    record = _patch_distributed(
        monkeypatch,
        world_size=1,
        destroy_failure=cleanup,
    )

    def fail_mesh_construction(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        del device_type, mesh_shape, mesh_dim_names
        raise primary

    monkeypatch.setattr(runtime, "init_device_mesh", fail_mesh_construction)

    with pytest.raises(BaseExceptionGroup) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert raised.value.exceptions == (primary, cleanup)
    assert record["destroy_calls"] == 1
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


def test_failure_does_not_destroy_preexisting_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(
        monkeypatch,
        world_size=2,
        preinitialized=True,
    )

    with pytest.raises(RuntimeError, match="World size 2"):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert "init_kwargs" not in record
    assert "destroy_calls" not in record
    assert torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


def test_destroy_preserves_process_group_borrowed_during_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(
        monkeypatch,
        world_size=1,
        preinitialized=True,
    )

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )
    runtime.destroy_global_device_mesh()

    assert "init_kwargs" not in record
    assert "destroy_calls" not in record
    assert torch.distributed.is_initialized()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


def test_rollback_failure_is_preserved_with_world_size_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(
        monkeypatch,
        world_size=2,
        destroy_failure=RuntimeError("process group cleanup failed"),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert {str(error) for error in raised.value.exceptions} == {
        (
            "World size 2 does not match mesh topology "
            "{'dp': 1, 'pp': 1, 'tp': 1} (expected 1)"
        ),
        "process group cleanup failed",
    }
    assert record["destroy_calls"] == 1
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None


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
