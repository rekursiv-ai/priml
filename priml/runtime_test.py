"""Tests for runtime strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import os

import pytest
import torch

from priml import runtime
from priml.runtime import (
    MultiProcess,
    RuntimeProtocol,
    SingleProcess,
    get_device,
    initialize_global_device_mesh,
    is_rank_zero,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


def test_get_device_explicit() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_none_returns_the_torch_default() -> None:
    assert get_device(None) == torch.get_default_device()


def test_get_device_has_no_auto_spelling() -> None:
    with pytest.raises(RuntimeError, match="auto"):
        get_device("auto")


@pytest.mark.parametrize(
    ("accelerator", "available", "expected"),
    [
        (torch.device("cuda"), True, "cuda"),
        (torch.device("mps"), True, "mps"),
        # A CUDA wheel on a machine with no GPU, as on a CPU CI runner.
        (torch.device("cuda"), False, "cpu"),
        (None, False, "cpu"),
    ],
)
@pytest.mark.parametrize("runtime_config", [SingleProcess.Config, MultiProcess.Config])
def test_a_runtime_without_a_device_takes_the_best_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    accelerator: torch.device | None,
    available: bool,
    expected: str,
    runtime_config: type[SingleProcess.Config | MultiProcess.Config],
) -> None:
    def current_accelerator(*, check_available: bool = False) -> torch.device | None:
        """Torch's contract: the build's accelerator, or ``None`` if unusable."""
        return accelerator if available or not check_available else None

    monkeypatch.setattr(torch.accelerator, "current_accelerator", current_accelerator)
    config = runtime_config()
    assert config.device is None
    assert config.make().device == torch.device(expected)


class _StubRuntime:
    """Borrows the protocol's stub bodies, which acquire nothing."""

    device = torch.device("cpu")
    initialize = RuntimeProtocol.initialize
    destroy = RuntimeProtocol.destroy


def test_runtime_protocol_stub_bodies_are_inert() -> None:
    strategy: RuntimeProtocol = _StubRuntime()
    assert strategy.initialize() is None
    assert strategy.destroy() is None
    assert not runtime.runtime_initialized()


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


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_single_process_resolves_device(device: str) -> None:
    process = SingleProcess.Config(device=device).make()
    assert process.device == torch.device(device)


@pytest.mark.parametrize(("device", "backend"), [("cpu", "gloo"), ("cuda", "nccl")])
def test_multiprocess_backend_defaults_from_device(device: str, backend: str) -> None:
    runtime = MultiProcess.Config(device=device).make()
    assert runtime.device == torch.device(device)
    assert runtime.backend == backend


def test_multiprocess_backend_respects_explicit() -> None:
    runtime = MultiProcess.Config(device="cpu", backend="gloo").make()
    assert runtime.backend == "gloo"


def _fallback_local_rank(*, fallback_rank: int | None) -> int:
    return fallback_rank if fallback_rank is not None else 0


def test_multiprocess_initialize_passes_configured_device_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1)
    monkeypatch.setattr(
        torch.distributed,
        "get_node_local_rank",
        _fallback_local_rank,
    )
    process = MultiProcess.Config(
        device="cuda",
        backend="custom",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    ).make()

    process.initialize()

    assert "mesh_device_type" in record
    assert record["mesh_device_type"] == "cuda"
    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
    assert init_kwargs["backend"] == "custom"
    assert init_kwargs["device_id"] == torch.device("cuda", 0)
    # The runtime's device is the rank's own GPU, so children built in its
    # context land on the bound device rather than whatever ``cuda`` means.
    assert process.device == torch.device("cuda", 0)


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
        device="cpu",
        float32_matmul_precision="high",
    ).make().initialize()
    SingleProcess.Config(
        device="cpu",
        float32_matmul_precision="high",
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
        device="cpu",
        float32_matmul_precision="high",
    ).make().initialize()

    with pytest.raises(
        RuntimeError,
        match=r"^Runtime already initialized with different settings ",
    ):
        SingleProcess.Config(
            device="cpu",
            float32_matmul_precision="highest",
        ).make().initialize()


def test_single_process_initialize_enables_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime, "enable_determinism", lambda: calls.append("on"))
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    SingleProcess.Config(device="cpu", deterministic=True).make().initialize()

    assert calls == ["on"]


def test_single_process_destroy_rejects_a_live_device_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mesh means a multi-process runtime owns the process; do not clobber it."""
    monkeypatch.setattr(runtime, "_device_mesh", object())
    with pytest.raises(
        RuntimeError,
        match=r"^Device mesh initialized but single process\.$",
    ):
        SingleProcess.Config(device="cpu").make().destroy()


def test_single_process_initialize_rejects_multiprocess_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live device mesh is not a single-process runtime to piggyback on."""
    monkeypatch.setattr(runtime, "_runtime_initialized", True)
    monkeypatch.setattr(runtime, "_single_process_settings", None)

    with pytest.raises(
        RuntimeError,
        match=r"^Runtime already initialized by a multi-process strategy\.$",
    ):
        SingleProcess.Config(device="cpu").make().initialize()


@pytest.mark.usefixtures("torch_globals_restored")
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

    assert runtime._runtime_initialized is False
    assert runtime._single_process_settings is None
    # A conflicting config now initializes cleanly rather than raising.
    SingleProcess.Config(device="cpu", deterministic=True).make().initialize()


@pytest.mark.parametrize(
    "mesh_topology",
    [
        {"dp": 0, "pp": 1, "tp": 1},
        {"dp": -1, "pp": -1, "tp": 1},
        {"dp": -2, "pp": 1, "tp": 2},
    ],
)
def test_multiprocess_rejects_invalid_mesh_dimensions(
    mesh_topology: dict[str, int],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Mesh topology dimensions must be positive except for at most one "
        r"-1 \(auto\); got .+\.$",
    ) as raised:
        MultiProcess.Config(device="cpu", mesh_topology=mesh_topology).make()

    assert str(raised.value) == (
        "Mesh topology dimensions must be positive except for at most one "
        f"-1 (auto); got {mesh_topology}."
    )


def test_resolve_mesh_topology_uses_exact_integer_auto_dimension() -> None:
    topology = {"dp": -1, "pp": 1, "tp": 2}

    resolved = runtime._resolve_mesh_topology(topology, 12)

    assert resolved == {"dp": 6, "pp": 1, "tp": 2}
    assert type(resolved["dp"]) is int


def test_multiprocess_finalize_leaves_device_unresolved() -> None:
    # Finalize stays hermetic: it must not probe hardware, so a pprint golden
    # is identical on a CUDA box and a CPU CI runner. Resolution happens in
    # ``__init__``.
    config = MultiProcess.Config().finalize()
    assert config.device is None
    assert config.make().device.type in {"cpu", "cuda", "mps"}


def test_multiprocess_initialize_and_destroy_drive_the_global_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=2)
    precision_calls: list[str] = []
    determinism_calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision_calls.append)
    monkeypatch.setattr(
        runtime,
        "enable_determinism",
        lambda: determinism_calls.append("enabled"),
    )
    process = MultiProcess.Config(
        device="cpu",
        backend="gloo",
        deterministic=True,
        float32_matmul_precision="high",
        mesh_topology={"dp": -1, "pp": 1, "tp": 1},
    ).make()

    process.initialize()
    assert runtime.runtime_initialized()
    assert runtime.global_device_mesh() is not None
    assert "mesh_device_type" in record
    assert record["mesh_device_type"] == "cpu"
    assert "mesh_shape" in record
    assert record["mesh_shape"] == (2, 1, 1)
    assert "mesh_dim_names" in record
    assert record["mesh_dim_names"] == ("dp", "pp", "tp")
    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
    assert init_kwargs["backend"] == "gloo"
    assert init_kwargs["device_id"] is None
    assert init_kwargs["timeout"] is torch.distributed.default_pg_timeout
    assert precision_calls == ["high"]
    assert determinism_calls == ["enabled"]
    with pytest.raises(RuntimeError, match=r"^Runtime already initialized\.$"):
        process.initialize()

    process.destroy()
    assert not runtime.runtime_initialized()
    assert runtime.global_device_mesh() is None
    assert "destroy_calls" in record
    assert record["destroy_calls"] == 1
    assert not torch.distributed.is_initialized()


def test_destroy_without_an_initialized_runtime_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1, preinitialized=True)
    monkeypatch.setattr(runtime, "_process_group_owned", True)

    runtime.destroy_global_device_mesh()

    assert "destroy_calls" not in record
    assert torch.distributed.is_initialized()
    assert runtime._process_group_owned is False


def test_destroy_clears_all_owned_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1)
    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    runtime.destroy_global_device_mesh()

    assert "destroy_calls" in record
    assert record["destroy_calls"] == 1
    assert runtime._runtime_initialized is False
    assert runtime._device_mesh is None
    assert runtime._single_process_settings is None
    assert runtime._process_group_owned is False


def test_multiprocess_initialize_enables_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime, "enable_determinism", lambda: calls.append("on"))
    _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        deterministic=True,
    )

    assert calls == ["on"]


def test_multiprocess_initialize_preserves_default_nondeterminism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime, "enable_determinism", lambda: calls.append("on"))
    _patch_distributed(monkeypatch, world_size=1)

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    assert calls == []


@pytest.mark.parametrize(
    ("device", "backend"),
    [(torch.device("cpu"), "gloo"), (torch.device("cuda"), "nccl")],
)
def test_initialize_defaults_the_backend_from_the_device(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    backend: str,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1)
    monkeypatch.setattr(
        torch.distributed,
        "get_node_local_rank",
        _fallback_local_rank,
    )

    initialize_global_device_mesh(
        device=device,
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
    )

    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
    assert init_kwargs["backend"] == backend


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

    with pytest.raises(
        ValueError,
        match=r"^mesh_topology cannot be empty\. Specify dimensions, "
        r"e\.g\., \{'dp': -1, 'pp': 1, 'tp': 1\}$",
    ):
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={},
        )

    assert "init_kwargs" not in record
    assert runtime._process_group_owned is False


@pytest.mark.parametrize(
    "mesh_topology",
    [
        {"dp": -1, "pp": -1, "tp": 1},
        {"dp": -2, "pp": 1, "tp": 2},
    ],
)
def test_initialize_rejects_invalid_auto_dimensions_before_acquiring(
    monkeypatch: pytest.MonkeyPatch,
    mesh_topology: dict[str, int],
) -> None:
    record = _patch_distributed(monkeypatch, world_size=4)

    with pytest.raises(
        ValueError,
        match=r"^Mesh topology dimensions must be positive except for at most one "
        r"-1 \(auto\); got .+\.$",
    ) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology=mesh_topology,
        )

    assert str(raised.value) == (
        "Mesh topology dimensions must be positive except for at most one "
        f"-1 (auto); got {mesh_topology}."
    )
    assert "init_kwargs" not in record
    assert runtime._process_group_owned is False


def test_initialize_rejects_zero_dimension_before_acquiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=1)
    topology = {"dp": -1, "pp": 0, "tp": 1}

    with pytest.raises(
        ValueError,
        match=r"^Mesh topology dimensions must be positive except for at most one "
        r"-1 \(auto\); got .+\.$",
    ) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology=topology,
        )

    assert str(raised.value) == (
        "Mesh topology dimensions must be positive except for at most one "
        f"-1 (auto); got {topology}."
    )
    assert "init_kwargs" not in record
    assert runtime._process_group_owned is False


# Records ``set_device`` and ``init_process_group`` arguments for assertion.
class _DistributedRecord(TypedDict, total=False):
    set_device: torch.device
    init_kwargs: dict[str, object]
    mesh_device_type: str
    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    destroy_calls: int


def _patch_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    preinitialized: bool = False,
    destroy_failure: BaseException | None = None,
) -> _DistributedRecord:
    """Stub the distributed/CUDA surface so the cuda branch runs off-GPU."""
    record: _DistributedRecord = {}
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
        record["mesh_shape"] = mesh_shape
        record["mesh_dim_names"] = mesh_dim_names
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

    assert "destroy_calls" in record
    assert record["destroy_calls"] == 1
    assert not torch.distributed.is_initialized()
    assert runtime._runtime_initialized is False
    assert runtime._device_mesh is None
    assert runtime._single_process_settings is None
    assert runtime._process_group_owned is False


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

    assert "destroy_calls" in record
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
    assert "destroy_calls" in record
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

    with pytest.raises(
        BaseExceptionGroup,
        match=r"^Distributed runtime initialization and rollback failed$",
    ) as raised:
        initialize_global_device_mesh(
            device=torch.device("cpu"),
            backend="gloo",
            mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        )

    assert raised.value.exceptions == (primary, cleanup)
    assert "destroy_calls" in record
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
    assert "destroy_calls" in record
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
    assert "mesh_device_type" in record
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

    assert "set_device" in record
    assert record["set_device"] == torch.device("cuda", 3)
    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
    assert init_kwargs["device_id"] == torch.device("cuda", 3)
    # init_device_mesh forbids a device index; the type alone is passed.
    assert "mesh_device_type" in record
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

    assert "set_device" in record
    assert record["set_device"] == torch.device("cuda", 0)
    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
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
    assert "init_kwargs" in record
    init_kwargs = record["init_kwargs"]
    assert init_kwargs["device_id"] is None
    assert "mesh_device_type" in record
    assert record["mesh_device_type"] == "cpu"


@pytest.fixture
def torch_globals_restored() -> Iterator[None]:
    """Put back the real torch settings a test drives through the runtime."""
    before = runtime._snapshot_torch_globals()
    yield
    before.restore()


@pytest.mark.usefixtures("torch_globals_restored")
def test_single_process_destroy_restores_the_torch_globals_it_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)
    monkeypatch.setattr(runtime, "_torch_globals_before", None)
    precision = torch.get_float32_matmul_precision()
    assert not torch.are_deterministic_algorithms_enabled()

    first = SingleProcess.Config(
        device="cpu",
        deterministic=True,
        float32_matmul_precision="medium",
    ).make()
    first.initialize()
    assert torch.are_deterministic_algorithms_enabled()
    first.destroy()

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision() == precision
    SingleProcess.Config(device="cpu").make().initialize()
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.usefixtures("torch_globals_restored")
def test_multiprocess_destroy_restores_the_torch_globals_it_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed(monkeypatch, world_size=1)
    monkeypatch.setattr(runtime, "_torch_globals_before", None)
    precision = torch.get_float32_matmul_precision()

    initialize_global_device_mesh(
        device=torch.device("cpu"),
        backend="gloo",
        mesh_topology={"dp": 1, "pp": 1, "tp": 1},
        deterministic=True,
        float32_matmul_precision="medium",
    )
    assert torch.are_deterministic_algorithms_enabled()
    runtime.destroy_global_device_mesh()

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision() == precision


@pytest.mark.usefixtures("torch_globals_restored")
def test_failed_initialize_rolls_back_torch_globals_and_cuda_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _patch_distributed(monkeypatch, world_size=2)
    monkeypatch.setattr(runtime, "_torch_globals_before", None)
    monkeypatch.setenv("LOCAL_RANK", "3")
    bound: list[object] = []
    precision = torch.get_float32_matmul_precision()

    # Scoped: the suite's CUDA reclaim teardown must see the real device state.
    with monkeypatch.context() as cuda:
        cuda.setattr(torch.cuda, "is_initialized", lambda: True)
        cuda.setattr(torch.cuda, "current_device", lambda: 5)
        cuda.setattr(torch.cuda, "set_device", bound.append)
        with pytest.raises(RuntimeError, match="World"):
            initialize_global_device_mesh(
                device=torch.device("cuda"),
                backend="gloo",
                mesh_topology={"dp": 1, "pp": 1, "tp": 1},
                deterministic=True,
                float32_matmul_precision="medium",
            )

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision() == precision
    assert bound == [torch.device("cuda", 3), 5]
    assert "destroy_calls" in record
    assert record["destroy_calls"] == 1
    assert runtime._torch_globals_before is None


@pytest.mark.usefixtures("torch_globals_restored")
@pytest.mark.parametrize("lifecycle", ["single", "multi", "failed"])
@pytest.mark.parametrize("prior", [None, ":16:8"])
def test_runtime_restores_cublas_environment(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
    prior: str | None,
) -> None:
    _patch_distributed(monkeypatch, world_size=2 if lifecycle == "failed" else 1)
    monkeypatch.setattr(runtime, "_torch_globals_before", None)
    if prior is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", prior)
    if lifecycle == "single":
        process = SingleProcess.Config(device="cpu", deterministic=True).make()
        process.initialize()
        process.destroy()
    elif lifecycle == "failed":
        with pytest.raises(RuntimeError, match="World size"):
            initialize_global_device_mesh(
                device="cpu",
                mesh_topology={"dp": 1},
                deterministic=True,
            )
    else:
        initialize_global_device_mesh(
            device="cpu",
            mesh_topology={"dp": 1},
            deterministic=True,
        )
        runtime.destroy_global_device_mesh()
    restored = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    assert restored == prior


@pytest.mark.usefixtures("torch_globals_restored")
def test_single_process_failed_settings_application_restores_prior_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_runtime_initialized", False)
    monkeypatch.setattr(runtime, "_single_process_settings", None)
    monkeypatch.setattr(runtime, "_torch_globals_before", None)
    precision = torch.get_float32_matmul_precision()

    def fail_determinism() -> None:
        raise RuntimeError("Determinism setup failed")

    monkeypatch.setattr(runtime, "enable_determinism", fail_determinism)
    process = SingleProcess.Config(
        device="cpu",
        deterministic=True,
        float32_matmul_precision="medium",
    ).make()
    with pytest.raises(RuntimeError, match="Determinism setup failed"):
        process.initialize()
    assert torch.get_float32_matmul_precision() == precision
    assert not runtime.runtime_initialized()
    assert runtime._torch_globals_before is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
