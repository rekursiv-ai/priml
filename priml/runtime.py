"""Process-global training runtime (device mesh, seeding, distributed init)."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal, Protocol, override

import math

from configgle import Fig


if TYPE_CHECKING:
    from typing import Self

    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    import torch

    from priml.math.seed import enable_determinism
else:
    from wrapt import lazy_import

    # Torch costs seconds to import; deferring it keeps this module importable
    # by consumers that only touch its non-torch surface.
    torch = lazy_import("torch")
    init_device_mesh = lazy_import("torch.distributed.device_mesh", "init_device_mesh")
    enable_determinism = lazy_import("priml.math.seed", "enable_determinism")


Float32MatmulPrecision = Literal["highest", "high", "medium"]


class RuntimeProtocol(Protocol):
    """Protocol for runtime strategies (process-global initialization).

    Handles process-global setup: distributed backend init, device mesh creation.
    For model-specific parallelism, use ParallelStrategyProtocol.
    """

    device: torch.device

    def initialize(self) -> None:
        """Acquire distributed backend and hardware resources."""
        ...

    def destroy(self) -> None:
        """Release distributed backend and hardware resources."""
        ...


__all__ = [
    "MultiProcess",
    "RuntimeProtocol",
    "SingleProcess",
    "destroy_global_device_mesh",
    "get_device",
    "global_device_mesh",
    "initialize_global_device_mesh",
    "is_rank_zero",
    "runtime_initialized",
]


def get_device(device: torch.device | str | None = "auto") -> torch.device:
    """Resolve a device specification to a ``torch.device``.

    Args:
      device: One of:

        - ``"auto"`` (default): probe hardware and return the best
          available backend: CUDA > MPS > CPU.
        - ``None``: return ``torch.get_default_device()``, i.e.
          whatever was set via ``torch.set_default_device()``.
        - A device string (``"cuda"``, ``"cuda:1"``, ``"mps"``,
          ``"cpu"``, …) or an existing ``torch.device``: passed
          through unchanged.

    Returns:
      device: Resolved ``torch.device``.

    Examples::

        get_device()           # "auto" → cuda on GPU box, mps on Mac
        get_device("cuda:1")   # explicit GPU
        get_device(None)       # torch global default

    """
    if device is None:
        return torch.get_default_device()
    if isinstance(device, str) and device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


_device_mesh: DeviceMesh | None = None
_runtime_initialized: bool = (
    False  # config-globals: ignore -- mutable process-init state flag, not a knob
)
_single_process_settings: tuple[bool, Float32MatmulPrecision | None] | None = (
    None  # config-globals: ignore -- mutable process-init state, not a knob
)


class SingleProcess:
    """Single-process runtime strategy (non-distributed).

    Use this for:
    - Single GPU training
    - CPU training

    Launch:
      python train.py  # Single Python process

    Example:
      SingleProcess.Config(device="cuda")  # Uses default GPU
      SingleProcess.Config(device="cuda:0")  # Uses GPU 0
      SingleProcess.Config(device="cpu")  # CPU training

    Note:
      No device mesh is created, so distributed parallel strategies
      (DataParallel, FullySharded, HybridSharded, RecursiveSharded) raise at
      construction -- they require ``MultiProcess``. Use ``NoParallel`` here.

    """

    class Config(Fig["SingleProcess"]):
        device: torch.device | str | None = "auto"
        """Target device ("auto" = best available, None = torch default)."""
        deterministic: bool = False
        """Enable deterministic CUDA operations."""
        float32_matmul_precision: Float32MatmulPrecision | None = None
        """Float32 matmul precision override; None preserves PyTorch's default."""

    def __init__(self, config: Config):
        self.device = get_device(config.device)
        self.deterministic = config.deterministic
        self.float32_matmul_precision: Float32MatmulPrecision | None = (
            config.float32_matmul_precision
        )

    def initialize(self) -> None:
        """Initialize execution resources (set deterministic mode if configured).

        Idempotent for an equivalent config. A single-process runtime owns no
        process group or device mesh, so re-initializing is a no-op -- and one
        process legitimately builds several runtimes (an eval sweep constructs
        a fresh trainer per round). Raising there would kill every construction
        after the first. A *different* config still raises: settings apply
        process-globally, so the second one would silently not take effect.

        Raises:
          RuntimeError: A multi-process runtime is already initialized, or a
            single-process runtime was initialized with different settings.

        """
        global _runtime_initialized, _single_process_settings  # noqa: PLW0603
        settings = (self.deterministic, self.float32_matmul_precision)
        if _runtime_initialized:
            if _single_process_settings is None:
                raise RuntimeError(
                    "Runtime already initialized by a multi-process strategy.",
                )
            if _single_process_settings != settings:
                raise RuntimeError(
                    "Runtime already initialized with different settings "
                    f"{_single_process_settings}; cannot re-initialize as "
                    f"{settings}. These apply process-globally.",
                )
            return
        _set_float32_matmul_precision(self.float32_matmul_precision)
        if self.deterministic:
            enable_determinism()
        _single_process_settings = settings
        _runtime_initialized = True

    def destroy(self) -> None:
        """Cleanup runtime resources."""
        global _runtime_initialized, _single_process_settings  # noqa: PLW0603
        if global_device_mesh() is not None:
            raise RuntimeError("Device mesh initialized but single process.")
        _single_process_settings = None
        _runtime_initialized = False


class MultiProcess:
    """Multi-process distributed training with device mesh.

    Use this for:
    - Multi-GPU on one host (DDP, TP, PP, or combinations)
    - Multi-host training (distributed across machines)
    - Any combination of Data/Tensor/Pipeline parallelism

    Launch:
      # Single host with 4 GPUs
      torchrun --nproc_per_node=4 train.py

      # Multi-host (2 nodes, 8 GPUs each)
      torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d train.py

    Parallelism strategies:
    - TP (Tensor Parallel): Split model width-wise
    - PP (Pipeline Parallel): Split model depth-wise (not yet implemented)
    - DP (Data Parallel): Minibatch comprised of microbatches (on each replica)

    Mesh ordering (assumes consecutive ranks have fastest interconnect):
    - TP (frequent communication) → fastest interconnect
    - PP (moderate communication) → medium interconnect
    - DP (minimal communication) → slowest interconnect

    Examples:
      # Pure DDP (8 GPUs, all data parallel)
      MultiProcess.Config(mesh={"dp": 8, "pp": 1, "tp": 1})

      # Pure TP (4 GPUs on one host, one data replica)
      MultiProcess.Config(mesh={"dp": 1, "pp": 1, "tp": 4})

      # DP + TP (8 GPUs = 2 DP replicas x 4 TP per replica)
      MultiProcess.Config(mesh={"dp": 2, "pp": 1, "tp": 4})

      # DP + PP + TP (16 GPUs = 2 DP replicas x 2 PP stages x 4 TP per stage)
      MultiProcess.Config(mesh={"dp": 2, "pp": 2, "tp": 4})

      # Auto-infer DP size
      MultiProcess.Config(mesh={"dp": -1, "pp": 1, "tp": 4})

    Note:
      All three dimensions (dp, pp, tp) are mandatory. Use 1 for unused dimensions.

    """

    class Config(Fig["MultiProcess"]):
        device: torch.device | str | None = "auto"
        """Target device ("auto" = best available, None = torch default)."""
        deterministic: bool = False
        """Enable deterministic CUDA operations."""
        float32_matmul_precision: Float32MatmulPrecision | None = None
        """Float32 matmul precision override; None preserves PyTorch's default."""
        backend: str | None = None
        """Distributed backend (None = auto: nccl for CUDA, gloo for CPU)."""
        mesh_topology: dict[str, int] = field(
            default_factory=lambda: {"dp": -1, "pp": 1, "tp": 1},
        )
        """Device mesh dimensions (-1 = auto-infer from world size)."""

        @override
        def finalize(self) -> Self:
            self.device = get_device(self.device)
            # Backend is resolved once, in ``MultiProcess.__init__`` -- do not
            # duplicate that logic here (the two copies silently drift).
            return super().finalize()

    def __init__(self, config: Config):
        self.device = get_device(config.device)
        if config.backend is None:
            self.backend = "nccl" if self.device.type.startswith("cuda") else "gloo"
        else:
            self.backend = config.backend
        self.deterministic = config.deterministic
        self.float32_matmul_precision: Float32MatmulPrecision | None = (
            config.float32_matmul_precision
        )
        if (
            any(s == 0 for s in config.mesh_topology.values())
            or sum(s < 0 for s in config.mesh_topology.values()) > 1
        ):
            raise ValueError(
                f"Mesh shapes {config.mesh_topology} must be positive "
                "except for at most one negative.",
            )
        self.mesh_topology = dict(config.mesh_topology)

    def initialize(self) -> None:
        """Initialize distributed backend and device mesh."""
        initialize_global_device_mesh(
            backend=self.backend,
            device=self.device,
            mesh_topology=self.mesh_topology,
            deterministic=self.deterministic,
            float32_matmul_precision=self.float32_matmul_precision,
        )

    def destroy(self) -> None:
        """Cleanup distributed backend."""
        destroy_global_device_mesh()


def global_device_mesh() -> DeviceMesh | None:
    """Get device mesh set by TrainLoop or manual setup.

    Returns None if not set (single process mode).
    """
    return _device_mesh


def runtime_initialized() -> bool:
    """Return whether a process-global runtime is initialized.

    Returns:
      initialized: True if distributed runtime resources are currently
        initialized.

    """
    return _runtime_initialized


def is_rank_zero() -> bool:
    """Return whether this process is the global rank-0 (or non-distributed).

    The single predicate behind rank-0-only side effects: high-frequency
    narrative logging, retention purges, and similar work that must run once
    per job rather than once per rank. Non-distributed processes are rank 0 by
    definition, so the helper is safe to call before any process group exists.

    Returns:
      rank_zero: True when distributed is uninitialized or the global rank is 0.

    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def initialize_global_device_mesh(
    *,
    device: torch.device | str | None = None,
    backend: str | None = None,
    mesh_topology: dict[str, int] | None = None,
    deterministic: bool = False,
    float32_matmul_precision: Float32MatmulPrecision | None = None,
) -> DeviceMesh:
    """Initialize global device mesh and distributed runtime (process group + device mesh).

    Args:
        device: Device type for mesh ('cuda', 'cpu')
        backend: Distributed backend ('nccl', 'gloo')
        mesh_topology: Mesh dimensions e.g. {"dp": 2, "tp": 4}.
        deterministic: Enable deterministic CUDA ops.
        float32_matmul_precision: Float32 matmul precision override.

    Returns:
        DeviceMesh.

    """
    global _runtime_initialized  # noqa: PLW0603

    if _runtime_initialized:
        raise RuntimeError("Runtime already initialized.")

    # Validate the topology BEFORE acquiring anything. These checks used to run
    # after init_process_group, so a config error left an initialized process
    # group behind that nothing owned -- _runtime_initialized stays False on
    # the raise path, so no later destroy() would tear it down.
    if not mesh_topology:
        raise ValueError(
            "mesh_topology cannot be empty. "
            "Specify dimensions, e.g., {'dp': -1, 'pp': 1, 'tp': 1}",
        )
    if sum(size < 0 for size in mesh_topology.values()) > 1:
        raise ValueError(
            f"At most one mesh dimension can be -1 (auto), got {mesh_topology}",
        )

    device = get_device(device)

    _set_float32_matmul_precision(float32_matmul_precision)

    if deterministic:
        enable_determinism()

    if backend is None:
        backend = "nccl" if device.type.startswith("cuda") else "gloo"

    # Bind this rank to its own GPU before initializing NCCL. torchrun launches
    # every rank with the same default device (cuda:0); without an explicit
    # bind, NCCL sees the same device on multiple ranks and raises "Duplicate
    # GPU detected" (Issue#368). ``get_node_local_rank`` reads ``LOCAL_RANK``
    # and falls back to 0 for single-process / non-torchrun launches.
    if device.type == "cuda":
        local_rank = torch.distributed.get_node_local_rank(fallback_rank=0)
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            timeout=torch.distributed.default_pg_timeout,
            device_id=device if device.type == "cuda" else None,
        )

    world_size_actual = torch.distributed.get_world_size()

    if any(s < 0 for s in mesh_topology.values()):
        world_size_expected = math.prod(s for s in mesh_topology.values() if s > 0)
        n = world_size_actual // world_size_expected
        mesh_topology = {k: n if v < 0 else v for k, v in mesh_topology.items()}

    world_size_expected = math.prod(mesh_topology.values())
    if world_size_actual != world_size_expected:
        raise RuntimeError(
            f"World size {world_size_actual} does not match mesh topology "
            f"{mesh_topology} (expected {world_size_expected})",
        )

    global _device_mesh  # noqa: PLW0603
    _device_mesh = init_device_mesh(
        device.type,
        tuple(mesh_topology.values()),
        mesh_dim_names=tuple(mesh_topology.keys()),
    )

    _runtime_initialized = True

    return _device_mesh


def _set_float32_matmul_precision(
    precision: Float32MatmulPrecision | None,
) -> None:
    """Apply the requested float32 matmul precision override."""
    if precision is not None:
        torch.set_float32_matmul_precision(precision)


def destroy_global_device_mesh() -> None:
    """Cleanup global device mesh and distributed runtime (destroy process group).

    WARNING: Must be called manually before process exit in distributed training.
    Do NOT rely on garbage collection to call this.
    """
    global _runtime_initialized, _device_mesh, _single_process_settings  # noqa: PLW0603
    if not _runtime_initialized:
        return
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _device_mesh = None
    _single_process_settings = None
    _runtime_initialized = False
