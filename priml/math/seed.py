"""Random seed utilities for reproducible training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import hashlib
import logging
import os
import random
import secrets

import numpy as np
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)

numpy_rng: np.random.Generator = np.random.default_rng()


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


# Opaque round-trip types: the formats are implementation-defined by
# ``random.getstate`` / ``numpy_rng.bit_generator.state`` respectively,
# and we never inspect them -- only round-trip through ``set_rng_state``.
PythonRngState = tuple[Any, ...]
NumpyRngState = Any


class RngState(TypedDict):
    """Per-component RNG state captured by ``get_rng_state``."""

    python: PythonRngState
    """``random.getstate()``. Opaque round-trip value."""

    torch: torch.Tensor
    """``torch.get_rng_state()`` byte tensor."""

    numpy: NotRequired[NumpyRngState]
    """``numpy_rng.bit_generator.state``. Opaque round-trip value.

    ``get_rng_state`` always writes it, but it is not required on read:
    checkpoints minted by a capture that predates numpy tracking omit it,
    and ``set_rng_state`` must still resume them rather than raising."""

    cuda: NotRequired[list[torch.Tensor]]
    """``torch.cuda.get_rng_state_all()``. Present iff CUDA was
    available and initialized at capture."""

    cuda_uuids: NotRequired[list[str]]
    """Per-device stable identifier (UUID, PCI bus, or name) parallel
    to ``cuda``. Used by ``set_rng_state`` to warn when
    ``CUDA_VISIBLE_DEVICES`` was remapped between capture and restore."""


def enable_determinism(*, cudnn: bool = True, sdpa: bool = True) -> None:
    """Enable deterministic behaviour for reproducible results.

    Call once at program start, before any CUDA operations. The
    ``CUBLAS_WORKSPACE_CONFIG`` env var must be set before CUDA
    initializes or it has no effect; calling after init logs a warning
    so the failure mode is visible.

    Args:
      cudnn: Disable cuDNN benchmark and enable deterministic mode.
      sdpa: Disable flash and memory-efficient SDPA backends.

    References:
      https://docs.pytorch.org/docs/stable/notes/randomness.html

    """
    if torch.cuda.is_initialized():
        logger.warning(
            "enable_determinism called after CUDA initialization; "
            "CUBLAS_WORKSPACE_CONFIG and SDPA backend settings may not "
            "take effect on this process.",
        )
    cublas_default = ":4096:8"
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_default
    elif existing != cublas_default:
        logger.warning(
            "Keeping caller-set CUBLAS_WORKSPACE_CONFIG=%r; "
            "deterministic default would have been %r.",
            existing,
            cublas_default,
        )
    torch.use_deterministic_algorithms(True)
    if cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


_STABLE_REPR_TYPES = (str, bytes, int, float, bool, type(None))


def salt(*args: object) -> int:
    """Hash args to a deterministic non-negative 31-bit salt.

    Args:
      *args: Stable-repr primitives only -- ``str``, ``bytes``, ``int``,
        ``float``, ``bool``, or ``None``. Arbitrary objects are
        rejected because Python's default ``__repr__`` embeds the
        object address, which would make the salt non-deterministic
        across runs.

    Returns:
      salt: Non-negative 31-bit integer derived from the args.

    Raises:
      TypeError: If any argument is not a stable-repr primitive.

    """
    for arg in args:
        if not isinstance(arg, _STABLE_REPR_TYPES):
            raise TypeError(
                "salt() requires primitive args with stable repr "
                f"(str/bytes/int/float/bool/None); got {type(arg).__name__}.",
            )
    return int(hashlib.md5(str(args).encode()).hexdigest(), 16) & 0x7FFFFFFF  # noqa: S324 -- non-cryptographic seed salt; collision risk is acceptable, weak-hash flag does not apply


def make_seed() -> int:
    """Generate a 63-bit seed from OS entropy.

    Returns:
      seed: Non-zero 63-bit integer suitable for ``torch.manual_seed``
        and ``numpy.random.PCG64``.

    """
    return secrets.randbits(63) or 1


def set_seed_local(seed: int | None = None) -> int:
    """Seed Python, NumPy, torch CPU, and visible CUDA devices.

    Each RNG is salted independently to avoid correlated streams.
    The legacy ``np.random`` module is also reseeded so codepaths
    that call ``np.random.rand`` / ``randn`` directly remain
    reproducible. Per-CUDA-device seeding runs only when CUDA is
    already initialized -- otherwise torch's lazy-init path will
    apply the seed on first use.

    Args:
      seed: Seed value. If None, a fresh seed is drawn from OS entropy.

    Returns:
      seed: The seed value used (the argument if non-None, else the
        freshly generated value).

    """
    if seed is None:
        seed = make_seed()

    random.seed(salt("python", seed))
    numpy_rng.bit_generator.state = np.random.PCG64(salt("numpy", seed)).state
    np.random.seed(salt("numpy_legacy", seed))  # noqa: NPY002 -- deliberate legacy reseed
    torch.manual_seed(salt("torch", seed))

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # ``torch.cuda.manual_seed`` seeds the current device, so the
        # ``torch.cuda.device(i)`` context manager is required for the
        # per-device-unique salt to actually land on device ``i``.
        # Seeds every visible CUDA device; callers that own only a
        # subset must restrict ``CUDA_VISIBLE_DEVICES`` before calling.
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.manual_seed(salt("cuda", i, seed))

    logger.info("Using seed: %d", seed)

    return seed


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Reseed Python / NumPy generators per DataLoader worker.

    PyTorch's ``DataLoader`` reseeds torch's RNG in each forked worker
    automatically, but does NOT touch Python's ``random``,
    ``priml.math.seed.numpy_rng``, or the legacy ``np.random``
    module. Without this hook every forked worker draws identical
    correlated sequences from those generators.

    The CUDA RNG is intentionally left untouched: dataloader workers run on
    CPU and should not initialize a CUDA context, so per-worker CUDA reseeding
    is out of scope for this hook.

    Pass as ``worker_init_fn=`` to ``DataLoader``::

        DataLoader(..., worker_init_fn=dataloader_worker_init_fn)

    Args:
      worker_id: The DataLoader worker index, supplied by PyTorch.

    """
    # ``torch.initial_seed`` inside a worker returns the per-worker
    # value torch has already set; mod to ``2**32`` to keep numpy happy.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(salt("python_worker", worker_id, worker_seed))
    numpy_rng.bit_generator.state = np.random.PCG64(
        salt("numpy_worker", worker_id, worker_seed),
    ).state
    np.random.seed(salt("numpy_legacy_worker", worker_id, worker_seed))  # noqa: NPY002 -- deliberate legacy reseed for fork-safety


def _local_salt(
    *,
    base_seed: int,
    salt_by_rank: bool,
    mesh: DeviceMesh | None,
    global_rank: int,
) -> int:
    """Resolve the per-rank salted seed for ``set_seed_distributed``."""
    if not salt_by_rank:
        logger.info("seed=%d not salted.", base_seed)
        return base_seed
    if mesh is not None:
        local_rank = mesh.get_local_rank()
        this_seed = salt("rank", local_rank, base_seed)
        logger.info(
            "seed=%d salted by (mesh) local_rank=%d; this_seed=%d.",
            base_seed,
            local_rank,
            this_seed,
        )
        return this_seed
    this_seed = salt("rank", global_rank, base_seed)
    logger.info(
        "seed=%d salted by global_rank=%d; this_seed=%d.",
        base_seed,
        global_rank,
        this_seed,
    )
    return this_seed


def set_seed_distributed(
    seed: int | None = None,
    mesh: DeviceMesh | None = None,
    *,
    salt_by_rank: bool = True,
) -> tuple[int, int]:
    """Seed all ranks of the default process group from rank 0.

    Broadcasts the base seed from rank 0 to every rank, then optionally
    salts by rank. The user-supplied ``seed`` argument is informational
    on rank>0 -- rank 0's broadcast wins.

    Assumes ``mesh`` (when given) is built over the same default process group
    used for the broadcast, so that ``mesh.get_local_rank()`` indexes the same
    ranks the broadcast reached. A mesh over a disjoint group would salt by an
    unrelated local rank rather than corrupt the broadcast itself.

    Args:
      seed: Base seed value. If None, rank 0 generates from OS entropy.
      mesh: Device mesh for local-rank salting, built over the default
        process group. If None, salts by global rank.
      salt_by_rank: If True, derive a per-rank salted seed; if False,
        every rank ends up with the same seed.

    Returns:
      base_seed: The base seed value used before rank-specific salting.
      local_seed: The seed used after rank-specific salting.

    Raises:
      AssertionError: If the default process group is not initialized.

    """
    assert dist.is_initialized(), (
        "set_seed_distributed requires an initialized default process group; "
        "call dist.init_process_group(...) first or use set_seed_local."
    )
    global_rank = dist.get_rank()
    # NCCL requires CUDA tensors; gloo accepts CPU. Pick once per call
    # to keep rank 0 and rank>0 in lockstep on shape, dtype, and device.
    if dist.get_backend() == "nccl":
        assert torch.cuda.is_available(), (
            "NCCL backend declared but no CUDA devices are available; "
            "use gloo for CPU-only distributed training."
        )
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    if global_rank == 0:
        if seed is None:
            seed = make_seed()
        seed_tensor = torch.tensor(seed, dtype=torch.long, device=device)
    else:
        seed_tensor = torch.zeros((), dtype=torch.long, device=device)
    dist.broadcast(seed_tensor, src=0)
    base_seed = int(seed_tensor.item())
    this_seed = _local_salt(
        base_seed=base_seed,
        salt_by_rank=salt_by_rank,
        mesh=mesh,
        global_rank=global_rank,
    )
    return base_seed, set_seed_local(this_seed)


def _cuda_device_identity(index: int) -> str:
    """Return a stable identifier for CUDA device ``index``."""
    props = torch.cuda.get_device_properties(index)
    for attr in ("uuid", "pci_bus_id", "name"):
        value = getattr(props, attr, None)
        if value:
            return str(value)
    return f"cuda:{index}"


def _warn_unsupported_backend(name: str) -> None:
    logger.warning(
        "%s backend is active but %s RNG state is not captured by "
        "get_rng_state; checkpoint will not round-trip %s reproducibility.",
        name,
        name,
        name,
    )


def get_rng_state() -> RngState:
    """Capture the current RNG state of every backend we manage.

    Returns:
      state: ``python``, ``numpy``, and ``torch`` are always present.
        ``cuda`` and ``cuda_uuids`` are present iff CUDA is available
        and initialized; ``cuda_uuids`` lets ``set_rng_state`` detect
        a ``CUDA_VISIBLE_DEVICES`` remap at restore time. Active MPS
        or XPU backends are logged as warnings -- those backends are
        not yet supported by this module.

        A pure CPU run never initializes a CUDA context, so it omits the
        CUDA entries on a GPU host too. The exception is a process that
        touched CUDA incidentally (a capability probe, an earlier GPU eval)
        and then checkpoints a CPU run: the entries appear despite nothing
        advancing them, which makes resume equality host-dependent. Pop
        them at that call site.

    """
    state: RngState = {
        "python": random.getstate(),
        "numpy": numpy_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
        state["cuda_uuids"] = [
            _cuda_device_identity(i) for i in range(torch.cuda.device_count())
        ]
    if torch.backends.mps.is_available():
        _warn_unsupported_backend("MPS")
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        _warn_unsupported_backend("XPU")
    return state


def set_rng_state(state: RngState) -> None:
    """Restore RNG state captured by ``get_rng_state``.

    Args:
      state: A previously captured ``RngState``. The python/torch keys
        are required; ``numpy`` is restored when present and skipped
        otherwise, so a checkpoint minted before numpy tracking still
        resumes. If ``cuda`` is present, its length must match the
        visible CUDA device count exactly; callers who want to load a
        multi-GPU checkpoint on fewer devices must slice
        ``state['cuda']`` themselves. If ``cuda_uuids`` is present and
        differs from the current devices' identifiers, a warning is
        logged but the restore proceeds.

    Raises:
      KeyError: If either required ``python``/``torch`` key is absent.
      ValueError: If ``len(state['cuda']) != torch.cuda.device_count()``.

    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        numpy_rng.bit_generator.state = state["numpy"]
    if "cuda" in state:
        cuda_states = state["cuda"]
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(cuda_states) != n_devices:
            raise ValueError(
                f"CUDA RNG state mismatch: state has {len(cuda_states)} "
                f"device entries but {n_devices} CUDA devices visible. "
                "Slice state['cuda'] to the desired length if this is "
                "intentional.",
            )
        saved_uuids = state.get("cuda_uuids")
        if saved_uuids is not None and n_devices:
            current_uuids = [_cuda_device_identity(i) for i in range(n_devices)]
            if saved_uuids != current_uuids:
                logger.warning(
                    "CUDA device identities changed since checkpoint: "
                    "saved=%r vs current=%r. RNG state will be restored "
                    "positionally, which may not match the original devices.",
                    saved_uuids,
                    current_uuids,
                )
        if n_devices:
            torch.cuda.set_rng_state_all(cuda_states)
