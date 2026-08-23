"""Parallelism strategies.

A single ``ParallelStrategy`` protocol owns the full placement lifecycle for a
module, applied in a fixed order inside ``__call__``:

  1. Shard application -- ``fully_shard`` / ``replicate`` / per-block sharding.
  2. Placement -- :func:`place`, which materializes a meta module onto
     ``self.device`` or moves an already-allocated one there.

Placement runs AFTER sharding in the sharded strategies, so each rank
allocates and initializes only its own shard through a DTensor-aware
``reset_parameters``. ``NoParallel`` has no shard step to order against.

:func:`place` is the one entry point for step 2 because its two halves are
mutually exclusive and each is wrong for the other's input: ``Module.to``
cannot copy out of meta storage, and ``to_empty`` would discard an eager
module's weights. Choosing per strategy is what let four of them skip eager
placement while a fifth skipped materialization.

Tensor parallelism lives beside this module in ``train/tensor_parallel.py``:
its plan is read off the model's own ``shard`` declarations rather than
configured here. Pipeline parallelism has no strategy yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import logging

from configgle import Fig
from torch import Tensor, nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._composable.replicate import replicate
from torch.nn.modules.batchnorm import _BatchNorm

import torch

from priml.runtime import get_device, global_device_mesh


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)


__all__ = [
    "DataParallel",
    "FullySharded",
    "HybridSharded",
    "NoParallel",
    "RecursiveSharded",
    "materialize_meta",
    "named_meta_state",
    "place",
]


class NoParallel:
    """Single-device placement (no sharding or replication).

    Compatible with torch.compile.
    """

    class Config(Fig["NoParallel"]):
        device: torch.device | str | None = None
        """Target device; ``None`` takes the runtime's.

        Left unset the loop injects ``runtime.device`` at finalize, so ONE
        field names the device a single-process run uses. Set it to override
        that for this model alone (a CPU reference beside a GPU run)."""

    def __init__(self, config: Config) -> None:
        self.device = get_device(config.device)

    def __call__(self, model: nn.Module) -> nn.Module:
        return place(model, self.device)


class DataParallel:
    """DDP via composable replicate() API.

    Replicates the model across the data-parallel mesh dimension and
    all-reduces gradients. Compatible with torch.compile.
    """

    class Config(Fig["DataParallel"]):
        mesh_dim: str = "dp"
        """Device mesh dimension for data parallelism."""
        bucket_cap_mb: int = 25
        """Gradient all-reduce bucket size in MB."""
        find_unused_parameters: bool = False
        """Reduce grads for params absent from a given backward graph.

        Required for models whose forward uses a data-dependent subset of
        parameters each step (e.g. ACT / recursive reasoning): with the default
        ``False``, DDP assumes every parameter receives a gradient and silently
        skips the all-reduce when one does not, leaving replicas to diverge.
        Carries the usual DDP overhead of an extra graph traversal, so leave it
        off for static models that use all parameters every step."""
        gradient_as_bucket_view: bool = False
        """If True, make gradients views into DDP all-reduce buckets."""

    def __init__(self, config: Config) -> None:
        mesh = global_device_mesh()
        if mesh is None:
            raise RuntimeError(
                "DataParallel requires distributed mode. "
                "Initialize with MultiProcess runtime.",
            )
        if mesh.mesh_dim_names is None or config.mesh_dim not in mesh.mesh_dim_names:
            raise ValueError(
                f"Mesh dimension '{config.mesh_dim}' not in {mesh.mesh_dim_names}. "
                f"Configure runtime with mesh_topology containing '{config.mesh_dim}'.",
            )
        self.device = _mesh_device(mesh)
        self.process_group = mesh.get_group(config.mesh_dim)
        self.bucket_cap_mb = config.bucket_cap_mb
        self.find_unused_parameters = config.find_unused_parameters
        self.gradient_as_bucket_view = config.gradient_as_bucket_view
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        model = place(model, self.device)
        replicate(
            model,
            process_group=self.process_group,
            bucket_cap_mb=self.bucket_cap_mb,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )
        logger.info(
            f"Applied DataParallel: mesh_dim={self.config.mesh_dim}, "
            f"bucket_cap_mb={self.bucket_cap_mb}, "
            f"gradient_as_bucket_view={self.gradient_as_bucket_view}",
        )
        return model


class FullySharded:
    """FSDP via composable fully_shard() API.

    Shards params + grads + optimizer state across the mesh dimension.
    Compatible with torch.compile.
    """

    class Config(Fig["FullySharded"]):
        mesh_dim: str = "dp"
        """Device mesh dimension for sharding."""
        reshard_after_forward: bool = True
        """Re-shard parameters after forward pass to save memory."""
        mp_param_dtype: torch.dtype | None = None
        """Mixed precision dtype for parameters."""
        mp_reduce_dtype: torch.dtype | None = None
        """Mixed precision dtype for gradient reduction."""
        mp_output_dtype: torch.dtype | None = None
        """Mixed precision dtype for output."""

    def __init__(self, config: Config) -> None:
        mesh = global_device_mesh()
        if mesh is None:
            raise RuntimeError(
                "FullySharded requires distributed mode. "
                "Initialize with MultiProcess runtime.",
            )
        if mesh.mesh_dim_names is None or config.mesh_dim not in mesh.mesh_dim_names:
            raise ValueError(
                f"Mesh dimension '{config.mesh_dim}' not in {mesh.mesh_dim_names}. "
                f"Configure runtime with mesh_topology containing '{config.mesh_dim}'.",
            )

        self.device = _mesh_device(mesh)
        self.mesh = mesh[config.mesh_dim]
        self.reshard_after_forward = config.reshard_after_forward
        self.mp_policy = _create_mp_policy(
            param_dtype=config.mp_param_dtype,
            reduce_dtype=config.mp_reduce_dtype,
            output_dtype=config.mp_output_dtype,
        )
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        _shard(
            model,
            mesh=self.mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=self.reshard_after_forward,
        )
        # Placed AFTER sharding so each rank initializes only its local shard
        # with the correct (DTensor-aware) parameter init.
        model = place(model, self.device)
        logger.info(
            f"Applied FullySharded: mesh_dim={self.config.mesh_dim}, "
            f"reshard_after_forward={self.reshard_after_forward}",
        )
        return model


class HybridSharded:
    """2D FSDP: shard within replicas, replicate across replicas (HSDP).

    Uses fully_shard() with a 2D mesh (replicate_dim x shard_dim).
    Compatible with torch.compile.
    """

    class Config(Fig["HybridSharded"]):
        replicate_dim: str = "dp"
        """Mesh dimension for replication across replicas."""
        shard_dim: str = "tp"
        """Mesh dimension for sharding within replicas."""
        reshard_after_forward: bool = True
        """Re-shard parameters after forward pass to save memory."""
        mp_param_dtype: torch.dtype | None = None
        """Mixed precision dtype for parameters."""
        mp_reduce_dtype: torch.dtype | None = None
        """Mixed precision dtype for gradient reduction."""
        mp_output_dtype: torch.dtype | None = None
        """Mixed precision dtype for output."""

    def __init__(self, config: Config) -> None:
        mesh = global_device_mesh()
        if mesh is None:
            raise RuntimeError(
                "HybridSharded requires distributed mode. "
                "Initialize with MultiProcess runtime.",
            )

        missing: list[str] = []
        if (
            mesh.mesh_dim_names is None
            or config.replicate_dim not in mesh.mesh_dim_names
        ):
            missing.append(config.replicate_dim)
        if mesh.mesh_dim_names is None or config.shard_dim not in mesh.mesh_dim_names:
            missing.append(config.shard_dim)
        if missing:
            raise ValueError(
                f"Mesh dimensions {missing} not in {mesh.mesh_dim_names}. "
                f"HybridSharded requires 2D mesh. "
                f"Configure runtime with mesh_topology containing both "
                f"'{config.replicate_dim}' and '{config.shard_dim}'.",
            )

        self.device = _mesh_device(mesh)
        self.mesh = mesh[config.replicate_dim, config.shard_dim]
        self.reshard_after_forward = config.reshard_after_forward
        self.mp_policy = _create_mp_policy(
            param_dtype=config.mp_param_dtype,
            reduce_dtype=config.mp_reduce_dtype,
            output_dtype=config.mp_output_dtype,
        )
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        _shard(
            model,
            mesh=self.mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=self.reshard_after_forward,
        )
        model = place(model, self.device)
        logger.info(
            f"Applied HybridSharded: replicate_dim={self.config.replicate_dim}, "
            f"shard_dim={self.config.shard_dim}, mesh_shape={self.mesh.shape}",
        )
        return model


class RecursiveSharded:
    """Recursive per-block FSDP sharding.

    Shards modules matching module_types in bottom-up order, then shards root.
    Compatible with torch.compile.
    """

    class Config(Fig["RecursiveSharded"]):
        mesh_dim: str = "dp"
        """Device mesh dimension for sharding."""
        module_types: Sequence[type[nn.Module]] = ()
        """Module classes to shard recursively (e.g., (TransformerBlock,))."""
        reshard_after_forward: bool = True
        """Re-shard parameters after forward pass to save memory."""
        mp_param_dtype: torch.dtype | None = None
        """Mixed precision dtype for parameters."""
        mp_reduce_dtype: torch.dtype | None = None
        """Mixed precision dtype for gradient reduction."""
        mp_output_dtype: torch.dtype | None = None
        """Mixed precision dtype for output."""

    def __init__(self, config: Config) -> None:
        mesh = global_device_mesh()
        if mesh is None:
            raise RuntimeError(
                "RecursiveSharded requires distributed mode. "
                "Initialize with MultiProcess runtime.",
            )
        if mesh.mesh_dim_names is None or config.mesh_dim not in mesh.mesh_dim_names:
            raise ValueError(
                f"Mesh dimension '{config.mesh_dim}' not in {mesh.mesh_dim_names}. "
                f"Configure runtime with mesh_topology containing '{config.mesh_dim}'.",
            )
        if not config.module_types:
            raise ValueError(
                "RecursiveSharded requires module_types to be specified. "
                "Provide tuple of module classes to shard (e.g., (TransformerBlock,)).",
            )

        self.device = _mesh_device(mesh)
        self.mesh = mesh[config.mesh_dim]
        self.module_types = tuple(config.module_types)
        self.reshard_after_forward = config.reshard_after_forward
        self.mp_policy = _create_mp_policy(
            param_dtype=config.mp_param_dtype,
            reduce_dtype=config.mp_reduce_dtype,
            output_dtype=config.mp_output_dtype,
        )
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        # Standard per-block FSDP sharding pattern from PyTorch composable API
        # docs. Shard matching submodules in reverse order (leaves first).
        # BatchNorm is sharded individually (leaves-first ordering reaches it
        # before its enclosing block) so ``_shard`` can apply the float32
        # override to its statistics even when it is nested inside a matched
        # block running under a reduced-precision policy.
        matched_count = 0
        for child in reversed(list(model.modules())):
            if not isinstance(child, (*self.module_types, _BatchNorm)):
                continue
            if isinstance(child, self.module_types):
                matched_count += 1
            # The root is sharded once, below, with its own
            # ``reshard_after_forward``. It is counted above rather than
            # skipped outright: a root that is itself a matched type is still
            # a match, and composable FSDP refuses a second application.
            if child is model:
                continue
            _shard(
                child,
                mesh=self.mesh,
                mp_policy=self.mp_policy,
                reshard_after_forward=self.reshard_after_forward,
            )

        if matched_count == 0:
            raise ValueError(
                f"RecursiveSharded found 0 modules matching {self.module_types}. "
                f"Verify module_types contains correct classes.",
            )

        # Shard root module
        _shard(
            model,
            mesh=self.mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=False,
        )

        model = place(model, self.device)
        logger.info(
            f"Applied RecursiveSharded: sharded {matched_count} modules matching "
            f"{[t.__name__ for t in self.module_types]}, mesh_dim={self.config.mesh_dim}",
        )
        return model


def place(model: nn.Module, device: torch.device) -> nn.Module:
    """Put ``model`` on ``device``, by materializing it or by moving it.

    The single answer to "get this module onto its device", because the two
    ways of doing it are mutually exclusive and each is wrong for the other's
    input: ``Module.to`` cannot copy out of meta storage (torch raises and
    names ``to_empty``), and ``to_empty`` would discard the weights an eager
    module already holds. Strategies call this instead of choosing, so a
    strategy cannot forget the case it does not use -- which is how the
    distributed four came to skip eager placement entirely while the
    single-device one skipped materialization.

    Args:
      model: Module to place, meta-constructed or already allocated.
      device: Target device.

    Returns:
      model: The placed module. Materialization is in-place, so the return is
        the same object there; ``Module.to`` is also in-place for parameters,
        and the return keeps one calling shape for both.

    """
    if any(t.is_meta for _, t in named_meta_state(model)):
        materialize_meta(model, device)
        return model
    return model.to(device)


def named_meta_state(model: nn.Module) -> list[tuple[str, Tensor]]:
    """Return every named parameter and buffer (the full materializable state).

    Both parameters and buffers can live on the meta device after lazy
    construction and both must be materialized and initialized; a forgotten
    buffer is as fatal as a forgotten parameter.
    """
    return [*model.named_parameters(), *model.named_buffers()]


def materialize_meta(model: nn.Module, device: torch.device) -> None:
    """Materialize a meta module onto ``device`` and re-init its parameters.

    No-op when the module has no meta state. Otherwise allocates real
    (uninitialized) storage via ``to_empty`` and then makes a single call to
    ``model.reset_parameters()`` so the freshly allocated memory holds a valid
    init rather than ``to_empty``'s garbage.

    Ownership contract (mirrors how configgle composition constructs the tree):
    a module owns re-initializing whatever it constructs. If ``__init__``
    ``.make()``s or otherwise registers a child, that module's
    ``reset_parameters`` must call the child's ``reset_parameters`` -- the same
    way the destructor side of ``new``/``delete`` is owned by whoever
    allocated. ``model.reset_parameters()`` therefore recurses through the
    ownership tree with no external module walk; each parameter is initialized
    by exactly the module that made it.

    Args:
      model: Module to materialize (possibly on the meta device).
      device: Target device for real storage.

    """
    meta_tensors = [t for _, t in named_meta_state(model) if t.is_meta]
    if not meta_tensors:
        return
    model.to_empty(device=device)
    # Poison the freshly-allocated (garbage) storage with NaN, then let the
    # model recurse into its own ownership tree. Any floating-point tensor with
    # ANY remaining NaN was never written (or only partially written) by a
    # reset_parameters -- some module forgot to re-initialize a child it
    # constructed, or wrote only a slice -- so fail loudly rather than train on
    # garbage. ``.any()`` (not ``.all()``) also catches partial-slice writes.
    # Buffers are poisoned and audited too: a forgotten buffer is just as fatal
    # as a forgotten parameter. Integer buffers (e.g. ``num_batches_tracked``)
    # cannot hold NaN, so they are skipped -- ``to_empty`` zero-fills them and
    # there is no garbage-vs-init signal to check.
    with torch.no_grad():
        for _, tensor in named_meta_state(model):
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    getattr(model, "reset_parameters", lambda: None)()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    uninitialized = [
        name
        for name, tensor in named_meta_state(model)
        if tensor.is_floating_point() and torch.isnan(tensor).any()
    ]
    if uninitialized:
        raise RuntimeError(
            "State not initialized after materialize (a module did not reset a "
            f"parameter or buffer it constructed): {uninitialized}",
        )


def _mesh_device(mesh: DeviceMesh) -> torch.device:
    """Resolve the concrete device this rank occupies in ``mesh``."""
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(mesh.device_type)


def _create_mp_policy(
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> MixedPrecisionPolicy | None:
    """Create MixedPrecisionPolicy if any dtype is specified."""
    if param_dtype is None and reduce_dtype is None and output_dtype is None:
        return None
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )


def _module_mp_policy(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
) -> MixedPrecisionPolicy | None:
    """Resolve the mixed-precision policy for one module, overriding BatchNorm.

    BatchNorm accumulates running statistics by reduction; doing that reduction
    in a low-precision ``param_dtype`` (e.g. bfloat16) loses precision and can
    drift the stats. When a reduced-precision base policy is active, force
    BatchNorm to full float32 (params, reduction, and output) so its statistics
    stay exact, mirroring standard FSDP mixed-precision practice. Non-BatchNorm
    modules keep the base policy unchanged.
    """
    if mp_policy is None or not isinstance(module, _BatchNorm):
        return mp_policy
    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )


def _shard(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    reshard_after_forward: bool,
) -> None:
    """Apply fully_shard with optional mixed precision (BatchNorm forced fp32)."""
    policy = _module_mp_policy(module, mp_policy)
    if policy is not None:
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
    else:
        fully_shard(module, mesh=mesh, reshard_after_forward=reshard_after_forward)
