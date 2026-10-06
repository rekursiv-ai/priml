"""Parallelism strategies.

Each strategy applies the placement lifecycle in ``__call__``:

  1. Shard application -- ``fully_shard`` / ``replicate`` / per-block sharding.
  2. Placement -- :func:`place`, which materializes a meta module onto
     ``self.device`` or moves an already-allocated one there.

Placement follows sharding, so each rank allocates and initializes its own
shard through a DTensor-aware ``reset_parameters``. :func:`place` chooses
materialization for meta modules and preserves existing weights for eager ones.

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
from torch.distributed.tensor import DTensor
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
        """Target device; ``None`` takes torch's default device.

        Left unset, the loop sets ``runtime.device`` when it is constructed,
        before it makes its children, so ONE field names the device a
        single-process run uses. Set it to override that for this model alone
        (a CPU reference beside a GPU run)."""

    def __init__(self, config: Config) -> None:
        self.device = get_device(config.device)

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply to the input."""
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
        mesh = _require_mesh("DataParallel")
        _require_mesh_dimension(mesh, config.mesh_dim)
        self.device = _mesh_device(mesh)
        self.process_group = mesh.get_group(config.mesh_dim)
        self.bucket_cap_mb = config.bucket_cap_mb
        self.find_unused_parameters = config.find_unused_parameters
        self.gradient_as_bucket_view = config.gradient_as_bucket_view
        self.mesh_dim = config.mesh_dim

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply to the input."""
        model = place(model, self.device)
        replicate(
            model,
            process_group=self.process_group,
            bucket_cap_mb=self.bucket_cap_mb,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )
        logger.info(
            "Applied DataParallel: mesh_dim=%s, bucket_cap_mb=%s, gradient_as_bucket_view=%s",
            self.mesh_dim,
            self.bucket_cap_mb,
            self.gradient_as_bucket_view,
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
        mesh = _require_mesh("FullySharded")
        _require_mesh_dimension(mesh, config.mesh_dim)

        self.device = _mesh_device(mesh)
        self.mesh = mesh[config.mesh_dim]
        self.reshard_after_forward = config.reshard_after_forward
        self.mp_policy = _create_mp_policy(
            param_dtype=config.mp_param_dtype,
            reduce_dtype=config.mp_reduce_dtype,
            output_dtype=config.mp_output_dtype,
        )
        self.mesh_dim = config.mesh_dim

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply to the input."""
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
            "Applied FullySharded: mesh_dim=%s, reshard_after_forward=%s",
            self.mesh_dim,
            self.reshard_after_forward,
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
        mesh = _require_mesh("HybridSharded")

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
        self.replicate_dim = config.replicate_dim
        self.shard_dim = config.shard_dim

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply to the input."""
        _shard(
            model,
            mesh=self.mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=self.reshard_after_forward,
        )
        model = place(model, self.device)
        logger.info(
            "Applied HybridSharded: replicate_dim=%s, shard_dim=%s, mesh_shape=%s",
            self.replicate_dim,
            self.shard_dim,
            self.mesh.shape,
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
        mesh = _require_mesh("RecursiveSharded")
        _require_mesh_dimension(mesh, config.mesh_dim)
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
        self.mesh_dim = config.mesh_dim

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply to the input."""
        modules = list(model.modules())
        matched_count = sum(isinstance(child, self.module_types) for child in modules)
        if matched_count == 0:
            raise ValueError(
                f"RecursiveSharded found 0 modules matching {self.module_types}. "
                f"Verify module_types contains correct classes.",
            )
        for child in reversed(modules[1:]):
            if not isinstance(child, self.module_types) and not (
                self.mp_policy is not None and isinstance(child, _BatchNorm)
            ):
                continue
            _shard(
                child,
                mesh=self.mesh,
                mp_policy=self.mp_policy,
                reshard_after_forward=self.reshard_after_forward,
            )

        # Shard root module.
        _shard(
            model,
            mesh=self.mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=False,
        )

        model = place(model, self.device)
        logger.info(
            "Applied RecursiveSharded: sharded %s modules matching %s, mesh_dim=%s",
            matched_count,
            [t.__name__ for t in self.module_types],
            self.mesh_dim,
        )
        return model


def place(model: nn.Module, device: torch.device) -> nn.Module:
    """Put ``model`` on ``device``, by materializing it or by moving it.

    ``Module.to`` preserves eager weights but cannot copy meta storage;
    ``to_empty`` allocates meta weights but would discard eager ones.

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

    Args:
      model: Neural network module to inspect.

    Returns:
      state: List of (name, tensor) pairs for all parameters and buffers
        (e.g., [("layer1.weight", Tensor(...)), ...]).

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
    state = named_meta_state(model)
    if not any(t.is_meta for _, t in state):
        return
    if any(not t.is_meta for _, t in state):
        raise ValueError("Cannot materialize a module with mixed meta and eager state.")
    error: Exception | None = None
    try:
        model.to_empty(device=device)
        # Poison floating state with NaN so a missing or partial reset cannot
        # train on garbage. Integer and bool state has no value that cannot also
        # be a legal initial one, so ``to_empty``'s contents stand for it.
        with torch.no_grad():
            for _, tensor in named_meta_state(model):
                if _nan_capable(tensor):
                    tensor.fill_(torch.nan)
        reset_parameters = getattr(model, "reset_parameters", None)
        if reset_parameters is not None:
            reset_parameters()
    except (RuntimeError, ValueError, TypeError, OSError, LookupError) as exc:
        error = exc
    if error is None:
        uninitialized = [
            name
            for name, tensor in named_meta_state(model)
            if _nan_capable(tensor) and bool(torch.isnan(_local(tensor)).any())
        ]
        if uninitialized:
            error = RuntimeError(
                "State not initialized after materialize (a module did not reset a "
                f"parameter or buffer it constructed): {uninitialized}",
            )
    if torch.distributed.is_initialized():
        errors: list[str | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            errors,
            None if error is None else str(error),
        )
        if any(message is not None for message in errors):
            raise RuntimeError(
                f"Distributed materialization failed: {errors}",
            ) from error
    elif error is not None:
        raise error


def _nan_capable(tensor: Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _require_mesh(strategy: str) -> DeviceMesh:
    mesh = global_device_mesh()
    if mesh is None:
        raise RuntimeError(
            f"{strategy} requires distributed mode. "
            "Initialize with MultiProcess runtime.",
        )
    return mesh


def _require_mesh_dimension(mesh: DeviceMesh, name: str) -> None:
    if mesh.mesh_dim_names is None or name not in mesh.mesh_dim_names:
        raise ValueError(
            f"Mesh dimension '{name}' not in {mesh.mesh_dim_names}. "
            f"Configure runtime with mesh_topology containing '{name}'.",
        )


def _mesh_device(mesh: DeviceMesh) -> torch.device:
    """Resolve the concrete device this rank occupies in ``mesh``."""
    if mesh.device_type == "cuda":
        return torch.device("cuda", index=torch.cuda.current_device())
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


# BatchNorm accumulates running statistics by reduction; doing that reduction in a low-
# precision ``param_dtype`` (e.g. bfloat16) loses precision and can drift the stats.
# When a reduced-precision base policy is active, force BatchNorm to full float32
# (params, reduction, and output) so its statistics stay exact, mirroring standard FSDP
# mixed-precision practice. Non-BatchNorm modules keep the base policy unchanged.
def _module_mp_policy(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
) -> MixedPrecisionPolicy | None:
    """Resolve the mixed-precision policy for one module, overriding BatchNorm."""
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
    if mp_policy is not None:
        for child in reversed(list(module.modules())[1:]):
            if isinstance(child, _BatchNorm) and not hasattr(child, "_get_fsdp_state"):
                _shard(
                    child,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
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
