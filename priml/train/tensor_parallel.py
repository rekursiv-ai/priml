"""Generic tensor-parallel applier driven by per-module shard styles.

The sharding plan is the configgle config tree: each building-block ``Config``
carries a ``shard`` style (``"none"``/``"colwise"``/``"rowwise"``/``"vocab"``)
that its runtime module records as ``module.shard``. ``apply_tensor_parallel``
walks the module tree, reads those styles, and applies the matching
``ParallelStyle`` over the ``tp`` mesh dimension via ``parallelize_module``.

Standard ``nn.Linear``/``nn.Embedding`` blocks use the built-in
``ColwiseParallel``/``RowwiseParallel`` styles. Custom layers (hand-written
forwards over DTensor operands, e.g. ``EnsembleLinear``) provide their own
``ParallelStyle`` and expose it as ``module.tensor_parallel_style()`` so the
applier stays generic.

A ``tp`` size of 1 is a structural no-op: the plan is empty-by-effect and the
single-device forward stays bit-for-bit identical to the unsharded model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import logging

from configgle import Fig
from torch import nn
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

import torch

from priml.model.custom_types import ShardStyle
from priml.runtime import global_device_mesh


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)


__all__ = [
    "ShardAware",
    "TensorParallel",
    "TensorParallelStyleProvider",
    "apply_tensor_parallel",
]


@runtime_checkable
class ShardAware(Protocol):
    """A module that declares how it shards under tensor parallelism."""

    @property
    def shard(self) -> ShardStyle | None:
        """The declared style, or ``None`` to stay replicated.

        Read-only, so a module may hold it at a narrower type: MLA admits only
        ``"colwise"``, and a mutable member would be invariant and reject it.
        """
        ...


@runtime_checkable
class TensorParallelStyleProvider(Protocol):
    """A custom layer that supplies its own ``ParallelStyle``.

    Layers whose forward is not a plain ``nn.Linear``/``nn.Embedding`` matmul
    (e.g. ``EnsembleLinear``'s einsum) cannot be sharded by the built-in
    styles -- the forward must see consistent DTensor operands. Such a layer
    implements this method to redistribute its own input/output.
    """

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the ``ParallelStyle`` that shards this layer over ``tp``."""
        ...


@runtime_checkable
class TensorParallelValidator(Protocol):
    """A module that validates its own tensor-parallel preconditions.

    The applier calls this on every submodule after sharding so a layer can
    raise a clear error for an unsupported configuration (e.g. an attention
    block whose fused kernel has no DTensor sharding strategy) instead of
    failing later with a cryptic deep-stack error.
    """

    def assert_tensor_parallel_compatible(self) -> None:
        """Raise if this module cannot run under tensor parallelism."""
        ...


def apply_tensor_parallel(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Shard each submodule per its declared shard style over ``mesh['tp']``.

    Walks the module tree, reads every ``ShardAware`` submodule's ``shard``
    style, and applies the matching ``ParallelStyle`` via
    ``parallelize_module``. Custom layers (``TensorParallelStyleProvider``)
    supply their own style.

    Args:
      model: Module whose submodules declare ``shard`` styles.
      mesh: Device mesh containing a ``tp`` dimension.

    Returns:
      model: The same module, sharded in place. A ``tp`` size of 1 is a
        structural no-op (forward bit-for-bit unchanged).

    """
    tp_mesh = mesh["tp"]
    if tp_mesh.size() == 1:
        return model
    plan: dict[str, ParallelStyle] = {}
    for name, submodule in model.named_modules():
        style = _shard_style(submodule)
        if style is not None:
            plan[name] = style
    if not plan:
        return model
    parallelize_module(model, tp_mesh, plan)
    for submodule in model.modules():
        if isinstance(submodule, TensorParallelValidator):
            submodule.assert_tensor_parallel_compatible()
    logger.info(f"Applied tensor parallel: {len(plan)} sharded submodules.")
    return model


class TensorParallel:
    """Tensor-parallel placement strategy (``ParallelStrategyProtocol``).

    Records the placement device and shards ``model`` over the ``tp`` mesh
    dimension per the shard styles declared on its building-block configs.
    Compatible with torch.compile.
    """

    class Config(Fig["TensorParallel"]):
        mesh_dim: str = "tp"
        """Device mesh dimension for tensor parallelism."""

    def __init__(self, config: Config) -> None:
        mesh = global_device_mesh()
        if mesh is None:
            raise RuntimeError(
                "TensorParallel requires distributed mode. "
                "Initialize with MultiProcess runtime.",
            )
        if mesh.mesh_dim_names is None or config.mesh_dim not in mesh.mesh_dim_names:
            raise ValueError(
                f"Mesh dimension '{config.mesh_dim}' not in {mesh.mesh_dim_names}. "
                f"Configure runtime with mesh_topology containing '{config.mesh_dim}'.",
            )
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if mesh.device_type == "cuda"
            else torch.device(mesh.device_type)
        )
        self.mesh = mesh
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        return apply_tensor_parallel(model, self.mesh)


def _shard_style(module: nn.Module) -> ParallelStyle | None:
    """Resolve the ``ParallelStyle`` for a single submodule, or ``None``.

    Custom layers take priority via ``tensor_parallel_style``; standard
    ``nn.Linear``/``nn.Embedding`` blocks dispatch on their ``shard`` field.
    """
    if isinstance(module, TensorParallelStyleProvider):
        if not isinstance(module, ShardAware) or not module.shard:
            return None
        return module.tensor_parallel_style()
    if not isinstance(module, ShardAware):
        return None
    # Reached with an unlisted style: the annotation binds no value here --
    # ShardAware's isinstance check is hasattr alone, and an override or
    # deserialized config carries unchecked text. Falling through instead
    # would silently leave the layer replicated.
    if module.shard is None:
        return None
    if module.shard == "colwise":
        return ColwiseParallel()
    if module.shard == "rowwise":
        return RowwiseParallel()
    if module.shard == "vocab":
        return _vocab_style(module)
    raise ValueError(
        f"Unknown shard style {module.shard!r} on {type(module).__name__}.",
    )


def _vocab_style(module: nn.Module) -> ParallelStyle:
    """Vocab-dim shard: row-wise for embeddings, col-wise for the lm head.

    An ``nn.Embedding`` shards its table over the vocabulary (dim 0) and takes
    a replicated token-id input; an ``nn.Linear`` head shards its output over
    the vocabulary (dim 0) and returns replicated logits.
    """
    if isinstance(module, nn.Embedding):
        return RowwiseParallel(input_layouts=Replicate())
    return ColwiseParallel(output_layouts=Replicate())
