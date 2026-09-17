"""Runtime-checkable Protocols for model channel attributes."""

from __future__ import annotations

from typing import Literal, Protocol, Self, TypeGuard, runtime_checkable
from typing_extensions import ParamSpec, TypeVar

from configgle import Makeable
from torch import Tensor, nn

import torch

from priml.math.custom_types import TensorFn


__all__ = [
    "ActivationFn",
    "AttentionKernel",
    "CachedAttention",
    "ChannelsHead",
    "ChannelsIn",
    "ChannelsInOut",
    "ChannelsInOutConfig",
    "ChannelsOut",
    "DeepModelConfig",
    "DepthIndex",
    "HasAttention",
    "HasDepthIndex",
    "HasForwardCached",
    "HasResetParameters",
    "LatentAttentionKernel",
    "LookupTable",
    "NumHeads",
    "RotaryFactors",
    "ShardStyle",
    "Shardable",
    "TensorModule",
    "WeightedTensorModule",
    "flatten_depth_index",
    "has_forward_cached",
    "has_weight",
    "infer_same_width",
    "is_cached_attention",
    "propagate_attr",
]

_Kwargs = ParamSpec("_Kwargs", default=...)
"""The open keyword bus a module accepts beyond its positional contract.

Defaults to ``...`` (gradual), so a bare ``TensorModule`` accepts every
module whose ``forward`` starts with the named tensors: ``nn.Module.__call__``
is generic over the subclass's ``forward``, and a fixed ``**kwargs: object``
rejects a kernel that names its own keywords (``window: int = -1``).
"""

_ModuleT_co = TypeVar(
    "_ModuleT_co",
    bound="TensorModule",
    default="TensorModule",
    covariant=True,
)
"""The module a width-configurable config builds; a plain tensor module by default."""

type DepthIndex = tuple[tuple[int, int], ...]
"""Global-to-local ``(index, count)`` pairs; empty means unspecified."""

ShardStyle = Literal["colwise", "rowwise", "vocab"]
"""Tensor-parallel shard style for a building-block config.

The styles themselves; a slot that may decline to shard spells that
``ShardStyle | None``, so absence reads where it is declared rather than being
folded into the name. ``get_args`` therefore yields exactly the real styles.
"""

type ActivationFn = Makeable[nn.Module | TensorFn] | TensorFn
"""An activation: a config that builds one, or a plain ``Tensor -> Tensor``.

A bare ``nn.Module`` is deliberately not an arm: ``Module.__call__`` is untyped,
so a Module is not statically a :data:`TensorFn` and admitting it would make
every ``self.act(x)`` infer ``Any``. Pass a Module through a ``Makeable``.
"""


@runtime_checkable
class HasResetParameters(Protocol):
    """Owns resettable parameters or buffers."""

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        ...


class TensorModule(HasResetParameters, Protocol[_Kwargs]):
    """A module that maps a Tensor to a Tensor.

    ``nn.Module.__call__`` is untyped, so a plain ``nn.Module`` annotation makes
    every call site ``Any``. This names the contract those call sites need.
    """

    def __call__(
        self,
        x: Tensor,
        /,
        *args: _Kwargs.args,
        **kwargs: _Kwargs.kwargs,
    ) -> Tensor:
        """Apply to the input."""
        ...


class WeightedTensorModule(TensorModule[_Kwargs], Protocol[_Kwargs]):
    """A tensor module exposing its ``weight``.

    Not ``runtime_checkable``: ``nn.Module`` stores parameters in ``_parameters``
    and resolves them via ``__getattr__``, so a data-member protocol check
    (``inspect.getattr_static``) misses every real weight and accepts a plain
    class attribute; use :func:`has_weight` at runtime.
    """

    weight: Tensor


@runtime_checkable
class RotaryFactors(Protocol):
    """Maps positions to the ``(cos, sin)`` factors a rotation applies.

    Not a :class:`TensorModule`: a rotary embedding returns a PAIR, and its
    consumer unpacks it. Naming the pair is what lets the slot hold a learned
    or NTK-scaled variant rather than pinning ``RoPE`` itself.

    Fill the slot with an ``nn.Module``: every consumer binds what it builds as
    a child attribute, so a plain object is left out of the module tree.
    """

    def __call__(self, positions: Tensor, /) -> tuple[Tensor, Tensor]:
        """Apply to the input."""
        ...


@runtime_checkable
class HasForwardCached[CacheT](Protocol):
    """A layer with an explicit cached decode path.

    Method-only on purpose: ``isinstance`` against a runtime-checkable Protocol
    resolves data members through ``inspect.getattr_static``, which never sees
    an ``nn.Module`` submodule (those surface only via ``Module.__getattr__``).
    A protocol naming ``attn`` as a member would therefore be False for every
    real block; read the attribute with a plain ``getattr`` instead.
    """

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: CacheT,
        **kwargs: object,
    ) -> tuple[Tensor, CacheT]:
        """Run the layer while reading and updating ``cache``."""
        ...


@runtime_checkable
class CachedAttention[CacheT](HasForwardCached[CacheT], Protocol):
    """An attention sublayer that also allocates its own cache.

    Two shapes because the two roles differ: a BLOCK has a cached path but
    delegates allocation to its ``attn`` (which alone knows whether the cache
    holds K/V, a compressed latent, or a recurrent state), so a block is only a
    :class:`HasForwardCached`; the attention is the one that allocates.
    """

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CacheT:
        """Allocate an empty cache sized for this attention."""
        ...


@runtime_checkable
class AttentionKernel(Protocol[_Kwargs]):
    """Windowed causal attention over ``[..., S, num_heads, channels_head]``.

    The layout is the one every priml projection emits and the one a fused
    kernel wants, so a kernel needing SDPA's ``[..., num_heads, S, ...]`` transposes
    internally -- a stride view, not a copy. The window is an argument rather
    than a mask because a mask disqualifies every flash backend; a kernel that
    cannot express one builds the mask itself.

    Keyword arguments are an open message bus from the model boundary. Each
    layer consumes what it owns and forwards the remainder unchanged; the
    concrete kernel reads only the messages it understands. The protocol does
    not enumerate that open set because doing so would make every new modular
    consumer a change to this shared interface.
    """

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *args: _Kwargs.args,
        **kwargs: _Kwargs.kwargs,
    ) -> Tensor:
        """Apply to the input."""
        ...


@runtime_checkable
class LatentAttentionKernel(Protocol[_Kwargs]):
    """Attends over a COMPRESSED kv latent rather than materialized K and V.

    A separate protocol from :class:`AttentionKernel` because the inputs differ
    in kind, not in spelling: there is no K or V to pass. What exists is the
    latent ``c_kv``, the head-shared rope key, and the projection that would
    expand them -- so a kernel may either fold that projection into the
    contraction (never forming K/V, which is what makes the cache ~25x
    smaller) or apply it and delegate to an ordinary
    :class:`AttentionKernel`.

    ``w_kr`` and ``w_uv`` arrive as per-head views rather than as the module
    that owns them: under tensor parallelism only this rank's head rows are
    valid, and slicing them in ONE place keeps that correctness argument out
    of every kernel.

    Args:
      q_nope: ``[..., S, num_heads, qk_nope]`` non-rotary query part.
      q_pe: ``[..., S, num_heads, qk_rope]`` rotary query part.
      c_kv: ``[..., T, kv_lora_rank]`` compressed kv latent, head-shared.
      k_pe: ``[..., T, qk_rope]`` rotary key, head-shared.
      w_kr: ``[num_heads, qk_nope, kv_lora_rank]`` latent-to-key projection.
      w_uv: ``[num_heads, v, kv_lora_rank]`` latent-to-value projection.
      **kwargs: Open message bus forwarded unchanged to the attention kernel.

    Returns:
      out: ``[..., S, num_heads, v]`` per-head attention output.

    """

    def __call__(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        c_kv: Tensor,
        k_pe: Tensor,
        *args: _Kwargs.args,
        **kwargs: _Kwargs.kwargs,
    ) -> Tensor:
        """Apply to the input."""
        ...


class LookupTable(WeightedTensorModule[_Kwargs], Protocol[_Kwargs]):
    """A table mapping integer ids to rows, whose rows are readable.

    Wider than :class:`TensorModule` in the two ways a wrapper needs: the
    weight, because an init the recipe states applies to the table itself, and
    ``to``, because holding a table at a narrower width is a move rather than a
    computation.
    """

    def to(self, *, dtype: torch.dtype) -> Self:
        """To."""
        ...


@runtime_checkable
class HasAttention(Makeable[nn.Module], Protocol):
    """A module config exposing residual-stream attention as ``attn``.

    ``Makeable`` already says it builds a module; this adds the one thing a
    STACK needs beyond that -- the attention, to push what follows from DEPTH
    (how far back this layer attends, whether a table feeds its gate) and to
    size the tensors the attention heads consume.

    Declared as a shape rather than as a block CLASS, so a wrapper or a
    replacement qualifies by exposing an attention instead of by inheriting.
    Builds a bare ``nn.Module``, not a :class:`TensorModule`: the stack holds
    its blocks in a ``ModuleList`` and resets only those that are
    :class:`HasResetParameters`, so a block without ``reset_parameters`` is a valid
    layer here.
    """

    attn: Makeable[TensorModule]
    """The attention sublayer; the stack narrows it to the one it requires."""


@runtime_checkable
class NumHeads(Protocol):
    """Has a uniform attention-head count."""

    num_heads: int


@runtime_checkable
class ChannelsHead(Protocol):
    """Has a uniform per-head channel width."""

    channels_head: int


@runtime_checkable
class HasDepthIndex(Protocol):
    """Carries global-to-local positions within nested module stacks."""

    depth_index: DepthIndex


@runtime_checkable
class ChannelsIn(Protocol):
    """Has a channels_in attribute."""

    channels_in: int


@runtime_checkable
class ChannelsOut(Protocol):
    """Has a channels_out attribute."""

    channels_out: int


@runtime_checkable
class ChannelsInOut(ChannelsIn, ChannelsOut, Protocol):
    """Has both channels_in and channels_out attributes."""


@runtime_checkable
class ChannelsInOutConfig(
    Makeable[_ModuleT_co],
    ChannelsInOut,
    Protocol[_ModuleT_co],
):
    """A config with input/output widths that builds a tensor module."""


class DeepModelConfig(ChannelsInOutConfig, Protocol):
    """A buildable deep model exposing its projections and block stack."""

    num_layers: int
    proj_in: Makeable[TensorModule] | None
    block: ChannelsInOutConfig | list[ChannelsInOutConfig]
    proj_out: Makeable[TensorModule] | None


@runtime_checkable
class Shardable(Protocol):
    """A building-block config that declares a tensor-parallel shard style."""

    shard: ShardStyle | None


def has_weight(module: object) -> TypeGuard[WeightedTensorModule]:
    """Check tensor weights, including parameters registered through nn.Module.

    Runtime protocol checks use static lookup, missing registered parameters.

    Args:
      module: Candidate to check for a weight attribute; any object qualifies
        because the tie path is resolved by name at runtime.

    Returns:
      guard: True if module has a weight attribute that is a Tensor.

    """
    return isinstance(getattr(module, "weight", None), Tensor)


def flatten_depth_index(depth_index: DepthIndex) -> int:
    """Flatten a hierarchical depth index using global-to-local mixed radix.

    Args:
      depth_index: Global-to-local ``(index, count)`` pairs. Empty means the
        position is unspecified.

    Returns:
      index: Zero-based flattened index, or ``-1`` when unspecified.

    Raises:
      ValueError: A count is non-positive or an index is outside its level.

    """
    flattened: int = -1
    for level, pair in enumerate(depth_index):
        index: int = pair[0]
        count: int = pair[1]
        if count < 1 or index < 0 or index >= count:
            raise ValueError(
                f"depth_index level {level} must satisfy 0 <= index < count; "
                f"got ({index}, {count}).",
            )
        flattened = index if flattened == -1 else flattened * count + index
    return flattened


# ``isinstance`` against a generic Protocol leaves ``CacheT`` unsolved, so
# every downstream cache is Unknown; these guards name the erased parameter.
def has_forward_cached(module: object) -> TypeGuard[HasForwardCached[object]]:
    """Check for the cached decode path, with an opaque cache type."""
    return isinstance(module, HasForwardCached)


def is_cached_attention(module: object) -> TypeGuard[CachedAttention[object]]:
    """Check for cache allocation plus the cached decode path."""
    return isinstance(module, CachedAttention)


def propagate_attr(
    config: object,
    name: str,
    value: object,
    *,
    protocol: type | None = None,
) -> None:
    """Propagate a parent value to a child config attribute.

    Used by composite-module configs to push shared dimensions down to
    their child configs during ``finalize``. Participation is decided by
    ``protocol``: a child that implements it MUST have the attribute, so a
    MISSING one (a typo, a misdeclared field) raises rather than silently
    producing a child built with a default sentinel dimension. A child that
    does not implement ``protocol`` legitimately opts out and is skipped.

    A participating attribute is assigned normally. Transparent Config wrappers
    may explicitly inherit the runtime Protocol while delegating reads and writes
    through their passthrough mixin.

    Args:
      config: Child config to mutate.
      name: Attribute name to set.
      value: Value to assign.
      protocol: Runtime-checkable Protocol gating participation. ``None``
        means every child participates (best-effort attributes such as
        ``depth`` that no Protocol governs); absence then skips silently.

    Raises:
      AttributeError: ``config`` implements ``protocol`` yet has no such
        attribute at all -- a typo or a misdeclared field.

    """
    participates = protocol is None or isinstance(config, protocol)
    if not hasattr(config, name):
        if not participates or protocol is None:
            return
        raise AttributeError(
            f"{type(config).__name__} satisfies {protocol.__name__} but has "
            f"no attribute {name!r}; cannot propagate value {value!r}.",
        )
    setattr(config, name, value)


def infer_same_width(config: ChannelsInOut) -> None:
    """Fill a width-preserving layer's missing width; both set and unequal raises.

    A norm, a residual block, an attention writing back into its own stream --
    none widens or narrows, so a parent may push either width down and
    ``finalize`` fills the other. Raised at finalize so the ``pprint`` of the
    offending tree names both values.

    Args:
      config: A config declaring ``channels_in`` and ``channels_out`` with ``-1``
        sentinels.

    Raises:
      ValueError: Both widths are set and differ.

    """
    if config.channels_in == -1:
        config.channels_in = config.channels_out
    if config.channels_out == -1:
        config.channels_out = config.channels_in
    if config.channels_in != config.channels_out:
        raise ValueError(
            f"channels_in={config.channels_in} must equal "
            f"channels_out={config.channels_out}.",
        )
