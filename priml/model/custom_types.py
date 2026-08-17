"""Runtime-checkable Protocols for model channel attributes."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from configgle import Makeable
from torch import Tensor, nn

from priml.math.custom_types import TensorFn


__all__ = [
    "ActivationFn",
    "AttentionBlock",
    "AttentionKernel",
    "ChannelsIn",
    "ChannelsInOut",
    "ChannelsOut",
    "HasDepth",
    "HeadGeometry",
    "LookupTable",
    "RotaryFactors",
    "ShardStyle",
    "ShardableConfig",
    "TensorModule",
    "propagate_attr",
]

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


class TensorModule(Protocol):
    """A module that maps a Tensor to a Tensor.

    ``nn.Module.__call__`` is untyped, so a plain ``nn.Module`` annotation makes
    every call site ``Any``. This names the contract those call sites need.
    """

    def __call__(self, x: Tensor, /, *args: Any, **kwargs: Any) -> Tensor: ...
    def reset_parameters(self) -> None: ...


@runtime_checkable
class RotaryFactors(Protocol):
    """Maps positions to the ``(cos, sin)`` factors a rotation applies.

    Not a :class:`TensorModule`: a rotary embedding returns a PAIR, and its
    consumer unpacks it. Naming the pair is what lets the slot hold a learned
    or NTK-scaled variant rather than pinning ``RoPE`` itself.

    Fill the slot with an ``nn.Module``: every consumer binds what it builds as
    a child attribute, so a plain object is left out of the module tree.
    """

    def __call__(self, positions: Tensor, /) -> tuple[Tensor, Tensor]: ...


@runtime_checkable
class AttentionKernel(Protocol):
    """Windowed causal attention over ``[..., S, heads, channels_head]``.

    The layout is the one every priml projection emits and the one a fused
    kernel wants, so a kernel needing SDPA's ``[..., heads, S, ...]`` transposes
    internally -- a stride view, not a copy. The window is an argument rather
    than a mask because a mask disqualifies every flash backend; a kernel that
    cannot express one builds the mask itself.
    """

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor: ...


@runtime_checkable
class LookupTable(Protocol):
    """A table mapping integer ids to rows, whose rows are readable.

    Wider than :class:`TensorModule` in the two ways a wrapper needs: the
    weight, because an init the recipe states applies to the table itself, and
    ``to``, because holding a table at a narrower width is a move rather than a
    computation.
    """

    weight: Tensor

    def __call__(self, tokens: Tensor, /, *args: Any, **kwargs: Any) -> Tensor: ...
    def reset_parameters(self) -> None: ...
    def to(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class AttentionBlock(Makeable[nn.Module], Protocol):
    """A block config a stack can reach the attention of.

    ``Makeable`` already says it builds a module; this adds the one thing a
    STACK needs beyond that -- the attention, to push what follows from DEPTH
    (how far back this layer attends, whether a table feeds its gate) and to
    size the tensors the heads consume.

    Declared as a shape rather than as a block CLASS, so a wrapper or a
    replacement qualifies by exposing an attention instead of by inheriting.
    """

    attn: Makeable[nn.Module]
    """The attention sublayer; the stack narrows it to the one it requires."""


@runtime_checkable
class HeadGeometry(Protocol):
    """Answers for the head geometry of whatever attention it composes.

    A model that owns a tensor the heads consume -- rotary factors, sized per
    head; a value embedding, added to the VALUES and so spanning every head --
    has to ask the block, because the two widths are decoupled: an attention
    wider than the residual stream is legal, so dividing the model width would
    misreport every model where they differ.

    Answered BY THE BLOCK rather than read off ``block.attn``, because where
    the attention sits is the block's business: a wrapper (an output gate, an
    adapter, a router) composes one without being one, and a reader reaching
    for a fixed attribute silently gets the wrong answer from every wrapper
    ever written.
    """

    @property
    def heads(self) -> int:
        """Attention heads the block's queries are split into."""
        ...

    @property
    def channels_head(self) -> int:
        """Channels per head."""
        ...


@runtime_checkable
class HasDepth(Protocol):
    """Carries its INDEX in the stack, for anything that scales with position.

    Not a count: ``depth`` says which layer, so a depth-scaled init divides by
    it and an attention reads its reach off a pattern at it. Only blocks whose
    behavior varies with position declare it, which is why a parent pushing one
    gates on this rather than setting it blind.
    """

    depth: int


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
class ShardableConfig(Protocol):
    """A building-block config that declares a tensor-parallel shard style."""

    shard: ShardStyle | None


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
    ``protocol``: a child that implements it MUST accept the value, so a
    failed set (typo, or a value-typed field shadowed by a read-only
    property) raises rather than silently producing a child built with a
    default sentinel dimension. A child that does not implement ``protocol``
    legitimately opts out and is skipped.

    Args:
      config: Child config to mutate.
      name: Attribute name to set.
      value: Value to assign.
      protocol: Runtime-checkable Protocol gating participation. ``None``
        means every child participates (best-effort attributes such as
        ``depth`` that no Protocol governs); absence then skips silently.

    Raises:
      AttributeError: ``config`` implements ``protocol`` yet the attribute
        cannot be set -- a typo or a misdeclared field.

    """
    if protocol is not None and not isinstance(config, protocol):
        return
    if not hasattr(config, name):
        if protocol is None:
            return
        raise AttributeError(
            f"{type(config).__name__} satisfies {protocol.__name__} but has "
            f"no attribute {name!r}; cannot propagate value {value!r}.",
        )
    # A derived read-only property (e.g. channels_out == channels_in) declares
    # the value rather than storing it; leave it untouched.
    descriptor = getattr(type(config), name, None)
    if isinstance(descriptor, property) and descriptor.fset is None:
        return
    setattr(config, name, value)
