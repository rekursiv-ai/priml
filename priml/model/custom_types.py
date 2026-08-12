"""Runtime-checkable Protocols for model channel attributes."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from configgle import Makeable
from torch import Tensor, nn

from priml.math.custom_types import TensorFn


__all__ = [
    "ActivationFn",
    "ChannelsIn",
    "ChannelsInOut",
    "ChannelsOut",
    "ShardableConfig",
    "TensorModule",
    "propagate_attr",
]

ShardStyle = Literal["none", "colwise", "rowwise", "vocab"]
"""Tensor-parallel shard style for a building-block config; none = replicated."""

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

    shard: ShardStyle


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
