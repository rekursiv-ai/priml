"""Dispatch eager tensor operations without loading compiler internals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from torch.utils._python_dispatch import TorchDispatchMode

import torch


if TYPE_CHECKING:
    from torch._ops import OpOverload


class EagerDispatchMode(TorchDispatchMode, ABC):
    """Disable tracing of dispatch handlers only when a caller is compiling."""

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Route an operation through the handler without eager compiler imports."""
        del types
        if torch.compiler.is_compiling():
            return torch.compiler.disable(self._dispatch)(func, args, kwargs or {})
        return self._dispatch(func, args, kwargs or {})

    # Torch's automatic disable wrapper imports Dynamo on first eager dispatch.
    @classmethod
    @override
    def _should_skip_dynamo(cls) -> bool:
        return False

    @abstractmethod
    def _dispatch(
        self,
        func: OpOverload[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Handle one operation outside compiler tracing."""
