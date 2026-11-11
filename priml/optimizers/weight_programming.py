"""Numerical weight programming through the existing optimizer contract.

A :class:`WeightProgramming` member inside a :class:`CompositeOptimizer`
edits model weights from ``step`` without a mandatory forward or backward.
Subclasses own calibration forwards, objectives, and persistent statistics.
Callers bind the full model and a lazy ``next_batch`` supplier through
:meth:`CompositeOptimizer.programming` before stepping.

The base excludes owned parameters from ordinary backward via
``requires_grad=False``, temporarily restores original eligibility during
:meth:`program_weights`, and restores ownership state afterward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast, overload, override

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping

    from torch import nn
    from torch.optim.optimizer import StateDict as OptimizerStateDict


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateOutput:
    """Report a numerical update and algorithm-specific metrics."""

    updated: bool = False
    """Whether an update was applied."""

    metrics: dict[str, float | Tensor] = field(default_factory=dict)
    """Metrics emitted by the update."""


class WeightProgramming(Optimizer, ABC):
    """Program live weights from ``step`` without a mandatory forward or backward.

    Subclasses own calibration forwards, objectives, and persistent statistics.
    Store persistent method state in ``state`` or ``param_groups``.
    Callers supply the model and lazy calibration input through :meth:`bind`.
    """

    class Config(Fig["Callable[..., WeightProgramming]"]):
        """The programmer's injectable settings."""

        @override
        def make(self) -> Callable[..., WeightProgramming]:
            """Return a constructor awaiting the routed parameters."""
            return partial(
                cast("type[WeightProgramming]", self.parent_class),
                config=self.finalized(),
            )

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        config: Config,
    ) -> None:
        del config
        param_list = list(params)
        super().__init__([{"params": param_list}], {"lr": 0.0})
        self._model: nn.Module | None = None
        self._next_batch: Callable[[], dict[str, object]] = _no_batch
        self.output = UpdateOutput()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Execute one numerical programming call with optional calibration input."""
        if self._model is None:
            raise RuntimeError("step() requires a bound model")
        loss: Tensor | float | None = None
        with self.restore_requires_grad():
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            self.output = self.program_weights(
                self._model,
                next_batch=self._next_batch,
            )
        return loss

    @contextmanager
    def bind(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> Generator[None]:
        """Bind model and input supplier for the programming scope.

        Args:
          model: The full model whose weights are being programmed.
          next_batch: Lazy calibration-input supplier for input-dependent
            programming.

        Yields:
          nothing: Used as a context manager.

        """
        if self._model is not None:
            raise RuntimeError("already bound")
        self._model = model
        self._next_batch = next_batch
        self.output = UpdateOutput()
        try:
            yield
        finally:
            self._model = None
            self._next_batch = _no_batch

    @abstractmethod
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        """Edit owned weights and report the numerical result.

        The method may read the whole model but must preserve sibling
        parameters and their gradients. Owned parameters have their
        ``requires_grad`` temporarily restored during this call.

        Args:
          model: The full model; owned parameters are editable.
          next_batch: Lazy calibration-input supplier; call it to fetch
            the next training batch for input-dependent calibration.

        Returns:
          output: Whether the update was applied and its metrics.

        """
        ...

    def on_epoch_end(self) -> None:
        """Keep no epoch state by default."""
        return

    @contextmanager
    def restore_requires_grad(self) -> Generator[None]:
        """Temporarily restore saved autograd eligibility for owned parameters.

        Yields:
          nothing: Parameters carry their original ``requires_grad`` inside
            the scope, and are refrozen on exit regardless of exceptions.

        """
        try:
            self._restore_requires_grad()
            yield
        finally:
            self._freeze_params()

    @override
    def add_param_group(self, param_group: dict[str, object]) -> None:
        """Extend routed ownership: save requires_grad eligibility and freeze."""
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        params = cast("list[Tensor]", group["params"])
        group["saved_requires_grad"] = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)

    @override
    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore native optimizer state; reinitialize transient binding state."""
        super().__setstate__(state)
        self._model = None
        self._next_batch = _no_batch
        self.output = UpdateOutput()

    @override
    def state_dict(self) -> OptimizerStateDict:
        """Checkpoint optimizer state with programmer type identity."""
        state = super().state_dict()
        state["programmer"] = f"{type(self).__module__}.{type(self).__qualname__}"
        return state

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore optimizer state with programmer type identity verification."""
        expected = f"{type(self).__module__}.{type(self).__qualname__}"
        loaded = cast("dict[str, object]", state_dict)
        saved_type = loaded.get("programmer")
        if saved_type is None:
            raise ValueError(
                f"State has no programmer tag; expected {expected!r}.",
            )
        if saved_type != expected:
            raise ValueError(
                f"State was saved by {saved_type!r}, not {expected!r}.",
            )
        super().load_state_dict(cast("OptimizerStateDict", state_dict))

    def _restore_requires_grad(self) -> None:
        for group in self.param_groups:
            saved = cast("list[bool]", group["saved_requires_grad"])
            for p, s in zip(
                cast("list[Tensor]", group["params"]),
                saved,
                strict=True,
            ):
                p.requires_grad_(s)

    def _freeze_params(self) -> None:
        for group in self.param_groups:
            for p in cast("list[Tensor]", group["params"]):
                p.requires_grad_(False)


def _no_batch() -> dict[str, object]:
    raise RuntimeError("Calibration input is unavailable outside a training update.")
