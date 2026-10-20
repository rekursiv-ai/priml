"""Observe fitted numerical state and random draws without replacing arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from traceback import walk_stack
from types import FrameType
from typing import TYPE_CHECKING, Self, cast, overload, override
from unittest.mock import patch

import inspect
import sys

from numpy.random import Generator
from torch import Tensor

import numpy as np
import torch

from priml.baselines.name_that_llm.svm import LinearSVC


if TYPE_CHECKING:
    from types import CodeType, TracebackType

    from numpy.typing import ArrayLike, NDArray


class NumericalTrace:
    """Capture one sequential native replay; restore all hooks on exit."""

    def __init__(self) -> None:
        self.record: dict[str, Tensor] = {}
        self.sites: list[str] = []
        self.states: list[dict[str, object]] = []
        self.counts: dict[str, int] = {}
        self.stack = ExitStack()
        self.hooks: _Hooks | None = None

    def __enter__(self) -> Self:
        """Begin observing fitted estimators and the continuing chunk generator."""
        self.hooks = _Hooks(
            previous=sys.getprofile(),
            factory=cast("Callable[[object], Generator]", np.random.default_rng),
        )
        self.stack.enter_context(
            patch.object(np.random, attribute="default_rng", new=self._generator),
        )
        try:
            self._install_observer()
        except BaseException:
            sys.setprofile(self.previous)
            self.stack.close()
            self.hooks = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore profiling and generator bindings even if the observed call fails."""
        del exc_type, exc_value, traceback
        sys.setprofile(self.previous)
        self.stack.close()
        self.hooks = None

    def put(self, prefix: str, value: object) -> None:
        """Record every numeric leaf without rounding, sampling, or hashing.

        Args:
          prefix: Stable name identifying the operation and leaf.
          value: A tensor, NumPy array/scalar, or nested list/dictionary.

        """
        if isinstance(value, Tensor):
            self.record[prefix] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise TypeError(f"Unsupported array dtype at {prefix}: {value.dtype}.")
            self.record[prefix] = torch.from_numpy(value.copy())
        elif isinstance(value, (int, float, np.generic)):
            self.record[prefix] = torch.from_numpy(np.asarray(value))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(
                cast("list[object] | tuple[object, ...]", value),
            ):
                self.put(f"{prefix}/{index}", value=item)
        elif isinstance(value, dict):
            for name, item in cast("dict[str, object]", value).items():
                self.put(f"{prefix}/{name}", value=item)

    @property
    def observing(self) -> bool:
        """Report whether generator operations currently belong to this trace."""
        return self.hooks is not None

    @property
    def previous(self) -> Callable[..., object] | None:
        """Return the profiler acquired by the active trace, if any."""
        return self.hooks.previous if self.hooks is not None else None

    def _generator(self, seed: object = None) -> Generator:
        if self.hooks is None:
            raise ValueError("Expected self.hooks is not None.")
        generator = self.hooks.factory(seed)
        return (
            generator
            if generator is seed
            else _PermutationRecorder(generator, trace=self)
        )

    def _install_observer(self) -> None:
        tool = next(
            index
            for index in range(5, -1, -1)
            if sys.monitoring.get_tool(index) is None
        )
        sys.monitoring.use_tool_id(tool, "name-that-llm-trace")
        self.stack.callback(self._release_monitoring, tool=tool)
        self._configure_monitoring(tool)

    def _configure_monitoring(self, tool: int) -> None:
        for event in (sys.monitoring.events.PY_RETURN, sys.monitoring.events.PY_YIELD):
            sys.monitoring.register_callback(tool, event, self._returned)
        sys.monitoring.register_callback(
            tool,
            sys.monitoring.events.PY_UNWIND,
            self._unwound,
        )
        self._enable_returns(tool)

    def _enable_returns(self, tool: int) -> None:
        sys.monitoring.set_events(
            tool,
            sys.monitoring.events.PY_RETURN
            | sys.monitoring.events.PY_YIELD
            | sys.monitoring.events.PY_UNWIND,
        )

    def _release_monitoring(self, *, tool: int) -> None:
        sys.monitoring.set_events(tool, 0)
        for event in (
            sys.monitoring.events.PY_RETURN,
            sys.monitoring.events.PY_YIELD,
            sys.monitoring.events.PY_UNWIND,
        ):
            sys.monitoring.register_callback(tool, event, None)
        sys.monitoring.free_tool_id(tool)

    def _returned(self, code: CodeType, offset: int, value: object) -> None:
        del offset
        if self._watches_return(code):
            self._profile(frame_for(code), event="return", arg=value)

    def _unwound(self, code: CodeType, offset: int, exception: BaseException) -> None:
        """Preserve the profiler's final return event when an operation raises."""
        del exception
        self._returned(code, offset=offset, value=None)

    def _watches_return(self, code: CodeType) -> bool:
        """Select every return consumed by this observer's observation hooks."""
        return code.co_name in {"fit", "__call__"}

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and self._watches_return(frame.f_code):
            instance = frame.f_locals.get("self")
            if frame.f_code.co_name == "fit" and isinstance(instance, LinearSVC):
                seed = frame.f_locals.get("seed_solver")
                if isinstance(seed, int):
                    index = self.counts.get("liblinear", 0)
                    self.counts["liblinear"] = index + 1
                    self.put(f"rng/liblinear/{index}", value=seed)
            self._observe(
                instance,
                name=frame.f_code.co_name,
                result=arg,
            )

    def _observe(self, instance: object, name: str, result: object) -> None:
        kind = type(instance).__name__
        if kind == "SigmoidCalibration":
            kind = "_SigmoidCalibration"
        if name == "fit" and kind in {
            "StandardScaler",
            "LogisticRegression",
            "LinearSVC",
            "TfidfTransformer",
            "_SigmoidCalibration",
        }:
            index = self.counts.get(kind, 0)
            self.counts[kind] = index + 1
            for key, value in cast("dict[str, object]", vars(instance)).items():
                if key.endswith("_"):
                    self.put(f"{kind}/{index}/{key}", value=value)
        if name == "__call__" and type(result).__name__ == "BatchEncoding":
            index = self.counts.get("tokenizer", 0)
            self.counts["tokenizer"] = index + 1
            self.put(
                f"tokenizer/{index}",
                value=cast("dict[str, object]", vars(result))["data"],
            )


def frame_for(code: CodeType) -> FrameType:
    """Return the executing frame for a monitored code object.

    Args:
      code: Code whose return event is currently executing.

    Returns:
      frame: Live operation frame before its stack is released.

    """
    frame = inspect.currentframe()
    assert isinstance(frame, FrameType)
    while frame.f_code is not code:
        parent = frame.f_back
        assert isinstance(parent, FrameType)
        frame = parent
    return frame


class _PermutationRecorder(Generator):
    def __init__(self, generator: Generator, trace: NumericalTrace) -> None:
        super().__init__(generator.bit_generator)
        self.trace = trace

    @overload
    def permutation(self, x: int, axis: int = 0) -> NDArray[np.int64]: ...

    @overload
    def permutation(self, x: ArrayLike, axis: int = 0) -> NDArray[np.generic]: ...

    @override
    def permutation(self, x: ArrayLike, axis: int = 0) -> NDArray[np.generic]:
        if not self.trace.observing:
            return super().permutation(x, axis=axis)
        self.trace.states.append(dict(self.bit_generator.state))
        result = super().permutation(x, axis=axis)
        self.trace.states.append(dict(self.bit_generator.state))
        index = self.trace.counts.get("permutation", 0)
        self.trace.counts["permutation"] = index + 1
        self.trace.put(f"rng/permutation/{index}", value=result)
        frames = walk_stack(inspect.currentframe())
        next(frames)
        caller, line = next(frames)
        self.trace.sites.append(
            f"{caller.f_code.co_filename}:{line}:permutation",
        )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class _Hooks:
    previous: Callable[..., object] | None
    factory: Callable[[object], Generator]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
