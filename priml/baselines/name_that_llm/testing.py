"""Numerical tracing and miniature fixtures shared by the numerical tests."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Self, cast, overload, override
from unittest.mock import patch

import inspect
import sys

from numpy.random import Generator
from torch import Tensor

import numpy as np
import torch

from priml.baselines.name_that_llm.experiments import exp000
from priml.baselines.name_that_llm.model import (
    CenteredHeads,
    QwenHeads,
    Split,
    SvmBranch,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import CodeType, TracebackType

    from numpy.typing import ArrayLike, NDArray

    from priml.baselines.name_that_llm.benchmark import Benchmark


class NumericalTrace:
    """Capture one sequential native replay; restore all hooks on exit."""

    def __init__(self) -> None:
        self.record: dict[str, Tensor] = {}
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
            if value.dtype.kind not in ("b", "i", "u", "f"):
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
        # Python 3.12's ``free_tool_id`` leaves the tool's events set; 3.14 clears them.
        sys.monitoring.set_events(tool, sys.monitoring.events.NO_EVENTS)
        for event in (
            sys.monitoring.events.PY_RETURN,
            sys.monitoring.events.PY_YIELD,
            sys.monitoring.events.PY_UNWIND,
        ):
            sys.monitoring.register_callback(tool, event, None)
        sys.monitoring.free_tool_id(tool)

    def _returned(self, code: CodeType, offset: int, value: object) -> None:
        del offset, value
        self._deliver(code)

    def _unwound(self, code: CodeType, offset: int, exception: BaseException) -> None:
        """Record a fit that raises with whatever state it reached."""
        del offset, exception
        self._deliver(code)

    def _deliver(self, code: CodeType) -> None:
        if code.co_name == "fit":
            self._observe(frame_for(code).f_locals.get("self"))

    def _observe(self, instance: object) -> None:
        kind = type(instance).__name__
        if kind in {
            "StandardScaler",
            "LogisticRegression",
            "LinearSVC",
            "TfidfTransformer",
            "SigmoidCalibration",
        }:
            index = self.counts.get(kind, 0)
            self.counts[kind] = index + 1
            for key, value in cast("dict[str, object]", vars(instance)).items():
                if key.endswith("_"):
                    self.put(f"{kind}/{index}/{key}", value=value)


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
        result = super().permutation(x, axis=axis)
        if self.trace.observing:
            index = self.trace.counts.get("permutation", 0)
            self.trace.counts["permutation"] = index + 1
            self.trace.put(f"rng/permutation/{index}", value=result)
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class _Hooks:
    previous: Callable[..., object] | None
    factory: Callable[[object], Generator]


def recipe() -> Benchmark.Config:
    """Return exp000 finalized, with its saved layers and input paths resolved."""
    return exp000().finalized()


def svm_branch() -> SvmBranch.Config:
    """Return exp000's calibrated lexical branch."""
    branch = recipe().branches["svm"]
    assert isinstance(branch, SvmBranch.Config)
    return branch


def plain_branch() -> QwenHeads.Config:
    """Return exp000's plain Qwen heads, saved layers resolved."""
    branch = recipe().branches["plain"]
    assert isinstance(branch, QwenHeads.Config)
    return branch


def centred_branch() -> CenteredHeads.Config:
    """Return exp000's chunk-centred Qwen heads, saved layers resolved."""
    branch = recipe().branches["centred"]
    assert isinstance(branch, CenteredHeads.Config)
    return branch


def miniature() -> tuple[
    NDArray[np.object_],
    NDArray[np.int64],
    NDArray[np.str_],
    NDArray[np.float32],
]:
    """Build small geometry with all layers, branches and chunk sizes intact.

    Returns:
      texts: Deterministic short training passages.
      targets: Three repeated class indices.
      groups: Six complete prompt groups.
      embeddings: Rows by the four saved layers by hidden width two.

    """
    texts = np.array(
        [f"a{label} b{group}" for group in range(6) for label in range(3)],
        dtype=object,
    )
    targets = np.tile(np.arange(3), reps=6).astype(np.int64)
    groups = np.repeat(np.array(["a", "b", "c", "d", "e", "f"]), repeats=3)
    # One rounding from float64 keeps the inputs identical on every host.
    embeddings = (
        np.sin(np.arange(18 * 4 * 2) / 5).reshape(18, 4, 2)
        + 0.5 * targets[:, None, None]
    ).astype(np.float32)
    return texts, targets, groups, embeddings


def miniature_split() -> Split:
    """Fit on the miniature and predict its first three rows in two clusters.

    Returns:
      split: Complete miniature rows with the first three also predicted.

    """
    texts, targets, groups, embeddings = miniature()
    return Split(
        texts=texts,
        targets=targets,
        groups=groups,
        embeddings=embeddings,
        predict_texts=texts[:3],
        predict_embeddings=embeddings[:3],
        clusters=np.array([2, 2, 3]),
    )


def completed_record(
    trace: NumericalTrace,
    outputs: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Retain every output, fitted numeric field, permutation and operation count.

    Args:
      trace: Observer holding all fitted numeric attributes and random draws.
      outputs: Complete component outputs to append to the record.

    Returns:
      record: Every captured value and operation count.

    """
    for name, value in outputs.items():
        trace.put(name, value=value)
    for name, count in trace.counts.items():
        trace.put(f"counts/{name}", value=count)
    return trace.record


def float32_comparands(record: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Round float64 values to float32 precision, keeping the float64 dtype pinned."""
    return {
        name: value.float().double() if value.dtype == torch.float64 else value
        for name, value in record.items()
    }
