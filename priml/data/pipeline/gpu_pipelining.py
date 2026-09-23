"""Pipelined GPU processor for overlapping data transfer with compute.

Implements double-buffering pattern to hide CPU→GPU transfer latency by
transferring batch N+1 while computing on batch N.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from dataclasses import field
from typing import Protocol, TypeVar, cast

import logging

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.data.custom_types import Processor
from priml.lib.traverse import (
    could_path_lead_to_pattern,
    path_matches_pattern,
    recursively_iterate_over_object_descendants,
)


logger = logging.getLogger(__name__)


__all__ = [
    "InferenceMode",
    "PipelinedGPUProcessor2Stream",
    "PipelinedGPUProcessor3Stream",
]


_S = TypeVar("_S")


# Named rather than lambdas: a config field holding a closure makes two
# otherwise-identical configs unequal (closures compare by identity), and its
# repr carries an address, so the printed tree names nothing a reader can act
# on. A module-level function is one object, so equality and the print both
# hold.
def to_cuda_non_blocking(x: object) -> object:
    """Move a tensor to CUDA without blocking; pass anything else through.

    Args:
      x: X.

    Returns:
      result: The object.

    """
    return x.to("cuda", non_blocking=True) if isinstance(x, Tensor) else x


def to_cpu(x: object) -> object:
    """Move a tensor to CPU, blocking until the copy lands.

    Args:
      x: X.

    Returns:
      result: The Tensor.

    """
    return x.to("cpu") if isinstance(x, Tensor) else x


def to_cpu_non_blocking(x: object) -> object:
    """Move a tensor to CPU without blocking; pass anything else through.

    Args:
      x: X.

    Returns:
      result: The object.

    """
    return x.to("cpu", non_blocking=True) if isinstance(x, Tensor) else x


class InferenceMode:
    """Wrap sample iterator to run under torch.inference_mode() context.

    Disables autograd tracking for faster inference. Use as a standalone
    processor to wrap iterators. Typically goes last in the pipeline so that
    upstream GPU processing happens inside the inference mode context.

    Example:
        samples = load_samples()
        samples = PipelinedGPUProcessor(config)(samples)
        samples = InferenceMode()(samples)  # Goes last

    """

    class Config(Fig["InferenceMode"]): ...

    def __init__(self, config: Config) -> None:
        pass

    def __call__(self, samples: Iterator[object]) -> Iterator[object]:
        """Apply to the input."""
        with torch.inference_mode():
            yield from samples


type Sample = Mapping[str, object]
"""One batch in flight through a pipelined stage.

These stages move and reshape whatever fields their configured processors
produce; they add no keys of their own. A closed ``TypedDict`` would name a
field set neither stage controls, which is what forced a cast at every yield
and at every call site.
"""


class _SpecConfig(Protocol):
    """The config fields every pipelined stage shares."""

    processors: MutableSequence[Makeable[Processor[Sample, Sample]]]

    input_spec: dict[str, Callable[[object], object]]

    output_spec: dict[str, Callable[[object], object]]

    input_require_tensor: bool

    output_require_tensor: bool


class _SpecTransformMixin:
    """Shared spec-based transform logic for pipelined GPU processors."""

    def __init__(self, config: _SpecConfig) -> None:
        # Built here rather than declared and left to each subclass: both set
        # these identically, and an unassigned declaration is a half-built
        # object every method below would have to tolerate.
        #
        # The processor chain is heterogeneous -- each link's output feeds the
        # next link's input -- so it is typed by the protocol's widest
        # instantiation rather than by any one stage's pair.
        self.processors: list[Processor[Sample, Sample]] = [
            p.make() for p in config.processors
        ]
        self.input_spec = config.input_spec
        self.output_spec = config.output_spec
        self.input_require_tensor = config.input_require_tensor
        self.output_require_tensor = config.output_require_tensor

    def _process_batch(self, batch: Sample) -> Sample:
        """Chain all processors over a single batch."""
        batch_iter: Iterator[Sample] = iter([batch])
        for processor in self.processors:
            batch_iter = processor(batch_iter)
        return next(batch_iter)

    def _apply_input_stream(self, samples: Iterator[_S]) -> Iterator[_S]:
        """Apply the input spec to each sample in a stream."""
        for sample in samples:
            yield self._apply_input(sample)

    def _apply_input(self, sample: _S) -> _S:
        """Apply the input spec, honoring ``input_require_tensor``."""
        return self._apply_spec(
            sample,
            self.input_spec,
            require_tensor=self.input_require_tensor,
        )

    def _apply_output(self, sample: _S) -> _S:
        """Apply the output spec, honoring ``output_require_tensor``."""
        return self._apply_spec(
            sample,
            self.output_spec,
            require_tensor=self.output_require_tensor,
        )

    def _apply_spec(
        self,
        sample: _S,
        spec: dict[str, Callable[[object], object]],
        *,
        require_tensor: bool,
    ) -> _S:
        if not spec:
            return sample

        patterns = set(spec.keys())

        for path, value in recursively_iterate_over_object_descendants(
            sample,
            recurse=lambda p, _: self._should_recurse_for_spec(p, patterns),
        ):
            if require_tensor and not isinstance(value, Tensor):
                continue

            if not path:
                continue

            transform = self._get_transform_for_path(path, spec)
            if not transform:
                continue

            parent: object = sample
            for step in path[:-1]:
                if isinstance(parent, Mapping):
                    parent = cast(Mapping[object, object], parent)[step]
                elif isinstance(parent, Sequence) and isinstance(step, int):
                    parent = parent[step]
                else:
                    break

            final_key = path[-1]
            # Written through the narrowed arm, not a union: the membership
            # test that proves the write legal is the one that types it.
            if isinstance(parent, MutableMapping):
                cast(MutableMapping[object, object], parent)[final_key] = transform(
                    value,
                )
            elif isinstance(parent, MutableSequence) and isinstance(final_key, int):
                cast(MutableSequence[object], parent)[final_key] = transform(value)

        return sample

    def _get_transform_for_path(
        self,
        path: tuple[int | str, ...],
        spec: dict[str, Callable[[object], object]],
    ) -> Callable[[object], object] | None:
        for pattern, transform in spec.items():
            if path_matches_pattern(path, pattern):
                return transform
        return None

    def _should_recurse_for_spec(
        self,
        path: tuple[int | str, ...],
        patterns: set[str],
    ) -> bool:
        if not path:
            return True

        if "**" in patterns:
            return True

        path_str = ".".join(str(p) for p in path)

        return any(
            could_path_lead_to_pattern(path_str, pattern) for pattern in patterns
        )


class PipelinedGPUProcessor3Stream(_SpecTransformMixin):
    """Process batches on GPU with 3-stream pipelined data transfers.

    Overlaps CPU→GPU data transfer of batch N+1 with GPU compute on batch N
    using separate CUDA streams. This hides transfer latency and improves
    throughput for GPU-bound processing.

    Architecture:
        - stream_input: Transfers next batch to GPU asynchronously
        - stream_work: GPU processing
        - stream_output: Transfers results to CPU asynchronously

    Usage:
        config = PipelinedGPUProcessor3Stream.Config()
        config.processors.append(MyGPUProcessor.Config())
        config.input_spec = {
            "media_tensor": lambda x: rgb2float(
                x.to("cuda", torch.float16), inplace=True
            )
        }
        config.output_spec = {
            "embeddings.*": lambda x: x.to("cpu")
        }
        processor = config.make()

    Example:
        # Before: Sequential transfer + compute
        for batch in batches:
            batch = batch.to("cuda")  # Wait for transfer
            output = model(batch)     # Then compute

        # After: Pipelined transfer + compute
        for batch in PipelinedGPUProcessor3Stream(config)(batches):
            # Transfer N+1 happens while computing N
            output = ...  # tensors transformed per spec

    """

    class Config(Fig["PipelinedGPUProcessor3Stream"]):
        processors: MutableSequence[Makeable[Processor[Sample, Sample]]] = field(
            default_factory=list[Makeable[Processor[Sample, Sample]]],
        )
        """GPU-resident processors run on each batch, in order."""

        input_spec: dict[str, Callable[[object], object]] = field(
            default_factory=lambda: {"**": to_cuda_non_blocking},
        )
        """Field-pattern to transform, applied on ``stream_input``."""

        output_spec: dict[str, Callable[[object], object]] = field(
            default_factory=lambda: {"**": to_cpu},
        )
        """Field-pattern to transform for output (blocking transfers)."""

        input_require_tensor: bool = True
        """Apply input transforms only to tensor values."""

        output_require_tensor: bool = True
        """Apply output transforms only to tensor values."""

    def __init__(self, config: _SpecConfig):
        super().__init__(config)

        # Create separate streams for pipelined GPU operations.
        if torch.cuda.is_available():
            self.stream_input = torch.cuda.Stream()  # CPU→GPU transfers.
            self.stream_work = torch.cuda.Stream()  # GPU processing.
            self.stream_output = torch.cuda.Stream()  # GPU→CPU transfers.
        else:
            self.stream_input = None
            self.stream_work = None
            self.stream_output = None

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Process batches with pipelined GPU transfers.

        Requires:
            - Batched samples with tensor fields

        Yields:
            Processed batches with results on GPU

        """
        # An empty processor list still runs the specs: they are independent
        # transforms, not a stage of the pipeline.
        if not self.processors or not torch.cuda.is_available():
            pipeline: Iterator[Sample] = self._apply_input_stream(samples)
            for processor in self.processors:
                pipeline = processor(pipeline)

            # Apply output transforms to each sample.
            for output_sample in pipeline:
                yield self._apply_output(output_sample)
            return

        # Three-stream pipelined processing with async transfers.
        if self.stream_input is None:
            raise ValueError("Expected self.stream_input is not None.")
        if self.stream_work is None:
            raise ValueError("Expected self.stream_work is not None.")
        if self.stream_output is None:
            raise ValueError("Expected self.stream_output is not None.")

        # Prime: Get first batch.
        try:
            raw_batch = next(samples)
        except StopIteration:
            return

        try:
            # Transfer first batch to GPU.
            with torch.cuda.stream(self.stream_input):
                gpu_batch = self._apply_input(raw_batch)
            self.stream_input.synchronize()

            with torch.cuda.stream(self.stream_work):
                processed_batch = self._process_batch(gpu_batch)

            # Main loop: overlap transfers and compute.
            for next_raw_batch in samples:
                # Start input transfer of N+1 (overlaps with GPU work on N)
                with torch.cuda.stream(self.stream_input):
                    next_gpu_batch = self._apply_input(next_raw_batch)

                # Wait for GPU work on N to complete.
                self.stream_work.synchronize()

                # Start output transfer of N (async, uses current processed_batch)
                with torch.cuda.stream(self.stream_output):
                    cpu_batch = self._apply_output(processed_batch)

                # Wait for input transfer of N+1 to complete before using it.
                self.stream_input.synchronize()

                # GPU work on N+1 overlaps output transfer of N.
                with torch.cuda.stream(self.stream_work):
                    processed_batch = self._process_batch(next_gpu_batch)

                # Wait for output transfer to complete.
                self.stream_output.synchronize()
                yield cpu_batch

            # Final batch.
            self.stream_work.synchronize()
            with torch.cuda.stream(self.stream_output):
                final_cpu = self._apply_output(processed_batch)
            self.stream_output.synchronize()
            yield final_cpu
        finally:
            torch.cuda.synchronize()


class PipelinedGPUProcessor2Stream(_SpecTransformMixin):
    """Process batches on GPU with 2-stream pipelined data transfers.

    Uses wait_stream() for dependencies between compute and transfer streams.
    This is the original origin/main implementation.

    Architecture:
        - stream_output: GPU work + output transfer (sequential on same stream)
        - stream_input: Input transfer (overlaps with stream_output)

    Overlap achieved: Input transfer N+1 happens during (GPU work N + output transfer N).
    """

    class Config(Fig["PipelinedGPUProcessor2Stream"]):
        processors: MutableSequence[Makeable[Processor[Sample, Sample]]] = field(
            default_factory=list[Makeable[Processor[Sample, Sample]]],
        )
        """GPU-resident processors run on each batch, in order."""

        input_spec: dict[str, Callable[[object], object]] = field(
            default_factory=lambda: {"**": to_cuda_non_blocking},
        )
        """Field-pattern to transform, applied on ``stream_input``."""

        output_spec: dict[str, Callable[[object], object]] = field(
            default_factory=lambda: {"**": to_cpu_non_blocking},
        )
        """Field-pattern to transform, applied on ``stream_output``."""

        input_require_tensor: bool = True
        """Apply input transforms only to tensor values."""

        output_require_tensor: bool = True
        """Apply output transforms only to tensor values."""

    def __init__(self, config: _SpecConfig):
        super().__init__(config)

        if torch.cuda.is_available():
            self.stream_output = torch.cuda.Stream()
            self.stream_input = torch.cuda.Stream()
        else:
            self.stream_output = None
            self.stream_input = None

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Process batches with 2-stream pipelined GPU transfers."""
        # An empty processor list still runs the specs: they are independent
        # transforms, not a stage of the pipeline.
        if not self.processors or not torch.cuda.is_available():
            pipeline: Iterator[Sample] = self._apply_input_stream(samples)
            for processor in self.processors:
                pipeline = processor(pipeline)

            # Apply output transforms to each sample.
            for output_sample in pipeline:
                yield self._apply_output(output_sample)
            return

        if self.stream_output is None:
            raise ValueError("Expected self.stream_output is not None.")
        if self.stream_input is None:
            raise ValueError("Expected self.stream_input is not None.")

        try:
            next_sample = next(samples)
        except StopIteration:
            return

        try:
            with torch.cuda.stream(self.stream_input):
                next_sample = self._apply_input(next_sample)

            for incoming_sample in samples:
                self.stream_output.wait_stream(self.stream_input)

                with torch.cuda.stream(self.stream_output):
                    current_sample = self._apply_output(
                        self._process_batch(next_sample),
                    )

                with torch.cuda.stream(self.stream_input):
                    next_sample = self._apply_input(incoming_sample)

                yield current_sample

            self.stream_output.wait_stream(self.stream_input)
            with torch.cuda.stream(self.stream_output):
                final_sample = self._apply_output(
                    self._process_batch(next_sample),
                )
            yield final_sample
        finally:
            torch.cuda.synchronize()
