"""Profiling wrapper for measuring processor GPU execution time."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging

from configgle import Fig, Makeable

import torch

from priml.data.custom_types import Processor


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


__all__ = ["ProfiledProcessor"]


class ProfiledProcessor:
    """Wrap a processor to measure GPU execution time using CUDA events.

    Records CUDA events before and after each sample is processed by the
    wrapped processor. Measures actual GPU kernel execution time.

    Example:
        ProfiledProcessor.Config(
            processor=CLIPEmbedding.Config(),
            name="CLIP",
        )

    """

    class Config(Fig["ProfiledProcessor"]):
        processor: Makeable[object] | None = None
        """The processor being timed; required."""

        name: str = ""
        """Label in the timing log; empty takes the processor's class name."""

    Input = dict[str, object]
    Output = dict[str, object]

    def __init__(self, config: Config):
        if config.processor is None:
            raise ValueError("Must specify `processor`.")
        self.processor = cast(
            Processor[dict[str, object], dict[str, object]],
            config.processor.make(),
        )
        self.name = config.name or type(self.processor).__name__
        self.use_cuda: bool = torch.cuda.is_available()

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Profile processor execution time per sample.

        Records CUDA events before and after processing each sample.
        Logs GPU execution time at DEBUG level.

        Requires:
            - (any fields required by underlying processor)

        Adds:
            - (any fields added by underlying processor)

        """
        processed = self.processor(samples)

        if not self.use_cuda:
            yield from processed
            return

        while True:
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                sample = next(processed)

                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms: float = start_event.elapsed_time(end_event)
                logger.debug("%s: %.1fms", self.name, elapsed_ms)

                yield sample

            except StopIteration:
                break
