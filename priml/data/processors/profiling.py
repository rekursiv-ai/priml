"""Profiling wrapper for measuring processor GPU execution time."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        processor: Makeable[Processor[dict[str, object], dict[str, object]]] | None = (
            None
        )
        """The processor being timed; required."""

        name: str = ""
        """Label in the timing log; empty takes the processor's class name."""

    Input = dict[str, object]
    Output = dict[str, object]

    def __init__(self, config: Config):
        if config.processor is None:
            raise ValueError("Must specify `processor`.")
        self.processor: Processor[dict[str, object], dict[str, object]] = (
            config.processor.make()
        )
        self.name = config.name or type(self.processor).__name__
        self.use_cuda: bool = torch.cuda.is_available()

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Profile processor execution time per output.

        Times from the processor's most recent pull of an input to its next
        output, so the time upstream stages spend producing that input is
        excluded. Each measurement synchronizes the device, which serializes
        the pipeline: profile, do not train, with this. Logs at DEBUG.

        Requires:
            - (any fields required by underlying processor)

        Adds:
            - (any fields added by underlying processor)

        """
        if not self.use_cuda:
            yield from self.processor(samples)
            return

        timer = _PullTimer()
        for sample in self.processor(timer.watch(samples)):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms: float = timer.start.elapsed_time(end_event)
            logger.debug("%s: %.1fms", self.name, elapsed_ms)
            yield sample


class _PullTimer:
    """Records a CUDA start event each time the timed processor pulls an input."""

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)

    def watch(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Pass ``samples`` through, restarting the clock after each upstream pull."""
        for sample in samples:
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record()
            yield sample
