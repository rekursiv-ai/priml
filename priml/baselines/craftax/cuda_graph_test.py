"""Tests for replaying a procedure as a captured CUDA graph."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.cuda_graph import CudaGraphed


pytestmark = [
    pytest.mark.gpu_torch_cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


class _Noise:
    """A procedure drawing from a generator into a buffer it owns."""

    def __init__(self, generator: torch.Generator) -> None:
        self.generator = generator
        self.buffer = torch.zeros(64, device="cuda")
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        drawn = torch.rand(64, generator=self.generator, device="cuda")
        self.buffer.copy_(self.buffer * 0.5 + drawn)


def test_replays_reproduce_the_eager_calls_bit_for_bit() -> None:
    graphed_generator = torch.Generator(device="cuda").manual_seed(3)
    eager_generator = torch.Generator(device="cuda").manual_seed(3)
    graphed = _Noise(graphed_generator)
    eager = _Noise(eager_generator)
    replay = CudaGraphed(graphed, generators=(graphed_generator,))
    for _ in range(5):
        replay()
        eager()
        assert torch.equal(graphed.buffer, eager.buffer)
    assert torch.equal(graphed_generator.get_state(), eager_generator.get_state())


def test_python_runs_only_for_the_eager_call_and_the_capture() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    procedure = _Noise(generator)
    replay = CudaGraphed(procedure, generators=(generator,))
    for _ in range(4):
        replay()
    assert procedure.calls == 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
