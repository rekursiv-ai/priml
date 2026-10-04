"""Tests for replaying a procedure as a captured CUDA graph."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from priml.baselines.craftax.cuda_graph import CudaGraphed


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


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
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


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_python_runs_only_for_the_eager_call_and_the_capture() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    procedure = _Noise(generator)
    replay = CudaGraphed(procedure, generators=(generator,))
    for _ in range(4):
        replay()
    assert procedure.calls == 2


class _CpuStream:
    def __init__(self) -> None:
        self.waited_on: list[object] = []

    def wait_stream(self, stream: object) -> None:
        self.waited_on.append(stream)


class _CpuGraph:
    def __init__(self) -> None:
        self.generators: list[torch.Generator] = []
        self.replays = 0

    def register_generator_state(self, generator: torch.Generator) -> None:
        self.generators.append(generator)

    def replay(self) -> None:
        self.replays += 1


def test_constructor_starts_not_warmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", _CpuStream)

    replay = CudaGraphed(lambda: None)

    assert replay._warm is False


def test_capture_runs_the_procedure_and_replay_reruns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    capture_stream = _CpuStream()
    caller_stream = _CpuStream()
    graph = _CpuGraph()
    graph_context_args: list[tuple[_CpuGraph, _CpuStream]] = []

    def procedure() -> None:
        nonlocal calls
        calls += 1

    def current_stream() -> _CpuStream:
        return caller_stream

    def stream_context(stream: _CpuStream) -> object:
        assert stream is capture_stream
        return nullcontext(stream)

    def graph_context(captured_graph: _CpuGraph, stream: _CpuStream) -> object:
        graph_context_args.append((captured_graph, stream))
        return nullcontext(captured_graph)

    monkeypatch.setattr(torch.cuda, "Stream", lambda: capture_stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    generator = torch.Generator().manual_seed(0)
    replay = CudaGraphed(procedure, generators=(generator,))
    replay()
    replay()
    replay()
    assert calls == 2
    assert graph.generators == [generator]
    assert capture_stream.waited_on == [caller_stream]
    assert caller_stream.waited_on == [capture_stream]
    assert graph_context_args == [(graph, capture_stream)]
    assert graph.replays == 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
