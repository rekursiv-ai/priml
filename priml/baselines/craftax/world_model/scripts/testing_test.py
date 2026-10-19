"""Tests for the world-model scripts' shared test support."""

from __future__ import annotations

from functools import partial

import itertools

import pytest
import torch

from priml.baselines.craftax.world_model.scripts.testing import (
    Clock,
    CpuTorch,
    FakeCuda,
)


def test_a_graph_replays_the_call_captured_into_it() -> None:
    cuda = FakeCuda()
    calls: list[int] = []
    graph = cuda.CUDAGraph()
    stream = cuda.Stream()
    with cuda.graph(graph, stream=stream):
        assert cuda.capturing is graph
        graph.body = partial(calls.append, 7)
    assert cuda.capturing is None
    graph.replay()
    graph.replay()
    assert calls == [7, 7]
    assert cuda.replays == 2
    assert cuda.graphs == [graph]


def test_a_capture_without_a_call_and_an_empty_replay_are_refused() -> None:
    cuda = FakeCuda()
    graph = cuda.CUDAGraph()
    with (
        pytest.raises(ValueError, match="made no engine call"),
        cuda.graph(graph, stream=cuda.Stream()),
    ):
        pass
    with pytest.raises(ValueError, match="nothing captured"):
        graph.replay()


def test_events_time_a_quarter_millisecond_per_replay_between_them() -> None:
    cuda = FakeCuda()
    graph = cuda.CUDAGraph()
    graph.body = list[int]
    first = cuda.Event(enable_timing=True)
    first.record()
    graph.replay()
    begin, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    begin.record()
    assert first.elapsed_time(begin) == 0.25
    for _ in range(4):
        graph.replay()
    order = r"^Both events must be recorded, this one first\.$"
    with pytest.raises(ValueError, match=order):
        begin.elapsed_time(end)
    end.record()
    assert begin.elapsed_time(end) == 1.0
    assert end.elapsed_time(end) == 0.0
    with pytest.raises(ValueError, match=order):
        end.elapsed_time(begin)
    with pytest.raises(
        ValueError,
        match=r"^A timing event needs enable_timing=True\.$",
    ):
        cuda.Event(enable_timing=False)


def test_streams_order_nothing_on_the_cpu() -> None:
    cuda = FakeCuda()
    side = cuda.Stream()
    side.wait_stream(cuda.current_stream())
    with cuda.stream(side):
        pass
    assert cuda.calls == []


def test_the_allocator_reports_fixed_sizes_and_records_each_call() -> None:
    cuda = FakeCuda()
    device = torch.device("cpu")
    cuda.synchronize()
    assert cuda.memory_allocated(device) == 3 << 30
    cuda.reset_peak_memory_stats(device)
    cuda.synchronize(device)
    assert cuda.max_memory_allocated(device) == 5 << 30
    assert cuda.max_memory_reserved(device) == 6 << 30
    assert cuda.get_device_name(device) == "Fake H200"
    assert cuda.calls == [
        "synchronize",
        "memory_allocated",
        "reset_peak_memory_stats",
        "synchronize",
    ]
    with pytest.raises(ValueError, match="not meta"):
        cuda.synchronize(torch.device("meta"))


def test_cpu_torch_maps_cuda_to_the_cpu_and_records_compiles() -> None:
    fake = CpuTorch()
    assert fake.device("cuda") == torch.device("cpu")
    with pytest.raises(ValueError, match="not mps"):
        fake.device("mps")
    assert fake.compile(sum, fullgraph=True) is sum
    assert fake.compiled == [{"fullgraph": True}]
    assert torch.equal(
        fake.zeros(2, dtype=fake.int16),
        torch.zeros(2, dtype=torch.int16),
    )
    assert isinstance(fake.cuda, FakeCuda)


def test_the_clock_reads_its_readings_in_order() -> None:
    clock = Clock([0.0, 1.5])
    assert [clock.perf_counter(), clock.perf_counter()] == [0.0, 1.5]
    ticking = Clock(itertools.count(0.0, 0.5))
    assert [ticking.perf_counter() for _ in range(3)] == [0.0, 0.5, 1.0]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
