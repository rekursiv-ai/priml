from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.testing import cuda_graphs
from priml.testing.cuda_graphs import (
    CaptureError,
    FakeCuda,
    FakeGraph,
    FakeStream,
    cpu_cuda_graphs,
)


if TYPE_CHECKING:
    from collections.abc import Callable


class _Step:
    """A step with a counter, a carry, and a Python call count."""

    def __init__(self) -> None:
        self.counter = torch.zeros((), dtype=torch.int64)
        self.carry = torch.arange(3.0)
        self.calls = 0

    def __call__(self) -> Tensor:
        self.calls += 1
        self.counter.add_(1)
        self.carry.mul_(2)
        return self.carry + self.counter


def _capture(step: _Step, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeGraph, Tensor]:
    cpu_cuda_graphs(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        out = step()
    assert isinstance(graph, FakeGraph)
    return graph, out


def test_a_capture_runs_python_once_and_leaves_every_tensor_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _Step()

    graph, _ = _capture(step, monkeypatch)

    assert step.calls == 1
    assert step.counter.item() == 0
    assert step.carry.tolist() == [0.0, 1.0, 2.0]
    assert graph.ops == ["aten::add_.Tensor", "aten::mul_.Tensor", "aten::add.Tensor"]


def test_each_replay_reruns_the_ops_into_the_captured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _Step()
    graph, out = _capture(step, monkeypatch)

    graph.replay()
    assert out.tolist() == [1.0, 3.0, 5.0]
    graph.replay()

    assert step.calls == 1
    assert graph.replays == 2
    assert step.counter.item() == 2
    assert out.tolist() == [2.0, 6.0, 10.0]


def test_a_replay_draws_as_the_eager_step_does(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(5)
    noise = torch.empty(4)
    cpu_cuda_graphs(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        noise.uniform_(generator=generator)
    assert isinstance(graph, FakeGraph)
    assert generator.initial_seed() == 5

    eager = torch.Generator().manual_seed(5)
    for _ in range(2):
        graph.replay()
        assert torch.equal(noise, torch.empty(4).uniform_(generator=eager))


def test_each_call_sizes_its_results_by_its_own_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One op met at two shapes, or two dtypes, makes results of each."""
    x = torch.arange(3.0)
    y = torch.arange(6.0).view(2, 3)
    cpu_cuda_graphs(monkeypatch)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        added = (x + 1, y + 1)
        moved = (x.to(torch.float64), x.to(torch.float16))
    assert [t.shape for t in added] == [(3,), (2, 3)]
    assert [t.dtype for t in moved] == [torch.float64, torch.float16]


def test_a_view_is_taken_once_at_capture_and_reads_each_replays_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A view launches no kernel, so the graph records none; an in-place one too."""
    x = torch.arange(6.0).view(2, 3)
    y = torch.zeros(2, 3)
    cpu_cuda_graphs(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        doubled = x[1] * 2
        y.t_()
    assert isinstance(graph, FakeGraph)
    assert graph.ops == ["aten::mul.Tensor"]
    assert y.shape == (3, 2)
    x.add_(1)

    for _ in range(2):
        graph.replay()

    assert doubled.tolist() == [8.0, 10.0, 12.0]
    assert y.shape == (3, 2)


def test_a_floating_pointwise_call_is_sized_without_a_meta_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.arange(6.0).view(2, 3)
    cpu_cuda_graphs(monkeypatch)
    monkeypatch.setattr(cuda_graphs, "_shaped", _no_meta_run)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.sigmoid(x.t() * 2.0)
    assert isinstance(graph, FakeGraph)
    assert (out.shape, out.stride()) == ((3, 2), (1, 3))

    graph.replay()

    assert torch.equal(out, torch.sigmoid(x.t() * 2.0))


def test_an_integer_pointwise_call_is_sized_on_meta_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run at capture, a division by the zeros a replay finds filled would raise."""
    numerators = torch.arange(3) + 4
    divisors = torch.zeros(3, dtype=torch.int64)
    cpu_cuda_graphs(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quotients = torch.div(numerators, divisors, rounding_mode="floor")
    assert isinstance(graph, FakeGraph)
    divisors.fill_(2)

    graph.replay()

    assert quotients.tolist() == [2, 2, 3]


def test_ops_returning_several_results_or_their_arguments_replay_into_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.tensor([[1.0, 5.0, 2.0], [4.0, 0.0, 6.0]])
    counts = [torch.zeros(2)]
    best, where = torch.empty(3), torch.empty(3, dtype=torch.int64)
    cpu_cuda_graphs(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        made = x.max(dim=0)
        written = torch.max(x, dim=0, out=(best, where))
        torch._foreach_add_(counts, 1.0)
    assert isinstance(graph, FakeGraph)
    assert written[0] is best
    assert written[1] is where
    assert counts[0].tolist() == [0.0, 0.0]

    graph.replay()

    for result in (made, written):
        assert result[0].tolist() == [4.0, 5.0, 6.0]
        assert result[1].tolist() == [1, 0, 1]
    assert counts[0].tolist() == [1.0, 1.0]


def test_a_capture_leaves_the_generator_where_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state()
    cpu_cuda_graphs(monkeypatch)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        torch.rand(3, generator=generator)
    assert torch.equal(generator.get_state(), before)


# Defined ahead of the test: ``parametrize`` reads them as the module loads.
def _item(x: Tensor) -> object:
    return x.sum().item()


def _nonzero(x: Tensor) -> object:
    return x.nonzero()


def _mask(x: Tensor) -> object:
    return x[x > 1]


def _branch(x: Tensor) -> object:
    return bool(x.any())


def _to_numpy(x: Tensor) -> object:
    return x.numpy()


def _to_list(x: Tensor) -> object:
    return x.tolist()


@pytest.mark.parametrize(
    "unsafe",
    [_item, _nonzero, _mask, _branch, _to_numpy, _to_list],
    ids=["item", "nonzero", "mask", "branch", "numpy", "tolist"],
)
def test_a_host_read_or_data_dependent_shape_breaks_the_capture(
    monkeypatch: pytest.MonkeyPatch,
    unsafe: Callable[[Tensor], object],
) -> None:
    x = torch.arange(3.0)
    cpu_cuda_graphs(monkeypatch)
    with (
        pytest.raises(CaptureError, match="not permitted while a stream captures"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        unsafe(x)


def test_a_graph_captures_once_and_replays_only_once_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _Step()
    graph, _ = _capture(step, monkeypatch)
    with pytest.raises(CaptureError, match="once"), FakeCuda().graph(graph):
        pass
    with pytest.raises(CaptureError, match="nothing captured"):
        FakeGraph().replay()


def test_the_stand_ins_put_cuda_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = cpu_cuda_graphs(monkeypatch)

    graph = torch.cuda.CUDAGraph(keep_graph=True)

    assert cuda.graphs == [graph]
    assert isinstance(graph, FakeGraph)
    assert graph.keep_graph
    assert graph.pool() == (0, id(graph))
    assert torch.cuda.default_generators[0] is torch.default_generator
    assert isinstance(torch.cuda.current_stream(), FakeStream)
    assert isinstance(torch.cuda.default_stream(0), FakeStream)
    with torch.cuda.device(torch.device("cpu")), torch.cuda.stream(None):
        torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=torch.device("cuda"), priority=-1)
    stream.wait_stream(torch.cuda.current_stream())
    stream.synchronize()
    assert stream.cuda_stream == 0
    assert stream.priority == -1
    assert torch.cuda.Stream().priority == 0
    assert torch.equal(torch.cuda.get_rng_state(0), torch.get_rng_state())
    # Outside a capture the host reads read, through an instance or the class.
    assert torch.arange(2).tolist() == [0, 1]
    assert Tensor.tolist(torch.arange(2)) == [0, 1]
    assert torch.arange(2).numpy().tolist() == [0, 1]


def _no_meta_run(*args: object) -> object:
    del args
    raise AssertionError("a meta run sized a call")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
