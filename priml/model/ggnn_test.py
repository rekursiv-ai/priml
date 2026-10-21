"""Check the gated graph network against a per-edge loop reference and torch cost."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Final, cast

import re

from torch import Tensor, nn
from treekle import from_plain

import pytest
import torch

from priml.cost import cost
from priml.model.ggnn import GGNN, GraphBatch, RecurrentCell
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_tensor_golden, rng_fingerprint


_CWD: Final = Path(__file__).resolve().parent
HIDDEN: Final = 5
INPUT: Final = 3
EDGE_TYPES: Final = 4


def tiny_graph() -> GGNN.Config:
    config = GGNN.Config()
    config.channels_hidden = HIDDEN
    config.num_layers = 2
    config.num_edge_types = EDGE_TYPES
    return config


def topology() -> GraphBatch:
    """Three disconnected graphs with ordered directed edges of every type."""
    return GraphBatch(
        source=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 5]),
        destination=torch.tensor([1, 0, 3, 4, 2, 6, 7, 8, 5, 7]),
        edge_type=torch.arange(10).remainder(EDGE_TYPES),
        node_counts=torch.tensor([2, 3, 4]),
    )


def features() -> Tensor:
    return torch.arange(9 * INPUT, dtype=torch.float32).reshape(9, INPUT) / 27


def reference(model: GGNN, graph: GraphBatch, values: Tensor) -> Tensor:
    """Run the recurrence one edge at a time, in the order the edges are given."""
    state = torch.cat([values, values.new_zeros(len(values), HIDDEN - INPUT)], dim=-1)
    for _ in range(model.config.num_layers):
        aggregate = torch.zeros_like(state)
        for source, destination, kind in zip(
            from_plain(cast(object, graph.source.tolist()), list[int]),
            from_plain(cast(object, graph.destination.tolist()), list[int]),
            from_plain(cast(object, graph.edge_type.tolist()), list[int]),
            strict=True,
        ):
            aggregate[destination] += model.linears[kind](state[source])
        state = model.gru(aggregate, state)
    return state


def test_defaults_size_every_child_from_one_hidden_width() -> None:
    config = GGNN.Config().copy_tree().finalize()
    assert (config.channels_hidden, config.num_layers, config.num_edge_types) == (
        128,
        4,
        1,
    )
    assert config.init_edge is None
    assert config.edge.channels_in == config.edge.channels_out == 128
    assert config.cell.channels_in == config.cell.channels_out == 128
    assert config.edge.bias
    assert config.cell.bias


def test_forward_matches_a_per_edge_reference_for_every_edge_layout() -> None:
    _ = torch.manual_seed(0)
    model = tiny_graph().make()
    _ = model.eval()
    empty = torch.zeros(0, dtype=torch.long)
    for graph in (
        topology(),
        replace(topology(), edge_type=topology().edge_type.remainder(3)),
        replace(topology(), source=empty, destination=empty, edge_type=empty),
    ):
        assert torch.allclose(
            model(graph, features()),
            reference(model, graph, features()),
            atol=1e-6,
        )


def test_forward_carries_gradients_to_every_parameter() -> None:
    model = tiny_graph().make()
    model(topology(), features()).square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_forward_zero_pads_narrow_features_to_the_hidden_width() -> None:
    model = tiny_graph().make()
    wide = torch.cat([features(), torch.zeros(9, HIDDEN - INPUT)], dim=-1)
    assert torch.equal(model(topology(), features()), model(topology(), wide))


def test_state_dict_layout_is_stable() -> None:
    names = list(tiny_graph().make().state_dict())
    assert names[:2] == ["linears.0.weight", "linears.0.bias"]
    assert {re.sub(r"\.\d+\.", ".N.", name) for name in names} == {
        "linears.N.weight",
        "linears.N.bias",
        "gru.weight_ih",
        "gru.weight_hh",
        "gru.bias_ih",
        "gru.bias_hh",
    }
    assert len(names) == 2 * EDGE_TYPES + 4


def test_a_second_edge_initializer_redraws_weights_and_zeroes_biases() -> None:
    config = tiny_graph()
    config.init_edge = partial(nn.init.normal_, std=3.0)
    _ = torch.manual_seed(0)
    model = config.make()
    first = [linear.weight.clone() for linear in model.linears]
    assert all(1.0 < weight.std() < 6.0 for weight in first)
    state: dict[str, Tensor] = dict(model.state_dict())
    assert all(
        state[f"linears.{index}.bias"].count_nonzero() == 0
        for index in range(EDGE_TYPES)
    )
    _ = torch.manual_seed(0)
    model.reset_parameters()
    assert all(
        not torch.equal(before, linear.weight)
        for before, linear in zip(first, model.linears, strict=True)
    )
    plain = tiny_graph().make()
    assert all(0.0 < linear.weight.std() < 1.0 for linear in plain.linears)


def test_construction_builds_the_children_then_resets_the_cell_and_edge_maps() -> None:
    config = tiny_graph().finalize()
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(2)
        model = config.make()
        _ = torch.manual_seed(2)
        linears = [config.edge.make() for _ in range(EDGE_TYPES)]
        cell = config.cell.make()
        cell.reset_parameters()
        for linear in linears:
            linear.reset_parameters()
    expected = {
        **{
            f"linears.{index}.{name}": value
            for index, linear in enumerate(linears)
            for name, value in linear.state_dict().items()
        },
        **{f"gru.{name}": value for name, value in cell.state_dict().items()},
    }
    assert all(
        torch.equal(expected[name], value) for name, value in model.state_dict().items()
    )


def test_initialization_draws_match_the_recorded_state_and_generator_position() -> None:
    config = tiny_graph()
    config.init_edge = partial(nn.init.xavier_normal_, gain=2**0.5)
    with host_agnostic_numerics():
        _ = torch.manual_seed(0)
        model = config.make()
        state: dict[str, Tensor] = dict(model.state_dict())
        record = {**state, "rng": rng_fingerprint()}
    assert_tensor_golden(_CWD / "testdata" / "ggnn_init.pt", record)


def test_pooling_averages_each_graph_and_returns_zero_for_an_empty_one() -> None:
    # A one-node graph divides by exactly one, so its pool is its node state.
    graph = replace(topology(), node_counts=torch.tensor([0, 2, 3, 4]))
    values = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4)
    pooled = graph.average(values)
    assert pooled.shape == (4, 4)
    assert pooled[0].count_nonzero() == 0
    assert torch.equal(pooled[1], values[:2].mean(0))
    assert torch.equal(pooled[2], values[2:5].mean(0))
    assert torch.equal(pooled[3], values[5:].mean(0))
    single = replace(topology(), node_counts=torch.tensor([1, 8]))
    assert torch.equal(single.average(values)[0], values[0])
    stacked = graph.average(torch.ones(9, 5, 2))
    assert stacked.shape == (4, 5, 2)
    assert torch.equal(stacked[3], torch.ones(5, 2))


def test_pooling_rounds_counts_in_the_feature_dtype() -> None:
    graph = replace(topology(), node_counts=torch.tensor([257, 2, 3]))
    # BF16 first loses integer resolution at 257, so the count is cast before dividing.
    values = torch.zeros(257 + 2 + 3, 4, dtype=torch.bfloat16)
    values[[0, 257, 257 + 2]] = 1
    expected = torch.tensor(
        [[1 / 256] * 4, [1 / 2] * 4, [1 / 3] * 4],
        dtype=torch.bfloat16,
    )
    assert torch.equal(graph.average(values), expected)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_requires_grad", [False, True])
@pytest.mark.parametrize("state_requires_grad", [False, True])
def test_cell_cost_matches_the_complete_torch_invocation(
    bias: bool,
    input_requires_grad: bool,
    state_requires_grad: bool,
) -> None:
    config = RecurrentCell.Config()
    config.channels_in = 5
    config.channels_out = 7
    config.bias = bias
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(6, 5, requires_grad=input_requires_grad),
            torch.randn(6, 7, requires_grad=state_requires_grad),
        ),
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        input_requires_grad=input_requires_grad,
        state_requires_grad=state_requires_grad,
    )
    assert execution.params == 3 * 7 * (5 + 7) + (6 * 7 if bias else 0)
    assert execution.bytes_state == 0


@pytest.mark.parametrize("edge_counts", [(2, 3, 4, 5), (2, 3, 0, 0), (0, 0, 0, 0)])
def test_message_cost_keeps_unused_parameters_and_actual_edge_work(
    edge_counts: tuple[int, ...],
) -> None:
    config = tiny_graph()
    edge_type = torch.repeat_interleave(torch.arange(4), torch.tensor(edge_counts))
    source = torch.arange(sum(edge_counts)).remainder(6)
    graph = GraphBatch(
        source=source,
        destination=(source + 1).remainder(6),
        edge_type=edge_type,
        node_counts=torch.tensor([2, 4]),
    )
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(6, 5, requires_grad=True),
        run=lambda module, values: cast(Tensor, module(graph, values)),
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=edge_counts,
    )
    assert execution.params == 4 * (5 * 5 + 5) + 3 * 5 * (5 + 5) + 6 * 5
    one_step = config.copy_tree()
    one_step.num_layers = 1
    single = cost(
        one_step.finalize(),
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=edge_counts,
    )
    assert execution.params == single.params
    assert execution["flops", "matmul"].sum() == 2 * single["flops", "matmul"].sum()
    assert execution["bytes", "matmul"].sum() == 2 * single["bytes", "matmul"].sum()
    assert execution.params_active == (
        sum(count > 0 for count in edge_counts) * (5 * 5 + 5) + 3 * 5 * (5 + 5) + 6 * 5
    )


def test_message_cost_without_message_steps_has_no_active_parameters() -> None:
    config = tiny_graph()
    config.num_layers = 0
    edge_counts = (2, 3, 4, 5)
    source = torch.arange(sum(edge_counts)).remainder(6)
    graph = GraphBatch(
        source=source,
        destination=(source + 1).remainder(6),
        edge_type=torch.repeat_interleave(torch.arange(4), torch.tensor(edge_counts)),
        node_counts=torch.tensor([2, 4]),
    )
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(6, 5, requires_grad=True),
        run=lambda module, values: cast(Tensor, module(graph, values)),
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=edge_counts,
    )
    assert execution.params_active == 0


def test_message_cost_reads_a_narrower_input_width() -> None:
    config = tiny_graph().copy_tree().finalize()
    wide = cost(
        config,
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=(2, 3, 4, 5),
    )
    narrow = cost(
        config,
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=(2, 3, 4, 5),
        channels=INPUT,
    )
    assert narrow["bytes", "selection"].sum() > wide["bytes", "selection"].sum()


def test_message_cost_requires_complete_nonnegative_edge_counts() -> None:
    config = tiny_graph().finalize()
    with pytest.raises(TypeError, match="edge_counts"):
        _ = cost(config, seq_len=2, batch_size=3, dtype=torch.float32)
    for counts in ((2, 3), (2, 3, -1, 0)):
        with pytest.raises(ValueError, match="one nonnegative count per edge type"):
            _ = cost(
                config,
                seq_len=2,
                batch_size=3,
                dtype=torch.float32,
                edge_counts=counts,
            )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
