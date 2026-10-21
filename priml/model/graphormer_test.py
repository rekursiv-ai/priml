"""Check Graphormer on small synthetic graphs: bias, init draws, cost and layout."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import re

from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaFused
from priml.model.graphormer import (
    GraphAttention,
    GraphBlock,
    GraphKernel,
    Graphormer,
    feed_forward,
)
from priml.model.linear import Linear
from priml.model.special import Dropout
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_tensor_golden, rng_fingerprint


_CWD: Final = Path(__file__).resolve().parent
BATCH: Final = 2
NODES: Final = 3
FEATURES: Final = 5
EDGE_FEATURES: Final = 6
HOPS: Final = 7
MAX_HOPS: Final = 3
HEADS: Final = 2
HIDDEN: Final = 8
LAYERS: Final = 2


def _config() -> Graphormer.Config:
    config = Graphormer.Config()
    config.channels_in = FEATURES
    config.channels_hidden = HIDDEN
    config.heads = HEADS
    config.num_layers = LAYERS
    config.edge.channels_in = EDGE_FEATURES
    config.degree.channels_in = 9
    config.spatial.channels_in = 10
    config.edge_distances = 4
    config.max_hops = MAX_HOPS
    return config


def _graph() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(BATCH, NODES, FEATURES, generator=generator),
        torch.randn(BATCH, NODES + 1, NODES + 1, generator=generator),
        torch.arange(BATCH * NODES * NODES).reshape(BATCH, NODES, NODES) % 10,
        torch.arange(BATCH * NODES).reshape(BATCH, NODES) % 9,
        torch.randn(BATCH, NODES, NODES, HOPS, EDGE_FEATURES, generator=generator),
    )


def test_defaults_are_the_canonical_graphormer_encoder() -> None:
    config = Graphormer.Config().copy_tree().finalize()
    assert (config.channels_hidden, config.heads, config.num_layers) == (768, 32, 12)
    assert (config.edge_distances, config.max_hops, config.init_std) == (128, 5, 0.02)
    assert (config.degree.channels_in, config.spatial.channels_in) == (512, 512)
    assert config.degree.padding_idx == config.spatial.padding_idx == 0
    assert config.node.channels_out == config.degree.channels_out == 768
    assert config.token.channels_out == config.block.channels_in == 768
    assert config.edge.channels_out == config.virtual.channels_out == 32
    assert config.spatial.channels_out == 32
    assert config.distance.channels_in == 128 * 32 * 32
    assert isinstance(config.block.attn, GraphAttention.Config)
    assert config.block.attn.attention.num_heads == 32
    assert not config.block.prenorm


def test_forward_returns_the_graph_token_and_node_states_with_gradients() -> None:
    model = _config().make()
    _ = model.eval()
    features, bias, spatial, degrees, paths = _graph()
    graph, nodes = model(
        features,
        bias=bias,
        spatial=spatial,
        degrees=degrees,
        paths=paths,
    )
    assert graph.shape == (BATCH, HIDDEN)
    assert nodes.shape == (BATCH, NODES, HIDDEN)
    (graph.sum() + nodes.square().sum()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_forward_composes_the_encoders_with_the_graph_token_first() -> None:
    model = _config().make()
    _ = model.eval()
    features, bias, spatial, degrees, paths = _graph()
    graph, nodes = model(
        features,
        bias=bias,
        spatial=spatial,
        degrees=degrees,
        paths=paths,
    )
    states = torch.cat(
        [
            model.token.weight.expand(BATCH, -1, -1),
            model.node(features) + model.degree(degrees),
        ],
        dim=1,
    ).transpose(0, 1)
    attention_bias = model.attention_bias(bias, spatial, paths)
    for block in model.blocks:
        states = block(states, attn_mask=attention_bias)
    assert torch.equal(graph, states[0])
    assert torch.equal(nodes, states[1:].transpose(0, 1))
    assert not torch.equal(graph, nodes[:, 0])


def test_attention_bias_is_the_closed_form_of_its_three_terms() -> None:
    model = _config().make()
    with torch.no_grad():
        for table in (model.spatial, model.virtual, model.distance):
            _ = table.weight.copy_(torch.randn(table.weight.shape))
    _, bias, spatial, _, paths = _graph()
    actual = model.attention_bias(bias, spatial, paths)
    assert actual.shape == (BATCH, HEADS, NODES + 1, NODES + 1)
    expected = 2 * bias[:, None].expand(-1, HEADS, -1, -1).clone()
    virtual = model.virtual.weight.reshape(HEADS)[:, None]
    expected[:, :, 1:, 0] += virtual
    expected[:, :, 0, :] += virtual
    expected[:, :, 1:, 1:] += model.spatial(spatial).permute(0, 3, 1, 2)
    encoded = functional.linear(
        paths[:, :, :, :MAX_HOPS],
        model.edge.weight,
        model.edge.bias,
    )
    matrices = model.distance.weight.reshape(-1, HEADS, HEADS)[:MAX_HOPS]
    products = torch.einsum("bijkh,khg->bijg", encoded, matrices)
    divisor = torch.where(spatial > 1, spatial - 1, spatial.clamp(min=1)).clamp(
        max=MAX_HOPS,
    )
    expected[:, :, 1:, 1:] += (products / divisor[..., None]).permute(0, 3, 1, 2)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_attention_bias_ignores_path_positions_beyond_max_hops() -> None:
    model = _config().make()
    _, bias, spatial, _, paths = _graph()
    changed = paths.clone()
    changed[:, :, :, MAX_HOPS:] += 1.0
    assert torch.equal(
        model.attention_bias(bias, spatial, paths),
        model.attention_bias(bias, spatial, changed),
    )
    changed[:, :, :, 0] += 1.0
    assert not torch.equal(
        model.attention_bias(bias, spatial, paths),
        model.attention_bias(bias, spatial, changed),
    )


def test_state_dict_layout_is_stable() -> None:
    names = list(_config().make().state_dict())
    assert names[:9] == [
        "node.weight",
        "node.bias",
        "degree.weight",
        "token.weight",
        "edge.weight",
        "edge.bias",
        "distance.weight",
        "spatial.weight",
        "virtual.weight",
    ]
    assert {re.sub(r"\.\d+\.", ".N.", name) for name in names[9:]} == {
        "blocks.N.attn.attention.proj_qkv.weight",
        "blocks.N.attn.attention.proj_qkv.bias",
        "blocks.N.attn.attention.proj_out.weight",
        "blocks.N.attn.attention.proj_out.bias",
        "blocks.N.ffn.N.up_proj.weight",
        "blocks.N.ffn.N.up_proj.bias",
        "blocks.N.ffn.N.down_proj.weight",
        "blocks.N.ffn.N.down_proj.bias",
        "blocks.N.norm1.weight",
        "blocks.N.norm1.bias",
        "blocks.N.norm2.weight",
        "blocks.N.norm2.bias",
    }
    assert len(names) == 9 + LAYERS * 12


def test_initialization_zeroes_padding_rows_and_biases() -> None:
    _ = torch.manual_seed(0)
    model = _config().make()
    state: dict[str, Tensor] = dict(model.state_dict())
    assert state["degree.weight"][0].count_nonzero() == 0
    assert state["spatial.weight"][0].count_nonzero() == 0
    assert state["degree.weight"][1:].count_nonzero() > 0
    assert state["node.bias"].count_nonzero() == state["edge.bias"].count_nonzero() == 0
    for index in range(LAYERS):
        prefix = f"blocks.{index}."
        assert state[prefix + "attn.attention.proj_out.bias"].count_nonzero() == 0
        assert state[prefix + "ffn.0.up_proj.bias"].count_nonzero() == 0
        assert state[prefix + "ffn.0.up_proj.weight"].std() < 0.5
    assert 0.003 < state["token.weight"].std() < 0.1


def test_initialization_scale_follows_init_std() -> None:
    config = _config()
    config.init_std = 2.0
    _ = torch.manual_seed(0)
    state: dict[str, Tensor] = dict(config.make().state_dict())
    assert 1.0 < state["degree.weight"][1:].std() < 4.0
    assert 1.0 < state["blocks.0.attn.attention.proj_qkv.weight"].std() < 4.0


def test_initialization_draws_match_the_recorded_state_and_generator_position() -> None:
    with host_agnostic_numerics():
        _ = torch.manual_seed(0)
        model = _config().make()
        state: dict[str, Tensor] = dict(model.state_dict())
        record = {**state, "rng": rng_fingerprint()}
    assert_tensor_golden(_CWD / "testdata" / "graphormer_init.pt", record)


def test_attention_reset_repeats_the_constructor_draws() -> None:
    config = GraphAttention.Config()
    config.channels_in = HIDDEN
    config.attention.num_heads = HEADS
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(1)
        attention = config.make()
        initial = {
            name: value.clone() for name, value in attention.state_dict().items()
        }
        _ = torch.manual_seed(1)
        attention.reset_parameters()
        final: dict[str, Tensor] = dict(attention.state_dict())
    assert all(torch.equal(initial[name], final[name]) for name in initial)
    assert final["attention.proj_out.bias"].count_nonzero() == 0


def test_attention_config_infers_one_residual_width() -> None:
    from_out = GraphAttention.Config()
    from_out.channels_out = HIDDEN
    from_out = from_out.copy_tree().finalize()
    assert from_out.channels_in == from_out.attention.channels_in == HIDDEN
    assert from_out.dropout.channels_in == HIDDEN
    from_kernel = GraphAttention.Config()
    from_kernel.attention.channels_out = HIDDEN
    from_kernel = from_kernel.copy_tree().finalize()
    assert from_kernel.channels_in == from_kernel.dropout.channels_in == HIDDEN
    with pytest.raises(ValueError, match="must equal"):
        _ = GraphAttention.Config(channels_in=2, channels_out=HIDDEN).finalize()


def test_graph_kernel_attends_over_sequence_major_heads() -> None:
    _ = torch.manual_seed(0)
    q, k, v = (torch.randn(NODES, BATCH, HEADS, HIDDEN) for _ in range(3))
    mask = torch.randn(BATCH, HEADS, NODES, NODES)
    expected = SdpaFused.Config().make()(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
    )
    actual = GraphKernel.Config().make()(q, k, v, attn_mask=mask)
    assert torch.equal(actual, expected.transpose(0, 1))


def test_residual_dropout_acts_on_the_attention_update_only() -> None:
    config = GraphAttention.Config()
    config.channels_in = HIDDEN
    config.attention.num_heads = HEADS
    config.attention.dropout = 0.0
    config.dropout.p = 1.0
    attention = config.make()
    _ = attention.train()
    mask = torch.zeros(BATCH, HEADS, NODES, NODES)
    assert (
        attention(torch.ones(NODES, BATCH, HIDDEN), attn_mask=mask).count_nonzero() == 0
    )


def test_block_is_post_norm_with_a_two_dropout_feed_forward() -> None:
    config = GraphBlock.Config()
    config.channels_in = HIDDEN
    assert not config.prenorm
    assert isinstance(config.attn, GraphAttention.Config)
    config.attn.attention.num_heads = HEADS
    config.attn.attention.dropout = 0.0
    block = config.make()
    _ = block.eval()
    x = torch.randn(NODES, BATCH, HIDDEN)
    mask = torch.zeros(BATCH, HEADS, NODES, NODES)
    attended = block.norm1(x + block.attn(x, attn_mask=mask))
    expected = block.norm2(attended + block.ffn(attended))
    assert torch.allclose(block(x, attn_mask=mask), expected, atol=1e-6)
    elements = feed_forward().elements
    assert isinstance(elements, list)
    assert isinstance(elements[1], Dropout.Config)
    assert elements[1].p == 0.1


def test_finalize_leaves_a_foreign_attention_slot_alone() -> None:
    config = _config()
    foreign = Attention.Config(proj_qkv=Linear.Config(bias=True))
    foreign.num_heads = 4
    config.block.attn = foreign
    attn = config.copy_tree().finalize().block.attn
    assert isinstance(attn, Attention.Config)
    assert attn.num_heads == 4


def test_cost_matches_torch_execution_including_hop_products() -> None:
    def run(module: nn.Module, values: tuple[Tensor, ...]) -> Tensor:
        features, bias, spatial, degrees, paths = values
        graph, nodes = cast(
            tuple[Tensor, Tensor],
            module(
                features,
                bias=bias,
                spatial=spatial,
                degrees=degrees,
                paths=paths,
            ),
        )
        return torch.cat([graph[:, None], nodes], dim=1)

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            _config(),
            build_input=lambda: tuple(
                value.requires_grad_(value.is_floating_point()) for value in _graph()
            ),
            run=run,
            seq_len=NODES,
            batch_size=BATCH,
            hops=HOPS,
            dtype=torch.float32,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
