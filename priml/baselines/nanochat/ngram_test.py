"""Tests for NanoChat causal n-gram embeddings and fused value mixing."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat.ngram import (
    HashedNgramTables,
    NgramEmbedding,
    clear_marked_sinks,
    ngram_mix,
)
from priml.model.embedding import Embedding
from priml.testing.cost import assert_cost_matches_torch


def test_ngram_embedding_zeros_incomplete_prefix_and_receives_gradients() -> None:
    config = NgramEmbedding.Config()
    config.channels_in = 13
    config.channels_out = 2
    config.multipliers = (1, 3, 5)
    config.scale = 0.25
    built = config.make()
    assert isinstance(built.inner, Embedding)
    with torch.no_grad():
        built.inner.weight.copy_(torch.arange(26).reshape(13, 2))

    tokens = torch.tensor([[1, 2, 3, 4]])
    output = built(tokens)
    expected = torch.zeros(1, 4, 2)
    for position in range(2, 4):
        bucket = (
            sum(
                int(tokens[0, position - lag]) * multiplier
                for lag, multiplier in enumerate(config.multipliers)
            )
            % config.channels_in
        )
        expected[0, position] = torch.tensor([2 * bucket, 2 * bucket + 1]) * 0.25
    assert torch.equal(output, expected)
    output.sum().backward()
    assert isinstance(built.inner.weight.grad, Tensor)
    assert torch.count_nonzero(built.inner.weight.grad) > 0


def test_hash_indices_preserve_prefix_and_coefficient_order() -> None:
    config = HashedNgramTables.Config()
    config.channels_out = 4
    config.num_embeddings = 17
    config.hash_multipliers = ((3, 5, 7), (11, 13, 17))
    table = config.make()
    tokens = torch.tensor([[1, 2, 3, 4]])
    indices = table.indices(tokens)
    first = ((1 * 3) ^ (1 * 5) ^ (1 * 7)) % 17
    second = ((2 * 3) ^ (1 * 5) ^ (2 * 7)) % 17
    assert indices[0][0, :2].tolist() == [first, second]
    assert all(
        torch.equal(full[:, :3], prefix)
        for full, prefix in zip(indices, table.indices(tokens[:, :3]), strict=True)
    )


def test_hashed_tables_cost_is_one_gather_per_hash_and_matches_torch() -> None:
    """Two hashes gather two half-width rows; the integer hash is elementwise."""
    config = HashedNgramTables.Config()
    config.channels_out = 4
    config.num_embeddings = 17
    config.hash_multipliers = ((3, 5, 7), (11, 13, 17))
    priced = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 17, (2, 5)),
        num_tokens=10,
    )
    assert priced.params == 2 * 17 * 2
    assert priced.primal.bytes.selection == 4
    assert priced.adjoint.flops.selection == 4
    # Three multiplies, two XORs, one modulo per hash.
    assert priced.primal.flops.elementwise == 2 * 2 * 3


def test_table_initialization_transform_preserves_rng_draws() -> None:
    config = HashedNgramTables.Config()
    config.channels_out = 4
    config.num_embeddings = 7
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = config.make()
        expected_rng = torch.get_rng_state()
        config.init_after = nn.init.zeros_
        torch.manual_seed(42)
        candidate = config.make()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.count_nonzero(reference.tables[0].weight) > 0
        assert torch.count_nonzero(candidate.tables[0].weight) == 0


def test_gradient_buffers_follow_placement_without_narrowing() -> None:
    """Eager placement moves non-persistent sinks without quantizing their values."""
    tables = HashedNgramTables.Config(channels_out=4, num_embeddings=7).make()
    tables.prepare_gradient_sinks(dirty_bitmaps=True)
    tables.gradient_sinks[0].fill_(1.001)
    expected = tables.gradient_sinks[0].clone()
    tables.to(dtype=torch.bfloat16)
    assert tables.tables[0].weight.dtype == torch.bfloat16
    assert tables.gradient_sinks[0].dtype == torch.float32
    assert torch.equal(tables.gradient_sinks[0], expected)
    tables.to("meta")
    assert tables.gradient_sinks[0].device.type == "meta"
    assert tables.gradient_bitmaps[0].device.type == "meta"
    tables.to_empty(device="cpu")
    tables.prepare_gradient_sinks(dirty_bitmaps=True)
    assert tables.gradient_sinks[0].device.type == "cpu"
    assert tables.gradient_sinks[0].dtype == torch.float32
    assert tables.gradient_bitmaps[0].dtype == torch.uint8
    assert not torch.count_nonzero(tables.gradient_sinks[0])


def test_fused_mix_accumulates_fp32_sinks_without_weight_gradients() -> None:
    torch.manual_seed(17)
    values = torch.randn(1, 3, 2, 2, requires_grad=True)
    gate = torch.randn(1, 3, 2, requires_grad=True)
    weights = [torch.randn(5, 2, requires_grad=True) for _ in range(2)]
    indices = [torch.tensor([[1, 1, 3]]) for _ in weights]
    sinks = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
    bitmaps = [torch.zeros(weight.shape[0], dtype=torch.uint8) for weight in weights]

    output = ngram_mix(values, [gate], weights, indices, sinks, bitmaps)
    output.sum().backward()
    assert values.grad is not None
    assert gate.grad is not None
    assert all(weight.grad is None for weight in weights)
    assert all(sink.dtype == torch.float32 for sink in sinks)
    assert all(torch.count_nonzero(sink) > 0 for sink in sinks)
    assert [bitmap.nonzero().flatten().tolist() for bitmap in bitmaps] == [
        [1, 3],
        [1, 3],
    ]

    before = [weight.detach().clone() for weight in weights]
    with torch.no_grad():
        for weight, sink in zip(weights, sinks, strict=True):
            weight.add_(sink, alpha=-0.1)
    assert all(
        not torch.equal(weight, original)
        for weight, original in zip(weights, before, strict=True)
    )
    clear_marked_sinks(sinks, bitmaps)
    assert all(torch.count_nonzero(sink) == 0 for sink in sinks)
    assert all(torch.count_nonzero(bitmap) == 0 for bitmap in bitmaps)


@pytest.mark.gpu_torch_cuda
@pytest.mark.gpu_triton
@pytest.mark.parametrize("sources", [1, 2])
def test_cuda_fused_mix_matches_autograd_and_marks_rows(sources: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device.")
    torch.manual_seed(31)
    values = torch.randn(
        1,
        17,
        2,
        2,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gates = [
        torch.randn(1, 17, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(sources)
    ]
    weights = [
        torch.randn(32, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2 * sources)
    ]
    indices = [torch.arange(17, device="cuda")[None] % 5 for _ in weights]
    sinks = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
    bitmaps = [
        torch.zeros(weight.shape[0], device="cuda", dtype=torch.uint8)
        for weight in weights
    ]
    upstream = torch.randn_like(values)

    reference = values.float()
    expected_sinks = [
        torch.zeros_like(weight, dtype=torch.float32) for weight in weights
    ]
    half = weights[0].shape[1]
    for source, gate in enumerate(gates):
        rows = (
            torch.cat(
                [
                    weights[2 * source + part][indices[2 * source + part]]
                    for part in (0, 1)
                ],
                dim=-1,
            )
            .view_as(values)
            .float()
        )
        sigmoid = gate.float().sigmoid()
        reference = reference + 2 * sigmoid.unsqueeze(-1) * rows
        contribution = (upstream.float() * (2 * sigmoid.detach()).unsqueeze(-1)).view(
            -1,
            values.shape[-2] * values.shape[-1],
        )
        for part in (0, 1):
            table = 2 * source + part
            expected_sinks[table].index_add_(
                0,
                indices[table].reshape(-1),
                contribution[:, part * half : (part + 1) * half],
            )
    reference = reference.to(values.dtype)
    expected = torch.autograd.grad(reference, [values, *gates], upstream)

    actual = ngram_mix(values, gates, weights, indices, sinks, bitmaps)
    torch.testing.assert_close(actual, reference)
    actual.backward(upstream)
    torch.testing.assert_close(values.grad, expected[0])
    for gate, gradient in zip(gates, expected[1 : 1 + sources], strict=True):
        torch.testing.assert_close(gate.grad, gradient)
    for weight, sink, gradient in zip(weights, sinks, expected_sinks, strict=True):
        assert weight.grad is None
        torch.testing.assert_close(sink, gradient)
    assert all(
        bitmap.nonzero().flatten().tolist() == list(range(5)) for bitmap in bitmaps
    )
    clear_marked_sinks(sinks, bitmaps)
    assert all(torch.count_nonzero(sink) == 0 for sink in sinks)
    assert all(torch.count_nonzero(bitmap) == 0 for bitmap in bitmaps)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
