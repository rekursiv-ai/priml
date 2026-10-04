"""Tests for NanoChat causal n-gram embeddings and fused value mixing."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat import ngram
from priml.baselines.nanochat.ngram import (
    HashedNgramTables,
    NgramEmbedding,
    clear_marked_sinks,
    ngram_mix,
)
from priml.model.embedding import Embedding
from priml.testing.cost import assert_cost_matches_torch


class _FakeKernel:
    def __init__(self) -> None:
        self.grid: tuple[int, ...] = ()
        self.args: tuple[object, ...] = ()
        self.kwargs: dict[str, object] = {}

    def __getitem__(self, grid: tuple[int, ...]) -> _FakeKernel:
        self.grid = grid
        return self

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def _fake_kernel() -> _FakeKernel:
    return _FakeKernel()


def _kernel_buffers(kernel: _FakeKernel) -> list[Tensor]:
    value = cast(list[Tensor] | tuple[Tensor, ...], kernel.kwargs["buffers"])
    assert isinstance(value, (list, tuple))
    return list(value)


def _kernel_dimensions(kernel: _FakeKernel) -> tuple[int, ...]:
    return cast(tuple[int, ...], kernel.kwargs["dimensions"])


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

    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]])
    output = built(tokens)
    expected = torch.zeros(3, 4, 2)
    for position in range(2, 4):
        for row in range(3):
            bucket = (
                sum(
                    int(tokens[row, position - lag]) * multiplier
                    for lag, multiplier in enumerate(config.multipliers)
                )
                % config.channels_in
            )
            expected[row, position] = torch.tensor([2 * bucket, 2 * bucket + 1]) * 0.25
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
    costed = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 17, (2, 5)),
        seq_len=10,
        batch_size=1,
        dtype=None,
    )
    assert costed.params == 2 * 17 * 2
    rows = 10
    hashes = len(config.hash_multipliers)
    width_per_hash = config.channels_out // hashes
    # Each concrete invocation reads every hash index and shifted token ID.
    assert costed["bytes", "primal", "selection", torch.int64] == 8 * (
        hashes * rows + 2 * rows * (3 - 1)
    )
    assert costed["bytes", "primal", "selection", torch.float32] == 4 * (
        hashes * 2 * rows * width_per_hash + 2 * rows * config.channels_out
    )
    assert costed["flops", "adjoint", "selection"].sum() == (
        hashes * rows * width_per_hash
    )
    # Three multiplies, two XORs, one modulo per hash and row.
    assert costed["flops", "primal", "elementwise"].sum() == (2 * 3 * hashes * rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_hash_traffic_counts_integer_operands(dtype: torch.dtype) -> None:
    config = HashedNgramTables.Config()
    config.channels_out = 4
    config.hash_multipliers = ((3, 5, 7), (11, 13, 17))
    costed = config.finalize().cost(seq_len=1, batch_size=1, dtype=dtype)
    # The hash is integer arithmetic on token ids: int64 whatever the batch's dtype.
    assert costed["bytes", "primal", "elementwise", torch.int64] == 8 * 2 * (
        2 * 3 + 3 * 2 + 2
    )
    assert costed["bytes", "primal", "elementwise", dtype] == 0
    assert costed["bytes", "adjoint", "elementwise"].sum() == 0


def test_hashed_tables_cost_counts_shift_and_output_copies() -> None:
    config = HashedNgramTables.Config()
    config.channels_out = 4
    config.hash_multipliers = ((3, 5, 7), (11, 13, 17))
    costed = config.finalize().cost(seq_len=1, batch_size=1, dtype=torch.bfloat16)
    assert costed["bytes", "primal", "selection", torch.int64] == 8 * (2 + 2 * 2)
    assert costed["bytes", "primal", "selection", torch.bfloat16] == 2 * (
        2 * 2 * 2 + 2 * 4
    )


def test_ngram_embedding_cost_matches_torch_with_hash_scale_and_context() -> None:
    config = NgramEmbedding.Config()
    config.channels_in = 13
    config.channels_out = 4
    config.multipliers = (1, 3, 5)
    config.scale = 0.25
    context = NgramEmbedding.Config()
    context.channels_in = 7
    config.contexts = {"context": context}
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 7, (2, 4)),
        seq_len=4,
        batch_size=2,
        dtype=None,
    )


def test_ngram_cost_counts_padding_and_masked_prefix_copy() -> None:
    config = NgramEmbedding.Config()
    config.channels_in = 13
    config.channels_out = 2
    config.multipliers = (1, 3, 5)
    costed = config.finalize().cost(
        seq_len=8,
        batch_size=1,
        dtype=torch.bfloat16,
    )
    rows = 8
    order = len(config.multipliers)
    shifted = sum(2 * rows + lag for lag in range(1, order))
    prefix = min(order - 1, rows)
    output = 2 * config.channels_out * rows + prefix * config.channels_out
    # The gathered row moves at the table's dtype (torch's default, fp32); the
    # padded output is the batch's activation, bf16; the lookup index and the
    # shifted ids are int64.
    assert costed["bytes", "primal", "selection", torch.int64] == 8 * (rows + shifted)
    assert costed["bytes", "primal", "selection", torch.float32] == 4 * (
        2 * rows * config.channels_out
    )
    assert costed["bytes", "primal", "selection", torch.bfloat16] == (2 * output)


def test_ngram_cost_includes_context_table_and_scale() -> None:
    config = NgramEmbedding.Config()
    config.channels_in = 13
    config.channels_out = 2
    config.scale = 0.25
    context = NgramEmbedding.Config()
    context.channels_in = 7
    config.contexts = {"previous": context}
    costed = config.finalize().cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    rows = 4
    width = config.channels_out
    itemsize = torch.bfloat16.itemsize
    assert costed.params == (13 + 7) * 2
    assert costed["flops", "primal", "elementwise"].sum() == 2 * width * rows
    assert costed["bytes", "primal", "elementwise"].sum() == (
        itemsize * width * rows * (2 + 3)
    )


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
        with torch.no_grad():
            candidate.tables[0].weight.fill_(1)
        candidate.reset_parameters()
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
    values = torch.randn(3, 4, 2, 5, requires_grad=True)
    gate = torch.randn(3, 4, 2, requires_grad=True)
    weights = [torch.randn(6, 5, requires_grad=True) for _ in range(2)]
    indices = [torch.tensor([[1, 1, 3, 2]]).expand(3, -1).contiguous() for _ in weights]
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
        [1, 2, 3],
        [1, 2, 3],
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
        3,
        17,
        4,
        2,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gates = [
        torch.randn(3, 17, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(sources)
    ]
    weights = [
        torch.randn(32, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2 * sources)
    ]
    indices = [
        (torch.arange(17, device="cuda")[None] % 5).expand(3, -1).contiguous()
        for _ in weights
    ]
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


def test_ngram_configs_validate_hash_geometry() -> None:
    with pytest.raises(ValueError, match="divide"):
        HashedNgramTables.Config(channels_out=3, hash_multipliers=((1,), (1,))).make()
    with pytest.raises(ValueError, match="same n-gram"):
        HashedNgramTables.Config(channels_out=4, hash_multipliers=((1,), (1, 2))).make()


def test_ngram_mix_rejects_bad_inputs() -> None:
    value = torch.zeros(2, 3, 4, 6)
    gate = torch.zeros(2, 3, 4)
    weight = torch.zeros(5, 12)
    index = torch.zeros(2, 3, dtype=torch.long)
    sink = torch.zeros_like(weight)
    with pytest.raises(ValueError, match="1 <= len"):
        ngram_mix(value, [], [], [], [], [])
    with pytest.raises(ValueError, match="ndim"):
        ngram_mix(
            torch.zeros(2, 3, 4),
            [gate],
            [weight, weight],
            [index, index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match="dtype"):
        ngram_mix(
            value,
            [gate],
            [weight, weight],
            [index, index],
            [sink.half(), sink],
            [],
        )
    with pytest.raises(ValueError, match="shape"):
        ngram_mix(
            value,
            [gate],
            [weight, weight],
            [index, index],
            [torch.zeros(4, 2), sink],
            [],
        )
    with pytest.raises(ValueError, match=r"index\.numel"):
        ngram_mix(
            value,
            [gate],
            [weight, weight],
            [torch.zeros(2, 4, dtype=torch.long), index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match=r"gate\.shape"):
        ngram_mix(
            value,
            [torch.zeros(2, 3, 5)],
            [weight, weight],
            [index, index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match="len\\(weights\\)"):
        ngram_mix(value, [gate], [weight], [index], [sink], [])
    with pytest.raises(ValueError, match="contiguous"):
        ngram_mix(
            value.transpose(-1, -2),
            [gate],
            [weight, weight],
            [index, index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match=r"index\.dtype"):
        ngram_mix(
            value,
            [gate],
            [weight, weight],
            [index.to(torch.int32), index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match=r"w\.shape"):
        ngram_mix(
            value,
            [gate],
            [torch.zeros(3, 12), weight],
            [index, index],
            [sink, sink],
            [],
        )
    noncontiguous_sink = torch.zeros(12, 5).transpose(0, 1)
    with pytest.raises(ValueError, match=r"s\.is_contiguous"):
        ngram_mix(
            value,
            [gate],
            [weight, weight],
            [index, index],
            [noncontiguous_sink, sink],
            [],
        )
    with pytest.raises(ValueError, match=r"w\.dtype"):
        ngram_mix(
            value.half(),
            [gate],
            [weight, weight],
            [index, index],
            [sink, sink],
            [],
        )
    with pytest.raises(ValueError, match=r"index\.is_contiguous"):
        ngram_mix(value, [gate], [weight, weight], [index.t(), index], [sink, sink], [])
    with pytest.raises(ValueError, match=r"gate\.numel"):
        ngram_mix(
            value,
            [torch.zeros(2, 6, 4)],
            [weight, weight],
            [index, index],
            [sink, sink],
            [],
        )


def test_cpu_backward_reference_and_empty_sink_clear() -> None:
    values = torch.randn(2, 3, 4, 6)
    gate = torch.randn(2, 3, 4)
    weights = [torch.randn(13, 12) for _ in range(2)]
    indices = [torch.tensor([[1, 2, 3], [4, 5, 6]]) for _ in weights]
    sinks = [torch.zeros_like(weight) for weight in weights]
    gradients = ngram._mix_backward_reference(values, [gate], weights, indices, sinks)
    assert gradients[0].shape == gate.shape
    assert all(torch.count_nonzero(sink) > 0 for sink in sinks)
    clear_marked_sinks([], [])


def test_ngram_cuda_dispatch_pins_kernel_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    values = torch.randn(2, 17, 3, 8)
    gates = [torch.randn(2, 17, 3) for _ in range(2)]
    weights = [torch.randn(16, 12) for _ in range(4)]
    indices = [torch.arange(34).reshape(2, 17).remainder(16) for _ in weights]
    sinks = [torch.zeros_like(weight) for weight in weights]
    bitmaps = [torch.zeros(16, dtype=torch.uint8) for _ in weights]
    forward_kernel = _fake_kernel()
    backward_kernel = _fake_kernel()
    clear_kernel = _fake_kernel()
    monkeypatch.setattr(ngram, "_compiled_ngram_forward", lambda: forward_kernel)
    monkeypatch.setattr(ngram, "_compiled_ngram_backward", lambda: backward_kernel)
    monkeypatch.setattr(ngram, "_compiled_sink_clear", lambda: clear_kernel)

    ngram._mix_forward_cuda(values, gates, weights, indices)
    assert forward_kernel.grid == (3,)
    assert isinstance(forward_kernel.grid[0], int)
    assert len(forward_kernel.args) == 0
    forward_buffers = _kernel_buffers(forward_kernel)
    assert len(forward_buffers) == 12
    assert forward_buffers[0] is values
    assert all(forward_buffers[i] is gates[i - 2] for i in (2, 3))
    assert all(forward_buffers[i] is weights[i - 4] for i in range(4, 8))
    assert all(forward_buffers[i] is indices[i - 8] for i in range(8, 12))
    assert all(isinstance(value, int) for value in _kernel_dimensions(forward_kernel))
    assert {
        key: forward_kernel.kwargs[key]
        for key in (
            "n_rows",
            "n_head",
            "dimensions",
            "block",
            "num_warps",
            "num_stages",
        )
    } == {
        "n_rows": 34,
        "n_head": 3,
        "dimensions": (12, 4, 6, 8, 2),
        "block": 16,
        "num_warps": 4,
        "num_stages": 1,
    }
    output = forward_buffers[1]
    assert output.shape == values.shape
    assert output.dtype == values.dtype

    ngram._mix_backward_cuda(
        values,
        gates,
        weights,
        indices,
        sinks,
        bitmaps=bitmaps,
    )
    assert backward_kernel.grid == (3,)
    backward_buffers = _kernel_buffers(backward_kernel)
    assert len(backward_buffers) == 21
    assert backward_buffers[0] is values
    assert all(backward_buffers[i] is gates[i - 1] for i in (1, 2))
    assert all(backward_buffers[i] is weights[i - 3] for i in range(3, 7))
    assert all(backward_buffers[i] is indices[i - 7] for i in range(7, 11))
    assert all(backward_buffers[i].shape == gates[i - 11].shape for i in (11, 12))
    assert backward_buffers[11] is not backward_buffers[12]
    assert all(backward_buffers[i] is sinks[i - 13] for i in range(13, 17))
    assert all(backward_buffers[i] is bitmaps[i - 17] for i in range(17, 21))
    assert all(isinstance(value, int) for value in _kernel_dimensions(backward_kernel))
    assert {
        key: backward_kernel.kwargs[key]
        for key in (
            "mark",
            "n_rows",
            "n_head",
            "dimensions",
            "block",
            "num_warps",
            "num_stages",
        )
    } == {
        "mark": True,
        "n_rows": 34,
        "n_head": 3,
        "dimensions": (12, 4, 6, 8, 2),
        "block": 16,
        "num_warps": 4,
        "num_stages": 1,
    }

    ngram._mix_backward_cuda(values, gates[:1], weights[:2], indices[:2], sinks[:2])
    assert backward_kernel.kwargs["mark"] is False
    backward_buffers = _kernel_buffers(backward_kernel)
    assert all(backward_buffers[i] is sinks[i - 13] for i in (13, 14))
    assert all(backward_buffers[i] is sinks[i - 17] for i in (17, 18))

    ngram._clear_marked_sinks_cuda(sinks, bitmaps)
    assert clear_kernel.grid == (2,)
    assert isinstance(clear_kernel.grid[0], int)
    clear_buffers = _kernel_buffers(clear_kernel)
    assert clear_buffers[0] is sinks[3]
    assert clear_buffers[1] is bitmaps[3]
    assert {
        key: clear_kernel.kwargs[key]
        for key in ("n_cols", "block", "rows_per_program", "num_warps")
    } == {
        "n_cols": 12,
        "block": 16,
        "rows_per_program": 8,
        "num_warps": 1,
    }
    with pytest.raises(ValueError, match="shorter than argument"):
        ngram._clear_marked_sinks_cuda(sinks[:2], bitmaps[:1])

    with pytest.raises(ValueError, match="7 rows is not divisible by 8") as exc_info:
        ngram._clear_marked_sinks_cuda(
            [torch.zeros(7, 3)],
            [torch.zeros(7, dtype=torch.uint8)],
        )
    assert str(exc_info.value) == (
        "7 rows is not divisible by 8; the clear omits a bounds mask and would "
        "read past the table"
    )


def test_prepare_gradient_sinks_allocates_exact_shapes_and_dtypes() -> None:
    tables = HashedNgramTables.Config(
        channels_out=4,
        num_embeddings=7,
        hash_multipliers=((3, 5), (11, 13)),
    ).make()
    tables.prepare_gradient_sinks()
    assert [sink.shape for sink in tables.gradient_sinks] == [(7, 2), (7, 2)]
    assert all(sink.dtype == torch.float32 for sink in tables.gradient_sinks)
    assert all(
        sink.device == table.weight.device
        for sink, table in zip(tables.gradient_sinks, tables.tables, strict=True)
    )
    assert tables.gradient_bitmaps == []

    tables.prepare_gradient_sinks(dirty_bitmaps=True)
    assert [bitmap.shape for bitmap in tables.gradient_bitmaps] == [(7,), (7,)]
    assert all(bitmap.dtype == torch.uint8 for bitmap in tables.gradient_bitmaps)
    assert all(
        bitmap.device == table.weight.device
        for bitmap, table in zip(tables.gradient_bitmaps, tables.tables, strict=True)
    )
    tables.to("meta")
    tables.prepare_gradient_sinks(dirty_bitmaps=True)
    assert all(sink.device.type == "meta" for sink in tables.gradient_sinks)
    assert all(bitmap.device.type == "meta" for bitmap in tables.gradient_bitmaps)


def test_pad_ngram_sources_repeats_first_source_to_requested_length() -> None:
    first = torch.tensor([2, 3])
    second = torch.tensor([5, 7])
    padded = ngram._pad_ngram_sources([first, second], 4)
    assert len(padded) == 4
    assert padded[0] is first
    assert padded[1] is second
    assert padded[2] is first
    assert padded[3] is first
    padded = ngram._pad_ngram_sources([first], 2)
    assert len(padded) == 2
    assert all(value is first for value in padded)


def test_ngram_cuda_host_dispatch_accepts_cpu_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernels are faked, but the host dispatch still sizes its grid with
    # ``triton.cdiv``; Triton ships Linux wheels only.
    pytest.importorskip("triton")
    values = torch.randn(2, 3, 4, 6)
    gate = torch.randn(2, 3, 4)
    # _clear_marked_sinks_cuda needs rows divisible by its 8-row program.
    weights = [torch.randn(16, 12) for _ in range(2)]
    indices = [torch.tensor([[1, 2, 3], [4, 5, 6]]) for _ in weights]
    sinks = [torch.zeros_like(weight) for weight in weights]
    bitmaps = [torch.zeros(16, dtype=torch.uint8) for _ in weights]
    monkeypatch.setattr(ngram, "_compiled_ngram_forward", _fake_kernel)
    monkeypatch.setattr(ngram, "_compiled_ngram_backward", _fake_kernel)
    monkeypatch.setattr(ngram, "_compiled_sink_clear", _fake_kernel)
    assert (
        ngram._mix_forward_cuda(values, [gate], weights, indices).shape == values.shape
    )
    gradients = ngram._mix_backward_cuda(
        values,
        [gate],
        weights,
        indices,
        sinks,
        bitmaps=bitmaps,
    )
    assert gradients[0].shape == gate.shape
    ngram._clear_marked_sinks_cuda(sinks, bitmaps)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
