"""Tests for embedding module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

import multiprocessing

from torch import Tensor

import pytest
import torch

from priml.cost import Cost
from priml.model import embedding
from priml.model.embedding import Embedding, MultiHotEmbedding, _power_of_two
from priml.model.init import normal
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


_CWD: Final = Path(__file__).resolve().parent


class _FakeKernel:
    def __init__(self) -> None:
        self.grid: tuple[int, ...] | None = None
        self.args: tuple[object, ...] | None = None
        self.kwargs: dict[str, object] | None = None

    def __getitem__(self, grid: tuple[int, ...]) -> _FakeKernel:
        self.grid = grid
        return self

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeBackwardKernels:
    def __init__(self) -> None:
        self.accumulate = _FakeKernel()
        self.decode = _FakeKernel()


class _BackwardDispatch:
    def __init__(self) -> None:
        self.triton_calls: list[tuple[Tensor, Tensor]] = []
        self.torch_calls: list[tuple[Tensor, Tensor]] = []

    def triton(self, rows: Tensor, gradient: Tensor) -> Tensor:
        self.triton_calls.append((rows, gradient))
        return gradient

    def torch(self, rows: Tensor, gradient: Tensor) -> Tensor:
        self.torch_calls.append((rows, gradient))
        return gradient


class _FakeProperties:
    multi_processor_count = 2


def _fake_backward_kernels() -> _FakeBackwardKernels:
    return _FakeBackwardKernels()


def _fake_device_properties(device: object) -> _FakeProperties:
    del device
    return _FakeProperties()


def test_embedding_config_pprint() -> None:
    config = Embedding.Config(2, 4)
    assert_pprint_golden(
        test_file=__file__,
        name="embedding",
        config=config,
    )


def test_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="embedding",
        build_module=Embedding.Config(2, 4).make,
        build_input=lambda: torch.tensor([[0, 1, 0], [1, 0, 1]]),
        seed=0,
    )


def test_embedding():
    m = Embedding.Config(1000, 64).make()
    ids = torch.randint(0, 1000, (2, 8))
    assert m(ids).shape == (2, 8, 64)


def test_embedding_reset():
    m = Embedding.Config(1000, 64).make()
    m.reset_parameters()


def test_embedding_padding_idx():
    m = Embedding.Config(1000, 64, padding_idx=0).make()
    assert m(torch.zeros(1, dtype=torch.long)).abs().sum() == 0


def test_embedding_config_and_reset_dtype() -> None:
    config = Embedding.Config(8, 4, dtype=torch.float64, padding_idx=2)
    model = config.make()
    assert model.weight.dtype == torch.float64
    assert torch.equal(model.weight[2], torch.zeros(4, dtype=torch.float64))
    model.reset_parameters()
    assert torch.equal(model.weight[2], torch.zeros(4, dtype=torch.float64))


def test_multi_hot_torch_paths_and_kernel_validation() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    assert torch.equal(embedding(rows), embedding.forward_torch(rows))
    with pytest.raises(TypeError, match="stores bf16"):
        embedding.float().forward_triton(rows)
    with pytest.raises(ValueError, match="8 fields"):
        embedding.backward_triton(
            rows,
            torch.ones(2, config.channels_concat).bfloat16(),
        )
    wide = _eight_field_layout()
    wide.channels_out = 8
    invalid = wide.make()
    packed = _draw_packed(wide, batch=2)
    with pytest.raises(ValueError, match="2\\^k >= 16"):
        invalid.backward_triton(packed, torch.ones(2, wide.channels_concat).bfloat16())


def test_power_of_two_and_non_cuda_backward_dispatch() -> None:
    assert [_power_of_two(value) for value in (2, 3, 8, 9)] == [2, 4, 8, 16]
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    gradient = torch.ones(2, config.channels_concat).bfloat16()
    assert torch.equal(
        embedding.backward(rows, gradient),
        embedding.backward_torch(rows, gradient),
    )


def test_the_table_realizes_the_spread_it_was_asked_for():
    """A table must not be drawn narrower than its own initializer states.

    Every initializer here divides by ``sqrt(depth + 1)`` and DEFAULTS that
    depth to 1, so a ``reset_parameters`` that simply omits it draws at 0.707
    of the request -- a real change to the model, and one no shape, name, or
    dtype assertion can see. ``depth`` therefore has to be forwarded, exactly
    as ``Linear`` and ``Conv`` forward theirs.
    """
    torch.manual_seed(0)
    m = Embedding.Config(
        4096,
        256,
        init_weight=partial(normal, std=0.5),
    ).make()
    assert abs(float(m.weight.detach().std()) / 0.5 - 1.0) < 0.02


def test_a_depth_scales_the_table_down():
    """The field is not decorative: a stated depth still scales.

    Nothing in this repo asks a lookup table for depth scaling -- a table has
    no residual branch -- but the field exists so the default is a CHOICE
    rather than an omission, and a choice has to be honored to be one.
    """
    torch.manual_seed(0)
    flat = Embedding.Config(
        4096,
        256,
        init_weight=partial(normal, std=0.5),
    ).make()
    torch.manual_seed(0)
    scaled = Embedding.Config(
        4096,
        256,
        depth_index=((3, 4),),
        init_weight=partial(normal, std=0.5),
    ).make()
    assert torch.allclose(scaled.weight.detach(), flat.weight.detach() / 2.0)


def test_embedding_cost_is_a_gather() -> None:
    """A lookup gathers one row; the adjoint scatter-adds its four gradients."""
    config = Embedding.Config(8, 4)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 8, (3,)),
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    f32, i64 = torch.float32, torch.int64
    assert analytical == Cost(
        cells={
            ("flops", "adjoint", "selection", f32): 3 * 4,
            ("bytes", "primal", "selection", i64): 3 * 8,
            ("bytes", "primal", "selection", f32): 4 * 2 * 3 * 4,
            ("bytes", "adjoint", "selection", i64): 3 * 8,
            ("bytes", "adjoint", "selection", f32): 4 * (3 * 3 * 4 + 32),
        },
        params=32,
        params_active=4,
    )


def test_a_multi_hot_embedding_is_as_wide_as_its_layout() -> None:
    config = _small_layout()
    assert config.observation_size == 3 * 4 + 2
    assert config.channels_concat == 3 * 2 + 2


def test_multi_hot_embedding_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="multi_hot_embedding",
        config=_small_layout(),
    )


def test_multi_hot_embedding_bfb() -> None:
    config = _small_layout()
    config.dtype = torch.float32
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="multi_hot_embedding",
        build_module=config.make,
        build_input=lambda: _draw_packed(config),
        seed=0,
    )


def test_a_multi_hot_embedding_sums_each_cells_rows_then_appends_scalars() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    features = embedding(rows)
    assert features.shape == (2, config.channels_concat)
    assert features.dtype == torch.bfloat16
    ids = rows[:, :12].reshape(2, 3, 4).long() + torch.tensor(config.offsets)
    for row in range(2):
        for cell in range(3):
            total = torch.zeros(2)
            for field in range(4):
                total = total + embedding.weight[ids[row, cell, field]].float()
            assert torch.equal(features[row, 2 * cell : 2 * cell + 2], total.bfloat16())
    assert torch.equal(features[:, 6:], rows[:, 12:].bfloat16())


def test_a_multi_hot_backward_is_fixed_point_rounded_to_the_table_dtype() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    grad = torch.randn(2, config.channels_concat).bfloat16()
    result = embedding.backward(rows, grad)
    assert result.shape == (8, 2)
    assert result.dtype == torch.bfloat16
    # Every value is a multiple of 2^-24 rounded once to bf16, and the scalar
    # columns contribute nothing.
    zero = embedding.backward(rows, grad * 0)
    assert torch.equal(zero, torch.zeros(8, 2).bfloat16())
    ids = rows[:, :12].reshape(-1, 4).long() + torch.tensor(config.offsets)
    fixed = (grad[:, :6].float().reshape(-1, 2) * 2.0**24).round().to(torch.int64)
    total = torch.zeros(8, 2, dtype=torch.int64)
    for field in range(4):
        total.index_add_(0, ids[:, field], fixed)
    assert torch.equal(result, (total.double() * 2.0**-24).float().bfloat16())


def test_autograd_reaches_the_table_through_the_fixed_point_backward() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=6).reshape(2, 3, -1)
    grad = torch.randn(2, 3, config.channels_concat).bfloat16()
    embedding(rows).backward(grad)
    assert embedding.weight.grad is not None
    assert torch.equal(embedding.weight.grad, embedding.backward(rows, grad))
    # The ids carry no gradient, even when asked for one.
    rows.requires_grad_()
    embedding(rows).backward(grad)
    assert rows.grad is None


@pytest.mark.gpu_triton
def test_the_triton_backward_matches_the_int64_reference() -> None:
    """Every digit plane, int64 wrap-around, ragged tiles, fields sharing rows."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _eight_field_layout()
    embedding = config.make().cuda()
    generator = torch.Generator().manual_seed(1)
    rows = _draw_packed(config, batch=37, time=5)
    # Ids outside their field's vocabulary land in row ``offset + id``: field 1's
    # id 10 and field 3's id 0 are both row 74, which gets the cell's gradient
    # twice, and field 5's id -1 is row 105.
    rows[0, 0, 8 + 1] = 10.0
    rows[0, 0, 8 + 3] = 0.0
    rows[3, 2, 16 + 5] = -1.0
    grad = torch.randn(37, 5, config.channels_concat, generator=generator)
    # Exponents from 2^-30 (below the fixed point's 2^-24) to 2^36, so every
    # byte of the int64 contributions is exercised; 185 contributions of 2^61
    # to cell 0's rows make those sums wrap.
    grad *= 2.0 ** torch.randint(-30, 37, grad.shape, generator=generator)
    grad[:, :, :16] = 2.0**37
    rows = rows.cuda()
    grad = grad.bfloat16().cuda()
    reference = embedding.backward_torch(rows, grad)
    assert torch.equal(embedding.backward_triton(rows, grad), reference)
    assert torch.equal(embedding.backward(rows.bfloat16(), grad), reference)


@pytest.mark.gpu_triton
def test_the_triton_forward_matches_the_torch_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _eight_field_layout()
    embedding = config.make().cuda()
    rows = _draw_packed(config, batch=64, time=3).cuda()
    with torch.no_grad():
        reference = embedding.forward_torch(rows)
        assert torch.equal(embedding.forward_triton(rows), reference)
        # On CUDA with a bf16 table, ``forward`` is the Triton path.
        assert torch.equal(embedding(rows), reference)
        assert torch.equal(embedding(rows.bfloat16()), reference)


@pytest.mark.gpu_triton
def test_an_id_above_256_reads_the_row_its_gradient_lands_on_cuda() -> None:
    """fp32 ids index exactly on every path, where bf16 would round 257 to 256."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = MultiHotEmbedding.Config()
    config.channels_in = 512
    config.channels_out = 16
    config.num_cells = 3
    config.num_scalars = 1
    config.dtype = torch.bfloat16
    embedding = config.make().cuda()
    rows = torch.tensor([[257.0, 301.0, 511.0, 0.1]], device="cuda")
    with torch.no_grad():
        features = embedding(rows)
        gradient = embedding.backward(rows, torch.ones_like(features))
        assert torch.equal(features, embedding.forward_torch(rows))
        assert torch.equal(
            features[0, :48].reshape(3, 16),
            embedding.weight[[257, 301, 511]],
        )
    assert gradient.any(dim=1).nonzero().flatten().tolist() == [257, 301, 511]


@pytest.mark.gpu_triton
def test_an_embedding_of_any_width_runs_on_cuda_as_the_reference_does() -> None:
    """Rows of 3 lanes: the forward masks its lanes, the backward takes the torch sums."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _eight_field_layout()
    config.channels_out = 3
    embedding = config.make().cuda()
    rows = _draw_packed(config, batch=5, time=2).cuda()
    grad = torch.randn(5, 2, config.channels_concat, device="cuda").bfloat16()
    with torch.no_grad():
        assert torch.equal(embedding(rows), embedding.forward_torch(rows))
        gradient = embedding.backward(rows, grad)
        assert torch.equal(gradient, embedding.backward_torch(rows, grad))


def _small_layout() -> MultiHotEmbedding.Config:
    """Return three cells of four fields, 2 lanes, and 2 scalars."""
    config = MultiHotEmbedding.Config()
    config.channels_in = 8
    config.channels_out = 2
    config.offsets = (0, 2, 4, 6)
    config.num_cells = 3
    config.num_scalars = 2
    config.dtype = torch.bfloat16
    return config


def _eight_field_layout() -> MultiHotEmbedding.Config:
    """Return eight fields, as the Triton backward unrolls them, at a production width."""
    config = MultiHotEmbedding.Config()
    config.channels_in = 154
    config.channels_out = 16
    config.offsets = (0, 64, 72, 74, 90, 106, 122, 138)
    config.num_cells = 99
    config.num_scalars = 51
    config.dtype = torch.bfloat16
    return config


def _draw_packed(
    config: MultiHotEmbedding.Config,
    *,
    batch: int = 2,
    time: int = 1,
) -> Tensor:
    """Draw packed rows, ``[batch, time, size]`` or ``[batch, size]``: ids, then scalars."""
    generator = torch.Generator().manual_seed(0)
    offsets = torch.tensor(config.offsets)
    widths = torch.diff(torch.cat((offsets, torch.tensor([config.channels_in]))))
    ids = torch.floor(
        torch.rand(
            batch,
            time,
            config.num_cells,
            len(config.offsets),
            generator=generator,
        )
        * widths,
    )
    scalars = torch.rand(batch, time, config.num_scalars, generator=generator)
    return torch.cat((ids.flatten(-2), scalars), dim=-1).squeeze(1)


def test_multi_hot_embedding_kernel_host_dispatch_with_fake_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_layout()
    embedding_module = config.make()
    rows = _draw_packed(config, batch=2, time=3).to("meta")
    embed_kernel = _FakeKernel()
    monkeypatch.setattr(embedding, "_embed_kernel", lambda: embed_kernel)
    features = embedding_module.forward_triton(rows)
    assert features.shape == (2, 3, config.channels_concat)
    assert features.dtype == embedding_module.weight.dtype
    assert features.device == rows.device
    assert embed_kernel.grid == (6,)
    assert embed_kernel.args is not None
    observations, weight, offsets, output = embed_kernel.args[:4]
    assert isinstance(observations, Tensor)
    assert isinstance(output, Tensor)
    assert observations.shape == (6, config.observation_size)
    assert observations.device == rows.device
    assert weight is embedding_module.weight
    assert offsets is embedding_module.offsets
    assert output.shape == (6, config.channels_concat)
    assert output.dtype == features.dtype
    assert output.device == features.device
    assert embed_kernel.args[4:] == (config.num_cells, config.num_scalars)
    assert embed_kernel.kwargs == {
        "num_fields": len(config.offsets),
        "width": config.channels_out,
        "width_block": 2,
        "cell_block": 4,
        "scalar_block": 2,
        "table_rows": config.channels_in,
        "num_warps": 4,
    }

    wide = _eight_field_layout()
    wide_embedding = wide.make()
    wide_rows = _draw_packed(wide, batch=2, time=3)
    kernels = _fake_backward_kernels()
    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        _fake_device_properties,
    )
    gradient = torch.ones(
        2,
        3,
        wide.channels_concat,
        dtype=torch.bfloat16,
        device="meta",
    )
    result = wide_embedding.backward_triton(wide_rows, gradient)
    assert result.shape == (wide.channels_in, wide.channels_out)
    assert result.dtype == wide_embedding.weight.dtype
    assert result.device == wide_embedding.weight.device
    accumulate = kernels.accumulate
    assert accumulate.grid == (4,)
    assert accumulate.args is not None
    assert isinstance(accumulate.args[0], tuple)
    assert isinstance(accumulate.args[0][0], Tensor)
    assert isinstance(accumulate.args[0][1], Tensor)
    assert isinstance(accumulate.args[0][2], Tensor)
    assert isinstance(accumulate.args[0][3], Tensor)
    observations = accumulate.args[0][0]
    grad = accumulate.args[0][1]
    offsets = accumulate.args[0][2]
    partials = accumulate.args[0][3]
    assert observations.shape == (6, wide.observation_size)
    assert observations.device == wide_rows.device
    assert grad.shape == (6, wide.channels_concat)
    assert grad.device == torch.device("meta")
    assert offsets is wide_embedding.offsets
    assert partials.shape == (4, wide.channels_in * wide.channels_out)
    assert partials.dtype == torch.int64
    assert partials.device == torch.device("meta")
    assert accumulate.args[1:] == (
        594,
        5,
        wide.observation_size,
        wide.channels_concat,
        154,
    )
    assert accumulate.kwargs == {
        "num_cells": 99,
        "width": 16,
        "tile": 128,
        "vocab_first": 64,
        "vocab_rest": 16,
        "num_warps": 4,
    }
    decode = kernels.decode
    assert decode.grid == ((2464 + 127) // 128,)
    assert decode.args is not None
    assert isinstance(decode.args[1], Tensor)
    assert decode.args[1].shape == (154, 16)
    assert isinstance(decode.args[0], Tensor)
    assert decode.args[0].shape == (4, 2464)
    assert decode.args[2:] == (4, 2464, 2464)
    assert decode.kwargs == {
        "program_block": 64,
        "block": 128,
        "num_warps": 4,
    }


def test_multi_hot_backward_triton_uses_vocab_bounds_and_exact_program_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 31
    config.channels_out = 16
    config.offsets = (0, 2, 4, 6, 8, 10, 12, 14)
    config.num_cells = 2
    config.num_scalars = 1
    config.dtype = torch.bfloat16
    module = config.make()
    kernels = _fake_backward_kernels()
    property_devices: list[torch.device] = []

    def properties(device: torch.device) -> _FakeProperties:
        property_devices.append(device)
        return _FakeProperties()

    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    rows = torch.empty(2, 3, config.observation_size, device="meta")
    gradients = torch.empty(2, 3, config.channels_concat, device="meta")

    result = module.backward_triton(rows, gradients)

    accumulate = kernels.accumulate
    assert result.shape == module.weight.shape
    assert result.dtype == module.weight.dtype
    assert result.device == module.weight.device
    assert accumulate.grid == (4,)
    assert accumulate.args is not None
    assert isinstance(accumulate.args[0], tuple)
    assert isinstance(accumulate.args[0][0], Tensor)
    assert isinstance(accumulate.args[0][1], Tensor)
    assert isinstance(accumulate.args[0][2], Tensor)
    assert isinstance(accumulate.args[0][3], Tensor)
    observations = accumulate.args[0][0]
    grad = accumulate.args[0][1]
    offsets = accumulate.args[0][2]
    partials = accumulate.args[0][3]
    assert observations.shape == (6, config.observation_size)
    assert observations.device == rows.device
    assert grad.shape == (6, config.channels_concat)
    assert grad.device == gradients.device
    assert offsets is module.offsets
    assert partials.shape == (4, module.weight.numel())
    assert partials.dtype == torch.int64
    assert partials.device == gradients.device
    assert property_devices == [gradients.device]
    assert accumulate.args[1:6] == (
        12,
        1,
        config.observation_size,
        config.channels_concat,
        31,
    )
    assert accumulate.kwargs == {
        "num_cells": 2,
        "width": 16,
        "tile": 128,
        "vocab_first": 16,
        "vocab_rest": 32,
        "num_warps": 4,
    }


@pytest.mark.parametrize(
    ("pairs", "programs"),
    [(4 * (2**31 // 128) + 2, 5), (66_600_000, 4)],
)
def test_multi_hot_backward_triton_program_count_ceil_division(
    monkeypatch: pytest.MonkeyPatch,
    pairs: int,
    programs: int,
) -> None:
    config = _eight_field_layout()
    config.num_cells = 2
    module = config.make()
    kernels = _fake_backward_kernels()
    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties)
    rows = torch.empty(pairs // 2, config.observation_size, device="meta")
    gradients = torch.empty(pairs // 2, config.channels_concat, device="meta")

    module.backward_triton(rows, gradients)

    assert kernels.accumulate.grid == (programs,)
    assert kernels.accumulate.args is not None
    assert kernels.accumulate.args[1] == pairs
    assert kernels.accumulate.args[2] == (pairs + 127) // 128


def test_multi_hot_embedding_cost_counts_cpu_gather_and_scatter() -> None:
    config = _small_layout()
    cost = config.cost(seq_len=3, batch_size=2, dtype=torch.float32)
    assert cost.params == config.channels_in * config.channels_out
    assert cost.params_active == len(config.offsets) * config.channels_out
    assert cost["flops", "adjoint", "selection"].sum() == 3 * 2 * 3 * 4 * 2


@pytest.mark.gpu_triton
def test_multi_hot_triton_rejects_a_row_outside_the_table() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    # A device assertion poisons its CUDA context; isolate it from other tests.
    process = multiprocessing.get_context("spawn").Process(
        target=_check_triton_invalid_row,
    )
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0


def _check_triton_invalid_row() -> None:
    config = _small_layout()
    model = config.make()
    rows = _draw_packed(config)
    rows[0, 0] = config.channels_in
    with pytest.raises(IndexError):
        model.forward_torch(rows)
    model = model.cuda()
    with pytest.raises(RuntimeError, match="device-side assert"):
        _launch_invalid_row(model, rows.cuda())


def _launch_invalid_row(model: MultiHotEmbedding, rows: Tensor) -> None:
    model.forward_triton(rows)
    torch.cuda.synchronize()


def test_multi_hot_materialization_rebuilds_offsets() -> None:
    config = _small_layout()
    model = config.make().to("meta")
    model.to_empty(device="cpu")
    model.offsets.fill_(-1)
    model.reset_parameters()
    assert torch.equal(model.offsets, torch.tensor(config.offsets))


@pytest.mark.parametrize("offsets", [(), (3, 1), (0, 2, 2), (0, 8), (-1, 2)])
def test_multi_hot_invalid_field_offsets_rejected(offsets: tuple[int, ...]) -> None:
    config = _small_layout()
    config.offsets = offsets
    with pytest.raises(ValueError, match="offsets"):
        config.make()


def test_embedding_keeps_shard_and_requested_meta_device() -> None:
    model = Embedding.Config(
        3,
        2,
        device="meta",
        shard="vocab",
        init_weight=lambda _weight: None,
    ).make()
    assert model.shard == "vocab"
    assert model.weight.device == torch.device("meta")


def test_multi_hot_offsets_are_nonpersistent_buffers() -> None:
    model = _small_layout().make()
    assert torch.equal(model.offsets, torch.tensor((0, 2, 4, 6)))
    assert "offsets" not in model.state_dict()


def test_multi_hot_backward_dispatch_checks_every_triton_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (False, torch.bfloat16, 8, 16),
        (True, torch.float32, 8, 16),
        (True, torch.bfloat16, 7, 16),
        (True, torch.bfloat16, 9, 16),
        (True, torch.bfloat16, 8, 8),
        (True, torch.bfloat16, 8, 16),
        (True, torch.bfloat16, 8, 24),
    )

    for cuda, dtype, fields, width in cases:
        config = MultiHotEmbedding.Config()
        config.channels_in = fields
        config.channels_out = width
        config.offsets = tuple(range(fields))
        config.num_cells = 2
        config.dtype = dtype
        model = config.make()
        input = torch.zeros(2, config.observation_size)
        if cuda:
            input = _CudaInput(input)
        grad = torch.zeros(2, config.channels_concat)
        dispatch = _BackwardDispatch()
        monkeypatch.setattr(model, "backward_triton", dispatch.triton)
        monkeypatch.setattr(model, "backward_torch", dispatch.torch)

        result = model.backward(input, grad)

        should_use_triton = (
            cuda and dtype == torch.bfloat16 and fields == 8 and width == 16
        )
        assert result is grad
        assert len(dispatch.triton_calls) == should_use_triton
        assert len(dispatch.torch_calls) == (not should_use_triton)
        if should_use_triton:
            assert dispatch.triton_calls == [(input, grad)]
        else:
            assert dispatch.torch_calls == [(input, grad)]


def test_multi_hot_backward_torch_allocates_on_input_device() -> None:
    config = _small_layout()
    model = config.make()
    input = torch.empty(2, config.observation_size, device="meta")
    grad = torch.empty(2, config.channels_concat, device="meta")
    assert model.backward_torch(input, grad).device == input.device


def test_multi_hot_backward_triton_uses_second_field_vocab_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 80
    config.channels_out = 16
    config.offsets = (0, 2, 66, 68, 70, 72, 74, 76)
    config.num_cells = 2
    config.dtype = torch.bfloat16
    model = config.make()
    kernels = _fake_backward_kernels()
    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties)
    rows = torch.empty(2, config.observation_size, device="meta")
    grad = torch.empty(2, config.channels_concat, device="meta")

    model.backward_triton(rows, grad)

    assert kernels.accumulate.kwargs is not None
    assert kernels.accumulate.kwargs["vocab_rest"] == 64


def test_multi_hot_backward_triton_floors_small_vocab_tiles_at_sixteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 46
    config.channels_out = 16
    config.offsets = (0, 32, 34, 36, 38, 40, 42, 44)
    config.num_cells = 2
    config.dtype = torch.bfloat16
    model = config.make()
    kernels = _fake_backward_kernels()
    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties)
    rows = torch.empty(2, config.observation_size, device="meta")
    grad = torch.empty(2, config.channels_concat, device="meta")

    model.backward_triton(rows, grad)

    assert kernels.accumulate.kwargs is not None
    assert kernels.accumulate.kwargs["vocab_rest"] == 16


def test_multi_hot_forward_torch_sums_all_fields_with_multiple_leading_axes() -> None:
    config = _small_layout()
    model = config.make()
    with torch.no_grad():
        model.weight.copy_(torch.arange(16).reshape(8, 2).bfloat16())
    rows = _draw_packed(config, batch=2, time=3)
    features = model.forward_torch(rows)
    # MultiHotEmbedding packs three cells and four fields by contract.
    ids = rows[..., :12].reshape(2, 3, 3, 4).long() + model.offsets
    embedded = model.weight[ids].float()
    expected_cells = embedded[..., 0, :] + embedded[..., 1, :]
    expected_cells = expected_cells + embedded[..., 2, :] + embedded[..., 3, :]
    expected = torch.cat(
        (expected_cells.bfloat16().flatten(-2), rows[..., 12:].bfloat16()),
        dim=-1,
    )
    assert torch.equal(features, expected)


def test_multi_hot_reset_parameters_passes_empty_depth_index() -> None:
    seen: list[tuple[tuple[int, ...], ...]] = []
    config = _small_layout()

    def init_weight(
        weight: Tensor,
        *,
        depth_index: tuple[tuple[int, ...], ...] = ((1,),),
    ) -> None:
        del weight
        seen.append(depth_index)

    config.init_weight = init_weight
    model = config.make()
    model.reset_parameters()
    assert seen == [(), ()]


class _CudaInput(Tensor):
    """CPU storage with the CUDA flag set for dispatch-guard unit tests."""

    is_cuda = True


def test_multi_hot_backward_dispatch_uses_triton_for_eight_fields_at_width_sixteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _mutation_layout(offsets=tuple(range(8)), width=16, rows=16)
    input = _CudaInput(torch.zeros(2, module.num_cells * 8))
    grad = torch.zeros(2, module.num_cells * 16)
    dispatch = _BackwardDispatch()
    monkeypatch.setattr(module, "backward_triton", dispatch.triton)
    monkeypatch.setattr(module, "backward_torch", dispatch.torch)

    assert module.backward(input, grad) is grad
    assert dispatch.triton_calls == [(input, grad)]
    assert dispatch.torch_calls == []


@pytest.mark.parametrize(
    ("cuda", "dtype", "fields", "width"),
    [
        (False, torch.bfloat16, 8, 16),
        (True, torch.float32, 8, 16),
        (True, torch.bfloat16, 7, 16),
        (True, torch.bfloat16, 9, 16),
        (True, torch.bfloat16, 8, 8),
        (True, torch.bfloat16, 8, 24),
    ],
)
def test_multi_hot_backward_dispatch_uses_torch_outside_triton_layout(
    monkeypatch: pytest.MonkeyPatch,
    cuda: bool,
    dtype: torch.dtype,
    fields: int,
    width: int,
) -> None:
    module = _mutation_layout(
        offsets=tuple(range(fields)),
        width=width,
        rows=fields,
        dtype=dtype,
    )
    input = torch.zeros(2, module.num_cells * fields)
    if cuda:
        input = _CudaInput(input)
    grad = torch.zeros(2, module.num_cells * width)
    dispatch = _BackwardDispatch()
    monkeypatch.setattr(module, "backward_triton", dispatch.triton)
    monkeypatch.setattr(module, "backward_torch", dispatch.torch)

    assert module.backward(input, grad) is grad
    assert dispatch.triton_calls == []
    assert dispatch.torch_calls == [(input, grad)]


def test_multi_hot_offsets_move_with_module_but_stay_out_of_state_dict() -> None:
    module = _mutation_layout(
        offsets=tuple(range(8)),
        width=16,
        rows=16,
    ).to(device="meta", dtype=torch.float64)
    assert module.offsets.device.type == "meta"
    assert module.offsets.dtype == torch.int64
    assert module.weight.dtype == torch.float64
    assert "offsets" not in module.state_dict()


def test_multi_hot_backward_torch_allocates_on_meta_input_device() -> None:
    module = _mutation_layout(offsets=(0, 2), width=2, rows=4).to("meta")
    # backward_torch accepts arbitrary leading rows; only device is under test.
    rows = torch.empty(2, 3, device="meta")
    gradients = torch.empty(2, 3, device="meta")
    assert module.backward_torch(rows, gradients).device.type == "meta"


def test_multi_hot_forward_torch_sums_all_fields_and_preserves_scalars() -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 8
    config.channels_out = 2
    config.offsets = (0, 2, 4, 6)
    config.num_cells = 2
    config.num_scalars = 2
    model = config.make()
    with torch.no_grad():
        model.weight.copy_(torch.arange(16).reshape(8, 2).bfloat16())
    rows = torch.zeros(2, 3, config.observation_size)
    rows[..., -2] = 0.5
    rows[..., -1] = 0.25
    expected_row = torch.tensor([24, 28, 24, 28, 0.5, 0.25], dtype=torch.bfloat16)
    expected = expected_row.expand(2, 3, config.channels_concat)
    assert torch.equal(model.forward_torch(rows), expected)


def test_multi_hot_forward_triton_preserves_leading_input_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 4
    config.channels_out = 4
    config.offsets = (0, 2)
    config.num_cells = 1
    config.num_scalars = 2
    config.dtype = torch.bfloat16
    model = config.make()
    kernel = _FakeKernel()
    monkeypatch.setattr(embedding, "_embed_kernel", lambda: kernel)
    rows = torch.empty(3, 5, config.observation_size, device="meta")

    assert model.forward_triton(rows).shape == (3, 5, config.channels_concat)


def test_multi_hot_backward_triton_uses_every_nonfirst_field_vocab_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MultiHotEmbedding.Config()
    config.channels_in = 80
    config.channels_out = 16
    config.offsets = (0, 2, 66, 68, 70, 72, 74, 76)
    config.num_cells = 1
    config.dtype = torch.bfloat16
    model = config.make()
    kernels = _FakeBackwardKernels()
    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties)
    rows = torch.empty(2, config.observation_size, device="meta")
    gradients = torch.empty(2, config.channels_concat, device="meta")

    model.backward_triton(rows, gradients)

    assert kernels.accumulate.kwargs is not None
    assert kernels.accumulate.kwargs["vocab_rest"] == 64


def _mutation_layout(
    *,
    offsets: tuple[int, ...],
    width: int,
    rows: int,
    dtype: torch.dtype = torch.bfloat16,
) -> MultiHotEmbedding:
    config = MultiHotEmbedding.Config()
    config.channels_in = rows
    config.channels_out = width
    config.offsets = offsets
    config.num_cells = 2
    config.dtype = dtype
    return config.make()


def _launch_backward_mutation_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    offsets: tuple[int, ...],
    rows: int,
) -> tuple[_FakeKernel, Tensor, Tensor]:
    # The launch reads only metadata; a real 2^25-row table took 4.5s to draw.
    with torch.device("meta"):
        module = _mutation_layout(
            offsets=offsets,
            width=16,
            rows=max(rows, offsets[-1] + 1),
        )
    kernels = _FakeBackwardKernels()
    devices: list[torch.device] = []

    def properties(device: torch.device) -> _FakeProperties:
        devices.append(device)
        return _FakeProperties()

    monkeypatch.setattr(embedding, "_backward_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    input = torch.empty(3, rows // 3, module.num_cells * len(offsets), device="meta")
    grad = torch.empty(
        3,
        rows // 3,
        module.num_cells * module.weight.shape[1],
        device="meta",
    )
    result = module.backward_triton(input, grad)

    assert result.shape == module.weight.shape
    assert result.dtype == module.weight.dtype
    assert kernels.accumulate.args is not None
    assert isinstance(kernels.accumulate.args[0], tuple)
    assert isinstance(kernels.accumulate.args[0][0], Tensor)
    assert isinstance(kernels.accumulate.args[0][1], Tensor)
    assert isinstance(kernels.accumulate.args[0][2], Tensor)
    assert isinstance(kernels.accumulate.args[0][3], Tensor)
    observations = kernels.accumulate.args[0][0]
    gradient = kernels.accumulate.args[0][1]
    used_offsets = kernels.accumulate.args[0][2]
    partials = kernels.accumulate.args[0][3]
    assert observations.shape == (rows, input.shape[-1])
    assert observations.device == input.device
    assert gradient.shape == (rows, grad.shape[-1])
    assert gradient.device == grad.device
    assert used_offsets is module.offsets
    assert kernels.accumulate.grid is not None
    assert partials.shape == (kernels.accumulate.grid[0], module.weight.numel())
    assert partials.dtype is torch.int64
    assert partials.device == grad.device
    assert devices == [grad.device]
    assert kernels.accumulate.args[1] == rows * module.num_cells
    assert kernels.accumulate.args[2] == (rows * module.num_cells + 127) // 128
    assert kernels.accumulate.kwargs is not None
    assert kernels.accumulate.kwargs["num_cells"] == module.num_cells
    assert kernels.accumulate.kwargs["width"] == module.weight.shape[1]
    assert kernels.accumulate.kwargs["tile"] == 128
    assert kernels.accumulate.kwargs["num_warps"] == 4
    return kernels.accumulate, input, grad


def test_multi_hot_backward_triton_reshapes_rank_three_inputs_and_clamps_vocab_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small, _, _ = _launch_backward_mutation_case(
        monkeypatch,
        offsets=(0, 2, 4, 6, 8, 10, 12, 14),
        rows=6,
    )
    assert small.grid == (4,)
    assert small.kwargs is not None
    assert small.kwargs["vocab_first"] == 16
    assert small.kwargs["vocab_rest"] == 16

    larger_second_field, _, _ = _launch_backward_mutation_case(
        monkeypatch,
        offsets=(0, 2, 19, 21, 23, 25, 27, 29),
        rows=6,
    )
    assert larger_second_field.kwargs is not None
    assert larger_second_field.kwargs["vocab_first"] == 16
    assert larger_second_field.kwargs["vocab_rest"] == 32


@pytest.mark.parametrize(
    ("pairs", "programs"),
    [(4 * (2**31 // 128) + 2, 5), (66_600_000, 4)],
)
def test_multi_hot_backward_triton_program_count_uses_signed_ceil_division(
    monkeypatch: pytest.MonkeyPatch,
    pairs: int,
    programs: int,
) -> None:
    rows = pairs // 2
    assert rows * 2 == pairs
    accumulate, _, _ = _launch_backward_mutation_case(
        monkeypatch,
        offsets=(0, 2, 4, 6, 8, 10, 12, 14),
        rows=rows,
    )
    assert accumulate.grid == (programs,)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
