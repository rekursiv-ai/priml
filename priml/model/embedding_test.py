"""Tests for embedding module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.cost import Cost
from priml.model.embedding import Embedding, MultiHotEmbedding
from priml.model.init import normal
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_embedding_config_pprint() -> None:
    config = Embedding.Config(8, 4)
    assert_pprint_golden(
        test_file=__file__,
        name="embedding",
        config=config,
    )


def test_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="embedding",
        build_module=lambda: Embedding.Config(8, 4).make(),
        build_input=lambda: torch.tensor([[0, 3, 7]]),
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
    assert config.observation_size == 3 * 3 + 2
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
        build_input=lambda: _draw_packed(config, batch=2),
        seed=0,
    )


def test_a_multi_hot_embedding_sums_each_cells_rows_then_appends_scalars() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    features = embedding(rows)
    assert features.shape == (2, config.channels_concat)
    assert features.dtype == torch.bfloat16
    ids = rows[:, :9].reshape(2, 3, 3).long() + torch.tensor(config.offsets)
    for row in range(2):
        for cell in range(3):
            total = torch.zeros(2)
            for field in range(3):
                total = total + embedding.weight[ids[row, cell, field]].float()
            assert torch.equal(features[row, 2 * cell : 2 * cell + 2], total.bfloat16())
    assert torch.equal(features[:, 6:], rows[:, 9:].bfloat16())


def test_a_multi_hot_backward_is_fixed_point_rounded_to_the_table_dtype() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=2)
    grad = torch.randn(2, config.channels_concat).bfloat16()
    result = embedding.backward(rows, grad)
    assert result.shape == (9, 2)
    assert result.dtype == torch.bfloat16
    # Every value is a multiple of 2^-24 rounded once to bf16, and the scalar
    # columns contribute nothing.
    zero = embedding.backward(rows, grad * 0)
    assert torch.equal(zero, torch.zeros(9, 2).bfloat16())
    ids = rows[:, :9].reshape(-1, 3).long() + torch.tensor(config.offsets)
    fixed = (grad[:, :6].float().reshape(-1, 2) * 2.0**24).round().to(torch.int64)
    total = torch.zeros(9, 2, dtype=torch.int64)
    for field in range(3):
        total.index_add_(0, ids[:, field], fixed)
    assert torch.equal(result, (total.double() * 2.0**-24).float().bfloat16())


def test_autograd_reaches_the_table_through_the_fixed_point_backward() -> None:
    config = _small_layout()
    embedding = config.make()
    rows = _draw_packed(config, batch=4).reshape(2, 2, -1)
    grad = torch.randn(2, 2, config.channels_concat).bfloat16()
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
    """Return three cells of three fields (vocabularies 3, 2 and 4), 2 lanes, 2 scalars."""
    config = MultiHotEmbedding.Config()
    config.channels_in = 9
    config.channels_out = 2
    config.offsets = (0, 3, 5)
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
    batch: int,
    time: int = 1,
) -> torch.Tensor:
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
