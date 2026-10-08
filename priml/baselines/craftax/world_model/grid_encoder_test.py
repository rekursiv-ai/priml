"""Tests for the grid convolution frame encoder."""

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.game.state import OBS_COLS
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.grid_encoder import (
    GridConvEncoder,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    WorldModel,
)
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    craftax_schema,
)
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.cost import assert_cost_matches_torch


def test_encoder_maps_slots_to_pooled_and_per_slot_memory() -> None:
    encoder = _encoder()
    pooled, memory = encoder(torch.randn(2, 3, 150, 16))
    assert pooled.shape == (2, 3, 16)
    assert memory.shape == (2, 3, 150, 16)
    alone, _ = encoder(torch.randn(150, 16))
    assert alone.shape == (16,)


def test_board_slot_is_its_observation_grid_cell() -> None:
    # A frame's cells are the packed observation's (replay_test), which the game
    # lays out row-major: slot 10 is row 0, column 10.
    row, column = divmod(10, OBS_COLS)
    # One 3 x 3 block: perturbing a cell moves only its grid neighbours' memory.
    encoder = _encoder(num_blocks=1, kernel_size=3)
    slots = torch.randn(150, 16)
    _, before = encoder(slots)
    slots[row * 11 + column] += 1
    _, after = encoder(slots)
    moved = (after - before).abs().amax(-1) > 0
    want = torch.zeros(9, 11, dtype=torch.bool)
    want[max(row - 1, 0) : row + 2, max(column - 1, 0) : column + 2] = True
    assert torch.equal(moved[:99].unflatten(0, (9, 11)), want)
    assert not moved[99:].any()


def test_aux_slots_condition_every_cell_but_the_board_no_aux_memory() -> None:
    encoder = _encoder()
    with torch.no_grad():
        encoder.film.normal_()
    slots = torch.randn(150, 16)
    _, before = encoder(slots)
    slots[120] += 1
    _, after = encoder(slots)
    moved = (after - before).abs().amax(-1) > 0
    assert moved[:99].all()
    assert torch.equal(moved[99:], torch.arange(99, 150) == 120)


def test_memory_is_final_normed() -> None:
    encoder = _encoder()
    _, memory = encoder(torch.randn(5, 150, 16))
    torch.testing.assert_close(
        memory.square().mean(-1),
        torch.ones(5, 150),
        rtol=1e-4,
        atol=1e-4,
    )


def test_config_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="rows x columns"):
        GridConvEncoder.Config(channels_in=16, num_slots=99).make()
    with pytest.raises(ValueError, match="odd"):
        GridConvEncoder.Config(channels_in=16, num_slots=150, kernel_size=4).make()


def test_frame_macs_counts_convolutions_mlps_and_film() -> None:
    encoder = _encoder(num_blocks=2, kernel_size=3)
    per_cell = 2 * (16 * 9 + 3 * 16 * 8)
    want = 99 * per_cell + 51 * 3 * 16 * 8 + 2 * 2 * 16 * 16
    assert encoder.frame_macs() == want


def test_the_encoder_costs_the_convolutions_products_and_pooling_torch_runs() -> None:
    config = GridConvEncoder.Config(
        channels_in=12,
        num_slots=11,
        rows=2,
        columns=3,
        kernel_size=3,
        channels_hidden=7,
        num_heads=2,
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(4, 11, 12, requires_grad=True),
        run=_encoded,
        batch_size=4,
        dtype=None,
    )


def test_world_model_trains_and_reloads_with_a_grid_encoder() -> None:
    config = _tiny_world_model_config()
    torch.manual_seed(0)
    model = config.make()
    assert isinstance(model.encoder, GridConvEncoder)
    loss = model(_batch()).loss
    loss.backward()
    assert loss.isfinite()
    grads = {n: p.grad for n, p in model.named_parameters() if n.startswith("encoder.")}
    assert all(g is not None and g.abs().sum() > 0 for g in grads.values())
    reloaded = config.make()
    reloaded.load_state_dict(model.state_dict(), strict=True)


def _encoded(model: nn.Module, slots: Tensor) -> Tensor:
    """Sum both of the encoder's outputs, so backward reaches every product."""
    assert isinstance(model, GridConvEncoder)
    pooled, memory = model(slots)
    return pooled.sum() + memory.sum()


def _encoder(*, num_blocks: int = 2, kernel_size: int = 7) -> GridConvEncoder:
    torch.manual_seed(0)
    config = GridConvEncoder.Config(
        channels_in=16,
        num_slots=150,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        channels_hidden=8,
        num_heads=4,
    )
    return config.make()


def _tiny_world_model_config() -> WorldModel.Config:
    config = WorldModel.Config(
        encoder=GridConvEncoder.Config(channels_in=16, channels_hidden=8, num_heads=4),
    )
    config.decoder.channels_in = 16
    config.decoder.stack.num_layers = 1
    block = config.decoder.stack.block
    assert isinstance(block, DecoderBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 32
    config.transformer.channels_in = 36
    config.transformer.num_layers = 1
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48
    return config


def _batch() -> PackedBatch:
    segment = _random_segment(craftax_schema(), 4, seed=0)
    return pack_windows([[segment]], t_g=16, s_max=1)


def _random_segment(schema: FrameSchema, decisions: int, *, seed: int) -> Segment:
    """Return valid random tokens: a frame per decision, plus the next."""
    generator = torch.Generator().manual_seed(seed)
    frames = decisions + 1
    cells = torch.stack(
        [
            torch.randint(
                0,
                field.valid,
                (frames, schema.cell_slots),
                generator=generator,
            )
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (frames,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    return Segment(
        cells=cells.to(torch.uint8),
        aux=aux.to(torch.int16),
        actions=torch.randint(0, 43, (decisions,), generator=generator).to(torch.uint8),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).to(torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
        starts_episode=True,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
