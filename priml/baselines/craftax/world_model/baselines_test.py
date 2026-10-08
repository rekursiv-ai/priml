"""Check the trivial validation baselines a trained world model must beat."""

import math

import pytest
import torch

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.baselines import report, tally
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
    WorldModelLogits,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib.codec import from_plain
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


CELLS = 99
"""Cells of a Craftax frame, each of which a next frame scores."""


def test_copying_the_current_frame_scores_the_cells_it_repeats() -> None:
    segment, batch = _batch()
    counts = tally(tiny_model(), batch)
    # Three act jobs have a next frame, of 99 cells each; frame 1 -> 2 repeats.
    assert int(counts["cells"]) == 3 * CELLS
    copied = (segment.cells[1:] == segment.cells[:-1]).all(-1).sum()
    assert int(counts["copy_correct"]) == int(copied) >= CELLS


def test_a_model_that_predicts_the_next_frame_scores_every_cell() -> None:
    model = tiny_model()
    _, batch = _batch()
    with torch.no_grad():
        logits = model.logits(batch)
    # Put all mass on each job's target IDs.
    local = torch.full_like(logits.local, -30.0)
    prefix = len(model.schema.prefix_ranges)
    target = batch.cells[batch.job_next.clamp(min=0).long()].long()
    ids = target + model.cell_offsets
    cells = local[:, prefix : prefix + model.schema.cell_slots]
    cells.scatter_(-1, ids, 30.0)
    perfect = WorldModelLogits(action=logits.action, local=local)
    counts = tally(model, batch, logits=perfect)
    assert int(counts["model_correct"]) == int(counts["cells"]) == 3 * CELLS
    # A cell is right only when every field is: miss one field everywhere.
    visibility = [f.name for f in model.schema.cell_fields].index("visibility")
    cells.scatter_(-1, ids[..., visibility : visibility + 1], -60.0)
    missed = tally(model, batch, logits=perfect)
    assert int(missed["model_correct"]) == 0


def test_report_sets_model_nll_beside_the_targets_own_frequencies() -> None:
    model = tiny_model()
    _, batch = _batch()
    result = report([tally(model, batch), tally(model, batch)])
    empirical = from_plain(result["empirical_nll"], dict[str, float])
    # Actions 4, 4, 7 per window: entropy of (2/3, 1/3).
    entropy = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert from_plain(empirical["action"], float) == pytest.approx(entropy)
    assert from_plain(empirical["reward"], float) == pytest.approx(entropy)
    # Every act job's done is false: a certain target costs nothing.
    assert from_plain(empirical["done"], float) == 0.0
    nll = from_plain(result["model_nll"], dict[str, float])
    assert set(nll) == {"action", "reward", "done", "board", "hud"}
    assert from_plain(nll["action"], float) > 0
    accuracy = from_plain(result["cell_accuracy"], dict[str, float])
    assert set(accuracy) == {"model", "copy"}
    # An untrained model loses every comparison.
    assert from_plain(result["beats"], dict[str, bool]) == {
        "action": False,
        "reward": False,
        "done": False,
        "cells": False,
    }


def tiny_model() -> WorldModel:
    """Return a seeded world model at minimum width over the Craftax schema."""
    config = WorldModel.Config()
    config.encoder.channels_in = config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        stack.num_layers = 1
        assert isinstance(stack.block, TransformerBlock.Config | DecoderBlock.Config)
        assert isinstance(stack.block.ffn, SwiGLU.Config)
        stack.block.ffn.channels_hidden = 32
    config.transformer.channels_in = 36
    config.transformer.num_layers = 1
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48
    torch.manual_seed(0)
    return config.make().eval()


def _batch() -> tuple[Segment, PackedBatch]:
    """One window: a start job and three act jobs; frame 2 copies frame 1."""
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(3)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (4, CELLS), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    ).to(torch.uint8)
    cells[2] = cells[1]
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (4,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    ).to(torch.int16)
    segment = Segment(
        cells=cells,
        aux=aux,
        actions=torch.tensor([4, 4, 7], dtype=torch.uint8),
        reward=torch.tensor([0, 1, 0], dtype=torch.int16),
        done=torch.zeros(3, dtype=torch.bool),
        starts_episode=True,
    )
    return segment, pack_windows([[segment]], t_g=8, s_max=1)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
