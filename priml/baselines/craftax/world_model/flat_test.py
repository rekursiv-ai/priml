"""Check the flat model: its layout, target alignment, and bits-per-byte accounting.

The flat model must score exactly the targets the hierarchical model scores,
each from the positions before it and none after, so the reference here lays
each window out by hand from its segments, as the design's sequence
``start, o_0, a_0, r_1, d_1, o_1, ...``, and runs the same stack over it.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

import dataclasses

from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.data import _pad_counts
from priml.baselines.craftax.world_model.flat import (
    HEAD,
    FlatModel,
    flat_cost,
    flat_layout,
    flat_positions,
)
from priml.baselines.craftax.world_model.metric import craftax_target_nll
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
    WorldModelLogits,
)
from priml.baselines.craftax.world_model.schema import (
    action_id,
    craftax_schema,
    done_id,
    number_id,
)
from priml.baselines.craftax.world_model.testing import naive_attention
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.cost import assert_cost_matches_torch


T_G = 10
"""Global positions per test window."""

CONTEXT = flat_positions(T_G)
"""Flat positions per test window: the most a window of ``T_G`` can need."""


def _shrink_global(stack: Transformer.Config) -> None:
    """Cut a global stack to one layer of width 36, GQA 2:1 at 4 channels a head."""
    stack.channels_in = 36
    stack.num_layers = 1
    block = stack.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    # A flat window is ~900 positions under SdpaVarlen's dense mask, whose cost
    # grows with the heads: the design's 9:3 made every forward 4x slower here,
    # and the layout these tests check does not depend on the head count.
    block.attn.num_heads = 2
    block.attn.num_heads_kv = 1
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48


def _flat(context: int = CONTEXT) -> FlatModel:
    config = FlatModel.Config()
    config.context = context
    _shrink_global(config.lm.transformer)
    torch.manual_seed(0)
    return config.make()


def _world() -> WorldModel:
    config = WorldModel.Config()
    config.encoder.channels_in = config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        stack.num_layers = 1
        assert isinstance(stack.block, TransformerBlock.Config | DecoderBlock.Config)
        assert isinstance(stack.block.ffn, SwiGLU.Config)
        stack.block.ffn.channels_hidden = 32
    _shrink_global(config.transformer)
    torch.manual_seed(0)
    return config.make()


def _segment(decisions: int, *, frames: int, starts: bool, terminal: bool) -> Segment:
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(decisions + 10 * frames + starts)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (frames, 99), generator=generator)
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
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = terminal
    return Segment(
        cells=cells.byte(),
        aux=aux.short(),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).short(),
        done=done,
        starts_episode=starts,
    )


def _windows() -> list[list[Segment]]:
    """Two windows: a whole episode then a cut one; one opened mid-episode."""
    return [
        [
            _segment(2, frames=2, starts=True, terminal=True),
            _segment(2, frames=3, starts=True, terminal=False),
        ],
        [_segment(2, frames=3, starts=False, terminal=False)],
    ]


def _batch(windows: Sequence[Sequence[Segment]] | None = None) -> PackedBatch:
    return pack_windows(windows or _windows(), t_g=T_G, s_max=3)


def test_layout_places_each_job_after_the_frame_it_reads() -> None:
    layout = flat_layout(_batch(), context=CONTEXT)
    # Window 0: a start (153), an act (153), a terminal act (3); then a start
    # and two acts, the last one generating the appended third frame. Window 1
    # opens with its first frame (150), which no job generates.
    assert layout.job_row.tolist() == [0] * 6 + [1] * 2
    assert layout.job_start.tolist() == [0, 153, 306, 309, 462, 615, 150, 303]
    assert layout.frame_row.tolist() == [0] * 5 + [1] * 3
    assert layout.frame_start.tolist() == [3, 156, 312, 465, 618, 0, 153, 306]
    # Window 0's last segment runs to the end of its window: the pad is its tail.
    window_0 = [0, 309, CONTEXT, CONTEXT, CONTEXT]
    window_1 = [CONTEXT + 456, 2 * CONTEXT, 2 * CONTEXT, 2 * CONTEXT]
    assert layout.cu_seqlens.tolist() == [*window_0, *window_1]
    assert layout.positions[0, [0, 308, 309, 310]].tolist() == [0, 308, 0, 1]
    assert layout.positions[1, [0, 455, 456]].tolist() == [0, 455, 0]
    assert layout.fits.tolist() == [True, True]


def _branch_windows() -> list[list[Segment]]:
    """``_windows`` with window 0's second episode a branch: no start, mid-row."""
    return [
        [
            _segment(2, frames=2, starts=True, terminal=True),
            _segment(2, frames=3, starts=False, terminal=False),
        ],
        [_segment(2, frames=3, starts=False, terminal=False)],
    ]


@pytest.mark.parametrize("make_windows", [_windows, _branch_windows])
def test_logits_match_the_plans_sequence_laid_out_by_hand(
    make_windows: Callable[[], list[list[Segment]]],
) -> None:
    model, windows = _flat(), make_windows()
    batch = _batch(windows)
    with torch.no_grad():
        logits = model.logits(batch)
        expected, blocks = _reference(model, windows)
    table = model.lm.embedding.weight
    first = action_id(0)
    for job, (row, start) in enumerate(blocks):
        has_frame = bool(batch.job_next[job] >= 0)
        slots = model.schema.local_slots if has_frame else 2
        local = expected[row, start : start + slots] @ table.T
        torch.testing.assert_close(logits.local[job, :slots], local)
        if not batch.job_is_start[job]:
            # The action is predicted from the last slot of the frame before it.
            global_at = int(batch.job_at[job]) - 1
            action = expected[row, start - 1] @ table[first : first + 43].T
            torch.testing.assert_close(logits.action.flatten(0, 1)[global_at], action)


@pytest.mark.parametrize(
    ("field", "job", "slot"),
    [("reward", 1, 0), ("done", 1, 1), ("cells", 1, 2 + 40), ("aux", 1, 2 + 99 + 7)],
)
def test_no_target_is_predicted_from_itself(field: str, job: int, slot: int) -> None:
    model = _flat()
    batch = _batch()
    changed = _perturb(batch, field=field, job=job, slot=slot)
    with torch.no_grad():
        before, after = model.logits(batch).local, model.logits(changed).local
    # Everything up to the changed target's own prediction reads only earlier
    # positions; the prediction right after it reads the change.
    torch.testing.assert_close(before[job, : slot + 1], after[job, : slot + 1])
    assert not torch.allclose(before[job, slot + 1], after[job, slot + 1])


def test_an_action_is_predicted_before_it_is_read() -> None:
    model = _flat()
    batch = _batch()
    job = 1
    at = int(batch.job_at[job])
    action = batch.action.clone()
    action.view(-1)[at] = (action.view(-1)[at] + 1) % 43
    changed = dataclasses.replace(batch, action=action)
    with torch.no_grad():
        before, after = model.logits(batch), model.logits(changed)
    torch.testing.assert_close(
        before.action.flatten(0, 1)[at - 1],
        after.action.flatten(0, 1)[at - 1],
    )
    assert not torch.allclose(before.local[job, 0], after.local[job, 0])


def test_targets_are_scored_as_the_world_model_scores_them() -> None:
    flat, world, batch = _flat(), _world(), _batch()
    generator = torch.Generator().manual_seed(1)
    logits = WorldModelLogits(
        action=torch.randn(*batch.kind.shape, 43, generator=generator),
        local=torch.randn(len(batch.job_at), 152, 461, generator=generator),
    )
    ours, theirs = flat.target_terms(batch, logits), world.target_terms(batch, logits)
    assert ours.keys() == theirs.keys()
    for name, (value, mask) in theirs.items():
        torch.testing.assert_close(ours[name][0].nll, value.nll, msg=name)
        torch.testing.assert_close(ours[name][0].logz_sq, value.logz_sq, msg=name)
        assert torch.equal(ours[name][1], mask), name


def test_records_score_the_hierarchical_targets() -> None:
    flat, world, batch = _flat(), _world(), _batch()
    with torch.no_grad():
        ours, theirs = craftax_target_nll(flat, batch), craftax_target_nll(world, batch)
    assert torch.equal(ours != 0, theirs != 0)


def test_records_match_the_loss() -> None:
    flat, batch = _flat(), _batch()
    with torch.no_grad():
        ours, loss = craftax_target_nll(flat, batch), flat(batch)
    torch.testing.assert_close(ours.sum(), sum(loss.nll.values()), rtol=1e-5, atol=0)


def test_count_padding_moves_no_input_and_no_scored_target() -> None:
    # The stream pads jobs with unscored start jobs at position 0, where window
    # 0 here has its own start job, and frames with zero frames no job reads.
    batch = _batch()
    padded = _pad_counts(batch, 16)
    jobs, frames = len(batch.job_at), len(batch.aux)
    assert len(padded.job_at) > jobs
    assert len(padded.aux) > frames
    ours, theirs = (flat_layout(b, context=CONTEXT) for b in (batch, padded))
    assert torch.equal(theirs.job_start[:jobs], ours.job_start)
    assert torch.equal(theirs.frame_start[:frames], ours.frame_start)
    assert torch.equal(theirs.cu_seqlens, ours.cu_seqlens)
    model = _flat()
    with torch.no_grad():
        unpadded = craftax_target_nll(model, batch).view(jobs, -1)
        nll = craftax_target_nll(model, padded).view(len(padded.job_at), -1)
    assert torch.equal(nll[:jobs], unpadded)


def test_the_bound_is_reached_by_an_even_and_an_odd_window() -> None:
    # A window cut mid-episode on both sides needs its lead frame and a block
    # per act; an odd one can open an episode and end on a job's appended frame.
    even = pack_windows(
        [[_segment(6, frames=7, starts=False, terminal=False)]],
        t_g=12,
        s_max=1,
    )
    odd = pack_windows(
        [[_segment(6, frames=7, starts=True, terminal=False)]],
        t_g=13,
        s_max=1,
    )
    for batch in (even, odd):
        t_g = batch.kind.shape[-1]
        assert flat_layout(batch, context=flat_positions(t_g)).fits.all()
        assert not flat_layout(batch, context=flat_positions(t_g) - 1).fits.any()


def test_a_window_past_the_bound_scores_nan_not_fewer_targets() -> None:
    # Unfinished episodes back to back break the stream's contract: each needs
    # two blocks for three global positions.
    short = _segment(1, frames=2, starts=True, terminal=False)
    windows = [[short, short], [_segment(3, frames=4, starts=True, terminal=False)]]
    batch = pack_windows(windows, t_g=6, s_max=2)
    model = _flat(context=flat_positions(6))
    with torch.no_grad():
        logits = model.logits(batch)
        loss = model(batch)
    first = batch.job_at < 6
    assert logits.local[first].isnan().all()
    assert logits.local[~first].isfinite().all()
    assert loss.loss.isnan()


def test_a_window_longer_than_the_context_raises() -> None:
    model = _flat(context=flat_positions(T_G) - 1)
    with pytest.raises(ValueError, match="does not fit"):
        model.logits(_batch())


def test_cost_counts_every_matmul_but_attention() -> None:
    model, batch = _flat(), _batch()
    with FlopCounterMode(display=False) as counter:
        model(batch)
    counts = counter.get_flop_counts()
    matmuls = sum(
        flops
        for op, flops in cast("dict[object, int]", counts["Global"]).items()
        if "attention" not in str(op)
    )
    # ``SdpaVarlen`` forms its scores again, under no grad, for the max logit.
    probe = sum(
        flops
        for name, ops in counts.items()
        if name.endswith("attn_kernel")
        for op, flops in cast("dict[object, int]", ops).items()
        if "attention" not in str(op)
    )
    cost = flat_cost(model, batch)
    assert cost.flops == matmuls - probe
    assert cost.positions == 2 * CONTEXT


def test_the_config_cost_prices_the_products_torch_runs() -> None:
    config = FlatModel.Config()
    config.context = CONTEXT
    _shrink_global(config.lm.transformer)
    naive_attention(config.lm.transformer.block)
    # One short episode: the cost counts positions, so a bigger batch adds
    # only time.
    batch = _batch([[_segment(2, frames=2, starts=True, terminal=True)]])
    assert_cost_matches_torch(
        config,
        build_input=lambda: batch.kind,
        run=partial(_flat_loss, batch=batch),
        seq_len=T_G,
        batch_size=len(batch.kind),
        frames=len(batch.aux),
        jobs=len(batch.job_at),
        dtype=None,
    )


def test_default_model_is_the_plan_size_stack_over_craftax_ids() -> None:
    config = FlatModel.Config()
    assert config.lm.vocab_size == 461
    assert config.lm.transformer.num_layers == 20
    assert config.lm.transformer.channels_in == 1_152
    assert config.context == 8_192
    # The longest replay window whose flat form fits 8,192 positions.
    assert flat_positions(105) <= 8_192 < flat_positions(106)


def _flat_loss(model: nn.Module, inputs: Tensor, *, batch: PackedBatch) -> Tensor:
    """Return ``model``'s loss on ``batch``; ``inputs`` only stands in for it."""
    del inputs
    assert isinstance(model, FlatModel)
    return model(batch).loss


def _perturb(batch: PackedBatch, *, field: str, job: int, slot: int) -> PackedBatch:
    """Change the target ``job`` predicts at local ``slot``, and nothing else."""
    following = int(batch.job_next[job])
    if field == "reward":
        reward = batch.job_reward.clone()
        reward[job] += 1
        return dataclasses.replace(batch, job_reward=reward)
    if field == "done":
        done = batch.job_done.clone()
        done[job] = ~done[job]
        return dataclasses.replace(batch, job_done=done)
    if field == "cells":
        cells = batch.cells.clone()
        cells[following, slot - 2, 0] = (cells[following, slot - 2, 0] + 1) % 37
        return dataclasses.replace(batch, cells=cells)
    aux = batch.aux.clone()
    aux[following, slot - 2 - 99] = (aux[following, slot - 2 - 99] + 1) % 100
    return dataclasses.replace(batch, aux=aux)


def _reference(
    model: FlatModel,
    windows: Sequence[Sequence[Segment]],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return the stack's output over windows laid out by hand, and each job's block."""
    rows: list[torch.Tensor] = []
    ends: list[int] = [0]
    blocks: list[tuple[int, int]] = []
    for row, segments in enumerate(windows):
        parts: list[torch.Tensor] = []
        for segment in segments:
            if segment.starts_episode:
                blocks.append((row, sum(len(p) for p in parts)))
                parts.append(_head(model, action=None, reward=0, done=False))
            parts.append(_frame(model, segment, 0))
            for step in range(len(segment.actions)):
                blocks.append((row, sum(len(p) for p in parts)))
                parts.append(
                    _head(
                        model,
                        action=int(segment.actions[step]),
                        reward=int(segment.reward[step]),
                        done=bool(segment.done[step]),
                    ),
                )
                if step + 1 < len(segment.cells) and not segment.done[step]:
                    parts.append(_frame(model, segment, step + 1))
            ends.append(row * CONTEXT + sum(len(p) for p in parts))
        used = torch.cat(parts)
        rows.append(torch.cat([used, used.new_zeros(CONTEXT - len(used), 36)]))
        ends.append((row + 1) * CONTEXT)
    cu_seqlens = torch.tensor(ends, dtype=torch.int32)
    flat = torch.arange(len(rows) * CONTEXT)
    segment_of = torch.searchsorted(cu_seqlens.long(), flat, right=True) - 1
    positions = (flat - cu_seqlens[segment_of]).view(len(rows), CONTEXT).int()
    hidden = model.lm.transformer(
        torch.stack(rows),
        positions=positions,
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(hidden, torch.Tensor)
    return hidden, blocks


def _head(
    model: FlatModel,
    *,
    action: int | None,
    reward: int,
    done: bool,
) -> torch.Tensor:
    """Embed a head: the start vector or an action, then reward and done."""
    table = model.lm.embedding.weight
    first = model.start if action is None else table[action_id(action)]
    rows = torch.stack([first, table[number_id(reward)], table[done_id(done=done)]])
    return rows + model.slot_embedding[:HEAD]


def _frame(model: FlatModel, segment: Segment, index: int) -> torch.Tensor:
    """Embed one frame of a segment with its slot rows."""
    slots = model.frame_slots(segment.cells[index], segment.aux[index])
    return slots + model.slot_embedding[HEAD:]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
