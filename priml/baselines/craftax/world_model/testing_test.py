"""Tests for the world model's test support: its tiny models, frames and contexts."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    DEFAULT_MAX_TIMESTEPS,
    OBS_SIZE,
)
from priml.baselines.craftax.world_model import testing
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import pack_windows
from priml.baselines.craftax.world_model.engine import Engine
from priml.baselines.craftax.world_model.model import DecoderBlock
from priml.baselines.craftax.world_model.schema import done_id
from priml.baselines.craftax.world_model.testing import (
    NaiveVarlen,
    context_features,
    context_reference,
    final_hidden,
    naive_attention,
    played_frames,
    random_contexts,
    random_segment,
    row_state,
    row_zero_done,
    small_schema,
    tiny_model,
    window_segment,
)
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive, SdpaVarlen
from priml.model.transformer.block import TransformerBlock


def test_small_schema_cuts_cells_and_scalars() -> None:
    schema = small_schema()
    assert schema.cell_slots == 3
    assert len(schema.scalar_names) == len(schema.scalar_ranges) == 4


def test_tiny_models_are_equal_copies_and_restore_the_rng() -> None:
    schema = small_schema()
    first = tiny_model(schema)
    after_first = torch.rand(3)
    second = tiny_model(schema)
    assert torch.equal(torch.rand(3), after_first)
    assert first is not second
    for (name, a), (_, b) in zip(
        first.state_dict().items(),
        second.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(a, b), name
    assert not first.training
    untied = tiny_model(schema, untied_output=True, global_layers=2, local_layers=2)
    assert len(untied.transformer.blocks) == 2
    assert untied.state_dict().keys() != first.state_dict().keys()


def test_random_segment_fits_the_schema_and_its_terminal_flag() -> None:
    schema = small_schema()
    open_ = random_segment(schema, 3, seed=0)
    assert open_.cells.shape[:2] == (4, 3)
    assert open_.done.tolist() == [False] * 3
    assert open_.starts_episode
    closed = random_segment(schema, 3, seed=0, starts=False, terminal=True)
    assert closed.cells.shape[0] == 3
    assert closed.done.tolist() == [False, False, True]
    assert not closed.starts_episode
    for column, field in enumerate(schema.cell_fields):
        assert int(open_.cells[..., column].max()) < field.valid


def test_row_zero_done_forces_only_row_zeros_act_done() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator(),
    )
    control = row_zero_done(engine.control, done=True)
    assert control.job_forced[:, 1].tolist() == [True, False]
    assert int(control.job_tokens[0, 1, 0]) == done_id(done=True)


def test_state_of_copies_one_row_and_cuts_its_cache() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator(),
    )
    engine.state.length[1] = 3
    state = row_state(engine, row=1)
    assert state["keys"].shape[1] == 3
    assert state["values"].shape[1] == 3
    state["length"].fill_(7)
    assert int(engine.state.length[1]) == 3


def test_played_frames_record_each_decision_up_to_the_end() -> None:
    with eager(world=tiny_world):
        frames = played_frames(world_seed=3, decisions=4)
    assert frames.observations.shape == (4, OBS_SIZE)
    assert frames.masks.shape == (4, ATN_DIM)
    assert frames.cells.shape == (4, 99, 8)
    assert frames.aux.shape == (4, OBS_SIZE - 99 * 8)


def test_played_frames_stop_when_the_episode_ends() -> None:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 2)):
        frames = played_frames(world_seed=3, decisions=10)
    assert len(frames.observations) == 2


def _off_schema(state: object, cells: object, aux: object) -> int:
    """Stand in for ``token_frame_numba``, reporting a frame outside the schema."""
    del state, cells, aux
    return 1


def test_played_frames_refuse_a_frame_outside_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(testing, "token_frame_numba", _off_schema)
    with (
        eager(world=tiny_world),
        pytest.raises(
            ValueError,
            match=r"^World 3 shows a frame outside the schema\.$",
        ),
    ):
        played_frames(world_seed=3, decisions=2)


def test_random_contexts_split_prefix_from_steps() -> None:
    lengths = torch.tensor([[1, 2], [1, 1]])
    anchored = torch.tensor([[True, True], [False, False]])
    contexts = random_contexts(
        lengths,
        anchored,
        schema=small_schema(),
        slots=3,
        counts=torch.tensor([0, 2]),
        seed=5,
    )
    assert contexts.cells.shape[:2] == (2, 2)
    assert contexts.prefix_cells.shape[:2] == (2, 3)
    segment = random_segment(small_schema(), 5, seed=5)
    assert torch.equal(contexts.prefix_cells[0], segment.cells[:3])
    assert float(contexts.prefix_previous_actions[0, 0]) == 0.0
    assert float(contexts.previous_actions[0, 0]) == float(segment.actions[2])


def test_window_segment_takes_the_inclusive_range() -> None:
    cells = torch.arange(5 * 3 * 2).view(5, 3, 2)
    aux = torch.arange(5 * 4).view(5, 4)
    actions = torch.arange(5)
    segment = window_segment(cells, aux, actions, first=1, last=3, anchored=True)
    assert torch.equal(segment.cells, cells[1:4])
    assert segment.actions.tolist() == [1, 2, 3]
    assert segment.reward.tolist() == [0, 0, 0]
    assert segment.starts_episode


def test_final_hidden_stops_at_the_tap_and_restores_the_stack() -> None:
    model = tiny_model(small_schema(), global_layers=2)
    blocks = model.transformer.blocks
    windows = [random_segment(small_schema(), 2, seed=1)]
    with torch.no_grad():
        hidden = final_hidden(model, layers=1, windows=windows)
        full = final_hidden(model, layers=2, windows=windows)
    assert model.transformer.blocks is blocks
    assert hidden.shape == (1, 5, 36)
    assert not torch.equal(hidden, full)
    # The tap's hook is gone: a third call is not stopped where the first was.
    with torch.no_grad():
        logits = model.logits(pack_windows([windows], t_g=5, s_max=1))
    assert logits.action.shape[0] == 1


def test_context_reference_equals_each_windows_own_features() -> None:
    model = tiny_model(small_schema())
    contexts = random_contexts(
        torch.tensor([[1, 2]]),
        torch.tensor([[True, True]]),
        schema=small_schema(),
        slots=2,
        counts=torch.tensor([0]),
        seed=3,
    )
    with torch.no_grad():
        got = context_reference(model, contexts, layers=1)
        cells = torch.cat([contexts.prefix_cells, contexts.cells], dim=1)[0]
        aux = torch.cat([contexts.prefix_aux, contexts.aux], dim=1)[0]
        previous = torch.cat(
            [contexts.prefix_previous_actions, contexts.previous_actions],
            dim=1,
        ).long()[0]
        taken = torch.cat([previous[1:], previous[:1] * 0])
        second = window_segment(cells, aux, taken, first=2, last=3, anchored=True)
        want = context_features(model, layers=1, windows=[second])
    assert got.shape == (1, 2, 36)
    # One window packed alone versus two packed together: the padded batch
    # reduces in another order, as context_test.py's replay comparisons allow.
    torch.testing.assert_close(got[0, 1], want[0], rtol=0, atol=2e-6)


def test_naive_varlen_matches_sdpa_varlen() -> None:
    generator = torch.Generator().manual_seed(0)
    # Two segments of 2 and 3 tokens, 4 heads over 2 key heads, width 5.
    q = torch.randn(5, 4, 6, generator=generator)
    k = torch.randn(5, 2, 6, generator=generator)
    v = torch.randn(5, 2, 6, generator=generator)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    naive = NaiveVarlen(NaiveVarlen.Config())
    fused = SdpaVarlen(SdpaVarlen.Config())
    torch.testing.assert_close(
        naive(q, k, v, cu_seqlens=cu),
        fused(q, k, v, cu_seqlens=cu),
        rtol=1e-5,
        atol=1e-6,
    )


def test_naive_attention_swaps_every_kernel_of_a_block() -> None:
    plain = TransformerBlock.Config()
    plain.attn = Attention.Config()
    naive_attention(plain)
    assert isinstance(plain.attn, Attention.Config)
    assert isinstance(plain.attn.attn_kernel, SdpaNaive.Config)
    varlen = TransformerBlock.Config()
    varlen.attn = VarlenAttention.Config()
    naive_attention(varlen)
    assert isinstance(varlen.attn, VarlenAttention.Config)
    assert isinstance(varlen.attn.attn_kernel, NaiveVarlen.Config)
    decoder = DecoderBlock.Config()
    decoder.attn = VarlenAttention.Config()
    naive_attention(decoder)
    assert isinstance(decoder.attn, VarlenAttention.Config)
    assert isinstance(decoder.attn.attn_kernel, NaiveVarlen.Config)
    assert isinstance(decoder.cross_attn, Attention.Config)
    assert isinstance(decoder.cross_attn.attn_kernel, SdpaNaive.Config)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
