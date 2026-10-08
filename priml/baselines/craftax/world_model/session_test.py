"""Check play sessions: marked token streams, overrides, branches, save and reload."""

from pathlib import Path

import dataclasses
import functools

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.engine import Engine
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.session import (
    Mark,
    Session,
    Stream,
)
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    row_zero_done,
    small_schema,
    tiny_model,
)


def _engine(*, rows: int = 1, seed: int = 0) -> Engine:
    """Return a small engine over the tiny model."""
    model = tiny_model(small_schema())
    return Engine(
        model,
        rows=rows,
        t_max=16,
        generator=torch.Generator().manual_seed(seed),
    )


def test_new_session_generates_first_frame_then_acts() -> None:
    session = Session(_engine(), row=0)
    session.prefill()
    first = session.stream()
    assert first.starts_episode
    assert first.cells.shape == (1, 3, 8)
    assert bool((first.frame_marks == Mark.MODEL).all())
    assert first.action.shape == (0,)
    turn = session.act(5)
    stream = session.stream()
    assert stream.cells.shape[0] == 2
    assert int(stream.action[0]) == 5
    assert stream.decision_marks[0].tolist() == [Mark.FORCED, Mark.MODEL, Mark.MODEL]
    assert int(stream.reward[0]) == int(turn.reward)
    assert bool(stream.done[0]) == bool(turn.done)
    assert torch.equal(stream.cells[1].long(), turn.cells)
    assert bool((stream.decision_logp[0] <= 0).all())
    # Every slot of the frame the act led to was generated and scored.
    assert bool((stream.frame_logp[1] < 0).all())


def test_override_changes_the_frame_the_model_encodes_and_marks_it() -> None:
    session = Session(_engine(), row=0)
    session.prefill()
    session.override(4, 7)
    session.override(1, [5, 1, 1, 0, 0, 0, 0, 0])
    stream = session.stream()
    engine = session.engine
    assert int(stream.aux[0, 1]) == 7
    assert int(engine.state.aux[0, 1]) == 7
    assert stream.cells[0, 1].tolist() == [5, 1, 1, 0, 0, 0, 0, 0]
    assert engine.state.cells[0, 1].tolist() == [5, 1, 1, 0, 0, 0, 0, 0]
    marks, logp = stream.frame_marks[0], stream.frame_logp[0]
    assert (int(marks[4]), int(marks[1])) == (Mark.OVERRIDE, Mark.OVERRIDE)
    assert int(marks[0]) == Mark.MODEL
    assert (float(logp[4]), float(logp[1])) == (0.0, 0.0)
    assert float(logp[0]) < 0


# The policy reads whole Craftax observations, so three frames of the full
# 152-slot schema are sampled slot by slot: 0.17 s warm on x86.
@pytest.mark.compute_large_fixture
def test_policy_autoplay_reads_the_current_overridden_frame() -> None:
    engine = Engine(
        tiny_model(craftax_schema()),
        rows=1,
        t_max=16,
        generator=torch.Generator().manual_seed(4),
    )
    session = Session(engine, row=0)
    session.prefill()
    session.override(99 + 22, 7)
    seen: list[Tensor] = []

    def policy(observation: Tensor) -> Tensor:
        seen.append(observation)
        return torch.tensor([6])

    session.autoplay(2, policy=policy)
    stream = session.stream()
    assert int(stream.aux[0, 22]) == 7
    assert len(seen) == 2
    for step, observation in enumerate(seen):
        assert torch.equal(
            observation,
            decode(stream.cells[step, None], stream.aux[step, None]),
        )
    assert stream.action.tolist() == [6, 6]
    assert bool((stream.decision_marks[:, 0] == Mark.FORCED).all())


def test_autoplay_marks_model_actions() -> None:
    session = Session(_engine(), row=0)
    session.prefill()
    session.autoplay(3)
    stream = session.stream()
    assert stream.action.shape == (3,)
    assert bool((stream.decision_marks[:, 0] == Mark.MODEL).all())
    assert stream.cells.shape[0] == 4


def test_prefill_from_archived_segment_marks_data() -> None:
    segment = random_segment(small_schema(), 2, seed=1)
    session = Session(_engine(), row=0)
    session.prefill(segment)
    stream = session.stream()
    assert torch.equal(stream.cells, segment.cells)
    assert torch.equal(stream.action, segment.actions)
    assert bool((stream.frame_marks == Mark.DATA).all())
    assert bool((stream.decision_marks == Mark.DATA).all())
    assert torch.equal(session.engine.state.cells[0], segment.cells[-1].long())
    assert int(session.engine.state.length[0]) == 5


def test_prefill_from_a_stream_restarts_after_its_last_episode_end() -> None:
    session = Session(_engine(), row=0)
    session.prefill(random_segment(small_schema(), 4, seed=3, starts=False))
    # Decision 1 ends an episode, so frame 2 follows a ``start`` position.
    stream = dataclasses.replace(
        session.stream(),
        done=torch.tensor([False, True, False, False]),
    )
    reloaded = Session(_engine(), row=0)
    reloaded.prefill(stream)
    state = reloaded.engine.state
    assert int(state.length[0]) == 1 + 2 * 2
    assert int(state.count[0]) == 2
    assert torch.equal(state.cells[0], stream.cells[-1].long())
    assert not reloaded.stream().starts_episode


def test_saved_session_reloads_to_identical_tokens(tmp_path: Path) -> None:
    session = Session(_engine(seed=2), row=0)
    session.prefill()
    session.act(3)
    session.override(0, 2)
    session.autoplay(4)
    session.stream().save(tmp_path / "session.pt")
    loaded = Stream.load(tmp_path / "session.pt")
    # The random model ends episodes often; the reload must cross a boundary.
    assert bool(loaded.done.any())
    for field in dataclasses.fields(loaded):
        assert torch.equal(
            torch.as_tensor(getattr(loaded, field.name)),
            torch.as_tensor(getattr(session.stream(), field.name)),
        ), field.name
    reloaded = Session(_engine(), row=0)
    reloaded.prefill(loaded)
    assert torch.equal(reloaded.engine.state.cells, session.engine.state.cells)
    assert torch.equal(reloaded.engine.state.length, session.engine.state.length)
    session.engine.generator.manual_seed(9)
    reloaded.engine.generator.manual_seed(9)
    original, again = session.act(1), reloaded.act(1)
    assert torch.equal(original.cells, again.cells)
    assert torch.equal(original.aux, again.aux)
    torch.testing.assert_close(
        session.stream().decision_logp[-1],
        reloaded.stream().decision_logp[-1],
    )


def test_one_rows_episode_ends_leave_another_sessions_samples_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams: list[Stream] = []
    for done in (False, True):
        engine = _engine(rows=2, seed=5)
        first = Session(engine, row=0)
        first.prefill()
        second = first.branch(1)
        monkeypatch.setattr(
            engine,
            "control",
            functools.partial(row_zero_done, engine.control, done=done),
        )
        for _ in range(3):
            first.act(1)
            second.act(2)
        assert first.stream().done.tolist() == [done] * 3
        streams.append(second.stream())
    for field in dataclasses.fields(Stream):
        one, two = (torch.as_tensor(getattr(stream, field.name)) for stream in streams)
        assert torch.equal(one, two), field.name


def test_branch_copies_the_row_and_its_stream() -> None:
    session = Session(_engine(rows=2), row=0)
    session.prefill()
    session.act(2)
    branch = session.branch(1)
    state = session.engine.state
    length = int(state.length[0])
    assert int(state.length[1]) == length
    assert torch.equal(state.keys[:, 1, :length], state.keys[:, 0, :length])
    assert torch.equal(state.cells[1], state.cells[0])
    assert torch.equal(branch.stream().cells, session.stream().cells)
    branch.act(4)
    assert branch.stream().action.tolist()[-1] == 4
    assert session.stream().action.shape == (1,)
    assert int(state.length[0]) == length


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
