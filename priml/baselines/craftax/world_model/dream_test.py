"""Check batched generated episodes and the invalid-cell rule."""

from typing import cast

import dataclasses
import functools

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.dream import (
    Rollout,
    dream,
    invalid_cells,
)
from priml.baselines.craftax.world_model.engine import (
    Control,
    Engine,
    Prefix,
    StartResult,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts.engine_checks import (
    prefix_of,
)
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    row_zero_done,
    small_schema,
    tiny_model,
)


def test_dream_from_start_records_every_frame_and_decision() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator().manual_seed(0),
    )
    rollout = dream(engine, decisions=6)
    assert rollout.cells.shape == (2, 7, 3, 8)
    assert rollout.aux.shape == (2, 7, 4)
    assert rollout.action.shape == rollout.reward.shape == rollout.done.shape == (2, 6)
    assert bool(rollout.starts[:, 0].all())
    # A frame begins an episode exactly when the decision before it ended one.
    assert torch.equal(rollout.starts[:, 1:], rollout.done)
    assert bool(rollout.done.any())
    for logp in (rollout.action_logp, rollout.reward_logp, rollout.done_logp):
        assert bool((logp <= 0).all())
    assert bool((rollout.frame_logp < 0).all())
    assert rollout.invalid.shape == (2, 7, 3)


def test_dream_from_per_row_prefixes_follows_recorded_actions() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    prefixes = [
        prefix_of(random_segment(schema, 3, seed=1)),
        None,
        prefix_of(random_segment(schema, 1, seed=2, starts=False)),
    ]
    engine = Engine(model, rows=3, t_max=16, generator=torch.Generator().manual_seed(1))
    actions = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rollout = dream(engine, decisions=3, prefixes=prefixes, actions=actions)
    assert torch.equal(rollout.action.long(), actions)
    assert rollout.starts[:, 0].tolist() == [False, True, False]
    assert bool((rollout.frame_logp[1, 0] < 0).all())
    for row, prefix in enumerate(prefixes):
        if prefix is None:
            continue
        assert torch.equal(rollout.cells[row, 0], prefix.cells[0, -1])
        assert bool((rollout.frame_logp[row, 0] == 0).all())
        # The first action's log-probability reads only this row's prefix.
        alone = Engine(model, rows=1, t_max=16, generator=torch.Generator())
        alone.prefill(torch.tensor([0]), prefix)
        control = alone.control()
        control.action_forced.fill_(value=True)
        control.action.fill_(int(actions[row, 0]))
        torch.testing.assert_close(
            rollout.action_logp[row, 0],
            alone.decide(control).action_logp[0],
        )


def test_dream_runs_a_start_job_only_after_an_episode_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=1,
        t_max=16,
        generator=torch.Generator().manual_seed(3),
    )
    calls: list[Tensor] = []
    start = engine.start

    def spy(control: Control) -> StartResult:
        calls.append(engine.state.needs_start.clone())
        return start(control)

    monkeypatch.setattr(engine, "start", spy)
    rollout = dream(engine, decisions=6)
    ended = int(rollout.done[0].sum())
    # This seed ends some episodes and continues others, so both paths run.
    assert 0 < ended < 6
    assert len(calls) == 1 + ended
    assert all(bool(needs.all()) for needs in calls)
    assert torch.equal(rollout.starts[:, 1:], rollout.done)
    assert bool((rollout.frame_logp < 0).all())


def test_dream_rows_do_not_depend_on_another_rows_episode_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model(small_schema())
    rollouts: list[Rollout] = []
    for done in (False, True):
        engine = Engine(
            model,
            rows=2,
            t_max=16,
            generator=torch.Generator().manual_seed(7),
        )
        monkeypatch.setattr(
            engine,
            "control",
            functools.partial(row_zero_done, engine.control, done=done),
        )
        rollouts.append(dream(engine, decisions=4))
    # Row 0 begins an episode after every decision in one run and never in the other.
    assert rollouts[0].done[0].tolist() == [False] * 4
    assert rollouts[1].done[0].tolist() == [True] * 4
    for field in dataclasses.fields(Rollout):
        one, two = (
            cast("torch.Tensor", getattr(rollout, field.name))[1]
            for rollout in rollouts
        )
        assert torch.equal(one, two), field.name


def test_dream_rejects_a_prefix_list_of_the_wrong_length() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator(),
    )
    with pytest.raises(ValueError, match="one per row"):
        dream(engine, decisions=1, prefixes=[None])


def test_dream_rejects_no_decisions() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator(),
    )
    with pytest.raises(ValueError, match="must be positive"):
        dream(engine, decisions=0)


def test_dream_reprefills_a_full_row_from_its_recorded_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema),
        rows=1,
        t_max=8,
        generator=torch.Generator().manual_seed(6),
    )
    prefix = prefix_of(random_segment(schema, 3, seed=4, starts=False))
    calls: list[tuple[int, Prefix]] = []
    prefill = engine.prefill

    def spy(rows: Tensor, kept: Prefix) -> None:
        calls.append((int(engine.state.count[rows[0]]), kept))
        prefill(rows, kept)

    monkeypatch.setattr(engine, "prefill", spy)
    # The prefix leaves room for one decision, so decision 1 needs a re-prefill
    # unless decision 0 ends the episode, which this seed does not sample.
    rollout = dream(engine, decisions=2, prefixes=[prefix])
    assert not bool(rollout.done[0, 0])
    assert len(calls) == 2
    count, kept = calls[1]
    cells = torch.cat([prefix.cells[0, :-1], rollout.cells[0]])
    aux = torch.cat([prefix.aux[0, :-1], rollout.aux[0]])
    actions = torch.cat([prefix.actions[0], rollout.action[0]])
    keep = 8 // 4
    assert torch.equal(kept.cells[0], cells[count - keep : count + 1].long())
    assert torch.equal(kept.aux[0], aux[count - keep : count + 1].long())
    assert torch.equal(kept.actions[0], actions[count - keep : count])


# The policy reads whole Craftax observations, so three frames of the full
# 152-slot schema are sampled slot by slot for two rows: 0.18 s warm on x86.
@pytest.mark.compute_large_fixture
def test_dream_feeds_a_policy_decoded_observations() -> None:
    engine = Engine(
        tiny_model(craftax_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator().manual_seed(2),
    )
    seen: list[Tensor] = []

    def policy(observation: Tensor) -> Tensor:
        seen.append(observation)
        return torch.tensor([7, 9])

    rollout = dream(engine, decisions=2, actions=policy)
    assert len(seen) == 2
    assert seen[0].shape == (2, 843)
    assert seen[0].dtype == torch.float32
    for step, observation in enumerate(seen):
        expected = decode(rollout.cells[:, step], rollout.aux[:, step])
        assert torch.equal(observation, expected)
    assert rollout.action.tolist() == [[7, 7], [9, 9]]


def test_invalid_cells_flag_what_no_observation_holds() -> None:
    schema = craftax_schema()
    cells = torch.tensor(
        [
            [2, 1, 1, 0, 0, 0, 0, 0],  # Visible grass.
            [0, 0, 0, 0, 0, 0, 0, 0],  # Unseen.
            [0, 0, 0, 0, 3, 0, 0, 0],  # Unseen, yet a passive mob.
            [2, 1, 1, 1, 1, 0, 0, 0],  # Melee and passive in one cell.
            [2, 1, 1, 0, 0, 2, 4, 5],  # Ranged mob plus projectiles.
        ],
    )
    assert invalid_cells(cells, schema=schema).tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
