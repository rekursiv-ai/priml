"""Check episode cutting, padding, summary statistics, and their comparison."""

import dataclasses

import pytest
import torch

from priml.baselines.craftax.world_model.archive import Episode, Receipt
from priml.baselines.craftax.world_model.dream import Rollout
from priml.baselines.craftax.world_model.episodes import (
    Episodes,
    archived_episodes,
    compare,
    departures,
    first_episodes,
    summarize,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib.codec import from_plain


def _rollout(done: list[list[bool]]) -> Rollout:
    """Return a rollout whose frame ``d`` of row ``b`` holds ``10 * b + d`` in aux 0."""
    rows, decisions = len(done), len(done[0])
    aux = torch.zeros(rows, decisions + 1, 51, dtype=torch.int16)
    aux[..., 0] = 10 * torch.arange(rows)[:, None] + torch.arange(decisions + 1)
    flags = torch.tensor(done)
    starts = torch.cat([torch.ones(rows, 1, dtype=torch.bool), flags], dim=1)
    zeros = torch.zeros(rows, decisions)
    return Rollout(
        cells=torch.zeros(rows, decisions + 1, 99, 8, dtype=torch.uint8),
        aux=aux,
        starts=starts,
        frame_logp=torch.zeros(rows, decisions + 1, 150),
        invalid=torch.zeros(rows, decisions + 1, 99, dtype=torch.bool),
        action=torch.arange(decisions, dtype=torch.uint8).expand(rows, -1),
        reward=torch.ones(rows, decisions, dtype=torch.int16),
        done=flags,
        action_logp=zeros,
        reward_logp=zeros,
        done_logp=zeros,
    )


def test_first_episodes_cut_each_row_at_its_first_done() -> None:
    episodes = first_episodes(_rollout([[False, True, False, True], [False] * 4]))
    assert episodes.length.tolist() == [2, 4]
    assert episodes.ended.tolist() == [True, False]
    # Frame d is the one decision d observed; frame D, after the last, is dropped.
    assert episodes.aux[..., 0].tolist() == [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert episodes.action.shape == episodes.reward.shape == (2, 4)


def _archived(decisions: int) -> Episode:
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=1,
            sampling_seed=2,
            initial_state_hash=3,
            arm=0,
            split=1,
        ),
        actions=torch.arange(decisions, dtype=torch.uint8),
        hashes=torch.zeros(1, dtype=torch.int64),
        cells=torch.ones(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.full((decisions, 51), 2, dtype=torch.int16),
        reward=torch.ones(decisions, dtype=torch.int16),
        done=done,
        summary={},
    )


def test_archived_episodes_pad_short_and_cut_long_episodes() -> None:
    episodes = archived_episodes([_archived(3), _archived(6), _archived(4)], horizon=4)
    assert episodes.length.tolist() == [3, 4, 4]
    # An episode of exactly the horizon ends inside it.
    assert episodes.ended.tolist() == [True, False, True]
    assert episodes.action.tolist()[:2] == [[0, 1, 2, 0], [0, 1, 2, 3]]
    assert episodes.cells.shape == (3, 4, 99, 8)
    assert bool((episodes.aux[1] == 2).all())


def test_archived_episodes_cut_by_the_stall_cap_never_ended() -> None:
    capped = dataclasses.replace(
        _archived(3),
        done=torch.zeros(3, dtype=torch.bool),
        truncated=True,
    )
    episodes = archived_episodes([_archived(3), capped], horizon=4)
    assert episodes.length.tolist() == [3, 3]
    assert episodes.ended.tolist() == [True, False]


# Episode 0 sees a grass cell and a melee mob, gets reward 1 then dies (-1). Episode 1
# moves every decision, reaches floor 2 at decision 3, and at decision 2 shows health
# above the maximum its strength allows.
def _episodes() -> Episodes:
    """Two episodes over a horizon of 4: one dies at decision 1, one survives."""
    names = [field.name for field in craftax_schema().cell_fields]
    cells = torch.zeros(2, 4, 99, 8, dtype=torch.uint8)
    cells[..., names.index("visibility")] = 1
    cells[..., names.index("block")] = 2
    cells[0, :, 0, names.index("melee")] = 1
    # Episode 1's first cell turns to stone (4) at decision 1 and back at 3.
    cells[1, 1:3, 0, names.index("block")] = 4
    # Episode 1 never sees its last row: unseen cells are all zeros.
    cells[1, :, 90:] = 0
    aux = torch.zeros(2, 4, 51, dtype=torch.int16)
    scalars = craftax_schema().scalar_names
    for name in ("dexterity", "strength", "intelligence"):
        aux[..., scalars.index(name)] = 1
    aux[1, 3, scalars.index("floor")] = 2
    # Health at its maximum is legal; one grid step above it is not.
    aux[1, 1, scalars.index("health")] = 20 * 9
    aux[1, 2, scalars.index("health")] = 20 * 9 + 1
    # Past episode 0's end: a deep floor and impossible health, both ignored.
    aux[0, 3, scalars.index("floor")] = 5
    aux[0, 3, scalars.index("health")] = 999
    return Episodes(
        cells=cells,
        aux=aux,
        action=torch.tensor([[0, 5, 9, 9], [1, 1, 1, 2]], dtype=torch.uint8),
        reward=torch.tensor([[1, -1, 7, 7], [0, 0, 0, 3]], dtype=torch.int16),
        length=torch.tensor([2, 4]),
        ended=torch.tensor([True, False]),
    )


def test_summarize_measures_survival_return_and_floors() -> None:
    summary = summarize(_episodes())
    assert summary["episodes"] == 2
    assert summary["horizon"] == 4
    assert summary["ended"] == 0.5
    assert from_plain(summary["survival"], dict[str, float]) == {"1": 1.0, "2": 0.5}
    # Rewards past an episode's length are ignored.
    assert summary["return_mean"] == pytest.approx((0 + 3) / 2)
    assert summary["rewards_per_1000"] == pytest.approx(1000 * 2 / 6)
    reached = from_plain(summary["floor_reached"], list[float])
    assert reached[:6] == [1.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    occupancy = from_plain(summary["floor_occupancy"], list[float])
    assert occupancy[0] == pytest.approx(5 / 6)
    assert occupancy[2] == pytest.approx(1 / 6)


def test_summarize_measures_actions_board_and_hud() -> None:
    summary = summarize(_episodes())
    actions = from_plain(summary["action_frequency"], list[float])
    assert len(actions) == 43
    assert actions[1] == pytest.approx(3 / 6)
    assert actions[9] == 0.0
    blocks = from_plain(summary["block_frequency"], list[float])
    assert blocks[4] == pytest.approx(2 / (2 * 99 + 4 * 90))
    mobs = from_plain(summary["mobs_per_frame"], dict[str, float])
    assert from_plain(mobs["melee"], float) == pytest.approx(2 / 6)
    assert from_plain(mobs["passive"], float) == 0.0
    # Pairs within episodes: 1 in episode 0, 3 in episode 1; cell 0 changes
    # twice, at 0 -> 1 and at 2 -> 3.
    assert summary["cell_change_rate"] == pytest.approx(2 / (4 * 99))
    assert summary["invalid_cell_rate"] == 0.0
    assert summary["hud_bound_violation_rate"] == pytest.approx(1 / 6)
    assert from_plain(
        from_plain(summary["aux_mean"], dict[str, float])["floor"],
        float,
    ) == (pytest.approx(2 / 6))


def test_summarize_flags_cells_no_observation_holds() -> None:
    episodes = _episodes()
    names = [field.name for field in craftax_schema().cell_fields]
    # Cell 0 of episode 0 already holds a melee mob.
    episodes.cells[0, 0, 0, names.index("passive")] = 1
    assert summarize(episodes)["invalid_cell_rate"] == pytest.approx(1 / (6 * 99))


def test_compare_reports_both_summaries_and_total_variation() -> None:
    same = compare(_episodes(), _episodes())
    assert from_plain(same["distance"], dict[str, float]) == {
        "action_tv": 0.0,
        "block_tv": 0.0,
        "floor_occupancy_tv": 0.0,
    }
    moved = _episodes()
    moved.action.fill_(42)
    distance = from_plain(compare(moved, _episodes())["distance"], dict[str, float])
    assert distance["action_tv"] == pytest.approx(1.0)
    assert set(compare(moved, _episodes())) == {"generated", "real", "distance"}


def test_departures_count_hud_violations_and_floor_changes_without_a_ladder() -> None:
    found = departures(_episodes())
    assert (found["episodes"], found["decisions"]) == (2, 6)
    rates = from_plain(found["hud_violation_rate_by_field"], dict[str, float])
    assert rates == {
        "health": pytest.approx(1 / 6),
        "mana": 0.0,
        "food": 0.0,
        "drink": 0.0,
        "energy": 0.0,
    }
    assert found["episodes_with_violation"] == 0.5
    assert found["first_violation_decision_median"] == 2.0
    # One decision pair of episode 0 and three of episode 1: one floor change,
    # from 0 to 2 after a move, so without a ladder and a jump of two.
    assert found["floor_changes_per_1000"] == 250.0
    assert found["floor_change_without_ladder_action_fraction"] == 1.0
    assert (found["floor_jump_gt1"], found["floor_changes"]) == (1, 1)
    # Episode 0 shows floor 5 only past its end.
    reach = from_plain(found["first_reach"], dict[str, dict[str, object]])
    assert reach["5"] == {"fraction": 0.0, "median_decision": None}
    alignment = from_plain(found["ladder_alignment"], dict[str, float])
    assert alignment["ladder_at_t+0"] == 0.0
    assert alignment["descend_actions"] == 0


def test_departures_find_a_descend_followed_by_its_floor() -> None:
    episodes = _episodes()
    episodes.action[1, 2] = 18
    episodes.aux[1, 3, craftax_schema().scalar_names.index("floor")] = 1
    found = departures(episodes)
    assert found["floor_change_without_ladder_action_fraction"] == 0.0
    assert found["floor_jump_gt1"] == 0
    alignment = from_plain(found["ladder_alignment"], dict[str, float])
    assert alignment["ladder_at_t+0"] == 1.0
    assert alignment["ladder_at_t-1"] == 0.0
    assert alignment["descend_actions"] == 1
    assert alignment["descend_followed_by_floor_up"] == 1.0


def test_ladder_alignment_reads_no_decision_outside_the_episode() -> None:
    episodes = _episodes()
    floor = craftax_schema().scalar_names.index("floor")
    # Episode 0 changes floor after decision 0 and has a ladder action only past
    # its end; episode 1 changes floor after decision 2 and has one at decision 0.
    episodes.aux[0, 1, floor] = 1
    episodes.action[0, 2] = 18
    episodes.action[1, 0] = 18
    alignment = from_plain(departures(episodes)["ladder_alignment"], dict[str, float])
    # Only episode 1's ladder, two decisions before its change, counts.
    assert alignment["ladder_at_t-2"] == 0.5
    assert alignment["ladder_at_t+2"] == 0.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
