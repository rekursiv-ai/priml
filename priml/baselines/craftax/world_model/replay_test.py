"""Tests for token frames, record, replay, snapshots, branches, and truncation.

The original implementation checked its C replay library this way; the same
checks hold here without it.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import dataclasses

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game.observation import compute_observations_numba
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    INVENTORY_OBS_SIZE,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_MOB_TYPES,
    OBS_SIZE,
    env_state,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    read_manifest,
    read_shard,
    write_shard,
)
from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    number_id,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def episode() -> Episode:
    return replay.record(world_seed=6, sampling_seed=2, max_decisions=2_000)


def test_recorded_episode_is_complete(episode: Episode) -> None:
    decisions = len(episode.actions)
    assert decisions > 256
    assert episode.hashes.shape == (-(-decisions // 256) + 1,)
    assert episode.hashes[0] == _signed(episode.receipt.initial_state_hash)
    assert episode.done[-1]
    assert not episode.done[:-1].any()
    assert episode.cells.shape == (decisions, 99, 8)
    assert episode.aux.shape == (decisions, 51)


def test_replay_matches_every_hash(episode: Episode) -> None:
    assert replay.verify(episode) == replay.MATCHED


def test_initial_hash_mismatch(episode: Episode) -> None:
    receipt = dataclasses.replace(
        episode.receipt,
        initial_state_hash=episode.receipt.initial_state_hash ^ 1,
    )
    assert replay.verify(dataclasses.replace(episode, receipt=receipt)) == -1


def test_mutated_action_reports_first_mismatched_hash(episode: Episode) -> None:
    decisions = len(episode.actions)
    for check in range(1, len(episode.hashes)):
        decision = min(256 * check, decisions) - 1
        if not episode.aux[decision, 44:46].any():
            break
    else:
        pytest.fail("Every checkpoint follows a sleeping or resting decision.")
    facing = int(episode.aux[decision, 31:35].argmax()) + 1
    original = int(episode.actions[decision])
    move = next(a for a in range(1, 5) if a not in {facing, original})
    actions = episode.actions.clone()
    actions[decision] = move
    mutated = dataclasses.replace(episode, actions=actions)
    assert replay.verify(mutated) == check + 1


def test_regenerated_frames_equal_recorded(episode: Episode) -> None:
    regenerated = replay.replay(episode)
    for name in ("cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(regenerated, name)),
            cast("torch.Tensor", getattr(episode, name)),
        ), name


def test_record_is_deterministic(episode: Episode) -> None:
    replay.record(world_seed=8, sampling_seed=3, max_decisions=2_000)
    again = replay.record(world_seed=6, sampling_seed=2, max_decisions=2_000)
    assert again.receipt == episode.receipt
    for name in ("actions", "hashes", "cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(again, name)),
            cast("torch.Tensor", getattr(episode, name)),
        ), name


def test_different_worlds_differ(episode: Episode) -> None:
    other = replay.record(world_seed=8, sampling_seed=2, max_decisions=2_000)
    assert other.receipt.initial_state_hash != episode.receipt.initial_state_hash


def test_terminal_hash_depends_on_the_terminal_state(episode: Episode) -> None:
    other = replay.record(world_seed=6, sampling_seed=3, max_decisions=2_000)
    assert other.hashes[0] == episode.hashes[0]
    assert other.hashes[-1] != episode.hashes[-1]


@pytest.mark.parametrize("change", [-1, 1])
def test_episode_of_wrong_length_fails_last_hash(episode: Episode, change: int) -> None:
    decisions = len(episode.actions) + change
    assert -(-decisions // 256) + 1 == len(episode.hashes)
    actions = torch.cat([episode.actions, torch.zeros(1, dtype=torch.uint8)])
    changed = dataclasses.replace(episode, actions=actions[:decisions])
    assert replay.verify(changed) == len(episode.hashes)


def test_unfinished_episode_is_rejected() -> None:
    with pytest.raises(ValueError, match="terminal"):
        replay.record(world_seed=6, sampling_seed=2, max_decisions=3)


def test_invalid_arguments_are_rejected(episode: Episode) -> None:
    receipt = dataclasses.replace(episode.receipt, world_seed=1 << 32)
    with pytest.raises(ValueError, match="arguments"):
        replay.verify(dataclasses.replace(episode, receipt=receipt))
    with pytest.raises(ValueError, match="arguments"):
        replay.verify(dataclasses.replace(episode, hashes=episode.hashes[:-1]))
    actions = episode.actions.clone()
    actions[0] = 43
    with pytest.raises(ValueError, match="arguments"):
        replay.verify(dataclasses.replace(episode, actions=actions))


def test_frames_of_mismatched_episode_raise(episode: Episode) -> None:
    hashes = episode.hashes.clone()
    hashes[-1] ^= 1
    with pytest.raises(ValueError, match=f"hash {len(hashes) - 1} "):
        replay.replay(dataclasses.replace(episode, hashes=hashes))


def test_token_values_lie_in_schema(episode: Episode) -> None:
    schema = craftax_schema()
    for index, field in enumerate(schema.cell_fields):
        assert int(episode.cells[..., index].max()) < field.valid, field.name
    low, high = torch.tensor(schema.scalar_ranges).T - number_id(0)
    assert bool(((episode.aux >= low) & (episode.aux <= high)).all())
    assert int(episode.reward.min()) >= -1
    assert int(episode.reward[-1]) == -1


def test_archive_round_trip_replays(episode: Episode, tmp_path: Path) -> None:
    write_shard(tmp_path, index=0, episodes=[episode], provenance={"test": "1"})
    (restored,) = read_shard(tmp_path, read_manifest(tmp_path)[0])
    assert restored.receipt == episode.receipt
    assert replay.verify(restored) == replay.MATCHED
    regenerated = replay.replay(restored)
    for name in ("actions", "hashes", "cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(regenerated, name)),
            cast("torch.Tensor", getattr(episode, name)),
        ), name


def test_snapshots_lie_on_the_stride(episode: Episode) -> None:
    taken = replay.snapshots(episode, stride=256)
    decisions = len(episode.actions)
    assert [s.decision for s in taken] == list(range(0, decisions, 256))
    assert len({s.state for s in taken}) == len(taken)
    assert all(len(s.state) == replay.SNAPSHOT_BYTES for s in taken)


def test_a_snapshot_is_the_state_then_the_stream_as_the_c_library_lays_them_out() -> (
    None
):
    # The C library's replay_snapshot_size is sizeof(State) + sizeof(unsigned int).
    assert replay.SNAPSHOT_BYTES == 80_248 + 4
    states, rng = replay.reset_world(6)
    rng[0] = 0x0102_0304
    saved = replay.save(states, rng)
    assert saved[-4:] == bytes([4, 3, 2, 1])
    loaded, stream = replay.load(saved)
    assert loaded.tobytes() == states.tobytes()
    assert np.array_equal(stream, rng)


def test_initial_snapshot_is_the_reset_world(episode: Episode) -> None:
    assert replay.initial(episode) == replay.snapshots(episode, stride=256)[0]
    receipt = dataclasses.replace(
        episode.receipt,
        initial_state_hash=episode.receipt.initial_state_hash ^ 1,
    )
    with pytest.raises(ValueError, match="initial state hash"):
        replay.initial(dataclasses.replace(episode, receipt=receipt))


@pytest.mark.parametrize("offset", [0, 17])
def test_segment_from_every_snapshot_equals_recorded_frames(
    episode: Episode,
    offset: int,
) -> None:
    decisions = len(episode.actions)
    for snapshot in replay.snapshots(episode, stride=256):
        start = snapshot.decision + offset
        stop = min(start + 300, decisions)
        frames = replay.segment(episode, start=start, stop=stop, snapshot=snapshot)
        _assert_frames(frames, episode, start=start, stop=stop)


def test_segment_from_the_seed_equals_recorded_frames(episode: Episode) -> None:
    decisions = len(episode.actions)
    frames = replay.segment(episode, start=100, stop=decisions)
    _assert_frames(frames, episode, start=100, stop=decisions)


def test_damaged_snapshot_fails_its_hash(episode: Episode) -> None:
    snapshot = replay.snapshots(episode, stride=256)[1]
    state = bytearray(snapshot.state)
    state[0] ^= 1
    damaged = dataclasses.replace(snapshot, state=bytes(state))
    with pytest.raises(ValueError, match="hash 1 "):
        replay.segment(episode, start=256, stop=300, snapshot=damaged)


def test_snapshots_verify_the_whole_episode(episode: Episode) -> None:
    hashes = episode.hashes.clone()
    hashes[-1] ^= 1
    with pytest.raises(ValueError, match=f"hash {len(hashes) - 1} "):
        replay.snapshots(dataclasses.replace(episode, hashes=hashes), stride=256)


def test_segment_and_snapshot_arguments_are_checked(episode: Episode) -> None:
    decisions = len(episode.actions)
    snapshot = replay.snapshots(episode, stride=256)[1]
    with pytest.raises(ValueError, match="stride"):
        replay.snapshots(episode, stride=300)
    with pytest.raises(ValueError, match="segment"):
        replay.segment(episode, start=200, stop=300, snapshot=snapshot)
    with pytest.raises(ValueError, match="segment"):
        replay.segment(episode, start=10, stop=decisions + 1)
    with pytest.raises(ValueError, match="segment"):
        replay.segment(episode, start=10, stop=10)
    unaligned = dataclasses.replace(snapshot, decision=300)
    with pytest.raises(ValueError, match="snapshot"):
        replay.segment(episode, start=300, stop=301, snapshot=unaligned)
    with pytest.raises(ValueError, match="snapshot"):
        replay.segment(
            episode,
            start=256,
            stop=300,
            snapshot=dataclasses.replace(snapshot, state=b""),
        )


def test_truncated_record_replays_without_a_terminal(episode: Episode) -> None:
    # A stall cap cuts a training episode after decision 255, where hash 1 is
    # the state that follows it.
    cut = dataclasses.replace(
        episode,
        actions=episode.actions[:256],
        hashes=episode.hashes[:2],
    )
    truncated = dataclasses.replace(cut, truncated=True)
    regenerated = replay.replay(truncated)
    for name in ("cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(regenerated, name)),
            cast("torch.Tensor", getattr(episode, name))[:256],
        )
    assert not regenerated.done.any()
    assert len(replay.snapshots(truncated, stride=256)) == 1
    frames = replay.segment(truncated, start=100, stop=256)
    _assert_frames(frames, episode, start=100, stop=256)
    with pytest.raises(ValueError, match="end differs"):
        replay.replay(cut)
    with pytest.raises(ValueError, match="end differs"):
        replay.replay(dataclasses.replace(episode, truncated=True))


def test_an_end_the_record_lacks_is_named_and_verify_keeps_the_c_librarys_status(
    episode: Episode,
) -> None:
    # Every recorded hash matches; only the record's truncation flag is false.
    lying = dataclasses.replace(episode, truncated=True)
    last = len(episode.actions) - 1
    # The C library reports an end mismatch as the first hash after it (replay.h).
    assert replay.verify(lying) == last // 256 + 2
    with pytest.raises(ValueError, match="end differs from its record's"):
        replay.replay(lying)


def test_branch_replays_from_its_origin(episode: Episode) -> None:
    parent_state = replay.snapshots(episode, stride=256)[1].state
    branch = Episode(
        receipt=dataclasses.replace(
            episode.receipt,
            initial_state_hash=int(episode.hashes[1]) % (1 << 64),
        ),
        actions=episode.actions[256:],
        hashes=episode.hashes[1:],
        cells=episode.cells[256:],
        aux=episode.aux[256:],
        reward=episode.reward[256:],
        done=episode.done[256:],
        summary={},
        origin=replay.origin(episode, decision=256),
    )
    assert len(branch.origin) == len(parent_state)
    assert replay.initial(branch).state == parent_state
    assert replay.verify(branch) == replay.MATCHED
    regenerated = replay.replay(branch)
    for name in ("cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(regenerated, name)),
            cast("torch.Tensor", getattr(branch, name)),
        ), name
    taken = replay.snapshots(branch, stride=256)
    assert taken[0].state == parent_state
    tail = len(branch.actions)
    _assert_frames(
        replay.segment(branch, start=5, stop=tail),
        branch,
        start=5,
        stop=tail,
    )
    damaged = bytearray(branch.origin)
    damaged[0] ^= 1
    with pytest.raises(ValueError, match="initial state hash"):
        replay.replay(dataclasses.replace(branch, origin=bytes(damaged)))


def test_origin_of_a_later_decision_steps_from_a_snapshot(episode: Episode) -> None:
    snapshot = replay.snapshots(episode, stride=256)[1]
    assert replay.origin(episode, decision=300, snapshot=snapshot) == replay.origin(
        episode,
        decision=300,
    )
    with pytest.raises(ValueError, match="decision"):
        replay.origin(episode, decision=len(episode.actions))


def test_origin_at_a_snapshots_own_decision_checks_its_hash(episode: Episode) -> None:
    # The start's state presented as decision 256's: no step runs before it.
    wrong = replay.Snapshot(decision=256, state=replay.initial(episode).state)
    with pytest.raises(ValueError, match="hash 1 "):
        replay.origin(episode, decision=256, snapshot=wrong)


@pytest.mark.parametrize("seed", [0, 73, 8191])
def test_the_cells_are_the_packed_observations_integers(seed: int) -> None:
    cells, _, observation = _frame(seed)
    assert np.array_equal(cells, observation[: replay.CELL_VALUES].astype(np.uint8))
    assert cells.any()


def test_a_fresh_player_has_full_meters_and_attributes_of_one() -> None:
    _, aux, observation = _frame(0)
    meters = replay.CELL_VALUES + 22
    # Health 9 HP sits on the 0.05-HP grid at token 180; the rest are counts.
    assert aux[22] == round(observation.item(meters) * 10 * 20)
    assert np.equal(aux[28:31], 1).all()
    assert aux[:11].sum() == 0


def test_rewards_are_integers_from_minus_one_through_234() -> None:
    assert replay.reward_token_numba(np.float32(-1.0)) == -1
    assert replay.reward_token_numba(np.float32(234.0)) == 234
    assert replay.reward_token_numba(np.float32(0.5)) == -2
    assert replay.reward_token_numba(np.float32(235.0)) == -2
    assert replay.reward_token_numba(np.float32(np.nan)) == -2


def test_the_schema_admits_every_cell_value_the_game_writes() -> None:
    # A frame's bounds are the schema's; the game's constants must fit them.
    assert replay.CELL_LIMIT.tolist() == [
        NUM_BLOCK_TYPES,
        NUM_ITEM_TYPES + 1,
        2,
        *[NUM_MOB_TYPES + 1] * 5,
    ]
    assert len(replay.AUX_LOW) == len(replay.AUX_HIGH) == INVENTORY_OBS_SIZE


def test_the_state_hash_uses_the_c_librarys_offset_basis() -> None:
    # The C library's basis is the standard FNV-1a one without its last digit, so an
    # empty input hashes to the basis itself.
    assert replay.fnv1a_numba(np.zeros(0, dtype=np.uint8)) == np.uint64(
        1_469_598_103_934_665_603,
    )
    assert replay.fnv1a_numba(np.array([0], dtype=np.uint8)) == np.uint64(
        1_469_598_103_934_665_603 * 1_099_511_628_211 % 2**64,
    )


def test_a_seed_beyond_32_bits_is_refused() -> None:
    with pytest.raises(ValueError, match="32 bits"):
        replay.reset_world(2**32)


def test_a_frame_outside_the_schema_is_refused_at_its_first_offender() -> None:
    states, _ = replay.reset_world(6)
    cells = np.zeros(replay.CELL_VALUES, dtype=np.uint8)
    aux = np.zeros(INVENTORY_OBS_SIZE, dtype=np.int16)
    state = env_state(states, 0)
    states["inventory"]["wood"][0] = replay.AUX_HIGH[0] + 1
    # Status is 1 plus the observation index: aux 0, wood, follows the cells.
    assert replay.token_frame_numba(state, cells, aux) == 1 + replay.CELL_VALUES
    states["inventory"]["wood"][0] = replay.AUX_HIGH[0]
    level = int(state.player_level)
    row, col = (int(v) for v in state.player_position)
    states["map"][0, level, row, col] = replay.CELL_LIMIT[0]
    # The player stands at the window's centre, row 4 and column 5 of 9 x 11.
    assert replay.token_frame_numba(state, cells, aux) == 1 + (4 * 11 + 5) * 8


def test_replay_raises_on_a_frame_outside_the_schema(episode: Episode) -> None:
    seed = episode.receipt.world_seed
    states, rng = replay.reset_world(seed)
    states["inventory"]["wood"][0] = replay.AUX_HIGH[0] + 1
    initial = int(replay.fnv1a_numba(states.view(np.uint8)))
    branch = dataclasses.replace(
        episode,
        receipt=dataclasses.replace(episode.receipt, initial_state_hash=initial),
        actions=episode.actions[:1],
        hashes=torch.tensor([_signed(initial), 0]),
        origin=replay.xor_bytes(
            replay.save(states, rng),
            right=replay.save(*replay.reset_world(seed)),
        ),
        truncated=True,
    )
    with pytest.raises(ValueError, match="outside the schema"):
        replay.segment(branch, start=0, stop=1)


def _frame(
    seed: int,
) -> tuple[NDArray[np.uint8], NDArray[np.int16], NDArray[np.float32]]:
    states, _ = replay.reset_world(seed)
    state = env_state(states, 0)
    cells = np.zeros(replay.CELL_VALUES, dtype=np.uint8)
    aux = np.zeros(INVENTORY_OBS_SIZE, dtype=np.int16)
    assert replay.token_frame_numba(state, cells, aux) == 0
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(ATN_DIM, dtype=np.uint8)
    compute_observations_numba(state, observation, mask)
    return cells, aux, observation


def _assert_frames(
    frames: replay.Frames,
    episode: Episode,
    *,
    start: int,
    stop: int,
) -> None:
    for name in ("cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(frames, name)),
            cast("torch.Tensor", getattr(episode, name))[start:stop],
        )


def _signed(value: int) -> int:
    """Return the int64 bits of a uint64 hash."""
    return value - (1 << 64) if value >= 1 << 63 else value


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
