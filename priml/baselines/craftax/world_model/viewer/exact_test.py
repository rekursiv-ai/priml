"""Check exact game bundles: an episode's frames, creature health, hash checks, and files."""

from pathlib import Path
from typing import Final

import dataclasses
import gzip
import hashlib
import shutil
import subprocess

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game.mobs import (
    MELEE_HEALTH,
    PASSIVE_HEALTH,
    RANGED_HEALTH,
)
from priml.baselines.craftax.game.state import (
    MAX_ACHIEVEMENT_RETURN,
    Action,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, play_numba
from priml.baselines.craftax.lib.arrays import int_rows, ints, typed
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Episode
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.viewer import exact
from priml.baselines.craftax.world_model.viewer.exact import (
    Manifest,
    Replayed,
    frame_dtype,
    health_dtype,
    read_frame,
    read_health,
    read_manifest,
    replay_episode,
    replay_frames,
    write_bundle,
)


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def episode() -> Episode:
    """Return a 404-decision random-play episode: two stride hashes and a final one."""
    return replay.record(world_seed=4, sampling_seed=2, max_decisions=1_000)


@pytest.fixture(scope="module")
def played(episode: Episode) -> Replayed:
    return replay_episode(episode)


def test_each_sleep_gives_a_frame_every_stride_ticks_it_sleeps_through(
    episode: Episode,
    played: Replayed,
) -> None:
    slept = replay_episode(episode, sleep_stride=4)
    # The decision frames do not change, and only a sleep's step adds frames.
    assert slept.frames.tobytes() == played.frames.tobytes()
    assert (len(played.sleeps), len(played.sleep_frames)) == (0, 0)
    actions = episode.actions.numpy()
    assert len(slept.sleeps) >= 3
    sleeps = int_rows(slept.sleeps)
    assert all(actions[d] == Action.SLEEP.value for d, _, _ in sleeps)
    counts = [(ticks - 1) // 4 for _, ticks, _ in sleeps]
    assert slept.sleeps[:, 2].tolist() == np.cumsum([0, *counts[:-1]]).tolist()
    assert len(slept.sleep_frames) == len(slept.sleep_health) == sum(counts)
    assert slept.sleep_facings.shape == (sum(counts), 99)
    frames = slept.sleep_frames
    assert frames["sleeping"].all()
    for (decision, _, first), count in zip(sleeps, counts, strict=True):
        own = frames[first : first + count]
        before = read_frame(slept.frames, decision).tick_before
        # A frame after the 4th tick, the 8th, ...: the world's tick counts them.
        assert own["tick_before"].tolist() == [
            before + 4 * k for k in range(1, count + 1)
        ]
        assert np.equal(own["step"], decision).all()
        assert np.equal(own["row"], read_frame(slept.frames, decision).row).all()


def test_a_sleep_whose_ticks_end_elsewhere_is_refused(episode: Episode) -> None:
    states, rng = replay.load(replay.initial(episode).state)
    stats = new_stats(1)
    actions = ints(episode.actions.numpy())
    first = actions.index(Action.SLEEP.value)
    for action in actions[:first]:
        play_numba(env_state(states, 0), rng, env_stats(stats, 0), action, Rules())
    # The world before the sleep is not where its ticks end.
    with pytest.raises(ValueError, match="ends off its collapsed step"):
        exact._sleep_frames(
            states.copy(),
            rng.copy(),
            stats.copy(),
            action=Action.SLEEP.value,
            stride=4,
            after=(states, rng),
        )


def test_the_facings_hold_each_projectile_shown_in_view_by_its_cell() -> None:
    states, _ = replay.reset_world(4)
    state = env_state(states, 0)
    level = state.player_level
    row, col = state.player_position
    typed(states["light_map"], np.uint8)[0, level, :, :] = 255
    # Two arrows of the player's and an enemy's arrow in view, and one out of it.
    arrows, enemy = state.player_projectiles[level], state.mob_projectiles[level]
    arrow_directions = state.player_projectile_directions
    enemy_directions = state.mob_projectile_dirs
    for slot, (at, direction) in enumerate(
        (((row, col - 2), (0, -1)), ((row + 1, col), (1, 0)), ((row + 6, col), (0, 1))),
    ):
        arrows.mask[slot], arrows.type_id[slot] = 1, 0
        arrows.position[slot, 0], arrows.position[slot, 1] = at
        arrow_directions[level, slot, 0], arrow_directions[level, slot, 1] = direction
    enemy.mask[0], enemy.type_id[0] = 1, 0
    enemy.position[0, 0], enemy.position[0, 1] = row - 1, col + 3
    enemy_directions[level, 0, 0], enemy_directions[level, 0, 1] = 0, 1
    frames, health = np.zeros(1, frame_dtype()), np.zeros(1, health_dtype())
    frame, creatures = exact._record(frames, 0), exact._record(health, 0)
    facings = np.zeros(99, np.uint8)
    exact._before(frame, creatures, facings, states)
    expected = np.zeros(99, np.uint8)
    expected[4 * 11 + 3] = Action.LEFT.value << 4
    expected[5 * 11 + 5] = Action.DOWN.value << 4
    expected[3 * 11 + 8] = Action.RIGHT.value
    np.testing.assert_array_equal(facings, expected)
    enemy_directions[level, 0, 0], enemy_directions[level, 0, 1] = 1, 1
    with pytest.raises(ValueError, match="flies"):
        exact._before(frame, creatures, facings, states)


def test_frame_and_health_records_keep_their_byte_layout() -> None:
    assert (frame_dtype().itemsize, health_dtype().itemsize) == (7_858, 120)
    fields = frame_dtype().fields
    assert fields is not None
    # games.mjs reads the action, tick, score and achievement count at these offsets.
    offsets = {name: entry[1] for name, entry in fields.items()}
    assert (offsets["action"], offsets["tick_before"], offsets["score"]) == (12, 4, 20)
    assert (offsets["achievements"], offsets["map"], offsets["mobs"]) == (
        832,
        885,
        7_798,
    )


def test_every_decision_has_a_frame_and_the_last_board_follows(
    episode: Episode,
    played: Replayed,
) -> None:
    frames = played.frames
    decisions = len(episode.actions)
    assert decisions == 404
    assert len(frames) == len(played.health) == decisions + 1
    assert played.facings.shape == (decisions + 1, 99)
    assert frames["step"].tolist() == list(range(decisions + 1))
    assert frames["action"].tolist() == [*episode.actions.tolist(), 255]
    assert frames["terminal"].tolist() == [0] * (decisions - 1) + [1, 1]
    assert played.ended
    # A decision's after-values are the next frame's before-values.
    for name in ("tick", "floor", "health", "mana"):
        after, before = frames[f"{name}_after"], frames[f"{name}_before"]
        assert np.array_equal(after[:-1], before[1:])
        assert after[-1] == before[-1]
    # Random play crafts no armour, so its rewards are the achievements it
    # unlocks, and -1 at its death.
    unlocked = episode.reward.clamp(min=0).cumsum(0).tolist()
    assert frames["score"].tolist() == [*unlocked, unlocked[-1]]
    hashes = episode.hashes.numpy().view(np.uint64)
    assert np.array_equal(played.hashes[[255, -1]], hashes[1:])


def test_frames_hold_what_the_policy_saw(episode: Episode, played: Replayed) -> None:
    frames = played.frames[:-1]
    decisions = len(episode.actions)
    cells = episode.cells.reshape(decisions, -1).numpy()
    assert np.array_equal(frames["observation"], cells)
    names = craftax_schema().scalar_names
    aux = episode.aux.long().numpy()
    for name in ("food", "drink", "energy", "floor", "sleeping"):
        field = "floor_before" if name == "floor" else name
        assert np.array_equal(frames[field], aux[:, names.index(name)]), name
    for slot, name in enumerate(
        ("wood", "stone", "coal", "iron", "diamond", "sapling"),
    ):
        assert np.array_equal(
            frames["inventory"][:, slot],
            aux[:, names.index(name)],
        ), name
    facing = aux[
        :,
        [names.index(f"facing_{d}") for d in ("left", "right", "up", "down")],
    ]
    assert np.array_equal(frames["direction"], facing.argmax(-1) + 1)
    assert np.array_equal(frames["health_max"], 8 + aux[:, names.index("strength")])


def test_the_map_and_creatures_are_the_players_floor(played: Replayed) -> None:
    seen = 0
    for t in range(len(played.frames)):
        frame, health = read_frame(played.frames, t), read_health(played.health, t)
        # ``read_frame``'s map: the floor's 48 x 48 cells, each block, item and light.
        board = np.asarray(frame.map).reshape(48, 48, 3)
        view = np.asarray(frame.observation).reshape(9, 11, 8)
        row, col = int(frame.row), int(frame.col)
        centre = view[4, 5, :]
        if centre[2]:
            assert (centre[0], centre[1]) == (
                board[row, col, 0],
                board[row, col, 1] + 1,
            )
        count = int(frame.mob_count)
        mobs = int_rows(np.asarray(frame.mobs, np.int64)[: 4 * count].reshape(count, 4))
        listed: set[tuple[int, int, int]] = set()
        for i, (klass, kind, mob_row, mob_col) in enumerate(mobs):
            r, c = mob_row - row + 4, mob_col - col + 5
            if 0 <= r < 9 and 0 <= c < 11 and view[r, c, 2]:
                listed.add((r, c, klass))
            if klass < 3:
                table = (MELEE_HEALTH, PASSIVE_HEALTH, RANGED_HEALTH)[klass]
                assert health.maximum[i] == table[kind] > 0
                assert 0 < health.current[i] <= health.maximum[i]
            else:
                assert health.maximum[i] == health.current[i] == 0
        assert not np.any(health.maximum[count:])
        cells = {(r, c, k) for r, c, k in int_rows(np.argwhere(view[:, :, 3:]))}
        assert cells == listed
        seen += len(listed)
    assert seen > 0


def test_a_prefix_ends_where_it_is_cut(episode: Episode, played: Replayed) -> None:
    prefix = replay_episode(episode, limit=300)
    assert len(prefix.frames) == 301
    assert prefix.frames[:-1].tobytes() == played.frames[:300].tobytes()
    assert not prefix.ended
    final = read_frame(prefix.frames, -1)
    assert (final.step, final.action, final.terminal) == (300, 255, 0)
    for name in ("tick_before", "health_before", "observation", "map", "mobs"):
        assert np.array_equal(prefix.frames[-1:][name], played.frames[300:301][name]), (
            name
        )
    assert prefix.health[-1:].tobytes() == played.health[300:301].tobytes()


@pytest.mark.parametrize("index", [0, 1, -1])
def test_a_recorded_hash_the_replay_misses_is_refused(
    episode: Episode,
    index: int,
) -> None:
    hashes = episode.hashes.clone()
    hashes[index] ^= 1
    with pytest.raises(ValueError, match="hash"):
        replay_episode(dataclasses.replace(episode, hashes=hashes))


def test_an_episode_that_ends_before_its_last_action_is_refused(
    episode: Episode,
) -> None:
    with pytest.raises(ValueError, match="ended at decision 403"):
        replay_frames(
            *replay.load(replay.initial(episode).state),
            actions=np.append(episode.actions.numpy(), 0),
        )
    with pytest.raises(ValueError, match="at least one decision"):
        replay_episode(episode, limit=0)


def test_a_complete_record_must_end_at_its_last_decision(
    episode: Episode,
    played: Replayed,
) -> None:
    with pytest.raises(ValueError, match="did not end at its last decision"):
        replay_episode(_cut(episode, played, decisions=300))


def test_a_truncated_record_must_not_end(episode: Episode) -> None:
    with pytest.raises(ValueError, match="though its record was truncated"):
        replay_episode(dataclasses.replace(episode, truncated=True))


def test_a_truncated_record_replays_to_its_cut(
    episode: Episode,
    played: Replayed,
) -> None:
    record = dataclasses.replace(_cut(episode, played, decisions=300), truncated=True)
    truncated = replay_episode(record)
    assert truncated.frames[:-1].tobytes() == played.frames[:300].tobytes()
    assert not truncated.ended


def test_a_creature_of_no_species_is_refused() -> None:
    states, rng = replay.reset_world(7)
    melee = env_state(states, 0).melee_mobs[0]
    melee.mask[0], melee.type_id[0] = 1, -1
    with pytest.raises(ValueError, match="class 0 has species -1"):
        replay_frames(states, rng, actions=np.zeros(1, dtype=np.uint8))


def test_a_bundle_holds_frames_health_and_their_digests(
    tmp_path: Path,
    played: Replayed,
) -> None:
    output = tmp_path / "game"
    write_bundle(played, output, title="Random play", mean_score=12.5)
    manifest = read_manifest((output / "manifest.json").read_text(), Manifest)
    final = read_frame(played.frames, -1)
    assert manifest.schema_name == "craftax-exact-game/v1"
    assert (manifest.frames, manifest.actions) == (405, 404)
    assert (manifest.tick, manifest.score) == (final.tick_before, final.score)
    assert manifest.achievements == final.achievements > 0
    assert manifest.description == "404 recorded actions"
    assert manifest.end_label == "Episode end"
    assert manifest.mean_score == 12.5
    assert manifest.mean_return_pct == 100 * 12.5 / float(MAX_ACHIEVEMENT_RETURN)
    for name, raw, packed, payload in (
        ("frames", manifest.frames_sha256, manifest.gzip_sha256, played.frames),
        ("health", manifest.health_sha256, manifest.health_gzip_sha256, played.health),
    ):
        compressed = (output / f"{name}.bin.gz").read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == packed
        assert gzip.decompress(compressed) == payload.tobytes()
        assert hashlib.sha256(payload.tobytes()).hexdigest() == raw
    with pytest.raises(FileExistsError):
        write_bundle(played, output, title="Again")


@pytest.mark.cli_node
@pytest.mark.skipif(shutil.which("node") is None, reason="The viewer builds with Node.")
def test_viewer_builder_accepts_an_exact_bundle(
    tmp_path: Path,
    played: Replayed,
) -> None:
    write_bundle(played, tmp_path / "game", title="Random play", mean_score=3.0)
    page = tmp_path / "page.html"
    node = shutil.which("node")
    assert node is not None
    command = [node, str(_CWD / "games.mjs"), "build", "--output", str(page)]
    command.append(str(tmp_path / "game"))
    subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603 -- Fixed argv: resolved node, our games.mjs.
    html = page.read_text()
    assert '"kind":"exact"' in html
    assert '"meanScore":3' in html
    (tmp_path / "game" / "health.bin.gz").write_bytes(gzip.compress(b"x"))
    broken = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603 -- Fixed argv: resolved node, our games.mjs.
    assert broken.returncode != 0
    assert "Health payload hash mismatch" in broken.stderr


def _cut(episode: Episode, played: Replayed, *, decisions: int) -> Episode:
    """Return ``episode``'s record cut after ``decisions``, its last hash the State then."""
    last = played.hashes[decisions - 1 : decisions].view(np.int64)
    kept = -(-decisions // 256)
    return dataclasses.replace(
        episode,
        actions=episode.actions[:decisions],
        hashes=torch.cat([episode.hashes[:kept], torch.from_numpy(last)]),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
