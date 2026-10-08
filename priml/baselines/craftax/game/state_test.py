"""Tests for the constants against the C header and the state against C ``State``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    ACTION_OBS_SIZE,
    ATN_DIM,
    INVENTORY_DTYPE,
    LOG_DTYPE,
    MAX_ACHIEVEMENT_RETURN,
    MOB_SLOTS,
    MOBS_DTYPE,
    NO_ACTION,
    NOISE_PI2,
    NOISE_SQRT2,
    NUM_ACHIEVEMENTS,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_LEVELS,
    OBS_SIZE,
    OBS_TILE_CHANNELS,
    PADDING,
    PI,
    STATE_DTYPE,
    STATS_DTYPE,
    TRAINING_STATS_DTYPE,
    Achievement,
    Action,
    BlockType,
    EnvState,
    ItemType,
    env_state,
    env_stats,
    new_logs,
    new_states,
    new_stats,
    semantic_bytes,
)
from priml.lib.codec import loads


if TYPE_CHECKING:
    from collections.abc import Callable


def test_observation_geometry_matches_the_c_header() -> None:
    assert OBS_TILE_CHANNELS == 8
    assert OBS_SIZE == 843
    assert ATN_DIM == len(Action) == 43
    assert NUM_ACHIEVEMENTS == len(Achievement) == 67
    assert NUM_BLOCK_TYPES == len(BlockType) == 37
    assert NUM_ITEM_TYPES == len(ItemType) == 5


def test_enums_are_dense_from_zero() -> None:
    for enum in (BlockType, ItemType, Achievement, Action):
        assert [member.value for member in enum] == list(range(len(enum)))


def test_achievement_rewards_sum_to_the_ceiling_in_fp32() -> None:
    rewards = ACHIEVEMENT_REWARD_MAP
    assert rewards.dtype == np.float32
    assert rewards.shape == (NUM_ACHIEVEMENTS,)
    # The C max_achievement_return() is a sequential fp32 sum.
    total = np.float32(0.0)
    for i in range(len(rewards)):
        total = np.float32(total + np.float32(rewards.item(i)))
    assert total == MAX_ACHIEVEMENT_RETURN == np.float32(226.0)


def test_reward_map_spot_values_match_constants_h() -> None:
    rewards = ACHIEVEMENT_REWARD_MAP
    assert rewards[Achievement.COLLECT_WOOD] == 1.0
    assert rewards[Achievement.MAKE_DIAMOND_SWORD] == 3.0
    assert rewards[Achievement.ENTER_SEWERS] == 5.0
    assert rewards[Achievement.ENTER_FIRE_REALM] == 8.0
    assert rewards[Achievement.DEFEAT_LIZARD] == 5.0
    assert rewards[Achievement.DEFEAT_NECROMANCER] == 8.0
    assert rewards[Achievement.EAT_BAT] == 3.0
    assert rewards[Achievement.LEARN_FIREBALL] == 5.0
    assert rewards[Achievement.COLLECT_RUBY] == 3.0
    assert rewards[Achievement.DEFEAT_KNIGHT] == 5.0
    assert rewards[Achievement.DEFEAT_ARCHER] == 5.0


def test_float_literals_are_float32() -> None:
    assert NOISE_PI2.dtype == np.float32
    assert NOISE_SQRT2.dtype == np.float32
    assert PI.dtype == np.float32
    # The C literal 6.28318530717958647692f rounds to this float32.
    assert NOISE_PI2.view(np.uint32) == 0x40C90FDB


_CWD: Final = Path(__file__).resolve().parent
_LAYOUT_GOLDEN = _CWD.parent / "testdata" / "state_layout.json"
"""``offsetof``/``sizeof`` of every field, measured from the C structs."""


def _layout(struct: str) -> dict[str, tuple[int, int]]:
    golden = loads(_LAYOUT_GOLDEN.read_text(encoding="utf-8"))
    assert isinstance(golden, dict)
    table = golden[struct]
    assert isinstance(table, dict)
    out: dict[str, tuple[int, int]] = {}
    for name, entry in table.items():
        assert isinstance(entry, list)
        offset, size = entry
        assert isinstance(offset, int)
        assert isinstance(size, int)
        out[name] = (offset, size)
    return out


def _sizeof(struct: str) -> int:
    golden = loads(_LAYOUT_GOLDEN.read_text(encoding="utf-8"))
    assert isinstance(golden, dict)
    sizes = golden["sizeof"]
    assert isinstance(sizes, dict)
    size = sizes[struct]
    assert isinstance(size, int)
    return size


@pytest.mark.parametrize(
    ("struct", "dtype"),
    [
        ("Inventory", INVENTORY_DTYPE),
        ("Mobs", MOBS_DTYPE),
        ("State", STATE_DTYPE),
        ("Log", LOG_DTYPE),
    ],
)
def test_every_field_sits_at_its_c_offset_with_its_c_size(
    struct: str,
    dtype: np.dtype[np.void],
) -> None:
    expected = _layout(struct)
    assert dtype.fields is not None
    actual = {
        str(name): (entry[1], entry[0].itemsize)
        for name, entry in dtype.fields.items()
        if name != PADDING
    }
    assert list(actual) == list(expected), "field order differs from C"
    assert actual == expected
    assert dtype.itemsize == _sizeof(struct)
    # The C fields leave gaps; PADDING fills exactly those, so no byte is no field's.
    covered = np.zeros(dtype.itemsize, dtype=np.int64)
    for offset, size in actual.values():
        covered[offset : offset + size] += 1
    if PADDING in dtype.fields:
        padding, offset = dtype.fields[PADDING][:2]
        assert not covered[offset : offset + padding.itemsize].any()
        covered[offset : offset + padding.itemsize] += 1
    assert np.equal(covered, 1).all()


def test_state_is_80248_bytes_with_the_c_tail_padding() -> None:
    assert STATE_DTYPE.itemsize == 80_248
    assert STATE_DTYPE.fields is not None
    timestep_dtype, timestep_offset = STATE_DTYPE.fields["timestep"][:2]
    assert timestep_offset + timestep_dtype.itemsize == 80_244


def test_mobs_mask_is_one_byte_per_slot_like_c_bool() -> None:
    assert MOBS_DTYPE.fields is not None
    mask_dtype = MOBS_DTYPE.fields["mask"][0]
    assert mask_dtype.base == np.uint8
    assert mask_dtype.shape == (MOB_SLOTS,)


def test_the_out_of_range_aliases_are_the_next_field() -> None:
    # The chest's loot may write potions[6]; the potion shuffle may write
    # potion_mapping[6]. Each lands on the field the port aliases explicitly.
    assert INVENTORY_DTYPE.fields is not None
    potions_dtype, potions_offset = INVENTORY_DTYPE.fields["potions"][:2]
    assert potions_offset + potions_dtype.itemsize == INVENTORY_DTYPE.fields["books"][1]
    assert STATE_DTYPE.fields is not None
    mapping_dtype, mapping_offset = STATE_DTYPE.fields["potion_mapping"][:2]
    assert (
        mapping_offset + mapping_dtype.itemsize
        == STATE_DTYPE.fields["learned_spells"][1]
    )


def test_semantic_bytes_excludes_exactly_the_c_padding() -> None:
    mask = semantic_bytes()
    assert mask.shape == (80_248,)
    padding = np.flatnonzero(~mask).tolist()
    mobs_padding = [
        offset + level * 64 + 39
        for offset in (76_424, 77_000, 77_576, 78_152, 78_944)
        for level in range(NUM_LEVELS)
    ]
    assert padding == sorted([*mobs_padding, 80_244, 80_245, 80_246, 80_247])
    assert semantic_bytes(MOBS_DTYPE).tolist() == [*[True] * 39, False, *[True] * 24]
    assert semantic_bytes(INVENTORY_DTYPE).all()


def _method_copy(states: np.ndarray) -> np.ndarray:
    return states.copy()


def _fancy_index(states: np.ndarray) -> np.ndarray:
    return states[[0, 1]]


def _concatenate(states: np.ndarray) -> np.ndarray:
    return np.concatenate([states[:1], states[1:]])


@pytest.mark.parametrize(
    "copy",
    [np.copy, np.array, _method_copy, _fancy_index, _concatenate],
    ids=["np.copy", "np.array", "ndarray.copy", "fancy index", "concatenate"],
)
def test_a_numpy_copy_of_states_carries_their_padding(
    copy: Callable[[np.ndarray], np.ndarray],
) -> None:
    # The C env's state hash covers the padding bytes, so a copy must carry them;
    # numpy copies a structured array field by field and leaves a byte that is
    # no field's as the new allocation found it.
    states = new_states(2)
    states.view(np.uint8).reshape(2, -1)[:, ~semantic_bytes()] = 0xA5
    assert copy(states).tobytes() == states.tobytes()


def test_new_states_are_zeroed_and_independent() -> None:
    states = new_states(2)
    assert states.shape == (2,)
    assert states.tobytes() == bytes(2 * 80_248)
    states[0]["timestep"] = 5
    assert states[1]["timestep"] == 0


def test_new_logs_are_zeroed_fp32_sums() -> None:
    logs = new_logs(3)
    assert logs.tobytes() == bytes(3 * 328)
    assert logs["achievements"].shape == (3, NUM_ACHIEVEMENTS)


def test_the_previous_action_field_follows_the_packed_view_and_names_no_action() -> (
    None
):
    assert ACTION_OBS_SIZE == OBS_SIZE + 1 == 844
    assert NO_ACTION == ATN_DIM == len(Action)


def test_training_stats_lead_with_their_fields_then_hold_the_stats() -> None:
    """Led by the training fields, so Numba never takes them for a stats subtype."""
    assert STATS_DTYPE.fields is not None
    assert TRAINING_STATS_DTYPE.fields is not None
    extra = {
        name: (entry[1], entry[0].itemsize)
        for name, entry in TRAINING_STATS_DTYPE.fields.items()
        if name not in STATS_DTYPE.fields
    }
    assert extra == {
        "stall_limit": (0, 4),
        "last_gain": (4, 4),
        "escape": (8, 4),
        "branch": (12, 4),
        "branch_steps": (16, 8),
    }
    for name, entry in STATS_DTYPE.fields.items():
        assert TRAINING_STATS_DTYPE.fields[name][0] == entry[0], name
        assert TRAINING_STATS_DTYPE.fields[name][1] == entry[1] + 24, name
    assert TRAINING_STATS_DTYPE.itemsize == STATS_DTYPE.itemsize + 24


@pytest.mark.parametrize("dtype", [STATS_DTYPE, TRAINING_STATS_DTYPE])
def test_new_stats_record_no_undefined_event_in_either_layout(
    dtype: np.dtype[np.void],
) -> None:
    stats = new_stats(2, dtype)
    assert stats.dtype == dtype
    assert stats["first_undefined_step"].tolist() == [-1, -1]
    env_stats(stats, 1)


def test_env_stats_refuses_another_record() -> None:
    with pytest.raises(ValueError, match="STATS_DTYPE"):
        env_stats(new_logs(2), 0)


@jit
def _touch(state: EnvState, level: int, slot: int) -> None:
    state.melee_mobs[level].mask[slot] = np.uint8(1)
    state.inventory.potions[5] = 7
    state.map[level, 1, 2] = np.uint8(BlockType.STONE)


@pytest.mark.compute_large_fixture
def test_a_kernel_writes_nested_fields_at_their_c_offsets() -> None:
    states = new_states(2)
    _touch(env_state(states, 1), 4, 2)
    raw = states[1:2].tobytes()
    assert raw[76_424 + 4 * 64 + 36 + 2] == 1
    assert int.from_bytes(raw[76_328 + 68 + 20 : 76_328 + 68 + 24], "little") == 7
    assert raw[4 * 2304 + 48 + 2] == BlockType.STONE
    assert states[0:1].tobytes() == bytes(80_248)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
