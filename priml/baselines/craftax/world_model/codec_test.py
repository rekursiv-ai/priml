"""Check the observation codec against the game's float32 formulas and its frames."""

from collections.abc import Callable
from typing import override

import functools
import math
import struct

from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.world_model.codec import (
    decode,
    encode,
    tokens_and_flags,
)
from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    number_id,
)
from priml.baselines.craftax.world_model.testing import played_frames
from priml.lib.codec import from_plain


SQRT_FIELDS = frozenset(
    {
        *("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby"),
        *("sapling", "torches", "arrows"),
        *(f"potion_{c}" for c in ("red", "green", "blue", "pink", "cyan", "yellow")),
    },
)
TENTHS_FIELDS = frozenset(
    {
        *("food", "drink", "energy", "mana", "xp", "dexterity", "strength"),
        *("intelligence", "floor"),
    },
)
QUARTER_FIELDS = frozenset(["pickaxe", "sword"])
HALF_FIELDS = frozenset({"books", *(f"armour_{i}" for i in range(4))})

# Aux values of the design's worked example (record 608), the game's float32 as
# printed there, followed by the vocabulary IDs the plan lists for them.
WORKED_AUX_FLOATS = {
    0: 0.223606795,
    1: 0.141421348,
    2: 0.100000001,
    11: 0.5,
    12: 0.25,
    22: 0.899999976,
    23: 0.5,
    24: 0.600000024,
    25: 0.800000012,
    26: 0.899999976,
    28: 0.100000001,
    29: 0.100000001,
    30: 0.100000001,
    34: 1.0,
    43: 0.760345697,
    49: 1.0,
}
WORKED_AUX_IDS = (
    [160, 157, 156] + [155] * 8 + [157, 156] + [155] * 9
    + [335, 160, 161, 163, 164, 155, 156, 156, 156]
    + [155] * 3 + [156] + [155] * 8 + [349] + [155] * 5 + [156, 155]
)  # fmt: skip


def test_encode_inverts_every_aux_value_and_cell() -> None:
    obs, cells, aux = _synthetic_frames()
    got_cells, got_aux = encode(obs)
    assert got_cells.dtype == torch.uint8
    assert got_aux.dtype == torch.int16
    assert torch.equal(got_cells, cells)
    assert torch.equal(got_aux, aux)


def test_decode_reproduces_the_games_floats_bit_for_bit() -> None:
    obs, cells, aux = _synthetic_frames()
    decoded = decode(cells, aux)
    assert decoded.dtype == torch.float32
    assert torch.equal(decoded.view(torch.int32), obs.view(torch.int32))


def test_played_observations_encode_to_the_states_token_frames() -> None:
    # The game's kernels as Python on the tiny world: its sixteen random
    # decisions show what a frame of play holds, the synthetic frames the rest.
    with eager(world=tiny_world):
        played = played_frames(world_seed=11, decisions=16)
    assert len(played.observations) == 16
    got_cells, got_aux = encode(played.observations)
    assert torch.equal(got_cells, played.cells)
    assert torch.equal(got_aux, played.aux)
    decoded = decode(played.cells, played.aux)
    # Health decodes to its 0.05-HP grid point, light to its 1/255 bin.
    exact = torch.ones(843, dtype=torch.bool)
    exact[792 + 22] = exact[792 + 43] = False
    assert torch.equal(decoded[:, exact], played.observations[:, exact])


def test_codec_is_rank_agnostic() -> None:
    obs, _, aux = _synthetic_frames()
    obs = obs[:258].reshape(3, 86, 843)
    got_cells, got_aux = encode(obs)
    assert got_cells.shape == (3, 86, 99, 8)
    assert got_aux.shape == (3, 86, 51)
    assert torch.equal(got_aux.reshape(-1, 51), aux[:258])
    single_cells, single_aux = encode(obs[1, 2])
    assert single_cells.shape == (99, 8)
    assert torch.equal(single_aux, aux[88])
    assert decode(single_cells, single_aux).shape == (843,)
    assert torch.equal(decode(got_cells, got_aux), obs)


def test_health_tolerates_float32_drift_from_the_games_damage() -> None:
    # 9 HP, two iron-armour hits of 2 * 0.9, then one recover, in float32.
    health = _f32(9.0)
    for delta in (-_f32(2 * 0.9), -_f32(2 * 0.9), 1.0):
        health = _f32(health + delta)
    assert health != _f32(6.4)
    assert _encode_aux({22: _f32(health / 10)})[22] == 128


# Grid steps off each health token on both sides of the token frame's
# quarter-step margin and at the half steps.
@pytest.mark.parametrize(
    "steps",
    [-0.3, -0.26, -0.24, -0.1, 0.1, 0.24, 0.26, 0.3, 0.5, 0.74, 0.76],
)
@pytest.mark.parametrize("token", [0, 1, 128, 259, 260])
def test_health_matches_the_token_frame(token: int, steps: float) -> None:
    hp = _f32((token + steps) / 20)
    assert _encode_or_none(22, _f32(hp / 10)) == _frame_health(hp)


def test_every_half_step_of_health_takes_the_token_above_as_frames_do() -> None:
    # A melee hit on a sleeping player on the boss floor moves HP by half steps.
    half = torch.arange(260, dtype=torch.float32).mul(2).add(1).div(40)
    on_grid = torch.arange(261, dtype=torch.float32).mul(0.05)
    obs = _synthetic_frames()[0][:1].repeat(len(half) + len(on_grid), 1)
    obs[:, 792 + 22] = torch.cat([half, on_grid]) / 10
    _, aux = encode(obs)
    expected = [*range(1, 261), *range(261)]
    assert aux[:, 22].tolist() == expected
    assert [_frame_health(float(hp)) for hp in torch.cat([half, on_grid])] == (expected)


# Light levels on both sides of the token frame's acceptance bounds.
@pytest.mark.parametrize(
    "light",
    [-0.6 / 255, -0.4 / 255, 0.4 / 255, 0.5, 0.760, 1.0, 1 + 0.4 / 255, 1 + 0.6 / 255],
)
def test_light_matches_the_token_frame(light: float) -> None:
    light = _f32(light)
    token = _round_half_away(light * 255)
    assert _encode_or_none(43, light) == (token if 0 <= token <= 255 else None)


def test_light_encodes_continuous_level_to_nearest_bin() -> None:
    assert _encode_aux({43: _f32(0.760345697)})[43] == 194
    assert _encode_aux({43: 1.0})[43] == 255
    assert _encode_aux({43: _f32(0.4 / 255)})[43] == 0
    assert _encode_aux({43: 0.5})[43] == 128


def test_negative_zero_encodes_as_zero_and_decodes_positive() -> None:
    obs = _synthetic_frames()[0][0].clone()
    zeros = (obs == 0).nonzero().flatten()
    assert {792 + 1, 792 + 22, 792 + 43} <= set(zeros.tolist())
    assert zeros.min() < 792
    obs[zeros] = -0.0
    cells, aux = encode(obs)
    assert torch.equal(aux, _synthetic_frames()[2][0])
    decoded = decode(cells, aux)
    assert not decoded.signbit().any()
    assert torch.equal(decoded.abs().view(torch.int32), obs.abs().view(torch.int32))


def test_encode_synchronizes_once_and_decode_never() -> None:
    obs = _synthetic_frames()[0][:4]
    with _SyncRecorder() as recorder:
        cells, aux = encode(obs)
    assert recorder.syncs == ["aten.nonzero"]
    with _SyncRecorder() as recorder:
        decode(cells, aux)
    assert recorder.syncs == []


def test_tokens_and_flags_equal_encode_with_no_flags_and_no_sync() -> None:
    obs = _synthetic_frames()[0]
    with _SyncRecorder() as recorder:
        cells, aux, bad = tokens_and_flags(obs)
    assert recorder.syncs == []
    assert bad.shape == (261, 843)
    assert not bad.any()
    want_cells, want_aux = encode(obs)
    assert torch.equal(cells, want_cells)
    assert torch.equal(aux, want_aux)


def test_tokens_and_flags_flag_exactly_the_bad_values_and_keep_tokens_in_range() -> (
    None
):
    obs, want_cells, want_aux = (t[:3].clone() for t in _synthetic_frames())
    planted = [(0, 8 * 40), (1, 792 + 22), (1, 792 + 5), (2, 8 * 7 + 4)]
    for (row, index), value in zip(
        planted,
        [37.0, 1.305, math.nan, 256.0],
        strict=True,
    ):
        obs[row, index] = value
    cells, aux, bad = tokens_and_flags(obs)
    assert torch.equal(bad.nonzero(), torch.tensor(sorted(planted)))
    bounds = torch.tensor(_aux_bounds())
    limit = torch.tensor([field.valid for field in craftax_schema().cell_fields])
    assert (cells.long() < limit).all()
    assert ((aux >= bounds[:, 0]) & (aux <= bounds[:, 1])).all()
    keep = ~bad
    assert torch.equal(
        cells.flatten(-2)[keep[:, :792]],
        want_cells.flatten(-2)[keep[:, :792]],
    )
    assert torch.equal(aux[keep[:, 792:]], want_aux[keep[:, 792:]])
    with pytest.raises(ValueError, match=r"at index \(0, 320\)"):
        encode(obs)


def test_off_grid_health_is_a_token_and_only_out_of_range_health_is_flagged() -> None:
    obs = _synthetic_frames()[0][:3].clone()
    obs[:, 792 + 22] = torch.tensor([1.0112, 1.0125, 1e9])
    _, aux, bad = tokens_and_flags(obs)
    assert aux[:2, 22].tolist() == [202, 203]
    assert bad[:, 792 + 22].tolist() == [False, False, True]
    assert not bad[:2].any()
    with pytest.raises(ValueError, match=r"'health' at index \(2, 814\)"):
        encode(obs)


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_codec_accepts_every_value_the_cpu_codec_accepts() -> None:
    obs, cells, aux = (t.cuda() for t in _synthetic_frames())
    got_cells, got_aux, bad = tokens_and_flags(obs)
    assert not bad.any()
    assert torch.equal(got_cells, cells)
    assert torch.equal(got_aux, aux)
    assert torch.equal(decode(cells, aux).view(torch.int32), obs.view(torch.int32))


def test_bfloat16_observations_lose_health_above_10_hp() -> None:
    obs, _, aux = _synthetic_frames()
    health = aux[:, 22]
    high = obs[(health >= 200) & (health <= 260)]
    assert len(high) == 61
    rounded = high.to(torch.bfloat16).float()
    _, got, bad = tokens_and_flags(rounded)
    lost = bad[:, 792 + 22] | (got[:, 22] != aux[(health >= 200) & (health <= 260), 22])
    assert int(lost.sum()) == 22


def test_rejection_reports_caller_multi_index() -> None:
    obs = _synthetic_frames()[0][:6].reshape(2, 3, 843).clone()
    obs[1, 0, 8 * 40 + 3] = 9.0
    obs[1, 2, 792 + 22] = 1.4
    with pytest.raises(ValueError, match=r"'melee' at index \(1, 0, 323\)"):
        encode(obs)
    with pytest.raises(ValueError, match=r"'health' at index \(1, 1, 814\) holds"):
        encode(obs[:, 1:])
    with pytest.raises(ValueError, match=r"'health' at index \(814,\) holds 1\.39999"):
        encode(obs[1, 2])


@pytest.mark.parametrize(
    ("slot", "value"),
    [
        (0, 0.2246),
        (0, 1.0),
        (10, 0.25),
        (11, 1.25),
        (22, -0.0051),
        (22, 1.305),
        (28, 0.0),
        (31, 0.5),
        (43, 1 + 0.6 / 255),
        (43, -0.6 / 255),
        (48, 0.9),
        (5, math.nan),
        (43, math.nan),
        (22, math.inf),
    ],
)
def test_encode_rejects_aux_the_game_never_writes(slot: int, value: float) -> None:
    obs = _synthetic_frames()[0][:2].clone()
    obs[1, 792 + slot] = value
    name = craftax_schema().scalar_names[slot]
    with pytest.raises(ValueError, match=name):
        encode(obs)


@pytest.mark.parametrize(
    ("channel", "value"),
    [
        (0, 37.0),
        (0, 2.5),
        (1, 6.0),
        (2, 2.0),
        (3, 9.0),
        (7, -1.0),
        (4, 256.0),
        (5, math.nan),
    ],
)
def test_encode_rejects_cells_the_game_never_writes(channel: int, value: float) -> None:
    obs = _synthetic_frames()[0][:2].clone()
    obs[1, 8 * 40 + channel] = value
    name = craftax_schema().cell_fields[channel].name
    with pytest.raises(ValueError, match=name):
        encode(obs)


def test_encode_rejects_wrong_width_and_dtype() -> None:
    with pytest.raises(ValueError, match="843"):
        encode(torch.zeros(2, 842))
    with pytest.raises(ValueError, match="float32"):
        encode(torch.zeros(2, 843, dtype=torch.bfloat16))


def test_decode_marks_out_of_field_values_as_nan() -> None:
    _, cells, aux = _synthetic_frames()
    cells = cells[:1].clone()
    aux = aux[:1].clone()
    cells[0, 3, 0] = 37
    aux[0, 28] = 0
    aux[0, 23] = 50
    aux[0, 22] = -1
    aux[0, 43] = 300
    decoded = decode(cells, aux)
    nan_at = decoded[0].isnan().nonzero().flatten().tolist()
    assert nan_at == [3 * 8, 792 + 22, 792 + 23, 792 + 28, 792 + 43]


@pytest.mark.gpu_torch_mps
def test_decode_on_mps_equals_the_cpus() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    _, cells, aux = _synthetic_frames()
    mps = torch.device("mps")
    assert torch.equal(decode(cells.to(mps), aux.to(mps)).cpu(), decode(cells, aux))


def test_worked_example_ids() -> None:
    obs = torch.zeros(843)
    for slot, value in WORKED_AUX_FLOATS.items():
        obs[792 + slot] = value
    obs[8 * 29 : 8 * 30] = torch.tensor([7, 1, 1, 0, 1, 0, 0, 0])
    obs[8 * 74 : 8 * 75] = torch.tensor([7, 3, 1, 0, 0, 0, 0, 0])
    cells, aux = encode(obs)
    offsets = torch.tensor([field.offset for field in craftax_schema().cell_fields])
    cell_ids, aux_ids = cells.long() + offsets, aux.long() + number_id(0)
    assert aux_ids.tolist() == WORKED_AUX_IDS
    assert cell_ids[29].tolist() == [7, 65, 73, 74, 91, 106, 122, 138]
    assert cell_ids[74].tolist() == [7, 67, 73, 74, 90, 106, 122, 138]
    assert cell_ids[77].tolist() == [0, 64, 72, 74, 90, 106, 122, 138]


def _f32(value: float) -> float:
    """Round a Python float to the nearest float32, as a C cast does."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


# Every one of the game's expressions is one correctly rounded float32 operation
# on exact inputs, so rounding its float64 result once to float32 matches it.
def _game_aux(name: str, value: int) -> float:
    """Return ``compute_observations_numba``'s float32 for one aux token value."""
    if name in SQRT_FIELDS:
        return _f32(_f32(value**0.5) / 10)
    if name == "health":
        return _f32(_f32(value / 20) / 10)
    divisor = 255 if name == "light" else 1
    divisor = 10 if name in TENTHS_FIELDS else divisor
    divisor = 4 if name in QUARTER_FIELDS else divisor
    divisor = 2 if name in HALF_FIELDS else divisor
    return _f32(value / divisor)


@functools.cache
def _synthetic_frames() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 261 frames of the game's floats whose aux fields sweep their ranges."""
    schema = craftax_schema()
    frames = 261
    generator = torch.Generator().manual_seed(0)
    cells = torch.stack(
        [
            torch.randint(field.valid, (frames, 99), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    ).to(torch.uint8)
    aux = torch.tensor(
        [
            [low + k % (high - low + 1) for low, high in _aux_bounds()]
            for k in range(frames)
        ],
        dtype=torch.int16,
    )
    aux_obs = torch.tensor(
        [
            [
                _game_aux(name, v)
                for name, v in zip(
                    schema.scalar_names,
                    from_plain(row, list[int]),
                    strict=True,
                )
            ]
            for row in aux.tolist()
        ],
        dtype=torch.float32,
    )
    obs = torch.cat([cells.reshape(frames, 792).float(), aux_obs], dim=-1)
    return obs, cells, aux


def _aux_bounds() -> list[tuple[int, int]]:
    """Return each aux field's inclusive value range."""
    base = number_id(0)
    return [(low - base, high - base) for low, high in craftax_schema().scalar_ranges]


def _encode_aux(values: dict[int, float]) -> list[int]:
    """Encode an otherwise-zero legal frame with ``values`` at aux slots."""
    obs = torch.zeros(843)
    obs[792 + 28 : 792 + 31] = _f32(0.1)
    for slot, value in values.items():
        obs[792 + slot] = value
    return from_plain(encode(obs)[1].tolist(), list[int])


def _encode_or_none(slot: int, value: float) -> int | None:
    """Return the token ``encode`` gives ``value`` at aux ``slot``; None if rejected."""
    try:
        return _encode_aux({slot: value})[slot]
    except ValueError:
        return None


def _frame_health(hp: float) -> int | None:
    """Return a token frame's health token for float32 ``hp``; None if out of range."""
    token = math.floor(hp * 20 + 0.75)
    return token if 0 <= token <= 260 else None


def _round_half_away(x: float) -> int:
    """Round half away from zero, as C ``lround`` does."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


class _SyncRecorder(TorchDispatchMode):
    """Record the dispatched ops that read a device value on the host.

    Each one waits for the stream on an accelerator. Device-to-host copies are
    not dispatched for CPU tensors, so this sees every synchronization but those.
    """

    def __init__(self) -> None:
        super().__init__()
        self.syncs: list[str] = []

    @override
    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        name = str(func).rpartition(".")[0]
        sync_ops = {"aten.nonzero", "aten._local_scalar_dense", "aten.is_nonzero"}
        if name in sync_ops | {"aten.equal", "aten.masked_select", "aten._unique2"}:
            self.syncs.append(name)
        return func(*args, **(kwargs or {}))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
