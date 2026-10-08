"""Episodes from their first decision, generated or captured, and their statistics.

``Episodes`` holds ``N`` episodes cut at a shared horizon of ``H`` decisions,
so a generated first episode and a real one are measured alike: survival and
return up to the horizon, the floors reached, the action mix, the board's
blocks and mobs, how much of the board changes per decision, and how often a
frame breaks one of the game's invariants -- an impossible cell or a HUD value
above its maximum. ``compare`` sets two summaries side by side with the total
variation distance of their categorical distributions.
"""

from collections.abc import Sequence
from typing import Final

import dataclasses

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.game.state import Action
from priml.baselines.craftax.world_model.archive import Episode
from priml.baselines.craftax.world_model.dream import (
    Rollout,
    invalid_cells,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib.codec import PlainTree, from_plain
from priml.math.stats import total_variation


DEPARTURE_INVENTORY: Final = (
    *("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling"),
    *("potion_red", "arrows", "torches", "xp"),
)
"""The auxiliary fields whose changes ``departures`` counts."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Episodes:
    """``N`` episodes from their first decision, cut at a horizon of ``H``.

    Entries past an episode's ``length`` are ignored.

    Attributes:
      cells: Cell values of each decision's frame, uint8 ``[N, H, 99, 8]``.
      aux: Auxiliary values of each decision's frame, int16 ``[N, H, 51]``.
      action: Executed actions, uint8 ``[N, H]``.
      reward: Rewards, int16 ``[N, H]``.
      length: Decisions observed, long ``[N]``, at most ``H``.
      ended: Whether the last observed decision ended the episode, bool ``[N]``.

    """

    cells: Tensor
    aux: Tensor
    action: Tensor
    reward: Tensor
    length: Tensor
    ended: Tensor


def first_episodes(rollout: Rollout) -> Episodes:
    """Return each row's first episode of a rollout that began every row at ``start``.

    Args:
      rollout: A ``dream`` rollout of ``D`` decisions whose rows all began at
        ``start``, without prefixes; a prefixed row's frame 0 is real data.

    Returns:
      episodes: One per row, cut at its first terminal or at ``D``.

    """
    decisions = rollout.done.shape[-1]
    ended = rollout.done.any(-1)
    first_done = rollout.done.long().argmax(-1)
    return Episodes(
        cells=rollout.cells[:, :decisions],
        aux=rollout.aux[:, :decisions],
        action=rollout.action,
        reward=rollout.reward,
        length=torch.where(ended, first_done + 1, decisions),
        ended=ended,
    )


def archived_episodes(episodes: Sequence[Episode], *, horizon: int) -> Episodes:
    """Return captured episodes cut, or zero-padded, to ``horizon`` decisions.

    Args:
      episodes: Archived episodes, each ending at its last decision unless
        capture's stall cap truncated it there.
      horizon: Decisions kept per episode.

    Returns:
      episodes: The first ``horizon`` decisions of each; a truncated episode
        never ended.

    """
    length = torch.tensor([min(len(e.actions), horizon) for e in episodes])
    fields = {
        "cells": [e.cells for e in episodes],
        "aux": [e.aux for e in episodes],
        "action": [e.actions for e in episodes],
        "reward": [e.reward for e in episodes],
    }
    padded = {name: _pad(values, horizon=horizon) for name, values in fields.items()}
    ended = torch.tensor(
        [len(e.actions) <= horizon and not e.truncated for e in episodes],
    )
    return Episodes(**padded, length=length, ended=ended)


def summarize(episodes: Episodes) -> dict[str, PlainTree]:
    """Return statistics of episodes over their observed decisions.

    Args:
      episodes: The episodes.

    Returns:
      summary: ``episodes``, ``horizon``, the ``ended`` fraction, ``survival``
        (the fraction longer than each power of two below the horizon),
        ``return_mean``, ``rewards_per_1000`` (positive rewards), per-floor
        ``floor_reached`` and ``floor_occupancy``, ``action_frequency``,
        ``aux_mean`` by field, ``block_frequency`` over visible cells,
        ``mobs_per_frame`` by class, ``cell_change_rate`` between consecutive
        frames, ``invalid_cell_rate``, and ``hud_bound_violation_rate``.

    """
    schema = craftax_schema()
    horizon = episodes.action.shape[-1]
    valid = (
        torch.arange(horizon, device=episodes.length.device) < episodes.length[:, None]
    )
    count = valid.sum().double()
    aux = episodes.aux.long()
    floor = aux[..., schema.scalar_names.index("floor")]
    reward = episodes.reward.long() * valid
    return {
        "episodes": len(episodes.length),
        "horizon": horizon,
        "ended": episodes.ended.double().mean().item(),
        "survival": {
            str(2**k): (episodes.length > 2**k).double().mean().item()
            for k in range(horizon.bit_length())
            if 2**k < horizon
        },
        "return_mean": reward.sum(-1).double().mean().item(),
        "rewards_per_1000": (1000 * (reward > 0).sum() / count).item(),
        "floor_reached": [
            (floor.masked_fill(~valid, -1).amax(-1) >= k).double().mean().item()
            for k in range(9)
        ],
        "floor_occupancy": _frequency(floor[valid], size=9),
        "action_frequency": _frequency(episodes.action[valid].long(), size=43),
        "aux_mean": dict(
            zip(
                schema.scalar_names,
                from_plain(aux[valid].double().mean(0).tolist(), list[float]),
                strict=True,
            ),
        ),
        **_board(episodes.cells, valid=valid),
        "hud_bound_violation_rate": (
            (_hud_violations(aux) & valid).sum() / count
        ).item(),
    }


def compare(generated: Episodes, real: Episodes) -> dict[str, PlainTree]:
    """Summarize generated and real episodes and the distances between them.

    Args:
      generated: Episodes the model generated.
      real: Captured episodes at the same horizon.

    Returns:
      comparison: ``generated`` and ``real`` summaries, and ``distance``: the
        total variation distance of the action, visible-block, and floor
        occupancy distributions.

    """
    ours, theirs = summarize(generated), summarize(real)
    names = ("action_frequency", "block_frequency", "floor_occupancy")
    keys = ("action_tv", "block_tv", "floor_occupancy_tv")
    distance: dict[str, PlainTree] = {
        key: total_variation(
            torch.tensor(ours[name]),
            torch.tensor(theirs[name]),
        ).item()
        for key, name in zip(keys, names, strict=True)
    }
    return {"generated": ours, "real": theirs, "distance": distance}


def departures(episodes: Episodes) -> dict[str, PlainTree]:
    """Return where episodes depart from the game's rules, over observed decisions.

    Args:
      episodes: The episodes.

    Returns:
      departures: ``episodes`` and ``decisions``; the per-field HUD bound
        violation rate (health, mana, food, drink, energy), the fraction of
        episodes with any, and the median decision of an episode's first;
        floor changes per 1,000 decision pairs, the fraction made without a
        ladder action, and the count jumping more than one floor; XP increases
        per 1,000; per inventory field, increases, decreases, and jumps by more
        than one per 1,000; for floors 5 to 8, the fraction of episodes
        reaching it and their median decision of arrival; and
        ``ladder_alignment``, where ladder actions sit around floor changes.

    """
    aux = episodes.aux.long()
    n, h = episodes.action.shape
    valid = torch.arange(h) < episodes.length[:, None]
    names = craftax_schema().scalar_names
    f = {name: aux[..., i] for i, name in enumerate(names)}
    need = 7 + 2 * f["dexterity"]
    rules = {
        "health": f["health"] > 20 * (8 + f["strength"]),
        "mana": f["mana"] > 6 + 3 * f["intelligence"],
        "food": f["food"] > need,
        "drink": f["drink"] > need,
        "energy": f["energy"] > need,
    }
    violated = torch.stack(list(rules.values())).any(0) & valid
    first = torch.where(
        violated.any(-1),
        violated.float().argmax(-1),
        torch.full((n,), -1),
    )
    floor = f["floor"]
    pair = valid[:, 1:] & valid[:, :-1]
    delta = (floor[:, 1:] - floor[:, :-1])[pair]
    act = episodes.action[:, :-1].long()[pair]
    changed = delta != 0
    ladder = (act == Action.DESCEND) | (act == Action.ASCEND)
    xp_delta = (f["xp"][:, 1:] - f["xp"][:, :-1])[pair]
    return {
        "episodes": n,
        "decisions": int(valid.sum()),
        "hud_violation_rate_by_field": {
            name: float((rule & valid).sum() / valid.sum())
            for name, rule in rules.items()
        },
        "episodes_with_violation": float((first >= 0).float().mean()),
        "first_violation_decision_median": _median(first[first >= 0]),
        "floor_changes_per_1000": float(1000 * changed.sum() / pair.sum()),
        "floor_change_without_ladder_action_fraction": float(
            (changed & ~ladder).sum() / changed.sum().clamp(min=1),
        ),
        "floor_jump_gt1": int((delta.abs() > 1).sum()),
        "floor_changes": int(changed.sum()),
        "xp_increase_per_1000": float(1000 * (xp_delta > 0).sum() / pair.sum()),
        "inventory": {
            name: _changes(f[name], pair=pair) for name in DEPARTURE_INVENTORY
        },
        "first_reach": {
            str(k): _first_reach(floor >= k, valid=valid) for k in (5, 6, 7, 8)
        },
        "ladder_alignment": _ladder_alignment(episodes, floor=floor, valid=valid),
    }


def _changes(values: Tensor, *, pair: Tensor) -> dict[str, PlainTree]:
    """Return one field's increases, decreases, and jumps per 1,000 decision pairs."""
    d = (values[:, 1:] - values[:, :-1])[pair]
    pairs = pair.sum()
    return {
        "increase_per_1000": float(1000 * (d > 0).sum() / pairs),
        "decrease_per_1000": float(1000 * (d < 0).sum() / pairs),
        "jump_gt1_per_1000": float(1000 * (d.abs() > 1).sum() / pairs),
    }


def _first_reach(reached: Tensor, *, valid: Tensor) -> dict[str, PlainTree]:
    """Return the fraction of episodes reaching a floor and their median arrival."""
    hit = reached & valid
    t = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full((len(hit),), -1))
    ok = t >= 0
    return {"fraction": float(ok.float().mean()), "median_decision": _median(t[ok])}


def _ladder_alignment(
    episodes: Episodes,
    *,
    floor: Tensor,
    valid: Tensor,
) -> dict[str, PlainTree]:
    """Return how often a ladder action sits at each offset from a floor change."""
    act = episodes.action.long()
    ladder = ((act == Action.DESCEND) | (act == Action.ASCEND)) & valid
    change = torch.zeros_like(valid)
    change[:, :-1] = (floor[:, 1:] != floor[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    # Padded, not rolled: a roll reads decision 0 past the horizon and the
    # horizon's last decisions before decision 0.
    padded = functional.pad(ladder, (2, 2))
    alignment: dict[str, PlainTree] = {
        f"ladder_at_t{shift:+d}": float(
            (change & padded[:, 2 + shift : 2 + shift + ladder.shape[1]]).sum()
            / change.sum().clamp(min=1),
        )
        for shift in (-2, -1, 0, 1, 2)
    }
    descend = (act == Action.DESCEND) & valid
    descend[:, -1] = False
    up = torch.zeros_like(valid)
    up[:, :-1] = floor[:, 1:] > floor[:, :-1]
    alignment["descend_actions"] = int(descend.sum())
    alignment["descend_followed_by_floor_up"] = float(
        (descend & up).sum() / descend.sum().clamp(min=1),
    )
    return alignment


def _median(values: Tensor) -> float | None:
    """Return the lower median of ``values`` as a float, None when empty."""
    return float(values.float().median()) if len(values) else None


def _pad(values: Sequence[Tensor], *, horizon: int) -> Tensor:
    """Stack the first ``horizon`` rows of each tensor, zero-padded to ``horizon``."""
    out = values[0].new_zeros(len(values), horizon, *values[0].shape[1:])
    for row, value in enumerate(values):
        out[row, : len(value)] = value[:horizon]
    return out


def _frequency(values: Tensor, *, size: int) -> list[float]:
    """Return the relative frequency of each integer in ``0 ... size - 1``."""
    counts = torch.bincount(values.flatten(), minlength=size)[:size].double()
    return [float(p) for p in counts / counts.sum().clamp(min=1)]


def _board(cells: Tensor, *, valid: Tensor) -> dict[str, PlainTree]:
    """Return the block mix, mob counts, change rate, and invalid-cell rate."""
    schema = craftax_schema()
    names = [field.name for field in schema.cell_fields]
    frames = cells[valid]
    visible = frames[..., names.index("visibility")] == 1
    pairs = valid[:, 1:] & valid[:, :-1]
    changed = (cells[:, 1:] != cells[:, :-1]).any(-1)[pairs]
    return {
        "block_frequency": _frequency(
            frames[..., names.index("block")][visible].long(),
            size=37,
        ),
        "mobs_per_frame": {
            name: ((frames[..., names.index(name)] != 0).sum() / len(frames)).item()
            for name in ("melee", "passive", "ranged")
        },
        "cell_change_rate": changed.double().mean().item(),
        "invalid_cell_rate": invalid_cells(frames, schema=schema)
        .double()
        .mean()
        .item(),
    }


def _hud_violations(aux: Tensor) -> Tensor:
    """Flag frames whose health, food, drink, energy, or mana exceeds its maximum."""
    names = craftax_schema().scalar_names
    fields = dict(zip(names, aux.unbind(-1), strict=True))
    need = 7 + 2 * fields["dexterity"]
    # Health is tokenized on a 0.05-HP grid; its maximum is 8 + strength HP.
    return (
        (fields["health"] > 20 * (8 + fields["strength"]))
        | (fields["mana"] > 6 + 3 * fields["intelligence"])
        | (fields["food"] > need)
        | (fields["drink"] > need)
        | (fields["energy"] > need)
    )
