"""The game's legal actions, read from a decoded frame instead of the State.

``game.observation.compute_action_mask_numba`` decides legality from the State; a
generated frame has none, so ``legal_actions`` reads the same rules off the
frame's tokens. The corpus holds only legal actions, so an evaluation counts a
generated episode's illegal ones.
"""

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.schema import craftax_schema


def legal_actions(cells: Tensor, aux: Tensor) -> Tensor:
    """Return the actions ``compute_action_mask_numba`` allows in decoded frames.

    Everything the mask reads is in the frame, except the item under a player
    standing in the dark: the observation writes that cell as all zeros, so
    both ladder actions are left allowed there rather than counted against.

    Args:
      cells: Cell values ``[..., 99, 8]``.
      aux: Auxiliary values ``[..., 51]``.

    Returns:
      legal: Bool ``[..., 43]``, indexed by action.

    """
    v = dict(zip(craftax_schema().scalar_names, aux.long().unbind(-1), strict=True))
    wood, stone, coal = v["wood"] > 0, v["stone"] > 0, v["coal"] > 0
    iron, diamond, mana = v["iron"] > 0, v["diamond"], v["mana"]
    pickaxe, sword = v["pickaxe"], v["sword"]
    armour = torch.stack([v[f"armour_{i}"] for i in range(4)], dim=-1)
    item = cells[..., 49, 1].long()
    ladder = {"down": item == 3, "up": item == 4, "unknown": item == 0}
    enchant = (mana >= 9) & ((v["ruby"] > 0) | (v["sapphire"] > 0))
    xp = v["xp"] > 0
    free = v["sleeping"] + v["resting"] == 0
    rules = {
        6: v["energy"] < 7 + 2 * v["dexterity"],
        7: stone,
        8: v["wood"] >= 2,
        9: stone,
        10: v["sapling"] > 0,
        11: wood & (pickaxe < 1),
        12: wood & stone & (pickaxe < 2),
        13: wood & stone & iron & coal & (pickaxe < 3),
        14: wood & (sword < 1),
        15: wood & stone & (sword < 2),
        16: wood & stone & iron & coal & (sword < 3),
        # Health is in 0.05-HP steps; the game's maximum is 8 + strength HP.
        17: v["health"] < 20 * (8 + v["strength"]),
        18: ladder["unknown"]
        | (ladder["down"] & (v["floor_clear"] > 0) & (v["floor"] < 8)),
        19: ladder["unknown"] | (ladder["up"] & (v["floor"] > 0)),
        20: wood & (diamond >= 3) & (pickaxe < 4),
        21: wood & (diamond >= 2) & (sword < 4),
        22: (armour < 1).any(-1) & (v["iron"] >= 3) & (v["coal"] >= 3),
        23: (armour < 2).any(-1) & (diamond >= 3),
        24: (v["bow"] > 0) & (v["arrows"] > 0),
        25: wood & stone & (v["arrows"] < 99),
        26: (v["learned_fireball"] > 0) & (mana >= 2),
        27: (v["learned_iceball"] > 0) & (mana >= 2),
        28: v["torches"] > 0,
        **{
            29 + k: v[f"potion_{color}"] > 0
            for k, color in enumerate(
                ("red", "green", "blue", "pink", "cyan", "yellow"),
            )
        },
        35: v["books"] > 0,
        36: enchant & (sword > 0),
        37: enchant & (armour.sum(-1) > 0),
        38: wood & coal & (v["torches"] < 99),
        39: xp & (v["dexterity"] < 5),
        40: xp & (v["strength"] < 5),
        41: xp & (v["intelligence"] < 5),
        42: enchant & (v["bow"] > 0),
    }
    always = torch.ones_like(free)
    mask = torch.stack(
        [free & rules.get(action, always) for action in range(43)],
        dim=-1,
    )
    mask[..., 0] = True
    return mask
