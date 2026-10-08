"""Slot schemas: the fixed token vocabulary and the per-slot allowed IDs.

A frame is ``cell_slots`` cell positions, each carrying one ID per cell field,
followed by one position per scalar slot. The local decoder generates a frame
after a prefix of scalar slots: the reward and done of the decision before it.
Every table here is derived from the schema, so the tokenizer, the loss, the
sampler, and the viewer cannot disagree about which IDs a slot may hold.
"""

import dataclasses
import functools

import torch


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CellField:
    """One categorical channel of a board cell.

    Attributes:
      name: Channel name, e.g. ``block``.
      offset: First vocabulary ID of the channel; ``ID = offset + game value``.
      size: Reserved ID count, matching the game's channel-offset table.
      valid: Leading game values that can occur; the rest are reserved.

    """

    name: str
    offset: int
    size: int
    valid: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FrameSchema:
    """Slot layout of frames and local-decoder jobs over one vocabulary.

    Attributes:
      vocab_size: Number of token IDs.
      cell_slots: Cell positions per frame; each holds one ID per cell field.
      cell_fields: The categorical channels of every cell.
      scalar_names: Names of the single-ID frame slots after the cells.
      scalar_ranges: Inclusive allowed ID range of each scalar slot.
      prefix_names: Names of the single-ID local slots before the frame.
      prefix_ranges: Inclusive allowed ID range of each prefix slot.

    """

    vocab_size: int
    cell_slots: int
    cell_fields: tuple[CellField, ...]
    scalar_names: tuple[str, ...]
    scalar_ranges: tuple[tuple[int, int], ...]
    prefix_names: tuple[str, ...] = ()
    prefix_ranges: tuple[tuple[int, int], ...] = ()

    @property
    def frame_slots(self) -> int:
        """Positions per frame: cells then scalars."""
        return self.cell_slots + len(self.scalar_ranges)

    @property
    def local_slots(self) -> int:
        """Positions per local job: prefix then frame."""
        return len(self.prefix_ranges) + self.frame_slots

    def local_allowed(self) -> torch.Tensor:
        """Return allowed IDs per single-ID local slot; cell rows are all False.

        Returns:
          allowed: Bool tensor ``[local_slots, vocab_size]``.

        """
        allowed = torch.zeros(self.local_slots, self.vocab_size, dtype=torch.bool)
        prefix = len(self.prefix_ranges)
        first_scalar = prefix + self.cell_slots
        rows = [*enumerate(self.prefix_ranges)]
        rows += [(first_scalar + i, r) for i, r in enumerate(self.scalar_ranges)]
        for row, (low, high) in rows:
            allowed[row, low : high + 1] = True
        return allowed

    def cell_index_table(self) -> torch.Tensor:
        """Return each cell field's allowed IDs, padded with ``vocab_size``.

        Gathering logits extended by one ``-inf`` column at index ``vocab_size``
        through this table gives ``[..., fields, width]`` per-field logits whose
        padding never wins a softmax.

        Returns:
          table: Long tensor ``[len(cell_fields), max field size]``.

        """
        width = max(field.size for field in self.cell_fields)
        table = torch.full((len(self.cell_fields), width), self.vocab_size)
        for row, field in enumerate(self.cell_fields):
            table[row, : field.valid] = torch.arange(field.valid) + field.offset
        return table


@functools.cache
def craftax_schema() -> FrameSchema:
    """Return the Craftax schema of the game at source revision ``6ffa5b10``.

    Returns:
      schema: 99 cells of 8 fields, 51 auxiliary slots, reward and done prefix.

    """
    fields = (
        ("block", 0, 64, 37),
        ("item", 64, 8, 6),
        ("visibility", 72, 2, 2),
        ("melee", 74, 16, 9),
        ("passive", 90, 16, 9),
        ("ranged", 106, 16, 9),
        ("mob_projectile", 122, 16, 9),
        ("player_projectile", 138, 16, 9),
    )
    aux = _craftax_aux_fields()
    return FrameSchema(
        vocab_size=461,
        cell_slots=99,
        cell_fields=tuple(
            CellField(name=name, offset=offset, size=size, valid=valid)
            for name, offset, size, valid in fields
        ),
        scalar_names=tuple(name for name, _, _ in aux),
        scalar_ranges=tuple((number_id(low), number_id(high)) for _, low, high in aux),
        prefix_names=("reward", "done"),
        prefix_ranges=(
            (number_id(-1), number_id(234)),
            (done_id(done=False), done_id(done=True)),
        ),
    )


def number_id(value: int) -> int:
    """Return the Craftax ``number.*`` ID of an integer in ``-1…260``."""
    if value < -1 or value > 260:
        raise ValueError(value)
    return 155 + value


def action_id(action: int) -> int:
    """Return the Craftax ``action.*`` ID of a game action in ``0…42``."""
    if action < 0 or action >= 43:
        raise ValueError(action)
    return 416 + action


def done_id(*, done: bool) -> int:
    """Return the Craftax ``done.*`` ID."""
    return 460 if done else 459


def _craftax_aux_fields() -> list[tuple[str, int, int]]:
    """Return the 51 auxiliary fields in the game's order with value bounds."""
    counts = ["wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby"]
    counts += ["sapling", "torches", "arrows", "books"]
    potions = ["red", "green", "blue", "pink", "cyan", "yellow"]
    fields = [(name, 0, 99) for name in counts]
    fields += [("pickaxe", 0, 4), ("sword", 0, 4)]
    fields += [("sword_enchantment", 0, 2), ("bow_enchantment", 0, 2), ("bow", 0, 1)]
    fields += [(f"potion_{name}", 0, 99) for name in potions]
    fields += [("health", 0, 260), ("food", 0, 17), ("drink", 0, 17)]
    fields += [("energy", 0, 17), ("mana", 0, 21), ("xp", 0, 8)]
    fields += [(name, 1, 5) for name in ("dexterity", "strength", "intelligence")]
    fields += [(f"facing_{d}", 0, 1) for d in ("left", "right", "up", "down")]
    fields += [(f"armour_{i}", 0, 2) for i in range(4)]
    fields += [(f"armour_enchantment_{i}", 0, 2) for i in range(4)]
    fields += [("light", 0, 255), ("sleeping", 0, 1), ("resting", 0, 1)]
    fields += [("learned_fireball", 0, 1), ("learned_iceball", 0, 1)]
    fields += [("floor", 0, 8), ("floor_clear", 0, 1), ("boss_vulnerable", 0, 1)]
    return fields
