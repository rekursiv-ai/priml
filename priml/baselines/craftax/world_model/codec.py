"""Observation codec: the game's 843-float Craftax observations to token values.

``game.observation.compute_observations_numba`` writes 792 cell values (9 rows by
11 columns, row-major, the 8 channels innermost) and then the 51 auxiliary
values in ``schema.py`` order. ``encode`` inverts that float32 buffer into
cell values ``[..., 99, 8]`` and aux token values ``[..., 51]``; ``decode``
rebuilds the float32 buffer with the game's formulas. Every value
round-trips bit for bit with two exceptions. Health and light are quantized:
health decodes to its 0.05-HP grid point and light to its 1/255 bin; health off
the grid, which the boss floor reaches by half steps, takes the token a frame
read from the game's State gives it (``replay._health_numba``: the grid point
above). And -0.0 compares equal to 0.0, so ``encode`` accepts it wherever 0
is valid and ``decode`` writes +0.0 in its place.

Validation policy: the codec is the fail-closed boundary of the tokenizer, so
``encode`` rejects, never clips, and it accepts what a State's token frame can
hold. It flags all 843 values in one vectorized pass and locates offenders
with a single ``nonzero``: one host synchronization per call on an
accelerator, where a device's first call also copies its tables over, and none
on CPU. Only a rejection reads more back, to name the offender.
``tokens_and_flags`` is that pass without the ``nonzero``. ``decode`` never
inspects values: it maps a value outside its field's range to NaN, the float
contract.
"""

from typing import cast

import dataclasses
import functools

import torch

from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    number_id,
)
from priml.lib.codec import from_plain


def encode(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the game's float32 observations into cell and aux token values.

    Args:
      obs: Observations, float32 ``[..., 843]``.

    Returns:
      cells: Cell values, uint8 ``[..., 99, 8]``.
      aux: Auxiliary token values, int16 ``[..., 51]``.

    Raises:
      ValueError: On a wrong width or dtype, or on any value that is not the
        game's float32 of an in-range token value; the message gives the first
        offender's index into ``obs``.

    """
    cells, aux, bad = tokens_and_flags(obs)
    _reject(bad, obs=obs, tables=_tables(obs.device))
    return cells, aux


def tokens_and_flags(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize like ``encode``, flagging bad values instead of raising.

    Reads no device value, so it can run inside a CUDA graph. Every returned
    token is in its field's range: a flagged value's token is a substitute
    (cell value 0, the field's lowest aux value), never an out-of-range one.

    Args:
      obs: Observations, float32 ``[..., 843]``.

    Returns:
      cells: Cell values, uint8 ``[..., 99, 8]``.
      aux: Auxiliary token values, int16 ``[..., 51]``.
      bad: Whether each of the 843 values is not the game's float32 of an
        in-range token value, bool ``[..., 843]``.

    Raises:
      ValueError: On a wrong width or dtype.

    """
    if obs.shape[-1:] != (843,):
        msg = f"Observations must be [..., 843], got {tuple(obs.shape)}."
        raise ValueError(msg)
    if obs.dtype != torch.float32:
        msg = f"Observations must be float32, got {obs.dtype}."
        raise ValueError(msg)
    tables = _tables(obs.device)
    x = obs[..., :792].unflatten(-1, (99, 8))
    # The cast is unspecified for out-of-range floats, but whatever byte it
    # yields, only an integral x in 0..255 compares equal to it afterwards.
    cells = x.to(torch.uint8)
    cells_bad = (cells.to(torch.float32) != x) | (cells >= tables.cell_limit)
    aux, aux_ok = _encode_aux(obs[..., 792:], tables=tables)
    cells = cells.masked_fill(cells_bad, 0)
    aux = torch.where(aux_ok, aux, tables.aux_low.to(torch.int16))
    return cells, aux, torch.cat([cells_bad.flatten(-2), ~aux_ok], dim=-1)


def decode(cells: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Rebuild the game's float32 observations from cell and aux token values.

    Args:
      cells: Cell values, uint8 ``[..., 99, 8]``.
      aux: Auxiliary token values, integer ``[..., 51]``.

    Returns:
      obs: Observations, float32 ``[..., 843]``; a value outside its field's
        range decodes to NaN.

    """
    tables = _tables(aux.device)
    cell_x = tables.cell_float[tables.cell_channel, cells.long()]
    column = aux.long().clamp(-1, 261) + 1
    aux_x = tables.aux_float[tables.aux_field, column]
    return torch.cat([cell_x.flatten(-2), aux_x], dim=-1)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Tables:
    """Per-field codec constants on one device.

    Attributes:
      obs_names: Field name of each of the 843 observation values.
      cell_channel: Channel index, long ``[8]``.
      cell_limit: Valid value count per channel, uint8 ``[8]``.
      cell_float: The game's float per channel and value, NaN when reserved,
        float32 ``[8, 256]``.
      aux_field: Field index, long ``[51]``.
      aux_scale: Multiplier from the game's float to the token value (before
        the square for count fields), float64 ``[51]``.
      aux_square: Whether the field is ``sqrt(n) / 10``, bool ``[51]``.
      aux_low: Lowest valid token value, float64 ``[51]``.
      aux_high: Highest valid token value, float64 ``[51]``.
      aux_exact: Whether the game's float must equal the decoded one bit for
        bit, bool ``[51]``; false for health and light.
      aux_health: Whether the field is health, bool ``[51]``.
      aux_float: The game's float per field and value ``-1…261`` (column
        ``value + 1``), NaN outside the field's range, float32 ``[51, 263]``.

    """

    obs_names: tuple[str, ...]
    cell_channel: torch.Tensor
    cell_limit: torch.Tensor
    cell_float: torch.Tensor
    aux_field: torch.Tensor
    aux_scale: torch.Tensor
    aux_square: torch.Tensor
    aux_low: torch.Tensor
    aux_high: torch.Tensor
    aux_exact: torch.Tensor
    aux_health: torch.Tensor
    aux_float: torch.Tensor


@functools.cache
def _tables(device: torch.device) -> _Tables:
    """Build the codec constants for ``device`` from the Craftax schema."""
    # Computed on the CPU and moved: CUDA divides a tensor by a scalar as a
    # multiply by its reciprocal, one ulp off the game's float32 of counts such
    # as sqrt(11) / 10, and tables built there reject valid frames.
    if device.type != "cpu":
        cpu = _tables(torch.device("cpu"))
        tables = {
            field.name: cast("torch.Tensor", getattr(cpu, field.name))
            for field in dataclasses.fields(cpu)
            if field.name != "obs_names"
        }
        # MPS holds no float64. Only ``encode`` reads those tables, and it
        # computes in float64, so it cannot run there; ``decode`` can.
        moved = {
            name: table.to(device)
            for name, table in tables.items()
            if device.type != "mps" or table.dtype != torch.float64
        }
        return dataclasses.replace(cpu, **moved)
    schema = craftax_schema()
    fields = schema.cell_fields
    names = schema.scalar_names
    base = number_id(0)
    with device:
        limit = torch.tensor([field.valid for field in fields])
        cell_float = torch.arange(256.0).expand(len(fields), 256)
        low = torch.tensor([low - base for low, _ in schema.scalar_ranges])
        high = torch.tensor([high - base for _, high in schema.scalar_ranges])
        values = torch.arange(-1, 262, dtype=torch.float64)
        in_range = (values >= low[:, None]) & (values <= high[:, None])
        floats = torch.stack([_float_row(name, values) for name in names])
        return _Tables(
            obs_names=tuple(field.name for field in fields) * schema.cell_slots + names,
            cell_channel=torch.arange(len(fields)),
            cell_limit=limit.to(torch.uint8),
            cell_float=cell_float.where(
                torch.arange(256) < limit[:, None],
                torch.nan,
            ),
            aux_field=torch.arange(len(names)),
            aux_scale=torch.tensor(
                [float(_aux_scale(name)) for name in names],
            ).double(),
            aux_square=torch.tensor([_is_count(name) for name in names]),
            aux_low=low.double(),
            aux_high=high.double(),
            aux_exact=torch.tensor([name not in {"health", "light"} for name in names]),
            aux_health=torch.tensor([name == "health" for name in names]),
            aux_float=floats.where(in_range, torch.nan),
        )


def _is_count(name: str) -> bool:
    """Whether the aux field is a ``sqrt(n) / 10`` count."""
    counts = {"wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby"}
    counts |= {"sapling", "torches", "arrows"}
    return name in counts or name.startswith("potion_")


def _aux_scale(name: str) -> int:
    """Return the factor that maps the game's float to the token value."""
    if _is_count(name):
        return 10
    if name == "books" or (
        name.startswith("armour_") and not name.startswith("armour_enchantment")
    ):
        return 2
    tenths = {"food", "drink", "energy", "mana", "xp", "floor"}
    tenths |= {"dexterity", "strength", "intelligence"}
    scales = {"pickaxe": 4, "sword": 4, "health": 200, "light": 255}
    return 10 if name in tenths else scales.get(name, 1)


# Each of the game's expressions is one correctly rounded float32 operation on an
# exact input, and rounding the float64 result of one square root or division to
# float32 gives the same value, so these rows match ``compute_observations_numba``.
def _float_row(name: str, values: torch.Tensor) -> torch.Tensor:
    """Return the game's float32 of every token value of one aux field."""
    if _is_count(name):
        return values.clamp(min=0).sqrt().float() / 10
    if name == "health":
        # The game's health is float32 HP, written as ``HP / 10.0f``.
        return (values / 20).float() / 10
    return values.float() / _aux_scale(name)


# Health and light take the token a State's frame gives them. Light is the nearest
# token in range. Health is ``floor(HP * 20 + 0.75)`` (``replay._health_numba``): the grid
# point within a quarter step, else the one above. That reads HP from the State, this
# reads the float32 ``HP / 10``, whose rounding moves the scaled value by at most
# ``20 * 13 * 2**-24``, under 2e-5 steps, so the two can differ only for health that
# close to a quarter step off the grid, where play never puts it: HP moves by whole
# and half steps.
def _encode_aux(
    x: torch.Tensor,
    *,
    tables: _Tables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the game's aux floats ``[..., 51]`` into int16 token values and validity."""
    y = x.double() * tables.aux_scale
    y = torch.where(tables.aux_square, y * y, y)
    value = torch.where(tables.aux_health, (y + 0.75).floor(), y.round())
    in_range = (value >= tables.aux_low) & (value <= tables.aux_high)
    column = torch.where(in_range, value, tables.aux_low).long() + 1
    expected = tables.aux_float[tables.aux_field, column]
    ok = in_range & (~tables.aux_exact | (expected == x))
    return value.to(torch.int16), ok


def _reject(bad: torch.Tensor, *, obs: torch.Tensor, tables: _Tables) -> None:
    """Raise naming the first flagged value of ``obs`` by its index there."""
    where = bad.nonzero()
    if where.shape[0]:
        index = tuple(from_plain(where[0].tolist(), list[int]))
        msg = (
            f"Observation field {tables.obs_names[index[-1]]!r} at index {index} "
            f"holds {float(obs[index])!r}, which is not the game's float32 of a "
            "valid token value."
        )
        raise ValueError(msg)
