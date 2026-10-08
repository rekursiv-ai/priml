"""Model bundles: decoded token streams laid out for the replay viewer.

The viewer's policy view reads 7,858-byte frames, the layout an exact game
replay records (``exact.frame_dtype``; ``games.mjs`` checks it). A model bundle fills every field a
decoded frame determines and leaves zero the fields only the game's State
knows: game ticks, the player's map position, the 48x48 map, mobs, the
achievement count, and entity health. The manifest lists those as absent and
labels the source ``model``, so the viewer hides the full-map tab and heatmaps.
``games.mjs build`` turns bundles into one self-contained page.

Beside the frames, a bundle stores the exact token frames (894 bytes each:
792 cell bytes, then 51 little-endian int16 aux values, the design's
uncompressed frame), optionally the real token frames of the same actions for the
side-by-side comparison, and per-decision annotations: where each action came
from, reward and done probabilities, invalid cells, and play overrides.
"""

from pathlib import Path

import dataclasses
import gzip
import hashlib

from torch import Tensor

import numpy as np
import torch

from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.dream import (
    Rollout,
    invalid_cells,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.session import Mark, Stream
from priml.baselines.craftax.world_model.viewer.exact import (
    frame_dtype,
    manifest_json,
)
from priml.lib.codec import from_plain
from priml.paths import validated_output_path


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Annotations:
    """Per-decision and per-frame notes the viewer shows beside model frames.

    Attributes:
      action_marks: Source of each action: ``model``, ``data``, ``forced``.
      reward: Reward of each decision.
      done: Terminal flag of each decision.
      reward_prob: Model probability of each reward; None unless generated.
      done_prob: Model probability that each decision ends the episode; None
        unless generated.
      frame_source: Source of each frame's tokens, ignoring overrides.
      invalid: Cells of each frame that no observation of the game can hold.
      overrides: Frame slots of each frame a person replaced.

    """

    action_marks: list[str]
    reward: list[int]
    done: list[bool]
    reward_prob: list[float | None]
    done_prob: list[float | None]
    frame_source: list[str]
    invalid: list[list[int]]
    overrides: list[list[int]]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """``manifest.json`` of a model bundle; ``games.mjs build`` verifies it."""

    schema_name: str = "craftax-model-game/v1"
    source: str = "model"
    title: str
    description: str
    end_label: str
    provenance: str
    frames: int
    actions: int
    tick: int = 0
    score: int
    achievements: int = 0
    absent: tuple[str, ...] = (
        *("tick", "position", "map", "mobs", "achievements", "entity_health"),
    )
    frames_sha256: str
    gzip_sha256: str
    tokens_sha256: str
    tokens_gzip_sha256: str
    reference_frames: int | None = None
    reference_sha256: str | None = None
    reference_gzip_sha256: str | None = None
    annotations: Annotations


def stream_of(rollout: Rollout, *, row: int, action_mark: Mark = Mark.MODEL) -> Stream:
    """Return one row of a ``dream`` rollout as a marked stream.

    Args:
      rollout: The rollout.
      row: Engine row to take.
      action_mark: Source of the row's actions: ``MODEL`` for the action head,
        ``FORCED`` for a policy or a recorded action stream.

    Returns:
      stream: The row's frames and decisions; a prefixed row's first frame,
        which is real, is marked ``DATA``.

    """
    frames, slots = rollout.frame_logp.shape[1:]
    first = Mark.MODEL if bool(rollout.starts[row, 0]) else Mark.DATA
    frame_marks = torch.full((frames, slots), int(Mark.MODEL), dtype=torch.uint8)
    frame_marks[0] = first
    decision_marks = torch.full((frames - 1, 3), int(Mark.MODEL), dtype=torch.uint8)
    decision_marks[:, 0] = action_mark
    return Stream(
        starts_episode=bool(rollout.starts[row, 0]),
        cells=rollout.cells[row],
        aux=rollout.aux[row],
        frame_marks=frame_marks,
        frame_logp=rollout.frame_logp[row],
        action=rollout.action[row],
        reward=rollout.reward[row],
        done=rollout.done[row],
        decision_marks=decision_marks,
        decision_logp=torch.stack(
            [
                rollout.action_logp[row],
                rollout.reward_logp[row],
                rollout.done_logp[row],
            ],
            dim=-1,
        ),
    )


def frame_records(stream: Stream, *, start: int = 0) -> bytes:
    """Lay out frames ``start`` onward of ``stream`` as the viewer's frames.

    Frame ``i`` holds decision ``i``'s action, its terminal flag, and the
    cumulative reward of its episode through that decision; its after-values
    are frame ``i + 1``'s unless decision ``i`` ended the episode. The final
    frame is marked by action 255 and carries the last decision's score and
    terminal flag, as an exact replay's final frame does.

    Args:
      stream: Frames and decisions.
      start: First frame to lay out.

    Returns:
      records: 7,858 bytes per frame.

    """
    v = _aux_fields(stream.aux)
    done = stream.done.bool()
    ended = torch.cat([done, torch.ones(1, dtype=torch.bool)])
    facing = torch.stack(
        [v[f"facing_{d}"] for d in ("left", "right", "up", "down")],
        -1,
    )
    health = v["health"].float() / 20
    record = np.zeros(len(stream.aux), dtype=frame_dtype())
    record["step"] = np.arange(len(record))
    record["action"] = torch.cat([stream.action.long(), torch.tensor([255])])
    record["floor_before"] = v["floor"]
    record["floor_after"] = torch.where(ended, v["floor"], v["floor"].roll(-1))
    record["direction"] = torch.where(facing.any(-1), facing.argmax(-1) + 1, 0)
    record["sleeping"] = v["sleeping"]
    record["terminal"] = torch.cat([done, torch.cat([done.new_zeros(1), done])[-1:]])
    record["score"] = _scores(stream.reward, done=done)
    record["health_before"] = health
    record["health_after"] = torch.where(ended, health, health.roll(-1))
    record["mana_before"] = v["mana"]
    record["mana_after"] = torch.where(ended, v["mana"], v["mana"].roll(-1))
    record["observation"] = stream.cells.flatten(1)
    record["health_max"] = 8 + v["strength"]
    record["mana_max"] = 6 + 3 * v["intelligence"]
    record["need_max"] = 7 + 2 * v["dexterity"]
    for name in ("food", "drink", "energy"):
        record[name] = v[name]
    record["inventory"] = torch.stack([v[name] for name in _inventory_names()], -1)
    record["spells"] = torch.stack([v["learned_fireball"], v["learned_iceball"]], -1)
    record["sword_enchant"] = v["sword_enchantment"]
    record["bow_enchant"] = v["bow_enchantment"]
    return record[start:].tobytes()


def token_records(cells: Tensor, aux: Tensor) -> bytes:
    """Return 894-byte token frames: 792 cell bytes, then 51 little-endian int16."""
    frames = len(cells)
    board = cells.to(torch.uint8).numpy().reshape(frames, -1)
    scalars = aux.numpy().astype("<i2").view(np.uint8).reshape(frames, -1)
    return np.concatenate([board, scalars], axis=1).tobytes()


def annotations(stream: Stream, *, start: int = 0) -> Annotations:
    """Return the viewer's notes on frames and decisions ``start`` onward.

    Args:
      stream: Frames and decisions.
      start: First frame, and first decision, to annotate.

    Returns:
      annotations: Sources, probabilities, invalid cells, and overrides.

    """
    generated = stream.decision_marks[start:] == Mark.MODEL
    p = stream.decision_logp[start:].double().exp()
    done = stream.done[start:].bool()
    done_prob = torch.where(done, p[:, 2], 1 - p[:, 2])
    invalid = invalid_cells(stream.cells[start:], schema=craftax_schema())
    overrides = stream.frame_marks[start:] == Mark.OVERRIDE
    return Annotations(
        action_marks=_mark_names(stream.decision_marks[start:, 0]),
        reward=from_plain(stream.reward[start:].tolist(), list[int]),
        done=from_plain(done.tolist(), list[bool]),
        reward_prob=_optional(p[:, 1], present=generated[:, 1]),
        done_prob=_optional(done_prob, present=generated[:, 2]),
        frame_source=_mark_names(stream.frame_marks[start:].amin(-1)),
        invalid=[
            from_plain(row.nonzero()[:, 0].tolist(), list[int]) for row in invalid
        ],
        overrides=[
            from_plain(row.nonzero()[:, 0].tolist(), list[int]) for row in overrides
        ],
    )


def write_bundle(
    stream: Stream,
    output: Path,
    *,
    title: str,
    reference: Segment | None = None,
    description: str = "",
    provenance: str = "",
) -> None:
    """Write ``stream`` as a model bundle directory for ``games.mjs build``.

    Args:
      stream: Frames and decisions to show.
      output: New directory, under ``/opt/scratch/artifacts/``.
      title: Picker title.
      reference: Real frames under the same actions, frame ``i`` beside the
        stream's frame ``i``, for the side-by-side comparison.
      description: Page description; defaults to the decision count.
      provenance: Source note; defaults to a generic model note.

    Raises:
      FileExistsError: If ``output`` exists.

    """
    output = validated_output_path(output)
    output.mkdir(parents=True)
    payloads = {
        "frames": frame_records(stream),
        "tokens": token_records(stream.cells, stream.aux),
    }
    if reference is not None:
        payloads["reference"] = token_records(reference.cells, reference.aux)
    digests: dict[str, str] = {}
    for name, raw in payloads.items():
        packed = gzip.compress(raw, mtime=0)
        (output / f"{name}.bin.gz").write_bytes(packed)
        digests[name] = hashlib.sha256(raw).hexdigest()
        digests[f"{name}_gzip"] = hashlib.sha256(packed).hexdigest()
    decisions = len(stream.action)
    manifest = Manifest(
        title=title,
        description=description or f"{decisions:,} generated decisions",
        end_label="End of generation",
        provenance=provenance
        or "World-model generation; forced actions and overrides are marked.",
        frames=decisions + 1,
        actions=decisions,
        score=int(_scores(stream.reward, done=stream.done.bool())[-1]),
        frames_sha256=digests["frames"],
        gzip_sha256=digests["frames_gzip"],
        tokens_sha256=digests["tokens"],
        tokens_gzip_sha256=digests["tokens_gzip"],
        reference_frames=None if reference is None else len(reference.cells),
        reference_sha256=digests.get("reference"),
        reference_gzip_sha256=digests.get("reference_gzip"),
        annotations=annotations(stream),
    )
    (output / "manifest.json").write_text(manifest_json(manifest))


def _aux_fields(aux: Tensor) -> dict[str, Tensor]:
    """Return each auxiliary field of ``aux [F, 51]`` by schema name, as long."""
    names = craftax_schema().scalar_names
    return dict(zip(names, aux.long().unbind(-1), strict=True))


def _inventory_names() -> tuple[str, ...]:
    """Return the aux field of each of the frame record's 24 inventory slots."""
    potions = ("red", "green", "blue", "pink", "cyan", "yellow")
    return (
        *("wood", "stone", "coal", "iron", "diamond", "sapling", "pickaxe"),
        *("sword", "bow", "arrows", "armour_0", "armour_1", "armour_2"),
        *("armour_3", "torches", "ruby", "sapphire"),
        *(f"potion_{color}" for color in potions),
        "books",
    )


def _scores(reward: Tensor, *, done: Tensor) -> Tensor:
    """Return each frame's episode reward so far, floored at 0, long ``[N + 1]``."""
    total = reward.long().cumsum(0)
    before = torch.cat([total.new_zeros(1), total[:-1]])
    first = torch.cat([done.new_ones(1), done[:-1]])
    episode = first.long().cumsum(0) - 1
    score = (total - before[first][episode]).clamp(min=0)
    return torch.cat([score, score[-1:] if len(score) else score.new_zeros(1)])


def _mark_names(marks: Tensor) -> list[str]:
    """Return the lower-case ``Mark`` name of each element of ``marks [N]``."""
    return [Mark(m).name.lower() for m in from_plain(marks.tolist(), list[int])]


def _optional(values: Tensor, *, present: Tensor) -> list[float | None]:
    """Return ``values`` as floats, None where not ``present``."""
    return [
        x if keep else None
        for x, keep in zip(
            from_plain(values.tolist(), list[float]),
            from_plain(present.tolist(), list[bool]),
            strict=True,
        )
    ]
