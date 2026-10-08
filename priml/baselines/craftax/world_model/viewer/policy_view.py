"""Policy-view bundles: an exact game cut to what its policy saw, for one panel.

The policy-view panel (``policy_view.js``) draws, before each decision, the
policy's 9x11 window and its HUD, and nothing of the map. A policy-view bundle
therefore keeps of each exact frame (``exact.py``) its first 885 bytes, every
field before the floor's 48x48 map, at the frame's own offsets, so ``hud.js``
reads both alike, then the frame's 99 projectile facings (``exact.py``), so
the panel turns each projectile the way it flies: a 984-byte record.
``policy-view.bin.gz`` holds the ``T + 1`` records back to back, as an exact
game does: before each decision, then the final State. Schema v1 had no
facings, 885-byte records.
A replay with sleep frames (``exact.replay_episode(sleep_stride=...)``) adds
``policy-sleep.bin.gz``: each sleep's frames, cut alike, back to back, which
the manifest's ``sleeps`` index. ``manifest.json`` holds their SHA-256s, the
record they were replayed from and the game build that replayed it. ``scripts/games.py panel`` writes a bundle and
``games.mjs panel`` the panel's script.
"""

from pathlib import Path

import dataclasses
import gzip
import hashlib

import numpy as np

from priml.baselines.craftax.game import jit
from priml.baselines.craftax.game.jit import package_digest, platform_key
from priml.baselines.craftax.lib.arrays import int_rows
from priml.baselines.craftax.world_model.replay import Replayable
from priml.baselines.craftax.world_model.viewer.exact import (
    Replayed,
    frame_dtype,
    manifest_json,
    read_frame,
)
from priml.paths import validated_output_path


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """``manifest.json`` of a policy-view bundle.

    ``record_bytes`` is a record's size and ``decisions`` the frames less one.
    The record is ``episode`` below the capture root ``archive``, named as
    ``scripts/games.py`` names it; the seeds and initial State hash are its
    receipt's, the 64-bit ones as decimal strings, which a JavaScript number
    cannot hold; ``record_decisions`` is its length, of which the bundle keeps
    the first ``decisions``, and ``actions_sha256`` and ``hashes_sha256`` digest
    its actions and State hashes as stored. ``game`` is the digest of the game
    build that replayed it (``jit.package_digest``) and ``platform`` the libm it
    ran on. ``end`` (floor, row, column, facing), ``tick`` and ``score`` (the
    achievement return) are the final frame's. ``sleeps`` lists each sleep's
    decision, ticks and first frame in ``policy-sleep.bin.gz``, which holds
    ``sleep_frames`` frames, one every ``sleep_stride`` ticks the player
    sleeps through (``(ticks - 1) // sleep_stride`` per sleep); all are empty
    or None for a bundle without sleep frames.
    """

    schema_name: str = "craftax-policy-view/v2"
    record_bytes: int
    decisions: int
    frames_sha256: str
    gzip_sha256: str
    archive: str
    episode: str
    world_seed: int
    sampling_seed: str
    initial_state_hash: str
    arm: int
    split: int
    record_decisions: int
    actions_sha256: str
    hashes_sha256: str
    game: str
    platform: str
    end: tuple[int, int, int, int]
    tick: int
    score: int
    sleep_stride: int | None = None
    sleeps: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    sleep_frames: int = 0
    sleep_frames_sha256: str | None = None
    sleep_gzip_sha256: str | None = None


def write_bundle(
    replayed: Replayed,
    output: Path,
    *,
    record: Replayable,
    archive: Path,
    episode: str,
    sleep_stride: int | None = None,
) -> Manifest:
    """Write ``replayed`` as a policy-view bundle.

    Args:
      replayed: The exact frames of ``record``'s first decisions
        (``exact.replay_episode``).
      output: New directory, under ``/opt/scratch/artifacts/``.
      record: The record they were replayed from.
      archive: Its capture root.
      episode: Its name below ``archive``: worker directory, shard and index.
      sleep_stride: The ticks between ``replayed``'s sleep frames, when it has
        them.

    Returns:
      manifest: What ``manifest.json`` holds.

    Raises:
      FileExistsError: If ``output`` exists.

    """
    output = validated_output_path(output)
    output.mkdir(parents=True)
    layout, frames = frame_dtype(), replayed.frames
    # The map and the creatures close a frame, so the rest is its first bytes.
    dropped = sum(layout[name].itemsize for name in ("map", "mob_count", "mobs"))
    cut = layout.itemsize - dropped
    raw = _records(frames, replayed.facings, cut=cut)
    packed = gzip.compress(raw, mtime=0)
    (output / "policy-view.bin.gz").write_bytes(packed)
    sleep_frames = replayed.sleep_frames
    slept = (
        _records(sleep_frames, replayed.sleep_facings, cut=cut)
        if len(sleep_frames)
        else b""
    )
    slept_packed = gzip.compress(slept, mtime=0) if slept else b""
    if slept:
        (output / "policy-sleep.bin.gz").write_bytes(slept_packed)
    final, receipt = read_frame(frames, -1), record.receipt
    manifest = Manifest(
        record_bytes=cut + replayed.facings.shape[1],
        decisions=len(frames) - 1,
        frames_sha256=hashlib.sha256(raw).hexdigest(),
        gzip_sha256=hashlib.sha256(packed).hexdigest(),
        archive=str(archive),
        episode=episode,
        world_seed=receipt.world_seed,
        sampling_seed=str(receipt.sampling_seed),
        initial_state_hash=str(receipt.initial_state_hash),
        arm=receipt.arm,
        split=receipt.split,
        record_decisions=len(record.actions),
        actions_sha256=hashlib.sha256(record.actions.numpy().tobytes()).hexdigest(),
        hashes_sha256=hashlib.sha256(record.hashes.numpy().tobytes()).hexdigest(),
        game=package_digest(Path(jit.__file__).parent),
        platform=platform_key(),
        end=(
            int(final.floor_before),
            int(final.row),
            int(final.col),
            int(final.direction),
        ),
        tick=int(final.tick_before),
        score=int(final.score),
        sleep_stride=sleep_stride if slept else None,
        sleeps=[(d, k, f) for d, k, f in int_rows(replayed.sleeps)],
        sleep_frames=len(sleep_frames),
        sleep_frames_sha256=hashlib.sha256(slept).hexdigest() if slept else None,
        sleep_gzip_sha256=hashlib.sha256(slept_packed).hexdigest() if slept else None,
    )
    (output / "manifest.json").write_text(manifest_json(manifest))
    return manifest


def _records(frames: np.ndarray, facings: np.ndarray, *, cut: int) -> bytes:
    """Return each frame's first ``cut`` bytes and its facings, back to back."""
    head = frames.view(np.uint8).reshape(len(frames), -1)[:, :cut]
    return np.concatenate([head, facings], axis=1).tobytes()
