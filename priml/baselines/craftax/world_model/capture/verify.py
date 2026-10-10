"""Replay verifier: replay a sample of every closed shard and halt capture on a mismatch.

The verifier polls every worker manifest under the archive root. For each
newly published shard it requires all of its files to match their manifest
SHA-256s, so damage anywhere in the shard is caught, then replays a
deterministic sample of ``fraction`` of the episodes (at least one) from the
receipt and actions alone, and requires every state hash to match and, in a
frame shard, the regenerated token frames to equal the stored ones, or, in a
replay shard, the snapshots replay reaches to equal the stored ones (its
frames were compared with capture's when it was written). A mismatch, or a
shard it cannot read (a SHA-256 or CRC-32 failure, a missing or undecodable
file), halts every capture worker of every arm through the archive-wide
``HALT.json`` (``control.py``) and raises ``ReplayMismatchError``. Verified
shards are appended to the launch's own JSONL log, so a restarted verifier of
that launch skips them. A log shared between launches would make a later
launch skip any shard whose name an earlier one verified, in whichever
archive root that was, and a launch's workers are counted by markers under
the root named by its id: so launch ids are never reused, and each carries
its launch's UTC time.

The verifier of a launch runs until every worker of that launch has written its
completion marker and every shard they published is verified, or until the
archive is halted, by it or by another launch's verifier, whose halted workers
never complete. Draining workers can go a whole longest episode without
closing a shard, so it gives up only after that long without a new shard, at
the slowest per-environment rate, and only while a worker that wrote its start
marker is incomplete: workers still queued by Slurm never expire it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import dataclasses
import hashlib
import json
import logging
import math
import struct
import time
import zlib

from configgle import Fig

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Record,
    read_episodes,
    read_manifest,
    read_records,
    read_snapshots,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.control import (
    check_halt,
    completed,
    halt,
    started,
)
from priml.baselines.craftax.world_model.snapshots import (
    verify_snapshots,
)
from priml.lib.codec import from_plain, loads, to_plain
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only shard sampling needs it.


logger = logging.getLogger(__name__)


class ReplayMismatchError(RuntimeError):
    """A published shard does not replay to its records, or cannot be read."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ShardVerdict:
    """The outcome of verifying one shard.

    Attributes:
      shard: Shard path relative to the archive root.
      episodes: Indices of the replayed episodes within the shard.
      decisions: Decisions replayed.
      seconds: Wall time of the replay.
      mismatch: Empty when every replayed episode matched; otherwise the first
        mismatch, naming the episode's world seed, or the read failure.

    """

    shard: str
    episodes: list[int]
    decisions: int
    seconds: float
    mismatch: str


def verify_shard(
    directory: Path,
    line: ManifestLine,
    *,
    fraction: float,
    seed: int,
    name: str,
) -> ShardVerdict:
    """Replay a deterministic sample of one published shard.

    Args:
      directory: The shard's worker directory.
      line: The shard's manifest line.
      fraction: Share of the shard's episodes to replay; at least one is.
      seed: Seed of the sample, combined with ``name``.
      name: The shard's path relative to the archive root.

    Returns:
      verdict: What was replayed and the first mismatch, if any. A shard that
        cannot be read, or whose files differ from their manifest SHA-256s, is
        a mismatch.

    """
    clock = time.monotonic()
    try:
        # The readers check the CRC-32s of the sampled episodes only; damage to
        # every other episode is caught by the whole-file SHA-256 alone.
        _check_sha256(directory, line)
        summaries = read_summaries(directory, line)
        count = min(len(summaries), max(1, math.ceil(fraction * len(summaries))))
        generator = torch.Generator().manual_seed(zlib.crc32(f"{seed}/{name}".encode()))
        permutation = torch.randperm(len(summaries), generator=generator)
        chosen = sorted(int(index) for index in permutation[:count])
        picked = [summaries[i] for i in chosen]
        stride = line.snapshot_stride
        # A replay shard keeps frames only for episodes replay does not
        # reproduce: their CRC-32s are read, and replaying them would mismatch.
        framed = [s for s in picked if s.frames is not None]
        replayed = [s for s in picked if s.snapshots is not None]
        episodes = read_episodes(directory, line, summaries=framed) if framed else []
        records = read_records(directory, line, summaries=replayed)
        stored = read_snapshots(directory, line, summaries=replayed) if replayed else []
    except _READ_ERRORS as error:
        return ShardVerdict(
            shard=name,
            episodes=[],
            decisions=0,
            seconds=time.monotonic() - clock,
            mismatch=f"unreadable: {type(error).__name__}: {error}",
        )
    mismatch = (
        _snapshot_mismatch(records, stored, stride=stride)
        if stride
        else _frame_mismatch(episodes)
    )
    return ShardVerdict(
        shard=name,
        episodes=chosen,
        decisions=sum(s.decisions for s in picked),
        seconds=time.monotonic() - clock,
        mismatch=mismatch,
    )


class ReplayVerifier:
    """Verify every shard published under an archive root as it closes."""

    class Config(Fig["ReplayVerifier"]):
        """The archive, the launch it waits for, and the sample of each shard."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root ``root`` and ``log_dir`` resolve beneath; None takes them as given."""

        root: Path = Path("/datasets/craftax/world-model/archive")
        """Archive root holding ``{train,val}/arm{arm}/w{worker}/MANIFEST.jsonl``."""

        log_dir: Path = Path("/artifacts/craftax/world-model/verifier")
        """Directory of the append-only verified-shard logs, ``{launch}.jsonl``."""

        fraction: float = 0.01
        """Share of each shard's episodes to replay; at least one is."""

        seed: int = 0
        """Seed of the per-shard episode sample."""

        poll_seconds: float = 30.0
        """Wait between manifest polls that find no new shard."""

        launch: str = ""
        """Launch whose workers this verifier waits for; required, and never
        reused by another launch."""

        workers: int = 1
        """Completion markers of ``launch`` to wait for before stopping."""

        max_episode_decisions: int = 100_000
        """Longest episode, the timeout: a drain can publish nothing that long."""

        slowest_decisions_per_second: float = 10.0
        """Slowest per-environment rate of a capture source."""

        @override
        def finalize(self) -> Self:
            self.root = resolve_working_dir(self.base_dir, self.root)
            self.log_dir = resolve_working_dir(self.base_dir, self.log_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.fraction <= 0 or config.fraction > 1:
            raise ValueError("Verifier fraction must lie in (0, 1].")
        if not config.launch:
            raise ValueError("Verifier launch must name the launch it waits for.")
        if config.workers <= 0:
            raise ValueError("Verifier workers must be positive.")
        if config.slowest_decisions_per_second <= 0:
            raise ValueError("Verifier slowest_decisions_per_second must be positive.")
        self.config = config
        self.log = config.log_dir / f"{config.launch}.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.verified: set[str] = (
            {
                from_plain(
                    from_plain(loads(text), dict[str, object])["shard"],
                    str,
                )
                for text in self.log.read_text().splitlines()
            }
            if self.log.exists()
            else set[str]()
        )
        self.unreadable: set[Path] = set()

    def run(self, *args: str) -> None:
        """Verify until every worker of the launch is complete, as a priml job.

        Args:
          *args: Rejected; use Configgle overrides.

        Raises:
          ReplayMismatchError: A shard did not replay or read; capture is halted.
          CaptureHaltedError: The archive was halted, by another launch's
            verifier or an earlier run of this one.
          TimeoutError: A started worker of the launch is incomplete, and no
            shard closed and no worker started for a longest episode at the
            slowest rate.

        """
        if args:
            raise ValueError("Use Configgle --override for verifier settings.")
        cfg = self.config
        quiet_seconds = cfg.max_episode_decisions / cfg.slowest_decisions_per_second
        progress = time.monotonic()
        known_starts = 0
        while True:
            check_halt(cfg.root)
            # Markers first: a worker writes its marker after its last shard, so
            # the poll that follows sees every shard a counted worker published.
            done = completed(cfg.root, launch=cfg.launch)
            starts = started(cfg.root, launch=cfg.launch)
            if starts > known_starts:
                known_starts, progress = starts, time.monotonic()
            if self.poll_once():
                progress = time.monotonic()
                continue
            # A manifest awaiting its retry may hold a shard not yet verified, so
            # the run ends only once the retry has read it or halted capture.
            if done >= cfg.workers and not self.unreadable:
                return
            # Only a started, incomplete worker can be silent: one still queued
            # by Slurm publishes nothing, however long it waits for a GPU.
            if starts > done and time.monotonic() - progress > quiet_seconds:
                raise TimeoutError(
                    f"{done} of {cfg.workers} workers of launch {cfg.launch} are "
                    f"complete and {starts - done} running, and no shard closed "
                    f"for {quiet_seconds:.0f} s.",
                )
            time.sleep(cfg.poll_seconds)

    def poll_once(self) -> list[ShardVerdict]:
        """Verify every published shard not yet verified.

        A manifest that cannot be read is retried at the next poll, since a
        worker may be appending to it; a second failure halts capture.

        Returns:
          verdicts: One per newly verified shard.

        Raises:
          ReplayMismatchError: A shard did not replay or read; capture is halted.

        """
        verdicts: list[ShardVerdict] = []
        for manifest in sorted(self.config.root.glob("*/*/*/MANIFEST.jsonl")):
            try:
                lines = read_manifest(manifest.parent)
            except _READ_ERRORS as error:
                self._unreadable_manifest(manifest, error)
                continue
            self.unreadable.discard(manifest)
            verdicts += self._verify_new(manifest.parent, lines)
        return verdicts

    def _verify_new(
        self,
        directory: Path,
        lines: list[ManifestLine],
    ) -> list[ShardVerdict]:
        """Verify the shards of one manifest not yet verified."""
        cfg = self.config
        verdicts: list[ShardVerdict] = []
        for line in lines:
            name = str(directory.relative_to(cfg.root) / line.shard)
            if name in self.verified:
                continue
            verdict = verify_shard(
                directory,
                line,
                fraction=cfg.fraction,
                seed=cfg.seed,
                name=name,
            )
            if verdict.mismatch:
                halt(cfg.root, shard=name, reason=verdict.mismatch)
                raise ReplayMismatchError(f"{name}: {verdict.mismatch}")
            with self.log.open("a") as log:
                log.write(json.dumps(to_plain(verdict)) + "\n")
            self.verified.add(name)
            verdicts.append(verdict)
            logger.info(
                "Verified %s: %d episodes, %.0f decisions/s.",
                name,
                len(verdict.episodes),
                verdict.decisions / max(verdict.seconds, 1e-9),
            )
        return verdicts

    def _unreadable_manifest(self, manifest: Path, error: Exception) -> None:
        """Retry a manifest once; halt capture when it fails twice in a row."""
        name = str(manifest.relative_to(self.config.root))
        reason = f"unreadable: {type(error).__name__}: {error}"
        if manifest in self.unreadable:
            halt(self.config.root, shard=name, reason=reason)
            raise ReplayMismatchError(f"{name}: {reason}")
        self.unreadable.add(manifest)
        logger.warning("Retrying %s: %s", name, reason)


def _frame_mismatch(episodes: list[Episode]) -> str:
    """Return the first frame-shard episode that does not replay to its frames."""
    for episode in episodes:
        try:
            regenerated = replay.replay(episode)
        except ValueError as error:
            return f"world seed {episode.receipt.world_seed}: {error}"
        different = [
            name
            for name, ours, theirs in (
                ("cells", regenerated.cells, episode.cells),
                ("aux", regenerated.aux, episode.aux),
                ("reward", regenerated.reward, episode.reward),
                ("done", regenerated.done, episode.done),
            )
            if not torch.equal(ours, theirs)
        ]
        if different:
            return f"world seed {episode.receipt.world_seed}: {different} differ."
    return ""


def _snapshot_mismatch(
    records: list[Record],
    stored: list[bytes],
    *,
    stride: int,
) -> str:
    """Return the first replay-shard episode whose replay misses its snapshots."""
    for record, snapshots in zip(records, stored, strict=True):
        try:
            verify_snapshots(record, snapshots, stride=stride)
        except ValueError as error:
            return f"world seed {record.receipt.world_seed}: {error}"
    return ""


def _check_sha256(directory: Path, line: ManifestLine) -> None:
    """Raise unless the shard's ``.zst`` files match their manifest SHA-256s."""
    for suffix in sorted(set(line.sha256) - {"meta"}):
        path = directory / f"{line.shard}.{suffix}.zst"
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != line.sha256[suffix]:
                raise ValueError(f"SHA-256 mismatch for {path}.")


# Everything a damaged or missing file raises while a shard is read: the file
# system, the SHA-256 and CRC-32 checks and JSON, the codecs, zstd, and headers.
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, struct.error)
