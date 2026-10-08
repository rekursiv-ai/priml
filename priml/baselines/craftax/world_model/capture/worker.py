"""Capture worker: turn a source's completed episodes into published shards.

One worker owns one arm, one seed range, and two shard streams,
``{train,val}/arm{arm}/w{worker}/`` under the archive root. Episodes arrive
from an ``EpisodeSource`` as they end, in completion order. Each split's
pending episodes close as a shard at the first episode boundary after
``shard_decisions`` decisions. ``shards.ShardWriter`` encodes episodes as replay
shards on a thread pool, snapshotting each by replay and checking every state
hash, and publishes shards in close order, so the rollout never waits on replay, zstd,
or the filesystem beyond the bound on the raw episodes the worker holds. A
worker directory holds replay shards only. Only the owning copy resumes: a
directory missing a shard below its last published one holds shards relayed
for a corpus, and would repeat the owner's world seeds.

The source applies the decision budget where it records episodes: once its
ended episodes hold the budget it starts none and drains the ones in flight,
so long episodes are never cut off, and which episodes are recorded depends on
when they start, never on how long they last. So a worker overshoots its
budget by the episodes in flight then, one per environment, and long episodes
are the likeliest to be in flight: in the end-to-end check's archive (64
environments, budgets of 700k, 100k, 100k and 100k decisions) the four arms
published 6.27M, 0.21M, 2.30M and 1.79M, a 59/2/22/17 mix rather than the
budgets' 70/10/10/10. A corpus frozen with ``--decisions``
takes the arms' shares back out of an archive (``scripts/freeze_corpus.py``);
one frozen ``--all`` or ``--combine`` keeps the archive's realized mix.
Stopping at the budget instead would truncate the episodes in flight, or drop
them and so keep short episodes over long ones.

A restarted worker resumes from its manifests: the budget counts published
decisions, episode ordinals continue after the largest published one, and
shard indices skip any unpublished file a crash left behind. When capture
fails, every episode already taken from the
source is still published, in both splits, before the failure is raised.
Every poll checks the archive-wide halt the replay verifier writes
(``control.py``). A worker started by a launch marks itself started as its job
begins and complete once everything it started is published.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

import dataclasses
import logging
import time

from configgle import Fig, Makeable

from priml.baselines.craftax.game import jit
from priml.baselines.craftax.game.jit import package_digest
from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_manifest,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.control import (
    check_halt,
    mark_complete,
    mark_started,
)
from priml.baselines.craftax.world_model.capture.env import Schedule
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.shards import (
    ShardStream,
    ShardWriter,
)
from priml.baselines.craftax.world_model.capture.source import (
    EpisodeSource,
    PolicySource,
    make_source,
)
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CaptureReport:
    """What one run of a worker published.

    Attributes:
      episodes: Episodes published by this run.
      decisions: Their decisions.
      shards: Published shards, as paths relative to the archive root.

    """

    episodes: int
    decisions: int
    shards: tuple[Path, ...]


def shard_directory(root: Path, *, split: int, arm: int, worker: int) -> Path:
    """Return a worker's shard directory for one split."""
    return (
        root / ("val" if split == VALIDATION else "train") / f"arm{arm}" / f"w{worker}"
    )


class CaptureWorker:
    """Publish one worker's episodes as shards until its decision budget is met."""

    class Config(Fig["CaptureWorker"]):
        """The archive, the worker's identity and budget, the writer and the source."""

        root: Path = Path("/opt/scratch/datasets/craftax/world-model/archive")
        """Archive root holding ``{train,val}/arm{arm}/w{worker}/``."""

        arm: int = 0
        """Behaviour-mixture arm, 0-3."""

        worker: int = 0
        """Worker index within the arm, 0-3; selects the seed range."""

        generation: int = 0
        """Seed generation (``seeds.py``), one per dataset version and kind."""

        decisions: int = 10_000_000
        """Decision budget, counting decisions already published by this worker;
        the episodes in flight when it is met still finish, beyond it."""

        shard_decisions: int = 10_000_000
        """A shard closes at the first episode boundary at or after this many."""

        snapshot_stride: int = 16_384
        """Decisions between an episode's stored snapshots: 0.68 bytes per
        decision on the first dataset's mix, against 20.9 for frames."""

        poll_seconds: float = 0.05
        """Wait between polls that return no episode."""

        launch: str = ""
        """Launch that started this worker, named in its markers; empty writes none."""

        run_root: Path = Path("/opt/scratch/runs/craftax/world-model/capture")
        """Parent of the sources' run directories, ``{launch}/arm{arm}-w{worker}``."""

        compressors: int = 4
        """Threads encoding episodes: each replays its episode twice, once to
        check its frames and once to snapshot it."""

        buffer_decisions: int = 4_000_000
        """Decisions waiting for encoding at most: how far capture may run ahead
        of the encoders."""

        source: Makeable[EpisodeSource] = field(default_factory=PolicySource.Config)
        """What plays the arm's episodes."""

    def __init__(self, config: Config) -> None:
        for name, high in (("arm", 3), ("worker", 3), ("generation", 9)):
            if getattr(config, name) < 0 or getattr(config, name) > high:
                raise ValueError(f"Capture {name} must lie in 0-{high}.")
        for name in ("decisions", "shard_decisions", "compressors", "buffer_decisions"):
            if getattr(config, name) <= 0:
                raise ValueError(f"Capture {name} must be positive.")
        if config.snapshot_stride <= 0 or config.snapshot_stride % 256:
            raise ValueError(
                f"Snapshot stride {config.snapshot_stride} is not a multiple of 256.",
            )
        self.config = config

    def run(self, *args: str) -> None:
        """Run capture as a priml job and log what it published.

        Args:
          *args: Rejected; use Configgle overrides.

        """
        if args:
            raise ValueError("Use Configgle --override for capture settings.")
        report = self.capture()
        logger.info(
            "Capture published %d episodes, %d decisions, %d shards.",
            report.episodes,
            report.decisions,
            len(report.shards),
        )

    def capture(self) -> CaptureReport:
        """Capture until the budget is met and every started episode is published.

        Returns:
          report: What this run published.

        Raises:
          CaptureHaltedError: The replay verifier halted capture.

        """
        cfg = self.config
        check_halt(cfg.root)
        if cfg.launch:
            mark_started(cfg.root, launch=cfg.launch, arm=cfg.arm, worker=cfg.worker)
        directories = {
            split: shard_directory(
                cfg.root,
                split=split,
                arm=cfg.arm,
                worker=cfg.worker,
            )
            for split in (TRAIN, VALIDATION)
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        resume = _resume(directories)
        report = (
            self._run_source(resume, directories=directories)
            if resume.decisions < cfg.decisions
            else CaptureReport(episodes=0, decisions=0, shards=())
        )
        if cfg.launch:
            mark_complete(
                cfg.root,
                launch=cfg.launch,
                arm=cfg.arm,
                worker=cfg.worker,
                decisions=resume.decisions + report.decisions,
            )
        return report

    def _run_source(
        self,
        resume: _Resume,
        *,
        directories: dict[int, Path],
    ) -> CaptureReport:
        """Play the source from ``resume`` until the budget is met and published."""
        cfg = self.config
        source = make_source(cfg.source)
        try:
            source.start(
                schedule=Schedule(
                    arm=cfg.arm,
                    worker=cfg.worker,
                    generation=cfg.generation,
                    first_episode=resume.next_episode,
                    budget=cfg.decisions - resume.decisions,
                ),
                run_dir=cfg.run_root / cfg.launch / f"arm{cfg.arm}-w{cfg.worker}",
            )
            writer = ShardWriter(
                provenance={
                    **source.provenance(),
                    "game_sha1": package_digest(Path(jit.__file__).parent),
                },
                stride=cfg.snapshot_stride,
                compressors=cfg.compressors,
                buffer_decisions=cfg.buffer_decisions,
            )
            try:
                streams = {
                    split: writer.stream(
                        directory,
                        index=resume.next_shard[split],
                        threshold=cfg.shard_decisions,
                    )
                    for split, directory in directories.items()
                }
                try:
                    self._capture(source, streams=streams)
                finally:
                    # Also on failure: every episode taken is complete, so it is
                    # published rather than lost with the process.
                    for stream in streams.values():
                        stream.close_shard()
                    writer.finish()
            finally:
                writer.shutdown()
        finally:
            source.close()
        return CaptureReport(
            episodes=sum(line.episodes for _, line in writer.published),
            decisions=sum(line.decisions for _, line in writer.published),
            shards=tuple(
                d.relative_to(cfg.root) / line.shard for d, line in writer.published
            ),
        )

    def _capture(
        self,
        source: EpisodeSource,
        *,
        streams: dict[int, ShardStream],
    ) -> None:
        """Hand every episode to its split's stream until the source has finished."""
        cfg = self.config
        while True:
            check_halt(cfg.root)
            # Sampled before the poll: episodes that end after it are still
            # handed over by a later poll.
            finished = source.finished()
            episodes = source.poll()
            for episode in episodes:
                if episode.receipt.arm != cfg.arm:
                    raise ValueError("Expected episode.receipt.arm == cfg.arm.")
                streams[episode.receipt.split].add(episode)
            if finished and not episodes:
                return
            if not episodes:
                time.sleep(cfg.poll_seconds)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Resume:
    """Where a restarted worker continues."""

    next_episode: int
    next_shard: dict[int, int]
    decisions: int


def _resume(directories: dict[int, Path]) -> _Resume:
    """Read a worker's manifests and directories to find where it continues."""
    next_episode = 0
    decisions = 0
    next_shard: dict[int, int] = {}
    for split, directory in directories.items():
        lines = read_manifest(directory)
        if any(not line.snapshot_stride for line in lines):
            raise ValueError(f"{directory} holds frame shards; capture writes none.")
        indices = {
            int(path.name.split(".")[0].removeprefix("shard-"))
            for path in directory.glob("shard-*")
        }
        _require_owner(directory, lines, indices=indices)
        decisions += sum(line.decisions for line in lines)
        for line in lines:
            for summary in read_summaries(directory, line):
                ordinal = from_plain(summary.summary["episode"], int)
                next_episode = max(next_episode, ordinal + 1)
        next_shard[split] = max(indices, default=-1) + 1
    return _Resume(
        next_episode=next_episode,
        next_shard=next_shard,
        decisions=decisions,
    )


def _require_owner(
    directory: Path,
    lines: Sequence[ManifestLine],
    *,
    indices: set[int],
) -> None:
    """Raise unless ``directory`` holds every shard below its last published one."""
    published = [int(line.shard.removeprefix("shard-")) for line in lines]
    # The owner holds every shard it started, published or left by a crash; a
    # copy that relayed some of its shards lacks the others, and resuming it
    # would repeat the owner's episode ordinals, and so its world seeds.
    absent = set(range(max(published, default=0))) - indices
    if absent:
        raise ValueError(
            f"{directory} holds no shard-{min(absent):06d} below its published "
            f"shard-{max(published):06d}: it is not the owning copy of this "
            "worker; resume the owner's directory.",
        )
