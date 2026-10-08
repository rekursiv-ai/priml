"""Branch capture: restart a behaviour policy from states of archived episodes.

A branch pool is a JSONL file of ``BranchPoint`` s: an archived parent episode,
named by its worker directory, shard, and index in the shard, and a decision
of it. Branch ``n`` of a capture worker starts from pool entry ``n`` (its
episode ordinal): the state before that decision, with the game stream it had
there, so the parent's world plays on under the worker's policy and a new
sampling seed. A branch capture records only branches (``env.py``), each a
training episode of its parent's world seed.

``BranchFeeder`` replays each parent from its nearest stored snapshot to the
branch decision on a pool of threads, with every hash on the way checked,
``ahead`` branches ahead of need, and hands the capture branch ``n``'s start
when it resets into it, waiting for it if it is not ready; so which branches
a capture records never depends on how fast the feeder runs. Each start
carries its ``origin``, the state XORed with the reset world of its world
seed, which the branch's record stores so it replays alone (``archive.py``),
and its point, which its summary records under ``"branch"``. Once the pool is
used up the capture drains.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import collections
import dataclasses
import itertools
import json
import threading

from configgle import Fig

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    EpisodeSummary,
    ManifestLine,
    read_manifest,
    read_records,
    read_snapshots,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.env import BranchStart
from priml.baselines.craftax.world_model.capture.seeds import TRAIN
from priml.baselines.craftax.world_model.snapshots import (
    decode_snapshots,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BranchPoint:
    """Where one branch starts.

    Attributes:
      directory: The parent's worker directory.
      shard: The parent's shard, ``shard-NNNNNN``.
      episode: The parent's index within the shard.
      decision: The parent decision the branch starts before.
      floor: The floor the parent is on there.
      kind: Why the point was chosen: ``entry`` to its floor or ``time`` on it.

    """

    directory: Path
    shard: str
    episode: int
    decision: int
    floor: int
    kind: str

    def to_json(self) -> dict[str, object]:
        """Return the point as its pool line and summary entry hold it.

        Returns:
          line: The point's fields, the directory as a string.

        """
        return {
            "directory": str(self.directory),
            "shard": self.shard,
            "episode": self.episode,
            "decision": self.decision,
            "floor": self.floor,
            "kind": self.kind,
        }


def select_points(
    entries: Sequence[tuple[Path, ManifestLine]],
    *,
    arm: int,
    floors: Collection[int],
    spacing: int,
    windows: Mapping[int, int],
) -> list[BranchPoint]:
    """Return the branch points of one arm's training episodes on chosen floors.

    An episode that reaches a chosen floor gives an ``entry`` point at its
    first decision there and ``time`` points every ``spacing`` decisions of
    its time on the floor after that, counted over all its visits, and, for a
    floor in ``windows``, within the first that many of those decisions. So
    points follow time on the floor, however often play steps on and off it.
    Only replay-shard episodes that replay count; one stored as frames is
    skipped, and so is a branch, whose summary holds ``"branch"``: it starts
    from its origin, not from the reset world a branch's origin is stored
    against, so the feeder refuses it as a parent.

    Args:
      entries: Each parent shard's directory and manifest line.
      arm: The arm whose episodes are parents.
      floors: Floors to branch on.
      spacing: Decisions on the floor between its ``time`` points.
      windows: Per floor, decisions on it that ``time`` points span.

    Returns:
      points: In shard, episode, and decision order.

    """
    points: list[BranchPoint] = []
    for directory, line in entries:
        for index, summary in enumerate(read_summaries(directory, line)):
            receipt = summary.receipt
            if receipt.arm != arm or receipt.split != TRAIN or summary.floors is None:
                continue
            if summary.frames is not None or "branch" in summary.summary:
                continue
            changes = summary.floors.changes
            ends = [start for start, _ in changes[1:]] + [summary.decisions]
            on: dict[int, list[range]] = collections.defaultdict(list)
            for (start, floor), end in zip(changes, ends, strict=True):
                on[floor].append(range(start, end))
            for floor in sorted(on.keys() & set(floors)):
                time = list(itertools.chain.from_iterable(on[floor]))
                time = time[: windows.get(floor, len(time))]
                points += [
                    BranchPoint(
                        directory=directory,
                        shard=line.shard,
                        episode=index,
                        decision=time[k],
                        floor=floor,
                        kind="time" if k else "entry",
                    )
                    for k in range(0, len(time), spacing)
                ]
    return sorted(points, key=lambda p: (p.directory, p.shard, p.episode, p.decision))


def write_pool(path: Path, points: Sequence[BranchPoint]) -> None:
    """Write a branch pool, one point per line in branch order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p.to_json()) + "\n" for p in points))


def read_pool(path: Path) -> list[BranchPoint]:
    """Return a branch pool's points in branch order."""
    return [_point(from_plain(loads(text), dict[str, object])) for text in path.open()]


class BranchFeeder:
    """Compute branch starts ahead of need and hand them to a capture in order."""

    class Config(Fig["BranchFeeder"]):
        """Where the pools are, and how far ahead their starts are computed."""

        pools: Path = Path()
        """Directory of the branch pools (``write_pool``), ``arm{A}-w{W}.jsonl``
        for worker W of arm A, whose parents are on this node."""

        ahead: int = 256
        """Branch starts computed ahead of need, about 80 kB each."""

        threads: int = 4
        """Threads replaying parents."""

    def __init__(
        self,
        config: Config,
        *,
        arm: int,
        worker: int,
        first_episode: int,
    ) -> None:
        if config.ahead <= 0 or config.threads <= 0:
            raise ValueError("Branch feeder ahead and threads must be positive.")
        self.config = config
        self.points = read_pool(config.pools / f"arm{arm}-w{worker}.jsonl")
        self._next = first_episode
        self._futures: dict[int, Future[BranchStart]] = {}
        self._parents = _Parents()
        self._replay = ThreadPoolExecutor(
            config.threads,
            thread_name_prefix="branch-replay",
        )

    def start(self, ordinal: int) -> BranchStart | None:
        """Return branch ``ordinal``'s start, waiting for it; None once the pool is used up.

        Args:
          ordinal: The capture's next episode ordinal; ordinals come in order.

        Returns:
          start: The parent's world seed, the origin, and the point.

        """
        stop = min(ordinal + self.config.ahead, len(self.points))
        for queued in range(max(self._next, ordinal), stop):
            self._futures[queued] = self._replay.submit(
                self._parents.start,
                self.points[queued],
            )
        self._next = max(self._next, stop)
        future = self._futures.pop(ordinal, None)
        return None if future is None else future.result()

    def close(self) -> None:
        """Stop the replay threads, dropping starts not yet begun."""
        self._replay.shutdown(wait=True, cancel_futures=True)


class _Parents:
    """Parent episodes of a pool, their manifests and summaries read once; thread-safe."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: dict[Path, list[ManifestLine]] = {}
        self.summaries: dict[tuple[Path, str], list[EpisodeSummary]] = {}

    def start(self, point: BranchPoint) -> BranchStart:
        """Replay a branch's parent from its nearest snapshot to the branch decision.

        Args:
          point: The branch point.

        Returns:
          start: The parent's world seed, the state's origin, and the point.

        """
        line, summary = self._locate(point)
        (record,) = read_records(point.directory, line, summaries=[summary])
        # A parent that is itself a branch starts from its origin, not from the
        # reset world its branch's origin is stored against.
        if record.origin:
            raise ValueError("A branch pool's parents start at their reset.")
        (stored,) = read_snapshots(point.directory, line, summaries=[summary])
        snapshots = decode_snapshots(
            stored,
            base=replay.initial(record),
            stride=line.snapshot_stride,
        )
        k = point.decision // line.snapshot_stride
        return BranchStart(
            world_seed=record.receipt.world_seed,
            origin=replay.origin(
                record,
                decision=point.decision,
                snapshot=snapshots[k - 1] if k else None,
            ),
            point=point.to_json(),
        )

    def _locate(self, point: BranchPoint) -> tuple[ManifestLine, EpisodeSummary]:
        """Return the parent's manifest line and summary, reading each shard once."""
        with self.lock:
            if point.directory not in self.lines:
                self.lines[point.directory] = read_manifest(point.directory)
            line = next(
                (li for li in self.lines[point.directory] if li.shard == point.shard),
                None,
            )
            if line is None:
                raise FileNotFoundError(
                    f"{point.directory} publishes no {point.shard}: a pool's "
                    "parents are read on the node and at the paths it names.",
                )
            key = (point.directory, point.shard)
            if key not in self.summaries:
                self.summaries[key] = read_summaries(point.directory, line)
            return line, self.summaries[key][point.episode]


def _point(line: Mapping[str, object]) -> BranchPoint:
    """Decode one pool line."""
    return BranchPoint(
        directory=Path(from_plain(line["directory"], str)),
        shard=from_plain(line["shard"], str),
        episode=from_plain(line["episode"], int),
        decision=from_plain(line["decision"], int),
        floor=from_plain(line["floor"], int),
        kind=from_plain(line["kind"], str),
    )
