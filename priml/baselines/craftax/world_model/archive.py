"""Sharded trajectory archive: episode records, frames or snapshots, manifest, corpora.

A shard is three or four files written together and closed atomically. Every
shard holds

- ``shard-NNNNNN.bin.zst``: per episode, a fixed header, one action byte per
  decision, and the sparse state hashes. Enough to replay the episode.
- ``shard-NNNNNN.meta.jsonl``: one summary line per episode, with the offset,
  size, and CRC-32 of the episode's bytes in the other files.

The rest is one of two formats, told apart by the manifest line's
``snapshot_stride``:

- A frame shard (``snapshot_stride`` 0) holds ``shard-NNNNNN.frames.zst``: per
  episode, the token frames the loader reads: cells, auxiliary tokens, reward,
  and done, about 20 bytes per decision.
- A replay shard (``snapshot_stride`` S) holds ``shard-NNNNNN.snap.zst``: per
  episode, the simulator snapshots before decisions S, 2S, ... below its
  length, each XORed with the episode's reset world and compressed as its own
  zstd frame, back to back; an episode of at most S decisions has none. Replay
  regenerates the frames from the records and snapshots byte for byte
  (``snapshots.py``). Its summary lines also carry ``floors``, where the floor
  token changes and whether the episode ended in death, so the stratum index
  needs neither frames nor replay. An episode replay does not reproduce keeps
  its token frames instead, in the shard's own ``.frames.zst``, a fourth file
  that exists only for such episodes: 2 episodes in the 1.96B v1 decisions
  converted, of 80,552 and 45,232 decisions, whose replay departs from the
  captured trajectory at decision 66,993 and before 4,352.

A record is version 1, as in dataset v1, unless the episode is a branch or
truncated; then it is version 2, whose header's reserved word holds the flags
``TRUNCATED`` (1) and ``BRANCH`` (2). A branch starts from a state of another
episode rather than its world's reset: its record ends with the ``origin``, that
state's snapshot (``replay.Snapshot``) XORed with the reset world of its world
seed, the parent's, and its initial state hash is the origin's. A truncated episode was
cut after its last decision without ending there, by capture's stall cap or a
derived corpus's cap (``scripts/data_derive.py``): its last hash is the state
after that decision and its last ``done`` is false.

Each episode's bytes are one span of each ``.zst`` file, recorded with its
CRC-32 in the summary line, so a reader decompresses one episode without
decoding the shard, and every file is a valid concatenated zstd stream. The
records of a replay shard are byte for byte those of the frame shard of the
same episodes. A snapshot is the game State's bytes and then its stream
(``replay.Snapshot``), valid only for the game whose State layout wrote it;
every restored snapshot is checked against the state hash of its decision.

A shard exists for readers only once its line is in ``MANIFEST.jsonl``: files a
crashed writer left without one are no shard. The first datasets were written
with these same writers, and capture (``capture/``) writes with them too.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    BinaryIO,
    Final,
    cast,
)

import dataclasses
import hashlib
import itertools
import json
import os
import struct
import sys
import zlib

import torch

from priml.lib import zstd_compat
from priml.lib.codec import from_plain, loads, to_plain


_HEADER = struct.Struct("<4sHHQQQIIBB2x")
TRUNCATED: Final = 1
"""Record flag: the episode was cut after its last decision, which is not terminal."""
BRANCH: Final = 2
"""Record flag: the episode starts from its ``origin``, not its world's reset."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Receipt:
    """Everything needed to regenerate an episode, besides its actions.

    Attributes:
      world_seed: Seed passed to ``generate_world_numba`` at this episode's reset.
      sampling_seed: Seed of the policy's action sampling.
      initial_state_hash: FNV-1a hash of the state after reset.
      arm: Behaviour-mixture arm index.
      split: 0 for training, 1 for validation.

    """

    world_seed: int
    sampling_seed: int
    initial_state_hash: int
    arm: int
    split: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Episode:
    """One complete episode of ``T`` decisions.

    Attributes:
      receipt: How to regenerate the episode.
      actions: Executed actions, uint8 ``[T]``.
      hashes: FNV-1a state hashes before decisions 0, 256, 512, ... and after the
        last decision, as uint64 bits in int64 ``[ceil(T / 256) + 1]``.
      cells: The game's cell values of each pre-decision frame, uint8 ``[T, 99, 8]``.
      aux: Auxiliary token values of each frame, int16 ``[T, 51]``.
      reward: Realized reward of each decision, int16 ``[T]``.
      done: Terminal flag of each decision, bool ``[T]``.
      summary: Progression record and achievement mask from capture.
      origin: A branch's start state: its snapshot (``replay.Snapshot``) XORed
        with the reset world of ``receipt.world_seed``; empty for an episode that starts
        at its world's reset.
      truncated: Whether the episode was cut after its last decision, which
        then is not terminal.

    """

    receipt: Receipt
    actions: torch.Tensor
    hashes: torch.Tensor
    cells: torch.Tensor
    aux: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    summary: Mapping[str, object]
    origin: bytes = b""
    truncated: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    """One episode's ``.bin.zst`` record: what replay regenerates it from.

    Attributes:
      receipt: How to regenerate the episode.
      actions: Executed actions, uint8 ``[T]``.
      hashes: State hashes, as ``Episode.hashes``.
      origin: A branch's start state, as ``Episode.origin``.
      truncated: Whether the last decision is not terminal, as ``Episode``'s.

    """

    receipt: Receipt
    actions: torch.Tensor
    hashes: torch.Tensor
    origin: bytes = b""
    truncated: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FloorTrace:
    """What an episode's stratum index needs of its frames (``index.py``).

    Attributes:
      changes: ``(decision, floor)`` at decision 0 and wherever the floor token
        differs from the decision before, in decision order.
      died: Whether the episode ended in death: its terminal reward is -1.

    """

    changes: tuple[tuple[int, int], ...]
    died: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ReplayEpisode:
    """One complete episode as a replay shard stores it.

    Attributes:
      receipt: How to regenerate the episode.
      actions: Executed actions, uint8 ``[T]``.
      hashes: State hashes, as ``Episode.hashes``.
      snapshots: Its snapshot frames back to back (``snapshots.snapshot_episode``).
      floors: Its floor trace (``index.floor_trace``).
      summary: Progression record and achievement mask from capture.
      frames: Its token frames (``token_frame``), stored instead of snapshots
        when replay does not reproduce them; empty otherwise.
      origin: A branch's start state, as ``Episode.origin``.
      truncated: Whether the last decision is not terminal, as ``Episode``'s.

    """

    receipt: Receipt
    actions: torch.Tensor
    hashes: torch.Tensor
    snapshots: bytes
    floors: FloorTrace
    summary: Mapping[str, object]
    frames: bytes = b""
    origin: bytes = b""
    truncated: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """Where one episode's bytes lie in a shard file.

    Attributes:
      offset: Byte offset in the file.
      size: Byte count.
      crc32: CRC-32 of those bytes.

    """

    offset: int
    size: int
    crc32: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeSummary:
    """One episode's ``.meta.jsonl`` line: what it is and where its bytes are.

    Attributes:
      receipt: How to regenerate the episode.
      decisions: Decision count.
      bin: Its record in ``.bin.zst``.
      frames: Its frames in ``.frames.zst``; None for a replay-shard episode
        that replay reproduces.
      snapshots: Its snapshots in ``.snap.zst``; None in a frame shard and
        for a replay-shard episode stored as frames.
      floors: Its floor trace; None in a frame shard, whose frames hold it.
      summary: Progression record and achievement list from capture.

    """

    receipt: Receipt
    decisions: int
    bin: Span
    frames: Span | None = None
    snapshots: Span | None = None
    floors: FloorTrace | None = None
    summary: Mapping[str, object]

    @property
    def replayed(self) -> bool:
        """Whether replay regenerates the frames: a replay-shard episode stored without them."""
        return self.frames is None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ManifestLine:
    """A closed shard as recorded in ``MANIFEST.jsonl``.

    Attributes:
      shard: File stem, ``shard-NNNNNN``.
      episodes: Episode count.
      decisions: Decision count over all episodes.
      sha256: Hex digest of each file, keyed by suffix (``bin``, ``meta``, and
        ``frames`` or ``snap``).
      provenance: The capture's source, build, and checkpoint hashes.
      snapshot_stride: Decisions between the stored snapshots of a replay
        shard; 0 for a frame shard, which lines written before replay shards
        existed leave out.

    """

    shard: str
    episodes: int
    decisions: int
    sha256: dict[str, str]
    provenance: dict[str, str]
    snapshot_stride: int = 0


def write_shard(
    directory: Path,
    *,
    index: int,
    episodes: Sequence[Episode],
    provenance: dict[str, str],
) -> ManifestLine:
    """Write, close, and publish one frame shard.

    Args:
      directory: The worker's shard directory; it must exist.
      index: Shard index within the directory.
      episodes: Complete episodes, in the order readers will see them.
      provenance: Source, build, and checkpoint hashes.

    Returns:
      line: The manifest line appended for this shard.

    Raises:
      FileExistsError: A shard with this index is already published.

    """
    bins = [record_frame(e) for e in episodes]
    frames = [token_frame(e) for e in episodes]
    meta = [
        _meta_line(e, record=b, frames=f)
        for e, b, f in zip(episodes, _spans(bins), _spans(frames), strict=True)
    ]
    return _publish(
        directory,
        index=index,
        payloads={
            "bin": b"".join(bins),
            "frames": b"".join(frames),
            "meta": "".join(meta).encode(),
        },
        episodes=len(episodes),
        decisions=sum(len(e.actions) for e in episodes),
        provenance=provenance,
        snapshot_stride=0,
    )


def write_replay_shard(
    directory: Path,
    *,
    index: int,
    episodes: Sequence[ReplayEpisode],
    stride: int,
    provenance: dict[str, str],
) -> ManifestLine:
    """Write, close, and publish one replay shard.

    Args:
      directory: The worker's shard directory; it must exist.
      index: Shard index within the directory.
      episodes: Complete episodes with their snapshots, in the order readers
        will see them.
      stride: Decisions between the episodes' snapshots, a positive multiple
        of 256.
      provenance: Source, build, and checkpoint hashes, and the replay
        build that took the snapshots.

    Returns:
      line: The manifest line appended for this shard.

    Raises:
      FileExistsError: A shard with this index is already published.

    """
    if stride <= 0:
        raise ValueError("Expected stride > 0.")
    if stride % 256 != 0:
        raise ValueError("Expected stride % 256 == 0.")
    bins = [record_frame(e) for e in episodes]
    snapshots = [e.snapshots for e in episodes]
    frames = [e.frames for e in episodes]
    meta = [
        _replay_meta_line(e, record=b, snapshots=s, frames=f)
        for e, b, s, f in zip(
            episodes,
            _spans(bins),
            _spans(snapshots),
            _spans(frames),
            strict=True,
        )
    ]
    payloads = {"bin": b"".join(bins), "snap": b"".join(snapshots)}
    if any(frames):
        payloads["frames"] = b"".join(frames)
    return _publish(
        directory,
        index=index,
        payloads={**payloads, "meta": "".join(meta).encode()},
        episodes=len(episodes),
        decisions=sum(len(e.actions) for e in episodes),
        provenance=provenance,
        snapshot_stride=stride,
    )


def read_manifest(directory: Path) -> list[ManifestLine]:
    """Return the published shards of one worker directory, in close order.

    Only newline-terminated lines are published: a writer that crashed while
    appending leaves an unterminated tail, which the next append drops.

    Args:
      directory: The worker's shard directory.

    Returns:
      lines: One per published shard.

    """
    path = directory / "MANIFEST.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().split("\n")[:-1]
    return [
        _manifest_line(from_plain(loads(text), dict[str, object])) for text in lines
    ]


def read_shard(directory: Path, line: ManifestLine) -> list[Episode]:
    """Read and verify one whole published frame shard.

    Args:
      directory: The worker's shard directory.
      line: The shard's manifest line.

    Returns:
      episodes: The shard's episodes in written order.

    Raises:
      ValueError: A file's SHA-256 differs from its manifest line, or the shard
        is a replay shard, which stores no frames.

    """
    _require_frames(directory, line)
    payloads: dict[str, bytes] = {}
    for suffix in line.sha256:
        path = directory / _file_name(line.shard, suffix=suffix)
        payloads[suffix] = path.read_bytes()
        if hashlib.sha256(payloads[suffix]).hexdigest() != line.sha256[suffix]:
            raise ValueError(f"SHA-256 mismatch for {path}.")
    summaries = [_summary(text) for text in payloads["meta"].decode().splitlines()]
    return [
        _parse_episode(
            _slice(payloads["bin"], span=s.bin),
            _slice(payloads["frames"], span=_frames_span(s)),
            summary=s,
        )
        for s in summaries
    ]


def read_summaries(directory: Path, line: ManifestLine) -> list[EpisodeSummary]:
    """Return a published shard's episode summaries, verifying ``.meta.jsonl``.

    Args:
      directory: The worker's shard directory.
      line: The shard's manifest line.

    Returns:
      summaries: One per episode, in written order.

    Raises:
      ValueError: The summary file's SHA-256 differs from its manifest line.

    """
    path = directory / _file_name(line.shard, suffix="meta")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != line.sha256["meta"]:
        raise ValueError(f"SHA-256 mismatch for {path}.")
    return [_summary(text) for text in payload.decode().splitlines()]


def read_episodes(
    directory: Path,
    line: ManifestLine,
    *,
    summaries: Sequence[EpisodeSummary],
) -> list[Episode]:
    """Read chosen episodes' stored frames, decompressing only their own bytes.

    Each episode's compressed frames are checked against the CRC-32s in its
    summary, so a damaged episode fails here even though the whole-file
    SHA-256 is not recomputed.

    Args:
      directory: The worker's shard directory.
      line: The shard's manifest line.
      summaries: The episodes to read, from ``read_summaries``.

    Returns:
      episodes: The chosen episodes, in the order given.

    Raises:
      ValueError: An episode is damaged, or stores no frames: every episode of
        a replay shard that replay reproduces.

    """
    missing = [s.receipt for s in summaries if s.replayed]
    if missing:
        raise ValueError(
            f"Shard {directory / line.shard} stores no frames of {missing[0]}; "
            "replay regenerates them (snapshots.py).",
        )
    records = directory / _file_name(line.shard, suffix="bin")
    frames = directory / _file_name(line.shard, suffix="frames")
    with records.open("rb") as bin_file, frames.open("rb") as frame_file:
        return [
            _parse_episode(
                _pread(bin_file, span=s.bin),
                _pread(frame_file, span=_frames_span(s)),
                summary=s,
            )
            for s in summaries
        ]


def read_records(
    directory: Path,
    line: ManifestLine,
    *,
    summaries: Sequence[EpisodeSummary],
) -> list[Record]:
    """Read chosen episodes' records alone, from a shard of either format.

    Args:
      directory: The worker's shard directory.
      line: The shard's manifest line.
      summaries: The episodes to read, from ``read_summaries``.

    Returns:
      records: The chosen episodes' records, in the order given, each checked
        against the CRC-32 of its compressed record.

    """
    with (directory / _file_name(line.shard, suffix="bin")).open("rb") as bin_file:
        frames = [_pread(bin_file, span=s.bin) for s in summaries]
    return [
        _parse_record(frame, summary=summary)
        for frame, summary in zip(frames, summaries, strict=True)
    ]


def read_snapshots(
    directory: Path,
    line: ManifestLine,
    *,
    summaries: Sequence[EpisodeSummary],
) -> list[bytes]:
    """Read chosen episodes' stored snapshots from a replay shard.

    Args:
      directory: The worker's shard directory.
      line: The shard's manifest line.
      summaries: The episodes to read, from ``read_summaries``.

    Returns:
      snapshots: Each chosen episode's snapshot frames, as written, in the order
        given; decode them with ``snapshots.decode_snapshots``.

    Raises:
      ValueError: The shard is a frame shard, an episode stores its frames
        instead (``EpisodeSummary.replayed``), or its snapshots fail their CRC-32.

    """
    if not line.snapshot_stride:
        raise ValueError(f"Shard {directory / line.shard} stores no snapshots.")
    snapshots: list[bytes] = []
    with (directory / _file_name(line.shard, suffix="snap")).open("rb") as snap_file:
        for summary in summaries:
            span = summary.snapshots
            if span is None:
                raise ValueError(
                    f"Episode {summary.receipt} stores its frames, not snapshots.",
                )
            payload = _pread(snap_file, span=span)
            if zlib.crc32(payload) != span.crc32:
                raise ValueError(
                    f"Damaged snapshots {summary.receipt}: CRC-32 mismatch.",
                )
            snapshots.append(payload)
    return snapshots


def write_corpus(path: Path, *, entries: Sequence[tuple[Path, ManifestLine]]) -> None:
    """Freeze a list of published shards as a named corpus file.

    Args:
      path: Corpus file to write, e.g. ``corpora/base.json``.
      entries: Each shard's directory and manifest line.

    """
    shards = [
        {"directory": str(directory), "line": to_plain(line)}
        for directory, line in entries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps({"shards": shards}).encode())


def read_corpus(path: Path) -> list[tuple[Path, ManifestLine]]:
    """Return the shards a corpus file names.

    Args:
      path: Corpus file written by ``write_corpus``.

    Returns:
      entries: Each shard's directory and manifest line.

    """
    shards = from_plain(
        from_plain(loads(path.read_text()), dict[str, object])["shards"],
        list[dict[str, object]],
    )
    return [
        (
            Path(from_plain(shard["directory"], str)),
            _manifest_line(from_plain(shard["line"], dict[str, object])),
        )
        for shard in shards
    ]


def record_frame(episode: Episode | ReplayEpisode | Record) -> bytes:
    """Return an episode's record as both formats store it: one zstd frame.

    Args:
      episode: The episode; only its receipt, actions, and hashes are read.

    Returns:
      frame: Its header, actions, and hashes, compressed at the default level.

    """
    return zstd_compat.compress(_record_bytes(episode))


def token_frame(episode: Episode) -> bytes:
    """Return an episode's token frames as a frame shard stores them: one zstd frame.

    Args:
      episode: The episode; only its cells, aux, reward, and done are read.

    Returns:
      frame: Its frames column by column, compressed at the default level.

    """
    return zstd_compat.compress(_frame_bytes(episode))


def _publish(
    directory: Path,
    *,
    index: int,
    payloads: dict[str, bytes],
    episodes: int,
    decisions: int,
    provenance: dict[str, str],
    snapshot_stride: int,
) -> ManifestLine:
    """Write a shard's files atomically, then append its manifest line."""
    if sys.byteorder != "little":
        raise ValueError('Expected sys.byteorder == "little".')
    shard = f"shard-{index:06d}"
    if not episodes:
        raise ValueError("A shard holds at least one episode.")
    # Only a manifest line publishes a shard: files a crashed writer left
    # without one are no shard, and are overwritten.
    if shard in {line.shard for line in read_manifest(directory)}:
        raise FileExistsError(f"Shard {directory / shard} is already published.")
    for suffix, payload in payloads.items():
        _write_atomic(directory / _file_name(shard, suffix=suffix), payload)
    line = ManifestLine(
        shard=shard,
        episodes=episodes,
        decisions=decisions,
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
        provenance=dict(provenance),
        snapshot_stride=snapshot_stride,
    )
    _append_line(directory / "MANIFEST.jsonl", json.dumps(to_plain(line)))
    return line


def _record_bytes(episode: Episode | ReplayEpisode | Record) -> bytes:
    """Serialize one episode's header, actions, hashes, and any origin."""
    r = episode.receipt
    flags = TRUNCATED * episode.truncated | BRANCH * bool(episode.origin)
    header = _HEADER.pack(
        b"CXE1",
        2 if flags else 1,
        flags,
        r.world_seed,
        r.sampling_seed,
        r.initial_state_hash,
        len(episode.actions),
        len(episode.hashes),
        r.arm,
        r.split,
    )
    parts = (episode.actions, episode.hashes.to(torch.int64))
    return header + b"".join(_tensor_bytes(t) for t in parts) + episode.origin


def _frame_bytes(episode: Episode) -> bytes:
    """Serialize one episode's token frames column by column."""
    columns = (
        episode.cells.to(torch.uint8),
        episode.aux.to(torch.int16),
        episode.reward.to(torch.int16),
        episode.done.to(torch.uint8),
    )
    return b"".join(_tensor_bytes(column) for column in columns)


def _meta_line(episode: Episode, *, record: Span, frames: Span) -> str:
    """Serialize one frame-shard summary line."""
    line = {
        "receipt": to_plain(episode.receipt),
        "decisions": len(episode.actions),
        "bin": _span_json(record),
        "frames": _span_json(frames),
        "summary": episode.summary,
    }
    return json.dumps(line) + "\n"


def _replay_meta_line(
    episode: ReplayEpisode,
    *,
    record: Span,
    snapshots: Span,
    frames: Span,
) -> str:
    """Serialize one replay-shard summary line: its snapshots or its frames."""
    if episode.snapshots and episode.frames:
        raise ValueError("Expected not (episode.snapshots and episode.frames).")
    line = {
        "receipt": to_plain(episode.receipt),
        "decisions": len(episode.actions),
        "bin": _span_json(record),
        **(
            {"frames": _span_json(frames)}
            if episode.frames
            else {"snap": _span_json(snapshots)}
        ),
        "floors": [list(change) for change in episode.floors.changes],
        "died": episode.floors.died,
        "summary": episode.summary,
    }
    return json.dumps(line) + "\n"


def _summary(text: str) -> EpisodeSummary:
    """Decode one ``.meta.jsonl`` line of either format."""
    line = from_plain(loads(text), dict[str, object])
    changes = [
        from_plain(c, list[int])
        for c in from_plain(line.get("floors"), list[list[int]], default=[])
    ]
    floors = (
        FloorTrace(
            changes=tuple((decision, floor) for decision, floor in changes),
            died=from_plain(line["died"], bool),
        )
        if "floors" in line
        else None
    )
    return EpisodeSummary(
        receipt=from_plain(from_plain(line["receipt"], dict[str, object]), Receipt),
        decisions=from_plain(line["decisions"], int),
        bin=_span(line["bin"]),
        frames=_span(line["frames"]) if "frames" in line else None,
        snapshots=_span(line["snap"]) if "snap" in line else None,
        floors=floors,
        summary=from_plain(line["summary"], dict[str, object]),
    )


def _span(value: object) -> Span:
    """Decode one ``[offset, size, crc32]`` span."""
    offset, size, crc32 = from_plain(value, list[int])
    return Span(offset=offset, size=size, crc32=crc32)


def _span_json(span: Span) -> list[int]:
    """Encode one span as ``[offset, size, crc32]``."""
    return [span.offset, span.size, span.crc32]


def _spans(parts: Sequence[bytes]) -> list[Span]:
    """Return the span of each part once concatenated."""
    ends = itertools.accumulate(len(p) for p in parts)
    return [
        Span(offset=end - len(p), size=len(p), crc32=zlib.crc32(p))
        for end, p in zip(ends, parts, strict=True)
    ]


def _require_frames(directory: Path, line: ManifestLine) -> None:
    """Raise unless the shard is a frame shard."""
    if line.snapshot_stride:
        raise ValueError(
            f"Shard {directory / line.shard} stores no frames; replay regenerates "
            "them (snapshots.py).",
        )


def _frames_span(summary: EpisodeSummary) -> Span:
    """Return a frame-shard episode's frames span."""
    if summary.frames is None:
        raise ValueError("Expected summary.frames is not None.")
    return summary.frames


def _parse_episode(
    record: bytes,
    token_frame: bytes,
    *,
    summary: EpisodeSummary,
) -> Episode:
    """Check and parse one episode's two zstd frames against its summary."""
    if zlib.crc32(token_frame) != _frames_span(summary).crc32:
        raise ValueError(f"Damaged episode {summary.receipt}: CRC-32 mismatch.")
    parsed = _parse_record(record, summary=summary)
    count = summary.decisions
    frames = memoryview(zstd_compat.decompress(token_frame))
    cells, frames = _take(frames, shape=(count, 99, 8), dtype=torch.uint8)
    aux, frames = _take(frames, shape=(count, 51), dtype=torch.int16)
    reward, frames = _take(frames, shape=(count,), dtype=torch.int16)
    done, frames = _take(frames, shape=(count,), dtype=torch.uint8)
    if frames:
        raise ValueError("Expected not frames.")
    return Episode(
        receipt=parsed.receipt,
        actions=parsed.actions,
        hashes=parsed.hashes,
        cells=cells,
        aux=aux,
        reward=reward,
        done=done.bool(),
        summary=summary.summary,
        origin=parsed.origin,
        truncated=parsed.truncated,
    )


def _parse_record(record: bytes, *, summary: EpisodeSummary) -> Record:
    """Check and parse one episode's ``.bin.zst`` frame against its summary."""
    if zlib.crc32(record) != summary.bin.crc32:
        raise ValueError(f"Damaged episode {summary.receipt}: CRC-32 mismatch.")
    records = memoryview(zstd_compat.decompress(record))
    # The format fixes the fields' types: four bytes, then nine integers.
    magic, version, flags, world, sampling, initial, count, hash_count, arm, split = (
        cast(
            "tuple[bytes, int, int, int, int, int, int, int, int, int]",
            _HEADER.unpack_from(records),
        )
    )
    if magic != b"CXE1" or version not in {1, 2} or (version == 1) != (flags == 0):
        raise ValueError(
            f"Unsupported episode record {magic!r} version {version} flags {flags}.",
        )
    if count != summary.decisions:
        raise ValueError(f"Episode {summary.receipt} disagrees with its summary.")
    records = records[_HEADER.size :]
    actions, records = _take(records, shape=(count,), dtype=torch.uint8)
    hashes, records = _take(records, shape=(hash_count,), dtype=torch.int64)
    if bool(flags & BRANCH) != bool(records):
        raise ValueError(f"Episode {summary.receipt} disagrees with its flags.")
    receipt = Receipt(
        world_seed=world,
        sampling_seed=sampling,
        initial_state_hash=initial,
        arm=arm,
        split=split,
    )
    return Record(
        receipt=receipt,
        actions=actions,
        hashes=hashes,
        origin=bytes(records),
        truncated=bool(flags & TRUNCATED),
    )


def _take(
    stream: memoryview,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, memoryview]:
    """Copy a tensor of ``shape`` off the front of ``stream``."""
    numel = torch.Size(shape).numel()
    size = numel * dtype.itemsize
    tensor = torch.frombuffer(bytearray(stream[:size]), dtype=dtype, count=numel)
    return tensor.reshape(shape), stream[size:]


def _slice(payload: bytes, *, span: Span) -> bytes:
    """Return the bytes of ``payload`` that ``span`` covers."""
    return payload[span.offset : span.offset + span.size]


def _pread(handle: BinaryIO, *, span: Span) -> bytes:
    """Read the bytes of an open file that ``span`` covers."""
    handle.seek(span.offset)
    return handle.read(span.size)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's contiguous little-endian bytes."""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _manifest_line(data: Mapping[str, object]) -> ManifestLine:
    """Decode one manifest line."""
    return from_plain(data, ManifestLine)


def _file_name(shard: str, *, suffix: str) -> str:
    """Return the file name of one shard component."""
    return f"{shard}.{suffix}.jsonl" if suffix == "meta" else f"{shard}.{suffix}.zst"


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through a fsynced temporary file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)
    _fsync_directory(path.parent)


def _append_line(path: Path, text: str) -> None:
    """Append one line to ``path``, first dropping a torn tail, and make it durable."""
    existing = path.read_bytes() if path.exists() else b""
    kept = existing.rfind(b"\n") + 1
    with path.open("r+b" if existing else "wb") as handle:
        handle.truncate(kept)
        handle.seek(kept)
        handle.write(text.encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Persist a rename by fsyncing its directory."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
