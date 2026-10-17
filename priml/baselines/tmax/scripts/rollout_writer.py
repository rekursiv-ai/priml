"""Persist live and fixed-corpus rollout updates exactly once each.

TMax's released writer appends, which double-trains an update replayed after a
restart. This writer indexes what is already on disk, publishes each live
update behind a write intent and an atomic commit marker, and on resume
truncates only the uncommitted physical suffix.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import json
import os

from treekle import from_plain, json_loads


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch
else:
    from wrapt import lazy_import

    # ~90 ms and ~1050 ms; only ``_json_default`` reaches either.
    np = lazy_import("numpy")
    torch = lazy_import("torch")


ROLLOUT_SHARD_SIZE: Final = 10_000
"""TMax's released writer chooses a new shard per 10,000 prior samples."""


class RolloutWriter:
    """Persist rollout updates in TMax's shard layout, exactly once each.

    TMax's released writer appends, which is right for a single forward run
    and wrong for a resume: re-running a step would append a second copy of
    its records, and the learner would train the update twice. This writer
    reads what is already on disk, reports which steps are complete so a
    resumed campaign can skip collecting them, and refuses to write a
    ``(step, sample_idx)`` it already holds.
    """

    def __init__(self, directory: Path, run_name: str) -> None:
        """Index the shards already staged in ``directory``."""
        self.directory = Path(directory)
        self.run_name = run_name
        self._reset_index()
        self._load_index()

    def _reset_index(self) -> None:
        """Clear the in-memory view before loading or after tail recovery."""
        self._written: set[tuple[int, int]] = set()
        self._indices: dict[int, set[int]] = {}
        self._records: dict[int, list[dict[str, object]]] = {}
        self._locations: list[tuple[Path, int, int]] = []
        self._boundaries: dict[Path, set[int]] = {}
        self._markers: dict[int, dict[str, object]] = {}
        self._pending: dict[int, dict[str, object]] = {}
        self._torn_tail: tuple[Path, int] | None = None
        self._samples = 0

    def _load_index(self) -> None:
        """Index shards, then commit markers, then write intents."""
        self._load_shards()
        self._load_markers()
        self._load_intents()

    def _load_shards(self) -> None:
        """Load complete records and retain a recoverable final torn write."""
        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        for shard_index, shard in enumerate(shards):
            payload = shard.read_bytes()
            lines = payload.splitlines(keepends=True)
            self._boundaries[shard] = {0}
            offset = 0
            for line_index, encoded in enumerate(lines):
                end = offset + len(encoded)
                if not encoded.strip():
                    self._boundaries[shard].add(end)
                    offset = end
                    continue
                is_final_line = (
                    shard_index == len(shards) - 1 and line_index == len(lines) - 1
                )
                try:
                    record = json_loads(encoded.decode("utf-8"), dict[str, object])
                except ValueError as error:
                    if is_final_line:
                        self._torn_tail = (shard, offset)
                        break
                    raise ValueError(
                        f"Invalid rollout JSON before the recoverable tail in {shard}.",
                    ) from error
                if is_final_line and not encoded.endswith((b"\n", b"\r")):
                    self._torn_tail = (shard, offset)
                    break
                key = (
                    from_plain(record["step"], int),
                    from_plain(record["sample_idx"], int),
                )
                if key in self._written:
                    raise ValueError(
                        "Duplicate staged rollout record at step "
                        f"{key[0]} sample {key[1]} in {self.directory}.",
                    )
                self._written.add(key)
                self._indices.setdefault(key[0], set()).add(key[1])
                self._records.setdefault(key[0], []).append(record)
                self._locations.append((shard, offset, key[0]))
                self._boundaries[shard].add(end)
                self._samples += 1
                offset = end

    def _load_markers(self) -> None:
        """Load and validate every live commit marker."""
        for marker_path in sorted(
            self.directory.glob(f"{self.run_name}_live_step_*.json"),
        ):
            if marker_path.name.endswith(".pending.json"):
                continue
            try:
                marker = json_loads(marker_path.read_text(), dict[str, object])
                step = from_plain(marker["step"], int)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid live rollout commit marker {marker_path}.",
                ) from error
            expected_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.json"
            )
            if marker.get("version") != 1 or marker_path != expected_path:
                raise ValueError(
                    f"Invalid live rollout commit marker {marker_path}: "
                    "unsupported version or filename/step mismatch.",
                )
            if step in self._markers:
                raise ValueError(f"Duplicate live rollout marker for step {step}.")
            self._markers[step] = marker

    def _load_intents(self) -> None:
        """Load and validate every pending live write intent."""
        for pending_path in sorted(
            self.directory.glob(f"{self.run_name}_live_step_*.pending.json"),
        ):
            try:
                pending = json_loads(
                    pending_path.read_text(encoding="utf-8"),
                    dict[str, object],
                )
                step = from_plain(pending["step"], int)
                shard_name = from_plain(pending["shard"], str)
                offset = from_plain(pending["byte_offset"], int)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid live rollout write intent {pending_path}.",
                ) from error
            expected_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
            )
            shard_path = self.directory / shard_name
            if (
                pending.get("version") != 1
                or pending_path != expected_path
                or shard_path.parent != self.directory
                or not shard_path.name.startswith(f"{self.run_name}_rollouts_")
                or shard_path.suffix != ".jsonl"
                or offset < 0
                or (shard_path.exists() and offset > shard_path.stat().st_size)
                or (
                    shard_path.exists()
                    and offset not in self._boundaries.get(shard_path, set())
                )
                or (not shard_path.exists() and offset != 0)
            ):
                raise ValueError(
                    f"Invalid live rollout write intent {pending_path}.",
                )
            if step in self._pending:
                raise ValueError(
                    f"Duplicate live rollout write intent for step {step}.",
                )
            self._pending[step] = pending

    def completed_steps(self, records_per_update: int) -> set[int]:
        """Return steps already holding a full update.

        Args:
          records_per_update: Records one complete update writes.

        Returns:
          steps: Steps a resumed campaign must not collect again.

        """
        self._require_complete_tail()
        expected = set(range(records_per_update))
        return {step for step, indices in self._indices.items() if indices == expected}

    def incomplete_steps(self, records_per_update: int) -> set[int]:
        """Return steps with records but fewer than one complete update."""
        self._require_complete_tail()
        expected = set(range(records_per_update))
        return {step for step, indices in self._indices.items() if indices != expected}

    def _require_complete_tail(self) -> None:
        """Reject non-live use of a shard whose final append was interrupted."""
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            raise ValueError(
                f"Rollout shard {path} has an interrupted final record at byte "
                f"{offset}; only live resume can recover that speculative tail.",
            )

    def committed_live_steps(self, records_per_update: int) -> set[int]:
        """Return marker-backed steps whose record contract validates.

        Args:
          records_per_update: Records one complete update holds.

        Returns:
          steps: Steps whose marker, indices, and digest all validate.

        Raises:
          ValueError: A marker disagrees with the records staged under it.

        """
        committed: set[int] = set()
        expected = list(range(records_per_update))
        for step, marker in self._markers.items():
            marker_count = marker.get("records_per_update")
            marker_indices = marker.get("sample_indices")
            cursor = marker.get("prompt_loader_state_after")
            if (
                marker_count != records_per_update
                or marker_indices != expected
                or not isinstance(cursor, Mapping)
            ):
                raise ValueError(
                    f"Live rollout marker for step {step} has an invalid "
                    "record or prompt-cursor contract.",
                )
            actual = sorted(self._indices.get(step, set()))
            if actual != expected:
                raise ValueError(
                    f"Live rollout marker for step {step} claims a complete "
                    f"update, but staged sample indices are {actual}.",
                )
            actual_digest = _records_digest(self._records.get(step, []))
            if marker.get("records_sha256") != actual_digest:
                raise ValueError(
                    f"Live rollout marker for step {step} does not match the "
                    "staged record contents.",
                )
            committed.add(step)
        return committed

    def live_cursor_after(
        self,
        step: int,
        records_per_update: int,
    ) -> Mapping[str, object]:
        """Return the validated prompt-loader cursor committed after ``step``.

        Args:
          step: A committed live step.
          records_per_update: Records one complete update holds.

        Returns:
          cursor: The prompt-loader state recorded after ``step``.

        Raises:
          ValueError: ``step`` is not committed.

        """
        if step not in self.committed_live_steps(records_per_update):
            raise ValueError(f"Live rollout step {step} is not committed.")
        return from_plain(
            self._markers[step]["prompt_loader_state_after"],
            dict[str, object],
        )

    def records_for_step(
        self,
        step: int,
        records_per_update: int,
    ) -> list[dict[str, object]]:
        """Return one complete indexed step without reparsing rollout shards.

        Args:
          step: An indexed step.
          records_per_update: Records one complete update holds.

        Returns:
          records: The step's records, ordered by ``sample_idx``.

        Raises:
          ValueError: The step lacks a sample index or holds one twice.

        """
        expected = list(range(records_per_update))
        records = self._records.get(step, [])
        indices = sorted(from_plain(record["sample_idx"], int) for record in records)
        if indices != expected:
            raise ValueError(
                f"Staged live rollout step {step} must contain each sample index "
                f"0..{records_per_update - 1} exactly once.",
            )
        return sorted(
            (dict(record) for record in records),
            key=lambda record: from_plain(record["sample_idx"], int),
        )

    def recover_uncommitted_live_tail(
        self,
        first_missing_step: int,
        records_per_update: int,
    ) -> set[int]:
        """Roll back only the speculative physical suffix after the frontier.

        A live commit marker is the durable boundary. A write intent records
        the exact shard offset before a new update is appended; older runs
        without intents are recovered from the first indexed uncommitted
        record. Recovery refuses any layout where committed or earlier-step
        data appears after that boundary.

        Args:
          first_missing_step: The first step with no commit marker.
          records_per_update: Records one complete update holds.

        Returns:
          recovered: Steps whose uncommitted records or intents were removed.

        Raises:
          ValueError: Uncommitted data precedes the frontier, or committed data
            follows the truncation point.

        """
        committed = self.committed_live_steps(records_per_update)
        uncommitted = set(self._indices).difference(committed)
        invalid = sorted(step for step in uncommitted if step < first_missing_step)
        if invalid:
            raise ValueError(
                "Cannot recover uncommitted rollout records before the durable "
                f"frontier {first_missing_step}: {invalid}.",
            )
        pending_uncommitted = {step for step in self._pending if step not in committed}
        invalid_pending = sorted(
            step for step in pending_uncommitted if step < first_missing_step
        )
        if invalid_pending:
            raise ValueError(
                "Cannot recover rollout write intents before the durable "
                f"frontier {first_missing_step}: {invalid_pending}.",
            )

        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        order = {path: index for index, path in enumerate(shards)}
        candidates = [
            (order[path], offset, path)
            for path, offset, step in self._locations
            if step in uncommitted
        ]
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            candidates.append((order[path], offset, path))
        for step in pending_uncommitted:
            pending = self._pending[step]
            path = self.directory / from_plain(pending["shard"], str)
            if path.exists():
                candidates.append(
                    (order[path], from_plain(pending["byte_offset"], int), path),
                )

        recovered = uncommitted | pending_uncommitted
        if candidates:
            shard_index, byte_offset, boundary = min(candidates)
            committed_after_boundary = sorted(
                step
                for path, offset, step in self._locations
                if step in committed
                and (order[path], offset) >= (shard_index, byte_offset)
            )
            if committed_after_boundary:
                raise ValueError(
                    "Cannot recover a non-suffix rollout tail because committed "
                    f"steps follow it: {committed_after_boundary}.",
                )
            with boundary.open("r+b") as handle:
                handle.truncate(byte_offset)
                handle.flush()
                os.fsync(handle.fileno())
            for shard in shards[shard_index + 1 :]:
                shard.unlink()

        removed_metadata = False
        for step in tuple(self._pending):
            pending_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
            )
            if step in committed or step in pending_uncommitted:
                pending_path.unlink(missing_ok=True)
                self._pending.pop(step)
                removed_metadata = True
        for temporary in self.directory.glob(
            f"{self.run_name}_live_step_*.json.tmp",
        ):
            temporary.unlink()
            removed_metadata = True
        if candidates or recovered or removed_metadata:
            _fsync_directory(self.directory)
        if candidates or recovered:
            self._reset_index()
            self._load_index()
        return recovered

    def recover_incomplete_fixed_tail(self, records_per_update: int) -> set[int]:
        """Remove incomplete fixed-corpus updates from the physical suffix.

        The fixed writer appends one whole update at a time. After interruption,
        every incomplete update must therefore be in the physical suffix.
        Recovery refuses to truncate if any complete update follows it.

        Args:
          records_per_update: Records one complete update holds.

        Returns:
          recovered: Incomplete steps removed from the suffix.

        Raises:
          ValueError: A complete update follows an incomplete one.

        """
        expected = set(range(records_per_update))
        incomplete = {
            step for step, indices in self._indices.items() if indices != expected
        }
        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        order = {path: index for index, path in enumerate(shards)}
        candidates = [
            (order[path], offset, path)
            for path, offset, step in self._locations
            if step in incomplete
        ]
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            candidates.append((order[path], offset, path))
        if not candidates:
            return set()

        shard_index, byte_offset, boundary = min(candidates)
        complete_after_boundary = sorted(
            step
            for path, offset, step in self._locations
            if self._indices[step] == expected
            and (order[path], offset) >= (shard_index, byte_offset)
        )
        if complete_after_boundary:
            raise ValueError(
                "Cannot recover a non-suffix fixed rollout update because "
                f"complete steps follow it: {complete_after_boundary}.",
            )
        with boundary.open("r+b") as handle:
            handle.truncate(byte_offset)
            handle.flush()
            os.fsync(handle.fileno())
        for shard in shards[shard_index + 1 :]:
            shard.unlink()
        _fsync_directory(self.directory)
        self._reset_index()
        self._load_index()
        return incomplete

    def write(self, step: int, records: Sequence[Mapping[str, object]]) -> Path:
        """Append one update's records under TMax's shard name.

        Args:
          step: The learner step these records belong to.
          records: The update's rollout records.

        Returns:
          path: The shard written.

        Raises:
          ValueError: A record duplicates one already written, which would
            train its update twice.

        """
        self._require_complete_tail()
        keys = [
            (from_plain(record["step"], int), from_plain(record["sample_idx"], int))
            for record in records
        ]
        if any(record_step != step for record_step, _ in keys):
            raise ValueError(
                f"Step {step} received records belonging to another step.",
            )
        duplicates = sorted(self._written.intersection(keys))
        if duplicates:
            raise ValueError(
                f"Refusing to write {len(duplicates)} rollout record(s) already "
                f"staged under {self.directory}, starting at step "
                f"{duplicates[0][0]} sample {duplicates[0][1]}: appending them "
                "would train one update twice.",
            )
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Step {step} produced duplicate (step, sample_idx) records.",
            )
        shard_idx = self._samples // ROLLOUT_SHARD_SIZE
        path = self.directory / f"{self.run_name}_rollouts_{shard_idx:06d}.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(_encode(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for key in keys:
            self._written.add(key)
            self._indices.setdefault(key[0], set()).add(key[1])
        for record in records:
            self._records.setdefault(step, []).append(
                json_loads(_encode(record), dict[str, object]),
            )
        self._samples += len(keys)
        return path

    def write_live(
        self,
        step: int,
        records: Sequence[Mapping[str, object]],
        *,
        prompt_loader_state_after: Mapping[str, object],
    ) -> Path:
        """Append records, then atomically publish their durable commit marker.

        Args:
          step: The learner step these records train.
          records: One complete update, sample indices ``0..n-1``.
          prompt_loader_state_after: Prompt-loader cursor after this update.

        Returns:
          path: The shard the records were appended to.

        Raises:
          ValueError: The records do not hold each sample index exactly once.

        """
        indices = sorted(from_plain(record["sample_idx"], int) for record in records)
        if indices != list(range(len(records))):
            raise ValueError(
                f"Live rollout step {step} must contain sample indices "
                f"0..{len(records) - 1} exactly once.",
            )
        shard_idx = self._samples // ROLLOUT_SHARD_SIZE
        path = self.directory / f"{self.run_name}_rollouts_{shard_idx:06d}.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        pending_path = (
            self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
        )
        pending_temporary = pending_path.with_suffix(pending_path.suffix + ".tmp")
        pending: dict[str, object] = {
            "version": 1,
            "step": step,
            "shard": path.name,
            "byte_offset": path.stat().st_size if path.exists() else 0,
        }
        with pending_temporary.open("w", encoding="utf-8") as handle:
            handle.write(_encode(pending) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        pending_temporary.replace(pending_path)
        _fsync_directory(self.directory)
        self._pending[step] = pending

        path = self.write(step, records)
        marker: dict[str, object] = {
            "version": 1,
            "step": step,
            "records_per_update": len(records),
            "sample_indices": indices,
            "records_sha256": _records_digest(self._records[step]),
            "prompt_loader_state_after": prompt_loader_state_after,
        }
        marker_path = self.directory / f"{self.run_name}_live_step_{step:08d}.json"
        temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_encode(marker) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(marker_path)
        _fsync_directory(self.directory)
        self._markers[step] = marker
        pending_path.unlink()
        self._pending.pop(step)
        _fsync_directory(self.directory)
        return path


def _fsync_directory(directory: Path) -> None:
    """Make file creation, replacement, and removal durable on Linux."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encode(record: Mapping[str, object]) -> str:
    """Serialize one record with the released TMax writer's conversions."""
    return json.dumps(record, default=_json_default)


def _json_default(value: object) -> object:
    """Match the released TMax writer's JSON conversion rules."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return cast("object", value.value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray | np.generic):
        plain: object = value.tolist()
        return plain
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return str(value)


def _records_digest(records: Sequence[Mapping[str, object]]) -> str:
    """Hash records by sample index using the writer's JSON conversion."""
    normalized = sorted(
        (json_loads(_encode(record), dict[str, object]) for record in records),
        key=lambda record: from_plain(record["sample_idx"], int),
    )
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
