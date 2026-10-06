# ruff: noqa: INP001 (Fixture tooling that lives beside the artifacts it rewrites.)
"""Rebuild two test fixtures from a published TMax rollout archive.

Both fixtures come from one shard of the ``allenai/tmax-9b`` run. The shard
contains four training updates, steps 90 through 93. Each step has 8 prompts
and 32 samples per prompt, for 1,024 records in total.

The 9B source is intentional. As of 2026-10-06, Ai2 publishes TMax models at
4B, 8B, 9B, and 27B, but only the 9B repository provides rollout and training
log-probability archives. These fixtures test the rollout format and advantage
calculation without loading the 9B model. They are not training data for the
4B ``exp000`` run, which must collect fresh rollouts from its own policy.

- ``recorded_rollout.jsonl`` is the exact JSONL record for step 91,
  ``sample_idx`` 216, and ``prompt_idx`` 6. It preserves the published format,
  including the missing ``tool_mask``.
- ``recorded_advantages.json`` groups all 1,024 records by step and prompt.
  It copies the published rewards and advantages so ``rollouts_test.py`` can
  check PriML's centered-advantage calculation against upstream TMax.

The script checks the archive against a fixed SHA-256 digest before reading it.
By default, it rebuilds both fixtures in memory and compares them with the
committed files. Pass ``--write`` to replace the files, or ``--archive PATH``
to use an archive that is already downloaded.

Run from the repository root:

    uv --quiet run --frozen -- python \
        priml/baselines/tmax/testdata/regenerate_recorded_fixtures.py

Python 3.14 or newer is required because its standard ``tarfile`` module can
read zstd archives. The repository environment uses Python 3.14.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request


FIXTURES = Path(__file__).resolve().parent
RECORDED_LINE = FIXTURES / "recorded_rollout.jsonl"
RECORDED_ADVANTAGES = FIXTURES / "recorded_advantages.json"

ARCHIVE_URL = (
    "https://huggingface.co/allenai/tmax-9b/resolve/main/"
    "rollouts/archives/swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677/"
    "rollouts/rollouts.tar.zst.part-000"
)
ARCHIVE_SHA256 = "597f4f36ab78ff13d6917445eb1cd2f5ad4c219ff7b987de2110a59ac86b8632"
SHARD_RECORDS = 1024
GROUP_SIZE = 32
RECORDED_KEY = (91, 216, 6)
SOURCE_NOTE = (
    "https://huggingface.co/allenai/tmax-9b "
    "rollouts/archives/swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677"
)


def _archive(explicit: Path | None) -> Path:
    """Locate the pinned archive, downloading it once if no copy is given."""
    if explicit is not None:
        return explicit
    cache = Path(tempfile.gettempdir()) / "tmax_rollouts_part000.tar.zst"
    if not cache.exists():
        print(  # noqa: T201 -- A fixture script's product is this report.
            f"downloading {ARCHIVE_URL}",
        )
        urllib.request.urlretrieve(ARCHIVE_URL, cache)
    return cache


def _verify_sha256(archive: Path) -> None:
    """Fail unless the archive matches the pinned sha256."""
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError(
            f"{archive} is not the pinned archive: {digest.hexdigest()}",
        )


def _shard_lines(archive: Path) -> list[bytes]:
    """Read the archive's single rollout JSONL member as raw lines."""
    # Transparent mode: typeshed knows it, and 3.14 detects zstd through it.
    with tarfile.open(archive, "r:*") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".jsonl")
        ]
        if len(members) != 1:
            raise ValueError(f"Expected one JSONL shard, found {len(members)}.")
        stream = tar.extractfile(members[0])
        if stream is None:
            raise TypeError(f"{members[0].name} could not be extracted.")
        return stream.read().splitlines()


def _record(line: bytes) -> dict[str, object]:
    """Parse one raw JSONL record."""
    return cast(dict[str, object], json.loads(line))


def _int(record: dict[str, object], key: str) -> int:
    """Read one record field as an int, refusing anything else."""
    value = record[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}.")
    return value


def _recorded_line(lines: list[bytes], records: list[dict[str, object]]) -> bytes:
    """Return the raw line of the pinned (step, sample_idx, prompt_idx) record."""
    indexes = [
        index
        for index, record in enumerate(records)
        if (
            _int(record, "step"),
            _int(record, "sample_idx"),
            _int(record, "prompt_idx"),
        )
        == RECORDED_KEY
    ]
    if len(indexes) != 1:
        raise LookupError(
            f"Expected exactly one record at {RECORDED_KEY}, found {len(indexes)}.",
        )
    return lines[indexes[0]]


def _advantages_bytes(records: list[dict[str, object]]) -> bytes:
    """Rebuild the grouped published rewards and advantages, compact JSON."""
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(
            (_int(record, "step"), _int(record, "prompt_idx")),
            [],
        ).append(record)

    ordered: list[dict[str, object]] = []
    for key in sorted(groups):
        members = groups[key]
        sample_ids = [_int(member, "sample_idx") for member in members]
        if len(members) != GROUP_SIZE or sample_ids != sorted(sample_ids):
            raise ValueError(
                f"Group {key} is not {GROUP_SIZE} records in ascending file order.",
            )
        ordered.append(
            {
                "step": key[0],
                "prompt_idx": key[1],
                "sample_idx": sample_ids,
                "reward": [member["reward"] for member in members],
                "advantage": [member["advantage"] for member in members],
            },
        )

    if len(ordered) != SHARD_RECORDS // GROUP_SIZE:
        raise ValueError(
            f"Expected {SHARD_RECORDS // GROUP_SIZE} groups, built {len(ordered)}.",
        )
    payload: dict[str, object] = {
        "source": SOURCE_NOTE,
        "archive_sha256": ARCHIVE_SHA256,
        "num_samples_per_prompt_rollout": GROUP_SIZE,
        "advantage_normalization_type": "centered",
        "groups": ordered,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _check_or_write(path: Path, rebuilt: bytes, write: bool) -> bool:
    """Compare the rebuilt bytes with the committed fixture, or rewrite it."""
    if path.exists() and path.read_bytes() == rebuilt:
        print(  # noqa: T201 -- A fixture script's product is this report.
            f"{path.name}: identical",
        )
        return True
    if not write:
        state = "missing" if not path.exists() else "differs from the committed fixture"
        print(  # noqa: T201 -- A fixture script's product is this report.
            f"{path.name}: {state}; pass --write to regenerate it",
        )
        return False
    path.write_bytes(rebuilt)
    print(  # noqa: T201 -- A fixture script's product is this report.
        f"{path.name}: rewrote {len(rebuilt)} bytes",
    )
    return True


def main() -> None:
    """Rebuild both recorded fixtures from the pinned archive and verify."""
    parser = argparse.ArgumentParser(
        description="Regenerate the recorded TMax fixtures from the pinned archive.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Reuse a downloaded archive instead of fetching the pinned one.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the fixtures; without it the script only verifies.",
    )
    args = parser.parse_args()
    # argparse leaves every attribute as Any; narrow at the boundary.
    archive = _archive(cast("Path | None", args.archive))
    write = cast(bool, args.write)

    _verify_sha256(archive)
    lines = _shard_lines(archive)
    if len(lines) != SHARD_RECORDS:
        raise ValueError(f"Expected {SHARD_RECORDS} records, found {len(lines)}.")
    records = [_record(line) for line in lines]

    ok_line = _check_or_write(
        RECORDED_LINE,
        _recorded_line(lines, records) + b"\n",
        write,
    )
    ok_advantages = _check_or_write(
        RECORDED_ADVANTAGES,
        _advantages_bytes(records),
        write,
    )
    if not (ok_line and ok_advantages):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
