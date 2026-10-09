#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prepare pinned inputs for TMax exp000 and a separate published rollout shard.

The 4B checkpoint comes from the repository named in TMax's training script.
The ``tmax-4b`` model card points to a different base model, so it is not used
here. The training rows and terminal tasks come from a pinned dataset revision.

The published rollout shard comes from a 9B run, not exp000's 4B model.
It is kept for comparison and debugging, not used to train exp000.

The live runner uses the local training rows and tasks through TMax's own
loaders, rather than downloading them again.

The script writes a manifest with file hashes and source revisions. On later
runs, it checks the recorded files and refuses missing, changed, or extra
files. Without a manifest, it refuses to reuse existing assets.

Examples:
  prepare_data.py
  prepare_data.py --checkpoint-dir /models/qwen --rollouts-dir /data/rollouts

'''
# fmt: on

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tarfile
import urllib.request

from huggingface_hub import hf_hub_download, list_repo_files

import pyarrow as pa

from priml.baselines.tmax.experiments import exp000
from priml.baselines.tmax.scripts import upstream
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# The released 4B launcher names this checkpoint.
CHECKPOINT_REPO: Final = "hamishivi/Qwen3.5-4B"
CHECKPOINT_REVISION: Final = "956c2718b66ff38a3b8427ce824704f5a467407f"
DATASET_REPO: Final = "allenai/tmax-15k-open-instruct"
DATASET_REVISION: Final = "7b090eca98bf351356bc1c64290c5c4a09f2f98c"

# The recorded-fixture script uses this same archive and hash.
ARCHIVE_URL: Final = (
    "https://huggingface.co/allenai/tmax-9b/resolve/main/"
    "rollouts/archives/swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677/"
    "rollouts/rollouts.tar.zst.part-000"
)
ARCHIVE_SHA256: Final = (
    "597f4f36ab78ff13d6917445eb1cd2f5ad4c219ff7b987de2110a59ac86b8632"
)
# The rollout loader looks for *_rollouts_*.jsonl.
SHARD_NAME: Final = "tmax_9b_rollouts_part000.jsonl"
MANIFEST_NAME: Final = "exp000_assets_manifest.json"
"""Manifest stored beside the published rollout shard."""
TASK_DATA_DIRNAME: Final = "task-data.tar.gz.extracted"
"""Directory name used by TMax for extracted terminal tasks."""


class _Flags(Protocol):
    """The parsed command line."""

    base_dir: Path
    checkpoint_dir: Path | None
    rollouts_dir: Path | None
    dataset_dir: Path | None
    upstream_dir: Path | None
    manifest_path: Path | None


def default_directories(
    base_dir: Path | str = "/opt/scratch",
) -> tuple[Path, Path]:
    """Return separate locations for the checkpoint and published rollout."""
    config = exp000()
    config.base_dir = base_dir
    config = config.finalize()
    return (
        Path(cast("Path | str", config.step.model_path)),
        resolve_working_dir(base_dir, "/datasets/tmax/rollouts/published"),
    )


def main() -> int:
    """Stage the TMax inputs and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/scratch"),
        help="Root for every default asset directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Where the policy checkpoint lives (default: exp000's model_path).",
    )
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=None,
        help=(
            "Where the published 9B parity shard is staged; exp000's own live"
            " rollouts are written to a different directory"
            " (default: base_dir/datasets/tmax/rollouts/published)."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Where to write the deterministic asset manifest "
            "(default: rollouts-dir/" + MANIFEST_NAME + ")."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Where the pinned RL dataset is staged.",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help=(
            "Where TMax is staged "
            f"({upstream.UPSTREAM_REPO}@{upstream.UPSTREAM_SHORT})."
        ),
    )
    flags = cast(_Flags, parser.parse_args())
    checkpoint_default, rollouts_default = default_directories(flags.base_dir)
    flags.checkpoint_dir = flags.checkpoint_dir or checkpoint_default
    flags.rollouts_dir = flags.rollouts_dir or rollouts_default
    flags.dataset_dir = flags.dataset_dir or resolve_working_dir(
        flags.base_dir,
        "/datasets/tmax/tmax-15k-open-instruct",
    )
    flags.upstream_dir = flags.upstream_dir or resolve_working_dir(
        flags.base_dir,
        "/datasets/tmax/upstream/tmax",
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest_target = flags.manifest_path or (flags.rollouts_dir / MANIFEST_NAME)
    _verify_staged_assets(
        manifest_target,
        flags.checkpoint_dir,
        flags.rollouts_dir,
        flags.dataset_dir,
    )
    _stage_checkpoint(flags.checkpoint_dir)
    _stage_rollouts(flags.rollouts_dir)
    _stage_dataset(flags.dataset_dir)
    _stage_upstream(flags.upstream_dir)
    manifest = _write_manifest(
        flags.checkpoint_dir,
        flags.rollouts_dir,
        dataset_dir=flags.dataset_dir,
        upstream_dir=flags.upstream_dir,
        path=flags.manifest_path,
    )
    logger.info(
        "Staged. exp000 collects its own rollouts through the live TMax seam,"
        " and the released writer's tool_mask omission is already covered"
        " there, so no sidecar is needed for it. The staged shard is the"
        " published 9B parity fixture, which predates that field; training on"
        " it (rather than on live 4B rollouts) would need"
        " --override dataset.tool_mask_path=... .",
    )
    logger.info("manifest: %s", manifest)
    return 0


def _verify_staged_assets(
    manifest_path: Path,
    checkpoint_dir: Path,
    rollouts_dir: Path,
    dataset_dir: Path,
) -> None:
    """Check that existing assets match the manifest.

    Without a manifest, existing files cannot be tied to the pinned sources,
    so the script refuses to reuse them. With one, it checks each recorded
    file's hash and rejects files the manifest does not list.

    Args:
      manifest_path: The manifest to check, when one exists.
      checkpoint_dir: The staged checkpoint directory.
      rollouts_dir: The staged rollout fixture directory.
      dataset_dir: The staged dataset directory.

    Raises:
      ValueError: The manifest is unreadable or the assets do not match it.
      TypeError: The manifest is not an object with a file map.

    """
    if not manifest_path.is_file():
        existing: dict[str, list[str]] = {}
        for label, directory in (
            ("checkpoint", checkpoint_dir),
            ("rollouts", rollouts_dir),
            ("dataset", dataset_dir),
        ):
            files = _relative_files(directory)
            if files:
                existing[label] = files
        if existing:
            raise ValueError(
                "Refusing to label existing assets as pinned without a prior "
                f"manifest: {existing}. Re-stage into clean directories.",
            )
        return
    try:
        raw_payload = cast(
            "object",
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{manifest_path} could not be read: {error}.") from error
    if not isinstance(raw_payload, dict):
        raise TypeError(f"{manifest_path} must contain an object.")
    payload = cast("dict[str, object]", raw_payload)
    _verify_manifest_identity(payload, manifest_path)
    files = payload.get("files", {})
    if not isinstance(files, Mapping):
        raise TypeError(f"{manifest_path} records no file map.")
    file_items = cast("Mapping[object, object]", files)
    # The writer records each SHA-256 hash as a 64-character string.
    if not all(
        isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
        for name, digest in file_items.items()
    ):
        raise TypeError(f"{manifest_path} contains an invalid file digest map.")
    recorded = cast("dict[str, str]", files)
    for label, directory in (
        ("checkpoint", checkpoint_dir),
        ("rollouts", rollouts_dir),
        ("dataset", dataset_dir),
    ):
        prefix = f"{label}/"
        expected = {
            name.removeprefix(prefix): digest
            for name, digest in recorded.items()
            if name.startswith(prefix)
        }
        changed = sorted(
            name
            for name, digest in expected.items()
            if not _matches_digest(directory / name, digest)
        )
        if changed:
            raise ValueError(
                f"Staged {label} assets no longer match {manifest_path}: "
                f"{changed}. Re-stage into a clean directory, or pass a fresh "
                "--manifest-path to record the new state deliberately.",
            )
        actual = set(_relative_files(directory, exclude=manifest_path))
        if actual != set(expected):
            raise ValueError(
                f"Staged {label} file inventory no longer matches "
                f"{manifest_path}: expected {sorted(expected)}, got "
                f"{sorted(actual)}. Re-stage into a clean directory.",
            )


def _verify_manifest_identity(payload: dict[str, object], path: Path) -> None:
    """Check that the manifest names the expected pinned sources."""
    expected: dict[str, object] = {
        "schema": 1,
        "experiment": "exp000",
        "upstream_tmax_commit": upstream.UPSTREAM_SHORT,
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
        },
        "rollouts": {
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "shard_name": SHARD_NAME,
        },
    }
    changed = [key for key, value in expected.items() if payload.get(key) != value]
    if changed:
        raise ValueError(
            f"{path} was produced from different source pins: {changed}.",
        )


def _relative_files(directory: Path, *, exclude: Path | None = None) -> list[str]:
    """List files below ``directory`` by relative path."""
    if not directory.is_dir():
        return []
    return [
        file.relative_to(directory).as_posix()
        for file in sorted(directory.rglob("*"))
        if file.is_file() and file != exclude
    ]


def _matches_digest(path: Path, digest: str) -> bool:
    """Return whether ``path`` exists with the recorded sha256."""
    if not path.is_file():
        return False
    return _sha256(path) == digest


def _stage_checkpoint(directory: Path) -> None:
    """Download the pinned checkpoint if its config file is missing."""
    if (directory / "config.json").is_file():
        logger.info("checkpoint: already staged at %s", directory)
        return
    logger.info("checkpoint: %s @ %s", CHECKPOINT_REPO, CHECKPOINT_REVISION)
    for name in list_repo_files(CHECKPOINT_REPO, revision=CHECKPOINT_REVISION):
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=name,
            revision=CHECKPOINT_REVISION,
            local_dir=directory,
        )


def _stage_rollouts(directory: Path) -> None:
    """Reuse a staged rollout file, or check its archive and extract it."""
    target = directory / SHARD_NAME
    if target.is_file():
        logger.info("rollouts: already staged at %s", target)
        return
    archive = _archive_path(directory)
    if archive.exists():
        try:
            _verify_sha256(archive)
        except ValueError:
            logger.warning("rollouts: replacing corrupt cached archive at %s", archive)
            archive.unlink()
    if not archive.exists():
        logger.info("rollouts: downloading %s", ARCHIVE_URL)
        archive.parent.mkdir(parents=True, exist_ok=True)
        _download(ARCHIVE_URL, archive)
    directory.mkdir(parents=True, exist_ok=True)
    _extract_shard(archive, target)
    logger.info("rollouts: staged %s", target)


def _stage_dataset(directory: Path) -> None:
    """Download training rows and terminal tasks from the pinned dataset."""
    marker = directory / "data" / "train-00000-of-00001.parquet"
    task_archive = directory / "task-data.tar.gz"
    task_data = directory / TASK_DATA_DIRNAME
    if marker.is_file() and task_archive.is_file() and task_data.is_dir():
        logger.info("dataset: already staged at %s", directory)
        return
    if not marker.is_file() or not task_archive.is_file():
        logger.info("dataset: %s @ %s", DATASET_REPO, DATASET_REVISION)
        for name in list_repo_files(
            DATASET_REPO,
            revision=DATASET_REVISION,
            repo_type="dataset",
        ):
            hf_hub_download(
                repo_id=DATASET_REPO,
                filename=name,
                revision=DATASET_REVISION,
                repo_type="dataset",
                local_dir=directory,
            )
    _extract_task_data(task_archive, task_data)


def _extract_task_data(archive: Path, target: Path) -> None:
    """Extract terminal tasks before moving them to their final directory."""
    if target.is_dir():
        return
    if not archive.is_file():
        raise FileNotFoundError(f"Pinned task archive is missing: {archive}")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError(
            f"Incomplete task extraction exists at {partial}; re-stage into a "
            "clean dataset directory.",
        )
    partial.mkdir(parents=True)
    # Leave incomplete extraction in place so the next run refuses it.
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(partial, filter="data")
    partial.replace(target)


def _stage_upstream(directory: Path) -> None:
    """Clone TMax at the pinned commit, or check an existing copy.

    Args:
      directory: Destination for the pinned TMax checkout.

    Raises:
      ValueError: The existing copy is at another commit or has local changes.
      subprocess.CalledProcessError: Cloning or checkout fails.

    """
    try:
        upstream.verify_checkout(directory)
    except FileNotFoundError:
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603 -- Fixed Git executable and repository URL.
            ["git", "clone", upstream.UPSTREAM_REPO, str(directory)],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed Git executable and revision.
            ["git", "-C", str(directory), "checkout", upstream.UPSTREAM_COMMIT],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        upstream.verify_checkout(directory)
    else:
        logger.info("upstream: already staged at %s", directory)


def _download(url: str, target: Path) -> None:
    """Check the downloaded archive's hash before moving it into place."""
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        try:
            _verify_sha256(partial)
        except ValueError:
            partial.unlink()
        else:
            partial.replace(target)
            _fsync_directory(target.parent)
            return
    urllib.request.urlretrieve(  # noqa: S310 -- The URL is a fixed HTTPS dataset endpoint.
        url,
        partial,
    )
    _verify_sha256(partial)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    partial.replace(target)
    _fsync_directory(target.parent)


def _fsync_directory(directory: Path) -> None:
    """Flush the cache directory after moving a downloaded archive."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive_path(rollouts_dir: Path) -> Path:
    """Keep the downloaded archive beside the published rollout directory."""
    return rollouts_dir.parent / "cache" / "tmax_rollouts_part000.tar.zst"


def _extract_shard(archive: Path, target: Path) -> None:
    """Extract one rollout file and move it into place when complete.

    The published archive is a zstd-compressed tar. PyArrow decompresses it
    because Python 3.12 cannot read this format directly. A file in another
    format is rejected rather than treated as a rollout archive.
    """
    members: list[str] = []
    payload: bytes | None = None
    with _zstd_tar(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            members.append(member.name)
            stream = tar.extractfile(member)
            if stream is None:
                raise TypeError(f"{member.name} could not be extracted.")
            content = stream.read()
            if payload is None:
                payload = content
    if len(members) != 1:
        raise ValueError(f"Expected one JSONL shard, found {len(members)}.")
    if payload is None:
        raise TypeError(f"{members[0]} could not be extracted.")
    # Write to a temporary file so a failed extraction cannot look complete.
    partial = target.with_suffix(".jsonl.partial")
    partial.write_bytes(payload)
    partial.replace(target)


@contextmanager
def _zstd_tar(archive: Path) -> Generator[tarfile.TarFile]:
    """Read the zstd-compressed archive as a stream.

    Args:
      archive: The ``.tar.zst`` file to read.

    Yields:
      tar: The tar archive. Its files must be read in order because the
        decompressed stream cannot seek backward.

    Raises:
      tarfile.ReadError: The file is not a zstd-compressed tar.

    """
    source = pa.CompressedInputStream(pa.OSFile(str(archive), "rb"), "zstd")
    try:
        try:
            # The decompressed stream cannot seek, so read the tar in order.
            with tarfile.open(fileobj=source, mode="r|") as tar:
                yield tar
        except OSError as error:
            # Give callers one clear error for an archive with the wrong format.
            raise tarfile.ReadError(
                f"{archive} is not a zstd-compressed tar: {error}",
            ) from error
    finally:
        source.close()


def _verify_sha256(archive: Path) -> None:
    """Check the downloaded archive against its pinned SHA-256 hash."""
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError(
            f"{archive} is not the pinned archive: {digest.hexdigest()}",
        )


def _write_manifest(
    checkpoint_dir: Path,
    rollouts_dir: Path,
    *,
    dataset_dir: Path | None = None,
    upstream_dir: Path | None = None,
    path: Path | None = None,
) -> Path:
    """Record asset hashes and their pinned sources in a manifest."""
    target = path or (rollouts_dir / MANIFEST_NAME)
    files: dict[str, str] = {}
    for label, directory in (
        ("checkpoint", checkpoint_dir),
        ("rollouts", rollouts_dir),
        *((("dataset", dataset_dir),) if dataset_dir is not None else ()),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Cannot manifest missing {label} directory: {directory}",
            )
        for name in _relative_files(directory, exclude=target):
            files[f"{label}/{name}"] = _sha256(directory / name)
    payload = {
        "schema": 1,
        "experiment": "exp000",
        "upstream_tmax_commit": upstream.UPSTREAM_SHORT,
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
        },
        "rollouts": {
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "shard_name": SHARD_NAME,
        },
        "files": files,
    }
    if upstream_dir is not None:
        payload["upstream"] = {
            "repo": upstream.UPSTREAM_REPO,
            "commit": upstream.UPSTREAM_COMMIT,
            "path": str(upstream_dir),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(target)
    return target


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
