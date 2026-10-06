#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download released checkpoints from the Hub into the local run-dir layout.

Fetches the released checkpoints and places each one EXACTLY where the
evaluation experiments (exp011-exp014) default their checkpoint paths:

    <scratch>/runs/<name>/checkpoints/step_<n:08d>.pt

so eval-only flows run without local training. The model-only flavor
(default) is placed as the step file; ``--flavor full`` fetches the full
training state instead (resume flows). Existing destination files are skipped
only while their source receipt and content still match, so reruns are
idempotent without hiding source updates or local corruption.

When the Hub repo publishes a ``SHA256SUMS`` file (sha256sum format, paths
relative to the repo root), every downloaded file is verified against it: a
digest mismatch deletes the bad file and aborts with a clear message. A repo
without ``SHA256SUMS`` (or a file without an entry) downloads with a logged
note instead of failing.

Usage:
    python -m priml.baselines.sudoku.scripts.download_checkpoints \
        --names exp006:exp010 \
        [--repo-id rekursiv-ai/sudoku-trm] [--dest DIR] [--step N] \
        [--flavor model|full]
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
)


if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Download the named checkpoints and log a placement summary.

    Args:
      argv: Argument list; None reads ``sys.argv``.

    Returns:
      exit_code: 0 on success.

    Raises:
      SystemExit: A requested name or step is absent from the Hub repo.

    """
    args = _parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    dest = Path(args.dest) if args.dest is not None else Path("/opt/scratch")
    filename = {"model": "model.pt", "full": "full_state.pt"}[args.flavor]
    files = [str(f) for f in list_repo_files(args.repo_id)]
    checksums = _load_checksums(args.repo_id, files)
    rows: list[str] = []
    for requested in args.names:
        name, _, run_name = requested.partition(":")
        run_name = run_name or name
        steps = available_steps(files, name=name, filename=filename)
        if not steps:
            named = sorted({f.partition("/")[0] for f in files if "/step_" in f})
            raise SystemExit(
                f"No {filename!r} checkpoints for {name!r} in {args.repo_id}; "
                f"available names: {', '.join(named) or 'none'}.",
            )
        step = steps[-1] if args.step is None else args.step
        if step not in steps:
            raise SystemExit(
                f"Step {step} not available for {name!r} in {args.repo_id} "
                f"(available steps: {steps}).",
            )
        target = destination_path(dest, name=run_name, step=step)
        repo_path = f"{name}/step_{step}/{filename}"
        expected_sha256 = None if checksums is None else checksums.get(repo_path)
        provenance_path = target.with_suffix(f"{target.suffix}.source.json")
        provenance = _read_provenance(provenance_path)
        if _existing_matches_source(
            target,
            provenance,
            repo_id=args.repo_id,
            repo_path=repo_path,
            expected_sha256=expected_sha256,
        ):
            status = "skipped, exists"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            cached = Path(str(hf_hub_download(args.repo_id, repo_path)))
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                staged = Path(temporary_file.name)
            try:
                shutil.copy2(cached, staged)
                note = _verify_download(staged, repo_path, checksums)
                downloaded_sha256 = _file_sha256(staged)
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
            expected_provenance = {
                "repo_id": args.repo_id,
                "repo_path": repo_path,
                "sha256": downloaded_sha256,
            }
            _write_provenance(provenance_path, expected_provenance)
            status = f"downloaded, {target.stat().st_size / 1024**2:.1f} MB{note}"
        rows.append(f"  {name} step {step}: {target} [{status}]")
    logger.info(
        "Checkpoints from https://huggingface.co/%s (%s):",
        args.repo_id,
        filename,
    )
    for row in rows:
        logger.info("%s", row)
    return 0


def available_steps(files: Iterable[str], *, name: str, filename: str) -> list[int]:
    """List the steps of ``name`` that carry ``filename`` in the repo.

    Args:
      files: Repo file listing (paths relative to the repo root).
      name: Checkpoint name (top-level directory in the Hub repo).
      filename: Flavor file to require (``model.pt`` or ``full_state.pt``).

    Returns:
      steps: Ascending step numbers with that flavor present.

    """
    pattern = re.compile(rf"{re.escape(name)}/step_(\d+)/{re.escape(filename)}")
    return sorted(int(match.group(1)) for f in files if (match := pattern.fullmatch(f)))


def destination_path(dest: Path, *, name: str, step: int) -> Path:
    """Compute the run-dir-convention placement for one checkpoint.

    Args:
      dest: Scratch root (``--dest``, default ``/opt/scratch``).
      name: Checkpoint name; becomes the run (experiment) name.
      step: Training step.

    Returns:
      path: ``<dest>/runs/<name>/checkpoints/step_<step:08d>.pt`` -- where the
        eval configs default their checkpoint paths.

    """
    return dest / "runs" / name / "checkpoints" / f"step_{step:08d}.pt"


class _Flags(Protocol):
    """Parsed command-line flags."""

    names: list[str]
    repo_id: str
    dest: Path | None
    step: int | None
    flavor: str


def _parse_args(argv: list[str] | None) -> _Flags:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
    )
    _add_arguments(parser)
    return cast(_Flags, parser.parse_args(argv))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Checkpoint names to fetch, e.g. exp006 verifier_s0; "
        "HUB:RUN places Hub name HUB under run name RUN (exp006:exp010 "
        "puts the released generator where exp011 reads it).",
    )
    parser.add_argument(
        "--repo-id",
        default="rekursiv-ai/sudoku-trm",
        help="Source HuggingFace repo (default: rekursiv-ai/sudoku-trm).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination scratch root; defaults to /opt/scratch.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step to fetch for every name; default: the latest available.",
    )
    parser.add_argument(
        "--flavor",
        choices=["full", "model"],
        default="model",
        help="model: EMA-applied eval weights (default); full: the original "
        "training checkpoint for resume.",
    )


def _read_provenance(path: Path) -> dict[str, str] | None:
    """Read a checkpoint placement's source identity, or None when invalid."""
    try:
        value = cast(object, json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    typed_value = cast(dict[object, object], value)
    repo_id = typed_value.get("repo_id")
    repo_path = typed_value.get("repo_path")
    sha256 = typed_value.get("sha256")
    if not isinstance(repo_id, str):
        return None
    if not isinstance(repo_path, str):
        return None
    if not isinstance(sha256, str):
        return None
    return {"repo_id": repo_id, "repo_path": repo_path, "sha256": sha256}


def _existing_matches_source(
    target: Path,
    provenance: dict[str, str] | None,
    *,
    repo_id: str,
    repo_path: str,
    expected_sha256: str | None,
) -> bool:
    """Return whether an existing placement still matches its source receipt."""
    if target.is_symlink() or not target.is_file() or provenance is None:
        return False
    if provenance.get("repo_id") != repo_id or provenance.get("repo_path") != repo_path:
        return False
    recorded_sha256 = provenance.get("sha256")
    if expected_sha256 is not None and recorded_sha256 != expected_sha256:
        return False
    try:
        return recorded_sha256 == _file_sha256(target)
    except OSError:
        return False


def _write_provenance(path: Path, value: dict[str, str]) -> None:
    """Atomically replace one source receipt without following destination links."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(json.dumps(value, sort_keys=True) + "\n")
        temporary = Path(temporary_file.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    """Hash one checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checksums(repo_id: str, files: list[str]) -> dict[str, str] | None:
    """Fetch and parse the repo's ``SHA256SUMS``; None when absent."""
    if "SHA256SUMS" not in files:
        logger.info(
            "Note: %s publishes no %s; downloads are not checksum-verified.",
            repo_id,
            "SHA256SUMS",
        )
        return None
    sums_path = Path(str(hf_hub_download(repo_id, "SHA256SUMS")))
    checksums: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue  # Not a sha256sum-format row.
        digest, path = parts
        checksums[path.lstrip("*")] = digest.lower()
    return checksums


def _verify_download(
    target: Path,
    repo_path: str,
    checksums: dict[str, str] | None,
) -> str:
    """Verify one downloaded file against the repo checksums."""
    if checksums is None:
        return ""
    expected = checksums.get(repo_path)
    if expected is None:
        logger.warning(
            "Note: %s has no entry for %s; file not verified.",
            "SHA256SUMS",
            repo_path,
        )
        return ", no checksum entry"
    actual = _file_sha256(target)
    if actual != expected:
        target.unlink()
        raise SystemExit(
            f"SHA-256 mismatch for {repo_path}:\n"
            f"  expected {expected} (from SHA256SUMS)\n"
            f"  got      {actual}\n"
            f"The corrupt file was deleted ({target}); rerun to re-download.",
        )
    return ", sha256 ok"


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
