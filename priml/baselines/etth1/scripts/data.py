"""Prepare the checksum-pinned public ETTh1 CSV before training."""
# ruff: noqa: S310 -- This is one fixed HTTPS URL.

from __future__ import annotations

from http.client import HTTPResponse
from pathlib import Path
from typing import Final, Protocol, cast
from urllib import request

import argparse
import hashlib
import tempfile

from priml.baselines.etth1.experiments import exp000


DATASET_COMMIT: Final = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
DATASET_URL: Final = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    f"{DATASET_COMMIT}/ETT-small/ETTh1.csv"
)
DATASET_SHA256: Final = (
    "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
)


def prepare(directory: Path, *, source: Path | None = None) -> Path:
    """Verify an existing dataset or atomically install the pinned public CSV.

    Args:
      directory: Destination directory, also passed to the experiment.
      source: Optional local copy for offline preparation.

    Returns:
      path: Verified ETTh1.csv.

    """
    path = directory / "ETTh1.csv"
    if path.exists():
        _verify(path.read_bytes())
        return path
    if source is not None:
        payload = source.read_bytes()
    else:
        # Only the pinned dataset URL is downloaded.
        with cast(
            HTTPResponse,
            request.urlopen(DATASET_URL, timeout=60),
        ) as response:
            payload = response.read()
    _verify(payload)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="data-", dir=directory) as staging:
        temporary = Path(staging) / "ETTh1.csv"
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path


def main() -> int:
    """Download or verify the dataset at the experiment's resolved location."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    print(prepare(flags.directory, source=flags.source))
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(exp000().copy_tree().finalize().dataset.working_dir),
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Use a local CSV instead of downloading.",
    )


def _verify(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise ValueError(
            "ETTh1 SHA-256 mismatch; refusing to use or replace unverified data.",
        )


class _Flags(Protocol):
    directory: Path
    source: Path | None
