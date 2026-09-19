#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prepare the pinned ARC-AGI1 ``arc1concept-aug-1000`` dataset.

The default destination matches ``ArcData.Config`` under a default
``TrainLoop``. The source revision is immutable; a local ``--input-prefix``
keeps tests and offline rebuilds hermetic.

Examples:
  prepare_data.py
  prepare_data.py --directory /datasets/my-arcagi1

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import logging
import subprocess
import tempfile

from priml.baselines.arcagi1.data import ArcData
from priml.baselines.arcagi1.scripts.build_dataset import build_arc_dataset
from priml.lib.custom_json import DictCodec, IntCodec, loads
from priml.train.train_loop import TrainLoop


SOURCE_URL: Final = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git"
"""Pinned upstream source repository."""

SOURCE_REVISION: Final = "c01103738605ba39d1430519b1ee0c62f4c707f8"
"""Immutable upstream commit containing the ARC source files."""


def main() -> int:
    """Prepare the dataset; return the process exit code.

    Returns:
      result: Process exit code.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = ArcData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    if flags.directory is not None:
        config.working_dir = flags.directory
    config.augmentation.num_aug = flags.num_aug
    config.augmentation.seed = flags.seed
    prepare(config, input_file_prefix=flags.input_prefix)
    return 0


def default_directory() -> Path:
    """Return the directory a default ``TrainLoop`` resolves for ARC data."""
    config = ArcData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def prepare(
    config: ArcData.Config,
    *,
    input_file_prefix: Path | str | None = None,
) -> Path:
    """Build the ARC tree with a pinned source and deterministic metadata.

    Args:
      config: Dataset location and offline augmentation recipe.
      input_file_prefix: Local ``*_challenges.json`` prefix for hermetic builds.

    Returns:
      directory: Prepared dataset root.

    """
    config = config.copy_tree().finalize()
    target = Path(config.working_dir)
    augmentation = config.augmentation.make()
    if input_file_prefix is not None:
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(input_file_prefix),
            augmentation=augmentation,
        )
        return target

    with _pinned_source() as prefix:
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(prefix),
            augmentation=augmentation,
        )
    return target


def num_puzzle_identifiers(directory: Path | str) -> int:
    """Read the identifier-table size recorded by a prepared dataset."""
    metadata = DictCodec.coerce(
        loads((Path(directory) / "train" / "dataset.json").read_text()),
    )
    return IntCodec.coerce(metadata["num_puzzle_identifiers"])


class _PinnedSource:
    """Clone the source at one revision and verify the checked-out digest."""

    def __init__(self) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory(prefix="arcagi1-source-")
        clone = Path(self._temporary.name) / "TinyRecursiveModels"
        subprocess.run(  # noqa: S603 -- Fixed Git executable and repository URL.
            ["git", "clone", "--no-checkout", SOURCE_URL, str(clone)],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed Git executable and revision.
            ["git", "-C", str(clone), "checkout", SOURCE_REVISION],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        actual = subprocess.run(  # noqa: S603 -- Fixed Git executable and revision.
            ["git", "-C", str(clone), "rev-parse", "HEAD"],  # noqa: S607 -- Fixed Git executable.
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != SOURCE_REVISION:
            raise RuntimeError(
                f"source revision mismatch: got {actual}, expected {SOURCE_REVISION}",
            )
        return clone / "kaggle" / "combined" / "arc-agi"

    def __exit__(self, *exc: object) -> None:
        del exc
        if self._temporary is not None:
            self._temporary.cleanup()


def _pinned_source() -> _PinnedSource:
    """Return a context manager for the verified source checkout."""
    return _PinnedSource()


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument("--input-prefix", type=Path, default=None)
    parser.add_argument("--num-aug", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)


class _Flags(Protocol):
    """Parsed command-line flags."""

    directory: Path | None
    input_prefix: Path | None
    num_aug: int
    seed: int


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
