#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Prepare the pinned ARC-AGI-1 ``arc1concept-aug-1000`` dataset.

The default destination matches ``exp000``'s dataset under a default
``TrainLoop``. The source revision is immutable; a local ``--input-prefix``
keeps tests and offline rebuilds hermetic. The ARC-AGI-2 rungs build their
own tree on first use.

Examples:
  prepare_data.py
  prepare_data.py --base-dir /data/scratch --experiment exp005
  prepare_data.py --directory /datasets/my-arcagi
  prepare_data.py --experiment exp005

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import logging

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.puzzle_data import Arc2PuzzleDataset
from priml.baselines.arcagi.scripts.build_dataset import build_arc_dataset


if TYPE_CHECKING:
    from priml.baselines.arcagi.data import PuzzleData


def main() -> int:
    """Prepare the dataset; return the process exit code.

    Returns:
      result: Process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = dataset_config(flags.experiment)
    if flags.base_dir is not None:
        config.base_dir = flags.base_dir
    if flags.directory is not None:
        # An explicit directory is a filesystem path, not a logical one under the
        # experiment's resource root.
        config.base_dir = None
        config.working_dir = flags.directory
    if flags.num_aug is not None:
        config.augmentation.num_aug = flags.num_aug
    if flags.seed is not None:
        config.augmentation.seed = flags.seed
    prepare(config, input_file_prefix=flags.input_prefix)
    return 0


def dataset_config(experiment: str) -> PuzzleData.Config:
    """Return the dataset config the named ARC experiment trains on.

    Args:
      experiment: Name of a factory in :mod:`~priml.baselines.arcagi.experiments`.

    Returns:
      config: A copy of that experiment's dataset config, rooted at its base.

    """
    loop = cast(_Experiment, getattr(experiments, experiment))()
    config = loop.dataset.copy_tree()
    config.base_dir = loop.base_dir
    return config


def default_directory() -> Path:
    """Return the directory ``exp000`` resolves for ARC data."""
    return Path(dataset_config("exp000").finalize().working_dir)


def prepare(
    config: PuzzleData.Config,
    *,
    input_file_prefix: Path | str | None = None,
) -> Path:
    """Build the ARC tree with a pinned source and deterministic metadata.

    Args:
      config: Dataset location and offline augmentation recipe.
      input_file_prefix: Local ``*_challenges.json`` prefix for hermetic builds.

    Returns:
      directory: Prepared dataset root.

    Raises:
      TypeError: If ``config`` reads an ARC-AGI-2 tree, which this builder
        does not write.

    """
    if isinstance(config, Arc2PuzzleDataset.Config):
        raise TypeError(
            "an ARC-AGI-2 rung builds its own tree on first use; this builder "
            "writes only ARC-AGI-1 trees.",
        )
    config = config.copy_tree().finalize()
    target = Path(config.working_dir)
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=None if input_file_prefix is None else str(input_file_prefix),
        augmentation=config.augmentation.make(),
    )
    return target


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="resource root; the tree lands where the experiment reads it",
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument(
        "--experiment",
        default="exp000",
        help="experiment whose dataset recipe to build",
    )
    parser.add_argument("--input-prefix", type=Path)
    parser.add_argument("--num-aug", type=int)
    parser.add_argument("--seed", type=int)


class _Flags(Protocol):
    """Parsed command-line flags."""

    base_dir: Path | None
    directory: Path | None
    experiment: str
    input_prefix: Path | None
    num_aug: int | None
    seed: int | None


class _Loop(Protocol):
    """The two fields of an experiment config the preparer reads."""

    base_dir: Path | str | None

    @property
    def dataset(self) -> PuzzleData.Config:
        """Return the dataset config."""
        ...


class _Experiment(Protocol):
    def __call__(self) -> _Loop: ...


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
