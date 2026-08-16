#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download CIFAR-10 and cache it as normalized tensors.

Run once before the first experiment. Idempotent: a split already present is
left alone, so re-running costs nothing.

The default destination matches the one ``Cifar10Data.Config`` resolves under a
default ``TrainLoop``, so preparing and training agree without either naming a
path. Point both elsewhere with ``--directory`` here and
``--override dataset.working_dir=...`` at launch.

Examples:
  prepare_data.py
  prepare_data.py --directory /datasets/my-cifar10

'''
# fmt: on

from __future__ import annotations

from pathlib import Path

import argparse
import logging

from priml.baselines.cifar10.data import Cifar10Data, prepare
from priml.train.train_loop import TrainLoop


def main() -> int:
    """Prepare the dataset; return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(args.directory)
    return 0


def default_directory() -> Path:
    """Return the dataset directory a default ``TrainLoop`` would resolve.

    Returns:
      directory: Prepared-tensor location shared by the preparer and the loop.

    """
    config = Cifar10Data.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_directory(),
        help="Destination for train.pt and test.pt.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
