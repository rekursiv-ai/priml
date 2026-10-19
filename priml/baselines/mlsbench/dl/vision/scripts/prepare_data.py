#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download a dataset, CIFAR-100 by default, and cache it as 8-bit tensors.

Run once before the first experiment. Idempotent: a split already present is
left alone, so re-running costs nothing.

The default destination matches the one the dataset's recipe resolves under a
default ``TrainLoop``, so preparing and training agree without either naming a
path. Point both elsewhere with ``--directory`` here and
``--override dataset.working_dir=...`` at launch.

Examples:
  prepare_data.py
  prepare_data.py --dataset fmnist
  prepare_data.py --directory /datasets/my-cifar100

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, cast

import argparse
import logging

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, prepare
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.settings import mobilenetv2_fmnist
from priml.train.train_loop import TrainLoop


def main() -> int:
    """Prepare the dataset; return the process exit code.

    Returns:
      code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(flags.directory or default_directory(flags.dataset), dataset=flags.dataset)
    return 0


def default_directory(dataset: Dataset = "cifar100") -> Path:
    """Return the directory a default ``TrainLoop`` resolves for ``dataset``'s recipe.

    Args:
      dataset: The dataset whose recipe is resolved: ``exp000`` for CIFAR-100,
        its ``mobilenetv2-fmnist`` setting for Fashion-MNIST, and the
        ``dl-activation-function`` recipe's directory for CIFAR-10.

    Returns:
      directory: Prepared-tensor location shared by the preparer and the loop.

    """
    config = Cifar100Data.Config()
    if dataset == "fmnist":
        config = mobilenetv2_fmnist(exp000()).dataset
    elif dataset == "cifar10":
        config.working_dir = "/datasets/cifar10"
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


type Dataset = Literal["cifar10", "cifar100", "fmnist"]
"""A ``--dataset`` of the reference script."""


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "fmnist"),
        default="cifar100",
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Destination for train.pt and test.pt; the recipe's own by default.",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    dataset: Dataset
    directory: Path | None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
