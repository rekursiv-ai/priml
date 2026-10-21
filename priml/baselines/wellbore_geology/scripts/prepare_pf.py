#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prepare an experiment's particle-filter and backward-trajectory caches.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import logging

from configgle.launch import resolve_config

from priml.baselines.wellbore_geology.data import discover_wells, load_raw
from priml.baselines.wellbore_geology.particles import prepare_particles
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.paths import validated_output_path


logger = logging.getLogger(__name__)


def main() -> int:
    """Prepare every particle-bearing family of the named experiment.

    Returns:
      status: Zero after complete preparation succeeds.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    study = resolve_config(flags.experiment)
    assert isinstance(study, WellboreStudy.Config)
    prepare(study, check_only=flags.check_only)
    return 0


class Flags(Protocol):
    """Explicit experiment and mode."""

    experiment: str
    check_only: bool


def prepare(study: WellboreStudy.Config, *, check_only: bool) -> None:
    """Generate the particle caches of every family the study declares.

    Args:
      study: Experiment whose dataset directory and family caches are staged.
      check_only: Validate paired well inventory without generating records.

    """
    study = study.finalized()
    assert isinstance(study.dataset_directory, Path)
    for split in ("train", "test"):
        names = discover_wells(study.dataset_directory / split)
        logger.info("%s: %d complete wells", split, len(names))
        if check_only:
            continue
        wells = {
            name: load_raw(study.dataset_directory / split, name) for name in names
        }
        for family in study.families:
            if family.particle is None or family.pf_directory is None:
                continue
            prepare_particles(
                wells,
                config=family.particle,
                directory=validated_output_path(
                    family.pf_directory / split,
                    protected=(study.dataset_directory,),
                ),
                has_target=split == "train",
            )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "experiment",
        help="Dotted path of the experiment factory, such as the exp000 function.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate complete paired inputs without preparing caches.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
