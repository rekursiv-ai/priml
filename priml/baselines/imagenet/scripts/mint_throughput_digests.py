#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Re-mint the frozen reference digests the throughput experiments check against.

Drains the throughput ``exp000`` reference pipeline once, in a fresh process,
over the image set ``prepare_throughput_data.py`` staged, and writes the image
set's hash and the SHA-256 of every batch's ``image`` and ``label`` tensors.

The digests pin bytes, so they hold only where the staged JPEGs and the decode
stack (libjpeg-turbo, OpenCV) match the minting machine. To check a machine,
run this at the commit that last changed the digest file and diff the output
against it; to adopt a new image set or a new decode stack, commit the
re-minted file. A re-mint means the reference itself moved.

Examples:
  mint_throughput_digests.py
  mint_throughput_digests.py --base-dir /tmp/scratch --output /tmp/digests.sha256

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import logging

from priml.baselines import imagenet
from priml.baselines.imagenet.throughput_experiments import exp000


logger = logging.getLogger(__name__)


def main() -> int:
    """Mint the digests; return the process exit code.

    Returns:
      code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = exp000()
    if flags.base_dir is not None:
        cfg.base_dir = flags.base_dir
    output = flags.output or Path(imagenet.__file__).with_name(
        str(cfg.reference_digests),
    )
    cfg.make().mint_reference().write(
        output,
        header=(
            "Throughput exp000 reference on the prepare_throughput_data.py set.\n"
            "Re-mint with scripts/mint_throughput_digests.py; never edit by hand."
        ),
    )
    logger.info("Wrote %s.", output)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Resource root the staged image set resolves beneath (exp000's).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the digests; defaults to the committed file.",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    base_dir: Path | None
    output: Path | None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
