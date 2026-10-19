#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the optimization-gradient-compression script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``optimization-gradient-compression`` script and
each baseline's edit of it from a clone of the benchmark, runs the port on the
same inputs under the same pins, and compares with exact equality: every
compressor on three rounds of fixed gradients with its error-feedback
residuals, every setting's network (ResNet-20, VGG-11, ResNet-56) initial state
and forward, where the generator stands after building each, the rate of all
200 epochs and the optimizer of each setting, the epochs evaluated, the
reported accuracy, the test metrics, the test pipeline, every pixel both
datasets' loaders produce, and every parameter after each update of a short
run of the script's own ``train`` per compressor. It counts the random draws
on both sides. Any mismatch is printed and the exit code is 1; otherwise the
values the default test replays are written under ``--out``.

The script is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import tempfile

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.opt.gradient_compression import (
    reference,
    upstream,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from torch import Tensor


REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "random_@data/dataloader.py",
    "randperm@data/sampler.py",
    "randint@transforms/transforms.py",
    "rand@transforms/transforms.py",
    "rand_like@",
)
"""Random call sites of the script's stack that a capture run must reach."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "random_@vision/data.py",
    "randperm@vision/data.py",
    "randint@vision/data.py",
    "rand@vision/data.py",
    "rand_like@gradient_compression/compressors.py",
)
"""Random call sites of the port that a capture run must reach."""


def main() -> int:
    """Capture and compare; return the process exit code.

    Returns:
      code: 0 when every comparison is exact and the goldens were written, 1
        when any comparison differs.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    with tempfile.TemporaryDirectory(prefix="capture-reference-") as scratch:
        captured = capture(flags.root, Path(scratch))
    if captured.report:
        print("\n".join(captured.report))
        print(f"{len(captured.report)} mismatches; nothing written.")
        return 1
    capture_reference.write_records(flags.out, {"reference.pt": captured.record})
    capture_reference.write_sites(flags.out, captured.sites)
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader splits go to.

    Returns:
      captured: The script's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    pristine = scripts["none"]
    record: dict[str, Tensor] = {}
    report: list[str] = []
    with vision_upstream.hidden_accelerators():
        for name in reference.COMPRESSORS:
            record |= upstream.upstream_compressor(scripts[name], name)
        for name in reference.SETTINGS:
            network, state = upstream.upstream_network(pristine, name)
            record |= network
            report += compare_full_size(name, state)
        lockstep = {
            name: upstream.upstream_lockstep(script, name)
            for name, script in scripts.items()
        }
        for values, _ in lockstep.values():
            record |= values
        record |= upstream.upstream_trace(pristine)
        record |= upstream.upstream_metrics(pristine)
        record |= upstream.upstream_transforms(pristine)
        batches = {
            name: upstream.upstream_batches(pristine, upstream.SETTINGS[name][1])
            for name in ("exp000", "exp004")
        }
        record |= vision_reference.decode_batches(
            batches["exp004"][0],
            config=reference.SETTINGS["exp004"]().dataset,
            source=vision_reference.synthetic_split()[0],
        )
        sites = upstream.upstream_sites(pristine, scripts["qsgd"])
    port = reference.port_record(scratch)
    report += mismatches(record, port)
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    for name, theirs in batches.items():
        report += [
            f"{name}/{line}"
            for line in capture_reference.compare_loaders(
                theirs,
                reference.loader_batches(name, scratch),
            )
        ]
    port_sites = reference.draw_sites(scratch)
    report += check_sites(sites, port_sites)
    return vision_reference.Captured(
        record=record,
        sites={"reference": sites, "port": port_sites},
        report=report,
    )


def compare_full_size(name: str, theirs: dict[str, Tensor]) -> list[str]:
    """Compare one setting's whole initial state with the script's.

    Args:
      name: A key of :data:`~priml.baselines.mlsbench.opt.gradient_compression.reference.SETTINGS`.
      theirs: The script's network state after its seeded build.

    Returns:
      report: One line per tensor that differs.

    """
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        ours = reference.full_step(reference.SETTINGS[name]).model.state_dict()
    return [f"{name}/{line}" for line in capture_reference.compare_states(theirs, ours)]


def check_sites(
    upstream_sites: dict[str, int],
    port_sites: dict[str, int],
) -> list[str]:
    """Report every required random call site that a run did not reach.

    Args:
      upstream_sites: The script's draw counts.
      port_sites: The port's draw counts.

    Returns:
      report: One line per required site a side never drew from.

    """
    return [
        f"{side}: random call site {site} was never reached"
        for side, counts, required in (
            ("reference", upstream_sites, REQUIRED_UPSTREAM),
            ("port", port_sites, REQUIRED_PORT),
        )
        for site in required
        if not any(key.startswith(site) and count for key, count in counts.items())
    ]


class _Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    out: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Clone of the benchmark at the pinned commit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="The testdata directory the goldens are written to.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
