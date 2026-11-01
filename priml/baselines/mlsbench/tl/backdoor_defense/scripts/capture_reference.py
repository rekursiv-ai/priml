#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Capture the security-backdoor-defense driver's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``security-backdoor-defense`` driver, once per
defense (the pristine one and each baseline's edit), from a clone of the
benchmark, runs the port on the same inputs under the same pins, and compares
with exact equality: the poisoned positions and Python's generator after them;
both triggers' pixels; every defense's scores on fixed features; ``main`` run
whole at each setting on a small split with training skipped -- which images
it poisons, when it evaluates, its rates and optimizer, and, for every
defense, the scores, the images kept and the recall; each setting's shallowest
network built, featurized and measured for clean accuracy and attack success;
each full network's initial state and forward; every pixel the poisoned and
analysis loaders produce; and every parameter after two updates. It counts the
torch draws on both sides. Any mismatch is printed and the exit code is 1;
otherwise the values the default test replays are written under ``--out``.

The driver is imported from the clone, not vendored. Nothing runs on a GPU.

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
from priml.baselines.mlsbench.tl.backdoor_defense import reference, upstream
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, rng_fingerprint


if TYPE_CHECKING:
    from torch import Tensor, nn


REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@data/dataloader.py",
    "randperm@data/sampler.py",
    "randint@transforms/transforms.py",
    "rand@transforms/transforms.py",
)
"""Random call sites of the driver's stack that a capture run must reach."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@vision/data.py",
    "randperm@vision/data.py",
    "randint@vision/data.py",
    "rand@vision/data.py",
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
    write_goldens(flags.out, captured)
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited defenses and the splits go to.

    Returns:
      captured: The driver's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    pristine = scripts["pristine"]
    record: dict[str, Tensor] = {}
    report: list[str] = []
    with vision_upstream.hidden_accelerators():
        filters = {
            name: upstream.trace_driver(script, reference.FIRST)
            for name, script in scripts.items()
        }
        record |= upstream.filter_record(filters)
        inputs = upstream.filter_inputs(filters["pristine"])
        for setting in reference.SETTINGS:
            trace = (
                filters["pristine"]
                if setting == reference.FIRST
                else upstream.trace_driver(pristine, setting)
            )
            record |= upstream.main_record(trace, setting)
            record |= upstream.tiny_record(pristine, setting)
            model, draws, network = upstream.setting_record(pristine, setting)
            record |= network
            report += compare_network(model, draws, setting)
            passes = upstream.poisoned_batches(pristine, setting)
            record |= reference.decode_record(setting, passes)
            report += [
                f"loader/{setting}/{line}"
                for line in capture_reference.compare_loaders(
                    passes,
                    reference.loader_batches(setting, scratch),
                )
            ]
        record |= upstream.poison_record(pristine)
        record |= upstream.trigger_record(pristine)
        record |= upstream.defense_record(scripts)
        lockstep = upstream.lockstep(pristine, filters["pristine"])
        record |= lockstep[0]
        sites = upstream.upstream_sites(pristine)
    port_sites = reference.draw_sites(scratch)
    report += mismatches(
        record,
        reference.port_record(scratch) | reference.filter_record(inputs),
    )
    report += capture_reference.compare_lockstep(
        {"pristine": lockstep},
        run=reference.lockstep_record,
    )
    report += check_sites(sites, port_sites)
    report += reference.poison_at_tied_cut(record | inputs)
    return vision_reference.Captured(
        record=record | inputs,
        sites={"reference": sites, "port": port_sites},
        report=report,
    )


def compare_network(model: nn.Module, draws: Tensor, setting: str) -> list[str]:
    """Compare a setting's full network, as the driver initialized it, with the port's.

    Args:
      model: The driver's network at the task's seed.
      draws: Where the generator stood after the driver's build.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      report: One line per tensor that differs, in registration order, then
        one if the generator stands elsewhere after the build.

    """
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        ours = vision_reference.full_step(reference.SETTINGS[setting].factory).model
        ours_draws = rng_fingerprint()
    return [
        f"settings/{setting}/state: {line}"
        for line in capture_reference.compare_states(
            model.state_dict(),
            ours.state_dict(),
        )
    ] + [
        f"settings/{setting}/full/{line}"
        for line in mismatches({"draws": draws}, {"draws": ours_draws})
    ]


def check_sites(
    upstream_sites: dict[str, int],
    port_sites: dict[str, int],
) -> list[str]:
    """Report every required random call site that a run did not reach.

    Args:
      upstream_sites: The driver's draw counts.
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


def write_goldens(out: Path, captured: vision_reference.Captured) -> None:
    """Write the driver's values under ``out``, within the size ceiling.

    The networks' records, whose features fill most of a file, are stored
    apart from the poisoning, defense and filter records.

    Args:
      out: The ``testdata`` directory.
      captured: The result of :func:`capture`.

    """
    record = captured.record
    networks = ("settings/", "tiny/")
    capture_reference.write_records(
        out,
        {
            "reference.pt": {
                k: v for k, v in record.items() if not k.startswith(networks)
            },
            "reference_networks.pt": {
                k: v for k, v in record.items() if k.startswith(networks)
            },
        },
    )
    capture_reference.write_sites(out, captured.sites)


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
