#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the security-machine-unlearning harness's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``security-machine-unlearning`` harness, its pristine
method and each baseline's edit of it from a clone of the benchmark, runs the
port on the same inputs under the same pins, and compares with exact equality:
the setting's launch flags against ``exp000``; the forget split; three updates
of every method on fixed batches, the network after each and every teacher;
the evaluation's accuracy, confidences and membership AUC on fixed inputs;
each setting's full network's initial state, draws and forward, and its
shallowest network's forward; and, per variant, the harness's whole
``main`` on a synthetic dataset with its four-worker loaders: the initial and
pretrained network, every unlearning step's batches, loss and network, the
evaluation, and where torch's generator stands afterwards. It counts the
random draws of the main process on both sides, and checks that Python's and
NumPy's generators are never drawn from. Any mismatch is printed and the exit
code is 1; otherwise the values the default test replays are written under
``--out``.

The harness is imported from the clone, not vendored. Nothing runs on a GPU.

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
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts.capture_reference import (
    compare_states,
    write_records,
)
from priml.baselines.mlsbench.tl.machine_unlearning import reference, upstream
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.tl.machine_unlearning.reference import (
        MainTrace,
    )


REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@data/dataloader.py",
    "random_@data/sampler.py",
)
"""Main-process random call sites of the harness a capture run must reach."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@machine_unlearning/data.py",
    "randperm@machine_unlearning/data.py",
    "randint@vision/data.py",
    "rand@vision/data.py",
)
"""Random call sites of the port a capture run must reach."""


def main() -> int:
    """Capture and compare; return the process exit code.

    Returns:
      code: 0 when every comparison is exact and the golden was written, 1 when
        any comparison differs.

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
    write_records(flags.out, {"reference.pt": captured.record})
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited methods and the synthetic dataset go to.

    Returns:
      captured: The harness's values, keyed as the port's are.

    """
    script = upstream.driver(root)
    modules = upstream.variants(root, scratch)
    with vision_upstream.hidden_accelerators():
        record = upstream.upstream_recipe(script, root) | upstream.upstream_split(
            script,
        )
        record |= upstream.upstream_evaluation(script)
        states: dict[str, dict[str, Tensor]] = {}
        report: list[str] = []
        for key in reference.SETTINGS:
            full, states[key] = upstream.upstream_full_size(script, key)
            record |= full
            report += compare_full_forward(script, key)
        for name, module in modules.items():
            values, theirs = upstream.upstream_method_lockstep(script, module, name)
            record |= values
            report += compare_lockstep(name, theirs, reference.method_lockstep(name)[1])
        sites: dict[str, dict[str, int]] = {}
        for name, module in modules.items():
            trace, untouched = upstream.upstream_main(script, module)
            if not untouched:
                report.append(f"main/{name}: Python's or NumPy's generator was drawn")
            record |= reference.main_record(name, trace)
            ours = reference.port_main(name, scratch)
            report += compare_main(name, trace, ours)
            sites[f"reference-{name}"] = trace.sites
            sites[f"port-{name}"] = ours.sites
            report += check_sites(name, trace.sites, ours.sites)
    port, our_states = reference.port_record(scratch)
    report += [
        f"full/{key}/state: {line}"
        for key, state in states.items()
        for line in compare_states(state, our_states[key])
    ]
    report += mismatches(record, port)
    return vision.Captured(record=record, sites=sites, report=report)


def compare_full_forward(script: upstream.Driver, key: str) -> list[str]:
    """Compare a setting's full network's draws and forwards; too slow to replay.

    Args:
      script: The harness.
      key: A key of :data:`reference.SETTINGS`.

    Returns:
      report: One line per differing logits, under ``full/<key>/``.

    """
    theirs, their_draws = upstream.upstream_full_network(script, key)
    ours, our_draws = reference.full_network(key)
    return mismatches(
        reference.full_size_values(theirs, their_draws, key=key),
        reference.full_size_values(ours, our_draws, key=key),
    )


def compare_lockstep(
    name: str,
    theirs: list[dict[str, Tensor]],
    ours: list[dict[str, Tensor]],
) -> list[str]:
    """Compare a method's student after each update and its teachers after the last.

    Args:
      name: The variant.
      theirs: The script's states.
      ours: The port's states.

    Returns:
      report: One line per differing count or tensor.

    """
    if len(theirs) != len(ours):
        return [f"methods/{name}: {len(ours)} states against {len(theirs)}"]
    return [
        f"methods/{name}/{index}: {line}"
        for index, (want, got) in enumerate(zip(theirs, ours, strict=True), start=1)
        for line in compare_states(want, got)
    ]


def compare_main(name: str, theirs: MainTrace, ours: MainTrace) -> list[str]:
    """Compare every network state and batch two end-to-end runs produced.

    Args:
      name: The variant.
      theirs: The harness's run.
      ours: The port's run.

    Returns:
      report: One line per differing tensor, step count or teacher set.

    """
    prefix = f"main/{name}"
    report = [
        f"{prefix}/initial: {line}"
        for line in compare_states(theirs.initial, ours.initial)
    ]
    report += [
        f"{prefix}/pretrained: {line}"
        for line in compare_states(theirs.pretrained, ours.pretrained)
    ]
    if len(theirs.steps) != len(ours.steps):
        return [
            *report,
            f"{prefix}: {len(ours.steps)} steps against {len(theirs.steps)}",
        ]
    for index, (want, got) in enumerate(zip(theirs.steps, ours.steps, strict=True)):
        batches = mismatches(
            dict(zip(("rx", "ry", "fx", "fy"), want.batches, strict=True)),
            dict(zip(("rx", "ry", "fx", "fy"), got.batches, strict=True)),
        )
        lines = batches + compare_states(want.state, got.state)
        report += [f"{prefix}/step{index}: {line}" for line in lines]
    if list(theirs.teachers) != list(ours.teachers):
        return [
            *report,
            f"{prefix}: teachers {list(ours.teachers)} against {list(theirs.teachers)}",
        ]
    for role, want_state in theirs.teachers.items():
        report += [
            f"{prefix}/{role}: {line}"
            for line in compare_states(want_state, ours.teachers[role])
        ]
    return report


def check_sites(
    name: str,
    upstream_sites: dict[str, int],
    port_sites: dict[str, int],
) -> list[str]:
    """Report every required random call site an end-to-end run did not reach.

    Args:
      name: The variant.
      upstream_sites: The harness's main-process draw counts.
      port_sites: The port's draw counts.

    Returns:
      report: One line per required site a side never drew from.

    """
    return [
        f"main/{name}: {side} random call site {site} was never reached"
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
        help="The testdata directory the golden is written to.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
