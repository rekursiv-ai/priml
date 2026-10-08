#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Compare seeded runs' data_eval.py reports, arm by arm, against a control.

Each GROUP is NAME=REPORT[,REPORT...], one data_eval.py report per seed of a
run scored on the same tiles; the first group is the control. Every headline
number of a report -- natural nats per decision, bits per byte overall and per
modality, floor, and event, nats per decision per modality, per arm, and per
class (timeout or ended episode, repeated or fresh frame), and "mix", the arms
reweighted to --mix -- gets each group's seed mean and standard deviation and
its difference from the control's mean.

Seed noise is the standard deviation pooled over every group with two or more
seeds. z is a difference over its standard error, noise * sqrt(1/n_control +
1/n_group); the table flags |z| > 2 with "*". Without a group of two seeds
there is no noise estimate and no z. A few seeds estimate the noise poorly:
two control seeds alone give one degree of freedom, and a metric on which
they happen to agree gets a large z from any difference.

Examples:
  priml/baselines/craftax/world_model/scripts/data_compare.py ctrl=base-s0.json,base-s1.json,base-s2.json treated=treated-s0.json,treated-s1.json --output /opt/scratch/artifacts/craftax/world-model/compare-50m.json

'''
# fmt: on

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import json
import statistics

from priml.baselines.craftax.world_model.metric import MODALITIES
from priml.lib.codec import from_plain, loads
from priml.paths import validated_output_path


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the table is printed.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output) if flags.output else None
    groups = {
        name: [
            headline(
                from_plain(loads(Path(p).read_text()), dict[str, object]),
                mix=flags.mix,
            )
            for p in paths.split(",")
        ]
        for name, paths in (group.split("=", 1) for group in flags.groups)
    }
    result = compare(groups)
    names = list(groups)
    print(f"{'metric':<16}" + "".join(f"{name:>30}" for name in names))
    for metric, rows in result.items():
        print(f"{metric:<16}" + "".join(_cell(rows[name]) for name in names))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=1) + "\n")
        print(f"Wrote {output}.")
    return 0


def headline(report: Mapping[str, object], *, mix: Sequence[float]) -> dict[str, float]:
    """Return the headline numbers of one data_eval.py report.

    Args:
      report: The report's JSON object.
      mix: Weight of each arm in ``mix``; arms the report lacks are left out.

    Returns:
      values: ``nats``, ``bpb`` and the metric's ``bpb/*``, ``nats/<modality>``,
        nats per decision of each class label (``arm0``, ``timeout``, ``fresh``,
        ...), and ``mix``.

    """
    metric = from_plain(report.get("metric"), dict[str, object])
    values = {
        "nats": from_plain(metric.get("nats_per_decision"), float),
        "bpb": from_plain(metric.get("bpb"), float),
    } | {k: from_plain(v, float) for k, v in metric.items() if k.startswith("bpb/")}
    count, modality = 0.0, dict.fromkeys(MODALITIES, 0.0)
    labels: dict[str, tuple[float, float]] = {}
    for name, value in from_plain(report.get("classes"), dict[str, object]).items():
        row = from_plain(value, dict[str, object])
        decisions = from_plain(row.get("decisions"), float)
        nats = decisions * from_plain(row.get("nats_per_decision"), float)
        count += decisions
        for m in MODALITIES:
            modality[m] += decisions * from_plain(row.get(m), float)
        # A class is arm/ending/frame, e.g. arm0/timeout/repeated; each part pools.
        for label in name.split("/"):
            seen, total = labels.get(label, (0.0, 0.0))
            labels[label] = (seen + decisions, total + nats)
    values |= {f"nats/{m}": nats / count for m, nats in modality.items()}
    values |= {label: nats / seen for label, (seen, nats) in sorted(labels.items())}
    arms = [(w, values[f"arm{a}"]) for a, w in enumerate(mix) if f"arm{a}" in values]
    values["mix"] = sum(w * v for w, v in arms) / sum(w for w, _ in arms)
    return values


def compare(
    groups: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return each metric's per-group statistics against the first group.

    Args:
      groups: Each group's headline values, one mapping per seed; the first
        group is the control.

    Returns:
      rows: Per metric every run holds, per group: ``n``, ``mean``, ``sd``
        (two or more seeds), and against the control ``delta`` and ``z``
        (when some group has two or more seeds).

    """
    control, *_ = groups
    runs = [run for seeds in groups.values() for run in seeds]
    result: dict[str, dict[str, dict[str, float]]] = {}
    for metric in (k for k in runs[0] if all(k in run for run in runs)):
        values = {
            name: [run[metric] for run in seeds] for name, seeds in groups.items()
        }
        means = {name: statistics.fmean(v) for name, v in values.items()}
        degrees = sum(len(v) - 1 for v in values.values())
        squares = sum((x - means[n]) ** 2 for n, v in values.items() for x in v)
        rows: dict[str, dict[str, float]] = {}
        for name, v in values.items():
            row = {"n": float(len(v)), "mean": means[name]}
            if len(v) > 1:
                row["sd"] = statistics.stdev(v)
            if name != control:
                row["delta"] = means[name] - means[control]
            if name != control and degrees:
                error = (
                    squares / degrees * (1 / len(v) + 1 / len(values[control]))
                ) ** 0.5
                row["z"] = row["delta"] / error if error else 0.0
            rows[name] = row
        result[metric] = rows
    return result


class Flags(Protocol):
    """Parsed command-line flags."""

    groups: list[str]
    mix: tuple[float, ...]
    output: Path | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "groups",
        nargs="+",
        metavar="NAME=REPORT[,REPORT...]",
        help="A group's reports, one per seed; the first group is the control.",
    )
    parser.add_argument(
        "--mix",
        type=lambda text: tuple(float(w) for w in text.split(",")),
        default=(70.0, 10.0, 10.0, 10.0),
        help="Arm weights of the mix row, comma-separated; default 70,10,10,10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison JSON to write.",
    )


def _cell(row: Mapping[str, float]) -> str:
    """Return one group's table cell: mean, then sd or relative delta and z."""
    mean = f"{row['mean']:.5g}"
    if "delta" not in row:
        spread = f" sd {row['sd']:.2g}" if "sd" in row else ""
        return f"{mean + spread:>30}"
    base = row["mean"] - row["delta"]
    relative = f" {row['delta'] / base:+.1%}" if base else ""
    z = row.get("z")
    score = "" if z is None else f" z{z:+.1f}{'*' if abs(z) > 2 else ' '}"
    return f"{mean + relative + score:>30}"


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
