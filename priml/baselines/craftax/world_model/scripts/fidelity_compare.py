#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Compare fidelity.py reports of checkpoints scored on the same data with a base.

Every report must hold the same settings, spans, and continuation windows,
which fidelity.py fixes by corpus and seed. Each OTHER report is compared with
BASE, OTHER minus BASE, with a 95% bootstrap interval:

- teacher_forced: nats per decision, overall and per modality, paired over
  spans and pooled over their target decisions; ``lower`` counts the spans on
  which OTHER scores lower.
- continuations: each divergence metric at each step, the model's
  continuations paired over the windows whose real continuation holds the step.
- dreams: every count per 1,000 decisions. Dreams start in new worlds, so rows
  do not pair; each report's rows are resampled on their own.
- model_minus_frozen: each report's model against its own frozen frame, paired
  over windows.

With two or more OTHER reports, the seeds of one recipe, ``mean`` compares
their mean with BASE the same way. The intervals resample spans, windows, and
rows, not training runs. ``seeds`` gives data_compare.py's z of every summary
number: the difference over the seed noise pooled from the OTHER reports, with
one degree of freedom fewer than their count, as if BASE's recipe had the same
seed noise.

Examples:
  priml/baselines/craftax/world_model/scripts/fidelity_compare.py pilot.json s0.json s1.json s2.json --output /opt/scratch/artifacts/craftax/world-model/fidelity/compare/stage0-vs-pilot.json

'''
# fmt: on

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import functools
import json
import math

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.scripts import data_compare
from priml.baselines.craftax.world_model.scripts.dream_eval import (
    compare_ratio,
    ratio_interval,
)
from priml.lib.codec import PlainTree, from_plain, loads
from priml.paths import validated_output_path


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the comparison is printed, and written with ``--output``.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output) if flags.output else None
    if output and output.exists():
        raise FileExistsError(f"Output {output} exists; choose a new file.")
    base, *others = (
        from_plain(loads(Path(path).read_text()), dict[str, object])
        for path in flags.reports
    )
    result = compare(base, others, resamples=flags.resamples, seed=flags.seed)
    print("\n".join(_lines(result)))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=1) + "\n")
    return 0


def compare(
    base: Mapping[str, object],
    others: Sequence[Mapping[str, object]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, PlainTree]:
    """Compare fidelity reports scored on the same data with a base report.

    Args:
      base: The report every other one is compared with.
      others: Reports of other checkpoints, keyed by their provenance tags.
      resamples: Bootstrap resamples per interval.
      seed: Seed of the resamples.

    Returns:
      comparison: ``teacher_forced``, ``continuations``, and ``dreams``, each
        metric's difference per other tag, and ``mean`` with two or more;
        ``model_minus_frozen`` per report tag; and ``seeds``, data_compare's
        row of the others per summary number.

    Raises:
      ValueError: If a report holds other settings, spans, or windows.

    """
    reads: tuple[tuple[str, Callable[[Mapping[str, object]], object]], ...] = (
        ("settings", _settings),
        ("spans", _span_keys),
        ("windows", _windows),
    )
    for other in others:
        for what, reader in reads:
            if reader(other) != reader(base):
                raise ValueError(
                    f"{_tag(other)} was scored on other {what} than {_tag(base)}.",
                )
    generator = torch.Generator().manual_seed(seed)
    paired = functools.partial(_paired, resamples=resamples, generator=generator)
    _, targets, spans = _spans(base)
    other_spans = {_tag(o): _spans(o)[2] for o in others}
    model = {_tag(o): _per_pair(o, "model") for o in others}
    seeds = data_compare.compare(
        {"base": [_numbers(base)], "others": [_numbers(o) for o in others]},
    )
    return {
        "base": _tag(base),
        "others": [_tag(other) for other in others],
        "teacher_forced": {
            name: paired(
                values,
                {tag: s[name] for tag, s in other_spans.items()},
                targets,
            )
            for name, values in spans.items()
        },
        "continuations": {
            metric: {
                k: paired(
                    values[~values.isnan()],
                    {tag: m[metric][k][~values.isnan()] for tag, m in model.items()},
                    torch.ones(int((~values.isnan()).sum())),
                )
                for k, values in steps.items()
            }
            for metric, steps in _per_pair(base, "model").items()
        },
        "dreams": _dreams(base, others, resamples=resamples, generator=generator),
        "model_minus_frozen": {
            _tag(r): _against_frozen(r, paired=paired) for r in (base, *others)
        },
        "seeds": {metric: groups["others"] for metric, groups in seeds.items()},
    }


class Flags(Protocol):
    """Parsed command-line flags."""

    reports: list[Path]
    output: Path | None
    resamples: int
    seed: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        metavar="REPORT",
        help="fidelity.py reports: BASE, then each report compared with it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New comparison JSON to write.",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=4_000,
        help="Bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the resamples.")


def _tag(report: Mapping[str, object]) -> str:
    """Return a report's provenance tag."""
    return str(from_plain(report["provenance"], dict[str, object])["tag"])


def _settings(report: Mapping[str, object]) -> object:
    """Return a report's settings."""
    return report["settings"]


def _span_keys(report: Mapping[str, object]) -> object:
    """Return each span's episode, first target, and target count."""
    return _spans(report)[0]


def _spans(
    report: Mapping[str, object],
) -> tuple[list[tuple[str, int, int]], Tensor, dict[str, Tensor]]:
    """Return each span's key and targets, and its summed nats overall and per modality."""
    rows = from_plain(
        from_plain(report["teacher_forced"], dict[str, object])["per_span"],
        list[dict[str, object]],
    )
    keys = [
        (str(r["episode"]), from_plain(r["start"], int), from_plain(r["targets"], int))
        for r in rows
    ]
    targets = torch.tensor([key[2] for key in keys], dtype=torch.float64)
    nats = {
        name: [from_plain(value, float) for value in values]
        for name, values in [
            ("total", [r["nats_per_decision"] for r in rows]),
            *(
                (
                    name,
                    [
                        from_plain(r["modalities"], dict[str, object])[name]
                        for r in rows
                    ],
                )
                for name in from_plain(rows[0]["modalities"], dict[str, object])
            ),
        ]
    }
    return (
        keys,
        targets,
        {
            name: torch.tensor(values, dtype=torch.float64) * targets
            for name, values in nats.items()
        },
    )


def _per_pair(report: Mapping[str, object], side: str) -> dict[str, dict[str, Tensor]]:
    """Return each window's value per metric and step, NaN where it is not held."""
    continuations = from_plain(report["continuations"], dict[str, object])
    per_pair = from_plain(
        from_plain(continuations[side], dict[str, object])["per_pair"],
        dict[str, object],
    )
    return {
        metric: {
            k: torch.tensor(
                [math.nan if v is None else from_plain(v, float) for v in values],
                dtype=torch.float64,
            )
            for k, values in (
                (k, from_plain(values, list[object]))
                for k, values in from_plain(steps, dict[str, object]).items()
            )
        }
        for metric, steps in per_pair.items()
    }


def _windows(report: Mapping[str, object]) -> tuple[list[str], list[list[bool]]]:
    """Return the window names and which windows each side holds at each step."""
    continuations = from_plain(report["continuations"], dict[str, object])
    windows = from_plain(continuations["windows"], list[dict[str, object]])
    held = [
        [value is not None for value in from_plain(values, list[object])]
        for side in ("model", "frozen")
        for steps in from_plain(
            from_plain(continuations[side], dict[str, object])["per_pair"],
            dict[str, object],
        ).values()
        for values in from_plain(steps, dict[str, object]).values()
    ]
    return [str(w["name"]) for w in windows], held


def _paired(
    base: Tensor,
    others: Mapping[str, Tensor],
    den: Tensor,
    *,
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Return each other's pooled paired difference from ``base`` over units."""
    differences = {tag: other - base for tag, other in others.items()}
    if len(differences) > 1:
        differences["mean"] = torch.stack(list(differences.values())).mean(0)
    return {
        tag: {
            **ratio_interval(d, den, resamples=resamples, generator=generator)[0],
            "lower": int((d < 0).sum()),
        }
        for tag, d in differences.items()
    }


def _dreams(
    base: Mapping[str, object],
    others: Sequence[Mapping[str, object]],
    *,
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Return each dream count's difference per 1,000 decisions, rows unpaired."""
    rows = {_tag(r): _dream_rows(r) for r in (base, *others)}
    decisions = from_plain(
        from_plain(base["dreams"], dict[str, object])["decisions"],
        float,
    )
    result: dict[str, PlainTree] = {}
    for name, counted in rows.pop(_tag(base)).items():
        samples = {tag: values[name] for tag, values in rows.items()}
        if len(samples) > 1:
            samples["mean"] = torch.cat(list(samples.values()))
        entries: dict[str, PlainTree] = {}
        for tag, sample in samples.items():
            entry = compare_ratio(
                counted,
                torch.full_like(counted, decisions),
                sample,
                torch.full_like(sample, decisions),
                resamples=resamples,
                generator=generator,
            )
            diff = from_plain(entry["diff"], dict[str, object])
            entries[tag] = {
                "base": _thousand(
                    from_plain(entry["real"], dict[str, object])["value"],
                ),
                "other": _thousand(
                    from_plain(entry["dream"], dict[str, object])["value"],
                ),
                **{key: _thousand(diff[key]) for key in ("value", "low", "high")},
                "p_value": entry["p_value"],
            }
        result[name] = entries
    return result


def _dream_rows(report: Mapping[str, object]) -> dict[str, Tensor]:
    """Return each dream row's counts per name."""
    per_row = from_plain(
        from_plain(report["dreams"], dict[str, object])["per_row"],
        dict[str, object],
    )
    return {
        name: torch.tensor(from_plain(values, list[float]), dtype=torch.float64)
        for name, values in per_row.items()
    }


def _against_frozen(
    report: Mapping[str, object],
    *,
    paired: Callable[..., dict[str, PlainTree]],
) -> dict[str, PlainTree]:
    """Return the model's paired difference from the frozen frame per metric and step."""
    model, still = _per_pair(report, "model"), _per_pair(report, "frozen")
    return {
        metric: {
            k: paired(
                values[~values.isnan()],
                {"model": model[metric][k][~values.isnan()]},
                torch.ones(int((~values.isnan()).sum())),
            )["model"]
            for k, values in steps.items()
        }
        for metric, steps in still.items()
    }


def _numbers(report: Mapping[str, object]) -> dict[str, float]:
    """Return the summary's numbers, leaving out those it could not measure."""
    return {
        name: from_plain(value, float)
        for name, value in from_plain(report["summary"], dict[str, object]).items()
        if value is not None
    }


def _thousand(value: object) -> float:
    """Return a per-decision rate per 1,000 decisions."""
    return 1_000 * from_plain(value, float)


def _lines(tree: object, path: str = "") -> list[str]:
    """Return one line per comparison entry, under its slash-joined path."""
    if not isinstance(tree, dict):
        return []
    node = from_plain(cast(dict[str, object], tree), dict[str, object])
    if "delta" in node:
        z = node.get("z")
        score = "" if z is None else f" z {from_plain(z, float):+.2f}"
        return [f"{path:<64} delta {from_plain(node['delta'], float):+.4g}{score}"]
    if "value" in node:
        parts = [_number(node.get(key)) for key in ("value", "low", "high")]
        line = f"{path:<64} {parts[0]} [{parts[1]}, {parts[2]}]"
        if "lower" in node:
            line += f" lower {node['lower']}/{node['n']}"
        return [line]
    return [
        line
        for key, value in node.items()
        for line in _lines(value, f"{path}/{key}" if path else key)
    ]


def _number(value: object) -> str:
    """Return a number with four significant digits, or n/a."""
    return "n/a" if value is None else f"{from_plain(value, float):+.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
