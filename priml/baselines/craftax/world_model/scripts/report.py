#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Evaluate trained world-model checkpoints and write their report.

The checkpoints are runs of one experiment, the seeds of one recipe or a
single run; checkpoint NAME was trained with dataset.sampler_seed set to its
--sampler-seeds entry, which names its own validation windows. Each step
writes its JSON to OUTPUT:

1. baselines: each checkpoint against the trivial validation baselines on its
   own windows (baselines.py), baselines-NAME.json;
2. tiles: each checkpoint on --tiles fixed natural-mix validation tiles, the
   same decisions for every checkpoint (data_eval.py), data-eval-NAME.json;
3. matrix: each checkpoint's in-loop validation metric on the windows of every
   sampler seed of --window-seeds, which separates model-seed noise from
   window noise, evalonly-NAME-on-vV.json;
4. engine: the engine's correctness checks on each checkpoint
   (engine_checks.py beside this script), engine-checks-NAME.json;
5. dreams: open-loop dreams of each checkpoint beside real validation episodes
   (dream.py beside this script), dreams-NAME/.

Steps 1-3 score with the kernels and autocast the experiment trained under on
CUDA (scoring.py), every checkpoint's before any child runs; steps 4 and 5 run
their scripts in a child process, and a failed one is named and exits 1 with
everything else written, an engine check that crashed as a failed criterion.
With --wandb-runs, each checkpoint's training run is read from W&B, before
anything is scored, for the stability criteria, wandb-report.json. Every flag
is checked before OUTPUT is created. summary.json then holds the criteria
verdicts and the seed spreads of the primary metric and of each modality's NLL
three ways: each checkpoint on its own windows, every checkpoint on the same
windows, and every checkpoint on the same natural-mix tiles; report.md renders
it. --summarize rebuilds summary.json and report.md from an existing OUTPUT,
whose settings.json names the checkpoints and their seeds, so a host with W&B
access can add --wandb-runs to an evaluation run where it had none.

Examples:
  priml/baselines/craftax/world_model/scripts/report.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --skip dreams --output /opt/scratch/artifacts/craftax/world-model/stage0
  priml/baselines/craftax/world_model/scripts/report.py s0.pt s1.pt s2.pt --sampler-seeds 0,1,2 --wandb-runs RUN0,RUN1,RUN2 --output /opt/scratch/artifacts/craftax/world-model/stage0

'''
# fmt: on

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import dataclasses
import json
import math
import statistics
import subprocess
import sys
import time

import torch

from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
)
from priml.baselines.craftax.world_model.metric import (
    HEAD_BYTES,
    MODALITIES,
    SCALAR_BYTES,
    CraftaxBitsPerByte,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.baselines.craftax.world_model.scripts import (
    baselines,
    curves,
    data_eval,
)
from priml.lib.codec import from_plain, loads, to_plain
from priml.paths import validated_output_path


_THIS: Final = Path(__file__).resolve()

STEPS: Final = ("baselines", "tiles", "matrix", "engine", "dreams")
"""The evaluation steps, in the order each checkpoint runs them."""

PRIMARY: Final = "nats_per_decision_natural"
"""The design's primary metric, as the metric names it."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """One checkpoint to evaluate.

    Attributes:
      name: Its name in file names and the report.
      path: The ``TrainLoop`` checkpoint.
      sampler_seed: The ``dataset.sampler_seed`` its run trained with; its own
        validation windows are this seed's.

    """

    name: str
    path: Path
    sampler_seed: int


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the summary and report are written; 1 when an engine
        check or dream run failed, the summary and report written all the same.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    names, seeds, windows = _settings(parser, flags, output=output)
    # Read before anything is scored or written, so a wrong run fails first.
    wandb = _wandb_report(flags) if flags.wandb_runs else None
    failed: list[str] = []
    if not flags.summarize:
        output.mkdir(parents=True)
        settings = {**vars(flags), "names": names, "windows": windows}
        (output / "settings.json").write_text(json.dumps(settings, default=str) + "\n")
        checkpoints = [
            Checkpoint(name=name, path=path, sampler_seed=seed)
            for name, path, seed in zip(names, flags.checkpoints, seeds, strict=True)
        ]
        # Every in-process score is written before a child runs, so a failing
        # child costs only its own output.
        corpora = [
            _score(c, flags=flags, windows=windows, output=output) for c in checkpoints
        ]
        failed = [
            step
            for checkpoint, corpus in zip(checkpoints, corpora, strict=True)
            for step in _children(checkpoint, flags=flags, corpus=corpus, output=output)
        ]
    if wandb is not None:
        _write(output / "wandb-report.json", wandb)
    summary = summarize(output, names=names, own=seeds, windows=windows)
    _write(output / "summary.json", summary)
    (output / "report.md").write_text(render(summary, names=names))
    print(f"Wrote {output / 'summary.json'} and {output / 'report.md'}.")
    if failed:
        print(f"Failed: {', '.join(failed)}.")
    return int(bool(failed))


def validation_metric(
    model: WorldModel,
    config: WorldModelLoop.Config,
    *,
    device: torch.device,
    sampler_seed: int,
) -> dict[str, object]:
    """Score ``model`` as its loop's evaluation does, on one sampler seed's windows.

    The loop's ``eval`` over the experiment's own stream with
    ``dataset.sampler_seed`` replaced, as one process: the step's target scorer
    under inference mode and its autocast, and the ``val`` metric with the
    validation split's natural counts, zstd reference included. Under
    ``EvalSpans`` it scores every span, the decisions all of a run's ranks
    scored together, whatever the seed. Otherwise it scores rank 0's stratified
    micro-batches, which for a run trained on several ranks are a share of the
    ones its loop scored: each rank draws its own.

    Args:
      model: The trained model, on ``device``.
      config: Its experiment's finalized config.
      device: Where the model scores.
      sampler_seed: Whose validation windows to score.

    Returns:
      metrics: The metric's ``compute``, each key under ``val/`` as the run
        logs it.

    """
    stream = config.dataset.copy_tree()
    stream.sampler_seed = sampler_seed
    stream.device = device
    dataset = stream.make()
    metric = config.metrics_eval["val"].make()
    assert isinstance(metric, CraftaxBitsPerByte)
    metric.natural_decisions = dataset.eval_sampler.counts.double()
    target_nll = config.step.target_nll_fn.make()
    for batch in dataset.eval_dataloader():
        with torch.inference_mode(), autocast(config, device):
            nll = target_nll(model, batch["media"])
        metric.update(nll, **batch)
    return {f"val/{key}": value for key, value in metric.compute().items()}


def summarize(
    directory: Path,
    *,
    names: Sequence[str],
    own: Sequence[int],
    windows: Sequence[int],
) -> dict[str, object]:
    """Return the criteria verdicts and seed spreads of an evaluated directory.

    Reads whatever the steps wrote; a criterion no step measured is None.

    Args:
      directory: The report's directory.
      names: Each checkpoint's name.
      own: Each checkpoint's own sampler seed.
      windows: The sampler seeds of the eval-only matrix.

    Returns:
      summary: ``training``, ``timing``, and criteria 1-2 from
        ``wandb-report.json``; ``criterion_3_beats_baselines`` and
        ``criterion_4_engine_checks`` per checkpoint; ``sigma``, the spreads of
        the primary metric, of bits per byte, and of each modality's NLL in
        nats per target, on each checkpoint's own windows, on each window set,
        and on the common tiles; and ``dreams``, each checkpoint's dream
        distances and settings.

    """
    wandb = _read(directory / "wandb-report.json")
    runs = [_object(run) for run in _list(wandb["runs"])] if wandb is not None else []
    stable = [
        from_plain(r["loss_finite"], bool)
        and from_plain(r["grad_norm_finite"], bool)
        and from_plain(r["unrecovered_spikes"], int) == 0
        for r in runs
    ]
    trends = [_object(r["modalities"]) for r in runs]
    decreased = [
        set(trend) == set(MODALITIES)
        and all(from_plain(_object(v)["decreased"], bool) for v in trend.values())
        for trend in trends
    ]
    criterion3: dict[str, object] = {}
    criterion4: dict[str, object] = {}
    dreams: dict[str, object] = {}
    for name in names:
        baseline = _read(directory / f"baselines-{name}.json")
        keys = ("beats", "model_nll", "empirical_nll", "cell_accuracy")
        criterion3[name] = (
            {k: baseline[k] for k in keys} if baseline is not None else None
        )
        engine = _read(directory / f"engine-checks-{name}.json")
        criterion4[name] = engine["all_passed"] if engine is not None else None
        dream = _read(directory / f"dreams-{name}" / "report.json")
        dreams[name] = (
            {k: dream[k] for k in ("distance", "run")} if dream is not None else None
        )
    return {
        "training": {from_plain(r["run"], str): _training(r) for r in runs},
        "timing": {from_plain(r["run"], str): r.get("timing") for r in runs},
        "criterion_1_stable": all(stable) if runs else None,
        "criterion_2_every_modality_decreased": all(decreased) if runs else None,
        "criterion_3_beats_baselines": criterion3,
        "criterion_4_engine_checks": criterion4,
        "sigma": _sigma(directory, names=names, own=own, windows=windows),
        "dreams": dreams,
    }


def render(summary: Mapping[str, object], *, names: Sequence[str]) -> str:
    """Return ``summary`` as a markdown report.

    Args:
      summary: ``summarize``'s result.
      names: Each checkpoint's name, in order.

    Returns:
      text: The report.

    """
    lines = ["# World-model evaluation", "", "## Criteria", ""]
    lines += ["| Criterion | Verdict |", "|---|---|"]
    lines.append(f"| 1. Training stable | {_verdict(summary['criterion_1_stable'])} |")
    fell = _verdict(summary["criterion_2_every_modality_decreased"])
    lines.append(f"| 2. Every modality's validation NLL fell | {fell} |")
    criterion3 = _object(summary["criterion_3_beats_baselines"])
    criterion4 = _object(summary["criterion_4_engine_checks"])
    for name in names:
        entry = criterion3[name]
        beats = (
            all(from_plain(v, bool) for v in _object(_object(entry)["beats"]).values())
            if entry
            else None
        )
        lines.append(f"| 3. {name} beats the trivial baselines | {_verdict(beats)} |")
        engine = _verdict(criterion4[name])
        lines.append(f"| 4. {name} passes the engine checks | {engine} |")
    lines += ["", "## Baselines (nats per target; next-frame cells exactly right)", ""]
    lines += ["| Checkpoint | Target | Model | Baseline |", "|---|---|---|---|"]
    for name in names:
        if (entry := criterion3[name]) is None:
            continue
        model, empirical = (
            _object(_object(entry)[key]) for key in ("model_nll", "empirical_nll")
        )
        for target, nll in empirical.items():
            lines.append(
                f"| {name} | {target} | {_number(model[target])} | {_number(nll)} |",
            )
        cells = _object(_object(entry)["cell_accuracy"])
        lines.append(
            f"| {name} | cells (copy baseline) | {_number(cells['model'])} | "
            f"{_number(cells['copy'])} |",
        )
    lines += _training_lines(summary)
    lines += _sigma_lines(_object(summary["sigma"]))
    return "\n".join([*lines, *_dream_lines(_object(summary["dreams"]))]) + "\n"


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoints: list[Path]
    experiment: str
    override: list[str]
    names: str
    sampler_seeds: str
    window_seeds: str
    corpus: Path | None
    tiles: int
    tile_seed: int
    skip: list[str]
    dream_rows: int
    dream_decisions: int
    dream_seed: int
    wandb_runs: str
    wandb_project: str
    device: str
    summarize: bool
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="TrainLoop checkpoints (.pt), the runs of one experiment.",
    )
    parser.add_argument(
        "--experiment",
        default="priml.baselines.craftax.world_model.experiments.exp001",
        help="Dotted path of the experiment factory that trained the checkpoints.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Config override applied to every checkpoint's experiment; repeatable.",
    )
    parser.add_argument(
        "--names",
        default="",
        help="Comma-separated checkpoint names; default s0, s1, ... by position.",
    )
    parser.add_argument(
        "--sampler-seeds",
        default="",
        help=(
            "Each checkpoint's training dataset.sampler_seed, comma-separated; "
            "default 0 for each."
        ),
    )
    parser.add_argument(
        "--window-seeds",
        default="",
        help=(
            "Sampler seeds whose windows the matrix scores, comma-separated; "
            "default the checkpoints' own."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus whose validation shards the tiles cut; default the run's.",
    )
    parser.add_argument("--tiles", type=int, default=512, help="Tiles scored.")
    parser.add_argument("--tile-seed", type=int, default=0, help="Seed of the tiles.")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=STEPS,
        help="A step not to run; repeatable.",
    )
    parser.add_argument("--dream-rows", type=int, default=256, help="Dream rows.")
    parser.add_argument(
        "--dream-decisions",
        type=int,
        default=4_000,
        help="Decisions per dream row.",
    )
    parser.add_argument("--dream-seed", type=int, default=0, help="Dream seed.")
    parser.add_argument(
        "--wandb-runs",
        default="",
        help=(
            "Each checkpoint's W&B training run id, comma-separated, read for the "
            "stability criteria; off by default."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="craftax-world-model",
        help="W&B project of --wandb-runs, as entity/project or under the default entity.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help=(
            "Only rebuild summary.json and report.md from an existing OUTPUT, "
            "whose settings.json names the checkpoints and their seeds."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="New directory.")


# Every check here runs before anything is scored, so a bad flag leaves no OUTPUT behind
# and the same command runs once it is fixed.
def _settings(
    parser: argparse.ArgumentParser,
    flags: Flags,
    *,
    output: Path,
) -> tuple[list[str], list[int], list[int]]:
    """Return the checkpoints' names and own seeds and the matrix's window seeds."""
    if flags.summarize:
        if any(
            (flags.checkpoints, flags.names, flags.sampler_seeds, flags.window_seeds),
        ):
            parser.error("--summarize reads the checkpoints and seeds from OUTPUT.")
        settings = _object(loads((output / "settings.json").read_text()))
        names = from_plain(settings["names"], list[str])
        text = from_plain(settings["sampler_seeds"], str)
        seeds = _integers(text) or [0] * len(names)
        windows = from_plain(settings["windows"], list[int])
    else:
        count = len(flags.checkpoints)
        names = (
            flags.names.split(",") if flags.names else [f"s{i}" for i in range(count)]
        )
        seeds = _integers(flags.sampler_seeds) or [0] * count
        windows = _integers(flags.window_seeds) or sorted(set(seeds))
        if not count or len(names) != count or len(seeds) != count:
            parser.error("Give one or more checkpoints, a name and a seed for each.")
        if missing := [str(path) for path in flags.checkpoints if not path.is_file()]:
            parser.error(f"No checkpoint at {', '.join(missing)}.")
    runs = flags.wandb_runs.split(",") if flags.wandb_runs else names
    if len(runs) != len(names):
        parser.error(
            f"--wandb-runs names {len(runs)} runs for {len(names)} checkpoints.",
        )
    return names, seeds, windows


def _score(
    checkpoint: Checkpoint,
    *,
    flags: Flags,
    windows: Sequence[int],
    output: Path,
) -> Path:
    """Run one checkpoint's in-process steps not skipped; return the tiles' corpus."""
    name, device = checkpoint.name, torch.device(flags.device)
    overrides = [*flags.override, f"dataset.sampler_seed={checkpoint.sampler_seed}"]
    clock = time.monotonic()
    model, config = load_trained(
        flags.experiment,
        checkpoint.path,
        overrides=overrides,
        device=device,
    )
    corpus = flags.corpus or Path(config.dataset.corpus)
    provenance: dict[str, object] = {
        "checkpoint": str(checkpoint.path),
        "experiment": flags.experiment,
        "overrides": overrides,
    }
    if "baselines" not in flags.skip:
        result, run = baselines.evaluate(model, config, device=device)
        _write(output / f"baselines-{name}.json", {**result, "run": provenance | run})
        _log(name, "baselines", clock)
    if "tiles" not in flags.skip:
        start = time.monotonic()
        with autocast(config, device):
            result, available = data_eval.evaluate(
                model,
                corpus,
                t_g=config.dataset.t_g,
                s_max=config.dataset.s_max,
                count=flags.tiles,
                seed=flags.tile_seed,
                cached_decisions=config.dataset.cached_decisions,
            )
        run = {
            "corpus": str(corpus),
            "t_g": config.dataset.t_g,
            "seed": flags.tile_seed,
            "tiles_available": available,
            "seconds": time.monotonic() - start,
        }
        _write(output / f"data-eval-{name}.json", {**result, "run": provenance | run})
        _log(name, "tiles", clock)
    if "matrix" not in flags.skip:
        for seed in windows:
            metrics = validation_metric(model, config, device=device, sampler_seed=seed)
            _write(output / f"evalonly-{name}-on-v{seed}.json", metrics)
            _log(name, f"matrix v{seed}", clock)
    del model
    torch.cuda.empty_cache()
    return corpus


def _children(
    checkpoint: Checkpoint,
    *,
    flags: Flags,
    corpus: Path,
    output: Path,
) -> list[str]:
    """Run one checkpoint's engine checks and dreams not skipped; return those failed."""
    name, clock = checkpoint.name, time.monotonic()
    overrides = [*flags.override, f"dataset.sampler_seed={checkpoint.sampler_seed}"]
    options = ["--experiment", flags.experiment, "--device", flags.device]
    options += [f"--override={value}" for value in overrides]
    checks = output / f"engine-checks-{name}.json"
    steps: list[tuple[str, list[str]]] = []
    if "engine" not in flags.skip:
        steps.append(("engine_checks.py", [str(corpus), str(checks), *options]))
    if "dreams" not in flags.skip:
        dreams = [*options, "--corpus", str(corpus), "--rows", str(flags.dream_rows)]
        dreams += ["--decisions", str(flags.dream_decisions)]
        dreams += ["--seed", str(flags.dream_seed)]
        dreams += ["--output", str(output / f"dreams-{name}")]
        steps.append(("dream.py", dreams))
    failed: list[str] = []
    for script, arguments in steps:
        status = _child(
            script,
            [str(checkpoint.path), *arguments],
            name=name,
            clock=clock,
        )
        if status:
            failed.append(f"{name} {script}")
        # The checks write their report whatever they find, so a crashed run
        # wrote none, and its criterion would read as never measured.
        if status and script == "engine_checks.py" and not checks.exists():
            _write(checks, {"all_passed": False, "exit_status": status})
    return failed


def _child(script: str, arguments: Sequence[str], *, name: str, clock: float) -> int:
    """Run a sibling script in a child process; return its exit status."""
    command = [sys.executable, str(_THIS.with_name(script)), *arguments]
    status = subprocess.run(command, check=False).returncode  # noqa: S603 -- This package's own script, given this run's flags.
    _log(name, f"{script} (exit {status})", clock)
    return status


def _wandb_report(flags: Flags) -> dict[str, object]:
    """Return each W&B training run's stability report, under ``runs``."""
    rule = curves.SpikeRule()
    runs: list[object] = []
    for run in flags.wandb_runs.split(","):
        rows = curves.run_history(flags.wandb_project, run)
        runs.append({"run": run, **curves.training_report(rows, rule=rule)})
    return {
        "project": flags.wandb_project,
        "rule": to_plain(rule),
        "runs": runs,
    }


def _training(run: Mapping[str, object]) -> dict[str, object]:
    """Return the stability and trend entries of one run's W&B report."""
    trend = _object(run["modalities"])
    return {
        "last_logged_step": run["last_step"],
        "eval_steps": run["eval_steps"],
        "loss_finite": run["loss_finite"],
        "grad_norm_finite": run["grad_norm_finite"],
        "unrecovered_spikes": run["unrecovered_spikes"],
        "spikes": len(_list(run["spikes"])),
        "logged": run["logged"],
        "grad_norm_first_last_max": run["grad_norm_first_last_max"],
        "clip_fraction_final": run["clip_fraction_last"],
        "max_attention_logit_max": run["max_attention_logit_max"],
        "modality_decreased": {m: _object(v)["decreased"] for m, v in trend.items()},
        "modality_monotone": {m: _object(v)["monotone"] for m, v in trend.items()},
    }


def _sigma(
    directory: Path,
    *,
    names: Sequence[str],
    own: Sequence[int],
    windows: Sequence[int],
) -> dict[str, object]:
    """Return the seed spreads on own windows, on each window set, and on the tiles."""
    result: dict[str, object] = {}
    diagonal = _all_read(
        directory / f"evalonly-{n}-on-v{v}.json"
        for n, v in zip(names, own, strict=True)
    )
    if diagonal:
        result["own_windows"] = _spreads(diagonal, prefix="val/")
    sets: dict[str, object] = {}
    seed_sd: list[float] = []
    means: list[float] = []
    for v in windows:
        row = _all_read(directory / f"evalonly-{n}-on-v{v}.json" for n in names)
        if row:
            sets[f"v{v}"] = {
                **_spreads(row, prefix="val/"),
                "zstd19_bpb": row[0].get("val/zstd19_bpb"),
            }
            primary = [from_plain(m[f"val/{PRIMARY}"], float) for m in row]
            means.append(statistics.fmean(primary))
            seed_sd += [statistics.stdev(primary)] if len(primary) > 1 else []
    if sets:
        result["same_windows"] = sets
        result["same_windows_pooled_seed_sd"] = (
            statistics.fmean(sd**2 for sd in seed_sd) ** 0.5 if seed_sd else None
        )
        result["window_set_sd_of_means"] = (
            statistics.stdev(means) if len(means) > 1 else None
        )
    tiles = _all_read(directory / f"data-eval-{n}.json" for n in names)
    if tiles:
        metrics = [_object(t["metric"]) for t in tiles]
        result["common_natural_tiles"] = {
            "decisions": tiles[0]["decisions"],
            "nats_per_decision": _spread(
                [from_plain(m["nats_per_decision"], float) for m in metrics],
            ),
            "bpb": _spread([from_plain(m["bpb"], float) for m in metrics]),
            "nll": _nll_spreads(metrics, prefix=""),
        }
    return result


def _spreads(
    metrics: Sequence[Mapping[str, object]],
    *,
    prefix: str,
) -> dict[str, object]:
    """Return the spreads of the primary metric, bits per byte, and per-modality NLL."""
    return {
        PRIMARY: _spread([from_plain(m[f"{prefix}{PRIMARY}"], float) for m in metrics]),
        "bpb": _spread([from_plain(m[f"{prefix}bpb"], float) for m in metrics]),
        "nll": _nll_spreads(metrics, prefix=prefix),
    }


def _nll_spreads(
    metrics: Sequence[Mapping[str, object]],
    *,
    prefix: str,
) -> dict[str, object]:
    """Return each modality's NLL spread in nats per target, from its bits per byte."""
    schema = craftax_schema()
    width = {**HEAD_BYTES, "board": len(schema.cell_fields), "hud": SCALAR_BYTES}
    return {
        m: _spread(
            [
                from_plain(metric[f"{prefix}bpb/{m}"], float) * width[m] * math.log(2)
                for metric in metrics
            ],
        )
        for m in MODALITIES
    }


def _spread(values: Sequence[float]) -> dict[str, object]:
    """Return the values, their mean, standard deviation, and its share of the mean."""
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "values": list(values),
        "mean": mean,
        "sd": sd,
        "sd_over_mean": None if sd is None else sd / mean,
    }


def _training_lines(summary: Mapping[str, object]) -> list[str]:
    """Return the report's section of each W&B training run, empty without one."""
    runs = _object(summary["training"])
    if not runs:
        return []
    timing = _object(summary["timing"])
    lines = ["", "## Training runs (W&B)", ""]
    lines += [
        (
            "| Run | Last step | Loss finite | Unrecovered spikes | Modalities fell "
            "| Median update s | Evaluations s | Runtime s |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, entry in runs.items():
        stats, times = _object(entry), _object(timing.get(run) or {})
        fell = _object(stats["modality_decreased"])
        lines.append(
            f"| {run} | {stats['last_logged_step']} | {stats['loss_finite']} "
            f"| {stats['unrecovered_spikes']} | {sum(map(bool, fell.values()))}/{len(fell)} "
            f"| {_number(times.get('dt_median_seconds'))} "
            f"| {_number(times.get('eval_total_seconds'))} "
            f"| {_number(times.get('runtime_seconds'))} |",
        )
    return lines


def _sigma_lines(sigma: Mapping[str, object]) -> list[str]:
    """Return the report's seed-spread section."""
    lines = ["", "## Seed spread of nats per decision", ""]
    lines += ["| Scored on | Values | Mean | SD |", "|---|---|---|---|"]
    rows: list[tuple[str, object]] = []
    if "own_windows" in sigma:
        rows.append(
            ("own windows, natural mix", _object(sigma["own_windows"])[PRIMARY]),
        )
    for key, value in _object(sigma.get("same_windows", {})).items():
        rows.append((f"windows {key}, natural mix", _object(value)[PRIMARY]))
    if "common_natural_tiles" in sigma:
        tiles = _object(sigma["common_natural_tiles"])["nats_per_decision"]
        rows.append(("common tiles", tiles))
    for label, entry in rows:
        spread = _object(entry)
        values = ", ".join(_number(v) for v in _list(spread["values"]))
        mean, sd = _number(spread["mean"]), _number(spread["sd"])
        lines.append(f"| {label} | {values} | {mean} | {sd} |")
    for key in ("same_windows_pooled_seed_sd", "window_set_sd_of_means"):
        if key in sigma:
            lines += ["", f"{key.replace('_', ' ')}: {_number(sigma[key])}"]
    return lines + _modality_lines(sigma)


def _modality_lines(sigma: Mapping[str, object]) -> list[str]:
    """Return each modality's NLL, mean and SD over checkpoints, as each spread scores it."""
    columns = {
        "own windows": sigma.get("own_windows"),
        **{
            f"windows {key}": value
            for key, value in _object(sigma.get("same_windows", {})).items()
        },
        "common tiles": sigma.get("common_natural_tiles"),
    }
    measured = {label: _object(_object(c)["nll"]) for label, c in columns.items() if c}
    if not measured:
        return []
    lines = [
        "",
        "## NLL per target by modality (nats, mean +- SD over checkpoints)",
        "",
    ]
    lines += ["| Modality | " + " | ".join(measured) + " |"]
    lines += ["|---|" + "---|" * len(measured)]
    for modality in MODALITIES:
        cells = [_object(nll[modality]) for nll in measured.values()]
        shown = [f"{_number(c['mean'])} +- {_number(c['sd'])}" for c in cells]
        lines.append(f"| {modality} | " + " | ".join(shown) + " |")
    return lines


def _dream_lines(dreams: Mapping[str, object]) -> list[str]:
    """Return the report's section of dream distances from real episodes."""
    measured = {name: _object(entry) for name, entry in dreams.items() if entry}
    if not measured:
        return []
    lines = ["", "## Dreams: distance from real validation episodes", ""]
    lines += ["| Checkpoint | Statistic | Distance |", "|---|---|---|"]
    for name, entry in measured.items():
        lines += [
            f"| {name} | {statistic} | {_number(distance)} |"
            for statistic, distance in _object(entry["distance"]).items()
        ]
    return lines


def _all_read(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Return every file's JSON object, or nothing when one of them is absent."""
    documents = [_read(path) for path in paths]
    present = [d for d in documents if d is not None]
    return present if len(present) == len(documents) else []


def _integers(text: str) -> list[int]:
    """Return a comma-separated list of integers; empty for empty text."""
    return [int(part) for part in text.split(",")] if text else []


def _read(path: Path) -> dict[str, object] | None:
    """Return a step's JSON object, or None when the step did not write it."""
    if not path.exists():
        return None
    return _object(loads(path.read_text()))


def _object(value: object) -> dict[str, object]:
    """Return a JSON object this report reads, which every step writes as one."""
    return from_plain(value, dict[str, object])


def _list(value: object) -> list[object]:
    """Return a JSON array this report reads."""
    return from_plain(value, list[object])


def _write(path: Path, value: Mapping[str, object]) -> None:
    """Write a JSON object with one-space indents."""
    path.write_text(json.dumps(value, indent=1) + "\n")


def _log(name: str, step: str, clock: float) -> None:
    """Print a step's completion and the checkpoint's elapsed seconds."""
    print(f"{name}: {step} done at {time.monotonic() - clock:.0f} s.", flush=True)


def _verdict(value: object) -> str:
    """Return a criterion's verdict in words."""
    if value is None:
        return "not measured"
    return "pass" if value else "FAIL"


def _number(value: object) -> str:
    """Return a number with five significant digits, or n/a."""
    return "n/a" if value is None else f"{from_plain(value, float):.5g}"


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
