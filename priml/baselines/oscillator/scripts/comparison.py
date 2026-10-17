"""Run the preliminary comparison and save its settings and scores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time

from configgle.launch import resolve_config
from treekle import to_plain

from priml.baselines.oscillator import experiments
from priml.baselines.oscillator.benchmark import ForecastStudy, StudyRecord
from priml.paths import validated_output_path


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


_CWD: Final = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    """Run fixed recipes, refusing to overwrite an earlier result.

    Args:
      argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
      code: 0 on success.

    Raises:
      RuntimeError: Git is not installed.
      TypeError: A named experiment does not build an oscillator study.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args(argv))
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is needed to record the code revision.")
    directory = validated_output_path(flags.directory)
    recipes: list[ForecastStudy.Config] = []
    for name in flags.experiments:
        recipe = resolve_config(f"{experiments.__name__}.{name}")
        if not isinstance(recipe, ForecastStudy.Config):
            raise TypeError(f"{name} does not build an oscillator study.")
        recipes.append(recipe)
    directory.mkdir(mode=0o755, parents=True, exist_ok=False)
    root = _CWD.parents[3]
    records: list[ComparisonRecord] = []
    for matched in (True, False):
        for seed in flags.seeds:
            for recipe in recipes:
                cfg = recipe.copy_tree()
                cfg.seed = seed
                if matched:
                    cfg.data.reference = cfg.data.cheap.copy_tree()
                case = "matched" if matched else "mismatched"
                label = f"{case}-{cfg.experiment_name}-{seed}"
                (directory / f"{label}.txt").write_text(
                    cfg.pformat(hide_default_values=False) + "\n",
                )
                started = time.perf_counter()
                record: ComparisonRecord = {
                    **cfg.make().run(),
                    "case": case,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                records.append(record)
                _write_json(directory / f"{label}.json", record)
                print(f"Finished {label}", flush=True)
    sources = sorted(_CWD.parent.rglob("*.py"))
    sources.extend(
        root / "priml/baselines/etth1" / name for name in ("model.py", "train_step.py")
    )
    _write_json(
        directory / "comparison.json",
        {
            "status": "preliminary synthetic pilot; no agent or real-data evaluation",
            "experiments": flags.experiments,
            "seeds": flags.seeds,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "git_head": _git(git, "rev-parse", "HEAD", cwd=root),
            "git_status": _git(git, "status", "--short", cwd=root),
            "source_sha256": {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes(),
                ).hexdigest()
                for path in sources
            },
            "runs": records,
        },
    )
    print(directory / "comparison.json")
    return 0


class ComparisonRecord(StudyRecord):
    """One study run, labelled by simulator case and timed."""

    case: str
    elapsed_seconds: float


class _Flags(Protocol):
    directory: Path
    experiments: list[str]
    seeds: list[int]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/opt/scratch/runs/oscillator/pilot"),
        help="New run directory; an existing one is never overwritten.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["exp000", "exp001", "exp002"],
        help="Recipes from priml.baselines.oscillator.experiments; exp_smoke checks wiring.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Model seeds applied to every recipe.",
    )


def _git(git: str, *args: str, cwd: Path) -> str:
    return subprocess.check_output(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
        [git, *args],
        cwd=cwd,
        text=True,
    ).strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(to_plain(value), indent=2, allow_nan=False) + "\n")
