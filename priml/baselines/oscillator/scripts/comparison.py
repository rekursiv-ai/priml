"""Run the preliminary comparison and save its settings and scores."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
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
    # Each combination names its own output files; a repeat would overwrite one.
    for values in (
        flags.experiments,
        flags.seeds,
        flags.data_seed_offsets,
        flags.cases,
    ):
        if len(set(values)) != len(values):
            parser.error(
                "Experiments, seeds, data seed offsets, and cases must be unique.",
            )
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
    # The export drops the monorepo prefix from this path literal, so the depth to
    # the repository root follows each layout; a fixed ``parents[3]`` fell one short.
    root = _CWD.parents[len(Path("priml/baselines/oscillator/scripts").parts) - 1]
    records: list[ComparisonRecord] = []
    for offset, case, seed, recipe in product(
        flags.data_seed_offsets,
        flags.cases,
        flags.seeds,
        recipes,
    ):
        cfg = recipe.copy_tree()
        cfg.seed = seed
        for stream in (
            cfg.data.train,
            cfg.data.validation,
            cfg.data.synthetic,
            cfg.data.test,
            cfg.data.shifted,
        ):
            stream.seed += offset
        if case == "matched":
            cfg.data.reference = cfg.data.cheap.copy_tree()
        label = f"{case}-{cfg.experiment_name}-{seed}"
        if offset:
            label += f"-data{offset}"
        (directory / f"{label}.txt").write_text(
            cfg.pformat(hide_default_values=False) + "\n",
        )
        started = time.perf_counter()
        record: ComparisonRecord = {
            **cfg.make().run(),
            "case": case,
            "data_seed_offset": offset,
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
            "data_seed_offsets": flags.data_seed_offsets,
            "cases": flags.cases,
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
                for path in sorted({*sources, *_imported_sources(_CWD.parents[2])})
            },
            "runs": records,
        },
    )
    print(directory / "comparison.json")
    return 0


class ComparisonRecord(StudyRecord):
    """One study run, labelled by simulator case and timed."""

    case: str
    data_seed_offset: int
    elapsed_seconds: float


class _Flags(Protocol):
    directory: Path
    experiments: list[str]
    seeds: list[int]
    data_seed_offsets: list[int]
    cases: list[str]


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
        help=(
            "Recipes from priml.baselines.oscillator.experiments; "
            "exp_smoke checks wiring."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Model seeds applied to every recipe.",
    )
    parser.add_argument(
        "--data-seed-offsets",
        nargs="+",
        type=int,
        default=[0],
        help="Offsets added to every split seed; recipes share data within each offset.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["matched", "mismatched"],
        default=["matched", "mismatched"],
        help="Simulator controls to run; the default keeps both.",
    )


def _git(git: str, *args: str, cwd: Path) -> str:
    return subprocess.check_output(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
        [git, *args],
        cwd=cwd,
        text=True,
    ).strip()


def _imported_sources(package: Path) -> set[Path]:
    """Return the file of each imported module under ``package``."""
    # A string prefix test, not ``Path.is_relative_to``: thousands of modules
    # are loaded, and building a path for each one cost ~40 ms per run.
    prefix = f"{package}{os.sep}"
    specs = (module.__spec__ for module in tuple(sys.modules.values()))
    return {
        Path(spec.origin)
        for spec in specs
        if spec is not None
        and spec.origin is not None
        and spec.origin.startswith(prefix)
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(to_plain(value), indent=2) + "\n")
