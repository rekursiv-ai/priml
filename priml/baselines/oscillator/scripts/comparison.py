"""Run the preliminary comparison and save its settings and scores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time

import torch

from priml.baselines.oscillator.experiments import exp000, exp001, exp002, exp_smoke


class _Flags(Protocol):
    directory: Path
    smoke: bool


def main(argv: list[str] | None = None) -> int:
    """Run fixed recipes, refusing to overwrite an earlier result."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args(argv))
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is needed to record the code revision.")
    flags.directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    records: list[dict[str, object]] = []
    recipes = (exp_smoke,) if flags.smoke else (exp000, exp001, exp002)
    seeds = (2021,) if flags.smoke else (2021, 2022, 2023)
    for matched in (True, False):
        for seed in seeds:
            for recipe in recipes:
                cfg = recipe()
                cfg.seed = seed
                if matched:
                    cfg.data.reference = cfg.data.cheap.copy_tree()
                case = "matched" if matched else "mismatched"
                label = f"{case}-{cfg.experiment_name}-{seed}"
                (flags.directory / f"{label}.txt").write_text(
                    cfg.pformat(hide_default_values=False) + "\n",
                )
                started = time.perf_counter()
                record = cfg.make().run()
                record["case"] = case
                record["elapsed_seconds"] = time.perf_counter() - started
                records.append(record)
                (flags.directory / f"{label}.json").write_text(
                    json.dumps(record, indent=2, allow_nan=False) + "\n",
                )
                print(f"Finished {label}", flush=True)
    sources = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    sources.extend(
        root / "priml/baselines/etth1" / name for name in ("model.py", "train_step.py")
    )
    report = {
        "status": "preliminary synthetic pilot; no agent or real-data evaluation",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "git_head": subprocess.check_output(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
            [git, "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "git_status": subprocess.check_output(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
            [git, "status", "--short"],
            cwd=root,
            text=True,
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes(),
            ).hexdigest()
            for path in sources
        },
        "runs": records,
    }
    (flags.directory / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(flags.directory / "comparison.json")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/opt/scratch/runs/oscillator/pilot"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Check the wiring with two tiny runs.",
    )
