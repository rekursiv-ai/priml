"""Stage pinned weights, freeze a protocol, and run its development or audit suite."""

from __future__ import annotations

from http.client import HTTPResponse
from importlib import metadata
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import tempfile
import urllib.request

from configgle.launch import resolve_config

import torch

from priml.baselines.oscillator import experiments
from priml.baselines.oscillator.pretrained import (
    BoltPipeline,
    Chronos2Pipeline,
    ModelFiles,
    PretrainedForecast,
)
from priml.baselines.oscillator.suite import BenchmarkSuite, suite_digest
from priml.paths import validated_output_path


_CWD: Final = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    """Run a declared protocol, keeping audit output and its lock immutable.

    Args:
      argv: Command arguments; None reads the process arguments.

    Returns:
      code: Zero for success; failed model runs make a completed audit nonzero.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args(argv))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = resolve_config(f"{experiments.__name__}.{flags.config}")
    if not isinstance(cfg, BenchmarkSuite.Config):
        parser.error("The selected experiment must build a BenchmarkSuite.")
    assets = model_files(cfg)
    if flags.model_directory is not None:
        for files in assets:
            files.directory = flags.model_directory / files.model_id.rsplit("/", 1)[1]
    if flags.action == "stage":
        for files in assets:
            stage_model(files)
        return 0
    suite = cfg.make()
    snapshot = protocol_snapshot(
        cfg,
        root=_CWD.parents[len(Path("priml/baselines/oscillator/scripts").parts) - 1],
    )
    if flags.action == "freeze":
        path = validated_output_path(flags.lock)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Frozen {cfg.version}: {path}")
        return 0
    if flags.phase == "audit" or flags.lock.exists():
        verify_lock(flags.lock, snapshot=snapshot)
    runtime = environment()
    records, diagnostics = suite.run(
        flags.directory,
        phase=flags.phase,
        resume=flags.resume,
        provenance={"protocol": snapshot, "environment": runtime},
    )
    summary = suite.summarize(records, phase=flags.phase)
    report = {
        "version": cfg.version,
        "phase": flags.phase,
        "protocol": snapshot,
        "environment": runtime,
        "diagnostics": diagnostics,
        "summary": summary,
        "runs": records,
    }
    (flags.directory / "protocol.txt").write_text(
        cfg.pformat(hide_default_values=False) + "\n",
    )
    (flags.directory / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(flags.directory / "results.json")
    return int(any(record["status"] != "complete" for record in records))


def model_files(config: BenchmarkSuite.Config) -> list[ModelFiles.Config]:
    """Collect each pinned pretrained asset once without loading its model."""
    assets: dict[tuple[str, str], ModelFiles.Config] = {}
    for recipe in config.recipes:
        method = recipe.study.method
        if isinstance(method, PretrainedForecast.Config):
            pipeline = method.pipeline
            if isinstance(pipeline, (BoltPipeline.Config, Chronos2Pipeline.Config)):
                files = pipeline.files
                assets[files.model_id, files.revision] = files
    return list(assets.values())


def protocol_snapshot(
    config: BenchmarkSuite.Config,
    *,
    root: Path,
) -> dict[str, object]:
    """Pin config, project sources, dependency lock, and staged weight manifests."""
    assets: dict[str, object] = {}
    for files in model_files(config):
        model = files.make()
        assets[files.model_id] = json.loads(
            (model.directory / "source.json").read_text(),
        )
    paths = [
        *root.joinpath("priml").rglob("*.py"),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "priml/baselines/oscillator/PROTOCOL.md",
        root / "priml/baselines/oscillator/pretrained-requirements.txt",
    ]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return {
        "version": config.version,
        "config_sha256": suite_digest(config),
        "source_sha256": sources,
        "assets": assets,
        "packages": environment()["packages"],
    }


def verify_lock(path: Path, *, snapshot: dict[str, object]) -> None:
    """Reject audit runs whose configuration, sources, or weights changed."""
    locked = cast(dict[str, object], json.loads(path.read_text()))
    if locked != snapshot:
        raise ValueError(
            "The protocol lock is stale: config, source, or model files changed.",
        )


def stage_model(config: ModelFiles.Config) -> None:
    """Download only the two model files at the declared immutable revision.

    Args:
      config: Model identity, revision, and new local staging directory.

    Raises:
      ValueError: An existing directory lacks a verified matching manifest.

    """
    directory = config.directory.expanduser().resolve()
    if directory.exists():
        config.make()
        print(f"Already verified {config.model_id}")
        return
    if len(config.revision) != 40 or any(
        character not in "0123456789abcdef" for character in config.revision
    ):
        raise ValueError("Model revisions must be full hexadecimal commit identities.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A failed download must not leave a final directory that looks staged.
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-",
        dir=directory.parent,
    ) as temporary:
        staged = Path(temporary)
        digests: dict[str, str] = {}
        for name in ("config.json", "model.safetensors"):
            url = f"https://huggingface.co/{config.model_id}/resolve/{config.revision}/{name}"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "priml-oscillator-benchmark"},
            )
            path = staged / name
            with (
                cast(
                    HTTPResponse,
                    urllib.request.urlopen(  # noqa: S310 -- Fixed HTTPS model host and scheme above.
                        request,
                        timeout=120,
                    ),
                ) as response,
                path.open("xb") as destination,
            ):
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
            with path.open("rb") as source:
                digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
        record = {
            "model_id": config.model_id,
            "revision": config.revision,
            "files": digests,
        }
        (staged / "source.json").write_text(json.dumps(record, indent=2) + "\n")
        staged.rename(directory)
    config.make()
    print(f"Staged {config.model_id} at {config.revision}")


def environment() -> dict[str, object]:
    """Record hardware, the Git revision, and relevant installed packages."""
    versions: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "scipy",
        "configgle",
        "chronos-forecasting",
        "transformers",
        "accelerate",
        "pandas",
        "python-dateutil",
        "six",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "packages": versions,
        "git_head": subprocess.check_output(  # noqa: S603 -- Fixed read-only Git arguments; no shell.
            [shutil.which("git") or "git", "rev-parse", "HEAD"],
            cwd=_CWD,
            text=True,
        ).strip(),
    }


class _Flags(Protocol):
    action: str
    config: str
    phase: Literal["development", "audit"]
    directory: Path
    model_directory: Path | None
    lock: Path
    resume: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action", choices=["stage", "freeze", "run"], default="run")
    parser.add_argument(
        "--config",
        default="exp040",
        help="Named BenchmarkSuite factory.",
    )
    parser.add_argument(
        "--phase",
        choices=["development", "audit"],
        default="development",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/opt/scratch/artifacts/oscillator/release-development"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("/opt/scratch/artifacts/oscillator/release-lock.json"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        help="Staged model root; changes paths, not model identities.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain completed attempts only when provenance matches.",
    )
