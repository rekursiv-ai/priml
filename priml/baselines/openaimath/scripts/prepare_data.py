#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prepare pinned inputs for the openai-math baseline.

The policy checkpoint is the same Qwen3.5-4B release the tmax baseline uses, so
it is shared rather than downloaded twice. The tasks come from the pinned
``FineEnvs/openai-math`` revision, whose own generator also fixes the Daytona
snapshot name that the runner passes to Harbor.

Training draws lemma tasks, not those challenges: the script checks out
openai/math's Lean library at the commit the dataset copied its challenges from
(about 2.1 GB and a minute, beside the dataset) and cuts it into tasks in the dataset's own
format under ``lemmas/`` (see ``lemma_tasks.py``; about 1.3 GB). Their counts
are pinned, so a changed cutter is caught.

The script writes a manifest with file hashes and source revisions. On later
runs, it checks the recorded files and refuses missing, changed, or extra
files. Without a manifest, it refuses to reuse existing assets.

Building the snapshot needs Docker and a Daytona API key, so the script prints
the generator command instead of running it. The name it prints must match the
manifest's, which is how a changed image recipe is caught.

Examples:
  prepare_data.py
  prepare_data.py --base-dir /opt/scratch --dataset-dir /data/openai-math

'''
# fmt: on

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tomllib

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    snapshot_download,  # ty: ignore[unresolved-import] -- The local stub covers this API for basedpyright.
)

from priml.baselines.openaimath.experiments import exp000
from priml.baselines.openaimath.live import LiveOpenMathRolloutData
from priml.baselines.openaimath.scripts import lemma_tasks
from priml.baselines.openaimath.scripts.runner import SNAPSHOT_NAME
from priml.baselines.tmax.scripts.prepare_data import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
)


logger = logging.getLogger(__name__)


DATASET_REPO: Final = "FineEnvs/openai-math"
DATASET_REVISION: Final = "d7056fd31c19fd1a0aed0a0bc3e99026d412b2ca"
SOURCE_COMMIT: Final = "adc7f1241b42e322a6451854ab7e4b4c146bf78a"
"""The openai/math commit the pinned revision copied its challenges from."""
EXPECTED_TASKS: Final = 369
SOURCE_REPO: Final = "https://github.com/openai/math.git"
SOURCE_PATHS: Final = ("/lean/OAI/", "/lean/ComparatorChallenges/")
"""The parts of openai/math the lemma tasks are cut from: the library and the
challenge configs, whose reference modules decide the family split."""
SOURCE_DIRNAME: Final = "openai-math-source"
"""The source checkout's default directory, beside the dataset's."""
LEMMAS_DIRNAME: Final = "lemmas"
"""Where the lemma tasks are written inside the dataset directory."""
EXPECTED_TRAIN_LEMMAS: Final = 20_652
EXPECTED_TEST_LEMMAS: Final = 5_341
"""Lemma counts the pinned source yields; a change means the cutter changed."""
MANIFEST_NAME: Final = "exp000_assets_manifest.json"
"""Manifest stored beside the staged dataset."""
CACHE_DIRNAME: Final = ".cache"
"""Hugging Face transport metadata, which is not part of the pinned assets."""


class _Flags(Protocol):
    """The parsed command line."""

    base_dir: Path
    checkpoint_dir: Path | None
    dataset_dir: Path | None
    source_dir: Path | None
    manifest_path: Path | None


def default_directories(base_dir: Path | str = "/opt/scratch") -> tuple[Path, Path]:
    """Return where exp000 reads the checkpoint and the pinned dataset.

    Both come from the finalized experiment, so the staged assets land exactly
    where the learner and the rollout runner look for them.

    Raises:
      TypeError: exp000 no longer collects live rollouts.

    """
    config = exp000()
    config.base_dir = base_dir
    config = config.finalize()
    if not isinstance(config.dataset, LiveOpenMathRolloutData.Config):
        raise TypeError("exp000 must collect live openai/math rollouts.")
    # tasks_dir is <dataset>/lemmas/train.
    return (
        Path(cast("Path | str", config.step.model_path)),
        Path(config.dataset.runner.tasks_dir).parents[1],
    )


def main() -> int:
    """Stage the openai-math inputs and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/scratch"),
        help="Root for every default asset directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Where the policy checkpoint lives (default: exp000's model_path).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Where the pinned task dataset is staged "
            "(default: the parent of exp000's runner tasks_dir)."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Where openai/math's Lean library is checked out at the pinned commit"
            " (default: beside dataset-dir, as " + SOURCE_DIRNAME + ")."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Where to write the deterministic asset manifest "
            "(default: dataset-dir/" + MANIFEST_NAME + ")."
        ),
    )
    flags = cast(_Flags, parser.parse_args())
    checkpoint_default, dataset_default = default_directories(flags.base_dir)
    flags.checkpoint_dir = flags.checkpoint_dir or checkpoint_default
    flags.dataset_dir = flags.dataset_dir or dataset_default
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest_target = flags.manifest_path or (flags.dataset_dir / MANIFEST_NAME)
    _verify_staged_assets(manifest_target, flags.checkpoint_dir, flags.dataset_dir)
    _stage_checkpoint(flags.checkpoint_dir)
    _stage_dataset(flags.dataset_dir)
    tasks = _verify_tasks(flags.dataset_dir)
    source_dir = flags.source_dir or flags.dataset_dir.parent / SOURCE_DIRNAME
    _stage_source(source_dir)
    _stage_lemmas(flags.dataset_dir, source_dir)
    snapshot = _generator_snapshot_name(flags.dataset_dir)
    if snapshot != SNAPSHOT_NAME:
        raise ValueError(
            f"The staged generator builds snapshot {snapshot}, but this "
            f"baseline pins {SNAPSHOT_NAME}. Re-pin the revision, the snapshot"
            " name, and the runner's snapshot field together.",
        )
    manifest = _write_manifest(
        flags.checkpoint_dir,
        flags.dataset_dir,
        tasks=tasks,
        snapshot=snapshot,
        path=flags.manifest_path,
    )
    logger.info("tasks: %d, every one validated by the dataset's oracle", tasks)
    logger.info(
        "lemma tasks: %d train, %d held out (not oracle-validated; run"
        " eval.py --oracle on a sample)",
        EXPECTED_TRAIN_LEMMAS,
        EXPECTED_TEST_LEMMAS,
    )
    logger.info("manifest: %s", manifest)
    _log_next_steps(flags.base_dir, flags.dataset_dir, snapshot)
    return 0


def _log_next_steps(base_dir: Path, dataset_dir: Path, snapshot: str) -> None:
    """Print the commands a fresh machine still needs."""
    logger.info(
        "snapshot: build it once per image recipe, then pass the name to"
        " Harbor:\n    cd %s && uv --quiet run --frozen --no-project"
        " --with daytona-sdk python -m openai_math_harbor.snapshot\n"
        "    (or %s, which only prints the name)",
        dataset_dir / "generator",
        "uv --quiet run --frozen --no-project python -m "
        "openai_math_harbor.snapshot --print-name",
    )
    if not dataset_dir.is_relative_to(base_dir):
        logger.info(
            "grade with scripts/eval.py; it resolves --tasks-dir and"
            " --task-list under --base-dir, and %s is outside %s",
            dataset_dir,
            base_dir,
        )
        return
    logical = Path("/") / dataset_dir.relative_to(base_dir)
    eval_command = (
        "uv --quiet run --frozen -- python -m"
        f" priml.baselines.openaimath.scripts.eval --base-dir {base_dir}"
        f" --snapshot {snapshot}"
    )
    logger.info(
        "check the lemma cutting, then grade a run (progress metric, then the"
        " protected score):\n"
        "    %s --oracle --tasks-dir %s --limit 64\n"
        "    %s --base-url <server> --tasks-dir %s --limit 256\n"
        "    %s --base-url <server> --attempts 8 --tasks-dir %s --task-list %s",
        eval_command,
        logical / LEMMAS_DIRNAME / "test",
        eval_command,
        logical / LEMMAS_DIRNAME / "test",
        eval_command,
        logical / "tasks",
        logical / LEMMAS_DIRNAME / lemma_tasks.OFFICIAL_TEST_TASKS_NAME,
    )


def _verify_staged_assets(
    manifest_path: Path,
    checkpoint_dir: Path,
    dataset_dir: Path,
) -> None:
    """Check that existing assets match the manifest.

    Without a manifest, existing dataset files cannot be tied to the pinned
    revision, so the script refuses to reuse them. The checkpoint is shared
    with the tmax baseline, which pins the same release, so an existing
    checkpoint is accepted and then recorded here.

    Args:
      manifest_path: The manifest to check, when one exists.
      checkpoint_dir: The staged checkpoint directory.
      dataset_dir: The staged dataset directory.

    Raises:
      ValueError: The manifest is unreadable or the assets do not match it.
      TypeError: The manifest is not an object with a file map.

    """
    if not manifest_path.is_file():
        existing = _relative_files(dataset_dir)
        if existing:
            raise ValueError(
                "Refusing to label existing dataset assets as pinned without a"
                f" prior manifest: {existing[:8]} ({len(existing)} files)."
                " Re-stage into a clean directory.",
            )
        return
    try:
        raw_payload = cast(
            "object",
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{manifest_path} could not be read: {error}.") from error
    if not isinstance(raw_payload, dict):
        raise TypeError(f"{manifest_path} must contain an object.")
    payload = cast("dict[str, object]", raw_payload)
    _verify_manifest_identity(payload, manifest_path)
    files = payload.get("files", {})
    if not isinstance(files, Mapping):
        raise TypeError(f"{manifest_path} records no file map.")
    file_items = cast("Mapping[object, object]", files)
    # The writer records each SHA-256 hash as a 64-character string.
    if not all(
        isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
        for name, digest in file_items.items()
    ):
        raise TypeError(f"{manifest_path} contains an invalid file digest map.")
    recorded = cast("dict[str, str]", files)
    for label, directory, strict in (
        ("checkpoint", checkpoint_dir, False),
        ("dataset", dataset_dir, True),
    ):
        prefix = f"{label}/"
        expected = {
            name.removeprefix(prefix): digest
            for name, digest in recorded.items()
            if name.startswith(prefix)
        }
        changed = sorted(
            name
            for name, digest in expected.items()
            if not _matches_digest(directory / name, digest)
        )
        if changed:
            raise ValueError(
                f"Staged {label} assets no longer match {manifest_path}: "
                f"{changed[:8]}. Re-stage into a clean directory, or pass a"
                " fresh --manifest-path to record the new state deliberately.",
            )
        if not strict:
            continue
        actual = set(_relative_files(directory, exclude=manifest_path))
        if actual != set(expected):
            raise ValueError(
                f"Staged {label} file inventory no longer matches "
                f"{manifest_path}: expected {len(expected)} files, got"
                f" {len(actual)}. Re-stage into a clean directory.",
            )


def _verify_manifest_identity(payload: dict[str, object], path: Path) -> None:
    """Check that the manifest names the expected pinned sources."""
    expected: dict[str, object] = {
        "schema": 1,
        "experiment": "exp000",
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "source_commit": SOURCE_COMMIT,
        },
        "snapshot": SNAPSHOT_NAME,
        "tasks": EXPECTED_TASKS,
        "lemmas": {"train": EXPECTED_TRAIN_LEMMAS, "test": EXPECTED_TEST_LEMMAS},
    }
    changed = [key for key, value in expected.items() if payload.get(key) != value]
    if changed:
        raise ValueError(
            f"{path} was produced from different source pins: {changed}.",
        )


def _stage_checkpoint(directory: Path) -> None:
    """Fetch every file at the pinned revision, including interrupted downloads.

    A config file alone is not evidence that the model weights arrived. The
    checkpoint is shared with TMax, so reuse the hub's local files while asking
    it to finish or revalidate each file in the pinned inventory.
    """
    names = list_repo_files(CHECKPOINT_REPO, revision=CHECKPOINT_REVISION)
    if not names or "config.json" not in names:
        raise ValueError(
            f"{CHECKPOINT_REPO}@{CHECKPOINT_REVISION} has no model config.",
        )
    logger.info("checkpoint: %s @ %s", CHECKPOINT_REPO, CHECKPOINT_REVISION)
    for name in names:
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=name,
            revision=CHECKPOINT_REVISION,
            local_dir=directory,
        )
        if not (directory / name).is_file():
            raise ValueError(f"Checkpoint download did not stage {directory / name}.")


def _stage_dataset(directory: Path) -> None:
    """Download every file of the pinned dataset revision."""
    registry = directory / "registry.json"
    tasks = directory / "tasks"
    if registry.is_file() and tasks.is_dir():
        logger.info("dataset: already staged at %s", directory)
        return
    logger.info("dataset: %s @ %s", DATASET_REPO, DATASET_REVISION)
    snapshot_download(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        repo_type="dataset",
        local_dir=directory,
        max_workers=8,
    )


def _verify_tasks(directory: Path) -> int:
    """Check the staged tasks against the pinned revision's own record.

    The dataset only publishes tasks whose reference proof the oracle accepted
    in a separate verifier sandbox, so that field is the integrity check that
    matters: a task without it would grade differently than the baseline says.

    Args:
      directory: The staged dataset directory.

    Returns:
      The number of verified tasks.

    Raises:
      ValueError: The staged tasks disagree with the registry or the pins.
      TypeError: A staged JSON or TOML file has the wrong shape.

    """
    registry = _read_json(directory / "registry.json")
    entries = registry.get("tasks")
    if not isinstance(entries, list):
        raise TypeError(f"{directory / 'registry.json'} lists no tasks.")
    names = sorted(
        _entry_name(cast("dict[str, object]", entry))
        for entry in cast("list[object]", entries)
    )
    if len(names) != EXPECTED_TASKS:
        raise ValueError(
            f"The pinned revision should hold {EXPECTED_TASKS} tasks, found"
            f" {len(names)}.",
        )
    staged = sorted(path.name for path in (directory / "tasks").iterdir())
    if staged != names:
        raise ValueError(
            "The staged task directories do not match registry.json: "
            f"{sorted(set(names) ^ set(staged))[:8]}.",
        )
    for name in names:
        _verify_task(directory / "tasks" / name, name)
    return len(names)


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object, reporting the file when it is malformed."""
    try:
        payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} could not be read: {error}.") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object.")
    return cast("dict[str, object]", payload)


def _entry_name(entry: dict[str, object]) -> str:
    """Read one registry entry's task name."""
    name = entry.get("name")
    if not isinstance(name, str):
        raise TypeError("A registry entry has no task name.")
    return name


def _verify_task(task_dir: Path, name: str) -> None:
    """Check one staged task against the layout the runner expects."""
    for relative in ("instruction.md", "tests/test.sh", "tests/Challenge.lean"):
        if not (task_dir / relative).is_file():
            raise ValueError(f"{task_dir} is missing {relative}.")
    path = task_dir / "task.toml"
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path} could not be read: {error}.") from error
    task = cast("dict[str, object]", payload.get("task", {}))
    metadata = cast("dict[str, object]", payload.get("metadata", {}))
    verifier = cast("dict[str, object]", payload.get("verifier", {}))
    agent = cast("dict[str, object]", payload.get("agent", {}))
    expected: dict[str, object] = {
        "task.name": f"openai-math/{name}",
        "metadata.oracle_result": "accepted",
        "metadata.source_commit": SOURCE_COMMIT,
        "verifier.environment_mode": "separate",
        "agent.network_mode": "no-network",
    }
    found = {
        "task.name": task.get("name"),
        "metadata.oracle_result": metadata.get("oracle_result"),
        "metadata.source_commit": metadata.get("source_commit"),
        "verifier.environment_mode": verifier.get("environment_mode"),
        "agent.network_mode": agent.get("network_mode"),
    }
    changed = [key for key, value in expected.items() if found[key] != value]
    if changed:
        raise ValueError(f"{path} disagrees with the pins: {changed}.")


def _git(directory: Path, *args: str) -> str:
    """Run git in ``directory`` and return its stripped output.

    Raises:
      ValueError: git exited nonzero.

    """
    completed = subprocess.run(  # noqa: S603 -- Fixed git arguments, not input.
        ["git", "-C", str(directory), *args],  # noqa: S607 -- git from PATH.
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed in {directory}: {completed.stderr.strip()}",
        )
    return completed.stdout.strip()


def _stage_source(directory: Path) -> None:
    """Check out openai/math's Lean library at the pinned commit.

    Git is its own integrity check here: an existing checkout must sit at the
    pinned commit with no local changes, or it is refused.

    Raises:
      ValueError: An existing checkout is at another commit or has changes.

    """
    if (directory / ".git").is_dir():
        head = _git(directory, "rev-parse", "HEAD")
        dirty = _git(directory, "status", "--porcelain")
        if head != SOURCE_COMMIT or dirty:
            raise ValueError(
                f"{directory} is at {head} with changes {dirty[:200]!r}; the"
                f" lemma tasks need a clean checkout of {SOURCE_COMMIT}.",
            )
        logger.info("source: already checked out at %s", directory)
        return
    logger.info("source: %s @ %s", SOURCE_REPO, SOURCE_COMMIT)
    directory.mkdir(parents=True, exist_ok=True)
    _git(directory, "init", "-q")
    _git(directory, "remote", "add", "origin", SOURCE_REPO)
    _git(directory, "sparse-checkout", "set", "--no-cone", *SOURCE_PATHS)
    _git(
        directory,
        "fetch",
        "-q",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        SOURCE_COMMIT,
    )
    _git(directory, "checkout", "-q", "FETCH_HEAD")


def _stage_lemmas(dataset_dir: Path, source_dir: Path) -> None:
    """Cut the lemma tasks out of the source, once, and check their counts.

    A ``lemmas/`` directory without both completion files, or with a task list
    that disagrees with the split, is an interrupted run. Regenerate it from
    the pinned source and staged tasks rather than trusting a partial result.

    Raises:
      ValueError: The counts differ from the pinned ones.

    """
    out = dataset_dir / LEMMAS_DIRNAME
    split_path = out / "split.json"
    task_list = out / lemma_tasks.OFFICIAL_TEST_TASKS_NAME
    record = _read_json(split_path) if split_path.is_file() else {}
    official = record.get("official_test_tasks")
    complete = isinstance(official, list) and task_list.is_file()
    if complete:
        complete = task_list.read_text(encoding="utf-8").splitlines() == official
    if complete:
        logger.info("lemmas: already generated at %s", out)
    else:
        if out.exists():
            shutil.rmtree(out)
        logger.info("lemmas: cutting tasks from %s", source_dir / "lean")
        record = lemma_tasks.generate(
            source_dir / "lean",
            dataset_dir / "tasks",
            out,
            source_commit=SOURCE_COMMIT,
        )
    found = (record.get("train_lemmas"), record.get("test_lemmas"))
    if found != (EXPECTED_TRAIN_LEMMAS, EXPECTED_TEST_LEMMAS):
        raise ValueError(
            f"{split_path} records {found} train and held-out lemmas; the pins"
            f" are {(EXPECTED_TRAIN_LEMMAS, EXPECTED_TEST_LEMMAS)}. Re-pin them"
            " only together with a deliberate change to lemma_tasks.py.",
        )


def _generator_snapshot_name(dataset_dir: Path) -> str:
    """Ask the staged generator which snapshot its image recipe builds.

    Raises:
      ValueError: The generator could not be run or printed nothing.

    """
    generator = dataset_dir / "generator"
    completed = subprocess.run(
        [sys.executable, "-m", "openai_math_harbor.snapshot", "--print-name"],
        cwd=generator,
        capture_output=True,
        text=True,
        check=False,
    )
    name = completed.stdout.strip()
    if completed.returncode != 0 or not name:
        raise ValueError(
            f"{generator} could not print its snapshot name:"
            f" {completed.stderr.strip()}",
        )
    return name


def _relative_files(directory: Path, *, exclude: Path | None = None) -> list[str]:
    """List files below ``directory`` by relative path."""
    if not directory.is_dir():
        return []
    return [
        file.relative_to(directory).as_posix()
        for file in sorted(directory.rglob("*"))
        if file.is_file() and file != exclude and CACHE_DIRNAME not in file.parts
    ]


def _matches_digest(path: Path, digest: str) -> bool:
    """Return whether ``path`` exists with the recorded sha256."""
    if not path.is_file():
        return False
    return _sha256(path) == digest


def _write_manifest(
    checkpoint_dir: Path,
    dataset_dir: Path,
    *,
    tasks: int,
    snapshot: str,
    path: Path | None = None,
) -> Path:
    """Record asset hashes and their pinned sources in a manifest."""
    target = path or (dataset_dir / MANIFEST_NAME)
    files: dict[str, str] = {}
    for label, directory in (("checkpoint", checkpoint_dir), ("dataset", dataset_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Cannot manifest missing {label} directory: {directory}",
            )
        for name in _relative_files(directory, exclude=target):
            files[f"{label}/{name}"] = _sha256(directory / name)
    payload = {
        "schema": 1,
        "experiment": "exp000",
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "source_commit": SOURCE_COMMIT,
        },
        "snapshot": snapshot,
        "tasks": EXPECTED_TASKS,
        "lemmas": {"train": EXPECTED_TRAIN_LEMMAS, "test": EXPECTED_TEST_LEMMAS},
        "task_count": tasks,
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(target)
    return target


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
