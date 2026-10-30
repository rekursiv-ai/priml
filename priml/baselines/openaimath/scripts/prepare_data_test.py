"""Tests for the pinned-asset staging script."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json

import pytest

from priml.baselines.openaimath.scripts import lemma_tasks, prepare_data
from priml.baselines.openaimath.scripts.runner import SNAPSHOT_NAME
from priml.baselines.tmax.scripts.prepare_data import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
)


TASK_FILES = {
    "instruction.md": "# Prove it\n",
    "tests/test.sh": "#!/bin/sh\n",
    "tests/Challenge.lean": "theorem t : True := by trivial\n",
}

TASK_TOML = """\
schema_version = "1.4"

[task]
name = "openai-math/{name}"

[metadata]
oracle_result = "accepted"
source_commit = "{commit}"

[agent]
network_mode = "no-network"

[verifier]
environment_mode = "separate"
"""


def _write_task(
    root: Path,
    name: str,
    *,
    commit: str = prepare_data.SOURCE_COMMIT,
) -> Path:
    """Write one synthetic task directory."""
    task = root / "tasks" / name
    for relative, content in TASK_FILES.items():
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (task / "task.toml").write_text(
        TASK_TOML.format(name=name, commit=commit),
        encoding="utf-8",
    )
    return task


def _stage_dataset(root: Path, names: list[str]) -> Path:
    """Write a synthetic staged dataset with a matching registry."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        _write_task(root, name)
    (root / "registry.json").write_text(
        json.dumps(
            {
                "name": "openai-math",
                "version": "1.0",
                "tasks": [{"name": name, "path": f"tasks/{name}"} for name in names],
            },
        ),
        encoding="utf-8",
    )
    return root


def test_verify_tasks_accepts_the_published_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset that matches the registry and the pins verifies."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 2)
    root = _stage_dataset(tmp_path / "dataset", ["alpha", "beta"])
    assert prepare_data._verify_tasks(root) == 2


def test_verify_tasks_rejects_a_task_the_oracle_never_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unvalidated task is refused rather than trained on."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 1)
    root = _stage_dataset(tmp_path / "dataset", ["alpha"])
    toml = root / "tasks" / "alpha" / "task.toml"
    toml.write_text(
        toml.read_text(encoding="utf-8").replace('"accepted"', '"rejected"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="oracle_result"):
        prepare_data._verify_tasks(root)


def test_verify_tasks_rejects_another_upstream_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Challenges must come from the pinned openai/math commit."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 1)
    root = _stage_dataset(tmp_path / "dataset", ["alpha"])
    _write_task(root, "alpha", commit="0" * 40)
    with pytest.raises(ValueError, match="source_commit"):
        prepare_data._verify_tasks(root)


def test_verify_tasks_rejects_a_registry_that_disagrees_with_the_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray task directory cannot slip past the registry."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 2)
    root = _stage_dataset(tmp_path / "dataset", ["alpha", "beta"])
    _write_task(root, "gamma")
    with pytest.raises(ValueError, match=r"registry\.json"):
        prepare_data._verify_tasks(root)


def test_verify_tasks_rejects_a_task_count_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published task count is part of the pin."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 3)
    root = _stage_dataset(tmp_path / "dataset", ["alpha", "beta"])
    with pytest.raises(ValueError, match="3 tasks"):
        prepare_data._verify_tasks(root)


def test_verify_tasks_rejects_a_task_without_its_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every task needs the graded files the runner stages."""
    monkeypatch.setattr(prepare_data, "EXPECTED_TASKS", 1)
    root = _stage_dataset(tmp_path / "dataset", ["alpha"])
    (root / "tasks" / "alpha" / "tests" / "Challenge.lean").unlink()
    with pytest.raises(ValueError, match=r"Challenge\.lean"):
        prepare_data._verify_tasks(root)


def test_generator_snapshot_name_uses_the_staged_generator(tmp_path: Path) -> None:
    """The snapshot name comes from the dataset's own generator."""
    generator = tmp_path / "dataset" / "generator" / "openai_math_harbor"
    generator.mkdir(parents=True)
    (generator / "__init__.py").write_text("", encoding="utf-8")
    (generator / "snapshot.py").write_text(
        "import sys\n\n"
        "if __name__ == '__main__':\n"
        "    print('openai-math-lean-0123456789ab')\n"
        "    sys.exit(0)\n",
        encoding="utf-8",
    )
    name = prepare_data._generator_snapshot_name(tmp_path / "dataset")
    assert name == "openai-math-lean-0123456789ab"


def test_generator_snapshot_name_reports_a_broken_generator(tmp_path: Path) -> None:
    """A generator that cannot run is an error, not an empty name."""
    (tmp_path / "dataset" / "generator").mkdir(parents=True)
    with pytest.raises(ValueError, match="could not print"):
        prepare_data._generator_snapshot_name(tmp_path / "dataset")


def test_manifest_round_trip_accepts_unchanged_assets(tmp_path: Path) -> None:
    """A second run over unchanged assets verifies and continues."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha", "beta"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=2,
        snapshot=SNAPSHOT_NAME,
    )
    prepare_data._verify_staged_assets(manifest, checkpoint, dataset)
    payload = cast(dict[str, object], json.loads(manifest.read_text(encoding="utf-8")))
    dataset_payload = cast(dict[str, object], payload["dataset"])
    assert payload["task_count"] == 2
    assert payload["tasks"] == prepare_data.EXPECTED_TASKS
    assert payload["snapshot"] == SNAPSHOT_NAME
    assert dataset_payload["revision"] == prepare_data.DATASET_REVISION
    assert "dataset/tasks/alpha/task.toml" in cast(list[object], payload["files"])


def test_manifest_verification_rejects_a_changed_task(tmp_path: Path) -> None:
    """An edited task is caught by its hash."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=1,
        snapshot=SNAPSHOT_NAME,
    )
    (dataset / "tasks" / "alpha" / "instruction.md").write_text(
        "# Something else\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no longer match"):
        prepare_data._verify_staged_assets(manifest, checkpoint, dataset)


def test_manifest_verification_rejects_an_extra_task(tmp_path: Path) -> None:
    """A task added after staging is caught by the inventory check."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=1,
        snapshot=SNAPSHOT_NAME,
    )
    _write_task(dataset, "beta")
    with pytest.raises(ValueError, match="inventory"):
        prepare_data._verify_staged_assets(manifest, checkpoint, dataset)


def test_manifest_verification_tolerates_a_shared_checkpoint(tmp_path: Path) -> None:
    """The tmax baseline's checkpoint directory may hold extra files."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=1,
        snapshot=SNAPSHOT_NAME,
    )
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    prepare_data._verify_staged_assets(manifest, checkpoint, dataset)


def test_manifest_verification_rejects_a_changed_checkpoint(tmp_path: Path) -> None:
    """The recorded checkpoint files still have to match."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=1,
        snapshot=SNAPSHOT_NAME,
    )
    (checkpoint / "config.json").write_text('{"architectures": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no longer match"):
        prepare_data._verify_staged_assets(manifest, checkpoint, dataset)


def test_manifest_verification_refuses_unrecorded_dataset_files(
    tmp_path: Path,
) -> None:
    """Existing dataset files without a manifest are not assumed pinned."""
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    with pytest.raises(ValueError, match="without a prior manifest"):
        prepare_data._verify_staged_assets(
            dataset / prepare_data.MANIFEST_NAME,
            tmp_path / "models",
            dataset,
        )


def test_stage_checkpoint_finishes_an_interrupted_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config already on disk must not skip missing model weights."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    fetched: list[str] = []

    def list_files(repo: str, *, revision: str) -> list[str]:
        assert (repo, revision) == (
            CHECKPOINT_REPO,
            CHECKPOINT_REVISION,
        )
        return [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: Path,
    ) -> Path:
        assert (repo_id, revision, local_dir) == (
            CHECKPOINT_REPO,
            CHECKPOINT_REVISION,
            checkpoint,
        )
        fetched.append(filename)
        target = local_dir / filename
        target.write_text(filename, encoding="utf-8")
        return target

    monkeypatch.setattr(prepare_data, "list_repo_files", list_files)
    monkeypatch.setattr(prepare_data, "hf_hub_download", download)
    prepare_data._stage_checkpoint(checkpoint)
    assert fetched == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all((checkpoint / name).is_file() for name in fetched)


def test_stage_checkpoint_refuses_a_missing_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not manifest a checkpoint that the downloader did not finish."""

    def list_files(_repo: str, *, revision: str) -> list[str]:
        del revision
        return ["config.json", "model.safetensors"]

    monkeypatch.setattr(prepare_data, "list_repo_files", list_files)

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: Path,
    ) -> None:
        del repo_id, revision
        if filename == "config.json":
            local_dir.mkdir()
            (local_dir / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(prepare_data, "hf_hub_download", download)
    with pytest.raises(ValueError, match=r"model\.safetensors"):
        prepare_data._stage_checkpoint(tmp_path / "models")


def test_manifest_identity_rejects_a_repinned_snapshot(tmp_path: Path) -> None:
    """A manifest from other pins cannot vouch for this revision."""
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    dataset = _stage_dataset(tmp_path / "dataset", ["alpha"])
    manifest = prepare_data._write_manifest(
        checkpoint,
        dataset,
        tasks=1,
        snapshot=SNAPSHOT_NAME,
    )
    payload = cast(dict[str, object], json.loads(manifest.read_text(encoding="utf-8")))
    payload["snapshot"] = "openai-math-lean-deadbeefcafe"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different source pins"):
        prepare_data._verify_staged_assets(manifest, checkpoint, dataset)


def test_relative_files_hides_transport_metadata(tmp_path: Path) -> None:
    """Hugging Face download metadata is not part of the pinned assets."""
    (tmp_path / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "download" / "x.metadata").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / prepare_data.MANIFEST_NAME
    manifest.write_text("{}", encoding="utf-8")
    assert prepare_data._relative_files(tmp_path, exclude=manifest) == [
        "registry.json",
    ]


def _fake_generate(
    calls: list[Path],
    counts: tuple[int, int],
) -> object:
    """Return a stand-in for the lemma cutter that records where it wrote."""

    def generate(
        _lean_dir: Path,
        _tasks_dir: Path,
        out_dir: Path,
        *,
        source_commit: str,
    ) -> dict[str, object]:
        assert source_commit == prepare_data.SOURCE_COMMIT
        calls.append(out_dir)
        out_dir.mkdir(parents=True)
        record: dict[str, object] = {
            "train_lemmas": counts[0],
            "test_lemmas": counts[1],
            "official_test_tasks": ["alpha"],
        }
        (out_dir / lemma_tasks.OFFICIAL_TEST_TASKS_NAME).write_text(
            "alpha\n",
            encoding="utf-8",
        )
        (out_dir / "split.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    return generate


def test_stage_lemmas_regenerates_an_interrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lemmas directory without its split record is cut again from scratch."""
    calls: list[Path] = []
    counts = (prepare_data.EXPECTED_TRAIN_LEMMAS, prepare_data.EXPECTED_TEST_LEMMAS)
    monkeypatch.setattr(
        lemma_tasks,
        "generate",
        _fake_generate(calls, counts),
    )
    stale = tmp_path / "dataset" / prepare_data.LEMMAS_DIRNAME / "train" / "lemma-x"
    stale.mkdir(parents=True)
    prepare_data._stage_lemmas(tmp_path / "dataset", tmp_path / "source")
    assert calls == [tmp_path / "dataset" / prepare_data.LEMMAS_DIRNAME]
    assert not stale.exists()

    # A finished run is reused, not cut again.
    prepare_data._stage_lemmas(tmp_path / "dataset", tmp_path / "source")
    assert len(calls) == 1


def test_stage_lemmas_regenerates_when_the_final_task_list_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split from an interrupted older cutter cannot bless a missing list."""
    calls: list[Path] = []
    counts = (prepare_data.EXPECTED_TRAIN_LEMMAS, prepare_data.EXPECTED_TEST_LEMMAS)
    monkeypatch.setattr(lemma_tasks, "generate", _fake_generate(calls, counts))
    out = tmp_path / "dataset" / prepare_data.LEMMAS_DIRNAME
    out.mkdir(parents=True)
    (out / "split.json").write_text(
        json.dumps(
            {
                "train_lemmas": counts[0],
                "test_lemmas": counts[1],
                "official_test_tasks": ["alpha"],
            },
        ),
        encoding="utf-8",
    )
    stale = out / "test" / "stale"
    stale.mkdir(parents=True)
    prepare_data._stage_lemmas(tmp_path / "dataset", tmp_path / "source")
    assert calls == [out]
    assert not stale.exists()
    assert (out / lemma_tasks.OFFICIAL_TEST_TASKS_NAME).read_text(
        encoding="utf-8",
    ) == "alpha\n"


def test_stage_lemmas_regenerates_a_truncated_task_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A written but incomplete list must not be accepted as a completed run."""
    calls: list[Path] = []
    counts = (prepare_data.EXPECTED_TRAIN_LEMMAS, prepare_data.EXPECTED_TEST_LEMMAS)
    monkeypatch.setattr(lemma_tasks, "generate", _fake_generate(calls, counts))
    out = tmp_path / "dataset" / prepare_data.LEMMAS_DIRNAME
    out.mkdir(parents=True)
    (out / "split.json").write_text(
        json.dumps(
            {
                "train_lemmas": counts[0],
                "test_lemmas": counts[1],
                "official_test_tasks": ["alpha"],
            },
        ),
        encoding="utf-8",
    )
    (out / lemma_tasks.OFFICIAL_TEST_TASKS_NAME).write_text("", encoding="utf-8")
    prepare_data._stage_lemmas(tmp_path / "dataset", tmp_path / "source")
    assert calls == [out]


def test_stage_lemmas_refuses_unpinned_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cutter that yields other counts changed, and the pins must say so."""
    calls: list[Path] = []
    monkeypatch.setattr(
        lemma_tasks,
        "generate",
        _fake_generate(calls, (1, 2)),
    )
    with pytest.raises(ValueError, match="Re-pin"):
        prepare_data._stage_lemmas(tmp_path / "dataset", tmp_path / "source")


@pytest.mark.parametrize(
    ("head", "status", "accepted"),
    [
        (prepare_data.SOURCE_COMMIT, "", True),
        ("0" * 40, "", False),
        (prepare_data.SOURCE_COMMIT, " M lean/OAI/Leaf.lean", False),
    ],
)
def test_stage_source_reuses_only_a_clean_pinned_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    status: str,
    accepted: bool,
) -> None:
    """An existing checkout at another commit, or with edits, is refused."""
    (tmp_path / "source" / ".git").mkdir(parents=True)
    ran: list[tuple[str, ...]] = []

    def git(_directory: Path, *args: str) -> str:
        ran.append(args)
        return {"rev-parse": head, "status": status}[args[0]]

    monkeypatch.setattr(prepare_data, "_git", git)
    if accepted:
        prepare_data._stage_source(tmp_path / "source")
    else:
        with pytest.raises(ValueError, match="clean checkout"):
            prepare_data._stage_source(tmp_path / "source")
    # Reuse only inspects; it never fetches over an existing checkout.
    assert [args[0] for args in ran] == ["rev-parse", "status"]


def test_default_directories_follow_exp000() -> None:
    """The staging defaults are the paths exp000's runner reads."""
    checkpoint, dataset = prepare_data.default_directories("/opt/scratch")
    assert checkpoint == Path("/opt/scratch/models/Qwen3.5-4B")
    assert dataset == Path("/opt/scratch/datasets/openaimath/openai-math")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
