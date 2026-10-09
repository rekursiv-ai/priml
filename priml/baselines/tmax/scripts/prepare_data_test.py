"""Tests for the TMax data preparation CLI."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hashlib
import json
import subprocess
import tarfile
import tempfile
import urllib.request

import pyarrow as pa
import pytest

from priml.baselines.tmax.scripts import prepare_data, upstream


def _sha256(path: Path) -> str:
    """Return the digest the script pins the archive to."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zstd_tar(archive: Path, members: dict[str, str]) -> None:
    """Write a zstd-compressed tar without tarfile's 3.14-only ``w:zst`` mode.

    The suite runs on the repository's 3.12 floor as well as on 3.14, and only
    3.14 can WRITE zstd through tarfile. Compressing a plain tar with pyarrow
    keeps the fixture available on both.
    """
    with tempfile.TemporaryDirectory() as scratch:
        plain = Path(scratch) / "plain.tar"
        with tarfile.open(plain, "w:") as tar:
            for name, content in members.items():
                member = Path(scratch) / Path(name).name
                member.write_text(content, encoding="utf-8")
                tar.add(member, arcname=name)
        with pa.CompressedOutputStream(pa.OSFile(str(archive), "wb"), "zstd") as sink:
            sink.write(plain.read_bytes())


def _no_stage(_directory: Path) -> None:
    """Stub a network-backed stage in CLI orchestration tests."""


def _manifest_path(
    _checkpoint: Path,
    rollout: Path,
    **_kwargs: object,
) -> Path:
    """Stub manifest creation in CLI orchestration tests."""
    return rollout / prepare_data.MANIFEST_NAME


def test_defaults_are_the_paths_exp000_uses(tmp_path: Path) -> None:
    """The preparer and exp000 resolve these paths independently.

    A divergence would let a successful preparation be followed by a run that
    cannot find what it needs.
    """
    assert prepare_data.default_directories() == (
        Path("/opt/scratch/models/Qwen3.5-4B"),
        Path("/opt/scratch/datasets/tmax/rollouts/published"),
    )
    assert prepare_data.default_directories(tmp_path) == (
        tmp_path / "models/Qwen3.5-4B",
        tmp_path / "datasets/tmax/rollouts/published",
    )


def test_main_stages_the_requested_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage the checkpoint, rollouts, and upstream in the requested paths."""
    checkpoints = tmp_path / "checkpoints"
    rollouts = tmp_path / "rollouts"
    upstream_dir = tmp_path / "upstream"
    staged: list[tuple[str, Path]] = []

    def record_checkpoint(directory: Path) -> None:
        staged.append(("checkpoint", directory))

    def record_rollouts(directory: Path) -> None:
        staged.append(("rollouts", directory))

    def record_upstream(directory: Path) -> None:
        staged.append(("upstream", directory))

    monkeypatch.setattr(prepare_data, "_stage_checkpoint", record_checkpoint)
    monkeypatch.setattr(prepare_data, "_stage_rollouts", record_rollouts)
    monkeypatch.setattr(prepare_data, "_stage_dataset", _no_stage)
    monkeypatch.setattr(prepare_data, "_stage_upstream", record_upstream)
    monkeypatch.setattr(
        prepare_data,
        "_write_manifest",
        _manifest_path,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data",
            "--checkpoint-dir",
            str(checkpoints),
            "--rollouts-dir",
            str(rollouts),
            "--upstream-dir",
            str(upstream_dir),
        ],
    )
    assert prepare_data.main() == 0
    assert staged == [
        ("checkpoint", checkpoints),
        ("rollouts", rollouts),
        ("upstream", upstream_dir),
    ]


def test_main_falls_back_to_the_default_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exp000's default directories when no paths are given."""
    staged: list[tuple[str, Path]] = []

    def record_checkpoint(directory: Path) -> None:
        staged.append(("checkpoint", directory))

    def record_rollouts(directory: Path) -> None:
        staged.append(("rollouts", directory))

    monkeypatch.setattr(prepare_data, "_stage_checkpoint", record_checkpoint)
    monkeypatch.setattr(prepare_data, "_stage_rollouts", record_rollouts)
    monkeypatch.setattr(prepare_data, "_stage_dataset", _no_stage)
    monkeypatch.setattr(prepare_data, "_stage_upstream", _no_stage)
    monkeypatch.setattr(
        prepare_data,
        "_write_manifest",
        _manifest_path,
    )
    monkeypatch.setattr("sys.argv", ["prepare_data"])
    assert prepare_data.main() == 0
    checkpoints, rollouts = prepare_data.default_directories()
    assert staged == [("checkpoint", checkpoints), ("rollouts", rollouts)]


def test_a_staged_checkpoint_is_left_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotence: a second run must not re-download 8 GB."""

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "a staged checkpoint must not be re-fetched"
        raise AssertionError(message)

    monkeypatch.setattr(prepare_data, "list_repo_files", fail)
    (tmp_path / "config.json").write_text("{}")
    prepare_data._stage_checkpoint(tmp_path)


def test_the_checkpoint_is_fetched_at_the_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every repo file is fetched, all at one revision."""
    fetched: list[tuple[str, str]] = []

    def repo_files(_repo: str, *, revision: str) -> list[str]:
        assert revision == prepare_data.CHECKPOINT_REVISION
        return ["config.json", "model.safetensors"]

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: Path,
    ) -> str:
        assert repo_id == prepare_data.CHECKPOINT_REPO
        fetched.append((filename, revision))
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(prepare_data, "list_repo_files", repo_files)
    monkeypatch.setattr(prepare_data, "hf_hub_download", download)
    prepare_data._stage_checkpoint(tmp_path)
    assert fetched == [
        ("config.json", prepare_data.CHECKPOINT_REVISION),
        ("model.safetensors", prepare_data.CHECKPOINT_REVISION),
    ]


def test_a_staged_shard_is_left_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not download an already-staged shard again."""

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "a staged shard must not be re-downloaded"
        raise AssertionError(message)

    monkeypatch.setattr(prepare_data, "_download", fail)
    (tmp_path / prepare_data.SHARD_NAME).write_text("")
    prepare_data._stage_rollouts(tmp_path)


def test_the_dataset_is_fetched_at_its_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch the parquet and task archive from the pinned revision."""
    fetched: list[tuple[str, str, Path]] = []
    extracted: list[tuple[Path, Path]] = []

    def repo_files(_repo: str, *, revision: str, repo_type: str) -> list[str]:
        assert _repo == prepare_data.DATASET_REPO
        assert revision == prepare_data.DATASET_REVISION
        assert repo_type == "dataset"
        return ["data/train-00000-of-00001.parquet", "task-data.tar.gz"]

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        repo_type: str,
        local_dir: Path,
    ) -> str:
        assert repo_type == "dataset"
        fetched.append((repo_id, revision, Path(filename)))
        return str(local_dir / filename)

    monkeypatch.setattr(prepare_data, "list_repo_files", repo_files)
    monkeypatch.setattr(prepare_data, "hf_hub_download", download)

    def extract(archive: Path, target: Path) -> None:
        extracted.append((archive, target))

    monkeypatch.setattr(prepare_data, "_extract_task_data", extract)
    prepare_data._stage_dataset(tmp_path)

    assert fetched == [
        (
            prepare_data.DATASET_REPO,
            prepare_data.DATASET_REVISION,
            Path("data/train-00000-of-00001.parquet"),
        ),
        (
            prepare_data.DATASET_REPO,
            prepare_data.DATASET_REVISION,
            Path("task-data.tar.gz"),
        ),
    ]
    assert extracted == [
        (
            tmp_path / "task-data.tar.gz",
            tmp_path / prepare_data.TASK_DATA_DIRNAME,
        ),
    ]


def test_task_data_is_extracted_to_upstreams_layout(tmp_path: Path) -> None:
    """Extract task data into the layout expected by upstream."""
    archive = tmp_path / "task-data.tar.gz"
    source = tmp_path / "task"
    source.mkdir()
    (source / "instruction.md").write_text("fix it", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="task-1")

    target = tmp_path / prepare_data.TASK_DATA_DIRNAME
    prepare_data._extract_task_data(archive, target)

    assert (target / "task-1" / "instruction.md").read_text() == "fix it"
    assert not target.with_name(target.name + ".partial").exists()


def test_a_pinned_upstream_checkout_is_left_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified checkout must not be cloned again."""

    def verified(directory: Path) -> Path:
        return directory

    monkeypatch.setattr(
        upstream,
        "verify_checkout",
        verified,
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a staged checkout must not be cloned")

    monkeypatch.setattr(subprocess, "run", fail)
    prepare_data._stage_upstream(tmp_path)


def test_an_upstream_checkout_at_the_wrong_commit_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different checkout must not be silently replaced."""

    def wrong_commit(_directory: Path) -> Path:
        raise ValueError("wrong TMax commit")

    monkeypatch.setattr(upstream, "verify_checkout", wrong_commit)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a wrong checkout must not be cloned over")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="wrong TMax commit"):
        prepare_data._stage_upstream(tmp_path)


def test_the_pinned_archive_is_verified_then_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: download, check the sha256, extract the shard."""
    archive = tmp_path / "downloaded.tar.zst"
    _write_zstd_tar(
        archive,
        {"rollouts/rollouts_part000.jsonl": '{"prompt": "ls"}\n'},
    )

    def archive_path(_directory: Path) -> Path:
        return archive

    monkeypatch.setattr(prepare_data, "_archive_path", archive_path)
    monkeypatch.setattr(prepare_data, "ARCHIVE_SHA256", _sha256(archive))
    directory = tmp_path / "rollouts"
    prepare_data._stage_rollouts(directory)
    staged = directory / prepare_data.SHARD_NAME
    assert staged.read_text() == '{"prompt": "ls"}\n'


def test_rollout_archive_cache_stays_under_the_asset_root(tmp_path: Path) -> None:
    rollouts = tmp_path / "datasets/tmax/rollouts/published"
    assert prepare_data._archive_path(rollouts) == (
        tmp_path / "datasets/tmax/rollouts/cache/tmax_rollouts_part000.tar.zst"
    )


def test_an_archive_that_fails_its_pin_is_refused(tmp_path: Path) -> None:
    """The sha256 pin is what makes this a reproduction, not a lookalike."""
    archive = tmp_path / "rollouts.tar.zst"
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="is not the pinned archive"):
        prepare_data._verify_sha256(archive)


def test_archive_download_never_publishes_an_interrupted_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep interrupted downloads in a partial file."""
    target = tmp_path / "rollouts.tar.zst"
    partial = target.with_suffix(target.suffix + ".partial")
    complete = b"complete pinned archive"
    monkeypatch.setattr(
        prepare_data,
        "ARCHIVE_SHA256",
        hashlib.sha256(complete).hexdigest(),
    )

    def interrupt(_url: str, destination: Path) -> None:
        destination.write_bytes(b"incomplete")
        raise OSError("network interrupted")

    monkeypatch.setattr(urllib.request, "urlretrieve", interrupt)
    with pytest.raises(OSError, match="network interrupted"):
        prepare_data._download(prepare_data.ARCHIVE_URL, target)
    assert not target.exists()
    assert partial.read_bytes() == b"incomplete"

    def finish(_url: str, destination: Path) -> None:
        destination.write_bytes(complete)

    monkeypatch.setattr(urllib.request, "urlretrieve", finish)
    prepare_data._download(prepare_data.ARCHIVE_URL, target)
    assert target.read_bytes() == complete
    assert not partial.exists()


def test_the_shard_is_extracted_from_the_archive(tmp_path: Path) -> None:
    archive = tmp_path / "rollouts.tar.zst"
    _write_zstd_tar(
        archive,
        {"rollouts/rollouts_part000.jsonl": '{"prompt": "ls"}\n'},
    )
    target = tmp_path / prepare_data.SHARD_NAME
    prepare_data._extract_shard(archive, target)
    assert target.read_text() == '{"prompt": "ls"}\n'
    # The partial file is renamed over the target, never left behind.
    assert not target.with_suffix(".jsonl.partial").exists()


def test_a_gzip_archive_is_refused(tmp_path: Path) -> None:
    """Reject gzip archives because the rollout extractor expects zstd."""
    archive = tmp_path / "rollouts.tar.gz"
    member = tmp_path / "rollouts_part000.jsonl"
    member.write_text('{"prompt": "ls"}\n')
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(member, arcname="rollouts/rollouts_part000.jsonl")

    with pytest.raises(tarfile.ReadError):
        prepare_data._extract_shard(archive, tmp_path / prepare_data.SHARD_NAME)


def test_an_archive_without_exactly_one_shard_is_refused(
    tmp_path: Path,
) -> None:
    """Two shards would silently stage the wrong one."""
    archive = tmp_path / "rollouts.tar.zst"
    _write_zstd_tar(archive, {"a.jsonl": "", "b.jsonl": ""})
    with pytest.raises(ValueError, match="Expected one JSONL shard, found 2"):
        prepare_data._extract_shard(archive, tmp_path / prepare_data.SHARD_NAME)


def test_manifest_records_all_staged_files_and_source_pins(tmp_path: Path) -> None:
    """Record the experiment, source pins, and staged file digests."""
    checkpoint = tmp_path / "checkpoint"
    rollouts = tmp_path / "rollouts"
    checkpoint.mkdir()
    rollouts.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (rollouts / prepare_data.SHARD_NAME).write_text("{}\n")

    manifest = prepare_data._write_manifest(checkpoint, rollouts)
    payload = cast(dict[str, object], json.loads(manifest.read_text(encoding="utf-8")))
    dataset = cast(dict[str, object], payload["dataset"])

    assert payload["experiment"] == "exp000"
    assert payload["upstream_tmax_commit"] == "6d3d606"
    assert dataset["revision"] == prepare_data.DATASET_REVISION
    assert payload["files"] == {
        "checkpoint/config.json": _sha256(checkpoint / "config.json"),
        "checkpoint/model.safetensors": _sha256(checkpoint / "model.safetensors"),
        f"rollouts/{prepare_data.SHARD_NAME}": _sha256(
            rollouts / prepare_data.SHARD_NAME,
        ),
    }


def test_manifest_records_the_upstream_pin_without_hashing_it(
    tmp_path: Path,
) -> None:
    """Pin the upstream checkout by commit instead of hashing its files."""
    checkpoint = tmp_path / "checkpoint"
    rollouts = tmp_path / "rollouts"
    checkpoint.mkdir()
    rollouts.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (rollouts / prepare_data.SHARD_NAME).write_text("{}\n")
    upstream_dir = tmp_path / "upstream"

    manifest = prepare_data._write_manifest(
        checkpoint,
        rollouts,
        upstream_dir=upstream_dir,
    )
    payload = cast(dict[str, object], json.loads(manifest.read_text()))

    assert payload["upstream"] == {
        "repo": upstream.UPSTREAM_REPO,
        "commit": upstream.UPSTREAM_COMMIT,
        "path": str(upstream_dir),
    }
    files = cast(dict[str, str], payload["files"])
    assert not any(name.startswith("upstream/") for name in files)


def test_a_staged_asset_that_drifted_from_its_manifest_is_refused(
    tmp_path: Path,
) -> None:
    """Existence is not identity: the manifest is a pin, not a log.

    Marker-file acceptance would take a half-written or hand-modified staging
    as the pinned revision; re-hashing every recorded file is what makes a
    re-run refuse a changed asset instead of silently training on it.
    """
    checkpoint = tmp_path / "checkpoint"
    rollouts = tmp_path / "rollouts"
    dataset = tmp_path / "dataset"
    for directory in (checkpoint, rollouts, dataset):
        directory.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (rollouts / prepare_data.SHARD_NAME).write_text("{}\n")
    (dataset / "task-data.tar.gz").write_bytes(b"tasks")
    manifest = prepare_data._write_manifest(
        checkpoint,
        rollouts,
        dataset_dir=dataset,
        path=tmp_path / prepare_data.MANIFEST_NAME,
    )

    prepare_data._verify_staged_assets(manifest, checkpoint, rollouts, dataset)

    (rollouts / prepare_data.SHARD_NAME).write_text("tampered\n")
    with pytest.raises(ValueError, match="no longer match"):
        prepare_data._verify_staged_assets(manifest, checkpoint, rollouts, dataset)
    (checkpoint / "config.json").unlink()
    with pytest.raises(ValueError, match="no longer match"):
        prepare_data._verify_staged_assets(manifest, checkpoint, rollouts, dataset)


def test_verification_passes_silently_without_a_manifest(tmp_path: Path) -> None:
    """A first run has no manifest to verify against; staging writes one."""
    prepare_data._verify_staged_assets(
        tmp_path / prepare_data.MANIFEST_NAME,
        tmp_path / "checkpoint",
        tmp_path / "rollouts",
        tmp_path / "dataset",
    )


def test_verification_refuses_existing_assets_without_a_manifest(
    tmp_path: Path,
) -> None:
    """Reject existing assets that have no manifest."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="without a prior manifest"):
        prepare_data._verify_staged_assets(
            tmp_path / prepare_data.MANIFEST_NAME,
            checkpoint,
            tmp_path / "rollouts",
            tmp_path / "dataset",
        )


def test_verification_refuses_a_manifest_with_different_source_pins(
    tmp_path: Path,
) -> None:
    """Reject a manifest with different source pins."""
    checkpoint = tmp_path / "checkpoint"
    rollouts = tmp_path / "rollouts"
    dataset = tmp_path / "dataset"
    for directory in (checkpoint, rollouts, dataset):
        directory.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (rollouts / prepare_data.SHARD_NAME).write_text("{}\n", encoding="utf-8")
    manifest = prepare_data._write_manifest(
        checkpoint,
        rollouts,
        dataset_dir=dataset,
        path=tmp_path / prepare_data.MANIFEST_NAME,
    )
    payload = cast(
        "dict[str, object]",
        json.loads(manifest.read_text(encoding="utf-8")),
    )
    dataset_payload = cast("dict[str, object]", payload["dataset"])
    dataset_payload["revision"] = "different"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different source pins"):
        prepare_data._verify_staged_assets(manifest, checkpoint, rollouts, dataset)


def test_verification_refuses_an_unrecorded_extra_file(tmp_path: Path) -> None:
    """Reject staged files that the manifest did not record."""
    checkpoint = tmp_path / "checkpoint"
    rollouts = tmp_path / "rollouts"
    dataset = tmp_path / "dataset"
    for directory in (checkpoint, rollouts, dataset):
        directory.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (rollouts / prepare_data.SHARD_NAME).write_text("{}\n", encoding="utf-8")
    manifest = prepare_data._write_manifest(
        checkpoint,
        rollouts,
        dataset_dir=dataset,
        path=tmp_path / prepare_data.MANIFEST_NAME,
    )
    (checkpoint / "foreign.safetensors").write_bytes(b"wrong")

    with pytest.raises(ValueError, match="inventory"):
        prepare_data._verify_staged_assets(manifest, checkpoint, rollouts, dataset)


def test_a_manifest_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """A corrupt manifest cannot verify anything, so it must not pass quietly."""
    manifest = tmp_path / prepare_data.MANIFEST_NAME
    manifest.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        prepare_data._verify_staged_assets(
            manifest,
            tmp_path / "checkpoint",
            tmp_path / "rollouts",
            tmp_path / "dataset",
        )


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """Only the object the writer records can be verified against."""
    manifest = tmp_path / prepare_data.MANIFEST_NAME
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        prepare_data._verify_staged_assets(
            manifest,
            tmp_path / "checkpoint",
            tmp_path / "rollouts",
            tmp_path / "dataset",
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
