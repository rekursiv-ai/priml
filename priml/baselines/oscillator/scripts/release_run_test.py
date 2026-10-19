"""Check release locks, local staging, and command argument handling."""

from importlib import metadata
from io import BytesIO
from pathlib import Path
from typing import cast

import json
import urllib.request

import pytest

from priml.baselines.oscillator.experiments import exp040
from priml.baselines.oscillator.pretrained import AssetManifest, ModelFiles
from priml.baselines.oscillator.pretrained_test import _assets
from priml.baselines.oscillator.scripts import release_run
from priml.baselines.oscillator.scripts.release_run import (
    main,
    model_files,
    protocol_snapshot,
    stage_model,
    verify_lock,
)
from priml.baselines.oscillator.suite import BenchmarkSuite, suite_digest
from priml.baselines.oscillator.suite_test import RepeatForecast, tiny_suite


def test_protocol_lock_covers_config_sources_and_dependency_lock(
    tmp_path: Path,
) -> None:
    source = tmp_path / "priml/baselines/oscillator"
    source.mkdir(parents=True)
    (source / "PROTOCOL.md").write_text("A declared protocol.")
    (source / "pretrained-requirements.txt").write_text("Pinned external requirements.")
    (source / "model.py").write_text("value = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n")
    cfg = tiny_suite()
    snapshot = protocol_snapshot(cfg, root=tmp_path)
    locked = tmp_path / "lock.json"
    locked.write_text(json.dumps(snapshot))
    verify_lock(locked, snapshot=snapshot)
    lockfile.write_text("version = 2\n")
    with pytest.raises(ValueError, match="stale"):
        verify_lock(locked, snapshot=protocol_snapshot(cfg, root=tmp_path))
    lockfile.write_text("version = 1\n")
    (source / "model.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="stale"):
        verify_lock(locked, snapshot=protocol_snapshot(cfg, root=tmp_path))
    (source / "model.py").write_text("value = 1\n")
    cfg.recipes[0].seeds.append(19)
    with pytest.raises(ValueError, match="stale"):
        verify_lock(locked, snapshot=protocol_snapshot(cfg, root=tmp_path))


def test_staging_verifies_existing_files_without_using_the_network(
    tmp_path: Path,
) -> None:
    cfg = _assets(tmp_path)
    original = (tmp_path / "source.json").read_bytes()
    stage_model(cfg)
    assert (tmp_path / "source.json").read_bytes() == original
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        stage_model(cfg)
    cfg.directory = tmp_path / "new"
    cfg.revision = "main"
    with pytest.raises(ValueError, match="revision"):
        stage_model(cfg)
    assert not cfg.directory.exists()


def test_release_assets_are_distinct_pinned_models() -> None:
    assets = model_files(exp040())
    assert {files.model_id for files in assets} == {
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-2",
    }
    assert all(len(files.revision) == 40 for files in assets)


def test_release_arguments_expose_help_and_reject_non_suite_configs() -> None:
    with pytest.raises(SystemExit) as status:
        main(["--help"])
    assert status.value.code == 0
    with pytest.raises(SystemExit) as status:
        main(["--config", "exp000"])
    assert status.value.code == 2


def test_release_cli_freezes_runs_resumes_and_rejects_stale_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes = cfg.recipes[:1]
    cfg.recipes[0].seeds = [11]

    def resolve(name: str) -> BenchmarkSuite.Config:
        del name
        return cfg.copy_tree()

    monkeypatch.setattr(release_run, "resolve_config", resolve)

    def snapshot(config: BenchmarkSuite.Config, *, root: Path) -> dict[str, object]:
        assert root.is_dir()
        return {"config_sha256": suite_digest(config)}

    monkeypatch.setattr(release_run, "protocol_snapshot", snapshot)
    lock = tmp_path / "lock.json"
    base = ["--lock", str(lock)]
    assert main([*base, "--action", "freeze"]) == 0
    with pytest.raises(FileExistsError):
        main([*base, "--action", "freeze"])
    directory = tmp_path / "audit"
    command = [*base, "--phase", "audit", "--directory", str(directory)]
    assert main(command) == 0
    assert main([*command, "--resume"]) == 0
    report = cast(
        dict[str, object],
        json.loads((directory / "results.json").read_text()),
    )
    assert report["phase"] == "audit"
    assert cast(list[dict[str, object]], report["summary"])[0]["failed_runs"] == 0
    assert (directory / "protocol.txt").is_file()
    cfg.recipes[0].study.method = RepeatForecast.Config(factor=float("nan"))
    with pytest.raises(ValueError, match="stale"):
        main(command)
    assert (
        main(
            [
                "--directory",
                str(tmp_path / "development"),
                "--lock",
                str(tmp_path / "absent.json"),
            ],
        )
        == 1
    )


def test_staged_assets_enter_the_snapshot_and_cli_uses_the_requested_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp040()
    assets = model_files(cfg)
    for files in assets:
        path = tmp_path / files.model_id.rsplit("/", 1)[1]
        path.mkdir()
        _assets(path)
        record = cast(AssetManifest, json.loads((path / "source.json").read_text()))
        record.update(model_id=files.model_id, revision=files.revision)
        (path / "source.json").write_text(json.dumps(record))

    def resolve(name: str) -> BenchmarkSuite.Config:
        del name
        return cfg

    monkeypatch.setattr(release_run, "resolve_config", resolve)
    assert main(["--action", "stage", "--model-directory", str(tmp_path)]) == 0
    root = tmp_path / "repo"
    source = root / "priml/baselines/oscillator"
    source.mkdir(parents=True)
    (source / "PROTOCOL.md").write_text("Protocol")
    (source / "pretrained-requirements.txt").write_text("Pinned requirements")
    (root / "pyproject.toml").write_text("project")
    (root / "uv.lock").write_text("lock")
    snapshot = protocol_snapshot(cfg, root=root)
    assert set(cast(dict[str, object], snapshot["assets"])) == {
        files.model_id for files in assets
    }


@pytest.mark.parametrize("interrupted", [False, True])
def test_model_staging_is_atomic_and_hashes_the_actual_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
) -> None:
    cfg = ModelFiles.Config(directory=tmp_path / "weights")

    def open_file(
        request: urllib.request.Request,
        *,
        timeout: int,
    ) -> BytesIO:
        assert timeout == 120
        assert request.full_url.startswith(
            f"https://huggingface.co/{cfg.model_id}/resolve/{cfg.revision}/",
        )
        if interrupted and request.full_url.endswith("model.safetensors"):
            raise OSError("Download interrupted")
        return BytesIO(b"downloaded content")

    monkeypatch.setattr(urllib.request, "urlopen", open_file)
    if interrupted:
        with pytest.raises(OSError, match="interrupted"):
            stage_model(cfg)
        assert not cfg.directory.exists()
        assert not list(tmp_path.iterdir())
    else:
        stage_model(cfg)
        assert cfg.make().directory == cfg.directory
        assert (
            cfg.directory / "model.safetensors"
        ).read_bytes() == b"downloaded content"


def test_environment_records_missing_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "accelerate":
            raise metadata.PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(metadata, "version", version)
    record = release_run.environment()
    assert cast(dict[str, str], record["packages"])["accelerate"] == "not installed"
    assert record["git_head"]
