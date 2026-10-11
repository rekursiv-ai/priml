"""Tests for the task-data stager (synthetic clones; the real one is opt-in)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import os

import pytest

from priml.baselines.mlsbench.scripts import prepare_data as pd


def _clone(tmp: Path, *, external: bool = False) -> Path:
    for name, files in {
        "has-data": {"a.json": "[1]", "sub/b.txt": "bee"},
        "no-data": {},
        "net-data": {"c.txt": "c"},
    }.items():
        d = tmp / "tasks" / name
        d.mkdir(parents=True)
        cfg = (
            {"requires_network": True}
            if name == "net-data" or (external and name == "no-data")
            else {}
        )
        (d / "config.json").write_text(json.dumps(cfg))
        for rel, text in files.items():
            (d / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / "data" / rel).write_text(text)
    return tmp


def test_stage_copies_and_records_digests(tmp_path: Path) -> None:
    root, dest = _clone(tmp_path / "up"), tmp_path / "out"
    s = pd.stage_task(root, "has-data", dest)
    assert s.copied == 2
    assert (dest / "has-data/data/sub/b.txt").read_text() == "bee"
    manifest = cast("dict[str, object]", json.loads((dest / pd.MANIFEST).read_text()))
    assert manifest["has-data"] == s.files
    assert s.files["a.json"] == pd.sha256(root / "tasks/has-data/data/a.json")


def test_stage_is_idempotent_and_repairs_a_corrupt_copy(tmp_path: Path) -> None:
    root, dest = _clone(tmp_path / "up"), tmp_path / "out"
    _ = pd.stage_task(root, "has-data", dest)
    assert pd.stage_task(root, "has-data", dest).copied == 0
    (dest / "has-data/data/a.json").write_text("tampered")
    assert pd.stage_task(root, "has-data", dest).copied == 1
    assert (dest / "has-data/data/a.json").read_text() == "[1]"
    assert not list(dest.rglob("*.part"))


def test_verify_only_writes_nothing_and_flags_drift(tmp_path: Path) -> None:
    root, dest = _clone(tmp_path / "up"), tmp_path / "out"
    with pytest.raises(pd.StageError, match="missing"):
        _ = pd.stage_task(root, "has-data", dest, verify_only=True)
    assert not dest.exists()
    _ = pd.stage_task(root, "has-data", dest)
    (dest / "has-data/data/a.json").write_text("x")
    with pytest.raises(pd.StageError, match="differs"):
        _ = pd.stage_task(root, "has-data", dest, verify_only=True)


def test_a_task_without_data_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(pd.StageError, match="no data"):
        _ = pd.stage_task(_clone(tmp_path / "up"), "no-data", tmp_path / "o")


def test_survey_flags_external_inputs(tmp_path: Path) -> None:
    info = pd.survey(_clone(tmp_path / "up"))
    assert info["has-data"] == {"ships_data": True, "files": 2, "external": False}
    assert info["no-data"]["ships_data"] is False
    assert info["net-data"]["external"] is True


def test_verify_manifest_catches_missing_and_changed_files(tmp_path: Path) -> None:
    root, dest = _clone(tmp_path / "up"), tmp_path / "out"
    assert pd.verify_manifest(root, dest) == [f"no {pd.MANIFEST} in {dest}"]
    _ = pd.stage_task(root, "has-data", dest)
    assert pd.verify_manifest(root, dest) == []
    (dest / "has-data/data/a.json").write_text("x")
    (dest / "has-data/data/sub/b.txt").unlink()
    assert sorted(pd.verify_manifest(root, dest)) == [
        "has-data/a.json: digest mismatch",
        "has-data/sub/b.txt: missing",
    ]
    (root / "tasks/has-data/data/a.json").write_text("[2]")
    assert any("disagrees with the clone" in p for p in pd.verify_manifest(root, dest))


def test_resolve_root(tmp_path: Path) -> None:
    with pytest.raises(pd.StageError, match="no tasks"):
        _ = pd.resolve_root(tmp_path)
    assert pd.resolve_root(_clone(tmp_path / "up")) == (tmp_path / "up").resolve()


def test_defaults_are_logical_paths_beneath_base_dir() -> None:
    assert pd.default_path(pd.UPSTREAM_DIR) == Path("/opt/scratch/mlsbench")
    assert pd.default_path(pd.STAGED_DIR) == Path(
        "/opt/scratch/datasets/mlsbench_staged",
    )


def test_smoke_builds_the_synthetic_clone(tmp_path: Path) -> None:
    dest = tmp_path / "smoke"
    assert pd.main(["--smoke", str(dest)]) == 0
    assert pd.main(["--smoke", str(dest)]) == 0  # idempotent
    assert (dest / "tasks" / "synth-three-setting" / "config.json").is_file()


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, dest = str(_clone(tmp_path / "up")), str(tmp_path / "out")
    assert pd.main(["--root", root, "--list"]) == 0
    listing = capsys.readouterr().out
    assert "has-data\t2 files" in listing
    assert "net-data" in listing
    assert "no-data" not in listing
    assert pd.main(["--root", root, "--dest", dest, "--all", "--verify"]) == 1
    assert pd.main(["--root", root, "--dest", dest, "--all"]) == 0
    assert "has-data: 2 files ok, 2 written" in capsys.readouterr().out
    assert pd.main(["--root", root, "--dest", dest, "--task", "net-data"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert pd.main(["--root", root, "--dest", dest, "--verify"]) == 0
    assert pd.main(["--root", root, "--dest", dest, "--task", "nope"]) == 1
    assert pd.main(["--root", root, "--dest", dest, "--task", "no-data"]) == 1


_REAL = os.environ.get("MLSBENCH_ROOT", "")


@pytest.mark.integration
@pytest.mark.skipif(not _REAL, reason="set MLSBENCH_ROOT to a read-only clone")
def test_real_clone_stages_and_verifies(tmp_path: Path) -> None:
    root = Path(_REAL)
    info = pd.survey(root)
    shipping = [n for n, i in info.items() if i["ships_data"]]
    assert "llm-dllm-demask-strategy" in shipping
    s = pd.stage_task(root, "llm-dllm-demask-strategy", tmp_path)
    assert {"math_test.json", "HumanEval.jsonl.gz", "c4_texts.json"} <= set(s.files)
    assert pd.verify_manifest(root, tmp_path) == []


# -- dataset binds and staged-data binds ---------------------------------------


def _pkg(root: Path, name: str, cfg: dict[str, object]) -> None:
    d = root / "vendor" / "pkg_configs" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))


def test_dataset_binds_expand_templates_and_skip_absent_hosts(tmp_path: Path) -> None:
    root, dr = tmp_path / "up", tmp_path / "data"
    for sub in ("a", "b", "c"):
        (dr / sub).mkdir(parents=True)
    _pkg(
        root,
        "LLaDA",
        {
            "data_bind": ["{data_root}/a:/data/a", "{data_root}/gone:/data/gone"],
            "data_deps": [
                {"host_path": "{data_root}/a", "container_path": "/data/a"},  # dup
                {"host_path": "{data_root}/b", "container_path": "/data/b"},
                {"host_path": "{project_root}/vendor", "container_path": "/v"},
                {"name": "no-paths"},
            ],
        },
    )
    got = pd.dataset_binds(root, "llada", dr)  # case-insensitive package
    assert got.binds == (f"{dr}/a:/data/a", f"{dr}/b:/data/b", f"{root}/vendor:/v")
    assert got.skipped == (f"{dr}/gone:/data/gone",)


def test_dataset_binds_accept_a_scalar_data_bind(tmp_path: Path) -> None:
    root, dr = tmp_path / "up", tmp_path / "data"
    dr.mkdir()
    _pkg(root, "sk", {"data_bind": "{data_root}:/data"})
    assert pd.dataset_binds(root, "sk", dr).binds == (f"{dr}:/data",)


def test_dataset_binds_without_a_data_root_or_config_are_empty(tmp_path: Path) -> None:
    root = tmp_path / "up"
    _pkg(root, "sk", {"data_bind": "{data_root}:/data"})
    assert pd.dataset_binds(root, "sk", "") == pd.DatasetBinds((), ())
    assert pd.dataset_binds(root, "unknown", tmp_path).binds == ()
    assert pd.dataset_binds(tmp_path / "no-vendor", "sk", tmp_path).binds == ()


def test_a_broken_pkg_config_is_a_stage_error(tmp_path: Path) -> None:
    root = tmp_path / "up"
    d = root / "vendor/pkg_configs/bad"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{")
    with pytest.raises(pd.StageError, match="unreadable pkg config"):
        pd.dataset_binds(root, "bad", tmp_path)


def test_staged_bind_is_read_only_and_digest_checked(tmp_path: Path) -> None:
    root, dest = _clone(tmp_path / "up"), tmp_path / "out"
    assert pd.staged_tasks(dest) == frozenset()
    pd.stage_task(root, "has-data", dest)
    assert pd.staged_tasks(dest) == {"has-data"}
    spec = pd.staged_bind(dest, "has-data")
    assert spec == f"{(dest / 'has-data/data').resolve()}:/workspace/_task/data:ro"
    (dest / "has-data/data/a.json").write_text("[2]")
    with pytest.raises(pd.StageError, match="differs from the manifest"):
        pd.staged_bind(dest, "has-data")
    (dest / "has-data/data/a.json").unlink()
    with pytest.raises(pd.StageError, match="is missing"):
        pd.staged_bind(dest, "has-data")
    with pytest.raises(pd.StageError, match="not staged"):
        pd.staged_bind(dest, "no-data")
    with pytest.raises(pd.StageError, match=r"no manifest\.json"):
        pd.staged_bind(tmp_path / "empty", "has-data")
