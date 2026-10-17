"""Tests for rebuilding the recorded TMax fixtures from a rollout archive."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib import request

import hashlib
import io
import json
import tarfile
import tempfile

from treekle import json_loads

import pytest

from priml.baselines.tmax.scripts import regenerate_recorded_fixtures as regen


if TYPE_CHECKING:
    from pathlib import Path


def _records() -> list[dict[str, object]]:
    return [
        {
            "step": 1,
            "prompt_idx": prompt,
            "sample_idx": sample,
            "reward": float(sample),
            "advantage": sample - 0.5,
        }
        for prompt in (0, 1)
        for sample in (0, 1)
    ]


def _lines(records: list[dict[str, object]]) -> list[bytes]:
    return [json.dumps(record).encode() for record in records]


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin a tiny archive holding two groups of two samples."""
    path = _tar(
        tmp_path / "rollouts.tar",
        {"r/shard.jsonl": b"\n".join(_lines(_records())) + b"\n"},
    )
    monkeypatch.setattr(regen, "ARCHIVE_SHA256", _sha256(path))
    monkeypatch.setattr(regen, "SHARD_RECORDS", 4)
    monkeypatch.setattr(regen, "GROUP_SIZE", 2)
    monkeypatch.setattr(regen, "RECORDED_KEY", (1, 1, 0))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    monkeypatch: pytest.MonkeyPatch,
    archive: Path,
    fixtures: Path,
    *extra: str,
) -> int:
    monkeypatch.setattr(
        "sys.argv",
        ["regen", "--archive", str(archive), "--fixtures", str(fixtures), *extra],
    )
    return regen.main()


def test_write_then_verify_round_trips(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    assert _run(monkeypatch, archive, fixtures, "--write") == 0
    assert "rewrote" in capsys.readouterr().out
    line = (fixtures / "recorded_rollout.jsonl").read_bytes()
    assert json.loads(line) == _records()[1]
    payload = json_loads(
        (fixtures / "recorded_advantages.json").read_bytes(),
        dict[str, object],
    )
    assert payload["groups"] == [
        {
            "step": 1,
            "prompt_idx": prompt,
            "sample_idx": [0, 1],
            "reward": [0.0, 1.0],
            "advantage": [-0.5, 0.5],
        }
        for prompt in (0, 1)
    ]

    assert _run(monkeypatch, archive, fixtures) == 0
    assert capsys.readouterr().out.count("identical") == 2


def test_verify_fails_on_missing_or_drifted_fixtures(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "recorded_rollout.jsonl").write_bytes(b"stale\n")
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, archive, tmp_path)
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "recorded_rollout.jsonl: differs from the committed fixture" in out
    assert "recorded_advantages.json: missing" in out
    assert (tmp_path / "recorded_rollout.jsonl").read_bytes() == b"stale\n"


def test_main_refuses_a_shard_with_the_wrong_record_count(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(regen, "SHARD_RECORDS", 5)
    with pytest.raises(ValueError, match="Expected 5 records, found 4"):
        _run(monkeypatch, archive, tmp_path)


def test_archive_prefers_an_explicit_path(tmp_path: Path) -> None:
    assert regen._archive(tmp_path / "a.tar") == tmp_path / "a.tar"


def test_archive_downloads_once_into_the_temp_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    fetched: list[tuple[str, Path]] = []

    def fake_retrieve(url: str, target: Path) -> None:
        fetched.append((url, target))
        target.write_bytes(b"x")

    monkeypatch.setattr(request, "urlretrieve", fake_retrieve)
    first = regen._archive(None)
    second = regen._archive(None)
    assert first == second == tmp_path / "tmax_rollouts_part000.tar.zst"
    assert fetched == [(regen.ARCHIVE_URL, first)]


def test_verify_sha256_rejects_an_unpinned_archive(tmp_path: Path) -> None:
    path = tmp_path / "other.tar"
    path.write_bytes(b"not it")
    with pytest.raises(ValueError, match="is not the pinned archive"):
        regen._verify_sha256(path)


@pytest.mark.parametrize("count", [0, 2])
def test_shard_lines_requires_exactly_one_jsonl_member(
    tmp_path: Path,
    count: int,
) -> None:
    members = {f"r/{index}.jsonl": b"{}\n" for index in range(count)}
    path = _tar(tmp_path / "a.tar", {"r/readme.txt": b"hi", **members})
    with pytest.raises(ValueError, match=f"found {count}"):
        regen._shard_lines(path)


def test_int_refuses_a_non_int_field() -> None:
    with pytest.raises(TypeError, match="step must be an int, got str"):
        regen._int({"step": "1"}, "step")


def test_recorded_line_requires_exactly_one_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    monkeypatch.setattr(regen, "RECORDED_KEY", (9, 9, 9))
    with pytest.raises(LookupError, match="found 0"):
        regen._recorded_line(_lines(records), records)


@pytest.mark.parametrize(
    "records",
    [
        _records()[:3],
        [_records()[1], _records()[0], *_records()[2:]],
    ],
    ids=["short-group", "out-of-order"],
)
def test_advantages_refuse_a_malformed_group(
    archive: Path,
    records: list[dict[str, object]],
) -> None:
    del archive
    with pytest.raises(ValueError, match="ascending file order"):
        regen._advantages_bytes(records)


def test_advantages_refuse_the_wrong_group_count(archive: Path) -> None:
    del archive
    with pytest.raises(ValueError, match="Expected 2 groups, built 1"):
        regen._advantages_bytes(_records()[:2])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
