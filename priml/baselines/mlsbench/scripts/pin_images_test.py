"""Tests for pin_images: synthetic Harbor tree + mocked docker; real root opt-in."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

import pytest


if TYPE_CHECKING:
    from collections.abc import Sequence

from priml.baselines.mlsbench.executor import LocalDocker
from priml.baselines.mlsbench.scripts import pin_images as pi


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "up"
    for task, line in {
        "a": "FROM bohanlyu2022/mlsbench-harbor-alpha:latest\nENV OMP=1",
        "b": "FROM bohanlyu2022/mlsbench-harbor-alpha:latest AS base",
        "c": "FROM bohanlyu2022/mlsbench-harbor-beta",
        "d": "FROM ubuntu:22.04\nRUN echo hi",
    }.items():
        df = root / "harbor" / "tasks-docker" / f"mls-bench__{task}" / "environment"
        df.mkdir(parents=True)
        (df / "Dockerfile").write_text(line + "\n")
    return root


class _Docker:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail or set()

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        ref = cmd[-1]
        if any(f in ref for f in self.fail):
            return subprocess.CompletedProcess(cmd, 1, "", "denied")
        if cmd[1] == "pull":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        repo = ref.rsplit(":", 1)[0]
        digests = json.dumps(["other/x@sha256:00", f"{repo}@sha256:{'ab' * 32}"])
        return subprocess.CompletedProcess(cmd, 0, digests, "")


def test_find_bases_dedupes_and_ignores_non_harbor(tmp_path: Path) -> None:
    assert pi.find_bases(_tree(tmp_path)) == {
        "alpha": "bohanlyu2022/mlsbench-harbor-alpha:latest",
        "beta": "bohanlyu2022/mlsbench-harbor-beta:latest",
    }


def test_find_bases_errors(tmp_path: Path) -> None:
    with pytest.raises(pi.PinError, match="no Harbor"):
        pi.find_bases(tmp_path)
    root = _tree(tmp_path)
    df = root / "harbor/tasks-docker/mls-bench__a/environment/Dockerfile"
    df.write_text("FROM bohanlyu2022/mlsbench-harbor-alpha:v2\n")
    with pytest.raises(pi.PinError, match="conflicting"):
        pi.find_bases(root)


def test_pin_one_pulls_then_inspects_and_picks_matching_repo() -> None:
    d = _Docker()
    ref = pi.pin_one("bohanlyu2022/mlsbench-harbor-alpha:latest", runner=d)
    assert ref == f"bohanlyu2022/mlsbench-harbor-alpha@sha256:{'ab' * 32}"
    assert [c[1] for c in d.calls] == ["pull", "image"]


def test_pin_one_failures() -> None:
    with pytest.raises(pi.PinError, match="pull"):
        pi.pin_one("x/y:latest", runner=_Docker({"x/y"}))

    def no_digest(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "[]", "")

    with pytest.raises(pi.PinError, match="no RepoDigest"):
        pi.pin_one("x/y:latest", runner=no_digest)


def test_pin_merges_keeps_existing_and_reports_failures(tmp_path: Path) -> None:
    root, lock = _tree(tmp_path), tmp_path / "lock.json"
    lock.write_text(json.dumps({"alpha": "old@sha256:11", "zzz": "z@sha256:22"}))
    d = _Docker(fail={"beta"})
    out, errs = pi.pin(root, lock, None, runner=d)
    assert out["alpha"] == "old@sha256:11"  # kept, not re-pulled
    assert set(errs) == {"beta"}
    assert not any("alpha" in c[-1] for c in d.calls)
    out, errs = pi.pin(root, lock, ["alpha"], repin=True, runner=_Docker())
    assert out["alpha"].endswith("ab" * 32)
    assert not errs
    assert json.loads(lock.read_text()) == out
    assert list(out) == sorted(out)


def test_pin_unknown_package(tmp_path: Path) -> None:
    with pytest.raises(pi.PinError, match="unknown"):
        pi.pin(_tree(tmp_path), tmp_path / "l.json", ["nope"], runner=_Docker())


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, lock = _tree(tmp_path), tmp_path / "l.json"
    assert pi.main(["--root", str(root), "--lock", str(lock), "--list"]) == 0
    assert capsys.readouterr().out.count("unpinned") == 2
    assert pi.main(["--root", str(root), "--lock", str(lock)]) == 2  # nothing asked
    missing = str(tmp_path / "no-clone")
    assert pi.main(["--root", missing, "--lock", str(lock), "--list"]) == 2


@pytest.mark.integration
def test_real_upstream_has_65_bases_and_matches_lock() -> None:
    root = os.environ.get("MLSBENCH_ROOT", "")
    if not root:
        pytest.skip("MLSBENCH_ROOT not set")
    bases = pi.find_bases(Path(root))
    assert len(bases) == 65
    assert "llada" in bases
    lock = pi.read_lock(pi.DEFAULT_LOCK)
    assert set(lock) <= set(bases)
    for pkg, ref in lock.items():
        assert ref.startswith(f"{pi.BASE_PREFIX}{pkg}@sha256:")


# --- registry mode: fake fetcher, no network ---------------------------------

_LIST = {
    "mediaType": "application/vnd.oci.image.index.v1+json",
    "manifests": [
        {"digest": "sha256:aa", "platform": {"os": "linux", "architecture": "amd64"}},
        {
            "digest": "sha256:bb",
            "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"},
        },
        {
            "digest": "sha256:cc",
            "platform": {"os": "unknown", "architecture": "unknown"},
        },
    ],
}
_SINGLE = {
    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
    "config": {"digest": "sha256:cfg"},
}


class _Hub:
    """Fake auth + registry keyed by repo name."""

    def __init__(self, docs: dict[str, object], *, bad_digest: bool = False) -> None:
        self.docs = docs
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.bad_digest = bad_digest

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
    ) -> tuple[int, dict[str, str], bytes]:
        self.urls.append(url)
        self.headers.append(headers)
        if url.startswith(pi._AUTH_URL):
            return 200, {}, b'{"token": "tok"}'
        repo = re.split(r"/(?:manifests|blobs)/", url.split("/v2/")[1])[0]
        name = repo.rsplit("/", 1)[-1]
        if "/blobs/" in url:
            return 200, {}, b'{"os": "linux", "architecture": "amd64"}'
        if name not in self.docs:
            return 404, {}, b'{"errors": "unknown"}'
        body = json.dumps(self.docs[name]).encode()
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        if self.bad_digest:
            digest = "sha256:" + "0" * 64
        return 200, {"docker-content-digest": digest}, body


def _sha(doc: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(doc).encode()).hexdigest()


def test_resolve_registry_index_lists_platforms_and_skips_attestations() -> None:
    hub = _Hub({"mlsbench-harbor-alpha": _LIST})
    res = pi.resolve_registry("bohanlyu2022/mlsbench-harbor-alpha:latest", hub)
    assert res.digest == _sha(_LIST)
    assert res.platforms == {"linux/amd64": "sha256:aa", "linux/arm64/v8": "sha256:bb"}
    assert (
        "scope=repository%3Abohanlyu2022%2Fmlsbench-harbor-alpha%3Apull" in hub.urls[0]
    )
    assert hub.headers[1]["Authorization"] == "Bearer tok"
    assert "manifest.list.v2" in hub.headers[1]["Accept"]
    assert hub.urls[1].endswith("/manifests/latest")


def test_resolve_registry_accepts_a_digest_reference() -> None:
    hub = _Hub({"mlsbench-harbor-alpha": _LIST})
    ref = f"bohanlyu2022/mlsbench-harbor-alpha@{_sha(_LIST)}"
    res = pi.resolve_registry(ref, hub)
    assert res.digest == _sha(_LIST)
    assert (
        "scope=repository%3Abohanlyu2022%2Fmlsbench-harbor-alpha%3Apull" in hub.urls[0]
    )
    assert hub.urls[1].endswith(f"/manifests/{_sha(_LIST)}")


def test_resolve_registry_single_manifest_reads_platform_from_config() -> None:
    hub = _Hub({"mlsbench-harbor-alpha": _SINGLE})
    res = pi.resolve_registry("bohanlyu2022/mlsbench-harbor-alpha:latest", hub)
    assert res.digest == _sha(_SINGLE)
    assert res.platforms == {"linux/amd64": res.digest}


def test_resolve_registry_errors() -> None:
    with pytest.raises(pi.PinError, match="HTTP 404"):
        pi.resolve_registry("bohanlyu2022/mlsbench-harbor-zzz:latest", _Hub({}))
    bad = _Hub({"mlsbench-harbor-alpha": _LIST}, bad_digest=True)
    with pytest.raises(pi.PinError, match="Docker-Content-Digest"):
        pi.resolve_registry("bohanlyu2022/mlsbench-harbor-alpha:latest", bad)

    def no_token(
        _url: str,
        _headers: dict[str, str],
    ) -> tuple[int, dict[str, str], bytes]:
        return 401, {}, b""

    with pytest.raises(pi.PinError, match="token"):
        pi.resolve_registry("x/y:latest", no_token)


def test_pin_registry_writes_lock_and_platforms(tmp_path: Path) -> None:
    root, lock, plats = _tree(tmp_path), tmp_path / "l.json", tmp_path / "p.json"
    lock.write_text(json.dumps({"alpha": "old@sha256:11"}))
    hub = _Hub({"mlsbench-harbor-beta": _LIST})
    out, errs = pi.pin_registry(root, lock, None, platforms_path=plats, fetch=hub)
    assert not errs
    assert out["alpha"] == "old@sha256:11"  # kept
    assert out["beta"] == f"bohanlyu2022/mlsbench-harbor-beta@{_sha(_LIST)}"
    assert cast("object", json.loads(lock.read_text())) == out
    assert list(cast("dict[str, object]", json.loads(plats.read_text()))) == ["beta"]
    assert not any("alpha" in u for u in hub.urls)


def test_pin_registry_records_failures_and_continues(tmp_path: Path) -> None:
    root, lock = _tree(tmp_path), tmp_path / "l.json"
    hub = _Hub({"mlsbench-harbor-beta": _SINGLE})  # alpha -> 404
    out, errs = pi.pin_registry(root, lock, None, fetch=hub)
    assert set(errs) == {"alpha"}
    assert "404" in errs["alpha"]
    assert set(out) == {"beta"}


def test_urllib_fetch_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Opener:
        def open(self, _req: object, timeout: float) -> object:
            del timeout
            calls.append("x")
            raise urllib.error.URLError("boom")

    def _build_opener(*handlers: object) -> _Opener:
        del handlers
        return _Opener()

    monkeypatch.setattr(urllib.request, "build_opener", _build_opener)
    sleeps: list[float] = []
    with pytest.raises(pi.PinError, match="after 3 attempts"):
        pi.urllib_fetch("https://x/", {}, attempts=3, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_cli_registry_mode_uses_injected_network_only_via_fetcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock, plats = _tree(tmp_path), tmp_path / "l.json", tmp_path / "p.json"
    hub = _Hub({"mlsbench-harbor-alpha": _LIST, "mlsbench-harbor-beta": _SINGLE})
    monkeypatch.setattr(pi, "urllib_fetch", hub)
    argv = ["--root", str(root), "--lock", str(lock), "--platforms", str(plats)]
    assert pi.main([*argv, "--registry", "--all"]) == 0
    assert set(cast("dict[str, str]", json.loads(lock.read_text()))) == {
        "alpha",
        "beta",
    }
    assert set(cast("dict[str, object]", json.loads(plats.read_text()))) == {
        "alpha",
        "beta",
    }


def test_real_lock_is_fully_pinned_and_consumable() -> None:
    lock = pi.read_lock(pi.DEFAULT_LOCK)
    if len(lock) < 65:
        pytest.skip("lock not fully pinned")
    ex = LocalDocker(LocalDocker.Config())  # type: ignore[call-arg]
    for pkg, ref in lock.items():
        assert ex.image_ref(pkg) == ref
        assert re.fullmatch(rf"{pi.BASE_PREFIX}{pkg}@sha256:[0-9a-f]{{64}}", ref)
