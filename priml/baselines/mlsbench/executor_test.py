"""Tests for the executor seam. No docker: ``subprocess.run`` is mocked."""

# ruff: noqa: S603

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, cast

import dataclasses
import json
import os
import shutil
import subprocess
import time

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from priml.baselines.mlsbench import executor as ex
from priml.baselines.mlsbench.executor import (
    Executor,
    ExecutorError,
    ExecutorUnavailable,
    FakeExecutor,
    LocalDocker,
    Outcome,
    RunRequest,
)
from priml.baselines.mlsbench.task import Setting, TaskSpec
from priml.baselines.mlsbench.workspace import Workspace


def typed_request(
    *,
    package: str,
    root: Path,
    workspace: Path,
    label: str,
    script: str,
    time_limit: float,
    seed: int,
    log_path: Path,
    env: tuple[tuple[str, str], ...] = (),
    compute: float = 1.0,
) -> RunRequest:
    """Build a RunRequest from real (minimal) task, setting and workspace values."""
    task = TaskSpec(
        name=package,
        family="test",
        package=package,
        root=root,
        seeds=(seed,),
        files=(),
        settings=(),
        baselines=(),
        leaderboard=(),
        parser=root / "parser.py",
        score_spec=None,
    )
    return RunRequest(
        task=task,
        setting=Setting(
            label=label,
            script=script,
            time_limit=time_limit,
            compute=compute,
        ),
        workspace=Workspace(root=workspace, task=task, edit=None, digest=""),
        seed=seed,
        log_path=log_path,
        env=env,
    )


def _request(tmp_path: Path, *, label: str = "s1", limit: float = 5.0) -> RunRequest:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return typed_request(
        package="synth-pkg",
        root=tmp_path / "task",
        workspace=ws,
        label=label,
        script="scripts/s1.sh",
        time_limit=limit,
        seed=42,
        log_path=tmp_path / "logs" / f"{label}.log",
    )


def _docker(
    tmp_path: Path,
    lock: dict[str, str] | None = None,
    *,
    gpus: str = "all",
    network: str = "auto",
    user: str = "host",
    platform: str = "",
    extra_env: dict[str, str] | None = None,
    binds: list[str] | None = None,
) -> LocalDocker:
    lockfile = tmp_path / "lock.json"
    lockfile.write_text(
        json.dumps({"synth-pkg": "repo/img@sha256:abc"} if lock is None else lock),
    )
    return LocalDocker(
        LocalDocker.Config(
            image_lock=lockfile,
            docker_bin="docker",
            gpus=gpus,
            network=network,
            user=user,
            platform=platform,
            extra_env={} if extra_env is None else extra_env,
            binds=[] if binds is None else binds,
        ),
    )


def _stdout(kw: dict[str, object]) -> BinaryIO:
    return cast("BinaryIO", kw["stdout"])


def _ok(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
    del kw
    return subprocess.CompletedProcess(cmd, 0)


def _local_image(
    fake: Callable[..., subprocess.CompletedProcess[bytes]],
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """Answer ``docker image inspect`` as present; pass every other call to fake."""

    def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if cmd[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0)
        return fake(cmd, **kw)

    return run


@pytest.fixture(autouse=True)
def _fake_which(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if cast("pytest.Item", request.node).get_closest_marker("integration"):
        return  # real docker: keep the real PATH lookup

    def which(binary: str) -> str:
        return f"/usr/bin/{binary}"

    monkeypatch.setattr(shutil, "which", which)


def test_fake_executor_satisfies_protocol_and_records(tmp_path: Path) -> None:
    fake = FakeExecutor(FakeExecutor.Config(outputs={"s1": "acc=1"}, seconds=2.0))
    assert isinstance(fake, Executor)
    req = _request(tmp_path)
    res = fake.run(req)
    assert res.outcome is Outcome.OK
    assert res.returncode == 0
    assert res.log_path.read_text() == "acc=1"
    assert fake.calls == [req]


def test_fake_executor_outcomes(tmp_path: Path) -> None:
    fake = FakeExecutor(FakeExecutor.Config(outcomes={"s1": Outcome.TIMEOUT}))
    assert fake.run(_request(tmp_path)).returncode is None
    fake = FakeExecutor(FakeExecutor.Config(outcomes={"s1": Outcome.CRASHED}))
    assert fake.run(_request(tmp_path)).outcome is Outcome.CRASHED


def test_local_docker_command_and_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.update(cmd=cmd, **kw)
        _stdout(kw).write(b"hello log")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("MLSBENCH_ROOT", "/secret")
    res = _docker(tmp_path, extra_env={"X": "1"}).run(_request(tmp_path))
    cmd = cast("list[str]", seen["cmd"])
    env = cast("dict[str, str]", seen["env"])
    assert res.outcome is Outcome.OK
    assert res.log_path.read_text() == "hello log"
    assert "repo/img@sha256:abc" in cmd
    assert not any(":latest" in c for c in cmd)
    assert "SEED=42" in cmd
    assert "X=1" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert cmd[-2:] == ["bash", f"{ex._TASK_MOUNT}/scripts/s1.sh"]
    assert 4.0 < cast("float", seen["timeout"]) <= 5.0  # the budget, minus elapsed
    assert env["PATH"] == "/usr/bin:/bin"
    assert "MLSBENCH_ROOT" not in env
    assert seen["stderr"] == subprocess.STDOUT


def test_cpu_task_omits_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd)
        return _ok(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    _docker(tmp_path, gpus="").run(_request(tmp_path))
    assert "--gpus" not in seen[0]


def test_cpu_only_setting_omits_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd)
        return _ok(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    req = _request(tmp_path)
    req = dataclasses.replace(
        req,
        setting=dataclasses.replace(req.setting, compute=0.0),
    )
    _docker(tmp_path).run(req)
    assert "--gpus" not in seen[0]


@pytest.mark.parametrize(
    ("gpus", "compute", "wanted"),
    [(0, 0.33, False), (2, 0.0, True), (None, 0.33, True), (None, 0.0, False)],
)
def test_bundle_gpu_count_decides_gpus(
    tmp_path: Path,
    gpus: int | None,
    compute: float,
    wanted: bool,
) -> None:
    req = _request(tmp_path)
    req = dataclasses.replace(
        req,
        gpus=gpus,
        setting=dataclasses.replace(req.setting, compute=compute),
    )
    cmd = _docker(tmp_path).command(req, "i@sha256:a", "n")
    assert ("--gpus" in cmd) is wanted


def test_timeout_kills_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd, cast("float", kw["timeout"]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    res = _docker(tmp_path).run(_request(tmp_path))
    assert res.outcome is Outcome.TIMEOUT
    assert res.returncode is None
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1][1:] == ["kill", name]


def test_timeout_survives_failing_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        del kw
        raise subprocess.TimeoutExpired(cmd, 1.0)

    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    assert _docker(tmp_path).run(_request(tmp_path)).outcome is Outcome.TIMEOUT


def test_crashed_script_is_a_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        del kw
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    res = _docker(tmp_path).run(_request(tmp_path))
    assert (res.outcome, res.returncode) == (Outcome.CRASHED, 3)


def test_missing_digest_refuses_to_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kw: object) -> None:
        del args, kw
        raise AssertionError("docker must not be invoked")

    monkeypatch.setattr(subprocess, "run", boom)
    for lock in ({}, {"synth-pkg": "repo/img:latest"}):
        with pytest.raises(ExecutorError, match="no pinned image digest"):
            _docker(tmp_path, lock=lock).run(_request(tmp_path))


def test_missing_lock_file(tmp_path: Path) -> None:
    d = LocalDocker(LocalDocker.Config(image_lock=tmp_path / "nope.json"))
    with pytest.raises(ExecutorError, match="not found"):
        d.run(_request(tmp_path))


def test_shipped_lock_is_digest_only() -> None:
    lock = cast(
        "dict[str, str]",
        json.loads((ex._TESTDATA / "image_lock.json").read_text()),
    )
    assert len(lock) <= 65  # one digest per upstream base image (65 published)
    assert lock
    assert all("@sha256:" in v and ":latest" not in v for v in lock.values())
    d = LocalDocker(LocalDocker.Config())
    with pytest.raises(ExecutorError, match="no pinned image digest"):
        d.image_ref("synth-pkg-not-pinned")


def test_lock_lookup_is_case_insensitive_on_package(tmp_path: Path) -> None:
    d = _docker(tmp_path, lock={"synthpkg": "r/synthpkg@sha256:1"})
    assert d.image_ref("SynthPkg") == "r/synthpkg@sha256:1"


def test_isolation_flags_and_argv_order(tmp_path: Path) -> None:
    cmd = _docker(tmp_path, gpus="").command(
        _request(tmp_path),
        "repo/img@sha256:abc",
        "n",
    )
    assert cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "no-new-privileges" in cmd
    assert "HOME=/tmp" in cmd
    assert "--platform" not in cmd
    assert cmd[cmd.index("-w") + 1] == "/workspace"
    assert cmd.index("repo/img@sha256:abc") < cmd.index("bash")
    assert not any(c.endswith(":/workspace") for c in cmd)  # never hide the image's


def _layout_request(tmp_path: Path, package: str = "SynthPkg") -> RunRequest:
    task = tmp_path / "task"
    (task / "scripts").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / "SynthPkg").mkdir(parents=True)
    (ws / "SynthPkg" / "custom.py").write_text("x")
    (ws / "SynthPkg" / "pkg").mkdir()
    (ws / "SynthPkg" / "pkg" / "b.py").write_text("y")
    (ws / "SynthPkg" / "link.py").symlink_to(ws / "SynthPkg" / "custom.py")
    return typed_request(
        package=package,
        root=task,
        workspace=ws,
        label="set-a",
        script="scripts/m.sh",
        time_limit=5.0,
        seed=7,
        log_path=tmp_path / "logs" / "set-a.log",
        env=(("PYTHONPATH", "/workspace/SynthPkg"),),
    )


def test_native_layout_task_script_package_cwd_and_files(tmp_path: Path) -> None:
    req = _layout_request(
        tmp_path,
        package="synthpkg",
    )  # config says SynthPkg; case differs
    cmd = _docker(
        tmp_path,
        lock={"synthpkg": "r/i@sha256:1"},
        binds=["/h/d:/data:ro"],
    ).command(req, "r/i@sha256:1", "n")
    mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
    ws = (tmp_path / "ws").resolve()
    task = (tmp_path / "task").resolve()
    # the script lives in the task dir, mounted at /workspace/_task (not in the workspace)
    assert f"{task}:/workspace/_task:ro" in mounts
    assert cmd[-2:] == ["bash", "/workspace/_task/scripts/m.sh"]
    # cwd is the package dir; only files (not dirs) are laid over the image, read-only
    assert cmd[cmd.index("-w") + 1] == "/workspace/SynthPkg"
    assert f"{ws}/SynthPkg/custom.py:/workspace/SynthPkg/custom.py:ro" in mounts
    assert f"{ws}/SynthPkg/pkg/b.py:/workspace/SynthPkg/pkg/b.py:ro" in mounts
    assert not any("link.py" in m for m in mounts)  # symlinks are never followed
    assert not any(m.endswith((":/workspace", ":/workspace/SynthPkg")) for m in mounts)
    assert "/h/d:/data:ro" in mounts
    out = (tmp_path / "logs" / "set-a.out").resolve()
    assert f"{out}:/output" in mounts
    env = {
        cmd[i + 1].split("=")[0]: cmd[i + 1].split("=", 1)[1]
        for i, c in enumerate(cmd)
        if c == "-e"
    }
    assert env["OUTPUT_DIR"] == "/output"
    assert env["ENV"] == "set-a"
    assert env["SEED"] == "7"
    assert env["MLSBENCH_TASK_DIR"] == "/workspace/_task"
    assert env["MLSBENCH_PKG_DIR"] == "/workspace/SynthPkg"
    assert env["PYTHONPATH"] == "/workspace/SynthPkg"


def test_isolation_flags_are_configurable(tmp_path: Path) -> None:
    d = _docker(tmp_path, gpus="0,1", network="bridge", user="", platform="linux/amd64")
    cmd = d.command(_request(tmp_path), "img@sha256:abc", "n")
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert "--user" not in cmd
    assert cmd[cmd.index("--platform") + 1] == "linux/amd64"
    assert cmd[cmd.index("--gpus") + 1] == "0,1"
    assert (
        _docker(tmp_path, network="")
        .command(_request(tmp_path), "i@sha256:a", "n")
        .count("--network")
        == 0
    )


def test_missing_docker_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def which(binary: str) -> None:
        del binary

    monkeypatch.setattr(shutil, "which", which)
    with pytest.raises(ExecutorUnavailable):
        _docker(tmp_path).run(_request(tmp_path))


def test_daemon_down_and_image_pull_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make(msg: bytes) -> Callable[..., subprocess.CompletedProcess[bytes]]:
        def fake_run(
            cmd: list[str],
            **kw: object,
        ) -> subprocess.CompletedProcess[bytes]:
            _stdout(kw).write(msg)
            return subprocess.CompletedProcess(cmd, 125)

        return fake_run

    monkeypatch.setattr(
        subprocess,
        "run",
        make(b"Cannot connect to the Docker daemon at x"),
    )
    with pytest.raises(ExecutorUnavailable):
        _docker(tmp_path).run(_request(tmp_path))
    monkeypatch.setattr(subprocess, "run", make(b"manifest unknown"))
    with pytest.raises(ExecutorError, match="manifest unknown") as info:
        _docker(tmp_path).run(_request(tmp_path))
    assert not isinstance(info.value, ExecutorUnavailable)


def test_a_missing_image_is_pulled_outside_the_time_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _request(tmp_path)
    clock = [0.0]
    calls: list[list[str]] = []
    timeouts: list[object] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        timeouts.append(kw.get("timeout"))
        if cmd[1] == "pull":
            clock[0] += 10 * req.setting.time_limit  # a slow first pull
        return subprocess.CompletedProcess(cmd, 1 if cmd[1] == "image" else 0)

    def monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = _docker(tmp_path).run(req)
    assert res.outcome is Outcome.OK
    assert [c[1] for c in calls] == ["image", "pull", "run"]
    assert calls[1][2] == "repo/img@sha256:abc"
    assert timeouts == [None, None, req.setting.time_limit]


def test_a_failed_pull_is_an_executor_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        _stdout(kw).write(b"pull access denied")
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExecutorError, match="pull access denied") as info:
        _docker(tmp_path).run(_request(tmp_path))
    assert not isinstance(info.value, ExecutorUnavailable)
    assert [c[1] for c in calls] == ["image", "pull"]  # the script never ran


# --- real docker (opt-in: ``-m integration``; skipped without a daemon/image) ---

_LOCK = ex._TESTDATA / "image_lock.json"


def _local_cpu_image() -> str:
    """Return a pinned CPU image present locally, else skip (never pulls)."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker binary not found")
    if subprocess.run([docker, "info"], capture_output=True, check=False).returncode:
        pytest.skip("docker daemon not running")
    lock = cast("dict[str, str]", json.loads(_LOCK.read_text()))
    for ref in lock.values():
        if not subprocess.run(
            [docker, "image", "inspect", ref],
            capture_output=True,
            check=False,
        ).returncode:
            return ref
    pytest.skip(
        "no pinned image present locally (run scripts/pin_images.py --package gplearn)",
    )


@pytest.fixture
def real(tmp_path: Path) -> tuple[LocalDocker, Path]:
    ref = _local_cpu_image()
    lockfile = tmp_path / "lock.json"
    lockfile.write_text(json.dumps({"synth-pkg": ref}))
    ws = tmp_path / "ws"
    ws.mkdir()
    return LocalDocker(LocalDocker.Config(image_lock=lockfile, gpus="")), ws


def _real_request(
    tmp_path: Path,
    ws: Path,
    script: str,
    limit: float = 120.0,
) -> RunRequest:
    task = tmp_path / "task"
    (task / "scripts").mkdir(parents=True, exist_ok=True)
    (task / "scripts" / "s.sh").write_text(script)
    return typed_request(
        package="synth-pkg",
        root=task,
        workspace=ws,
        label="s",
        script="scripts/s.sh",
        time_limit=limit,
        seed=42,
        log_path=tmp_path / "logs" / "s.log",
    )


@pytest.mark.integration
def test_real_docker_end_to_end(tmp_path: Path, real: tuple[LocalDocker, Path]) -> None:
    d, ws = real
    script = (
        "pwd; echo seed=$SEED; id -u; echo PATH=$PATH; echo err >&2\n"
        "echo written > $OUTPUT_DIR/out.txt\n"
        "python -c \"import urllib.request as u; u.urlopen('http://1.1.1.1', timeout=3)\""
        " 2>/dev/null && echo NET=yes || echo NET=no\n"
    )
    res = d.run(_real_request(tmp_path, ws, script))
    log = res.log_path.read_text()
    assert res.outcome is Outcome.OK, log
    assert "/workspace\n" in log
    assert "seed=42" in log
    assert f"\n{os.getuid()}\n" in log  # non-root, host uid
    assert "err" in log  # stderr captured in the same log
    assert "NET=no" in log
    assert (tmp_path / "logs" / "s.out" / "out.txt").read_text() == "written\n"


@pytest.mark.integration
def test_real_docker_crash_and_timeout(
    tmp_path: Path,
    real: tuple[LocalDocker, Path],
) -> None:
    d, ws = real
    res = d.run(_real_request(tmp_path, ws, "echo before; exit 7\n"))
    assert (res.outcome, res.returncode) == (Outcome.CRASHED, 7)
    assert "before" in res.log_path.read_text()
    res = d.run(_real_request(tmp_path, ws, "echo start; sleep 300\n", limit=8.0))
    assert res.outcome is Outcome.TIMEOUT
    assert res.seconds < 60
    docker = shutil.which("docker") or "docker"
    ps = subprocess.run(
        [docker, "ps", "-q", "--filter", "name=mlsbench-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ps.stdout.strip() == "", "timed-out container was left running"


def test_auto_network_follows_the_bundle_and_defaults_to_none(tmp_path: Path) -> None:
    d = _docker(tmp_path, gpus="")
    req = _request(tmp_path)
    assert d.network_for(req) == "none"  # no bundle information
    for value, want in ((True, "bridge"), (False, "none"), (None, "none")):
        req = dataclasses.replace(req, allow_internet=value)
        cmd = d.command(req, "repo/img@sha256:abc", "n")
        assert cmd[cmd.index("--network") + 1] == want
    req = dataclasses.replace(req, allow_internet=True)
    assert _docker(tmp_path, gpus="", network="none").network_for(req) == "none"


# -- package env, per-request binds, held-out input generation -----------------


def _env_of(cmd: list[str]) -> dict[str, str]:
    return {
        cmd[i + 1].split("=")[0]: cmd[i + 1].split("=", 1)[1]
        for i, c in enumerate(cmd)
        if c == "-e"
    }


def test_package_env_cannot_override_harness_owned_variables(tmp_path: Path) -> None:
    req = dataclasses.replace(
        _layout_request(tmp_path),
        env=(
            ("OUTPUT_DIR", "/elsewhere/output"),
            ("SEED", "0"),
            ("SAVE_PATH", "/x"),
            ("ENV", "other"),
            ("MLSBENCH_PKG_DIR", "/elsewhere"),
            ("HF_HOME", "/opt/hf"),
            ("HOME", "/root"),
        ),
    )
    cmd = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}).command(
        req,
        "r/i@sha256:1",
        "n",
    )
    env = _env_of(cmd)
    assert env["OUTPUT_DIR"] == "/output"  # as in Harbor: the harness sets it last
    assert env["SEED"] == "7"
    assert env["SAVE_PATH"] == "/output"
    assert env["ENV"] == "set-a"
    assert env["MLSBENCH_PKG_DIR"] == "/workspace/SynthPkg"
    assert env["HF_HOME"] == "/opt/hf"  # everything else passes through
    assert env["HOME"] == "/tmp"  # noqa: S108 - /root is unwritable for the host uid
    assert cmd.count("HOME=/tmp") == 1
    assert not any(c.startswith("HOME=/root") for c in cmd)


def test_home_from_the_package_env_is_kept_when_running_as_the_image_user(
    tmp_path: Path,
) -> None:
    req = dataclasses.replace(_layout_request(tmp_path), env=(("HOME", "/root"),))
    cmd = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}, user="").command(
        req,
        "r/i@sha256:1",
        "n",
    )
    assert _env_of(cmd)["HOME"] == "/root"


def test_request_binds_follow_the_task_mount_and_the_global_binds(
    tmp_path: Path,
) -> None:
    req = dataclasses.replace(
        _layout_request(tmp_path),
        binds=("/h/staged:/workspace/_task/data:ro",),
    )
    cmd = _docker(
        tmp_path,
        lock={"SynthPkg": "r/i@sha256:1"},
        binds=["/h/d:/data:ro"],
    ).command(req, "r/i@sha256:1", "n")
    mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
    task = f"{(tmp_path / 'task').resolve()}:/workspace/_task:ro"
    assert mounts.index(task) < mounts.index("/h/d:/data:ro")
    assert mounts.index("/h/d:/data:ro") < mounts.index(
        "/h/staged:/workspace/_task/data:ro",
    )  # the staged data overlays the clone's


def test_generated_inputs_are_laid_over_the_workspace_read_only(
    tmp_path: Path,
) -> None:
    gen = tmp_path / "stage" / "SynthPkg" / "_in" / "x.b64"
    gen.parent.mkdir(parents=True)
    gen.write_text("blob")
    cmd = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}).command(
        _layout_request(tmp_path),
        "r/i@sha256:1",
        "n",
        [("SynthPkg/_in/x.b64", gen)],
    )
    assert f"{gen}:/workspace/SynthPkg/_in/x.b64:ro" in [
        cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"
    ]


def _inputgen_request(tmp_path: Path) -> RunRequest:
    gen = tmp_path / "bundle" / "_inputgen"
    gen.mkdir(parents=True)
    (gen / "apply.py").write_text("")
    return dataclasses.replace(_layout_request(tmp_path), inputgen=gen)


def test_inputgen_container_is_isolated_and_stages_outside_the_image(
    tmp_path: Path,
) -> None:
    req = _inputgen_request(tmp_path)
    docker = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}, network="bridge")
    stage = tmp_path / "stage"
    stage.mkdir()
    cmd = docker.inputgen_command(req, "r/i@sha256:1", "n", stage)
    mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
    assert (
        f"{req.inputgen}:/tests/eval/_inputgen:ro" in mounts
    )  # bundle stays read-only
    assert f"{stage.resolve()}:/stage" in mounts
    assert cmd[cmd.index("--network") + 1] == "none"  # generator code gets no network
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert cmd[-3:] == ["/tests/eval/_inputgen/apply.py", "SynthPkg", "/stage"]
    assert cmd[-4] == "python"
    assert not any(":/workspace" in m for m in mounts)


def test_inputs_are_generated_then_mounted_and_a_failure_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _inputgen_request(tmp_path)
    calls: list[list[str]] = []
    rcs = iter([0, 0])

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        if "apply.py" in " ".join(cmd):
            stage = req.log_path.with_suffix(".inputs")
            (stage / "SynthPkg" / "_in").mkdir(parents=True)
            (stage / "SynthPkg" / "_in" / "x.b64").write_text("blob")
            _stdout(kw).write(b"[inputgen] 1 file")
        return subprocess.CompletedProcess(cmd, next(rcs))

    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    docker = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"})
    res = docker.run(req)
    assert res.outcome is Outcome.OK
    assert len(calls) == 2
    assert "apply.py" in " ".join(calls[0])
    assert any(
        m.endswith(":/workspace/SynthPkg/_in/x.b64:ro")
        for m in (calls[1][i + 1] for i, c in enumerate(calls[1]) if c == "-v")
    )
    assert "bash" in calls[1]

    # a generator failure never runs the script
    calls.clear()

    def failing_gen(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        _stdout(kw).write(b"Traceback")
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(subprocess, "run", _local_image(failing_gen))
    res = docker.run(req)
    assert res.outcome is Outcome.MISSING_INPUT
    assert res.detail == "inputgen exited 3"
    assert res.returncode is None
    assert len(calls) == 1
    assert "Traceback" in res.log_path.read_text()


def test_inputgen_daemon_down_and_pull_failure_are_executor_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def make(msg: bytes) -> Callable[..., subprocess.CompletedProcess[bytes]]:
        def fake_run(
            cmd: list[str],
            **kw: object,
        ) -> subprocess.CompletedProcess[bytes]:
            calls.append(cmd)
            _stdout(kw).write(msg)
            return subprocess.CompletedProcess(cmd, 125)

        return fake_run

    docker = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"})
    req = _inputgen_request(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _local_image(make(b"Cannot connect to the Docker daemon at x")),
    )
    with pytest.raises(ExecutorUnavailable):
        docker.run(req)
    monkeypatch.setattr(subprocess, "run", _local_image(make(b"manifest unknown")))
    with pytest.raises(ExecutorError, match="manifest unknown") as info:
        docker.run(req)
    assert not isinstance(info.value, ExecutorUnavailable)
    assert all("apply.py" in " ".join(c) for c in calls)  # the script never ran


def test_a_generator_that_writes_nothing_is_a_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", _ok)
    res = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}).run(
        _inputgen_request(tmp_path),
    )
    assert res.outcome is Outcome.MISSING_INPUT
    assert res.detail == "inputgen produced no files"


def test_a_generator_timeout_is_a_missing_input_and_kills_its_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        del kw
        calls.append(cmd)
        if cmd[1] == "kill":
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    res = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"}).run(
        _inputgen_request(tmp_path),
    )
    assert res.outcome is Outcome.MISSING_INPUT
    assert res.detail == "inputgen timed out"
    assert calls[-1][1] == "kill"


def test_inputgen_and_script_share_one_time_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _inputgen_request(tmp_path)
    limit = req.setting.time_limit
    clock = [0.0]
    timeouts: list[float] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        timeouts.append(cast("float", kw["timeout"]))
        if "apply.py" in " ".join(cmd):
            stage = req.log_path.with_suffix(".inputs")
            (stage / "SynthPkg").mkdir(parents=True, exist_ok=True)
            (stage / "SynthPkg" / "x").write_text("blob")
            clock[0] += spent
        return subprocess.CompletedProcess(cmd, 0)

    def monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(subprocess, "run", _local_image(fake_run))
    docker = _docker(tmp_path, lock={"SynthPkg": "r/i@sha256:1"})

    spent = limit / 4
    assert docker.run(req).outcome is Outcome.OK
    assert timeouts == [limit, limit - spent]

    # a generator that uses the whole budget leaves none for the script
    timeouts.clear()
    spent = limit
    res = docker.run(req)
    assert res.outcome is Outcome.TIMEOUT
    assert timeouts == [limit]  # the script never ran


def test_fake_executor_models_missing_input(tmp_path: Path) -> None:
    fake = FakeExecutor(FakeExecutor.Config(outcomes={"s1": Outcome.MISSING_INPUT}))
    res = fake.run(_request(tmp_path))
    assert res.outcome is Outcome.MISSING_INPUT
    assert res.returncode is None
