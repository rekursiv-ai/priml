"""Where a task's setting script actually runs.

:class:`Executor` is the one seam between the harness and a container runtime.
:class:`LocalDocker` shells out to the ``docker`` CLI (no SDK dependency);
:class:`FakeExecutor` replays canned logs so everything above it is testable
with no container. Only infrastructure problems raise (:class:`ExecutorError`
and its subclass :class:`ExecutorUnavailable`); a timeout or a crashing script
is a *value* (:class:`Outcome`), because a benchmark run must record it.

Images are never referenced by tag. ``image_lock.json`` maps a task package to
``repo@sha256:...``; a package with no digest refuses to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, cast, runtime_checkable

import contextlib
import json
import os
import shutil
import subprocess
import time
import uuid

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.baselines.mlsbench.task import Setting, TaskSpec  # noqa: TC004
    from priml.baselines.mlsbench.workspace import Workspace  # noqa: TC004


_TESTDATA = Path(__file__).parent / "testdata"
_SANITISED_PATH = "/usr/bin:/bin"
_PASSTHROUGH_ENV = ("HOME", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT")
_CONTAINER_ROOT = "/workspace"
"""Where the image keeps its package sources; also the container's working root."""
_TASK_MOUNT = f"{_CONTAINER_ROOT}/_task"
"""Native runs bind the task directory here (scripts read ``/workspace/_task/data``)."""
_OUTPUT_MOUNT = "/output"
_INPUTGEN_MOUNT = "/tests/eval/_inputgen"
"""Where Harbor's verifier mounts the held-out input generator."""
_DOCKER_FAILED = 125
"""``docker run`` exits 125 when docker itself (not the container) fails."""
_DAEMON_DOWN = ("cannot connect to the docker daemon", "is the docker daemon running")


class ExecutorError(RuntimeError):
    """The executor could not run the request (not a timeout or a crash)."""


class ExecutorUnavailable(ExecutorError):  # noqa: N818
    """The docker binary is missing or its daemon is down."""


class Outcome(StrEnum):
    """How a run ended."""

    OK = "ok"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    MISSING_INPUT = "missing_input"
    """An input the setting needs (held-out blobs, staged data) was absent or could not
    be materialized; the script was never run. ``RunResult.detail`` says which."""


@dataclass(frozen=True)
class RunRequest:
    """One setting of one task, in a materialized workspace."""

    task: TaskSpec
    setting: Setting
    workspace: Workspace
    seed: int
    log_path: Path
    """Where the executor writes combined stdout and stderr."""

    allow_internet: bool | None = None
    """The task bundle's ``allow_internet``; ``None`` = unknown (treated as no network)."""

    gpus: int | None = None
    """The task bundle's GPU count; ``0`` = CPU-only task, ``None`` = unknown."""

    env: tuple[tuple[str, str], ...] = ()
    """Extra ``(name, value)`` container variables, e.g. the package's env block
    (``PYTHONPATH``, ``HF_HOME``...) that native runs take from the package config
    and that the digest-pinned image does not bake in. Applied after ``extra_env``. The
    harness-owned names (``SEED``, ``ENV``, ``OUTPUT_DIR``, ``SAVE_PATH``,
    ``MLSBENCH_TASK_DIR``, ``MLSBENCH_PKG_DIR``) always win over it, as in Harbor's
    verifier (a package env of ``OUTPUT_DIR=/tmp/output`` must not redirect the
    harness's output mount), and ``HOME`` is ignored when the container runs as the
    host uid (``/root`` is unwritable there)."""

    binds: tuple[str, ...] = ()
    """Per-request ``-v`` specs (``host:container[:ro]``), applied after the task
    mount so a staged ``data/`` overlays the clone's. Complements the global
    ``LocalDocker.Config.binds``."""

    inputgen: Path | None = None
    """Host directory of the bundle's verifier-only ``tests/eval/_inputgen`` (with
    ``apply.py``). When set, the executor materializes the held-out input blobs in a
    throwaway container first, as Harbor's eval scripts do in-container, and lays
    them over ``/workspace`` read-only. Read at run time, never copied into a tree."""


@dataclass(frozen=True)
class RunResult:
    """What happened. ``returncode`` is ``None`` on timeout."""

    outcome: Outcome
    returncode: int | None
    log_path: Path
    seconds: float
    detail: str = ""
    """Why, for ``MISSING_INPUT``; empty otherwise."""


@runtime_checkable
class Executor(Protocol):
    """Runs one setting. Must not raise for a timeout or a crash."""

    def run(self, request: RunRequest) -> RunResult:
        """Run ``request`` and return its outcome."""
        ...


def _read_lock(path: Path) -> dict[str, str]:
    try:
        raw = cast("object", json.loads(path.read_text()))
    except FileNotFoundError as e:
        raise ExecutorError(f"image lock file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ExecutorError(f"image lock file is not valid JSON: {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ExecutorError(f"image lock must map package -> ref: {path}")
    lock = cast("dict[object, object]", raw)
    out = {k: v for k, v in lock.items() if isinstance(k, str) and isinstance(v, str)}
    if len(out) != len(lock):
        raise ExecutorError(f"image lock must map package -> ref: {path}")
    return out


class LocalDocker:
    """Runs a setting script in a digest-pinned container via the docker CLI."""

    class Config(Fig["LocalDocker"]):
        """Docker invocation parameters."""

        image_lock: Path | str = "image_lock.json"
        """package -> ``repo@sha256:...``. Relative paths resolve under ``testdata/``."""

        docker_bin: str = "docker"
        """Resolved on the caller's PATH; the child then runs with a sanitised one."""

        gpus: str = "all"
        """``--gpus`` value; ``""`` = never. Not passed for a CPU-only task (see
        :meth:`LocalDocker.wants_gpus`), so those run without the NVIDIA runtime."""

        network: str = "auto"
        """``--network`` value. ``"auto"`` follows the task bundle's ``allow_internet``
        (``bridge`` when true) and falls back to ``none`` when the request carries no
        bundle information. Any other value (``none``, ``bridge``, ``""`` = docker's
        default) is used as given, for every task."""

        user: str = "host"
        """``--user``: ``"host"`` = the caller's ``uid:gid`` (non-root, and the
        bind-mounted workspace stays writable); ``""`` = the image default."""

        platform: str = ""
        """``--platform`` (e.g. ``linux/amd64`` on an arm host); ``""`` = default."""

        extra_env: dict[str, str] = field(default_factory=dict[str, str])
        """Extra ``-e`` variables passed into the container."""

        binds: list[str] = field(default_factory=list[str])
        """Extra ``-v`` specs (``host:container[:ro]``), e.g. a ``/data`` model or
        dataset mount. Nothing is mounted for data by default."""

        kill_grace: float = 10.0
        """Seconds allowed for ``docker kill`` after a timeout."""

    def __init__(self, config: Config) -> None:
        """Store the config; nothing touches docker until :meth:`run`."""
        self.config = config

    def image_ref(self, package: str) -> str:
        """Return the digest-pinned image for ``package``.

        Raises:
          ExecutorError: The lock has no digest for it. Never falls back to a tag.

        """
        lock = Path(self.config.image_lock)
        if not lock.is_absolute():
            lock = _TESTDATA / lock
        entries = _read_lock(lock)
        # Task configs say "LLaDA"; upstream image repos are lowercase ("llada").
        ref = entries.get(package) or entries.get(package.lower())
        if ref is None or "@sha256:" not in ref:
            raise ExecutorError(
                f"no pinned image digest for package {package!r} in {lock}",
            )
        return ref

    def _docker(self) -> str:
        found = shutil.which(self.config.docker_bin)
        if found is None:
            raise ExecutorUnavailable(
                f"docker binary not found: {self.config.docker_bin!r}",
            )
        return found

    def _env(self) -> dict[str, str]:
        env = {"PATH": _SANITISED_PATH}
        env.update({k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ})
        return env

    @staticmethod
    def wants_gpus(request: RunRequest) -> bool:
        """Whether the setting gets ``--gpus``: not for a CPU-only task.

        The bundle's GPU count decides (``compute`` can be a CPU share on a CPU-only
        task); without a bundle, a setting with ``compute == 0`` is CPU-only.
        """
        if request.gpus is not None:
            return request.gpus > 0
        return request.setting.compute > 0

    def network_for(self, request: RunRequest) -> str:
        """Resolve ``--network``: explicit config wins, ``auto`` reads the bundle."""
        if self.config.network != "auto":
            return self.config.network
        return "bridge" if request.allow_internet is True else "none"

    def command(
        self,
        request: RunRequest,
        image: str,
        name: str,
        generated: Sequence[tuple[str, Path]] = (),
    ) -> list[str]:
        """Build the ``docker run`` command line.

        Args:
          request: The setting to run.
          image: Digest-pinned image reference.
          name: Container name.
          generated: ``(relative path, host path)`` input files materialized by
            :meth:`generate_inputs`, laid over ``/workspace`` read-only.

        Returns:
          argv: The full ``docker run`` argv.

        """
        cfg = self.config
        cmd = [self._docker(), "run", "--rm", "--name", name]
        cmd += ["--security-opt", "no-new-privileges"]
        network = self.network_for(request)
        if network:
            cmd += ["--network", network]
        user = f"{os.getuid()}:{os.getgid()}" if cfg.user == "host" else cfg.user
        if user:
            # A non-root uid has no passwd entry or writable /root: give it a HOME.
            cmd += ["--user", user, "-e", "HOME=/tmp"]
        if cfg.platform:
            cmd += ["--platform", cfg.platform]
        if cfg.gpus and self.wants_gpus(request):
            cmd += ["--gpus", cfg.gpus]
        pkg_dir = self.package_dir(request)
        workdir = f"{_CONTAINER_ROOT}/{pkg_dir}" if pkg_dir else _CONTAINER_ROOT
        out = self.output_dir(request)
        owned = {
            "SEED": str(request.seed),
            "ENV": request.setting.label,
            "OUTPUT_DIR": _OUTPUT_MOUNT,
            "SAVE_PATH": _OUTPUT_MOUNT,
            "MLSBENCH_TASK_DIR": _TASK_MOUNT,
        }
        if pkg_dir:
            owned["MLSBENCH_PKG_DIR"] = workdir
        requested = dict(request.env)
        if user:
            requested.pop("HOME", None)  # the host uid cannot write /root
        env = {**owned, **cfg.extra_env, **requested, **owned}
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        # Layout of a native run: the image's /workspace/<pkg> is kept and only the
        # workspace's own files are laid over it, the task directory is /workspace/_task
        # and the script is ``bash /workspace/_task/<script>`` run from the package dir.
        for rel, src in self.workspace_files(request):
            cmd += ["-v", f"{src}:{_CONTAINER_ROOT}/{rel}:ro"]
        task_root = cast("Path | None", getattr(request.task, "root", None))
        if task_root is not None:
            cmd += ["-v", f"{Path(task_root).resolve()}:{_TASK_MOUNT}:ro"]
        cmd += ["-v", f"{out.resolve()}:{_OUTPUT_MOUNT}"]
        for rel, src in generated:
            cmd += ["-v", f"{src}:{_CONTAINER_ROOT}/{rel}:ro"]
        for spec in (*cfg.binds, *request.binds):
            cmd += ["-v", spec]
        script = request.setting.script
        if task_root is not None and not Path(script).is_absolute():
            script = f"{_TASK_MOUNT}/{script}"
        cmd += ["-w", workdir, image, "bash", script]
        return cmd

    def output_dir(self, request: RunRequest) -> Path:
        """Host directory mounted as ``$OUTPUT_DIR`` (next to the log; created)."""
        out = request.log_path.with_suffix(".out")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def workspace_files(self, request: RunRequest) -> list[tuple[str, Path]]:
        """``(relative path, host path)`` of every regular file in the workspace.

        Files are mounted one by one so the image's own ``/workspace/<pkg>`` sources
        stay visible: the workspace holds only the task's declared files, and a
        directory mount would hide the rest of the package. Symlinks are skipped.
        """
        root = Path(request.workspace.root).resolve()
        return [
            (p.relative_to(root).as_posix(), p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink()
        ]

    def package_dir(self, request: RunRequest) -> str:
        """Top-level workspace directory of the setting's package, or ``""``.

        Matched like native runs do (case, ``-`` and ``_`` insensitive).
        """

        def norm(n: str) -> str:
            return n.lower().replace("-", "").replace("_", "")

        want = norm(request.task.package)
        root = Path(request.workspace.root)
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and norm(d.name) == want:
                return d.name
        return ""

    def inputgen_command(
        self,
        request: RunRequest,
        image: str,
        name: str,
        stage: Path,
    ) -> list[str]:
        """Build the throwaway ``docker run`` that materializes held-out inputs.

        Mirrors the ``python /tests/eval/_inputgen/apply.py <task> /workspace`` line of
        Harbor's eval scripts, but writes into ``stage`` (mounted at ``/stage``) because
        the image's package directory is not writable by the host uid. The bundle's
        ``_inputgen`` directory is mounted read-only at ``/tests/eval/_inputgen``.
        """
        assert request.inputgen is not None
        cfg = self.config
        cmd = [self._docker(), "run", "--rm", "--name", name]
        cmd += ["--security-opt", "no-new-privileges", "--network", "none"]
        user = f"{os.getuid()}:{os.getgid()}" if cfg.user == "host" else cfg.user
        if user:
            cmd += ["--user", user, "-e", "HOME=/tmp"]
        if cfg.platform:
            cmd += ["--platform", cfg.platform]
        cmd += ["-v", f"{Path(request.inputgen).resolve()}:{_INPUTGEN_MOUNT}:ro"]
        cmd += ["-v", f"{stage.resolve()}:/stage"]
        cmd += ["-w", "/stage", image]
        cmd += ["python", f"{_INPUTGEN_MOUNT}/apply.py", request.task.name, "/stage"]
        return cmd

    def generate_inputs(
        self,
        request: RunRequest,
        image: str,
        env: dict[str, str],
    ) -> tuple[list[tuple[str, Path]], str]:
        """Materialize the bundle's held-out input blobs under ``<log>.inputs``.

        Returns:
          files: ``(path relative to /workspace, host path)`` of every generated file.
          error: ``""`` on success, else why materialization failed (output goes to
            the setting's log).

        """
        stage = request.log_path.with_suffix(".inputs")
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        name = f"mlsbench-inputgen-{uuid.uuid4().hex}"
        cmd = self.inputgen_command(request, image, name, stage)
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        with request.log_path.open("wb") as log:
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=request.setting.time_limit,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                self._kill(name, env)
                return [], "inputgen timed out"
            except FileNotFoundError as e:
                raise ExecutorUnavailable(f"cannot execute docker: {e}") from e
        if proc.returncode == _DOCKER_FAILED:
            self._raise_docker_failed(request.log_path, image)
        if proc.returncode != 0:
            return [], f"inputgen exited {proc.returncode}"
        files = [
            (p.relative_to(stage).as_posix(), p)
            for p in sorted(stage.rglob("*"))
            if p.is_file() and not p.is_symlink()
        ]
        if not files:
            return [], "inputgen produced no files"
        return files, ""

    def run(self, request: RunRequest) -> RunResult:
        """Run the setting script; timeout and crash are returned, not raised.

        With ``request.inputgen`` the held-out inputs are materialized first; a failure
        there is ``Outcome.MISSING_INPUT`` (the script is not run).

        Raises:
          ExecutorUnavailable: docker is missing or its daemon is down.
          ExecutorError: No pinned image, or docker failed to start the container.

        """
        image = self.image_ref(request.task.package)
        name = f"mlsbench-{uuid.uuid4().hex}"
        env = self._env()
        generated: list[tuple[str, Path]] = []
        # A first run pulls the image (GBs): never inside the setting's budget.
        self._ensure_image(request, image, env)
        # One time budget per setting covers input generation and the script, so a
        # setting never exceeds ``time_limit`` (``max_eval_time`` is sized on it).
        start = time.monotonic()
        if request.inputgen is not None:
            generated, error = self.generate_inputs(request, image, env)
            if error:
                return RunResult(
                    Outcome.MISSING_INPUT,
                    None,
                    request.log_path,
                    time.monotonic() - start,
                    error,
                )
            # The log now holds the generator's output; the script's replaces it.
        remaining = request.setting.time_limit - (time.monotonic() - start)
        if remaining <= 0:
            return RunResult(
                Outcome.TIMEOUT,
                None,
                request.log_path,
                time.monotonic() - start,
            )
        cmd = self.command(request, image, name, generated)
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        with request.log_path.open("wb") as log:
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                self._kill(name, env)
                return RunResult(
                    Outcome.TIMEOUT,
                    None,
                    request.log_path,
                    time.monotonic() - start,
                )
            except FileNotFoundError as e:
                raise ExecutorUnavailable(f"cannot execute docker: {e}") from e
        seconds = time.monotonic() - start
        rc = proc.returncode
        if rc == _DOCKER_FAILED:
            self._raise_docker_failed(request.log_path, image)
        outcome = Outcome.OK if rc == 0 else Outcome.CRASHED
        return RunResult(outcome, rc, request.log_path, seconds)

    def _ensure_image(
        self,
        request: RunRequest,
        image: str,
        env: dict[str, str],
    ) -> None:
        """Pull ``image`` unless it is already local; its output goes to the log.

        Raises:
          ExecutorUnavailable: docker is missing or its daemon is down.
          ExecutorError: The pull failed.

        """
        docker = self._docker()
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        with request.log_path.open("wb") as log:
            try:
                for cmd in (
                    [docker, "image", "inspect", image],
                    [docker, "pull", image],
                ):
                    proc = subprocess.run(  # noqa: S603
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                    if proc.returncode == 0:
                        return
            except FileNotFoundError as e:
                raise ExecutorUnavailable(f"cannot execute docker: {e}") from e
        self._raise_docker_failed(request.log_path, image)

    @staticmethod
    def _raise_docker_failed(log_path: Path, image: str) -> NoReturn:
        """Classify a docker-level failure (rc 125, or a failed pull): daemon down or not."""
        tail = log_path.read_text(errors="replace")[-2000:]
        if any(s in tail.lower() for s in _DAEMON_DOWN):
            raise ExecutorUnavailable(f"docker daemon unavailable: {tail.strip()}")
        raise ExecutorError(f"docker failed to run {image}: {tail.strip()}")

    def _kill(self, name: str, env: dict[str, str]) -> None:
        """Best-effort ``docker kill``; the client process is already dead."""
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            subprocess.run(  # noqa: S603
                [self._docker(), "kill", name],
                capture_output=True,
                env=env,
                timeout=self.config.kill_grace,
                check=False,
            )


class FakeExecutor:
    """Replays canned logs; records every request. For tests and smoke runs."""

    class Config(Fig["FakeExecutor"]):
        """What each setting label should produce."""

        outputs: dict[str, str] = field(default_factory=dict[str, str])
        """Setting label -> log text to write."""

        outcomes: dict[str, Outcome] = field(default_factory=dict[str, Outcome])
        """Setting label -> outcome; default ``OK``."""

        seconds: float = 0.0

    def __init__(self, config: Config) -> None:
        """Start with no recorded calls."""
        self.config = config
        self.calls: list[RunRequest] = []

    def run(self, request: RunRequest) -> RunResult:
        """Write the canned log and return the canned outcome."""
        self.calls.append(request)
        label = request.setting.label
        outcome = Outcome(self.config.outcomes.get(label, Outcome.OK))
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        request.log_path.write_text(self.config.outputs.get(label, ""))
        rc = {
            Outcome.OK: 0,
            Outcome.TIMEOUT: None,
            Outcome.CRASHED: 1,
            Outcome.MISSING_INPUT: None,
        }[outcome]
        return RunResult(outcome, rc, request.log_path, self.config.seconds)
