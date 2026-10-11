"""Opt-in: does the native script layout resolve inside a real container.

Needs docker plus the pinned gplearn image already present locally (never pulled here).
Runs a *fake* task whose scripts use the relative-path patterns of the real ones, then
(with ``MLSBENCH_ROOT``) the real ``ml-symbolic-regression`` script from a real
materialized workspace. Marked ``integration``; skipped when anything is missing.
"""

# ruff: noqa: S603

from __future__ import annotations

from pathlib import Path
from typing import cast

import dataclasses
import json
import os
import shutil
import subprocess

import pytest

from priml.baselines.mlsbench.executor import (
    _TESTDATA,  # pyright: ignore[reportPrivateUsage]
    LocalDocker,
    Outcome,
    RunRequest,
)
from priml.baselines.mlsbench.executor_test import typed_request


pytestmark = pytest.mark.integration

_PKG = "gplearn"


def _image() -> str:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker binary not found")
    if subprocess.run([docker, "info"], capture_output=True, check=False).returncode:
        pytest.skip("docker daemon not running")
    lock = cast(
        "dict[str, str]",
        json.loads((_TESTDATA / "image_lock.json").read_text()),
    )
    ref = lock.get(_PKG)
    if (
        ref is None
        or subprocess.run(
            [docker, "image", "inspect", ref],
            capture_output=True,
            check=False,
        ).returncode
    ):
        pytest.skip("pinned gplearn image not present locally (never pulled here)")
    return ref


def _docker(tmp_path: Path, ref: str) -> LocalDocker:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({_PKG: ref}))
    return LocalDocker(
        LocalDocker.Config(image_lock=lock, gpus="", platform="linux/amd64"),
    )


def _request(
    tmp_path: Path,
    task: Path,
    ws: Path,
    script: str,
    limit: float,
) -> RunRequest:
    return typed_request(
        package=_PKG,
        root=task,
        workspace=ws,
        label="lab",
        script=script,
        time_limit=limit,
        seed=42,
        log_path=tmp_path / "logs" / "lab.log",
    )


_PROBE = """#!/bin/bash
cd /workspace
export TASK_DIR=/workspace/_task
python gplearn/custom.py --seed ${SEED:-42} --data-path /workspace/_task/data/d.txt \\
    --output-dir ${OUTPUT_DIR:-.}
cd gplearn && python custom.py --relative && ls gplearn/genetic.py
echo pkgdir=$MLSBENCH_PKG_DIR task=$MLSBENCH_TASK_DIR env=$ENV
(echo x > gplearn/custom.py) 2>/dev/null && echo EDIT_WRITABLE=yes || echo EDIT_WRITABLE=no
"""


def test_fake_task_with_real_script_patterns(tmp_path: Path) -> None:
    ref = _image()
    task = tmp_path / "task"
    (task / "scripts").mkdir(parents=True)
    (task / "data").mkdir()
    (task / "data" / "d.txt").write_text("payload")
    (task / "scripts" / "probe.sh").write_text(_PROBE)
    ws = tmp_path / "ws"
    (ws / "gplearn").mkdir(parents=True)
    (ws / "gplearn" / "custom.py").write_text(
        "import os, sys\n"
        "print('custom', os.getcwd(), sys.argv[1:])\n"
        "if '--data-path' in sys.argv:\n"
        "    p = sys.argv[sys.argv.index('--data-path') + 1]\n"
        "    print('data', open(p).read())\n"
        "    out = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "    open(os.path.join(out, 'res.txt'), 'w').write('ok')\n",
    )
    res = _docker(tmp_path, ref).run(
        _request(tmp_path, task, ws, "scripts/probe.sh", 120),
    )
    log = res.log_path.read_text()
    assert res.outcome is Outcome.OK, log
    assert "data payload" in log  # /workspace/_task/data resolves
    assert "custom /workspace ['--seed', '42'" in log  # ran from /workspace
    assert "custom /workspace/gplearn ['--relative']" in log  # relative to the pkg dir
    assert "gplearn/genetic.py" in log  # the image's package source stayed visible
    assert "pkgdir=/workspace/gplearn task=/workspace/_task env=lab" in log
    assert "EDIT_WRITABLE=no" in log  # laid-over files are read-only
    assert (
        tmp_path / "logs" / "lab.out" / "res.txt"
    ).read_text() == "ok"  # $OUTPUT_DIR
    assert (ws / "gplearn" / "custom.py").read_text().startswith("import os")


def test_real_task_script_resolves(tmp_path: Path) -> None:
    """The real ``ml-symbolic-regression`` script starts (finds custom_sr.py, gplearn)."""
    ref = _image()
    root = os.environ.get("MLSBENCH_ROOT", "")
    if not root:
        pytest.skip("MLSBENCH_ROOT not set")
    from priml.baselines.mlsbench.task import load_task  # noqa: PLC0415
    from priml.baselines.mlsbench.workspace import materialize  # noqa: PLC0415

    task = load_task("ml-symbolic-regression", root=Path(root))
    ws = materialize(
        task,
        tmp_path / "run",
        edit=None,
        bundle=Path(root) / "harbor/tasks-docker/mls-bench__ml-symbolic-regression",
    )
    setting = task.settings[0]
    req = RunRequest(
        task=task,
        setting=setting,
        workspace=ws,
        seed=42,
        log_path=tmp_path / "logs" / "real.log",
    )
    # Short deadline: the point is that the command starts, not that it finishes.
    req = dataclasses.replace(
        req,
        setting=dataclasses.replace(setting, time_limit=60.0),
    )
    res = _docker(tmp_path, ref).run(req)
    log = res.log_path.read_text(errors="replace")
    # custom_sr.py was found from the package dir and gplearn imported. The public
    # clone lacks the held-out input blobs the harness would stage (``_sr_inputs``),
    # so the only acceptable failure is that specific missing blob.
    assert "can't open file" not in log, log[-1500:]
    assert "ModuleNotFoundError" not in log, log[-1500:]
    assert (
        'File "/workspace/gplearn/custom_sr.py"' in log
        or res.outcome is not Outcome.CRASHED
    )
    if res.outcome is Outcome.CRASHED:
        assert "_sr_inputs" in log, log[-1500:]


def test_real_task_gets_its_held_out_inputs_and_package_env(tmp_path: Path) -> None:
    """With the bundle's generator the run gets past the missing ``_sr_inputs`` blob."""
    ref = _image()
    root = os.environ.get("MLSBENCH_ROOT", "")
    if not root:
        pytest.skip("MLSBENCH_ROOT not set")
    from priml.baselines.mlsbench.metric import (  # noqa: PLC0415
        inputgen_dir,
        package_env,
    )
    from priml.baselines.mlsbench.task import load_task  # noqa: PLC0415
    from priml.baselines.mlsbench.workspace import materialize  # noqa: PLC0415

    bundle = Path(root) / "harbor/tasks-docker/mls-bench__ml-symbolic-regression"
    task = load_task("ml-symbolic-regression", root=Path(root))
    ws = materialize(task, tmp_path / "run", edit=None, bundle=bundle)
    setting = dataclasses.replace(task.settings[0], time_limit=90.0)
    req = RunRequest(
        task=task,
        setting=setting,
        workspace=ws,
        seed=123,
        log_path=tmp_path / "logs" / "real.log",
        env=package_env(task, setting.package, bundle),
        inputgen=inputgen_dir(bundle),
    )
    assert req.inputgen is not None
    res = _docker(tmp_path, ref).run(req)
    log = res.log_path.read_text(errors="replace")
    assert res.outcome is Outcome.OK, (res.outcome, res.detail, log[-1500:])
    assert res.returncode == 0
    assert "FileNotFoundError" not in log, log[-1500:]
