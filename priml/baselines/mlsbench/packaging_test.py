"""Packaging: the data files the harness reads at runtime must ship in the wheel.

The fast tests read ``pyproject.toml`` (never modified here) and emulate
hatchling's wheel ``exclude`` globs. The slow test really builds a wheel and
sdist with ``uv build`` into a temp dir (needs ``uv``; hatchling may be
fetched from the network), so it is opt-in via the ``integration`` marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import json
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

import pytest


_PKG: Final = Path(__file__).resolve().parent
_REPO: Final = _PKG.parents[2]
_PYPROJECT: Final = _REPO / "pyproject.toml"
_REL: Final = "priml/baselines/mlsbench"

# Files read at import/runtime (test_runtime_data_matches_source_scan derives this).
_RUNTIME_DATA: Final = (
    "eval_script_pins.json",
    "registry.json",
    "testdata/image_lock.json",
    "testdata/image_platforms.json",
)


def _glob_re(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob (``**/``, ``*``) to a regex."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + "$")


def _config(*keys: str) -> object:
    """Return the ``pyproject.toml`` value at the nested table ``keys``."""
    value: object = tomllib.loads(_PYPROJECT.read_text())
    for key in keys:
        value = cast("dict[str, object]", value)[key]
    return value


def _wheel_cfg(key: str) -> list[str]:
    return cast("list[str]", _config("tool", "hatch", "build", "targets", "wheel", key))


def _data_files() -> list[Path]:
    return sorted(
        p
        for p in _PKG.rglob("*")
        if p.is_file()
        and p.suffix not in {".py", ".pyc"}
        and "__pycache__" not in p.parts
    )


def test_backend_is_hatchling_with_package_root() -> None:
    assert _config("build-system", "build-backend") == "hatchling.build"
    assert "priml" in _wheel_cfg("packages")


def test_wheel_excludes_only_tests_not_data() -> None:
    excludes = [_glob_re(e) for e in _wheel_cfg("exclude")]
    data = _data_files()
    assert data, "no data files found"
    for f in data:
        rel = f"priml/{f.relative_to(_PKG.parent.parent).as_posix()}"
        assert not any(e.match(rel) for e in excludes), f"wheel excludes {rel}"
    # sanity: the exclude patterns do drop this very test file
    assert any(e.match(f"{_REL}/packaging_test.py") for e in excludes)


def test_runtime_data_files_exist_and_are_valid_json() -> None:
    for rel in _RUNTIME_DATA:
        path = _PKG / rel
        assert path.is_file(), rel
        assert isinstance(json.loads(path.read_text()), dict)


def test_runtime_data_matches_source_scan() -> None:
    """Every packaged data file that non-test sources name is in ``_RUNTIME_DATA``."""
    text = "\n".join(
        p.read_text()
        for p in _PKG.rglob("*.py")
        if not p.name.endswith("_test.py")
        and "testdata" not in p.parts
        and p.name != "testing.py"
    )
    named = {
        m.group(0) for m in re.finditer(r"[\w./-]+\.json\b", text.replace("``", " "))
    }
    named = {Path(n).name for n in named}
    found = sorted(
        f.relative_to(_PKG).as_posix()
        for f in _data_files()
        if f.name in named and f.suffix == ".json"
    )
    assert found == sorted(_RUNTIME_DATA), (found, sorted(_RUNTIME_DATA))


def test_data_files_not_gitignored() -> None:
    git = shutil.which("git")
    if git is None or not (_REPO / ".git").exists():
        pytest.skip("no git checkout")
    rels = [str(f.relative_to(_REPO)) for f in _data_files()]
    r = subprocess.run(  # noqa: S603
        [git, "check-ignore", "--", *rels],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    # hatchling honours .gitignore: an ignored file would be dropped from builds.
    assert r.returncode == 1, f"ignored data files:\n{r.stdout}"


@pytest.mark.integration
@pytest.mark.slow
def test_real_build_contains_data_and_omits_tests(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    r = subprocess.run(  # noqa: S603
        [uv, "build", "--out-dir", str(tmp_path)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        pytest.skip(f"uv build unavailable (offline?): {r.stderr[-300:]}")
    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))
    names = set(zipfile.ZipFile(wheel).namelist())
    for f in _data_files():
        rel = f"priml/{f.relative_to(_PKG.parent.parent).as_posix()}"
        assert rel in names, f"missing from wheel: {rel}"
    for rel in _RUNTIME_DATA:
        assert f"{_REL}/{rel}" in names, f"runtime file missing from wheel: {rel}"
    assert not any(n.endswith("_test.py") and _REL in n for n in names)
    with tarfile.open(sdist) as t:
        snames = t.getnames()
    for rel in _RUNTIME_DATA:
        assert any(n.endswith(f"{_REL}/{rel}") for n in snames), rel
