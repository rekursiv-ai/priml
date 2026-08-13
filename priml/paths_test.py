"""Tests for config-time path composition and write-time destination checks."""

from __future__ import annotations

from pathlib import Path

import os
import subprocess
import sys

import pytest

from priml.paths import resolve_working_dir, validated_output_path


def test_resolve_working_dir_without_base_is_its_own_root() -> None:
    assert resolve_working_dir(None, "/checkpoints") == Path("/checkpoints")


def test_resolve_working_dir_nests_absolute_working_dir_under_base() -> None:
    """An absolute ``working_dir`` must still land beneath ``base_dir``.

    ``Path("/a") / "/b" == Path("/b")`` -- POSIX makes an absolute right
    operand discard the base. Without the leading-slash strip, a logical
    root like ``"/checkpoints"`` would silently escape its owner's
    ``base_dir`` and write to the filesystem root.
    """
    assert resolve_working_dir("/base", "/checkpoints") == Path("/base/checkpoints")


def test_resolve_working_dir_accepts_relative_working_dir() -> None:
    assert resolve_working_dir("/base", "runs/exp1") == Path("/base/runs/exp1")


def test_composing_a_path_does_not_load_torch() -> None:
    """Config-time resolution must stay usable in torch-free processes.

    This module is imported by render scripts and data configs that never
    train; a module-top torch import would add multi-second startup to every
    one of them.

    The probe resolves the package by its own import name rather than a
    ``parents[N]`` walk: this file ships to the standalone export, where the
    package sits at a different depth and a hardcoded walk points outside it.
    """
    module = resolve_working_dir.__module__
    package = __import__(module.split(".", 1)[0])
    assert package.__file__ is not None
    source = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "from MODULE import resolve_working_dir; "
        "resolve_working_dir('/opt/scratch', '/datasets/probe'); "
        "assert 'torch' not in sys.modules"
    ).replace("MODULE", module)
    probe = subprocess.run(  # noqa: S603 -- argv is this module's own import path.
        [sys.executable, "-c", source],
        cwd=Path(package.__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_validated_output_path_expands_tilde(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A tilde argument names the home directory, not a literal ``~`` dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    assert validated_output_path("~/artifacts/x.json") == home / "artifacts" / "x.json"


def test_validated_output_path_returns_absolute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The caller writes to the RETURN value, so it must be canonical."""
    monkeypatch.chdir(tmp_path)

    assert (
        validated_output_path("artifacts/x.json") == tmp_path / "artifacts" / "x.json"
    )


def test_validated_output_path_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="must not be root"):
        validated_output_path(Path("/"))


def test_validated_output_path_rejects_non_normalized_path() -> None:
    with pytest.raises(ValueError, match="must be normalized"):
        validated_output_path("/opt/scratch/../repo/output.json")


def test_validated_output_path_rejects_symlink_resolving_to_root(
    tmp_path: Path,
) -> None:
    """A link to ``/`` is lexically fine but names the filesystem root."""
    link = tmp_path / "root-link"
    link.symlink_to("/", target_is_directory=True)

    with pytest.raises(ValueError, match="must not be root"):
        validated_output_path(link)


def test_validated_output_path_allows_a_checkout_path(tmp_path: Path) -> None:
    """Version control is .gitignore's concern, not this function's."""
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    output = checkout / "artifacts" / "x.json"

    assert validated_output_path(output) == output


def test_validated_output_path_rejects_a_resolved_alias(tmp_path: Path) -> None:
    """The obvious alias: the destination names a protected input outright."""
    protected = tmp_path / "input.npz"
    protected.write_bytes(b"payload")

    with pytest.raises(ValueError, match="aliases protected input"):
        validated_output_path(protected, protected=[protected])


def test_validated_output_path_rejects_a_symlinked_alias(tmp_path: Path) -> None:
    protected = tmp_path / "input.npz"
    protected.write_bytes(b"payload")
    link = tmp_path / "alias.npz"
    link.symlink_to(protected)

    with pytest.raises(ValueError, match="aliases protected input"):
        validated_output_path(link, protected=[protected])


def test_validated_output_path_rejects_a_hardlinked_alias(tmp_path: Path) -> None:
    """A hard link resolves to a distinct name yet shares the inode."""
    protected = tmp_path / "input.npz"
    protected.write_bytes(b"payload")
    alias = tmp_path / "hardlink.npz"
    os.link(protected, alias)

    with pytest.raises(ValueError, match="aliases protected input"):
        validated_output_path(alias, protected=[protected])


def test_validated_output_path_compares_tilde_spellings_after_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guarding the raw argument misses every ``~``-spelled alias.

    ``Path("~/x").exists()`` is False, so an unexpanded destination silently
    skips the inode check while an unexpanded protected entry never matches
    by path either.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    protected = home / "input.npz"
    protected.write_bytes(b"payload")
    alias = home / "hardlink.npz"
    os.link(protected, alias)

    with pytest.raises(ValueError, match="aliases protected input"):
        validated_output_path("~/hardlink.npz", protected=[Path("~/input.npz")])


def test_validated_output_path_allows_a_fresh_destination(tmp_path: Path) -> None:
    protected = tmp_path / "input.npz"
    protected.write_bytes(b"payload")
    output = tmp_path / "report.json"

    assert validated_output_path(output, protected=[protected]) == output


def test_callable_protected_is_evaluated_after_destination_validation(
    tmp_path: Path,
) -> None:
    """An unusable destination fails before a run reads its inputs.

    Deriving the protected set means reading the artifacts; that work must not
    happen for a path the checks would reject anyway.
    """
    del tmp_path
    calls: list[int] = []

    def protected() -> list[Path]:
        calls.append(1)
        return []

    with pytest.raises(ValueError, match="must be normalized"):
        validated_output_path("/opt/scratch/../x/score.json", protected=protected)

    assert not calls


def test_callable_protected_still_guards(tmp_path: Path) -> None:
    """A deferred protected set is enforced exactly like an eager one."""
    protected = tmp_path / "input.npz"
    protected.write_bytes(b"payload")

    with pytest.raises(ValueError, match="aliases protected input"):
        validated_output_path(protected, protected=lambda: [protected])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
