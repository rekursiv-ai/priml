"""Tests for TMax checkout identity and import-root resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import subprocess

import pytest

from priml.baselines.tmax.scripts import upstream


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "tmax"
    (root / upstream.OPEN_INSTRUCT).mkdir(parents=True)
    return root


def test_resolve_checkout_requires_open_instruct(tmp_path: Path) -> None:
    # A checkout is recognized by the open-instruct tree inside it.
    with pytest.raises(FileNotFoundError, match="No TMax checkout"):
        upstream.resolve_checkout(tmp_path / "missing")
    root = tmp_path / "not-tmax"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="not a TMax checkout"):
        upstream.resolve_checkout(root)


def test_verify_checkout_accepts_only_the_pinned_commit(tmp_path: Path) -> None:
    # Only the pinned commit passes; another tree is a different experiment.
    root = _checkout(tmp_path)
    assert (
        upstream.verify_checkout(
            root,
            read_commit=lambda _root: upstream.UPSTREAM_COMMIT,
            read_status=lambda _root: "",
        )
        == root
    )
    with pytest.raises(ValueError, match="reproduces"):
        upstream.verify_checkout(
            root,
            read_commit=lambda _root: "deadbeef",
            read_status=lambda _root: "",
        )


def test_verify_checkout_refuses_a_dirty_pinned_tree(tmp_path: Path) -> None:
    # Local modifications disqualify even the right commit.
    root = _checkout(tmp_path)
    with pytest.raises(ValueError, match="local modifications"):
        upstream.verify_checkout(
            root,
            read_commit=lambda _root: upstream.UPSTREAM_COMMIT,
            read_status=lambda _root: " M open_instruct/grpo_fast.py",
        )


def test_open_instruct_is_inside_verified_checkout(tmp_path: Path) -> None:
    # The import root lives inside the verified checkout.
    root = _checkout(tmp_path)
    assert upstream.open_instruct(root) == root / upstream.OPEN_INSTRUCT


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603, S607 -- Fixed Git executable in a tmp repo.


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a one-commit repo under an isolated git config."""
    config = tmp_path / "gitconfig"
    config.write_text("[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "init")
    return root


def test_head_commit_reads_the_full_sha(git_repo: Path) -> None:
    commit = upstream.head_commit(git_repo)
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_working_tree_status_reports_untracked_and_modified_files(
    git_repo: Path,
) -> None:
    assert not upstream.working_tree_status(git_repo)
    (git_repo / "a.txt").write_text("b\n", encoding="utf-8")
    (git_repo / "new.txt").write_text("n\n", encoding="utf-8")
    status = upstream.working_tree_status(git_repo)
    assert "a.txt" in status
    assert "new.txt" in status


@pytest.mark.parametrize(
    "reader",
    [upstream.head_commit, upstream.working_tree_status],
)
def test_git_readers_refuse_a_non_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: Callable[[Path], str],
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"checkout|inspected"):
        reader(tmp_path)


@pytest.mark.parametrize(
    "reader",
    [upstream.head_commit, upstream.working_tree_status],
)
def test_git_readers_report_a_missing_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: Callable[[Path], str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="git is unavailable"):
        reader(tmp_path)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
