"""Tests for TMax checkout identity and import-root resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from priml.baselines.tmax.scripts import upstream


if TYPE_CHECKING:
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
