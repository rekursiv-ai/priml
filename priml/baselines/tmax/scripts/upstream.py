"""Locate and verify the pinned TMax checkout this baseline drives.

TMax owns the terminal environments, the vLLM engines, the tool parser, the
reward functions, and the Terminal-Bench evaluation. PriML drives that code
rather than reimplementing it, which makes the checkout a staged ASSET of the
experiment -- like the policy checkpoint and the rollout shards -- and not a
file-system coincidence.

So the location is configured and the identity is verified. There is no
guess from this file's own position: a sibling-directory default resolves to
whatever happens to sit beside the clone, silently picks up an unrelated or
differently-versioned tree, and cannot be reproduced on a cluster whose
layout differs. ``scripts/prepare_data.py`` stages the checkout at
:data:`DEFAULT_CHECKOUT` beside the other pinned assets; ``--upstream-root``
overrides it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import subprocess


if TYPE_CHECKING:
    from collections.abc import Callable


UPSTREAM_REPO: Final = "https://github.com/hamishivi/tmax"
"""The TMax repository this baseline reproduces."""

UPSTREAM_COMMIT: Final = "6d3d606c0f9de6ceec9481f9ccc793936818fef0"
"""The pinned TMax commit: every rollout and evaluation claim is against it."""

UPSTREAM_SHORT: Final = UPSTREAM_COMMIT[:7]
"""The short commit this package's documentation and manifests quote."""

DEFAULT_CHECKOUT: Final = Path("/opt/scratch/datasets/tmax/upstream/tmax")
"""Where ``prepare_data.py`` stages the checkout, beside the other assets."""

OPEN_INSTRUCT: Final = Path("training/open-instruct")
"""The in-repo Open-Instruct fork holding TMax's rollout runtime."""


def resolve_checkout(path: Path | str | None = None) -> Path:
    """Return a staged TMax checkout root, without verifying its commit.

    Args:
      path: Explicit checkout root; ``None`` takes :data:`DEFAULT_CHECKOUT`.

    Returns:
      root: The checkout root, which exists and carries TMax's runtime.

    Raises:
      FileNotFoundError: Nothing is staged there, or the directory is not a
        TMax checkout.

    """
    root = Path(path) if path is not None else DEFAULT_CHECKOUT
    if not root.is_dir():
        raise FileNotFoundError(
            f"No TMax checkout at {root}. Stage it with "
            "`python -m priml.baselines.tmax.scripts.prepare_data`, or pass "
            "--upstream-root; it is cloned from "
            f"{UPSTREAM_REPO} at {UPSTREAM_SHORT}.",
        )
    if not (root / OPEN_INSTRUCT).is_dir():
        raise FileNotFoundError(
            f"{root} is not a TMax checkout: it carries no {OPEN_INSTRUCT.as_posix()}.",
        )
    return root


def head_commit(root: Path) -> str:
    """Return the checkout's HEAD commit.

    Args:
      root: A git checkout root.

    Returns:
      commit: The full 40-character HEAD commit.

    Raises:
      FileNotFoundError: ``root`` is not a git checkout, or git is absent.

    """
    try:
        completed = subprocess.run(  # noqa: S603 -- Fixed Git executable.
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607 -- Fixed executable.
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise FileNotFoundError(
            f"Cannot read {root}'s commit: git is unavailable.",
        ) from error
    if completed.returncode:
        raise FileNotFoundError(
            f"{root} is not a git checkout, so its TMax commit cannot be "
            f"verified: {completed.stderr.strip()}",
        )
    return completed.stdout.strip()


def working_tree_status(root: Path) -> str:
    """Return porcelain status for tracked and untracked checkout changes."""
    try:
        completed = subprocess.run(  # noqa: S603 -- Fixed Git executable.
            [  # noqa: S607 -- Fixed executable.
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise FileNotFoundError(
            f"Cannot inspect {root}'s working tree: git is unavailable.",
        ) from error
    if completed.returncode:
        raise FileNotFoundError(
            f"{root}'s working tree cannot be inspected: {completed.stderr.strip()}",
        )
    return completed.stdout.strip()


def verify_checkout(
    path: Path | str | None = None,
    *,
    read_commit: Callable[[Path], str] = head_commit,
    read_status: Callable[[Path], str] = working_tree_status,
) -> Path:
    """Return the staged checkout root, refusing any commit but the pinned one.

    Called where the checkout is USED rather than where it is configured, so
    one unverified tree cannot produce rollouts or a score under this
    baseline's name.

    Args:
      path: Explicit checkout root; ``None`` takes :data:`DEFAULT_CHECKOUT`.
      read_commit: Reads a checkout's HEAD commit; injected by tests.
      read_status: Reads checkout modifications; injected by tests.

    Returns:
      root: The verified checkout root.

    Raises:
      FileNotFoundError: Nothing is staged there, or the commit is unreadable.
      ValueError: The checkout is at a different commit or has local changes.

    """
    root = resolve_checkout(path)
    commit = read_commit(root)
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"{root} is at TMax commit {commit or '(unknown)'}, but this "
            f"baseline reproduces {UPSTREAM_COMMIT}. Check out the pinned "
            "commit, or stage a fresh checkout with "
            "`python -m priml.baselines.tmax.scripts.prepare_data`.",
        )
    status = read_status(root)
    if status:
        preview = status.splitlines()[:5]
        raise ValueError(
            f"{root} is at the pinned commit but has local modifications: "
            f"{preview}. Use a clean checkout so the recorded commit identifies "
            "the code that runs.",
        )
    return root


def open_instruct(root: Path) -> Path:
    """Return the Open-Instruct import root inside a verified checkout.

    Args:
      root: A verified checkout root.

    Returns:
      path: The directory holding the importable ``open_instruct`` package.

    """
    return root / OPEN_INSTRUCT
