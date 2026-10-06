"""The pinned karpathy/autoresearch reference and the kernel handed to it.

Shared by every script that runs the reference beside this package
(``karpathy_parity``, ``karpathy_data_parity``, ``karpathy_race``), so all
three name the same commit and substitute the same attention kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import subprocess
import types

from priml.model.attention.value_gated_attention import sdpa_attention


if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


UPSTREAM_URL: Final = "https://github.com/karpathy/autoresearch.git"
"""Repository the reference is cloned from."""

UPSTREAM_COMMIT: Final = "b11d6f283f866eb7e10fb776a4b8553fef873fd5"
"""Revision every comparison is against."""


def clone_upstream(
    root: Path,
    *,
    url: str = UPSTREAM_URL,
    commit: str = UPSTREAM_COMMIT,
) -> Path:
    """Clone the reference at its pinned commit, or verify an existing clone.

    Args:
      root: Directory the clone lives in.
      url: Repository to clone.
      commit: Revision the comparison is against.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: An existing clone is dirty or at another commit, so what
        it contains is no longer the reference this comparison names.

    """
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603 -- Fixed git argv; url and root come from this script's own flags.
            ["git", "clone", "--quiet", url, str(root)],  # noqa: S607 -- git is resolved from PATH like every developer tool.
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed git argv; the commit is the pinned constant.
            ["git", "checkout", "--quiet", commit],  # noqa: S607 -- git is resolved from PATH like every developer tool.
            cwd=root,
            check=True,
        )
    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"clone is at {head}, expected {commit}")
    if dirty := _git(root, "status", "--porcelain"):
        raise RuntimeError(f"clone has local modifications:\n{dirty}")
    return root


def their_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> Tensor:
    """``exp001``'s own kernel, behind FlashAttention-3's call signature.

    The reference calls ``fa3.flash_attn_func(q, k, v, causal=True,
    window_size=(w, 0))``; this hands that call to the very function
    ``exp001`` puts in its kernel slot, so both sides issue ONE kernel and
    what remains between them is the recipe rather than the backend.

    Args:
      q: ``[B, S, heads, channels_head]`` queries.
      k: Keys, same shape.
      v: Values, same shape.
      causal: Whether the mask is causal; the recipe always passes True.
      window_size: ``(history, future)``; the recipe always passes future 0.

    Returns:
      out: Attention output, same shape as ``q``.

    """
    if not causal:
        raise ValueError("the recipe attends causally")
    if window_size[1] != 0:
        raise ValueError(f"unexpected future window {window_size[1]}")
    return sdpa_attention(q, k, v, window=window_size[0])


def kernels_stub() -> types.ModuleType:
    """Return a ``kernels`` module whose ``get_kernel`` yields :func:`their_attention`.

    Their ``train.py`` imports FlashAttention-3 through ``kernels``, which
    builds only for SM90; installing this in ``sys.modules`` lets it run on
    any card.

    Returns:
      module: The stub, ready for ``sys.modules["kernels"]``.

    """
    module = types.ModuleType("kernels")
    module.__dict__["get_kernel"] = _get_kernel
    return module


def _get_kernel(name: str) -> types.SimpleNamespace:
    """Resolve their FlashAttention-3 request to the portable kernel."""
    if "flash-attention-3" not in name:
        raise ValueError(name)
    return types.SimpleNamespace(
        flash_attn_interface=types.SimpleNamespace(flash_attn_func=their_attention),
    )


def _git(root: Path, *arguments: str) -> str:
    """Run a read-only git command in the clone."""
    return subprocess.run(  # noqa: S603 -- Read-only git subcommands chosen by this module, no shell.
        ["git", *arguments],  # noqa: S607 -- git is resolved from PATH like every developer tool.
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
