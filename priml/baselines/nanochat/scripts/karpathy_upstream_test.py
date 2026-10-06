"""Tests for the pinned upstream reference shared by the comparison scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import subprocess

from torch import Tensor

import pytest
import torch

from priml.baselines.nanochat.scripts import karpathy_upstream
from priml.model.attention.value_gated_attention import sdpa_attention


if TYPE_CHECKING:
    from pathlib import Path


def _git_run(
    root: Path,
    *,
    head: str,
    status: str = "",
    calls: list[tuple[list[str], Path | None, bool, bool, bool]],
) -> object:
    """Return a ``subprocess.run`` fake that records calls and creates the clone."""

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, cwd, check, capture_output, text))
        if arguments[1] == "clone":
            (root / ".git").mkdir(parents=True)
        output = {"rev-parse": head, "status": status}.get(arguments[1], "")
        return subprocess.CompletedProcess(arguments, 0, output, "")

    return run


def test_clone_upstream_clones_checks_out_and_verifies_the_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outer" / "nested" / "reference"
    commit = karpathy_upstream.UPSTREAM_COMMIT
    calls: list[tuple[list[str], Path | None, bool, bool, bool]] = []
    monkeypatch.setattr(subprocess, "run", _git_run(root, head=commit, calls=calls))
    assert karpathy_upstream.clone_upstream(root) == root
    assert calls == [
        (
            ["git", "clone", "--quiet", karpathy_upstream.UPSTREAM_URL, str(root)],
            None,
            True,
            False,
            False,
        ),
        (["git", "checkout", "--quiet", commit], root, True, False, False),
        (["git", "rev-parse", "HEAD"], root, True, True, True),
        (["git", "status", "--porcelain"], root, True, True, True),
    ]


def test_clone_upstream_reuses_a_clean_existing_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "existing"
    (root / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], Path | None, bool, bool, bool]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _git_run(root, head=karpathy_upstream.UPSTREAM_COMMIT, calls=calls),
    )
    assert karpathy_upstream.clone_upstream(root) == root
    assert [call[0][1] for call in calls] == ["rev-parse", "status"]


def test_clone_upstream_rejects_wrong_revision_and_dirty_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "existing"
    (root / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], Path | None, bool, bool, bool]] = []
    monkeypatch.setattr(subprocess, "run", _git_run(root, head="other", calls=calls))
    with pytest.raises(RuntimeError, match=r"^clone is at other, expected pinned$"):
        karpathy_upstream.clone_upstream(root, commit="pinned")

    monkeypatch.setattr(
        subprocess,
        "run",
        _git_run(root, head="pinned", status="M train.py", calls=calls),
    )
    with pytest.raises(
        RuntimeError,
        match=r"^clone has local modifications:\nM train.py$",
    ):
        karpathy_upstream.clone_upstream(root, commit="pinned")


class _FlashAttentionFunction(Protocol):
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> Tensor: ...


class _FlashAttentionInterface(Protocol):
    flash_attn_func: _FlashAttentionFunction


class _KernelAdapter(Protocol):
    flash_attn_interface: _FlashAttentionInterface


@runtime_checkable
class _KernelRegistry(Protocol):
    def get_kernel(self, name: str) -> _KernelAdapter: ...


def test_kernel_stub_hands_their_fa3_call_the_portable_kernel() -> None:
    kernel = karpathy_upstream.kernels_stub()
    assert kernel.__name__ == "kernels"
    assert isinstance(kernel, _KernelRegistry)
    attention = kernel.get_kernel("flash-attention-3").flash_attn_interface
    assert attention.flash_attn_func is karpathy_upstream.their_attention
    with pytest.raises(ValueError, match=r"^other$"):
        kernel.get_kernel("other")


def test_their_attention_is_the_windowed_sdpa_and_rejects_other_masks() -> None:
    q = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    k, v = q.flip(1), q.flip(-1)
    assert torch.equal(
        karpathy_upstream.their_attention(q, k, v, causal=True, window_size=(1, 0)),
        sdpa_attention(q, k, v, window=1),
    )
    with pytest.raises(ValueError, match=r"^the recipe attends causally$"):
        karpathy_upstream.their_attention(q, k, v, causal=False, window_size=(1, 0))
    with pytest.raises(ValueError, match=r"^unexpected future window 1$"):
        karpathy_upstream.their_attention(q, k, v, causal=True, window_size=(1, 1))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
