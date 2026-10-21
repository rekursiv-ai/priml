"""Check that CUDA housekeeping runs on CUDA devices and only there."""

from __future__ import annotations

import gc

import pytest
import torch

from priml.baselines.rsna_knee.device import release_memory, synchronize


@pytest.mark.parametrize("name", ["cpu", "cuda"])
def test_synchronize_waits_only_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    waited: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", waited.append)
    synchronize(torch.device(name))
    assert waited == ([torch.device("cuda")] if name == "cuda" else [])


@pytest.mark.parametrize("name", ["cpu", "cuda"])
def test_release_memory_collects_and_empties_the_cache_only_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc") or 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("cache"))
    release_memory(torch.device(name))
    assert events == (["gc", "cache"] if name == "cuda" else [])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
