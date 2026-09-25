"""Distributed Torch Hub cache ordering for the DINOv2 teacher."""

from __future__ import annotations

from torch import nn

import pytest
import torch

from priml.hub import load_torch_hub_distributed
from priml.model.dinov2 import DinoV2Teacher


def test_frozen_teacher_cost_has_no_backward_work() -> None:
    estimate = DinoV2Teacher.Config().cost(batch_size=2, dtype=torch.float32)
    assert estimate["flops", "primal", "matmul", torch.float32] > 0
    assert estimate["flops", "adjoint", "matmul", torch.float32] == 0


def test_rank_zero_populates_hub_before_other_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rank = 0
    encoder = nn.Identity()

    def load(_repository: str, _variant: str) -> nn.Module:
        events.append(f"load:{rank}")
        return encoder

    def broadcast(status: list[str | None], *, src: int) -> None:
        assert src == 0
        assert status == [None]
        events.append(f"broadcast:{rank}")

    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    assert (
        load_torch_hub_distributed("facebookresearch/dinov2", "dinov2_vitb14")
        is encoder
    )
    rank = 1
    assert (
        load_torch_hub_distributed("facebookresearch/dinov2", "dinov2_vitb14")
        is encoder
    )
    assert events == ["load:0", "broadcast:0", "broadcast:1", "load:1"]


def test_rank_zero_hub_failure_reaches_other_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 0
    status_message = None

    def load(_repository: str, _variant: str) -> nn.Module:
        raise OSError("hub unavailable")

    def broadcast(status: list[str | None], *, src: int) -> None:
        nonlocal status_message
        assert src == 0
        if rank == 0:
            status_message = status[0]
        else:
            status[0] = status_message

    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    with pytest.raises(OSError, match="hub unavailable"):
        load_torch_hub_distributed("facebookresearch/dinov2", "dinov2_vitb14")
    rank = 1
    with pytest.raises(
        RuntimeError, match="rank 0 could not load facebookresearch/dinov2"
    ):
        load_torch_hub_distributed("facebookresearch/dinov2", "dinov2_vitb14")
