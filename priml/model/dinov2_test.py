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

    def load(repository: str, variant: str) -> nn.Module:
        del repository, variant
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


def test_dino_cost_rejects_unknown_variant() -> None:
    config = DinoV2Teacher.Config()
    config.variant = "tiny"
    with pytest.raises(ValueError, match="no cost model"):
        config.cost(batch_size=2, dtype=torch.float32)


def test_dino_teacher_resizes_position_and_clamps_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # DINO's learned position table is batch-one by production contract.
            self.pos_embed = nn.Parameter(torch.randn(1, 5, 3))
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

        def get_intermediate_layers(
            self,
            image: torch.Tensor,
            *,
            n: list[int],
            reshape: bool,
            return_class_token: bool,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            assert not reshape
            assert return_class_token
            return tuple(
                (
                    torch.full((image.shape[0], 4, 3), float(index)),
                    torch.full((image.shape[0], 3), float(index + 1)),
                )
                for index in n
            )

    def load(repository: str, variant: str) -> TinyEncoder:
        del repository, variant
        return TinyEncoder()

    monkeypatch.setattr(
        "priml.model.dinov2.load_torch_hub_distributed",
        load,
    )
    config = DinoV2Teacher.Config()
    config.image_size = 256
    config.layer_indices = (-3, 0, 9)
    teacher = DinoV2Teacher(config)
    assert teacher.encoder.pos_embed.shape == (1, 257, 3)
    # The teacher explicitly requires square image_size x image_size inputs.
    output = teacher(torch.zeros(2, 3, 256, 256, dtype=torch.uint8))
    assert len(output) == 3
    assert all(value.shape == (2, 5, 3) for value in output)
    assert torch.equal(output[0], output[1])
    assert not torch.equal(output[2], output[1])
    with pytest.raises(ValueError, match="image size"):
        # The invalid square is intentional: the guard compares both axes.
        teacher(torch.zeros(2, 3, 128, 128, dtype=torch.uint8))


def test_rank_zero_hub_failure_reaches_other_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 0
    status_message = None

    def load(repository: str, variant: str) -> nn.Module:
        del repository, variant
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
        RuntimeError,
        match="rank 0 could not load facebookresearch/dinov2",
    ):
        load_torch_hub_distributed("facebookresearch/dinov2", "dinov2_vitb14")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
