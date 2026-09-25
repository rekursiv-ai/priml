"""Shape, routing, and objective checks for the SpeedrunDiT backbone."""

# PyTorch optimizer parameter groups are dynamically typed by its stubs.
# pyright: reportAny=false

from __future__ import annotations

from typing import override

from torch import nn

import pytest
import torch

from priml.baselines.speedrundit.model import ModelOutput, SpeedrunDiT
from priml.baselines.speedrundit.objective import SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.baselines.speedrundit.sampling import sample_latents
from priml.math.diffusion.time_shift import time_shift
from priml.math.position_embedding import image_token_positions
from priml.model.attention.rope import RoPE
from priml.optimizers.muon import Muon


def tiny_model() -> SpeedrunDiT:
    config = SpeedrunDiT.Config(
        input_size=4,
        in_channels=2,
        patch_size=1,
        hidden_size=32,
        depth=6,
        num_heads=4,
        cls_channels=8,
        projector_hidden=16,
        projection_depths=(2, 3, 6),
        drop_ratio=0.5,
        path_drop_prob=0.0,
        class_dropout_prob=0.1,
    )
    return config.make()


def test_cost_counts_shared_projector_once() -> None:
    model = tiny_model()
    estimate = model.config.cost(batch_size=2, dtype=torch.float32)
    assert estimate.params == sum(parameter.numel() for parameter in model.parameters())
    assert estimate["flops", "primal", "matmul", torch.float32] > 0


def test_rope_leaves_cls_untouched_and_uses_original_positions() -> None:
    rope = RoPE.Config(channels_head=(4, 4)).make()
    positions = image_token_positions(4, torch.device("cpu")).expand(2, -1, -1)
    kept = torch.tensor([[0, 6, 3], [0, 7, 2]])
    selected = positions.gather(1, kept[..., None].expand(-1, -1, 2))
    assert selected[0].tolist() == [[0, 0], [1, 1], [0, 2]]
    x = torch.randn(2, 3, 4, 8)
    full_cos, full_sin = rope(image_token_positions(4, torch.device("cpu")))
    cos, sin = (
        factor.expand(2, -1, -1, -1).gather(
            1,
            kept[:, :, None, None].expand(-1, -1, 1, factor.shape[-1]),
        )
        for factor in (full_cos, full_sin)
    )
    expected_cos, expected_sin = rope(selected)
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)
    rotated, _ = RoPE.rotate(x, x, cos, sin, interleave=True)
    assert torch.equal(rotated[:, 0], x[:, 0])
    assert not torch.equal(rotated[:, 1], x[:, 1])
    single_cos, single_sin = rope(selected[0:1, 1:2])
    single, _ = RoPE.rotate(
        x[0:1, 1:2],
        x[0:1, 1:2],
        single_cos,
        single_sin,
        interleave=True,
    )
    assert torch.equal(rotated[0:1, 1:2], single)


def test_training_routing_alignment_and_backward() -> None:
    model = tiny_model().train()
    image = torch.randn(2, 2, 4, 4)
    labels = torch.tensor([1, 2])
    teacher = tuple(torch.randn(2, 17, 8) for _ in range(3))
    objective = SpeedrunObjective(shift_time=False)
    terms = objective(model, image, labels, teacher, time=torch.full((2,), 0.5))
    assert terms.loss.shape == (2,)
    assert terms.output.velocity.shape == image.shape
    assert terms.output.cls_velocity.shape == (2, 8)
    assert [p.tokens.shape[1] for p in terms.output.projections] == [17, 8, 17]
    kept = terms.output.projections[1].ids_keep
    assert kept is not None
    assert kept.shape == (2, 8)
    terms.loss.mean().backward()
    assert model.final_layer.linear.weight.grad is not None
    assert model.projector[0].weight.grad is not None


def test_position_table_promotes_bfloat16_tokens_to_float32() -> None:
    model = tiny_model().eval()
    model.wg_norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
    captured = None

    class StopForwardError(Exception):
        pass

    def capture_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal captured
        captured = inputs[0]
        raise StopForwardError

    model.blocks[0].register_forward_pre_hook(capture_input)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(StopForwardError):
        model(
            torch.randn(1, 2, 4, 4),
            torch.full((1,), 0.5),
            torch.tensor([1]),
            torch.randn(1, 8),
            route_tokens=False,
        )
    assert captured is not None
    assert captured.dtype == torch.float32


def test_optimizer_partitions_hidden_matrices_from_heads() -> None:
    model = tiny_model()
    optimizer = speedrundit_optimizer().make()(model)
    assert len(optimizer.optimizers) == 2
    adam, muon = optimizer.optimizers
    assert isinstance(muon, Muon)
    assert any(
        p is model.final_layer.linear.weight for p in adam.param_groups[0]["params"]
    )
    assert any(
        p is model.blocks[0].attn.qkv.weight for p in muon.param_groups[0]["params"]
    )
    assert any(p is model.projector[0].weight for p in muon.param_groups[0]["params"])


def test_shift_and_sampler_return_expected_latent_shapes() -> None:
    assert torch.allclose(
        time_shift(torch.tensor([0.0, 1.0]), 8192),
        torch.tensor([0.0, 1.0]),
    )
    model = tiny_model().eval()
    latents = torch.randn(2, 2, 4, 4)
    cls = torch.randn(2, 8)
    sampled, sampled_cls = sample_latents(
        model,
        latents,
        cls,
        torch.tensor([1, 2]),
        num_steps=3,
        shift_time=False,
    )
    assert sampled.shape == latents.shape
    assert sampled_cls.shape == cls.shape
    assert torch.isfinite(sampled).all()
    # The source model includes a null class when guidance dropout is enabled.
    guided, _ = sample_latents(
        model,
        latents,
        cls,
        torch.tensor([1, 2]),
        num_steps=2,
        cfg_scale=2,
        shift_time=False,
    )
    assert torch.isfinite(guided).all()


def test_zero_cls_guidance_keeps_conditional_cls_drift() -> None:
    class ConstantVelocityModel(nn.Module):
        config = type("Config", (), {"num_classes": 2})()

        @override
        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            cls: torch.Tensor,
            **_kwargs: object,
        ) -> ModelOutput:
            del t
            cls_velocity = torch.where(y[:, None] == 2, -1.0, 1.0).expand_as(cls)
            return ModelOutput(torch.zeros_like(x), cls_velocity, ())

    model = ConstantVelocityModel()
    latents = torch.zeros(1, 2, 4, 4)
    cls = torch.zeros(1, 8)
    labels = torch.tensor([1])
    torch.manual_seed(7)
    _, conditional_cls = sample_latents(
        model,  # pyright: ignore[reportArgumentType] -- model API test double
        latents,
        cls,
        labels,
        num_steps=2,
        shift_time=False,
    )
    torch.manual_seed(7)
    _, zero_guidance_cls = sample_latents(
        model,  # pyright: ignore[reportArgumentType] -- model API test double
        latents,
        cls,
        labels,
        num_steps=2,
        cfg_scale=2,
        cls_cfg_scale=0,
        shift_time=False,
    )
    assert torch.equal(zero_guidance_cls, conditional_cls)
