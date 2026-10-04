"""Shape, routing, and objective checks for the SpeedrunDiT backbone."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast, override

import math

from torch import Tensor, nn

import pytest
import torch
import torch.distributed as dist

from priml.baselines.speedrundit.model import (
    ModelOutput,
    SpeedrunDiT,
    _linear_cost,
)
from priml.baselines.speedrundit.objective import SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.baselines.speedrundit.sampling import sample_latents
from priml.math.diffusion.time_shift import time_shift
from priml.math.position_embedding import image_token_positions
from priml.model.attention.rope import RoPE
from priml.optimizers.muon import Muon
from priml.train.parallelism import materialize_meta


if TYPE_CHECKING:
    from pathlib import Path


def tiny_model() -> SpeedrunDiT:
    config = SpeedrunDiT.Config(
        input_size=4,
        in_channels=2,
        hidden_size=32,
        depth=6,
        num_heads=4,
        cls_channels=8,
        projector_hidden=16,
        projection_depths=(2, 3, 6),
        drop_ratio=0.5,
        path_drop_prob=0.0,
    )
    return config.make()


def test_linear_cost_preserves_rows_dtype_and_bias() -> None:
    estimate = _linear_cost(3, channels_out=5, rows=2, dtype=torch.float64)
    assert estimate.params == 20
    assert estimate.params_active == 20
    assert estimate["flops", "primal", "matmul", torch.float64] == 60
    assert estimate["flops", "primal", "elementwise", torch.float64] == 10
    assert estimate["bytes", "primal", "matmul", torch.float64] == 248


def test_cost_counts_shared_projector_once() -> None:
    model = tiny_model()
    estimate = model.config.cost(batch_size=2, dtype=torch.float32)
    assert estimate.params == sum(parameter.numel() for parameter in model.parameters())
    assert estimate["flops", "primal", "matmul", torch.float32] > 0


@pytest.mark.parametrize(
    "overrides",
    [{}, {"drop_ratio": 0.0, "path_drop_prob": 0.0}, {"path_drop_prob": 1.0}],
)
def test_every_trainable_parameter_receives_a_gradient(
    overrides: dict[str, object],
) -> None:
    """Replicated ranks desync on a trainable parameter whose gradient never arrives."""
    config = tiny_model().config.copy_tree()
    for name, value in overrides.items():
        setattr(config, name, value)
    model = config.make().train()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.2 * torch.randn(parameter.shape, generator=generator))
    output = model(
        torch.randn(2, 2, 4, 4, generator=generator),
        t=torch.rand(2, generator=generator),
        y=torch.tensor([0, 1]),
        cls_token=torch.randn(2, 8, generator=generator),
    )
    aligned = torch.stack([p.tokens.square().mean() for p in output.projections])
    loss = output.velocity.square().mean() + output.cls_velocity.square().mean()
    (loss + aligned.sum()).backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not unused


def test_cost_counts_the_projector_even_when_no_depth_projects() -> None:
    config = tiny_model().config.copy_tree()
    config.projection_depths = ()
    model = config.make()
    estimate = config.cost(batch_size=2, dtype=torch.float32)
    assert estimate.params == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("reference_rope", [True, False])
def test_meta_construction_materializes_to_the_eager_constants(
    reference_rope: bool,
) -> None:
    """Every deterministic tensor matches an eager build; random ones are finite."""
    config = tiny_model().config.copy_tree()
    config.reference_rope = reference_rope
    eager = config.make()
    with torch.device("meta"):
        model = config.make()
    materialize_meta(model, device=torch.device("cpu"))
    random = {
        name
        for name, module in eager.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding))
        for name in (f"{name}.weight", f"{name}.bias")
    }
    for name, tensor in model.state_dict(keep_vars=False).items():
        assert bool(torch.isfinite(tensor).all()), name
        if name not in random:
            torch.testing.assert_close(tensor, eager.state_dict()[name], msg=name)


def test_rope_leaves_cls_untouched_and_uses_original_positions() -> None:
    rope = RoPE.Config(channels_head=(4, 4)).make()
    positions = image_token_positions(4, device=torch.device("cpu")).expand(2, -1, -1)
    kept = torch.tensor([[0, 6, 3], [0, 7, 2]])
    selected = positions.gather(1, kept[..., None].expand(-1, -1, 2))
    assert selected[0].tolist() == [[0, 0], [1, 1], [0, 2]]
    x = torch.randn(2, 3, 4, 8)
    full_cos, full_sin = rope(image_token_positions(4, device=torch.device("cpu")))
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
    rotated, _ = RoPE.rotate(x, k=x, cos=cos, sin=sin, interleave=True)
    assert torch.equal(rotated[:, 0], x[:, 0])
    assert not torch.equal(rotated[:, 1], x[:, 1])
    single_cos, single_sin = rope(selected[0:1, 1:2])
    single, _ = RoPE.rotate(
        x[0:1, 1:2],
        k=x[0:1, 1:2],
        cos=single_cos,
        sin=single_sin,
        interleave=True,
    )
    assert torch.equal(rotated[0:1, 1:2], single)


def test_training_routing_alignment_and_backward() -> None:
    model = tiny_model().train()
    image = torch.randn(3, 2, 4, 4)
    labels = torch.tensor([1, 2, 3])
    teacher = tuple(torch.randn(3, 17, 8) for _ in range(3))
    objective = SpeedrunObjective(shift_time=False)
    terms = objective(
        model,
        latents=image,
        labels=labels,
        teacher_features=teacher,
        time=torch.full((3,), 0.5),
    )
    assert terms.loss.shape == (3,)
    assert terms.output.velocity.shape == image.shape
    assert terms.output.cls_velocity.shape == (3, 8)
    assert [p.tokens.shape[1] for p in terms.output.projections] == [17, 8, 17]
    kept = terms.output.projections[1].ids_keep
    assert kept is not None
    assert kept.shape == (3, 8)
    terms.loss.mean().backward()
    assert model.final_layer.linear.weight.grad is not None
    assert model.projector[0].weight.grad is not None


def test_position_table_promotes_bfloat16_tokens_to_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model().eval()
    monkeypatch.setattr(model, "wg_norm", nn.Identity())
    model.blocks[0].register_forward_pre_hook(_stop_forward)
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(_StopForwardError) as stopped,
    ):
        model(
            torch.randn(2, 2, 4, 4),
            t=torch.full((1,), 0.5),
            y=torch.tensor([1]),
            cls_token=torch.randn(2, 8),
            route_tokens=False,
        )
    assert stopped.value.tokens.dtype == torch.float32


class _StopForwardError(Exception):
    """Carries the first block's input out of the aborted forward."""

    def __init__(self, tokens: Tensor) -> None:
        super().__init__()
        self.tokens = tokens


def _stop_forward(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
    del module
    raise _StopForwardError(inputs[0])


def test_optimizer_partitions_hidden_matrices_from_heads() -> None:
    model = tiny_model()
    optimizer = speedrundit_optimizer().make()(model)
    assert len(optimizer.optimizers) == 2
    adam, muon = optimizer.optimizers
    assert isinstance(muon, Muon)
    adam_params = cast("list[Tensor]", adam.param_groups[0]["params"])
    muon_params = cast("list[Tensor]", muon.param_groups[0]["params"])
    assert any(p is model.final_layer.linear.weight for p in adam_params)
    assert any(p is model.blocks[0].attn.qkv.weight for p in muon_params)
    assert any(p is model.projector[0].weight for p in muon_params)


def test_shift_and_sampler_return_expected_latent_shapes() -> None:
    assert torch.allclose(
        time_shift(torch.tensor([0.0, 1.0]), latent_dimensions=8192),
        torch.tensor([0.0, 1.0]),
    )
    model = tiny_model().eval()
    latents = torch.randn(3, 2, 4, 4)
    cls = torch.randn(3, 8)
    sampled, sampled_cls = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=torch.tensor([1, 2, 3]),
        num_steps=3,
        shift_time=False,
    )
    assert sampled.shape == latents.shape
    assert sampled_cls.shape == cls.shape
    assert torch.isfinite(sampled).all()
    # The source model includes a null class when guidance dropout is enabled.
    guided, _ = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=torch.tensor([1, 2, 3]),
        num_steps=2,
        cfg_scale=2,
        shift_time=False,
    )
    assert torch.isfinite(guided).all()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"encoder_blocks": 4, "decoder_blocks": 3}, "shorter than SPRINT"),
        ({"projection_depths": (3, 2)}, "strictly increasing"),
        ({"projection_depths": (2, 7)}, "outside the model"),
        ({"patch_size": 3}, "divisible by patch_size"),
        ({"encoder_blocks": 0}, "encoder_blocks"),
        ({"drop_ratio": -0.1}, "drop_ratio"),
        ({"drop_ratio": 1.0}, "drop_ratio"),
        ({"drop_ratio": math.nan}, "drop_ratio"),
        ({"path_drop_prob": -0.2}, "path_drop_prob"),
        ({"path_drop_prob": 1.5}, "path_drop_prob"),
        ({"path_drop_prob": math.nan}, "path_drop_prob"),
    ],
)
def test_inconsistent_geometry_is_rejected(
    overrides: dict[str, object],
    match: str,
) -> None:
    config = tiny_model().config.copy_tree()
    for name, value in overrides.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match=match):
        config.make()


def test_forward_rejects_a_latent_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="latent shape does not match"):
        tiny_model()(
            torch.randn(3, 2, 4, 5),
            t=torch.rand(3),
            y=torch.tensor([0, 1, 2]),
            cls_token=torch.randn(3, 8),
        )


def test_forward_rejects_a_cls_token_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match=r"cls_token must have shape"):
        tiny_model()(
            torch.randn(3, 2, 4, 4),
            t=torch.rand(3),
            y=torch.tensor([0, 1, 2]),
            cls_token=torch.randn(3, 5),
        )


def test_computed_rope_matches_the_reference_buffers() -> None:
    """Weights are randomized: zero-initialized heads would hide attention entirely."""
    reference = tiny_model().eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(0.2 * torch.randn(parameter.shape, generator=generator))
    config = reference.config.copy_tree()
    config.reference_rope = False
    computed = config.make().eval()
    computed.load_state_dict(reference.state_dict())
    args = (
        torch.randn(3, 2, 4, 4),
        torch.rand(3),
        torch.tensor([0, 1, 2]),
        torch.randn(3, 8),
    )
    with torch.no_grad():
        expected = reference(*args)
        actual = computed(*args)
    assert expected.velocity.abs().amax() > 0
    assert torch.allclose(actual.velocity, expected.velocity)
    assert torch.allclose(actual.cls_velocity, expected.cls_velocity)


def test_training_path_drop_broadcasts_the_coin_across_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 0's coin decides: this rank's own coin, 0.5, would keep the branch."""
    config = tiny_model().config.copy_tree()
    config.path_drop_prob = 0.25
    model = config.make().train()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.2 * torch.randn(parameter.shape, generator=generator))
    args = (
        torch.randn(3, 2, 4, 4, generator=generator),
        torch.rand(3, generator=generator),
        torch.tensor([0, 1, 2]),
        torch.randn(3, 8, generator=generator),
    )
    labels = torch.zeros(3, dtype=torch.bool)
    kept = model(
        *args,
        route_tokens=False,
        drop_sparse_path=False,
        force_drop_labels=labels,
    )
    forced = model(
        *args,
        route_tokens=False,
        drop_sparse_path=True,
        force_drop_labels=labels,
    )
    assert not torch.equal(kept.velocity, forced.velocity)
    monkeypatch.setattr(torch, "rand", partial(_coin, value=0.5))
    monkeypatch.setattr(dist, "broadcast", partial(_broadcast_from_rank_zero, coin=0.0))
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "gloo-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        dropped = model(*args, route_tokens=False, force_drop_labels=labels)
    finally:
        dist.destroy_process_group()
    assert torch.equal(dropped.velocity, forced.velocity)


def _coin(*size: int, value: float, **kwargs: object) -> Tensor:
    del size
    return torch.full(
        (),
        value,
        device=cast("torch.device | None", kwargs.get("device")),
    )


def _broadcast_from_rank_zero(tensor: Tensor, src: int, *, coin: float) -> None:
    assert src == 0
    _ = tensor.fill_(coin)


def test_sampler_requires_two_steps() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sample_latents(
            tiny_model(),
            latents=torch.randn(3, 2, 4, 4),
            cls_latents=torch.randn(3, 8),
            labels=torch.tensor([1, 2, 3]),
            num_steps=1,
        )


def test_sampler_restores_training_mode_and_shifts_time() -> None:
    model = tiny_model().train()
    latents = torch.randn(3, 2, 4, 4)
    cls = torch.randn(3, 8)
    labels = torch.tensor([1, 2, 3])
    torch.manual_seed(0)
    shifted, _ = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=labels,
        num_steps=2,
    )
    torch.manual_seed(0)
    plain, _ = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=labels,
        num_steps=2,
        shift_time=False,
    )
    assert model.training
    assert shifted.shape == latents.shape
    # 2 * 4 * 4 latent dimensions against a 4096 base shift every interior time.
    assert not torch.equal(shifted, plain)


def test_zero_cls_guidance_keeps_conditional_cls_drift() -> None:
    model = _ConstantVelocityModel()
    latents = torch.zeros(3, 2, 4, 4)
    cls = torch.zeros(3, 8)
    labels = torch.tensor([0, 1, 0])
    torch.manual_seed(7)
    _, conditional_cls = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=labels,
        num_steps=2,
        shift_time=False,
    )
    torch.manual_seed(7)
    _, zero_guidance_cls = sample_latents(
        model,
        latents=latents,
        cls_latents=cls,
        labels=labels,
        num_steps=2,
        cfg_scale=2,
        cls_cfg_scale=0,
        shift_time=False,
    )
    assert torch.equal(zero_guidance_cls, conditional_cls)


class _ConstantVelocityConfig:
    num_classes: int = 2


class _ConstantVelocityModel(nn.Module):
    config = _ConstantVelocityConfig()

    @override
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        cls_token: Tensor,
        **_kwargs: object,
    ) -> ModelOutput:
        del t
        cls_velocity = torch.where(y[:, None] == 2, -1.0, 1.0).expand_as(cls_token)
        return ModelOutput(
            velocity=torch.zeros_like(x),
            cls_velocity=cls_velocity,
            projections=(),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
