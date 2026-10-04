"""Probability paths, REG alignment, and objective input validation."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import math

from torch import Tensor
from torch.nn import functional

import pytest
import torch

from priml.baselines.speedrundit.model import ModelOutput, Projection
from priml.baselines.speedrundit.objective import (
    SpeedrunObjective,
    interpolant,
    projection_loss,
)
from priml.math.diffusion.time_shift import time_shift


def test_linear_path_has_constant_derivatives() -> None:
    t = torch.tensor([0.25, 0.75])

    alpha, sigma, d_alpha, d_sigma = interpolant(t, path="linear")

    assert torch.equal(alpha, 1 - t)
    assert torch.equal(sigma, t)
    assert torch.equal(d_alpha, -torch.ones_like(t))
    assert torch.equal(d_sigma, torch.ones_like(t))


def test_cosine_path_is_a_quarter_turn_with_its_derivatives() -> None:
    t = torch.tensor([0.0, 0.25, 1.0])
    alpha, sigma, d_alpha, d_sigma = interpolant(t, path="cosine")
    angle = math.pi / 2 * t
    assert torch.allclose(alpha, angle.cos())
    assert torch.allclose(sigma, angle.sin())
    assert torch.allclose(d_alpha, -math.pi / 2 * angle.sin())
    assert torch.allclose(d_sigma, math.pi / 2 * angle.cos())
    assert torch.allclose(alpha.square() + sigma.square(), torch.ones(3))


def test_projection_loss_gathers_the_tokens_sparse_routing_kept() -> None:
    teacher = torch.randn(2, 5, 3)
    kept = torch.tensor([[0, 3, 1, 4], [4, 1, 2, 0]])
    student = torch.randn(2, 4, 3)
    loss = projection_loss(
        (Projection(tokens=student, ids_keep=kept),),
        teacher_features=(teacher,),
    )
    selected = torch.stack([teacher[0, [0, 3, 1, 4]], teacher[1, [4, 1, 2, 0]]])
    expected = -functional.cosine_similarity(student, selected, dim=-1).mean(dim=1)
    assert torch.allclose(loss, expected)


def test_projection_loss_rejects_mismatched_depth_counts() -> None:
    student = Projection(tokens=torch.randn(2, 5, 3), ids_keep=None)
    with pytest.raises(
        ValueError,
        match=r"^teacher and student projection depths must match$",
    ):
        projection_loss(
            (student,),
            teacher_features=(torch.randn(2, 5, 3), torch.randn(2, 5, 3)),
        )


def test_projection_loss_averages_across_projection_depths() -> None:
    teacher = torch.eye(4)[:3].expand(2, -1, -1).clone()
    predictions = (
        Projection(tokens=-teacher, ids_keep=None),
        Projection(tokens=-teacher, ids_keep=None),
    )

    assert torch.equal(
        projection_loss(predictions, teacher_features=(teacher, teacher)),
        torch.ones(2),
    )


def test_projection_loss_rejects_mismatched_token_shapes() -> None:
    student = Projection(tokens=torch.randn(2, 4, 3), ids_keep=None)
    with pytest.raises(
        ValueError,
        match=r"^teacher tokens do not match the student projection$",
    ):
        projection_loss((student,), teacher_features=(torch.randn(2, 5, 3),))


@pytest.mark.parametrize("path", ["linear", "cosine"])
def test_lognormal_time_maps_a_lognormal_sigma_per_path(
    path: Literal["linear", "cosine"],
) -> None:
    latents = torch.zeros(3, 2, 4, 5)
    torch.manual_seed(0)
    sigma = torch.randn(3).exp()
    expected = sigma / (1 + sigma) if path == "linear" else 2 * sigma.atan() / math.pi
    objective = SpeedrunObjective(path=path, weighting="lognormal", shift_time=False)
    torch.manual_seed(0)
    assert torch.allclose(objective.sample_time(latents), expected)


def test_shifted_time_follows_the_latent_dimension() -> None:
    latents = torch.zeros(3, 2, 4, 5)
    torch.manual_seed(0)
    uniform = torch.rand(3)
    torch.manual_seed(0)
    shifted = SpeedrunObjective(shift_base=8).sample_time(latents)
    assert torch.allclose(
        shifted,
        time_shift(uniform, latent_dimensions=2 * 4 * 5, reference_dimensions=8),
    )


@pytest.mark.parametrize("path", ["linear", "cosine"])
@pytest.mark.parametrize("cfm_weighting", ["uniform", "linear"])
def test_objective_builds_inputs_and_returns_each_loss_term(
    path: Literal["linear", "cosine"],
    cfm_weighting: Literal["uniform", "linear"],
) -> None:
    latents = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) / 13
    labels = torch.tensor([1, 2])
    time = torch.tensor([0.25, 0.75])
    noise = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) / 17
    cls_noise = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]])
    teacher_sparse = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 11
    teacher_cls = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3) / 19
    student_sparse = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 7
    student_dense = -torch.arange(36, dtype=torch.float32).reshape(2, 6, 3) / 23
    ids_keep = torch.tensor([[0, 3, 1, 4], [4, 1, 3, 0]])
    output = ModelOutput(
        velocity=torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) / 29,
        cls_velocity=torch.arange(6, dtype=torch.float32).reshape(2, 3) / 31,
        projections=(
            Projection(tokens=student_sparse, ids_keep=ids_keep),
            Projection(tokens=student_dense, ids_keep=None),
        ),
    )
    model = Mock(return_value=output)
    objective = SpeedrunObjective(
        path=path,
        cfm_weighting=cfm_weighting,
        projection_coeff=0.7,
        cls_coeff=0.2,
        cfm_coeff=0.1,
        shift_time=False,
    )

    terms = objective(
        model,
        latents=latents,
        labels=labels,
        teacher_features=(teacher_sparse, teacher_cls),
        time=time,
        noise=noise,
        cls_noise=cls_noise,
    )

    if path == "linear":
        alpha, sigma = 1 - time, time
        d_alpha, d_sigma = -torch.ones_like(time), torch.ones_like(time)
    else:
        angle = math.pi * time / 2
        alpha, sigma = angle.cos(), angle.sin()
        d_alpha = -(math.pi / 2) * angle.sin()
        d_sigma = (math.pi / 2) * angle.cos()
    latent_broadcast = (slice(None), None, None, None)
    expected_noisy = alpha[latent_broadcast] * latents + sigma[latent_broadcast] * noise
    expected_target = (
        d_alpha[latent_broadcast] * latents + d_sigma[latent_broadcast] * noise
    )
    expected_cls_noisy = alpha[:, None] * teacher_cls[:, 0] + sigma[:, None] * cls_noise
    expected_cls_target = (
        d_alpha[:, None] * teacher_cls[:, 0] + d_sigma[:, None] * cls_noise
    )
    model.assert_called_once()
    model_args = model.call_args.args
    assert len(model_args) == 4
    noisy_arg, time_arg, labels_arg, cls_arg = model_args
    assert isinstance(noisy_arg, Tensor)
    assert isinstance(time_arg, Tensor)
    assert isinstance(labels_arg, Tensor)
    assert isinstance(cls_arg, Tensor)
    assert torch.equal(noisy_arg, expected_noisy)
    assert torch.equal(time_arg, time)
    assert torch.equal(labels_arg, labels)
    assert torch.equal(cls_arg, expected_cls_noisy)

    selected_sparse = teacher_sparse.gather(
        1,
        ids_keep[..., None].expand(-1, -1, teacher_sparse.shape[-1]),
    )
    expected_projection = (
        -(
            functional.normalize(student_sparse, dim=-1)
            * functional.normalize(selected_sparse, dim=-1)
        )
        .sum(dim=-1)
        .mean(dim=1)
        - (
            functional.normalize(student_dense, dim=-1)
            * functional.normalize(teacher_cls, dim=-1)
        )
        .sum(dim=-1)
        .mean(dim=1)
    ) / 2
    expected_velocity = (output.velocity - expected_target).square().mean(dim=(1, 2, 3))
    expected_cls = (output.cls_velocity - expected_cls_target).square().mean(dim=1)
    cfm_error = (output.velocity - torch.roll(expected_target, 1, 0)).square()
    if cfm_weighting == "linear":
        cfm_error = cfm_error * time.reshape(2, 1, 1, 1)
    expected_cfm = -cfm_error.mean()
    expected_loss = (
        expected_velocity
        + 0.7 * expected_projection
        + 0.2 * expected_cls
        + 0.1 * expected_cfm
    )
    expected_mean_loss = (
        expected_velocity.mean()
        + 0.7 * expected_projection.mean()
        + 0.2 * expected_cls.mean()
        + 0.1 * expected_cfm
    )
    assert torch.allclose(terms.velocity, expected_velocity)
    assert torch.allclose(terms.cls, expected_cls)
    assert torch.allclose(terms.projection, expected_projection)
    assert torch.allclose(terms.cfm, expected_cfm)
    assert torch.allclose(terms.loss, expected_loss)
    assert torch.allclose(terms.mean_loss, expected_mean_loss)
    assert terms.output is output


def test_objective_samples_missing_time_and_noise_from_their_inputs() -> None:
    latents = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) / 11
    labels = torch.tensor([2, 3])
    teacher_features = tuple(
        torch.arange(36, dtype=torch.float32).reshape(2, 6, 3) / divisor
        for divisor in (7, 11, 13)
    )
    output = ModelOutput(
        velocity=torch.zeros_like(latents),
        cls_velocity=torch.zeros(2, 3),
        projections=tuple(
            Projection(tokens=features, ids_keep=None) for features in teacher_features
        ),
    )
    model = Mock(return_value=output)
    torch.manual_seed(19)
    expected_time = torch.rand(2)
    expected_noise = torch.randn_like(latents)
    expected_cls_noise = torch.randn_like(teacher_features[-1][:, 0])
    torch.manual_seed(19)

    SpeedrunObjective(shift_time=False)(
        model,
        latents=latents,
        labels=labels,
        teacher_features=teacher_features,
    )

    model.assert_called_once()
    model_args = model.call_args.args
    assert len(model_args) == 4
    noisy_arg, time_arg, labels_arg, cls_arg = model_args
    assert isinstance(noisy_arg, Tensor)
    assert isinstance(time_arg, Tensor)
    assert isinstance(labels_arg, Tensor)
    assert isinstance(cls_arg, Tensor)
    assert torch.equal(time_arg, expected_time)
    assert torch.equal(labels_arg, labels)
    assert torch.equal(
        noisy_arg,
        (1 - expected_time)[..., None, None, None] * latents
        + expected_time[..., None, None, None] * expected_noise,
    )
    assert torch.equal(
        cls_arg,
        (1 - expected_time)[:, None] * teacher_features[-1][:, 0]
        + expected_time[:, None] * expected_cls_noise,
    )


@pytest.mark.parametrize("weighting", ["uniform", "lognormal"])
def test_sampled_time_uses_the_latent_device(
    weighting: Literal["uniform", "lognormal"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latents = torch.zeros(2, 3, 4)
    rand = Mock(wraps=torch.rand)
    randn = Mock(wraps=torch.randn)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randn", randn)

    SpeedrunObjective(weighting=weighting, shift_time=False).sample_time(latents)

    sample = rand if weighting == "uniform" else randn
    unused_sample = randn if weighting == "uniform" else rand
    sample.assert_called_once_with(2, device=latents.device)
    unused_sample.assert_not_called()


def test_objective_rejects_mismatched_label_count() -> None:
    with pytest.raises(
        ValueError,
        match=r"^labels and latents have different batch sizes$",
    ):
        SpeedrunObjective()(
            _unused_model,
            latents=torch.zeros(2, 3, 4, 5),
            labels=torch.zeros(3, dtype=torch.int64),
            teacher_features=(torch.zeros(2, 6, 7),),
        )


def _unused_model(*args: Tensor) -> ModelOutput:
    raise AssertionError(f"validation must precede the model; got {len(args)} args")


def test_objective_requires_a_teacher_feature_map() -> None:
    with pytest.raises(
        ValueError,
        match=r"^REG requires at least one teacher feature map$",
    ):
        SpeedrunObjective()(
            _unused_model,
            latents=torch.zeros(2, 3, 4, 5),
            labels=torch.zeros(2, dtype=torch.int64),
            teacher_features=(),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
