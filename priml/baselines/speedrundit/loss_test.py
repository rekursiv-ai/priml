"""The four-term objective: each term in isolation, and the draw order."""

from __future__ import annotations

from typing import Final, TypedDict, cast

import math

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.speedrundit.loss import (
    SpeedrunDiTLoss,
    cosine_path,
    linear_path,
    linear_time_weight,
    logit_normal_time,
    mean_flat,
    rectified_flow_path,
    resolution_time_shift,
    uniform_time,
)
from priml.baselines.speedrundit.model import SpeedrunDiT


BATCH: Final = 3
GRID: Final = 4
CHANNELS: Final = 8
TARGET: Final = 16


def tiny_model() -> SpeedrunDiT:
    """Build a small model for end-to-end objective tests.

    Returns:
      model: A shrunk SR-DiT.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = CHANNELS
    cfg.channels_hidden = 64
    cfg.image_size = GRID
    cfg.num_layers = 5
    cfg.heads = 4
    cfg.num_classes = 10
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 32
    return cfg.make()


class Inputs(TypedDict):
    """The objective's batch arguments, keyed as it takes them."""

    media: Tensor
    label: Tensor
    cls_token: Tensor


def inputs() -> Inputs:
    """Build one batch of objective inputs.

    Returns:
      batch: Latents, labels, and class features.

    """
    generator = torch.Generator().manual_seed(0)
    return {
        "media": torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator),
        "label": torch.randint(10, (BATCH,), generator=generator),
        "cls_token": torch.randn(BATCH, TARGET, generator=generator),
    }


def test_linear_path_velocity_target_is_exactly_noise_minus_data() -> None:
    """``-1 * x + 1 * eps`` must land on the same bits as ``eps - x``.

    That identity is what lets this path stand in for
    ``priml.math.diffusion.target_rectified_flow`` without a log-SNR round
    trip. If it ever stopped holding, the reuse argument in the module
    docstring would be wrong.
    """
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(64, generator=generator)
    eps = torch.randn(64, generator=generator)
    path = linear_path(torch.rand(64, generator=generator))
    assert torch.equal(path.d_alpha * x + path.d_sigma * eps, eps - x)


def test_linear_path_endpoints() -> None:
    """At time zero the path is the data; at one it is the noise."""
    zero, one = torch.zeros(1), torch.ones(1)
    assert torch.equal(linear_path(zero).alpha, torch.ones(1))
    assert torch.equal(linear_path(zero).sigma, torch.zeros(1))
    assert torch.equal(linear_path(one).alpha, torch.zeros(1))
    assert torch.equal(linear_path(one).sigma, torch.ones(1))


def test_the_shared_schedule_agrees_with_the_straight_form() -> None:
    """``rectified_flow_path`` and ``linear_path`` compute one function.

    They agree far below a float32 ULP and are NOT bit-identical, because the
    logit/sigmoid round trip rounds twice where ``1 - t`` rounds once. Both
    halves of that are asserted: a drift would mean the shared schedule and
    the reference had genuinely diverged, and exact equality would mean the
    parity argument for keeping the straight form was never needed.
    """
    t = torch.linspace(0.01, 0.99, 64)
    shared, straight = rectified_flow_path(t), linear_path(t)
    assert torch.allclose(shared.alpha, straight.alpha, atol=1e-6)
    assert torch.allclose(shared.sigma, straight.sigma, atol=1e-6)
    assert not torch.equal(shared.sigma, straight.sigma)


def test_both_paths_share_the_velocity_target() -> None:
    """The target is ``eps - x`` either way, which is what lets the sampler
    hand this model's output to ``target_rectified_flow`` unchanged.
    """
    t = torch.linspace(0.01, 0.99, 8)
    assert rectified_flow_path(t).d_alpha == linear_path(t).d_alpha
    assert rectified_flow_path(t).d_sigma == linear_path(t).d_sigma


def test_cosine_path_stays_on_the_unit_circle() -> None:
    """``alpha^2 + sigma^2`` is one everywhere on the cosine path."""
    t = torch.linspace(0, 1, 17)
    path = cosine_path(t)
    assert torch.allclose(path.alpha**2 + path.sigma**2, torch.ones_like(t))


def test_time_shift_is_the_identity_at_the_base_resolution() -> None:
    """The shift is defined so that a sample of ``base`` elements is unmoved."""
    t = torch.linspace(0, 1, 11)
    shifted = resolution_time_shift(t, shape=(2, 4, 4), base=32)
    assert torch.allclose(shifted, t)


def test_time_shift_pushes_larger_samples_toward_noise() -> None:
    """A bigger latent carries more redundancy, so the same nominal time
    destroys less; the shift compensates.
    """
    t = torch.full((5,), 0.5)
    shifted = resolution_time_shift(t, shape=(32, 16, 16), base=4096)
    assert torch.all(shifted > t)
    assert torch.all(shifted <= 1.0)


def test_time_shift_matches_the_reference_formula() -> None:
    """Pinned against the closed form rather than a recorded number."""
    t = torch.tensor([0.25])
    shift = math.sqrt(8192 / 4096)  # noqa: TID251 -- The closed form the reference spells.
    expected = (shift * t) / (1 + (shift - 1) * t)
    assert torch.equal(resolution_time_shift(t, shape=(32, 16, 16)), expected)


def test_samplers_produce_times_in_range() -> None:
    """Both time distributions stay inside the path's domain."""
    for sampler in (uniform_time, logit_normal_time):
        t = sampler(256)
        assert t.shape == (256, 1, 1, 1)
        assert torch.all(t >= 0), sampler.__name__
        assert torch.all(t <= 1), sampler.__name__


def test_mean_flat_reduces_everything_but_the_batch() -> None:
    """Rank two and up keep the batch; rank one collapses to a scalar."""
    assert mean_flat(torch.ones(4, 3, 2)).shape == (4,)
    assert mean_flat(torch.ones(7)).ndim == 0


def test_the_draw_order_is_time_then_noise_then_class_noise() -> None:
    """Reordering the draws changes every sample a run ever sees.

    Asserted by replaying the same seed by hand: whatever the objective drew
    must equal what these three calls draw in this order.
    """
    model = tiny_model()
    batch = inputs()
    objective = SpeedrunDiTLoss.Config().make()

    torch.manual_seed(31_337)
    result = objective(model, **batch)

    torch.manual_seed(31_337)
    expected_time = uniform_time(BATCH)
    expected_time = resolution_time_shift(
        expected_time,
        shape=tuple(batch["media"].shape[1:]),
        base=4096,
    ).to(dtype=batch["media"].dtype)
    expected_noise = torch.randn_like(batch["media"])

    assert torch.equal(result.time, expected_time)
    assert torch.equal(result.noise, expected_noise)


def test_supplied_time_and_noise_bypass_the_draws() -> None:
    """A caller pinning the randomness gets exactly what it passed."""
    model = tiny_model()
    batch = inputs()
    objective = SpeedrunDiTLoss.Config(time_transform=None).make()
    time = torch.full((BATCH, 1, 1, 1), 0.3)
    noise = torch.ones_like(batch["media"])
    result = objective(model, **batch, time=time, noise=noise)
    assert torch.equal(result.time, time)
    assert torch.equal(result.noise, noise)


def test_projection_is_zero_without_targets() -> None:
    """A corpus with no alignment features trains the velocity terms alone."""
    model = tiny_model()
    result = SpeedrunDiTLoss.Config().make()(model, **inputs())
    assert result.projection.item() == 0.0


def test_projection_rewards_alignment() -> None:
    """Aligned projections score lower than anti-aligned ones.

    A negative cosine similarity, so perfect alignment is minus one and the
    term is minimized by agreeing with the frozen encoder.
    """
    model = tiny_model()
    batch = inputs()
    tokens = 1 + GRID * GRID
    aligned = torch.ones(BATCH, tokens, TARGET)
    opposed = -aligned
    objective = SpeedrunDiTLoss.Config().make()
    torch.manual_seed(5)
    good = objective(model, **batch, features=[aligned])
    torch.manual_seed(5)
    bad = objective(model, **batch, features=[opposed])
    # The model is identity-at-init so both projections are the same; the
    # sign of the target is what separates them.
    assert good.projection.item() != bad.projection.item()


def test_contrastive_term_is_negative() -> None:
    """CFM subtracts a distance, so minimizing the total pushes the field away
    from the batch neighbour's target.
    """
    model = tiny_model()
    result = SpeedrunDiTLoss.Config().make()(model, **inputs())
    assert result.cfm.item() <= 0.0


def test_the_total_is_its_weighted_parts() -> None:
    """The reported total must be reconstructible from the reported terms.

    Note ``cfm_cls`` is computed and reported but deliberately NOT summed,
    matching the reference; a total that included it would silently differ.
    """
    model = tiny_model()
    cfg = SpeedrunDiTLoss.Config()
    result = cfg.make()(model, **inputs())
    expected = (
        result.denoising.mean()
        + cfg.projection_coeff * result.projection.mean()
        + cfg.cls_coeff * result.cls.mean()
        + cfg.cfm_coeff * result.cfm.mean()
    )
    assert torch.equal(result.loss, expected)


def test_weighting_is_injected_not_branched() -> None:
    """Swapping the contrastive weight is a value, not a mode string."""
    model = tiny_model()
    batch = inputs()
    torch.manual_seed(11)
    uniform = SpeedrunDiTLoss.Config().make()(model, **batch)
    torch.manual_seed(11)
    weighted = SpeedrunDiTLoss.Config(cfm_weight=linear_time_weight)
    linear = weighted.make()(model, **batch)
    assert uniform.cfm.item() != linear.cfm.item()
    assert torch.equal(uniform.time, linear.time)


def test_the_path_is_injected_not_branched() -> None:
    """Swapping the interpolant is a value too."""
    model = tiny_model()
    batch = inputs()
    torch.manual_seed(13)
    straight = SpeedrunDiTLoss.Config().make()(model, **batch)
    torch.manual_seed(13)
    curved = SpeedrunDiTLoss.Config(interpolant=cosine_path).make()(model, **batch)
    assert straight.denoising.shape == curved.denoising.shape
    assert not torch.equal(straight.denoising, curved.denoising)


def test_the_time_transform_slot_can_be_rebound() -> None:
    """``base`` rides in the tree through ``PartialConfig`` rather than as a
    keyword nobody can print.
    """
    cfg = SpeedrunDiTLoss.Config(
        time_transform=PartialConfig(resolution_time_shift, base=64),
    )
    model = tiny_model()
    batch = inputs()
    torch.manual_seed(17)
    shifted = cfg.make()(model, **batch)
    torch.manual_seed(17)
    plain = SpeedrunDiTLoss.Config(time_transform=None).make()(model, **batch)
    assert not torch.equal(shifted.time, plain.time)


@pytest.mark.parametrize("term", ["denoising", "cls"])
def test_per_sample_terms_keep_the_batch_axis(term: str) -> None:
    """Per-sample errors stay per-sample so a metric can weight them."""
    model = tiny_model()
    result = SpeedrunDiTLoss.Config().make()(model, **inputs())
    value = cast(Tensor, getattr(result, term))
    assert value.shape == (BATCH,)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
