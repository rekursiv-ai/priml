"""Latent normalizers keep each reference's exact operation order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.model.ae.checkpoint import LocalFile
from priml.model.ae.latent_norm import (
    BatchNormLatents,
    ChannelLatentStats,
    ElementwiseLatentStats,
    InverseStdLatents,
    LatentNormalizer,
    ScaleLatents,
    ShiftScaleLatents,
    StdLatents,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _latent() -> Tensor:
    return torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(0))


def _stats_file(tmp_path: Path, **stats: Tensor | None) -> LocalFile.Config:
    path = tmp_path / "stats.pt"
    torch.save(stats, path)
    return LocalFile.Config(path=path)


def test_every_normalizer_satisfies_the_protocol(tmp_path: Path) -> None:
    # ``ChannelLatentStats`` refuses statistics not shaped [1, C, 1, 1].
    stats = _stats_file(
        tmp_path,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
    )
    assert isinstance(ScaleLatents.Config().make(), LatentNormalizer)
    assert isinstance(ChannelLatentStats.Config(stats=stats).make(), LatentNormalizer)
    var = _stats_file(tmp_path, var=torch.ones(3, 4, 5))
    assert isinstance(ElementwiseLatentStats.Config(stats=var).make(), LatentNormalizer)


def test_scale_is_one_multiply_each_way() -> None:
    norm = ScaleLatents.Config(scale=0.3099).make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), latent * 0.3099)
    assert torch.equal(norm.denormalize(latent), latent / 0.3099)


def test_scale_refuses_zero() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _ = ScaleLatents.Config(scale=0.0).make()


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf")])
def test_scale_refuses_a_nonfinite_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScaleLatents.Config(scale=scale).make()


@pytest.mark.parametrize("multiplier", [0.0, float("nan"), float("inf"), -float("inf")])
def test_channel_stats_require_invertible_multiplier(
    tmp_path: Path,
    multiplier: float,
) -> None:
    config = ChannelLatentStats.Config(
        stats=_stats_file(tmp_path, mean=torch.zeros(1), std=torch.ones(1)),
        multiplier=multiplier,
    )
    with pytest.raises(ValueError, match="finite nonzero"):
        config.make()


def test_elementwise_stats_follow_rae_order(tmp_path: Path) -> None:
    """``(z - mean) / sqrt(var + eps)`` and ``z * sqrt(var + eps) + mean``."""
    mean = torch.rand(3, 4, 5, generator=torch.Generator().manual_seed(1))
    var = torch.rand(3, 4, 5, generator=torch.Generator().manual_seed(2)) + 0.1
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, mean=mean, var=var),
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - mean) / torch.sqrt(var + 1e-5))
    assert torch.equal(norm.denormalize(latent), latent * torch.sqrt(var + 1e-5) + mean)


def test_elementwise_stats_skip_a_missing_mean(tmp_path: Path) -> None:
    """RAE's 256px statistics carry no mean; the reference subtracts 0, exactly."""
    var = torch.full((3, 4, 5), 4.0)
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, mean=None, var=var),
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - 0) / torch.sqrt(var + 1e-5))
    assert torch.equal(norm.denormalize(latent), latent * torch.sqrt(var + 1e-5) + 0)


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), -2.0])
def test_elementwise_stats_refuse_an_eps_outside_finite_nonnegatives(
    tmp_path: Path,
    eps: float,
) -> None:
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, var=torch.ones(3, 4, 5)),
        eps=eps,
    )
    with pytest.raises(ValueError, match="finite nonnegative eps"):
        _ = config.make()


def test_elementwise_stats_refuse_a_missing_var(tmp_path: Path) -> None:
    """The reference's ``else 1`` fallback raises in ``torch.sqrt``; so does this."""
    config = ElementwiseLatentStats.Config(stats=_stats_file(tmp_path, mean=None))
    with pytest.raises(ValueError, match="needs a var"):
        _ = config.make()


def test_channel_stats_follow_lightningdit_order(tmp_path: Path) -> None:
    """``(z - mean) / std * m`` and ``z * std / m + mean``."""
    # ``ChannelLatentStats`` refuses statistics not shaped [1, C, 1, 1].
    mean = torch.tensor([0.1, -0.2, 0.3]).view(1, 3, 1, 1)
    std = torch.tensor([0.08, 0.2, 0.13]).view(1, 3, 1, 1)
    config = ChannelLatentStats.Config(
        stats=_stats_file(tmp_path, mean=mean, std=std),
        multiplier=1.5,
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - mean) / std * 1.5)
    assert torch.equal(norm.denormalize(latent), (latent * std) / 1.5 + mean)


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (torch.zeros(3), torch.ones(3)),
        # ChannelLatentStats.__init__ accepts only [1, C, 1, 1]. Each row keeps the
        # leading 1 and breaks one other part: the std, the spatial axes, the rank.
        (torch.zeros(1, 3, 1, 1), torch.ones(3)),
        (torch.zeros(1, 3, 4, 5), torch.ones(1, 3, 4, 5)),
        (torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)),
    ],
    ids=["bare", "mismatched", "spatial", "five-axis"],
)
def test_channel_stats_refuse_statistics_not_shaped_per_channel(
    tmp_path: Path,
    mean: Tensor,
    std: Tensor,
) -> None:
    """A bare ``[C]`` would broadcast against the width axis, not the channels."""
    config = ChannelLatentStats.Config(stats=_stats_file(tmp_path, mean=mean, std=std))
    with pytest.raises(ValueError, match=r"shape \[1, C, 1, 1\]"):
        _ = config.make()


def _elementwise(tmp_path: Path) -> LatentNormalizer:
    stats = _stats_file(tmp_path, mean=torch.zeros(3, 4, 5), var=torch.ones(3, 4, 5))
    return ElementwiseLatentStats.Config(stats=stats).make()


def _channel(tmp_path: Path) -> LatentNormalizer:
    # ``ChannelLatentStats`` refuses statistics not shaped [1, C, 1, 1].
    stats = _stats_file(
        tmp_path,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
    )
    return ChannelLatentStats.Config(stats=stats).make()


@pytest.mark.parametrize(
    "build",
    [_elementwise, _channel],
    ids=["elementwise", "channel"],
)
def test_statistics_copy_to_a_device_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[Path], LatentNormalizer],
) -> None:
    """Every later batch on a device reuses the copies the first one made."""
    norm = build(tmp_path)
    copies: list[torch.device] = []
    to = Tensor.to

    def counted(tensor: Tensor, device: torch.device) -> Tensor:
        copies.append(device)
        return to(tensor, device)

    monkeypatch.setattr(Tensor, "to", counted)
    latent = _latent()
    for _ in range(3):
        _ = norm.denormalize(norm.normalize(latent))
    assert len(copies) == 2


def test_statistics_files_are_required() -> None:
    with pytest.raises(ValueError, match="stats file"):
        _ = ElementwiseLatentStats.Config().make()
    with pytest.raises(ValueError, match="stats file"):
        _ = ChannelLatentStats.Config().make()


def test_shift_scale_shifts_then_scales_and_back() -> None:
    norm = ShiftScaleLatents.Config(shift=0.1159, scale=0.3611).make()
    assert isinstance(norm, LatentNormalizer)
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - 0.1159) * 0.3611)
    assert torch.equal(norm.denormalize(latent), latent / 0.3611 + 0.1159)


@pytest.mark.parametrize(
    ("shift", "scale", "match"),
    [(0.0, 0.0, "nonzero"), (float("nan"), 1.0, "finite shift")],
)
def test_shift_scale_refuses_a_degenerate_pair(
    shift: float,
    scale: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = ShiftScaleLatents.Config(shift=shift, scale=scale).make()


def _batch_norm_file(tmp_path: Path, **stats: Tensor) -> LocalFile.Config:
    """Save statistics under ``bn.<name>``, as FLUX.2's checkpoint names them."""
    path = tmp_path / "bn.pt"
    torch.save({f"bn.{name}": value for name, value in stats.items()}, path)
    return LocalFile.Config(path=path)


def test_batch_norm_is_the_frozen_module_and_its_inverse(tmp_path: Path) -> None:
    """The reference runs an eval ``BatchNorm2d``; leading dimensions pass through."""
    generator = torch.Generator().manual_seed(1)
    mean = torch.randn(3, generator=generator)
    var = 0.5 + torch.rand(3, generator=generator)
    stats = _batch_norm_file(tmp_path, running_mean=mean, running_var=var)
    norm = BatchNormLatents.Config(stats=stats, eps=1e-4).make()
    assert isinstance(norm, LatentNormalizer)
    reference = torch.nn.BatchNorm2d(3, eps=1e-4, affine=False).eval()
    assert reference.running_mean is not None
    assert reference.running_var is not None
    reference.running_mean.copy_(mean)
    reference.running_var.copy_(var)
    latent = torch.randn(2, 4, 3, 5, 6, generator=generator)
    expected = reference(latent.flatten(0, 1)).unflatten(0, (2, 4))
    assert torch.equal(norm.normalize(latent), expected)
    inverse = latent * torch.sqrt(var.view(-1, 1, 1) + 1e-4) + mean.view(-1, 1, 1)
    assert torch.equal(norm.denormalize(latent), inverse)


def test_batch_norm_refuses_missing_or_misshapen_statistics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stats file"):
        _ = BatchNormLatents.Config().make()
    no_var = _batch_norm_file(tmp_path, running_mean=torch.zeros(3))
    with pytest.raises(ValueError, match="running_var"):
        _ = BatchNormLatents.Config(stats=no_var).make()
    misshapen = _batch_norm_file(
        tmp_path,
        running_mean=torch.zeros(3),
        running_var=torch.ones(4),
    )
    with pytest.raises(ValueError, match="shape"):
        _ = BatchNormLatents.Config(stats=misshapen).make()
    stats = _batch_norm_file(
        tmp_path,
        running_mean=torch.zeros(3),
        running_var=torch.ones(3),
    )
    with pytest.raises(ValueError, match="eps"):
        _ = BatchNormLatents.Config(stats=stats, eps=-1.0).make()


def test_inverse_std_multiplies_by_the_reciprocal_and_divides_back() -> None:
    """Wan's order: ``(z - mean) * (1 / std)``, and ``z / (1 / std) + mean``."""
    mean, std = [0.5, -0.25, 1.5], [2.8184, 1.4541, 0.3523]
    norm = InverseStdLatents.Config(mean=mean, std=std).make()
    assert isinstance(norm, LatentNormalizer)
    latent = torch.randn(2, 4, 3, 5, 6, 7, generator=torch.Generator().manual_seed(2))
    shift = torch.tensor(mean).view(-1, 1, 1, 1)
    reciprocal = (1.0 / torch.tensor(std)).view(-1, 1, 1, 1)
    assert torch.equal(norm.normalize(latent), (latent - shift) * reciprocal)
    assert torch.equal(norm.denormalize(latent), latent / reciprocal + shift)


@pytest.mark.parametrize(
    ("mean", "std", "match"),
    [([], [], "one mean and one std"), ([0.0], [0.0], "nonzero")],
)
def test_inverse_std_refuses_degenerate_statistics(
    mean: list[float],
    std: list[float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = InverseStdLatents.Config(mean=mean, std=std).make()


def test_std_divides_and_multiplies_back() -> None:
    """Mochi's order: ``(z - mean) / std``, and ``z * std + mean``."""
    mean, std = [0.5, -0.25, 1.5], [0.9263, 1.3720, 0.8813]
    norm = StdLatents.Config(mean=mean, std=std).make()
    assert isinstance(norm, LatentNormalizer)
    latent = torch.randn(2, 4, 3, 5, 6, 7, generator=torch.Generator().manual_seed(3))
    shift, scale = (torch.tensor(v).view(-1, 1, 1, 1) for v in (mean, std))
    assert torch.equal(norm.normalize(latent), (latent - shift) / scale)
    assert torch.equal(norm.denormalize(latent), latent * scale + shift)


def test_std_refuses_mismatched_statistics() -> None:
    with pytest.raises(ValueError, match="one mean and one std"):
        _ = StdLatents.Config(mean=[0.0, 1.0], std=[1.0]).make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
