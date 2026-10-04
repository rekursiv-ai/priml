"""Latent normalizers keep each reference's exact operation order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.model.vision_ae.checkpoint import LocalFile
from priml.model.vision_ae.custom_types import LatentNormalizer
from priml.model.vision_ae.latent_norm import (
    ChannelLatentStats,
    ElementwiseLatentStats,
    ScaleLatents,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _latent() -> Tensor:
    return torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0))


def _stats_file(tmp_path: Path, **stats: Tensor | None) -> LocalFile.Config:
    path = tmp_path / "stats.pt"
    torch.save(stats, path)
    return LocalFile.Config(path=path)


def test_every_normalizer_satisfies_the_protocol(tmp_path: Path) -> None:
    stats = _stats_file(
        tmp_path,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
    )
    assert isinstance(ScaleLatents.Config().make(), LatentNormalizer)
    assert isinstance(ChannelLatentStats.Config(stats=stats).make(), LatentNormalizer)
    var = _stats_file(tmp_path, var=torch.ones(3, 4, 4))
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
    mean = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(1))
    var = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(2)) + 0.1
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, mean=mean, var=var),
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - mean) / torch.sqrt(var + 1e-5))
    assert torch.equal(norm.denormalize(latent), latent * torch.sqrt(var + 1e-5) + mean)


def test_elementwise_stats_skip_a_missing_mean(tmp_path: Path) -> None:
    """RAE's 256px statistics carry no mean; the reference subtracts 0, exactly."""
    var = torch.full((3, 4, 4), 4.0)
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
        stats=_stats_file(tmp_path, var=torch.ones(3, 4, 4)),
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
        (torch.zeros(1, 3, 1, 1), torch.ones(3)),
        (torch.zeros(1, 3, 4, 1), torch.ones(1, 3, 4, 1)),
        (torch.zeros(1, 1, 3, 1, 1), torch.ones(1, 1, 3, 1, 1)),
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
    stats = _stats_file(tmp_path, mean=torch.zeros(3, 4, 4), var=torch.ones(3, 4, 4))
    return ElementwiseLatentStats.Config(stats=stats).make()


def _channel(tmp_path: Path) -> LatentNormalizer:
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
