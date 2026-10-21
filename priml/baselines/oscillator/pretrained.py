"""Evaluate pinned local Chronos weights without fitting reference targets."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from pkgutil import resolve_name
from typing import TYPE_CHECKING, Protocol, TypedDict

import hashlib

from configgle import Fig, InlineConfig, Makeable
from torch import Tensor
from treekle import json_loads

import torch

from priml.baselines.oscillator.data import STATE_CHANNELS
from priml.baselines.oscillator.train_step import FitReport


if TYPE_CHECKING:
    from priml.baselines.oscillator.data import Oscillator, Windows


class RawPipeline(Protocol):
    """The common quantile interface of the two upstream pipelines."""

    def predict_quantiles(
        self,
        inputs: Tensor,
        *,
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs: object,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]]:
        """Return quantiles and upstream point forecasts."""
        ...


class PipelineFactory(Protocol):
    """Load a pipeline from files already staged on disk."""

    def from_pretrained(self, path: str, **kwargs: object) -> RawPipeline:
        """Build the model without downloading anything."""
        ...


class ForecastPipeline(Protocol):
    """Map observed multivariate histories to a point forecast."""

    def __call__(self, media: Tensor, *, horizon: int) -> Tensor:
        """Return forecasts shaped (trajectories, horizon, channels)."""
        ...


class AssetManifest(TypedDict):
    """Pinned identity and complete digests for the two staged files."""

    model_id: str
    revision: str
    files: dict[str, str]


class ModelFiles:
    """Verify the staged model identity and every recorded file digest."""

    class Config(Fig["ModelFiles"]):
        directory: Path = Path("/opt/scratch/caches/chronos/chronos-bolt-tiny")
        """Local files; staging is a separate command."""

        model_id: str = "amazon/chronos-bolt-tiny"
        """Upstream repository identity expected by this recipe."""

        revision: str = "a0e552de83495b5c28c14c71c374f3e33280b340"
        """Pinned upstream model revision."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.directory = config.directory.expanduser().resolve()
        record = json_loads(
            (self.directory / "source.json").read_text(),
            AssetManifest,
        )
        if (
            record["model_id"] != config.model_id
            or record["revision"] != config.revision
        ):
            raise ValueError("Staged weights do not match the pinned model identity.")
        if set(record["files"]) != {"config.json", "model.safetensors"}:
            raise ValueError("The model manifest must cover config and safetensors.")
        for name, expected in record["files"].items():
            with (self.directory / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Staged model digest changed: {name}")


class BoltPipeline:
    """Forecast the two channels independently with a pinned Chronos-Bolt model."""

    class Config(Fig["BoltPipeline"]):
        files: ModelFiles.Config = field(default_factory=ModelFiles.Config)
        """Pinned model files, verified before loading."""

        factory: Makeable[PipelineFactory] = field(
            default_factory=lambda: InlineConfig(
                resolve_name,
                "chronos:BaseChronosPipeline",
            ),
        )
        """Upstream pipeline constructor; replaceable for offline tests."""

        batch_size: int = 64
        """Independent channel histories per inference batch."""

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError("The inference batch size must be positive.")
        self.config = config
        files = config.files.make()
        self.pipeline = config.factory.make().from_pretrained(
            str(files.directory),
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=True,
        )

    def __call__(self, media: Tensor, *, horizon: int) -> Tensor:
        """Return upstream median forecasts using only the supplied context."""
        context = media.transpose(1, 2).reshape(-1, media.shape[1])
        forecasts: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(context), self.config.batch_size):
                _, point = self.pipeline.predict_quantiles(
                    context[start : start + self.config.batch_size],
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                )
                if not isinstance(point, Tensor):
                    raise TypeError("Chronos-Bolt must return a batched point tensor.")
                forecasts.append(point)
        return (
            torch.cat(forecasts)
            .reshape(len(media), STATE_CHANNELS, horizon)
            .transpose(1, 2)
            .to(media)
        )


class Chronos2Pipeline:
    """Keep both state channels together within each Chronos-2 input group."""

    class Config(Fig["Chronos2Pipeline"]):
        files: ModelFiles.Config = field(
            default_factory=lambda: ModelFiles.Config(
                directory=Path("/opt/scratch/caches/chronos/chronos-2"),
                model_id="amazon/chronos-2",
                revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
            ),
        )
        """Pinned multivariate model, staged before evaluation."""

        factory: Makeable[PipelineFactory] = field(
            default_factory=lambda: InlineConfig(
                resolve_name,
                "chronos:Chronos2Pipeline",
            ),
        )
        """Upstream constructor; tests inject a small local implementation."""

        batch_size: int = 64
        """Maximum target channels in each upstream batch."""

    def __init__(self, config: Config) -> None:
        if config.batch_size < STATE_CHANNELS:
            raise ValueError("An inference batch must fit one complete state pair.")
        self.config = config
        files = config.files.make()
        self.pipeline = config.factory.make().from_pretrained(
            str(files.directory),
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=True,
        )

    def __call__(self, media: Tensor, *, horizon: int) -> Tensor:
        """Forecast each trajectory as a separate multivariate group."""
        with torch.inference_mode():
            _, point = self.pipeline.predict_quantiles(
                media.transpose(1, 2),
                prediction_length=horizon,
                quantile_levels=[0.5],
                batch_size=self.config.batch_size,
                cross_learning=False,
            )
        if not isinstance(point, list) or len(point) != len(media):
            raise TypeError("Chronos-2 must return one point tensor per trajectory.")
        return torch.stack(point).transpose(1, 2).to(media)


class PretrainedFitReport(FitReport):
    """Zero-shot methods consume no adaptation targets."""

    reference_training_used: int
    reference_validation_used: int


class PretrainedForecast:
    """Use a fixed pretrained pipeline as a zero-shot comparison."""

    class Config(Fig["PretrainedForecast"]):
        pipeline: Makeable[ForecastPipeline] = field(
            default_factory=BoltPipeline.Config,
        )
        """Point forecaster loaded only from verified local files."""

    def __init__(self, config: Config) -> None:
        self.pipeline = config.pipeline.make()
        self.horizon = 0

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> PretrainedFitReport:
        """Read the forecast geometry without using any reference target values.

        Args:
          train: Windows supplying the forecast horizon through their label shape.
          validation: Unused reference validation windows.
          synthetic: Unused cheap-simulator windows.
          simulator: Unused cheap simulator.

        Returns:
          report: Zero optimizer updates and zero reference trajectories consumed.

        """
        del validation, synthetic, simulator
        self.horizon = train.label.shape[1]
        return {
            "updates": 0,
            "best_update": 0,
            "pretrain_updates": 0,
            "reference_training_used": 0,
            "reference_validation_used": 0,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Forecast from the visible context, never another trajectory's future.

        Args:
          media: Observations shaped (trajectories, context timesteps, channels).

        Returns:
          forecast: Predictions shaped (trajectories, horizon, channels).

        """
        if self.horizon <= 0:
            raise RuntimeError("Set the forecast geometry before predicting.")
        result = self.pipeline(media, horizon=self.horizon)
        expected = (len(media), self.horizon, STATE_CHANNELS)
        if result.shape != expected:
            raise ValueError(
                "The pretrained forecast must match the target geometry.",
            )
        return result
