"""Forecast observed motion without receiving the episode's hidden dynamics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

import math

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.etth1.model import DLinear
from priml.loss.simple_loss import SimpleLoss, mse
from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.swiglu import SwiGLU
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.baselines.oscillator.regimes import ObservedWindows, RegimeDynamics


@runtime_checkable
class ObservedGeometry(Protocol):
    """Dimensions shared by the observed task and a forecasting model."""

    seq_len: int
    pred_len: int
    channels: int


class ObservedForecaster(Protocol):
    """Fit to training windows and forecast from observed history alone."""

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        """Return the fitting budget and any model-selection diagnostics."""
        ...

    def predict(self, media: Tensor) -> Tensor:
        """Return future observations without hidden states or test targets."""
        ...


class ObservedDLinear(nn.Module):
    """Adapt the existing DLinear forward to the shared training batch names."""

    class Config(Fig["ObservedDLinear"]):
        model: DLinear.Config = field(default_factory=DLinear.Config)
        """Original DLinear implementation and its full configuration."""

        @property
        def seq_len(self) -> int:
            """Observed timesteps owned by DLinear."""
            return self.model.seq_len

        @seq_len.setter
        def seq_len(self, value: int) -> None:
            self.model.seq_len = value

        @property
        def pred_len(self) -> int:
            """Forecast timesteps owned by DLinear."""
            return self.model.pred_len

        @pred_len.setter
        def pred_len(self, value: int) -> None:
            self.model.pred_len = value

        @property
        def channels(self) -> int:
            """Observed variables owned by DLinear."""
            return self.model.channels

        @channels.setter
        def channels(self, value: int) -> None:
            self.model.channels = value

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.model = config.model.make()

    @override
    def forward(self, media: Tensor, **kwargs: object) -> Tensor:
        """Pass observed histories through unchanged, ignoring loss-only fields."""
        del kwargs
        return self.model(media)


class ObservedHistory(nn.Module):
    """Mix the observed channels through a replaceable nonlinear block."""

    class Config(Fig["ObservedHistory"]):
        seq_len: int = 96
        """Observed timesteps per history."""

        pred_len: int = 64
        """Forecast timesteps."""

        channels: int = 2
        """Variables visible to the method."""

        block: Makeable[nn.Module] = field(
            default_factory=lambda: SwiGLU.Config(channels_hidden=128, bias=True),
        )
        """Nonlinear map over the observed history."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.block,
                "channels_in",
                self.seq_len * self.channels,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.block,
                "channels_out",
                self.pred_len * self.channels,
                protocol=ChannelsOut,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.seq_len, config.pred_len, config.channels) <= 0:
            raise ValueError("Forecast geometry must be positive.")
        self.config = config
        self.block = config.block.make()

    @override
    def forward(self, media: Tensor, **kwargs: object) -> Tensor:
        """Predict every observed variable at each future timestep."""
        del kwargs
        result = cast("Tensor", self.block(media.flatten(start_dim=1)))
        assert isinstance(result, Tensor)
        return result.reshape(len(media), self.config.pred_len, self.config.channels)


class ObservedNeuralForecast:
    """Fit an observed-variable model through PRIML's training lifecycle."""

    class Config(Fig["ObservedNeuralForecast"]):
        step: TrainStep.Config = field(
            default_factory=lambda: TrainStep.Config[Makeable[nn.Module]](
                model=ObservedDLinear.Config(),
                optimizer=PartialConfig(torch.optim.Adam, lr=1e-3),
                loss=SimpleLoss.Config(loss_fn=mse),
                train_budget_steps=1200,
                compile=None,
                parallelism=NoParallel.Config(device="cpu"),
            ),
        )
        """Model, optimizer, and shared training controls."""

        batch_size: int = 32
        """Training histories per microbatch."""

        validation_every: int = 50
        """Successful optimizer attempts between checkpoint comparisons."""

        sample_seed: int = 606
        """Training-index stream, independent of data generation."""

        @property
        def seq_len(self) -> int:
            """Observed timesteps, owned by the model."""
            return cast(ObservedGeometry, self.step.model).seq_len

        @seq_len.setter
        def seq_len(self, value: int) -> None:
            cast(ObservedGeometry, self.step.model).seq_len = value

        @property
        def pred_len(self) -> int:
            """Forecast timesteps, owned by the model."""
            return cast(ObservedGeometry, self.step.model).pred_len

        @pred_len.setter
        def pred_len(self, value: int) -> None:
            cast(ObservedGeometry, self.step.model).pred_len = value

        @property
        def channels(self) -> int:
            """Visible variables, owned by the model."""
            return cast(ObservedGeometry, self.step.model).channels

        @channels.setter
        def channels(self, value: int) -> None:
            cast(ObservedGeometry, self.step.model).channels = value

    def __init__(self, config: Config) -> None:
        if not isinstance(config.step.model, ObservedGeometry):
            raise TypeError("The model must declare its observed forecast geometry.")
        budget = config.step.train_budget_steps
        if not math.isfinite(budget) or budget < 1 or budget != int(budget):
            raise ValueError("The optimizer budget must be a positive whole count.")
        if min(config.batch_size, config.validation_every) <= 0:
            raise ValueError("Batch and validation counts must be positive.")
        if (
            config.step.train_budget_sec != math.inf
            or config.step.train_budget_epochs != math.inf
        ):
            raise ValueError("This study counts updates, not time or epochs.")
        if config.step.dtype_autocast is not None:
            raise ValueError("The observed float32 loss does not support autocast.")
        self.config = config
        self.step = config.step.make()
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        """Normalize on training only and restore the best validation state."""
        del dynamics
        if self.step.global_step:
            raise RuntimeError("Build a fresh forecaster for every fit.")
        cfg = self.config
        for windows in (train, validation):
            if windows.media.shape[1:] != (
                cfg.seq_len,
                cfg.channels,
            ) or windows.label.shape[1:] != (cfg.pred_len, cfg.channels):
                raise ValueError("Observed data does not match the model geometry.")
        self.mean, self.scale = train.channel_stats()
        generator = torch.Generator().manual_seed(cfg.sample_seed)
        best_loss, best_update = math.inf, 0
        best_state: TrainStep.StateDict | None = None
        for _ in range(
            int(cfg.step.train_budget_steps) * cfg.step.accumulate_grad_batches,
        ):
            indices = torch.randint(
                len(train.media),
                (cfg.batch_size,),
                generator=generator,
            )
            self.step.train_step(
                **self.step.preprocess_batch(
                    {
                        "media": (
                            (train.media[indices] - self.mean) / self.scale
                        ).float(),
                        "label": (
                            (train.label[indices] - self.mean) / self.scale
                        ).float(),
                    },
                ),
            )
            if not self.step.accumulation_steps and (
                self.step.global_step % cfg.validation_every == 0
                or self.step.global_step == cfg.step.train_budget_steps
            ):
                with torch.no_grad():
                    loss = float(
                        (
                            (self.predict(validation.media) - validation.label)
                            / self.scale
                        )
                        .square()
                        .mean(),
                    )
                if math.isfinite(loss) and loss < best_loss:
                    best_loss, best_update = loss, self.step.global_step
                    best_state = deepcopy(self.step.state_dict())
        if best_state is None:
            self.mean = self.scale = None
            raise ValueError("No finite validation checkpoint was found.")
        attempts, skipped = self.step.global_step, self.step.skipped_steps
        self.step.load_state_dict(best_state)
        return {
            "attempted_updates": attempts,
            "skipped_updates": skipped,
            "best_update": best_update,
            "validation_scaled_mse": best_loss,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Use the shared evaluation path and return original observation units."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit the observed forecaster before predicting.")
        with torch.no_grad():
            prediction = self.step.call_eval(
                **self.step.preprocess_batch(
                    {"media": ((media - self.mean) / self.scale).float()},
                ),
            )
        assert isinstance(prediction, Tensor)
        return prediction.to(media) * self.scale + self.mean


class ObservedRidge:
    """Fit one affine history-to-future map across the training episodes."""

    class Config(Fig["ObservedRidge"]):
        penalty: float = 0.01
        """Ridge penalty in normalized history coordinates."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.penalty) or config.penalty <= 0:
            raise ValueError("The ridge penalty must be finite and positive.")
        self.config = config
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None
        self.target_mean: Tensor | None = None
        self.weights: Tensor | None = None
        self.shape: tuple[int, int] = (0, 0)

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        """Fit without reading validation targets or any test episode."""
        del validation, dynamics
        features = train.media.flatten(start_dim=1).double()
        self.mean = features.mean(0)
        self.scale = features.std(0, correction=0).clamp_min(1e-6)
        features = (features - self.mean) / self.scale
        targets = train.label.flatten(start_dim=1).double()
        self.target_mean = targets.mean(0)
        self.weights = torch.linalg.solve(
            features.T @ features
            + self.config.penalty * torch.eye(features.shape[1], dtype=features.dtype),
            features.T @ (targets - self.target_mean),
        )
        self.shape = (train.label.shape[1], train.label.shape[2])
        return {"attempted_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Apply the pooled affine map to an unseen observed history."""
        if (
            self.weights is None
            or self.mean is None
            or self.scale is None
            or self.target_mean is None
        ):
            raise RuntimeError("Fit the observed ridge model before predicting.")
        features = (media.flatten(start_dim=1).double() - self.mean) / self.scale
        return (
            (features @ self.weights + self.target_mean)
            .reshape(len(media), *self.shape)
            .to(media)
        )


class RecentPhysics:
    """Estimate a cubic force law from visible history, then roll it forward."""

    class Config(Fig["RecentPhysics"]):
        visible_steps: int = 24
        """Newest observed states used for parameter identification."""

        penalty: float = 1e-7
        """Ridge penalty after scaling the local force columns."""

        lower: tuple[float, float, float] = (0.05, 0.0, 0.0)
        """Lower physical bounds for stiffness, damping, and cubic restoration."""

        upper: tuple[float, float, float] = (4.0, 1.0, 2.0)
        """Upper physical bounds; these are shared across all tested domains."""

    def __init__(self, config: Config) -> None:
        if (
            config.visible_steps < 7
            or not math.isfinite(config.penalty)
            or config.penalty <= 0
        ):
            raise ValueError(
                "Identification needs at least seven states and a positive penalty.",
            )
        if any(
            not math.isfinite(a + b) or a < 0 or b <= a
            for a, b in zip(config.lower, config.upper, strict=True)
        ):
            raise ValueError("Physical parameter bounds must be finite and ordered.")
        self.config = config
        self.dynamics: RegimeDynamics | None = None
        self.pred_len = 0

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        """Take only time geometry; the equation form is a declared prior."""
        del validation
        self.dynamics = dynamics
        self.pred_len = train.label.shape[1]
        return {"attempted_updates": 0}

    def identify(self, media: Tensor) -> tuple[Tensor, Tensor]:
        """Fit coefficients without a change time, hidden velocity, or future target.

        Five-point finite differences use only observed cells. The solve and
        integration use float64 because acceleration divides by the time step
        squared. The missing velocity is inferred from backward differences.

        """
        if self.dynamics is None:
            raise RuntimeError("Fit the physics control before predicting.")
        if media.ndim != 3 or media.shape[1] < 7 or media.shape[2] not in (1, 2):
            raise ValueError(
                "Expected at least seven position-only or state observations.",
            )
        history = media[:, -self.config.visible_steps :].double()
        step = self.dynamics.config.step_size
        position = history[..., 0]
        x = position[:, 2:-2]
        velocity = (
            position[:, :-4]
            - 8 * position[:, 1:-3]
            + 8 * position[:, 3:-1]
            - position[:, 4:]
        ) / (12 * step)
        if media.shape[2] == 2:
            v = history[..., 1]
            velocity = v[:, 2:-2]
            acceleration = (v[:, :-4] - 8 * v[:, 1:-3] + 8 * v[:, 3:-1] - v[:, 4:]) / (
                12 * step
            )
        else:
            acceleration = (
                -position[:, 4:]
                + 16 * position[:, 3:-1]
                - 30 * x
                + 16 * position[:, 1:-3]
                - position[:, :-4]
            ) / (12 * step**2)
        features = -torch.stack((x, velocity, x.pow(3)), dim=-1)
        scale = features.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        normalized = features / scale
        gram = normalized.mT @ normalized
        coefficients = torch.linalg.solve(
            gram + self.config.penalty * torch.eye(3, dtype=gram.dtype),
            normalized.mT @ acceleration[..., None],
        ).squeeze(-1) / scale.squeeze(1)
        coefficients = coefficients.clamp(
            min=torch.tensor(self.config.lower),
            max=torch.tensor(self.config.upper),
        )
        return self.observed_state(media), coefficients

    def observed_state(self, media: Tensor) -> Tensor:
        """Infer the newest state using only coordinates in the observed history."""
        if self.dynamics is None:
            raise RuntimeError("Fit the physics control before predicting.")
        if media.ndim != 3 or media.shape[1] < 7 or media.shape[2] not in (1, 2):
            raise ValueError(
                "Expected at least seven position-only or state observations.",
            )
        position = media[..., 0].double()
        velocity = (
            media[:, -1, 1].double()
            if media.shape[2] == 2
            else (
                25 * position[:, -1]
                - 48 * position[:, -2]
                + 36 * position[:, -3]
                - 16 * position[:, -4]
                + 3 * position[:, -5]
            )
            / (12 * self.dynamics.config.step_size)
        )
        return torch.stack((position[:, -1], velocity), dim=-1)

    def predict(self, media: Tensor) -> Tensor:
        """Predict observed coordinates using parameters inferred from this history."""
        if self.dynamics is None:
            raise RuntimeError("Fit the physics control before predicting.")
        state, coefficients = self.identify(media)
        return self.dynamics.forecast(
            state,
            coefficients=coefficients,
            steps=self.pred_len,
        )[..., : media.shape[-1]].to(media)


class FixedPhysics(RecentPhysics):
    """Keep the initial-state inference but never fit the missing dynamics."""

    class Config(Makes["FixedPhysics"], RecentPhysics.Config):
        coefficients: tuple[float, float, float] = (1.0, 0.15, 0.0)
        """Fixed approximate stiffness, damping, and cubic force."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if (
            any(not math.isfinite(v) or v < 0 for v in config.coefficients)
            or config.coefficients[0] <= 0
        ):
            raise ValueError("The approximate force coefficients must be physical.")
        self.coefficients = config.coefficients

    @override
    def identify(self, media: Tensor) -> tuple[Tensor, Tensor]:
        """Use the same observed state estimate and the declared approximate law."""
        state = self.observed_state(media)
        return state, torch.tensor(self.coefficients, dtype=state.dtype).expand(
            len(media),
            -1,
        )


class LastObservation:
    """Repeat the newest observed value as a simple forecasting control."""

    class Config(Fig["LastObservation"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.pred_len = 0

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        """Record the output length without fitting any parameters."""
        del validation, dynamics
        self.pred_len = train.label.shape[1]
        return {"attempted_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Return a constant future from observed coordinates alone."""
        if not self.pred_len:
            raise RuntimeError("Fit the persistence control before predicting.")
        return media[:, -1:].expand(-1, self.pred_len, -1).clone()
