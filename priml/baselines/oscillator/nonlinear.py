"""Compare nonlinear history models without supplying the delayed equation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, Protocol, Self, cast, override

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.oscillator.data import STATE_CHANNELS, Windows
from priml.baselines.oscillator.train_step import (
    FitReport,
    ForecastGeometry,
    ForecastMethod,
)
from priml.cost import Cost, cost
from priml.loss.simple_loss import SimpleLoss, mse
from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.swiglu import SwiGLU
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.baselines.oscillator.data import Oscillator


class HistoryKernel(Protocol):
    """Compare two batches of normalized observed histories."""

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """Return a (left samples, right samples) kernel matrix."""
        ...


class PolynomialKernel:
    """Include history interactions up to a set degree through a polynomial kernel."""

    class Config(Fig["PolynomialKernel"]):
        degree: int = 2
        """Largest interaction order represented by the kernel."""

        offset: float = 1.0
        """Constant that keeps lower-order features in the kernel."""

    def __init__(self, config: Config) -> None:
        if config.degree < 1 or not math.isfinite(config.offset) or config.offset < 0:
            raise ValueError(
                "The kernel needs a positive degree and nonnegative offset.",
            )
        self.config = config

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """Return normalized polynomial similarities for flattened histories."""
        return (self.config.offset + left @ right.T / left.shape[1]).pow(
            self.config.degree,
        )


class KernelCorrection:
    """Fit a nonlinear history correction through a small dual ridge solve."""

    class Config(Fig["KernelCorrection"]):
        kernel: Makeable[HistoryKernel] = field(default_factory=PolynomialKernel.Config)
        """Similarity function; the default includes quadratic interactions."""

        penalty: float = 1e-3
        """Fixed ridge penalty; no validation or test tuning is performed."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.penalty) or config.penalty <= 0:
            raise ValueError("The kernel penalty must be finite and positive.")
        self.config = config
        self.kernel = config.kernel.make()
        self.simulator: Oscillator | None = None
        self.features: Tensor | None = None
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None
        self.target_mean: Tensor | None = None
        self.weights: Tensor | None = None
        self.horizon = 0

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        """Fit residuals using only training histories and targets.

        Args:
          train: Reference trajectories used for the solve and normalization.
          validation: Unused; the penalty is fixed.
          synthetic: Unused.
          simulator: Supplies the uncorrected forecast.

        Returns:
          report: Zero gradient updates; the method uses one direct solve.

        """
        del validation, synthetic
        self.simulator = simulator
        self.horizon = train.label.shape[1]
        features = train.media.flatten(start_dim=1).double()
        self.mean = features.mean(dim=0)
        self.scale = features.std(dim=0, correction=0).clamp_min(1e-6)
        self.features = (features - self.mean) / self.scale
        target = (
            (train.label - simulator.forecast_history(train.media, steps=self.horizon))
            .flatten(start_dim=1)
            .double()
        )
        self.target_mean = target.mean(dim=0)
        gram = self.kernel(self.features, self.features)
        self.weights = torch.linalg.solve(
            gram + self.config.penalty * torch.eye(len(gram), dtype=gram.dtype),
            target - self.target_mean,
        )
        return {"updates": 0, "best_update": 0, "pretrain_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Add the learned nonlinear correction to the supplied simulator.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Corrected forecasts in the input dtype.

        Raises:
          RuntimeError: ``fit`` has not run.

        """
        if (
            self.simulator is None
            or self.features is None
            or self.mean is None
            or self.scale is None
            or self.weights is None
            or self.target_mean is None
        ):
            raise RuntimeError("Fit the kernel correction before predicting.")
        features = (media.flatten(start_dim=1).double() - self.mean) / self.scale
        correction = (
            self.kernel(features, self.features) @ self.weights + self.target_mean
        )
        return self.simulator.forecast_history(
            media,
            steps=self.horizon,
        ) + correction.to(media.dtype).reshape(len(media), self.horizon, STATE_CHANNELS)


class HistoryNetwork(nn.Module):
    """Forecast from the full history using a replaceable nonlinear block."""

    class Config(Fig["HistoryNetwork"]):
        seq_len: int = 96
        """Observed timesteps per trajectory."""

        pred_len: int = 48
        """Future timesteps per trajectory."""

        block: Makeable[nn.Module] = field(
            default_factory=lambda: SwiGLU.Config(channels_hidden=64),
        )
        """Network over flattened history; owns its hidden width and activation."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.block,
                "channels_in",
                self.seq_len * STATE_CHANNELS,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.block,
                "channels_out",
                self.pred_len * STATE_CHANNELS,
                protocol=ChannelsOut,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one flattened history per trajectory through the chosen block.

            Args:
              seq_len: Observed timesteps; must match the model's history.
              batch_size: Trajectories per batch.
              dtype: Activation dtype; ``None`` uses torch's default.
              **kwargs: Forwarded to the block's cost.

            Returns:
              cost: Block operations and parameters; reshaping uses no arithmetic.

            Raises:
              ValueError: The supplied history length does not match the model.

            """
            if seq_len != self.seq_len:
                raise ValueError("Cost history length must match the model geometry.")
            return cost(
                self.block,
                seq_len=1,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.seq_len, config.pred_len) <= 0:
            raise ValueError("History and forecast lengths must be positive.")
        self.config = config
        self.block = config.block.make()

    @override
    def forward(self, media: Tensor, **kwargs: object) -> Tensor:
        """Map the observed history to a complete future window.

        Args:
          media: Histories shaped (trajectories, seq_len, 2).
          **kwargs: Extra batch values passed through the common train step.

        Returns:
          prediction: Forecasts shaped (trajectories, pred_len, 2).

        Raises:
          ValueError: The input does not match the configured history geometry.

        """
        del kwargs
        if media.ndim != 3 or media.shape[1:] != (self.config.seq_len, STATE_CHANNELS):
            raise ValueError("Input does not match the history network geometry.")
        output = cast(Tensor, self.block(media.flatten(start_dim=1)))
        return output.reshape(len(media), self.config.pred_len, STATE_CHANNELS)


class NeuralFitReport(FitReport):
    """Successful updates and attempt counts before restoring the selected checkpoint."""

    attempted_updates: int
    """Optimizer attempts before restoring the selected checkpoint."""

    skipped_updates: int
    """Attempts that performed no optimizer update."""

    best_attempt: int
    """Attempt index of the checkpoint; best_update counts successful updates."""


class NeuralForecast:
    """Train a nonlinear control through PRIML's shared training lifecycle."""

    class Config(Fig["NeuralForecast"]):
        step: TrainStep.Config[HistoryNetwork.Config] = field(
            default_factory=lambda: TrainStep.Config(
                model=HistoryNetwork.Config(),
                optimizer=PartialConfig(torch.optim.Adam, lr=1e-3),
                loss=SimpleLoss.Config(loss_fn=mse),
                train_budget_steps=300,
                compile=None,
            ),
        )
        """Model, optimizer, and supported shared training controls."""

        batch_size: int = 8
        """Reference trajectories drawn per microbatch."""

        validation_every: int = 25
        """Optimizer attempts between validation checks."""

        sample_seed: int = 606
        """Independent random stream for sampling training trajectories."""

        @property
        def seq_len(self) -> int:
            """Observed timesteps, owned by the model config."""
            return int(self.step.model.seq_len)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.

        @seq_len.setter
        def seq_len(self, value: int) -> None:
            self.step.model.seq_len = value

        @property
        def pred_len(self) -> int:
            """Future timesteps, owned by the model config."""
            return int(self.step.model.pred_len)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.

        @pred_len.setter
        def pred_len(self, value: int) -> None:
            self.step.model.pred_len = value

        @property
        def updates(self) -> int:
            """Optimizer-attempt budget and schedule horizon."""
            return int(self.step.train_budget_steps)

        @updates.setter
        def updates(self, value: int) -> None:
            self.step.train_budget_steps = value

    def __init__(self, config: Config) -> None:
        budget = config.step.train_budget_steps
        if not math.isfinite(budget) or budget <= 0 or budget != int(budget):
            raise ValueError(
                "The network needs a finite positive integer update budget.",
            )
        if min(config.batch_size, config.validation_every) <= 0:
            raise ValueError("Batch size and validation interval must be positive.")
        # ``fit`` counts attempts itself, so time and epoch budgets would be
        # ignored; bf16 autocast fails in this float32 loss's backward pass.
        unsupported = {
            "dtype_autocast": config.step.dtype_autocast is not None,
            "train_budget_sec": config.step.train_budget_sec != math.inf,
            "train_budget_epochs": config.step.train_budget_epochs != math.inf,
        }
        for name, enabled in unsupported.items():
            if enabled:
                raise ValueError(f"The history network does not support {name}.")
        self.config = config
        self.step = config.step.make()
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> NeuralFitReport:
        """Fit on reference data and choose a full training state using validation.

        Args:
          train: Reference windows used for training and normalization.
          validation: Independent windows used to choose a checkpoint.
          synthetic: Unused; this control has no pretraining stage.
          simulator: Unused; the network sees only observed windows.

        Returns:
          report: Successful updates, attempts, selected checkpoint, and validation loss.

        Raises:
          RuntimeError: The method was already fitted.
          ValueError: Input geometry is wrong or no finite checkpoint is found.

        """
        del synthetic, simulator
        if self.step.global_step:
            raise RuntimeError("Build a fresh network for each run.")
        cfg = self.config
        for windows in (train, validation):
            if windows.media.shape[1:] != (
                cfg.seq_len,
                STATE_CHANNELS,
            ) or windows.label.shape[1:] != (cfg.pred_len, STATE_CHANNELS):
                raise ValueError("Data does not match the history network geometry.")
        self.mean, self.scale = train.channel_stats()
        generator = torch.Generator().manual_seed(cfg.sample_seed)
        best_loss, best_update, best_attempt = math.inf, 0, 0
        best_state: TrainStep.StateDict | None = None
        for _ in range(cfg.updates * cfg.step.accumulate_grad_batches):
            indices = torch.randint(
                len(train.media),
                (cfg.batch_size,),
                generator=generator,
            )
            self.step.train_step(
                **self.step.preprocess_batch(
                    {
                        "media": (train.media[indices] - self.mean) / self.scale,
                        "label": (train.label[indices] - self.mean) / self.scale,
                    },
                ),
            )
            update = self.step.global_step
            if not self.step.accumulation_steps and (
                update % cfg.validation_every == 0 or update == cfg.updates
            ):
                loss = float(
                    ((self.predict(validation.media) - validation.label) / self.scale)
                    .square()
                    .mean(),
                )
                if math.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_update = update - self.step.skipped_steps
                    best_attempt = update
                    best_state = deepcopy(self.step.state_dict())
        if best_state is None:
            self.mean = self.scale = None
            raise ValueError("No finite validation checkpoint was found.")
        attempts, skipped = self.step.global_step, self.step.skipped_steps
        self.step.load_state_dict(best_state)
        return {
            "updates": attempts - skipped,
            "attempted_updates": attempts,
            "skipped_updates": skipped,
            "best_attempt": best_attempt,
            "best_update": best_update,
            "pretrain_updates": 0,
            "validation_scaled_mse": best_loss,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Use the shared evaluation path, including EMA when configured.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Forecasts on the device of ``media``.

        Raises:
          RuntimeError: ``fit`` has not run.

        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit the network before predicting.")
        prediction = self.step.call_eval(
            **self.step.preprocess_batch({"media": (media - self.mean) / self.scale}),
        )
        assert isinstance(prediction, Tensor)
        return prediction.to(media.device) * self.scale + self.mean


class HistoryAblation:
    """Remove older cues during both fitting and prediction."""

    class Config(Fig["HistoryAblation"]):
        method: Makeable[ForecastMethod] = field(
            default_factory=KernelCorrection.Config,
        )
        """Method receiving only the retained part of each history."""

        visible_steps: int = 32
        """Newest observations retained; older entries are replaced by zero."""

        seq_len: int = 96
        """Observed timesteps, forwarded to a wrapped method with a geometry."""

        pred_len: int = 48
        """Future timesteps, forwarded to a wrapped method with a geometry."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.method,
                "seq_len",
                self.seq_len,
                protocol=ForecastGeometry,
            )
            propagate_attr(
                self.method,
                "pred_len",
                self.pred_len,
                protocol=ForecastGeometry,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.visible_steps <= 0:
            raise ValueError("The visible history must be positive.")
        self.config = config
        self.method = config.method.make()

    def mask(self, media: Tensor) -> Tensor:
        """Return a copy with old observations removed."""
        result = media.clone()
        result[:, : max(0, media.shape[1] - self.config.visible_steps)] = 0
        return result

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        """Fit the wrapped method after masking every supplied history.

        Args:
          train: Reference windows; only their histories are masked.
          validation: Model-selection windows, masked the same way.
          synthetic: Cheap-simulator windows, masked the same way.
          simulator: Passed through unchanged.

        Returns:
          report: The wrapped method's fit report.

        """
        return self.method.fit(
            Windows(media=self.mask(train.media), label=train.label),
            validation=Windows(
                media=self.mask(validation.media),
                label=validation.label,
            ),
            synthetic=Windows(media=self.mask(synthetic.media), label=synthetic.label),
            simulator=simulator,
        )

    def predict(self, media: Tensor) -> Tensor:
        """Apply the same history mask used during fitting."""
        return self.method.predict(self.mask(media))
