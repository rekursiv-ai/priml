"""Tests for TrainLoop."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, override

import functools
import json
import logging
import math
import tempfile
import time

from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import torch


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter

from configgle import Fig, Makeable, PartialConfig

from priml.custom_types import CheckpointableProtocol
from priml.data.custom_types import DatasetProtocol
from priml.data.dummy import DummyDataset
from priml.loss.custom_types import LossOutput
from priml.metrics.binary_accuracy import BinaryAccuracy
from priml.runtime import SingleProcess, runtime_initialized
from priml.timer import CheckpointableStepTimer
from priml.train import train_loop
from priml.train.checkpointing import Checkpointer, _agreed_across_ranks
from priml.train.custom_types import TrainStepOutput
from priml.train.parallelism import NoParallel
from priml.train.profiling import PhaseTimer, TorchProfiling
from priml.train.tracker import FileTracker
from priml.train.train_loop import (
    EvalTimeLimitError,
    TrainLoop,
    _set_loader_epoch,
)
from priml.train.train_step import TrainStep


_RuntimeEvents = list[str]
"""Recorded lifecycle events; provided per-test via the ``runtime_events`` fixture."""


@pytest.fixture
def runtime_events() -> _RuntimeEvents:
    """Hermetic per-test recorder for runtime/model lifecycle events."""
    return []


class _RecordingRuntime:
    """Runtime that records TrainLoop lifecycle events into a caller-supplied list."""

    _events: _RuntimeEvents | None = None
    """Class-level handle set by tests via ``set_events`` before make()."""

    @classmethod
    def set_events(cls, events: _RuntimeEvents) -> None:
        """Bind a per-test event recorder; must be called before ``Config.make()``."""
        cls._events = events

    class Config(Fig["_RecordingRuntime"]): ...

    def __init__(self, config: Config) -> None:
        del config
        self.device = torch.device("cpu")

    def initialize(self) -> None:
        """Record runtime initialization."""
        if self._events is not None:
            self._events.append("runtime_initialize")

    def destroy(self) -> None:
        """Record runtime cleanup."""
        if self._events is not None:
            self._events.append("runtime_destroy")


class _RuntimeAwareModel(nn.Module):
    """Model that records construction after runtime initialization."""

    class Config(Fig["_RuntimeAwareModel"], make_with_kwargs=True):
        in_features: int = -1
        """Input feature count."""
        out_features: int = -1
        """Output feature count."""

    def __init__(self, in_features: int, out_features: int) -> None:
        if _RecordingRuntime._events is not None:
            _RecordingRuntime._events.append("model_init")
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    @override
    def forward(
        self,
        media: Tensor,
        **_kwargs: Any,
    ) -> Tensor:
        """Forward pass."""
        return self.linear(media)


class _HasTimer(Protocol):
    """Minimal timer attribute protocol for test narrowing."""

    timer: PhaseTimer


class _WarmupDataset:
    """Dataset exposing two eval batches for compile-warmup tests."""

    class Config(Fig["_WarmupDataset"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> list[dict[str, Tensor]]:
        """Return an unused train loader."""
        return []

    def eval_dataloader(self) -> list[dict[str, Tensor]]:
        """Return eval batches."""
        return [
            {"media": torch.tensor([[1.0]]), "label": torch.tensor([1])},
            {"media": torch.tensor([[2.0]]), "label": torch.tensor([0])},
        ]

    def state_dict(self) -> dict[str, Any]:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataset state for checkpointing."""
        del state_dict


class _ScopedEvalDataset:
    """Dataset that records bounded versus full eval loader use."""

    class Config(Fig["_ScopedEvalDataset"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.timer_epoch = CheckpointableStepTimer()
        self.eval_scopes: list[str] = []

    def train_dataloader(self) -> list[dict[str, Tensor]]:
        """Return train batches."""
        return [
            {"media": torch.tensor([[1.0, 0.0]]), "label": torch.tensor([0])},
            {"media": torch.tensor([[0.0, 1.0]]), "label": torch.tensor([1])},
        ]

    def eval_dataloader(self) -> list[dict[str, Tensor]]:
        """Return bounded eval batches."""
        self.eval_scopes.append("bounded")
        return [{"media": torch.tensor([[1.0, 0.0]]), "label": torch.tensor([0])}]

    def full_eval_dataloader(self) -> list[dict[str, Tensor]]:
        """Return full eval batches."""
        self.eval_scopes.append("full")
        return [
            {"media": torch.tensor([[1.0, 0.0]]), "label": torch.tensor([0])},
            {"media": torch.tensor([[0.0, 1.0]]), "label": torch.tensor([1])},
        ]

    def state_dict(self) -> dict[str, Any]:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataset state for checkpointing."""
        del state_dict


class _WeightedEvalDataset:
    """Dataset exposing uneven eval batches with valid example counts."""

    class Config(Fig["_WeightedEvalDataset"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> list[dict[str, Tensor]]:
        """Return an unused train loader."""
        return []

    def eval_dataloader(self) -> list[dict[str, Any]]:
        """Return eval batches with uneven valid counts."""
        return [
            {"media": torch.tensor([[1.0]]), "valid_count": 4},
            {"media": torch.tensor([[0.0]]), "valid_count": 1},
        ]

    def state_dict(self) -> dict[str, Any]:
        """Get dataset state for checkpointing."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataset state for checkpointing."""
        self.timer_epoch.load_state_dict(state_dict["timer_epoch"])


class _WeightedEvalStep:
    """Train step that returns the batch media as an eval scalar metric."""

    class Config(Fig["_WeightedEvalStep"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.global_step = 0
        self.local_step = 0

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return the batch unchanged."""
        return batch

    def eval_loss(self, **preprocessed_batch: Any) -> dict[str, Any]:
        """Return loss and metric equal to the batch media scalar."""
        media = cast(Tensor, preprocessed_batch["media"])
        return {"loss": media.flatten(), "model": media, "metrics": {"score": media}}

    def train_loss(self, **preprocessed_batch: Any) -> dict[str, Any]:
        """Delegate to eval_loss."""
        return self.eval_loss(**preprocessed_batch)

    def train_step(self, **preprocessed_batch: Any) -> dict[str, Tensor]:
        """Unused train step."""
        del preprocessed_batch
        return {"loss": torch.zeros(1), "model": torch.zeros(1, 1)}

    def state_dict(self) -> dict[str, Any]:
        """Get train-step state."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load train-step state."""
        self.global_step = int(state_dict["global_step"])


class _WarmupStep:
    """Train step that records eval warmup calls."""

    class Config(Fig["_WarmupStep"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.global_step = 0
        self.local_step = 0
        self.eval_calls: list[Tensor] = []

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return the batch unchanged."""
        return batch

    def eval_loss(self, **preprocessed_batch: Any) -> dict[str, Tensor]:
        """Record eval batch media."""
        media = cast(Tensor, preprocessed_batch["media"])
        self.eval_calls.append(media.clone())
        return {"loss": torch.zeros(1), "model": media}

    def train_loss(self, **preprocessed_batch: Any) -> dict[str, Tensor]:
        """Delegate to eval_loss."""
        return self.eval_loss(**preprocessed_batch)

    def train_step(self, **preprocessed_batch: Any) -> dict[str, Tensor]:
        """Advance one train step."""
        del preprocessed_batch
        self.global_step += 1
        self.local_step += 1
        return {"loss": torch.zeros(1), "model": torch.zeros(1, 1)}

    def call_eval(self, **preprocessed_batch: Any) -> Tensor:
        """Return eval logits placeholder."""
        return cast(Tensor, preprocessed_batch["media"])

    def on_epoch_end(self) -> None:
        """No-op epoch hook."""

    def state_dict(self) -> dict[str, Any]:
        """Get train-step state."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load train-step state."""
        self.global_step = int(state_dict["global_step"])


class _RecordingTracker:
    """Tracker that records metric payloads for TrainLoop tests."""

    class Config(Fig["_RecordingTracker"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.metrics_by_step: list[tuple[dict[str, Any], int]] = []
        self.notes: list[str] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Record prefixed metrics for assertions."""
        self.metrics_by_step.append(
            ({f"{prefix}{key}": value for key, value in metrics.items()}, step),
        )

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """Ignore image payloads in scalar tracker tests."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """Record run notes for assertions."""
        self.notes.append(notes)

    def close(self) -> None:
        """Record tracker cleanup."""
        self.closed = True


class _ExtrasMetric:
    """Metric whose ``compute`` carries a non-scalar ``extras`` payload."""

    class Config(Fig["_ExtrasMetric"]):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def update(self, logits: Tensor, **batch: Any) -> None:
        """Ignore batches; the payload is constant."""
        del logits, batch

    def compute(self) -> dict[str, Any]:
        """Return one scalar plus a non-scalar ``extras`` payload."""
        return {"metric_score": 2.0, "extras": {"payload": ("opaque",)}}

    def reset(self) -> None:
        """Stateless."""

    def state_dict(self) -> dict[str, Any]:
        """Stateless."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Stateless."""
        del state_dict


def _cross_entropy(output: Tensor, *, label: Tensor, **_kwargs: Any) -> LossOutput:
    """Wrapper for cross_entropy that extracts label from kwargs."""
    return {"loss": torch.nn.functional.cross_entropy(output, label, reduction="none")}


class _LinearModel(nn.Module):
    """Simple linear model for testing."""

    class Config(Fig["_LinearModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @override
    def forward(self, media: Tensor, **_kwargs: Any) -> Tensor:
        """Forward pass."""
        return self.linear(media)


def test_train_loop_basic():
    """Test TrainLoop runs without errors."""
    torch.manual_seed(42)

    # Create config
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config = TrainLoop.Config(
        step=step_config,
        dataset=DummyDataset.Config(
            input_shape=(2,),
            num_classes=2,
            num_samples=20,
            batch_size=4,
            device="cpu",
        ),
    )
    config.metrics = {}
    config.max_steps = 10
    config.num_steps_eval = 5
    config.seed = 42

    with tempfile.TemporaryDirectory() as tmp:
        assert isinstance(config.checkpointing, Checkpointer.Config)
        config.checkpointing.base_dir = "/"
        config.checkpointing.working_dir = Path(tmp)
        config.checkpointing.save_every = 5
        loop = config.make()
        loop.train()
        assert loop.step.global_step == 10
        assert (Path(tmp) / "step_00000005.pt").exists()
        assert (Path(tmp) / "step_00000010.pt").exists()


def test_train_loop_with_max_epochs():
    """Test TrainLoop stops at max_epochs."""
    torch.manual_seed(42)

    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config = TrainLoop.Config(
        step=step_config,
        dataset=DummyDataset.Config(
            input_shape=(2,),
            num_classes=2,
            num_samples=20,
            batch_size=4,
            device="cpu",
        ),
    )
    config.max_steps = 1000  # High limit
    config.max_epochs = 2  # Should stop after 2 epochs (10 steps)
    config.num_steps_eval = 5
    config.seed = 42

    with tempfile.TemporaryDirectory() as tmp:
        assert isinstance(config.checkpointing, Checkpointer.Config)
        config.checkpointing.base_dir = "/"
        config.checkpointing.working_dir = Path(tmp)
        loop = config.make()
        loop.train()
        assert loop.current_epoch == 2


def _binary_cross_entropy_with_logits(
    output: Tensor,
    *,
    label: Tensor,
    **_kwargs: Any,
) -> LossOutput:
    """Wrapper for binary_cross_entropy_with_logits that extracts label from kwargs."""
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            output,
            label,
            reduction="none",
        ),
    }


class _LogisticModel(nn.Module):
    """Logistic regression model for binary classification."""

    class Config(Fig["_LogisticModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @override
    def forward(self, media: Tensor, **_kwargs: Any) -> Tensor:
        """Forward pass."""
        return self.linear(media).squeeze(-1)


class _BinaryDataset:
    """Binary classification dataset for testing."""

    class Config(Fig["_BinaryDataset"], make_with_kwargs=True):
        pass

    def __init__(self):
        torch.manual_seed(42)
        self.timer_epoch = CheckpointableStepTimer()
        n_samples = 100
        n_features = 2

        # Generate linearly separable data: y = 1 if x1 + x2 > 0 else 0
        self.X = torch.randn(n_samples, n_features)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).float()

    def train_dataloader(self):
        """Get training dataloader."""
        dataset = TensorDataset(self.X, self.y)

        def collate_train(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
            return {
                "media": torch.stack([x[0] for x in batch]),
                "label": torch.stack([x[1] for x in batch]),
            }

        return DataLoader(
            dataset,
            batch_size=20,
            shuffle=True,
            collate_fn=collate_train,
        )

    def eval_dataloader(self):
        """Get eval dataloader."""
        dataset = TensorDataset(self.X, self.y)

        def collate_eval(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
            return {
                "media": torch.stack([x[0] for x in batch]),
                "label": torch.stack([x[1] for x in batch]),
            }

        return DataLoader(
            dataset,
            batch_size=20,
            shuffle=False,
            collate_fn=collate_eval,
        )

    def state_dict(self) -> dict[str, Any]:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataset state from checkpoint."""
        _ = state_dict


def test_train_loop_comprehensive():
    """Comprehensive test with checkpointing and metrics."""
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"

        # Create config with checkpointing and metrics
        step_config = TrainStep.Config()
        step_config.model = _LogisticModel.Config(in_features=2, out_features=1)
        step_config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
        step_config.loss = PartialConfig(_binary_cross_entropy_with_logits)
        step_config.parallelism = NoParallel.Config(device="cpu")
        step_config.compile = None
        config = TrainLoop.Config(step=step_config, dataset=_BinaryDataset.Config())
        config.metrics = {"accuracy": BinaryAccuracy.Config()}
        config.max_steps = 20
        config.num_steps_eval = 10
        config.checkpointing = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        config.seed = 42

        # First training run - train for 20 steps
        loop1 = config.make()
        loop1.train()

        # Check training completed
        assert loop1.step.global_step == 20

        # Check checkpoints were created
        assert (checkpoint_dir / "step_00000010.pt").exists()
        assert (checkpoint_dir / "step_00000020.pt").exists()

        # Get final loss
        ds1: Any = loop1.dataset
        final_loss_result_1 = loop1.step.eval_loss(media=ds1.X, label=ds1.y)
        final_loss_1 = final_loss_result_1["loss"]

        # Second training run - should resume from step 20
        step_config2 = TrainStep.Config()
        step_config2.model = _LogisticModel.Config(in_features=2, out_features=1)
        step_config2.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
        step_config2.loss = PartialConfig(_binary_cross_entropy_with_logits)
        step_config2.parallelism = NoParallel.Config(device="cpu")
        step_config2.compile = None
        config2 = TrainLoop.Config(step=step_config2, dataset=_BinaryDataset.Config())
        config2.metrics = {"accuracy": BinaryAccuracy.Config()}
        config2.max_steps = 30
        config2.num_steps_eval = 10
        config2.checkpointing = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        config2.seed = 42
        assert isinstance(config2.checkpointing, Checkpointer.Config)
        config2.checkpointing.resume = True

        loop2 = config2.make()

        # Should have resumed from step 20
        assert loop2.step.global_step == 20

        # Should have same loss as end of first run
        ds2: Any = loop2.dataset
        resumed_loss_result = loop2.step.eval_loss(media=ds2.X, label=ds2.y)
        torch.testing.assert_close(final_loss_1, resumed_loss_result["loss"])

        # Continue training to step 30
        loop2.train()
        assert loop2.step.global_step == 30

        # Check that loss improved
        final_loss_result_2 = loop2.step.eval_loss(media=ds2.X, label=ds2.y)
        final_loss_2 = final_loss_result_2["loss"]
        assert final_loss_2.mean().item() < final_loss_1.mean().item()

        # Check that accuracy improved
        eval_metrics = loop2.eval()
        assert "accuracy_accuracy" in eval_metrics
        assert eval_metrics["accuracy_accuracy"] > 0.75  # Should get > 75% accuracy


def _eval_only_step_config() -> TrainStep.Config:
    step_config = TrainStep.Config()
    step_config.model = _LogisticModel.Config(in_features=2, out_features=1)
    step_config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step_config.loss = PartialConfig(_binary_cross_entropy_with_logits)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    return step_config


def test_eval_only_loads_checkpoint_and_skips_training():
    """eval_only loads a checkpoint, evals once, and runs no training step."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"

        train_cfg = TrainLoop.Config(
            step=_eval_only_step_config(),
            dataset=_BinaryDataset.Config(),
        )
        train_cfg.metrics = {"accuracy": BinaryAccuracy.Config()}
        train_cfg.max_steps = 20
        train_cfg.num_steps_eval = float("inf")
        train_cfg.checkpointing = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        train_cfg.seed = 42
        train_loop = train_cfg.make()
        train_loop.train()
        assert train_loop.step.global_step == 20
        assert (checkpoint_dir / "step_00000020.pt").exists()

        # eval_only run: loads the step-20 checkpoint, evals, no training.
        eval_cfg = TrainLoop.Config(
            step=_eval_only_step_config(),
            dataset=_BinaryDataset.Config(),
        )
        eval_cfg.metrics = {"accuracy": BinaryAccuracy.Config()}
        eval_cfg.max_steps = 20
        eval_cfg.num_steps_eval = float("inf")
        eval_cfg.checkpointing = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        eval_cfg.seed = 42
        eval_cfg.eval_only = True

        eval_loop = eval_cfg.make()
        # resume defaults on, so the checkpoint loads without eval_only touching
        # the checkpointer's read policy.
        assert eval_loop.step.global_step == 20  # loaded, not trained
        eval_loop.train()  # dispatches to the eval-only path
        # Still 20: eval_only must not advance the optimizer step.
        assert eval_loop.step.global_step == 20
        # No new checkpoint is written by eval_only.
        assert not (checkpoint_dir / "step_00000030.pt").exists()


def test_train_raises_when_no_finite_stop_condition() -> None:
    """Construction refuses a training config with every stop bound infinite."""
    config = TrainLoop.Config(
        step=_eval_only_step_config(),
        dataset=_BinaryDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = math.inf
    config.max_epochs = math.inf
    config.max_time = math.inf
    config.seed = 42

    with pytest.raises(ValueError, match="no finite stop condition"):
        config.make()


def _resume_table_config(checkpoint_dir: Path):
    """A minimal trainable config writing into ``checkpoint_dir``."""
    cfg = TrainLoop.Config(
        step=_eval_only_step_config(),
        dataset=_BinaryDataset.Config(),
    )
    cfg.metrics = {}
    cfg.max_steps = 20
    cfg.num_steps_eval = float("inf")
    cfg.eval_every_epoch = False
    cfg.checkpointing = Checkpointer.Config(
        base_dir="/",
        working_dir=checkpoint_dir,
        save_every=10,
        keep_last_n=5,
    )
    cfg.seed = 42
    return cfg


def _seed_checkpoints(checkpoint_dir: Path) -> None:
    """Train a fresh run to populate ``checkpoint_dir`` with step_10, step_20."""
    cfg = _resume_table_config(checkpoint_dir)
    assert isinstance(cfg.checkpointing, Checkpointer.Config)
    cfg.checkpointing.resume = False
    loop = cfg.make()
    loop.train()
    assert (checkpoint_dir / "step_00000020.pt").exists()


def test_resume_latest_uses_largest_when_checkpoints_exist():
    """resume=True, resume_step=-1, checkpoints exist -> load largest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = -1
        loop = cfg.make()
        assert loop.step.global_step == 20  # largest on disk


def test_resume_latest_starts_fresh_when_no_checkpoints():
    """resume=True, resume_step=-1, no checkpoints -> start at 0, no error.

    -1 means "resume from latest if any"; an empty dir implies resume from
    nothing, i.e. a fresh start. Must NOT raise (the prior contract did).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "nope"  # never written to
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = -1
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_resume_explicit_step_loads_that_step():
    """resume=True, resume_step>0 present -> load exactly that step."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = 10
        # step_20 exists and a later save would re-mint it; that overwrite
        # guard is exercised elsewhere -- here we isolate the explicit load.
        cfg.checkpointing.allow_checkpoint_overwrite = True
        loop = cfg.make()
        assert loop.step.global_step == 10


def test_resume_explicit_step_missing_raises():
    """resume=True, resume_step>0 absent -> hard error (named step not found)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)  # has 10, 20 -- not 999

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = 999
        with pytest.raises(RuntimeError, match="requested but not found"):
            cfg.make()


def test_resume_explicit_step_no_checkpoints_raises():
    """resume=True, resume_step>0, empty dir -> hard error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "nope"
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = 5
        with pytest.raises(RuntimeError, match="requested but not found"):
            cfg.make()


def test_resume_false_starts_at_zero_into_empty_dir():
    """resume=False, empty dir -> start at 0 (resume_step ignored)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "fresh"
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = False
        cfg.checkpointing.resume_step = 10  # must be ignored
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_resume_defaults_to_true():
    """The resume default is True (preemption-restart is the common case)."""
    finalized = TrainLoop.Config().finalize()
    assert isinstance(finalized.checkpointing, Checkpointer.Config)
    assert finalized.checkpointing.resume is True


def test_fresh_run_refuses_to_overwrite_existing_checkpoints():
    """A from-scratch run whose saves would land on existing files is refused.

    The colleague's footgun: a fresh run reusing a name silently overwrote the
    prior run's checkpoints. With save_every=10/max_steps=20, this run would
    mint step_10/step_20 -- both already on disk -- so it must refuse.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)  # steps 10, 20

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = False
        cfg.checkpointing.allow_checkpoint_overwrite = False
        with pytest.raises(RuntimeError, match="would overwrite existing"):
            cfg.make()


def test_rewind_resume_refuses_to_overwrite_newer_checkpoints():
    """Resuming an older step is refused when later saves would clobber newer.

    Orthogonal to resume: resuming step 10 then training to 20 would mint
    step_20 over the existing step_20. allow_checkpoint_overwrite=False
    refuses, and the check is upfront (at make()), not after training.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)  # steps 10, 20

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = 10  # rewind: start_step=10, step_20 is newer
        cfg.checkpointing.allow_checkpoint_overwrite = False
        with pytest.raises(RuntimeError, match="would overwrite existing"):
            cfg.make()


def test_resume_latest_does_not_trip_overwrite_guard():
    """Resuming the latest checkpoint never collides -- no save step exceeds it."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)  # steps 10, 20

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = True
        cfg.checkpointing.resume_step = -1  # start_step=20; no save step in (20, 20]
        cfg.checkpointing.allow_checkpoint_overwrite = False
        loop = cfg.make()
        assert loop.step.global_step == 20


def test_fresh_run_into_off_cadence_dir_is_allowed():
    """A populated dir whose steps this run never re-mints does not trip."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "step_5.pt").write_bytes(b"x")  # off the save cadence

        cfg = _resume_table_config(checkpoint_dir)  # save_every=10 -> 10, 20
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = False
        cfg.checkpointing.allow_checkpoint_overwrite = False
        loop = cfg.make()  # 5 is never a save step -> no collision
        assert loop.step.global_step == 0


def test_allow_checkpoint_overwrite_permits_clobber():
    """allow_checkpoint_overwrite=True lets a run mint over existing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume = False
        cfg.checkpointing.allow_checkpoint_overwrite = True
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_allow_checkpoint_overwrite_defaults_to_false():
    """allow_checkpoint_overwrite must default False -- clobbering is opt-in."""
    finalized = TrainLoop.Config().finalize()
    assert isinstance(finalized.checkpointing, Checkpointer.Config)
    assert finalized.checkpointing.allow_checkpoint_overwrite is False


def test_eval_only_never_trips_overwrite_guard():
    """eval_only writes no checkpoint, so the overwrite guard must not fire.

    Loading an older explicit step (10) while a newer one (20) exists would
    trip the cadence-collision guard for a training run, but eval_only runs no
    training step and saves nothing -- the guard is vacuous and must be skipped.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir)  # steps 10, 20

        cfg = _resume_table_config(checkpoint_dir)
        cfg.eval_only = True
        assert isinstance(cfg.checkpointing, Checkpointer.Config)
        cfg.checkpointing.resume_step = (
            10  # older than latest; would-collide for training
        )
        cfg.checkpointing.allow_checkpoint_overwrite = False
        loop = cfg.make()  # must not raise
        assert loop.step.global_step == 10


def test_available_steps_lists_checkpoints():
    """available_steps surfaces the on-disk checkpoint steps for diagnostics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt = Checkpointer.Config(working_dir=Path(temp_dir), save_every=10).make()
        assert ckpt.available_steps() == []
        (Path(temp_dir) / "step_10.pt").write_bytes(b"x")
        (Path(temp_dir) / "step_4000.pt").write_bytes(b"x")
        assert ckpt.available_steps() == [10, 4000]


def _make_recording_train_loop_config(
    events: _RuntimeEvents,
    base_dir: Path,
    *,
    step_factory: Callable[[], TrainStep.Config] | None = None,
) -> TrainLoop.Config:
    """Build a TrainLoop.Config wired to a recording runtime + minimal model."""
    _RecordingRuntime.set_events(events)
    config = TrainLoop.Config(
        dataset=DummyDataset.Config(
            input_shape=(2,),
            num_classes=2,
            num_samples=4,
            batch_size=2,
            device="cpu",
        ),
    )
    if step_factory is not None:
        config.step = step_factory()
    else:
        step_config = TrainStep.Config()
        model_config = _RuntimeAwareModel.Config()
        model_config.in_features = 2
        model_config.out_features = 2
        step_config.model = model_config
        step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
        step_config.loss = PartialConfig(_cross_entropy)
        step_config.parallelism = NoParallel.Config(device="cpu")
        step_config.compile = None
        config.step = step_config
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 1
    config.num_steps_eval = math.inf
    config.base_dir = base_dir
    config.runtime = _RecordingRuntime.Config()
    return config


def test_eval_weights_scalar_metrics_by_valid_count() -> None:
    """Eval scalar means weight partial batches by valid example count."""
    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False

    loop = config.make()

    assert loop.eval()["score"] == 0.8


def test_eval_fails_when_exceeding_max_eval_time() -> None:
    """An eval pass over its wall-clock budget raises EvalTimeLimitError."""
    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.max_eval_time = 0.0  # any elapsed batch trips the deadline

    loop = config.make()

    with pytest.raises(EvalTimeLimitError, match="max_eval_time"):
        loop.eval()


def test_eval_stop_on_time_limit_publishes_partial_results() -> None:
    """Opt-in data-generation eval stops at the budget instead of raising."""
    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.max_eval_time = 0.0
    config.eval_stop_on_time_limit = True

    loop = config.make()

    assert loop.eval() == {}


def test_eval_respects_generous_max_eval_time() -> None:
    """A large budget leaves eval unaffected (no spurious failure)."""
    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.max_eval_time = 3_600.0

    loop = config.make()

    assert loop.eval()["score"] == 0.8


def test_eval_warmup_runs_configured_eval_batches() -> None:
    """Eval warmup runs eval_loss once per configured batch before training."""
    config = TrainLoop.Config(
        step=cast(Any, _WarmupStep.Config()),
        dataset=_WarmupDataset.Config(),
    )
    config.metrics = {}
    config.checkpointing = None
    config.eval_warmup_batches = 1
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False

    loop = config.make()
    step = cast(_WarmupStep, loop.step)

    assert len(step.eval_calls) == 1
    torch.testing.assert_close(step.eval_calls[0], torch.tensor([[1.0]]))


def test_train_loop_initializes_runtime_before_model_init(
    runtime_events: _RuntimeEvents,
    tmp_path: Path,
) -> None:
    """TrainLoop initializes runtime before constructing the step/model."""
    config = _make_recording_train_loop_config(runtime_events, tmp_path)

    loop = config.make()
    loop.train()

    assert runtime_events == [
        "runtime_initialize",
        "model_init",
        "runtime_destroy",
    ]


def test_train_loop_skips_runtime_init_when_caller_already_initialized(
    runtime_events: _RuntimeEvents,
    tmp_path: Path,
) -> None:
    """If caller initialized the runtime, TrainLoop must not re-init or destroy.

    This is the runtime_initialized()/_owns_runtime contract. When the
    sentinel reports the runtime is already up, ``_owns_runtime`` must be
    False and ``_cleanup`` must not call ``runtime.destroy()``.
    """
    # Use a SingleProcess runtime that the caller initializes explicitly.
    runtime = SingleProcess.Config(device="cpu").make()
    runtime.initialize()
    try:
        assert runtime_initialized(), (
            "regression: SingleProcess.initialize() must flip the global flag"
        )
        config = _make_recording_train_loop_config(runtime_events, tmp_path)
        # Override runtime config with the pre-initialized one's class so the
        # loop tries (and must skip) the init/destroy pair.
        config.runtime = SingleProcess.Config(device="cpu")
        loop = config.make()
        loop.train()
        # Recording runtime is NOT in use here (we swapped to SingleProcess);
        # the assertion is on the borrowed-vs-owned flag.
        assert not loop._owns_runtime, (
            "TrainLoop must not claim ownership when runtime already initialized",
        )
    finally:
        runtime.destroy()


def test_train_loop_releases_runtime_when_init_raises(
    runtime_events: _RuntimeEvents,
    tmp_path: Path,
) -> None:
    """TrainLoop must call runtime.destroy() if __init__ raises after init.

    Otherwise distributed resources (NCCL process group, device mesh) leak
    for the process lifetime. Reproduces by injecting a failing
    ``step.make()``.
    """

    class _FailingStepConfig(TrainStep.Config):
        @override
        def make(self) -> TrainStep:
            raise RuntimeError("boom from step.make()")

    def _bad_step_factory() -> TrainStep.Config:
        step = _FailingStepConfig()
        model_config = _RuntimeAwareModel.Config()
        model_config.in_features = 2
        model_config.out_features = 2
        step.model = model_config
        step.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
        step.loss = PartialConfig(_cross_entropy)
        step.parallelism = NoParallel.Config(device="cpu")
        step.compile = None
        return step

    config = _make_recording_train_loop_config(
        runtime_events,
        tmp_path,
        step_factory=_bad_step_factory,
    )

    with pytest.raises(RuntimeError, match=r"boom from step\.make"):
        config.make()

    # The runtime must have been initialized AND torn down even though the
    # construction failed mid-way.
    assert runtime_events == [
        "runtime_initialize",
        "runtime_destroy",
    ]


class TestFinalize:
    def test_project_working_dir_scopes_leaf_working_directories(self):
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointing = Checkpointer.Config()
        cfg.phase_timer = PhaseTimer.Config(
            base_dir="/",
            working_dir="/explicit/child",
        )

        finalized = cfg.finalize()

        assert finalized.working_dir == Path("/opt/scratch/runs/my_project/exp000")
        assert isinstance(finalized.checkpointing, Checkpointer.Config)
        assert finalized.checkpointing.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/checkpoints"
        )
        assert isinstance(finalized.phase_timer, PhaseTimer.Config)
        assert finalized.phase_timer.working_dir == Path("/explicit/child")

    def test_profiling_output_is_scoped_below_run(self) -> None:
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.profiling = TorchProfiling.Config(torch_profile=False)

        finalized = cfg.finalize()

        assert isinstance(finalized.profiling, TorchProfiling.Config)
        assert finalized.profiling.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/profiling"
        )

    def test_explicit_profiling_base_dir_wins(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "profiling"
        cfg = TrainLoop.Config()
        cfg.profiling = TorchProfiling.Config(
            torch_profile=False,
            base_dir=tmp_path,
            working_dir="/profiling",
        )

        finalized = cfg.finalize()

        assert isinstance(finalized.profiling, TorchProfiling.Config)
        assert finalized.profiling.working_dir == working_dir

    def test_checkpoint_working_dir_is_scoped_below_run(self):
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointing = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointing, Checkpointer.Config)
        assert finalized.checkpointing.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/checkpoints"
        )

    def test_explicit_base_dir_propagates_to_checkpointing(
        self,
        tmp_path: Path,
    ) -> None:
        scratch = tmp_path / "explicit-scratch"
        cfg = TrainLoop.Config()
        cfg.base_dir = scratch
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointing = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointing, Checkpointer.Config)
        assert finalized.checkpointing.working_dir == (
            scratch / "runs/my_project/exp000/checkpoints"
        )

    def test_checkpoint_base_dir_not_overwritten_when_set(self):
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointing = Checkpointer.Config(
            base_dir="/",
            working_dir="/custom/dir",
        )
        finalized = cfg.finalize()
        assert isinstance(finalized.checkpointing, Checkpointer.Config)
        assert finalized.checkpointing.working_dir == Path("/custom/dir")

    def test_checkpoint_working_dir_without_run_name(self):
        cfg = TrainLoop.Config()
        cfg.checkpointing = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointing, Checkpointer.Config)
        assert finalized.checkpointing.working_dir == Path(
            "/opt/scratch/runs/checkpoints"
        )

    def test_doc_is_sent_to_tracker_via_log_notes(self):
        """``doc`` reaches the built tracker through ``log_notes`` at init time.

        The launcher only sets ``doc``; TrainLoop forwards it to the tracker's
        ``log_notes`` after make (no config-internals mutation).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointing = None
            cfg.tracker = _RecordingTracker.Config()
            cfg.doc = "Hypothesis: X. Change: Y. Result: TODO."
            loop = cfg.make()
            tracker = cast(_RecordingTracker, loop.tracker)
            assert tracker.notes == ["Hypothesis: X. Change: Y. Result: TODO."]

    def test_empty_doc_does_not_call_log_notes(self):
        """No description means no notes call -- the tracker keeps its own."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointing = None
            cfg.tracker = _RecordingTracker.Config()
            cfg.doc = ""
            loop = cfg.make()
            tracker = cast(_RecordingTracker, loop.tracker)
            assert tracker.notes == []

    def test_doc_without_tracker_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointing = None
            cfg.tracker = None
            cfg.doc = "docstring note"
            loop = cfg.make()  # must not raise
            assert loop.tracker is None


def test_phase_timer_instruments_data_load_and_model_init():
    """PhaseTimer records data_load and model_init phases."""
    torch.manual_seed(42)
    config = TrainLoop.Config()
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config.step = step_config
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=20,
        batch_size=4,
        device="cpu",
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 5
    config.num_steps_eval = math.inf
    config.seed = 42
    config.phase_timer = PhaseTimer.Config(enabled=True)

    loop = config.make()
    loop.train()

    s = loop.phase_timer.summary()
    assert "data_load" in s
    assert "model_init" in s
    assert s["data_load"] > 0
    assert s["model_init"] > 0


def _make_step_logging_loop_config() -> TrainLoop.Config:
    """Minimal CPU loop that logs a per-step loss line on every step."""
    config = TrainLoop.Config()
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config.step = step_config
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=8,
        batch_size=4,
        device="cpu",
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 2
    config.num_steps_eval = math.inf
    config.num_steps_log = 1  # log every step
    config.seed = 42
    return config


def test_result_line_accounts_for_every_second(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RESULT decomposes wall time, so no clock can be moved unseen.

    A budget the run reports itself is only auditable when the parts add up:
    training seconds charged, seconds excluded from the charge, evaluation,
    and everything else must reconstruct the wall clock.
    """
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    config = _make_step_logging_loop_config()
    config.num_steps_eval = 1
    loop = config.make()
    # A measurable eval: at zero seconds every term is zero and the sum holds
    # however the parts are counted, so the invariant would not bite.
    inner_eval = loop.eval

    def slow_eval() -> dict[str, Any]:
        time.sleep(0.05)
        return inner_eval()

    monkeypatch.setattr(loop, "eval", slow_eval)

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        loop.train()

    result = next(r.message for r in caplog.records if r.message.startswith("RESULT:"))
    parts = dict(
        field.split("=", 1) for field in result.removeprefix("RESULT:").split(" | ")
    )
    seconds = {
        key: float(parts[key].removesuffix("s"))
        for key in ("train_sec", "train_unbilled_sec", "eval_sec", "other_sec", "time")
    }
    assert seconds["eval_sec"] >= 0.1  # two evals, 0.05s each
    accounted = (
        seconds["train_sec"]
        + seconds["train_unbilled_sec"]
        + seconds["eval_sec"]
        + seconds["other_sec"]
    )
    assert accounted == pytest.approx(seconds["time"], abs=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_the_runtime_device_places_the_model() -> None:
    """The runtime's device decides placement when nothing narrower is set.

    One process names its device ONCE, on the runtime. A placement default
    that probes the hardware itself answers a question the runtime already
    answered, so a CPU-pinned run silently trains on the GPU -- and two fields
    that resolve "which device" independently can only agree by luck.
    """
    config = _make_step_logging_loop_config()
    config.runtime = SingleProcess.Config(device="cpu")
    config.step.parallelism = NoParallel.Config()  # unset: defer to the runtime
    loop = config.copy_tree().finalize().make()

    assert loop.runtime.device == torch.device("cpu")
    assert next(loop.step.model.parameters()).device == torch.device("cpu")


def test_result_line_shows_seconds_a_budget_declined_to_charge(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup seconds land in ``train_unbilled_sec``, not in ``other_sec``.

    A budgeted recipe reads its own clock, so the loop must decompose against
    the clock it keeps itself -- otherwise the excluded seconds are invisible,
    which is exactly the lever the account exists to expose: widening the
    warmup would buy free training and move nothing the line reports.
    """
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_step_logging_loop_config()
    config.num_steps_eval = 100  # finite (RESULT fires) but no cadence eval in 2 steps
    loop = config.make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=100.0)
    )
    # A budgeted step: it charges only part of the second its update took.
    loop.step.elapsed_sec = 0.4
    monkeypatch.setattr(loop, "_train_elapsed", lambda: loop.step.elapsed_sec)

    def timed_eval() -> dict[str, Any]:
        clock.now += 5.0
        return {"score": 1.0}

    monkeypatch.setattr(loop, "eval", timed_eval)

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        loop.train()

    result = next(r.message for r in caplog.records if r.message.startswith("RESULT:"))
    parts = dict(
        field.split("=", 1) for field in result.removeprefix("RESULT:").split(" | ")
    )
    seconds = {k: float(v.removesuffix("s")) for k, v in parts.items() if k != "steps"}
    # Two 1s steps after the 100s compile; the step charged 0.4s of them.
    assert seconds["train_sec"] == pytest.approx(0.4)
    assert seconds["train_unbilled_sec"] == pytest.approx(0.6)
    assert seconds["eval_sec"] == pytest.approx(5.0)
    assert seconds["other_sec"] == pytest.approx(100.0)
    assert seconds["time"] == pytest.approx(106.0)


def test_per_step_loss_logged_on_rank_zero(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-step loss narrative emits on rank 0 (or non-distributed)."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    loop = _make_step_logging_loop_config().make()

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        loop.train()
    assert any("loss=" in r.message and "Step " in r.message for r in caplog.records)


def test_startup_logs_before_entering_train_step(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup diagnostics emit before the first train_step call.

    At DEBUG: they name the phase a wedged run is stuck in, so they cost
    nothing on a healthy console and are turned on when one hangs.
    """
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    loop = _make_step_logging_loop_config().make()

    with caplog.at_level(logging.DEBUG, logger="priml.train.train_loop"):
        loop.train()

    messages = [r.message for r in caplog.records]
    enter_index = next(
        i for i, m in enumerate(messages) if "Entering train step 1/" in m
    )
    step_index = next(i for i, m in enumerate(messages) if "Step 1/" in m)
    assert enter_index < step_index


def test_per_step_log_includes_step_metrics(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-step console line carries the step's scalar metrics.

    So training accuracy is greppable from the job log alone when the tracker
    backend is unavailable, not only on the dashboard.
    """
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    loop = _make_step_logging_loop_config().make()

    inner = loop.step.train_step

    def _with_metrics(**batch: Any) -> dict[str, Any]:
        out = dict(inner(**batch))
        out["metrics"] = {"cell_accuracy": 0.875, "act_steps": 4.0}
        return out

    monkeypatch.setattr(loop.step, "train_step", _with_metrics)

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        loop.train()
    step_lines = [
        r.message
        for r in caplog.records
        if "Step " in r.message and "loss=" in r.message
    ]
    assert step_lines
    assert any(
        "cell_accuracy=0.8750" in m and "act_steps=4.0000" in m for m in step_lines
    )


def test_per_step_loss_suppressed_on_non_zero_rank(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero ranks suppress the per-step loss line (avoid N-fold console spam)."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)
    loop = _make_step_logging_loop_config().make()

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        loop.train()
    assert not any(
        "loss=" in r.message and "Step " in r.message for r in caplog.records
    )


def test_logged_train_loss_is_all_reduced_before_rank_zero_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank participates in logged tensor reductions on log steps."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)
    calls: list[tuple[float, ...]] = []

    def all_reduce(tensor: Tensor) -> None:
        calls.append(tuple(float(value) for value in tensor.tolist()))
        tensor.mul_(8)

    loop = _make_step_logging_loop_config().make()

    def _with_metrics(**batch: Any) -> dict[str, Any]:
        del batch
        # Advance the step counter the way the real step does -- by charging
        # the timer global_step reads -- since it is a read-only property.
        cast(TrainStep, loop.step).timer_step.global_count += 1
        return {
            "loss": torch.tensor([1.0]),
            "model": torch.zeros(1, 2),
            "metrics": {
                "grad_norm": torch.tensor(2.0),
                "param_norm": torch.tensor(3.0),
            },
        }

    monkeypatch.setattr(loop.step, "train_step", _with_metrics)

    loader = loop.dataset.train_dataloader()
    iterator = iter(loader)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    for _ in range(2):
        loop._do_train_step(loop.step.preprocess_batch(next(iterator)))

    assert calls == [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)]


def test_error_logs_on_non_zero_rank(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ERROR-level logging is never rank-gated -- each rank records its own."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)

    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        logging.getLogger("priml.train.train_loop").error("rank-local failure")
    assert any("rank-local failure" in r.message for r in caplog.records)


class _FakeClock:
    """Deterministic ``time.perf_counter`` stand-in advanced explicitly by tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _make_max_time_loop_config() -> TrainLoop.Config:
    """Loop config for max_time stop tests: 10s budget, ample steps, no evals."""
    config = _make_step_logging_loop_config()
    config.max_steps = 100
    config.max_time = 10.0
    config.eval_every_epoch = False
    return config


def _timed_train_step(
    loop: TrainLoop, clock: _FakeClock, *, first_step_time: float
) -> Callable[..., dict[str, Any]]:
    """Wrap the loop's train_step to advance ``clock`` by a scripted duration.

    The first step takes ``first_step_time`` (simulating the backward-graph
    compile); every later step takes 1s.
    """
    inner = loop.step.train_step

    def timed_step(**batch: Any) -> dict[str, Any]:
        clock.now += first_step_time if loop.step.global_step == 0 else 1.0
        return inner(**batch)

    return timed_step


def test_max_time_kind_defaults_to_wall() -> None:
    """max_time stays a wall-clock cap unless a config opts into "train"."""
    assert TrainLoop.Config().max_time_kind == "wall"


def test_max_time_wall_counts_first_step_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default "wall": the first step's compile time eats the whole budget."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    loop = _make_max_time_loop_config().make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=100.0)
    )

    loop.train()

    assert loop.step.global_step == 1  # the 100s "compile" alone exceeds 10s


def test_max_time_train_kind_excludes_first_step_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kind="train": the budget clock starts after the first (compile) step."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_max_time_loop_config()
    config.max_time_kind = "train"
    loop = config.make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=100.0)
    )

    loop.train()

    # The 100s first step is excluded; 1s steps 2..11 then fill the 10s budget.
    assert loop.step.global_step == 11


def test_max_time_train_kind_excludes_cadence_eval_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kind="train": cadence evals pause the budget clock, not consume it."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_max_time_loop_config()
    config.max_time_kind = "train"
    config.num_steps_eval = 5
    loop = config.make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=1.0)
    )
    eval_calls: list[int] = []

    def timed_eval() -> dict[str, Any]:
        eval_calls.append(loop.step.global_step)
        clock.now += 50.0  # each eval alone would blow the 10s budget
        return {"score": 1.0}

    monkeypatch.setattr(loop, "eval", timed_eval)

    loop.train()

    # Two 50s cadence evals are excluded, so the run reaches the same step as
    # an eval-free one; the trailing call is the (post-budget) final eval.
    assert loop.step.global_step == 11
    assert eval_calls == [5, 10, 11]


def test_max_time_train_kind_excludes_epoch_boundary_eval_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kind="train": epoch-boundary evals pause the budget clock too."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_max_time_loop_config()
    config.max_time_kind = "train"
    config.eval_every_epoch = True  # 8 samples / batch 4 -> eval every 2 steps
    loop = config.make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=1.0)
    )
    eval_calls: list[int] = []

    def timed_eval() -> dict[str, Any]:
        eval_calls.append(loop.step.global_step)
        clock.now += 50.0  # each eval alone would blow the 10s budget
        return {"score": 1.0}

    monkeypatch.setattr(loop, "eval", timed_eval)

    loop.train()

    # Five 50s epoch-boundary evals are excluded; only the 1s steps count.
    assert loop.step.global_step == 11
    assert eval_calls == [2, 4, 6, 8, 10]


def test_train_metrics_include_pure_train_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trackers get ``train/elapsed``: pure-train seconds, compile excluded."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_step_logging_loop_config()
    config.tracker = _RecordingTracker.Config()
    loop = config.make()
    monkeypatch.setattr(
        loop.step, "train_step", _timed_train_step(loop, clock, first_step_time=100.0)
    )

    loop.train()

    tracker = cast(_RecordingTracker, loop.tracker)
    elapsed_by_step = {
        step: payload["train/elapsed"]
        for payload, step in tracker.metrics_by_step
        if "train/elapsed" in payload
    }
    wall_by_step = {
        step: payload["train/time_since_start"]
        for payload, step in tracker.metrics_by_step
        if "train/time_since_start" in payload
    }
    # Step 1's 100s compile is excluded from the pure-train clock (rebased to
    # zero); step 2's 1s counts. The wall clock keeps counting both.
    assert elapsed_by_step == {1: 0.0, 2: 1.0}
    assert wall_by_step == {1: 100.0, 2: 101.0}


def test_phase_timer_disabled_no_overhead():
    """When disabled, no phases recorded."""
    torch.manual_seed(42)
    config = TrainLoop.Config()
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config.step = step_config
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=20,
        batch_size=4,
        device="cpu",
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_steps = 5
    config.num_steps_eval = math.inf
    config.seed = 42

    loop = config.make()
    loop.train()

    s = loop.phase_timer.summary()
    assert "data_load" not in s
    assert "model_init" not in s


def test_phase_timer_passed_to_step():
    """TrainLoop passes timer to step."""
    torch.manual_seed(42)
    config = TrainLoop.Config()
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config.step = step_config
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=20,
        batch_size=4,
        device="cpu",
    )
    config.metrics = {}
    config.max_steps = 1
    config.num_steps_eval = math.inf
    config.seed = 42
    config.phase_timer = PhaseTimer.Config(enabled=True)

    loop = config.make()
    assert hasattr(loop.step, "timer")
    assert cast(_HasTimer, loop.step).timer is loop.phase_timer


# -- Regression tests (Issue#286 trainloop + checkpoint-loop group) ----------


def _simple_dummy_dataset() -> DummyDataset.Config:
    """The default small CPU dummy dataset for simple-loop helpers."""
    return DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=8,
        batch_size=4,
        device="cpu",
    )


def _make_simple_loop_config(
    tmp: str,
    *,
    dataset: Makeable[DatasetProtocol] | None = None,
) -> TrainLoop.Config:
    """Build a minimal CPU TrainLoop.Config over the supplied dataset."""
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config = TrainLoop.Config(
        step=step_config,
        dataset=dataset if dataset is not None else _simple_dummy_dataset(),
    )
    config.metrics = {}
    config.checkpointing = Checkpointer.Config(
        base_dir="/",
        working_dir=Path(tmp),
        save_every=5,
    )
    config.max_steps = 10
    config.seed = 42
    return config


def test_no_eval_or_checkpoint_at_step_zero() -> None:
    """T-015: no eval / checkpoint should fire before the first train step."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 1
        config.num_steps_eval = 5  # step 0 would be a multiple of 5
        loop = config.make()

        eval_steps: list[int] = []
        orig_eval = loop.eval

        def spy_eval() -> dict[str, Any]:
            eval_steps.append(loop.step.global_step)
            return orig_eval()

        loop.eval = spy_eval  # ty: ignore[invalid-assignment] -- test spy patches a bound method
        loop.train()

        # No eval should have run while global_step == 0.
        assert 0 not in eval_steps, f"eval ran at step 0: {eval_steps}"
        # No checkpoint dir for step 0.
        assert not (Path(tmp) / "step_00000000.pt").exists()


def test_resume_does_not_eval_or_checkpoint_before_first_new_step() -> None:
    """A cadence-step resume must advance training before save/eval side effects."""
    with tempfile.TemporaryDirectory() as tmp:
        initial = _make_simple_loop_config(tmp)
        initial.max_steps = 5
        initial.num_steps_eval = math.inf
        initial.make().train()
        assert (Path(tmp) / "step_00000005.pt").exists()
        config = _make_simple_loop_config(tmp)
        config.max_steps = 6
        config.num_steps_eval = 5
        loop = config.make()
        assert loop.step.global_step == 5

        maybe_save_steps: list[int] = []
        checkpointing = loop.checkpointing
        assert checkpointing is not None
        original_maybe_save = checkpointing.maybe_save

        def spy_maybe_save(target: CheckpointableProtocol, step: int) -> bool:
            maybe_save_steps.append(step)
            return original_maybe_save(target, step)

        checkpointing.maybe_save = spy_maybe_save  # ty: ignore[invalid-assignment] -- test spy patches a bound method
        eval_steps: list[int] = []
        orig_eval = loop.eval

        def spy_eval() -> dict[str, Any]:
            eval_steps.append(loop.step.global_step)
            return orig_eval()

        loop.eval = spy_eval  # ty: ignore[invalid-assignment] -- test spy patches a bound method
        loop.train()

        assert loop.step.global_step == 6
        assert 5 not in maybe_save_steps
        assert 5 not in eval_steps


def test_train_step_logs_gpu_memory_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train-step tracker logs process CUDA memory peaks when CUDA is available."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 1
        config.num_steps_eval = math.inf
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        original_train_step = loop.step.train_step

        def train_step_with_cuda_metrics(**batch: Any) -> TrainStepOutput:
            result = original_train_step(**batch)
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(
                torch.cuda,
                "max_memory_allocated",
                lambda: 2_000_000_000,
            )
            monkeypatch.setattr(
                torch.cuda,
                "max_memory_reserved",
                lambda: 3_000_000_000,
            )
            return result

        monkeypatch.setattr(loop.step, "train_step", train_step_with_cuda_metrics)
        loop.train()

    train_log = next(
        metrics
        for metrics, _step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert train_log["train/total_loss"] >= 0.0
    assert train_log["train/step_time"] >= 0.0
    assert train_log["train/time_since_start"] >= 0.0
    assert train_log["train/gpu_mem_allocated_gb"] == 2.0
    assert train_log["train/gpu_mem_reserved_gb"] == 3.0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU guard is host-specific")
def test_train_step_omits_gpu_memory_on_cpu_tracker() -> None:
    """CPU train-step tracker logs keep running without CUDA memory keys."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 1
        config.num_steps_eval = math.inf
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    train_log = next(
        metrics
        for metrics, _step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert "train/gpu_mem_allocated_gb" not in train_log
    assert "train/gpu_mem_reserved_gb" not in train_log


def test_train_tracker_logging_respects_num_steps_log_cadence() -> None:
    """Train metrics upload on the num_steps_log cadence, not every step.

    Logging hundreds of history keys every sub-second step floods the tracker's
    ingestion and lags the dashboard tens of thousands of steps behind live.
    With num_steps_log=5 over 12 steps, train rows land at steps 1, 5, 10 only.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 12
        config.num_steps_eval = math.inf
        config.num_steps_log = 5
        config.early_train_log_steps = 0
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    train_steps = sorted(
        step
        for metrics, step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert train_steps == [1, 5, 10]


def test_accumulation_logs_once_per_update_not_once_per_microbatch() -> None:
    """Logging runs per microbatch; the cadences count optimizer updates.

    Without the gate every pass of an accumulation reports the same step, the
    last of them alone carrying that update's metrics -- four identical console
    lines per step here, eight in the nanochat recipe.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Sized so one pass holds whole accumulations: the epoch boundary drops
        # a partial one, so a shorter pass would reset the count every time and
        # never complete an update.
        config = _make_simple_loop_config(
            tmp,
            dataset=DummyDataset.Config(
                input_shape=(2,),
                num_classes=2,
                num_samples=64,
                batch_size=4,
                device="cpu",
            ),
        )
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.step.accumulate_grad_batches = 4
        config.max_steps = 3
        config.num_steps_eval = math.inf
        config.num_steps_log = 1
        config.early_train_log_steps = 100
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    train_steps = sorted(
        step
        for metrics, step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert train_steps == [1, 2, 3]


def test_accumulation_evaluates_once_per_update_not_once_per_microbatch() -> None:
    """The eval cadence counts updates; the loop body runs per microbatch.

    Measured before the guard on a five-minute budget: an eval costing 23
    seconds ran eight times at one step -- once per accumulation pass -- and
    spent three of those five minutes re-scoring identical weights.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(
            tmp,
            dataset=DummyDataset.Config(
                input_shape=(2,),
                num_classes=2,
                num_samples=64,
                batch_size=4,
                device="cpu",
            ),
        )
        config.checkpointing = None
        config.step.accumulate_grad_batches = 4
        config.max_steps = 4
        config.num_steps_eval = 2
        loop = config.make()

        evaluated: list[int] = []
        inner = loop.eval

        def spy() -> dict[str, Any]:
            evaluated.append(loop.step.global_step)
            return inner()

        loop.eval = spy  # ty: ignore[invalid-assignment] -- test spy patches a bound method
        loop.train()

    # Step 2 on the cadence, then the final eval. The while-loop exits once
    # ``max_steps`` is reached, so step 4's cadence eval never comes due.
    assert evaluated == [2, 4]


def test_train_tracker_logs_every_startup_step_then_cadence() -> None:
    """Startup diagnostics log every early step before falling back to cadence."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 12
        config.num_steps_eval = math.inf
        config.num_steps_log = 5
        config.early_train_log_steps = 3
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    train_steps = sorted(
        step
        for metrics, step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert train_steps == [1, 2, 3, 5, 10]


def test_epoch_eval_logs_eval_time_to_tracker() -> None:
    """Epoch-triggered eval logs duration with epoch eval metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 100
        config.max_epochs = 1
        config.num_steps_eval = math.inf
        config.eval_every_epoch = True
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    epoch_logs = [metrics for metrics, step in tracker.metrics_by_step if step == 1]
    assert any("eval/total_loss" in metrics for metrics in epoch_logs)
    assert any(
        isinstance(metrics.get("eval/time"), float) and metrics["eval/time"] >= 0.0
        for metrics in epoch_logs
    )
    assert tracker.closed


def test_step_eval_logs_eval_time_to_tracker() -> None:
    """Step-triggered eval logs duration with eval metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        config.num_steps_eval = 1
        config.eval_every_epoch = False
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    eval_logs = [metrics for metrics, step in tracker.metrics_by_step if step == 1]
    assert any("eval/total_loss" in metrics for metrics in eval_logs)
    assert any(
        isinstance(metrics.get("eval/time"), float) and metrics["eval/time"] >= 0.0
        for metrics in eval_logs
    )
    # Per-batch eval step time is logged like train/step_time.
    assert any(
        isinstance(metrics.get("eval/mean_batch_time"), float)
        and metrics["eval/mean_batch_time"] >= 0.0
        for metrics in eval_logs
    )
    assert tracker.closed


def test_final_eval_uses_same_bounded_loader_as_cadence() -> None:
    """Every eval -- cadence and final -- uses the one (bounded) eval loader.

    There is no separate uncapped final pass: the dataset's own eval config
    (caps or none) decides the eval scope, and the final eval reuses it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        config.num_steps_eval = 1
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = cast(_ScopedEvalDataset, loop.dataset)
    # Cadence eval at step 1, then the final eval; both bounded, never "full".
    assert dataset.eval_scopes == ["bounded", "bounded"]
    assert "full" not in dataset.eval_scopes


def test_phase_timer_summary_logs_before_final_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timing summary prints before the final eval starts."""
    events: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointing = None
        config.max_steps = 2
        # Finite cadence so a final eval runs (num_steps_eval=inf disables eval).
        config.num_steps_eval = 2
        config.eval_every_epoch = False
        config.phase_timer = PhaseTimer.Config(enabled=True)
        loop = config.make()
        dataset = cast(_ScopedEvalDataset, loop.dataset)
        timer = cast(PhaseTimer, loop.phase_timer)
        eval_dataloader = dataset.eval_dataloader
        log_summary = timer.log_summary

        def record_eval_dataloader() -> list[dict[str, Tensor]]:
            events.append("final_eval")
            return eval_dataloader()

        def record_log_summary() -> None:
            before = timer._summary_logged
            log_summary()
            if not before and timer._summary_logged:
                events.append("summary")

        # The cadence eval at step 2 and the final eval both call
        # eval_dataloader; the last call is the post-loop final eval, which must
        # follow the phase-timer summary.
        monkeypatch.setattr(dataset, "eval_dataloader", record_eval_dataloader)
        monkeypatch.setattr(loop.phase_timer, "log_summary", record_log_summary)

        loop.train()

    assert events[-2:] == ["summary", "final_eval"]


@pytest.mark.parametrize("never", [float("inf"), 0])
def test_eval_disabled_when_num_steps_eval_says_never(never: float) -> None:
    """``inf`` and ``0`` both skip every eval, including the final one.

    Throughput/profiling runs care only about train-step timing; an eval (the
    whole eval set) is wasted work that holds the GPU long after the last train
    step. ``0`` is the readable spelling and used to divide by zero in the
    cadence check; ``inf`` is the older one and stays valid.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointing = None
        config.max_steps = 2
        config.num_steps_eval = never
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = cast(_ScopedEvalDataset, loop.dataset)
    assert dataset.eval_scopes == []


def test_num_steps_eval_minus_one_runs_the_final_eval_only() -> None:
    """``-1`` scores once, at the end, with no mid-run cadence.

    The third regime, because "when to score" is two questions. Before it
    existed this was written as a cadence too large to fire -- ``100_000``,
    ``1_000_000_000``, ``max_steps`` -- which breaks silently the moment a run
    grows past the number somebody guessed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointing = None
        config.max_steps = 6
        config.num_steps_eval = -1
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = cast(_ScopedEvalDataset, loop.dataset)
    assert len(dataset.eval_scopes) == 1


def test_final_post_training_eval_logs_to_tracker() -> None:
    """The end-of-training eval reaches the tracker, not just the console.

    The final eval is the run's reported number (post-training pass@K); it must
    land in the tracker at the final step, not only the RESULT log line.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        # Cadence wider than the run (2 % 5 != 0): no mid-run eval fires, so the
        # only eval is the post-training final one (num_steps_eval=inf would
        # disable eval entirely, so it must stay finite).
        config.num_steps_eval = 5
        config.eval_every_epoch = False
        loop = config.make()
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    final_logs = [
        metrics
        for metrics, step in tracker.metrics_by_step
        if step == loop.step.global_step
    ]
    assert any("eval/total_loss" in metrics for metrics in final_logs), (
        "final eval not logged to tracker"
    )


def test_cadence_eval_runs_once_per_optimizer_step() -> None:
    """Gradient accumulation cannot repeat eval at one optimizer step."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointing = None
        config.num_steps_eval = 5
        loop = config.make()
        assert isinstance(loop.step, TrainStep)
        loop.step.timer_step.global_count = 5

        eval_count = 0
        original_eval = loop.eval

        def count_eval() -> dict[str, Any]:
            nonlocal eval_count
            eval_count += 1
            return original_eval()

        loop.eval = count_eval  # ty: ignore[invalid-assignment] -- bound test spy
        loop._maybe_eval()
        loop._maybe_eval()

    assert eval_count == 1


def test_no_post_loop_eval_when_no_training() -> None:
    """T-019: with max_steps=0, the post-loop eval must not run."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 0
        config.num_steps_eval = math.inf
        loop = config.make()

        eval_count = [0]
        orig_eval = loop.eval

        def spy_eval() -> dict[str, Any]:
            eval_count[0] += 1
            return orig_eval()

        loop.eval = spy_eval  # ty: ignore[invalid-assignment] -- test spy patches a bound method
        loop.train()

        assert loop.step.global_step == 0
        assert eval_count[0] == 0, "post-loop eval ran despite zero training"


def test_retention_keeps_last_n_after_training() -> None:
    """Retention prunes to keep_last_n across cadence + forced end-of-run saves."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 25
        config.num_steps_eval = math.inf
        assert isinstance(config.checkpointing, Checkpointer.Config)
        config.checkpointing.save_every = 5
        config.checkpointing.keep_last_n = 2
        loop = config.make()
        loop.train()

        assert loop.checkpointing is not None
        # Saves land at 5,10,15,20,25; retention keeps only the newest two.
        assert loop.checkpointing.available_steps() == [20, 25]


def _make_accum_epoch_loop_config(
    *,
    drop_partial: bool,
    samples: int,
    batch_size: int,
    accumulate: int,
) -> TrainLoop.Config:
    """Loop whose per-epoch micro-batch count leaves a partial accumulation."""
    config = TrainLoop.Config()
    step_config = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    step_config.accumulate_grad_batches = accumulate
    step_config.drop_partial_accumulation_on_epoch_end = drop_partial
    config.step = step_config
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=2,
        num_samples=samples,
        batch_size=batch_size,
        device="cpu",
    )
    config.metrics = {}
    config.checkpointing = None
    config.max_epochs = 1
    config.max_steps = 1000
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.seed = 42
    return config


def test_partial_accumulation_dropped_at_epoch_end_by_default() -> None:
    """T-021: default flushes+discards partial accumulation at epoch boundary.

    Epoch 0 yields 3 micro-batches; with accumulate_grad_batches=8 none reach
    an optimizer step. At the boundary the 3 pending micro-batches must be
    discarded, so the new epoch's first micro-batch starts a *fresh*
    accumulation (count == 1), proving no cross-epoch mixing.
    """
    config = _make_accum_epoch_loop_config(
        drop_partial=True,
        samples=12,
        batch_size=4,
        accumulate=8,
    )
    loop = config.make()
    loop.train()

    assert loop.current_epoch == 1
    assert loop.step.global_step == 0
    # Boundary dropped epoch-0's 3 pending; only epoch-1's first micro-batch
    # remains accumulated.
    assert cast(TrainStep, loop.step).accumulation_steps == 1


def test_partial_accumulation_carries_across_epoch_when_opted_out() -> None:
    """T-021: flag=False carries the partial accumulation across the boundary.

    Epoch 0's 3 pending micro-batches survive the boundary and the new epoch's
    first micro-batch adds to them (count == 4), proving cross-epoch carry.
    """
    config = _make_accum_epoch_loop_config(
        drop_partial=False,
        samples=12,
        batch_size=4,
        accumulate=8,
    )
    loop = config.make()
    loop.train()

    assert loop.current_epoch == 1
    assert cast(TrainStep, loop.step).accumulation_steps == 4
    assert loop.step.global_step == 0


def _collective_skip_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Diverge per-rank verdicts; record the collective decision.

    Only rank 0's verdict is True. A correct collective decision broadcasts
    rank 0's view so every rank agrees, so no rank can return early and strand
    the others at the save barrier.
    """
    del mesh
    rank = torch.distributed.get_rank()
    try:
        skip = _agreed_across_ranks(rank == 0)
        (Path(result_dir) / f"rank_{rank}").write_text("skip" if skip else "save")
    except Exception as e:  # noqa: BLE001  -- surface worker error to parent
        (Path(result_dir) / f"rank_{rank}").write_text(f"FAIL:{e!r}")


@pytest.mark.cli_python_subprocess
def test_force_checkpoint_skip_is_collective(warm_pools: WarmPoolGetter) -> None:
    """F5: the force-checkpoint skip decision must be collective, not per-rank.

    With per-rank ``path.exists()`` the ranks disagree (rank 0 skips, rank 1
    saves), so rank 0 returns before the save barrier and deadlocks rank 1.
    A collective decision makes all ranks take the same branch.
    """
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_collective_skip_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    decisions = {v for k, v in results.items() if k.startswith("rank_")}
    assert decisions in ({"skip"}, {"save"}), results


def test_set_loader_epoch_tolerates_loader_without_dataset() -> None:
    """A loader lacking a ``.dataset`` attribute must be a no-op."""

    class _LoaderWithoutDataset:
        def __iter__(self) -> Iterator[dict[str, Tensor]]:
            return iter(())

    _set_loader_epoch(cast("DataLoader[Any]", _LoaderWithoutDataset()), epoch=3)


def test_dataset_receives_the_step_before_the_first_batch() -> None:
    """A generating dataset is bound to the step, and bound before iterating.

    An on-policy dataset produces its batches by acting with the current
    policy, so the binding must land before any batch is drawn -- otherwise
    the first rollout has no model to act with.
    """
    config = TrainLoop.Config(
        step=cast(Any, _WarmupStep.Config()),
        dataset=_BindingDataset.Config(),
    )
    config.checkpointing = None
    config.max_steps = 1
    loop = config.make()
    dataset = cast(_BindingDataset, loop.dataset)

    assert dataset.bound is loop.step
    assert dataset.batches_before_binding == 0

    loop.train()
    assert dataset.batches_drawn > 0


def test_dataset_without_the_hook_is_left_alone() -> None:
    """A dataset that reads a corpus must not be required to accept a step."""
    config = TrainLoop.Config(
        step=cast(Any, _WarmupStep.Config()),
        dataset=_WarmupDataset.Config(),
    )
    config.checkpointing = None
    config.max_steps = 0
    assert config.make().dataset is not None


class _BindingDataset:
    """Dataset that records when it was handed the train step."""

    class Config(Fig["_BindingDataset"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.timer_epoch = CheckpointableStepTimer()
        self.bound: object = None
        self.batches_drawn = 0
        self.batches_before_binding = 0

    def bind_step(self, step: object) -> None:
        """Record the train step supplied by the loop."""
        self.bound = step

    def train_dataloader(self) -> list[dict[str, Tensor]]:
        """Return one batch, recording whether the step was bound first."""
        if self.bound is None:
            self.batches_before_binding += 1
        self.batches_drawn += 1
        return [{"media": torch.tensor([[1.0]]), "label": torch.tensor([1])}]

    def eval_dataloader(self) -> list[dict[str, Tensor]]:
        """Return one eval batch."""
        return [{"media": torch.tensor([[1.0]]), "label": torch.tensor([1])}]

    def state_dict(self) -> dict[str, Any]:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataset state for checkpointing."""
        del state_dict


def _make_extras_publish_config() -> TrainLoop.Config:
    """Eval-publish config with a payload metric and a recording tracker."""
    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics = {"": _ExtrasMetric.Config()}
    config.tracker = _RecordingTracker.Config()
    config.checkpointing = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    return config


def test_eval_extras_only_passed_on_final_eval() -> None:
    """Cadence evals forward scalars only; the final eval forwards ``extras``.

    The loop owns final-ness: it strips the non-scalar ``extras`` on cadence and
    includes it only on the final eval so a payload-consuming tracker sees it.
    A FileTracker (scalars only) ignores ``extras`` either way.
    """
    loop = _make_extras_publish_config().make()
    tracker = cast(_RecordingTracker, loop.tracker)
    loop.step.global_step = 12

    loop._maybe_eval(force=True)
    assert len(tracker.metrics_by_step) == 1
    cadence, cadence_step = tracker.metrics_by_step[0]
    assert cadence_step == 12
    assert cadence["eval/metric_score"] == 2.0
    assert "eval/extras" not in cadence

    loop._maybe_eval(is_final=True, force=True)
    assert len(tracker.metrics_by_step) == 2
    final, final_step = tracker.metrics_by_step[1]
    assert final_step == 12
    assert final["eval/metric_score"] == 2.0
    assert final["eval/extras"] == {"payload": ("opaque",)}


def test_eval_extras_every_eval_forwards_payload_on_cadence_evals() -> None:
    """``eval_extras_every_eval=True`` forwards ``extras`` on every eval.

    Opt-in for runs whose payload consumer (e.g. an ARC signal-dump tracker)
    wants a per-eval artifact, not just the final one. The consumer then owns
    retention; the loop only stops stripping the payload on cadence evals.
    """
    config = _make_extras_publish_config()
    config.eval_extras_every_eval = True
    loop = config.make()
    tracker = cast(_RecordingTracker, loop.tracker)
    loop.step.global_step = 12

    loop._maybe_eval(force=True)  # A non-final (cadence-style) eval.
    assert len(tracker.metrics_by_step) == 1
    cadence, _ = tracker.metrics_by_step[0]
    assert cadence["eval/metric_score"] == 2.0
    assert cadence["eval/extras"] == {"payload": ("opaque",)}


def test_load_state_dict_can_skip_rng_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval-only checkpoint scoring can load state without restoring RNG."""
    calls: list[object] = []

    def fail_rng_restore(state: object) -> None:
        calls.append(state)
        raise AssertionError("RNG restore should be skipped.")

    monkeypatch.setattr(train_loop, "set_rng_state", fail_rng_restore)

    config = TrainLoop.Config(
        step=cast(Any, _WeightedEvalStep.Config()),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.checkpointing = None
    config.max_steps = 0
    config.restore_rng_state = False
    loop = config.make()

    loop.load_state_dict(
        {
            "step": {"global_step": 12},
            # The pass count rides the DATASET's own state, not a second copy
            # on the loop: only the loader knows when the data ran out.
            "dataset": {"timer_epoch": {"global_count": 3, "global_sec": 0.0}},
            "metrics": {},
            "rng": {"cuda": object()},
        },
    )

    assert calls == []
    assert loop.step.global_step == 12
    assert loop.current_epoch == 3


def test_file_tracker_writes_json_with_explicit_context(tmp_path: Path) -> None:
    tracker = FileTracker.Config(
        working_dir=tmp_path / "final_metrics.json",
    ).make()

    tracker.log_metrics(
        {
            "pass@2": 0.41,
            "pass@1": 0.33,
            "cell_accuracy": 0.87,
            "extras": {"artifact": object()},
        },
        12,
        prefix="eval/",
    )

    out = tmp_path / "final_metrics.json"
    assert json.loads(out.read_text()) == {
        "eval/pass@2": 0.41,
        "eval/pass@1": 0.33,
        "eval/cell_accuracy": 0.87,
    }
    # No temp file is left behind after the atomic replace.
    assert not list(tmp_path.glob("*.tmp.*"))


def test_file_tracker_noop_when_path_empty(tmp_path: Path) -> None:
    """An empty FileTracker path writes nothing."""
    FileTracker.Config(working_dir="").make().log_metrics(
        {"pass@2": 0.41},
        12,
        prefix="eval/",
    )

    assert not list(tmp_path.iterdir())


def test_file_tracker_creates_parent_dirs(tmp_path: Path) -> None:
    """A nested FileTracker path has its parent directories created."""
    target = tmp_path / "nested" / "dir" / "metrics.json"
    FileTracker.Config(working_dir=str(target)).make().log_metrics(
        {"score": 1.0}, 12, prefix="eval/"
    )

    assert json.loads(target.read_text()) == {"eval/score": 1.0}


def test_phase_heartbeat_fires_on_stall_and_names_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stalled phase logs ``STILL IN PHASE`` with the phase label and rank.

    Guards the eval-stall observability: a block that outlives the interval must
    emit a per-rank heartbeat naming the exact phase, so a distributed hang is
    self-diagnosing from the logs alone (no external py-spy).
    """
    from priml.train.train_loop import _phase_heartbeat  # noqa: PLC0415

    with (
        caplog.at_level(logging.WARNING, logger="priml.train.train_loop"),
        _phase_heartbeat("eval batch 5 eval_loss", interval_s=0.02),
    ):
        time.sleep(0.06)
    messages = [r.getMessage() for r in caplog.records]
    assert any("STILL IN PHASE" in m for m in messages)
    assert any("eval batch 5 eval_loss" in m for m in messages)
    assert any("[rank 0]" in m for m in messages)


def test_phase_heartbeat_silent_when_block_is_fast(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A block that returns before the interval emits no heartbeat (zero cost)."""
    from priml.train.train_loop import _phase_heartbeat  # noqa: PLC0415

    with (
        caplog.at_level(logging.WARNING, logger="priml.train.train_loop"),
        _phase_heartbeat("fast phase", interval_s=5.0),
    ):
        time.sleep(0.02)
    assert not any("STILL IN PHASE" in r.getMessage() for r in caplog.records)


def test_phase_heartbeat_watchdog_never_fires_while_healthy(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A slow but healthy phase must never trigger the faulthandler dump.

    ``faulthandler.dump_traceback_later`` walks every thread's frame stack from
    a C watchdog thread WITHOUT the GIL; against a running interpreter that
    walk reads mutating frames and can segfault the process (exp9010 full-set
    eval r1: 67 dumps over 12 healthy ~250s batches, garbage ``<invalid
    frame>`` reads, then SIGSEGV mid-dump). The contract: while the Python
    beat thread can run (the GIL is periodically available), the watchdog
    deadline is pushed forward and the dump never fires -- it may only fire
    for a genuine GIL-holding native wedge, whose frames are static.
    """
    from priml.train.train_loop import _phase_heartbeat  # noqa: PLC0415

    with _phase_heartbeat("eval batch 12 eval_loss", interval_s=0.05):
        deadline = time.perf_counter() + 0.35  # >3 watchdog periods.
        while time.perf_counter() < deadline:
            time.sleep(0.01)  # Healthy: the GIL is released constantly.
    assert "Timeout (" not in capfd.readouterr().err


@pytest.mark.compute_torch_compile
def test_phase_heartbeat_watchdog_fires_on_gil_holding_stall(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A GIL-holding native stall still gets the faulthandler stack dump.

    Companion to the never-fires-while-healthy contract: the watchdog must
    keep diagnosing the failure it exists for -- a native call that holds the
    GIL so long the Python beat thread cannot run (or push the deadline).
    Big-int multiplication is a single GIL-holding C call with a size knob;
    calibrate it to this machine, then wedge for several watchdog periods.
    """
    from priml.train.train_loop import _phase_heartbeat  # noqa: PLC0415

    def _timed(bits: int) -> float:
        start = time.perf_counter()
        x = 1 << bits
        _ = x * x
        return time.perf_counter() - start

    bits = 1 << 22
    while True:  # Calibrate one GIL-holding op to >=0.15s on this machine.
        # Best-of-5, not one sample: a single timing inflated by scheduler
        # preemption (seen 5x on a loaded box) sets interval_s so high the
        # real wedge below finishes before 2*interval_s and never arms the
        # dump. The minimum is the preemption-free cost, which the wedge
        # cannot undershoot.
        duration = min(_timed(bits) for _ in range(5))
        if duration >= 0.15:
            break
        bits *= 2
    with _phase_heartbeat("wedged phase", interval_s=duration / 8.0):
        x = 1 << bits
        _ = x * x  # Holds the GIL ~8 intervals; the watchdog fires at 2.
    assert "Timeout (" in capfd.readouterr().err


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
