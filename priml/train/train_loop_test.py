"""Tests for TrainLoop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast, override

import faulthandler
import functools
import gc
import json
import logging
import math
import shutil
import tempfile
import threading
import time

from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from torch.distributed.device_mesh import DeviceMesh

    from priml.custom_types import CheckpointableProtocol
    from priml.data.custom_types import DatasetProtocol
    from priml.distributed.testing import WarmPoolGetter
    from priml.loss.custom_types import LossOutput
    from priml.train.custom_types import TrainStepOutput

from configgle import Fig, Makeable, Makes, PartialConfig

from priml.cost import Cost, matmul_cost
from priml.data.dummy import DummyDataset
from priml.lib.custom_json import ListCodec
from priml.math.seed import RngState, get_rng_state, salt
from priml.metrics.binary_accuracy import BinaryAccuracy
from priml.metrics.topk import TopK
from priml.metrics.utilization import Utilization
from priml.runtime import SingleProcess, runtime_initialized
from priml.timer import CheckpointableStepTimer
from priml.train import train_loop
from priml.train.checkpointer import Checkpointer, _agreed_across_ranks
from priml.train.parallelism import NoParallel
from priml.train.profiler import PhaseTimer, TorchProfiler
from priml.train.tracker import FileTracker
from priml.train.train_loop import (
    EvalTimeLimitError,
    TrainLoop,
    _barrier_if_distributed,
    _compile_heartbeat,
    _HasTimer,
    _phase_heartbeat,
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
        **_kwargs: object,
    ) -> Tensor:
        """Forward pass."""
        return self.linear(media)


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

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
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

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
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

    def eval_dataloader(self) -> list[dict[str, object]]:
        """Return eval batches with uneven valid counts."""
        return [
            {"media": torch.tensor([[1.0]]), "valid_count": 4},
            {"media": torch.tensor([[0.0]]), "valid_count": 1},
        ]

    class StateDict(TypedDict):
        """The epoch timer's state."""

        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Get dataset state for checkpointing."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load dataset state for checkpointing."""
        state = cast(_WeightedEvalDataset.StateDict, state_dict)
        self.timer_epoch.load_state_dict(state["timer_epoch"])


class _WeightedEvalStep:
    """Train step that returns the batch media as an eval scalar metric."""

    class Config(Fig["_WeightedEvalStep"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.global_step = 0
        self.local_step = 0

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Return the batch unchanged."""
        return batch

    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Return loss and metric equal to the batch media scalar."""
        media = preprocessed_batch["media"]
        assert isinstance(media, Tensor)
        return {"loss": media.flatten(), "model": media, "metrics": {"score": media}}

    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Delegate to eval_loss."""
        return self.eval_loss(**preprocessed_batch)

    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Unused train step."""
        del preprocessed_batch
        return {"loss": torch.zeros(1), "model": torch.zeros(1, 1)}

    def call_eval(self, **preprocessed_batch: object) -> object:
        """Return the batch media."""
        return preprocessed_batch["media"]

    def on_epoch_end(self) -> None:
        """No-op epoch hook."""

    def state_dict(self) -> _StepStateDict:
        """Get train-step state."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load train-step state."""
        self.global_step = cast(_StepStateDict, state_dict)["global_step"]


class _StepStateDict(TypedDict):
    """The step counter a fake train step checkpoints."""

    global_step: int


class _WarmupStep:
    """Train step that records eval warmup calls."""

    class Config(Fig["_WarmupStep"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.global_step = 0
        self.local_step = 0
        self.eval_calls: list[Tensor] = []

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Return the batch unchanged."""
        return batch

    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Record eval batch media."""
        media = preprocessed_batch["media"]
        assert isinstance(media, Tensor)
        self.eval_calls.append(media.clone())
        return {"loss": torch.zeros(1), "model": media}

    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Delegate to eval_loss."""
        return self.eval_loss(**preprocessed_batch)

    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Advance one train step."""
        del preprocessed_batch
        self.global_step += 1
        self.local_step += 1
        return {"loss": torch.zeros(1), "model": torch.zeros(1, 1)}

    def call_eval(self, **preprocessed_batch: object) -> object:
        """Return eval logits placeholder."""
        return preprocessed_batch["media"]

    def on_epoch_end(self) -> None:
        """No-op epoch hook."""

    def state_dict(self) -> _StepStateDict:
        """Get train-step state."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load train-step state."""
        self.global_step = cast(_StepStateDict, state_dict)["global_step"]


class _RecordingTracker:
    """Tracker that records metric payloads for TrainLoop tests."""

    class Config(Fig["_RecordingTracker"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.metrics_by_step: list[tuple[dict[str, object], int]] = []
        self.notes: list[str] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Record prefixed metrics for assertions."""
        self.metrics_by_step.append(
            ({f"{prefix}{key}": value for key, value in metrics.items()}, step),
        )

    def log_images(self, key: str, images: list[object], step: int) -> None:
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

    def update(self, logits: Tensor, **batch: object) -> None:
        """Ignore batches; the payload is constant."""
        del logits, batch

    def compute(self) -> dict[str, object]:
        """Return one scalar plus a non-scalar ``extras`` payload."""
        return {"metric_score": 2.0, "extras": {"payload": ("opaque",)}}

    def reset(self) -> None:
        """Stateless."""

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        """Stateless."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Stateless."""
        del state_dict


def _cross_entropy(output: Tensor, *, label: Tensor, **_kwargs: object) -> LossOutput:
    """Call cross_entropy with the label extracted from kwargs."""
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
    def forward(self, media: Tensor, **_kwargs: object) -> Tensor:
        """Forward pass."""
        return self.linear(media)


@pytest.mark.compute_training
def test_train_loop_basic():
    """Test TrainLoop runs without errors."""
    torch.manual_seed(42)

    # Create config.
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
    config.metrics_eval = {}
    config.max_steps = 10
    config.num_steps_eval = 5
    config.seed = 42

    with tempfile.TemporaryDirectory() as tmp:
        assert isinstance(config.checkpointer, Checkpointer.Config)
        config.checkpointer.base_dir = "/"
        config.checkpointer.working_dir = Path(tmp)
        config.checkpointer.save_every = 5
        loop = config.make()
        loop.train()
        assert loop.step.global_step == 10
        assert (Path(tmp) / "step_00000005.pt").exists()
        assert (Path(tmp) / "step_00000010.pt").exists()


@pytest.mark.compute_training
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
    config.max_steps = 1000  # High limit.
    config.max_epochs = 2  # Should stop after 2 epochs (10 steps)
    config.num_steps_eval = 5
    config.seed = 42

    with tempfile.TemporaryDirectory() as tmp:
        assert isinstance(config.checkpointer, Checkpointer.Config)
        config.checkpointer.base_dir = "/"
        config.checkpointer.working_dir = Path(tmp)
        loop = config.make()
        loop.train()
        assert loop.current_epoch == 2


def _binary_cross_entropy_with_logits(
    output: Tensor,
    *,
    label: Tensor,
    **_kwargs: object,
) -> LossOutput:
    """Call binary_cross_entropy_with_logits with the label extracted from kwargs."""
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
    def forward(self, media: Tensor, **_kwargs: object) -> Tensor:
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

        # Generate linearly separable data: y = 1 if x1 + x2 > 0 else 0.
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

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load dataset state from checkpoint."""
        _ = state_dict


@pytest.mark.compute_training
def test_train_loop_comprehensive():
    """Comprehensive test with checkpointing and metrics."""
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"

        # Create config with checkpointing and metrics.
        step_config = TrainStep.Config()
        step_config.model = _LogisticModel.Config(in_features=2, out_features=1)
        step_config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
        step_config.loss = PartialConfig(_binary_cross_entropy_with_logits)
        step_config.parallelism = NoParallel.Config(device="cpu")
        step_config.compile = None
        config = TrainLoop.Config(step=step_config, dataset=_BinaryDataset.Config())
        config.metrics_eval = {"accuracy": BinaryAccuracy.Config()}
        config.max_steps = 20
        config.num_steps_eval = 10
        config.checkpointer = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        config.seed = 42

        # First training run - train for 20 steps.
        loop1 = config.make()
        loop1.train()

        # Check training completed.
        assert loop1.step.global_step == 20

        # Check checkpoints were created.
        assert (checkpoint_dir / "step_00000010.pt").exists()
        assert (checkpoint_dir / "step_00000020.pt").exists()

        # Get final loss.
        ds1 = loop1.dataset
        assert isinstance(ds1, _BinaryDataset)
        final_loss_result_1 = loop1.step.eval_loss(media=ds1.X, label=ds1.y)
        final_loss_1 = final_loss_result_1["loss"]

        # Second training run - should resume from step 20.
        step_config2 = TrainStep.Config()
        step_config2.model = _LogisticModel.Config(in_features=2, out_features=1)
        step_config2.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
        step_config2.loss = PartialConfig(_binary_cross_entropy_with_logits)
        step_config2.parallelism = NoParallel.Config(device="cpu")
        step_config2.compile = None
        config2 = TrainLoop.Config(step=step_config2, dataset=_BinaryDataset.Config())
        config2.metrics_eval = {"accuracy": BinaryAccuracy.Config()}
        config2.max_steps = 30
        config2.num_steps_eval = 10
        config2.checkpointer = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        config2.seed = 42
        assert isinstance(config2.checkpointer, Checkpointer.Config)
        config2.checkpointer.resume = True

        loop2 = config2.make()

        # Should have resumed from step 20.
        assert loop2.step.global_step == 20

        # Should have same loss as end of first run.
        ds2 = loop2.dataset
        assert isinstance(ds2, _BinaryDataset)
        resumed_loss_result = loop2.step.eval_loss(media=ds2.X, label=ds2.y)
        torch.testing.assert_close(final_loss_1, resumed_loss_result["loss"])

        # Continue training to step 30.
        loop2.train()
        assert loop2.step.global_step == 30

        # Check that loss improved.
        final_loss_result_2 = loop2.step.eval_loss(media=ds2.X, label=ds2.y)
        final_loss_2 = final_loss_result_2["loss"]
        assert final_loss_2.mean().item() < final_loss_1.mean().item()

        # Check that accuracy improved.
        eval_metrics = loop2.eval()
        assert "accuracy_accuracy" in eval_metrics
        accuracy = eval_metrics["accuracy_accuracy"]
        assert isinstance(accuracy, (int, float))
        assert accuracy > 0.75  # Should get > 75% accuracy.


def _eval_only_step_config() -> TrainStep.Config:
    step_config = TrainStep.Config()
    step_config.model = _LogisticModel.Config(in_features=2, out_features=1)
    step_config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step_config.loss = PartialConfig(_binary_cross_entropy_with_logits)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    return step_config


def test_eval_only_loads_checkpoint_and_skips_training(seeded_checkpoints: Path):
    """eval_only loads a checkpoint, evals once, and runs no training step.

    Reads the session's checkpoints rather than training its own: the claim is
    about what ``eval_only`` does GIVEN a checkpoint, so producing one is setup,
    and the session run writes it under the same recipe, seed, and cadence.

    """
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)
        assert (checkpoint_dir / "step_00000020.pt").exists()

        # eval_only run: loads the step-20 checkpoint, evals, no training.
        eval_cfg = TrainLoop.Config(
            step=_eval_only_step_config(),
            dataset=_BinaryDataset.Config(),
        )
        eval_cfg.metrics_eval = {"accuracy": BinaryAccuracy.Config()}
        eval_cfg.max_steps = 20
        eval_cfg.num_steps_eval = float("inf")
        eval_cfg.checkpointer = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
            keep_last_n=2,
        )
        eval_cfg.seed = 42
        eval_cfg.eval_only = True

        eval_loop = eval_cfg.make()
        # Resume defaults on, so the checkpoint loads without eval_only touching
        # the checkpointer's read policy.
        assert eval_loop.step.global_step == 20  # Loaded, not trained.
        eval_loop.train()  # Dispatches to the eval-only path.
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
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = math.inf
    config.max_epochs = math.inf
    config.max_time = math.inf
    config.seed = 42

    with pytest.raises(ValueError, match="no finite stop condition"):
        config.make()


def _resume_table_config(checkpoint_dir: Path):
    """Return a minimal trainable config writing into ``checkpoint_dir``."""
    cfg = TrainLoop.Config(
        step=_eval_only_step_config(),
        dataset=_BinaryDataset.Config(),
    )
    cfg.metrics_eval = {}
    cfg.max_steps = 20
    cfg.num_steps_eval = float("inf")
    cfg.eval_every_epoch = False
    cfg.checkpointer = Checkpointer.Config(
        base_dir="/",
        working_dir=checkpoint_dir,
        save_every=10,
        keep_last_n=5,
    )
    cfg.seed = 42
    return cfg


@pytest.fixture(scope="session")
def seeded_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train ONE 20-step run per session; every resume test copies its output."""
    source = tmp_path_factory.mktemp("seeded-checkpoints") / "ck"
    cfg = _resume_table_config(source)
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.resume = False
    loop = cfg.make()
    loop.train()
    assert (source / "step_00000020.pt").exists()
    # Several one-time torch imports hide under this path: the first optimizer
    # build pulls in torch._dynamo (417ms on colossus against 3ms warm), the
    # first RESUME pulls in torch.load's deserialization machinery, and the
    # first EVAL pulls in its own. Each is per-process, so leaving them to
    # whichever test runs first bills an arbitrary one -- a different one per
    # worker under xdist, which is what put these tests in the slow report at
    # -n 24. Paying them here makes the cost attributable to this fixture and
    # paid once. Warmed by REPLAYING the real build-resume-eval path rather
    # than by importing torch internals directly, so it stays correct if what
    # that path touches changes.
    warm = source.parent / "warm"
    shutil.copytree(source, warm)
    warm_config = _resume_table_config(warm)
    assert isinstance(warm_config.checkpointer, Checkpointer.Config)
    warm_config.checkpointer.resume = True
    warm_config.metrics_eval = {"accuracy": BinaryAccuracy.Config()}
    warm_config.eval_only = True
    warm_config.make().train()
    return source


def _seed_checkpoints(checkpoint_dir: Path, seeded: Path) -> None:
    """Populate ``checkpoint_dir`` with the session run's step_10 and step_20."""
    shutil.copytree(seeded, checkpoint_dir)


def test_resume_latest_uses_largest_when_checkpoints_exist(seeded_checkpoints: Path):
    """resume=True, resume_step=-1, checkpoints exist -> load largest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = -1
        loop = cfg.make()
        assert loop.step.global_step == 20  # Largest on disk.


def test_resume_latest_starts_fresh_when_no_checkpoints():
    """resume=True, resume_step=-1, no checkpoints -> start at 0, no error.

    -1 means "resume from latest if any"; an empty dir implies resume from
    nothing, i.e. a fresh start. Must NOT raise (the prior contract did).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "nope"  # `never` written to.
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = -1
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_resume_explicit_step_loads_that_step(seeded_checkpoints: Path):
    """resume=True, resume_step>0 present -> load exactly that step."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = 10
        # step_20 exists and a later save would re-mint it; that overwrite
        # guard is exercised elsewhere -- here we isolate the explicit load.
        cfg.checkpointer.allow_checkpoint_overwrite = True
        loop = cfg.make()
        assert loop.step.global_step == 10


def test_resume_explicit_step_missing_raises(seeded_checkpoints: Path):
    """resume=True, resume_step>0 absent -> hard error (named step not found)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)  # Has 10, 20 -- not 999.

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = 999
        with pytest.raises(RuntimeError, match="requested but not found"):
            cfg.make()


def test_resume_explicit_step_no_checkpoints_raises():
    """resume=True, resume_step>0, empty dir -> hard error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "nope"
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = 5
        with pytest.raises(RuntimeError, match="requested but not found"):
            cfg.make()


def test_resume_false_starts_at_zero_into_empty_dir():
    """resume=False, empty dir -> start at 0 (resume_step ignored)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "fresh"
        cfg = _resume_table_config(empty_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = False
        cfg.checkpointer.resume_step = 10  # Must be ignored.
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_resume_defaults_to_true():
    """The resume default is True (preemption-restart is the common case)."""
    finalized = TrainLoop.Config().finalize()
    assert isinstance(finalized.checkpointer, Checkpointer.Config)
    assert finalized.checkpointer.resume is True


def test_a_resume_with_nothing_left_to_do_says_so(
    caplog: pytest.LogCaptureFixture,
    seeded_checkpoints: Path,
) -> None:
    """A completed experiment re-run must explain itself, not just exit 0.

    Resume defaults on and a run is identified by its directory, so re-running
    a finished experiment restores its last step, finds the budget already
    spent, and returns in seconds with no RESULT line. That is correct
    behaviour and unreadable output: measured on a cifar10 re-run, the only
    signal was a 4-second job log with nothing in it. The warning names the
    step it resumed and what to do instead; it stays a warning because
    resume-by-default is the intended policy.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(
            checkpoint_dir,
            seeded_checkpoints,
        )  # Trains to max_steps and saves step_20.

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = -1
        loop = cfg.make()
        assert loop.step.global_step == cfg.max_steps  # Nothing left to run.
        with caplog.at_level(logging.WARNING, logger=train_loop.__name__):
            loop.train()
        assert loop.step.global_step == 20  # Exited cleanly, trained nothing.
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("already completed at step 20" in message for message in warnings), (
            warnings
        )
        assert any("experiment_name" in message for message in warnings), warnings


def test_fresh_run_refuses_to_overwrite_existing_checkpoints(seeded_checkpoints: Path):
    """A from-scratch run whose saves would land on existing files is refused.

    The colleague's footgun: a fresh run reusing a name silently overwrote the
    prior run's checkpoints. With save_every=10/max_steps=20, this run would
    mint step_10/step_20 -- both already on disk -- so it must refuse.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)  # Steps 10, 20.

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = False
        cfg.checkpointer.allow_checkpoint_overwrite = False
        with pytest.raises(RuntimeError, match="would overwrite existing"):
            cfg.make()


def test_rewind_resume_refuses_to_overwrite_newer_checkpoints(
    seeded_checkpoints: Path,
):
    """Resuming an older step is refused when later saves would clobber newer.

    Orthogonal to resume: resuming step 10 then training to 20 would mint
    step_20 over the existing step_20. allow_checkpoint_overwrite=False
    refuses, and the check is upfront (at make()), not after training.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)  # Steps 10, 20.

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = 10  # Rewind: start_step=10, step_20 is newer.
        cfg.checkpointer.allow_checkpoint_overwrite = False
        with pytest.raises(RuntimeError, match="would overwrite existing"):
            cfg.make()


def test_resume_latest_does_not_trip_overwrite_guard(seeded_checkpoints: Path):
    """Resuming the latest checkpoint never collides -- no save step exceeds it."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)  # Steps 10, 20.

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = True
        cfg.checkpointer.resume_step = -1  # start_step=20; no save step in (20, 20].
        cfg.checkpointer.allow_checkpoint_overwrite = False
        loop = cfg.make()
        assert loop.step.global_step == 20


def test_fresh_run_into_off_cadence_dir_is_allowed():
    """A populated dir whose steps this run never re-mints does not trip."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "step_5.pt").write_bytes(b"x")  # Off the save cadence.

        cfg = _resume_table_config(checkpoint_dir)  # save_every=10 -> 10, 20.
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = False
        cfg.checkpointer.allow_checkpoint_overwrite = False
        loop = cfg.make()  # 5 is never a save step -> no collision.
        assert loop.step.global_step == 0


def test_allow_checkpoint_overwrite_permits_clobber(seeded_checkpoints: Path):
    """allow_checkpoint_overwrite=True lets a run mint over existing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)

        cfg = _resume_table_config(checkpoint_dir)
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume = False
        cfg.checkpointer.allow_checkpoint_overwrite = True
        loop = cfg.make()
        assert loop.step.global_step == 0


def test_allow_checkpoint_overwrite_defaults_to_false():
    """allow_checkpoint_overwrite must default False -- clobbering is opt-in."""
    finalized = TrainLoop.Config().finalize()
    assert isinstance(finalized.checkpointer, Checkpointer.Config)
    assert finalized.checkpointer.allow_checkpoint_overwrite is False


def test_eval_only_never_trips_overwrite_guard(seeded_checkpoints: Path):
    """eval_only writes no checkpoint, so the overwrite guard must not fire.

    Loading an older explicit step (10) while a newer one (20) exists would
    trip the cadence-collision guard for a training run, but eval_only runs no
    training step and saves nothing -- the guard is vacuous and must be skipped.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "ck"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)  # Steps 10, 20.

        cfg = _resume_table_config(checkpoint_dir)
        cfg.eval_only = True
        assert isinstance(cfg.checkpointer, Checkpointer.Config)
        cfg.checkpointer.resume_step = (
            10  # Older than latest; would-collide for training.
        )
        cfg.checkpointer.allow_checkpoint_overwrite = False
        loop = cfg.make()  # Must not raise.
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
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = 1
    config.num_steps_eval = math.inf
    config.base_dir = base_dir
    config.runtime = _RecordingRuntime.Config()
    return config


def test_eval_weights_scalar_metrics_by_valid_count() -> None:
    """Eval scalar means weight partial batches by valid example count."""
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False

    loop = config.make()

    assert loop.eval()["score"] == 0.8


def test_eval_fails_when_exceeding_max_eval_time() -> None:
    """An eval pass over its wall-clock budget raises EvalTimeLimitError."""
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.max_eval_time = 0.0  # `any` elapsed batch trips the deadline.

    loop = config.make()

    with pytest.raises(EvalTimeLimitError, match="max_eval_time"):
        loop.eval()


def test_eval_stop_on_time_limit_publishes_partial_results() -> None:
    """Opt-in data-generation eval stops at the budget instead of raising."""
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
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
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False
    config.max_eval_time = 3_600.0

    loop = config.make()

    assert loop.eval()["score"] == 0.8


def test_eval_warmup_runs_configured_eval_batches() -> None:
    """Eval warmup runs eval_loss once per configured batch before training."""
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_WarmupDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
    config.eval_warmup_batches = 1
    config.max_steps = 0
    config.num_steps_eval = math.inf
    config.eval_every_epoch = False

    loop = config.make()
    step = loop.step
    assert isinstance(step, _WarmupStep)

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
        cfg.checkpointer = Checkpointer.Config()
        cfg.phase_timer = PhaseTimer.Config(
            base_dir="/",
            working_dir="/explicit/child",
        )

        finalized = cfg.finalize()

        assert finalized.working_dir == Path("/opt/scratch/runs/my_project/exp000")
        assert isinstance(finalized.checkpointer, Checkpointer.Config)
        assert finalized.checkpointer.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/checkpoints",
        )
        assert isinstance(finalized.phase_timer, PhaseTimer.Config)
        assert finalized.phase_timer.working_dir == Path("/explicit/child")

    def test_profiling_output_is_scoped_below_run(self) -> None:
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.profiler = TorchProfiler.Config(torch_profile=False)

        finalized = cfg.finalize()

        assert isinstance(finalized.profiler, TorchProfiler.Config)
        assert finalized.profiler.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/profiling",
        )

    def test_explicit_profiling_base_dir_wins(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "profiling"
        cfg = TrainLoop.Config()
        cfg.profiler = TorchProfiler.Config(
            torch_profile=False,
            base_dir=tmp_path,
            working_dir="/profiling",
        )

        finalized = cfg.finalize()

        assert isinstance(finalized.profiler, TorchProfiler.Config)
        assert finalized.profiler.working_dir == working_dir

    def test_checkpoint_working_dir_is_scoped_below_run(self):
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointer = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointer, Checkpointer.Config)
        assert finalized.checkpointer.working_dir == Path(
            "/opt/scratch/runs/my_project/exp000/checkpoints",
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
        cfg.checkpointer = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointer, Checkpointer.Config)
        assert finalized.checkpointer.working_dir == (
            scratch / "runs/my_project/exp000/checkpoints"
        )

    def test_checkpoint_base_dir_not_overwritten_when_set(self):
        cfg = TrainLoop.Config()
        cfg.study_name = "my_project/"
        cfg.experiment_name = "exp000"
        cfg.checkpointer = Checkpointer.Config(
            base_dir="/",
            working_dir="/custom/dir",
        )
        finalized = cfg.finalize()
        assert isinstance(finalized.checkpointer, Checkpointer.Config)
        assert finalized.checkpointer.working_dir == Path("/custom/dir")

    def test_checkpoint_working_dir_without_run_name(self):
        cfg = TrainLoop.Config()
        cfg.checkpointer = Checkpointer.Config()

        finalized = cfg.finalize()

        assert isinstance(finalized.checkpointer, Checkpointer.Config)
        assert finalized.checkpointer.working_dir == Path(
            "/opt/scratch/runs/checkpoints",
        )

    def test_doc_is_sent_to_tracker_via_log_notes(self):
        """``doc`` reaches the built tracker through ``log_notes`` at init time.

        The launcher only sets ``doc``; TrainLoop forwards it to the tracker's
        ``log_notes`` after make (no config-internals mutation).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointer = None
            cfg.tracker = _RecordingTracker.Config()
            cfg.doc = "Hypothesis: X. Change: Y. Result: TODO."
            loop = cfg.make()
            tracker = loop.tracker
            assert isinstance(tracker, _RecordingTracker)
            assert tracker.notes == ["Hypothesis: X. Change: Y. Result: TODO."]

    def test_empty_doc_does_not_call_log_notes(self):
        """No description means no notes call -- the tracker keeps its own."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointer = None
            cfg.tracker = _RecordingTracker.Config()
            cfg.doc = ""
            loop = cfg.make()
            tracker = loop.tracker
            assert isinstance(tracker, _RecordingTracker)
            assert tracker.notes == []

    def test_doc_without_tracker_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_simple_loop_config(tmp)
            cfg.checkpointer = None
            cfg.tracker = None
            cfg.doc = "docstring note"
            loop = cfg.make()  # Must not raise.
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
    config.metrics_eval = {}
    config.checkpointer = None
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


def _make_step_logging_loop_config(
    *,
    step_config: TrainStep.Config | None = None,
) -> TrainLoop.Config:
    """Minimal CPU loop that logs a per-step loss line on every step."""
    config = TrainLoop.Config()
    if step_config is None:
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
    config.metrics_eval = {}
    config.checkpointer = None
    config.max_steps = 2
    config.num_steps_eval = math.inf
    config.num_steps_log = 1  # Log every step.
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
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_step_logging_loop_config()
    config.num_steps_eval = 1
    loop = config.make()
    # A measurable eval: at zero seconds every term is zero and the sum holds
    # however the parts are counted, so the invariant would not bite.
    inner_eval = loop.eval

    def slow_eval() -> dict[str, object]:
        metrics = inner_eval()
        clock.now += 0.05
        return metrics

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
    assert seconds["eval_sec"] >= 0.1  # Two evals, 0.05s each.
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
    step_config = config.step
    assert isinstance(step_config, TrainStep.Config)
    step_config.parallelism = NoParallel.Config()  # Unset: defer to the runtime.
    loop = config.copy_tree().finalize().make()
    step = loop.step
    assert isinstance(step, TrainStep)

    assert loop.runtime.device == torch.device("cpu")
    assert next(step.model.parameters()).device == torch.device("cpu")


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
    config.num_steps_eval = 100  # Finite (RESULT fires) but no cadence eval in 2 steps.
    loop = config.make()
    monkeypatch.setattr(
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=100.0),
    )
    # A budgeted step: it charges only part of the second its update took.
    step = loop.step
    assert isinstance(step, TrainStep)
    monkeypatch.setattr(step, "elapsed_sec", 0.4, raising=False)
    monkeypatch.setattr(loop, "_train_elapsed", lambda: 0.4)

    def timed_eval() -> dict[str, object]:
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

    def _with_metrics(**batch: object) -> dict[str, object]:
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
        calls.append(tuple(ListCodec.coerce(tensor.tolist(), float)))
        tensor.mul_(8)

    loop = _make_step_logging_loop_config().make()
    step = loop.step
    assert isinstance(step, TrainStep)

    def _with_metrics(**batch: object) -> dict[str, object]:
        del batch
        # Advance the step counter the way the real step does -- by charging
        # the timer global_step reads -- since it is a read-only property.
        step.timer_step.global_count += 1
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
        batch = cast(dict[str, object], next(iterator))
        loop._do_train_step(loop.step.preprocess_batch(batch))

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


# The first step takes ``first_step_time`` (simulating the backward-graph compile);
# every later step takes 1s.
def _timed_train_step(
    loop: TrainLoop,
    clock: _FakeClock,
    *,
    first_step_time: float,
) -> Callable[..., TrainStepOutput]:
    """Wrap the loop's train_step to advance ``clock`` by a scripted duration."""
    inner = loop.step.train_step

    def timed_step(**batch: object) -> TrainStepOutput:
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
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=100.0),
    )

    loop.train()

    assert loop.step.global_step == 1  # The 100s "compile" alone exceeds 10s.


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
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=100.0),
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
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=1.0),
    )
    eval_calls: list[int] = []

    def timed_eval() -> dict[str, object]:
        eval_calls.append(loop.step.global_step)
        clock.now += 50.0  # Each eval alone would blow the 10s budget.
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
    config.eval_every_epoch = True  # 8 samples / batch 4 -> eval every 2 steps.
    loop = config.make()
    monkeypatch.setattr(
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=1.0),
    )
    eval_calls: list[int] = []

    def timed_eval() -> dict[str, object]:
        eval_calls.append(loop.step.global_step)
        clock.now += 50.0  # Each eval alone would blow the 10s budget.
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
        loop.step,
        "train_step",
        _timed_train_step(loop, clock, first_step_time=100.0),
    )

    loop.train()

    tracker = loop.tracker
    assert isinstance(tracker, _RecordingTracker)
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
    config.metrics_eval = {}
    config.checkpointer = None
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
    config.metrics_eval = {}
    config.max_steps = 1
    config.num_steps_eval = math.inf
    config.seed = 42
    config.phase_timer = PhaseTimer.Config(enabled=True)

    loop = config.make()
    assert isinstance(loop.step, _HasTimer)
    assert loop.step.timer is loop.phase_timer


# -- Regression tests (Issue#286 trainloop + checkpoint-loop group) ----------


def _simple_dummy_dataset() -> DummyDataset.Config:
    """Return the default small CPU dummy dataset for simple-loop helpers."""
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
) -> TrainLoop.Config[TrainStep.Config[_LinearModel.Config], Makeable[DatasetProtocol]]:
    """Build a minimal CPU TrainLoop.Config over the supplied dataset."""
    step_config: TrainStep.Config[_LinearModel.Config] = TrainStep.Config()
    step_config.model = _LinearModel.Config(in_features=2, out_features=2)
    step_config.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step_config.loss = PartialConfig(_cross_entropy)
    step_config.parallelism = NoParallel.Config(device="cpu")
    step_config.compile = None
    config: TrainLoop.Config[
        TrainStep.Config[_LinearModel.Config],
        Makeable[DatasetProtocol],
    ] = TrainLoop.Config(
        step=step_config,
        dataset=dataset if dataset is not None else _simple_dummy_dataset(),
    )
    config.metrics_eval = {}
    config.checkpointer = Checkpointer.Config(
        base_dir="/",
        working_dir=Path(tmp),
        save_every=5,
    )
    config.max_steps = 10
    config.seed = 42
    return config


def test_no_eval_or_checkpoint_at_step_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-015: no eval / checkpoint should fire before the first train step."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 1
        config.num_steps_eval = 5  # `step` 0 would be a multiple of 5.
        loop = config.make()

        eval_steps: list[int] = []
        orig_eval = loop.eval

        def spy_eval() -> dict[str, object]:
            eval_steps.append(loop.step.global_step)
            return orig_eval()

        monkeypatch.setattr(loop, "eval", spy_eval)
        loop.train()

        # No eval should have run while global_step == 0.
        assert 0 not in eval_steps, f"eval ran at step 0: {eval_steps}"
        # No checkpoint dir for step 0.
        assert not (Path(tmp) / "step_00000000.pt").exists()


def test_resume_does_not_eval_or_checkpoint_before_first_new_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        checkpointer = loop.checkpointer
        assert checkpointer is not None
        original_maybe_save = checkpointer.maybe_save

        def spy_maybe_save(target: CheckpointableProtocol, step: int) -> bool:
            maybe_save_steps.append(step)
            return original_maybe_save(target, step)

        monkeypatch.setattr(checkpointer, "maybe_save", spy_maybe_save)
        eval_steps: list[int] = []
        orig_eval = loop.eval

        def spy_eval() -> dict[str, object]:
            eval_steps.append(loop.step.global_step)
            return orig_eval()

        monkeypatch.setattr(loop, "eval", spy_eval)
        loop.train()

        assert loop.step.global_step == 6
        assert 5 not in maybe_save_steps
        assert 5 not in eval_steps


def test_resume_does_not_rewrite_completed_checkpoint_with_partial_accumulation() -> (
    None
):
    """A resumed partial accumulation must not overwrite its completed prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        initial = _make_simple_loop_config(tmp)
        initial.max_steps = 1
        assert isinstance(initial.step, TrainStep.Config)
        initial.step.accumulate_grad_batches = 2
        assert isinstance(initial.checkpointer, Checkpointer.Config)
        initial.checkpointer.save_every = 1
        initial.make().train()

        resumed = _make_simple_loop_config(tmp)
        resumed.max_steps = 2
        assert isinstance(resumed.step, TrainStep.Config)
        resumed.step.accumulate_grad_batches = 2
        assert isinstance(resumed.checkpointer, Checkpointer.Config)
        resumed.checkpointer.save_every = 1
        resumed.make().train()

        checkpoint = _load_loop_checkpoint(Path(tmp) / "step_00000001.pt")
        step_state = cast(TrainStep.StateDict, checkpoint["step"])
        assert step_state["accumulation_steps"] == 0


@pytest.mark.parametrize(
    ("consume_eval_rng", "num_steps_eval"),
    [(False, 1_000), (True, 1)],
    ids=("no_eval", "stochastic_eval"),
)
def test_cadence_checkpoint_replays_next_batch_and_rng(
    consume_eval_rng: bool,
    num_steps_eval: int,
) -> None:
    """A cadence checkpoint resumes at its exact data and RNG prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        reference_config = _make_simple_loop_config(
            str(Path(tmp) / "reference"),
            dataset=_ReplayDataset.Config(consume_eval_rng=consume_eval_rng),
        )
        reference_config.max_steps = 3
        reference_config.num_steps_eval = num_steps_eval
        assert isinstance(reference_config.checkpointer, Checkpointer.Config)
        reference_config.checkpointer.save_every = 1
        reference = reference_config.make()
        reference.train()
        assert isinstance(reference.dataset, _ReplayDataset)
        expected_batches = reference.dataset.batches[1:]
        expected_rng = get_rng_state()["torch"]

        resumed_config = _make_simple_loop_config(
            str(Path(tmp) / "resumed"),
            dataset=_ReplayDataset.Config(consume_eval_rng=consume_eval_rng),
        )
        resumed_config.max_steps = 3
        resumed_config.num_steps_eval = num_steps_eval
        assert isinstance(resumed_config.checkpointer, Checkpointer.Config)
        resumed_config.checkpointer.save_every = 1
        checkpoint_dir = Path(tmp) / "resumed"
        checkpoint_dir.mkdir()
        shutil.copy2(
            Path(tmp) / "reference" / "step_00000001.pt",
            checkpoint_dir / "step_00000001.pt",
        )
        resumed = resumed_config.make()
        assert resumed.step.global_step == 1
        resumed.train()
        assert isinstance(resumed.dataset, _ReplayDataset)

        assert resumed.dataset.cursor == reference.dataset.cursor
        assert len(resumed.dataset.batches) == len(expected_batches)
        for actual, expected in zip(
            resumed.dataset.batches,
            expected_batches,
            strict=True,
        ):
            assert torch.equal(actual, expected)
        assert torch.equal(get_rng_state()["torch"], expected_rng)


def test_final_checkpoint_replays_rng_after_final_eval() -> None:
    """Extending a completed run must continue after its final evaluation."""
    with tempfile.TemporaryDirectory() as tmp:
        reference_config = _make_simple_loop_config(
            str(Path(tmp) / "reference"),
            dataset=_ReplayDataset.Config(consume_eval_rng=True),
        )
        reference_config.max_steps = 2
        reference_config.num_steps_eval = -1
        reference = reference_config.make()
        reference.train()
        assert isinstance(reference.dataset, _ReplayDataset)
        expected_batch = next(reference.dataset)["media"]

        resumed_config = _make_simple_loop_config(
            str(Path(tmp) / "resumed"),
            dataset=_ReplayDataset.Config(consume_eval_rng=True),
        )
        resumed_config.max_steps = 3
        resumed_config.num_steps_eval = -1
        checkpoint_dir = Path(tmp) / "resumed"
        checkpoint_dir.mkdir()
        shutil.copy2(
            Path(tmp) / "reference" / "step_00000002.pt",
            checkpoint_dir / "step_00000002.pt",
        )
        resumed = resumed_config.make()
        resumed.train()
        assert isinstance(resumed.dataset, _ReplayDataset)

        assert len(resumed.dataset.batches) == 1
        assert torch.equal(resumed.dataset.batches[0], expected_batch)


@pytest.mark.parametrize(
    ("is_final", "expected_step"),
    [(False, 1), (True, 2)],
    ids=("cadence", "final"),
)
def test_eval_timeout_saves_and_restores_interrupted_prefix(
    tmp_path: Path,
    is_final: bool,
    expected_step: int,
) -> None:
    """A timed-out scheduled eval preserves its completed optimizer prefix."""
    config = _make_simple_loop_config(
        str(tmp_path),
        dataset=_ReplayDataset.Config(consume_eval_rng=True),
    )
    config.max_steps = 2
    config.num_steps_eval = -1 if is_final else 1
    config.max_eval_time = 0.0
    assert isinstance(config.checkpointer, Checkpointer.Config)
    config.checkpointer.save_every = 1
    loop = config.make()

    with pytest.raises(EvalTimeLimitError, match="max_eval_time"):
        loop.train()

    interrupted = loop.state_dict()
    checkpoint = tmp_path / f"step_{expected_step:08d}.pt"
    _assert_checkpoint_matches_interrupted_state(checkpoint, interrupted=interrupted)

    resumed_config = _make_simple_loop_config(
        str(tmp_path),
        dataset=_ReplayDataset.Config(consume_eval_rng=True),
    )
    resumed_config.max_steps = expected_step + 1
    resumed_config.num_steps_eval = math.inf
    assert isinstance(resumed_config.checkpointer, Checkpointer.Config)
    resumed_config.checkpointer.save_every = 1
    resumed = resumed_config.make()

    assert resumed.step.global_step == expected_step
    _assert_state_dict_equal(resumed.state_dict(), expected=interrupted)


def test_cadence_eval_error_after_work_saves_interrupted_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after an eval batch saves the RNG state that eval consumed."""
    config = _make_simple_loop_config(
        str(tmp_path),
        dataset=_ReplayDataset.Config(consume_eval_rng=True),
    )
    config.max_steps = 2
    config.num_steps_eval = 1
    assert isinstance(config.checkpointer, Checkpointer.Config)
    config.checkpointer.save_every = 1
    loop = config.make()
    monkeypatch.setattr(loop, "eval", functools.partial(_raise_after_eval, loop.eval))

    with pytest.raises(EvalTimeLimitError, match="after eval work"):
        loop.train()

    _assert_checkpoint_matches_interrupted_state(
        tmp_path / "step_00000001.pt",
        interrupted=loop.state_dict(),
    )


@pytest.mark.parametrize("is_final", [False, True], ids=("cadence", "final"))
def test_eval_error_logs_immediate_checkpoint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    is_final: bool,
) -> None:
    """A checkpoint failure never replaces the scheduled eval's exception."""
    config = _make_simple_loop_config(str(tmp_path))
    config.max_steps = 2
    config.num_steps_eval = -1 if is_final else 1
    config.max_eval_time = 0.0
    assert isinstance(config.checkpointer, Checkpointer.Config)
    config.checkpointer.save_every = 1
    loop = config.make()
    checkpointer = loop.checkpointer
    assert checkpointer is not None
    if is_final:
        monkeypatch.setattr(checkpointer, "save", _raise_checkpoint_write)
    else:
        monkeypatch.setattr(checkpointer, "maybe_save", _raise_checkpoint_write)

    with (
        caplog.at_level(logging.ERROR, logger="priml.train.train_loop"),
        pytest.raises(EvalTimeLimitError, match="max_eval_time"),
    ):
        loop.train()

    recovery_records = [
        record
        for record in caplog.records
        if record.message == "Failed to save checkpoint after evaluation error."
    ]
    assert len(recovery_records) == 1
    assert recovery_records[0].exc_info is not None


@pytest.mark.parametrize("is_final", [False, True], ids=("cadence", "final"))
@pytest.mark.parametrize(
    ("world_size", "expects_checkpoint_io"),
    [(1, True), (2, False)],
    ids=("one_rank", "multi_rank"),
)
def test_eval_error_recovery_skips_checkpoint_io_only_for_multirank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    is_final: bool,
    world_size: int,
    expects_checkpoint_io: bool,
) -> None:
    """Recovery remains local-only when a distributed group has multiple ranks."""
    config = _make_simple_loop_config(str(tmp_path))
    loop = config.make()
    checkpointer = loop.checkpointer
    assert checkpointer is not None
    calls: list[str] = []
    monkeypatch.setattr(
        checkpointer,
        "maybe_save",
        functools.partial(_record_maybe_save, calls),
    )
    monkeypatch.setattr(
        checkpointer,
        "save",
        functools.partial(_record_save, calls),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)

    loop._save_after_evaluation_error(is_final=is_final)

    expected = ["save" if is_final else "maybe_save"] if expects_checkpoint_io else []
    assert calls == expected


def test_interrupted_final_eval_does_not_save_partial_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Interrupted eval cannot make pending gradients look resumable."""
    config = _make_simple_loop_config(str(tmp_path))
    config.max_steps = 2
    config.num_steps_eval = -1
    config.max_eval_time = 0.0
    assert isinstance(config.step, TrainStep.Config)
    config.step.accumulate_grad_batches = 2
    loop = config.make()
    monkeypatch.setattr(loop, "_time_limit_reached", lambda: loop.local_step >= 1)

    with (
        caplog.at_level(logging.ERROR, logger="priml.train.train_loop"),
        pytest.raises(EvalTimeLimitError, match="max_eval_time"),
    ):
        loop.train()

    assert list(tmp_path.iterdir()) == []
    assert any(
        record.message == "Failed to save checkpoint after evaluation error."
        and record.exc_info is not None
        for record in caplog.records
    )


def test_terminal_partial_accumulation_fails_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal partial accumulation must not publish a resumable checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 2
        assert isinstance(config.step, TrainStep.Config)
        config.step.accumulate_grad_batches = 2
        loop = config.make()
        monkeypatch.setattr(loop, "_time_limit_reached", lambda: loop.local_step >= 1)

        with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
            loop.train()

        assert list(Path(tmp).iterdir()) == []


def test_terminal_partial_accumulation_preserves_prior_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed partial termination leaves the latest completed checkpoint intact."""
    with tempfile.TemporaryDirectory() as tmp:
        initial = _make_simple_loop_config(tmp)
        initial.max_steps = 1
        assert isinstance(initial.step, TrainStep.Config)
        initial.step.accumulate_grad_batches = 2
        assert isinstance(initial.checkpointer, Checkpointer.Config)
        initial.checkpointer.save_every = 1
        initial.make().train()

        resumed = _make_simple_loop_config(tmp)
        resumed.max_steps = 2
        assert isinstance(resumed.step, TrainStep.Config)
        resumed.step.accumulate_grad_batches = 2
        assert isinstance(resumed.checkpointer, Checkpointer.Config)
        resumed.checkpointer.save_every = 1
        resumed.checkpointer.allow_checkpoint_overwrite = True
        loop = resumed.make()
        monkeypatch.setattr(loop, "_time_limit_reached", lambda: loop.local_step >= 1)

        with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
            loop.train()

        checkpoint = _load_loop_checkpoint(Path(tmp) / "step_00000001.pt")
        step_state = cast(TrainStep.StateDict, checkpoint["step"])
        assert step_state["accumulation_steps"] == 0


def test_terminal_no_update_with_complete_accumulation_saves_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped train call with no pending gradients still saves its prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 2
        loop = config.make()
        original_train_step = loop._do_train_step
        calls = [0]

        monkeypatch.setattr(
            loop,
            "_do_train_step",
            functools.partial(_skip_after_first_train_step, original_train_step, calls),
        )
        monkeypatch.setattr(loop, "_time_limit_reached", lambda: calls[0] >= 2)
        loop.train()

        assert calls[0] == 2
        step = loop.step
        assert isinstance(step, TrainStep)
        assert step.global_step == 1
        assert step.accumulation_steps == 0
        assert loop.checkpointer is not None
        assert loop.checkpointer.available_steps() == [1]


class _CostedLinearModel(nn.Module):
    """A linear model whose config can cost itself, so an MFU meter binds."""

    class Config(Fig["_CostedLinearModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            del kwargs
            return matmul_cost(
                channels_in=self.in_features,
                channels_out=self.out_features,
                bias=True,
                rows=seq_len * batch_size,
                dtype=dtype,
            )

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    @override
    def forward(self, media: Tensor, **_kwargs: object) -> Tensor:
        """Forward pass."""
        return self.linear(media)


@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_device_timing_metric_waits_for_every_microbatch(
    accumulate_grad_batches: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")
    step = TrainStep.Config()
    config = _make_step_logging_loop_config(step_config=step)
    step.model = _CostedLinearModel.Config(in_features=2, out_features=2)
    step.parallelism = NoParallel.Config(device="cuda")
    step.accumulate_grad_batches = accumulate_grad_batches
    config.runtime = SingleProcess.Config(device="cuda")
    config.tracker = _RecordingTracker.Config()
    config.early_train_log_steps = 0
    metric = Utilization.Config()
    metric.tokens_key = "media"
    metric.peak_flops_per_sec = 1e6
    config.metrics_train = {"": metric}
    loop = config.make()
    batch: dict[str, object] = {
        "media": torch.randn(4, 2, device="cuda"),
        "label": torch.zeros(4, dtype=torch.long, device="cuda"),
    }
    for _ in range(accumulate_grad_batches):
        loop._do_train_step(batch)
    measured = cast(Utilization, loop.metrics_train[""])
    measured.reset()
    assert isinstance(loop.step, TrainStep)
    handle = loop.step.model.register_forward_pre_hook(_enqueue_cuda_delay)
    try:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loop._do_train_step(batch)
        end.record()
        end.synchronize()
        gpu_seconds = start.elapsed_time(end) / 1000
        seconds = measured.seconds
        if accumulate_grad_batches == 1:
            tracker = cast(_RecordingTracker, loop.tracker)
            payload = next(
                values
                for values, _ in reversed(tracker.metrics_by_step)
                if "train/step_time" in values
            )
            seconds = cast(float, payload["train/step_time"])
        assert seconds >= 0.8 * gpu_seconds
    finally:
        handle.remove()
        loop._destroy_runtime_once()


def _enqueue_cuda_delay(module: nn.Module, inputs: tuple[object, ...]) -> None:
    del module, inputs
    torch.cuda._sleep(100_000_000)


def test_cpu_timing_metric_does_not_synchronize_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = TrainStep.Config()
    config = _make_step_logging_loop_config(step_config=step)
    step.model = _CostedLinearModel.Config(in_features=2, out_features=2)
    config.runtime = SingleProcess.Config(device="cpu")
    metric = Utilization.Config()
    metric.tokens_key = "media"
    config.metrics_train = {"": metric}
    loop = config.make()
    calls: list[object] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", calls.append)
    try:
        loop._do_train_step(
            {"media": torch.randn(4, 2), "label": torch.zeros(4, dtype=torch.long)},
        )
        assert calls == []
    finally:
        loop._destroy_runtime_once()


@pytest.mark.gpu_torch_cuda
def test_loop_without_device_timing_metric_does_not_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")
    step = TrainStep.Config()
    config = _make_step_logging_loop_config(step_config=step)
    step.parallelism = NoParallel.Config(device="cuda")
    step.accumulate_grad_batches = 2
    config.runtime = SingleProcess.Config(device="cuda")
    config.metrics_train = {}
    config.early_train_log_steps = 0
    loop = config.make()
    calls: list[object] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", calls.append)
    try:
        loop._do_train_step(
            {
                "media": torch.randn(4, 2, device="cuda"),
                "label": torch.zeros(4, dtype=torch.long, device="cuda"),
            },
        )
        assert calls == []
    finally:
        loop._destroy_runtime_once()


def test_a_train_metric_publishes_on_the_train_payload() -> None:
    """A metric in ``metrics_train`` sees every step's batch and its wall time."""
    # ``_make_simple_loop_config`` pins its model slot to ``_LinearModel``, so
    # the costed model gets its own step; no checkpointer, so no directory.
    step: TrainStep.Config[_CostedLinearModel.Config] = TrainStep.Config()
    step.model = _CostedLinearModel.Config(in_features=2, out_features=2)
    step.optimizer = PartialConfig(torch.optim.Adam, lr=0.1)
    step.loss = PartialConfig(_cross_entropy)
    step.parallelism = NoParallel.Config(device="cpu")
    step.compile = None
    config = TrainLoop.Config(step=step, dataset=_simple_dummy_dataset())
    config.metrics_eval = {}
    config.checkpointer = None
    config.seed = 42
    config.tracker = _RecordingTracker.Config()
    mfu = Utilization.Config()
    mfu.tokens_key = "media"
    mfu.peak_flops_per_sec = 1e6
    config.metrics_train = {"": mfu}
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
    step_time = train_log["train/step_time"]
    tokens_per_sec = train_log["train/tokens_per_sec"]
    assert isinstance(step_time, float)
    assert isinstance(tokens_per_sec, float)
    # One [4, 2] media batch is 8 tokens; a 2x2 matmul is 24 FLOPs per token.
    assert tokens_per_sec == pytest.approx(8 / step_time)
    assert train_log["train/mfu"] == pytest.approx(24 * tokens_per_sec / 1e6)


def test_train_metrics_accumulate_over_the_log_cadence() -> None:
    """``update`` runs every step; ``compute``/``reset`` bracket each log."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.metrics_train = {"acc": TopK.Config(k_values=[1])}
        config.max_steps = 6
        config.num_steps_log = 3
        config.early_train_log_steps = 0
        config.num_steps_eval = math.inf
        loop = config.make()
        metric = loop.metrics_train["acc"]
        assert isinstance(metric, TopK)
        totals_at_compute: list[int] = []
        original_compute = metric.compute

        def spy_compute() -> dict[str, float]:
            totals_at_compute.append(metric.total)
            return original_compute()

        metric.compute = spy_compute  # ty: ignore[invalid-assignment] -- The test spies on a bound method.
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    # Step 1 always logs; then steps 3 and 6 on the cadence. Each ``compute``
    # saw only the steps since the previous log, four rows apiece.
    assert totals_at_compute == [4, 8, 12]
    logs = [
        m["train/acc_top1"] for m, _ in tracker.metrics_by_step if "train/acc_top1" in m
    ]
    assert len(logs) == 3
    for accuracy in logs:
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


def test_train_metrics_compute_and_reset_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero ranks bracket the window too, only the log is rank-0.

    A metric whose ``compute`` all-reduces would otherwise wait on rank 0
    alone, and one that does not would accumulate across the whole run.
    """
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.metrics_train = {"acc": TopK.Config(k_values=[1])}
        config.max_steps = 6
        config.num_steps_log = 3
        config.early_train_log_steps = 0
        config.num_steps_eval = math.inf
        loop = config.make()
        metric = loop.metrics_train["acc"]
        assert isinstance(metric, TopK)
        totals_at_compute: list[int] = []
        original_compute = metric.compute

        def spy_compute() -> dict[str, float]:
            totals_at_compute.append(metric.total)
            return original_compute()

        metric.compute = spy_compute  # ty: ignore[invalid-assignment] -- The test spies on a bound method.
        tracker = cast(_RecordingTracker, loop.tracker)

        loop.train()

    assert totals_at_compute == [4, 8, 12]
    assert not any("train/acc_top1" in m for m, _ in tracker.metrics_by_step)


def test_one_config_may_serve_both_phases() -> None:
    """The same protocol, so one config can sit in either dict."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        shared = TopK.Config(k_values=[1])
        config.metrics_train = {"acc": shared}
        config.metrics_eval = {"acc": shared}
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 1
        loop = config.make()
        assert isinstance(loop.metrics_train["acc"], TopK)
        assert isinstance(loop.metrics_eval["acc"], TopK)
        tracker = cast(_RecordingTracker, loop.tracker)
        loop.train()
    keys = {key for payload, _ in tracker.metrics_by_step for key in payload}
    assert {"train/acc_top1", "eval/acc_top1"} <= keys


def test_a_utilization_metric_binds_to_the_model_config_late() -> None:
    """No loop wiring: ``LateBound`` hands the built loop to the metric."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.metrics_train = {"": Utilization.Config()}
        config.max_steps = 0
        with pytest.raises(TypeError, match=r"_LinearModel\.Config"):
            config.make()


def test_complete_update_still_writes_terminal_checkpoint_and_resumes() -> None:
    """A completed update still receives the usual final resumable artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 1
        assert isinstance(config.step, TrainStep.Config)
        config.step.accumulate_grad_batches = 2
        loop = config.make()
        loop.train()

        assert loop.checkpointer is not None
        assert loop.checkpointer.available_steps() == [1]
        resumed = _make_simple_loop_config(tmp).make()
        assert resumed.step.global_step == 1


def test_train_step_logs_gpu_memory_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train-step tracker logs process CUDA memory peaks when CUDA is available."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 1
        config.num_steps_eval = math.inf
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

        original_train_step = loop.step.train_step

        def train_step_with_cuda_metrics(**batch: object) -> TrainStepOutput:
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
    assert _metric_float(train_log, "train/total_loss") >= 0.0
    assert _metric_float(train_log, "train/step_time") >= 0.0
    assert _metric_float(train_log, "train/time_since_start") >= 0.0
    assert train_log["train/gpu_mem_allocated_gb"] == 2.0
    assert train_log["train/gpu_mem_reserved_gb"] == 3.0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU guard is host-specific")
def test_train_step_omits_gpu_memory_on_cpu_tracker() -> None:
    """CPU train-step tracker logs keep running without CUDA memory keys."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 1
        config.num_steps_eval = math.inf
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

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
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 12
        config.num_steps_eval = math.inf
        config.num_steps_log = 5
        config.early_train_log_steps = 0
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

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
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        step_config = config.step
        assert isinstance(step_config, TrainStep.Config)
        step_config.accumulate_grad_batches = 4
        config.max_steps = 3
        config.num_steps_eval = math.inf
        config.num_steps_log = 1
        config.early_train_log_steps = 100
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

        loop.train()

    train_steps = sorted(
        step
        for metrics, step in tracker.metrics_by_step
        if "train/total_loss" in metrics
    )
    assert train_steps == [1, 2, 3]


def test_accumulation_evaluates_once_per_update_not_once_per_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        config.checkpointer = None
        step_config = config.step
        assert isinstance(step_config, TrainStep.Config)
        step_config.accumulate_grad_batches = 4
        config.max_steps = 4
        config.num_steps_eval = 2
        loop = config.make()

        evaluated: list[int] = []
        inner = loop.eval

        def spy() -> dict[str, object]:
            evaluated.append(loop.step.global_step)
            return inner()

        monkeypatch.setattr(loop, "eval", spy)
        loop.train()

    # Step 2 on the cadence, then the final eval. The while-loop exits once
    # ``max_steps`` is reached, so step 4's cadence eval never comes due.
    assert evaluated == [2, 4]


def test_train_tracker_logs_every_startup_step_then_cadence() -> None:
    """Startup diagnostics log every early step before falling back to cadence."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 12
        config.num_steps_eval = math.inf
        config.num_steps_log = 5
        config.early_train_log_steps = 3
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

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
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 100
        config.max_epochs = 1
        config.num_steps_eval = math.inf
        config.eval_every_epoch = True
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

        loop.train()

    epoch_logs = [metrics for metrics, step in tracker.metrics_by_step if step == 1]
    assert any("eval/total_loss" in metrics for metrics in epoch_logs)
    assert any(_metric_float(metrics, "eval/time") >= 0.0 for metrics in epoch_logs)
    assert tracker.closed


def test_step_eval_logs_eval_time_to_tracker() -> None:
    """Step-triggered eval logs duration with eval metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        config.num_steps_eval = 1
        config.eval_every_epoch = False
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

        loop.train()

    eval_logs = [metrics for metrics, step in tracker.metrics_by_step if step == 1]
    assert any("eval/total_loss" in metrics for metrics in eval_logs)
    assert any(_metric_float(metrics, "eval/time") >= 0.0 for metrics in eval_logs)
    # Per-batch eval step time is logged like train/step_time.
    assert any(
        _metric_float(metrics, "eval/mean_batch_time") >= 0.0 for metrics in eval_logs
    )
    assert tracker.closed


def test_final_eval_uses_same_bounded_loader_as_cadence() -> None:
    """Every eval -- cadence and final -- uses the one (bounded) eval loader.

    There is no separate uncapped final pass: the dataset's own eval config
    (caps or none) decides the eval scope, and the final eval reuses it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        config.num_steps_eval = 1
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = loop.dataset
    assert isinstance(dataset, _ScopedEvalDataset)
    # Cadence eval at step 1, then the final eval; both bounded, never "full".
    assert dataset.eval_scopes == ["bounded", "bounded"]
    assert "full" not in dataset.eval_scopes


def test_phase_timer_summary_logs_after_final_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timing summary includes the final eval before it is published."""
    events: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointer = None
        config.max_steps = 2
        # Finite cadence so a final eval runs (num_steps_eval=inf disables eval).
        config.num_steps_eval = 2
        config.eval_every_epoch = False
        config.phase_timer = PhaseTimer.Config(enabled=True)
        loop = config.make()
        dataset = loop.dataset
        assert isinstance(dataset, _ScopedEvalDataset)
        timer = loop.phase_timer
        assert isinstance(timer, PhaseTimer)
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
        # precede the phase-timer summary.
        monkeypatch.setattr(dataset, "eval_dataloader", record_eval_dataloader)
        monkeypatch.setattr(loop.phase_timer, "log_summary", record_log_summary)

        loop.train()

    assert events[-2:] == ["final_eval", "summary"]


def test_phase_timer_publishes_interval_and_summary_metrics() -> None:
    """Enabled timing reaches tracker sinks during and after training."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp, dataset=_ScopedEvalDataset.Config())
        config.checkpointer = None
        config.max_steps = 2
        config.num_steps_log = 1
        config.num_steps_eval = 0
        config.eval_every_epoch = False
        config.phase_timer = PhaseTimer.Config(enabled=True)
        config.tracker = _RecordingTracker.Config()
        loop = config.make()

        loop.train()

    tracker = loop.tracker
    assert isinstance(tracker, _RecordingTracker)
    timing_payloads = [
        metrics
        for metrics, _step in tracker.metrics_by_step
        if any(key.startswith("timing/") for key in metrics)
    ]
    assert any("timing/interval_wall_sec" in payload for payload in timing_payloads)
    assert any("timing/total_sec" in payload for payload in timing_payloads)
    assert tracker.closed


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
        config.checkpointer = None
        config.max_steps = 2
        config.num_steps_eval = never
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = loop.dataset
    assert isinstance(dataset, _ScopedEvalDataset)
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
        config.checkpointer = None
        config.max_steps = 6
        config.num_steps_eval = -1
        config.eval_every_epoch = False
        loop = config.make()

        loop.train()

    dataset = loop.dataset
    assert isinstance(dataset, _ScopedEvalDataset)
    assert len(dataset.eval_scopes) == 1


def test_final_post_training_eval_logs_to_tracker() -> None:
    """The end-of-training eval reaches the tracker, not just the console.

    The final eval is the run's reported number (post-training pass@K); it must
    land in the tracker at the final step, not only the RESULT log line.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.tracker = _RecordingTracker.Config()
        config.max_steps = 2
        # Cadence wider than the run (2 % 5 != 0): no mid-run eval fires, so the
        # only eval is the post-training final one (num_steps_eval=inf would
        # disable eval entirely, so it must stay finite).
        config.num_steps_eval = 5
        config.eval_every_epoch = False
        loop = config.make()
        tracker = loop.tracker
        assert isinstance(tracker, _RecordingTracker)

        loop.train()

    final_logs = [
        metrics
        for metrics, step in tracker.metrics_by_step
        if step == loop.step.global_step
    ]
    assert any("eval/total_loss" in metrics for metrics in final_logs), (
        "final eval not logged to tracker"
    )


def test_cadence_eval_runs_once_per_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gradient accumulation cannot repeat eval at one optimizer step."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.checkpointer = None
        config.num_steps_eval = 5
        loop = config.make()
        assert isinstance(loop.step, TrainStep)
        loop.step.timer_step.global_count = 5

        eval_count = 0
        original_eval = loop.eval

        def count_eval() -> dict[str, object]:
            nonlocal eval_count
            eval_count += 1
            return original_eval()

        monkeypatch.setattr(loop, "eval", count_eval)
        loop._maybe_eval()
        loop._maybe_eval()

    assert eval_count == 1


def test_no_post_loop_eval_when_no_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-019: with max_steps=0, the post-loop eval must not run."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 0
        config.num_steps_eval = math.inf
        loop = config.make()

        eval_count = [0]
        orig_eval = loop.eval

        def spy_eval() -> dict[str, object]:
            eval_count[0] += 1
            return orig_eval()

        monkeypatch.setattr(loop, "eval", spy_eval)
        loop.train()

        assert loop.step.global_step == 0
        assert eval_count[0] == 0, "post-loop eval ran despite zero training"


def test_eval_improvement_saves_off_cadence_checkpoint() -> None:
    """An eval that improves ``best_metric`` saves at that (off-cadence) step."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 6
        config.num_steps_eval = 3
        assert isinstance(config.checkpointer, Checkpointer.Config)
        config.checkpointer.save_every = 100
        config.checkpointer.best_metric = "total_loss"
        config.checkpointer.best_mode = "min"
        loop = config.make()
        loop.train()

        assert loop.checkpointer is not None
        # The first eval (step 3) sets the best; the end-of-run save adds 6.
        assert loop.checkpointer.available_steps() == [3, 6]


def test_final_eval_improvement_keeps_the_end_of_run_save() -> None:
    """The final eval saves the best at the last step; the end-of-run save adds nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 3
        config.num_steps_eval = -1
        assert isinstance(config.checkpointer, Checkpointer.Config)
        config.checkpointer.save_every = 100
        config.checkpointer.best_metric = "total_loss"
        config.checkpointer.best_mode = "min"
        writes: list[int] = []
        loop = config.make()
        assert isinstance(loop.checkpointer, Checkpointer)
        original_write = loop.checkpointer._write

        def spy_write(target: CheckpointableProtocol, step: int) -> None:
            writes.append(step)
            original_write(target, step)

        loop.checkpointer._write = spy_write  # ty: ignore[invalid-assignment] -- The test spies on a bound method.
        loop.train()

        assert writes == [3]
        assert loop.checkpointer.best_step == 3


def test_eval_only_never_saves_a_best_checkpoint(seeded_checkpoints: Path) -> None:
    """An eval-only run writes nothing, however good its score."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        _seed_checkpoints(checkpoint_dir, seeded_checkpoints)
        eval_cfg = TrainLoop.Config(
            step=_eval_only_step_config(),
            dataset=_BinaryDataset.Config(),
        )
        eval_cfg.metrics_eval = {"accuracy": BinaryAccuracy.Config()}
        eval_cfg.max_steps = 20
        eval_cfg.num_steps_eval = math.inf
        eval_cfg.checkpointer = Checkpointer.Config(
            base_dir="/",
            working_dir=checkpoint_dir,
            save_every=10,
        )
        eval_cfg.checkpointer.best_metric = "accuracy_accuracy"
        eval_cfg.eval_only = True
        before = {p: p.stat().st_mtime_ns for p in checkpoint_dir.iterdir()}

        loop = eval_cfg.make()
        loop.train()

        assert loop.checkpointer is not None
        assert loop.checkpointer.available_steps() == [10, 20]
        assert {p: p.stat().st_mtime_ns for p in checkpoint_dir.iterdir()} == before


def test_retention_keeps_last_n_after_training() -> None:
    """Retention prunes to keep_last_n across cadence + forced end-of-run saves."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_simple_loop_config(tmp)
        config.max_steps = 25
        config.num_steps_eval = math.inf
        assert isinstance(config.checkpointer, Checkpointer.Config)
        config.checkpointer.save_every = 5
        config.checkpointer.keep_last_n = 2
        loop = config.make()
        loop.train()

        assert loop.checkpointer is not None
        # Saves land at 5,10,15,20,25; retention keeps only the newest two.
        assert loop.checkpointer.available_steps() == [20, 25]


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
    config.metrics_eval = {}
    config.checkpointer = None
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
    step = loop.step
    assert isinstance(step, TrainStep)
    assert step.accumulation_steps == 1


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
    step = loop.step
    assert isinstance(step, TrainStep)
    assert step.accumulation_steps == 4
    assert step.global_step == 0


# Only rank 0's verdict is True. A correct collective decision broadcasts rank 0's view
# so every rank agrees, so no rank can return early and strand the others at the save
# barrier.
def _collective_skip_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Diverge per-rank verdicts; record the collective decision."""
    del mesh
    rank = torch.distributed.get_rank()
    try:
        skip = _agreed_across_ranks(rank == 0)
        (Path(result_dir) / f"rank_{rank}").write_text("skip" if skip else "save")
    except Exception as e:  # noqa: BLE001 -- The test surfaces any worker failure to its parent process.
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

    _set_loader_epoch(_LoaderWithoutDataset(), epoch=3)


def test_dataset_receives_the_step_before_the_first_batch() -> None:
    """A generating dataset is bound to the step, and bound before iterating.

    An on-policy dataset produces its batches by acting with the current
    policy, so the binding must land before any batch is drawn -- otherwise
    the first rollout has no model to act with.
    """
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_BindingDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 1
    loop = config.make()
    dataset = loop.dataset
    assert isinstance(dataset, _BindingDataset)

    assert dataset.bound is loop.step
    assert dataset.batches_before_binding == 0

    loop.train()
    assert dataset.batches_drawn > 0


def test_dataset_without_the_hook_is_left_alone() -> None:
    """A dataset that reads a corpus must not be required to accept a step."""
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_WarmupDataset.Config(),
    )
    config.checkpointer = None
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

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        """Get dataset state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load dataset state for checkpointing."""
        del state_dict


def _make_extras_publish_config() -> TrainLoop.Config:
    """Eval-publish config with a payload metric and a recording tracker."""
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {"": _ExtrasMetric.Config()}
    config.tracker = _RecordingTracker.Config()
    config.checkpointer = None
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
    tracker = loop.tracker
    assert isinstance(tracker, _RecordingTracker)
    step = cast(_WeightedEvalStep, loop.step)
    step.global_step = 12

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
    tracker = loop.tracker
    assert isinstance(tracker, _RecordingTracker)
    step = cast(_WeightedEvalStep, loop.step)
    step.global_step = 12

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
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.checkpointer = None
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
        {"score": 1.0},
        12,
        prefix="eval/",
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
    from priml.train.train_loop import (  # noqa: PLC0415 -- The test imports the private heartbeat seam only inside this scenario.
        _phase_heartbeat,
    )

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
    from priml.train.train_loop import (  # noqa: PLC0415 -- The test imports the private heartbeat seam only inside this scenario.
        _phase_heartbeat,
    )

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
    from priml.train.train_loop import (  # noqa: PLC0415 -- The test imports the private heartbeat seam only inside this scenario.
        _phase_heartbeat,
    )

    with _phase_heartbeat("eval batch 12 eval_loss", interval_s=0.01):
        deadline = time.perf_counter() + 0.08  # >3 watchdog periods.
        while time.perf_counter() < deadline:
            time.sleep(0.002)  # Healthy: the GIL is released constantly.
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
    from priml.train.train_loop import (  # noqa: PLC0415 -- The test imports the private heartbeat seam only inside this scenario.
        _phase_heartbeat,
    )

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


class _PathedDataset(_WarmupDataset):
    """A dataset that inherits its corpus root from the loop."""

    class Config(Makes["_PathedDataset"], _WarmupDataset.Config):
        base_dir: Path | str | None = None
        working_dir: Path | str = "/datasets/corpus"


class _PathedMetric(_ExtrasMetric):
    """A metric whose location is decided by what its logical path names."""

    class Config(Makes["_PathedMetric"], _ExtrasMetric.Config):
        base_dir: Path | str | None = None
        working_dir: Path | str = "/dump"


def test_finalize_routes_each_path_owner_to_the_root_it_reads() -> None:
    """The corpus and a ``/datasets`` metric read the bare root; dumps read the run."""
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_PathedDataset.Config(),
    )
    config.base_dir = "/opt/scratch"
    config.working_dir = Path("/runs/study/exp000")
    config.metrics_eval = {
        "shared": _PathedMetric.Config(working_dir="/datasets/arc"),
        "dump": _PathedMetric.Config(working_dir="/dump"),
    }
    config.metrics_train = {
        "preset": _PathedMetric.Config(base_dir="/elsewhere", working_dir="/x"),
    }

    finalized = config.finalize()

    assert finalized.working_dir == Path("/opt/scratch/runs/study/exp000")
    assert isinstance(finalized.dataset, _PathedDataset.Config)
    assert finalized.dataset.base_dir == "/opt/scratch"
    shared = finalized.metrics_eval["shared"]
    dump = finalized.metrics_eval["dump"]
    preset = finalized.metrics_train["preset"]
    assert isinstance(shared, _PathedMetric.Config)
    assert isinstance(dump, _PathedMetric.Config)
    assert isinstance(preset, _PathedMetric.Config)
    assert shared.base_dir == "/opt/scratch"
    assert dump.base_dir == Path("/opt/scratch/runs/study/exp000")
    assert preset.base_dir == "/elsewhere"


class _MeshAxis:
    def __init__(self, local_rank: int) -> None:
        self._local_rank = local_rank

    def get_local_rank(self) -> int:
        return self._local_rank


class _SeedMesh:
    """The two axes the loop salts seeds along: pipeline stage and data replica."""

    def __getitem__(self, name: str) -> _MeshAxis:
        return _MeshAxis({"pp": 1, "dp": 2}[name])


def test_a_mesh_salts_the_model_seed_by_stage_and_the_data_seed_by_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broadcast fixes the base seed; every later derivation is local."""
    seeded: list[tuple[object, ...]] = []

    def set_seed_distributed(
        seed: int | None,
        *,
        mesh: _MeshAxis,
        salt_by_rank: bool,
    ) -> tuple[int, int]:
        seeded.append(("distributed", seed, mesh.get_local_rank(), salt_by_rank))
        return 7, 8

    def set_seed_local(seed: int | None = None) -> int:
        seeded.append(("local", seed))
        assert seed is not None
        return seed

    monkeypatch.setattr(train_loop, "global_device_mesh", _SeedMesh)
    monkeypatch.setattr(train_loop, "set_seed_distributed", set_seed_distributed)
    monkeypatch.setattr(train_loop, "set_seed_local", set_seed_local)
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_WarmupDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 0
    config.seed = 42

    config.make()

    assert seeded == [
        ("distributed", 42, 1, True),
        ("local", salt("rank", 2, salt("dp", 7))),
    ]


def test_a_finite_gc_cadence_disables_automatic_collection_while_training(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Manual passes run on the cadence with the automatic collector off."""
    config = _make_step_logging_loop_config()
    config.num_steps_garbage_collect = 1
    config.max_steps = 3
    collected: list[bool] = []

    def collect(generation: int = 2) -> int:
        del generation
        collected.append(gc.isenabled())
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    loop = config.make()
    # Every rank collects together, so the pass ends on a barrier when a
    # group is live. Faked only around the GC call: a fake group seen by the
    # loader would send it looking for a distributed sampler.
    barriers: list[int] = []
    inner_gc = loop._maybe_garbage_collect

    def gc_under_a_live_group() -> None:
        with pytest.MonkeyPatch.context() as scoped:
            scoped.setattr(torch.distributed, "is_initialized", lambda: True)
            scoped.setattr(torch.distributed, "barrier", lambda: barriers.append(1))
            inner_gc()

    monkeypatch.setattr(loop, "_maybe_garbage_collect", gc_under_a_live_group)
    try:
        with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
            loop.train()
    finally:
        gc.enable()

    assert collected == [False, False]
    assert barriers == [1, 1]
    assert gc.isenabled()
    assert any("GC at local_step 1" in r.message for r in caplog.records)


def test_an_eval_error_without_a_checkpointer_still_surfaces(tmp_path: Path) -> None:
    config = _make_simple_loop_config(str(tmp_path))
    config.checkpointer = None
    config.max_steps = 1
    config.num_steps_eval = -1
    config.max_eval_time = 0.0
    loop = config.make()
    with pytest.raises(EvalTimeLimitError, match="max_eval_time"):
        loop.train()
    assert list(tmp_path.iterdir()) == []


def test_an_empty_loader_cannot_supply_a_batch_after_one_epoch_reset() -> None:
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_WarmupDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 1
    config.eval_every_epoch = False
    loop = config.make()
    with pytest.raises(RuntimeError, match="after epoch reset"):
        loop.train()
    assert loop.current_epoch == 2


def _distributed_flag_broadcast(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank_zero_verdict: float | None,
) -> list[float]:
    """Fake a live group whose rank-0 broadcast either records or overrides."""
    broadcasts: list[float] = []

    def broadcast(flag: Tensor, src: int) -> None:
        assert src == 0
        broadcasts.append(float(flag.item()))
        if rank_zero_verdict is not None:
            flag.fill_(rank_zero_verdict)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    return broadcasts


def test_time_limit_is_synced_on_the_log_cadence_and_latched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 0 reads the clock; the verdict is broadcast, then sticks."""
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_max_time_loop_config()
    config.num_steps_log = 5
    loop = config.make()
    broadcasts = _distributed_flag_broadcast(monkeypatch, rank_zero_verdict=None)
    step = loop.step
    assert isinstance(step, TrainStep)
    try:
        step.timer_step.global_count = 3  # Off the cadence: no collective.
        clock.now = 11.0
        assert not loop._time_limit_reached()
        assert broadcasts == []
        step.timer_step.global_count = 5
        clock.now = 5.0
        assert not loop._time_limit_reached()
        clock.now = 11.0
        assert loop._time_limit_reached()
        clock.now = 0.0
        assert loop._time_limit_reached()  # Latched.
        assert broadcasts == [0.0, 1.0]
    finally:
        loop._destroy_runtime_once()


def test_a_non_zero_rank_adopts_rank_zeros_time_limit_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_max_time_loop_config()
    config.num_steps_log = 1
    loop = config.make()
    broadcasts = _distributed_flag_broadcast(monkeypatch, rank_zero_verdict=1.0)
    step = loop.step
    assert isinstance(step, TrainStep)
    try:
        step.timer_step.global_count = 1
        clock.now = 999.0  # This rank's own clock never counts.
        assert loop._time_limit_reached()
        assert broadcasts == [0.0]
    finally:
        loop._destroy_runtime_once()


def test_eval_time_limit_is_rank_agreed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: True)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_step_logging_loop_config()
    config.max_eval_time = 1.0
    loop = config.make()
    broadcasts = _distributed_flag_broadcast(monkeypatch, rank_zero_verdict=None)
    try:
        clock.now = 0.5
        assert not loop._eval_time_limit_reached(0.0)
        clock.now = 2.0
        assert loop._eval_time_limit_reached(0.0)
        assert broadcasts == [0.0, 1.0]
    finally:
        loop._destroy_runtime_once()


def test_a_non_zero_rank_adopts_rank_zeros_eval_time_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("priml.train.train_loop.is_rank_zero", lambda: False)
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    config = _make_step_logging_loop_config()
    config.max_eval_time = 1.0
    loop = config.make()
    _distributed_flag_broadcast(monkeypatch, rank_zero_verdict=1.0)
    try:
        clock.now = 0.0
        assert loop._eval_time_limit_reached(0.0)
    finally:
        loop._destroy_runtime_once()


def test_eval_only_without_a_checkpoint_warns_and_scores_fresh_weights(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {}
    config.checkpointer = None
    config.tracker = _RecordingTracker.Config()
    config.eval_only = True
    loop = config.make()
    tracker = loop.tracker
    assert isinstance(tracker, _RecordingTracker)

    with caplog.at_level(logging.WARNING, logger="priml.train.train_loop"):
        loop.train()

    assert any("eval_only at global_step=0" in r.message for r in caplog.records)
    assert tracker.metrics_by_step[0][0]["eval/score"] == 0.8


class _RecordingProfiler:
    """Profiler fake that records which steps it bracketed."""

    class Config(Fig["_RecordingProfiler"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.cleaned = False

    def on_step_start(self, step: int) -> None:
        self.starts.append(step)

    def on_step_end(self, step: int) -> None:
        self.ends.append(step)

    def cleanup(self) -> None:
        self.cleaned = True


def test_the_profiler_brackets_every_step_and_is_cleaned_up() -> None:
    config = _make_step_logging_loop_config()
    config.profiler = _RecordingProfiler.Config()
    loop = config.make()
    profiler = loop.profiler
    assert isinstance(profiler, _RecordingProfiler)

    loop.train()

    assert profiler.starts == [0, 1]
    assert profiler.ends == [1, 2]
    assert profiler.cleaned


class _RecordingMetric:
    """Metric that keeps every output it was handed."""

    class Config(Fig["_RecordingMetric"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.seen: list[Tensor] = []

    def update(self, logits: Tensor, **batch: object) -> None:
        del batch
        self.seen.append(logits.detach().clone())

    def compute(self) -> dict[str, object]:
        return {"count": float(len(self.seen))}

    def reset(self) -> None:
        self.seen.clear()

    class StateDict(TypedDict):
        """Stateless."""

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


class _MetricOnlyEvalDataset(_WeightedEvalDataset):
    """Eval batches whose first carries no model input, only metric material."""

    class Config(Makes["_MetricOnlyEvalDataset"], _WeightedEvalDataset.Config):
        pass

    @override
    def eval_dataloader(self) -> list[dict[str, object]]:
        return [
            {"media": torch.tensor([[1.0]]), "metric_only": True},
            {"media": torch.tensor([[2.0]])},
        ]


def test_warm_eval_compile_skips_metric_only_batches() -> None:
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_MetricOnlyEvalDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 0
    config.eval_warmup_batches = 2
    loop = config.make()
    step = loop.step
    assert isinstance(step, _WarmupStep)
    assert len(step.eval_calls) == 1
    torch.testing.assert_close(step.eval_calls[0], torch.tensor([[2.0]]))


def test_eval_feeds_metric_only_batches_to_metrics_without_a_forward() -> None:
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_MetricOnlyEvalDataset.Config(),
    )
    config.metrics_eval = {"seen": _RecordingMetric.Config()}
    config.checkpointer = None
    config.max_steps = 0
    loop = config.make()
    step = loop.step
    metric = loop.metrics_eval["seen"]
    assert isinstance(step, _WarmupStep)
    assert isinstance(metric, _RecordingMetric)

    results = loop.eval()

    assert len(step.eval_calls) == 1
    assert [t.numel() for t in metric.seen] == [0, 1]
    assert results["seen_count"] == 2.0


class _ZeroWeightEvalDataset(_WeightedEvalDataset):
    """An eval batch with no valid examples beside one that has some."""

    class Config(Makes["_ZeroWeightEvalDataset"], _WeightedEvalDataset.Config):
        pass

    @override
    def eval_dataloader(self) -> list[dict[str, object]]:
        return [
            {"media": torch.tensor([[1.0]]), "valid_count": 0},
            {"media": torch.tensor([[0.5]]), "valid_count": 2},
        ]


def test_eval_skips_a_batch_with_no_valid_examples() -> None:
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_ZeroWeightEvalDataset.Config(),
    )
    config.metrics_eval = {"seen": _RecordingMetric.Config()}
    config.checkpointer = None
    config.max_steps = 0
    loop = config.make()
    metric = loop.metrics_eval["seen"]
    assert isinstance(metric, _RecordingMetric)

    results = loop.eval()

    assert results["score"] == 0.5
    assert len(metric.seen) == 1


class _VotingStep(_WeightedEvalStep):
    """A step whose eval offers one extra candidate per batch."""

    class Config(Makes["_VotingStep"], _WeightedEvalStep.Config):
        pass

    @override
    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        media = preprocessed_batch["media"]
        assert isinstance(media, Tensor)
        return {
            "loss": media.flatten(),
            "model": media,
            "eval_extra_votes": [(media * 2, {"media": media})],
        }


def test_eval_feeds_extra_votes_to_every_metric() -> None:
    config = TrainLoop.Config(
        step=_VotingStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_eval = {"seen": _RecordingMetric.Config()}
    config.checkpointer = None
    config.max_steps = 0
    loop = config.make()
    metric = loop.metrics_eval["seen"]
    assert isinstance(metric, _RecordingMetric)

    loop.eval()

    assert [float(t) for t in metric.seen] == [1.0, 2.0, 0.0, 0.0]


def test_run_ignores_its_arguments_and_trains() -> None:
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_BindingDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 1
    loop = config.make()
    loop.run("--flag", "value")
    assert loop.step.global_step == 1


def test_load_state_dict_restores_the_metrics_it_knows_and_skips_the_rest() -> None:
    config = TrainLoop.Config(
        step=_WeightedEvalStep.Config(),
        dataset=_WeightedEvalDataset.Config(),
    )
    config.metrics_train = {"acc": TopK.Config(k_values=[1])}
    config.metrics_eval = {"acc": TopK.Config(k_values=[1])}
    config.checkpointer = None
    config.max_steps = 0
    loop = config.make()

    loop.load_state_dict(
        {
            "step": {"global_step": 1},
            "dataset": {"timer_epoch": {"global_count": 0, "global_sec": 0.0}},
            "metrics_eval": {
                "acc": {"correct": {1: 3}, "total": 4},
                "ghost": {"correct": {1: 9}, "total": 9},
            },
            "metrics_train": {
                "acc": {"correct": {1: 1}, "total": 2},
                "ghost": {"correct": {1: 9}, "total": 9},
            },
        },
    )

    eval_metric = loop.metrics_eval["acc"]
    train_metric = loop.metrics_train["acc"]
    assert isinstance(eval_metric, TopK)
    assert isinstance(train_metric, TopK)
    assert eval_metric.total == 4
    assert train_metric.total == 2


class _EpochAwareDataset:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _EpochAwareLoader:
    def __init__(self) -> None:
        self.dataset = _EpochAwareDataset()


def test_set_loader_epoch_informs_a_dataset_that_listens() -> None:
    loader = _EpochAwareLoader()
    _set_loader_epoch(loader, 3)
    assert loader.dataset.epochs == [3]


def test_startup_barrier_waits_when_a_group_is_live(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    barriers: list[int] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))
    with caplog.at_level(logging.INFO, logger="priml.train.train_loop"):
        _barrier_if_distributed("tracker startup")
    assert barriers == [1]
    assert any("all ranks passed tracker startup" in r.message for r in caplog.records)


def test_compile_heartbeat_reports_a_long_running_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger="priml.train.train_loop"),
        _compile_heartbeat("train step 1", interval_s=0.02),
    ):
        time.sleep(0.06)
    assert any("train step 1: still running after" in r.message for r in caplog.records)


def test_phase_heartbeat_names_this_rank_when_a_group_is_live(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    with (
        caplog.at_level(logging.WARNING, logger="priml.train.train_loop"),
        _phase_heartbeat("eval batch 1 eval_loss", interval_s=0.02),
    ):
        time.sleep(0.06)
    assert any("[rank 3] STILL IN PHASE" in r.message for r in caplog.records)


def test_an_infinite_heartbeat_interval_arms_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed: list[float] = []

    def arm(timeout: float, **kwargs: object) -> None:
        del kwargs
        armed.append(timeout)

    monkeypatch.setattr(faulthandler, "dump_traceback_later", arm)
    before = threading.active_count()
    with _phase_heartbeat("single process", interval_s=math.inf):
        assert threading.active_count() == before
    assert armed == []


class _FakeCudaEvent:
    def __init__(self, *, enable_timing: bool) -> None:
        self.enable_timing = enable_timing
        self.recorded = 0

    def record(self) -> None:
        self.recorded += 1

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end_event: _FakeCudaEvent) -> float:
        del end_event
        return 1.0


def test_cuda_event_pairs_are_recorded_into_the_phase_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With event timing on, a metric phase is bracketed by two events."""
    config = TrainLoop.Config(
        step=_WarmupStep.Config(),
        dataset=_WarmupDataset.Config(),
    )
    config.checkpointer = None
    config.max_steps = 0
    config.phase_timer = PhaseTimer.Config(enabled=True, cuda_events=True)
    loop = config.make()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)

    events = loop._cuda_event_pair()

    assert events is not None
    start, end = events
    assert isinstance(start, _FakeCudaEvent)
    assert isinstance(end, _FakeCudaEvent)
    assert start.recorded == 1
    assert end.recorded == 0
    loop._record_cuda_timing("eval_metric_acc_update", events)
    assert end.recorded == 1
    timer = loop.phase_timer
    assert isinstance(timer, PhaseTimer)
    assert timer._cuda_events == {"eval_metric_acc_update": [(start, end)]}


class _ReplayDataset:
    """Stateful test dataset whose batches consume and expose RNG progression."""

    class Config(Fig["_ReplayDataset"], make_with_kwargs=True):
        consume_eval_rng: bool = False
        """Whether evaluation consumes the training RNG stream."""

    def __init__(self, consume_eval_rng: bool = False) -> None:
        self.timer_epoch = CheckpointableStepTimer()
        self.cursor = 0
        self.batches: list[Tensor] = []
        self.consume_eval_rng = consume_eval_rng

    def train_dataloader(self) -> Iterator[dict[str, Tensor]]:
        """Return the cursor-preserving train iterator."""
        return self

    def eval_dataloader(self) -> Iterator[dict[str, Tensor]]:
        """Return validation data, optionally consuming the shared RNG stream."""
        if self.consume_eval_rng:
            return iter(
                [
                    {
                        "media": torch.stack(
                            (torch.rand(()), torch.zeros(())),
                        ).unsqueeze(0),
                        "label": torch.tensor([0]),
                    },
                ],
            )
        return iter(())

    def __iter__(self) -> _ReplayDataset:
        return self

    def __next__(self) -> dict[str, Tensor]:
        """Yield one cursor-labelled random batch."""
        if self.cursor == 4:
            raise StopIteration
        batch = torch.tensor([[float(self.cursor), torch.rand(()).item()]])
        self.cursor += 1
        self.batches.append(batch.clone())
        return {"media": batch, "label": torch.tensor([self.cursor % 2])}

    class StateDict(TypedDict):
        """The next unconsumed record."""

        cursor: int

    def state_dict(self) -> StateDict:
        """Persist the next unconsumed record."""
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the next unconsumed record."""
        self.cursor = cast(_ReplayDataset.StateDict, state_dict)["cursor"]


def _skip_after_first_train_step(
    original_train_step: Callable[[dict[str, object]], None],
    calls: list[int],
    batch: dict[str, object],
) -> None:
    """Run only the first patched train step."""
    calls[0] += 1
    if calls[0] == 1:
        original_train_step(batch)


def _raise_after_eval(eval_fn: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Raise after the real eval consumes its batch and RNG."""
    eval_fn()
    raise EvalTimeLimitError("injected failure after eval work")


def _raise_checkpoint_write(target: CheckpointableProtocol, step: int) -> None:
    """Fail a recovery save without replacing the primary eval error."""
    del target, step
    raise RuntimeError("injected checkpoint persistence failure")


def _record_maybe_save(
    calls: list[str],
    target: CheckpointableProtocol,
    step: int,
) -> bool:
    """Record a recovery cadence-save call."""
    del target, step
    calls.append("maybe_save")
    return True


def _record_save(calls: list[str], target: CheckpointableProtocol, step: int) -> None:
    """Record a recovery forced-save call."""
    del target, step
    calls.append("save")


def _assert_checkpoint_matches_interrupted_state(
    checkpoint: Path,
    interrupted: TrainLoop.StateDict,
) -> None:
    """Assert a durable checkpoint is the state observed after the eval error."""
    assert checkpoint.exists()
    _assert_state_dict_equal(_load_loop_checkpoint(checkpoint), expected=interrupted)


def _load_loop_checkpoint(path: Path) -> TrainLoop.StateDict:
    """Read a checkpoint the loop wrote; its schema is the loop's."""
    saved = cast(object, torch.load(path, weights_only=True))
    assert isinstance(saved, dict)
    return cast(TrainLoop.StateDict, saved)


def _assert_state_dict_equal(
    actual: TrainLoop.StateDict,
    expected: TrainLoop.StateDict,
) -> None:
    """Compare the managed model, optimizer, data, and RNG checkpoint state."""
    torch.testing.assert_close(actual["step"], expected["step"], rtol=0, atol=0)
    assert actual["dataset"] == expected["dataset"]
    assert actual["metrics_train"] == expected["metrics_train"]
    assert actual["metrics_eval"] == expected["metrics_eval"]
    assert "rng" in actual
    assert "rng" in expected
    _assert_rng_state_equal(actual["rng"], expected=expected["rng"])


def _assert_rng_state_equal(actual: RngState, expected: RngState) -> None:
    """Compare every managed CPU and optional CUDA RNG stream exactly."""
    assert actual.keys() == expected.keys()
    assert actual["python"] == expected["python"]
    assert torch.equal(actual["torch"], expected["torch"])
    if "numpy" in actual:
        assert "numpy" in expected
        assert actual["numpy"] == expected["numpy"]
    if "cuda" in actual:
        assert "cuda" in expected
        for actual_state, expected_state in zip(
            actual["cuda"],
            expected["cuda"],
            strict=True,
        ):
            assert torch.equal(actual_state, expected_state)
    if "cuda_uuids" in actual:
        assert "cuda_uuids" in expected
        assert actual["cuda_uuids"] == expected["cuda_uuids"]


def _metric_float(metrics: Mapping[str, object], key: str) -> float:
    """Return a numeric tracker metric, or a failing sentinel when absent."""
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else -1.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
