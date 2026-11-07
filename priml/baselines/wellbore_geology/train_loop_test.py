"""Check the fold loop, its selection rules and complete native fold training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from configgle import Fig, InlineConfig, Makes
from torch.utils.data import DataLoader

import pytest
import torch

from priml.baselines.wellbore_geology.data import (
    FoldObservations,
    WellDataset,
    WellLoader,
)
from priml.baselines.wellbore_geology.folds import GeographicPrior
from priml.baselines.wellbore_geology.metrics import PooledRMSE
from priml.baselines.wellbore_geology.testing import (
    BatchedFoldData,
    probe_recipe,
    tiny_family,
    write_study_inputs,
)
from priml.baselines.wellbore_geology.train_loop import (
    FoldData,
    WellboreFoldLoop,
    require_fresh_runtime,
)
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import SingleProcess, runtime_initialized
from priml.train.epoch_loop import EarlyStopping, EpochTrainLoop
from priml.train.profiler import PhaseTimer


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from priml.baselines.wellbore_geology.data import WellRecord
    from priml.baselines.wellbore_geology.folds import RelativeSurface


def test_original_row_metric_is_bound_to_the_native_eval_slot() -> None:
    config = probe_recipe()
    assert config.metrics_eval
    loop = config.finalized().make()
    try:
        loop.eval()
        metric = loop.metrics_eval[""]
        assert isinstance(metric, PooledRMSE)
        state = metric.state_dict()
        predictions = state["predictions"]
        assert sum(value.numel() for value in predictions) == 32
        assert metric.compute() == {"rmse": loop.best_score}
    finally:
        loop.close()


@pytest.mark.parametrize("validation", [False, True])
def test_real_loop_owns_updates_and_complete_best_checkpoint(validation: bool) -> None:
    config = probe_recipe()
    assert isinstance(config.dataset, BatchedFoldData.Config)
    config.dataset.has_validation = validation
    if not validation:
        config.early_stopping = None
    loop = config.finalized().make()
    loop.train()
    step = loop.native_step
    assert step.global_step == step.ema.global_step == 6
    assert loop.best_epoch == 3
    if validation:
        assert [row["epoch"] for row in loop.val_history] == [1, 2, 3]
        assert loop.best_score == loop.val_history[-1]["rmse"]
    else:
        assert loop.val_history == []
        assert loop.best_score == float("inf")
        assert loop.fold_data.eval_dataloader() == ()
    assert loop.best_state is not None
    with step.ema.apply_to(step.model):
        for name, value in step.model.state_dict().items():
            assert torch.equal(value, loop.best_state[name])
            assert value.data_ptr() != loop.best_state[name].data_ptr()
    with pytest.raises(NotImplementedError, match="mid-fold native DataLoader"):
        loop.fold_data.load_state_dict({})


def test_caller_owned_runtime_is_rejected_before_model_work() -> None:
    runtime = SingleProcess.Config(device="cpu").make()
    runtime.initialize()
    try:
        with pytest.raises(ValueError, match="caller-owned runtime"):
            require_fresh_runtime()
        assert runtime_initialized()
    finally:
        runtime.destroy()


def test_real_loop_drives_and_closes_injected_timer_and_tracker() -> None:
    config = probe_recipe()
    config.phase_timer = PhaseTimer.Config(enabled=True, heartbeat_interval_sec=0)
    config.tracker = RecordingTracker.Config()
    config.num_steps_log = 1
    loop = config.finalized().make()
    loop.train()
    tracker = cast("RecordingTracker", loop.tracker)
    assert tracker.closed
    assert any(prefix == "train/" for prefix, _ in tracker.records)
    assert "model_init" in loop.phase_timer.summary()
    assert "data_load" in loop.phase_timer.summary()


class RecordingTracker:
    """Observe actual shared-loop publication without external services."""

    class Config(Fig["RecordingTracker"]):
        """Construct a fresh recording sink with no live state in configuration."""

    def __init__(self, config: Config) -> None:
        del config
        self.records: list[tuple[str, int]] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Record each delivery to the actual configured tracker."""
        del metrics
        self.records.append((prefix, step))

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Reject unexpected image logging."""
        del key, images, step
        raise AssertionError

    def log_notes(self, notes: str) -> None:
        """Accept startup notes."""
        del notes

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


@pytest.mark.parametrize("seconds", [float("nan"), -float("inf"), 0.0, -1.0])
def test_real_wall_slot_rejects_invalid_caps(seconds: float) -> None:
    config = probe_recipe()
    config.max_time = seconds
    with pytest.raises(ValueError, match="Training wall cap must be positive"):
        config.make()


@pytest.mark.parametrize("seconds", [float("inf"), 0.01, 300.0])
def test_real_wall_slot_accepts_positive_caps(seconds: float) -> None:
    config = probe_recipe()
    config.max_time = seconds
    assert config.finalized().max_time == seconds


@pytest.mark.parametrize(
    ("wells", "drop_last", "expected"),
    [(1, True, 3), (5, True, 3), (5, False, 6)],
)
def test_reconstructible_config_resolves_original_complete_update_horizon(
    wells: int,
    drop_last: bool,
    expected: int,
) -> None:

    config = WellboreFoldLoop.Config()
    config.max_epochs = 3
    config.step.lr_schedule = RecursiveCosine.Config()
    config.step.accumulate_grad_batches = 2
    config.dataset.loader.batch_size = 2
    config.dataset.loader.drop_last = drop_last
    observations = config.dataset.loader.dataset.observations
    assert isinstance(observations, FoldObservations.Config)
    observations.well_ids = tuple(str(index) for index in range(wells))
    finalized = config.finalized()
    schedule = finalized.step.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    assert schedule.T_max == expected


def test_family_seed_derives_unset_fold_streams() -> None:
    config = WellboreFoldLoop.Config()
    config.base_seed = 10
    config.fold = 2
    finalized = config.finalized()
    assert finalized.seed == 12
    assert finalized.dataset.loader.seed == 10 + 1000 * 2 + 1
    assert finalized.dataset.loader.dataset.streams.seed == 12


def test_family_seed_keeps_declared_fold_streams() -> None:
    config = WellboreFoldLoop.Config()
    config.base_seed = 10
    config.fold = 2
    config.seed = 5
    config.dataset.loader.seed = 6
    config.dataset.loader.dataset.streams.seed = 7
    finalized = config.finalized()
    assert finalized.seed == 5
    assert finalized.dataset.loader.seed == 6
    assert finalized.dataset.loader.dataset.streams.seed == 7


def test_without_a_family_seed_the_fold_streams_stay_unset() -> None:
    finalized = WellboreFoldLoop.Config().finalized()
    assert finalized.dataset.loader.seed == -1
    assert finalized.dataset.loader.dataset.streams.seed == -1


# ``current_epoch`` counts completed epochs, so the truncated pass scores as epoch 1.
@pytest.mark.parametrize(("truncated", "epochs"), [(False, [1]), (True, [1, 1])])
def test_a_truncated_last_epoch_is_scored_only_when_asked(
    *,
    truncated: bool,
    epochs: list[int],
) -> None:
    config = probe_recipe()
    # Two updates per epoch: the third stops partway through the second epoch.
    config.max_steps = 3
    config.evaluate_truncated = truncated
    loop = config.finalized().make()
    loop.train()
    assert loop.native_step.global_step == 3
    assert [row["epoch"] for row in loop.val_history] == epochs


def test_real_fold_data_builds_training_and_held_out_loaders(tmp_path: Path) -> None:
    write_study_inputs(tmp_path, rows=13)
    config = tiny_family(particle=False).training.dataset.copy_tree()
    train = config.loader.dataset.observations
    validation = config.validation
    assert isinstance(train, FoldObservations.Config)
    assert isinstance(validation, FoldObservations.Config)
    for observations, wells in ((train, ("well0", "well1")), (validation, ("well2",))):
        observations.directory = tmp_path / "train"
        observations.geography = GeographicPrior.Config()
        observations.support_well_ids = ("well0", "well1")
        observations.well_ids = wells
    config.validation_batch_size = 3
    config.loader.seed = 3
    config.loader.dataset.streams.seed = 4
    data = FoldData(config.finalized())
    assert isinstance(data.train, DataLoader)
    assert data.train.batch_size == 2
    assert isinstance(data.validation, DataLoader)
    assert data.validation.batch_size == 3
    assert data.eval_dataloader() is data.validation
    assert data.train_dataloader() is data.train
    assert set(data.held_out_surfaces) == {"well2"}


def test_unrun_loop_starts_without_a_selection() -> None:
    loop = probe_recipe().finalized().make()
    try:
        assert loop.best_epoch == "NA"
        assert loop.best_state is None
        assert loop.best_score == float("inf")
        assert loop.val_history == []
    finally:
        loop.close()


@pytest.mark.parametrize("seconds", [0.5, 1.0])
def test_small_positive_wall_caps_are_allowed(seconds: float) -> None:
    config = probe_recipe()
    config.max_time = seconds
    loop = config.finalized().make()
    loop.close()


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan")])
def test_nonpositive_wall_caps_are_rejected(seconds: float) -> None:
    config = probe_recipe()
    config.max_time = seconds
    with pytest.raises(
        ValueError,
        match=r"^Training wall cap must be positive or positive infinity\.$",
    ):
        WellboreFoldLoop(config.finalized())


@pytest.mark.parametrize(
    ("epochs", "accumulation"),
    [(0, 1), (1, 0), (0.5, 1)],
)
def test_epoch_and_accumulation_counts_must_be_positive(
    epochs: float,
    accumulation: int,
) -> None:
    config = probe_recipe()
    config.max_epochs = epochs
    config.step.accumulate_grad_batches = accumulation
    with pytest.raises(
        ValueError,
        match=r"^Epoch and accumulation counts must be positive\.$",
    ):
        WellboreFoldLoop(config.finalized())


def test_single_epoch_and_single_accumulation_are_allowed() -> None:
    config = probe_recipe()
    config.max_epochs = 1
    config.step.accumulate_grad_batches = 1
    WellboreFoldLoop(config.finalized()).close()


def test_early_stopping_must_follow_the_selection_metric() -> None:
    config = probe_recipe()
    config.early_stopping = EarlyStopping.Config(metric="total_loss")
    with pytest.raises(
        ValueError,
        match=r"^Early stopping must follow the selection metric 'rmse'\.$",
    ):
        WellboreFoldLoop(config.finalized())


def test_fold_data_without_validation_has_no_held_out_surfaces() -> None:
    config = probe_recipe()
    assert isinstance(config.dataset, BatchedFoldData.Config)
    config.dataset.has_validation = False
    config.early_stopping = None
    loop = config.finalized().make()
    try:
        assert loop.fold_data.state_dict().keys() == {"timer_epoch"}
        with pytest.raises(ValueError, match="_held_out"):
            _ = loop.fold_data.held_out_surfaces
    finally:
        loop.close()


def test_a_repeated_score_keeps_the_first_selection() -> None:
    loop = probe_recipe().finalized().make()
    try:
        loop.eval()
        epoch, state = loop.best_epoch, loop.best_state
        loop.eval()
        assert len(loop.val_history) == 2
        assert loop.best_epoch == epoch
        assert loop.best_state is state
    finally:
        loop.close()


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
@pytest.mark.parametrize("validation", [False, True])
def test_fold_data_builds_independent_training_and_validation_loaders(
    device: str | None,
    validation: bool,
) -> None:
    config = FoldData.Config()
    config.loader = RecordingLoader.Config()
    config.loader.seed = 11
    config.loader.batch_size = 2
    config.validation_batch_size = 3
    config.validation_seed_offset = 7
    config.device = None if device is None else torch.device(device)
    if not validation:
        config.validation = None
    data = FoldData(config)
    assert data.train_dataloader() is data.train
    assert isinstance(data.train, DataLoader)
    assert data.train.pin_memory == (device == "cuda")
    assert data.state_dict() == {"timer_epoch": {"global_count": 0, "global_sec": 0.0}}
    assert config.loader.pin_memory is False
    assert config.loader.shuffle
    assert config.loader.drop_last
    assert config.loader.dataset.transform.paths is not None
    assert config.loader.dataset.transform.images is not None
    if validation:
        held_out = data._held_out
        assert isinstance(held_out, RecordingLoader)
        assert held_out.config.seed == 18
        assert held_out.config.batch_size == 3
        assert held_out.config.pin_memory == (device == "cuda")
        assert held_out.config.shuffle is False
        assert held_out.config.drop_last is False
        assert held_out.config.dataset.transform.paths is None
        assert held_out.config.dataset.transform.images is None
        assert held_out.config.dataset.observations == config.validation
        assert data.eval_dataloader() is held_out.loader
        observations = held_out.dataset.observations
        assert isinstance(observations, FoldObservations)
        assert data.held_out_surfaces is observations.surfaces
        assert data.validation is not data.train
    else:
        assert data.validation is None
        assert data.eval_dataloader() == ()
        with pytest.raises(ValueError, match="_held_out"):
            _ = data.held_out_surfaces


class RecordingLoader(WellLoader):
    """Replace dataset and DataLoader construction at the fold's data boundary."""

    class Config(Makes["RecordingLoader"], WellLoader.Config, kw_only=True):
        """Keep the native loader settings available for inspection."""

    def __init__(self, config: WellLoader.Config) -> None:
        self.config = config
        self.loader = DataLoader[dict[str, object]](
            [],
            pin_memory=config.pin_memory,
        )
        dataset = config.dataset.copy_tree()
        dataset.observations = InlineConfig(EmptyFoldObservations)
        dataset.transform.paths = dataset.transform.images = None
        dataset.streams.seed = config.seed
        self.dataset = WellDataset(dataset.finalized())


class EmptyFoldObservations(FoldObservations):
    """Complete empty observation boundary with valid sequence behavior."""

    def __init__(self) -> None:
        self.wells: tuple[WellRecord, ...] = ()
        self.surfaces: dict[str, RelativeSurface] = {}


@pytest.mark.parametrize("explicit", [False, True])
def test_fold_seeds_fill_only_unset_random_streams(explicit: bool) -> None:
    config = WellboreFoldLoop.Config()
    config.base_seed = 7
    config.fold = 2
    config.fold_seed_stride = 13
    config.loader_seed_offset = 3
    if explicit:
        config.seed = 19
        config.dataset.loader.seed = 23
        config.dataset.loader.dataset.streams.seed = 29
    resolved = config.finalized()
    assert resolved.seed == (19 if explicit else 9)
    assert resolved.dataset.loader.seed == (23 if explicit else 36)
    assert resolved.dataset.loader.dataset.streams.seed == (29 if explicit else 9)
    assert config.seed == (19 if explicit else None)
    assert config.dataset.loader.seed == (23 if explicit else -1)
    assert config.dataset.loader.dataset.streams.seed == (29 if explicit else -1)


@pytest.mark.parametrize(
    ("is_final", "truncated", "evaluated", "force", "expected"),
    [
        (True, True, False, False, True),
        (False, True, False, False, False),
        (True, False, False, False, False),
        (True, True, True, False, False),
        (False, False, False, True, True),
    ],
)
def test_truncated_final_evaluation_forces_only_an_unevaluated_terminal_epoch(
    is_final: bool,
    truncated: bool,
    evaluated: bool,
    force: bool,
    expected: bool,
) -> None:
    loop = WellboreFoldLoop.__new__(WellboreFoldLoop)
    loop.config = WellboreFoldLoop.Config()
    loop.config.evaluate_truncated = truncated
    loop._terminal_epoch_evaluated = evaluated
    with patch.object(EpochTrainLoop, "_maybe_eval", autospec=True) as evaluate:
        loop._maybe_eval(is_final=is_final, force=force)
    evaluate.assert_called_once_with(loop, is_final=is_final, force=expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
