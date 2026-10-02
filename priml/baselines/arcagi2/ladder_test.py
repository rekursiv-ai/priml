"""The ARC-AGI-2 ladder: each rung's single delta, and bit-for-bit numerics.

Numerics: every rung's step reproduces the frozen ``testdata/<expNNN>.pt``
trajectory recorded from the implementation it was ported from, through the
arcagi1 recorder. Configs: each fork changes exactly what its docstring says.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PerOutputPass,
    SignalDumpTracker,
    StrictPass,
)
from priml.baselines.arcagi1.model import UrmRecurrence
from priml.baselines.arcagi1.train_step_test import (
    PortSubject,
    port_config,
    record,
    shrink_model,
)
from priml.baselines.arcagi2 import experiments
from priml.baselines.arcagi2.experiments import (
    LEAKED_TASKS,
    NUM_PUZZLE_IDENTIFIERS,
    TOTAL_TRAIN_STEPS,
)
from priml.baselines.arcagi2.warm_start import WarmStart
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors
from priml.train.checkpointer import Checkpointer
from priml.train.tracker import TrackerList

import priml.baselines.arcagi1.experiments


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.arcagi1.train_step import TrmTrainStep
    from priml.baselines.arcagi2.experiments import Arc2TrmTrainLoop


_CWD: Final = Path(__file__).resolve().parent

RUNGS: Final = ("exp001", "exp002", "exp003", "exp004", "exp005")
"""Every rung with its own numerics; exp006/exp007 differ only in initialization."""


def rung(name: str) -> Arc2TrmTrainLoop:
    """Build one rung of the ladder."""
    return cast("Callable[[], Arc2TrmTrainLoop]", getattr(experiments, name))()


def ladder_config(name: str) -> TrmTrainStep.Config:
    """Return ``name``'s step shrunk by size only, exactly as arcagi1's are."""
    step = rung(name).step
    shrunk = port_config("exp004")
    shrunk.model = step.model
    shrunk.optimizer = step.optimizer
    shrunk.signals = step.signals
    shrunk.emulate_precision_casts = step.emulate_precision_casts
    reference = port_config("exp004")
    for field in ("parallelism", "dtype_autocast", "total_train_steps", "ema"):
        setattr(shrunk, field, getattr(reference, field))
    shrunk.pool = step.pool
    assert isinstance(shrunk.pool, AtomicPool.Config)
    size = port_config("exp004").pool
    assert isinstance(size, AtomicPool.Config)
    shrunk.pool.batch_size = size.batch_size
    shrunk.pool.max_steps = size.max_steps
    shrunk.model.compile_core = None
    shrink_model(shrunk.model)
    return shrunk


def run_ladder(name: str) -> dict[str, Tensor]:
    """Record ``name``'s shrunk step from seed 0."""
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        return record(PortSubject.from_config(ladder_config(name)))


def golden_path(name: str) -> Path:
    """Where the golden for ``name`` lives."""
    return _CWD / "testdata" / f"{name}.pt"


@pytest.mark.parametrize("name", RUNGS)
def test_rung_replays_bit_for_bit(name: str) -> None:
    """The rung reproduces its frozen trajectory with zero mismatches."""
    report = mismatches(
        read_tensors(golden_path(name)),
        run_ladder(name),
    )
    assert not report, "\n".join(report)


@pytest.mark.parametrize("name", RUNGS)
def test_rung_golden_bites(name: str) -> None:
    """A one-ULP weight nudge is reported, not absorbed."""
    expected = read_tensors(golden_path(name))
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        subject = PortSubject.from_config(ladder_config(name))
        with torch.no_grad():
            weight = next(subject.step.model.parameters())
            weight.view(torch.int32).view(-1)[0] += 1
        actual = record(subject)
    assert mismatches(expected, actual)


@pytest.mark.parametrize("name", [*RUNGS, "exp006", "exp007"])
def test_every_rung_is_sized_for_arc2(name: str) -> None:
    cfg = rung(name).copy_tree().finalize()
    prefix = cfg.step.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    assert prefix.num_puzzles == NUM_PUZZLE_IDENTIFIERS
    assert cfg.dataset.num_puzzle_identifiers == NUM_PUZZLE_IDENTIFIERS
    pool = cfg.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert prefix.batch_size == pool.batch_size == cfg.dataset.batch_size
    assert cfg.max_steps == cfg.step.total_train_steps == TOTAL_TRAIN_STEPS
    assert cfg.dataset.epochs_per_iter == 4
    assert cfg.eval_extras_every_eval
    for metric in cfg.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        assert metric.working_dir == cfg.dataset.working_dir
        assert [type(rule) for rule in metric.rules] == [
            StrictPass.Config,
            PerOutputPass.Config,
        ]


@pytest.mark.parametrize(
    ("name", "source"),
    [("exp001", "exp004"), ("exp002", "exp005"), ("exp003", "exp006")],
)
def test_trm_rungs_keep_the_arcagi1_step(name: str, source: str) -> None:
    """Only the task-table size and the horizon move off the ARC-AGI-1 recipe."""
    ours = rung(name).step
    theirs = cast(
        "Callable[[], priml.baselines.arcagi1.experiments.TrmTrainLoop]",
        getattr(priml.baselines.arcagi1.experiments, source),
    )().step
    prefix = theirs.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    prefix.num_puzzles = NUM_PUZZLE_IDENTIFIERS
    theirs.total_train_steps = TOTAL_TRAIN_STEPS
    assert ours.pformat() == theirs.pformat()


def test_exp004_is_exp002_with_the_urm() -> None:
    cfg = rung("exp004")
    assert isinstance(cfg.step.model.recurrence, UrmRecurrence.Config)
    assert cfg.step.signals is None
    assert cfg.dataset.batch_size == cfg.dataset.eval_batch_size == 96
    assert cfg.dataset.spatial_eval_views == 0
    assert set(cfg.metrics_eval) == {""}


def test_exp005_adds_spatial_evaluation_and_rotating_dumps() -> None:
    cfg = rung("exp005")
    assert cfg.step.signals is not None
    assert cfg.dataset.spatial_eval_views == 2
    assert str(cfg.dataset.working_dir).endswith("-spatialeval-v2")
    assert str(cfg.dataset.working_dir).startswith(str(cfg.dataset.source_dataset_dir))
    assert cfg.dataset.eval_batch_size == 256
    assert set(cfg.metrics_eval) == {"", "spatial_eq", "spatial_big"}
    assert isinstance(cfg.tracker, TrackerList.Config)
    signals = cfg.tracker.trackers["signals"]
    assert isinstance(signals, SignalDumpTracker.Config)
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    assert signals.keep_last_n == cfg.checkpointer.keep_last_n == 8
    assert signals.keep_every == cfg.checkpointer.keep_every == 40_000
    assert signals.keep_every % cfg.num_steps_eval == 0


def test_exp006_warm_starts_and_scores_a_leak_free_slice() -> None:
    base, cfg = rung("exp005"), rung("exp006")
    assert isinstance(cfg.step.warm_start, WarmStart.Config)
    assert str(cfg.step.warm_start.path).endswith("step_00388670.pt")
    assert cfg.step.warm_start.rename is not None
    assert "leakage" in (experiments.exp006.__doc__ or "")
    leakfree = cfg.metrics_eval.pop("leakfree")
    assert isinstance(leakfree, CanonicalPassK.Config)
    assert leakfree.exclude_tasks == list(LEAKED_TASKS)
    cfg.step.warm_start = None
    cfg.experiment_name = base.experiment_name
    assert cfg.pformat() == base.pformat()


def test_exp007_changes_only_the_source_and_its_recurrence() -> None:
    base, cfg = rung("exp006"), rung("exp007")
    assert isinstance(cfg.step.warm_start, WarmStart.Config)
    assert isinstance(base.step.warm_start, WarmStart.Config)
    assert str(cfg.step.warm_start.path).endswith("exp029/checkpoints/step_00370000.pt")
    recurrence = cfg.step.model.recurrence
    assert isinstance(recurrence, DeepRecurrence.Config)
    assert recurrence.slow_cycles == 3
    base.step.warm_start.path = cfg.step.warm_start.path
    base_recurrence = base.step.model.recurrence
    assert isinstance(base_recurrence, DeepRecurrence.Config)
    base_recurrence.slow_cycles = 3
    base.experiment_name = cfg.experiment_name
    assert cfg.pformat() == base.pformat()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
