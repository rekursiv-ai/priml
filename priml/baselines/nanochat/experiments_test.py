"""Tests for the nanochat experiment ladder.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
ladder stays checkable on any machine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import json
import math

import numpy as np
import pytest

from priml.baselines.nanochat import experiments
from priml.baselines.nanochat.data import token_bytes_fingerprint
from priml.baselines.nanochat.experiments import NanoChatLoop
from priml.baselines.nanochat.metric import BitsPerByte
from priml.baselines.nanochat.train_step import NanoChatTrainStep


LADDER: list[tuple[str, Callable[[], NanoChatLoop.Config]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp_smoke", experiments.exp_smoke),
]


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
) -> None:
    """A config must build without a dataset or a GPU."""
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "nanochat"


def test_exp000_turns_both_mechanisms_off() -> None:
    """The baseline must be the bar, not a third variant.

    If exp000 already windowed its attention, exp001 would measure nothing and
    the ladder's comparison would be against an unstated recipe.
    """
    cfg = experiments.exp000()
    assert cfg.step.model.window_pattern == "L"
    assert cfg.step.model.value_embedding_stride == 0


def test_exp001_changes_only_the_window() -> None:
    base, fork = experiments.exp000(), experiments.exp001()
    assert base.step.model.window_pattern == "L"
    assert fork.step.model.window_pattern == "SSSL"
    assert fork.step.model.value_embedding_stride == (
        base.step.model.value_embedding_stride
    )
    assert fork.step.time_budget_sec == base.step.time_budget_sec
    assert fork.step.model.channels == base.step.model.channels


def test_exp002_changes_only_the_value_embeddings() -> None:
    base, fork = experiments.exp001(), experiments.exp002()
    assert base.step.model.value_embedding_stride == 0
    assert fork.step.model.value_embedding_stride == 2
    assert fork.step.model.window_pattern == base.step.model.window_pattern
    assert fork.step.time_budget_sec == base.step.time_budget_sec


def test_the_value_embedding_stride_follows_a_changed_depth() -> None:
    """A stride survives a fork that changes the depth; indices would not.

    Computing the layer list in the factory snapshots whatever ``num_layers``
    was at that moment, so a fork narrowing the model carries indices for a
    stack that no longer exists -- and the model rejects them.
    """
    cfg = experiments.exp002()
    cfg.step.model.num_layers = 4
    final = cfg.copy_tree().finalize()
    assert final.step.model.value_embedding_layers == [1, 3]


def test_the_budget_and_the_schedule_horizon_agree() -> None:
    """A schedule annealing past the stop wastes the tail; short of it, the
    run trains its last steps at a rate the recipe never intended.
    """
    for name, factory in LADDER:
        config = factory()
        assert config.max_time == config.step.time_budget_sec, name
        assert config.max_time_kind == "train", name


def test_the_loop_reads_the_steps_budget_clock() -> None:
    """Stop condition, reported time, and schedules must share one clock.

    ``TrainLoop`` rebases after one step; this baseline excludes a configured
    warmup. If the loop kept its own clock the run would anneal against one
    budget and stop on another, differing by the whole warmup.
    """
    loop = NanoChatLoop.__new__(NanoChatLoop)
    step = NanoChatTrainStep.__new__(NanoChatTrainStep)
    step.elapsed_sec = 12.5
    loop.step = step
    assert loop._train_elapsed() == 12.5


def test_the_dataset_batch_follows_the_steps_pass_size() -> None:
    """Two places naming the same number silently disagree; one propagates."""
    config = experiments.exp000()
    config.step.rows_per_pass = 8
    assert config.copy_tree().finalize().dataset.batch_size == 8


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_every_experiments_eval_geometry_is_constructible(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
    tmp_path: Path,
) -> None:
    """A shipped experiment must survive building its dataset, not just its config.

    Finalizing proves the tree is coherent; it does not run the validation that
    lives in ``__init__``. A cap that is not a whole number of eval batches, or
    a batch wider than the split, therefore passes every config-only test and
    fails the moment a run reaches for data.
    """
    config = factory()
    model = config.step.model
    # A split matching what this experiment declares, so the geometry check
    # passes and the EVAL GEOMETRY is what the test exercises.
    _prepared(tmp_path, rows=64, vocab=model.vocab_size, seq=model.max_seq_len)
    config.base_dir = "/"
    config.dataset.working_dir = str(tmp_path)
    config.dataset.device = "cpu"
    built = config.copy_tree().finalize().dataset.make()
    assert list(built.eval_dataloader()), name


def _prepared(root: Path, *, rows: int, vocab: int, seq: int) -> None:
    """Write a split at the given geometry."""
    for split in ("train", "val"):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(
            directory / "all__tokens.npy",
            np.zeros((rows, seq + 1), dtype=np.uint16),
        )
        lengths = np.ones(vocab, dtype=np.int32)
        lengths[0] = 0
        np.save(directory / "all__token_bytes.npy", lengths)
        (directory / "dataset.json").write_text(
            json.dumps(
                {
                    "vocab_size": vocab,
                    "max_seq_len": seq,
                    "token_bytes_sha256": token_bytes_fingerprint(lengths),
                },
            ),
        )


def test_the_dataset_inherits_the_models_geometry() -> None:
    """The model declares the geometry; the dataset verifies data against it.

    Without the push the two are independent copies agreeing only by coincident
    defaults, so ``exp_smoke`` -- which narrows the model -- would load rows of
    a width nothing checked.
    """
    config = experiments.exp_smoke().copy_tree().finalize()
    assert config.dataset.max_seq_len == config.step.model.max_seq_len
    assert config.dataset.vocab_size == config.step.model.vocab_size


def test_the_score_is_bits_per_byte() -> None:
    """A per-token score would rank a coarser tokenizer better for free."""
    assert isinstance(experiments.exp000().metrics["val"], BitsPerByte.Config)


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke, base = experiments.exp_smoke(), experiments.exp000()
    assert smoke.step.time_budget_sec < base.step.time_budget_sec
    assert smoke.step.model.channels < base.step.model.channels
    assert smoke.step.model.num_layers < base.step.model.num_layers
    assert smoke.step.model.max_seq_len < base.step.model.max_seq_len
    assert not smoke.step.compile
    # A finite bound: exp000 stops on its time budget and leaves max_steps at
    # infinity, against which any value would compare smaller.
    assert smoke.max_steps < 100
    assert math.isinf(base.max_steps)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
