"""Verify the full recipe, finalized resource paths, and printable defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

import numpy as np
import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.data import PuzzleData
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PerOutputPass,
    SignalDumpTracker,
    StrictPass,
)
from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.experiments import (
    LEAKED_TASKS,
    NUM_PUZZLE_IDENTIFIERS,
    TOTAL_TRAIN_STEPS,
    _on_arc2,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp_smoke,
)
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi2.scripts.build_dataset import (
    ARC2_DATASET_DIR,
    arc2_aug_policy_template,
    arc2_spatial_eval_template,
)
from priml.baselines.arcagi2.train_step import ArcDataParallel, ArcTrainStep
from priml.baselines.arcagi2.train_step_test import training_config
from priml.baselines.arcagi2.warm_start import WarmStart
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.swiglu import SwiGLU
from priml.paths import resolve_working_dir
from priml.runtime import MultiProcess, SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import TrackerList


if TYPE_CHECKING:
    from pathlib import Path


def test_arc2_retargets_a_complete_arc1_recipe() -> None:
    source = experiments.exp004()
    converted = _on_arc2(source, "probe")

    assert converted.experiment_name == "probe"
    assert converted.dataset.working_dir == "/datasets/arc2concept-aug-1000"


def test_arc2_retarget_skips_arc1_only_config_fields() -> None:
    class Arc1OnlyPuzzleData(PuzzleData.Config):
        arc1_only: int = 7
        """Field that the ARC2 data config does not declare."""

    class Arc1OnlyTrmTrainLoop(experiments.TrmTrainLoop):
        arc1_only: int = 11
        """Field that the ARC2 loop config does not declare."""

    source = Arc1OnlyTrmTrainLoop()
    reference = experiments.exp004()
    source.update(reference, skip_missing=True)
    source.dataset = Arc1OnlyPuzzleData().update(
        reference.dataset,
        skip_missing=True,
    )
    converted = _on_arc2(source, "probe")

    assert not hasattr(converted, "arc1_only")
    assert not hasattr(converted.dataset, "arc1_only")


def test_arc2_experiment_forks_retarget_the_complete_recipe() -> None:
    for index, (name, config) in enumerate(
        (
            ("exp001", exp001()),
            ("exp002", exp002()),
            ("exp003", exp003()),
            ("exp004", exp004()),
            ("exp005", exp005()),
            ("exp006", exp006()),
            ("exp007", exp007()),
        ),
    ):
        assert config.experiment_name == name
        assert config.dataset.num_puzzle_identifiers == NUM_PUZZLE_IDENTIFIERS
        assert config.max_steps == config.step.total_train_steps == TOTAL_TRAIN_STEPS
        assert config.study_name == "arcagi2"
        assert config.dataset.epochs_per_iter == 4
        assert config.eval_extras_every_eval
        assert isinstance(config.checkpointer, Checkpointer.Config)
        assert config.checkpointer.save_every == 5_000
        prefix = config.step.model.prefix
        assert isinstance(prefix, SparsePuzzleEmbedding.Config)
        assert prefix.batch_size == config.dataset.batch_size
        for metric in config.metrics_eval.values():
            assert isinstance(metric, CanonicalPassK.Config)
            assert metric.working_dir == config.dataset.working_dir
            assert [type(rule) for rule in metric.rules] == [
                StrictPass.Config,
                PerOutputPass.Config,
            ]
        if index < 4:
            assert config.dataset.working_dir == "/datasets/arc2concept-aug-1000"
        else:
            assert str(config.dataset.working_dir).endswith("-spatialeval-v2")
        assert prefix.num_puzzles == NUM_PUZZLE_IDENTIFIERS


def test_exp004_uses_the_urm_recipe() -> None:
    config = exp004()
    assert config.dataset.batch_size == config.dataset.eval_batch_size == 96
    assert config.dataset.spatial_eval_views == 0
    assert set(config.metrics_eval) == {""}
    assert config.step.optimizer is not None
    assert config.step.pool is not None


def test_exp005_wires_spatial_eval_and_signal_retention() -> None:
    config = exp005()
    assert config.dataset.source_dataset_dir == arc2_aug_policy_template(
        translation_prob=0.2,
        scale_prob=0.2,
    )
    assert config.dataset.working_dir == arc2_spatial_eval_template(
        spatial_views=2,
        translation_prob=0.2,
        scale_prob=0.2,
    )
    assert config.dataset.spatial_eval_views == 2
    assert config.dataset.eval_batch_size == 256
    assert set(config.metrics_eval) == {"", "spatial_eq", "spatial_big"}
    assert isinstance(config.tracker, TrackerList.Config)
    signals = config.tracker.trackers["signals"]
    assert isinstance(signals, SignalDumpTracker.Config)
    assert isinstance(config.checkpointer, Checkpointer.Config)
    assert signals.keep_last_n == config.checkpointer.keep_last_n == 8
    assert signals.keep_every == config.checkpointer.keep_every == 40_000


def test_exp006_excludes_the_leaked_tasks() -> None:
    config = exp006()
    assert isinstance(config.step.warm_start, WarmStart.Config)
    assert str(config.step.warm_start.path) == (
        "/opt/scratch/runs/arcagi1/exp028_aug_eval/checkpoints/step_00388670.pt"
    )
    assert config.step.warm_start.rename is not None
    leakfree = config.metrics_eval["leakfree"]
    assert isinstance(leakfree, CanonicalPassK.Config)
    assert leakfree.exclude_tasks == list(LEAKED_TASKS)


def test_exp007_uses_its_source_and_recurrence() -> None:
    config = exp007()
    assert isinstance(config.step.warm_start, WarmStart.Config)
    assert str(config.step.warm_start.path) == (
        "/opt/scratch/runs/arcagi1/exp029/checkpoints/step_00370000.pt"
    )
    assert isinstance(config.step.model.recurrence, DeepRecurrence.Config)
    assert config.step.model.recurrence.slow_cycles == 3


def test_finalized_reference_recipe() -> None:
    config = exp000().finalize()
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.mesh_topology == {"dp": -1, "pp": 1, "tp": 1}
    assert isinstance(config.step.parallelism, ArcDataParallel.Config)
    assert config.step.parallelism.gradient_as_bucket_view
    assert config.step.pool is not None
    assert config.dataset.batch_size == config.step.pool.batch_size == 256
    assert config.dataset.make().prepared.eval_batch_size == 256
    assert config.dataset.epochs_per_iter == 4
    assert config.max_steps == config.step.total_train_steps == 541_580
    # The directory the builder writes, so the reader never looks elsewhere.
    assert config.dataset.working_dir == resolve_working_dir(
        "/opt/scratch",
        ARC2_DATASET_DIR,
    )
    metric = config.metrics_eval[""]
    assert isinstance(metric, PassK.Config)
    assert metric.working_dir == config.dataset.working_dir


def test_model_geometry_follows_the_dataset_spec() -> None:
    default = exp000().finalize()
    default_embedding = default.step.model.embedding
    assert isinstance(default_embedding, GridEmbedding.Config)
    assert default_embedding.grid_shape == (900,) == default.dataset.spec.grid_shape
    assert default.step.model.vocab_size == 12 == default.dataset.spec.vocab_size

    config = exp000()
    config.dataset.spec.max_grid = 2
    finalized = config.finalize()
    embedding = finalized.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_shape == (4,)
    assert finalized.step.model.vocab_size == config.dataset.spec.vocab_size


def test_resource_paths_follow_base_dir(tmp_path: Path) -> None:
    config = exp000()
    config.base_dir = tmp_path
    config = config.finalize()
    metric = config.metrics_eval[""]
    assert isinstance(metric, PassK.Config)
    assert config.dataset.working_dir == tmp_path / "datasets/arc2concept-aug-1000"
    assert metric.working_dir == config.dataset.working_dir


def test_smoke_is_single_process() -> None:
    config = exp_smoke().finalize()
    assert config.experiment_name == "exp_smoke"
    assert config.num_steps_eval == 2
    assert isinstance(config.runtime, SingleProcess.Config)
    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert config.max_steps == config.step.total_train_steps == 4
    assert config.step.warmup_steps == 0
    assert config.step.use_ema is False
    assert config.step.compile is None
    model = config.step.model
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    assert isinstance(model.block.ffn, SwiGLU.Config)
    assert isinstance(model.block.rope, RoPE.Config)
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    assert model.channels_in == 32
    assert model.block.attn.num_heads == 2
    assert model.block.attn.channels_head == model.block.rope.channels_head == 16
    assert model.block.ffn.round_to == 32
    assert model.recurrence.slow_cycles == model.recurrence.fast_cycles == 1
    assert model.prefix.batch_size == 2
    assert isinstance(config.step.pool, AtomicPool.Config)
    assert config.step.pool.batch_size == 2
    assert config.step.pool.max_steps == 4
    assert config.dataset.batch_size == config.dataset.eval_batch_size == 2
    assert config.dataset.num_tasks == config.dataset.num_eval_tasks == 4
    assert config.checkpointer is None


def test_full_recipe_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_train_loop_evaluation_checkpoint_and_resume(tmp_path: Path) -> None:
    """Run the actual loop through a midpass checkpoint and its next update."""
    root = tmp_path / "datasets/arc2concept-aug-1000"
    root.mkdir(parents=True)
    (root / "identifiers.json").write_text('["<blank>", "puzzle"]')
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    (root / "test_puzzles.json").write_text(
        json.dumps({"puzzle": {"test": [{"input": grid, "output": grid}]}}),
    )
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            '{"ignore_label_id": 0, "blank_identifier_id": 0}',
        )
        for name, array in {
            "inputs": np.full((3, 9), 1, dtype=np.int32),
            "labels": np.full((3, 9), 1, dtype=np.int32),
            "puzzle_identifiers": np.array([1], dtype=np.int32),
            "puzzle_indices": np.array([0, 3], dtype=np.int64),
            "group_indices": np.array([0, 1], dtype=np.int64),
        }.items():
            np.save(directory / f"all__{name}.npy", array)
    config = exp_smoke()
    config.base_dir = tmp_path
    config.step = training_config(4, torch.bfloat16)
    assert isinstance(config.step.model.embedding, GridEmbedding.Config)
    config.dataset.spec.max_grid = 3
    config.runtime = SingleProcess.Config(device="cpu")
    config.dataset.device = "cpu"
    config.max_steps = 1
    config.num_steps_eval = 1
    config.checkpointer = Checkpointer.Config(save_every=1)
    loop = config.make()
    loop.run()
    assert isinstance(loop.step, ArcTrainStep)
    assert isinstance(loop.dataset, Arc2Data)
    assert loop.step.global_step == 1
    assert loop.dataset.state_dict()["next_batch"] == 1
    metric = loop.metrics_eval[""]
    assert isinstance(metric, PassK)
    assert (
        sum(
            len(records)
            for ballots in metric.votes.values()
            for records in ballots.values()
        )
        == 3
    )
    expected = {
        name: value.clone() for name, value in loop.step.model.state_dict().items()
    }
    checkpoint = tmp_path / "runs/arcagi2/exp_smoke/checkpoints/step_00000001.pt"
    assert checkpoint.is_file()
    config.max_steps = 2
    resumed = config.make()
    assert isinstance(resumed.step, ArcTrainStep)
    assert isinstance(resumed.dataset, Arc2Data)
    assert resumed.step.global_step == 1
    assert resumed.dataset.state_dict()["next_batch"] == 1
    for name, value in resumed.step.model.state_dict().items():
        assert torch.equal(value, expected[name])
    resumed.run()
    assert resumed.step.global_step == 2
    assert resumed.dataset.state_dict()["next_batch"] == 2
    assert (checkpoint.parent / "step_00000002.pt").is_file()
    assert any(
        not torch.equal(value, expected[name])
        for name, value in resumed.step.model.state_dict().items()
    )


def test_finalize_without_rope() -> None:
    config = exp000()
    config.step.model.rope = None
    finalized = config.finalize()
    assert finalized.step.model.rope is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
