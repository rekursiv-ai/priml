"""Verify the full recipe, finalized resource paths, and printable defaults."""

from __future__ import annotations

from pathlib import Path

import json

from configgle.testing import assert_pprint_golden

import numpy as np
import torch

from priml.baselines.arcagi2.bfb_test import miniature_config
from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.experiments import exp000, exp_smoke
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.train_step import ArcDataParallel, ArcTrainStep
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel


def test_finalized_reference_recipe() -> None:
    config = exp000().finalize()
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.mesh_topology == {"dp": -1, "pp": 1, "tp": 1}
    assert isinstance(config.step.parallelism, ArcDataParallel.Config)
    assert config.step.parallelism.gradient_as_bucket_view
    assert config.step.act is not None
    assert config.dataset.batch_size == config.step.act.batch_size == 256
    assert config.dataset.make().prepared.eval_batch_size == 256
    assert config.dataset.epochs_per_iter == 4
    assert config.max_steps == config.step.total_train_steps == 541_580
    assert config.dataset.working_dir == Path(
        "/opt/scratch/datasets/arcagi2/arc2concept-aug-1000",
    )
    metric = config.metrics_eval[""]
    assert isinstance(metric, PassK.Config)
    assert metric.working_dir == config.dataset.working_dir


def test_resource_paths_follow_base_dir(tmp_path: Path) -> None:
    config = exp000()
    config.base_dir = tmp_path
    config = config.finalize()
    metric = config.metrics_eval[""]
    assert isinstance(metric, PassK.Config)
    assert (
        config.dataset.working_dir == tmp_path / "datasets/arcagi2/arc2concept-aug-1000"
    )
    assert metric.working_dir == config.dataset.working_dir


def test_smoke_is_single_process() -> None:
    config = exp_smoke().finalize()
    assert isinstance(config.runtime, SingleProcess.Config)
    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert config.max_steps == 4
    assert config.step.act is not None
    assert config.dataset.batch_size == config.step.act.batch_size == 2


def test_full_recipe_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_train_loop_evaluation_checkpoint_and_resume(tmp_path: Path) -> None:
    """Run the actual loop through a midpass checkpoint and its next update."""
    root = tmp_path / "datasets/arcagi2/arc2concept-aug-1000"
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
            "inputs": np.full((3, 9), 2, dtype=np.int32),
            "labels": np.full((3, 9), 2, dtype=np.int32),
            "puzzle_identifiers": np.array([1], dtype=np.int32),
            "puzzle_indices": np.array([0, 3], dtype=np.int64),
            "group_indices": np.array([0, 1], dtype=np.int64),
        }.items():
            np.save(directory / f"all__{name}.npy", array)
    config = exp_smoke()
    config.base_dir = tmp_path
    config.step = miniature_config()
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
