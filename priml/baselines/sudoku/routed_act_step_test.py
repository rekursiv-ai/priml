"""Drive the train loop and its routed step at a size-only shrink of each recipe.

Every rung builds, trains, and evaluates on the 9x9 grid; the units below pin
the step's mechanisms one at a time. The pipeline goldens (``exp008``,
``exp012``) replay this step bit for bit inside their training segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast, override

import json

from configgle import PartialConfig
from torch import Tensor
from treekle import from_plain, json_loads

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import experiments
from priml.baselines.sudoku.act import CellCorruption, FeedbackCarry
from priml.baselines.sudoku.hypotheses import HypothesisRecursion
from priml.baselines.sudoku.latent_recursion import recipe_block
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.baselines.sudoku.routed_act_step import HaltRelease, RoutedActTrainStep
from priml.model.swiglu import SwiGLU
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA, NoEMA


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.sudoku.train_loop import SudokuTrainLoop


RECIPES: Final = (
    "exp000",
    "exp001",
    "exp002",
    "exp003",
    "exp004",
    "exp005",
    "exp006",
    "exp009",
    "exp010",
)
"""Every training rung."""

TRAIN_STEPS: Final = 3


def write_dataset(root: Path, *, grid_len: int = 16, vocab_size: int = 6) -> Path:
    """Write a square-grid fixture: 4 train rows (2 groups x 2), 2 test rows.

    Args:
      root: Dataset root; ``train/`` and ``test/`` splits are written below.
      grid_len: Number of cells per row.
      vocab_size: Number of token IDs, including pad and blank.

    Returns:
      root: The dataset root.

    """
    rng = np.random.default_rng(7)
    for split, rows, groups in (
        ("train", 4, [0, 2, 4]),
        ("test", 2, [0, 1, 2]),
    ):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(
            directory / "all__inputs.npy",
            rng.integers(1, vocab_size, (rows, grid_len)),
        )
        np.save(
            directory / "all__labels.npy",
            rng.integers(2, vocab_size, (rows, grid_len)),
        )
        np.save(directory / "all__group_indices.npy", np.asarray(groups, np.int32))
        (directory / "dataset.json").write_text(
            json.dumps({"vocab_size": vocab_size, "seq_len": grid_len}),
        )
    return root


def port_config(recipe: str, scratch: Path) -> SudokuTrainLoop:
    """Return ``recipe`` shrunk by size only, on a 4x4 grid of six tokens.

    Latent-hypothesis recipes run at width 12, FFN hidden 4 and batch 1, the
    sizes their source parity ran at: hypothesis alignment projects on the
    FFN's top-10 subspace, which is the whole space at width 10 or less and
    divides by a zero perpendicular norm there.
    """
    factory = cast("Callable[[], SudokuTrainLoop]", getattr(experiments, recipe))
    config = factory()
    config.experiment_name = "golden"
    config.base_dir = scratch
    config.checkpointer = None
    runtime = config.runtime = SingleProcess.Config()
    runtime.device = "cpu"
    config.max_steps = TRAIN_STEPS
    config.num_steps_eval = float("inf")
    config.eval_warmup_batches = 0
    step = config.step
    step.total_train_steps = TRAIN_STEPS
    # Two ACT steps, two slow cycles, one fast: every branch the recipe's sizes
    # reach (coverage.py lines and arcs, measured), and the fewest ops.
    step.pool.max_steps = 2
    ema = step.ema
    assert isinstance(ema, EMA.Config)
    ema.update_after_step = min(ema.update_after_step, 2)
    model = step.model
    model.channels_in = 4
    model.num_heads = 2
    model.num_layers = 1
    model.puzzle_emb_len = 2
    model.slow_cycles = 2
    model.fast_cycles = 1
    model.compile = False
    if model.block is None:
        model.block = recipe_block()
    assert isinstance(model.block.ffn, SwiGLU.Config)
    model.block.ffn.expansion = 1
    model.block.ffn.round_to = 1
    config.dataset.working_dir = scratch / "data"
    config.dataset.device = "cpu"
    config.dataset.batch_size = 2
    config.dataset.eval_batch_size = 2
    config.dataset.eval_num_instances = 2
    config.dataset.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    if isinstance(model, HypothesisRecursion.Config):
        model.channels_in = 12
        model.block.ffn.channels_hidden = 4
        config.dataset.batch_size = 1
        config.dataset.eval_batch_size = 1
        config.dataset.eval_num_instances = 1
    return config


@pytest.mark.parametrize("recipe", RECIPES)
def test_full_geometry_still_runs(recipe: str, tmp_path: Path) -> None:
    """Every rung builds, trains a step, and evaluates on the 9x9 grid."""
    config = port_config(recipe, tmp_path)
    config.dataset.spec = SudokuSpec()
    write_dataset(tmp_path / "data", grid_len=81, vocab_size=11)
    with torch.random.fork_rng(devices=[]):
        loop = config.make()
        try:
            batch = loop._get_next_batch()
            media = batch["media"]
            assert isinstance(media, Tensor)
            assert media.shape[-1] == 81
            loop.step.train_step(**batch)
            evaluation = loop.step.eval_loss(**batch)
        finally:
            loop.close()
    assert evaluation["model"].shape == (media.shape[0], 1 + 81)


def test_a_short_run_checkpoints_evaluates_and_resumes(tmp_path: Path) -> None:
    """Two steps with cadence evals, releases, and saves; a resume adds one."""
    write_dataset(tmp_path / "data")
    for steps in (2, 3):
        config = port_config("exp000", tmp_path)
        config.checkpointer = Checkpointer.Config(save_every=1)
        config.num_steps_eval = 1
        release = config.step.release = HaltRelease.Config()
        release.threshold = -1e9
        config.max_steps = steps
        loop = config.make()
        loop.train()
        assert loop.step.global_step == steps
    saved = sorted(path.name for path in (tmp_path / "runs").rglob("step_*.pt"))
    assert saved[-1] == "step_00000003.pt"
    metrics = from_plain(
        json_loads(next((tmp_path / "runs").rglob("metrics.json")).read_text()),
        dict[str, float],
    )
    assert metrics["eval/act_cap_frac"] == 0.0
    assert metrics["eval/act_steps"] == 1.0
    assert 0.0 <= metrics["eval/grid_exact"] <= 1.0


def test_dedicated_generators_survive_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """Resume continues the halt and corruption streams instead of restarting them."""
    config = port_config("exp006", tmp_path)
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    step.train_step(**loop._get_next_batch())
    state = step.state_dict()
    halting, carry = step.pool.halting, step.pool.carry
    assert halting is not None
    assert carry is not None
    assert "halt_rng" in state
    assert "corruption_rng" in state
    assert torch.equal(state["halt_rng"], halting.generator.get_state())
    assert torch.equal(state["corruption_rng"], carry.generator.get_state())
    assert "sparse_optimizer" in state
    expected = (
        torch.rand(4, generator=halting.generator),
        torch.rand(4, generator=carry.generator),
    )
    loop.close()
    restored = cast(RoutedActTrainStep, config.make().step)
    restored.load_state_dict(state)
    restored_halting, restored_carry = restored.pool.halting, restored.pool.carry
    assert restored_halting is not None
    assert restored_carry is not None
    assert torch.equal(torch.rand(4, generator=restored_halting.generator), expected[0])
    assert torch.equal(torch.rand(4, generator=restored_carry.generator), expected[1])
    assert restored.global_step == 1


def test_without_halt_training_the_head_is_frozen_and_unscored(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    config.step.pool.halting = None
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    assert all(not p.requires_grad for p in step.net.q_head.parameters())
    metrics = step.train_step(**loop._get_next_batch()).get("metrics", {})
    assert "q_halt_loss" not in metrics
    assert "halted_frac" not in metrics
    loop.close()


def test_without_an_ema_there_is_no_shadow(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    config.step.ema = NoEMA.Config()
    write_dataset(tmp_path / "data")
    loop = config.make()
    assert cast(RoutedActTrainStep, loop.step).ema_shadow is None
    loop.close()


def test_an_unclipped_step_still_reports_the_gradient_norm(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    config.step.gradient_clip_norm = float("inf")
    write_dataset(tmp_path / "data")
    loop = config.make()
    metrics = loop.step.train_step(**loop._get_next_batch()).get("metrics", {})
    assert {"grad_norm", "param_norm", "lr", "muon_lr"} <= metrics.keys()
    loop.close()


def test_evaluation_runs_eager_without_precision_cast_emulation(
    tmp_path: Path,
) -> None:
    config = port_config("exp000", tmp_path)
    config.step.emulate_precision_casts = False
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    # The guard reads only the flag, so it is raised after an eager build: a
    # compiled build would charge this test torch.compile's first import.
    step.net.config.compile = True
    with step._eval_compile_disabled():
        assert step.net.config.compile is False
    assert step.net.config.compile is True
    loop.close()


def test_an_all_padding_batch_scores_zero(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    write_dataset(tmp_path / "data")
    loop = config.make()
    empty = loop.step.eval_loss(
        media=torch.zeros(2, 16, dtype=torch.int32),
        label=torch.zeros(2, 16, dtype=torch.int32),
        valid_count=0,
    )
    assert empty["model"].shape == (2, 1 + 16)
    assert not empty["loss"].any()
    loop.close()


def test_release_keeps_each_rows_first_confident_step(tmp_path: Path) -> None:
    """A released row keeps the outputs of the step it cleared the threshold."""
    config = port_config("exp004", tmp_path)
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    batch = loop._get_next_batch()
    capped = step.eval_loss(**batch).get("metrics", {})
    step.release = HaltRelease(HaltRelease.Config(threshold=-1e9))
    released = step.eval_loss(**batch)
    step.release = None
    step.pool.config.max_steps = 1
    one_step = step.eval_loss(**batch)
    loop.close()
    assert torch.equal(released["model"], one_step["model"])
    released_metrics = released.get("metrics", {})
    assert released_metrics["act_steps"] == 1.0
    assert released_metrics["act_cap_frac"] == 0.0
    assert "act_cap_frac" not in capped
    assert capped["act_steps"] == float(config.step.pool.max_steps)
    assert torch.equal(
        HaltRelease(HaltRelease.Config(threshold=0.5))(torch.tensor([0.4, 0.5, 0.6])),
        torch.tensor([False, True, True]),
    )


class _ReleaseFirstRow(HaltRelease):
    """Release only the first surviving row at every step."""

    @override
    def __call__(self, q_halt: Tensor) -> Tensor:
        return torch.arange(q_halt.shape[0]) == 0


def test_release_drops_rows_one_at_a_time(tmp_path: Path) -> None:
    """Row 0 leaves at step 1 and row 1 at step 2, each with that step's grid."""
    config = port_config("exp004", tmp_path)
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    batch = loop._get_next_batch()
    step.release = _ReleaseFirstRow(HaltRelease.Config())
    staggered = step.eval_loss(**batch)
    step.release = None
    fixed: list[Tensor] = []
    for depth in (1, 2):
        step.pool.config.max_steps = depth
        fixed.append(step.eval_loss(**batch)["model"])
    loop.close()
    assert torch.equal(staggered["model"][0], fixed[0][0])
    assert torch.equal(staggered["model"][1], fixed[1][1])
    assert staggered.get("metrics", {})["act_steps"] == 1.5


def test_a_zero_depth_rollout_is_refused(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    step.pool.config.max_steps = 0
    with pytest.raises(ValueError, match="final_logits"):
        step.eval_loss(**loop._get_next_batch())
    loop.close()


def test_cardinality_waits_out_its_warmup(tmp_path: Path) -> None:
    config = port_config("exp006", tmp_path)
    cardinality = config.step.cardinality
    assert cardinality is not None
    cardinality.warmup_steps = 2
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    metrics = [
        step.train_step(**loop._get_next_batch()).get("metrics", {}) for _ in range(3)
    ]
    loop.close()
    assert [m["csp_loss"] for m in metrics[:2]] == [0.0, 0.0]
    assert isinstance(metrics[2]["csp_loss"], Tensor)


def test_without_a_puzzle_table_there_is_no_sparse_step(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    config.step.model.num_puzzle_identifiers = 0
    write_dataset(tmp_path / "data")
    loop = config.make()
    step = cast(RoutedActTrainStep, loop.step)
    assert step.net.puzzle_emb is None
    assert step.sparse_optimizer is None
    before = [p.detach().clone() for p in step._body_params]
    step.train_step(**loop._get_next_batch())
    loop.close()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, step._body_params, strict=True)
    )


def test_a_whole_model_compile_is_refused(tmp_path: Path) -> None:
    config = port_config("exp000", tmp_path)
    config.step.compile = PartialConfig(torch.compile)
    with pytest.raises(ValueError, match="never calls the compiled model"):
        config.make()


def test_a_configured_corruption_drives_the_carry(tmp_path: Path) -> None:
    config = port_config("exp006", tmp_path)
    feedback = config.step.pool.feedback
    assert isinstance(feedback, FeedbackCarry.Config)
    corruption = feedback.corruption = CellCorruption.Config()
    corruption.rate = 0.25
    write_dataset(tmp_path / "data")
    loop = config.make()
    carry = cast(RoutedActTrainStep, loop.step).pool.carry
    assert carry is not None
    assert isinstance(carry.corruption, CellCorruption)
    assert carry.corruption.config.rate == 0.25
    loop.close()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
