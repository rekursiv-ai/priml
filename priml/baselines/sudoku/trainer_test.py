"""Replay frozen training trajectories through the TRM trainer.

The goldens in ``testdata/exp0NN.pt`` were recorded from the reference
implementation this trainer was ported from; this module imports none of it,
so the proof outlives that code. Each recipe runs through the same size-only shrink
fixture while preserving every numerical choice the experiment makes.

Recorded per recipe, all compared with ``torch.equal``: every parameter and
persistent buffer after init and a fingerprint of the global RNG; an
evaluation rollout before and after training; three train steps (the batch the
dataset served, loss, probe, every metric, the ACT pool); the final state,
latents, and EMA shadow. Width 4 and batch 2: the goldens exercise every code
path, not kernel throughput, so they stay a few KB each.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import experiments, trm
from priml.baselines.sudoku.act import CellCorruption
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.baselines.sudoku.trainer import Trainer, sudoku_group_indices
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    mismatches,
    put_steps,
    read_tensors,
    rng_fingerprint,
    stored,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from configgle import Makeable
    from torch import nn

    from priml.model.transformer.block import TransformerBlock
    from priml.train.checkpointer import Checkpointer


_CWD: Final = Path(__file__).resolve().parent

RECIPES: Final = ("exp004", "exp005", "exp006", "exp007", "exp008", "exp009", "exp010")
"""Every rung of the training ladder."""

TRAIN_STEPS: Final = 3


class Loader(Protocol):
    """A dataset's batch source."""

    def eval_dataloader(self) -> Iterable[dict[str, object]]:
        """Return the eval batches."""
        ...


class Subject(Protocol):
    """The trainer surface the recorder reads; both ports expose it."""

    model: nn.Module
    dataset: Loader

    @property
    def ema_shadow(self) -> dict[str, Tensor] | None:
        """Return the name-keyed EMA shadow."""
        ...

    def _next_batch(self) -> dict[str, object]: ...

    def train_step(self, **batch: object) -> Mapping[str, object]:
        """Run one training call."""
        ...

    def eval_loss(self, **batch: object) -> Mapping[str, object]:
        """Run one evaluation call."""
        ...

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Move a batch to the trainer's device."""
        ...


class ShrinkableModel(Protocol):
    """The model fields a size-only shrink sets; both ports declare them."""

    channels_in: int
    num_heads: int
    num_layers: int
    puzzle_emb_len: int
    slow_cycles: int
    fast_cycles: int
    compile: bool
    dtype: torch.dtype | None
    block: TransformerBlock.Config | None


class ShrinkableData(Protocol):
    """The dataset fields a size-only shrink sets."""

    working_dir: Path | str
    device: str
    batch_size: int
    eval_batch_size: int | None
    eval_num_instances: int | None


class ShrinkableRuntime(Protocol):
    """The runtime field a size-only shrink sets."""

    device: torch.device | str | None


class Shrinkable(Protocol):
    """A trainer config of either port, seen through the fields shrunk.

    Child nodes are read-only properties: a property is covariant, so each
    port's concrete child config satisfies it where a mutable attribute,
    being invariant, would not.
    """

    experiment_name: str
    base_dir: Path | str | None
    max_steps: float
    num_steps_eval: float
    eval_warmup_batches: int
    total_train_steps: int
    max_act_steps: int
    ema_warmup_steps: int
    dtype_autocast: torch.dtype | None
    checkpointer: Checkpointer.Config | None

    @property
    def model(self) -> ShrinkableModel:
        """Return the model config."""
        ...

    @property
    def dataset(self) -> ShrinkableData:
        """Return the dataset config."""
        ...

    @property
    def runtime(self) -> ShrinkableRuntime:
        """Return the runtime config."""
        ...


class _FalsyCellCorruption(CellCorruption.Config):
    """A corruption config whose truth value is False."""

    def __bool__(self) -> bool:
        return False


def test_constraint_groups_follow_puzzle_spec() -> None:
    spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    groups = sudoku_group_indices(spec)
    assert groups.shape == (12, 4)
    assert torch.equal(groups[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(groups[4], torch.tensor([0, 4, 8, 12]))
    assert torch.equal(groups[8], torch.tensor([0, 1, 4, 5]))


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


def port_config(recipe: str, scratch: Path) -> Trainer.Config:
    """Return the priml ``recipe`` shrunk by size only."""
    factory = cast("Callable[[], Trainer.Config]", getattr(experiments, recipe))
    config = shrink(
        factory(),
        scratch=scratch,
        recipe_block=trm.recipe_block,
    )
    config.dataset.spec = SudokuSpec(
        grid_shape=(4, 4),
        box_shape=(2, 2),
        vocab_size=6,
    )
    return config


def shrink[ConfigT: Shrinkable](
    config: ConfigT,
    *,
    scratch: Path,
    recipe_block: Callable[[], TransformerBlock.Config],
) -> ConfigT:
    """Shrink a trainer config by size only, in place; precision as it runs.

    Args:
      config: A full-size recipe of either port; both share these field names.
      scratch: Holds the dataset (``data/``) and the run directory.
      recipe_block: The port's default block, materialized so its feed-forward
        width can shrink before anything finalizes it.

    Returns:
      config: The same config, CPU-sized.

    """
    config.experiment_name = "golden"
    config.base_dir = scratch
    config.checkpointer = None
    config.runtime.device = "cpu"
    config.max_steps = TRAIN_STEPS
    config.num_steps_eval = float("inf")
    config.eval_warmup_batches = 0
    config.total_train_steps = TRAIN_STEPS
    config.max_act_steps = 3
    config.ema_warmup_steps = min(config.ema_warmup_steps, 2)
    config.model.channels_in = 4
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.puzzle_emb_len = 2
    config.model.slow_cycles = 2
    config.model.fast_cycles = 2
    config.model.compile = False
    if config.model.block is None:
        config.model.block = recipe_block()
    assert isinstance(config.model.block.ffn, SwiGLU.Config)
    config.model.block.ffn.expansion = 1
    config.model.block.ffn.round_to = 1
    config.dataset.working_dir = scratch / "data"
    config.dataset.device = "cpu"
    config.dataset.batch_size = 2
    config.dataset.eval_batch_size = 2
    config.dataset.eval_num_instances = 2
    return config


def record(subject: Subject) -> dict[str, Tensor]:
    """Run the recorded protocol; call inside ``host_agnostic_numerics``.

    Args:
      subject: A freshly constructed trainer.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    out: dict[str, Tensor] = {"rng": rng_fingerprint()}
    _put(out, "init", _state(subject))
    evaluation = subject.preprocess_batch(
        next(iter(subject.dataset.eval_dataloader())),
    )
    _put(out, "eval_before", flatten(subject.eval_loss(**evaluation)))
    data: list[dict[str, Tensor]] = []
    train: list[dict[str, Tensor]] = []
    steps: list[dict[str, Tensor]] = []
    for _ in range(TRAIN_STEPS):
        batch = subject._next_batch()
        data.append(flatten(batch))
        train.append(flatten(subject.train_step(**batch)))
        steps.append({"steps": pool_state(subject)["steps"].clone()})
    put_steps(out, "data", data)
    put_steps(out, "train", train)
    put_steps(out, "pool", steps)
    _put(out, "pool/final", pool_state(subject))
    _put(out, "post", _state(subject))
    _put(out, "ema", dict(subject.ema_shadow or {}))
    _put(out, "eval_after", flatten(subject.eval_loss(**evaluation)))
    return out


def pool_state(subject: object) -> dict[str, Tensor]:
    """Return the recorded slot state of either port under the golden's names.

    Args:
      subject: This port's trainer, or the reference one it replaces.

    Returns:
      state: Inputs, labels, depth, halt mask, task ids, feedback, latents.

    """
    if isinstance(subject, Trainer):
        pool = subject.pool
        return {
            "inputs": pool.inputs,
            "labels": pool.labels,
            "steps": pool.steps,
            "halted": pool.halted,
            "puzzle_ids": pool.puzzle_ids,
            "feedback": pool.feedback,
            "z_slow": pool.z_slow,
            "z_fast": pool.z_fast,
        }
    return {
        name: cast(Tensor, getattr(subject, f"_pool_{source}"))
        for name, source in (
            ("inputs", "inputs"),
            ("labels", "labels"),
            ("steps", "h_step"),
            ("halted", "halted"),
            ("puzzle_ids", "puzzle_ids"),
            ("feedback", "feedback"),
            ("z_slow", "z_slow"),
            ("z_fast", "z_fast"),
        )
    }


def run(config: Makeable[object], scratch: Path) -> dict[str, Tensor]:
    """Write the dataset, build the trainer, and record it.

    Args:
      config: A trainer config prepared for the recorded fixture.
      scratch: The directory the config points at.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        write_dataset(scratch / "data")
        return record(cast(Subject, config.make()))


def golden_path(recipe: str) -> Path:
    """Where the golden for ``recipe`` lives."""
    return _CWD / "testdata" / f"{recipe}.pt"


def load_golden(recipe: str) -> dict[str, Tensor]:
    """Load a frozen golden."""
    return read_tensors(golden_path(recipe))


@pytest.mark.parametrize("recipe", RECIPES)
def test_golden_replays_bit_for_bit(recipe: str, tmp_path: Path) -> None:
    """The trainer reproduces the frozen trajectory with zero mismatches."""
    actual = run(port_config(recipe, tmp_path), tmp_path)
    assert_tensor_golden(golden_path(recipe), actual)


@pytest.mark.parametrize("recipe", RECIPES)
def test_full_geometry_still_runs(recipe: str, tmp_path: Path) -> None:
    config = port_config(recipe, tmp_path)
    config.dataset.spec = SudokuSpec()
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        write_dataset(tmp_path / "data", grid_len=81, vocab_size=11)
        result = record(cast(Subject, config.make()))
    assert result["data/media"].shape[-1] == 81


@pytest.mark.parametrize("recipe", RECIPES)
def test_one_ulp_weight_bites(recipe: str, tmp_path: Path) -> None:
    config = port_config(recipe, tmp_path)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        write_dataset(tmp_path / "data")
        subject = config.make()
        with torch.no_grad():
            weight = next(subject.model.parameters())
            bits = torch.int32 if weight.dtype == torch.float32 else torch.int16
            weight.view(bits).view(-1)[0] += 1
        actual = record(cast(Subject, subject))
    report = mismatches(load_golden(recipe), actual)
    assert any(line.startswith("init/") for line in report), report


def test_golden_bites(tmp_path: Path) -> None:
    """A changed loss weight is reported, not absorbed."""
    config = port_config("exp010", tmp_path)
    config.csp_loss_weight = 0.25
    report = mismatches(load_golden("exp010"), run(config, tmp_path))
    assert any(line.startswith("train/loss") for line in report), report


def flatten(values: Mapping[str, object]) -> dict[str, Tensor]:
    """Tensors as-is, numbers as int64/float64, nested mappings flattened.

    Args:
      values: A batch or a train/eval step output.

    Returns:
      flat: Slash-joined keys to tensors.

    """
    out: dict[str, Tensor] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            nested = cast("dict[str, object]", value)
            out |= {f"{key}/{k}": v for k, v in flatten(nested).items()}
        elif isinstance(value, Tensor):
            out[key] = value
        elif isinstance(value, int):
            out[key] = torch.tensor(value, dtype=torch.int64)
        else:
            out[key] = torch.tensor(cast(float, value), dtype=torch.float64)
    return out


def _state(subject: Subject) -> dict[str, Tensor]:
    """Parameters and persistent buffers."""
    return {k: v for k, v in subject.model.state_dict().items() if k != "_dummy"}


def _put(out: dict[str, Tensor], prefix: str, values: Mapping[str, Tensor]) -> None:
    """Store :func:`stored` copies of ``values`` under ``prefix``."""
    for key, value in values.items():
        out[f"{prefix}/{key}"] = stored(value)


def test_do_train_step_advances_training(tmp_path: Path) -> None:
    config = port_config("exp004", tmp_path)
    config.num_steps_log = 1
    write_dataset(tmp_path / "data")
    subject = config.make()

    subject._do_train_step(subject._next_batch())

    assert subject.global_step == 1
    assert subject.local_step == 1


def test_trainer_execution_variants(tmp_path: Path) -> None:
    config = port_config("exp004", tmp_path)
    config.dataset.working_dir = tmp_path / "data"
    config.grad_clip_max_norm = None
    config.log_body_norms = True
    config.dataset.eval_batch_size = 1
    write_dataset(tmp_path / "data")
    subject = config.make()
    assert isinstance(subject.evaluate(), dict)
    subject.config.emulate_precision_casts = False
    with subject._eval_compile_disabled():
        pass
    result = subject.train_step(**next(iter(subject.dataset.train_dataloader())))
    assert "metrics" in result
    assert "lr" in result["metrics"]
    assert isinstance(subject.evaluate(), dict)
    subject.config.eval_warmup_batches = 1
    subject._warm_eval_compile()
    empty = subject.eval_loss(
        media=torch.zeros(2, 16, dtype=torch.int32),
        label=torch.zeros(2, 16, dtype=torch.int32),
        valid_count=0,
    )
    assert empty["model"].shape[0] == 2
    subject.config.max_steps = 2
    subject.config.max_time = 0.0
    subject.run()


def test_trainer_optional_constructor_and_stop_guards(tmp_path: Path) -> None:
    config = port_config("exp004", tmp_path)
    config.dataset.working_dir = tmp_path / "data"
    config.train_q_halt = False
    config.doc = "notes"
    write_dataset(tmp_path / "data")
    subject = config.make()
    assert all(not p.requires_grad for p in subject.model.q_head.parameters())
    config = port_config("exp004", tmp_path)
    config.dataset.working_dir = tmp_path / "data"
    config.use_ema = False
    assert config.make().ema_shadow is None
    config = port_config("exp004", tmp_path)
    config.dataset.working_dir = tmp_path / "data"
    config.max_steps = float("inf")
    config.max_time = float("inf")
    with pytest.raises(ValueError, match="no finite stop"):
        config.make().run()


def test_trainer_run_and_resume_guard_branches(tmp_path: Path) -> None:
    config = port_config("exp004", tmp_path)
    config.dataset.working_dir = tmp_path / "data"
    write_dataset(tmp_path / "data")
    subject = config.make()
    subject._guard_resume_config(False)
    subject._guard_resume_config(True)
    subject._guard_resume_config(True)
    subject.config.doc = "changed"
    subject._guard_resume_config(True)
    subject.config.max_steps = 0
    subject.run("launcher-arg")
    assert subject.global_step == 0


def test_dedicated_generators_survive_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """Resume continues the halt and scramble streams instead of restarting them."""
    config = port_config("exp010", tmp_path)
    write_dataset(tmp_path / "data")
    subject = config.make()
    subject.train_step(**subject._next_batch())
    state = subject.state_dict()
    halting, carry = subject.pool.halting, subject.pool.carry
    assert halting is not None
    assert carry is not None
    step_state = state["step"]
    assert "halt_rng" in step_state
    assert "scramble_rng" in step_state
    assert torch.equal(step_state["halt_rng"], halting.generator.get_state())
    assert torch.equal(step_state["scramble_rng"], carry.generator.get_state())
    expected = (
        torch.rand(4, generator=halting.generator),
        torch.rand(4, generator=carry.generator),
    )
    restored = config.make()
    restored.load_state_dict(state)
    restored_halting, restored_carry = restored.pool.halting, restored.pool.carry
    assert restored_halting is not None
    assert restored_carry is not None
    assert torch.equal(torch.rand(4, generator=restored_halting.generator), expected[0])
    assert torch.equal(torch.rand(4, generator=restored_carry.generator), expected[1])


def test_feedback_corruption_overrides_the_slot_scramble(tmp_path: Path) -> None:
    config = port_config("exp010", tmp_path)
    write_dataset(tmp_path / "data")
    corruption = config.feedback_corruption = CellCorruption.Config()
    corruption.rate = 0.25
    carry = config.make().pool.carry
    assert carry is not None
    assert isinstance(carry.corruption, CellCorruption)
    assert carry.corruption.config.rate == 0.25


def test_a_falsy_feedback_corruption_still_overrides_the_slot_scramble(
    tmp_path: Path,
) -> None:
    """``or`` would discard a configured corruption whose truth value is False."""
    config = port_config("exp010", tmp_path)
    write_dataset(tmp_path / "data")
    corruption = _FalsyCellCorruption()
    config.feedback_corruption = corruption
    assert not corruption
    carry = config.make().pool.carry
    assert carry is not None
    assert isinstance(carry.corruption, CellCorruption)


def test_trainer_constructor_rejects_invalid_protocol_configs(tmp_path: Path) -> None:
    config = port_config("exp004", tmp_path)
    config.experiment_name = ""
    with pytest.raises(ValueError, match="experiment_name"):
        config.make()
    config = port_config("exp004", tmp_path)
    config.eval_act_steps = -1
    with pytest.raises(ValueError, match="eval_act_steps"):
        config.make()
    config = port_config("exp004", tmp_path)
    config.eval_min_act_steps = 0
    with pytest.raises(ValueError, match="eval_min_act_steps"):
        config.make()
    config = port_config("exp004", tmp_path)
    config.csp_loss_weight = 1.0
    config.dataset.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(3, 3), vocab_size=6)
    with pytest.raises(ValueError, match="csp_loss_weight"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
