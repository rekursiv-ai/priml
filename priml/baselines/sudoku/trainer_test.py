"""Replay frozen training trajectories through the TRM trainer.

The goldens in ``testdata/exp0NN.pt`` were recorded from the reference
implementation this trainer was ported from; this module imports none of it,
so the proof outlives that code. Each recipe is shrunk by SIZE only --
width, heads, cycles, batch, ACT cap, horizon, the EMA warmup -- so every
numerical choice the experiment makes is exercised.

Recorded per recipe, all compared with ``torch.equal``: every parameter and
persistent buffer after init and a fingerprint of the global RNG; an
evaluation rollout before and after training; five train steps (the batch the
dataset served, loss, probe, every metric, the ACT pool); the final state,
latents, and EMA shadow. Width 4 and batch 2: the goldens exercise every code
path, not kernel throughput, so they stay a few KB each.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import experiments, trm
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
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

    from priml.baselines.sudoku.trainer import Trainer
    from priml.model.transformer.block import TransformerBlock
    from priml.train.checkpointer import Checkpointer


_CWD: Final = Path(__file__).resolve().parent

RECIPES: Final = ("exp004", "exp005", "exp006", "exp007", "exp008", "exp009", "exp010")
"""Every rung of the training ladder."""

PRECISIONS: Final = ("fp32", "bf16_autocast")
"""``fp32`` runs without autocast; ``bf16_autocast`` is the recipe's own."""

CASES: Final = tuple(
    (recipe, precision)
    for precision in PRECISIONS
    for recipe in RECIPES
    if precision == "fp32" or recipe in {"exp004", "exp010"}
)
"""Every rung in fp32; the ladder's ends also under the recipe's bf16."""

TRAIN_STEPS: Final = 5


class Loader(Protocol):
    """A dataset's batch source."""

    def eval_dataloader(self) -> Iterable[dict[str, object]]:
        """Return the eval batches."""
        ...


class Subject(Protocol):
    """The trainer surface the recorder reads; both ports expose it."""

    model: nn.Module
    dataset: Loader
    _pool_inputs: Tensor
    _pool_labels: Tensor
    _pool_z_slow: Tensor
    _pool_z_fast: Tensor
    _pool_h_step: Tensor
    _pool_halted: Tensor
    _pool_puzzle_ids: Tensor
    _pool_feedback: Tensor

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


def write_dataset(root: Path) -> Path:
    """Write the 81-token fixture: 12 train rows (3 groups x 4), 6 test rows.

    Args:
      root: Dataset root; ``train/`` and ``test/`` splits are written below.

    Returns:
      root: The dataset root.

    """
    rng = np.random.default_rng(7)
    for split, rows, groups in (
        ("train", 12, [0, 4, 8, 12]),
        ("test", 6, list(range(7))),
    ):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "all__inputs.npy", rng.integers(1, 11, (rows, 81)))
        np.save(directory / "all__labels.npy", rng.integers(2, 11, (rows, 81)))
        np.save(directory / "all__group_indices.npy", np.asarray(groups, np.int32))
        (directory / "dataset.json").write_text('{"vocab_size": 11, "seq_len": 81}')
    return root


def port_config(recipe: str, precision: str, scratch: Path) -> Trainer.Config:
    """Return the priml ``recipe`` shrunk by size only."""
    factory = cast("Callable[[], Trainer.Config]", getattr(experiments, recipe))
    return shrink(
        factory(),
        precision=precision,
        scratch=scratch,
        recipe_block=trm.recipe_block,
    )


def shrink[ConfigT: Shrinkable](
    config: ConfigT,
    *,
    precision: str,
    scratch: Path,
    recipe_block: Callable[[], TransformerBlock.Config],
) -> ConfigT:
    """Shrink a trainer config by size only, in place.

    Args:
      config: A full-size recipe of either port; both share these field names.
      precision: One of :data:`PRECISIONS`.
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
    config.total_train_steps = 10
    config.max_act_steps = 3
    config.ema_warmup_steps = min(config.ema_warmup_steps, 2)
    config.model.channels_in = 4
    config.model.num_heads = 1
    config.model.num_layers = 1
    config.model.puzzle_emb_len = 2
    config.model.slow_cycles = 2
    config.model.fast_cycles = 2
    config.model.compile = False
    if config.model.block is None:
        config.model.block = recipe_block()
    assert isinstance(config.model.block.ffn, SwiGLU.Config)
    config.model.block.ffn.round_to = 4
    if precision == "fp32":
        config.dtype_autocast = config.model.dtype = None
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
        steps.append({"steps": subject._pool_h_step.clone()})
    put_steps(out, "data", data)
    put_steps(out, "train", train)
    put_steps(out, "pool", steps)
    _put(out, "pool/final", {**_pool(subject), **_latents(subject)})
    _put(out, "post", _state(subject))
    _put(out, "ema", dict(subject.ema_shadow or {}))
    _put(out, "eval_after", flatten(subject.eval_loss(**evaluation)))
    return out


def run(config: Makeable[object], scratch: Path) -> dict[str, Tensor]:
    """Write the dataset, build the trainer, and record it.

    Args:
      config: A shrunk trainer config of either port.
      scratch: The directory the config points at.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    write_dataset(scratch / "data")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        return record(cast(Subject, config.make()))


def golden_path(recipe: str, precision: str) -> Path:
    """Where the golden for ``recipe`` in ``precision`` lives."""
    suffix = "" if precision == "fp32" else f"_{precision}"
    return _CWD / "testdata" / f"{recipe}{suffix}.pt"


def load_golden(recipe: str, precision: str) -> dict[str, Tensor]:
    """Load a frozen golden."""
    return read_tensors(golden_path(recipe, precision))


@pytest.mark.parametrize(("recipe", "precision"), CASES)
def test_golden_replays_bit_for_bit(
    recipe: str,
    precision: str,
    tmp_path: Path,
) -> None:
    """The trainer reproduces the frozen trajectory with zero mismatches."""
    actual = run(port_config(recipe, precision, tmp_path), tmp_path)
    report = mismatches(load_golden(recipe, precision), actual)
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


def test_golden_bites(tmp_path: Path) -> None:
    """A changed loss weight is reported, not absorbed."""
    config = port_config("exp010", "fp32", tmp_path)
    config.csp_loss_weight = 0.25
    report = mismatches(load_golden("exp010", "fp32"), run(config, tmp_path))
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


def _pool(subject: Subject) -> dict[str, Tensor]:
    """Return the ACT slot state, latents aside."""
    return {
        "inputs": subject._pool_inputs,
        "labels": subject._pool_labels,
        "steps": subject._pool_h_step,
        "halted": subject._pool_halted,
        "puzzle_ids": subject._pool_puzzle_ids,
        "feedback": subject._pool_feedback,
    }


# Recorded once, after the last step: the pool is carried state, so its final value
# already depends on every step before it.
def _latents(subject: Subject) -> dict[str, Tensor]:
    """Return the ACT slots' carried latents."""
    return {"z_slow": subject._pool_z_slow, "z_fast": subject._pool_z_fast}


def _put(out: dict[str, Tensor], prefix: str, values: Mapping[str, Tensor]) -> None:
    """Store :func:`stored` copies of ``values`` under ``prefix``."""
    for key, value in values.items():
        out[f"{prefix}/{key}"] = stored(value)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
