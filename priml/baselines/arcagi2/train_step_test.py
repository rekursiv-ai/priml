"""Exact ARC2 optimizer trajectories against source-minted records.

The implementation this recipe was ported from was recorded once through
:func:`record_trajectory`, in every width, autocast, and clipping arm; the port
must reproduce every recorded tensor. Recording is under ``host_agnostic_numerics``
only: a native-numerics record would pin one host's last bits.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, Protocol

import math

from torch import Tensor, nn
from torch.optim import Optimizer

import pytest
import torch

from priml.baselines.arcagi2.experiments import exp000
from priml.baselines.arcagi2.model import RotaryBlock
from priml.baselines.arcagi2.record_test import assert_matches, load, reduce
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.self_attention import SelfAttention
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, mismatches, put_steps
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.arcagi2.train_step import ArcTrainStep
    from priml.train.custom_types import TrainStepOutput


def training_config(width: int, dtype: torch.dtype | None) -> ArcTrainStep.Config:
    """Shape-compact the reference recipe without replacing its components."""
    candidate = exp000().step
    model = candidate.model
    model.channels_in = width
    model.vocab_size = 4
    model.num_layers = 1
    assert isinstance(model.embedding, GridEmbedding.Config)
    model.embedding.grid_shape = (3,)
    assert isinstance(model.block, RotaryBlock.Config)
    model.block.channels_in = width
    assert isinstance(model.block.attn, SelfAttention.Config)
    model.block.attn.channels_in = width
    model.block.attn.num_heads = 2
    model.block.attn.channels_head = width // 2
    assert model.block.rope is not None
    model.block.rope.channels_head = width // 2
    assert isinstance(model.block.ffn, SwiGLU.Config)
    model.block.ffn.channels_in = width
    model.block.ffn.round_to = 1
    assert isinstance(model.prefix, SparsePuzzleEmbedding.Config)
    model.prefix.num_puzzles = 4
    model.prefix.batch_size = 2
    assert candidate.act is not None
    candidate.act.batch_size = 2
    candidate.act.max_steps = 2
    candidate.dtype_autocast = dtype
    candidate.total_train_steps = 3
    candidate.warmup_steps = 0
    candidate.use_ema = False
    candidate.compile = None
    candidate.parallelism = NoParallel.Config()
    candidate.parallelism.device = "cpu"
    return candidate


def capture_gradients(
    optimizer: Optimizer,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    model: nn.Module,
    sparse: Tensor,
    target: dict[str, Tensor],
) -> None:
    """Capture reduced, clipped dense gradients and unreduced sparse rows."""
    del optimizer, args, kwargs
    target.clear()
    for index, parameter in enumerate(model.parameters()):
        gradient = parameter.grad
        if gradient is not None:
            target[str(index)] = gradient.detach().clone()
    assert sparse.grad is not None
    target["sparse"] = sparse.grad.detach().clone()


def training_batch(index: int, rank: int) -> dict[str, object]:
    """Distinct rank inputs, one rank-exclusive ID, and one shared ID."""
    return {
        "media": (torch.arange(6).reshape(2, 3) + rank + index) % 2 + 2,
        "label": ((torch.arange(6).reshape(2, 3) + rank + 3) % 2) + 2,
        "puzzle_identifiers": torch.tensor([2 * rank + 1, 2]),
        "valid_count": 2,
    }


class Subject(Protocol):
    """One implementation of the ARC2 step, seen through the compared surface."""

    @property
    def model(self) -> nn.Module:
        """The dense body whose gradients and parameters are compared."""
        ...

    @property
    def dense_optimizer(self) -> Optimizer:
        """The optimizer whose step pre-hook captures gradients."""
        ...

    @property
    def sparse_local(self) -> Tensor:
        """The per-batch sparse rows receiving gradients."""
        ...

    @property
    def sparse_table(self) -> Tensor:
        """The master sparse table the optimizer updates."""
        ...

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        ...

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Run one evaluation call."""
        ...

    def resumed(self) -> Subject:
        """Return a fresh subject loaded from this one's state dict."""
        ...


class PortSubject:
    """The exported ARC2 step."""

    def __init__(self, config: ArcTrainStep.Config) -> None:
        self.config = config
        self.step: ArcTrainStep = config.make()

    @property
    def model(self) -> nn.Module:
        """The dense body."""
        return self.step.model

    @property
    def dense_optimizer(self) -> Optimizer:
        """The dense optimizer."""
        assert isinstance(self.step.optimizer, Optimizer)
        return self.step.optimizer

    @property
    def sparse_local(self) -> Tensor:
        """The per-batch sparse rows."""
        return self.step.puzzle_embedding.local_weights

    @property
    def sparse_table(self) -> Tensor:
        """The master sparse table."""
        return self.step.puzzle_embedding.weights

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        return self.step.train_step(**batch)

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Run one evaluation call."""
        return self.step.eval_loss(**batch)

    def resumed(self) -> PortSubject:
        """Return a fresh step loaded from this one's state dict."""
        other = PortSubject(self.config)
        other.step.load_state_dict(self.step.state_dict())
        return other


def record_trajectory(
    build: Callable[[], Subject],
    *,
    clip: float,
) -> dict[str, Tensor]:
    """Record three updates, an evaluation, and a resumed update from seed 0.

    Recorded per update, stacked on a step axis: output probe, loss, the
    every reduced and clipped dense gradient and the unreduced sparse rows,
    the finite clipping norm, and every dense parameter and the sparse table,
    all whole. Then a one-valid-row
    evaluation, and one update of a step restored from the state dict.

    Args:
      build: Constructs the subject; called seeded, under portable numerics.
      clip: Gradient clipping norm; ``inf`` disables it.

    Returns:
      record: Name-to-tensor record.

    """
    out: dict[str, Tensor] = {}
    steps: list[dict[str, Tensor]] = []
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        subject = build()
        gradients: dict[str, Tensor] = {}
        handle = subject.dense_optimizer.register_step_pre_hook(
            partial(
                capture_gradients,
                model=subject.model,
                sparse=subject.sparse_local,
                target=gradients,
            ),
        )
        batch: dict[str, object] = {}
        try:
            for index in range(3):
                batch = {
                    "media": (torch.arange(6).reshape(2, 3) + index) % 2 + 2,
                    "label": (torch.arange(6).reshape(2, 3) + 3) % 2 + 2,
                    "puzzle_identifiers": torch.tensor([index % 2, 1]),
                    "valid_count": 2,
                }
                result = subject.train_step(**batch)
                step = {"model": result["model"], "loss": result["loss"]}
                sparse_grad = gradients.pop("sparse")
                step["grad"] = joined(gradients.values())
                step["grad/sparse"] = rows(sparse_grad)
                if math.isfinite(clip):
                    norm = result.get("metrics", {})["grad_norm"]
                    assert isinstance(norm, Tensor)
                    step["grad_norm"] = norm
                step["param"] = joined(subject.model.parameters())
                step["sparse"] = rows(subject.sparse_table)
                steps.append(step)
        finally:
            handle.remove()
        batch["valid_count"] = 1
        evaluation = subject.eval_loss(**batch)
        out["eval/model"] = evaluation["model"]
        out["eval/loss"] = evaluation["loss"]
        resumed = subject.resumed()
        after = resumed.train_step(**batch)
        out["resumed/loss"] = after["loss"]
        out["resumed/model"] = after["model"]
        out["resumed/param"] = joined(resumed.model.parameters())
    record = reduce(out)
    put_steps(record, "step", [reduce(step) for step in steps])
    return record


def rows(table: Tensor) -> Tensor:
    """Return the first 2 elements of every row: each puzzle's own update."""
    return table.detach().flatten(1)[:, :2].clone()


CASES: Final = (
    *(
        (4, dtype, clip)
        for dtype in (None, torch.bfloat16)
        for clip in (math.inf, 0.01)
    ),
)
"""Every autocast and clipping arm at the minimum non-singleton width four."""


def case_name(width: int, dtype: torch.dtype | None, clip: float) -> str:
    """Key of one case in the frozen source table."""
    return f"w{width}-{dtype}-clip{clip}"


def run_port(
    width: int,
    dtype: torch.dtype | None,
    clip: float,
) -> dict[str, Tensor]:
    """Record the exported step for one case."""
    config = training_config(width, dtype)
    config.gradient_clip_norm = clip
    return record_trajectory(lambda: PortSubject(config), clip=clip)


@pytest.mark.parametrize(("width", "dtype", "clip"), CASES)
def test_reference_training_trajectory(
    width: int,
    dtype: torch.dtype | None,
    clip: float,
) -> None:
    """Losses, gradients, clipping, dense/sparse updates, and resume match."""
    assert_matches(
        "train_step",
        case_name(width, dtype, clip),
        run_port(width, dtype, clip),
    )


def test_reference_trajectory_bites() -> None:
    """A doubled sparse rate is reported, not absorbed."""
    config = training_config(4, None)
    config.gradient_clip_norm = math.inf
    config.sparse_optimizer.lr *= 2
    record = record_trajectory(lambda: PortSubject(config), clip=math.inf)
    expected = load("train_step")[case_name(4, None, math.inf)]
    assert mismatches(expected, record)


def test_arc_step_requires_atomic_act() -> None:
    config = training_config(4, None)
    config.act = None
    with pytest.raises(ValueError, match="requires an atomic ACT"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
