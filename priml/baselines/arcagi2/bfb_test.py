"""Replay an authentic-source ARC2 optimizer trajectory in exported priml."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi2.experiments import exp000
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.lib.custom_json import DictCodec
from priml.model.attention.self_attention import SelfAttention
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.arcagi2.train_step import ArcTrainStep
    from priml.train.custom_types import TrainStepOutput


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _source_golden_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent candidate replay from overwriting authentic source evidence."""
    monkeypatch.setenv("BFB_REGENERATE", "0")


class _Step(Protocol):
    @property
    def model(self) -> nn.Module: ...

    def train_step(self, **batch: object) -> TrainStepOutput: ...


class Trajectory(nn.Module):
    """Canonical checkpoint names independent of either implementation's tree."""

    def __init__(
        self,
        step: _Step,
        *,
        slow: Tensor,
        fast: Tensor,
        table: Tensor,
    ) -> None:
        super().__init__()
        self.step = step
        self.body = nn.ParameterList(list(step.model.parameters()))
        self.slow = nn.Buffer(slow)
        self.fast = nn.Buffer(fast)
        self.table = nn.Buffer(table)

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        """Capture logits, losses, and all body gradients through five updates."""
        gradients: dict[int, Tensor] = {}
        handles = [
            parameter.register_hook(_capture(gradients, index))
            for index, parameter in enumerate(self.body.parameters())
        ]
        snapshots: list[Tensor] = []
        try:
            for index, media in enumerate(inputs):
                result = self.step.train_step(
                    media=media,
                    label=(torch.arange(18).reshape(2, 9) + 3) % 12,
                    puzzle_identifiers=torch.tensor([index % 3 + 1, 2]),
                    valid_count=2,
                )
                snapshots.extend(
                    [
                        result["loss"].float().flatten(),
                        result["model"].float().flatten(),
                    ],
                )
                snapshots.extend(
                    gradients[key].float().flatten() for key in sorted(gradients)
                )
        finally:
            for handle in handles:
                handle.remove()
        return torch.cat(snapshots)


def _capture(gradients: dict[int, Tensor], index: int) -> Callable[[Tensor], None]:
    """Capture one parameter's gradient without changing backward."""

    def capture(gradient: Tensor) -> None:
        gradients[index] = gradient.detach().clone()

    return capture


def miniature_config() -> ArcTrainStep.Config:
    """Shrink the recipe without replacing its numerical choices."""
    candidate = exp000().step
    candidate.parallelism = NoParallel.Config(device="cpu")
    model = candidate.model
    assert isinstance(model.embedding, GridEmbedding.Config)
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, SelfAttention.Config)
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    model.channels_in = 8
    model.num_layers = 1
    model.embedding.grid_shape = (9,)
    model.block.attn.num_heads = 2
    model.block.attn.channels_head = 4
    model.block.rope.channels_head = 4
    model.prefix.num_puzzles = 8
    model.prefix.batch_size = 2
    assert candidate.act is not None
    candidate.act.batch_size = 2
    candidate.act.max_steps = 3
    candidate.total_train_steps = 10
    candidate.warmup_steps = 0
    candidate.use_ema = False
    candidate.compile = None
    candidate.parallelism.device = "cpu"
    return candidate


def build_port() -> Trajectory:
    """Build the exported implementation under canonical checkpoint names."""
    step = miniature_config().make()
    return Trajectory(
        step,
        slow=step.net.slow_init,
        fast=step.net.fast_init,
        table=step.puzzle_embedding.weights,
    )


def build_input() -> Tensor:
    """Five changing batches with recurring and newly seated task IDs."""
    return torch.stack(
        [(torch.arange(18).reshape(2, 9) + index) % 12 for index in range(5)],
    )


def test_source_trajectory_golden() -> None:
    """Replay source-minted gradients, logits, losses and post-update weights."""
    directory = _CWD / "testdata"
    assert (directory / "source_trajectory.pt").is_file(), (
        "Only the source oracle may mint this golden"
    )
    assert_bfb_against_golden(
        golden_dir=directory,
        golden_name="source_trajectory",
        build_module=build_port,
        build_input=build_input,
    )


def test_source_initialization_golden() -> None:
    """Compare every original initialized parameter, buffer and RNG byte."""
    expected = DictCodec.coerce(
        cast(
            object,
            torch.load(_CWD / "testdata" / "source_init.pt", weights_only=True),
        ),
        Tensor,
    )
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        port = build_port()
        actual = dict(port.state_dict())
        actual["rng"] = torch.get_rng_state()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), f"source initialization: {key}"
        assert actual[key].dtype == value.dtype


def _corrupt_run(module: nn.Module, inputs: object) -> Tensor:
    assert isinstance(module, Trajectory)
    assert isinstance(inputs, Tensor)
    with torch.no_grad():
        next(module.body.parameters()).add_(0.125)
    return module(inputs)


def test_source_golden_rejects_changed_embedding() -> None:
    """A changed embedding must fail the exact trajectory oracle."""
    with pytest.raises(AssertionError):
        assert_bfb_against_golden(
            golden_dir=_CWD / "testdata",
            golden_name="source_trajectory",
            build_module=build_port,
            build_input=build_input,
            run=_corrupt_run,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
