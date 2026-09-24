"""Bit-for-bit goldens for SR-DiT.

These freeze what ``scripts/parity.py`` established against the pinned
reference. ``source_init`` is at the parity script's geometry, since it IS the
reference's construction; the other two shrink the port further, which the
reference's hard-coded widths would not allow. The script needs a network and
a clone and runs once; these run on CPU in the ordinary suite and are what
catch a regression afterwards.

Three goldens, because they fail for different reasons. ``source_init`` is the
REFERENCE's initialization, minted by the parity script and never by this
file: reorder two submodules and the RNG stream shifts, and every weight moves.
``forward`` freezes the op order through routing, rotary positions, and the
value residual. ``five_steps`` drives the real train step, so the objective,
the drawn times and noise, the clip, AdamW's moments, and the EMA all reach
the compared state -- a golden over a hand-rolled loop beside it would stop
noticing when the recipe moved.

``forward`` and ``five_steps`` go through the harness, which randomizes the
parameters and LOADS them on replay; that is why initialization cannot be one
of them, since the load overwrites exactly what construction produced. It is
also why ``five_steps`` cannot see the EMA's construction-time seed, which
predates the load: ``scripts/parity.py`` is what compares that, against the
reference's own ``update_ema``.
Regenerate those two with ``BFB_REGENERATE=1``; a missing one is minted AND
fails, which is what forces someone to read it first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, override

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.scripts.parity import (
    INIT_GOLDEN,
    initialized_state,
    native_config,
)
from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.lib.custom_json import DictCodec
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent

STEPS: Final = 5
BATCH: Final = 2
CHANNELS: Final = 1
GRID: Final = 2
CLASSES: Final = 3
TARGET: Final = 2


def miniature() -> SpeedrunDiT.Config:
    """Shrink the recipe to the smallest model that still takes every path.

    One block per SPRINT stage, one head of four (the axial rotary ladder's
    floor), and every other width at two. Kept: the routing ratios, the
    path-drop probability, the value residual, the rotary positions, the qk
    norms, the zeroed modulation, and every initialization rule.

    Returns:
      cfg: The golden geometry.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = CHANNELS
    cfg.channels_hidden = 4
    cfg.image_size = GRID
    cfg.num_layers = 3
    cfg.heads = 1
    cfg.num_classes = CLASSES
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 2
    cfg.block.ffn.expansion = 2.0
    cfg.time_embedder.channels_frequency = 4
    assert cfg.sprint is not None
    cfg.sprint.num_encoder_layers = 1
    cfg.sprint.num_decoder_layers = 1
    return cfg


def miniature_step() -> SpeedrunDiTTrainStep.Config:
    """Wire the golden geometry into the recipe exp000 runs.

    Size only. The optimizer, the clip, the EMA decay and the objective's
    weights are left as the recipe sets them, which is what the golden is for.

    Returns:
      cfg: A train step at the golden geometry.

    """
    cfg = SpeedrunDiTTrainStep.Config()
    cfg.model = miniature()
    cfg.train_budget_steps = STEPS
    cfg.parallelism = NoParallel.Config(device="cpu")
    # fp32 on CPU: autograd's weight-gradient matmul has no bf16 kernel for the
    # transposed layout on most hosts and falls back to a scalar loop.
    cfg.dtype_autocast = None
    cfg.compile = None
    return cfg


def build_input() -> dict[str, Tensor]:
    """Draw the fixed inputs every golden replays against.

    Returns:
      batch: Latents, times, labels, class features, and alignment targets.

    """
    generator = torch.Generator().manual_seed(0)
    return {
        "media": torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator),
        "time": torch.rand(BATCH, generator=generator),
        "label": torch.randint(CLASSES, (BATCH,), generator=generator),
        "cls_token": torch.randn(BATCH, TARGET, generator=generator),
        "features": torch.randn(BATCH, 1 + GRID * GRID, TARGET, generator=generator),
    }


class _Forward(nn.Module):
    """Runs one forward and reports every output as one vector."""

    def __init__(self, model: SpeedrunDiT) -> None:
        super().__init__()
        self.inner = model

    @override
    def forward(
        self,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        features: Tensor,
    ) -> Tensor:
        """Run the model in eval mode and flatten its three outputs.

        Eval mode deliberately: routing, label dropout and path drop are all
        stochastic in training, and a golden over them would freeze a draw
        rather than an arithmetic.

        Args:
          media: Fixed latents.
          time: Fixed flow times.
          label: Fixed class indices.
          cls_token: Fixed class features.
          features: Unused; the batch is shared with the training golden.

        Returns:
          output: Velocity, projections, and class velocity, concatenated.

        """
        del features
        self.inner.eval()
        out = self.inner(media, time, label, cls_token)
        parts = [out.velocity.float().flatten(), out.cls_velocity.float().flatten()]
        parts.extend(p.float().flatten() for p in out.projections)
        return torch.cat(parts)


# Wrapping the TRAIN STEP rather than the model is what makes the golden cover
# the recipe: the objective, the drawn times and noise, the clip, the schedule,
# and AdamW's moments all reach the post-run weights the harness compares.
class _FiveSteps(nn.Module):
    """Runs the real update five times and reports the trajectory."""

    def __init__(self, config: SpeedrunDiTTrainStep.Config) -> None:
        super().__init__()
        self.step = config.make()
        self.inner = self.step.model

    @override
    def forward(
        self,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        features: Tensor,
    ) -> Tensor:
        """Take five updates and concatenate what each produced.

        Args:
          media: Fixed latents, reused every step.
          time: Unused; the objective draws its own, which this freezes.
          label: Fixed class indices.
          cls_token: Fixed class features.
          features: Fixed alignment targets.

        Returns:
          trajectory: The five losses, then the EMA-averaged weights. The live
          weights are the harness's post-run state, so they are not repeated.

        """
        del time
        # Seeded here, not by the harness: the harness's seed covers parameter
        # randomization, and the times and noise this objective draws have to
        # be reproducible independently of how many parameters it drew for.
        torch.manual_seed(4242)
        pieces: list[Tensor] = []
        for _ in range(STEPS):
            out = self.step.train_step(
                media=media,
                label=label,
                cls_token=cls_token,
                features=[features],
            )
            loss = out["loss"]
            assert isinstance(loss, Tensor)
            pieces.append(loss.float().reshape(1))
        # The shadow lives on the step, which is not a module, so no state
        # the harness compares would otherwise reach it. Cloned INSIDE the
        # swap: a lazy generator read after the context exits sees the
        # restored live weights instead of the shadow.
        with self.step.ema.apply_to(self.inner):
            pieces.extend(
                [p.detach().float().flatten().clone() for p in self.inner.parameters()],
            )
        return torch.cat(pieces)


def _assert_initialization(config: SpeedrunDiT.Config) -> None:
    """Compare a construction with the reference's, tensor by tensor."""
    expected = DictCodec.coerce(
        cast(object, torch.load(_CWD / "testdata" / INIT_GOLDEN, weights_only=True)),
        Tensor,
    )
    actual = initialized_state(config.make)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), f"initialization: {key}"
        assert actual[key].dtype == value.dtype


def test_source_initialization_golden() -> None:
    """Every initialized tensor, and the RNG left behind, are the reference's."""
    assert (_CWD / "testdata" / INIT_GOLDEN).is_file(), (
        "Only scripts/parity.py --mint may mint this golden"
    )
    _assert_initialization(native_config())


def test_source_initialization_golden_bites() -> None:
    """A changed initialization constant must fail the comparison."""
    cfg = native_config()
    assert cfg.value_residual is not None
    cfg.value_residual.initial = 0.25
    with pytest.raises(AssertionError, match="value_residual"):
        _assert_initialization(cfg)


def test_forward_bfb() -> None:
    """Freeze the op order of one eval forward."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speedrundit_forward",
        build_module=lambda: _Forward(miniature().make()),
        build_input=build_input,
        seed=0,
    )


@pytest.mark.compute_training
def test_five_steps_bfb() -> None:
    """Freeze the recipe: objective, draws, clip, AdamW, and EMA."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speedrundit_five_steps",
        build_module=lambda: _FiveSteps(miniature_step()),
        build_input=build_input,
        seed=42,
    )


def test_the_forward_golden_bites() -> None:
    """A golden nobody has seen fail is not evidence.

    Perturbs one parameter and asserts the comparison notices. Done through
    the harness's own ``run`` so the perturbation travels the compared path
    rather than a path built beside it.
    """

    def perturbed(module: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            next(iter(module.parameters())).add_(0.125)
        return cast(Tensor, module(**inputs))

    with pytest.raises(AssertionError):
        assert_bfb_against_golden(
            golden_dir=_CWD / "testdata",
            golden_name="speedrundit_forward",
            build_module=lambda: _Forward(miniature().make()),
            build_input=build_input,
            seed=0,
            run=perturbed,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
