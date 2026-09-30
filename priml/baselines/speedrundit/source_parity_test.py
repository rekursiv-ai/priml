"""SpeedrunDiT initialization and three-step training golden.

``testdata/reg_source.pt`` was minted from this recipe after the same run of
REG commit 3c51606's own ``SiT``, ``SILoss``, and ``CombinedOptimizer`` was
shown bit-for-bit equal to it at this config, with REG's hardcoded timestep
frequency (256) and decoder block count (2) set to 2 and 1 (see
``/opt/scratch/probes/puzzlespec/speedrundit/parity_tiny.py``), so the golden
pins REG's numerics without needing REG at test time.
The whole post-3-step state implies backward bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor

import pytest
import torch

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.objective import SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.compute_training
def test_initial_and_three_updates_golden() -> None:
    """Initial state, three losses, and final state, bitwise."""
    generator = torch.Generator().manual_seed(7)
    # Reference parity fixture; dimensions are recorded by the upstream golden.
    image = torch.randn(2, 2, 4, 4, generator=generator, dtype=torch.float64)
    time = torch.rand(2, generator=generator, dtype=torch.float64)
    cls = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    # Recorded against REG's own run; the golden fixes these sizes.
    teacher = torch.randn(3, 2, 17, 2, generator=generator, dtype=torch.float64)
    inputs = {
        "image": image.float(),
        "time": time.float(),
        "label": torch.arange(2),
        "cls": cls.float(),
        "teacher": teacher.float(),
    }
    training: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        torch.manual_seed(42)
        model = (
            SpeedrunDiT.Config(
                input_size=4,
                in_channels=2,
                patch_size=1,
                hidden_size=8,
                depth=4,
                num_heads=2,
                num_classes=2,
                cls_channels=2,
                timestep_frequencies=2,
                projector_hidden=2,
                projection_depths=(2, 3, 4),
                mlp_ratio_min=0.25,
                mlp_ratio_max=0.25,
                decoder_blocks=1,
            )
            .make()
            .train()
        )
        training |= {
            f"initial_state/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        optimizer = speedrundit_optimizer().make()(model)
        objective = SpeedrunObjective()
        losses: list[Tensor] = []
        for step in range(3):
            torch.manual_seed(1234 + step)
            terms = objective(
                model,
                inputs["image"],
                inputs["label"],
                tuple(inputs["teacher"].unbind()),
            )
            losses.append(terms.mean_loss.detach().clone())
            terms.mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    training["step_losses"] = torch.stack(losses)
    training |= {
        f"post_state/{name}": value.detach().clone()
        for name, value in model.state_dict().items()
    }
    assert_tensor_golden(_CWD / "testdata" / "reg_source.pt", training)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
