"""REG source model, backward, and five-step optimizer parity."""

# The source artifact is a nested torch.load dictionary without a static schema.
# pyright: reportAny=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.objective import SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.testing.bfb import host_agnostic_numerics


def _source_size_model() -> SpeedrunDiT:
    return SpeedrunDiT.Config(
        input_size=4,
        in_channels=2,
        patch_size=1,
        hidden_size=32,
        depth=6,
        num_heads=4,
        num_classes=4,
        cls_channels=8,
        projector_hidden=16,
        projection_depths=(2, 3, 6),
    ).make()


@pytest.mark.compute_training
def test_reg_source_forward_backward_and_five_updates() -> None:
    """Replay an artifact minted from REG commit 3c51606, independent of priml."""
    reference = torch.load(
        Path(__file__).parent / "testdata" / "reg_source.pt",
        weights_only=True,
        map_location="cpu",
    )
    assert reference["source_commit"] == "3c51606c801dd9e87ee9ef778782766ab7c379ca"
    with host_agnostic_numerics():
        torch.manual_seed(42)
        model = _source_size_model().train()
        assert model.state_dict().keys() == reference["state"].keys()
        for name, expected_tensor in reference["state"].items():
            assert torch.equal(model.state_dict()[name], expected_tensor), name
        model.load_state_dict(reference["state"])
        inputs = reference["input"]
        torch.manual_seed(123)
        output = model(inputs["image"], inputs["time"], inputs["label"], inputs["cls"])
        expected = reference["forward"]
        assert torch.equal(output.velocity, expected["velocity"])
        assert torch.equal(output.cls_velocity, expected["cls"])
        for projection, tokens, ids in zip(
            output.projections,
            expected["projections"],
            expected["ids"],
            strict=True,
        ):
            assert torch.equal(projection.tokens, tokens)
            if ids is None:
                assert projection.ids_keep is None
            else:
                assert projection.ids_keep is not None
                assert torch.equal(projection.ids_keep, ids)
        (
            output.velocity.sum()
            + output.cls_velocity.sum()
            + sum(p.tokens.sum() for p in output.projections)
        ).backward()
        for name, expected_grad in reference["backward"].items():
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected_grad), name

        model.zero_grad(set_to_none=True)
        optimizer = speedrundit_optimizer().make()(model)
        objective = SpeedrunObjective()
        teacher = tuple(reference["teacher"])
        for step, expected_loss in enumerate(reference["step_losses"]):
            torch.manual_seed(1234 + step)
            terms = objective(model, inputs["image"], inputs["label"], teacher)
            assert torch.equal(terms.mean_loss, expected_loss), step
            terms.mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for name, expected_tensor in reference["post_state"].items():
            assert torch.equal(model.state_dict()[name], expected_tensor), name
