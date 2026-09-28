"""REG source model, backward, and five-step optimizer parity.

``testdata/reg_source.pt`` holds SHA-256 digests minted from REG commit 3c51606:
each section is the raw 32-byte digests concatenated in this model's own order
(``state_dict`` for state, ``named_parameters`` for gradients).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import hashlib

import pytest
import torch

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.objective import SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent


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


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).digest()


def _digests(packed: torch.Tensor) -> list[bytes]:
    raw = packed.numpy().tobytes()
    return [raw[i : i + 32] for i in range(0, len(raw), 32)]


def _assert_digests(
    tensors: list[tuple[str, torch.Tensor]],
    packed: torch.Tensor,
) -> None:
    expected = _digests(packed)
    assert len(tensors) == len(expected)
    for (name, tensor), digest in zip(tensors, expected, strict=True):
        assert _tensor_digest(tensor) == digest, name


@pytest.mark.compute_training
def test_reg_source_forward_backward_and_five_updates() -> None:
    """Replay an artifact minted from REG commit 3c51606, independent of priml."""
    reference = read_tensors(_CWD / "testdata" / "reg_source.pt")
    commit = reference["source_commit"].numpy().tobytes().decode()
    assert commit == "3c51606c801dd9e87ee9ef778782766ab7c379ca"
    with host_agnostic_numerics():
        torch.manual_seed(42)
        model = _source_size_model().train()
        _assert_digests(list(model.state_dict().items()), reference["state"])
        torch.manual_seed(123)
        output = model(
            reference["input/image"],
            reference["input/time"],
            reference["input/label"],
            reference["input/cls"],
        )
        assert [_tensor_digest(output.velocity)] == _digests(
            reference["forward/velocity"],
        )
        assert [_tensor_digest(output.cls_velocity)] == _digests(
            reference["forward/cls"],
        )
        assert [_tensor_digest(p.tokens) for p in output.projections] == _digests(
            reference["forward/projections"],
        )
        present = reference["forward/ids_present"].tolist()
        assert [p.ids_keep is not None for p in output.projections] == present
        kept = [p.ids_keep for p in output.projections if p.ids_keep is not None]
        assert [_tensor_digest(ids) for ids in kept] == _digests(
            reference["forward/ids"],
        )
        (
            output.velocity.sum()
            + output.cls_velocity.sum()
            + sum(p.tokens.sum() for p in output.projections)
        ).backward()
        grads: list[tuple[str, torch.Tensor]] = []
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            grads.append((name, parameter.grad))
        _assert_digests(grads, reference["backward"])

        model.zero_grad(set_to_none=True)
        optimizer = speedrundit_optimizer().make()(model)
        objective = SpeedrunObjective()
        teacher = tuple(reference["teacher"].unbind())
        for step, expected_digest in enumerate(_digests(reference["step_losses"])):
            torch.manual_seed(1234 + step)
            terms = objective(
                model,
                reference["input/image"],
                reference["input/label"],
                teacher,
            )
            assert _tensor_digest(terms.mean_loss) == expected_digest, step
            terms.mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        _assert_digests(list(model.state_dict().items()), reference["post_state"])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
