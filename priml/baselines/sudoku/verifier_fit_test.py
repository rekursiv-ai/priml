from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    replay,
    weight_tensors,
)
from priml.baselines.sudoku.verifier import (
    DenoisingVerifier,
    SudokuVerifier,
    row_column_positions,
)
from priml.baselines.sudoku.verifier_fit import (
    DenoisingLoss,
    VerifierFit,
    VerifierLoss,
)
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_verifier_fit_needs_a_positive_step_budget() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        VerifierFit.Config(max_steps=0).make()


def _record_verifier(h: Harness) -> dict[str, Tensor]:
    h.harvest(h.generator("gen_a", seed=0))
    state = cast(
        "dict[str, dict[str, dict[str, Tensor]]]",
        torch.load(h.verifier(seed=0), weights_only=True),
    )
    return weight_tensors("model", state["step"]["model"])


def _record_denoising_fit(h: Harness) -> dict[str, Tensor]:
    h.harvest(h.generator("gen_a", seed=0))
    model = DenoisingVerifier.Config()
    model.channels_hidden = 4
    model.num_layers = 1
    model.heads = 2
    model.attention_mask = None
    model.position_index = row_column_positions
    state = cast(
        "dict[str, dict[str, dict[str, Tensor]]]",
        torch.load(
            h.verifier(seed=0, model=model, loss=DenoisingLoss.Config()),
            weights_only=True,
        ),
    )
    return weight_tensors("model", state["step"]["model"])


def test_denoising_loss_adds_masked_cross_entropy_on_correct_cells() -> None:
    label = torch.tensor([1.0, 0.0])
    cell_label = torch.zeros(2, 81)
    cell_label[1, 3] = 1.0
    cell_mask = torch.ones(2, 81)
    plain = torch.linspace(-1, 1, 2 * 82).reshape(2, 82)
    base = VerifierLoss.Config().make()(
        plain,
        label=label,
        cell_label=cell_label,
        cell_mask=cell_mask,
    )
    config = DenoisingLoss.Config()
    config.denoise_weight = 0.5
    loss = config.make()
    unchanged = loss(plain, label=label, cell_label=cell_label, cell_mask=cell_mask)
    assert torch.equal(unchanged, base)
    target = torch.full((2, 81), -1.0)
    target[0, :2] = 4.0
    target[1, 3] = 4.0  # A wrong candidate digit is never a target.
    block = torch.cat([plain, torch.zeros(2, 81 * 9), target], dim=-1)
    total = loss(block, label=label, cell_label=cell_label, cell_mask=cell_mask)
    # Uniform logits: each scored cell costs log 9; row 1 scores no cell.
    uniform = torch.log(torch.tensor(9.0))
    assert torch.equal(total, base + torch.stack([0.5 * uniform, torch.tensor(0.0)]))


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "verifier_fit": _record_verifier,
    "denoising_fit": _record_denoising_fit,
}


@pytest.mark.parametrize("case", _RECORDERS)
def test_golden_replays_bit_for_bit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    puzzle_data.dihedral_indices.cache_clear()
    report = mismatches(load_golden(case), replay(_RECORDERS[case], tmp_path))
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


def test_verifier_fit_needs_a_positive_clip_norm() -> None:
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        VerifierFit.Config(gradient_clip_norm=0.0).make()


def test_a_fit_step_under_autocast_updates_the_weights() -> None:
    """The bf16 forward feeds an fp32 loss and moves every trained matrix."""
    torch.manual_seed(0)
    fit = object.__new__(VerifierFit)
    fit.config = VerifierFit.Config(dtype_autocast=torch.bfloat16)
    fit.device = torch.device("cpu")
    fit.model = SudokuVerifier.Config(channels_hidden=4, num_layers=1, heads=2).make()
    fit.loss = VerifierLoss.Config().make()
    fit.optimizer = torch.optim.AdamW(fit.model.parameters(), lr=1e-2)
    fit.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fit.optimizer, T_max=2)
    before = fit.model.candidate_embedding.weight.detach().clone()
    candidate = torch.randint(2, 11, (3, 81))
    loss = fit._train_step(
        {
            "puzzle": torch.ones(3, 81, dtype=torch.long),
            "candidate": candidate,
            "label": torch.tensor([1.0, 0.0, 0.0]),
            "cell_label": (candidate % 2).float(),
            "cell_mask": torch.ones(3, 81),
        },
    )
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(fit.model.candidate_embedding.weight, before)
    assert all(p.grad is None for p in fit.model.parameters())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
