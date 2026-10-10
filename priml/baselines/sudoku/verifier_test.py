from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.baselines.sudoku.testing import sudoku_solution
from priml.baselines.sudoku.verifier import (
    DenoisingVerifier,
    SudokuVerifier,
    VerifierAcceptor,
    _gelu_cost,
    row_column_positions,
)
from priml.cost import cost


if TYPE_CHECKING:
    from pathlib import Path


def test_verifier_channels_must_divide_by_heads() -> None:
    with pytest.raises(ValueError, match="channels_hidden"):
        SudokuVerifier.Config(channels_hidden=5, heads=2).make()


def test_acceptor_rejects_an_empty_committee_or_chunk() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        VerifierAcceptor.Config(checkpoint_paths=("v.pt",), max_rows=0).make()
    with pytest.raises(ValueError, match=">= 1 checkpoint"):
        VerifierAcceptor.Config(checkpoint_paths=()).make()


def test_gelu_cost_counts_a_small_block() -> None:
    cost = _gelu_cost(channels=2, rows=3, dtype=torch.float32)
    assert cost["flops", "primal", "elementwise", torch.float32] == 48
    assert cost["flops", "adjoint", "elementwise", torch.float32] == 48


def test_gelu_cost_counts_exact_forward_and_backward_work() -> None:
    result = _gelu_cost(channels=3, rows=5, dtype=torch.float32)
    assert result["flops", "primal", "elementwise", torch.float32] == 120
    assert result["flops", "adjoint", "elementwise", torch.float32] == 120
    assert result["bytes", "primal", "elementwise", torch.float32] == 120
    assert result["bytes", "adjoint", "elementwise", torch.float32] == 180
    float64_result = _gelu_cost(channels=3, rows=5, dtype=torch.float64)
    assert float64_result["flops", "primal", "elementwise", torch.float64] == 120
    assert float64_result["bytes", "primal", "elementwise", torch.float64] == 240


def _denoising_verifier(seed: int) -> DenoisingVerifier:
    """Build a tiny prior-free verifier: full attention, row/column coordinates."""
    cfg = DenoisingVerifier.Config()
    cfg.channels_hidden = 4
    cfg.num_layers = 1
    cfg.heads = 2
    cfg.attention_mask = None
    cfg.position_index = row_column_positions
    torch.manual_seed(seed)
    return cfg.make()


def test_denoising_verifier_scores_like_its_parent_in_eval_mode() -> None:
    model = _denoising_verifier(0).eval()
    assert model.attention_mask is None
    assert model.position_embedding.weight.shape == (18, 4)
    parent = SudokuVerifier.Config()
    parent.channels_hidden = 4
    parent.num_layers = 1
    parent.heads = 2
    parent.attention_mask = None
    parent.position_index = row_column_positions
    torch.manual_seed(0)
    plain = parent.make().eval()
    puzzle = torch.stack([sudoku_solution(), sudoku_solution()])
    puzzle[:, ::3] = 1
    candidate = torch.stack([sudoku_solution(), sudoku_solution().roll(1)])
    assert torch.equal(model(puzzle, candidate), plain(puzzle, candidate))


def test_denoising_verifier_appends_masked_digit_targets_in_training() -> None:
    model = _denoising_verifier(0).train()
    puzzle = torch.stack([sudoku_solution(), sudoku_solution(), sudoku_solution()])
    puzzle[:, 40:] = 1
    candidate = puzzle.clone()
    candidate[:, 40:] = sudoku_solution()[40:]
    torch.manual_seed(1)
    output = model(puzzle, candidate)
    assert output.shape == (3, 82 + 81 * 9 + 81)
    target = output[:, 82 + 81 * 9 :]
    masked = target >= 0
    assert not masked[:, :40].any()
    assert masked[:, 40:].any()
    assert torch.equal(target[masked], (candidate - 2).float()[masked])
    assert bool((target[~masked] == -1).all())
    for fraction in (0.0, 1.0, float("nan")):
        cfg = DenoisingVerifier.Config()
        cfg.denoise_fraction = fraction
        with pytest.raises(ValueError, match="denoise_fraction"):
            cfg.make()


def test_verifier_cost_matches_torch() -> None:
    """Torch counts attention only through the math kernel, so pin it.

    A fused SDPA kernel is invisible to ``FlopCounterMode``, and which kernel
    CPU SDPA picks varies by host and by what ran before in the process.
    """
    config = SudokuVerifier.Config(channels_hidden=16, num_layers=1, heads=2).finalize()
    analytical = cost(config, batch_size=2, dtype=None)
    torch.manual_seed(0)
    module = config.make()
    module.train()
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        module(
            torch.randint(1, 11, (2, 81)),
            torch.randint(2, 11, (2, 81)),
        ).sum().backward()
    assert analytical.params == sum(p.numel() for p in module.parameters())
    assert analytical["flops", "matmul"].sum() == counter.get_total_flops()


def _member(seed: int) -> SudokuVerifier:
    """Build a tiny group-masked verifier member."""
    torch.manual_seed(seed)
    return SudokuVerifier.Config(channels_hidden=4, num_layers=1, heads=2).make().eval()


def test_acceptor_locks_only_when_every_member_strictly_clears(tmp_path: Path) -> None:
    """Members load from either checkpoint schema; the lock is their minimum."""
    first, second = _member(0), _member(1)
    torch.save(
        {"step": {"model": first.state_dict(), "global_step": 3}},
        tmp_path / "a.pt",
    )
    (tmp_path / "runs").mkdir()
    torch.save({"model": second.state_dict()}, tmp_path / "runs" / "b.pt")
    puzzle = torch.stack([sudoku_solution(), sudoku_solution()])
    puzzle[:, ::2] = 1
    candidate = torch.stack([sudoku_solution(), sudoku_solution().roll(1)])
    # One row per forward, as max_rows=1 scores them: a two-row batch takes a
    # different x86 BLAS blocking and differs in the last bit, so the exact
    # comparison only holds when both sides run the same shapes.
    with torch.no_grad():
        expected = torch.cat(
            [
                torch.minimum(
                    first(puzzle[row : row + 1], candidate[row : row + 1])[:, 0],
                    second(puzzle[row : row + 1], candidate[row : row + 1])[:, 0],
                )
                for row in range(candidate.shape[0])
            ],
        )
    config = VerifierAcceptor.Config(
        model=SudokuVerifier.Config(channels_hidden=4, num_layers=1, heads=2),
        base_dir=tmp_path,
        checkpoint_paths=(tmp_path / "a.pt", "/runs/b.pt"),
        max_rows=1,
        device="cpu",
    )
    acceptor = config.make()
    scores = acceptor.scores(candidate, puzzle)
    assert torch.equal(scores, expected)
    assert acceptor.scores(candidate[:0], puzzle[:0]).shape == (0,)
    config.threshold = float(scores.min())
    accepted = config.make()(candidate, puzzle)
    assert accepted.tolist() == (scores > scores.min()).tolist()
    assert not accepted[scores.argmin()]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
