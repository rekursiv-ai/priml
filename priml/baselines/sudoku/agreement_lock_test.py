from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import logging

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.agreement_lock import (
    AgreementLockEval,
    Member,
    modal_grid_predictions,
)
from priml.baselines.sudoku.runner import open_npz
from priml.baselines.sudoku.search_dump import learned_hps_output_width
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    metric_tensors,
    replay,
)
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.sudoku.puzzle_data import PuzzleDataset


def test_agreement_lock_needs_two_members_with_checkpoints() -> None:
    with pytest.raises(ValueError, match="at least two"):
        AgreementLockEval.Config(
            experiment_name="x",
            members=(Member("x", NINE_VIEWS[0]),),
        ).make()
    with pytest.raises(ValueError, match="needs a checkpoint"):
        AgreementLockEval.Config(experiment_name="x").make()


def test_agreement_lock_needs_a_name_and_a_positive_population() -> None:
    members = (Member("a.pt", NINE_VIEWS[0]), Member("a.pt", NINE_VIEWS[1]))
    with pytest.raises(ValueError, match="experiment_name is required"):
        AgreementLockEval.Config(members=members).make()
    with pytest.raises(ValueError, match="evaluation_count must be positive"):
        AgreementLockEval.Config(
            experiment_name="x",
            members=members,
            evaluation_count=0,
        ).make()


_SUBSET: Final = (5, 7, 9)
"""Global test indices of the targeted union run."""

_LABELS: Final = {5: [2, 3, 4, 5], 7: [9, 9, 9, 9], 9: [4, 5, 6, 8]}
"""Solution of each global index (a 4-cell toy grid)."""

_MEMBER_GRIDS: Final = (
    {5: [2, 3, 4, 5], 7: [3, 4, 5, 6], 9: [4, 5, 6, 7]},
    {5: [2, 3, 4, 5], 7: [9, 9, 9, 9], 9: [4, 5, 6, 7]},
    {7: [9, 9, 9, 9]},
)
"""Each member's grid per global index: the pair agrees on 5 and 9 only."""


class _FakeSearch:
    """A ``search_pass`` replaying ``_MEMBER_GRIDS`` over the requested subset."""

    def __init__(self, *, skewed_member: int = -1) -> None:
        self.skewed_member = skewed_member
        self.requested: list[tuple[int, tuple[int, ...]]] = []

    def __call__(self, **kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        dataset = cast("PuzzleDataset", kwargs["dataset"])
        indices = dataset.config.eval_instance_indices
        member = int(str(kwargs["label"]).split()[1])
        self.requested.append((member, indices))
        grids = torch.tensor([_MEMBER_GRIDS[member][i] for i in indices])
        rows = torch.zeros(len(indices), learned_hps_output_width(4))
        rows[:, -4:] = grids
        media = torch.ones(len(indices), 4, dtype=torch.int64)
        media += int(member == self.skewed_member)
        labels = torch.tensor([_LABELS[i] for i in indices])
        return rows, media, labels


def _subset_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search: _FakeSearch,
) -> AgreementLockEval:
    def eval_model(*_: object) -> object:
        return object()

    module = "priml.baselines.sudoku.agreement_lock"
    monkeypatch.setattr(f"{module}.eval_model", eval_model)
    monkeypatch.setattr(f"{module}.search_pass", search)
    cfg = AgreementLockEval.Config()
    cfg.experiment_name = "lock"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.eval_instance_indices = _SUBSET
    cfg.members = tuple(Member("gen.pt", view) for view in NINE_VIEWS[:3])
    return cfg.make()


def test_agreement_lock_locks_the_pair_and_ladders_survivors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pair locks 5 and 9; only global index 7 reaches the third member."""
    search = _FakeSearch()
    lock = _subset_lock(tmp_path, monkeypatch, search)
    with caplog.at_level(logging.INFO):
        metrics = lock.run("--passthrough")
    assert "--passthrough" in caplog.text
    assert search.requested == [(0, _SUBSET), (1, _SUBSET), (2, (7,))]
    assert metrics["eval/locked_count"] == 2
    assert metrics["eval/disagreement_count"] == 1
    assert metrics["eval/wrong_locks"] == 1
    assert metrics["eval/n_correct"] == 2
    assert metrics["eval/mean_passes"] == (2 * 2 + 3 * 1) / 3
    assert metrics["eval/member_2_exact_accuracy"] == 1.0
    dump_path = tmp_path / "runs" / "lock" / "dumps" / "agreement_lock_dump.npz"
    with open_npz(dump_path) as dump:
        final = torch.from_numpy(dump["final_grids"]).long()
        survivors = torch.from_numpy(dump["survivors"])
    assert torch.equal(final, torch.tensor([_MEMBER_GRIDS[1][i] for i in _SUBSET]))
    assert torch.equal(survivors, torch.tensor([1]))


@pytest.mark.parametrize(
    ("skewed_member", "message"),
    [(1, "first-pair member passes"), (2, "ladder member pass")],
)
def test_agreement_lock_refuses_misaligned_member_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skewed_member: int,
    message: str,
) -> None:
    lock = _subset_lock(
        tmp_path,
        monkeypatch,
        _FakeSearch(skewed_member=skewed_member),
    )
    with pytest.raises(RuntimeError, match=message):
        lock.run()


def test_modal_grid_breaks_ties_to_the_first_member() -> None:
    grid = torch.arange(2 * 81).reshape(2, 81) % 9 + 2
    assert modal_grid_predictions(
        torch.stack((grid[:2], grid[:2].flip(0), grid[:2])),
    )[1].tolist() == [0, 0]
    with pytest.raises(ValueError, match=r"\[members, puzzles, cells\]"):
        modal_grid_predictions(torch.zeros(2, 81))


def _record_lock(h: Harness) -> dict[str, Tensor]:
    gen_a = h.generator("gen_a", seed=0)
    gen_b = h.generator("gen_b", seed=1)
    cfg = AgreementLockEval.Config()
    cfg.experiment_name = "lock"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.model = h.model()
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.evaluation_count = 2
    views = NINE_VIEWS
    cfg.members = (
        Member(gen_a, views[0]),
        Member(gen_b, views[0], name="gen_b"),
        Member(gen_a, views[1]),
    )
    return {**metric_tensors(cfg.make().run()), **h.dumps("lock")}


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "agreement_lock": _record_lock,
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
