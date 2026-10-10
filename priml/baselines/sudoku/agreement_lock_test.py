from __future__ import annotations

from typing import TYPE_CHECKING, Final

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.agreement_lock import (
    AgreementLockEval,
    Member,
    modal_grid_predictions,
)
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


def test_agreement_lock_needs_two_members_with_checkpoints() -> None:
    with pytest.raises(ValueError, match="at least two"):
        AgreementLockEval.Config(
            experiment_name="x",
            members=(Member("x", NINE_VIEWS[0]),),
        ).make()
    with pytest.raises(ValueError, match="needs a checkpoint"):
        AgreementLockEval.Config(experiment_name="x").make()


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
    from priml.lib.testing.main import test_main

    test_main(__file__)
