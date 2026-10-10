from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import latent_recursion, puzzle_data
from priml.baselines.sudoku.hps_eval import HpsEval, search_pass
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.search import HpsSearch, SearchResult
from priml.baselines.sudoku.search_dump import learned_hps_output_width
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    metric_tensors,
    replay,
    write_dataset,
)
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def test_hps_eval_requires_a_name_and_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        HpsEval.Config().make()
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        HpsEval.Config(experiment_name="x").make()


def _record_hps(h: Harness) -> dict[str, Tensor]:
    checkpoint = h.generator("gen_a", seed=0)
    cfg = HpsEval.Config()
    cfg.experiment_name = "hps"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.checkpoint_path = checkpoint
    cfg.model = h.model()
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.evaluation_count = 2
    # A three-step generator's halt logits sit near its -5 init, far below the
    # recipe's 7.875. A probe pass that accepts every root reads them; the
    # thresholds are then placed among them -- the root one at the median, the
    # node one a quartile lower -- so the golden records root accepts, search
    # accepts, and search failures at any model size.
    probe = cfg.copy_tree()
    probe.experiment_name = "hps_probe"
    probe.search.acceptance_threshold = -1e9
    probe.make().run()
    roots = h.dumps("hps_probe")["dumps/hps_eval_predictions.npz/rows/scores"][:, 0]
    cfg.search.root_acceptance_threshold = float(roots.double().quantile(0.5))
    cfg.search.acceptance_threshold = float(roots.double().quantile(0.25))
    return {**metric_tensors(cfg.make().run()), **h.dumps("hps")}


def test_hps_eval_rejects_a_population_short_of_evaluation_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2-puzzle split clips ``evaluation_count=3``; the run must not publish."""
    cfg = HpsEval.Config()
    cfg.checkpoint_path = "/runs/generator.pt"
    cfg.experiment_name = "hps"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.evaluation_count = 3

    def search_pass(**_: object) -> tuple[Tensor, Tensor, Tensor]:
        labels = torch.full((2, 81), 2, dtype=torch.int64)
        return torch.zeros(2, learned_hps_output_width(81)), labels, labels

    def eval_model(*_: object) -> object:
        return object()

    monkeypatch.setattr("priml.baselines.sudoku.hps_eval.eval_model", eval_model)
    monkeypatch.setattr("priml.baselines.sudoku.hps_eval.search_pass", search_pass)
    with pytest.raises(RuntimeError, match=r"evaluated 2 puzzles.*population is 3"):
        cfg.make().run()
    assert not (tmp_path / "runs" / "hps" / "dumps").exists()


def test_search_pass_joins_solution_visited_in_view_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oracle solving the shifted puzzle emits the shifted solution."""
    view = NINE_VIEWS[1]
    assert view.name == "shift1"
    dataset = PuzzleDataset.Config(
        working_dir=write_dataset(tmp_path / "data"),
        batch_size=2,
        device="cpu",
    )

    def oracle(
        model: object,
        batch: Mapping[str, object],
        search: object,
    ) -> SearchResult:
        del model, search
        label = cast("Tensor", batch["label"])
        rows = label.shape[0]
        return SearchResult(
            accepted=torch.zeros(rows, dtype=torch.bool),
            root_predictions=view.apply(label),
            final_predictions=view.apply(label),
            scores=torch.zeros(rows),
            root_scores=torch.zeros(rows),
            nodes=torch.zeros(rows, dtype=torch.int64),
            depth=torch.full((rows,), -1),
            scored=torch.ones(rows, dtype=torch.bool),
            visited_predictions=[],
            visited_puzzles=[],
        )

    monkeypatch.setattr("priml.baselines.sudoku.hps_eval.run_search", oracle)
    rows, _, _ = search_pass(
        model=cast("latent_recursion.LatentRecursion", object()),
        dataset=dataset.finalize().make(),
        view=view,
        search=HpsSearch.Config(),
        device=torch.device("cpu"),
        deadline_seconds=float("inf"),
        label="probe",
        join_solution_visited=True,
    )
    assert rows[:, 5].tolist() == [1.0, 1.0]


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "hps": _record_hps,
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
