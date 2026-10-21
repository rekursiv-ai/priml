"""Checks for the metric module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig
from pyarrow import Table, parquet

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.metric import MoleculeMRR


if TYPE_CHECKING:
    from pathlib import Path


def identity(smiles: str) -> str | None:
    """Use small molecule labels to exercise ranking independently of RDKit."""
    return smiles.lower() if smiles != "invalid" else None


def metric_config(root: Path) -> MoleculeMRR.Config:
    """Write the smallest distinct candidate/feature geometry for two mass windows."""
    parquet.write_table(
        Table.from_pydict(
            {"mass": [100.0] * 3 + [200.0] * 5, "smiles": list("ABCDEFGH")},
        ),
        where=root / "pool_meta.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {"sid": [0, 1], "smiles": ["A", "E"], "metric_identity": ["a", "e"]},
        ),
        where=root / "train_metric_structs.parquet",
    )
    np.save(root / "pool_fp.npy", np.packbits(np.eye(8, 7, dtype=np.uint8), axis=1))
    config = MoleculeMRR.Config()
    config.working_dir = root
    config.identity = PartialConfig(identity)
    return config


def test_metric_averages_all_spectra_then_counts_molecules(tmp_path: Path) -> None:
    config = metric_config(tmp_path)
    metric = config.make()
    logits = torch.tensor(
        [[0, 3, 2, 0, 0, 0, 0], [0, 3, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 3, -1]],
        dtype=torch.float32,
    )
    metric.update(
        logits,
        sid=torch.tensor([0, 0, 1]),
        neutral_mass=torch.tensor([100.0, 100.0, 200.0], dtype=torch.float64),
    )
    expected = (1 / 3 + 1 / 2) / 2
    assert metric.compute() == {"mrr@25": expected, "molecules": 2.0}
    state = metric.state_dict()
    assert state["counts"] == {0: 2, 1: 1}
    assert torch.equal(state["sums"][0], logits[0] + logits[1])
    assert state["masses"] == {0: [100.0, 100.0], 1: [200.0]}
    restored = config.make()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()
    metric.reset()
    assert metric.compute() == {"mrr@25": 0.0, "molecules": 0.0}


def test_complete_model_output_scores_its_first_entry(tmp_path: Path) -> None:
    logits = torch.tensor(
        [[0, 3, 2, 0, 0, 0, 0], [0, 3, 2, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    batch = {
        "sid": torch.tensor([0, 0]),
        "neutral_mass": torch.tensor([100.0, 100.0]),
    }
    metric = metric_config(tmp_path).make()
    metric.update((logits, torch.zeros(2, 10)), **batch)
    expected = metric_config(tmp_path).make()
    expected.update(logits, **batch)
    assert metric.compute() == expected.compute()
    assert metric.compute()["mrr@25"] > 0


def test_identity_deduplication_and_cutoff(tmp_path: Path) -> None:
    config = metric_config(tmp_path)
    config.limit = 2
    metric = config.make()
    metric.smiles[:3] = ["A", "B", "b"]
    metric.update(
        torch.tensor(
            [[0, 3, 2, 0, 0, 0, 0], [0, 3, 2, 0, 0, 0, 0]],
            dtype=torch.float32,
        ),
        sid=torch.tensor([0, 0]),
        neutral_mass=torch.tensor([100.0, 100.0], dtype=torch.float64),
    )
    assert metric.compute()["mrr@2"] == 1 / 2
    config.limit = 1
    limited = config.make()
    limited.load_state_dict(metric.state_dict())
    assert limited.compute()["mrr@1"] == 0


def test_invalid_truth_and_absent_mass_window(tmp_path: Path) -> None:
    metric = metric_config(tmp_path).make()
    metric.update(
        torch.zeros(2, 7),
        sid=torch.tensor([0, 1]),
        neutral_mass=torch.tensor([500.0, 600.0], dtype=torch.float64),
    )
    assert metric.compute()["mrr@25"] == 0
    metric.truth[0] = "invalid"
    with pytest.raises(ValueError, match="Invalid truth"):
        _ = metric.compute()


def test_misaligned_spectrum_fields_are_rejected(tmp_path: Path) -> None:
    metric = metric_config(tmp_path).make()
    with pytest.raises(ValueError, match="zip"):
        metric.update(
            torch.zeros(3, 7),
            sid=torch.tensor([0, 1]),
            neutral_mass=torch.tensor([100.0, 200.0], dtype=torch.float64),
        )


def test_all_alias_structures_share_one_molecule_statistic(tmp_path: Path) -> None:
    config = metric_config(tmp_path)
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 1, 2],
                "smiles": ["A", "a", "E"],
                "metric_identity": ["a", "a", "e"],
            },
        ),
        where=tmp_path / "train_metric_structs.parquet",
    )
    metric = config.make()
    metric.update(
        torch.zeros(3, 7),
        sid=torch.tensor([0, 1, 2]),
        neutral_mass=torch.tensor([100.0, 100.0, 200.0], dtype=torch.float64),
    )
    assert metric.state_dict()["counts"] == {0: 2, 2: 1}
    assert metric.compute()["molecules"] == 2.0


def test_missing_masses_do_not_hide_finite_windows(tmp_path: Path) -> None:
    metric = metric_config(tmp_path).make()
    metric.update(
        torch.tensor([[3, 0, 0, 0, 0, 0, 0]] * 3, dtype=torch.float32),
        sid=torch.tensor([0, 0, 1]),
        neutral_mass=torch.tensor([100.0, float("nan"), float("inf")]),
    )
    assert metric.compute() == {"mrr@25": 0.5, "molecules": 2.0}


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limits_fail_before_loading(tmp_path: Path, limit: int) -> None:
    config = MoleculeMRR.Config()
    config.working_dir = tmp_path
    config.limit = limit
    with pytest.raises(ValueError, match="limit"):
        _ = config.make()


def test_score_ties_follow_identity_not_pool_order(tmp_path: Path) -> None:
    config = metric_config(tmp_path)
    metric = config.make()
    metric.smiles[:3] = ["C", "B", "A"]
    metric.update(
        torch.zeros(2, 7),
        sid=torch.tensor([0, 0]),
        neutral_mass=torch.tensor([100.0, 100.0]),
    )
    assert metric.compute()["mrr@25"] == 1.0


def test_tied_alias_order_cannot_change_reciprocal_rank(tmp_path: Path) -> None:
    metric = metric_config(tmp_path).make()
    metric.truth[0] = "C"
    metric.smiles[:3] = ["b", "B", "C"]
    metric.update(
        torch.zeros(2, 7),
        sid=torch.tensor([0, 0]),
        neutral_mass=torch.tensor([100.0, 100.0]),
    )
    expected = {"mrr@25": 0.5, "molecules": 1.0}
    assert metric.compute() == expected
    metric.smiles[:3] = ["B", "b", "C"]
    assert metric.compute() == expected


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
