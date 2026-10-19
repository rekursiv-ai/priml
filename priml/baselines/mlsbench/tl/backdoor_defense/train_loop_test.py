"""Tests for the victim-filter-retrain loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.experiments_test import (
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.tl.backdoor_defense.experiments import exp000
from priml.baselines.mlsbench.tl.backdoor_defense.poison import PoisonedData
from priml.train.checkpointer import Checkpointer


if TYPE_CHECKING:
    from pathlib import Path


def test_the_retrained_network_trains_on_the_kept_images_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_dataset(tmp_path, classes=10)
    cfg = shrink(exp000(), directory=tmp_path)
    cfg.dataset.poison_fraction = 0.25
    seen: list[torch.Tensor] = []
    keep = PoisonedData.keep

    def recording(self: PoisonedData, indices: torch.Tensor) -> None:
        seen.append(indices)
        keep(self, indices)

    monkeypatch.setattr(PoisonedData, "keep", recording)
    loop = cfg.make()
    loop.train()
    assert loop.filtered is not None
    (kept,) = seen
    assert torch.equal(kept, torch.from_numpy(loop.filtered.kept))
    assert not torch.isin(kept, torch.from_numpy(loop.filtered.removed)).any()
    assert loop.report is not None
    assert loop.report.poison_recall == loop.filtered.poison_recall


def test_a_checkpointer_would_resume_the_retraining_so_it_is_refused(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    cfg = shrink(exp000(), directory=tmp_path)
    cfg.checkpointer = Checkpointer.Config()
    with pytest.raises(
        ValueError,
        match=r"^BackdoorTrainLoop needs checkpointer=None\.$",
    ):
        cfg.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
