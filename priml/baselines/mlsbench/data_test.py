"""Tests for the per-setting batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from priml.baselines.mlsbench.data import SettingBatches
from priml.baselines.mlsbench.testing import build_task


if TYPE_CHECKING:
    from pathlib import Path


def test_hidden_settings_run_too(tmp_path: Path) -> None:
    r = build_task(tmp_path / "up")
    batches = SettingBatches(
        SettingBatches.Config(task="toy-task", mlsbench_root=str(r)),
    )
    labels = [b["setting"] for b in batches.eval_dataloader()]
    assert labels == ["a-x", "b"]  # "b" is hidden.


def test_settings_subset_selects_and_validates(tmp_path: Path) -> None:
    r = build_task(tmp_path / "up")
    cfg = SettingBatches.Config(task="toy-task", mlsbench_root=str(r), settings=("b",))
    assert [b["setting"] for b in SettingBatches(cfg).eval_dataloader()] == ["b"]
    bad = SettingBatches.Config(task="toy-task", mlsbench_root=str(r), settings=("z",))
    with pytest.raises(ValueError, match="no setting"):
        SettingBatches(bad).eval_dataloader()


def test_the_clone_is_required() -> None:
    with pytest.raises(ValueError, match="mlsbench_root"):
        SettingBatches(SettingBatches.Config(task="toy-task")).eval_dataloader()


def test_there_is_no_training_stream() -> None:
    with pytest.raises(NotImplementedError, match="eval_only"):
        SettingBatches(SettingBatches.Config(task="t")).train_dataloader()


def test_state_dict_round_trips() -> None:
    a = SettingBatches(SettingBatches.Config(task="t"))
    with a.timer_epoch:
        pass
    b = SettingBatches(SettingBatches.Config(task="t"))
    b.load_state_dict(a.state_dict())
    assert b.state_dict() == a.state_dict()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
