"""Tests that evaluation reports preserve their checkpoint inputs."""

from pathlib import Path

import sys

from configgle.launch import resolve_config
from treekle import json_loads

import pytest
import torch

from priml.baselines.etth1 import experiments
from priml.baselines.etth1.data_test import fixture_config
from priml.baselines.etth1.experiments import exp_smoke
from priml.baselines.etth1.scripts import evaluation
from priml.baselines.etth1.train_step import Etth1TrainLoop


@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_output_cannot_replace_best_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    cfg = exp_smoke()
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    cfg = cfg.copy_tree().finalize()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "step_00000000.pt"
    torch.save(
        {
            "step": {
                "model": cfg.step.model.make().state_dict(),
                "timer_step": {"global_count": 0},
            },
        },
        f=checkpoint,
    )
    selector = checkpoints / "best.json"
    selector.write_text('{"metric": "total_loss", "step": 0}\n')
    before = selector.read_bytes()
    output = selector
    if alias != "direct":
        output = tmp_path / "report.json"
        if alias == "hardlink":
            output.hardlink_to(selector)
        else:
            output.symlink_to(selector)
    monkeypatch.setattr(experiments, "exp000", lambda: cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--directory",
            str(cfg.dataset.working_dir),
            "--checkpoint",
            str(checkpoints),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="protected input artifact"):
        evaluation.main()
    assert selector.read_bytes() == before


@pytest.mark.parametrize("name", ["exp000", "exp001", "exp004"])
def test_evaluation_uses_selected_recipe(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = resolve_config(f"{experiments.__name__}.{name}")
    assert isinstance(cfg, Etth1TrainLoop.Config)
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    cfg = cfg.copy_tree().finalize()
    model = cfg.step.model.make()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"step": {"model": model.state_dict(), "timer_step": {"global_count": 0}}},
        f=checkpoint,
    )
    expected = evaluation.evaluate(model, batches=cfg.dataset.make().test_dataloader())
    monkeypatch.setattr(experiments, name, lambda: cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--experiment",
            name,
            "--checkpoint",
            str(checkpoint),
        ],
    )
    assert evaluation.main() == 0
    report = json_loads(capsys.readouterr().out, dict[str, object])
    assert report["experiment"] == name
    assert report["mse"] == expected["mse"]
    assert report["mae"] == expected["mae"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
