"""Check that a checkpoint is scored against the baselines on its validation batches.

The model's loading and its tallies have tests of their own (``scoring_test``,
and ``baselines_test`` beside ``baselines.py``); these tests stand in for them.
"""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

import dataclasses
import sys

import pytest
import torch

from priml.baselines.craftax.world_model.batch import PackedBatch
from priml.baselines.craftax.world_model.experiments import WorldModelLoop
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scripts import baselines
from priml.lib.codec import PlainTree, from_plain, loads


_MODEL = cast("WorldModel", object())
"""The model the stand-in loader returns."""

_REPORT: dict[str, PlainTree] = {
    "model_nll": {"action": 1.5},
    "empirical_nll": {"action": 1.0},
    "cell_accuracy": {"model": 0.5, "copy": 1.0},
    "beats": {"action": False},
}
"""The report the stand-in scoring gives."""


def test_main_writes_the_report_and_the_runs_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cast("WorldModelLoop.Config", object())
    loaded: list[tuple[str, Path, list[str], torch.device]] = []

    def load_trained(
        experiment: str,
        checkpoint: Path,
        *,
        overrides: list[str],
        device: torch.device,
    ) -> tuple[WorldModel, WorldModelLoop.Config]:
        loaded.append((experiment, checkpoint, overrides, device))
        return _MODEL, config

    def evaluate(
        model: WorldModel,
        given: WorldModelLoop.Config,
        *,
        device: torch.device,
    ) -> tuple[dict[str, PlainTree], dict[str, PlainTree]]:
        assert (model, given, device) == (_MODEL, config, torch.device("cpu"))
        return _REPORT, {"corpus": "smoke.json", "batches": 2}

    monkeypatch.setattr(baselines, "load_trained", load_trained)
    monkeypatch.setattr(baselines, "evaluate", evaluate)
    output = tmp_path / "out" / "baselines.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baselines.py",
            str(tmp_path / "step.pt"),
            *("--experiment", "experiments.exp_smoke"),
            *("--override", "base_dir=/b", "--override", "seed=2"),
            *("--device", "cpu", "--output", str(output)),
        ],
    )
    assert baselines.main() == 0
    overrides = ["base_dir=/b", "seed=2"]
    assert loaded == [
        ("experiments.exp_smoke", tmp_path / "step.pt", overrides, torch.device("cpu")),
    ]
    assert from_plain(loads(output.read_text()), dict[str, object]) == {
        **_REPORT,
        "run": {
            "checkpoint": str(tmp_path / "step.pt"),
            "experiment": "experiments.exp_smoke",
            "overrides": overrides,
            "corpus": "smoke.json",
            "batches": 2,
        },
    }


def test_evaluate_tallies_every_validation_batch_on_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [_batch(), _batch()]
    dataset = _Dataset(batches=batches)
    tallied: list[PackedBatch] = []

    def tally(model: WorldModel, media: PackedBatch) -> dict[str, torch.Tensor]:
        assert model is _MODEL
        tallied.append(media)
        return {"n": torch.ones(1)}

    monkeypatch.setattr(baselines, "tally", tally)
    monkeypatch.setattr(baselines, "report", _count)
    result, run = baselines.evaluate(
        _MODEL,
        cast(
            "WorldModelLoop.Config",
            SimpleNamespace(dataset=dataset, step=SimpleNamespace(dtype_autocast=None)),
        ),
        device=torch.device("cpu"),
    )
    assert dataset.device == torch.device("cpu")
    assert tallied == batches
    assert result == {"tallies": 2}
    assert (run["corpus"], run["batches"]) == ("corpora/smoke.json", 2)
    assert from_plain(run["seconds"], float) >= 0


@dataclasses.dataclass(kw_only=True, slots=True)
class _Dataset:
    """Stand in for an experiment's dataset config and the dataset it makes."""

    batches: list[PackedBatch]
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device("meta"),
    )
    corpus: str = "corpora/smoke.json"

    def make(self) -> Self:
        return self

    def eval_dataloader(self) -> Iterator[dict[str, PackedBatch]]:
        return iter([{"media": batch} for batch in self.batches])


def _batch() -> PackedBatch:
    """Return a micro-batch of empty tensors: the stand-in tally reads none."""
    return PackedBatch(
        **{field.name: torch.empty(0) for field in dataclasses.fields(PackedBatch)},
    )


def _count(tallies: list[dict[str, torch.Tensor]]) -> dict[str, PlainTree]:
    """Stand in for ``baselines.report``: count the tallies."""
    return {"tallies": len(tallies)}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
