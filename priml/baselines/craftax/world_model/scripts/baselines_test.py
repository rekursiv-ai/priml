"""Check that a checkpoint is scored against the baselines on its validation batches."""

from pathlib import Path

import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Receipt,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.data import StratifiedWindows
from priml.baselines.craftax.world_model.experiments import exp_smoke
from priml.baselines.craftax.world_model.scripts import baselines
from priml.lib.codec import from_plain, loads


@pytest.mark.compute_large_fixture
def test_main_scores_every_validation_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smoke_corpus(tmp_path)
    model = exp_smoke().step.model.make()
    torch.save({"step": {"model": model.state_dict()}}, tmp_path / "step.pt")
    output = tmp_path / "baselines.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baselines.py",
            str(tmp_path / "step.pt"),
            *(
                "--experiment",
                "priml.baselines.craftax.world_model.experiments.exp_smoke",
            ),
            *("--override", f"base_dir={tmp_path}", "--device", "cpu"),
            *("--output", str(output)),
        ],
    )
    assert baselines.main() == 0
    result = from_plain(loads(output.read_text()), dict[str, object])
    assert set(result) == {
        "model_nll",
        "empirical_nll",
        "cell_accuracy",
        "beats",
        "targets",
        "run",
    }
    run = from_plain(result["run"], dict[str, object])
    validation = exp_smoke().dataset.validation
    assert isinstance(validation, StratifiedWindows.Config)
    assert from_plain(run["batches"], int) == validation.batches
    assert from_plain(run["overrides"], list[str]) == [f"base_dir={tmp_path}"]
    assert from_plain(run["corpus"], str).endswith("smoke/corpora/smoke.json")
    targets = from_plain(result["targets"], dict[str, object])
    assert from_plain(targets["board"], int) > 0


def _smoke_corpus(root: Path) -> None:
    """Publish one 40-decision episode per split where ``exp_smoke`` reads them."""
    archive = root / "datasets" / "craftax" / "world-model" / "smoke"
    entries: list[tuple[Path, ManifestLine]] = []
    for split in (0, 1):
        directory = archive / str(split) / "w0"
        directory.mkdir(parents=True)
        done = torch.zeros(40, dtype=torch.bool)
        done[-1] = True
        episode = Episode(
            receipt=Receipt(
                world_seed=split,
                sampling_seed=0,
                initial_state_hash=0,
                arm=0,
                split=split,
            ),
            actions=torch.arange(40, dtype=torch.uint8) % 3,
            hashes=torch.zeros(1, dtype=torch.int64),
            cells=torch.zeros(40, 99, 8, dtype=torch.uint8),
            aux=torch.ones(40, 51, dtype=torch.int16),
            reward=torch.zeros(40, dtype=torch.int16),
            done=done,
            summary={},
        )
        line = write_shard(directory, index=0, episodes=[episode], provenance={})
        entries.append((directory, line))
    write_corpus(archive / "corpora" / "smoke.json", entries=entries)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
