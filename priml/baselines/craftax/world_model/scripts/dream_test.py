"""Check reference sampling and a generated-episode report with its bundles."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

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
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
    exp_smoke,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scripts import dream
from priml.baselines.craftax.world_model.testing import (
    small_schema,
    tiny_model,
)
from priml.lib.codec import from_plain, loads


def test_sample_real_draws_only_the_split_and_arms_asked_for(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    episodes = dream.sample_real(corpus, split=VALIDATION, arms=(3,), count=5, seed=0)
    assert sorted(e.receipt.world_seed for e in episodes) == [3, 4]
    both = dream.sample_real(corpus, split=VALIDATION, arms=(1, 3), count=2, seed=0)
    again = dream.sample_real(corpus, split=VALIDATION, arms=(1, 3), count=2, seed=0)
    assert [e.receipt.world_seed for e in both] == [e.receipt.world_seed for e in again]
    assert len(both) == 2
    assert {e.receipt.split for e in both} == {VALIDATION}


def test_sample_real_skips_branches_which_start_mid_game(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    episodes = dream.sample_real(corpus, split=TRAIN, arms=(3,), count=5, seed=0)
    assert sorted(e.receipt.world_seed for e in episodes) == [1, 2]
    for seed in range(4):
        drawn = dream.sample_real(corpus, split=TRAIN, arms=(3,), count=1, seed=seed)
        assert [e.receipt.world_seed for e in drawn] in ([1], [2])


def test_main_refuses_a_corpus_without_reference_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dream, "load_world_model", _tiny_trained)
    corpus, output = _corpus(tmp_path), tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *("dream.py", str(tmp_path / "step.pt"), "--corpus", str(corpus)),
            *("--arms", "2", "--rows", "2", "--decisions", "3"),
            *("--device", "cpu", "--output", str(output)),
        ],
    )
    with pytest.raises(ValueError, match="holds no val episode of arms"):
        dream.main()
    assert not output.exists()


@pytest.mark.compute_large_fixture
def test_main_writes_a_report_and_viewer_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _checkpoint(tmp_path / "step.pt")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dream.py",
            str(tmp_path / "step.pt"),
            "--experiment",
            "priml.baselines.craftax.world_model.experiments.exp_smoke",
            *("--rows", "2", "--decisions", "3"),
            *("--corpus", str(_corpus(tmp_path)), "--reference", "2"),
            *("--bundles", "1", "--device", "cpu", "--output", str(output)),
        ],
    )
    assert dream.main() == 0
    report = from_plain(
        loads((output / "report.json").read_text()),
        dict[str, object],
    )
    generated = from_plain(report["generated"], dict[str, object])
    real = from_plain(report["real"], dict[str, object])
    assert from_plain(generated["episodes"], int) == 2
    assert from_plain(real["episodes"], int) == 2
    assert generated["horizon"] == real["horizon"] == 3
    found = from_plain(report["departures"], dict[str, object])
    assert from_plain(found["generated"], dict[str, object])["episodes"] == 2
    # Both real episodes are longer than the horizon of 3.
    assert from_plain(found["real"], dict[str, object])["decisions"] == 2 * 3
    run = from_plain(report["run"], dict[str, object])
    assert from_plain(run["decisions_generated"], int) == 6
    assert run["dtype"] == "torch.float32"
    manifest = from_plain(
        loads((output / "bundles" / "generated-0" / "manifest.json").read_text()),
        dict[str, object],
    )
    assert manifest["actions"] == 3
    assert not (output / "bundles" / "generated-1").exists()
    samples = from_plain(
        cast("object", torch.load(output / "samples.pt", weights_only=True)),
        dict[str, dict[str, torch.Tensor]],
    )
    cells = samples["generated"]["cells"]
    assert cells.shape == (1, 3, 99, 8)
    # A saved view would carry every row's storage, and a [:, :H] slice of the
    # D + 1 frames the wrong row stride.
    assert cells.untyped_storage().nbytes() == cells.nbytes
    assert samples["real"]["length"].shape == (1,)


def _episode(decisions: int, *, split: int, arm: int, world_seed: int) -> Episode:
    """Return a terminal episode of constant frames, its actions the seed mod 43."""
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=arm,
            split=split,
        ),
        actions=torch.full((decisions,), world_seed % 43, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.ones(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={},
        # World 6's episode stands in for a branch: only its origin marks one.
        origin=bytes(8) if world_seed == 6 else b"",
    )


def _corpus(root: Path) -> Path:
    """Publish arm 3's training shard, with a branch, and two validation shards."""
    entries: list[tuple[Path, ManifestLine]] = []
    for split, arm, seeds in (
        (TRAIN, 3, [1, 2, 6]),
        (VALIDATION, 3, [3, 4]),
        (VALIDATION, 1, [5]),
    ):
        directory = root / ("train" if split == TRAIN else "val") / f"arm{arm}" / "w0"
        directory.mkdir(parents=True)
        episodes = [
            _episode(2 + seed, split=split, arm=arm, world_seed=seed) for seed in seeds
        ]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    path = root / "corpora" / "test.json"
    write_corpus(path, entries=entries)
    return path


def _checkpoint(path: Path) -> WorldModel:
    """Save a smoke-size model as ``TrainLoop.state_dict`` nests it; return it."""
    torch.manual_seed(1)
    model = exp_smoke().step.model.make()
    assert isinstance(model, WorldModel)
    torch.save({"step": {"model": model.state_dict()}, "rng": {}}, path)
    return model


def _tiny_trained(
    experiment: str,
    checkpoint: Path,
    *,
    overrides: Sequence[str] = (),
) -> tuple[WorldModel, WorldModelLoop.Config]:
    """Stand in for ``load_world_model``: a tiny model and the smoke experiment."""
    del experiment, checkpoint, overrides
    return tiny_model(small_schema()), exp_smoke()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
