"""Check the engine checks end to end on a smoke-size checkpoint, on the CPU."""

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
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.experiments import exp_smoke
from priml.baselines.craftax.world_model.scripts import engine_checks
from priml.lib.codec import from_plain, loads


@pytest.mark.compute_large_fixture
def test_every_check_passes_on_a_smoke_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)
    model = exp_smoke().step.model.make()
    torch.save({"step": {"model": model.state_dict()}}, tmp_path / "step.pt")
    output = tmp_path / "checks.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_checks.py",
            str(tmp_path / "step.pt"),
            str(_corpus(tmp_path)),
            str(output),
            "--experiment",
            "priml.baselines.craftax.world_model.experiments.exp_smoke",
            *("--decisions", "4", "--draws", "2000", "--device", "cpu"),
        ],
    )
    assert engine_checks.main() == 0
    results = from_plain(loads(output.read_text()), dict[str, object])
    assert list(results) == [
        "checkpoint",
        "corpus",
        "decisions",
        "1_teacher_forced_fp32",
        "1_teacher_forced_bf16_vs_training_numerics",
        "2_prefill_then_step_equals_step_only_fp32",
        "3_reset_one_row_leaves_others_bf16",
        "4_gumbel_max_matches_softmax_action_head",
        "5_fixed_seed_and_session_reload",
        "all_passed",
    ]
    teacher = from_plain(results["1_teacher_forced_fp32"], dict[str, object])
    # A start job and one act job per decision.
    assert teacher["jobs"] == 5
    assert from_plain(
        results["2_prefill_then_step_equals_step_only_fp32"],
        dict[str, object],
    )["free_decisions_identical_tokens"] == ("8/8")
    assert results["all_passed"] is True


@pytest.mark.parametrize("origin", [b"", b"branch state"])
def test_the_segment_is_the_first_long_enough_validation_episode(
    tmp_path: Path,
    origin: bytes,
) -> None:
    corpus = _corpus(tmp_path, origin=origin)
    segment = engine_checks.real_segment(corpus, decisions=7)
    # Episode 8 has 10 decisions, episode 9 has 11, and the training episode of
    # 12 is never read.
    assert len(segment.actions) == 7
    assert len(segment.cells) == 8
    assert int(segment.actions[0]) == 8
    # A branch's first decision continues its parent's episode.
    assert segment.starts_episode == (not origin)
    with pytest.raises(ValueError, match="long enough"):
        engine_checks.real_segment(corpus, decisions=10)


def _corpus(root: Path, *, origin: bytes = b"") -> Path:
    """Publish one long training episode, then validation episodes from ``origin``."""
    entries: list[tuple[Path, ManifestLine]] = []
    for split, seeds in ((TRAIN, [10]), (VALIDATION, [3, 8, 9])):
        directory = root / ("train" if split == TRAIN else "val") / "arm3" / "w0"
        directory.mkdir(parents=True)
        episodes = [
            _episode(
                2 + seed,
                split=split,
                world_seed=seed,
                origin=origin if split == VALIDATION else b"",
            )
            for seed in seeds
        ]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    path = root / "corpora" / "test.json"
    write_corpus(path, entries=entries)
    return path


def _episode(decisions: int, *, split: int, world_seed: int, origin: bytes) -> Episode:
    """Return a terminal episode of constant frames, its actions the seed."""
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=3,
            split=split,
        ),
        actions=torch.full((decisions,), world_seed, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.ones(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={},
        origin=origin,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
