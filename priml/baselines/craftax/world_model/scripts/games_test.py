"""Check ranking an archive's games and saving one as an exact game or policy view."""

from pathlib import Path

import dataclasses
import sys

import pytest

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    write_shard,
)
from priml.baselines.craftax.world_model.scripts import games
from priml.baselines.craftax.world_model.viewer import policy_view
from priml.baselines.craftax.world_model.viewer.exact import (
    Manifest,
    read_manifest,
)
from priml.lib.codec import from_plain, loads


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return an archive of five episodes, three complete, one truncated, one a branch.

    The complete ones return 5, 9 and 9 points, the second 9 in fewer decisions.
    """
    root = tmp_path_factory.mktemp("archive")
    short, shorter, longer = (
        replay.record(world_seed=seed, sampling_seed=1, max_decisions=500)
        for seed in (4, 7, 2)
    )
    assert len(shorter.actions) < len(short.actions) < len(longer.actions)
    train, val = root / "train" / "arm0" / "w0", root / "val" / "arm0" / "w0"
    for directory in (train, val):
        directory.mkdir(parents=True)
    write_shard(
        train,
        index=0,
        episodes=[_summarized(short, 5.0), _summarized(shorter, 9.0)],
        provenance={},
    )
    cut = _summarized(short, 20.0, truncated=1)
    branch = _summarized(shorter, 30.0, branch={"decision": 3})
    write_shard(
        val,
        index=0,
        episodes=[_summarized(longer, 9.0), cut, branch],
        provenance={},
    )
    return root


def test_rank_takes_the_highest_return_then_the_fewest_decisions(archive: Path) -> None:
    ranking = games.rank(archive)
    assert ranking.best == games.Candidate(
        episode="train/arm0/w0/shard-000000/1",
        score=9.0,
        decisions=25,
    )
    assert (ranking.complete, ranking.mean_score) == (3, (5.0 + 9.0 + 9.0) / 3)


def test_rank_refuses_an_archive_with_no_complete_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No complete episode"):
        games.rank(tmp_path)


def test_rank_prints_its_choice(
    archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["games.py", "rank", str(archive)])
    assert games.main() == 0
    printed = from_plain(loads(capsys.readouterr().out), dict[str, object])
    assert (
        from_plain(printed["best"], dict[str, object])["episode"]
        == "train/arm0/w0/shard-000000/1"
    )
    assert printed["complete"] == 3


def test_save_writes_the_best_game_beside_the_archive_average(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "best"
    argv = ["games.py", "save", str(archive), str(output), "--title", "Best game"]
    monkeypatch.setattr(sys, "argv", argv)
    assert games.main() == 0
    manifest = read_manifest((output / "manifest.json").read_text(), Manifest)
    assert (manifest.title, manifest.actions, manifest.end_label) == (
        "Best game",
        25,
        "Episode end",
    )
    assert manifest.mean_score == (5.0 + 9.0 + 9.0) / 3
    assert "train/arm0/w0/shard-000000/1" in manifest.provenance
    assert "3 complete episodes" in manifest.provenance


def test_save_takes_a_named_episode_and_a_prefix(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prefix"
    episode = "val/arm0/w0/shard-000000/0"
    argv = ["games.py", "save", str(archive), str(output), "--episode", episode]
    monkeypatch.setattr(sys, "argv", [*argv, "--limit", "20"])
    assert games.main() == 0
    manifest = read_manifest((output / "manifest.json").read_text(), Manifest)
    assert (manifest.title, manifest.actions) == (episode, 20)
    assert manifest.end_label == "End of recording"


def test_panel_writes_a_named_prefix_as_a_policy_view_bundle(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "panel"
    episode = "val/arm0/w0/shard-000000/0"
    argv = ["games.py", "panel", str(archive), str(output), "--episode", episode]
    monkeypatch.setattr(sys, "argv", [*argv, "--limit", "20"])
    assert games.main() == 0
    manifest = read_manifest(
        (output / "manifest.json").read_text(),
        policy_view.Manifest,
    )
    assert (manifest.decisions, manifest.archive, manifest.episode) == (
        20,
        str(archive),
        episode,
    )
    assert manifest.world_seed == 2
    assert (output / "policy-view.bin.gz").exists()


def test_an_unpublished_shard_is_named(archive: Path) -> None:
    with pytest.raises(ValueError, match="shard-000009"):
        games.read_episode(archive, "val/arm0/w0/shard-000009/0")


def _summarized(episode: Episode, score: float, **extra: object) -> Episode:
    """Return ``episode`` with a capture summary line of the given return."""
    summary = {"episode": 0, "return": score, "death": 1, "timeout": 0, **extra}
    return dataclasses.replace(episode, summary=summary)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
