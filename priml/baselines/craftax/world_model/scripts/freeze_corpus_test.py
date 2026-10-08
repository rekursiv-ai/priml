"""Check that a mixture corpus takes whole shards per arm in the shares asked."""

from collections.abc import Sequence
from pathlib import Path

import json
import sys

import pytest

from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_corpus,
)
from priml.baselines.craftax.world_model.capture.control import (
    CaptureHaltedError,
    halt,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.scripts import freeze_corpus
from priml.lib.codec import to_plain


def test_each_arm_takes_the_fewest_whole_shards_of_its_own_for_its_share(
    tmp_path: Path,
) -> None:
    _mixture(tmp_path)
    roots = [tmp_path / "a", tmp_path / "b"]
    selection = freeze_corpus.select_shards(
        roots,
        decisions=100,
        shares=(70, 10, 10, 10),
        seed=0,
    )
    assert list(selection) == [(s, a) for s in (TRAIN, VALIDATION) for a in range(4)]
    # Validation holds 1/19 of each arm's training share, rounded up.
    targets = [70, 10, 10, 10, 4, 1, 1, 1]
    for ((split, arm), entries), target in zip(selection.items(), targets, strict=True):
        own = {
            shard_directory(r, split=split, arm=arm, worker=w)
            for r in roots
            for w in (0, 1)
        }
        assert {directory for directory, _ in entries} <= own
        assert len(set(_names(tmp_path, entries))) == len(entries)
        decisions = [line.decisions for _, line in entries]
        assert sum(decisions[:-1]) < target <= sum(decisions)


def test_the_seed_fixes_each_split_and_arms_own_draw(tmp_path: Path) -> None:
    for arm in (0, 1):
        for worker in range(4):
            _publish(tmp_path, arm=arm, worker=worker, decisions=[1] * 25)
        _publish(tmp_path, split=VALIDATION, arm=arm, decisions=[1])
    first, again, other = (
        freeze_corpus.select_shards(
            [tmp_path],
            decisions=20,
            shares=(1, 1, 0, 0),
            seed=seed,
        )
        for seed in (0, 0, 1)
    )
    assert first == again
    assert first[TRAIN, 0] != other[TRAIN, 0]
    # Arms with equal shard counts must not draw the same shard positions.
    arm0, arm1 = ([(d.name, line.shard) for d, line in first[TRAIN, a]] for a in (0, 1))
    assert arm0 != arm1


def test_a_workers_early_shards_of_short_episodes_are_not_favoured(
    tmp_path: Path,
) -> None:
    # Shards fill in the order episodes end, so shard i holds 200 - i episodes.
    for split, size in ((TRAIN, 1900), (VALIDATION, 100)):
        for worker in range(2):
            _publish(
                tmp_path,
                split=split,
                arm=0,
                worker=worker,
                decisions=[size] * 200,
                episodes=range(200, 0, -1),
            )
    selection = freeze_corpus.select_shards(
        [tmp_path],
        decisions=190_000,
        shares=(1, 0, 0, 0),
        seed=0,
    )
    for split, size in ((TRAIN, 1900), (VALIDATION, 100)):
        taken = [line for _, line in selection[split, 0]]
        assert len(taken) == 100
        episodes = sum(line.episodes for line in taken)
        mean = sum(line.decisions for line in taken) / episodes
        # The population's mean episode: 200 shards of size over 20,100 episodes.
        # A worker's first 100 shards would give 100 * size / 15,050.
        assert mean == pytest.approx(200 * size / 20_100, rel=0.15)


def test_an_arm_short_of_its_share_fails_loudly(tmp_path: Path) -> None:
    _mixture(tmp_path)
    with pytest.raises(ValueError, match="Arm 3 has 18 published train"):
        freeze_corpus.select_shards(
            [tmp_path / "a", tmp_path / "b"],
            decisions=100,
            shares=(50, 10, 10, 30),
            seed=0,
        )


def test_validation_short_of_its_share_fails_loudly(tmp_path: Path) -> None:
    _mixture(tmp_path)
    with pytest.raises(ValueError, match="Arm 0 has 6 published val"):
        freeze_corpus.select_shards(
            [tmp_path / "a", tmp_path / "b"],
            decisions=120,
            shares=(1, 0, 0, 0),
            seed=0,
        )


def test_without_a_decision_count_every_published_shard_is_taken(
    tmp_path: Path,
) -> None:
    _mixture(tmp_path)
    _publish(tmp_path / "a", split=VALIDATION, arm=1, worker=1, decisions=[9])
    selection = freeze_corpus.select_shards(
        [tmp_path / "a", tmp_path / "b"],
        decisions=None,
        shares=(1, 1, 1, 0),
        seed=0,
    )
    # Every shard of arms 0-2 in both splits, however they compare with 1/19,
    # and none of arm 3's.
    assert [len(entries) for entries in selection.values()] == [5, 3, 3, 0, 2, 2, 1, 0]
    assert sum(line.decisions for _, line in selection[VALIDATION, 1]) == 11


def test_a_zero_share_leaves_an_arm_out(tmp_path: Path) -> None:
    _publish(tmp_path, arm=0, decisions=[30, 30])
    _publish(tmp_path, split=VALIDATION, arm=0, decisions=[3])
    selection = freeze_corpus.select_shards(
        [tmp_path],
        decisions=40,
        shares=(1, 0, 0, 0),
        seed=0,
    )
    assert [len(entries) for entries in selection.values()] == [2, 0, 0, 0, 1, 0, 0, 0]


def test_every_arm_with_a_share_gets_a_shard(tmp_path: Path) -> None:
    _mixture(tmp_path)
    selection = freeze_corpus.select_shards(
        [tmp_path / "a", tmp_path / "b"],
        decisions=1,
        shares=(70, 10, 10, 10),
        seed=0,
    )
    assert [len(entries) for entries in selection.values()] == [1] * 8


def test_a_worker_published_under_two_roots_is_refused(tmp_path: Path) -> None:
    _mixture(tmp_path)
    _publish(tmp_path / "b", arm=1, decisions=[6])
    with pytest.raises(ValueError, match="world seeds would repeat"):
        freeze_corpus.select_shards(
            [tmp_path / "a", tmp_path / "b"],
            decisions=100,
            shares=(70, 10, 10, 10),
            seed=0,
        )


def test_a_halted_root_is_refused(tmp_path: Path) -> None:
    _mixture(tmp_path)
    halt(tmp_path / "b", shard="train/arm0/w1/shard-000000", reason="Mismatch.")
    with pytest.raises(CaptureHaltedError):
        freeze_corpus.select_shards(
            [tmp_path / "a", tmp_path / "b"],
            decisions=100,
            shares=(70, 10, 10, 10),
            seed=0,
        )


@pytest.mark.parametrize("shares", [(1, 1, 1), (1, -1, 1, 1), (0, 0, 0, 0)])
def test_shares_name_four_nonnegative_arms(
    tmp_path: Path,
    shares: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shares"):
        freeze_corpus.select_shards([tmp_path], decisions=100, shares=shares, seed=0)


def test_main_freezes_the_corpus_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = [
        "freeze_corpus.py",
        "a",
        "b",
        "--corpus=base",
        "--decisions=100",
        "--seed=7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert freeze_corpus.main() == 0
    entries = read_corpus(tmp_path / "a" / "corpora" / "base.json")
    selection = freeze_corpus.select_shards(
        [tmp_path / "a", tmp_path / "b"],
        decisions=100,
        shares=(70, 10, 10, 10),
        seed=7,
    )
    assert entries == [entry for group in selection.values() for entry in group]
    assert len(entries) == 3 + 2 * 3 + 2 + 3
    assert all(directory.is_absolute() for directory, _ in entries)
    out = capsys.readouterr().out
    assert "train arm 0: 3 shards, " in out
    assert "drawn with seed 7." in out
    with pytest.raises(FileExistsError):
        freeze_corpus.main()


def test_main_freezes_every_published_shard_with_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["freeze_corpus.py", "a", "b", "--corpus=full", "--all"]
    monkeypatch.setattr(sys, "argv", argv)
    assert freeze_corpus.main() == 0
    entries = read_corpus(tmp_path / "a" / "corpora" / "full.json")
    assert len(entries) == 5 + 3 * 3 + 2 + 3
    assert "train arm 0: 5 shards, 130 decisions" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*argv[:-1], "--all", "--decisions=100"])
    with pytest.raises(SystemExit):
        freeze_corpus.main()


def test_combine_takes_every_shard_of_each_corpus_by_split_and_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One worker in three roots, as three generations: v1, v2 fresh, v2 branches.
    for root, rollout in (("v1", "73"), ("fresh", "16000073"), ("branch", "32000073")):
        _publish(tmp_path / root, arm=0, decisions=[5, 7], rollout_seed=rollout)
        if root != "branch":
            _publish(
                tmp_path / root,
                split=VALIDATION,
                arm=0,
                decisions=[1],
                rollout_seed=rollout,
            )
        _publish(tmp_path / root, arm=2, decisions=[3], rollout_seed=f"2{rollout}")
    monkeypatch.chdir(tmp_path)
    for root in ("v1", "fresh", "branch"):
        argv = ["freeze_corpus.py", root, f"--corpus={root}", "--all"]
        monkeypatch.setattr(sys, "argv", argv)
        assert freeze_corpus.main() == 0
    corpora = [
        tmp_path / r / "corpora" / f"{r}.json" for r in ("v1", "fresh", "branch")
    ]
    selection = freeze_corpus.combine_corpora(corpora)
    assert [len(entries) for entries in selection.values()] == [6, 0, 3, 0, 2, 0, 0, 0]
    assert sorted(_names(tmp_path, selection[TRAIN, 0])) == sorted(
        f"{root}/train/arm0/w0/shard-00000{i}"
        for root in ("v1", "fresh", "branch")
        for i in (0, 1)
    )
    argv = [
        "freeze_corpus.py",
        "fresh",
        "--corpus=v1v2",
        "--combine",
        *map(str, corpora),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()
    assert freeze_corpus.main() == 0
    entries = read_corpus(tmp_path / "fresh" / "corpora" / "v1v2.json")
    assert entries == [entry for group in selection.values() for entry in group]
    out = capsys.readouterr().out
    assert "train arm 0: 6 shards, 36 decisions" in out
    assert "val arm 0: 2 shards, 2 decisions" in out


def test_combining_two_captures_of_one_worker_generation_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for root in ("a", "b"):
        _publish(tmp_path / root, arm=1, decisions=[4], rollout_seed="1000073")
        monkeypatch.setattr(
            sys,
            "argv",
            ["freeze_corpus.py", str(tmp_path / root), "--corpus=c", "--all"],
        )
        assert freeze_corpus.main() == 0
    with pytest.raises(ValueError, match="world seeds would repeat"):
        freeze_corpus.combine_corpora(
            [tmp_path / root / "corpora" / "c.json" for root in ("a", "b")],
        )


def test_combining_a_corpus_twice_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["freeze_corpus.py", "a", "b", "--corpus=m", "--all"],
    )
    assert freeze_corpus.main() == 0
    corpus = tmp_path / "a" / "corpora" / "m.json"
    with pytest.raises(ValueError, match="named twice"):
        freeze_corpus.combine_corpora([corpus, corpus])


def _mixture(root: Path) -> None:
    """Publish arm 0 over two roots and arms 1-3 under the first."""
    _publish(root / "a", arm=0, worker=0, decisions=[30, 30, 30])
    _publish(root / "b", arm=0, worker=1, decisions=[20, 20])
    _publish(root / "a", split=VALIDATION, arm=0, worker=0, decisions=[3])
    _publish(root / "b", split=VALIDATION, arm=0, worker=1, decisions=[3])
    for arm in (1, 2, 3):
        _publish(root / "a", arm=arm, decisions=[6, 6, 6])
        _publish(root / "a", split=VALIDATION, arm=arm, decisions=[2])


def _publish(
    root: Path,
    *,
    split: int = TRAIN,
    arm: int,
    worker: int = 0,
    decisions: Sequence[int],
    episodes: Sequence[int] = (),
    rollout_seed: str = "",
) -> None:
    """Append manifest lines for shards of the given sizes; no shard files."""
    directory = shard_directory(root, split=split, arm=arm, worker=worker)
    directory.mkdir(parents=True, exist_ok=True)
    counts = zip(decisions, episodes or [1] * len(decisions), strict=True)
    with (directory / "MANIFEST.jsonl").open("a") as manifest:
        for index, (count, episode_count) in enumerate(counts):
            line = ManifestLine(
                shard=f"shard-{index:06d}",
                episodes=episode_count,
                decisions=count,
                sha256={},
                provenance={"rollout_seed": rollout_seed} if rollout_seed else {},
            )
            manifest.write(json.dumps(to_plain(line)) + "\n")


def _names(root: Path, entries: Sequence[tuple[Path, ManifestLine]]) -> list[str]:
    return [f"{d.relative_to(root)}/{line.shard}" for d, line in entries]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
