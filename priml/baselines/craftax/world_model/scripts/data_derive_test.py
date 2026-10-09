"""Check that derived corpora draw whole or capped episodes in the mixture shares."""

from functools import partial
from pathlib import Path

import dataclasses
import sys

import pytest
import torch

from priml.baselines.craftax.eager import eager, scripted, tiny_world
from priml.baselines.craftax.game.state import DEFAULT_MAX_TIMESTEPS
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Receipt,
    read_corpus,
    read_manifest,
    read_shard,
    read_summaries,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.data import ReplayStream
from priml.baselines.craftax.world_model.index import FLOOR_AUX, FLOORS
from priml.baselines.craftax.world_model.scripts import data_derive
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import from_plain


def test_each_arm_takes_its_share_of_whole_episodes_from_every_shard(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    entries = data_derive.derive(
        [source],
        tmp_path / "out",
        decisions=60,
        shares=(2, 1, 0, 0),
        max_decisions=0,
        seed=0,
        shard_decisions=25,
    )
    by_arm = _episodes(entries)
    assert sorted(by_arm) == [0, 1]
    for arm, target in ((0, 40), (1, 20)):
        lengths = [len(e.actions) for e in by_arm[arm]]
        assert sum(lengths[:-1]) < target <= sum(lengths)
        assert {e.receipt.arm for e in by_arm[arm]} == {arm}
    seeds = [e.receipt.world_seed for es in by_arm.values() for e in es]
    assert len(seeds) == len(set(seeds))
    # Episodes come from both of arm 0's source shards, not a prefix of one.
    assert {s // 1000 for s in seeds if s < 2000} == {0, 1}
    # Shards close at the first episode boundary after 25 decisions.
    lines = read_manifest(tmp_path / "out" / "train" / "arm0" / "w0")
    assert all(line.decisions - 12 < 25 for line in lines[:-1])


_DECISIONS = 6
"""Decisions the recorded episode plays before the tiny world's clock runs out."""

_BRANCH = 3
"""Where the branch leaves the recorded episode."""


@pytest.fixture(scope="module")
def recorded() -> list[Episode]:
    """Return a six-decision random-play episode and a three-decision branch of it.

    The game's kernels run as Python (``eager``) on the tiny world.
    """
    floors = [{"reached": 0, "decisions": 0, "kills": 0}] * FLOORS
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS)):
        played = replay.record(world_seed=12, sampling_seed=12, max_decisions=1_000)
        parent = dataclasses.replace(
            played,
            summary={
                "floors": [
                    {"reached": 1, "decisions": _DECISIONS, "kills": 3},
                    *floors[1:],
                ],
            },
        )
        # The state before the branch's first decision, by playing to it.
        before = scripted(ints(parent.actions[:_BRANCH].numpy()), world_seed=12)
        branch = dataclasses.replace(
            parent,
            receipt=dataclasses.replace(
                parent.receipt,
                sampling_seed=1,
                initial_state_hash=int(before.hashes[-1]),
            ),
            actions=parent.actions[_BRANCH:],
            hashes=torch.cat([before.hashes[-1:], parent.hashes[-1:]]),
            cells=parent.cells[_BRANCH:],
            aux=parent.aux[_BRANCH:],
            reward=parent.reward[_BRANCH:],
            done=parent.done[_BRANCH:],
            summary={**parent.summary, "branch": {"decision": _BRANCH}},
            origin=replay.origin(parent, decision=_BRANCH),
        )
        assert replay.verify(branch) == replay.MATCHED
    assert [len(e.actions) for e in (parent, branch)] == [_DECISIONS, 3]
    return [parent, branch]


# A cap of 2 cuts both episodes, replaying each prefix, the branch's from its
# origin; one of 4 cuts the parent alone.
@pytest.mark.parametrize("cap", [2, 4])
def test_a_capped_episode_keeps_its_prefix_and_replays_as_truncated(
    tmp_path: Path,
    recorded: list[Episode],
    cap: int,
) -> None:
    directory = shard_directory(tmp_path / "src", split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True)
    write_shard(directory, index=0, episodes=recorded, provenance={})
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS)):
        entries = data_derive.derive(
            [tmp_path / "src"],
            tmp_path / "out",
            decisions=sum(min(len(e.actions), cap) for e in recorded),
            shares=(1, 0, 0, 0),
            max_decisions=cap,
            seed=0,
            shard_decisions=1_000,
        )
        derived = _episodes(entries)[0]
        verified = [replay.verify(episode) for episode in derived]
    assert verified == [replay.MATCHED] * 2
    originals = {e.receipt.sampling_seed: e for e in recorded}
    for episode in derived:
        original = originals[episode.receipt.sampling_seed]
        kept = min(len(original.actions), cap)
        assert torch.equal(episode.actions, original.actions[:kept])
        assert torch.equal(episode.cells, original.cells[:kept])
        assert episode.truncated == (kept < len(original.actions))
        if episode.truncated:
            assert not episode.done.any()
            assert episode.summary["capped_from"] == len(original.actions)
            # Floor records describe the kept decisions; kills keep the whole's.
            floors = from_plain(episode.summary["floors"], list[dict[str, object]])
            visits = [
                int(n) for n in episode.aux[:, FLOOR_AUX].bincount(minlength=FLOORS)
            ]
            assert [f["decisions"] for f in floors] == visits
            assert [f["reached"] for f in floors] == [int(n > 0) for n in visits]
            assert floors[0]["kills"] == 3


def test_a_cap_at_a_multiple_of_256_ends_at_the_hash_the_episode_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unplayed(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("A cap at a held hash replays nothing.")

    monkeypatch.setattr(replay, "origin", unplayed)
    directory = shard_directory(tmp_path / "src", split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True)
    whole = dataclasses.replace(
        _episode(300, seed=1, arm=0, split=TRAIN),
        hashes=torch.tensor([10, 20, 30]),
    )
    write_shard(directory, index=0, episodes=[whole], provenance={})
    entries = data_derive.derive(
        [tmp_path / "src"],
        tmp_path / "out",
        decisions=256,
        shares=(1, 0, 0, 0),
        max_decisions=256,
        seed=0,
        shard_decisions=1_000,
    )
    (cut,) = _episodes(entries)[0]
    # The hash before decision 0, then the one before 256, after the last kept.
    assert cut.hashes.tolist() == [10, 20]
    assert (len(cut.actions), cut.truncated) == (256, True)


def test_an_archive_root_pools_every_published_training_episode(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    entries = data_derive.derive(
        [tmp_path / "src"],
        tmp_path / "out",
        decisions=138,
        shares=(2, 1, 0, 0),
        max_decisions=0,
        seed=0,
        shard_decisions=100,
        validation=source,
    )
    by_arm = _episodes(entries)
    # The root holds all 12 episodes of arm 0 and all 6 of arm 1.
    assert [len(by_arm[arm]) for arm in (0, 1)] == [12, 6]


def test_the_seed_fixes_the_draw(tmp_path: Path) -> None:
    source = _source(tmp_path)
    draws = [
        [
            e.receipt.world_seed
            for e in _episodes(
                data_derive.derive(
                    [source],
                    tmp_path / f"out{i}",
                    decisions=20,
                    shares=(1, 0, 0, 0),
                    max_decisions=0,
                    seed=seed,
                    shard_decisions=100,
                ),
            )[0]
        ]
        for i, seed in enumerate((0, 0, 1))
    ]
    assert draws[0] == draws[1]
    assert draws[0] != draws[2]


def test_several_sources_pool_into_one_draw_each_shard_once(tmp_path: Path) -> None:
    corpus = _source(tmp_path)
    _source(tmp_path / "more", base=10_000)
    entries = data_derive.derive(
        # The corpus lists shards of the first root: they are drawn once.
        [tmp_path / "src", corpus, tmp_path / "more" / "src"],
        tmp_path / "out",
        decisions=184,
        shares=(1, 1, 0, 0),
        max_decisions=0,
        seed=0,
        shard_decisions=100,
    )
    # Arm 1 needs 92 decisions: all 6 of its episodes in each source.
    seeds = [e.receipt.world_seed for e in _episodes(entries)[1]]
    assert sorted(s // 10_000 for s in seeds) == [0] * 6 + [1] * 6
    validation = [e for e in entries if e[0].parent.parent.name == "val"]
    assert len(validation) == 4


def test_a_larger_draw_extends_a_smaller_one(tmp_path: Path) -> None:
    source = _source(tmp_path)
    small, large = (
        [
            e.receipt.world_seed
            for e in _episodes(
                data_derive.derive(
                    [source],
                    tmp_path / f"out{decisions}",
                    decisions=decisions,
                    shares=(1, 0, 0, 0),
                    max_decisions=0,
                    seed=0,
                    shard_decisions=100,
                ),
            )[0]
        ]
        for decisions in (20, 80)
    )
    assert len(small) < len(large)
    assert large[: len(small)] == small


def test_each_branch_of_a_world_is_ranked_by_itself(tmp_path: Path) -> None:
    directory = shard_directory(tmp_path / "src", split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True)
    parent = _episode(4, seed=100, arm=0, split=TRAIN)
    # Branches share their parent's world seed; each has its own sampling seed.
    branches = [
        dataclasses.replace(
            parent,
            receipt=dataclasses.replace(parent.receipt, sampling_seed=1 + b),
            summary={**parent.summary, "branch": {"decision": b}},
        )
        for b in range(6)
    ]
    fresh = [_episode(4, seed=100 + w, arm=0, split=TRAIN) for w in range(6)]
    write_shard(directory, index=0, episodes=[*fresh, *branches], provenance={})
    entries = data_derive.derive(
        [tmp_path / "src"],
        tmp_path / "out",
        decisions=48,
        shares=(1, 0, 0, 0),
        max_decisions=0,
        seed=0,
        shard_decisions=100,
    )
    drawn = ["branch" in e.summary for e in _episodes(entries)[0]]
    assert drawn.count(True) == 6
    # Ranked by their shared world seed alone, they would tie and come as a block.
    first = drawn.index(True)
    assert drawn[first : first + 6] != [True] * 6


def test_replay_sources_derive_the_frame_sources_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    replay_twin(tmp_path / "src", tmp_path / "replay", monkeypatch)
    replayed = tmp_path / "replay" / "corpora" / "source.json"
    # Whole episodes: a cut replays its prefix for its last hash, and no game
    # replays these synthetic records.
    derived = [
        data_derive.derive(
            [corpus],
            tmp_path / out,
            decisions=60,
            shares=(2, 1, 0, 0),
            max_decisions=0,
            seed=0,
            shard_decisions=25,
        )
        for corpus, out in ((source, "from-frames"), (replayed, "from-replay"))
    ]
    written = [
        [
            line.sha256
            for directory, line in entries
            if directory.parent.parent.name == "train"
        ]
        for entries in derived
    ]
    assert written[0]
    assert written[0] == written[1]


def test_main_writes_a_corpus_the_loader_reads_with_the_sources_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    out = tmp_path / "out"
    # OUT as the shell may spell it: the corpus must still read from anywhere.
    monkeypatch.chdir(tmp_path)
    argv = ["data_derive.py", str(source), "out", "--name", "d"]
    argv += ["--decisions", "30", "--shares", "1,1,0,0"]
    monkeypatch.setattr(sys, "argv", argv)
    assert data_derive.main() == 0
    corpus = out / "corpora" / "d.json"
    entries = read_corpus(corpus)
    assert all(directory.is_absolute() for directory, _ in entries)
    validation = [e for e in read_corpus(source) if e[0].parent.parent.name == "val"]
    assert [e for e in entries if e[0].parent.parent.name == "val"] == validation
    config = ReplayStream.Config(
        working_dir=out,
        corpus="corpora/d.json",
        t_g=16,
        s_max=4,
        device="cpu",
    )
    stream = config.finalize().make()
    assert int(stream.train_sampler.counts.sum()) >= 30
    with pytest.raises(FileExistsError):
        data_derive.main()


# World seeds start at ``base``, so sources built with different bases differ.
def _source(root: Path, *, base: int = 0) -> Path:
    """Freeze a source corpus: arm 0 in two shards, arm 1 in one, and validation."""
    entries: list[tuple[Path, ManifestLine]] = []
    for arm, shards in ((0, 2), (1, 1)):
        directory = shard_directory(root / "src", split=TRAIN, arm=arm, worker=0)
        directory.mkdir(parents=True)
        for index in range(shards):
            first = base + 1000 * (index + 2 * arm)
            episodes = [
                _episode(n, seed=first + i, arm=arm, split=TRAIN)
                for i, n in enumerate([3, 12, 7, 9, 4, 11])
            ]
            line = write_shard(directory, index=index, episodes=episodes, provenance={})
            entries.append((directory, line))
        directory = shard_directory(root / "src", split=VALIDATION, arm=arm, worker=0)
        directory.mkdir(parents=True)
        episodes = [_episode(6, seed=base + 9000 + arm, arm=arm, split=VALIDATION)]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    path = root / "src" / "corpora" / "source.json"
    write_corpus(path, entries=entries)
    return path


def _episode(decisions: int, *, seed: int, arm: int, split: int) -> Episode:
    """Build an episode that ends in a terminal decision, frames numbered."""
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    numbers = (torch.arange(decisions) % 256).to(torch.uint8)
    cells = torch.zeros(decisions, 99, 8, dtype=torch.uint8)
    cells[:, 0, 0] = numbers
    aux = torch.ones(decisions, 51, dtype=torch.int16)
    # Three decisions on floor 0, the rest on floor 1.
    aux[:, 48] = (torch.arange(decisions) >= 3).to(torch.int16)
    floors = [
        {"reached": 1, "decisions": min(decisions, 3), "kills": 0},
        {"reached": int(decisions > 3), "decisions": max(decisions - 3, 0), "kills": 4},
        *[{"reached": 0, "decisions": 0, "kills": 0}] * 7,
    ]
    return Episode(
        receipt=Receipt(
            world_seed=seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=arm,
            split=split,
        ),
        actions=numbers,
        hashes=torch.zeros((decisions + 255) // 256 + 1, dtype=torch.int64),
        cells=cells,
        aux=aux,
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={"timeout": 0, "floors": floors},
    )


def _episodes(entries: list[tuple[Path, ManifestLine]]) -> dict[int, list[Episode]]:
    """Return the derived training episodes by arm, in written order."""
    by_arm: dict[int, list[Episode]] = {}
    for directory, line in entries:
        if read_summaries(directory, line)[0].receipt.split == TRAIN:
            for episode in read_shard(directory, line):
                by_arm.setdefault(episode.receipt.arm, []).append(episode)
    return by_arm


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
