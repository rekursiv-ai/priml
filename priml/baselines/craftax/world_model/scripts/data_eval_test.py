"""Check that validation tiles score each decision once and the report splits NLL."""

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
from priml.baselines.craftax.world_model.batch import pack
from priml.baselines.craftax.world_model.data import EpisodeCache, window
from priml.baselines.craftax.world_model.experiments import exp_smoke
from priml.baselines.craftax.world_model.scripts import (
    data_eval,
    data_stats,
)
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import from_plain, loads


SMOKE: str = "priml.baselines.craftax.world_model.experiments.exp_smoke"


@pytest.mark.parametrize(("t_g", "s_max"), [(8, 3), (9, 2), (32, 64)])
def test_tiles_score_every_action_and_first_frame_exactly_once(
    tmp_path: Path,
    t_g: int,
    s_max: int,
) -> None:
    lengths = [3, 10, 1, 7, 5]
    first = [sum(lengths[:i]) for i in range(len(lengths))]
    episodes = [_episode(n, offset=o) for n, o in zip(lengths, first, strict=True)]
    entry = _publish(tmp_path, episodes, split=1)
    cache = EpisodeCache([entry], capacity=100)
    scored: list[int] = []
    starts = 0
    for episode, start in data_eval.tiles(lengths, t_g=t_g, s_max=s_max):
        parts = window(
            cache,
            shard=0,
            episode=episode,
            start=start,
            t_g=t_g,
            s_max=s_max,
        )
        batch, _ = pack([parts], t_g=t_g, s_max=s_max)
        act = ~batch.job_is_start
        scored += [int(a) for a in batch.action.flatten()[batch.job_at[act].long()]]
        starts += int(batch.job_is_start.sum())
    # Each action byte is its decision's global index, so each appears once.
    assert sorted(scored) == list(range(sum(lengths)))
    # A window ending on a lone start position scores that first frame again.
    assert len(lengths) <= starts <= 2 * len(lengths)


@pytest.mark.compute_large_fixture
def test_main_reports_natural_metrics_strata_and_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smoke_archive(tmp_path)
    output = tmp_path / "eval.json"
    _run(tmp_path, monkeypatch, output=output)
    result = from_plain(loads(output.read_text()), dict[str, object])
    metric = from_plain(result["metric"], dict[str, object])
    assert from_plain(metric["bpb"], float) > 0
    assert "zstd19_bpb" not in metric
    assert from_plain(result["decisions"], int) == 42
    strata = from_plain(result["strata"], dict[str, object])
    assert (
        from_plain(
            from_plain(strata["floor0/entry"], dict[str, object])["decisions"],
            int,
        )
        > 0
    )
    classes = from_plain(result["classes"], dict[str, object])
    # The 30-decision episode repeats one frame after its first decision.
    repeated = from_plain(classes["arm1/ended/repeated"], dict[str, object])
    assert from_plain(repeated["decisions"], int) == 29
    fresh = from_plain(classes["arm1/ended/fresh"], dict[str, object])
    assert from_plain(fresh["decisions"], int) == 1
    timeout = from_plain(classes["arm1/timeout/fresh"], dict[str, object])
    assert from_plain(timeout["decisions"], int) == 12
    shares = [
        from_plain(from_plain(c, dict[str, object])["share_nats"], float)
        for c in classes.values()
    ]
    assert sum(shares) == pytest.approx(1)


@pytest.mark.compute_large_fixture
def test_a_replay_corpus_scores_as_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _smoke_archive(tmp_path)
    replay_twin(archive, tmp_path / "replay", monkeypatch)
    results: list[dict[str, object]] = []
    for root in (archive, tmp_path / "replay"):
        output = tmp_path / root.name / "eval.json"
        _run(
            tmp_path,
            monkeypatch,
            output=output,
            corpus=root / "corpora" / "smoke.json",
        )
        result = from_plain(loads(output.read_text()), dict[str, object])
        del result["run"]
        results.append(result)
    assert from_plain(results[0]["decisions"], int) == 42
    assert results[0] == results[1]


def test_frame_ids_ignore_light_and_repeats_look_back_a_window() -> None:
    episode = _episode(5, repeat=True)
    episode.aux[3, 43] = 7  # The light level.
    episode.cells[4, 0, 0] = 1
    ids = data_stats.frame_ids(episode).tolist()
    assert ids[:4] == [ids[0]] * 4
    assert ids[4] != ids[0]
    frames = torch.tensor(ids)
    assert data_stats.repeats(frames).tolist() == [False, True, True, True, False]
    assert data_stats.repeats(frames, window=2).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]
    spaced = torch.tensor([5, 6, 7, 5])
    assert data_stats.repeats(spaced, window=2).tolist() == [False] * 4
    assert data_stats.repeats(spaced, window=3).tolist() == [False, False, False, True]


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output: Path,
    corpus: Path | None = None,
) -> None:
    """Score a fresh ``exp_smoke`` model's checkpoint with ``data_eval.main``."""
    torch.manual_seed(0)
    model = exp_smoke().step.model.make()
    torch.save({"step": {"model": model.state_dict()}}, root / "step.pt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_eval.py",
            str(root / "step.pt"),
            *("--experiment", SMOKE),
            *("--override", f"base_dir={root}", "--device", "cpu"),
            *(("--corpus", str(corpus)) if corpus else ()),
            *("--tiles", "100", "--output", str(output)),
        ],
    )
    assert data_eval.main() == 0


def _smoke_archive(root: Path) -> Path:
    """Publish ``exp_smoke``'s corpus: one training and one validation shard."""
    archive = root / str(exp_smoke().dataset.working_dir).lstrip("/")
    train = _publish(archive, [_episode(40, offset=100, split=0)], split=0)
    val = _publish(
        archive,
        [_episode(30, repeat=True), _episode(12, offset=30, timeout=True)],
        split=1,
    )
    write_corpus(archive / "corpora" / "smoke.json", entries=[train, val])
    return archive


def _episode(
    decisions: int,
    *,
    offset: int = 0,
    repeat: bool = False,
    timeout: bool = False,
    split: int = 1,
) -> Episode:
    """Build an episode whose action bytes are its decisions' global indexes."""
    cells = torch.zeros(decisions, 99, 8, dtype=torch.uint8)
    if not repeat:
        cells[:, 0, 0] = torch.arange(decisions, dtype=torch.uint8)
    # Dexterity and its kin start at 1, so an all-ones frame is valid; floor 0.
    aux = torch.ones(decisions, 51, dtype=torch.int16)
    aux[:, 48] = 0
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=offset,
            sampling_seed=0,
            initial_state_hash=0,
            arm=1,
            split=split,
        ),
        actions=torch.arange(offset, offset + decisions, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=cells,
        aux=aux,
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={"timeout": int(timeout)},
    )


def _publish(
    root: Path,
    episodes: list[Episode],
    *,
    split: int,
) -> tuple[Path, ManifestLine]:
    """Publish ``episodes`` as one shard of arm 1, worker 0; return its entry."""
    directory = root / ("train", "val")[split] / "arm1" / "w0"
    directory.mkdir(parents=True)
    line = write_shard(directory, index=0, episodes=episodes, provenance={})
    return directory, line


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
