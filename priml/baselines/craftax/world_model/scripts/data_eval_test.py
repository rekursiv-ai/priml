"""Check that validation tiles score each decision once and the report splits NLL."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

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
from priml.baselines.craftax.world_model.batch import PackedBatch, pack
from priml.baselines.craftax.world_model.data import EpisodeCache, window
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts import (
    data_eval,
    data_stats,
)
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import PlainTree, from_plain, loads


if TYPE_CHECKING:
    from priml.baselines.craftax.world_model.model import WorldModel


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


def test_evaluate_reports_natural_metrics_strata_and_classes(
    tmp_path: Path,
    scored: None,
) -> None:
    del scored
    result = _evaluate(_smoke_archive(tmp_path) / "corpora" / "smoke.json")
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


def test_a_replay_corpus_scores_as_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scored: None,
) -> None:
    del scored
    archive = _smoke_archive(tmp_path)
    replay_twin(archive, tmp_path / "replay", monkeypatch)
    results = [
        _evaluate(root / "corpora" / "smoke.json")
        for root in (archive, tmp_path / "replay")
    ]
    assert from_plain(results[0]["decisions"], int) == 42
    assert results[0] == results[1]


@pytest.mark.parametrize(
    ("flags", "corpus", "t_g"),
    [((), "corpora/smoke.json", 32), (("--corpus=c.json", "--t-g=8"), "c.json", 8)],
)
def test_main_scores_the_runs_corpus_and_writes_the_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: tuple[str, ...],
    corpus: str,
    t_g: int,
) -> None:
    model = torch.nn.Linear(1, 1)
    dataset = SimpleNamespace(
        corpus="corpora/smoke.json",
        t_g=32,
        s_max=4,
        cached_decisions=77,
    )
    config = SimpleNamespace(dataset=dataset, step=SimpleNamespace(dtype_autocast=None))
    loaded: list[tuple[str, Path, list[str]]] = []
    evaluated: list[tuple[Path, int, int, int, int]] = []

    def load_trained(
        experiment: str,
        checkpoint: Path,
        *,
        overrides: list[str],
        device: torch.device,
    ) -> tuple[torch.nn.Module, object]:
        assert device == torch.device("cpu")
        loaded.append((experiment, checkpoint, overrides))
        return model, config

    def evaluate(
        given: torch.nn.Module,
        scored: Path,
        *,
        t_g: int,
        s_max: int,
        count: int,
        seed: int,
        cached_decisions: int,
    ) -> tuple[dict[str, PlainTree], int]:
        assert (given, cached_decisions) == (model, 77)
        evaluated.append((scored, t_g, s_max, count, seed))
        return {"tiles": count}, 9

    monkeypatch.setattr(data_eval, "load_trained", load_trained)
    monkeypatch.setattr(data_eval, "evaluate", evaluate)
    output = tmp_path / "out" / "eval.json"
    argv = ["data_eval.py", str(tmp_path / "step.pt"), "--experiment", SMOKE]
    argv += ["--override", "base_dir=/b", "--device", "cpu", *flags]
    argv += ["--tiles", "5", "--seed", "3", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert data_eval.main() == 0
    assert loaded == [(SMOKE, tmp_path / "step.pt", ["base_dir=/b"])]
    assert evaluated == [(Path(corpus), t_g, 4, 5, 3)]
    result = from_plain(loads(output.read_text()), dict[str, object])
    run = from_plain(result.pop("run"), dict[str, object])
    assert result == {"tiles": 5}
    assert from_plain(run.pop("seconds"), float) >= 0
    assert run == {
        "checkpoint": str(tmp_path / "step.pt"),
        "experiment": SMOKE,
        "overrides": ["base_dir=/b"],
        "corpus": corpus,
        "t_g": t_g,
        "seed": 3,
        "tiles_available": 9,
    }


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


@pytest.fixture
def scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score each tile with :func:`_nll` in place of a model's."""
    monkeypatch.setattr(data_eval, "craftax_target_nll", _nll)


# The model's own NLL has its tests; the report's sums need only some.
def _nll(model: torch.nn.Module, media: object) -> torch.Tensor:
    """Stand in for ``craftax_target_nll``: each job's record, a rising NLL per slot."""
    del model
    assert isinstance(media, PackedBatch)
    schema = craftax_schema()
    width = 1 + len(schema.prefix_names) + schema.frame_slots
    return torch.linspace(0.1, 1.0, len(media.job_at) * width)


def _evaluate(corpus: Path) -> dict[str, object]:
    """Score 100 tiles of ``corpus`` as ``exp_smoke`` would: 32 positions, 4 segments."""
    report, _ = data_eval.evaluate(
        cast("WorldModel", torch.nn.Linear(1, 1)),
        corpus,
        t_g=32,
        s_max=4,
        count=100,
        seed=0,
        cached_decisions=10_000,
    )
    return from_plain(report, dict[str, object])


def _smoke_archive(root: Path) -> Path:
    """Publish a corpus of one training and one validation shard; return its root."""
    archive = root / "archive"
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
