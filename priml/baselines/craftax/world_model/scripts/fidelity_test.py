"""Check the fidelity suite: fixed spans, rule counts, HUD dynamics, and the report."""

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import dataclasses
import functools
import json
import math
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
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.dream import Rollout
from priml.baselines.craftax.world_model.engine import Engine, Prefix
from priml.baselines.craftax.world_model.experiments import exp_smoke
from priml.baselines.craftax.world_model.metric import MODALITIES
from priml.baselines.craftax.world_model.model import WorldModelLogits
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts import fidelity
from priml.baselines.craftax.world_model.scripts.dream_eval import (
    Step,
    choose_windows,
    read_source,
    rollout,
    sampling_engine,
    window,
)
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    small_schema,
    tiny_model,
)
from priml.lib.codec import from_plain, loads


FIELDS = craftax_schema().scalar_names


def test_score_span_scores_the_same_targets_at_any_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = small_schema()
    model = tiny_model(schema)
    windows: list[PackedBatch] = []
    logits = model.logits

    def spy(batch: PackedBatch) -> WorldModelLogits:
        windows.append(batch)
        return logits(batch)

    monkeypatch.setattr(model, "logits", spy)
    episode = _episode(random_segment(schema, 12, seed=0, terminal=True), index=0)
    # A 16-position window holds 4 decisions of history, a 32-position one all 6
    # and the episode's start.
    for t_g, decisions, starts in ((16, 7, 0), (32, 9, 1)):
        sums = fidelity.score_span(model, episode, start=6, targets=3, t_g=t_g, s_max=4)
        assert _column(sums, "decisions") == 3, t_g
        assert _column(sums, "frames") == 3, t_g
        assert _column(sums, "cells") == 3 * schema.cell_slots, t_g
        assert _column(sums, "board") > 0, t_g
        assert int((~windows[-1].job_is_start).sum()) == decisions, t_g
        assert int(windows[-1].job_is_start.sum()) == starts, t_g


@pytest.mark.parametrize("origin", [b"", b"branch state"])
def test_score_span_sums_the_models_own_target_nll(origin: bytes) -> None:
    schema = small_schema()
    model = tiny_model(schema)
    real = random_segment(schema, 8, seed=1, terminal=True)
    sums = fidelity.score_span(
        model,
        dataclasses.replace(_episode(real, index=0), origin=origin),
        start=0,
        targets=5,
        t_g=16,
        s_max=4,
    )
    # From the episode start every scored target is one of the five decisions,
    # except the start job's first frame, which is history. A branch's first
    # decision continues its parent's episode, as training packs it: no start.
    segment = Segment(
        cells=real.cells[:6],
        aux=real.aux[:6],
        actions=real.actions[:5],
        reward=real.reward[:5],
        done=real.done[:5],
        starts_episode=not origin,
    )
    batch = pack_windows([[segment]], t_g=16, s_max=4)
    with torch.no_grad():
        terms = model.target_terms(batch, model.logits(batch))
    acts = ~batch.job_is_start
    expected = {
        "action": terms["action"][0].nll[terms["action"][1]].sum(),
        "reward": terms["reward"][0].nll[acts].sum(),
        "board": terms["board"][0].nll[acts].sum(),
        "hud": terms["hud"][0].nll[acts].sum(),
    }
    for name, value in expected.items():
        assert _column(sums, name) == pytest.approx(float(value), rel=1e-5), name
    assert _column(sums, "decisions") == 5


def test_score_span_counts_the_cells_a_copy_of_the_frame_gets_right() -> None:
    schema = small_schema()
    model = tiny_model(schema)
    moving = random_segment(schema, 8, seed=3, terminal=True)
    still = dataclasses.replace(
        moving,
        cells=moving.cells[:1].expand(8, -1, -1).clone(),
        aux=moving.aux[:1].expand(8, -1).clone(),
    )
    cells = 3 * schema.cell_slots
    for segment, copied in ((still, cells), (moving, 0)):
        sums = fidelity.score_span(
            model,
            _episode(segment, index=0),
            start=2,
            targets=3,
            t_g=32,
            s_max=4,
        )
        assert _column(sums, "cells") == cells
        assert _column(sums, "copy_correct") == copied
        assert 0 <= _column(sums, "model_correct") <= cells


def test_score_span_refuses_targets_that_do_not_fit_the_window() -> None:
    schema = small_schema()
    episode = _episode(random_segment(schema, 12, seed=0, terminal=True), index=0)
    with pytest.raises(ValueError, match="do not fit"):
        fidelity.score_span(
            tiny_model(schema),
            episode,
            start=0,
            targets=8,
            t_g=16,
            s_max=4,
        )


def test_score_span_stops_scoring_frames_at_the_death() -> None:
    schema = small_schema()
    model = tiny_model(schema)
    episode = _episode(random_segment(schema, 8, seed=2, terminal=True), index=0)
    sums = fidelity.score_span(model, episode, start=5, targets=3, t_g=32, s_max=4)
    assert _column(sums, "decisions") == 3
    assert _column(sums, "frames") == 2


def test_cell_accuracy_scores_the_argmax_against_the_next_frame() -> None:
    schema = small_schema()
    model = tiny_model(schema)
    batch = pack_windows([[random_segment(schema, 4, seed=3)]], t_g=16, s_max=4)
    jobs = batch.job_next >= 0
    # Logits whose per-field argmax is every job's next frame, which differs
    # from the frame it reads in every cell.
    upcoming = batch.cells[batch.job_next.clamp(min=0).long()].long()
    ids = model.cell_index[torch.arange(len(schema.cell_fields)), upcoming]
    local = torch.zeros(len(jobs), schema.local_slots, schema.vocab_size)
    prefix = len(schema.prefix_ranges)
    local[:, prefix : prefix + schema.cell_slots].scatter_(-1, ids, 1.0)
    cells, model_correct, copy_correct = fidelity._cell_accuracy(
        model,
        batch,
        local,
        jobs=jobs,
    )
    assert int(cells) == 4 * schema.cell_slots
    assert int(model_correct) == int(cells)
    assert int(copy_correct) == 0


def test_teacher_forced_pools_spans_into_rates_and_bits_per_byte(
    tmp_path: Path,
) -> None:
    model = tiny_model(craftax_schema())
    sources = fidelity.validation_sources(validation_corpus(tmp_path, lengths=(9, 12)))
    spans = [
        fidelity.Span(source=sources[0], start=2, targets=3),
        fidelity.Span(source=sources[1], start=0, targets=4),
    ]
    result = fidelity.teacher_forced(
        model,
        spans,
        t_g=32,
        s_max=4,
        resamples=16,
        generator=torch.Generator().manual_seed(0),
    )
    assert from_plain(result["decisions"], int) == 7
    assert from_plain(result["frames"], int) == 7
    sums = [
        fidelity.score_span(
            model,
            read_source(span.source),
            start=span.start,
            targets=span.targets,
            t_g=32,
            s_max=4,
        )
        for span in spans
    ]
    modality = {name: sum(_column(s, name) for s in sums) for name in MODALITIES}
    total = sum(modality.values())
    nats = from_plain(result["nats_per_decision"], dict[str, object])
    assert from_plain(nats["value"], float) == pytest.approx(total / 7)
    bpb = from_plain(result["bpb"], dict[str, object])
    # 4 bytes of action, reward, and done per decision, 894 per frame.
    expected = total / (math.log(2) * (4 * 7 + 894 * 7))
    assert from_plain(bpb["value"], float) == pytest.approx(expected)
    assert list(from_plain(result["hud_fields"], dict[str, object])) == list(FIELDS)
    per_span = [
        from_plain(r, dict[str, object])
        for r in from_plain(result["per_span"], list[object])
    ]
    assert [r["targets"] for r in per_span] == [3, 4]
    assert [r["episode_decisions"] for r in per_span] == [9, 12]
    assert sum(
        from_plain(r["nats_per_decision"], float) * t
        for r, t in zip(per_span, (3, 4), strict=True)
    ) == pytest.approx(total)
    # Each span's modalities, so two reports can be compared span by span.
    for name in MODALITIES:
        assert sum(
            from_plain(from_plain(r["modalities"], dict[str, object])[name], float) * t
            for r, t in zip(per_span, (3, 4), strict=True)
        ) == pytest.approx(modality[name]), name


def test_validation_sources_read_only_the_validation_split(tmp_path: Path) -> None:
    sources = fidelity.validation_sources(
        validation_corpus(tmp_path, lengths=(10, 400)),
    )
    assert [s.summary.decisions for s in sources] == [10, 400]
    assert {s.summary.receipt.split for s in sources} == {1}
    assert sources[1].name == "val/arm3/w0/shard-000000#1"


def test_validation_sources_of_a_replay_corpus_read_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = validation_corpus(tmp_path / "frames", lengths=(10, 400))
    replay_twin(tmp_path / "frames", tmp_path / "replay", monkeypatch)
    replayed = fidelity.validation_sources(
        tmp_path / "replay" / "corpora" / "test.json",
    )
    expected = fidelity.validation_sources(frames)
    assert [s.name for s in replayed] == [s.name for s in expected]
    for source, frame_source in zip(replayed, expected, strict=True):
        episode = read_source(source)
        want = read_source(frame_source)
        for name in ("actions", "cells", "aux", "reward", "done"):
            assert torch.equal(
                cast("torch.Tensor", getattr(episode, name)),
                cast("torch.Tensor", getattr(want, name)),
            ), name


def test_choose_spans_draws_episodes_by_length_and_fixes_them_by_seed(
    tmp_path: Path,
) -> None:
    sources = fidelity.validation_sources(
        validation_corpus(tmp_path, lengths=(10, 400)),
    )
    spans, again = (
        fidelity.choose_spans(
            sources,
            count=64,
            targets=20,
            generator=torch.Generator().manual_seed(0),
        )
        for _ in range(2)
    )
    assert spans == again
    lengths = [s.source.summary.decisions for s in spans]
    # Decision-weighted: the 400-decision episode holds 40 of every 41 decisions.
    assert lengths.count(400) >= 56
    for span in spans:
        assert span.start + span.targets <= span.source.summary.decisions
        assert span.targets == min(20, span.source.summary.decisions)


def test_counts_find_no_violation_in_played_frames() -> None:
    segment = _played(terminal=True)
    for name in fidelity.FIDELITY_CHECKS:
        assert _count(segment, name) == 0, name
    assert _count(segment, "deaths") == 1
    assert _count(segment, "reward") == 4


def test_counts_bound_each_health_rise_by_its_action() -> None:
    base = _with_aux(_played(terminal=False), 3, health=50)
    # Decision 3 moves left: only regeneration, one HP, can heal it.
    assert _count(_with_aux(base, 4, health=70), "health_rise_over_bound") == 0
    assert _count(_with_aux(base, 4, health=90), "health_rise_over_bound") == 1
    # A red potion adds 8 HP, and regeneration 1 more, in the same tick.
    potion = dataclasses.replace(
        base,
        actions=torch.tensor([5, 18, 11, 29], dtype=torch.uint8),
    )
    assert _count(_with_aux(potion, 4, health=230), "health_rise_over_bound") == 0
    assert _count(_with_aux(potion, 4, health=231), "health_rise_over_bound") == 1
    # Resting repeats ticks within one decision until health is full.
    rest = dataclasses.replace(
        base,
        actions=torch.tensor([5, 18, 11, 17], dtype=torch.uint8),
    )
    assert _count(_with_aux(rest, 4, health=180), "health_rise_over_bound") == 0


def test_counts_flag_a_living_frame_at_zero_health() -> None:
    segment = _with_aux(_played(terminal=False), 4, health=0)
    assert _count(segment, "alive_at_zero_health") == 1
    assert _count(_played(terminal=False), "alive_at_zero_health") == 0


def test_counts_allow_only_one_floor_per_ladder() -> None:
    segment = _played(terminal=False)
    for frame in (2, 3, 4):
        segment = _with_aux(segment, frame, floor=2)
    # Descending from floor 0 to 2 uses a ladder action but skips a floor.
    assert _count(segment, "floor_change_without_ladder") == 0
    assert _count(segment, "floor_change_invalid") == 1


def test_counts_bound_the_return_by_the_budget() -> None:
    segment = dataclasses.replace(
        _played(terminal=False),
        reward=torch.tensor([200, 30, 10, 0], dtype=torch.int16),
    )
    assert _count(segment, "reward_over_budget") == 1
    assert _count(segment, "reward") == 240
    # A -1 earns nothing back: 200 and 35 of positive reward exceed the budget.
    dipped = dataclasses.replace(
        segment,
        reward=torch.tensor([200, -1, 35, 0], dtype=torch.int16),
    )
    assert _count(dipped, "reward_over_budget") == 1
    # Earning exactly the budget is allowed.
    full = dataclasses.replace(
        segment,
        reward=torch.tensor([200, 34, 0, 0], dtype=torch.int16),
    )
    assert _count(full, "reward_over_budget") == 0


def test_counts_tally_need_decrements() -> None:
    segment = _with_aux(_played(terminal=False), 4, food=8, drink=7)
    assert _count(segment, "food_down") == 1
    assert _count(segment, "drink_down") == 1
    assert _count(segment, "energy_down") == 0


def test_counts_tell_a_death_from_a_timeout() -> None:
    dying = _played(terminal=True)
    timeout = dataclasses.replace(
        dying,
        reward=torch.tensor([1, 2, 1, 0], dtype=torch.int16),
    )
    assert _count(dying, "deaths") == 1
    assert _count(timeout, "deaths") == 0


def test_counts_tie_health_loss_and_deaths_to_needs_at_zero() -> None:
    # Health drops from frame 2 to 3, then regenerates one HP.
    hurt = _with_aux(_played(terminal=False), 3, health=160)
    assert _count(hurt, "health_down") == 1
    assert _count(hurt, "health_down_at_zero_need") == 0
    # Food at zero in the frame before the loss, not only in the frame after.
    starving = _with_aux(_with_aux(hurt, 2, food=0), 3, food=0)
    assert _count(starving, "health_down") == 1
    assert _count(starving, "health_down_at_zero_need") == 1
    assert _count(_with_aux(hurt, 3, food=0), "health_down_at_zero_need") == 0
    # The dying decision observes frame 3.
    dying = _played(terminal=True)
    assert _count(dying, "deaths_at_zero_need") == 0
    assert _count(_with_aux(dying, 3, drink=0), "deaths_at_zero_need") == 1
    assert _count(_with_aux(dying, 2, energy=0), "deaths_at_zero_need") == 0


def test_segments_split_a_row_at_every_terminal() -> None:
    real = _played(terminal=False)
    # The row continues a prefilled episode, then starts a new one.
    rollout = Rollout(
        cells=real.cells[None],
        aux=real.aux[None],
        starts=torch.tensor([[False, False, True, False, False]]),
        frame_logp=torch.zeros(1, 5, 150),
        invalid=torch.zeros(1, 5, 99, dtype=torch.bool),
        action=real.actions[None],
        reward=real.reward[None],
        done=torch.tensor([[False, True, False, False]]),
        action_logp=torch.zeros(1, 4),
        reward_logp=torch.zeros(1, 4),
        done_logp=torch.zeros(1, 4),
    )
    first, second = fidelity.segments(rollout, row=0)
    assert first.actions.tolist() == [5, 18]
    assert len(first.cells) == 2
    assert first.done.tolist() == [False, True]
    assert not first.starts_episode
    assert second.actions.tolist() == [11, 1]
    assert torch.equal(second.cells, real.cells[2:])
    assert second.starts_episode


def test_rates_are_per_thousand_decisions_pooled_over_units() -> None:
    result = fidelity.rates(
        torch.tensor([[1.0, 0.0], [3.0, 2.0]]),
        torch.tensor([100.0, 300.0]),
        names=("a", "b"),
        resamples=64,
        generator=torch.Generator().manual_seed(0),
    )
    a, b = (
        from_plain(result["a"], dict[str, object]),
        from_plain(result["b"], dict[str, object]),
    )
    assert from_plain(a["value"], float) == pytest.approx(10.0)
    assert from_plain(b["value"], float) == pytest.approx(5.0)
    assert from_plain(a["low"], float) <= 10.0 <= from_plain(a["high"], float)
    # One unit with no decisions, under the one name: its rate has no value.
    empty = fidelity.rates(
        torch.zeros(1, 1),
        torch.zeros(1),
        names=("a",),
        resamples=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert from_plain(empty["a"], dict[str, object])["value"] is None


def test_first_divergence_takes_the_median_with_never_last() -> None:
    # Pairs that never diverge rank last; an even count takes the lower median.
    firsts = {"frame": [3, None, 1], "board": [None, 2, None, 4], "hud": []}
    summary = fidelity._first_divergence({"first_divergence": firsts})
    frame = from_plain(summary["frame"], dict[str, object])
    assert frame["median"] == 3
    assert frame["diverged"] == pytest.approx(2 / 3)
    board = from_plain(summary["board"], dict[str, object])
    assert board["median"] == 4
    assert board["diverged"] == pytest.approx(1 / 2)
    assert from_plain(summary["hud"], dict[str, object]) == {
        "median": None,
        "diverged": 0.0,
    }


def test_frozen_repeats_the_first_frame_for_every_decision() -> None:
    real = _played(terminal=False)
    still = fidelity.frozen(real)
    assert still.cells.shape == real.cells.shape
    assert bool((still.cells == real.cells[0]).all())
    assert bool((still.aux == real.aux[0]).all())
    assert torch.equal(still.actions, real.actions)
    assert torch.equal(still.done, real.done)


def test_hud_dynamics_compare_decay_with_the_real_continuation() -> None:
    real = _played(terminal=False)
    for frame, food in enumerate((9, 9, 8, 8, 8)):
        real = _with_aux(real, frame, food=food)
    dreamed = real
    for frame, food in enumerate((9, 8, 7, 7, 6)):
        dreamed = _with_aux(dreamed, frame, food=food)
    result = fidelity.hud_dynamics(
        [dreamed],
        [real],
        checkpoints=(1, 4),
        resamples=16,
        generator=torch.Generator().manual_seed(0),
    )
    food = from_plain(result["food"], dict[str, object])
    down = from_plain(food["decrements_per_1000"], dict[str, object])
    assert from_plain(down["model"], dict[str, object])["value"] == pytest.approx(750.0)
    assert from_plain(down["real"], dict[str, object])["value"] == pytest.approx(250.0)
    assert from_plain(down["difference"], dict[str, object])["value"] == pytest.approx(
        500.0,
    )
    error = from_plain(food["abs_error"], dict[str, object])
    at_four = from_plain(error["4"], dict[str, object])
    assert from_plain(at_four["model"], dict[str, object])["value"] == pytest.approx(
        2.0,
    )
    assert from_plain(at_four["frozen"], dict[str, object])["value"] == pytest.approx(
        1.0,
    )
    at_one = from_plain(error["1"], dict[str, object])
    assert from_plain(at_one["frozen"], dict[str, object])["value"] == pytest.approx(
        0.0,
    )
    first = from_plain(food["first_divergence"], dict[str, object])
    assert from_plain(first["model"], dict[str, object]) == {
        "median": 1,
        "diverged": 1.0,
    }
    assert from_plain(first["frozen"], dict[str, object]) == {
        "median": 2,
        "diverged": 1.0,
    }


def test_hud_dynamics_score_an_ended_continuation_at_its_worst() -> None:
    real = _played(terminal=False)
    # This continuation dies in decision 1, so it holds frames 0-1 of real's 0-4.
    ended = Segment(
        cells=real.cells[:2],
        aux=real.aux[:2],
        actions=real.actions[:2],
        reward=torch.tensor([1, -1], dtype=torch.int16),
        done=torch.tensor([False, True]),
        starts_episode=True,
    )
    result = fidelity.hud_dynamics(
        [ended],
        [real],
        checkpoints=(1, 4),
        resamples=16,
        generator=torch.Generator().manual_seed(0),
    )
    food, health = (
        from_plain(result[name], dict[str, object]) for name in ("food", "health")
    )
    # Food holds 0-17 and health 0-260: the farthest from 9 and 180 are 0 and 0.
    for field, worst in ((food, 9.0), (health, 180.0)):
        at_four = from_plain(
            from_plain(field["abs_error"], dict[str, object])["4"],
            dict[str, object],
        )
        model = from_plain(at_four["model"], dict[str, object])
        assert model["value"] == pytest.approx(worst)
        assert model["n"] == 1
        assert from_plain(at_four["frozen"], dict[str, object])[
            "value"
        ] == pytest.approx(
            0.0,
        )
        first = from_plain(field["first_divergence"], dict[str, object])
        assert from_plain(first["model"], dict[str, object]) == {
            "median": 2,
            "diverged": 1.0,
        }
        assert from_plain(first["frozen"], dict[str, object]) == {
            "median": None,
            "diverged": 0.0,
        }


def test_hud_dynamics_count_changes_over_the_transitions_both_hold() -> None:
    real = _played(terminal=False)
    for frame, food in enumerate((9, 9, 8, 7, 6)):
        real = _with_aux(real, frame, food=food)
    # The continuation dies in decision 1, holding frames 0-1; the real food
    # falls three times after that.
    ended = Segment(
        cells=real.cells[:2],
        aux=real.aux[:2],
        actions=real.actions[:2],
        reward=torch.tensor([1, -1], dtype=torch.int16),
        done=torch.tensor([False, True]),
        starts_episode=True,
    )
    result = fidelity.hud_dynamics(
        [ended],
        [real],
        checkpoints=(1,),
        resamples=16,
        generator=torch.Generator().manual_seed(0),
    )
    down = from_plain(
        from_plain(result["food"], dict[str, object])["decrements_per_1000"],
        dict[str, object],
    )
    for side in ("model", "real", "difference"):
        entry = from_plain(down[side], dict[str, object])
        assert entry["denominator"] == 1, side
        assert entry["value"] == 0.0, side


@pytest.mark.compute_large_fixture
def test_dreams_rate_every_count_per_thousand_decisions(tmp_path: Path) -> None:
    corpus = validation_corpus(tmp_path, lengths=(9, 12, 10, 14, 11, 13), deaths=True)
    sources = fidelity.validation_sources(corpus)
    engine, step = sampling_engine(
        tiny_model(craftax_schema()),
        rows=2,
        t_max=16,
        seed=0,
    )
    settings = fidelity.Settings(rows=2, decisions=3, real_episodes=3, resamples=8)
    result = fidelity._dreams(engine, step, sources=sources, settings=settings)
    dreamed, real = (
        from_plain(result[side], dict[str, object]) for side in ("dream", "real")
    )
    per_row = from_plain(result["per_row"], dict[str, object])
    for name in fidelity.COUNTS:
        entry = from_plain(dreamed[name], dict[str, object])
        # Two rows of three decisions each.
        assert entry["denominator"] == 6, name
        rows = from_plain(per_row[name], list[float])
        assert len(rows) == 2, name
        assert entry["numerator"] == pytest.approx(sum(rows)), name
        assert from_plain(real[name], dict[str, object])["denominator"] == 9, name
    # Real episodes are a seeded draw, not the corpus's first ones.
    names = from_plain(result["real_names"], list[str])
    assert len(names) == 3
    assert set(names) != {s.name for s in sources[:3]}
    again = fidelity._dreams(engine, step, sources=sources, settings=settings)
    assert again["real_names"] == names


@pytest.mark.compute_large_fixture
def test_continuations_score_model_and_frozen_on_the_same_windows(
    tmp_path: Path,
) -> None:
    corpus = validation_corpus(tmp_path, lengths=(9, 12, 10, 14), deaths=True)
    windows = choose_windows(
        fidelity.validation_sources(corpus),
        count=4,
        prefix=2,
        decisions=3,
        generator=torch.Generator().manual_seed(0),
    )
    engine, step = sampling_engine(
        tiny_model(craftax_schema()),
        rows=4,
        t_max=32,
        seed=0,
    )
    settings = fidelity.Settings(rows=4, prefix=2, continuation=3, resamples=8)
    result = fidelity._continuations(engine, step, windows, settings=settings)
    at = functools.partial(fidelity._at, result)
    held = [w.real.cells for w in windows if len(w.real.cells) > 1]
    assert at("model", "board_mismatch", "1", "n") == len(held)
    assert at("frozen", "board_mismatch", "1", "n") == len(held)
    assert at("model", "ended", "1") is not None
    # The frozen frame is frame 0; the real board changes every decision.
    changed = [(cells[1] != cells[0]).any(-1) for cells in held]
    frozen = at("frozen", "board_mismatch", "1", "value")
    assert frozen == pytest.approx(float(torch.cat(changed).double().mean()))
    assert from_plain(frozen, float) > 0


@pytest.mark.compute_large_fixture
def test_continuations_force_the_recorded_actions_then_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forced: list[torch.Tensor] = []

    def spy(
        engine: Engine,
        step: Step,
        *,
        decisions: int,
        prefixes: Sequence[Prefix | None],
        actions: torch.Tensor,
    ) -> Rollout:
        forced.append(actions.clone())
        return rollout(
            engine,
            step,
            decisions=decisions,
            prefixes=prefixes,
            actions=actions,
        )

    monkeypatch.setattr(fidelity, "rollout", spy)
    segment = random_segment(craftax_schema(), 6, seed=4, terminal=True)
    # Three real decisions follow the prefix, then the episode ends.
    cut = window(
        _episode(segment, index=0),
        anchor=1,
        prefix=2,
        decisions=5,
        kind="k",
        name="e",
    )
    engine, step = sampling_engine(
        tiny_model(craftax_schema()),
        rows=2,
        t_max=32,
        seed=0,
    )
    settings = fidelity.Settings(rows=2, prefix=2, continuation=5, resamples=8)
    fidelity._continuations(engine, step, [cut], settings=settings)
    real = cut.real.actions.long().tolist()
    assert len(real) == 3
    assert forced[0][0].tolist() == [*real, 0, 0]
    # The row without a window samples its actions in a new world.
    assert forced[0][1].tolist() == [-1] * 5


def test_flatten_keeps_numbers_under_slash_joined_keys() -> None:
    flat = fidelity.flatten(
        {"a": {"b": 1.0, "c": [1, 2], "d": None, "e": {"f": 2}, "g": True}},
    )
    assert flat == {"a/b": 1.0, "a/e/f": 2.0}


@pytest.mark.compute_large_fixture
def test_measure_reports_every_part_on_a_tiny_model(tmp_path: Path) -> None:
    corpus = validation_corpus(
        tmp_path,
        lengths=(9, 12, 10, 14),
        deaths=True,
        still=True,
    )
    report = fidelity.measure(
        tiny_model(craftax_schema()),
        sources=fidelity.validation_sources(corpus),
        t_g=16,
        s_max=4,
        settings=fidelity.Settings(
            spans=3,
            targets=3,
            rows=4,
            decisions=4,
            real_episodes=4,
            prefix=2,
            continuation=3,
            resamples=16,
        ),
        precision=nullcontext(),
    )
    nll = from_plain(report["teacher_forced"], dict[str, object])
    assert from_plain(nll["decisions"], int) == 9
    real = from_plain(
        from_plain(report["dreams"], dict[str, object])["real"],
        dict[str, object],
    )
    assert set(real) == set(fidelity.COUNTS)
    continuation = from_plain(report["continuations"], dict[str, object])
    keys = ("model", "frozen", "hud", "first_divergence", "violations")
    for key in (*keys, "real_violations"):
        assert key in continuation, key
    summary = from_plain(report["summary"], dict[str, object])
    nats = from_plain(nll["nats_per_decision"], dict[str, object])
    assert summary["nll/nats_per_decision"] == nats["value"]
    at = functools.partial(fidelity._at, report, "continuations")
    for side, key in (("model", "violations"), ("real", "real_violations")):
        for name, stat in (
            ("health_down", "value"),
            ("health_down_at_zero_need", "value"),
            ("deaths", "numerator"),
            ("deaths_at_zero_need", "numerator"),
        ):
            assert summary[f"continuation/{name}/{side}"] == at(key, name, stat), name
    # The real boards never change, so only the model's differ from them.
    assert at("frozen", "board_mismatch", "1", "value") == 0.0
    assert from_plain(at("model", "board_mismatch", "1", "value"), float) > 0
    assert summary["continuation/ended/1"] == at("model", "ended", "1")
    assert len(from_plain(continuation["windows"], list[object])) == 4
    json.dumps(report)


def test_measure_refuses_episodes_too_short_to_continue(tmp_path: Path) -> None:
    corpus = validation_corpus(tmp_path, lengths=(9, 12))
    with pytest.raises(ValueError, match="long enough to continue"):
        fidelity.measure(
            tiny_model(craftax_schema()),
            sources=fidelity.validation_sources(corpus),
            t_g=16,
            s_max=4,
            settings=fidelity.Settings(
                spans=2,
                targets=3,
                rows=2,
                decisions=4,
                real_episodes=2,
                prefix=20,
                continuation=3,
                resamples=16,
            ),
            precision=nullcontext(),
        )


@pytest.mark.compute_large_fixture
def test_main_writes_the_report_of_a_smoke_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = exp_smoke()
    root = tmp_path / str(config.dataset.working_dir).lstrip("/")
    validation_corpus(
        root,
        lengths=(12, 17, 22),
        deaths=True,
        still=True,
        corpus=Path(str(config.dataset.corpus)),
    )
    model = config.step.model.make()
    checkpoint = tmp_path / "runs" / "smoke" / "checkpoints" / "step_00000004.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"step": {"model": model.state_dict()}}, checkpoint)
    output = tmp_path / "fidelity" / "smoke.json"
    settings = ["--spans", "2", "--targets", "4", "--rows", "2", "--decisions", "3"]
    settings += ["--real-episodes", "2", "--prefix", "2", "--continuation", "3"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fidelity.py",
            str(checkpoint),
            *(
                "--experiment",
                "priml.baselines.craftax.world_model.experiments.exp_smoke",
            ),
            *("--override", f"base_dir={tmp_path}", "--device", "cpu"),
            *("--resamples", "8", "--output", str(output)),
            *settings,
        ],
    )
    assert fidelity.main() == 0
    report = from_plain(loads(output.read_text()), dict[str, object])
    assert report["schema"] == fidelity.SCHEMA
    provenance = from_plain(report["provenance"], dict[str, object])
    assert provenance["tag"] == "smoke-step_00000004"
    assert from_plain(provenance["overrides"], list[str]) == [f"base_dir={tmp_path}"]
    summary = from_plain(report["summary"], dict[str, object])
    nats = from_plain(
        from_plain(report["teacher_forced"], dict[str, object])["nats_per_decision"],
        dict[str, object],
    )
    assert summary["nll/nats_per_decision"] == nats["value"]
    assert set(summary) >= {f"dream/{name}" for name in fidelity.HEADLINE}
    for stat in ("decrements_per_1000", "increments_per_1000", "abs_error_1"):
        assert f"continuation/food/{stat}/model" in summary, stat
    at = functools.partial(fidelity._at, report, "continuations")
    # The real boards never change, so only the model's differ from them.
    assert at("frozen", "board_mismatch", "1", "value") == 0.0
    assert summary["continuation/board_mismatch/1/n"] == at(
        "model",
        "board_mismatch",
        "1",
        "n",
    )


def test_main_rejects_an_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fidelity.py", str(tmp_path / "c.pt"), "--output", str(output)],
    )
    with pytest.raises(FileExistsError):
        fidelity.main()


def _column(sums: torch.Tensor, name: str) -> float:
    """Return one named column of ``score_span``'s sums."""
    return float(sums[fidelity.SPAN_COLUMNS.index(name)])


def _count(segment: Segment, name: str) -> int:
    """Return one named entry of ``counts``."""
    return int(fidelity.counts(segment)[fidelity.COUNTS.index(name)])


def _with_aux(segment: Segment, frame: int, **values: int) -> Segment:
    """Return ``segment`` with auxiliary fields of one frame replaced."""
    aux = segment.aux.clone()
    for name, value in values.items():
        aux[frame, FIELDS.index(name)] = value
    return dataclasses.replace(segment, aux=aux)


def _aux_frame(**values: int) -> torch.Tensor:
    """Return one frame's aux values: attributes 1, facing down, the rest 0 unless given."""
    merged = {
        "dexterity": 1,
        "strength": 1,
        "intelligence": 1,
        "facing_down": 1,
        "health": 180,
        "food": 9,
        "drink": 9,
        "energy": 9,
    } | values
    return torch.tensor([merged.get(name, 0) for name in FIELDS], dtype=torch.int16)


def _board(seed: int) -> torch.Tensor:
    """Return a visible 99-cell board with random blocks and no mobs."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.zeros(99, 8, dtype=torch.uint8)
    cells[:, 0] = torch.randint(2, 30, (99,), generator=generator).to(torch.uint8)
    cells[:, 1] = 1
    cells[:, 2] = 1
    return cells


def _shifted_left(cells: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return the view after a move left: every column moves one to the right."""
    grid = cells.view(9, 11, 8).clone()
    grid[:, 1:] = cells.view(9, 11, 8)[:, :-1]
    grid[:, 0] = _board(seed).view(9, 11, 8)[:, 0]
    return grid.view(99, 8)


# The move is consistent with the board; with ``terminal`` the last decision dies,
# otherwise the frame after it exists.
def _played(*, terminal: bool) -> Segment:
    """Return four decisions: collect wood, descend, craft a pickaxe, move left."""
    first = _board(0)
    frames = [first, first, _board(1), _board(1)]
    frames.append(_shifted_left(frames[-1], seed=2))
    aux = [
        _aux_frame(),
        _aux_frame(wood=1),
        _aux_frame(wood=1, floor=1, xp=1),
        _aux_frame(wood=0, pickaxe=1, floor=1, xp=1),
        _aux_frame(pickaxe=1, floor=1, xp=1, facing_down=0, facing_left=1),
    ]
    kept = 4 if terminal else 5
    return Segment(
        cells=torch.stack(frames[:kept]),
        aux=torch.stack(aux[:kept]),
        actions=torch.tensor([5, 18, 11, 1], dtype=torch.uint8),
        reward=torch.tensor([1, 2, 1, -1 if terminal else 0], dtype=torch.int16),
        done=torch.tensor([False, False, False, terminal]),
        starts_episode=True,
    )


def validation_corpus(
    root: Path,
    *,
    lengths: tuple[int, ...],
    deaths: bool = False,
    still: bool = False,
    corpus: Path = Path("corpora/test.json"),
) -> Path:
    """Publish a training and a validation shard of ``lengths``; return the corpus.

    With ``deaths``, every other episode dies; with ``still``, every frame of an
    episode shows its first board.
    """
    entries: list[tuple[Path, ManifestLine]] = []
    for split, name in enumerate(("train", "val")):
        directory = root / name / "arm3" / "w0"
        directory.mkdir(parents=True)
        episodes: list[Episode] = []
        for index, decisions in enumerate(lengths):
            segment = random_segment(
                craftax_schema(),
                decisions,
                seed=10 * split + index,
                terminal=True,
            )
            reward = segment.reward.clamp(min=0)
            reward[-1] = -1 if deaths and index % 2 == 0 else 0
            cells = (
                segment.cells[:1].expand_as(segment.cells) if still else segment.cells
            )
            segment = dataclasses.replace(segment, reward=reward, cells=cells.clone())
            episodes.append(_episode(segment, index=index, split=split))
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    write_corpus(root / corpus, entries=entries)
    return root / corpus


def _episode(segment: Segment, *, index: int, split: int = 1) -> Episode:
    """Return an archived episode of a complete segment."""
    died = bool(segment.done[-1]) and int(segment.reward[-1]) == -1
    return Episode(
        receipt=Receipt(
            world_seed=index + 100 * split,
            sampling_seed=1,
            initial_state_hash=2,
            arm=3,
            split=split,
        ),
        actions=segment.actions,
        hashes=torch.zeros(1, dtype=torch.int64),
        cells=segment.cells,
        aux=segment.aux,
        reward=segment.reward,
        done=segment.done,
        summary={"death": int(died), "achievements": []},
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
