"""Check which episodes each set takes, in what order, and the quiet-stretch timeline."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.craftax.ghosts.extract import (
    ENDED,
    FLOOR_CHANGED,
    FOUGHT,
    HELD,
    NEW_TILE,
    Ghost,
)
from priml.baselines.craftax.ghosts.layout import (
    Events,
    QuietRule,
    TimeRule,
)
from priml.baselines.craftax.ghosts.sets import (
    Pool,
    activity,
    all_set,
    keep_runs,
    quiet_segments,
    short_set,
    unbroken_set,
    wins_set,
)
from priml.baselines.craftax.lib.arrays import int_rows


def _ghost(ordinal: int, outcome: str, decisions: int, active: bytes = b"") -> Ghost:
    return Ghost(
        ordinal=ordinal,
        world_seed=15,
        sampling_seed=ordinal,
        decisions=decisions,
        outcome=outcome,
        end=(0, 24, 24, 3),
        floor_first=(0, *[-1] * 8),
        achievement_return=0,
        players=bytes(decisions),
        creatures=b"",
        samples=np.zeros(1, np.int64),
        events=Events(
            map=np.zeros((0, 6), np.int64),
            achievements=np.zeros((0, 2), np.int64),
            escapes=np.zeros((0, 5), np.int64),
        ),
        active=active or bytes(decisions),
        sleeps=np.zeros((0, 2), np.int64),
        sleep_samples=np.zeros(1, np.int64),
        sleep_creatures=b"",
        sleep_changes=np.zeros((0, 6), np.int64),
    )


def _pool(*ghosts: Ghost, capped: bool = False) -> Pool:
    return Pool(root="r", capped=capped, provenance={}, ghosts=ghosts)


def _picked(chosen: tuple[tuple[int, Ghost], ...]) -> list[tuple[int, int]]:
    return [(source, ghost.ordinal) for source, ghost in chosen]


def test_all_takes_the_main_captures_first_episodes() -> None:
    main = _pool(*(_ghost(i, "death", 50) for i in range(5)))
    chosen = all_set([main, _pool(_ghost(0, "win", 10))], count=3)
    assert _picked(chosen.episodes) == [(0, 0), (0, 1), (0, 2)]
    assert (chosen.composition.run, chosen.composition.qualified) == (5, 5)
    assert chosen.composition.deaths == 3


def test_short_takes_ends_within_the_cap_by_ordinal_then_capture() -> None:
    main = _pool(
        _ghost(0, "death", 120),
        _ghost(1, "timeout", 90),
        _ghost(2, "death", 99),
        _ghost(3, "death", 100),
    )
    capped = _pool(_ghost(0, "truncated", 100), _ghost(1, "death", 7), capped=True)
    chosen = short_set([main, capped], cap=100, count=10)
    assert _picked(chosen.episodes) == [(1, 1), (0, 2), (0, 3)]
    composition = chosen.composition
    assert (composition.run, composition.qualified) == (6, 3)
    assert (composition.deaths, composition.wins, composition.added_wins) == (3, 0, 0)
    assert composition.death_decisions == (7, 100)
    assert composition.win_decisions == ()
    assert _picked(short_set([main, capped], cap=100, count=2).episodes) == [
        (1, 1),
        (0, 2),
    ]


def test_short_adds_the_shortest_long_wins_up_to_their_natural_share() -> None:
    main = _pool(
        _ghost(0, "death", 10),
        _ghost(1, "death", 20),
        _ghost(2, "win", 500),
        _ghost(3, "death", 30),
        _ghost(4, "win", 300),
        _ghost(5, "win", 400),
        _ghost(6, "death", 900),
        _ghost(7, "timeout", 1_000),
    )
    chosen = short_set([main], cap=100, count=10)
    # Uncapped wins are 3 of 7 ended, so 3 deaths within the cap need
    # ceil(3 * 3 / 4) = 3 wins: all three, the shortest first.
    assert _picked(chosen.episodes) == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    composition = chosen.composition
    assert composition.added_wins == 3
    assert composition.natural_win_share == 3 / 7
    assert composition.win_decisions == (300, 500)
    fewer = short_set(
        [main, _pool(_ghost(9, "death", 5), capped=True)],
        cap=100,
        count=9,
    )
    assert fewer.composition.added_wins == 3
    main_two = _pool(*(g for g in main.ghosts if g.ordinal != 3))
    two = short_set([main_two], cap=100, count=10)
    # Now 3 of 6 ended are wins, so 2 deaths need 2 wins: 300 and 400.
    assert _picked(two.episodes) == [(0, 0), (0, 1), (0, 4), (0, 5)]


def test_short_takes_every_win_when_the_uncapped_captures_saw_no_death() -> None:
    main = _pool(_ghost(0, "win", 500), _ghost(1, "win", 50), _ghost(2, "win", 900))
    chosen = short_set(
        [main, _pool(_ghost(3, "death", 9), capped=True)],
        cap=100,
        count=9,
    )
    assert _picked(chosen.episodes) == [(0, 0), (0, 1), (0, 2), (1, 3)]
    assert chosen.composition.added_wins == 2


def test_activity_counts_active_live_and_dying_or_winning_episodes() -> None:
    ghosts = [
        _ghost(0, "death", 3, active=bytes([1, 0, 16])),
        _ghost(1, "win", 5, active=bytes([0, 2, 4, 0, 16])),
        _ghost(2, "timeout", 4, active=bytes([0, 0, 0, 16])),
    ]
    active, live, decisive = activity(ghosts)
    assert active.tolist() == [1, 1, 2, 1, 1]
    assert live.tolist() == [3, 3, 3, 2, 1]
    assert decisive.tolist() == [0, 0, 1, 0, 1]


def test_quiet_runs_collapse_to_their_ends_and_short_ones_stay() -> None:
    rule = QuietRule(per_live=50, min_run=64, keep=8)
    active = np.zeros(300, np.int64)
    active[[100, 130, 250]] = 1
    live = np.full(300, 10, np.int64)
    none = np.zeros(300, np.int64)
    segments = quiet_segments(active, live=live, decisive=none, rule=rule)
    # Quiet runs: [0, 100) and [131, 250) collapse; [101, 130) and [251, 300)
    # are shorter than 64 and stay.
    assert segments.tolist() == [[0, 8], [92, 139], [242, 300]]
    crowd = np.full(300, 100, np.int64)
    busy = np.full(300, 1, np.int64)
    # With 100 live a decision needs 2 active episodes, so every one is quiet.
    assert quiet_segments(busy, live=crowd, decisive=none, rule=rule).tolist() == [
        [0, 8],
        [292, 300],
    ]
    assert quiet_segments(busy, live=live, decisive=none, rule=rule).tolist() == [
        [0, 300],
    ]


def test_a_death_or_win_is_never_skipped_even_in_a_crowd() -> None:
    rule = QuietRule(per_live=50, min_run=64, keep=8)
    crowd = np.full(300, 100, np.int64)
    busy = np.full(300, 1, np.int64)
    decisive = np.zeros(300, np.int64)
    decisive[150] = 1
    # One death among 100 live breaks the quiet run: the run before it keeps
    # its last 8 decisions, so the walk into the death plays too.
    assert quiet_segments(busy, live=crowd, decisive=decisive, rule=rule).tolist() == [
        [0, 8],
        [142, 159],
        [292, 300],
    ]


def test_wins_takes_every_win_of_every_capture_by_ordinal_then_capture() -> None:
    main = _pool(_ghost(0, "death", 9), _ghost(1, "win", 40), _ghost(3, "win", 70))
    extra = _pool(_ghost(1, "win", 30), _ghost(2, "timeout", 99), _ghost(0, "win", 50))
    chosen = wins_set([main, extra])
    assert _picked(chosen.episodes) == [(1, 0), (0, 1), (1, 1), (0, 3)]
    assert (
        chosen.composition.run,
        chosen.composition.qualified,
        chosen.composition.wins,
    ) == (6, 4, 4)
    assert not wins_set([_pool(_ghost(0, "death", 9))]).episodes


def test_wins_puts_the_one_win_pinned_by_seed_first() -> None:
    main = _pool(_ghost(1, "win", 40), _ghost(3, "win", 70), _ghost(2, "death", 9))
    extra = _pool(_ghost(1, "win", 30), _ghost(0, "win", 50))
    pinned = wins_set([main, extra], unbroken=3)
    assert _picked(pinned.episodes) == [(0, 3), (1, 0), (0, 1), (1, 1)]
    for seed in (1, 2):  # Seed 1 sampled two wins, seed 2 a death.
        with pytest.raises(ValueError, match="one win sampled with seed"):
            wins_set([main, extra], unbroken=seed)


def test_unbroken_spreads_short_wins_over_their_lengths_after_the_pinned_one() -> None:
    # Wins of 10, 20, ..., 100 decisions (ordinal = decisions // 10), one too
    # long, the pinned one (ordinal 50, seed 50) among them, and a death.
    wins = [_ghost(k, "win", 10 * k) for k in range(1, 11)]
    pool = _pool(*wins, _ghost(50, "win", 45), _ghost(60, "death", 5))
    chosen = unbroken_set([pool], pinned=50, count=4, steps=95)
    # Fitting wins by length: 10 .. 90; ranks 0, 4, 8 of 9: 10, 50, 90.
    assert [g.decisions for _, g in chosen.episodes] == [45, 10, 50, 90]
    assert (chosen.composition.qualified, chosen.composition.added_wins) == (10, 0)
    # Too few fit: all that do, then the shortest longer ones.
    few = unbroken_set([pool], pinned=50, count=4, steps=25)
    assert [g.decisions for _, g in few.episodes] == [45, 10, 20, 30]
    assert few.composition.added_wins == 1
    with pytest.raises(ValueError, match="at least 3"):
        unbroken_set([pool], pinned=50, count=2, steps=95)


def _flags(decisions: int, **at: int) -> bytearray:
    flags = bytearray(decisions)
    for name, t in at.items():
        flags[t] |= {
            "tile": NEW_TILE,
            "floor": FLOOR_CHANGED,
            "held": HELD,
            "fight": FOUGHT,
        }[name.rstrip("0123456789")]
    flags[-1] |= ENDED
    return flags


def test_a_time_map_keeps_progress_with_context_and_spreads_idle_stretches() -> None:
    flags = _flags(100, tile=10, held=50, fight=51)
    runs, level = keep_runs(bytes(flags), rule=TimeRule(steps=100, levels=((2, 3),)))
    assert level == 0
    kept = sorted({t for a, b in int_rows(runs) for t in range(a, b)})
    # Decision 0, progress and the end with 2 decisions each side, and 3
    # decisions spread over each idle stretch between them.
    assert {0, 1, 3, 5, *range(8, 13), *range(48, 54), 97, 98, 99} <= set(kept)
    assert len(kept) == 1 + 3 + 5 + 3 + 6 + 3 + 3
    assert runs[0, 0] == 0
    assert runs[-1, 1] == 100
    assert (np.diff(runs.ravel()) > 0).all()


def test_a_time_map_tightens_until_it_fits_and_always_keeps_floors_and_the_end() -> (
    None
):
    flags = _flags(1000, **{f"tile{t}": t for t in range(0, 1000, 10)}, floor=505)
    rule = TimeRule(steps=250, levels=((2, 8), (1, 2), (0, 1)))
    runs, level = keep_runs(bytes(flags), rule=rule)
    kept = {t for a, b in int_rows(runs) for t in range(a, b)}
    assert level == 2
    assert len(kept) <= 250
    assert {0, 505, 999} <= kept
    # Too much progress for the budget: progress is thinned, the floor change kept.
    runs, level = keep_runs(bytes(flags), rule=TimeRule(steps=40, levels=((0, 0),)))
    kept = {t for a, b in int_rows(runs) for t in range(a, b)}
    assert level == 1
    assert len(kept) == 40
    assert {0, 505, 999} <= kept


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
