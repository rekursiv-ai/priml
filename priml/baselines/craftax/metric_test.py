"""Tests for the score: PufferLib's fp32 aggregation of an evaluation's logs."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.craftax.evaluation import Played
from priml.baselines.craftax.game.state import LOG_DTYPE, Achievement
from priml.baselines.craftax.metric import (
    LOG_FIELDS,
    CraftaxScore,
    aggregate_logs,
    log_metrics,
    report_metrics,
)


def _logs(rows: list[list[float]]) -> np.ndarray:
    """Build ``LOG_DTYPE`` records from rows of ``LOG_FIELDS`` floats."""
    fields = np.zeros((len(rows), LOG_FIELDS), dtype=np.float32)
    for index, row in enumerate(rows):
        fields[index, : len(row)] = row
    return fields.view(LOG_DTYPE).reshape(len(rows))


def _with_n(perf: float, n: float) -> list[float]:
    """Build a row whose ``perf`` and ``n`` are set; every other field is zero."""
    row = [0.0] * LOG_FIELDS
    row[0] = perf
    row[-1] = n
    return row


def test_log_fields_are_pufferlibs_log_nf() -> None:
    # 5 scalars, 9 floors, 67 achievements, n.
    assert LOG_FIELDS == 82


def test_the_sum_runs_in_environment_order_as_pufferlib_does() -> None:
    # fp32 addition is not associative: 1e8 + 1 + 1 ... rounds each 1 away when
    # added in order, while a pairwise sum adds the ones together first.
    perfs = [1e8] + [1.0] * 15
    logs = _logs([_with_n(perf, 1.0) for perf in perfs])

    mean = aggregate_logs(logs)

    sequential = np.float32(0.0)
    for perf in perfs:
        sequential = np.float32(sequential + np.float32(perf))
    pairwise = np.sum(np.asarray(perfs, dtype=np.float32))
    assert sequential != pairwise
    assert mean[0] == np.float32(sequential / np.float32(len(perfs)))


def test_every_field_is_the_fp32_sum_of_the_rows_in_environment_order() -> None:
    # Values over seven orders of magnitude, so a sum in any other order, or
    # a pairwise one, lands on other bits in most fields.
    generator = np.random.default_rng(0)
    fields = generator.standard_normal((2_048, LOG_FIELDS)).astype(np.float32)
    scales = np.array([1e-3, 1.0, 1e4], dtype=np.float32)
    fields *= generator.choice(scales, fields.shape)
    fields[:, -1] = generator.integers(0, 3, len(fields))
    sequential = np.zeros(LOG_FIELDS, dtype=np.float32)
    kept = fields[np.not_equal(fields[:, -1], 0)]
    for i in range(len(kept)):
        sequential += kept[i, :]

    mean = aggregate_logs(fields.view(LOG_DTYPE).reshape(len(fields)))

    count = np.float32(sequential.item(-1))
    assert np.array_equal(
        mean[:-1].view(np.uint32),
        (sequential / count)[:-1].view(np.uint32),
    )
    assert mean[-1] == count


def test_environments_without_a_finished_episode_are_skipped() -> None:
    # PufferLib's log_accum reads nothing from a log whose n is zero.
    logs = _logs([_with_n(0.5, 1.0), _with_n(9.0, 0.0), _with_n(0.25, 1.0)])

    mean = aggregate_logs(logs)

    assert mean[0] == np.float32(0.75) / np.float32(2.0)
    assert mean[-1] == 2.0


def test_n_is_reported_as_the_count_not_divided_by_itself() -> None:
    mean = aggregate_logs(_logs([_with_n(1.0, 3.0), _with_n(2.0, 4.0)]))

    assert mean[-1] == 7.0
    assert log_metrics(mean)["n"] == 7.0


def test_no_finished_episode_gives_zeros() -> None:
    assert not aggregate_logs(_logs([_with_n(0.0, 0.0)] * 3)).any()


def test_the_report_leaves_out_the_keys_that_repeat_perf() -> None:
    mean = aggregate_logs(_logs([_with_n(0.5, 2.0), _with_n(0.25, 1.0)]))

    reported, logged = report_metrics(mean), log_metrics(mean)

    assert set(logged) - set(reported) == {"score", "episode_return"}
    assert set(reported) - set(logged) == {"floor_9_finish"}
    assert all(
        reported[name] == value for name, value in logged.items() if name in reported
    )


@pytest.mark.parametrize("wins", [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (2.0, 3.0)])
def test_finish_rate_counts_boss_defeats_over_completed_episodes(
    wins: tuple[float, float],
) -> None:
    logs = _logs([_with_n(1.0, 2.0), _with_n(1.0, 3.0), _with_n(0.0, 0.0)])
    logs["achievements"][0, Achievement.DEFEAT_NECROMANCER] = wins[0]
    logs["achievements"][1, Achievement.DEFEAT_NECROMANCER] = wins[1]
    logs["achievements"][2, Achievement.DEFEAT_NECROMANCER] = 100.0
    logs["achievements"][:, Achievement.DAMAGE_NECROMANCER] = [2.0, 3.0, 100.0]
    score = CraftaxScore.Config().make()

    score.update(played=Played(logs=logs, rollouts=1, gameplay_seconds=1.0))
    metrics = score.compute()

    assert metrics["floor_9_finish"] == float(np.float32(sum(wins)) / np.float32(5.0))
    assert metrics["n"] == 5.0


def test_no_completed_episodes_has_zero_finish_rate() -> None:
    assert (
        report_metrics(aggregate_logs(_logs([_with_n(0.0, 0.0)])))["floor_9_finish"]
        == 0.0
    )


def test_the_score_is_the_mean_of_the_played_logs() -> None:
    logs = _logs([_with_n(0.5, 1.0), _with_n(9.0, 0.0), _with_n(0.25, 3.0)])
    score = CraftaxScore.Config().make()

    score.update(played=Played(logs=logs, rollouts=3, gameplay_seconds=1.5))
    metrics = score.compute()

    assert metrics["perf"] == np.float32(0.75) / np.float32(4.0)
    assert metrics["n"] == 4.0
    assert metrics["rollouts"] == 3.0
    assert metrics["gameplay_seconds"] == 1.5
    assert "score" not in metrics
    assert "episode_return" not in metrics


def test_the_score_round_trips_its_state() -> None:
    score = CraftaxScore.Config().make()
    played = Played(logs=_logs([_with_n(1.0, 1.0)]), rollouts=1, gameplay_seconds=1.0)
    played.logs["achievements"][0, Achievement.DEFEAT_NECROMANCER] = 1.0
    score.update(played=played)

    restored = CraftaxScore.Config().make()
    restored.load_state_dict(score.state_dict())

    assert restored.compute() == score.compute()


def test_the_score_requires_a_played_evaluation() -> None:
    with pytest.raises(TypeError, match="played"):
        CraftaxScore.Config().make().update(played=object())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
