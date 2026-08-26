"""Tests for the evaluation score."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import math

import pytest
import torch

from priml.baselines.craftax.game import constants
from priml.baselines.craftax.metric import CraftaxScore, crafter_score_pct
from priml.baselines.craftax.model import ActorCritic
from priml.metrics.custom_types import MetricProtocol


def _score(**overrides: object) -> CraftaxScore:
    config = CraftaxScore.Config()
    config.num_envs = 2
    config.steps = 5
    config.device = "cpu"
    config.seed = 3
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def _policy() -> ActorCritic:
    config = ActorCritic.Config()
    config.channels_in = 8
    config.num_layers = 1
    return config.make()


class _Actor:
    def __init__(self, policy: ActorCritic) -> None:
        self.model = policy
        self.observation_size = policy.policy[0].in_features
        self.device = next(policy.parameters()).device
        self.reset_count = 0
        self.previous_dones: list[torch.Tensor] = []
        self.draws: list[float] = []

    def reset(self, *, num_envs: int, device: torch.device) -> None:
        del num_envs, device
        self.reset_count += 1
        self.previous_dones = []

    def act(
        self,
        observation: torch.Tensor,
        previous_done: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        self.previous_dones.append(previous_done.clone())
        self.draws.append(float(torch.rand((), generator=generator)))
        logits, _ = self.model(observation)
        return torch.multinomial(
            logits.softmax(-1),
            1,
            generator=generator,
        ).squeeze(-1)


def _actor() -> _Actor:
    return _Actor(_policy())


def _episodes(count: int, *, unlocked: list[float] | None = None) -> dict[str, object]:
    """Build a saved state holding ``count`` identical episodes."""
    row = unlocked or [0.0] * len(constants.Achievement)
    return {
        "returns": [11.3] * count,
        "lengths": [7] * count,
        "unlocked": [list(row) for _ in range(count)],
        "rollout_index": 0,
    }


def test_it_satisfies_the_metric_protocol() -> None:
    assert isinstance(_score(), MetricProtocol)


def test_a_metric_that_saw_nothing_reports_no_episodes() -> None:
    # Not zero score: no episode finished, so there is nothing to average and
    # publishing a 0% would read as a policy that failed rather than as an
    # evaluation that has not happened.
    assert _score().compute() == {"episodes": 0.0}


def test_the_normalized_return_is_a_fraction_of_the_available_reward() -> None:
    score = _score()
    score.load_state_dict(_episodes(4))
    computed = score.compute()
    assert computed["mean_return"] == pytest.approx(11.3)
    assert computed["normalized_return_pct"] == pytest.approx(
        11.3 / constants.REWARD_CEILING * 100.0,
    )
    assert computed["episodes"] == 4.0
    assert computed["episode_length"] == pytest.approx(7.0)


def test_the_achievement_rate_is_the_mean_over_achievements() -> None:
    rates = [0.0] * len(constants.Achievement)
    rates[0] = 100.0
    rates[1] = 50.0
    score = _score()
    score.load_state_dict(_episodes(2, unlocked=rates))
    expected = 150.0 / len(constants.Achievement)
    assert score.compute()["achievements_pct"] == pytest.approx(expected)


def test_a_uniform_success_rate_scores_itself() -> None:
    # The geometric mean of identical values is that value, so the score is
    # readable on the same scale as the per-achievement rates.
    assert crafter_score_pct(torch.full((5,), 12.0).numpy()) == pytest.approx(12.0)


def test_one_unreached_achievement_does_not_annihilate_the_score() -> None:
    # Computed in log space: a plain geometric mean would return exactly zero
    # for any policy with a single achievement it never unlocks, which is
    # every policy anyone has trained.
    rates = torch.tensor([0.0, 50.0, 50.0, 50.0]).numpy()
    assert crafter_score_pct(rates) > 0.0


def test_breadth_beats_depth() -> None:
    # The property the Crafter score exists to express: unlocking several
    # achievements sometimes must beat farming one, at equal mean rate.
    broad = torch.tensor([25.0, 25.0, 25.0, 25.0]).numpy()
    narrow = torch.tensor([100.0, 0.0, 0.0, 0.0]).numpy()
    assert crafter_score_pct(broad) > crafter_score_pct(narrow)


def test_reset_forgets_every_episode() -> None:
    score = _score()
    score.load_state_dict(_episodes(3))
    score.reset()
    assert score.compute() == {"episodes": 0.0}


def test_a_checkpoint_round_trips() -> None:
    score = _score()
    score.load_state_dict(_episodes(2))
    expected = score.compute()

    restored = _score()
    restored.load_state_dict(score.state_dict())
    assert restored.compute() == expected


def test_a_saved_state_is_a_copy() -> None:
    # The loop checkpoints this dict; if it aliased the live lists, a later
    # episode would silently appear inside an already-written checkpoint.
    score = _score()
    score.load_state_dict(_episodes(1))
    saved = score.state_dict()
    score.load_state_dict(_episodes(5))
    assert len(saved["returns"]) == 1


def test_a_batch_without_an_actor_is_refused() -> None:
    with pytest.raises(TypeError, match="EvaluationActor"):
        _score().update(torch.zeros(2, 43))


@pytest.mark.compute_large_fixture
def test_playing_a_short_horizon_banks_nothing() -> None:
    # A truncated episode has an incomplete return, so counting it would drag
    # the mean toward zero by an amount set by the horizon, not the policy.
    score = _score(steps=2)
    score.update(torch.zeros(2, 43), actor=_actor())
    assert score.compute() == {"episodes": 0.0}


@pytest.mark.compute_large_fixture
def test_playing_banks_the_episodes_that_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shortening the episode limit is what makes a real rollout finish inside
    # a test; everything else about the play is the published evaluation.
    monkeypatch.setattr(constants, "MAX_TIMESTEPS", 2)
    score = _score(steps=5)
    score.update(torch.zeros(2, 43), actor=_actor())
    computed = score.compute()
    assert computed["episodes"] == 4.0
    assert computed["episode_length"] == pytest.approx(2.0)
    assert math.isfinite(computed["score_pct"])
    assert math.isfinite(computed["normalized_return_pct"])


@pytest.mark.compute_large_fixture
def test_the_same_seed_scores_the_same_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(constants, "MAX_TIMESTEPS", 2)
    policy = _policy()

    def played() -> dict[str, Any]:
        score = _score(steps=5)
        score.update(torch.zeros(2, 43), actor=_Actor(policy))
        return score.compute()

    assert played() == played()


@pytest.mark.compute_large_fixture
def test_repeated_batches_use_independent_evaluation_streams() -> None:
    score = _score(steps=1)
    actor = _actor()

    score.update(torch.zeros(2, 43), actor=actor)
    score.update(torch.zeros(2, 43), actor=actor)

    assert len(actor.draws) == 2
    assert actor.draws[0] != actor.draws[1]


@pytest.mark.compute_large_fixture
def test_view_mismatch_is_rejected_at_the_actor_boundary() -> None:
    actor = _actor()
    actor.observation_size += 1

    with pytest.raises(ValueError, match="observation_size"):
        _score(steps=1).update(torch.zeros(2, 43), actor=actor)


@pytest.mark.compute_large_fixture
def test_evaluation_switches_model_mode_once_per_rollout() -> None:
    score = _score(steps=5)
    actor = _actor()

    with patch.object(actor.model, "train", wraps=actor.model.train) as train:
        score.update(torch.zeros(2, 43), actor=actor)

    assert train.call_count == 2


@pytest.mark.compute_large_fixture
def test_two_plays_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The loop resets once per eval and updates once per batch, so several
    # passes must add episodes rather than replace them.
    monkeypatch.setattr(constants, "MAX_TIMESTEPS", 2)
    score = _score(steps=5)
    actor = _actor()
    score.update(torch.zeros(2, 43), actor=actor)
    score.update(torch.zeros(2, 43), actor=actor)
    assert actor.reset_count == 2
    assert score.compute()["episodes"] == 8.0


@pytest.mark.parametrize("field", ["num_envs", "steps"])
def test_an_empty_evaluation_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _score(**{field: 0})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
