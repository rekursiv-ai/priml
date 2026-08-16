"""The score: how well a policy plays, measured over whole episodes.

Two numbers matter. Normalized return is mean episodic reward as a percentage
of the 226 available, and is the figure the published baselines report. The
Crafter score is the geometric mean of the per-achievement success rates,
which rewards breadth instead: a policy that unlocks twenty achievements
sometimes scores better than one that farms a single lucrative one.

Both are computed only from episodes that finished inside the evaluation
horizon. A truncated episode has an incomplete return, so counting it would
drag the mean toward zero by an amount that depends on the horizon rather than
on the policy.
"""

from __future__ import annotations

from typing import Any

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game import constants
from priml.baselines.craftax.model import ActorCritic


class CraftaxScore:
    """Play fixed-length episodes and report the benchmark's metrics."""

    class Config(Fig["CraftaxScore"]):
        """Evaluation geometry.

        These fields define the score. Two runs are comparable only when
        their evaluation seed, worker count, and horizon all match.
        """

        num_envs: int = 64
        """Parallel workers the policy is evaluated across."""

        steps: int = 10_000
        """Steps each worker takes."""

        seed: int = 42
        """Seed for the evaluation worlds and the action sampling."""

        view: tuple[int, int] = (9, 11)
        """Tiles the evaluated policy can see, ``(rows, columns)``.

        Must match the view the policy TRAINED on: the observation is one
        one-hot vector per visible tile, so a different window is a different
        input width and the network cannot read it at all."""

        device: str = "auto"
        """Device the evaluation runs on."""

    def __init__(self, config: Config) -> None:
        """Prepare an empty score.

        Args:
          config: Evaluation geometry.

        Raises:
          ValueError: The geometry is not positive.

        """
        if config.num_envs <= 0 or config.steps <= 0:
            raise ValueError("Evaluation geometry must be positive")
        self.config = config
        self._returns: list[float] = []
        self._lengths: list[int] = []
        self._unlocked: list[list[float]] = []

    def update(self, logits: Tensor, **batch: Any) -> None:
        """Score the bound policy over a complete evaluation rollout.

        The metric plays its own episodes rather than reading the batch: a
        score is a property of the policy acting from fresh worlds, not of
        whatever transitions training happened to visit.

        Args:
          logits: Unused; present to satisfy the metric interface.
          **batch: Must carry ``policy``, the network to evaluate.

        """
        del logits
        policy = batch.get("policy")
        if policy is None:
            return
        self._play(policy)

    def compute(self) -> dict[str, Any]:
        """Summarize every episode seen since the last reset.

        Returns:
          metrics: ``normalized_return_pct`` is the headline; ``score_pct``
            is the geometric achievement score; the rest describe the
            distribution behind them.

        """
        if not self._returns:
            return {"episodes": 0.0}
        returns = np.asarray(self._returns, dtype=np.float64)
        rates = np.asarray(self._unlocked, dtype=np.float64).mean(axis=0)
        return {
            "normalized_return_pct": float(
                returns.mean() / constants.REWARD_CEILING * 100.0,
            ),
            "score_pct": crafter_score_pct(rates),
            "mean_return": float(returns.mean()),
            "achievements_pct": float(rates.mean()),
            "episodes": float(len(returns)),
            "episode_length": float(np.mean(self._lengths)),
        }

    def reset(self) -> None:
        """Forget every episode seen so far."""
        self._returns = []
        self._lengths = []
        self._unlocked = []

    def state_dict(self) -> dict[str, Any]:
        """Return the accumulated episodes."""
        return {
            "returns": list(self._returns),
            "lengths": list(self._lengths),
            "unlocked": [list(row) for row in self._unlocked],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore episodes saved by :meth:`state_dict`."""
        self._returns = list(state_dict["returns"])
        self._lengths = list(state_dict["lengths"])
        self._unlocked = [list(row) for row in state_dict["unlocked"]]

    @torch.no_grad()
    def _play(self, policy: ActorCritic) -> None:
        """Run the fixed evaluation rollout, banking finished episodes."""
        config = CraftaxEnv.Config()
        config.num_envs = self.config.num_envs
        config.device = self.config.device
        config.seed = self.config.seed
        config.view = self.config.view
        env = config.make()

        observation = env.reset()
        generator = torch.Generator(device=observation.device)
        generator.manual_seed(self.config.seed)
        episode_return = torch.zeros(self.config.num_envs, device=observation.device)
        episode_length = torch.zeros(
            self.config.num_envs,
            dtype=torch.int64,
            device=observation.device,
        )

        for _ in range(self.config.steps):
            logits, _ = policy(observation)
            action = torch.multinomial(
                logits.softmax(-1),
                1,
                generator=generator,
            ).squeeze(-1)
            transition = env.step(action)
            observation = transition.observation
            episode_return = episode_return + transition.reward
            episode_length = episode_length + 1

            if bool(transition.done.any()):
                finished = transition.done
                self._returns.extend(episode_return[finished].tolist())
                self._lengths.extend(episode_length[finished].tolist())
                unlocked = torch.stack(
                    [transition.info[name] for name in sorted(transition.info)],
                    dim=-1,
                )
                self._unlocked.extend(unlocked[finished].tolist())
                episode_return = episode_return * ~finished
                episode_length = episode_length * ~finished


def crafter_score_pct(success_rates_pct: np.ndarray) -> float:
    """Aggregate per-achievement success rates the way Crafter does.

    The geometric mean, computed in log space so that a zero rate does not
    annihilate the whole score. Breadth is what this rewards: unlocking many
    achievements rarely beats unlocking one reliably.

    Args:
      success_rates_pct: Per-achievement success percentages.

    Returns:
      score_pct: The aggregate score, in percent.

    References:
      https://arxiv.org/abs/2109.06780
        Hafner 2021. Benchmarking the spectrum of agent capabilities.

    """
    return float(np.expm1(np.log1p(success_rates_pct).mean()))
