"""The batched-environment protocol is a structural, runtime-checkable surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import Tensor

import torch

from priml.data.environment import BatchedEnvironmentProtocol


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(kw_only=True, slots=True)
class _Step:
    observation: Tensor
    reward: Tensor
    done: Tensor
    info: dict[str, Tensor] = field(default_factory=dict)


class _Counter:
    """Two workers counting up; each episode ends after ``horizon`` steps."""

    num_actions = 2
    observation_size = 1
    reward_ceiling = 3.0

    def __init__(self, horizon: int = 3) -> None:
        self.horizon = horizon
        self.count = torch.zeros(0)

    def reset(self, num_envs: int) -> Tensor:
        self.count = torch.zeros(num_envs)
        return self.count[:, None]

    def step(self, actions: Tensor) -> _Step:
        self.count = self.count + 1
        done = self.count >= self.horizon
        reward = actions.to(torch.float32)
        self.count = torch.where(done, torch.zeros_like(self.count), self.count)
        return _Step(observation=self.count[:, None], reward=reward, done=done)

    def state_dict(self) -> Mapping[str, object]:
        return {"count": self.count.clone()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        count = state_dict["count"]
        assert isinstance(count, torch.Tensor)
        self.count = count.clone()


def test_a_conforming_environment_satisfies_the_protocol() -> None:
    assert isinstance(_Counter(), BatchedEnvironmentProtocol)


def test_an_environment_without_step_does_not() -> None:
    class _NoStep:
        num_actions = 2
        observation_size = 1
        reward_ceiling = 1.0

        def reset(self, num_envs: int) -> Tensor:
            return torch.zeros(num_envs, 1)

    assert not isinstance(_NoStep(), BatchedEnvironmentProtocol)


def test_done_step_returns_the_reset_observation_and_state_resumes() -> None:
    env: BatchedEnvironmentProtocol = _Counter(horizon=2)
    env.reset(2)
    env.step(torch.tensor([0, 1]))
    saved = env.state_dict()
    step = env.step(torch.tensor([1, 1]))
    assert step.done.tolist() == [True, True]
    assert step.observation.tolist() == [[0.0], [0.0]]
    env.load_state_dict(saved)
    assert env.step(torch.tensor([0, 0])).done.tolist() == [True, True]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
