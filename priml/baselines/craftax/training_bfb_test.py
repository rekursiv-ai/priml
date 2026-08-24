r"""Bit-for-bit golden for the whole PPO training path.

What is pinned is not a forward pass but an UPDATE: collect a rollout from the
simulator, score it, and take every optimization pass over it. That covers the
world generator, the renderer, action sampling, the advantage recursion, the
clipped objective, the gradient clip, and Adam -- in the order a real run
performs them -- so a change anywhere in the port moves the golden.

Mirrors the JAX study's ``training_bfb_test``, with two deliberate departures:

* It trains against the REAL simulator rather than a synthetic environment.
  The JAX golden could not afford that; four workers for four steps here is
  cheap, and pinning the simulator is most of the value.
* It compares one trace tensor rather than twenty named arrays, because that
  is the shape ``assert_bfb_against_golden`` speaks. The post-run state_dict
  the harness also checks covers what the separate arrays covered: every
  weight the update touched.

Regenerate after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \\
        priml/baselines/craftax/training_bfb_test.py

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.train.parallelism import NoParallel


_GOLDEN_DIR = Path(__file__).parent.resolve() / "goldens"

_TRACED_METRICS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "learning_rate",
    "explained_variance",
    "episodes",
)
"""Every scalar one update reports, in a fixed order.

Fixed because the trace is a tensor: a reordering would silently compare one
metric against another rather than failing.
"""

_UPDATES = 2
"""Updates the trace covers.

Two, not one: the second is what proves the learning-rate anneal, the
optimizer's carried moments, and the environment's continuation across an
update boundary are pinned too.
"""


class _PPOTrace(nn.Module):
    """A module whose forward is a whole PPO run.

    The BFB harness randomizes and snapshots a MODULE's parameters, so the
    training step has to be reachable as one. Registering the policy as a
    child means the harness's load lands on the very tensors the optimizer
    holds, and the post-run state it compares is the trained weights.
    """

    def __init__(self, step: CraftaxTrainStep) -> None:
        super().__init__()
        self.step = step
        self.policy = step.model

    @override
    def forward(self, unused: Tensor) -> Tensor:
        """Run every update and return its scalars as one flat tensor.

        Args:
          unused: Ignored; the data comes from the environment.

        Returns:
          trace: ``[updates, 1 + len(_TRACED_METRICS)]``, loss first.

        """
        del unused
        rows: list[Tensor] = []
        for _ in range(_UPDATES):
            result = self.step.train_step()
            metrics = result.get("metrics", {})
            rows.append(
                torch.stack(
                    [
                        result["loss"].reshape(()),
                        *(
                            torch.as_tensor(
                                float(metrics[name]),
                                dtype=torch.float32,
                            )
                            for name in _TRACED_METRICS
                        ),
                    ],
                ),
            )
        return torch.stack(rows)


def _build_trace() -> nn.Module:
    """Build the pinned training configuration."""
    config = CraftaxTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 4
    config.env.seed = 7
    config.rollout_steps = 4
    config.num_epochs = 2
    config.num_minibatches = 2
    config.total_train_steps = 8
    config.learning_rate = 3e-3
    # A 3x3 view rather than the benchmark's 9x11. The observation is one
    # one-hot vector per visible tile, so the window is what sets the input
    # width -- 798 floats here against 8,268 -- and the first layer's weights
    # are almost the whole committed file. The encoding is identical either
    # way: same channels, same order, same arithmetic, fewer tiles.
    #
    # What the golden gives up is pinning the published geometry, and it never
    # was the thing pinning it: ``observation_test`` checks the real width
    # against the reference implementation directly.
    config.env.view = (3, 3)
    # TWO hidden units, and not one. A width-1 axis broadcasts against
    # anything, so a transposed or mis-ordered tensor still lines up and the
    # golden would record the wrong arithmetic as correct.
    config.model.channels_in = 2
    config.model.num_layers = 1
    # Pinned, not defaulted: compiling changes which random numbers are drawn,
    # so a golden minted uncompiled cannot be replayed compiled.
    config.compile = None
    config.seed = 7
    return _PPOTrace(config.make())


@pytest.mark.parametrize("device", bfb_devices())
@pytest.mark.compute_training
def test_a_whole_ppo_update_matches_the_golden(device: str) -> None:
    del device
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="craftax_ppo_training_v1",
        build_module=_build_trace,
        build_input=lambda: torch.zeros(1),
        seed=7,
    )


def test_the_golden_is_small_enough_to_keep_in_git() -> None:
    # Goldens live in the repository, so one that grew to a model checkpoint
    # would be a payload, not a fixture. The bound sits just above the actual
    # size, which is almost entirely first-layer weights, so a widening that
    # doubles it has to be deliberate.
    golden = _GOLDEN_DIR / "craftax_ppo_training_v1.pt"
    assert golden.stat().st_size < 40_000


def test_the_golden_covers_the_optimizer_not_just_the_forward() -> None:
    """Prove the trace bites: perturb one weight and the comparison fails.

    A golden that only pinned a forward pass would still match after the
    optimizer changed, which is the regression this file exists to catch.
    """
    payload: dict[str, Any] = torch.load(
        _GOLDEN_DIR / "craftax_ppo_training_v1.pt",
        weights_only=False,
        map_location="cpu",
    )
    before = payload["post_state_dict"]["policy.policy.0.weight"]
    after = payload["state_dict"]["policy.policy.0.weight"]
    assert not torch.equal(before, after)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
