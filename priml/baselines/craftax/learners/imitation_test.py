"""Unit tests for branch self-imitation: the loss, the archive, and the first window.

The archive is driven with hand-built rollouts whose observations name their
epoch and row, so which branch the loss replays can be read back from what the
policy was fed. On the GPU, the learner epoch with imitation on is captured,
replayed and held to the eager epoch bit for bit.
"""

from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
    override,
)

import io
import math

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.learners.imitation import (
    BranchImitation,
    imitation_loss,
)
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.rollout import PhiloxSampler
from priml.baselines.craftax.testing import (
    packed_observations,
    sole_feature_policy,
    tiny_board_policy,
    tiny_train_step,
)
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    LearnerRollout,
)
from priml.lib.codec import from_plain
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.model.min_gru import TritonScan
from priml.optimizers.fused_muon import FusedMuon


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.craftax.model import Policy


AGENTS: Final = 5
HORIZON: Final = 4
NUM_ACTIONS: Final = 6
LAYERS: Final = 2
WIDTH: Final = 7
FEATURE_WIDTH: Final = 5
"""A real policy's feature width, distinct from every other."""


def test_the_loss_is_the_masked_cross_entropy_mean_over_the_branch() -> None:
    """Two branch steps: an even pick of two legal actions, then of three.

    The third action of step 0 is illegal, so its logit of 5 is left out: at
    -inf, not PPO's -1e4, which would tie it with the legal logits of -1e4
    and give it a third of the mass. The steps past the branch have no legal
    action, which would be NaN unmasked.
    """
    logits = torch.zeros(HORIZON, 3)
    logits[0] = torch.tensor([-1e4, -1e4, 5.0])
    logits[2:] = torch.tensor([7.0, -3.0, 1.0])
    logits.requires_grad_(True)
    mask = torch.ones(HORIZON, 3)
    mask[0, 2] = 0
    mask[2:] = 0
    actions = torch.tensor([0.0, 1.0, 2.0, 2.0])
    loss = imitation_loss(logits, actions, mask, torch.tensor(2), coefficient=0.01)
    assert loss.dtype == torch.float32
    assert float(loss.detach()) == pytest.approx(0.01 * (math.log(2) + math.log(3)) / 2)
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert torch.equal(logits.grad[2:], torch.zeros(2, 3))
    assert float(logits.grad[0, 2]) == 0.0
    # An empty branch weighs nothing and learns nothing.
    fresh = logits.detach().requires_grad_(True)
    empty = imitation_loss(fresh, actions, mask, torch.tensor(0), coefficient=0.01)
    empty.backward()
    assert float(empty.detach()) == 0.0
    assert fresh.grad is not None
    assert torch.equal(fresh.grad, torch.zeros(HORIZON, 3))


def test_ingest_archives_the_legal_restored_branches_to_their_first_terminal() -> None:
    """Rows 0-3 are read; row 4 is not, though restored.

    Row 0 ends at its terminal on step 2, and an illegal action after it does
    not count. Row 1's terminal on step 0 belongs to the episode before the
    branch, so the branch runs the horizon. Row 2 was not restored, and row 3
    took an illegal action.
    """
    imitation = _imitation(rows=4, capacity=3)
    imitation.ingest(
        _rollout(
            epoch=1,
            starts=(0, 1, 3, 4),
            terminals=((0, 2), (1, 0)),
            illegal=((0, 3), (3, 1)),
        ),
    )
    state = imitation.state_dict()
    assert int(state["count"]) == int(state["inserted"]) == 2
    assert _ids(state["observations"][:2]) == [10, 11]
    assert state["lengths"].tolist() == [2, HORIZON, 0]
    assert torch.equal(
        state["initial_states"][:, :2, 0],
        torch.tensor([[10.0, 11.0]] * 2),
    )
    assert torch.equal(state["actions"][1], _rollout(epoch=1).actions[1])


def test_the_archive_keeps_the_newest_branches_in_completion_order() -> None:
    """Three branches complete in the first epoch, one in the second; two are kept.

    Row 1 ends first (step 2), then rows 0 and 2 at the horizon, so the first
    epoch keeps rows 0 and 2; the second epoch's branch displaces the older.
    """
    imitation = _imitation(rows=3, capacity=2)
    imitation.ingest(_rollout(epoch=1, starts=(0, 1, 2), terminals=((1, 2),)))
    policy = _RecordingPolicy()
    assert _replayed(imitation, policy, times=3) == [10, 12, 10]
    imitation.ingest(_rollout(epoch=2, starts=(1,), terminals=((1, 3),)))
    # The cursor stands at 3: past the oldest (row 2), at row 1 of epoch 2.
    assert _replayed(imitation, policy, times=2) == [21, 12]


def test_the_cursor_waits_while_the_archive_is_empty_and_replays_the_carry() -> None:
    imitation = _imitation(rows=3, capacity=2)
    imitation.ingest(_rollout(epoch=1, starts=()))
    policy = _RecordingPolicy()
    loss, metrics = imitation.loss(policy)
    assert float(loss.detach()) == 0.0
    assert float(metrics["imitation/branches"]) == 0.0
    assert int(imitation.state_dict()["cursor"]) == 0
    imitation.ingest(_rollout(epoch=2, starts=(2,)))
    loss, metrics = imitation.loss(policy)
    assert float(metrics["imitation/loss"]) == float(loss.detach()) > 0
    assert float(metrics["imitation/branches"]) == 1.0
    observations, state, starts = policy.fed[-1]
    assert observations.shape == (1, HORIZON, 3)
    assert torch.equal(state, torch.full((LAYERS, 1, WIDTH), 22.0))
    assert torch.equal(starts, torch.zeros(1, HORIZON))


def test_the_state_round_trips_and_an_empty_state_empties_the_archive() -> None:
    imitation = _imitation(rows=3, capacity=2)
    assert imitation.state_dict() == {}
    imitation.ingest(_rollout(epoch=1, starts=(0, 1, 2)))
    policy = _RecordingPolicy()
    _replayed(imitation, policy, times=1)
    saved = {name: value.clone() for name, value in imitation.state_dict().items()}
    restored = _imitation(rows=3, capacity=2)
    restored.load_state_dict(saved)
    assert _replayed(restored, policy, times=2) == _replayed(imitation, policy, times=2)
    # Loaded into a live archive, the tensors are copied, not rebound.
    live = restored.state_dict()["observations"]
    restored.load_state_dict(saved)
    assert restored.state_dict()["observations"] is live
    restored.load_state_dict({})
    assert int(restored.state_dict()["count"]) == 0
    assert float(restored.loss(policy)[0].detach()) == 0.0
    # Without a feature the state is what it was before features existed.
    assert "features" not in saved
    assert policy.read[-1] is None


def test_with_a_feature_the_archive_keeps_each_branchs_and_the_replay_reads_them() -> (
    None
):
    """The branches of the completion-order test, each replayed on its own features.

    Rows 0 and 2 are kept, row 2 in slot 0; the replay reads row 0's first.
    """
    imitation = _imitation(rows=3, capacity=2)
    rollout = _rollout(epoch=1, starts=(0, 1, 2), terminals=((1, 2),), featured=True)
    assert rollout.features is not None
    imitation.ingest(rollout)
    state = imitation.state_dict()
    assert list(state)[-1] == "features"
    assert torch.equal(state["features"], rollout.features[[2, 0]])
    policy = _RecordingPolicy()
    assert _replayed(imitation, policy, times=2) == [10, 12]
    for read, row in zip(policy.read, (0, 2), strict=True):
        assert read is not None
        assert torch.equal(read, rollout.features[row : row + 1])
    saved = {name: value.clone() for name, value in state.items()}
    restored = _imitation(rows=3, capacity=2)
    restored.load_state_dict(saved)
    _replayed(restored, policy, times=2)
    for ours, theirs in zip(policy.read[-2:], policy.read[:2], strict=True):
        assert ours is not None
        assert theirs is not None
        assert torch.equal(ours, theirs)


def test_a_feature_policy_relearns_a_branch_on_the_features_its_actor_read() -> None:
    """The loss is the branch's cross-entropy over the learner's window of it.

    A policy whose trunk reads its feature alone, which refuses a window
    without one: replayed from the branch's carry on its stored features, it
    scores bit for bit the window the learner would.
    """
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    torch.manual_seed(1)
    policy = config.make()
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(AGENTS, HORIZON, FEATURE_WIDTH, generator=generator)
    initial = torch.randn(policy.initial_state(AGENTS).shape, generator=generator)
    terminals = torch.zeros(AGENTS, HORIZON, dtype=torch.bfloat16)
    terminals[1, 3] = 1
    branch_starts = torch.zeros(AGENTS, dtype=torch.uint8)
    branch_starts[1] = 1
    rollout = LearnerRollout(
        observations=packed_observations(
            tiny_board_policy(),
            batch=AGENTS,
            time=HORIZON,
            seed=3,
        ).bfloat16(),
        actions=torch.randint(0, 43, (AGENTS, HORIZON), generator=generator).float(),
        logprobs=torch.zeros(AGENTS, HORIZON),
        rewards=torch.zeros(AGENTS, HORIZON),
        terminals=terminals,
        values=torch.zeros(AGENTS, HORIZON),
        action_mask=torch.ones(AGENTS, HORIZON, 43, dtype=torch.bfloat16),
        initial_states=initial.to(policy.state_dtype),
        branch_starts=branch_starts,
        features=features.bfloat16(),
    )
    assert rollout.features is not None
    imitation = _imitation(rows=3, capacity=2)
    imitation.ingest(rollout)
    loss, _ = imitation.loss(policy)
    decoded, _, _ = policy.forward_sequence(
        rollout.observations[1:2],
        rollout.initial_states[:, 1:2],
        torch.zeros(1, HORIZON, dtype=torch.bfloat16),
        features=rollout.features[1:2],
    )
    expected = imitation_loss(
        decoded[0, :, :43],
        rollout.actions[1],
        rollout.action_mask[1],
        torch.tensor(3),
        coefficient=imitation.coefficient,
    )
    assert torch.equal(loss, expected)
    loss.backward()
    assert policy.proj_feature is not None
    assert policy.proj_feature.weight.grad is not None
    assert policy.proj_feature.weight.grad.any()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("rows", 0, "rows"),
        ("capacity", -1, "capacity"),
        ("coefficient", 0.0, "coefficient"),
        ("coefficient", math.nan, "coefficient"),
        ("coefficient", math.inf, "coefficient"),
    ],
)
def test_a_bad_geometry_or_weight_is_refused(
    field: str,
    value: float,
    match: str,
) -> None:
    config = BranchImitation.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=match):
        config.make()


def test_rows_past_the_environments_are_refused_before_anything_is_built() -> None:
    config = tiny_train_step()
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.auxiliary = BranchImitation.Config()
    with pytest.raises(ValueError, match=r"rows \[0, 64\) of 4 environments"):
        config.make()


def test_the_first_window_alone_learns_the_auxiliary_loss() -> None:
    """At a zero rate every window scores the same weights, so gradients compare.

    With the auxiliary on, the first window's gradients move and the second's
    do not; its metrics and its state pass through the windows.
    """
    gradients: list[list[list[Tensor]]] = []
    for auxiliary in (None, _SumAuxiliary.Config()):
        config = tiny_train_step()
        config.optimizer = PartialConfig(torch.optim.SGD, lr=0.0)
        windows = config.learner
        assert isinstance(windows, AgentWindows.Config)
        windows.auxiliary = auxiliary
        step = _make(config)
        try:
            recorded: list[list[Tensor]] = []
            step.optimizer.register_step_pre_hook(
                partial(_record_gradients, recorded=recorded),
            )
            learner = step.learner
            assert isinstance(learner, AgentWindows)
            _, metrics = learner(step, _learner_rollout(step))
            gradients.append(recorded)
            assert float(metrics["auxiliary_loss"]) == 0.0
            if auxiliary is None:
                assert learner.state_dict() == {}
            else:
                assert "sum/loss" in metrics
                assert learner.state_dict() == {"ingested": torch.tensor(1)}
                learner.load_state_dict({"ingested": torch.tensor(5)})
                assert learner.state_dict() == {"ingested": torch.tensor(5)}
        finally:
            step.close()
    alone, joined = gradients
    assert len(alone) == len(joined) == 2
    assert all(not torch.equal(a, b) for a, b in zip(alone[0], joined[0], strict=True))
    assert all(torch.equal(a, b) for a, b in zip(alone[1], joined[1], strict=True))


def test_a_resumed_run_with_imitation_equals_an_uninterrupted_one() -> None:
    """Epoch 1's checkpoint, loaded into the step after epoch 2: epoch 2 runs again.

    Every row of the slot an epoch learns from starts a branch, so epoch 1's 3
    branches overflow the archive of 2 and epoch 2's replay it; the archive
    and its cursor must be saved for epoch 2 to match. Rollouts of 2 steps.
    """
    config = _imitation_step()
    config.rollout.horizon = 2
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.minibatch_size = 2 * config.env.num_envs
    config.train_budget_steps = 2
    step = _make(config)
    try:
        _train(step)
        saved = io.BytesIO()
        torch.save(step.state_dict(), saved)
        expected = [_train(step)]
        expected_state = {
            name: value.clone() for name, value in step.learner.state_dict().items()
        }
        step.load_state_dict(
            from_plain(
                cast(
                    "object",
                    torch.load(io.BytesIO(saved.getvalue()), weights_only=True),
                ),
                dict[str, object],
            ),
        )
        actual = [_train(step)]
        actual_state = step.learner.state_dict()
    finally:
        step.close()
    for ours, theirs in zip(actual, expected, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(ours, theirs, strict=True))
    assert actual_state.keys() == expected_state.keys()
    for name, value in actual_state.items():
        assert torch.equal(value, expected_state[name]), name
    assert float(expected[-1][-1]) > 0


@pytest.mark.gpu_triton
def test_the_captured_learner_epoch_with_imitation_equals_the_eager_one() -> None:
    """Five epochs of the production kernels: the first eager, the rest replayed.

    A twin forced eager every epoch must train to the same bits: every loss
    term and metric, the masters and momentum, and the archive and its cursor.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _imitation_step()
    config.parallelism.device = "cuda"
    config.sampler = PhiloxSampler.Config()
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.block.scan = TritonScan.Config()
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TritonPPO.Config().update(windows.objective, skip_missing=True)
    config.rollout.horizon = 8
    windows.minibatch_size = 16
    config.train_budget_steps = 5
    captured, eager = _make(config), _make(config)
    try:
        for _ in range(5):
            eager._warm = False
            ours, theirs = _train(captured), _train(eager)
            assert all(torch.equal(a, b) for a, b in zip(ours, theirs, strict=True))
        assert len(captured._graphs) == 2
        assert not eager._graphs
        for a, b in zip(
            _optimizer_state(captured.optimizer),
            _optimizer_state(eager.optimizer),
            strict=True,
        ):
            assert torch.equal(a, b)
        state = captured.learner.state_dict()
        for name, value in eager.learner.state_dict().items():
            assert torch.equal(value, state[name]), name
        assert int(state["cursor"]) >= 3
    finally:
        captured.close()
        eager.close()


class _RecordingPolicy(nn.Module):
    """Score every step with one learnable row, recording what it is fed."""

    def __init__(self) -> None:
        super().__init__()
        self.row = nn.Parameter(torch.arange(NUM_ACTIONS + 1, dtype=torch.float32))
        self.dtype = torch.float32
        self.fed: list[tuple[Tensor, Tensor, Tensor]] = []
        self.read: list[Tensor | None] = []
        """The features each window was fed, beside ``fed``."""

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return a zero carry."""
        return torch.zeros(LAYERS, num_envs, WIDTH, device=device)

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step: the row for every environment."""
        del episode_start, carry, features
        return self.row.expand(observations.shape[0], -1), state

    def forward_sequence(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        *,
        actions: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score a window: the row at every step; no loss of its own."""
        del actions
        self.fed.append((observations.clone(), state.clone(), episode_start.clone()))
        self.read.append(None if features is None else features.clone())
        batch, time = episode_start.shape
        return self.row.expand(batch, time, -1), state, self.row.new_zeros(())

    @override
    def forward(self, observations: Tensor) -> Tensor:
        return self.row.expand(observations.shape[0], -1)


class _SumAuxiliary:
    """An auxiliary loss: a weight times the sum of every policy weight."""

    class Config(Fig["_SumAuxiliary"]):
        """The loss's weight."""

        weight: float = 0.5
        """Scales the sum."""

    def __init__(self, config: Config) -> None:
        self.weight = config.weight
        self.ingested = 0

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Nothing to check."""
        del config

    def ingest(self, rollout: LearnerRollout) -> None:
        """Count the rollouts."""
        del rollout
        self.ingested += 1

    def loss(self, policy: Policy) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the weighted sum of the weights."""
        total = (
            self.weight
            * torch.stack(
                [parameter.float().sum() for parameter in policy.parameters()],
            ).sum()
        )
        return total, {"sum/loss": total.detach()}

    def state_dict(self) -> dict[str, Tensor]:
        """Return the rollouts counted."""
        return {"ingested": torch.tensor(self.ingested)}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the count."""
        self.ingested = int(state["ingested"])


def _record_gradients(
    optimizer: torch.optim.Optimizer,
    args: object,
    kwargs: object,
    *,
    recorded: list[list[Tensor]],
) -> None:
    """Record every parameter's gradient before a step: an optimizer pre-hook."""
    del args, kwargs
    recorded.append(
        [
            parameter.grad.clone()
            for group in optimizer.param_groups
            for parameter in cast("list[torch.Tensor]", group["params"])
            if parameter.grad is not None
        ],
    )


def _imitation(*, rows: int, capacity: int) -> BranchImitation:
    config = BranchImitation.Config()
    config.rows = rows
    config.capacity = capacity
    imitation = config.make()
    imitation.prepare(tiny_train_step())
    return imitation


# With ``featured``, step ``t`` of row ``r`` also reads a feature, in bf16, the policy's
# dtype: ``[10 * epoch + r, t]``.
def _rollout(
    *,
    epoch: int,
    starts: tuple[int, ...] = (),
    terminals: tuple[tuple[int, int], ...] = (),
    illegal: tuple[tuple[int, int], ...] = (),
    featured: bool = False,
) -> LearnerRollout:
    """Return a rollout whose row ``r`` observes and carries ``10 * epoch + r``."""
    names = (10 * epoch + torch.arange(AGENTS)).float()
    observations = torch.zeros(AGENTS, HORIZON, 3)
    observations[..., 0] = names[:, None]
    actions = (torch.arange(AGENTS * HORIZON) % NUM_ACTIONS).float()
    actions = actions.reshape(AGENTS, HORIZON)
    action_mask = torch.ones(AGENTS, HORIZON, NUM_ACTIONS)
    ended = torch.zeros(AGENTS, HORIZON)
    for row, time in illegal:
        action_mask[row, time, int(actions[row, time])] = 0
    for row, time in terminals:
        ended[row, time] = 1
    branch_starts = torch.zeros(AGENTS, dtype=torch.uint8)
    branch_starts[list(starts)] = 1
    features = torch.stack(
        (
            names[:, None].expand(AGENTS, HORIZON),
            torch.arange(HORIZON).float().expand(AGENTS, HORIZON),
        ),
        dim=-1,
    ).bfloat16()
    return LearnerRollout(
        observations=observations,
        actions=actions,
        logprobs=torch.zeros(AGENTS, HORIZON),
        rewards=torch.zeros(AGENTS, HORIZON),
        terminals=ended,
        values=torch.zeros(AGENTS, HORIZON),
        action_mask=action_mask,
        initial_states=names[None, :, None].expand(LAYERS, AGENTS, WIDTH).clone(),
        branch_starts=branch_starts,
        features=features if featured else None,
    )


def _ids(observations: Tensor) -> list[int]:
    """Return the names of archived or replayed branches."""
    return [int(value) for value in observations[:, 0, 0]]


def _replayed(
    imitation: BranchImitation,
    policy: _RecordingPolicy,
    *,
    times: int,
) -> list[int]:
    """Take ``times`` losses; return the names of the branches they replayed."""
    for _ in range(times):
        imitation.loss(policy)
    return [_ids(observations)[0] for observations, _, _ in policy.fed[-times:]]


def _learner_rollout(step: CraftaxTrainStep) -> LearnerRollout:
    """Return a random rollout of the step's agents and horizon, in its dtypes."""
    model = step.config.model
    assert isinstance(model, MinGRUPolicy.Config)
    agents, horizon = step.env.num_envs, step.config.rollout.horizon
    generator = torch.Generator().manual_seed(3)

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator).bfloat16()

    return LearnerRollout(
        observations=packed_observations(model, batch=agents, time=horizon),
        actions=torch.randint(0, 43, (agents, horizon), generator=generator).float(),
        logprobs=-draw(agents, horizon).abs(),
        rewards=draw(agents, horizon),
        terminals=torch.zeros(agents, horizon, dtype=torch.bfloat16),
        values=draw(agents, horizon),
        action_mask=torch.ones(agents, horizon, 43, dtype=torch.bfloat16),
        initial_states=step.model.initial_state(agents),
        branch_starts=torch.ones(agents, dtype=torch.uint8),
    )


def _imitation_step() -> CraftaxTrainStep.Config:
    """Return the tiny step with imitation of rows 0-2 into an archive of 2."""
    config = tiny_train_step()
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    imitation = windows.auxiliary = BranchImitation.Config()
    imitation.rows = 3
    imitation.capacity = 2
    return config


def _make(config: CraftaxTrainStep.Config) -> CraftaxTrainStep:
    torch.manual_seed(0)
    return config.make()


# The slot the epoch learns from was collected by the last epoch; the one it prefetches
# into is the other, so the marks are the learner's to read.
def _train(step: CraftaxTrainStep) -> list[Tensor]:
    """Mark every row of the next epoch's slot restored, then train it."""
    step.rollout.slots[step.ready].branch_starts.fill_(1)
    if step.device.type == "cuda":
        torch.cuda.synchronize(step.device)
    result = step.train_step()
    metrics = result.get("metrics", {})
    return [
        result["model"],
        *(
            torch.as_tensor(metrics[name]).float()
            for name in ("auxiliary_loss", "imitation/branches", "imitation/loss")
        ),
    ]


def _optimizer_state(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    assert isinstance(optimizer, FusedMuon)
    return [*optimizer.master_weights, *optimizer.momentum_buffers]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
