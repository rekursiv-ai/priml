"""Tests for the PPO training step."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import copy
import math

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.optimizers import learning_rate
from priml.testing.fixtures import torch_compiler_isolation
from priml.train.custom_types import TrainStepOutput, TrainStepProtocol
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable


pytestmark = pytest.mark.compute_training


def _config(**overrides: object) -> CraftaxTrainStep.Config:
    config = CraftaxTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 4
    config.rollout_steps = 3
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.channels_in = 4
    config.model.num_layers = 1
    for name, value in overrides.items():
        setattr(config, name, value)
    # The world and the policy draw from separate streams, so a reproducible
    # run has to pin both.
    if "seed" in overrides:
        config.env.seed = int(config.seed)
    return config


def _step() -> CraftaxTrainStep:
    step = _config().make()
    assert isinstance(step, CraftaxTrainStep)
    return step


def test_it_satisfies_the_training_step_protocol() -> None:
    assert isinstance(_step(), TrainStepProtocol)


def test_the_network_is_sized_from_the_environment() -> None:
    # An experiment that changes the environment must not have to remember to
    # resize the network by hand.
    step = _step()
    policy = step.model.policy
    first, last = policy[0], policy[-1]
    assert isinstance(first, nn.Linear)
    assert isinstance(last, nn.Linear)
    assert first.in_features == step.env.observation_size
    assert last.out_features == step.env.num_actions


def test_eval_scores_without_advancing_the_training_environment() -> None:
    """Evaluation must not step the env or bank episodes into training metrics.

    ``eval_loss`` delegates to ``train_loss``, which calls ``collect()`` --
    and ``collect`` advances ``_observation``/``_done`` and calls
    ``_record_episodes``. So merely SCORING moved the world the next update
    would train from, and folded eval episodes into the return/length averages
    an experiment reads as training progress.
    """
    step = _step()
    observation = step._observation.clone()
    done = step._done.clone()
    banked = len(step._finished_returns)

    _ = step.eval_loss()

    assert torch.equal(step._observation, observation), "eval moved the env"
    assert torch.equal(step._done, done), "eval moved the done flags"
    assert len(step._finished_returns) == banked, "eval banked episodes"


def test_one_step_consumes_the_declared_interactions() -> None:
    step = _step()
    assert step.steps_per_update == 4 * 3


def test_the_loops_batch_passes_through_untouched() -> None:
    # The rollout is collected inside the step, so whatever the loop hands
    # over is neither read nor copied.
    batch: dict[str, object] = {"observation": object()}
    assert _step().preprocess_batch(batch) is batch


def test_a_step_optimizes_and_reports_its_diagnostics() -> None:
    result = _step().train_step()
    assert math.isfinite(float(result["loss"]))
    metrics = _metrics(result)
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
        "explained_variance",
        "episodes",
    ):
        assert math.isfinite(float(metrics[name])), name


def test_a_step_changes_the_policy() -> None:
    step = _step()
    policy = step.model.policy
    layer = policy[0]
    assert isinstance(layer, nn.Linear)
    before = layer.weight.detach().clone()
    step.train_step()
    assert not torch.equal(before, layer.weight.detach())


def test_the_step_counter_advances() -> None:
    step = _step()
    step.train_step()
    step.train_step()
    assert step.global_step == 2


def test_the_learning_rate_anneals_toward_zero() -> None:
    step = _config(total_train_steps=4).make()
    rates: list[float] = []
    for _ in range(3):
        step.train_step()
        rates.append(learning_rate(step.optimizer))
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0]


def test_annealing_can_be_switched_off() -> None:
    step = _config(anneal_learning_rate=False, learning_rate=1e-3).make()
    step.train_step()
    assert step.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_a_rollout_has_the_declared_shape() -> None:
    step = _step()
    rollout = step.collect()
    assert rollout.observation.shape == (3, 4, step.env.observation_size)
    assert rollout.action.shape == (3, 4)
    assert rollout.advantage.shape == (3, 4)
    assert rollout.target.shape == (3, 4)


def test_a_rollout_is_collected_without_gradients() -> None:
    # Backpropagating through the collection would tie the policy to its own
    # sampling, which the clipped objective already accounts for.
    rollout = _step().collect()
    assert not rollout.observation.requires_grad
    assert not rollout.log_prob.requires_grad


def test_minibatches_partition_the_rollout_exactly() -> None:
    step = _step()
    rollout = step.collect()
    seen = [minibatch["action"].shape[0] for minibatch in rollout.minibatches(count=4)]
    assert sum(seen) == 4 * 3
    assert len(seen) == 4


def test_minibatches_are_shuffled() -> None:
    rollout = _step().collect()
    first = next(
        rollout.minibatches(count=1, generator=torch.Generator().manual_seed(0)),
    )
    second = next(
        rollout.minibatches(count=1, generator=torch.Generator().manual_seed(1)),
    )
    assert not torch.equal(first["action"], second["action"])


def test_the_same_seed_reproduces_the_same_update() -> None:
    def run() -> float:
        step = _config(seed=7).make()
        return float(step.train_step()["loss"])

    assert run() == run()


def test_evaluation_does_not_change_the_policy() -> None:
    step = _step()
    policy = step.model.policy
    layer = policy[0]
    assert isinstance(layer, nn.Linear)
    before = layer.weight.detach().clone()
    result = step.eval_loss()
    assert math.isfinite(float(result["loss"]))
    assert torch.equal(before, layer.weight.detach())


def test_action_logits_can_be_read_for_arbitrary_observations() -> None:
    step = _step()
    logits = step.call_eval(observation=torch.zeros(3, step.env.observation_size))
    assert logits.shape == (3, step.env.num_actions)
    assert not logits.requires_grad


def test_evaluation_actor_samples_policy_logits_and_preserves_mode() -> None:
    step = _step()
    for parameter in step.model.policy.parameters():
        torch.nn.init.zeros_(parameter)
    head = step.model.policy[-1]
    assert isinstance(head, torch.nn.Linear)
    assert head.bias is not None
    head.bias.data[7] = 100.0
    actor = step.make_evaluation_actor()
    actor.reset(num_envs=2, device=torch.device("cpu"))

    action = actor.act(
        torch.zeros(2, step.env.observation_size),
        torch.zeros(2, dtype=torch.bool),
        generator=torch.Generator().manual_seed(0),
    )

    assert action.tolist() == [7, 7]
    assert step.model.training


def test_a_checkpoint_resumes_an_identical_run() -> None:
    step = _config(seed=3).make()
    step.train_step()
    # Deep-copied because the live step keeps mutating these tensors: a
    # shallow snapshot would be rewritten by the very update it is meant to
    # be compared against.
    saved = copy.deepcopy(step.state_dict())

    expected = float(step.train_step()["loss"])

    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert float(resumed.train_step()["loss"]) == expected


def test_finished_episodes_are_summarized_once() -> None:
    step = _step()
    step.env.state.player_health[:] = 0.0
    metrics = _metrics(step.train_step())
    assert float(metrics["episodes"]) > 0.0
    assert math.isfinite(float(metrics["episode_return"]))
    # The bank is cleared, so the next update reports only its own episodes.
    assert float(_metrics(step.train_step())["episodes"]) < float(metrics["episodes"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_steps", 0),
        ("num_epochs", 0),
        ("num_minibatches", 0),
        ("total_train_steps", 0),
        ("clip_epsilon", 0.0),
        ("discount", 1.5),
        ("trace_decay", -0.1),
    ],
)
def test_an_invalid_setting_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"positive|at least one|between zero and one"):
        _config(**{field: value}).make()


@pytest.mark.compute_jax_jit
def test_compiling_agrees_with_eager_to_float32_rounding() -> None:
    """Compiling changes the last bits, and nothing above them.

    Measured, not assumed: at this width the two paths differ by about 4e-9,
    because inductor fuses the first matmul into a different reduction order.
    So the compiled run is the SAME experiment -- but it is not bit-for-bit,
    which is why the golden pins ``compile=False`` rather than defaulting it.
    """

    def loss(*, compiled: bool) -> float:
        setting = PartialConfig(torch.compile, fullgraph=True) if compiled else None
        with torch_compiler_isolation():
            return float(_config(seed=5, compile=setting).make().train_step()["loss"])

    eager = loss(compiled=False)
    assert loss(compiled=True) == pytest.approx(eager, abs=1e-6)


def test_cuda_graphs_leave_a_cpu_step_eager() -> None:
    # Nothing to capture on a CPU, so the optimizer keeps its eager form -- the
    # one the golden was minted with.
    step = _config(cuda_graphs=True).make()
    assert isinstance(step, CraftaxTrainStep)
    group = step.optimizer.param_groups[0]
    assert isinstance(group["lr"], float)
    assert not group["capturable"]
    step.train_step()
    assert isinstance(group["lr"], float)


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_graphed_update_replays_the_eager_procedures_bit_for_bit() -> None:
    """The graphs change launches, not arithmetic -- across a checkpoint too.

    The eager twin keeps the graphed step's capturable Adam, which is the one
    numerical difference graphs bring; everything else must match exactly,
    including after a load replaces the optimizer state the graph addressed.
    """
    steps: list[CraftaxTrainStep] = []
    for step_class in (CraftaxTrainStep, _EagerProcedures):
        config = _config(cuda_graphs=True, seed=3)
        config.parallelism = NoParallel.Config(device="cuda")
        config.env.device = "cuda"
        config.env.num_envs = 8
        config.rollout_steps = 4
        config.num_minibatches = 2
        steps.append(step_class(config.copy_tree().finalize()))
    for step in steps:
        step.train_step()
        saved = copy.deepcopy(step.state_dict())
        step.train_step()
        step.load_state_dict(saved)
        step.train_step()
    graphed, eager = steps
    for mine, theirs in zip(
        graphed.model.parameters(),
        eager.model.parameters(),
        strict=True,
    ):
        assert torch.equal(mine, theirs)


class _EagerProcedures(CraftaxTrainStep):
    """The graphed step's procedures, called directly instead of replayed."""

    @override
    def _procedure(self, procedure: Callable[[], None]) -> Callable[[], None]:
        return procedure


def _metrics(result: TrainStepOutput) -> dict[str, float | Tensor]:
    """Read the optional diagnostics a completed update always carries."""
    return result.get("metrics", {})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
