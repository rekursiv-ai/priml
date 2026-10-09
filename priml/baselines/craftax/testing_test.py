"""Tests for the port's shared test support."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import hashlib
import threading

import numpy as np
import pytest
import torch

from priml.baselines.craftax import testing
from priml.baselines.craftax.experiments import exp000, exp002
from priml.baselines.craftax.game.jit import platform_key
from priml.baselines.craftax.game.state import ATN_DIM, OBS_SIZE
from priml.baselines.craftax.model import (
    MinGRUPolicy,
    packed_observation_embedding,
)
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.baselines.craftax.policies.encoder import BoardEncoder
from priml.baselines.craftax.policies.rnn import ActorCriticRNN
from priml.baselines.craftax.rollout import StepGraph, TorchPhiloxSampler
from priml.baselines.craftax.testing import FakeEnv, _golden_path
from priml.baselines.craftax.train_step import AgentWindows
from priml.baselines.craftax.world_model.feature import InitialWeights, Refill
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import TorchPPO
from priml.model.min_gru import TorchScan
from priml.testing import regenerate


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.craftax.env import CraftaxEnv
    from priml.baselines.craftax.experiments import CraftaxTrainLoop
    from priml.baselines.craftax.train_step import CraftaxTrainStep
    from priml.model.embedding import MultiHotEmbedding


def test_fake_env_draws_ids_from_the_policy_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FakeEnv's ids stay inside each field's vocabulary, as the embedding defines it.

    It carried its own copy of the vocabulary widths, so an embedding laid out
    any other way was fed ids past the end of its fields.
    """
    monkeypatch.setattr(testing, "multi_hot_embedding", _one_id_per_field)

    env = FakeEnv.make(num_envs=2, num_buffers=1, seed=0)

    embedding = _one_id_per_field(MinGRUPolicy.Config())
    ids = env.observations[:, : embedding.num_cells * len(embedding.offsets)]
    assert torch.equal(ids, torch.zeros_like(ids))


def test_fake_env_buffers_are_as_wide_as_the_env_and_the_policy_read() -> None:
    env = FakeEnv.make(num_envs=2, num_buffers=1, seed=0)

    assert env.observations.shape == (2, OBS_SIZE)
    assert env.observations.shape[1] == (
        testing.multi_hot_embedding(MinGRUPolicy.Config()).observation_size
    )
    assert env.action_mask.shape == (2, ATN_DIM)


def test_assert_golden_passes_a_matching_host_golden_and_fails_a_changed_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = _golden_dir(tmp_path, minted=True)
    _set_overwrite(monkeypatch, overwrite=False)
    _assert_golden_without_skip(test_file=test_file, lines=["value 1"])
    with pytest.raises(AssertionError, match="probe_host changed"):
        _assert_golden_without_skip(test_file=test_file, lines=["value 2"])


def test_assert_golden_skips_a_host_without_its_golden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = _golden_dir(tmp_path, minted=False)
    _set_overwrite(monkeypatch, overwrite=False)
    with pytest.raises(pytest.skip.Exception, match="no probe_host golden"):
        testing.assert_golden(
            test_file=test_file,
            name="probe",
            lines=["value 1"],
            host="host",
        )


@pytest.mark.parametrize("minted", [False, True])
def test_golden_overwrite_writes_the_host_golden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minted: bool,
) -> None:
    test_file = _golden_dir(tmp_path, minted=minted)
    _set_overwrite(monkeypatch, overwrite=True)
    # A golden minted from nothing fails once, so a reviewer sees it.
    outcome = (
        nullcontext()
        if minted
        else pytest.raises(AssertionError, match="Missing golden regenerated")
    )
    with outcome:
        _assert_golden_without_skip(test_file=test_file, lines=["value 2", "value 3"])
    golden = tmp_path / "testdata" / "probe_host.txt"
    assert golden.read_text(encoding="utf-8") == "value 2\nvalue 3\n"


def test_a_changed_golden_fails_with_its_name_and_unified_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A golden with no host is read from ``<name>.txt``; the message diffs it against now."""
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "probe.txt").write_text("value 1\nvalue 2\n", encoding="utf-8")
    _set_overwrite(monkeypatch, overwrite=False)
    try:
        testing.assert_golden(
            test_file=str(tmp_path / "probe_test.py"),
            name="probe",
            lines=["value 1", "value 3"],
        )
    except AssertionError as changed:
        message = str(changed)
    except pytest.skip.Exception as skip:
        pytest.fail(f"assert_golden skipped: {skip}")
    else:
        pytest.fail("a changed golden passed")
    assert message == (
        "probe changed:\n--- golden\n+++ now\n@@ -1,2 +1,2 @@\n value 1\n-value 2\n+value 3"
    )


def test_a_fake_env_starts_zeroed_and_steps_each_buffer_from_its_own_seed() -> None:
    """Buffer ``b`` draws from ``seed + b``: buffer 1 of seed 5 is buffer 0 of seed 6."""
    env = FakeEnv.make(num_envs=4, num_buffers=2, seed=5)
    twin = FakeEnv.make(num_envs=4, num_buffers=2, seed=6)
    assert (env.rewards.shape, env.terminals.shape, env.actions.shape) == (
        (4,),
        (4,),
        (4, 1),
    )
    assert not env.rewards.any()
    assert not env.terminals.any()
    assert not env.actions.any()
    assert env.stepped == [[], []]
    assert torch.equal(env.observations[2:], twin.observations[:2])
    assert torch.equal(env.action_mask[2:], twin.action_mask[:2])
    env.actions[2:] = 7.0
    env.step_buffer(1)
    twin.step_buffer(0)
    assert [len(actions) for actions in env.stepped] == [0, 1]
    # ``[rows, 1]``: the env's actions, as ``FakeEnv.make`` lays them out.
    assert torch.equal(env.stepped[1][0], torch.full((2, 1), 7.0))
    assert torch.equal(env.rewards[2:], twin.rewards[:2])
    assert torch.equal(env.terminals[2:], twin.terminals[:2])
    assert torch.equal(env.observations[2:], twin.observations[:2])
    assert not env.rewards[:2].any()


def test_smoke_feature_is_a_float32_refill_over_the_smoke_world_model() -> None:
    config = testing.smoke_feature()
    weights, history = config.weights, config.history
    assert isinstance(weights, InitialWeights.Config)
    assert weights.experiment.endswith("world_model.experiments.exp_smoke")
    assert isinstance(history, Refill.Config)
    assert (history.t_max, history.keep) == (16, 4)
    assert (config.layers, config.hook_interval, config.dtype) == (1, 2, torch.float32)


@pytest.mark.parametrize(
    "policy",
    [testing.tiny_policy, testing.tiny_board_policy],
    ids=["multi_hot", "board"],
)
def test_packed_observations_hold_in_range_ids_then_scalars(
    policy: Callable[[], MinGRUPolicy.Config],
) -> None:
    config = policy()
    observations = testing.packed_observations(config, batch=3, time=2, seed=1)
    assert observations.shape == (3, 2, config.observation_size)
    assert torch.equal(
        observations,
        testing.packed_observations(config, batch=3, time=2, seed=1),
    )
    one_step = testing.packed_observations(config, batch=3, seed=1)
    assert one_step.shape == (3, config.observation_size)
    encoder = config.embedding
    if isinstance(encoder, BoardEncoder.Config):
        actions = observations[..., -1]
        assert bool((actions == actions.floor()).all())
        assert int(actions.max()) < encoder.status.embedding.channels_in
        embedding = encoder.cells
    else:
        embedding = testing.multi_hot_embedding(config)
    ids = observations[..., : embedding.num_cells * len(embedding.offsets)]
    assert bool((ids >= 0).all())
    assert int(ids.max()) < embedding.channels_in


def test_tiny_exp000_step_is_exp000_on_the_gpu_at_test_size() -> None:
    config = testing.tiny_exp000_step()
    model, windows = config.model, config.learner
    assert isinstance(model, MinGRUPolicy.Config)
    assert isinstance(windows, AgentWindows.Config)
    assert config.checkpoint is None
    assert (model.channels_hidden, model.num_layers) == (8, 2)
    assert config.parallelism.device == "cuda"
    assert windows.minibatch_size == 16
    assert (config.rollout.horizon, config.train_budget_steps) == (8, 4)
    assert config.env.num_envs == testing.tiny_env().num_envs
    assert type(config.sampler) is type(exp000().step.sampler)


def test_digests_name_the_dtype_shape_and_bytes() -> None:
    array = np.arange(6, dtype=np.int32)
    tensor = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    sha = hashlib.sha256(array.tobytes()).hexdigest()
    assert testing.digest(array) == f"<i4 (6,) {sha}"
    assert testing.digest(tensor) == f"torch.int32 (2, 3) {sha}"
    assert testing.fp32(1.0) == "0x3f800000 1.0"
    assert testing.fp32(torch.tensor(-0.5)) == "0xbf000000 -0.5"


def test_env_digests_cover_the_worlds_streams_stats_and_buffers() -> None:
    record = np.zeros(2, dtype=[("a", np.int32)])
    env = SimpleNamespace(
        states=record,
        rngs=np.zeros(2, dtype=np.uint32),
        stats=record,
        observations=torch.zeros(2, 3),
        action_mask=torch.ones(2, 3, dtype=torch.uint8),
        rewards=torch.zeros(2),
        terminals=torch.zeros(2),
        # ``[rows, 1]``: the env's actions, as ``CraftaxEnv`` lays them out.
        actions=torch.zeros(2, 1),
    )
    lines = testing.env_digests(cast("CraftaxEnv", env))
    assert [line.split()[0] for line in lines] == [
        "states",
        "rngs",
        "stats",
        "observations",
        "action_mask",
        "rewards",
        "terminals",
        "actions",
    ]
    assert lines[0] == f"states {testing.digest(record.view(np.uint8))}"


def test_optimizer_state_names_each_tensor_by_its_parameter() -> None:
    torch.manual_seed(0)
    model = testing.tiny_policy(dtype=torch.float32).make()
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.SGD(
        [parameters["proj_out.weight"], parameters["proj_in.weight"]],
        lr=0.1,
        momentum=0.9,
    )
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = testing.optimizer_state(model, optimizer, "momentum_buffer")
    assert list(state) == ["proj_out.weight", "proj_in.weight"]
    assert torch.equal(
        state["proj_in.weight"],
        torch.ones_like(parameters["proj_in.weight"]),
    )


@pytest.mark.parametrize(
    ("host", "file_name"),
    [("", "probe.txt"), ("h", "probe_h.txt")],
)
def test_a_golden_is_named_for_its_host_in_testdata_beside_its_test(
    tmp_path: Path,
    host: str,
    file_name: str,
) -> None:
    # Compared as a path, not found on disk: macOS matches `TESTDATA` to `testdata`.
    path = _golden_path(
        test_file=str(tmp_path / "probe_test.py"),
        name="probe",
        host=host,
    )
    assert path == tmp_path.resolve() / "testdata" / file_name


def test_a_golden_for_another_host_does_not_satisfy_this_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = _golden_dir(tmp_path, minted=True)
    _set_overwrite(monkeypatch, overwrite=False)
    with pytest.raises(pytest.skip.Exception, match="no probe_other golden"):
        testing.require_golden(test_file=test_file, name="probe", host="other")


def _one_id_per_field(config: MinGRUPolicy.Config) -> MultiHotEmbedding.Config:
    """Return the packed layout with a table unlike the policy's: one id per field."""
    del config
    embedding = packed_observation_embedding()
    embedding.offsets = (0, 1, 2, 3, 4, 5, 6, 7)
    embedding.channels_in = 8
    return embedding


def test_a_portable_checkpoint_holds_the_filled_policys_weights_as_fp32_masters(
    tmp_path: Path,
) -> None:
    """The file is the fill's weights by name, fp32, so a step started from it holds them."""
    config = testing.tiny_policy()
    path = tmp_path / "masters.pt"
    assert testing.portable_checkpoint(config, path, seed=5) == path
    masters = from_plain(
        cast("object", torch.load(path, weights_only=True)),
        dict[str, torch.Tensor],
    )
    filled = config.make()
    testing.fill_portable(filled, seed=5)
    expected = dict(filled.named_parameters())
    assert masters.keys() == expected.keys()
    for name, weight in expected.items():
        assert masters[name].dtype == torch.float32, name
        assert torch.equal(masters[name], weight.detach().float()), name
    other = testing.portable_checkpoint(config, tmp_path / "other.pt", seed=6)
    assert not torch.equal(
        from_plain(
            cast("object", torch.load(other, weights_only=True)),
            dict[str, torch.Tensor],
        )["proj_in.weight"],
        masters["proj_in.weight"],
    )


def _tiny_rnn() -> ActorCriticRNN.Config:
    """Return exp005's GRU actor-critic at test size."""
    config = ActorCriticRNN.Config()
    config.observation_size = 6
    config.channels_hidden = 4
    config.num_layers = 1
    return config


def _tiny_towers() -> ActorCritic.Config:
    """Return exp003's actor-critic towers at test size."""
    config = ActorCritic.Config()
    config.observation_size = 6
    config.channels_hidden = 4
    config.num_layers = 1
    return config


def _sole_feature() -> MinGRUPolicy.Config:
    """Return the sole-feature policy over a 3-wide feature."""
    return testing.sole_feature_policy(feature_width=3)


@pytest.mark.parametrize(
    "policy",
    [_tiny_rnn, _tiny_towers, _sole_feature, testing.tiny_board_policy],
    ids=["gru", "actor_critic", "sole_feature", "board"],
)
def test_fill_portable_overwrites_every_parameter_whatever_the_init(
    policy: Callable[
        [],
        ActorCriticRNN.Config | ActorCritic.Config | MinGRUPolicy.Config,
    ],
) -> None:
    """Policies built under different torch seeds hold the same weights once filled.

    It filled only a MinGRU's stages, so the GRU's cell and heads kept their
    orthogonal init, whose ``randn`` differs by host: the GRU's golden
    differed between macOS and x86.
    """
    config = policy()
    torch.manual_seed(0)
    first = config.make()
    torch.manual_seed(1)
    second = config.make()
    testing.fill_portable(first, seed=3)
    testing.fill_portable(second, seed=3)
    named = list(second.named_parameters())
    assert sorted(name for name, _ in testing.forward_parameters(first)) == sorted(
        name for name, _ in named
    )
    for (name, filled), (_, twin) in zip(first.named_parameters(), named, strict=True):
        assert torch.equal(filled, twin), name


@pytest.mark.parametrize(("minted", "overwrite"), [(True, False), (False, True)])
def test_require_golden_runs_on_with_a_golden_or_to_mint_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minted: bool,
    overwrite: bool,
) -> None:
    test_file = _golden_dir(tmp_path, minted=minted)
    _set_overwrite(monkeypatch, overwrite=overwrite)
    # A skip here would report this test skipped, not failed.
    try:
        testing.require_golden(test_file=test_file, name="probe", host="host")
    except pytest.skip.Exception as skip:
        pytest.fail(f"require_golden skipped: {skip}")


def test_require_golden_skips_before_the_work_without_a_golden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = _golden_dir(tmp_path, minted=False)
    _set_overwrite(monkeypatch, overwrite=False)
    with pytest.raises(pytest.skip.Exception, match="no probe_host golden"):
        testing.require_golden(test_file=test_file, name="probe", host="host")


def test_gpu_key_skips_without_a_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
        testing.gpu_key()


def test_gpu_key_names_the_device_model_then_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: " NVIDIA H200 (SXM) ")
    assert testing.gpu_key() == f"nvidia-h200-sxm_{platform_key()}"


@pytest.mark.parametrize(
    ("tiny", "recipe"),
    [(testing.tiny_train_step, exp000), (testing.tiny_exp002_step, exp002)],
)
def test_a_tiny_step_is_its_recipe_in_torch_forms_at_test_size(
    tiny: Callable[[], CraftaxTrainStep.Config],
    recipe: Callable[[], CraftaxTrainLoop],
) -> None:
    config, reference = tiny(), recipe().step
    assert config.checkpoint is None
    assert config.parallelism.device == "cpu"
    assert (config.rollout.horizon, config.train_budget_steps) == (4, 3)
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert (model.channels_hidden, model.num_layers) == (8, 2)
    assert isinstance(model.block.scan, TorchScan.Config)
    windows, reference_windows = config.learner, reference.learner
    assert isinstance(windows, AgentWindows.Config)
    assert isinstance(reference_windows, AgentWindows.Config)
    assert windows.minibatch_size == 8
    # The torch forms take the recipe's coefficients and seed, not their defaults.
    assert type(windows.objective) is TorchPPO.Config
    for name in (f.name for f in fields(TorchPPO.Config)):
        assert getattr(windows.objective, name) == getattr(
            reference_windows.objective,
            name,
        ), name
    sampler, reference_sampler = config.sampler, reference.sampler
    assert isinstance(sampler, TorchPhiloxSampler.Config)
    assert type(sampler) is TorchPhiloxSampler.Config
    assert isinstance(reference_sampler, TorchPhiloxSampler.Config)
    assert sampler.seed == reference_sampler.seed
    env = config.env
    assert (env.num_envs, env.num_buffers, env.threads_per_buffer) == (4, 2, 1)
    assert env.rules == reference.env.rules
    assert type(env.restart) is type(reference.env.restart)


def test_the_pipeline_replays_every_rollout_step_in_float64_on_its_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollout buffer steps on a worker thread, where torch's dispatch mode is unset.

    ``[1e8, 1, -1e8]`` sums to 0 in float32 and to 1 through float64 (measured
    on glibc x86 and macOS arm64), so the sum names the mode a replay ran in.
    """
    sums: list[float] = []

    def replay(graph: StepGraph) -> None:
        del graph
        sums.append(float(torch.tensor([1e8, 1.0, -1e8]).sum()))

    monkeypatch.setattr(StepGraph, "replay", replay)
    graph = cast("StepGraph", object())
    with testing.host_agnostic_pipeline():
        _on_a_thread(StepGraph.replay, graph)
    _on_a_thread(StepGraph.replay, graph)
    assert sums == [1.0, 0.0]


def _on_a_thread(replay: Callable[[StepGraph], None], graph: StepGraph) -> None:
    """Run ``replay(graph)`` on a new thread and wait for it."""
    thread = threading.Thread(target=replay, args=(graph,))
    thread.start()
    thread.join()


def _golden_dir(tmp_path: Path, *, minted: bool) -> str:
    """Return a test file's path beside ``testdata/``, holding ``probe_host`` if minted."""
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    if minted:
        (testdata / "probe_host.txt").write_text("value 1\n", encoding="utf-8")
    return str(tmp_path / "probe_test.py")


def _set_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overwrite: bool,
) -> None:
    """Make ``--regenerate-golden`` read as ``overwrite`` for this test."""
    regenerate.override(monkeypatch, golden=overwrite)


def _assert_golden_without_skip(
    *,
    test_file: str,
    lines: list[str],
) -> None:
    """Run ``assert_golden`` on ``probe_host``; a skip would hide a broken golden."""
    try:
        testing.assert_golden(
            test_file=test_file,
            name="probe",
            lines=lines,
            host="host",
        )
    except pytest.skip.Exception as skip:
        pytest.fail(f"assert_golden skipped: {skip}")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
