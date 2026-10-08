"""Tests for the port's shared test support."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields
from typing import TYPE_CHECKING, cast

import threading

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
from priml.baselines.craftax.rollout import StepGraph, TorchPhiloxSampler
from priml.baselines.craftax.testing import FakeEnv, _golden_path
from priml.baselines.craftax.train_step import AgentWindows
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import TorchPPO
from priml.model.min_gru import TorchScan
from priml.testing import regenerate


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
        _assert_golden_without_skip(test_file=test_file, lines=["value 2"])
    golden = tmp_path / "testdata" / "probe_host.txt"
    assert golden.read_text(encoding="utf-8") == "value 2\n"


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
