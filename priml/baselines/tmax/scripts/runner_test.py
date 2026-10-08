"""Tests for the thin TMax rollout driver."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import importlib
import json
import os

import pytest
import torch

from priml.baselines.tmax.scripts import (
    runner as runner_module,
    upstream,
)
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RayRuntimePolicy,
    RolloutWriter,
    RunnerConfig,
    UpstreamApi,
    WeightSync,
    build_session,
    collect_update,
    prime_prompts,
    run,
    run_updates,
    serialize_rollout_records,
    vllm_weight_name_mapper,
)


class _Rpc:
    """A bound engine method that records the call ``.remote()`` carries."""

    def __init__(self, record: Callable[..., object]) -> None:
        self._record = record

    def remote(self, *args: object, **kwargs: object) -> object:
        """Invoke the recorder exactly as Ray would on the engine."""
        return self._record(*args, **kwargs)


class _Engine:
    """Records the RPCs ``WeightSync`` issues, in upstream's argument shapes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.init_weight_transfer_engine = _Rpc(self._init)
        self.sleep = _Rpc(self._sleep)
        self.wake_up = _Rpc(self._wake)
        self.update_weights = _Rpc(self._update)
        self.set_model_step = _Rpc(self._set_model_step)

    def _sleep(self) -> None:
        self.events.append(("sleep", None))

    def _wake(self) -> None:
        self.events.append(("wake", None))

    def _set_model_step(self, step: int) -> None:
        self.events.append(("step", step))

    def _init(self, request: object) -> None:
        self.events.append(("init", request))

    def _update(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        *,
        packed: bool,
    ) -> None:
        self.events.append(("update", (names, dtype_names, shapes, packed)))

    def names_sent(self) -> list[str]:
        """Return the parameter names of the most recent ``update_weights``."""
        for name, payload in reversed(self.events):
            if name == "update":
                return cast(tuple[list[str], ...], payload)[0]
        return []


class _ActorManager:
    def __init__(self, events: list[tuple[str, object]] | None = None) -> None:
        self.events = events if events is not None else []
        self.set_should_stop = _Rpc(self._set_should_stop)

    def _set_should_stop(self, value: bool) -> list[object]:
        self.events.append(("gate", value))
        return []


def _init_request(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


def _trainer_init(info: object) -> dict[str, object]:
    return {"info": info}


def _ray_get(refs: object) -> list[object]:
    return list(cast("Iterable[object]", refs))


def _is_initialized() -> bool:
    return True


def _sync_api(sent: list[tuple[str, torch.Tensor]]) -> UpstreamApi:
    """Build the minimal upstream surface ``WeightSync`` touches."""

    def send_weights(*, iterator: object, trainer_args: object) -> None:
        del trainer_args
        sent.extend(cast("Iterable[tuple[str, torch.Tensor]]", iterator))

    weight_transfer = (
        SimpleNamespace(WeightTransferInitRequest=_init_request),
        SimpleNamespace(
            NCCLWeightTransferEngine=SimpleNamespace(
                trainer_init=_trainer_init,
                trainer_send_weights=send_weights,
            ),
            NCCLTrainerSendWeightsArgs=_init_request,
        ),
    )
    return UpstreamApi(
        ray=SimpleNamespace(get=_ray_get, is_initialized=_is_initialized),
        ray_queue=None,
        data_loader=ModuleType("data"),
        dataset_transformation=ModuleType("data"),
        environments_utils=ModuleType("data"),
        grpo_fast=ModuleType("data"),
        grpo_utils=ModuleType("data"),
        ground_truth_utils=ModuleType("data"),
        model_utils=ModuleType("data"),
        rl_utils=ModuleType("data"),
        utils=ModuleType("data"),
        vllm_utils=ModuleType("data"),
        weight_transfer=weight_transfer,
    )


def _sync_session(engines: list[_Engine], api: UpstreamApi) -> CollectionSession:
    return CollectionSession(
        api=api,
        tokenizer=None,
        dataset=None,
        iter_dataloader=None,
        inference_queue=None,
        prompt_queue=None,
        generation_config=None,
        base_env_config=None,
        model_dims=None,
        engines=list(engines),
        pools={},
        started_ray=False,
        actor_manager=_ActorManager(),
    )


def test_upstream_import_enables_released_reset_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Upstream imports with the released reset-failure-zero-reward flag on.
    observed: list[str | None] = []
    monkeypatch.setenv("SWERL_RESET_FAILURE_ZERO_REWARD", "0")

    def verified(_root: Path) -> Path:
        return tmp_path

    monkeypatch.setattr(upstream, "verify_checkout", verified)
    monkeypatch.setattr(upstream, "open_instruct", verified)

    def import_module(name: str) -> ModuleType:
        if name.endswith("vllm_utils"):
            observed.append(os.environ.get("SWERL_RESET_FAILURE_ZERO_REWARD"))
        if name.startswith("vllm.distributed.weight_transfer"):
            raise ImportError
        return ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    runner_module.load_upstream_api(tmp_path)
    assert observed == ["1"]


def test_runtime_policy_is_typed_printable_and_allowlisted() -> None:
    # The policy prints itself and carries exactly the released env allowlist.
    policy = RayRuntimePolicy()
    assert "vllm_use_v1='1'" in repr(policy)
    assert policy.env_vars() == {
        "SWERL_RESET_FAILURE_ZERO_REWARD": "1",
        "VLLM_USE_V1": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "NCCL_CUMEM_ENABLE": "0",
    }


def test_ray_init_forwards_only_the_runtime_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ray workers get the allowlist only; ambient credentials never travel.
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-forward")

    def init(**kwargs: object) -> None:
        calls.append(kwargs)

    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=init,
    )
    assert runner_module._initialize_ray(RunnerConfig(), ray)
    assert calls == [
        {"runtime_env": {"env_vars": RayRuntimePolicy().env_vars()}},
    ]
    assert "AWS_SECRET_ACCESS_KEY" not in cast(
        dict[str, str],
        cast(dict[str, object], calls[0]["runtime_env"])["env_vars"],
    )


def test_existing_ray_requires_a_matching_runtime_policy() -> None:
    # An already-running Ray job must carry the same runtime policy.
    ray = SimpleNamespace(
        is_initialized=lambda: True,
        get_runtime_context=lambda: SimpleNamespace(
            runtime_env={"env_vars": RayRuntimePolicy().env_vars()},
        ),
    )
    assert not runner_module._initialize_ray(RunnerConfig(), ray)
    ray.get_runtime_context = lambda: SimpleNamespace(runtime_env={})
    with pytest.raises(RuntimeError, match="already initialized"):
        runner_module._initialize_ray(RunnerConfig(), ray)


def test_checkpoint_must_match_on_every_live_ray_node(tmp_path: Path) -> None:
    # vLLM actors load this path remotely; a driver-local checkpoint is not enough.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    expected = runner_module._checkpoint_inventory(str(checkpoint))

    class _Probe:
        def options(self, **kwargs: object) -> object:
            resources = cast(dict[str, float], kwargs["resources"])
            node = next(iter(resources)).removeprefix("node:")

            def remote(path: str) -> tuple[str, str]:
                return node, path

            return SimpleNamespace(remote=remote)

    nodes = [
        {"Alive": True, "NodeManagerAddress": "10.0.0.1"},
        {"Alive": True, "NodeManagerAddress": "10.0.0.2"},
        {"Alive": False, "NodeManagerAddress": "10.0.0.3"},
    ]

    def get(refs: object) -> list[object]:
        return [
            expected if node == "10.0.0.1" else None
            for node, _path in cast("Iterable[tuple[str, str]]", refs)
        ]

    def remote(target: object) -> _Probe:
        if target is runner_module._checkpoint_inventory:
            return _Probe()
        pytest.fail("unexpected Ray task")

    ray = SimpleNamespace(nodes=lambda: nodes, remote=remote, get=get)
    with pytest.raises(RuntimeError, match=r"10\.0\.0\.2.*shared storage"):
        runner_module._require_checkpoint_on_ray_nodes(checkpoint, ray)


def test_checkpoint_probe_accepts_the_same_inventory_on_every_node(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    expected = runner_module._checkpoint_inventory(str(checkpoint))
    assert expected is not None

    class _Probe:
        def options(self, **_kwargs: object) -> object:
            def remote(_path: str) -> tuple[tuple[str, int, str], ...]:
                return expected

            return SimpleNamespace(remote=remote)

    def remote(_target: object) -> _Probe:
        return _Probe()

    ray = SimpleNamespace(
        nodes=lambda: [{"Alive": True, "NodeManagerAddress": "10.0.0.1"}],
        remote=remote,
        get=_ray_get,
    )
    runner_module._require_checkpoint_on_ray_nodes(checkpoint, ray)


def test_checkpoint_probe_rejects_same_size_different_contents(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text("left", encoding="utf-8")
    expected = runner_module._checkpoint_inventory(str(checkpoint))
    assert expected is not None
    config.write_text("rite", encoding="utf-8")
    different = runner_module._checkpoint_inventory(str(checkpoint))
    assert different is not None
    assert [(name, size) for name, size, _digest in expected] == [
        (name, size) for name, size, _digest in different
    ]
    assert expected != different
    config.write_text("left", encoding="utf-8")

    class _Probe:
        def options(self, **_kwargs: object) -> object:
            def remote(_path: str) -> tuple[tuple[str, int, str], ...]:
                return different

            return SimpleNamespace(remote=remote)

    def remote(_target: object) -> _Probe:
        return _Probe()

    ray = SimpleNamespace(
        nodes=lambda: [{"Alive": True, "NodeManagerAddress": "10.0.0.1"}],
        remote=remote,
        get=_ray_get,
    )
    with pytest.raises(RuntimeError, match=r"differs.*10\.0\.0\.1"):
        runner_module._require_checkpoint_on_ray_nodes(checkpoint, ray)


@dataclass
class _FakeBatch:
    """The upstream batch fields used by the serializer."""

    queries: Sequence[Sequence[int]]
    scores: Sequence[float] | None
    datasets: Sequence[object]
    ground_truths: Sequence[object]


@dataclass
class _FakeResult:
    """The upstream generation result fields used by the serializer."""

    responses: Sequence[Sequence[int]]
    finish_reasons: Sequence[str]
    masks: Sequence[Sequence[int]]
    logprobs: Sequence[Sequence[float]] | None
    request_info: object


def _batch() -> _FakeBatch:
    return _FakeBatch(
        queries=[[1, 2], [3]],
        scores=[0.0, 1.0],
        datasets=["task", "task"],
        ground_truths=[["x"], ["y"]],
    )


def _result() -> _FakeResult:
    return _FakeResult(
        responses=[[4, 5], [6]],
        finish_reasons=["stop", "length"],
        masks=[[1, 0], [1]],
        logprobs=[[-0.2, -0.3], [-0.4]],
        request_info=SimpleNamespace(),
    )


def _request_info(_request_info: object, index: int) -> dict[str, object]:
    return {"sample": index}


def _api() -> UpstreamApi:
    """Build an upstream-shaped fake without importing Ray."""
    modules = {name: ModuleType(name) for name in ("data", "rl")}
    modules["rl"].__dict__["_get_request_info_for_sample"] = _request_info
    return UpstreamApi(
        ray=None,
        ray_queue=None,
        data_loader=modules["data"],
        dataset_transformation=modules["data"],
        environments_utils=modules["data"],
        grpo_fast=modules["data"],
        grpo_utils=modules["data"],
        ground_truth_utils=modules["data"],
        model_utils=modules["data"],
        rl_utils=modules["rl"],
        utils=modules["data"],
        vllm_utils=modules["data"],
        weight_transfer=None,
    )


def test_serialization_matches_tmax_fields_and_adds_tool_mask() -> None:
    # Serialization keeps TMax's schema and adds the policy-versus-sandbox mask.
    records = serialize_rollout_records(
        _batch(),
        _result(),
        step=7,
        advantages=[-0.5, 0.5],
        num_samples_per_prompt=2,
        request_info_for_sample=_request_info,
    )

    assert records == [
        {
            "step": 7,
            "sample_idx": 0,
            "prompt_idx": 0,
            "prompt_tokens": [1, 2],
            "response_tokens": [4, 5],
            "reward": 0.0,
            "advantage": -0.5,
            "finish_reason": "stop",
            "dataset": "task",
            "ground_truth": ["x"],
            "request_info": {"sample": 0},
            "logprobs": [-0.2, -0.3],
            "tool_mask": [1, 0],
        },
        {
            "step": 7,
            "sample_idx": 1,
            "prompt_idx": 0,
            "prompt_tokens": [3],
            "response_tokens": [6],
            "reward": 1.0,
            "advantage": 0.5,
            "finish_reason": "length",
            "dataset": "task",
            "ground_truth": ["y"],
            "request_info": {"sample": 1},
            "logprobs": [-0.4],
            "tool_mask": [1],
        },
    ]


def test_serialization_refuses_misaligned_masks() -> None:
    # The tool mask must cover every response token exactly.
    result = _result()
    result.masks = [[1], [1]]
    with pytest.raises(ValueError, match="equal lengths"):
        serialize_rollout_records(
            _batch(),
            result,
            step=0,
            advantages=[-0.5, 0.5],
            num_samples_per_prompt=2,
            request_info_for_sample=_request_info,
        )


def test_serialization_refuses_missing_behavior_logprobs() -> None:
    # Without behavior logprobs there is no DPPO ratio to train on.
    result = _result()
    result.logprobs = None
    with pytest.raises(ValueError, match="no logprobs"):
        serialize_rollout_records(
            _batch(),
            result,
            step=0,
            advantages=[-0.5, 0.5],
            num_samples_per_prompt=2,
            request_info_for_sample=_request_info,
        )


def test_mocked_orchestration_collects_and_shards_updates(tmp_path: Path) -> None:
    # The fixed-corpus loop collects each update and appends its shard.
    api = _api()
    session = CollectionSession(
        api=api,
        tokenizer=None,
        dataset=None,
        iter_dataloader=None,
        inference_queue=None,
        prompt_queue=None,
        generation_config=None,
        base_env_config=None,
        model_dims=None,
        engines=[],
        pools={},
        started_ray=False,
    )
    calls: list[int] = []

    def collect(
        _session: CollectionSession,
        _config: RunnerConfig,
        step: int,
    ) -> tuple[_FakeBatch, _FakeResult, list[float]]:
        calls.append(step)
        return _batch(), _result(), [-0.5, 0.5]

    paths = run_updates(
        RunnerConfig(
            output_dir=tmp_path,
            run_name="mock",
            num_updates=2,
            num_prompts=1,
            samples_per_prompt=2,
        ),
        session,
        collect_update=collect,
    )

    assert calls == [0, 1]
    assert paths == [tmp_path / "mock_rollouts_000000.jsonl"] * 2
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[2])["step"] == 1
    assert json.loads(lines[2])["tool_mask"] == [1, 0]


def test_fixed_collection_recovers_an_incomplete_final_update(
    tmp_path: Path,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(
        0,
        [
            {"step": 0, "sample_idx": 0},
            {"step": 0, "sample_idx": 1},
        ],
    )
    writer.write(1, [{"step": 1, "sample_idx": 0}])
    calls: list[int] = []

    def collect(
        _session: CollectionSession,
        _config: RunnerConfig,
        step: int,
    ) -> tuple[_FakeBatch, _FakeResult, list[float]]:
        calls.append(step)
        return _batch(), _result(), [-0.5, 0.5]

    config = RunnerConfig(
        output_dir=tmp_path,
        run_name="mock",
        num_updates=2,
        num_prompts=1,
        samples_per_prompt=2,
    )
    run_updates(config, _sync_session([], _api()), collect_update=collect)

    assert calls == [1]
    records = [
        cast("dict[str, object]", json.loads(line))
        for line in (tmp_path / "mock_rollouts_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(record["step"], record["sample_idx"]) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_fixed_collection_refuses_a_non_suffix_partial_update(tmp_path: Path) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(0, [{"step": 0, "sample_idx": 0}])
    writer.write(
        1,
        [
            {"step": 1, "sample_idx": 0},
            {"step": 1, "sample_idx": 1},
        ],
    )
    writer = RolloutWriter(tmp_path, "mock")

    with pytest.raises(ValueError, match="complete steps follow"):
        writer.recover_incomplete_fixed_tail(records_per_update=2)


def test_rollout_writer_refuses_duplicates_and_detects_partial_steps(
    tmp_path: Path,
) -> None:
    # Rewriting a staged step is refused; short steps are flagged incomplete.
    writer = RolloutWriter(tmp_path, "mock")
    records = [
        {
            "step": 0,
            "sample_idx": 0,
            "prompt_idx": 0,
        },
    ]
    writer.write(0, records)
    assert writer.incomplete_steps(2) == {0}
    with pytest.raises(ValueError, match="already staged"):
        writer.write(0, records)


def test_rollout_writer_resume_indexes_complete_steps(tmp_path: Path) -> None:
    # A fresh writer rebuilds the complete/incomplete index from disk.
    writer = RolloutWriter(tmp_path, "mock")
    records = [
        {"step": 0, "sample_idx": 0},
        {"step": 0, "sample_idx": 1},
    ]
    writer.write(0, records)
    resumed = RolloutWriter(tmp_path, "mock")
    assert resumed.completed_steps(2) == {0}
    assert resumed.incomplete_steps(2) == set()


def test_rollout_writer_requires_the_exact_sample_index_set(tmp_path: Path) -> None:
    # A step is complete only with every sample index exactly once.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(
        0,
        [
            {"step": 0, "sample_idx": 0},
            {"step": 0, "sample_idx": 2},
        ],
    )
    assert writer.completed_steps(2) == set()
    assert writer.incomplete_steps(2) == {0}


def test_rollout_writer_rejects_duplicates_already_on_disk(tmp_path: Path) -> None:
    # A shard holding duplicate samples is corruption at load time.
    path = tmp_path / "mock_rollouts_000000.jsonl"
    path.write_text(
        '{"step": 0, "sample_idx": 0}\n{"step": 0, "sample_idx": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate staged rollout"):
        RolloutWriter(tmp_path, "mock")


def test_live_writer_atomically_commits_records_and_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A live step publishes by atomic renames: intent first, then the marker.
    replacements: list[tuple[Path, Path]] = []
    replace_file = os.replace

    def replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        replace_file(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(
        4,
        [{"step": 4, "sample_idx": 0}, {"step": 4, "sample_idx": 1}],
        prompt_loader_state_after={"position": 9},
    )

    assert replacements == [
        (
            tmp_path / "mock_live_step_00000004.pending.json.tmp",
            tmp_path / "mock_live_step_00000004.pending.json",
        ),
        (
            tmp_path / "mock_live_step_00000004.json.tmp",
            tmp_path / "mock_live_step_00000004.json",
        ),
    ]
    resumed = RolloutWriter(tmp_path, "mock")
    assert resumed.committed_live_steps(2) == {4}
    assert resumed.live_cursor_after(4, 2) == {"position": 9}
    marker = cast(
        "dict[str, object]",
        json.loads(
            (tmp_path / "mock_live_step_00000004.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    digest = marker["records_sha256"]
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_live_writer_returns_an_indexed_step_without_reparsing(tmp_path: Path) -> None:
    # Resume reuses the index RolloutWriter already built at startup.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(
        4,
        [{"step": 4, "sample_idx": 1}, {"step": 4, "sample_idx": 0}],
        prompt_loader_state_after={"position": 9},
    )
    assert [record["sample_idx"] for record in writer.records_for_step(4, 2)] == [
        0,
        1,
    ]
    with pytest.raises(ValueError, match="each sample index"):
        writer.records_for_step(4, 3)


def test_live_writer_recovers_only_the_uncommitted_physical_suffix(
    tmp_path: Path,
) -> None:
    # Recovery truncates only the bytes past the committed frontier.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(
        0,
        [{"step": 0, "sample_idx": 0}, {"step": 0, "sample_idx": 1}],
        prompt_loader_state_after={"position": 2},
    )
    shard = tmp_path / "mock_rollouts_000000.jsonl"
    committed_size = shard.stat().st_size
    pending = tmp_path / "mock_live_step_00000001.pending.json"
    pending.write_text(
        json.dumps(
            {
                "version": 1,
                "step": 1,
                "shard": shard.name,
                "byte_offset": committed_size,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    with shard.open("ab") as handle:
        handle.write(b'{"step": 1, "sample_idx": 0}\n{"step": 1')

    resumed = RolloutWriter(tmp_path, "mock")
    assert resumed.recover_uncommitted_live_tail(1, 2) == {1}
    assert shard.stat().st_size == committed_size
    assert not pending.exists()
    assert resumed.committed_live_steps(2) == {0}
    resumed.write_live(
        1,
        [{"step": 1, "sample_idx": 0}, {"step": 1, "sample_idx": 1}],
        prompt_loader_state_after={"position": 4},
    )
    assert resumed.committed_live_steps(2) == {0, 1}


def test_live_writer_refuses_to_recover_before_the_durable_frontier(
    tmp_path: Path,
) -> None:
    # Recovery needs the previous live commit marker, not just shard bytes.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(0, [{"step": 0, "sample_idx": 0}])

    resumed = RolloutWriter(tmp_path, "mock")
    with pytest.raises(ValueError, match="before the durable frontier"):
        resumed.recover_uncommitted_live_tail(1, 2)


def test_live_writer_rejects_an_intent_outside_record_boundaries(
    tmp_path: Path,
) -> None:
    # A write intent must point at a record boundary in its shard.
    writer = RolloutWriter(tmp_path, "mock")
    shard = writer.write(0, [{"step": 0, "sample_idx": 0}])
    (tmp_path / "mock_live_step_00000001.pending.json").write_text(
        json.dumps(
            {
                "version": 1,
                "step": 1,
                "shard": shard.name,
                "byte_offset": 1,
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid live rollout write intent"):
        RolloutWriter(tmp_path, "mock")


def test_live_writer_rejects_same_index_content_tampering(tmp_path: Path) -> None:
    # The marker's digest catches records edited under it.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(
        3,
        [
            {"step": 3, "sample_idx": 0, "reward": 0.0},
            {"step": 3, "sample_idx": 1, "reward": 1.0},
        ],
        prompt_loader_state_after={"position": 7},
    )
    shard = tmp_path / "mock_rollouts_000000.jsonl"
    records = [
        cast("dict[str, object]", json.loads(line))
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["reward"] = -1.0
    shard.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    resumed = RolloutWriter(tmp_path, "mock")
    with pytest.raises(ValueError, match=r"does not match.*record contents"):
        resumed.committed_live_steps(2)


def test_live_writer_rejects_marker_record_mismatch(tmp_path: Path) -> None:
    # A marker must agree with the records actually in the shard.
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(2, [{"step": 2, "sample_idx": 0}])
    (tmp_path / "mock_live_step_00000002.json").write_text(
        json.dumps(
            {
                "version": 1,
                "step": 2,
                "records_per_update": 2,
                "sample_indices": [0, 1],
                "prompt_loader_state_after": {"position": 4},
            },
        ),
        encoding="utf-8",
    )
    resumed = RolloutWriter(tmp_path, "mock")
    with pytest.raises(ValueError, match="claims a complete update"):
        resumed.committed_live_steps(2)


def test_runner_config_bounds_prompt_queue() -> None:
    # queue_size holds one update plus the async steps queued ahead.
    config = RunnerConfig(
        num_prompts=2,
        samples_per_prompt=3,
        async_steps=4,
    )
    assert config.queue_size == 10
    assert config.records_per_update == 6


def test_runner_config_preserves_released_batch_size() -> None:
    # The default update is the released 256: 8 prompts, 32 samples each.
    config = RunnerConfig()
    assert config.records_per_update == 8 * 32


def test_prime_prompts_fills_the_configured_inflight_window() -> None:
    calls: list[tuple[int, int, object, object, bool, object]] = []
    prompt_queue = object()
    generation_config = object()
    base_env_config = object()
    data = ModuleType("data")

    def add_prompt_to_generator(
        example: dict[str, int],
        epoch: int,
        queue: object,
        generation: object,
        *,
        is_eval: bool,
        base_env_config: object,
    ) -> None:
        calls.append(
            (
                example["index"],
                epoch,
                queue,
                generation,
                is_eval,
                base_env_config,
            ),
        )

    vars(data)["add_prompt_to_generator"] = add_prompt_to_generator
    session = _sync_session([], replace(_api(), data_loader=data))
    session.iter_dataloader = iter({"index": index} for index in range(6))
    session.prompt_queue = prompt_queue
    session.generation_config = generation_config
    session.base_env_config = base_env_config

    assert (
        prime_prompts(
            RunnerConfig(async_steps=2, num_prompts=3),
            session,
        )
        == 6
    )
    assert calls == [
        (index, 0, prompt_queue, generation_config, False, base_env_config)
        for index in range(6)
    ]


def test_run_primes_collects_and_closes_its_ray_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    paths = [tmp_path / "rollouts.jsonl"]
    ray = SimpleNamespace(
        is_initialized=lambda: True,
        shutdown=lambda: events.append("shutdown"),
    )
    api = replace(_api(), ray=ray)
    session = _sync_session([], api)
    session.started_ray = True

    def build(_config: RunnerConfig, _api: UpstreamApi) -> CollectionSession:
        events.append("build")
        return session

    def prime(_config: RunnerConfig, _session: CollectionSession) -> None:
        events.append("prime")

    def collect(_config: RunnerConfig, _session: CollectionSession) -> list[Path]:
        events.append("collect")
        return paths

    monkeypatch.setattr(runner_module, "build_session", build)
    monkeypatch.setattr(runner_module, "prime_prompts", prime)
    monkeypatch.setattr(runner_module, "run_updates", collect)

    assert run(RunnerConfig(output_dir=tmp_path), api) == paths
    assert events == ["build", "prime", "collect", "shutdown"]


def test_run_resolves_paths_before_loading_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path | str | None] = []
    config = RunnerConfig(base_dir=tmp_path)

    def load(
        root: Path | str | None,
        _runtime_policy: RayRuntimePolicy,
    ) -> UpstreamApi:
        observed.append(root)
        raise RuntimeError("stop after observing the path")

    monkeypatch.setattr(runner_module, "load_upstream_api", load)

    with pytest.raises(RuntimeError, match="stop after observing"):
        run(config)

    assert observed == [tmp_path / "datasets/tmax/upstream/tmax"]
    assert config.upstream_root == Path("/datasets/tmax/upstream/tmax")


def test_runner_validation_happens_when_the_runtime_is_made() -> None:
    # Validation waits for make(), after finalize has filled the values.
    config = RunnerConfig(num_updates=0)
    with pytest.raises(ValueError, match="num_updates"):
        config.make()


def test_runner_uses_the_prepared_dataset_and_task_assets() -> None:
    # The runner consumes exactly what prepare_data stages, under base_dir.
    config = RunnerConfig(base_dir="/opt/scratch").finalize()
    root = Path("/opt/scratch/datasets/tmax/tmax-15k-open-instruct")
    assert config.dataset_repo == "allenai/tmax-15k-open-instruct"
    assert config.dataset_path == root / "data/train-00000-of-00001.parquet"
    assert config.task_data_dir == root / "task-data.tar.gz.extracted"


def test_runner_cli_resolves_defaults_under_base_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[RunnerConfig] = []

    def run(config: RunnerConfig) -> list[Path]:
        captured.append(config)
        return []

    monkeypatch.setattr(runner_module, "run", run)
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--base-dir", str(tmp_path)],
    )

    assert runner_module.main() == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.working_dir == tmp_path / "runs/tmax/exp000"
    assert config.checkpoint == tmp_path / "models/Qwen3.5-4B"
    assert config.output_dir == tmp_path / "runs/tmax/exp000/rollouts"
    assert config.upstream_root == tmp_path / "datasets/tmax/upstream/tmax"


def test_session_refuses_missing_pinned_dataset_assets(tmp_path: Path) -> None:
    # File checks use the finalized copy without mutating the caller's config.
    api = replace(_api(), ray=SimpleNamespace(is_initialized=lambda: True))
    config = RunnerConfig(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="training rows") as error:
        build_session(config, api)
    assert str(
        tmp_path
        / "datasets/tmax/tmax-15k-open-instruct/data/train-00000-of-00001.parquet",
    ) in str(error.value)
    assert config.dataset_path == Path(
        "/datasets/tmax/tmax-15k-open-instruct/data/train-00000-of-00001.parquet",
    )


@pytest.mark.parametrize(
    ("already_initialized", "shutdowns"),
    [(False, 1), (True, 0)],
)
def test_session_failure_only_shuts_down_owned_ray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    already_initialized: bool,
    shutdowns: int,
) -> None:
    dataset = tmp_path / "train.parquet"
    dataset.write_text("rows", encoding="utf-8")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    initialized = already_initialized
    events: list[str] = []

    def is_initialized() -> bool:
        return initialized

    def init(**_kwargs: object) -> None:
        nonlocal initialized
        initialized = True

    def shutdown() -> None:
        nonlocal initialized
        events.append("shutdown")
        initialized = False

    ray = SimpleNamespace(
        is_initialized=is_initialized,
        init=init,
        shutdown=shutdown,
        get_runtime_context=lambda: SimpleNamespace(
            runtime_env={"env_vars": RayRuntimePolicy().env_vars()},
        ),
    )
    api = replace(_api(), ray=ray)
    config = RunnerConfig(dataset_path=dataset, task_data_dir=tasks)

    def fail(_checkpoint: Path, _ray: object) -> None:
        raise RuntimeError("setup failed")

    monkeypatch.setattr(runner_module, "_require_checkpoint_on_ray_nodes", fail)

    with pytest.raises(RuntimeError, match="setup failed"):
        build_session(config, api)

    assert events == ["shutdown"] * shutdowns


def test_the_qwen35_namespace_mapper_matches_upstream() -> None:
    """Map Hugging Face names to the prefix vLLM serves Qwen3.5/3.6 under."""
    mapper = vllm_weight_name_mapper("hamishivi/Qwen3.5-4B")
    assert mapper is not None
    assert mapper("model.embed_tokens.weight") == (
        "language_model.model.embed_tokens.weight"
    )
    assert vllm_weight_name_mapper("allenai/OLMo-2-1124-7B") is None


def test_weight_sync_maps_names_for_metadata_and_tensors_alike() -> None:
    """A name vLLM cannot match is a parameter that never arrives.

    Upstream applies its mapper to ``_collect_weight_metadata`` AND to the
    tensors ``_prepare_params_for_sync`` yields; mapping only one side would
    make the metadata and the payload disagree about the same parameter.
    """
    engines = [_Engine(), _Engine()]
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    sync = WeightSync(
        _sync_session(engines, api),
        tensor_parallel_size=1,
        name_mapper=vllm_weight_name_mapper("Qwen3.5-4B"),
    )
    sync.initialize(node_ip="127.0.0.1", port=29500)
    sync(
        [
            ("model.embed_tokens.weight", torch.zeros(2, 3)),
            ("lm_head.weight", torch.zeros(2, 3)),
        ],
        model_step=5,
    )

    expected = [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]
    assert [engine.names_sent() for engine in engines] == [expected, expected]
    assert [name for name, _ in sent] == expected
    assert all(("step", 5) in engine.events for engine in engines)


def test_weight_sync_without_a_mapper_sends_names_unchanged() -> None:
    # Non-Qwen checkpoints sync under their Hugging Face names as-is.
    engines = [_Engine()]
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    sync = WeightSync(_sync_session(engines, api), tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29500)
    sync([("model.embed_tokens.weight", torch.zeros(2, 3))], model_step=0)
    assert sent[0][0] == "model.embed_tokens.weight"


def test_weight_sync_gates_admissions_and_reopens_after_success() -> None:
    # A sync pauses admissions and engines, then restores both on success.
    engine = _Engine()
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    session = _sync_session([engine], api)
    manager = cast(_ActorManager, session.actor_manager)
    sync = WeightSync(session, tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29500)
    engine.events.clear()
    sync([("weight", torch.zeros(2))], model_step=7)
    assert manager.events == [("gate", True), ("gate", False)]
    assert [event for event, _ in engine.events] == [
        "sleep",
        "update",
        "wake",
        "step",
    ]


def test_weight_sync_failure_wakes_reopens_and_does_not_advance() -> None:
    # A failed sync wakes the engines for cleanup but keeps admissions closed.
    engine = _Engine()
    engine.update_weights = _Rpc(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("primary")),
    )
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    session = _sync_session([engine], api)
    manager = cast(_ActorManager, session.actor_manager)
    sync = WeightSync(session, tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29500)
    engine.events.clear()
    with pytest.raises(RuntimeError, match="primary"):
        sync([("weight", torch.zeros(2))], model_step=8)
    assert manager.events == [("gate", True)]
    assert [event for event, _ in engine.events] == ["sleep", "wake"]
    assert ("step", 8) not in engine.events
    with pytest.raises(RuntimeError, match="unusable after a failed transfer"):
        sync([("weight", torch.zeros(2))], model_step=8)


def test_collection_reports_to_the_session_actor_manager() -> None:
    # Upstream receives the session's actor manager and chosen age limit.
    seen: list[object] = []
    age_limits: list[int | None] = []
    data = ModuleType("data")

    def accumulate(
        *_args: object,
        actor_manager: object,
        max_result_age_steps: int | None,
        **_kwargs: object,
    ) -> tuple[_FakeResult, _FakeBatch, None, None]:
        seen.append(actor_manager)
        age_limits.append(max_result_age_steps)
        return _result(), _batch(), None, None

    vars(data)["accumulate_inference_batches"] = accumulate

    def compute_group_advantages(**_kwargs: object) -> torch.Tensor:
        return torch.tensor([-0.5, 0.5])

    vars(data)["compute_group_advantages"] = compute_group_advantages
    api = replace(_api(), data_loader=data)
    session = _sync_session([], api)
    collect_update(
        session,
        RunnerConfig(num_prompts=1, samples_per_prompt=2),
        5,
        max_result_age_steps=None,
    )
    assert seen == [cast(object, session.actor_manager)]
    assert age_limits == [None]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
