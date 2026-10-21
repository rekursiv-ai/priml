"""Tests for the thin TMax rollout driver."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import dataclasses
import importlib
import json
import os

from treekle import json_loads

import numpy as np
import pytest

from priml.baselines.tmax.scripts import custom_types, runner, testing, upstream
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RayRuntimePolicy,
    RunnerConfig,
    UpstreamApi,
    build_session,
    collect_update,
    prime_prompts,
    records_for_update,
    run,
    run_updates,
    serialize_rollout_records,
    vllm_weight_name_mapper,
)
from priml.baselines.tmax.scripts.testing import FakeLoader, FakeRay, unused


@dataclass(kw_only=True, slots=True)
class _FakeModelDims:
    def from_hf_config(self, model_name_or_path: str, /) -> object:
        return unused(model_name_or_path)


@dataclass(kw_only=True, slots=True)
class _FakeUpstream:
    """One stand-in for every open-instruct module the driver reads."""

    StreamingDataLoaderConfig: Callable[[], custom_types.StreamingConfig] = unused
    VLLMConfig: Callable[[], custom_types.VLLMConfig] = unused
    HFDataLoader: Callable[..., custom_types.PromptLoader] = unused
    single_example_collator: object = None
    add_prompt_to_generator: Callable[..., object] = unused
    accumulate_inference_batches: Callable[
        ...,
        tuple[
            custom_types.GenerationResult | None,
            custom_types.Batch | None,
            object,
            object,
        ],
    ] = unused
    compute_group_advantages: Callable[..., custom_types.FloatArray] = unused
    TokenizerConfig: Callable[..., object] = unused
    EnvsConfig: Callable[..., custom_types.EnvsConfig] = unused
    make_tokenizer: Callable[[object, object], object] = unused
    initialize_tools_and_envs: Callable[
        ...,
        tuple[dict[str, object], object, list[str]],
    ] = unused
    setup_datasets: Callable[..., tuple[object, object]] = unused
    build_base_env_config: Callable[[object, object], object] = unused
    GRPOExperimentConfig: Callable[[], custom_types.GRPOArgs] = unused
    RewardConfig: Callable[..., object] = unused
    build_all_verifiers: Callable[[object, object], object] = unused
    ModelConfig: Callable[..., object] = unused
    ModelDims: _FakeModelDims = field(default_factory=_FakeModelDims)
    SamplingConfig: Callable[..., object] = unused
    create_vllm_engines: Callable[..., list[custom_types.Engine]] = unused
    Queue: Callable[..., object] = unused

    def find_free_port(self) -> int:
        return unused()


def test_upstream_import_enables_released_reset_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Upstream imports with the released reset-failure-zero-reward flag on.
    observed: list[str | None] = []
    monkeypatch.setenv("SWERL_RESET_FAILURE_ZERO_REWARD", "0")

    def verified(root: Path) -> Path:
        del root
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
    runner.load_upstream_api(tmp_path)
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
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-forward")
    ray = FakeRay(initialized=False)
    assert runner._initialize_ray(RunnerConfig(), ray)
    assert ray.init_calls == [{"env_vars": RayRuntimePolicy().env_vars()}]
    assert "AWS_SECRET_ACCESS_KEY" not in cast(
        "dict[str, str]",
        ray.init_calls[0]["env_vars"],
    )


def test_existing_ray_requires_a_matching_runtime_policy() -> None:
    # An already-running Ray job must carry the same runtime policy.
    ray = FakeRay(runtime_env={"env_vars": RayRuntimePolicy().env_vars()})
    assert not runner._initialize_ray(RunnerConfig(), ray)
    ray.runtime_env = {}
    with pytest.raises(RuntimeError, match="already initialized"):
        runner._initialize_ray(RunnerConfig(), ray)


@dataclass(kw_only=True, slots=True)
class _NodeRef:
    node: str
    path: str


@dataclass(kw_only=True, slots=True)
class _PinnedProbe:
    """``remote`` for one node: returns a ref naming the node it ran on."""

    node: str

    def remote(self, *args: object, **_kwargs: object) -> object:
        return _NodeRef(node=self.node, path=str(args[0]))


@dataclass(kw_only=True, slots=True)
class _InventoryProbe:
    """The checkpoint-inventory task; ``options`` pins it to one node."""

    def remote(self, *_args: object, **_kwargs: object) -> object:
        return unused()

    def options(self, **kwargs: object) -> _PinnedProbe:
        resources = cast("dict[str, float]", kwargs["resources"])
        return _PinnedProbe(node=next(iter(resources)).removeprefix("node:"))


def _probe_task(target: object) -> custom_types.RemoteFunction:
    assert target is runner._checkpoint_inventory
    return _InventoryProbe()


def _probe_ray(
    nodes: list[Mapping[str, object]],
    inventory_on: Callable[[str], object],
) -> FakeRay:
    """Build a cluster whose node ``n`` reports ``inventory_on(n)``."""
    return FakeRay(
        live_nodes=nodes,
        tasks=_probe_task,
        resolve=lambda ref: inventory_on(cast("_NodeRef", ref).node),
    )


def _staged_checkpoint(tmp_path: Path, contents: str = "{}") -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(contents, encoding="utf-8")
    return checkpoint


def test_checkpoint_must_match_on_every_live_ray_node(tmp_path: Path) -> None:
    # vLLM actors load this path remotely; a driver-local checkpoint is not enough.
    checkpoint = _staged_checkpoint(tmp_path)
    expected = runner._checkpoint_inventory(str(checkpoint))
    ray = _probe_ray(
        [
            {"Alive": True, "NodeManagerAddress": "10.0.0.1"},
            {"Alive": True, "NodeManagerAddress": "10.0.0.2"},
            {"Alive": False, "NodeManagerAddress": "10.0.0.3"},
        ],
        lambda node: expected if node == "10.0.0.1" else None,
    )
    with pytest.raises(RuntimeError, match=r"10\.0\.0\.2.*shared storage"):
        runner._require_checkpoint_on_ray_nodes(checkpoint, ray)


def test_checkpoint_probe_accepts_the_same_inventory_on_every_node(
    tmp_path: Path,
) -> None:
    checkpoint = _staged_checkpoint(tmp_path)
    expected = runner._checkpoint_inventory(str(checkpoint))
    assert expected is not None
    ray = _probe_ray(
        [{"Alive": True, "NodeManagerAddress": "10.0.0.1"}],
        lambda _node: expected,
    )
    runner._require_checkpoint_on_ray_nodes(checkpoint, ray)


def test_checkpoint_probe_rejects_same_size_different_contents(
    tmp_path: Path,
) -> None:
    checkpoint = _staged_checkpoint(tmp_path, "left")
    config = checkpoint / "config.json"
    expected = runner._checkpoint_inventory(str(checkpoint))
    assert expected is not None
    config.write_text("rite", encoding="utf-8")
    different = runner._checkpoint_inventory(str(checkpoint))
    assert different is not None
    assert [(name, size) for name, size, _digest in expected] == [
        (name, size) for name, size, _digest in different
    ]
    assert expected != different
    config.write_text("left", encoding="utf-8")
    ray = _probe_ray(
        [{"Alive": True, "NodeManagerAddress": "10.0.0.1"}],
        lambda _node: different,
    )
    with pytest.raises(RuntimeError, match=r"differs.*10\.0\.0\.1"):
        runner._require_checkpoint_on_ray_nodes(checkpoint, ray)


@dataclass(kw_only=True, slots=True)
class _FakeBatch:
    """The upstream batch fields used by the serializer."""

    queries: Sequence[Sequence[int]]
    scores: Sequence[float] | None
    datasets: Sequence[object]
    ground_truths: Sequence[object]


@dataclass(kw_only=True, slots=True)
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


def _request_info(request_info: object, index: int) -> dict[str, object]:
    del request_info
    return {"sample": index}


class _RlUtils:
    """``rl_utils``: the released writer's per-sample request-info reader."""

    def _get_request_info_for_sample(
        self,
        request_info: object,
        index: int,
        /,
    ) -> dict[str, object]:
        return _request_info(request_info, index)


def _api(
    *,
    ray: FakeRay | None = None,
    upstream_modules: _FakeUpstream | None = None,
) -> UpstreamApi:
    """Bind one ``_FakeUpstream`` and the released request-info reader."""
    return testing.upstream_api(
        upstream_modules or _FakeUpstream(),
        ray=ray,
        rl_utils=_RlUtils(),
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
    session = testing.session(_api())
    calls: list[int] = []

    def collect(
        session: CollectionSession,
        config: RunnerConfig,
        step: int,
    ) -> tuple[_FakeBatch, _FakeResult, list[float]]:
        del session, config
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
        session: CollectionSession,
        config: RunnerConfig,
        step: int,
    ) -> tuple[_FakeBatch, _FakeResult, list[float]]:
        del session, config
        calls.append(step)
        return _batch(), _result(), [-0.5, 0.5]

    config = RunnerConfig(
        output_dir=tmp_path,
        run_name="mock",
        num_updates=2,
        num_prompts=1,
        samples_per_prompt=2,
    )
    run_updates(config, testing.session(_api()), collect_update=collect)

    assert calls == [1]
    records = [
        json_loads(line, dict[str, object])
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
    marker = json_loads(
        (tmp_path / "mock_live_step_00000004.json").read_text(
            encoding="utf-8",
        ),
        dict[str, object],
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
        json_loads(line, dict[str, object])
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

    session = testing.session(
        _api(
            upstream_modules=_FakeUpstream(
                add_prompt_to_generator=add_prompt_to_generator,
            ),
        ),
        loader=FakeLoader({"index": index} for index in range(6)),
    )
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
    ray = FakeRay(events=events)
    api = _api(ray=ray)
    session = testing.session(api)
    session.started_ray = True

    def build(config: RunnerConfig, api: UpstreamApi) -> CollectionSession:
        del config, api
        events.append("build")
        return session

    def prime(config: RunnerConfig, session: CollectionSession) -> None:
        del config, session
        events.append("prime")

    def collect(config: RunnerConfig, session: CollectionSession) -> list[Path]:
        del config, session
        events.append("collect")
        return paths

    monkeypatch.setattr(runner, "build_session", build)
    monkeypatch.setattr(runner, "prime_prompts", prime)
    monkeypatch.setattr(runner, "run_updates", collect)

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
        runtime_policy: RayRuntimePolicy,
    ) -> UpstreamApi:
        del runtime_policy
        observed.append(root)
        raise RuntimeError("stop after observing the path")

    monkeypatch.setattr(runner, "load_upstream_api", load)

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
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[RunnerConfig] = []

    def run(config: RunnerConfig) -> list[Path]:
        captured.append(config)
        return [tmp_path / "shard.jsonl"]

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--base-dir", str(tmp_path)],
    )

    assert runner.main() == 0
    assert capsys.readouterr().out == f"{tmp_path / 'shard.jsonl'}\n"
    assert len(captured) == 1
    config = captured[0]
    assert config.working_dir == tmp_path / "runs/tmax/exp000"
    assert config.checkpoint == tmp_path / "models/Qwen3.5-4B"
    assert config.output_dir == tmp_path / "runs/tmax/exp000/rollouts"
    assert config.upstream_root == tmp_path / "datasets/tmax/upstream/tmax"


def test_session_refuses_missing_pinned_dataset_assets(tmp_path: Path) -> None:
    # File checks use the finalized copy without mutating the caller's config.
    api = _api()
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
    ray = FakeRay(
        initialized=already_initialized,
        runtime_env={"env_vars": RayRuntimePolicy().env_vars()},
    )
    api = _api(ray=ray)
    config = RunnerConfig(dataset_path=dataset, task_data_dir=tasks)

    def fail(checkpoint: Path, ray: object) -> None:
        del checkpoint, ray
        raise RuntimeError("setup failed")

    monkeypatch.setattr(runner, "_require_checkpoint_on_ray_nodes", fail)

    with pytest.raises(RuntimeError, match="setup failed"):
        build_session(config, api)

    assert ray.events == ["shutdown"] * shutdowns


def test_the_qwen35_namespace_mapper_matches_upstream() -> None:
    """Map Hugging Face names to the prefix vLLM serves Qwen3.5/3.6 under."""
    mapper = vllm_weight_name_mapper("hamishivi/Qwen3.5-4B")
    assert mapper is not None
    assert mapper("model.embed_tokens.weight") == (
        "language_model.model.embed_tokens.weight"
    )
    assert vllm_weight_name_mapper("allenai/OLMo-2-1124-7B") is None


def test_collection_reports_to_the_session_actor_manager() -> None:
    # Upstream receives the session's actor manager and chosen age limit.
    seen: list[object] = []
    age_limits: list[int | None] = []

    def accumulate(
        *_args: object,
        actor_manager: object,
        max_result_age_steps: int | None,
        **_kwargs: object,
    ) -> tuple[_FakeResult, _FakeBatch, None, None]:
        seen.append(actor_manager)
        age_limits.append(max_result_age_steps)
        return _result(), _batch(), None, None

    def compute_group_advantages(**_kwargs: object) -> np.ndarray:
        return np.array([-0.5, 0.5])

    api = _api(
        upstream_modules=_FakeUpstream(
            accumulate_inference_batches=accumulate,
            compute_group_advantages=compute_group_advantages,
        ),
    )
    session = testing.session(api)
    records = records_for_update(
        session,
        RunnerConfig(num_prompts=1, samples_per_prompt=2),
        5,
        max_result_age_steps=3,
    )
    assert seen == [session.actor_manager]
    assert age_limits == [3]
    assert [record["advantage"] for record in records] == [-0.5, 0.5]


class _Settable:
    """A mutable upstream config object that accepts any attribute."""

    def __init__(self, stop_strings: list[str] | None = None) -> None:
        self.stop_strings = stop_strings


@dataclass(kw_only=True, slots=True)
class _Remote:
    value: object

    def remote(self, *_args: object, **_kwargs: object) -> object:
        return self.value

    def options(self, **_kwargs: object) -> _Remote:
        return self


def _actor_task(target: object) -> custom_types.RemoteFunction:
    return _Remote(value=("actor", target))


@dataclass(kw_only=True, slots=True)
class _Dims:
    def from_hf_config(self, model_name_or_path: str, /) -> object:
        return ("dims", model_name_or_path)


@dataclass(kw_only=True, slots=True)
class _Envs:
    pass_tools_to_chat_template: bool = True
    tool_call_names: object = ("bash",)
    tool_parser_type: str = "vllm_qwen3_xml"


@dataclass(kw_only=True, slots=True)
class _BuildUpstream:
    """Every open-instruct entry point the session build calls, recording each."""

    calls: dict[str, object] = field(default_factory=dict)
    single_example_collator: object = "collate"
    ModelDims: _Dims = field(default_factory=_Dims)

    def _record(self, name: str, value: object) -> object:
        self.calls[name] = value
        return value

    def ModelConfig(self, **kwargs: object) -> object:  # noqa: N802 -- Upstream's class name.
        return self._record("ModelConfig", kwargs)

    def TokenizerConfig(self, **kwargs: object) -> object:  # noqa: N802 -- Upstream's class name.
        return self._record("TokenizerConfig", kwargs)

    def make_tokenizer(self, tokenizer_config: object, model_config: object) -> object:
        return self._record("tokenizer", (tokenizer_config, model_config))

    def EnvsConfig(self, **kwargs: object) -> _Envs:  # noqa: N802 -- Upstream's class name.
        self.calls["EnvsConfig"] = kwargs
        return _Envs()

    def initialize_tools_and_envs(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], object, list[str]]:
        return {"pool": "p"}, "tool-definitions", ["</tool>"]

    def GRPOExperimentConfig(self) -> _Settable:  # noqa: N802 -- Upstream's class name.
        return cast("_Settable", self._record("args", _Settable()))

    def StreamingDataLoaderConfig(self) -> _Settable:  # noqa: N802 -- Upstream's class name.
        return cast("_Settable", self._record("streaming", _Settable(["<stop>"])))

    def setup_datasets(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[object, object]:
        return "train-dataset", None

    def build_base_env_config(self, tools_config: object, pools: object) -> object:
        return ("env", tools_config, pools)

    def RewardConfig(self, **kwargs: object) -> object:  # noqa: N802 -- Upstream's class name.
        return self._record("RewardConfig", kwargs)

    def build_all_verifiers(self, args: object, streaming: object) -> object:
        del args, streaming
        return "verifiers"

    def SamplingConfig(self, **kwargs: object) -> object:  # noqa: N802 -- Upstream's class name.
        return self._record("SamplingConfig", kwargs)

    def Queue(self, *, maxsize: int) -> object:  # noqa: N802 -- Upstream's class name.
        return ("queue", maxsize)

    def VLLMConfig(self) -> _Settable:  # noqa: N802 -- Upstream's class name.
        return _Settable()

    def create_vllm_engines(self, *args: object, **kwargs: object) -> list[object]:
        self.calls["engines"] = (args, kwargs)
        return ["engine"]

    def HFDataLoader(self, **kwargs: object) -> FakeLoader:  # noqa: N802 -- Upstream's class name.
        self.calls["HFDataLoader"] = kwargs
        return FakeLoader()


@dataclass(frozen=True, kw_only=True, slots=True)
class _ActorManagerModule:
    ActorManager: object = "ActorManagerClass"


def _actor_module() -> _ActorManagerModule:
    return _ActorManagerModule()


def _staged(tmp_path: Path) -> RunnerConfig:
    dataset = tmp_path / "train.parquet"
    dataset.write_text("rows", encoding="utf-8")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    return RunnerConfig(
        dataset_path=dataset,
        task_data_dir=tasks,
        output_dir=tmp_path / "out",
        num_prompts=2,
        samples_per_prompt=4,
    )


def test_build_session_wires_upstream_with_the_released_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_checkpoint_on_ray_nodes", _skip_probe)
    fake = _BuildUpstream()
    config = _staged(tmp_path)
    session = build_session(
        config,
        testing.upstream_api(
            fake,
            ray=FakeRay(initialized=False, tasks=_actor_task),
            actor_manager=_actor_module(),
        ),
    )

    assert session.started_ray
    assert session.engines == ["engine"]
    assert session.actor_manager == ("actor", "ActorManagerClass")
    streaming = cast("_Settable", fake.calls["streaming"])
    assert streaming.stop_strings == ["<stop>", "</tool>"]
    sampling = cast("dict[str, object]", fake.calls["SamplingConfig"])
    assert sampling["n"] == 4
    assert sampling["logprobs"] == 1
    engine_kwargs = cast("tuple[object, dict[str, object]]", fake.calls["engines"])[1]
    assert engine_kwargs["mask_tool_use"] is True
    assert engine_kwargs["prompt_queue"] == ("queue", 10)
    loader = cast("dict[str, object]", fake.calls["HFDataLoader"])
    output_dir = config.make().config.output_dir
    assert loader["work_dir"] == str(output_dir / "data-loader")


def _skip_probe(*_args: object) -> None:
    return None


def test_build_session_requires_the_actor_manager_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_checkpoint_on_ray_nodes", _skip_probe)
    ray = FakeRay(initialized=False, tasks=_actor_task)
    with pytest.raises(RuntimeError, match="does not expose ActorManager"):
        build_session(
            _staged(tmp_path),
            testing.upstream_api(_BuildUpstream(), ray=ray),
        )
    assert ray.events == ["shutdown"]


def test_build_session_requires_the_task_data(tmp_path: Path) -> None:
    config = _staged(tmp_path)
    (tmp_path / "tasks").rmdir()
    with pytest.raises(FileNotFoundError, match="task data are missing"):
        build_session(
            config,
            testing.upstream_api(
                _BuildUpstream(),
                ray=FakeRay(initialized=False, tasks=_actor_task),
            ),
        )


def test_a_ray_context_without_a_runtime_env_reads_as_empty() -> None:
    assert runner._initialized_runtime_env(FakeRay(context=object())) == {}


@dataclass(kw_only=True, slots=True)
class _GetterContext:
    value: object

    def get_runtime_env(self) -> object:
        return self.value


@pytest.mark.parametrize(
    ("value", "expected"),
    [({"env_vars": {"A": "1"}}, {"env_vars": {"A": "1"}}), ("not a mapping", {})],
)
def test_a_ray_context_exposing_get_runtime_env_is_read(
    value: object,
    expected: dict[str, object],
) -> None:
    ray = FakeRay(context=_GetterContext(value=value))
    assert runner._initialized_runtime_env(ray) == expected


def test_a_checkpoint_missing_on_the_driver_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing or empty on the driver"):
        runner._require_checkpoint_on_ray_nodes(tmp_path / "absent", FakeRay())


def test_a_cluster_without_live_nodes_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no live inference nodes"):
        runner._require_checkpoint_on_ray_nodes(tmp_path, FakeRay())


def test_a_node_without_an_address_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    ray = FakeRay(live_nodes=[{"Alive": True}], tasks=_actor_task)
    with pytest.raises(RuntimeError, match="no NodeManagerAddress"):
        runner._require_checkpoint_on_ray_nodes(tmp_path, ray)


# -- config -------------------------------------------------------------------


def test_the_released_system_prompt_lives_in_the_checkout() -> None:
    config = RunnerConfig()
    assert config.system_prompt() == config.upstream_root / runner.SYSTEM_PROMPT
    assert config.max_model_len == config.max_prompt_tokens + config.response_length


def test_a_system_prompt_override_resolves_under_base_dir(tmp_path: Path) -> None:
    config = RunnerConfig(
        base_dir=tmp_path,
        system_prompt_override_file=Path("/prompts/p.txt"),
    ).finalize()
    assert config.system_prompt() == tmp_path / "prompts/p.txt"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            dataclasses.replace(RunnerConfig(), start_step=-1),
            "start_step must be non-negative",
        ),
        (
            dataclasses.replace(RunnerConfig(), num_prompts=0),
            "num_prompts must be positive",
        ),
        (
            dataclasses.replace(RunnerConfig(), samples_per_prompt=1),
            "samples_per_prompt must be greater than one",
        ),
        (
            dataclasses.replace(RunnerConfig(), max_prompt_tokens=0),
            "lengths must be positive",
        ),
        (
            dataclasses.replace(RunnerConfig(), max_steps=0),
            "max_steps must be positive",
        ),
        (
            dataclasses.replace(RunnerConfig(), num_engines=0),
            "Engine counts must be positive",
        ),
        (
            dataclasses.replace(RunnerConfig(), pool_size=0),
            "pool_size must be positive",
        ),
        (
            dataclasses.replace(RunnerConfig(), dataset_repo=""),
            "dataset_repo must name",
        ),
        (
            dataclasses.replace(RunnerConfig(), filter_zero_std_samples=False),
            "active_sampling needs filter",
        ),
        (
            dataclasses.replace(RunnerConfig(), async_steps=1),
            "active_sampling needs async_steps",
        ),
        (
            dataclasses.replace(RunnerConfig(), active_sampling=False, async_steps=0),
            "async_steps must be positive",
        ),
    ],
)
def test_invalid_runner_settings_are_refused(
    config: RunnerConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        config.make()


# -- upstream import ----------------------------------------------------------


def test_upstream_import_binds_vllms_weight_transfer_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checkout(root: Path) -> Path:
        del root
        return tmp_path

    monkeypatch.setattr(upstream, "verify_checkout", checkout)
    monkeypatch.setattr(upstream, "open_instruct", checkout)
    monkeypatch.setattr(importlib, "import_module", ModuleType)
    api = runner.load_upstream_api(tmp_path)
    assert api.weight_transfer is not None
    base, nccl = api.weight_transfer
    assert cast("ModuleType", base).__name__.endswith("weight_transfer.base")
    assert cast("ModuleType", nccl).__name__.endswith("nccl_engine")


# -- collection ---------------------------------------------------------------


def test_collection_refuses_an_empty_retained_batch() -> None:
    session = testing.session(testing.upstream_api(_NoRollouts()))
    with pytest.raises(RuntimeError, match="no retained rollouts at training step 3"):
        collect_update(session, RunnerConfig(), 3, max_result_age_steps=None)


class _NoRollouts:
    def accumulate_inference_batches(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[None, None, None, None]:
        return None, None, None, None


@dataclass(kw_only=True, slots=True)
class _Batch:
    queries: Sequence[Sequence[int]] = ((1,),)
    scores: Sequence[float] | None = (1.0,)
    datasets: Sequence[object] = ("d",)
    ground_truths: Sequence[object] = ("g",)


@dataclass(kw_only=True, slots=True)
class _Result:
    responses: Sequence[Sequence[int]] = ((2,),)
    finish_reasons: Sequence[str] = ("stop",)
    masks: Sequence[Sequence[int]] = ((1,),)
    logprobs: Sequence[Sequence[float]] | None = ((-0.1,),)
    request_info: object = None


def _no_info(request_info: object, index: int) -> None:
    del request_info, index


@pytest.mark.parametrize(
    ("batch", "result", "match"),
    [
        (_Batch(scores=None), _Result(), "no reward scores"),
        (_Batch(), _Result(responses=((2,), (3,))), "different sample counts"),
    ],
)
def test_serialization_refuses_unusable_batches(
    batch: _Batch,
    result: _Result,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        serialize_rollout_records(
            batch,
            result,
            step=0,
            advantages=[0.0],
            num_samples_per_prompt=1,
            request_info_for_sample=_no_info,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
