"""Tests for building the TMax runtime and the runner's guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NoReturn, cast, overload

import dataclasses
import importlib

import pytest

from priml.baselines.tmax.scripts import (
    runner as runner_module,
    upstream,
)
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RayRuntimePolicy,
    RunnerConfig,
    UpstreamApi,
    build_session,
    collect_update,
    serialize_rollout_records,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.tmax.scripts import custom_types


def _unused(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("This test does not reach this upstream call.")


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


@dataclass(kw_only=True, slots=True)
class _BuildRay:
    """A Ray driver for session construction: no nodes are probed."""

    initialized: bool = False
    runtime_env: Mapping[str, object] = field(default_factory=dict[str, object])
    events: list[str] = field(default_factory=list)
    live_nodes: list[Mapping[str, object]] = field(
        default_factory=list[Mapping[str, object]],
    )
    context: object = field(default_factory=object)

    @property
    def _private(self) -> custom_types.RayPrivate:
        return _unused()

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, *, runtime_env: Mapping[str, object]) -> None:
        del runtime_env
        self.initialized = True

    def shutdown(self) -> None:
        self.events.append("shutdown")
        self.initialized = False

    def nodes(self) -> list[Mapping[str, object]]:
        return self.live_nodes

    def get_runtime_context(self) -> object:
        return self.context

    def remote(self, target: object, /) -> custom_types.RemoteFunction:
        return _Remote(value=("actor", target))

    @overload
    def get(self, refs: list[object], /) -> list[object]: ...
    @overload
    def get(self, refs: object, /) -> object: ...
    def get(self, refs: object, /) -> object:
        return refs


@dataclass(kw_only=True, slots=True)
class _Dims:
    def from_hf_config(self, model_name_or_path: str, /) -> object:
        return ("dims", model_name_or_path)


@dataclass(kw_only=True, slots=True)
class _Envs:
    pass_tools_to_chat_template: bool = True
    tool_call_names: object = ("bash",)
    tool_parser_type: str = "vllm_qwen3_xml"


class _Loader:
    _epoch: int = 0

    def __next__(self) -> object:
        return {}

    def state_dict(self) -> Mapping[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object], /) -> None:
        del state


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

    def HFDataLoader(self, **kwargs: object) -> _Loader:  # noqa: N802 -- Upstream's class name.
        self.calls["HFDataLoader"] = kwargs
        return _Loader()


def _build_api(
    fake: _BuildUpstream,
    ray: _BuildRay,
    *,
    actor_manager: custom_types.ActorManagerModule | None,
) -> UpstreamApi:
    module = cast("object", fake)
    return UpstreamApi(
        ray=ray,
        ray_queue=cast("custom_types.RayQueue", module),
        data_loader=cast("custom_types.DataLoaderModule", module),
        dataset_transformation=cast("custom_types.DatasetTransformationModule", module),
        environments_utils=cast("custom_types.EnvironmentsUtilsModule", module),
        grpo_fast=cast("custom_types.GrpoFastModule", module),
        grpo_utils=cast("custom_types.GrpoUtilsModule", module),
        ground_truth_utils=cast("custom_types.GroundTruthUtilsModule", module),
        model_utils=cast("custom_types.ModelUtilsModule", module),
        rl_utils=cast("custom_types.RlUtilsModule", module),
        utils=cast("custom_types.UtilsModule", module),
        vllm_utils=cast("custom_types.VllmUtilsModule", module),
        actor_manager=actor_manager,
    )


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
    monkeypatch.setattr(runner_module, "_require_checkpoint_on_ray_nodes", _unused_ok)
    fake = _BuildUpstream()
    ray = _BuildRay()
    config = _staged(tmp_path)
    session = build_session(
        config,
        _build_api(fake, ray, actor_manager=_actor_module()),
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


def _unused_ok(*_args: object) -> None:
    return None


def test_build_session_requires_the_actor_manager_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "_require_checkpoint_on_ray_nodes", _unused_ok)
    ray = _BuildRay()
    with pytest.raises(RuntimeError, match="does not expose ActorManager"):
        build_session(
            _staged(tmp_path),
            _build_api(_BuildUpstream(), ray, actor_manager=None),
        )
    assert ray.events == ["shutdown"]


def test_build_session_requires_the_task_data(tmp_path: Path) -> None:
    config = _staged(tmp_path)
    (tmp_path / "tasks").rmdir()
    with pytest.raises(FileNotFoundError, match="task data are missing"):
        build_session(
            config,
            _build_api(_BuildUpstream(), _BuildRay(), actor_manager=None),
        )


def test_a_ray_context_without_a_runtime_env_reads_as_empty() -> None:
    assert runner_module._initialized_runtime_env(_BuildRay(initialized=True)) == {}


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
    ray = _BuildRay(initialized=True, context=_GetterContext(value=value))
    assert runner_module._initialized_runtime_env(ray) == expected


def test_a_checkpoint_missing_on_the_driver_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing or empty on the driver"):
        runner_module._require_checkpoint_on_ray_nodes(tmp_path / "absent", _BuildRay())


def test_a_cluster_without_live_nodes_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no live inference nodes"):
        runner_module._require_checkpoint_on_ray_nodes(tmp_path, _BuildRay())


def test_a_node_without_an_address_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    ray = _BuildRay(live_nodes=[{"Alive": True}])
    with pytest.raises(RuntimeError, match="no NodeManagerAddress"):
        runner_module._require_checkpoint_on_ray_nodes(tmp_path, ray)


# -- config -------------------------------------------------------------------


def test_the_released_system_prompt_lives_in_the_checkout() -> None:
    config = RunnerConfig()
    assert config.system_prompt() == config.upstream_root / runner_module.SYSTEM_PROMPT
    assert config.max_model_len == config.max_prompt_tokens + config.response_length


def test_a_system_prompt_override_resolves_under_base_dir(tmp_path: Path) -> None:
    config = RunnerConfig(
        base_dir=tmp_path,
        system_prompt_override_file=Path("/prompts/p.txt"),
    ).finalize()
    assert config.system_prompt() == tmp_path / "prompts/p.txt"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"start_step": -1}, "start_step must be non-negative"),
        ({"num_prompts": 0}, "num_prompts must be positive"),
        ({"samples_per_prompt": 1}, "samples_per_prompt must be greater than one"),
        ({"max_prompt_tokens": 0}, "lengths must be positive"),
        ({"max_steps": 0}, "max_steps must be positive"),
        ({"num_engines": 0}, "Engine counts must be positive"),
        ({"pool_size": 0}, "pool_size must be positive"),
        ({"dataset_repo": ""}, "dataset_repo must name"),
        ({"filter_zero_std_samples": False}, "active_sampling needs filter"),
        ({"async_steps": 1}, "active_sampling needs async_steps"),
        ({"active_sampling": False, "async_steps": 0}, "async_steps must be positive"),
    ],
)
def test_invalid_runner_settings_are_refused(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(RunnerConfig(), **overrides).make()


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
    api = runner_module.load_upstream_api(tmp_path)
    assert api.weight_transfer is not None
    base, nccl = api.weight_transfer
    assert cast("ModuleType", base).__name__.endswith("weight_transfer.base")
    assert cast("ModuleType", nccl).__name__.endswith("nccl_engine")


# -- collection ---------------------------------------------------------------


def test_collection_refuses_an_empty_retained_batch() -> None:
    fake = cast("object", _NoRollouts())
    api = _build_api(_BuildUpstream(), _BuildRay(), actor_manager=None)
    session = CollectionSession(
        api=UpstreamApi(
            ray=api.ray,
            ray_queue=api.ray_queue,
            data_loader=cast("custom_types.DataLoaderModule", fake),
            dataset_transformation=api.dataset_transformation,
            environments_utils=api.environments_utils,
            grpo_fast=api.grpo_fast,
            grpo_utils=api.grpo_utils,
            ground_truth_utils=api.ground_truth_utils,
            model_utils=api.model_utils,
            rl_utils=api.rl_utils,
            utils=api.utils,
            vllm_utils=api.vllm_utils,
        ),
        tokenizer=None,
        dataset=None,
        iter_dataloader=_Loader(),
        inference_queue=None,
        prompt_queue=None,
        generation_config=None,
        base_env_config=None,
        model_dims=None,
        engines=[],
        pools={},
        started_ray=False,
        actor_manager=cast("custom_types.ActorManagerHandle", object()),
    )
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


def test_the_released_runtime_policy_is_applied_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy: Callable[[], dict[str, str]] = RayRuntimePolicy().env_vars
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    ray = _BuildRay()
    assert runner_module._initialize_ray(RunnerConfig(), ray)
    assert policy()["VLLM_USE_V1"] == "1"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
