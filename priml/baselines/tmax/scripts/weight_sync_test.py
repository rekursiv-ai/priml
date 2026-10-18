"""Tests for pushing learner weights into TMax's vLLM engines."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, cast, overload

from torch import Tensor

import pytest
import torch

from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    UpstreamApi,
    vllm_weight_name_mapper,
)
from priml.baselines.tmax.scripts.weight_sync import WeightSync


if TYPE_CHECKING:
    from priml.baselines.tmax.scripts import custom_types


def _unused(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("This test does not reach this upstream call.")


class _Ray:
    """A Ray driver whose ``get`` returns each reference unchanged."""

    @property
    def _private(self) -> custom_types.RayPrivate:
        return _unused()

    def is_initialized(self) -> bool:
        return True

    def init(self, *, runtime_env: Mapping[str, object]) -> object:
        return _unused(runtime_env)

    def shutdown(self) -> None:
        _unused()

    def nodes(self) -> list[Mapping[str, object]]:
        return _unused()

    def get_runtime_context(self) -> object:
        return _unused()

    def remote(self, target: object, /) -> custom_types.RemoteFunction:
        return _unused(target)

    @overload
    def get(self, refs: list[object], /) -> list[object]: ...
    @overload
    def get(self, refs: object, /) -> object: ...
    def get(self, refs: object, /) -> object:
        return refs


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
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.set_should_stop = _Rpc(self._set_should_stop)

    def _set_should_stop(self, value: bool) -> list[object]:
        self.events.append(("gate", value))
        return []


def _init_request(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


def _trainer_init(info: object) -> dict[str, object]:
    return {"info": info}


def _raise_primary(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("primary")


@dataclass(kw_only=True, slots=True)
class _FakeTransferEngine:
    """vLLM's trainer-side NCCL engine; records every tensor it sends."""

    sent: list[tuple[str, Tensor]]

    def trainer_init(self, init_info: Mapping[str, object], /) -> object:
        return _trainer_init(init_info)

    def trainer_send_weights(
        self,
        *,
        iterator: Iterator[tuple[str, Tensor]],
        trainer_args: object,
    ) -> None:
        del trainer_args
        self.sent.extend(iterator)


@dataclass(kw_only=True, slots=True)
class _FakeTransferBase:
    WeightTransferInitRequest: Callable[..., object] = _init_request


@dataclass(kw_only=True, slots=True)
class _FakeTransferNccl:
    NCCLWeightTransferEngine: _FakeTransferEngine
    NCCLTrainerSendWeightsArgs: Callable[..., object] = _init_request


def _api(
    *,
    weight_transfer: tuple[_FakeTransferBase, _FakeTransferNccl] | None = None,
) -> UpstreamApi:
    """Build the upstream surface ``WeightSync`` touches: Ray and the transfer."""
    module = object()
    return UpstreamApi(
        ray=_Ray(),
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
        weight_transfer=weight_transfer,
    )


def _sync_api(sent: list[tuple[str, torch.Tensor]]) -> UpstreamApi:
    return _api(
        weight_transfer=(
            _FakeTransferBase(),
            _FakeTransferNccl(NCCLWeightTransferEngine=_FakeTransferEngine(sent=sent)),
        ),
    )


def _sync_session(engines: list[_Engine], api: UpstreamApi) -> CollectionSession:
    loader = cast("custom_types.PromptLoader", object())
    return CollectionSession(
        api=api,
        tokenizer=None,
        dataset=None,
        iter_dataloader=loader,
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
    sync.initialize(node_ip="127.0.0.1", port=29_500)
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
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    sync([("model.embed_tokens.weight", torch.zeros(2, 3))], model_step=0)
    assert sent[0][0] == "model.embed_tokens.weight"


def test_weight_sync_sends_dtype_and_shape_metadata() -> None:
    # vLLM allocates receive buffers from the metadata, so it must match.
    engine = _Engine()
    sync = WeightSync(_sync_session([engine], _sync_api([])), tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    sync(
        [
            ("a", torch.zeros(2, 3, dtype=torch.bfloat16)),
            ("b", torch.zeros(4, dtype=torch.float32)),
        ],
        model_step=2,
    )
    assert ("update", (["a", "b"], ["bfloat16", "float32"], [[2, 3], [4]], True)) in (
        engine.events
    )


def test_weight_sync_gates_admissions_and_reopens_after_success() -> None:
    # A sync pauses admissions and engines, then restores both on success.
    engine = _Engine()
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    session = _sync_session([engine], api)
    manager = cast(_ActorManager, session.actor_manager)
    sync = WeightSync(session, tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
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
    engine.update_weights = _Rpc(_raise_primary)
    sent: list[tuple[str, torch.Tensor]] = []
    api = _sync_api(sent)
    session = _sync_session([engine], api)
    manager = cast(_ActorManager, session.actor_manager)
    sync = WeightSync(session, tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    engine.events.clear()
    with pytest.raises(RuntimeError, match="primary"):
        sync([("weight", torch.zeros(2))], model_step=8)
    assert manager.events == [("gate", True)]
    assert [event for event, _ in engine.events] == ["sleep", "wake"]
    assert ("step", 8) not in engine.events
    with pytest.raises(RuntimeError, match="unusable after a failed transfer"):
        sync([("weight", torch.zeros(2))], model_step=8)


def test_weight_sync_failure_after_wake_does_not_wake_again() -> None:
    # Engines already awake when the step record fails are not woken twice.
    engine = _Engine()
    session = _sync_session([engine], _sync_api([]))
    manager = cast(_ActorManager, session.actor_manager)
    sync = WeightSync(session, tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    engine.set_model_step = _Rpc(_raise_primary)
    engine.events.clear()
    with pytest.raises(RuntimeError, match="primary"):
        sync([("weight", torch.zeros(2))], model_step=3)
    assert [event for event, _ in engine.events] == ["sleep", "update", "wake"]
    assert manager.events == [("gate", True)]


def test_weight_sync_cleanup_wake_failure_keeps_the_primary_error() -> None:
    # A wake that also fails during cleanup must not mask the transfer error.
    engine = _Engine()
    engine.update_weights = _Rpc(_raise_primary)
    sync = WeightSync(_sync_session([engine], _sync_api([])), tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    engine.wake_up = _Rpc(_unused)
    with pytest.raises(RuntimeError, match="primary"):
        sync([("weight", torch.zeros(2))], model_step=1)


def test_weight_sync_requires_initialization() -> None:
    sync = WeightSync(_sync_session([_Engine()], _sync_api([])), tensor_parallel_size=1)
    with pytest.raises(RuntimeError, match="initialize must run before a sync"):
        sync([("weight", torch.zeros(2))], model_step=0)


def test_weight_sync_cannot_initialize_without_a_transfer_engine() -> None:
    sync = WeightSync(_sync_session([_Engine()], _api()), tensor_parallel_size=1)
    with pytest.raises(RuntimeError, match="no weight-transfer engine"):
        sync.initialize(node_ip="127.0.0.1", port=29_500)


def test_weight_sync_cannot_reinitialize_after_a_failed_transfer() -> None:
    engine = _Engine()
    engine.update_weights = _Rpc(_raise_primary)
    sync = WeightSync(_sync_session([engine], _sync_api([])), tensor_parallel_size=1)
    sync.initialize(node_ip="127.0.0.1", port=29_500)
    with pytest.raises(RuntimeError, match="primary"):
        sync([("weight", torch.zeros(2))], model_step=1)
    with pytest.raises(RuntimeError, match="unusable after a failed transfer"):
        sync.initialize(node_ip="127.0.0.1", port=29_500)


def test_weight_sync_assigns_each_engine_its_rank_offset() -> None:
    engines = [_Engine(), _Engine()]
    sync = WeightSync(_sync_session(engines, _sync_api([])), tensor_parallel_size=2)
    sync.initialize(node_ip="10.0.0.9", port=1234)
    requests = [
        cast("dict[str, object]", payload)
        for engine in engines
        for name, payload in engine.events
        if name == "init"
    ]
    assert [
        cast("dict[str, object]", request["init_info"])["rank_offset"]
        for request in requests
    ] == [1, 3]
    assert cast("dict[str, object]", requests[0]["init_info"])["world_size"] == 5


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
