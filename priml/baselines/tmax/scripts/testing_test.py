"""Tests for the TMax driver's shared test support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from priml.baselines.tmax.scripts import runner
from priml.baselines.tmax.scripts.testing import (
    FakeLoader,
    FakeRay,
    session,
    unused,
    upstream_api,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


def test_unused_fails_the_test_that_reaches_it() -> None:
    with pytest.raises(AssertionError, match="does not reach this upstream call"):
        unused(1, key=2)


def test_fake_ray_records_its_init_and_shutdown() -> None:
    ray = FakeRay(initialized=False)
    ray.init(runtime_env={"env_vars": {"A": "1"}})
    assert ray.is_initialized()
    assert ray.init_calls == [{"env_vars": {"A": "1"}}]
    ray.shutdown()
    assert not ray.is_initialized()
    assert ray.events == ["shutdown"]


def test_fake_ray_resolves_one_reference_or_each_of_a_list() -> None:
    ray = FakeRay(resolve=_tagged)
    assert ray.get(1) == ("ref", 1)
    assert ray.get([1, 2]) == [("ref", 1), ("ref", 2)]
    assert FakeRay().get([3]) == [3]


def test_fake_ray_context_carries_its_runtime_env_unless_given_one() -> None:
    ray = FakeRay(runtime_env={"env_vars": {"A": "1"}})
    assert runner._initialized_runtime_env(ray) == {"env_vars": {"A": "1"}}
    context = object()
    assert FakeRay(context=context).get_runtime_context() is context


def test_fake_ray_wraps_targets_with_its_task_factory_and_lists_nodes() -> None:
    nodes: list[Mapping[str, object]] = [{"Alive": True}]
    ray = FakeRay(live_nodes=nodes, tasks=_FixedTask)
    assert ray.nodes() == nodes
    assert ray.remote("target").remote() == "target"
    with pytest.raises(AssertionError, match="does not reach"):
        FakeRay().remote("target")
    with pytest.raises(AssertionError, match="does not reach"):
        _ = FakeRay()._private


def test_fake_loader_yields_its_examples_and_keeps_no_state() -> None:
    loader = FakeLoader([{"index": 0}, {"index": 1}], epoch=3)
    assert [next(loader), next(loader)] == [{"index": 0}, {"index": 1}]
    assert loader._epoch == 3
    assert loader.state_dict() == {}
    loader.load_state_dict({"position": 1})
    with pytest.raises(StopIteration):
        next(loader)


def test_upstream_api_binds_one_module_to_every_upstream_slot() -> None:
    modules = object()
    api = upstream_api(modules)
    assert {
        id(slot)
        for slot in (
            api.ray_queue,
            api.data_loader,
            api.dataset_transformation,
            api.environments_utils,
            api.grpo_fast,
            api.grpo_utils,
            api.ground_truth_utils,
            api.model_utils,
            api.rl_utils,
            api.utils,
            api.vllm_utils,
        )
    } == {id(modules)}
    assert isinstance(api.ray, FakeRay)
    assert (api.actor_manager, api.weight_transfer) == (None, None)


def test_upstream_api_takes_the_named_overrides() -> None:
    ray, rl_utils = FakeRay(), _RlUtils()
    api = upstream_api(object(), ray=ray, rl_utils=rl_utils)
    assert api.ray is ray
    assert api.rl_utils is rl_utils


def test_session_holds_the_engines_and_an_empty_runtime() -> None:
    api = upstream_api(object())
    engines = [_Engine(), _Engine()]
    built = session(api, engines=engines)
    assert built.api is api
    assert built.engines == engines
    assert built.engines is not engines
    assert isinstance(built.iter_dataloader, FakeLoader)
    assert (built.started_ray, built.pools) == (False, {})
    loader = FakeLoader()
    assert session(api, loader=loader).iter_dataloader is loader


def _tagged(ref: object) -> object:
    return ("ref", ref)


class _FixedTask:
    """A task factory whose ``remote`` returns the wrapped target."""

    def __init__(self, target: object) -> None:
        self.target = target

    def remote(self, *_args: object, **_kwargs: object) -> object:
        return self.target

    def options(self, **_kwargs: object) -> _FixedTask:
        return self


class _RlUtils:
    def _get_request_info_for_sample(
        self,
        request_info: object,
        index: int,
        /,
    ) -> dict[str, object]:
        del request_info
        return {"sample": index}


class _Rpc:
    def remote(self, *_args: object, **_kwargs: object) -> object:
        return None


class _Engine:
    def __init__(self) -> None:
        self.init_weight_transfer_engine = self.sleep = self.wake_up = _Rpc()
        self.update_weights = self.set_model_step = _Rpc()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
