"""Tests for when finished workers' fresh worlds are generated."""

from __future__ import annotations

from priml.baselines.craftax.restart import (
    RestartFromReserve,
    RestartOnDemand,
    RestartPlan,
)


def _on_demand() -> RestartOnDemand:
    return RestartOnDemand(RestartOnDemand.Config())


def _reserve() -> RestartFromReserve:
    return RestartFromReserve(RestartFromReserve.Config())


def test_on_demand_generates_one_world_per_finished_worker() -> None:
    plan = _on_demand().plan(finished=3, pool_size=16)
    assert plan == RestartPlan(generate=3, offset=0, modulus=3)


def test_on_demand_never_generates_more_than_the_pool() -> None:
    plan = _on_demand().plan(finished=40, pool_size=16)
    assert plan == RestartPlan(generate=16, offset=0, modulus=16)


def test_on_demand_generates_nothing_when_nobody_finished() -> None:
    plan = _on_demand().plan(finished=0, pool_size=16)
    assert plan == RestartPlan(generate=0, offset=0, modulus=1)


def test_on_demand_has_no_checkpoint_state() -> None:
    policy = _on_demand()
    assert policy.state_dict() == {"cursor": None}
    policy.load_state_dict({"cursor": 4})
    assert policy.state_dict() == {"cursor": None}


def test_reserve_fills_the_pool_once_and_deals_unused_worlds_in_order() -> None:
    policy = _reserve()
    assert policy.plan(finished=3, pool_size=8) == RestartPlan(8, 0, 8)
    assert policy.plan(finished=2, pool_size=8) == RestartPlan(0, 3, 8)
    assert policy.plan(finished=0, pool_size=8) == RestartPlan(0, 0, 1)
    assert policy.plan(finished=3, pool_size=8) == RestartPlan(0, 5, 8)
    assert policy.plan(finished=1, pool_size=8) == RestartPlan(8, 0, 8)


def test_reserve_refills_rather_than_deal_one_step_from_two_pools() -> None:
    policy = _reserve()
    policy.plan(finished=3, pool_size=8)
    policy.plan(finished=4, pool_size=8)
    assert policy.plan(finished=2, pool_size=8) == RestartPlan(8, 0, 8)


def test_reserve_shares_worlds_only_when_one_step_outnumbers_the_pool() -> None:
    policy = _reserve()
    assert policy.plan(finished=12, pool_size=8) == RestartPlan(8, 0, 8)
    assert policy.plan(finished=1, pool_size=8) == RestartPlan(8, 0, 8)


def test_reserve_resumes_dealing_where_a_checkpoint_stopped() -> None:
    original = _reserve()
    original.plan(finished=3, pool_size=8)
    resumed = _reserve()
    resumed.load_state_dict(original.state_dict())
    assert resumed.plan(finished=2, pool_size=8) == original.plan(
        finished=2,
        pool_size=8,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
