"""Tests for EMA -- the single class covering both
full-module (default) and parameter-only (TRM-style) shadows.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import os

from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import pytest
import torch
import torch.distributed as dist

from priml.distributed.testing import WorkerPool
from priml.train.ema import EMA, NoEMA, karras_decay


@pytest.fixture
def single_rank_group() -> Generator[None, None, None]:
    """Initialize a 1-rank gloo process group for DTensor-shadow tests.

    The rendezvous port is OS-assigned (not hardcoded) so concurrent
    xdist workers running this fixture never collide on a fixed port --
    the prior fixed 29557 made the second worker fail to bind with
    ``DistNetworkError: server socket has failed to listen``.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(WorkerPool.find_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


# -- baseline: legacy full-module shadow --------------------------------------


def test_ema_lerps_parameters_after_warmup() -> None:
    model = nn.Linear(1, 1, bias=True)
    assert model.bias is not None
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    ema = EMA.Config(decay=0.5).make()

    # First call lazy-initializes shadow at current params.
    ema(model)
    assert ema.shadow_model is not None
    shadow = cast(nn.Linear, ema.shadow_model)
    torch.testing.assert_close(shadow.weight, torch.full_like(model.weight, 1.0))

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema(model)
    # shadow = 0.5*1 + 0.5*3 = 2.0
    torch.testing.assert_close(shadow.weight, torch.full_like(model.weight, 2.0))


def test_ema_respects_update_after_step() -> None:
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA.Config(decay=0.5, update_after_step=1).make()

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema(model)
    # Initial call: lazy init, no update due to update_after_step=1.
    assert ema.shadow_model is not None
    shadow = cast(nn.Linear, ema.shadow_model)
    torch.testing.assert_close(shadow.weight, torch.full_like(model.weight, 3.0))

    ema(model)
    # Second call: still no update; counter advances to threshold.
    torch.testing.assert_close(shadow.weight, torch.full_like(model.weight, 3.0))


def test_ema_copies_buffers_when_track_buffers_true() -> None:
    """Default ``track_buffers=True`` copies buffers verbatim (no lerp)."""
    model = nn.BatchNorm1d(2)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.running_mean.fill_(0.5)
    ema = EMA.Config(decay=0.0).make()  # decay=0 => shadow := live
    ema(model)
    ema(model)
    assert ema.shadow_model is not None
    torch.testing.assert_close(
        ema.shadow_model.running_mean,
        torch.full_like(model.running_mean, 0.5),
    )


# -- TRM-style: parameter-only shadow -----------------------------------------


def test_ema_track_buffers_false_skips_buffers() -> None:
    """With ``track_buffers=False`` buffers are not propagated on update.

    Demonstrated by mutating the live buffer AFTER lazy-init and
    confirming the shadow does NOT follow.
    """
    model = nn.BatchNorm1d(2)
    ema = EMA.Config(decay=0.5, track_buffers=False).make()
    ema(model)  # lazy-init shadow
    assert ema.shadow_model is not None
    shadow_bn = cast(nn.BatchNorm1d, ema.shadow_model)
    snapshot = shadow_bn.running_mean.detach().clone()

    with torch.no_grad():
        model.running_mean.fill_(7.0)
    ema(model)
    torch.testing.assert_close(shadow_bn.running_mean, snapshot)


# -- apply_to context manager -------------------------------------------------


def test_ema_apply_to_swaps_params_and_restores() -> None:
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA.Config(decay=0.0).make()  # shadow := live at first call
    ema(model)
    with torch.no_grad():
        model.weight.fill_(7.0)

    pre = model.weight.detach().clone()
    with ema.apply_to(model):
        # Inside: model's weight is shadow's value (1.0).
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 1.0))
    # After: model restored to 7.0.
    torch.testing.assert_close(model.weight, pre)


def test_ema_apply_to_does_not_swap_buffers() -> None:
    """``apply_to`` must leave buffers at live-model values."""
    model = nn.BatchNorm1d(2)
    with torch.no_grad():
        model.running_mean.fill_(0.5)
    ema = EMA.Config(decay=0.0).make()
    ema(model)
    with torch.no_grad():
        model.running_mean.fill_(9.0)
    with ema.apply_to(model):
        # Inside: buffer must still be the live value, not shadowed.
        torch.testing.assert_close(
            model.running_mean,
            torch.full_like(model.running_mean, 9.0),
        )


def test_ema_apply_to_uses_preallocated_backup() -> None:
    """``apply_to`` reuses pre-allocated backup buffers across calls.

    Verifies by capturing the backup tensors' storage pointers after
    lazy init and confirming they don't change after subsequent
    ``apply_to`` calls (no fresh allocations).
    """
    model = nn.Linear(4, 4)
    ema = EMA.Config(decay=0.0).make()
    ema(model)  # lazy init populates EMA._backup
    backup_ptrs_before = {name: t.data_ptr() for name, t in ema._backup.items()}
    assert backup_ptrs_before, "backup buffers should be pre-allocated"

    for _ in range(3):
        with ema.apply_to(model):
            pass

    backup_ptrs_after = {name: t.data_ptr() for name, t in ema._backup.items()}
    assert backup_ptrs_after == backup_ptrs_before, (
        "apply_to re-allocated backup buffers"
    )


# -- param_filter -------------------------------------------------------------


def test_ema_param_filter_excludes_matching_params() -> None:
    """``param_filter`` returning False excludes the param from the shadow."""
    model = nn.Sequential(
        nn.Linear(2, 2),  # named "0"
        nn.Linear(2, 2),  # named "1"
    )

    def exclude_layer_1(name: str, _p: nn.Parameter) -> bool:
        return not name.startswith("1.")

    ema = EMA.Config(decay=0.0).make()
    ema.set_param_filter(exclude_layer_1)
    ema(model)
    # Mutate the excluded layer's weights; EMA should not track.
    layer_1_live = cast(nn.Linear, model[1])
    with torch.no_grad():
        layer_1_live.weight.fill_(9.0)
    ema(model)
    # Shadow's layer-1 weight stays at its initial value (whatever init was).
    # Concretely: shadow_model[1].weight must NOT equal 9.0.
    assert ema.shadow_model is not None
    layer_1_shadow = cast(
        nn.Linear,
        cast(nn.Sequential, ema.shadow_model)[1],
    )
    diff = (layer_1_shadow.weight - layer_1_live.weight).abs().max().item()
    assert diff > 0.0, "param_filter did not exclude layer 1"


# -- state_dict independence (REAL-6 carryover) -------------------------------


def test_ema_state_dict_returns_independent_tensors() -> None:
    model = nn.Linear(1, 1)
    ema = EMA.Config().make()
    ema(model)  # lazy init shadow
    state = ema.state_dict()
    # state["shadow_model"] is a state_dict (mapping); each tensor must be
    # an independent clone of the live shadow_model's parameter storage.
    assert ema.shadow_model is not None
    live_state = ema.shadow_model.state_dict()
    for name, value in state["shadow_model"].items():
        live = live_state[name]
        if torch.is_tensor(value) and torch.is_tensor(live):
            assert value.data_ptr() != live.data_ptr(), (
                f"state_dict()[shadow_model][{name}] aliases live shadow"
            )


def test_ema_two_instances_loaded_from_one_state_are_independent() -> None:
    model = nn.Linear(1, 1)
    src = EMA.Config().make()
    src(model)
    state = src.state_dict()

    a = EMA.Config().make()
    b = EMA.Config().make()
    a(model)  # init structure
    b(model)
    a.load_state_dict(state)
    b.load_state_dict(state)

    assert a.shadow_model is not None
    assert b.shadow_model is not None
    a_state = a.shadow_model.state_dict()
    b_state = b.shadow_model.state_dict()
    for name in a_state:
        if not torch.is_tensor(a_state[name]):
            continue
        assert a_state[name].data_ptr() != b_state[name].data_ptr(), (
            f"a.shadow_model[{name}] aliases b.shadow_model[{name}]"
        )


# -- config defaults ---------------------------------------------------------


def test_ema_config_make() -> None:
    ema = EMA.Config(decay=0.25).make()
    assert isinstance(ema, EMA)
    assert ema.decay == 0.25


def test_ema_config_defaults() -> None:
    cfg = EMA.Config()
    assert cfg.track_buffers is True
    assert cfg.decay == 0.9999
    assert cfg.update_after_step == 0
    assert cfg.update_every == 1


# -- Regression tests (Issue#286 EMA group) ----------------------------------


def test_ema_state_dict_preserves_metadata() -> None:
    """T-002: state_dict must preserve ``_metadata`` for versioned load."""
    model = nn.Linear(2, 2)
    ema = EMA.Config().make()
    ema(model)  # lazy init shadow

    state = ema.state_dict()
    shadow_sd = state["shadow_model"]

    # nn.Module.state_dict attaches _metadata used by load_state_dict
    # (module-version hooks). Round-tripping through a plain dict drops it.
    assert hasattr(shadow_sd, "_metadata"), (
        "EMA.state_dict() dropped shadow_model _metadata"
    )


def test_ema_apply_to_raises_on_missing_tracked_param() -> None:
    """T-003: a tracked param absent at swap time must not be silently skipped."""
    model = nn.Linear(2, 2)
    ema = EMA.Config(decay=0.0).make()
    ema(model)  # lazy init; tracks weight + bias

    # Inject a phantom tracked name absent from both live and shadow.
    ema._tracked_names.add("nonexistent.weight")

    with pytest.raises((KeyError, RuntimeError)), ema.apply_to(model):
        pass


def test_ema_apply_to_rolls_back_on_mid_swap_failure() -> None:
    """#338 regression: a mid-swap failure must leave the live model UN-swapped.

    The swap loop mutates live params in place. If the swap fails partway
    through, the params already swapped must be restored -- otherwise the live
    model is left partially EMA-swapped, silently corrupting subsequent
    training. Drive the shadow away from live, then force the SECOND swap to
    raise (after the first param is already swapped) and assert every param is
    bit-for-bit its pre-swap LIVE value.
    """
    model = nn.Linear(3, 3, bias=True)
    ema = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    ema(model)  # lazy init seeds shadow at current weights
    # Mutate live so shadow != live: a leaked swap would be detectable.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)

    pre = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Force the second swap to raise, guaranteeing the first param is already
    # swapped when the failure occurs (deterministic regardless of set order).
    real_shadow_param = ema._shadow_param
    calls = {"n": 0}

    def failing_shadow_param(name: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced mid-swap failure")
        return real_shadow_param(name)

    ema._shadow_param = failing_shadow_param  # ty: ignore[invalid-assignment] -- deliberately monkeypatches a bound method with a wrong-signature fake to force a mid-swap failure

    with pytest.raises(RuntimeError), ema.apply_to(model):
        pass

    # Every param must be restored to its pre-swap live value (no param left
    # holding a shadow value).
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, pre[name], rtol=0, atol=0)


# -- param-dict (name-keyed) shadow: FSDP/DTensor-survivable ------------------


def test_ema_param_dict_lerps_after_warmup() -> None:
    """param_dict mode lerps tracked params the same as module mode."""
    model = nn.Linear(1, 1, bias=True)
    assert model.bias is not None
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    ema = EMA.Config(decay=0.5, shadow_kind="param_dict").make()

    ema(model)  # lazy init seeds shadow at current params
    assert ema.shadow_model is None, "param_dict mode must not clone a module"
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(model.weight, 1.0),
    )

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema(model)
    # shadow = 0.5*1 + 0.5*3 = 2.0
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(model.weight, 2.0),
    )


def test_ema_param_dict_apply_to_swaps_and_restores() -> None:
    """param_dict mode's apply_to swaps shadow values in and restores."""
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA.Config(decay=0.0, shadow_kind="param_dict").make()
    ema(model)
    with torch.no_grad():
        model.weight.fill_(7.0)

    pre = model.weight.detach().clone()
    with ema.apply_to(model):
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 1.0))
    torch.testing.assert_close(model.weight, pre)


def test_ema_param_dict_excludes_buffers() -> None:
    """param_dict mode never shadows parameters that are buffers."""
    model = nn.BatchNorm1d(2)
    ema = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    ema(model)
    # running_mean / running_var are buffers, not parameters: absent from shadow.
    assert "running_mean" not in ema.shadow_params
    assert "running_var" not in ema.shadow_params
    assert "weight" in ema.shadow_params


def test_ema_param_dict_round_trips_under_dtensor(
    single_rank_group: None,
) -> None:
    """param_dict shadow clones LOCAL shards (no module deepcopy) for DTensor.

    A deepcopy-clone shadow breaks under ``fully_shard``; the param-dict
    shadow must clone ``param.detach()`` (the local shard) and round-trip.
    """
    del single_rank_group
    mesh = init_device_mesh("cpu", (1,))

    model = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        model.weight = nn.Parameter(
            distribute_tensor(model.weight, mesh, [Shard(0)]),
        )

    ema = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    ema(model)  # must NOT deepcopy the sharded module
    shadow = ema.shadow_params["weight"]
    assert isinstance(shadow, DTensor)
    live = model.weight
    assert isinstance(live, DTensor)
    # Shadow holds a DTensor cloned from the local shard; values match live.
    torch.testing.assert_close(shadow.to_local(), live.to_local())

    with torch.no_grad():
        model.weight.copy_(torch.full_like(model.weight, 8.0))
    ema(model)
    # 0.5*orig + 0.5*8 elementwise.
    expected = 0.5 * torch.arange(16, dtype=torch.float32).reshape(4, 4) + 4.0
    torch.testing.assert_close(shadow.to_local(), expected)


# -- warmup-seed --------------------------------------------------------------


def test_ema_warmup_seed_copies_live_at_boundary() -> None:
    """warmup_seed copies live weights AT the warmup boundary before averaging.

    Without seeding, the shadow holds the (stale) values captured at the
    first call. With seeding, the shadow is reset to the live weights at
    the warmup boundary, then averaging begins.
    """
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA.Config(
        decay=0.5,
        shadow_kind="param_dict",
        update_after_step=2,
        warmup_seed=True,
    ).make()

    ema(model)  # step 0: warmup, shadow seeded at 1.0
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema(model)  # step 1: warmup
    with torch.no_grad():
        model.weight.fill_(9.0)
    ema(model)  # step 2: boundary -> seed shadow to live (9.0), no lerp yet
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(model.weight, 9.0),
    )

    with torch.no_grad():
        model.weight.fill_(11.0)
    ema(model)  # step 3: lerp 0.5*9 + 0.5*11 = 10.0
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(model.weight, 10.0),
    )


def test_ema_no_warmup_seed_keeps_initial_shadow() -> None:
    """Without warmup_seed the boundary lerps against the first-call shadow."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA.Config(
        decay=0.5,
        shadow_kind="param_dict",
        update_after_step=1,
        warmup_seed=False,
    ).make()

    ema(model)  # step 0: warmup, shadow = 1.0
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema(model)  # step 1: boundary, lerp 0.5*1 + 0.5*3 = 2.0
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(model.weight, 2.0),
    )


# -- Karras / step-warmup decay schedule --------------------------------------


def test_ema_karras_decay_grows_with_step() -> None:
    """Karras schedule: effective decay = min(decay, (1+t)/(10+t)) per step.

    Hand-computed against the live update. ``t`` is the post-warmup local
    step index used in the averaging call (0-based).
    """
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA.Config(
        decay=0.99,
        shadow_kind="param_dict",
        decay_schedule=karras_decay,
    ).make()

    # Reference mirrors the implementation op sequence (mul_ then add_) so
    # the comparison is bit-for-bit against the hand-derived karras decays.
    # With update_after_step=0 the first call IS averaging step t=0
    # (decay 1/10) against live=0.0 -> shadow stays 0.0, local_step -> 1.
    ref = torch.zeros_like(model.weight)
    ema(model)  # averaging step t=0: eff = 1/10, live 0.0 -> shadow 0.0
    ref.mul_(1 / 10).add_(model.weight.detach(), alpha=1 - 1 / 10)
    torch.testing.assert_close(ema.shadow_params["weight"], ref, rtol=0, atol=0)

    # Averaging step t=1: eff = min(0.99, 2/11) = 2/11.
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema(model)
    ref.mul_(2 / 11).add_(model.weight.detach(), alpha=1 - 2 / 11)
    torch.testing.assert_close(ema.shadow_params["weight"], ref, rtol=0, atol=0)

    # Averaging step t=2: eff = min(0.99, 3/12) = 3/12.
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema(model)
    ref.mul_(3 / 12).add_(model.weight.detach(), alpha=1 - 3 / 12)
    torch.testing.assert_close(ema.shadow_params["weight"], ref, rtol=0, atol=0)


def test_ema_karras_resume_continues_the_ramp() -> None:
    """A resumed karras run must average as if never interrupted.

    The schedule reads ``local_step``; omitting it from the checkpoint
    restarts the ramp and diverges the shadow.
    """

    def advance(ema: EMA, model: nn.Module, values: list[float]) -> None:
        for value in values:
            with torch.no_grad():
                model.weight.fill_(value)
            ema(model)

    model = nn.Linear(1, 1, bias=False)
    uninterrupted = EMA.Config(
        decay=0.99,
        shadow_kind="param_dict",
        decay_schedule=karras_decay,
    ).make()
    advance(uninterrupted, model, [0.0, 1.0, 2.0, 3.0])

    source = EMA.Config(
        decay=0.99,
        shadow_kind="param_dict",
        decay_schedule=karras_decay,
    ).make()
    advance(source, model, [0.0, 1.0, 2.0])
    resumed = EMA.Config(
        decay=0.99,
        shadow_kind="param_dict",
        decay_schedule=karras_decay,
    ).make()
    resumed.load_state_dict(source.state_dict())
    advance(resumed, model, [3.0])

    torch.testing.assert_close(
        resumed.shadow_params["weight"],
        uninterrupted.shadow_params["weight"],
        rtol=0,
        atol=0,
    )


def test_ema_rejects_a_schedule_outside_the_unit_interval() -> None:
    """A decay above 1 extrapolates the shadow instead of averaging it.

    ``decay`` is validated to ``[0, 1]``, but the schedule's result is what
    the lerp uses.
    """

    def overshoot(decay: float, step: int) -> float:
        del decay, step
        return 2.0

    model = nn.Linear(1, 1, bias=False)
    ema = EMA.Config(
        decay=0.9,
        shadow_kind="param_dict",
        decay_schedule=overshoot,
    ).make()

    with pytest.raises(ValueError, match="decay"):
        ema(model)


def test_ema_karras_schedule_helper_values() -> None:
    """The schedule helper returns hand-computed effective decay values."""
    ema = EMA.Config(decay=0.999, decay_schedule=karras_decay).make()
    assert ema.effective_decay(0) == pytest.approx(1 / 10)
    assert ema.effective_decay(5) == pytest.approx(6 / 15)
    assert ema.effective_decay(90) == pytest.approx(91 / 100)
    # Caps at the configured decay.
    assert ema.effective_decay(1_000_000) == pytest.approx(0.999)


def test_ema_constant_schedule_ignores_step() -> None:
    """Default constant schedule returns the configured decay at every step."""
    ema = EMA.Config(decay=0.9).make()
    assert ema.effective_decay(0) == pytest.approx(0.9)
    assert ema.effective_decay(10_000) == pytest.approx(0.9)


# -- param-dict state_dict round-trip -----------------------------------------


def test_ema_param_dict_state_dict_round_trips() -> None:
    """param_dict mode checkpoints name-keyed shadow and reloads cleanly."""
    model = nn.Linear(2, 2)
    src = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    src(model)
    with torch.no_grad():
        model.weight.fill_(4.0)
    src(model)
    state = src.state_dict()

    dst = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    dst(model)
    dst.load_state_dict(state)
    for name, value in src.shadow_params.items():
        torch.testing.assert_close(dst.shadow_params[name], value)
        # Independent storage.
        assert dst.shadow_params[name].data_ptr() != value.data_ptr()


def test_ema_param_dict_load_before_init_is_immediately_queryable() -> None:
    """load_state_dict on a FRESH (uninitialized) param_dict EMA populates the
    shadow at once -- no first ``__call__`` required.

    Regression for the resume bug: the load used to defer a param_dict shadow
    into pending state until the next update, so ``shadow_params`` was EMPTY
    right after a resume. Eval-on-resume (which reads the shadow before any
    training step) then saw nothing. A param_dict shadow is self-describing, so
    it must be adopted directly on load.
    """
    model = nn.Linear(2, 2)
    src = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    src(model)
    with torch.no_grad():
        model.weight.fill_(4.0)
    src(model)
    state = src.state_dict()

    # Fresh EMA, never called -> uninitialized. Load must populate immediately.
    dst = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    dst.load_state_dict(state)

    assert dst.shadow_params, "shadow empty after load (deferred until first step?)"
    assert dst.shadow_params.keys() == src.shadow_params.keys()
    for name, value in src.shadow_params.items():
        torch.testing.assert_close(dst.shadow_params[name], value)

    # A subsequent update continues the RESTORED moving average rather than
    # re-cloning the live weights: shadow := loaded*decay + live*(1-decay).
    loaded = {n: t.detach().clone() for n, t in dst.shadow_params.items()}
    with torch.no_grad():
        model.weight.fill_(10.0)
    live = {n: t.detach().clone() for n, t in model.named_parameters()}
    dst(model)
    for name, prev in loaded.items():
        expected = prev * 0.5 + live[name] * 0.5
        torch.testing.assert_close(dst.shadow_params[name], expected)


def test_ema_param_dict_apply_to_after_load_before_init_swaps_shadow() -> None:
    """Eval-only resume swaps a restored param_dict shadow before any update."""
    source_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source_model.weight.fill_(1.0)
    src = EMA.Config(decay=0.0, shadow_kind="param_dict").make()
    src(source_model)
    with torch.no_grad():
        source_model.weight.fill_(4.0)
    src(source_model)
    state = src.state_dict()

    eval_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        eval_model.weight.fill_(10.0)
    dst = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    dst.load_state_dict(state)

    with dst.apply_to(eval_model):
        torch.testing.assert_close(
            eval_model.weight,
            torch.full_like(eval_model.weight, 4.0),
        )
    torch.testing.assert_close(
        eval_model.weight,
        torch.full_like(eval_model.weight, 10.0),
    )


def test_ema_module_apply_to_after_load_before_init_swaps_shadow() -> None:
    """Eval-only resume swaps a restored module shadow before any update."""
    source_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source_model.weight.fill_(1.0)
    src = EMA.Config(decay=0.0, shadow_kind="module").make()
    src(source_model)
    with torch.no_grad():
        source_model.weight.fill_(4.0)
    src(source_model)
    state = src.state_dict()

    eval_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        eval_model.weight.fill_(10.0)
    dst = EMA.Config(decay=0.5, shadow_kind="module").make()
    dst.load_state_dict(state)

    with dst.apply_to(eval_model):
        torch.testing.assert_close(
            eval_model.weight,
            torch.full_like(eval_model.weight, 4.0),
        )
    torch.testing.assert_close(
        eval_model.weight,
        torch.full_like(eval_model.weight, 10.0),
    )


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="needs a second device to reproduce the cross-device restore",
)
def test_ema_loaded_shadow_moves_to_live_param_device() -> None:
    """A shadow restored on the wrong device re-devices to the live params.

    Regression for the multi-rank resume crash: a checkpoint is saved by rank 0
    (shadow on ``cuda:0``), and ``torch.load`` without remapping restores that
    device on EVERY rank. Rank 4's first EMA update then mixes ``cuda:0``
    shadow tensors with ``cuda:4`` live params and raises a cross-device
    RuntimeError. The lazy-init of a loaded shadow must re-device each tensor
    to its live counterpart.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    cpu_model = nn.Linear(2, 2)
    src = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    src(cpu_model)
    state = src.state_dict()  # shadow tensors live on CPU

    # Live model on the accelerator; shadow restored from the CPU checkpoint.
    dev_model = nn.Linear(2, 2).to(device)
    dst = EMA.Config(decay=0.5, shadow_kind="param_dict").make()
    dst.load_state_dict(state)

    # First update must not raise a cross-device error and must land the
    # shadow on the live device.
    dst(dev_model)
    for name, tensor in dst.shadow_params.items():
        assert tensor.device == dev_model.get_parameter(name).device


# -- bit-for-bit regression: constant-decay CLONE path unchanged --------------


def test_ema_clone_path_bit_for_bit_default_config() -> None:
    """INVARIANT: default (module clone, constant decay) shadow is unchanged.

    Drives ten updates with deterministic params and asserts the shadow
    values match the closed-form constant-decay EMA bit-for-bit. Guards
    against the param-dict/karras/warmup additions altering the default.
    """
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    ema = EMA.Config().make()  # all defaults: module clone, decay=0.9999

    ema(model)  # seed
    assert ema.shadow_model is not None
    shadow = cast(nn.Linear, ema.shadow_model)
    expected = shadow.weight.detach().clone()
    decay = 0.9999
    for k in range(1, 11):
        with torch.no_grad():
            model.weight.copy_(torch.full_like(model.weight, float(k)))
        ema(model)
        # Mirror the implementation's exact op sequence (mul_ then add_)
        # so the comparison is bit-for-bit, not algebraically rearranged.
        expected.mul_(decay).add_(model.weight.detach(), alpha=1 - decay)
        torch.testing.assert_close(shadow.weight, expected, rtol=0, atol=0)


# -- NoEMA uniform interface --------------------------------------------------


def test_no_ema_apply_to_is_noop() -> None:
    """NoEMA.apply_to yields without swapping any weights."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema = NoEMA()
    ema(model)
    pre = model.weight.detach().clone()
    with ema.apply_to(model):
        torch.testing.assert_close(model.weight, pre)
    torch.testing.assert_close(model.weight, pre)


@pytest.mark.parametrize("decay", [-0.1, 1.5])
def test_ema_rejects_out_of_range_decay(decay: float) -> None:
    """#343: decay must lie in [0, 1]; an out-of-range lerp coefficient is a bug."""
    with pytest.raises(ValueError, match="decay"):
        EMA.Config(decay=decay).make()


@pytest.mark.parametrize("update_every", [0, -1])
def test_ema_rejects_nonpositive_update_every(update_every: int) -> None:
    """#343: update_every<=0 would divide by zero in the modulo; reject at init."""
    with pytest.raises(ValueError, match="update_every"):
        EMA.Config(update_every=update_every).make()


def test_ema_rejects_negative_update_after_step() -> None:
    """#343: update_after_step is a warmup length and must be non-negative."""
    with pytest.raises(ValueError, match="update_after_step"):
        EMA.Config(update_after_step=-1).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
