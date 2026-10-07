"""Tests for TrainStep."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, overload, override

import functools
import math
import tempfile

from configgle import Fig, MutableNamespace, PartialConfig
from torch import Tensor, nn

import pytest
import torch
import torch.distributed as dist

from priml import runtime
from priml.metrics.binary_accuracy import BinaryAccuracy
from priml.timer import CheckpointableStepTimer
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep, _assert_uniform_microbatch_count


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.loss.custom_types import LossOutput


class _LogitsOutput(Protocol):
    logits: Tensor


class _LinearModel(nn.Module):
    """Simple logistic regression model for testing.

    Owns a ``Linear`` rather than subclassing it: these models take ``**kwargs``
    and (below) return a dict, neither of which ``Linear.forward`` declares.
    """

    class Config(Fig["_LinearModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @override
    def forward(self, x: Tensor, **_kwargs: object) -> Tensor:
        return self.linear(x).squeeze(-1)

    def reset_parameters(self) -> None:
        # Owner resets what it constructs: ``materialize`` allocates empty
        # storage and calls this once, so a child left out stays uninitialized.
        self.linear.reset_parameters()


def test_trainable_logistic_regression():
    """Test TrainStep on toy logistic regression problem."""
    torch.manual_seed(42)

    X = torch.randn(100, 2)
    label = (X[:, 0] + X[:, 1] > 0).float()

    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    assert isinstance(config.optimizer, MutableNamespace)
    config.optimizer.weight_decay = 0.0
    config.optimizer.lr = 0.1
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    trainable = config.make()

    initial_loss: float | None = None
    final_loss: float = 0.0
    for _ in range(50):
        loss_result = trainable.train_step(x=X, label=label)
        loss = loss_result["loss"]
        if initial_loss is None:
            initial_loss = loss.mean().item()
        final_loss = loss.mean().item()

    assert initial_loss is not None
    assert final_loss < initial_loss * 0.5

    metric = BinaryAccuracy.Config().make()
    with torch.no_grad():
        model = trainable.model
        assert isinstance(model, _LinearModel)
        output = model(X)
        metric.update(output, label=label)

    accuracy = metric.compute()["accuracy"]
    assert accuracy > 0.9


@pytest.mark.parametrize(
    "placement",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_torch_cuda)],
)
def test_meta_construction_draws_on_the_placement_devices_generator(
    placement: str,
) -> None:
    """Under ``"meta"`` init runs where the model will live, not on the host.

    Eager construction allocates on the CPU and draws from the CPU generator,
    then the strategy copies the finished weights across -- so the run's init
    consumes a different random stream than the same recipe materialized on
    the device, and every parameter makes the trip over the bus.
    """
    device = torch.device(placement)
    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=4, out_features=4)
    config.parallelism = NoParallel.Config(device=str(device))
    config.device_init = "meta"
    config.compile = None

    cpu_before = torch.get_rng_state()
    device_before = _rng_state(device)
    step = config.make()

    assert next(step.model.parameters()).device.type == device.type
    assert not torch.equal(device_before, _rng_state(device))
    if device.type == "cuda":
        # When the placement device IS the host, the CPU generator is the one
        # that must advance, so the untouched-host half only exists on CUDA.
        assert torch.equal(cpu_before, torch.get_rng_state())


def _rng_state(device: torch.device) -> Tensor:
    """Snapshot the default generator state of ``device``."""
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.get_rng_state()


def test_eager_construction_allocates_during_the_build() -> None:
    """``"eager"`` (today's default) builds with storage, then places.

    A module whose ``__init__`` reads its own tensor values -- or one whose
    ``reset_parameters`` does not cover everything it builds -- cannot survive
    the meta path, so this arm is what keeps them constructible.
    """
    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=4, out_features=4)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    step = config.make()

    assert config.device_init == "eager"
    assert next(step.model.parameters()).device == torch.device("cpu")


def test_device_init_names_how_not_where() -> None:
    """The field chooses allocation strategy; ``parallelism`` owns the device.

    Two fields that each name a device disagree silently -- one builds on
    ``cuda:0`` while the other places on ``cuda:1`` -- so this one no longer
    accepts a device at all.
    """
    config = TrainStep.Config()
    # Assigned as an ``--override`` or a deserialized config delivers it: the
    # annotation rules this out statically, so the runtime guard is what
    # catches text that never met a type checker.
    object.__setattr__(config, "device_init", "cuda")
    with pytest.raises(ValueError, match="device_init"):
        config.make()


def test_trainable_train_step():
    """Test TrainStep.train_step with gradient accumulation."""
    torch.manual_seed(42)

    X = torch.randn(20, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    assert isinstance(config.optimizer, MutableNamespace)
    config.optimizer.weight_decay = 0.0
    config.optimizer.lr = 0.1
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.accumulate_grad_batches = 4

    trainable = config.make()

    losses: list[float] = []
    for _ in range(20):
        loss_result = trainable.train_step(x=X, label=label)
        losses.append(loss_result["loss"].mean().item())

    assert losses[-1] < losses[0]
    assert trainable.global_step == 5


def test_trainable_eval_loss():
    """Test TrainStep.eval_loss."""
    torch.manual_seed(42)

    X = torch.randn(10, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    trainable = config.make()

    result = trainable.eval_loss(x=X, label=label)
    assert "loss" in result
    assert result["loss"].mean().item() > 0


def test_trainable_checkpointing():
    """Test TrainStep state_dict saves and restores correctly."""
    torch.manual_seed(42)

    X = torch.randn(20, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    assert isinstance(config.optimizer, MutableNamespace)
    config.optimizer.weight_decay = 0.0
    config.optimizer.lr = 0.1
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    trainable = config.make()

    for _ in range(10):
        trainable.train_step(x=X, label=label)

    state = trainable.state_dict()

    with torch.no_grad():
        model = trainable.model
        assert isinstance(model, _LinearModel)
        output_before = model(X)
        loss_before = trainable.loss(output_before, label=label)

    trainable2 = config.make()
    assert trainable2.global_step == 0

    trainable2.load_state_dict(state)
    assert trainable2.global_step == 10

    with torch.no_grad():
        model = trainable2.model
        assert isinstance(model, _LinearModel)
        output_after = model(X)
        loss_after = trainable2.loss(output_after, label=label)

    torch.testing.assert_close(loss_before, loss_after)


def test_autocast_cache_enabled_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-045: autocast cache_enabled must be configurable (was hardcoded False)."""
    config = TrainStep.Config()
    # Field exists with a numerics-preserving default.
    assert config.autocast_cache_enabled is False

    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = torch.bfloat16
    config.autocast_cache_enabled = True

    step = config.make()

    seen: list[bool | None] = []
    orig = torch.amp.autocast

    def spy(*args: object, **kwargs: object) -> object:
        cache_enabled = kwargs.get("cache_enabled")
        assert cache_enabled is None or isinstance(cache_enabled, bool)
        device_type = kwargs.get("device_type")
        assert isinstance(device_type, str)
        enabled = kwargs.get("enabled", True)
        assert isinstance(enabled, bool)
        seen.append(cache_enabled)
        return orig(
            *args,
            device_type=device_type,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )

    monkeypatch.setattr(torch.amp, "autocast", spy)
    step(x=torch.randn(4, 2))

    assert seen == [True], seen


def test_call_eval_configures_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _linear_step(
        dtype_autocast=torch.bfloat16,
        autocast_cache_enabled=True,
    )
    seen: list[tuple[object, object, object]] = []
    orig = torch.amp.autocast

    def spy(*args: object, **kwargs: object) -> object:
        device_type = kwargs.get("device_type")
        dtype = kwargs.get("dtype")
        cache_enabled = kwargs.get("cache_enabled")
        enabled = kwargs.get("enabled", True)
        assert isinstance(device_type, str)
        assert dtype is None or isinstance(dtype, torch.dtype)
        assert cache_enabled is None or isinstance(cache_enabled, bool)
        assert isinstance(enabled, bool)
        seen.append((device_type, dtype, cache_enabled))
        return orig(
            *args,
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )

    monkeypatch.setattr(torch.amp, "autocast", spy)
    step.call_eval(torch.randn(4, 2))

    assert seen == [("cpu", torch.bfloat16, True)]


def test_train_step_refuses_to_checkpoint_pending_accumulation() -> None:
    """A checkpoint cannot resume gradients that its state does not contain."""
    torch.manual_seed(42)
    X = torch.randn(8, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.accumulate_grad_batches = 4

    trainable = config.make()
    # Two micro-batches: mid-accumulation (2 of 4).
    trainable.train_step(x=X, label=label)
    trainable.train_step(x=X, label=label)
    assert trainable.accumulation_steps == 2

    with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
        trainable.state_dict()


class _DictModel(nn.Module):
    """Model returning a dict output (multi-output contract for T-047)."""

    class Config(Fig["_DictModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @override
    def forward(self, x: Tensor, **_kwargs: object) -> dict[str, Tensor]:
        return {"logits": self.linear(x).squeeze(-1)}

    def reset_parameters(self) -> None:
        # Owner resets what it constructs: ``materialize`` allocates empty
        # storage and calls this once, so a child left out stays uninitialized.
        self.linear.reset_parameters()


def _loss_from_logits_dict(
    output: object,
    *,
    label: Tensor,
    **_kwargs: object,
) -> LossOutput:
    """Loss that consumes a ModelOutput by indexing its ``logits`` entry."""
    assert isinstance(output, dict)
    output_dict = cast(dict[str, object], output)
    logits: object = output_dict["logits"]
    assert isinstance(logits, Tensor)
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        ),
    }


def test_multi_output_model_conforms_to_model_output_protocol() -> None:
    """T-047: a dict/struct-returning model is typed via ModelOutput, not cast.

    A model returning a non-Tensor output must flow through the loss via the
    ModelOutput protocol path. Previously ``cast(Tensor, self(...))`` silently
    mis-typed the output as a bare Tensor.
    """
    torch.manual_seed(42)
    X = torch.randn(16, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _DictModel.Config(in_features=2, out_features=1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.loss = PartialConfig(_loss_from_logits_dict)

    step = config.make()
    result = step.train_step(x=X, label=label)
    assert "loss" in result
    assert result["loss"].mean().item() > 0


class _BadModel(nn.Module):
    """Model whose output violates the ModelOutput contract (None)."""

    class Config(Fig["_BadModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        del in_features, out_features, bias
        super().__init__()

    @override
    def forward(self, x: Tensor, **_kwargs: object) -> object:
        del x
        return None


def test_model_output_contract_violation_raises_clearly() -> None:
    """T-047: a model output that is neither Tensor nor ModelOutput raises."""
    torch.manual_seed(42)
    X = torch.randn(8, 2)
    label = (X.sum(dim=1) > 0).float()

    config = TrainStep.Config()
    config.model = _BadModel.Config(in_features=2, out_features=1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    step = config.make()
    with pytest.raises(TypeError, match=r"ModelOutput"):
        step.train_step(x=X, label=label)


def _unequal_micro_batches() -> tuple[Tensor, Tensor, list[int]]:
    """Build data plus a partition into UNEQUAL micro-batch sizes."""
    torch.manual_seed(7)
    x = torch.randn(10, 3)
    label = (x.sum(dim=1) > 0).float()
    # Deliberately unequal: 7 + 3, not 5 + 5.
    return x, label, [7, 3]


def test_grad_accum_equals_single_batch_unequal_micro_sizes() -> None:
    """T-020: accumulate(N unequal micro-batches) == one big batch, exactly.

    The load-bearing invariant. With per-element loss summed across
    micro-batches and grads divided ONCE by the grand-total element count,
    UNEQUAL micro-batch sizes must still reproduce the single-batch gradient.
    Mean-of-means would fail this specific case.
    """
    x, label, splits = _unequal_micro_batches()

    def build() -> TrainStep:
        config = TrainStep.Config()
        config.model = _LinearModel.Config(in_features=3, out_features=1)
        config.parallelism = NoParallel.Config(device="cpu")
        config.compile = None
        config.loss = PartialConfig(_binary_cross_entropy_with_logits)
        config.accumulate_grad_batches = len(splits)
        step = config.make()
        for p in step.model.parameters():
            torch.nn.init.zeros_(p)
        return step

    # Reference: one big batch, accumulate_grad_batches == 1.
    ref = build()
    ref.accumulate_grad_batches = 1
    ref.train_step(x=x, label=label)
    ref_grads = ref.last_microbatch_grads

    # Accumulated: unequal micro-batches 7 then 3.
    acc = build()
    offset = 0
    for n in splits:
        sl = slice(offset, offset + n)
        acc.train_step(x=x[sl], label=label[sl])
        offset += n
    acc_grads = acc.last_microbatch_grads

    assert len(ref_grads) == len(acc_grads)
    # The math is identical (sum of per-element grads / grand-total count). The
    # only difference is float reduction GROUPING: one 10-element reduction vs
    # a 7-element plus a 3-element reduction accumulated in .grad. That is
    # float32 reassociation, bounded by machine epsilon -- not an algorithmic
    # error like mean-of-means, which would diverge by O(1) for unequal sizes.
    for g_ref, g_acc in zip(ref_grads, acc_grads, strict=True):
        torch.testing.assert_close(g_ref, g_acc, rtol=1e-6, atol=1e-7)


def _binary_cross_entropy_with_logits(
    output: object,
    *,
    label: Tensor,
    **_kwargs: object,
) -> LossOutput:
    """Per-element BCE-with-logits loss (reduction='none')."""
    if isinstance(output, dict):
        output_dict = cast(dict[str, object], output)
        logits: object = output_dict["logits"]
    elif isinstance(output, Tensor):
        logits = output
    else:
        assert hasattr(output, "logits")
        logits = cast(_LogitsOutput, output).logits
    assert isinstance(logits, Tensor)
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        ),
    }


class _CountingModel(nn.Module):
    """Logistic-regression model that counts its forward calls."""

    forward_count = 0

    class Config(Fig["_CountingModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @override
    def forward(self, x: Tensor, **_kwargs: object) -> Tensor:
        self.forward_count += 1
        return self.linear(x).squeeze(-1)

    def reset_parameters(self) -> None:
        # Owner resets what it constructs: ``materialize`` allocates empty
        # storage and calls this once, so a child left out stays uninitialized.
        self.linear.reset_parameters()


def test_first_order_optimizer_runs_one_forward_per_step() -> None:
    """A first-order optimizer must NOT trigger a second (closure) forward.

    ``Learnable.step`` only forwards the loss-recompute closure to optimizers
    that set ``requires_closure``. A torch optimizer (default AdamW) executes
    any closure it receives, so a leaked closure would run a wasteful second
    forward every step and double-count BatchNorm stats. Exactly one forward
    per ``train_step`` is the contract.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 2)
    label = (x[:, 0] + x[:, 1] > 0).float()

    config = TrainStep.Config()
    config.model = _CountingModel.Config(in_features=2, out_features=1)
    config.loss = PartialConfig(_binary_cross_entropy_with_logits)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    trainable = config.make()

    model = trainable.model
    assert isinstance(model, _CountingModel)
    model.forward_count = 0
    trainable.train_step(x=x, label=label)
    assert model.forward_count == 1, (
        f"expected 1 forward, got {model.forward_count} "
        "(a closure leaked to a first-order optimizer?)"
    )


def _nan_loss(output: Tensor, **kwargs: object) -> LossOutput:
    """Per-element loss whose gradient is NaN everywhere."""
    del kwargs
    return {"loss": output * math.nan}


def _nan_grad_step(*, skip_step_on_nonfinite_grad: bool) -> TrainStep:
    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.loss = PartialConfig(_nan_loss)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.skip_step_on_nonfinite_grad = skip_step_on_nonfinite_grad
    return config.make()


def test_nonfinite_grad_skips_update_when_enabled() -> None:
    torch.manual_seed(0)
    trainable = _nan_grad_step(skip_step_on_nonfinite_grad=True)
    before = [p.detach().clone() for p in trainable.model.parameters()]

    result = trainable.train_step(x=torch.randn(4, 2))

    for param, original in zip(trainable.model.parameters(), before, strict=True):
        assert torch.equal(param, original)
        assert param.grad is None
    assert trainable.skipped_steps == 1
    assert trainable.global_step == 1
    assert result.get("metrics") == {"skipped_steps": 1}


def test_nonfinite_grad_reads_the_norm_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The skip check is one device sync, not one per log field."""
    torch.manual_seed(0)
    trainable = _nan_grad_step(skip_step_on_nonfinite_grad=True)
    reads = 0
    original = Tensor.item

    def counted(self: Tensor) -> float:
        nonlocal reads
        reads += 1
        return original(self)

    monkeypatch.setattr(Tensor, "item", counted)
    trainable.train_step(x=torch.randn(4, 2))
    assert reads == 1


def test_skipped_steps_survive_a_checkpoint_round_trip() -> None:
    """A resumed run keeps reporting the skips the checkpointed one made."""
    torch.manual_seed(0)
    trainable = _nan_grad_step(skip_step_on_nonfinite_grad=True)
    trainable.train_step(x=torch.randn(4, 2))
    state = trainable.state_dict()
    assert state.get("skipped_steps") == 1

    resumed = _nan_grad_step(skip_step_on_nonfinite_grad=True)
    resumed.load_state_dict(state)
    assert resumed.skipped_steps == 1
    resumed.train_step(x=torch.randn(4, 2))
    assert resumed.skipped_steps == 2


def test_a_checkpoint_without_skipped_steps_loads_at_zero() -> None:
    """A checkpoint written before the counter existed still resumes."""
    trainable = _nan_grad_step(skip_step_on_nonfinite_grad=True)
    state = trainable.state_dict()
    del state["skipped_steps"]
    trainable.skipped_steps = 3
    trainable.load_state_dict(state)
    assert trainable.skipped_steps == 0


def test_nonfinite_grad_corrupts_parameters_when_disabled() -> None:
    torch.manual_seed(0)
    trainable = _nan_grad_step(skip_step_on_nonfinite_grad=False)

    result = trainable.train_step(x=torch.randn(4, 2))

    assert all(p.isnan().all() for p in trainable.model.parameters())
    assert trainable.skipped_steps == 0
    assert "skipped_steps" not in result.get("metrics", {})


def _linear_step(**overrides: object) -> TrainStep:
    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.loss = PartialConfig(_binary_cross_entropy_with_logits)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gradient_clip_norm", 0.0, "gradient_clip_norm must be positive"),
        ("accumulate_grad_batches", 0, "accumulate_grad_batches must be positive"),
        ("train_budget_steps", 0.0, "train_budget_steps must be positive"),
        ("train_budget_sec", -1.0, "train_budget_sec must be positive"),
        ("train_budget_epochs", math.nan, "train_budget_epochs must be positive"),
    ],
)
def test_construction_rejects_a_non_positive_budget_or_clip(
    field: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _linear_step(**{field: value})


def test_local_step_counts_this_process_only() -> None:
    step = _linear_step()
    x = torch.randn(4, 2)
    label = (x.sum(dim=1) > 0).float()
    step.train_step(x=x, label=label)
    state = step.state_dict()
    resumed = _linear_step()
    resumed.load_state_dict(state)
    assert resumed.global_step == 1
    assert resumed.local_step == 0
    resumed.train_step(x=x, label=label)
    assert resumed.global_step == 2
    assert resumed.local_step == 1


def test_bound_epoch_timer_drives_the_epoch_budget() -> None:
    step = _linear_step(train_budget_epochs=4.0)
    timer = CheckpointableStepTimer()
    step.bind_epoch_timer(timer)
    assert step.progress_complete == 0.0
    timer.global_count = 3
    assert step.progress_complete == pytest.approx(0.75)
    assert step.progress_learning_schedule == pytest.approx(0.75)


def test_progress_uses_the_first_exhausted_budget_and_caps_at_one() -> None:
    step = _linear_step(
        train_budget_steps=8.0,
        train_budget_sec=20.0,
        train_budget_epochs=4.0,
    )
    epochs = CheckpointableStepTimer()
    step.bind_epoch_timer(epochs)

    step.timer_step.global_count = 2
    step.timer_step.global_sec = 10.0
    epochs.global_count = 3
    assert step.progress_complete == pytest.approx(0.75)

    epochs.global_count = 5
    assert step.progress_complete == 1.0


def test_preprocess_moves_tensors_and_leaves_other_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _linear_step()
    tensor = torch.zeros(2)
    calls: list[tuple[object, dict[str, object]]] = []
    original_to = Tensor.to

    def record_to(
        self: Tensor,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        args: tuple[object, ...] = () if device is None else (device,)
        kwargs: dict[str, object] = {"non_blocking": non_blocking}
        calls.append((args, kwargs))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(Tensor, "to", record_to)
    batch = step.preprocess_batch({"x": tensor, "name": "puzzle", "n": 3})
    moved = batch["x"]
    assert isinstance(moved, Tensor)
    assert moved.device == torch.device("cpu")
    assert calls == [((torch.device("cpu"),), {"non_blocking": False})]
    assert batch["name"] == "puzzle"
    assert batch["n"] == 3


def test_preprocess_uses_non_blocking_transfer_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _linear_step()
    step.parallelism.device = torch.device("cuda", 1)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_to = Tensor.to

    def record_to(
        self: Tensor,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        args: tuple[object, ...] = () if device is None else (device,)
        kwargs: dict[str, object] = {"non_blocking": non_blocking}
        calls.append((args, kwargs))
        return original_to(self, torch.device("cpu"), non_blocking=non_blocking)

    monkeypatch.setattr(Tensor, "to", record_to)
    moved = step.preprocess_batch({"x": torch.zeros(2)})["x"]

    assert isinstance(moved, Tensor)
    assert moved.device == torch.device("cpu")
    assert calls == [
        ((torch.device("cuda", 1),), {"non_blocking": True}),
    ]


def test_train_and_eval_losses_preserve_model_output() -> None:
    step = _linear_step()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    label = torch.tensor([0.0, 1.0])
    train_result = step.train_loss(x=x, label=label)
    eval_result = step.eval_loss(x=x, label=label)
    expected = cast(Tensor, step.model(x))

    assert torch.equal(train_result["model"], expected)
    assert torch.equal(eval_result["model"], expected)
    assert train_result["model"].shape == (2,)
    assert eval_result["model"].shape == (2,)


def test_call_eval_passes_positional_arguments_and_applies_ema() -> None:
    step = _linear_step(ema=EMA.Config(decay=0.5))
    step.ema(step.model)
    shadow = [p.detach().clone() for p in step.model.parameters()]
    with torch.no_grad():
        for parameter in step.model.parameters():
            parameter.add_(1)
    current = [p.detach().clone() for p in step.model.parameters()]
    seen: list[tuple[object, ...]] = []
    original_forward = step.model.forward

    def forward(x: Tensor) -> Tensor:
        args = (x,)
        seen.append(args)
        y = cast(object, original_forward(x))
        assert isinstance(y, Tensor)
        return y

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(step.model, "forward", forward)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with torch.no_grad():
        expected = cast(object, original_forward(x))
        assert isinstance(expected, Tensor)
    output = step.call_eval(x)

    with torch.no_grad():
        for parameter, value in zip(step.model.parameters(), shadow, strict=True):
            parameter.copy_(value)
        expected = cast(object, original_forward(x))
        assert isinstance(expected, Tensor)
        for parameter, value in zip(step.model.parameters(), current, strict=True):
            parameter.copy_(value)

    assert isinstance(output, Tensor)
    assert torch.equal(output, expected)
    assert len(seen) == 1
    assert seen[0] == (x,)
    assert not output.requires_grad
    assert all(
        torch.equal(p, value)
        for p, value in zip(step.model.parameters(), current, strict=True)
    )


def test_compile_slot_wraps_the_model_once() -> None:
    wrapped: list[object] = []

    def compile_fn(model: object) -> object:
        wrapped.append(model)
        assert callable(model)
        return model

    step = _linear_step(compile=PartialConfig(compile_fn))
    x = torch.randn(4, 2)
    step(x=x)
    step(x=x)
    assert wrapped == [step.model]


def test_closure_reaches_an_optimizer_that_requires_it() -> None:
    closures: list[object] = []

    class _ClosureOptimizer(torch.optim.SGD):
        requires_closure = True

        @overload
        def step(self, closure: None = None) -> None: ...

        @overload
        def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

        @override
        def step(
            self,
            closure: Callable[[], Tensor | float] | None = None,
        ) -> Tensor | float | None:
            closures.append(closure)
            super().step()
            return None

    step = _linear_step(optimizer=PartialConfig(_ClosureOptimizer, lr=0.1))
    step.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    assert isinstance(step.optimizer, _ClosureOptimizer)
    assert len(closures) == 1
    closure = closures[0]
    assert callable(closure)
    recomputed = closure()
    assert isinstance(recomputed, Tensor)
    assert recomputed.requires_grad


def _record_output(
    outputs: list[Tensor],
    module: nn.Module,
    args: tuple[object, ...],
    output: Tensor,
) -> None:
    del module, args
    outputs.append(output)


def test_train_on_output_matches_train_step_without_a_second_forward() -> None:
    torch.manual_seed(0)
    x, label = torch.randn(4, 2), torch.tensor([0.0, 1.0, 1.0, 0.0])
    reference = _linear_step()
    via_output = _linear_step()
    via_output.model.load_state_dict(reference.model.state_dict())
    model = via_output.model
    assert isinstance(model, _LinearModel)
    calls: list[Tensor] = []
    model.linear.register_forward_hook(functools.partial(_record_output, calls))

    expected = reference.train_step(x=x, label=label)
    result = via_output.train_on_output(via_output(x=x), x=x, label=label)

    assert len(calls) == 1
    assert via_output.global_step == 1
    torch.testing.assert_close(result["loss"], expected["loss"])
    for ours, theirs in zip(
        via_output.model.parameters(),
        reference.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(ours, theirs)


def _weighted_bce(
    output: object,
    *,
    label: Tensor,
    weight: Tensor,
    **_kwargs: object,
) -> LossOutput:
    """BCE scaled by a per-element ``weight`` the model never sees."""
    assert isinstance(output, Tensor)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        output,
        label,
        reduction="none",
    )
    return {"loss": bce * weight}


class _StrictLinear(nn.Module):
    """Takes ``x`` and nothing else, so a loss-only key reaching it raises."""

    class Config(Fig["_StrictLinear"]): ...

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.linear = nn.Linear(2, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)


def test_train_on_output_passes_loss_only_keys_to_the_loss_alone() -> None:
    step = _linear_step(
        model=_StrictLinear.Config(),
        loss=PartialConfig(_weighted_bce),
    )
    x = torch.randn(4, 2)
    result = step.train_on_output(
        step(x=x),
        label=torch.ones(4),
        weight=torch.full((4,), 2.0),
    )
    assert result["loss"].shape == (4,)


class _NormedModel(nn.Module):
    """Linear behind a BatchNorm, so training mode changes state."""

    class Config(Fig["_NormedModel"], make_with_kwargs=True): ...

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.linear = nn.Linear(2, 1)

    @override
    def forward(self, x: Tensor, **_kwargs: object) -> Tensor:
        return self.linear(self.norm(x)).squeeze(-1)

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.linear.reset_parameters()


def test_call_frozen_keeps_state_and_parameters_out_of_the_graph() -> None:
    torch.manual_seed(0)
    step = _linear_step(model=_NormedModel.Config())
    step.model.train()
    norm = step.model.get_submodule("norm")
    assert isinstance(norm, nn.BatchNorm1d)
    step.model.get_parameter("linear.bias").requires_grad_(False)
    running_mean = norm.running_mean
    assert running_mean is not None
    before = running_mean.clone()

    x = torch.randn(4, 2, requires_grad=True)
    output = step.call_frozen(x=x)
    assert isinstance(output, Tensor)
    output.sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad) > 0
    assert all(p.grad is None for p in step.model.parameters())
    torch.testing.assert_close(running_mean, before, rtol=0, atol=0)
    assert step.model.training
    assert norm.training
    assert step.model.get_parameter("linear.weight").requires_grad
    assert not step.model.get_parameter("linear.bias").requires_grad


def test_call_frozen_restores_state_when_the_forward_raises() -> None:
    step = _linear_step(model=_NormedModel.Config())
    step.model.train()
    step.model.get_submodule("norm").eval()  # A user-chosen submodule mode.
    frozen = step.model.get_parameter("linear.bias")
    frozen.requires_grad_(False)
    with pytest.raises(RuntimeError):
        step.call_frozen(x=torch.randn(4, 3))  # Wrong width.
    assert step.model.training
    assert not step.model.get_submodule("norm").training
    assert step.model.get_parameter("linear.weight").requires_grad
    assert not frozen.requires_grad


def test_call_eval_restores_submodule_modes() -> None:
    step = _linear_step(model=_NormedModel.Config())
    step.model.train()
    step.model.get_submodule("norm").eval()
    step.call_eval(x=torch.randn(4, 2))
    assert step.model.training
    assert not step.model.get_submodule("norm").training


def test_a_scalar_loss_is_refused() -> None:
    def reduced(output: object, **kwargs: object) -> LossOutput:
        del kwargs
        assert isinstance(output, Tensor)
        return {"loss": output.mean()}

    step = _linear_step(loss=PartialConfig(reduced))
    with pytest.raises(ValueError, match="Loss must be unreduced"):
        step.train_step(x=torch.randn(4, 2))


def test_train_loss_computes_without_an_update() -> None:
    step = _linear_step()
    before = [p.detach().clone() for p in step.model.parameters()]
    result = step.train_loss(x=torch.randn(4, 2), label=torch.zeros(4))
    assert result["loss"].shape == (4,)
    assert step.global_step == 0
    for param, original in zip(step.model.parameters(), before, strict=True):
        assert torch.equal(param, original)


def test_eval_loss_leaves_train_mode_as_it_found_it() -> None:
    step = _linear_step()
    step.model.eval()
    step.eval_loss(x=torch.randn(4, 2), label=torch.zeros(4))
    assert not step.model.training
    step.model.train()
    step.eval_loss(x=torch.randn(4, 2), label=torch.zeros(4))
    assert step.model.training


def test_epoch_end_discards_a_partial_accumulation_by_default() -> None:
    step = _linear_step(accumulate_grad_batches=3)
    step.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    assert step.accumulation_steps == 1
    step.on_epoch_end()
    assert step.accumulation_steps == 0
    assert step.accumulated_samples == 0
    assert all(p.grad is None for p in step.model.parameters())


def test_epoch_end_keeps_a_partial_accumulation_when_opted_out() -> None:
    step = _linear_step(
        accumulate_grad_batches=3,
        drop_partial_accumulation_on_epoch_end=False,
    )
    step.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    step.on_epoch_end()
    assert step.accumulation_steps == 1
    assert step.accumulated_samples == 4


def test_epoch_end_is_a_noop_with_nothing_pending() -> None:
    step = _linear_step()
    step.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    step.on_epoch_end()
    assert step.accumulation_steps == 0


def test_load_state_dict_can_remap_keys_and_skip_the_optimizer() -> None:
    torch.manual_seed(0)
    source = _linear_step()
    source.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    state = source.state_dict()
    renamed = {f"legacy.{key}": value for key, value in state["model"].items()}
    state["model"] = renamed

    target = _linear_step()
    target.load_state_dict(
        state,
        load_optimizer=False,
        remap=lambda model_state: {
            key.removeprefix("legacy."): value for key, value in model_state.items()
        },
    )

    for restored, original in zip(
        target.model.parameters(),
        source.model.parameters(),
        strict=True,
    ):
        assert torch.equal(restored, original)
    assert target.global_step == 1
    optimizer_state = target.optimizer.state_dict()["state"]
    assert isinstance(optimizer_state, dict)
    assert len(cast(dict[object, object], optimizer_state)) == 0


def test_load_state_dict_tolerates_a_checkpoint_without_timers() -> None:
    source = _linear_step()
    source.train_step(x=torch.randn(4, 2), label=torch.zeros(4))
    state: dict[str, object] = dict(source.state_dict())
    for key in ("timer_forward", "timer_eval", "timer_step", "ema"):
        del state[key]
    target = _linear_step()
    target.load_state_dict(state)
    assert target.global_step == 0


_MESH_SEAM: Final = "priml.train.train_step.global_device_mesh"
"""Patched by dotted path: a local named ``step`` shadows a module import."""


def _gloo_backend(group: object) -> str:
    del group
    return "gloo"


def _world_of_two(group: object = None) -> int:
    del group
    return 2


def test_uniform_count_guard_is_a_noop_on_a_single_rank_dp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-rank world whose ``dp`` axis is 1 wide has no peers to compare."""
    dp_group = object()

    class _Mesh:
        mesh_dim_names = ("dp", "tp")

        def get_group(self, name: str) -> object:
            assert name == "dp"
            return dp_group

    def world_size(group: object = None) -> int:
        return 1 if group is dp_group else 2

    reduced: list[object] = []

    def all_reduce(extremes: Tensor, op: object, group: object) -> None:
        del extremes, op, group
        reduced.append(1)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", world_size)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(_MESH_SEAM, _Mesh)
    _assert_uniform_microbatch_count(8)
    assert reduced == []


def test_uniform_count_guard_reduces_extremes_over_the_dp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MIN and MAX ride one MAX-reduce; unequal extremes raise."""
    dp_group = object()

    class _Mesh:
        mesh_dim_names = ("dp",)

        def get_group(self, name: str) -> object:
            assert name == "dp"
            return dp_group

    peer_count = 5

    def all_reduce(extremes: Tensor, op: object, group: object) -> None:
        assert group is dp_group
        assert op is dist.ReduceOp.MAX
        extremes[0] = max(int(extremes[0]), peer_count)
        extremes[1] = max(int(extremes[1]), -peer_count)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", _world_of_two)
    monkeypatch.setattr(dist, "get_backend", _gloo_backend)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(_MESH_SEAM, _Mesh)

    _assert_uniform_microbatch_count(5)
    with pytest.raises(ValueError, match="between 3 and 5 elements"):
        _assert_uniform_microbatch_count(3)


def test_uniform_count_guard_falls_back_to_world_without_a_dp_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups: list[object] = []

    def all_reduce(extremes: Tensor, op: object, group: object) -> None:
        del extremes, op
        groups.append(group)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", _world_of_two)
    monkeypatch.setattr(dist, "get_backend", _gloo_backend)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(_MESH_SEAM, lambda: None)
    _assert_uniform_microbatch_count(4)
    assert groups == [None]


def test_assert_uniform_microbatch_count_single_process_noop() -> None:
    """#340: the cross-rank guard is a no-op without an initialized group."""
    # No distributed group: must not raise regardless of the count.
    _assert_uniform_microbatch_count(3)
    _assert_uniform_microbatch_count(5)


def _uniform_count_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Worker: equal per-rank counts pass; unequal counts raise ValueError."""
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh

        # Equal counts across ranks: must pass.
        _assert_uniform_microbatch_count(8)

        # Unequal counts (rank 0 -> 3, rank 1 -> 5): must raise on every rank.
        local_count = 3 if rank == 0 else 5
        (result_dir / f"rank_{rank}").write_text(_unequal_outcome(local_count))
    except Exception as e:  # noqa: BLE001 -- The worker must surface any failure to its parent process.
        (result_dir / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


def _unequal_outcome(local_count: int) -> str:
    """``ok`` when the uniform-count assertion raises, else a FAIL record."""
    try:
        _assert_uniform_microbatch_count(local_count)
    except ValueError:
        return "ok"
    return "FAIL:no-raise-on-unequal"


@pytest.mark.cli_python_subprocess
def test_assert_uniform_microbatch_count_across_ranks(
    warm_pools: WarmPoolGetter,
) -> None:
    """#340: equal local-N passes; unequal local-N raises across a DP group."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_uniform_count_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
