"""Tests for TrainStep."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override

import functools
import math
import tempfile

from configgle import Fig, MutableNamespace, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml import runtime
from priml.loss.custom_types import LossOutput
from priml.metrics.binary_accuracy import BinaryAccuracy
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep, _assert_uniform_microbatch_count


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


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
