"""Tests for TrainStep."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import functools
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
    from priml.distributed.testing import WarmPoolGetter


class _LinearModel(nn.Linear):
    """Simple logistic regression model for testing."""

    class Config(Fig["_LinearModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> Tensor:  # ty: ignore[invalid-method-override]
        return super().forward(x).squeeze(-1)


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
        output = trainable.model(X)
        metric.update(output, label=label)

    accuracy = metric.compute()["accuracy"]
    assert accuracy > 0.9


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
    assert isinstance(result, dict)
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
        output_before = trainable.model(X)
        loss_before = trainable.loss(output_before, label=label)

    trainable2 = config.make()
    assert trainable2.global_step == 0

    trainable2.load_state_dict(state)
    assert trainable2.global_step == 10

    with torch.no_grad():
        output_after = trainable2.model(X)
        loss_after = trainable2.loss(output_after, label=label)

    torch.testing.assert_close(loss_before, loss_after)


def test_autocast_cache_enabled_is_configurable() -> None:
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

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("cache_enabled"))
        return orig(*args, **kwargs)

    torch.amp.autocast = spy  # ty: ignore[invalid-assignment]
    try:
        step(x=torch.randn(4, 2))
    finally:
        torch.amp.autocast = orig

    assert seen == [True], seen


def test_train_step_state_dict_records_accumulation_counters() -> None:
    """T-004: state_dict must record grad-accumulation counters."""
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

    state = trainable.state_dict()
    assert "accumulation_steps" in state
    assert "accumulated_samples" in state
    assert state["accumulation_steps"] == 2


class _DictModel(nn.Linear):
    """Model returning a dict output (multi-output contract for T-047)."""

    class Config(Fig["_DictModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> dict[str, Tensor]:  # ty: ignore[invalid-method-override] -- multi-output model returns a dict, not the nn.Linear Tensor
        return {"logits": super().forward(x).squeeze(-1)}


def _loss_from_logits_dict(
    output: Any,
    *,
    label: Tensor,
    **_kwargs: Any,
) -> LossOutput:
    """Loss that consumes a ModelOutput by indexing its ``logits`` entry."""
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"],
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


class _BadModel(nn.Linear):
    """Model whose output violates the ModelOutput contract (None)."""

    class Config(Fig["_BadModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> Any:  # ty: ignore[invalid-method-override] -- deliberately violates the ModelOutput contract for the negative test
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
    output: Any,
    *,
    label: Tensor,
    **_kwargs: Any,
) -> LossOutput:
    """Per-element BCE-with-logits loss (reduction='none')."""
    logits = output.logits if hasattr(output, "logits") else output
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        ),
    }


class _CountingModel(nn.Linear):
    """Logistic-regression model that counts its forward calls."""

    forward_count = 0

    class Config(Fig["_CountingModel"], make_with_kwargs=True):
        in_features: int = -1
        out_features: int = -1
        bias: bool = True

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> Tensor:  # ty: ignore[invalid-method-override]
        self.forward_count += 1
        return super().forward(x).squeeze(-1)


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


def test_assert_uniform_microbatch_count_single_process_noop() -> None:
    """#340: the cross-rank guard is a no-op without an initialized group."""
    # No distributed group: must not raise regardless of the count.
    _assert_uniform_microbatch_count(3)
    _assert_uniform_microbatch_count(5)


def _uniform_count_worker(result_dir_str: str, mesh: Any) -> None:
    """Worker: equal per-rank counts pass; unequal counts raise ValueError."""
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh

        # Equal counts across ranks: must pass.
        _assert_uniform_microbatch_count(8)

        # Unequal counts (rank 0 -> 3, rank 1 -> 5): must raise on every rank.
        local_count = 3 if rank == 0 else 5
        try:
            _assert_uniform_microbatch_count(local_count)
        except ValueError:
            (result_dir / f"rank_{rank}").write_text("ok")
        else:
            (result_dir / f"rank_{rank}").write_text("FAIL:no-raise-on-unequal")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        (result_dir / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.integration
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
