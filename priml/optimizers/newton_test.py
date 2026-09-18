"""Tests for Newton optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.metrics.binary_accuracy import BinaryAccuracy
from priml.optimizers.newton import Newton
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from pathlib import Path

    from priml.loss.custom_types import LossOutput


def _binary_cross_entropy_with_logits(
    output: Tensor,
    *,
    y: Tensor,
    **_kwargs: object,
) -> LossOutput:
    """Call binary_cross_entropy_with_logits with y extracted from kwargs."""
    return {
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            output,
            y,
            reduction="none",
        ),
    }


class _LinearModel(nn.Module):
    """Simple logistic regression model for testing.

    Owns a ``Linear`` rather than subclassing it: this model takes ``**kwargs``,
    which ``Linear.forward`` does not declare.
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


def test_newton_logistic_regression():
    """Newton converges on toy logistic regression via the TrainStep path.

    The train lib hosts the closure-based Newton optimizer with no hand-rolled
    loop: ``train_step`` builds the loss-recomputing closure and forwards it
    through ``Learnable.step`` to ``optimizer.step(closure)``. This is the
    forcing function proving the optimizer contract is general, not overfit to
    first-order ``step()``-without-closure optimizers.
    """
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2

    # Generate linearly separable data.
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).float()

    # Create trainable with Newton optimizer.
    config = TrainStep.Config()
    config.model = _LinearModel.Config(in_features=2, out_features=1)
    config.optimizer = Newton.Config()
    config.loss = PartialConfig(_binary_cross_entropy_with_logits)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None

    trainable = config.make()

    initial_loss = trainable.train_step(x=X, y=y)["loss"].mean().item()
    final_loss = initial_loss
    for _ in range(9):
        final_loss = trainable.train_step(x=X, y=y)["loss"].mean().item()

    assert final_loss < initial_loss * 0.01, (
        f"Newton via train_step should converge: {initial_loss} -> {final_loss}"
    )

    # Check accuracy on training data (should be near perfect)
    metric = BinaryAccuracy.Config().make()
    model = trainable.model
    assert isinstance(model, _LinearModel)
    with torch.no_grad():
        metric.update(model(X), label=y)

    metrics = metric.compute()
    accuracy = metrics["accuracy"]
    assert accuracy > 0.98, f"Accuracy should be > 98%, got {accuracy}"


def test_step_without_a_closure_is_a_caller_error() -> None:
    optimizer = Newton([nn.Parameter(torch.zeros(2))])
    with pytest.raises(ValueError, match="requires a loss-recomputing closure"):
        optimizer.step()


def test_step_rejects_a_closure_returning_a_bare_float() -> None:
    optimizer = Newton([nn.Parameter(torch.zeros(2))])
    with pytest.raises(TypeError, match="got float"):
        optimizer.step(lambda: 1.0)


def test_singular_hessian_falls_back_to_gradient_descent() -> None:
    """``(p0 + p1)^2`` has the rank-1 Hessian ``[[2, 2], [2, 2]]``.

    With no damping the solve is singular, so the step is plain descent.
    """
    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = Newton([param], lr=0.5, damping=0.0)

    optimizer.step(lambda: param.sum() ** 2)

    # ``grad = 2 (p0 + p1) [1, 1] = [-2, -2]``; ``delta = -grad``; ``p += lr delta``.
    torch.testing.assert_close(
        param.detach(),
        torch.tensor([2.0, -1.0], dtype=torch.float64),
    )


def test_step_returns_the_loss_it_differentiated() -> None:
    param = nn.Parameter(torch.tensor([2.0]))

    def closure() -> Tensor:
        return (param**2).sum()

    loss = Newton([param], lr=1.0).step(closure)
    assert isinstance(loss, Tensor)
    assert loss.item() == pytest.approx(4.0)


def test_step_with_no_parameter_groups_raises() -> None:
    optimizer = Newton([nn.Parameter(torch.zeros(1))])
    optimizer.param_groups.clear()
    with pytest.raises(RuntimeError, match="at least one parameter group"):
        optimizer.step(lambda: torch.zeros(()))


def test_newton_rejects_dtensor_params(tmp_path: Path) -> None:
    """#318: Newton must refuse sharded (DTensor) params, not silently corrupt.

    The exact-Hessian step flattens grads globally; a DTensor would contribute
    only its local shard, building a wrong Hessian. Newton is small-model-only
    and cannot be sharded, so it must fail loudly. A single-rank gloo group is
    enough to construct a DTensor; the guard fires before any collective.
    """
    from torch.distributed.device_mesh import (  # noqa: PLC0415 -- defer costly distributed import to its sole test.
        init_device_mesh,
    )
    from torch.distributed.tensor import (  # noqa: PLC0415 -- defer costly distributed import to its sole test.
        DTensor,
        Shard,
        distribute_tensor,
    )

    import torch.distributed as dist  # noqa: PLC0415 -- defer costly distributed import to its sole test.

    # A file rendezvous is collision-free across hosts and xdist workers; this
    # single-rank guard test does not need a TCP listener.
    rendezvous = (tmp_path / "gloo-rendezvous").resolve()
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        optimizer = Newton([param], lr=1.0, damping=0.0)

        def closure() -> Tensor:
            assert isinstance(param.data, DTensor)
            return param.data.to_local().pow(2).sum()

        with pytest.raises(
            NotImplementedError,
            match=r"sharded \(DTensor\) parameters",
        ):
            optimizer.step(closure)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
