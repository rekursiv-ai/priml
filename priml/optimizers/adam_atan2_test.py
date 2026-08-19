"""Tests for AdamATan2 parity with the reference package."""

from __future__ import annotations

from torch import Tensor
from torch.optim.optimizer import Optimizer

import pytest
import torch

from priml.optimizers.adam_atan2 import AdamATan2


def test_adam_atan2_matches_reference_bias_corrections() -> None:
    param = torch.tensor([0.5, -0.25, 0.125], dtype=torch.float64)
    expected = param.clone()
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    opt = AdamATan2([param], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    grads = [
        torch.tensor([0.01, -0.02, 0.04], dtype=torch.float64),
        torch.tensor([0.03, -0.01, -0.02], dtype=torch.float64),
        torch.tensor([-0.02, 0.05, 0.01], dtype=torch.float64),
    ]
    for step, grad in enumerate(grads, start=1):
        param.grad = grad.clone()
        expected, exp_avg, exp_avg_sq = _reference_step(
            expected,
            grad,
            exp_avg,
            exp_avg_sq,
            step=step,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
        )
        opt.step()
        torch.testing.assert_close(param, expected, rtol=0, atol=1e-15)

    state = opt.state[param]
    # ``step`` is intentionally a Python int (not a 0-dim Tensor) to avoid
    # per-step GPU<->CPU syncs; see adam_atan2.py for the rationale.
    assert isinstance(state["step"], int)
    assert state["step"] == len(grads)
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


@pytest.mark.gpu_torch_cuda
def test_adam_atan2_matches_external_package_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = pytest.importorskip("adam_atan2")
    if not torch.cuda.is_available():
        pytest.skip("external adam-atan2 is a CUDA package")
    # torch 2.11 renamed the capture guard to `_accelerator_...` and kept no
    # deprecated alias, so the reference package's `step()` raises before it
    # computes anything. 0.0.3 is the newest release on PyPI, so there is no
    # version to upgrade to; restore the name it calls rather than lose the
    # exact-parity comparison. The guard is a precondition check, not numerics.
    monkeypatch.setattr(
        Optimizer,
        "_cuda_graph_capture_health_check",
        Optimizer._accelerator_graph_capture_health_check,
        raising=False,
    )
    ref_param = torch.tensor([0.5, -0.25, 0.125], device="cuda")
    test_param = ref_param.clone()
    ref_opt = external.AdamATan2(
        [ref_param],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    test_opt = AdamATan2(
        [test_param],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    for grad in (
        torch.tensor([0.01, -0.02, 0.04], device="cuda"),
        torch.tensor([0.03, -0.01, -0.02], device="cuda"),
        torch.tensor([-0.02, 0.05, 0.01], device="cuda"),
    ):
        ref_param.grad = grad.clone()
        test_param.grad = grad.clone()
        ref_opt.step()
        test_opt.step()
    torch.testing.assert_close(test_param, ref_param, rtol=0, atol=0)


def _reference_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    param = param.clone()
    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()

    param.mul_(1.0 - lr * weight_decay)
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    step_size = lr / (1.0 - beta1**step)
    bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
    denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
    param.add_(torch.atan2(exp_avg, denom), alpha=-step_size)
    return param, exp_avg, exp_avg_sq
