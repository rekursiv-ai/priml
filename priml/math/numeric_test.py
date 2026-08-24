from __future__ import annotations

import math

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.math.numeric import (
    kahan_sum,
    log1mexp,
    log1psquare,
    log_arctan_exp,
    log_cosh,
    log_cumsum_exp,
    log_modulus,
    log_tan_exp,
    logerfc,
    logmeanexp,
    logsubexp,
    matrix_signum_via_newtonschulz,
    mesh_arange,
    power_transform,
    power_transform_inverse,
    safe_log,
    safe_pow,
    safe_rsqrt,
    safe_sqrt,
    safe_xlogy,
    sinh_arcsinh,
    sinh_arcsinh_inverse,
    smootherstep,
    smoothstep,
    smoothstep_inverse,
    soft_threshold,
    softcap,
    softmax_centered,
    softmax_centered_inverse,
    softplus_inverse,
    softsign,
    softsign_inverse,
    sqrt1pm1,
    ste_clamp,
    ste_round,
)


def test_log_arctan_exp_no_nan_inf():
    """Stable version produces no NaN/Inf where naive gives -inf."""

    def naive_fn(x: Tensor) -> Tensor:
        return torch.log(torch.arctan(torch.exp(x)))

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.linspace(-150, 150, 10_000, dtype=dtype)
        naive = naive_fn(x)
        stable = log_arctan_exp(x)
        naive_bad = naive.isnan() | naive.isinf()
        stable_bad = stable.isnan() | stable.isinf()
        assert naive_bad.any(), f"{dtype}: naive should have failures"
        assert not stable_bad.any(), (
            f"{dtype}: stable has {stable_bad.sum()} bad values"
        )


def test_log_arctan_exp_matches_reference():
    """Stable version matches float64 reference where naive also works."""
    x64 = torch.linspace(-10, 10, 1000, dtype=torch.float64)
    ref = torch.log(torch.arctan(torch.exp(x64)))
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x64.to(dtype)
        stable = log_arctan_exp(x)
        r = ref.to(dtype)
        torch.testing.assert_close(stable, r, rtol=0.02, atol=0.04)


def test_log_arctan_exp_asymptotics():
    """Check x -> -inf gives ~x, x -> +inf gives ~log(pi/2)."""
    x_neg = torch.tensor([-50.0, -100.0])
    assert torch.allclose(log_arctan_exp(x_neg), x_neg, atol=1e-6)
    x_pos = torch.tensor([50.0, 100.0])
    expected = torch.full_like(x_pos, math.log(math.pi / 2))
    assert torch.allclose(log_arctan_exp(x_pos), expected, atol=1e-6)


def test_log_tan_exp_no_nan_inf():
    """Stable version produces no NaN/Inf where naive gives -inf."""

    def naive_fn(x: Tensor) -> Tensor:
        return torch.log(torch.tan(torch.exp(x)))

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.linspace(-150, 0.4, 10_000, dtype=dtype)
        naive = naive_fn(x)
        stable = log_tan_exp(x)
        naive_bad = naive.isnan() | naive.isinf()
        stable_bad = stable.isnan() | stable.isinf()
        assert naive_bad.any(), f"{dtype}: naive should have failures"
        assert not stable_bad.any(), (
            f"{dtype}: stable has {stable_bad.sum()} bad values"
        )


def test_log_tan_exp_matches_reference():
    """Stable version matches float64 reference."""
    x64 = torch.linspace(-10, 0.4, 1000, dtype=torch.float64)
    ref = torch.log(torch.tan(torch.exp(x64)))
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x64.to(dtype)
        stable = log_tan_exp(x)
        r = ref.to(dtype)
        torch.testing.assert_close(stable, r, rtol=0.02, atol=0.04)


def test_log_tan_exp_asymptotics():
    """Check x -> -inf gives ~x."""
    x_neg = torch.tensor([-50.0, -100.0])
    assert torch.allclose(log_tan_exp(x_neg), x_neg, atol=1e-6)


def test_log1mexp():
    x = torch.tensor([-0.1, -0.7, -1.5, -3.0])
    expected = torch.log1p(-torch.exp(x))
    torch.testing.assert_close(log1mexp(x), expected, rtol=1e-5, atol=1e-5)


def test_log1mexp_branch_x_less_than_neg_log2():
    """Test the branch where x < -log(2)."""
    x = torch.tensor([-1.0, -2.0, -3.0])  # All less than -log(2) ≈ -0.693
    result = log1mexp(x)
    expected = torch.log1p(-torch.exp(x))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_log1mexp_branch_x_greater_than_neg_log2():
    """Test the branch where x >= -log(2)."""
    x = torch.tensor([-0.1, -0.3, -0.5])  # All greater than -log(2) ≈ -0.693
    result = log1mexp(x)
    expected = torch.log(-torch.expm1(x))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_log1mexp_with_numpy():
    """Test log1mexp with numpy input."""
    x = np.array([-0.5, -1.5])
    result = log1mexp(x)
    assert isinstance(result, Tensor)
    assert result.shape == (2,)


def test_logsubexp_without_return_sign():
    """Test logsubexp without return_sign."""
    x = torch.tensor([2.0, 3.0, 1.0])
    y = torch.tensor([1.0, 2.0, 0.5])
    result = logsubexp(x, y)
    assert isinstance(result, Tensor)
    # Verify: log(exp(x) - exp(y))
    expected = torch.log(torch.exp(x) - torch.exp(y))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_logsubexp_with_return_sign():
    """Test logsubexp with return_sign=True."""
    x = torch.tensor([2.0, 1.0])
    y = torch.tensor([1.0, 2.0])
    result_pair = logsubexp(x, y, return_sign=True)
    assert isinstance(result_pair, tuple)
    _, sign = result_pair
    # When x > y, sign should be +1; when x < y, sign should be -1
    expected_sign = torch.tensor([1, -1])
    torch.testing.assert_close(sign, expected_sign)


def test_logsubexp_equal_values():
    """Test logsubexp when x == y."""
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([1.0, 2.0])
    result = logsubexp(x, y)
    # log(exp(x) - exp(x)) = log(0) = -inf
    assert isinstance(result, Tensor)
    assert torch.all(torch.isinf(result))


def test_logsubexp_numpy_inputs():
    """Test logsubexp with numpy inputs."""
    x = np.array([2.0, 3.0])
    y = np.array([1.0, 2.0])
    result = logsubexp(x, y)
    assert isinstance(result, Tensor)


def test_softcap_basic():
    """Test basic softcap functionality."""
    x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    cap = 1.0
    result = softcap(x, cap)
    assert isinstance(result, Tensor)
    # Result should be bounded by cap
    assert torch.all(result <= cap)
    assert torch.all(result >= -cap)


def test_softcap_with_different_caps():
    """Test softcap with different cap values."""
    x = torch.tensor([10.0, -10.0, 0.0])
    for cap in [0.5, 1.0, 2.0, 5.0]:
        result = softcap(x, cap)
        # Check that values are bounded by cap
        assert isinstance(result, Tensor)
        assert torch.all(torch.abs(result) <= cap * 1.001)  # Small tolerance


def test_softcap_preserves_dtype():
    """Test that softcap preserves input dtype."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float16)
    result = softcap(x, 1.0)
    assert isinstance(result, Tensor)
    assert result.dtype == torch.float16


def test_softcap_with_numpy():
    """Test softcap with numpy input."""
    x = np.array([1.0, 2.0, 3.0])
    result = softcap(x, 1.5)
    assert isinstance(result, Tensor)


def test_softcap_zero():
    """Test softcap at zero."""
    x = torch.tensor([0.0])
    result = softcap(x, 1.0)
    torch.testing.assert_close(result, torch.tensor([0.0]), rtol=1e-5, atol=1e-5)


def test_softcap_does_not_truncate_integer_input():
    """An integer tensor must not round-trip through its own dtype.

    Capping is a real-valued operation: ``tanh(1/1) == 0.7616``. Casting that
    back to int64 floors every capped value to zero, so an integer input
    silently returns all zeros -- a bounded-output contract violated by the
    cast rather than by the maths.
    """
    result = softcap(torch.tensor([1, 2, 3]))

    assert result.dtype.is_floating_point
    assert torch.all(result != 0)


def test_softcap_preserves_float_dtypes_exactly():
    """Float inputs keep their dtype; only integers are promoted."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        assert softcap(torch.tensor([1.0, 2.0], dtype=dtype), 1.0).dtype == dtype


def test_ste_clamp():
    x = torch.tensor([-7.0, -2.5, 0.0, 1.5, 3.0, 5.5, 8.0])
    actual = ste_clamp(x, min=-2.0, max=5.0)
    expected = torch.clamp(x, min=-2.0, max=5.0)
    torch.testing.assert_close(actual, expected)


def test_ste_round():
    x = torch.tensor([-3.7, -1.2, 0.0, 0.6, 2.9, 4.1])
    for scale in [0.25, 3.0]:
        actual = ste_round(x, scale=scale)
        expected = torch.round(x * scale) / scale
        torch.testing.assert_close(actual, expected)


def test_safe_log():
    vals = torch.tensor([0.001, 0.1, 1.0, 5.0])
    torch.testing.assert_close(
        safe_log(vals),
        torch.log(vals),
        rtol=1e-5,
        atol=1e-5,
    )


def test_safe_sqrt():
    vals = torch.tensor([0.0, 0.01, 1.0, 9.0, 16.0])
    torch.testing.assert_close(
        safe_sqrt(vals),
        torch.sqrt(vals),
        rtol=1e-5,
        atol=1e-5,
    )


def test_safe_xlogy():
    expected = torch.tensor([0.0, float(np.log(2.0))])
    actual = safe_xlogy([0.0, 1.0], [0.0, 2.0])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_safe_rsqrt():
    vals = torch.tensor([0.04, 1.0, 4.0, 25.0])
    torch.testing.assert_close(
        safe_rsqrt(vals),
        torch.rsqrt(vals),
        rtol=1e-5,
        atol=1e-5,
    )


def test_safe_rsqrt_nonpositive():
    vals = torch.tensor([-1.0, 0.0, 0.25, -3.0])
    result = safe_rsqrt(vals)
    assert result[0] == 0.0  # negative -> 0 (undefined domain)
    assert result[1] == math.inf  # rsqrt(0) = 1/sqrt(0) = +inf
    assert result[3] == 0.0
    torch.testing.assert_close(result[2], torch.rsqrt(torch.tensor(0.25)))


def test_safe_rsqrt_gradient():
    x = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
    y = safe_rsqrt(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_safe_pow():
    base = torch.tensor([0.5, 2.0, 3.0])
    exp = torch.tensor([0.5, 0.5, 2.0])
    torch.testing.assert_close(
        safe_pow(base, exp),
        torch.pow(base, exp),
        rtol=1e-5,
        atol=1e-5,
    )


def test_safe_pow_nonpositive_base():
    base = torch.tensor([-1.0, 0.0, 4.0, -2.0])
    exp = torch.tensor(0.5)
    result = safe_pow(base, exp)
    assert result[0] == 0.0
    assert result[1] == 0.0  # 0 ** 0.5 = 0
    assert result[3] == 0.0
    torch.testing.assert_close(result[2], torch.tensor(2.0))


def test_safe_pow_zero_zero_is_one():
    """0 ** 0 == 1 (IEEE-754 / NumPy / torch.pow convention), not the safe-0."""
    assert safe_pow(0.0, 0.0).item() == 1.0
    # Vectorized: 0**0 = 1, 0**positive = 0, positive**0 = 1.
    base = torch.tensor([0.0, 0.0, 2.0, 0.0])
    exp = torch.tensor([0.0, 3.0, 0.0, 1.0])
    torch.testing.assert_close(
        safe_pow(base, exp),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    # Matches torch.pow on the positive/zero domain.
    assert safe_pow(0.0, 0.0).item() == torch.pow(torch.tensor(0.0), 0.0).item()


def test_safe_pow_gradient():
    base = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
    y = safe_pow(base, 0.5).sum()
    y.backward()
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()


def test_log_cumsum_exp():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = log_cumsum_exp(x)
    expected = torch.logcumsumexp(x, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_log_cumsum_exp_stability():
    """Should not overflow for large values."""
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    result = log_cumsum_exp(x)
    assert torch.isfinite(result).all()
    # First element should equal x[0].
    torch.testing.assert_close(result[0], x[0], rtol=1e-5, atol=1e-5)


def test_log_cumsum_exp_2d():
    x = torch.randn(3, 5)
    result = log_cumsum_exp(x, dim=-1)
    expected = torch.logcumsumexp(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_kahan_sum_accuracy():
    """Compensated sum is STRICTLY more accurate than naive for small terms
    swamped by a large running total.

    A big leading term followed by many tiny ones is the canonical case where
    naive fp32 drops the low-order bits of each small term once the running
    total grows; the compensation recovers them. Using ``<`` (not ``<=``) so
    the test actually proves the algorithm adds value -- on a benign sum where
    compensation is a no-op, ``<=`` would pass vacuously.
    """
    # 2**24 (fp32 ULP = 2) then 1000 copies of 1.0: each small term sits at
    # half an ULP, so naive accumulation drops it while compensation recovers
    # it. Far smaller N than a 1e7/1e-2 sum needs, and -- unlike that case,
    # which diverges only under the conftest-pinned MKL kernel (under default
    # threading both errors are 1.0) -- the swamping holds for any reduction
    # order, so the assertion is kernel-independent.
    x = torch.cat(
        [
            torch.tensor([2.0**24], dtype=torch.float32),
            torch.full((1000,), 1.0, dtype=torch.float32),
        ],
    )
    ref = torch.tensor(2.0**24 + 1000 * 1.0, dtype=torch.float32)
    kahan_err = (kahan_sum(x) - ref).abs()
    naive_err = (x.sum() - ref).abs()
    assert kahan_err < naive_err


def test_kahan_sum_at_least_as_accurate_general():
    """Compensated sum is never worse than naive on a benign many-term sum."""
    n = torch.arange(1, 1001, dtype=torch.float32)
    x = 1.0 / n
    ref = (1.0 / torch.arange(1, 1001, dtype=torch.float64)).sum().float()
    kahan_err = (kahan_sum(x) - ref).abs()
    naive_err = (x.sum() - ref).abs()
    assert kahan_err <= naive_err


def test_kahan_sum_neumaier_large_term_dominates():
    """Neumaier variant recovers terms larger than the running total.

    The classic ``[1, 1e100, 1, -1e100]`` (true sum 2) returns 0 under plain
    Kahan -- the compensation tracks the wrong operand's lost bits when a term
    dwarfs the running total -- but 2 under Kahan-Babuska-Neumaier. Pins the
    algorithm the docstring claims.
    """
    x = torch.tensor([1.0, 1e100, 1.0, -1e100], dtype=torch.float64)
    assert kahan_sum(x).item() == 2.0


def test_kahan_sum_empty_returns_zero():
    """Summing an empty tensor is 0, not an IndexError on x[0]."""
    torch.testing.assert_close(kahan_sum(torch.zeros(0)), torch.tensor(0.0))
    # Empty along a reduced dim collapses to a zero-filled result of that shape.
    result = kahan_sum(torch.zeros(3, 0), dim=1)
    torch.testing.assert_close(result, torch.zeros(3))


def test_kahan_sum_blocking_matches_a_strictly_sequential_recurrence() -> None:
    """Blocking must not change the answer, only the number of Python steps.

    The reduction runs ``sqrt(n)`` blocks in one vectorized step instead of one
    step per element (measured 2013ms -> 18ms at n=100_000). Each block's terms
    stay contiguous and in order, and the per-block compensations are folded
    back, so the result is the sequential one.
    """

    def sequential(x: Tensor) -> Tensor:
        total = torch.zeros((), dtype=x.dtype)
        compensation = torch.zeros((), dtype=x.dtype)
        for i in range(x.shape[0]):
            term = x[i]
            t = total + term
            compensation = compensation + (
                (total - t) + term
                if bool(total.abs() >= term.abs())
                else (term - t) + total
            )
            total = t
        return total + compensation

    # A length that is not a perfect square, so the final block is padded.
    torch.manual_seed(0)
    x = torch.randn(1_001, dtype=torch.float32)
    torch.testing.assert_close(kahan_sum(x), sequential(x), rtol=0, atol=0)
    # And on the swamping case the compensation exists for.
    hard = torch.cat(
        [
            torch.tensor([2.0**24], dtype=torch.float32),
            torch.full((1_000,), 1.0, dtype=torch.float32),
        ],
    )
    torch.testing.assert_close(kahan_sum(hard), sequential(hard), rtol=0, atol=0)


def test_kahan_sum_dim():
    x = torch.randn(4, 5)
    result = kahan_sum(x, dim=1)
    expected = x.sum(dim=1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_kahan_sum_keepdim():
    x = torch.randn(3, 4)
    result = kahan_sum(x, dim=0, keepdim=True)
    assert result.shape == (1, 4)


def test_kahan_sum_keepdim_all_dims_preserves_rank():
    """dim=None with keepdim=True must restore the full rank as size-1 dims."""
    result = kahan_sum(torch.ones(2, 3), keepdim=True)
    assert result.shape == (1, 1)
    torch.testing.assert_close(result, torch.full((1, 1), 6.0))


def test_smootherstep_boundaries():
    x = torch.tensor([0.0, 1.0])
    y = smootherstep(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 1.0]))


def test_smootherstep_midpoint():
    y = smootherstep(torch.tensor(0.5))
    torch.testing.assert_close(y, torch.tensor(0.5), rtol=1e-6, atol=1e-6)


def test_smootherstep_clamped():
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
    y = smootherstep(x)
    assert y[0] == 0.0
    assert y[-1] == 1.0


def test_smoothstep_boundaries():
    x = torch.tensor([0.0, 1.0])
    y = smoothstep(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 1.0]))


def test_smoothstep_midpoint():
    y = smoothstep(torch.tensor(0.5))
    torch.testing.assert_close(y, torch.tensor(0.5), rtol=1e-6, atol=1e-6)


def test_smoothstep_monotonic():
    x = torch.linspace(0, 1, 100)
    y = smoothstep(x)
    assert torch.all(y[1:] >= y[:-1])


def test_smoothstep_inverse_roundtrip():
    x = torch.linspace(0.01, 0.99, 20)
    y = smoothstep(x)
    x_back = smoothstep_inverse(y)
    torch.testing.assert_close(x_back, x, rtol=1e-4, atol=1e-4)


def test_smoothstep_inverse_boundaries():
    y = torch.tensor([0.0, 1.0])
    x = smoothstep_inverse(y)
    torch.testing.assert_close(x, torch.tensor([0.0, 1.0]), atol=1e-4, rtol=1e-4)


def test_smootherstep_monotonic():
    x = torch.linspace(0, 1, 100)
    y = smootherstep(x)
    assert torch.all(y[1:] >= y[:-1])


def test_soft_threshold_basic():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    result = soft_threshold(x, 2.0)
    expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_soft_threshold_zero_threshold():
    x = torch.tensor([-2.0, 0.0, 2.0])
    result = soft_threshold(x, 0.0)
    torch.testing.assert_close(result, x)


def test_sqrt1pm1_small():
    """For small x, should match x/2 to first order (Taylor expansion)."""
    x = torch.tensor([1e-8, 1e-10, -1e-8])
    result = sqrt1pm1(x)
    approx = x / 2  # First-order Taylor
    torch.testing.assert_close(result, approx, rtol=1e-4, atol=1e-12)


def test_sqrt1pm1_large():
    x = torch.tensor([3.0, 8.0, 99.0])
    result = sqrt1pm1(x)
    expected = torch.sqrt(1.0 + x) - 1.0
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_sqrt1pm1_zero():
    result = sqrt1pm1(torch.tensor(0.0))
    torch.testing.assert_close(result, torch.tensor(0.0), rtol=1e-7, atol=1e-7)


def test_sqrt1pm1_no_cancellation():
    """Direct computation loses precision for tiny x; sqrt1pm1 should not."""
    x = torch.tensor(1e-15, dtype=torch.float64)
    result = sqrt1pm1(x)
    # Direct: sqrt(1 + 1e-15) - 1 ≈ 0 due to cancellation in float64.
    direct = torch.sqrt(1.0 + x) - 1.0
    # Our result should be closer to the true value x/2.
    true_val = x / 2
    our_err = (result - true_val).abs()
    direct_err = (direct - true_val).abs()
    assert our_err <= direct_err + 1e-20


def test_softsign_roundtrip():
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = softsign(x)
    assert torch.all(y > -1)
    assert torch.all(y < 1)
    x_back = softsign_inverse(y)
    torch.testing.assert_close(x_back, x, rtol=1e-5, atol=1e-5)


def test_softsign_zero():
    torch.testing.assert_close(softsign(torch.tensor(0.0)), torch.tensor(0.0))


def test_softsign_inverse_boundary():
    """Values at ±1 should map to ±inf."""
    y = torch.tensor([0.999, -0.999])
    x = softsign_inverse(y)
    assert torch.all(torch.isfinite(x))
    assert x[0] > 0
    assert x[1] < 0


def test_power_transform_exp_limit():
    """power=0 should give exp(x)."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    result = power_transform(x, power=0)
    torch.testing.assert_close(result, torch.exp(x), rtol=1e-5, atol=1e-5)


def test_power_transform_identity():
    """power=1 should give 1 + x."""
    x = torch.tensor([0.0, 1.0, 2.0])
    result = power_transform(x, power=1.0)
    torch.testing.assert_close(result, 1.0 + x, rtol=1e-5, atol=1e-5)


def test_power_transform_roundtrip():
    x = torch.tensor([0.5, 1.0, 2.0])
    for c in [0.0, 0.5, 1.0, 2.0]:
        y = power_transform(x, power=c)
        x_back = power_transform_inverse(y, power=c)
        torch.testing.assert_close(x_back, x, rtol=1e-4, atol=1e-4)


def test_softmax_centered_simplex():
    x = torch.randn(3, 5)
    y = softmax_centered(x)
    assert y.shape == (3, 6)
    # Should sum to 1 along last dim.
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-5)
    # All positive.
    assert torch.all(y > 0)


def test_softmax_centered_roundtrip():
    x = torch.randn(2, 4)
    y = softmax_centered(x)
    x_back = softmax_centered_inverse(y)
    # Roundtrip should recover x up to a constant shift (softmax is shift-invariant).
    # So x_back - x should be approximately constant per row.
    diff = x_back - x
    spread = diff.max(dim=-1).values - diff.min(dim=-1).values
    torch.testing.assert_close(spread, torch.zeros(2), atol=1e-4, rtol=1e-4)


def test_sinh_arcsinh_identity():
    """skewness=0, tailweight=1 should be identity."""
    x = torch.tensor([-2.0, 0.0, 1.5, 3.0])
    y = sinh_arcsinh(x, skewness=0.0, tailweight=1.0)
    torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)


def test_sinh_arcsinh_roundtrip():
    x = torch.randn(10)
    for sk, tw in [(0.0, 1.0), (0.5, 1.5), (-1.0, 0.7)]:
        y = sinh_arcsinh(x, skewness=sk, tailweight=tw)
        x_back = sinh_arcsinh_inverse(y, skewness=sk, tailweight=tw)
        torch.testing.assert_close(x_back, x, rtol=1e-4, atol=1e-4)


def test_sinh_arcsinh_heavier_tails():
    """Tailweight > 1 should produce heavier tails (larger absolute values)."""
    x = torch.tensor([3.0, -3.0])
    light = sinh_arcsinh(x, tailweight=0.5).abs()
    heavy = sinh_arcsinh(x, tailweight=2.0).abs()
    assert torch.all(heavy > light)


def test_softplus_inverse_roundtrip() -> None:
    """softplus(softplus_inverse(x)) == x across small/mid/large branches."""
    x = torch.tensor([1e-6, 1e-2, 0.5, 1.0, 5.0, 30.0], dtype=torch.float64)
    recovered = torch.nn.functional.softplus(softplus_inverse(x))
    torch.testing.assert_close(recovered, x, rtol=1e-6, atol=1e-8)


def test_log1psquare_matches_naive_small() -> None:
    """In the non-overflow regime log1psquare == log(1 + x**2) exactly."""
    x = torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(log1psquare(x), torch.log1p(x * x))


def test_log1psquare_large_uses_2logabs() -> None:
    """For |x| past the overflow threshold the value is 2*log|x|."""
    x = torch.tensor([1e8, -1e10, 1e12], dtype=torch.float64)
    torch.testing.assert_close(log1psquare(x), 2 * x.abs().log())


def test_logerfc_matches_naive_negative() -> None:
    """For x < 0 logerfc == log(erfc(x)) directly."""
    x = torch.tensor([-3.0, -1.0, -0.1], dtype=torch.float64)
    torch.testing.assert_close(logerfc(x), torch.erfc(x).log())


def test_logerfc_positive_tail_stable() -> None:
    """For large x, erfc(x) underflows but logerfc stays finite and matches.

    Reference: log(erfc(x)) = log(erfcx(x)) - x**2 (the formulation used).
    """
    x = torch.tensor([1.0, 5.0, 20.0, 40.0], dtype=torch.float64)
    got = logerfc(x)
    expected = torch.special.erfcx(x).log() - x * x
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, expected)


def test_logerfc_positive_gradient_is_finite() -> None:
    """For x >= 0 the unselected log(erfc(x)) branch must not back-prop NaN.

    torch.where evaluates both branches; at large x, erfc(x) underflows to 0
    so log(erfc(x)) = -inf and its gradient is NaN. The double-where must keep
    that NaN out of the gradient of the selected erfcx branch.
    """
    x = torch.tensor([0.5, 5.0, 30.0], requires_grad=True)
    logerfc(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_log_cosh_matches_naive_midrange() -> None:
    """In the safe range log_cosh == log(cosh(x))."""
    x = torch.tensor([-2.0, -0.5, 0.3, 1.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(log_cosh(x), torch.cosh(x).log(), atol=1e-9, rtol=1e-9)


def test_log_cosh_is_even() -> None:
    """Cosh is even, so log_cosh(x) == log_cosh(-x)."""
    x = torch.tensor([0.2, 1.0, 7.0, 50.0], dtype=torch.float64)
    torch.testing.assert_close(log_cosh(x), log_cosh(-x))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_cosh_low_precision_accurate_and_finite(dtype: torch.dtype) -> None:
    """fp16/bf16 must stay accurate and NaN-free.

    Guards against a dtype-dependent crossover constant: a prior version used
    ``bound = 45 * tiny**(1/6)``, which in fp16 (large ``tiny``) selected a
    Taylor branch for |x| < ~8.9, returning values 2-4x wrong and NaN at x=5.
    """
    xs = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=dtype)
    got = log_cosh(xs).float()
    ref = torch.log(torch.cosh(xs.double())).float()
    assert torch.isfinite(got).all()
    # Tolerance ~ a few ulp of the dtype's epsilon.
    torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)
    grad_x = xs.clone().requires_grad_(True)
    log_cosh(grad_x).sum().backward()
    assert grad_x.grad is not None
    assert torch.isfinite(grad_x.grad).all()


def test_log_cosh_infinite_input_returns_positive_infinity() -> None:
    """cosh(+-inf) = inf, so log_cosh(+-inf) = +inf -- never NaN."""
    x = torch.tensor([float("inf"), float("-inf")])
    got = log_cosh(x)
    assert torch.equal(got, torch.tensor([float("inf"), float("inf")]))


def test_log_cosh_nan_input_propagates_nan() -> None:
    """NaN in must give NaN out (propagate, don't silently sanitize)."""
    got = log_cosh(torch.tensor([float("nan")]))
    assert torch.isnan(got).all()


def test_log_cosh_never_nan_for_finite_real_input() -> None:
    """log(cosh(x)) is finite for every finite real x; NaN there is a bug."""
    x = torch.linspace(-100.0, 100.0, steps=2001, dtype=torch.float64)
    assert torch.isfinite(log_cosh(x)).all()


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_log_cosh_gradient_never_nan(dtype: torch.dtype) -> None:
    """Gradient is tanh(x): finite everywhere (and +-1 at +-inf), never NaN."""
    x = torch.linspace(-50.0, 50.0, steps=1001, dtype=dtype, requires_grad=True)
    log_cosh(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(
        x.grad.float(), torch.tanh(x.detach().float()), atol=2e-2, rtol=2e-2
    )
    # +-inf inputs give finite +-1 gradients, not NaN.
    xinf = torch.tensor([float("inf"), float("-inf")], requires_grad=True)
    log_cosh(xinf).sum().backward()
    assert xinf.grad is not None
    assert torch.equal(xinf.grad, torch.tensor([1.0, -1.0]))


def test_log_cosh_large_is_finite() -> None:
    """Large |x| must not overflow; asymptote is |x| - log 2."""
    x = torch.tensor([100.0, 700.0], dtype=torch.float64)
    got = log_cosh(x)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, x - math.log(2), rtol=1e-9, atol=1e-9)


def test_log_cosh_zero_gradient_is_finite() -> None:
    """At x == 0 the small-|x| branch must not back-prop NaN."""
    x = torch.zeros(1, requires_grad=True)
    log_cosh(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_log_cosh_gradient_finite_for_large_x() -> None:
    """Gradient must be finite for |x| > sqrt(6).

    torch.where evaluates both branches for back-prop; the small-|x| branch
    contains log1p(-x*x/6), which is NaN once x*x > 6. That NaN must not leak
    into the gradient of the large-|x| branch (the classic where-NaN trap).
    The true gradient is tanh(x).
    """
    x = torch.tensor([2.5, 3.0, 5.0, 10.0], requires_grad=True)
    log_cosh(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, torch.tanh(x.detach()))


def test_mesh_arange_grid_coordinates() -> None:
    """2-D grid enumerates row-major (ij) coordinates."""
    grid = mesh_arange([2, 3])
    expected = torch.tensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
    )
    assert torch.equal(grid, expected)


def test_mesh_arange_start_step() -> None:
    """Start and step are honoured per dimension."""
    grid = mesh_arange([6, 4], start=[0, 1], step=[2, 1])
    expected = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [2, 1], [2, 2], [2, 3], [4, 1], [4, 2], [4, 3]],
    )
    assert torch.equal(grid, expected)


def test_matrix_signum_drives_singular_values_to_one() -> None:
    """The matrix sign function maps every singular value toward 1.

    This is the defining property of the orthogonal polar factor U V^T.
    Five bfloat16 Newton-Schulz steps reach it to ~0.35 absolute (the
    iteration's published convergence band), not to machine precision.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4)
    q = matrix_signum_via_newtonschulz(x).float()
    sv = torch.linalg.svdvals(q)
    torch.testing.assert_close(sv, torch.ones_like(sv), atol=0.35, rtol=0.0)


def test_matrix_signum_preserves_singular_vectors() -> None:
    """The result shares left/right singular vectors with the input.

    The Newton-Schulz iteration is a polynomial in ``X`` acting only on
    the singular values, so the output stays aligned with ``U V^T``.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 8, 4)
    q = matrix_signum_via_newtonschulz(x).float()
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    polar = u @ vh
    # Agreement is limited by bfloat16 + 5-step convergence, not alignment.
    torch.testing.assert_close(q, polar, atol=0.25, rtol=0.0)


def test_matrix_signum_preserves_input_dtype() -> None:
    """Output dtype must match the input, not leak the bfloat16 internals."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 8, 4, dtype=dtype)
        assert matrix_signum_via_newtonschulz(x).dtype == dtype


def test_power_transform_out_of_domain_is_finite() -> None:
    """1 + x*power <= 0 is outside the domain; result must not be NaN.

    safe_pow maps the non-positive base to 0, keeping value and gradient
    finite where torch.pow would yield NaN or a spurious magnitude.
    """
    x = torch.tensor([-3.0, -2.5], requires_grad=True)
    y = power_transform(x, power=0.5)  # 1 + x*0.5 = -0.5, -0.25 (both <= 0)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_log_modulus_values() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    expected = torch.tensor([-math.log(2.0), 0.0, math.log(2.0)], dtype=torch.float64)
    torch.testing.assert_close(log_modulus(x), expected)


def test_log_modulus_is_odd() -> None:
    x = torch.tensor([-1e6, -1.0, 0.0, 1.0, 1e6], dtype=torch.float64)
    torch.testing.assert_close(log_modulus(x), -log_modulus(-x))


def test_log_modulus_gradients() -> None:
    """d/dx = 1/(1+|x|) off the origin; sign(0) == 0 kills it at the kink."""
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    log_modulus(x).sum().backward()
    assert x.grad is not None
    expected = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64)
    torch.testing.assert_close(x.grad, expected)


def test_log_modulus_gradient_is_finite_in_the_tails() -> None:
    x = torch.tensor([-1e30, 1e30], dtype=torch.float32)
    _ = x.requires_grad_()
    log_modulus(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), x.grad


def test_log_modulus_preserves_dtype() -> None:
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        assert log_modulus(torch.zeros(3, dtype=dtype)).dtype == dtype


def test_logmeanexp_survives_an_empty_reduction() -> None:
    """An empty reduced axis must not raise: ``math.log(0)`` is a domain error.

    The identical bug in ``logmeanexp_all_to_all`` was fixed and pinned by
    ``test_logsumexp_all_to_all_empty_reduction``; this one was left.
    """
    result = logmeanexp(torch.zeros(3, 0), dim=-1)
    assert result.shape == (3,)
    assert bool(torch.isinf(result).all())


def test_out_of_domain_parameters_yield_nan_rather_than_a_device_sync() -> None:
    """These kernels do not inspect their arguments, and must not start.

    Checking ``cap`` or ``tailweight`` means ``bool(t.all())`` on a possibly
    tensor-valued argument, which synchronizes the device -- measured at 0.8x
    the cost of the whole kernel on a 1024x1024 CUDA tensor. The out-of-domain
    result is NaN or a collapsed map, the ordinary float contract, and the
    config layer above validates its own scalar field where the check is free
    (``softcap.py:79``).
    """
    x = torch.tensor([1.0, -2.0])
    assert bool(softcap(x, float("inf")).isnan().all())
    # A negative threshold expands magnitudes rather than shrinking them.
    assert float(soft_threshold(torch.tensor([1.0]), -2.0)) == 3.0
    # tailweight=0 collapses the map, so it is no longer invertible.
    assert float(sinh_arcsinh(torch.tensor([2.0]), 0.0, 0.0)) == 0.0


def test_documented_domains_propagate_nan_rather_than_syncing_to_check() -> None:
    """Off-domain input yields NaN; guarding it would cost a device sync.

    Both functions are elementwise and ``log1mexp`` is called from
    ``logsubexp``, so a ``bool((x < c).all())`` precondition measured 0.8x the
    cost of the whole kernel on a 1024x1024 CUDA tensor. NaN out for
    off-domain in is the ordinary float contract; the guard is not worth
    doubling the function.
    """
    assert bool(log_tan_exp(torch.tensor([1.0])).isnan())
    assert bool(log1mexp(torch.tensor([0.5])).isnan())
    # The documented domain is unaffected.
    assert torch.isfinite(log_tan_exp(torch.tensor([0.4]))).all()
    assert torch.isfinite(log1mexp(torch.tensor([-0.5]))).all()


def test_safe_helpers_propagate_nan_rather_than_inventing_a_value() -> None:
    """NaN is not "non-positive", so it must not become a plausible number.

    ``safe_log(nan)`` returned ``-inf`` and ``safe_sqrt(nan)`` returned ``0``:
    a NaN that entered upstream vanished into a finite-looking value instead of
    surfacing where it was introduced.
    """
    nan = torch.tensor([float("nan")])
    assert bool(safe_log(nan).isnan())
    assert bool(safe_sqrt(nan).isnan())
    assert bool(safe_rsqrt(nan).isnan())
    # The negative domain keeps its documented fallback.
    assert float(safe_sqrt(torch.tensor([-1.0]))) == 0.0
    assert float(safe_log(torch.tensor([-1.0]))) == float("-inf")


def test_safe_pow_propagates_nan_like_its_siblings() -> None:
    """NaN is not "non-positive", so it must not become the safe-zero.

    ``base > 0`` is False for NaN, which sent it down the fallback and
    returned 0.0 -- the same laundering
    ``test_safe_helpers_propagate_nan_rather_than_inventing_a_value``
    already pins for ``safe_log``/``safe_sqrt``/``safe_rsqrt``.
    """
    nan = torch.tensor([float("nan")])
    assert bool(safe_pow(nan, 0.5).isnan())
    assert bool(safe_pow(torch.tensor([2.0]), nan).isnan())
    # The documented non-positive fallback is unaffected.
    assert float(safe_pow(torch.tensor([-1.0]), 0.5)) == 0.0
    assert float(safe_pow(torch.tensor([0.0]), 0.0)) == 1.0


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16]
)
def test_smoothstep_inverse_is_finite_at_the_boundaries(dtype: torch.dtype) -> None:
    """The Newton guard must scale with the dtype, not be a fixed literal.

    ``df.clamp(min=1e-12)`` is below float16's smallest normal, so it rounded
    to exactly 0.0 and the division it exists to prevent happened anyway --
    NaN at y=0 and y=1, both inside the documented [0, 1] domain.
    """
    out = smoothstep_inverse(torch.tensor([0.0, 0.5, 1.0], dtype=dtype))
    assert bool(torch.isfinite(out).all())


def test_safe_sqrt_has_a_finite_gradient_at_zero() -> None:
    """The double-where must guard the BACKWARD pass, not just the forward.

    ``safe_sqrt`` clamped its operand to 0 and then took ``sqrt`` of it, whose
    derivative at 0 is infinite; ``where`` multiplies that by zero and yields
    NaN. The sibling ``safe_rsqrt`` moves the operand off the boundary instead.
    """
    x = torch.zeros(3, requires_grad=True)
    safe_sqrt(x).sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_ste_clamp_accepts_untensored_bounds() -> None:
    """Bounds arrive as plain floats, so they must convert like every sibling.

    ``ste_round`` and the rest run their arguments through
    ``convert_to_tensor``; ``ste_clamp`` passed ``min``/``max`` straight into
    ``torch.clamp``, which is what the four stacked type suppressions were
    holding up.
    """
    x = torch.tensor([-3.0, 0.5, 4.0], requires_grad=True)
    out = ste_clamp(x, -1.0, 1.0)
    torch.testing.assert_close(out, torch.tensor([-1.0, 0.5, 1.0]))
    out.sum().backward()
    # Straight-through: gradient is identity even where the forward clamped.
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones(3))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
