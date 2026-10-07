from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import functools
import math

from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

import pytest
import torch

from priml.math.stats import (
    SlidingWindow,
    _householder_qr,
    cov,
    ema_update,
    entropy_logits,
    entropy_logits_mean_all_to_all,
    entropy_probs,
    holm,
    jsd,
    pca,
    pca_eigh,
    pca_power,
    pca_svd,
    quantile_normalize,
    total_variation,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def _power_input() -> Tensor:
    return torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -2.0],
            [-1.0, 2.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [1.0, -2.0],
            [-1.0, 2.0],
            [-1.0, -2.0],
        ],
    )


def test_cov_matches_torch():
    torch.manual_seed(7)
    x = torch.randn(5, 8)
    # Compare with manual computation: X^T X / N after centering.
    xc = x - x.mean(dim=0, keepdim=True)
    expected = (xc.T @ xc) / x.shape[0]
    actual = cov(x, rowvar=False, bias=True)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("rowvar", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_cov_shape(rowvar: bool, bias: bool):
    x = torch.randn(3, 6, 4)
    result = cov(x, rowvar=rowvar, bias=bias)
    if rowvar:
        assert result.shape == (3, 6, 6)
    else:
        assert result.shape == (3, 4, 4)


def test_cov_cross():
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 5, 3)
    result = cov(x, y=y, rowvar=False)
    assert result.shape == (2, 3, 3)


def test_cov_unbiased():
    x = torch.randn(10, 4)
    biased = cov(x, bias=True)
    unbiased = cov(x, bias=False)
    # Unbiased should be N/(N-1) times biased.
    torch.testing.assert_close(
        unbiased,
        biased * x.shape[0] / (x.shape[0] - 1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_cov_two_observations_use_unbiased_denominator_one():
    x = torch.tensor([[0.0, 2.0, 4.0], [2.0, 6.0, 8.0]])
    biased = cov(x, bias=True)
    unbiased = cov(x, bias=False)
    torch.testing.assert_close(unbiased, biased * 2)


def test_cov_1d_matches_numpy():
    """A 1-D input is a single variable; cov returns its scalar variance."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    biased = cov(x, bias=True)
    assert biased.ndim == 0
    torch.testing.assert_close(biased, torch.tensor(1.25))
    unbiased = cov(x, bias=False)
    torch.testing.assert_close(unbiased, torch.tensor(5.0 / 3.0))


def test_cov_1d_cross_is_the_scalar_cross_covariance():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    result = cov(x, y=y, bias=True)
    assert result.ndim == 0
    torch.testing.assert_close(result, torch.tensor(2.5))


def test_cov_1d_unbiased_singleton_and_two_observations():
    one = cov(torch.tensor([2.0]), bias=False)
    two = cov(torch.tensor([1.0, 3.0]), bias=False)
    assert torch.isnan(one)
    torch.testing.assert_close(two, torch.tensor(2.0))


def test_jsd_is_zero_for_identical_members_and_log2_for_disjoint_ones():
    torch.manual_seed(2)
    base = torch.log_softmax(torch.randn(3, 5), dim=-1)
    identical = base[:1].expand(3, 5)
    torch.testing.assert_close(jsd(identical), torch.tensor(0.0), atol=1e-6, rtol=0)

    disjoint = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).log()
    torch.testing.assert_close(jsd(disjoint), torch.tensor(math.log(2.0)))


def test_jsd_with_negative_infinity_matches_exact_safe_log_entropy():
    logits = torch.tensor(
        [
            [-0.0809429288, 0.0512638986, -0.8687565327, -math.inf],
            [-1.2717219591, 0.8816379309, -0.6639689207, -math.inf],
            [0.1610462517, 1.1758316755, -0.3615134060, -math.inf],
        ],
        dtype=torch.float32,
    )
    result = jsd(torch.log_softmax(logits, dim=-1))
    assert torch.isfinite(result)
    assert torch.equal(result, torch.tensor(0.04872685670852661))


def test_jsd_squeezes_the_ensemble_and_event_dims():
    logp = torch.log_softmax(torch.randn(4, 3, 6), dim=-1)
    result = jsd(logp, ensemble_dim=1, event_dim=(-1,))
    assert result.shape == (4,)
    assert (result >= -1e-6).all()


def test_entropy_self_consistency():
    """entropy_logits and entropy_probs should agree."""
    torch.manual_seed(11)
    logits = torch.randn(4, 5, 8)
    h_logits = entropy_logits(logits)
    h_probs = entropy_probs(torch.softmax(logits, dim=-1))
    torch.testing.assert_close(h_logits, h_probs, atol=1e-5, rtol=1e-5)


def test_entropy_cross():
    logits_p = torch.randn(3, 4, 6)
    logits_q = torch.randn(3, 4, 6)
    h_logits = entropy_logits(logits_p, logits_q)
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    h_probs = entropy_probs(p, q)
    torch.testing.assert_close(h_logits, h_probs, atol=1e-5, rtol=1e-5)


def test_entropy_keepdim():
    x = torch.randn(3, 5, 7)
    h = entropy_logits(x, keepdim=True)
    assert h.shape == (3, 5, 1)


def test_entropy_probs_zeros():
    """Zero probabilities should not produce NaN."""
    p = torch.tensor([[0.0, 1.0, 0.0], [0.3, 0.7, 0.0]])
    h = entropy_probs(p)
    assert torch.isfinite(h).all()


def test_entropy_uniform_is_log_k():
    """Entropy of uniform distribution over k classes is log(k)."""
    k = 10
    logits = torch.zeros(k)
    h = entropy_logits(logits, dim=0)
    torch.testing.assert_close(
        h,
        torch.tensor(float(torch.tensor(k).log())),
        atol=1e-5,
        rtol=1e-5,
    )


def test_entropy_logits_mean_all_to_all_shape():
    x = torch.randn(3, 4, 6)
    h = entropy_logits_mean_all_to_all(x, dim=-1, dim_mean=0)
    assert h.shape == (4,)


def test_entropy_logits_mean_all_to_all_cross():
    x = torch.randn(3, 4, 6)
    y = torch.randn(3, 4, 6)
    h = entropy_logits_mean_all_to_all(x, y, dim=-1, dim_mean=0)
    assert h.shape == (4,)


def test_entropy_logits_mean_all_to_all_infer_dim_mean():
    x = torch.randn(2, 3, 5, 4)
    h = entropy_logits_mean_all_to_all(x, dim=-1, dim_mean=None)
    assert h.shape == torch.Size([])


def test_pca_shapes():
    torch.manual_seed(3)
    x = torch.randn(100, 5)
    eigenvalues, eigenvectors = pca(x)
    assert eigenvalues.shape == (5,)
    assert eigenvectors.shape == (5, 5)


def test_pca_eigenvalues_ascending():
    torch.manual_seed(4)
    x = torch.randn(100, 8)
    eigenvalues, _ = pca(x)
    assert (eigenvalues[1:] >= eigenvalues[:-1]).all()


def test_pca_whiten_unit_variance():
    torch.manual_seed(5)
    x = torch.randn(200, 5)
    mix = torch.randn(5, 6)[:, :5]
    x = x @ mix  # Add correlation.
    _, eigenvectors = pca(x, whiten=True, eps=1e-5)
    projected = (x - x.mean(0)) @ eigenvectors
    # Each component should have approximately unit variance.
    torch.testing.assert_close(projected.var(0), torch.ones(5), atol=0.15, rtol=0.15)


def test_pca_whitening_defaults_to_unregularized_unit_energy():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ],
    )
    eigenvalues, eigenvectors = pca(x, whiten=True)
    component_energy = eigenvalues * eigenvectors.square().sum(dim=0)
    torch.testing.assert_close(component_energy, torch.ones_like(eigenvalues))


def test_pca_power_matches_eigh():
    """Power iteration and eigh paths should produce equivalent results."""
    torch.manual_seed(7)
    x = torch.randn(32, 4)
    x_centered = (x - x.mean(0)).float()
    vals_eigh, vecs_eigh = pca_eigh(x_centered)
    vals_power, vecs_power = pca_power(x_centered, num_iters=100)
    torch.testing.assert_close(vals_power, vals_eigh, atol=1e-3, rtol=1e-3)
    # Eigenvectors may differ by sign; compare absolute values.
    torch.testing.assert_close(vecs_power.abs(), vecs_eigh.abs(), atol=1e-3, rtol=1e-3)


def test_pca_power_tol_early_exit_matches_full():
    """The ``tol`` early-exit still recovers the eigenvalues to tolerance."""
    torch.manual_seed(7)
    x = torch.randn(32, 4)
    x_centered = (x - x.mean(0)).float()
    vals_full, _ = pca_power(x_centered, num_iters=50, tol=0.0)
    vals_tol, _ = pca_power(x_centered, num_iters=50, tol=1e-6)
    torch.testing.assert_close(vals_tol, vals_full, atol=1e-3, rtol=1e-3)


def test_pca_power_stops_iterating_once_within_tol():
    """A loose ``tol`` exits after the first sweep; the result is that sweep's."""
    x = torch.randn(32, 4)
    x_centered = (x - x.mean(0)).float()
    torch.manual_seed(7)
    early, _ = pca_power(x_centered, num_iters=100, tol=1e3)
    torch.manual_seed(7)
    one_sweep, _ = pca_power(x_centered, num_iters=1, tol=0.0)
    torch.manual_seed(7)
    full, _ = pca_power(x_centered, num_iters=100, tol=0.0)
    assert torch.equal(early, one_sweep)
    assert not torch.equal(early, full)


def test_pca_power_default_is_two_hundred_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qr = Mock(wraps=_householder_qr)
    monkeypatch.setattr("priml.math.stats._householder_qr", qr)
    pca_power(torch.eye(2, 3))
    assert qr.call_count == 201


def test_pca_power_zero_tolerance_runs_every_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qr = Mock(wraps=_householder_qr)
    monkeypatch.setattr("priml.math.stats._householder_qr", qr)
    pca_power(_power_input(), num_iters=4, tol=0.0)
    assert qr.call_count == 5


def test_pca_power_zero_tol_uses_only_endpoint_rayleigh_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rayleigh(sigma: Tensor, basis: Tensor) -> Tensor:
        return (basis * (sigma @ basis)).sum(dim=0)

    estimates = Mock(side_effect=rayleigh)
    monkeypatch.setattr(
        "priml.math.stats._rayleigh",
        estimates,
        raising=False,
    )
    pca_power(_power_input(), num_iters=4, tol=0.0)
    assert estimates.call_count == 2


def test_pca_power_convergence_uses_each_component_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bases = (
        torch.tensor([[1.0, 0.2], [0.4, 0.9]]),
        torch.tensor([[1.0, 0.3], [0.1, 1.0]]),
    )
    calls = 0

    def qr(matrix: Tensor) -> tuple[Tensor, Tensor]:
        nonlocal calls
        basis = bases[min(calls, len(bases) - 1)]
        calls += 1
        return basis, torch.zeros_like(matrix)

    monkeypatch.setattr("priml.math.stats._householder_qr", qr)
    # A covariance and its basis are square by definition.
    pca_power(_power_input(), num_iters=2, tol=0.9)
    assert calls == 2


def test_pca_power_convergence_keeps_column_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bases = (
        torch.tensor([[1.0, 0.2], [0.4, 0.9]]),
        torch.tensor([[1.0, 0.3], [0.1, 1.0]]),
    )
    calls = 0

    def qr(matrix: Tensor) -> tuple[Tensor, Tensor]:
        nonlocal calls
        basis = bases[min(calls, len(bases) - 1)]
        calls += 1
        return basis, torch.zeros_like(matrix)

    monkeypatch.setattr("priml.math.stats._householder_qr", qr)
    pca_power(_power_input(), num_iters=2, tol=0.5)
    assert calls == 3


def test_pca_power_convergence_uses_a_strict_tolerance_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bases = (
        torch.eye(2),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    calls = 0

    def qr(matrix: Tensor) -> tuple[Tensor, Tensor]:
        nonlocal calls
        basis = bases[min(calls, len(bases) - 1)]
        calls += 1
        return basis, torch.zeros_like(matrix)

    monkeypatch.setattr("priml.math.stats._householder_qr", qr)
    pca_power(_power_input(), num_iters=2, tol=4.0)
    assert calls == 3


def test_householder_qr_leaves_a_zero_column_unreflected():
    """An all-zero column has no reflection to apply; Q stays identity there."""
    mat = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    q, r = _householder_qr(mat)
    assert torch.equal(q, torch.eye(2))
    assert torch.equal(r, mat)


def test_pca_accepts_injected_decompose():
    """A partial binds the power iteration's knobs without touching ``pca``."""
    torch.manual_seed(7)
    x = torch.randn(100, 8)
    vals, vecs = pca(
        x,
        decompose=functools.partial(pca_power, num_iters=50, tol=1e-6),
    )
    assert vals.shape == (8,)
    assert vecs.shape == (8, 8)


def test_pca_svd_matches_eigh():
    """The SVD decomposer agrees with eigh up to eigenvector sign."""
    torch.manual_seed(8)
    x = torch.randn(100, 6)
    vals_eigh, vecs_eigh = pca(x, decompose=pca_eigh)
    vals_svd, vecs_svd = pca(x, decompose=pca_svd)
    torch.testing.assert_close(vals_svd, vals_eigh, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(vecs_svd.abs(), vecs_eigh.abs(), atol=1e-4, rtol=1e-4)


def test_pca_svd_keeps_thin_shapes_when_observations_are_fewer_than_features():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    eigenvalues, eigenvectors = pca_svd(x)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (3, 2)
    covariance = x.T @ x / len(x)
    torch.testing.assert_close(
        eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T,
        covariance,
    )


def test_pca_svd_uses_thin_factorization_for_tall_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svd = Mock(wraps=torch.linalg.svd)
    monkeypatch.setattr(torch.linalg, "svd", svd)
    pca_svd(torch.randn(20, 3))
    assert svd.call_args.kwargs["full_matrices"] is False


def test_pca_svd_device_guard_message() -> None:
    with FakeTensorMode():
        mps_input = torch.empty((2, 3), device="mps")
        with pytest.raises(RuntimeError) as error:
            pca_svd(mps_input)
    assert str(error.value) == (
        "pca_svd is not supported on MPS; use pca_power instead."
    )


def test_pca_third_party_decompose_needs_no_library_change():
    """An arbitrary caller-supplied decomposer is honored verbatim."""

    def scaled_eigh(x_centered: Tensor) -> tuple[Tensor, Tensor]:
        eigenvalues, eigenvectors = pca_eigh(x_centered)
        return eigenvalues * 2, eigenvectors

    torch.manual_seed(9)
    x = torch.randn(50, 4)
    baseline, _ = pca(x)
    scaled, _ = pca(x, decompose=scaled_eigh)
    torch.testing.assert_close(scaled, baseline * 2)


def test_pca_zca_whitens():
    """ZCA via pca: V_raw @ diag(1/sqrt(lam)) @ V_raw^T should whiten."""
    torch.manual_seed(6)
    x = torch.randn(200, 4)
    mix = torch.randn(4, 5)[:, :4]
    x = x @ mix
    eigenvalues, V_raw = pca(x)
    # ZCA whitening matrix: V @ diag(1/sqrt(λ)) @ V^T.
    zca_matrix = V_raw * eigenvalues.rsqrt().unsqueeze(0) @ V_raw.T
    whitened = (x - x.mean(0)) @ zca_matrix
    c = cov(whitened, rowvar=False)
    torch.testing.assert_close(c, torch.eye(4), atol=0.15, rtol=0.15)


def test_quantile_normalize():
    x = torch.randn(4, 5, 6)
    result = quantile_normalize(x, q=0.01)
    assert result.shape == x.shape
    # Bulk of values should be in [0, 1].
    in_range = ((result >= -0.5) & (result <= 1.5)).float().mean()
    assert in_range > 0.9


def test_quantile_normalize_constant_input_is_finite():
    """Constant input collapses the range; map it to 0 instead of nan."""
    result = quantile_normalize(torch.full((10,), 5.0))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros(10))


def test_ema_update():
    val = ema_update(10.0, 20.0, alpha=0.5)
    assert val == pytest.approx(15.0)
    val2 = ema_update(10.0, 20.0, alpha=0.0)
    assert val2 == pytest.approx(10.0)
    val3 = ema_update(10.0, 20.0, alpha=1.0)
    assert val3 == pytest.approx(20.0)


def test_sliding_window():
    w = SlidingWindow(window_sec=10.0, pseudocount=0.0, pseudotime=0.0)
    w.add(0.0, 0.0)
    w.add(5.0, 100.0)
    rate = w.compute_rate(5.0, 100.0)
    assert rate == pytest.approx(20.0)


def test_sliding_window_prunes():
    w = SlidingWindow(window_sec=2.0)
    w.add(0.0, 0.0)
    w.add(1.0, 10.0)
    w.add(3.0, 30.0)  # Should prune t=0.0.
    assert len(w.samples) == 2


def test_sliding_window_insufficient_data():
    w = SlidingWindow()
    assert w.compute_rate(1.0, 5.0) == 0.0
    w.add(0.0, 0.0)
    assert w.compute_rate(1.0, 5.0) == 0.0


def test_total_variation_is_half_the_l1_distance() -> None:
    p = torch.tensor([[0.5, 0.25, 0.25], [1.0, 0.0, 0.0]])
    q = torch.tensor([[0.25, 0.25, 0.5], [0.0, 0.0, 1.0]])
    assert torch.equal(total_variation(p, q), torch.tensor([0.25, 1.0]))
    assert torch.equal(
        total_variation(p, q, dim=0, keepdim=True),
        torch.tensor([[(0.25 + 1.0) / 2, 0.0, (0.25 + 1.0) / 2]]),
    )


def test_holm_adjusts_in_input_order_through_ties_and_the_clamp() -> None:
    p = torch.tensor([0.01, 0.55, 0.03, 0.6, 0.03], dtype=torch.float64)
    # Ascending, 0.01, 0.03, 0.03, 0.55 and 0.6 are multiplied by 5, 4, 3, 2 and 1.
    # The second 0.03 and 0.6 rise to the running maximum; 2 * 0.55 clamps at 1.
    expected = [5 * 0.01, 1.0, 4 * 0.03, 1.0, 4 * 0.03]
    assert torch.equal(holm(p), torch.tensor(expected, dtype=torch.float64))


def test_holm_reads_python_floats_as_float64_and_each_row_as_a_family() -> None:
    assert holm([0.04, 0.01, 0.03]).tolist() == [2 * 0.03, 3 * 0.01, 2 * 0.03]
    rows = torch.tensor([[0.04, 0.01, 0.03], [0.2, 0.3, 0.1]], dtype=torch.float64)
    expected = [[2 * 0.03, 3 * 0.01, 2 * 0.03], [2 * 0.2, 2 * 0.2, 3 * 0.1]]
    assert holm(rows).tolist() == expected
    assert holm([]).shape == (0,)


def test_sliding_window_default_span_prunes_after_thirty_seconds():
    window = SlidingWindow()
    window.add(0.0, 0.0)
    window.add(30.5, 10.0)
    assert window.samples == [(30.5, 10.0)]


def test_cov_cross_centers_each_input_and_uses_observation_count():
    x = torch.tensor([[0.0, 2.0], [2.0, 0.0], [4.0, 4.0]])
    y = torch.tensor([[1.0, 3.0], [5.0, 1.0], [3.0, 8.0]])
    expected = (x - x.mean(0)).T @ (y - y.mean(0)) / 3
    torch.testing.assert_close(cov(x, y), expected)
    torch.testing.assert_close(cov(x, y, bias=False), expected * 1.5)


def test_cov_rowvar_cross_and_single_observation():
    x = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 8.0]])
    y = torch.tensor([[1.0, 4.0, 7.0], [2.0, 2.0, 6.0]])
    expected = (x - x.mean(-1, keepdim=True)) @ (y - y.mean(-1, keepdim=True)).T / 3
    torch.testing.assert_close(cov(x, y, rowvar=True), expected)
    one = cov(torch.tensor([[2.0, 8.0]]), bias=False)
    # `cov` returns a square variable-by-variable covariance matrix.
    assert one.shape == (2, 2)
    assert torch.isnan(one).all()


def test_cov_batched_rowvar_cross_centers_and_preserves_batch_axis():
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 7.0], [5.0, 4.0, 3.0, 2.0]]],
    )
    y = torch.tensor([[[3.0, 1.0, 2.0, 4.0], [5.0, 2.0, 1.0, 0.0]]])
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    expected = x_centered @ y_centered.transpose(-1, -2) / x.shape[-1]
    # A singleton batch axis distinguishes batched output from scalar output.
    result = cov(x, y, rowvar=True)
    assert result.shape == (1, 3, 2)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("add_feature_axis", [False, True])
def test_cov_cross_centers_large_means_before_multiplication(
    add_feature_axis: bool,
) -> None:
    x = torch.tensor([1.0, 1.0, 1.0, -3.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 4.0, 8.0], dtype=torch.float64) + 1e16
    if add_feature_axis:
        x = x[:, None]
        y = y[:, None]
    result = cov(x, y, bias=True)
    assert torch.equal(result.reshape(()), torch.tensor(-4.5, dtype=torch.float64))


def test_cov_cross_with_large_feature_means_preserves_precision():
    # Column sums divisible by six and steps of 8 above 1e16 keep every partial
    # sum, mean, centered value and product exact, so only the final division
    # rounds and the result is bit-identical in any accumulation order (x86 and
    # aarch64 matmuls differ there). Skipping the centering of ``y`` misses by 5.
    x = torch.tensor(
        [
            [10.0, 0.0, -10.0],
            [-10.0, -20.0, -10.0],
            [30.0, 0.0, 10.0],
            [20.0, -10.0, -30.0],
            [-20.0, 20.0, -10.0],
            [30.0, 10.0, -10.0],
        ],
        dtype=torch.float64,
    )
    y = (
        torch.tensor(
            [
                [0.0, 0.0, 24.0],
                [-16.0, 0.0, -16.0],
                [0.0, 8.0, -16.0],
                [16.0, 8.0, 16.0],
                [-16.0, -16.0, 0.0],
                [16.0, 0.0, 16.0],
            ],
            dtype=torch.float64,
        )
        + 1e16
    )
    expected = torch.tensor(
        [
            [213.33333333333334, 120.0, 80.0],
            [0.0, -66.66666666666667, 53.333333333333336],
            [-53.333333333333336, 0.0, -106.66666666666667],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(cov(x, y), expected)


def test_entropy_probs_respects_zero_mass_and_keepdim():
    p = torch.tensor([[0.0, 0.25, 0.75], [0.5, 0.5, 0.0]])
    q = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
    expected = -(p * q.log()).sum(-1, keepdim=True)
    torch.testing.assert_close(entropy_probs(p, q, keepdim=True), expected)


def test_entropy_probs_zero_mass_has_zero_probability_gradient():
    p = torch.tensor([0.0, 1.0], requires_grad=True)
    q = torch.tensor([0.25, 0.75])
    entropy_probs(p, q).backward()
    assert p.grad is not None
    assert p.grad[0] == 0.0


def test_entropy_logits_mean_all_to_all_exact_mean_distribution():
    x = torch.tensor([[[0.0, 2.0], [1.0, -1.0]], [[2.0, 0.0], [-1.0, 1.0]]])
    y = torch.tensor([[[1.0, 0.0], [-2.0, 1.0]], [[0.0, 2.0], [1.0, -1.0]]])
    mean_p = torch.softmax(x, -1).mean(0)
    mean_q = torch.softmax(y, -1).mean(0)
    expected = -(mean_p * mean_q.log()).sum(-1)
    torch.testing.assert_close(
        entropy_logits_mean_all_to_all(x, y, dim=-1, dim_mean=0),
        expected,
    )
    kept = entropy_logits_mean_all_to_all(
        x,
        y,
        dim=-1,
        dim_mean=0,
        keepdim=True,
        keepdim_mean=True,
    )
    assert kept.shape == (1, 2, 1)
    torch.testing.assert_close(kept.squeeze(0), expected.unsqueeze(-1))


def test_entropy_logits_mean_all_to_all_matches_torch_softmax_bits():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    mean_p = torch.softmax(x, dim=-1).mean(dim=0)
    log_q = torch.log_softmax(y, dim=-1)
    log_mean_q = torch.logsumexp(log_q, dim=0) - math.log(2)
    expected = -(mean_p * log_mean_q).sum(dim=-1)
    actual = entropy_logits_mean_all_to_all(x, y, dim=-1, dim_mean=0)
    assert torch.equal(actual, expected)


def test_entropy_logits_mean_all_to_all_infers_non_event_axes():
    x = torch.arange(120, dtype=torch.float64).reshape(2, 3, 4, 5) / 13
    y = torch.flip(x, dims=(0, 2))
    dim_mean = (0, 1, 2)
    mean_p = torch.softmax(x, dim=-1).mean(dim=dim_mean)
    log_q = torch.log_softmax(y, dim=-1)
    log_mean_q = torch.logsumexp(log_q, dim=dim_mean) - math.log(24)
    expected = -(mean_p * log_mean_q).sum(dim=-1)
    actual = entropy_logits_mean_all_to_all(x, y, dim=-1, dim_mean=None)
    explicit = entropy_logits_mean_all_to_all(x, y, dim=-1, dim_mean=dim_mean)
    assert torch.equal(actual, expected)
    assert torch.equal(explicit, expected)


def test_entropy_logits_mean_all_to_all_normalizes_multi_axis_events():
    x = torch.arange(120, dtype=torch.float64).reshape(2, 3, 4, 5) / 13
    y = torch.flip(x, dims=(0, 2))
    event_dims = (-2, -1)
    p = torch.softmax(x.reshape(2, 3, 20), dim=-1).reshape_as(x)
    mean_p = p.mean(dim=(0, 1))
    log_q = torch.log_softmax(y.reshape(2, 3, 20), dim=-1).reshape_as(y)
    log_mean_q = torch.logsumexp(log_q, dim=(0, 1)) - math.log(6)
    expected = -(mean_p * log_mean_q).sum(dim=event_dims)
    actual = entropy_logits_mean_all_to_all(
        x,
        y,
        dim=event_dims,
        dim_mean=None,
    )
    assert torch.equal(actual, expected)


def test_entropy_logits_mean_all_to_all_forwards_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7
    y = torch.flip(x, dims=(0, 1))

    def gather(outputs: list[Tensor], partial: Tensor) -> None:
        assert len(outputs) == 2
        outputs[0].copy_(partial)
        outputs[1].copy_(partial + 0.5)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    actual = entropy_logits_mean_all_to_all(
        x,
        y,
        dim=-1,
        dim_mean=1,
        keepdim=True,
        keepdim_mean=True,
        world_size=2,
    )
    mean_p = torch.softmax(x, dim=-1).mean(dim=1, keepdim=True)
    log_q = torch.log_softmax(y, dim=-1)
    partial = torch.logsumexp(log_q, dim=1, keepdim=True)
    log_mean_q = torch.logsumexp(torch.stack((partial, partial + 0.5)), dim=0)
    log_mean_q -= math.log(6)
    torch.testing.assert_close(
        actual,
        -(mean_p * log_mean_q).sum(dim=-1, keepdim=True),
    )


def test_entropy_logits_mean_all_to_all_preserves_middle_mean_axis():
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 11
    y = torch.flip(x, dims=(0, 2))
    mean_p = torch.softmax(x, dim=-1).mean(dim=1, keepdim=True)
    mean_q = torch.softmax(y, dim=-1).mean(dim=1, keepdim=True)
    expected = -(mean_p * mean_q.log()).sum(dim=-1, keepdim=True)
    actual = entropy_logits_mean_all_to_all(
        x,
        y,
        dim=-1,
        dim_mean=1,
        keepdim=True,
        keepdim_mean=True,
    )
    torch.testing.assert_close(actual, expected)


def test_jsd_nonleading_ensemble_and_event_dims():
    logits = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 0.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 1.0], [1.0, 0.0]],
        ],
    )
    logp = torch.log_softmax(logits, dim=2)
    probs = logp.exp()
    mixture = probs.mean(0)
    expected = -(mixture * mixture.log()).sum(-1) - (-(probs * logp).sum(-1)).mean(0)
    torch.testing.assert_close(jsd(logp, ensemble_dim=0, event_dim=2), expected)


def test_jsd_reduces_tuple_ensemble_and_event_axes():
    logits = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    ensemble_dims = (0, 1)
    event_dims = (2,)
    mixture = probs.mean(dim=ensemble_dims)
    expected = -(mixture * mixture.log()).sum(dim=-1) - (
        -(probs * logp).sum(dim=event_dims)
    ).mean(dim=ensemble_dims)
    torch.testing.assert_close(
        jsd(logp, ensemble_dim=ensemble_dims, event_dim=event_dims),
        expected,
    )


def test_pca_centers_float_input_and_whitening_regularizer():
    seen: list[Tensor] = []

    def decompose(centered: Tensor) -> tuple[Tensor, Tensor]:
        seen.append(centered)
        return torch.tensor([1.0, 4.0]), torch.eye(2)

    x = torch.tensor([[0, 1], [2, 5], [4, 3]], dtype=torch.float64)
    values, vectors = pca(x, whiten=True, eps=3.0, decompose=decompose)
    assert seen[0].dtype == torch.float32
    torch.testing.assert_close(seen[0], (x.float() - x.float().mean(0)))
    torch.testing.assert_close(values, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(vectors, torch.diag(torch.tensor([0.5, 1 / 7**0.5])))


def test_pca_power_uses_input_device_and_dtype(monkeypatch: pytest.MonkeyPatch):
    randn = Mock(wraps=torch.randn)
    monkeypatch.setattr(torch, "randn", randn)
    pca_power(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64),
        num_iters=0,
    )
    assert randn.call_args.args == (3, 3)
    assert randn.call_args.kwargs == {
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }


def test_householder_qr_reconstructs_rectangular_matrix_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    eye = Mock(wraps=torch.eye)
    monkeypatch.setattr(torch, "eye", eye)
    mat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=torch.float64)
    q, r = _householder_qr(mat)
    torch.testing.assert_close(q @ r, mat)
    torch.testing.assert_close(q.T @ q, torch.eye(3, dtype=torch.float64))
    assert q.dtype == mat.dtype
    assert q.device == mat.device
    assert eye.call_args_list[0].kwargs == {
        "device": mat.device,
        "dtype": mat.dtype,
    }


def test_householder_qr_sign_choice_avoids_cancellation():
    epsilon = torch.finfo(torch.float32).eps
    for sign in (-1.0, 1.0):
        matrix = torch.tensor(
            [[sign, 0.0], [epsilon, 1.0], [0.0, 0.0]],
            dtype=torch.float32,
        )
        q, r = _householder_qr(matrix)
        torch.testing.assert_close(q @ r, matrix, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.tril(r, diagonal=-1), torch.zeros_like(r))


def test_householder_qr_uses_strict_scaled_zero_threshold():
    epsilon = torch.finfo(torch.float32).eps
    matrix = torch.tensor(
        [[epsilon, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    _, r = _householder_qr(matrix)
    expected = torch.tensor(
        [[-epsilon, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(r, expected)


def test_householder_qr_threshold_tracks_matrix_scale():
    epsilon = torch.finfo(torch.float32).eps
    matrix = torch.tensor(
        [[1.25 * epsilon, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    q, r = _householder_qr(matrix)
    torch.testing.assert_close(q @ r, matrix)
    assert q[0, 0] == -1
    assert torch.equal(r, -matrix)


def test_householder_qr_continues_after_a_zero_column():
    matrix = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    q, r = _householder_qr(matrix)
    torch.testing.assert_close(q @ r, matrix)
    torch.testing.assert_close(torch.tril(r, diagonal=-1), torch.zeros_like(r))


def test_quantile_normalize_pins_quantile_endpoints():
    x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 100.0]])
    torch.testing.assert_close(quantile_normalize(x, q=0.1), (x - 0.5) / 51.5)
    torch.testing.assert_close(quantile_normalize(x, q=0.2), (x - 1.0) / 3.0)
    narrow = torch.tensor([0.0, 0.25, 0.5, 0.75])
    torch.testing.assert_close(
        quantile_normalize(narrow, q=0.25),
        (narrow - 0.1875) / 0.375,
    )


def test_ema_update_default_weights_the_new_observation():
    assert ema_update(10.0, 20.0) == pytest.approx(13.0)


def test_quantile_normalize_constant_input_has_finite_zero_gradient():
    x = torch.full((4,), 5.0, requires_grad=True)
    quantile_normalize(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_sliding_window_uses_elapsed_time_and_cumulative_count_deltas():
    window = SlidingWindow(window_sec=20.0)
    window.add(3.0, 10.0)
    window.add(8.0, 20.0)
    assert window.compute_rate(9.0, 30.0) == pytest.approx(22.0 / 6.1)


def test_householder_zero_pivot_is_triangular() -> None:
    # QR of a permutation is square by definition.
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    q, r = _householder_qr(matrix)
    torch.testing.assert_close(q @ r, matrix)
    torch.testing.assert_close(r.tril(-1), torch.zeros_like(r), atol=1e-14, rtol=0)


def test_pca_power_is_scale_equivariant() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, -2.0], [-1.0, 4.0]]) * 1e-5
    x = x - x.mean(0)
    expected, _ = pca_eigh(x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actual, _ = pca_power(x, num_iters=40)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)


def test_quantile_normalize_float64() -> None:
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    torch.testing.assert_close(quantile_normalize(x, q=0.2), (x - 1) / 3)


@pytest.mark.parametrize("keepdim", [False, True])
def test_entropy_logits_joint_event_axes(keepdim: bool) -> None:
    x = torch.zeros(2, 3, 4, dtype=torch.float64)
    result = entropy_logits(x, dim=(1, 2), keepdim=keepdim)
    expected = torch.full((2, 1, 1) if keepdim else (2,), math.log(12), dtype=x.dtype)
    torch.testing.assert_close(result, expected)


def test_entropy_mean_remaps_event_axis() -> None:
    x = torch.zeros(2, 3, 4, dtype=torch.float64)
    result = entropy_logits_mean_all_to_all(x, dim=1, dim_mean=0)
    torch.testing.assert_close(result, torch.full((4,), math.log(3), dtype=x.dtype))


def test_entropy_boundary_contracts() -> None:
    torch.testing.assert_close(
        entropy_logits(torch.tensor([0.0, -math.inf])),
        torch.tensor(0.0),
    )
    assert entropy_probs(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])).isposinf()
    assert cov(torch.tensor([2.0]), bias=False).isnan()
    assert cov(torch.tensor([[2.0, 3.0]]), bias=False).isnan().all()


@pytest.mark.parametrize("constant", [False, True])
def test_pca_rank_deficient_whitening_is_finite(constant: bool) -> None:
    x = (
        torch.zeros(2, 3)
        if constant
        else torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    )
    _, vectors = pca(x, whiten=True)
    assert vectors.isfinite().all()


_EMPTY_DIM_REDUCERS: list[Callable[[Tensor], Tensor]] = [
    functools.partial(entropy_logits, dim=()),
    functools.partial(entropy_probs, dim=()),
    functools.partial(total_variation, torch.zeros(2, 3), dim=()),
    functools.partial(entropy_logits_mean_all_to_all, dim=()),
    functools.partial(jsd, ensemble_dim=()),
    functools.partial(jsd, event_dim=()),
]


@pytest.mark.parametrize("reduce", _EMPTY_DIM_REDUCERS)
def test_reducers_reject_an_empty_dim(reduce: Callable[[Tensor], Tensor]) -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        reduce(torch.zeros(2, 3))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
