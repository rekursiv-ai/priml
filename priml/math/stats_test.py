from __future__ import annotations

import functools

import pytest
import torch

from priml.math.stats import (
    SlidingWindow,
    cov,
    ema_update,
    entropy_logits,
    entropy_logits_mean_all_to_all,
    entropy_probs,
    pca,
    pca_eigh,
    pca_power,
    pca_svd,
    quantile_normalize,
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


def test_cov_1d_matches_numpy():
    """A 1-D input is a single variable; cov returns its scalar variance."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    biased = cov(x, bias=True)
    assert biased.ndim == 0
    torch.testing.assert_close(biased, torch.tensor(1.25))
    unbiased = cov(x, bias=False)
    torch.testing.assert_close(unbiased, torch.tensor(5.0 / 3.0))


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
    mix = torch.randn(5, 5)
    x = x @ mix  # Add correlation.
    _, eigenvectors = pca(x, whiten=True, eps=1e-5)
    projected = (x - x.mean(0)) @ eigenvectors
    # Each component should have approximately unit variance.
    torch.testing.assert_close(projected.var(0), torch.ones(5), atol=0.15, rtol=0.15)


def test_pca_power_matches_eigh():
    """Power iteration and eigh paths should produce equivalent results."""
    torch.manual_seed(7)
    x = torch.randn(100, 8)
    x_centered = (x - x.mean(0)).float()
    vals_eigh, vecs_eigh = pca_eigh(x_centered)
    vals_power, vecs_power = pca_power(x_centered)
    torch.testing.assert_close(vals_power, vals_eigh, atol=1e-3, rtol=1e-3)
    # Eigenvectors may differ by sign; compare absolute values.
    torch.testing.assert_close(vecs_power.abs(), vecs_eigh.abs(), atol=1e-3, rtol=1e-3)


def test_pca_power_tol_early_exit_matches_full():
    """The ``tol`` early-exit still recovers the eigenvalues to tolerance."""
    torch.manual_seed(7)
    x = torch.randn(100, 8)
    x_centered = (x - x.mean(0)).float()
    vals_full, _ = pca_power(x_centered, num_iters=200, tol=0.0)
    vals_tol, _ = pca_power(x_centered, num_iters=200, tol=1e-6)
    torch.testing.assert_close(vals_tol, vals_full, atol=1e-3, rtol=1e-3)


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


def test_pca_third_party_decompose_needs_no_library_change():
    """An arbitrary caller-supplied decomposer is honored verbatim."""

    def scaled_eigh(x_centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    mix = torch.randn(4, 4)
    x = x @ mix
    eigenvalues, V_raw = pca(x)
    # ZCA whitening matrix: V @ diag(1/sqrt(λ)) @ V^T
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
    w.add(3.0, 30.0)  # Should prune t=0.0
    assert len(w.samples) == 2


def test_sliding_window_insufficient_data():
    w = SlidingWindow()
    assert w.compute_rate(1.0, 5.0) == 0.0
    w.add(0.0, 0.0)
    assert w.compute_rate(1.0, 5.0) == 0.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
