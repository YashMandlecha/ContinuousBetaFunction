"""Tests for betafn.processing.gamma: gamma_method_covariance and gamma_method_average.

Key properties checked:
  - For white noise (τ_int = 0.5), the gamma method reproduces the standard
    sample-mean variance σ²/N within sampling noise.
  - For AR(1) autocorrelated data, errors are inflated relative to σ²/N.
  - Cross-covariances within the same observable group are non-zero and
    consistent with the true correlation.
  - integrated_autocorrelation_time returns ≈ 0.5 for white noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from betafn.processing.gamma import (
    gamma_method_average,
    gamma_method_covariance,
    integrated_autocorrelation_time,
)


# ---------------------------------------------------------------------------
# gamma_method_covariance
# ---------------------------------------------------------------------------

class TestGammaMethodCovariance:
    def _white_noise(self, N: int = 2000, M: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        sigma = np.array([1.0, 2.0, 0.5, 3.0])[:M]
        means = np.array([5.0, 10.0, -1.0, 0.0])[:M]
        X = means + sigma * rng.standard_normal((N, M))
        return X, means, sigma

    def test_output_shapes(self):
        X, _, _ = self._white_noise(N=500, M=3)
        means, cov, W = gamma_method_covariance(X)
        assert means.shape == (3,)
        assert cov.shape == (3, 3)
        assert isinstance(W, int) and W >= 1

    def test_means_close_to_true(self):
        X, true_means, _ = self._white_noise(N=5000, M=4)
        means, _, _ = gamma_method_covariance(X)
        np.testing.assert_allclose(means, true_means, atol=0.1)

    def test_diagonal_covariance_white_noise(self):
        """For iid data, Cov(ā_α, ā_β) should be σ²/N on the diagonal."""
        N, M = 5000, 3
        X, _, sigma = self._white_noise(N=N, M=M)
        _, cov, _ = gamma_method_covariance(X)
        expected_diag = sigma[:M] ** 2 / N
        # Allow ±30% tolerance due to statistical estimation noise
        for i in range(M):
            assert abs(cov[i, i] - expected_diag[i]) / expected_diag[i] < 0.30, \
                f"Diagonal entry {i}: got {cov[i,i]:.2e}, expected {expected_diag[i]:.2e}"

    def test_covariance_matrix_is_symmetric(self):
        X, _, _ = self._white_noise(N=1000, M=4)
        _, cov, _ = gamma_method_covariance(X)
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_covariance_matrix_is_psd(self):
        """The covariance matrix should be positive semi-definite."""
        X, _, _ = self._white_noise(N=1000, M=4)
        _, cov, _ = gamma_method_covariance(X)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-12), f"Negative eigenvalue: {eigvals.min()}"

    def test_autocorrelated_data_inflates_errors(self):
        """AR(1) process with φ=0.6 has τ_int = 1/(1-φ²) - 0.5 ≈ 1.1.
        Errors should be roughly sqrt(2*τ_int) ≈ 1.5× larger than σ/√N."""
        rng = np.random.default_rng(7)
        N, phi = 5000, 0.6
        X = np.zeros((N, 1))
        noise = rng.standard_normal(N)
        for t in range(1, N):
            X[t, 0] = phi * X[t - 1, 0] + noise[t]

        sigma_sample = np.sqrt(1.0 / (1 - phi**2))   # stationary std
        _, cov_gm, _ = gamma_method_covariance(X)

        # True variance of mean: σ²/(N) * 2*tau_int
        tau_int = 0.5 / (1 - phi)  # exact for AR(1)
        expected_var_of_mean = sigma_sample**2 / N * 2 * tau_int

        ratio = cov_gm[0, 0] / expected_var_of_mean
        assert 0.5 < ratio < 2.0, \
            f"Variance-of-mean ratio = {ratio:.2f} (expected ≈ 1.0)"

    def test_cross_covariance_of_correlated_channels(self):
        """Two channels that are correlated should have non-zero off-diagonal covariance."""
        rng = np.random.default_rng(42)
        N = 3000
        common = rng.standard_normal(N)
        X = np.column_stack([common + 0.1 * rng.standard_normal(N),
                             common + 0.1 * rng.standard_normal(N)])
        _, cov, _ = gamma_method_covariance(X)
        # Off-diagonal should be large and positive
        assert cov[0, 1] > 0.8 * cov[0, 0], \
            f"Expected large positive cross-covariance, got {cov[0,1]:.4e}"


# ---------------------------------------------------------------------------
# gamma_method_average
# ---------------------------------------------------------------------------

class TestGammaMethodAverage:
    def _make_data(self, n_configs: int = 500, n_ft: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        data: dict[str, list] = {}
        for obs in ("Ep", "Es", "Q"):
            for j, ft in enumerate(["1.0", "2.0", "3.0"][:n_ft]):
                data[f"wilson_{obs}_{ft}"] = rng.normal(1.0, 0.01, n_configs).tolist()
        return data

    def test_returns_gvar_for_every_key(self):
        import gvar as gv
        data = self._make_data()
        result = gamma_method_average(data)
        assert set(result.keys()) == set(data.keys())
        for v in result.values():
            assert isinstance(v, gv.GVar)

    def test_means_close_to_true(self):
        import gvar as gv
        rng = np.random.default_rng(1)
        N = 2000
        data = {"wilson_Ep_1.0": rng.normal(5.0, 0.01, N).tolist()}
        result = gamma_method_average(data)
        assert abs(gv.mean(result["wilson_Ep_1.0"]) - 5.0) < 0.05

    def test_empty_input_returns_empty(self):
        assert gamma_method_average({}) == {}

    def test_cross_covariance_within_group_is_nonzero(self):
        """gvars from the same observable group should be correlated."""
        import gvar as gv
        rng = np.random.default_rng(5)
        N = 1000
        common = rng.standard_normal(N)
        data = {
            "wilson_Ep_1.0": (common + 0.01 * rng.standard_normal(N)).tolist(),
            "wilson_Ep_2.0": (common + 0.01 * rng.standard_normal(N)).tolist(),
        }
        result = gamma_method_average(data)
        cov = gv.evalcov([result["wilson_Ep_1.0"], result["wilson_Ep_2.0"]])
        # Off-diagonal should be close to the diagonal (high correlation)
        assert cov[0, 1] > 0.8 * cov[0, 0], \
            f"Expected strong cross-covariance within Ep group, got {cov[0,1]:.4e}"


# ---------------------------------------------------------------------------
# integrated_autocorrelation_time
# ---------------------------------------------------------------------------

class TestIntegratedAutocorrelationTime:
    def test_white_noise_gives_half(self):
        rng = np.random.default_rng(0)
        series = rng.standard_normal(5000)
        tau = integrated_autocorrelation_time(series)
        assert 0.4 < tau < 0.8, f"Expected τ_int ≈ 0.5 for white noise, got {tau:.3f}"

    def test_ar1_gives_larger_tau(self):
        rng = np.random.default_rng(1)
        phi = 0.7
        N = 5000
        X = np.zeros(N)
        for t in range(1, N):
            X[t] = phi * X[t - 1] + rng.standard_normal()
        tau = integrated_autocorrelation_time(X)
        # Exact τ_int for AR(1): 0.5 + φ/(1−φ) ≈ 2.83 for φ=0.7
        tau_exact = 0.5 + phi / (1 - phi)
        assert tau > 1.0, f"Expected τ_int > 1.0 for AR(1) with φ=0.7, got {tau:.3f}"
        assert abs(tau - tau_exact) / tau_exact < 0.4, \
            f"τ_int = {tau:.2f}, expected ≈ {tau_exact:.2f}"

    def test_constant_series_returns_half(self):
        tau = integrated_autocorrelation_time(np.ones(100))
        assert tau == pytest.approx(0.5)

    def test_very_short_series(self):
        tau = integrated_autocorrelation_time([1.0, 2.0, 3.0])
        assert tau == pytest.approx(0.5)
