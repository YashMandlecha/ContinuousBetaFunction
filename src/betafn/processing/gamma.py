"""Gamma-method (Madras-Sokal / Wolff UWerr) covariance estimation for MC data."""
from __future__ import annotations

import numpy as _numpy
import gvar as _gvar


def gamma_method_covariance(
    X: _numpy.ndarray,
    window_factor: float = 3.0,
) -> tuple[_numpy.ndarray, _numpy.ndarray, int]:
    """Compute the mean vector and covariance-of-means using the Gamma method.

    Standard Markov-chain averages underestimate statistical errors whenever
    successive configurations are autocorrelated (τ_int > 0.5).  For a single
    observable the fix is well-known — bin or multiply σ² by 2τ_int — but the
    off-diagonal entries of the covariance matrix between two observables α, β
    measured on the *same* MC history each receive a *different* correction that
    depends on the full cross-spectral density Γ_αβ(k).  No single τ_int can
    handle both diagonal and off-diagonal entries simultaneously; the only
    rigorous solution is the Gamma method.

    The Gamma method integrates the exact cross-correlation functions::

        Cov(ā_α, ā_β) = (1/N) * [Γ_αβ(0) + Σ_{k=1}^W (Γ_αβ(k) + Γ_βα(k))]

    where Γ_αβ(k) = (1/N) Σ_s δX_α(s) δX_β(s+k) is the lag-k cross-correlator
    of the centered series, and W is the Sokal automatic window.

    **Window selection.**  W is chosen to satisfy W ≥ window_factor × max_α τ_int(α)
    using the diagonal per-observable integrated autocorrelation times.  Using
    the same W for every pair (α, β) guarantees positive semi-definiteness.

    References: Madras & Sokal (1988) J.Stat.Phys. 50, 109;
    Wolff (2004) Comput.Phys.Commun. 156, 143.

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Raw MC time series in trajectory order.  N = configurations, M = observables.
    window_factor : float
        Safety factor S for the Sokal window: W = ceil(S × max_α τ_int(α)).
        Default 3.0 is conservative; Sokal's original recommendation is 1.5.

    Returns
    -------
    means : ndarray, shape (M,)
    cov_of_means : ndarray, shape (M, M)
        Covariance of sample means including all autocorrelations up to lag W.
    W : int
        Window width used (useful for diagnostics).
    """
    N, M = X.shape
    means = X.mean(axis=0)
    delta = X - means[_numpy.newaxis, :]

    # --- Step 1: estimate W from per-observable diagonal τ_int ---------------
    var_diag = (delta * delta).sum(axis=0) / N
    safe_var = _numpy.where(var_diag > 0.0, var_diag, 1.0)

    tau_diag = _numpy.full(M, 0.5)
    max_probe = min(N // 4, 1000)
    for k in range(1, max_probe):
        lag_corr = (delta[:N - k] * delta[k:]).sum(axis=0) / (N * safe_var)
        still_active = (lag_corr > 0.0) & (k < window_factor * tau_diag)
        tau_diag = _numpy.where(still_active, tau_diag + lag_corr, tau_diag)
        if not still_active.any():
            break

    W = max(4, int(_numpy.ceil(window_factor * float(_numpy.max(tau_diag)))))
    W = min(W, N // 4)

    # --- Step 2: accumulate the symmetrised cross-correlation sum -------------
    cov_sum = delta.T @ delta
    for k in range(1, W + 1):
        Gk = delta[: N - k].T @ delta[k:]
        cov_sum += Gk + Gk.T

    cov_of_means = cov_sum / (N * N)
    return means, cov_of_means, W


def gamma_method_average(
    data: dict,
    process=None,
    window_factor: float = 3.0,
) -> dict:
    """Compute averaged gvars via the Gamma method instead of binning.

    ``process`` is accepted for API compatibility with ``ProcessHooks.average``
    but is ignored; the Gamma method supersedes all binning.

    Observables are grouped by type — the second underscore-delimited token in
    each key (e.g. "Ep", "Es", "Ec", "Q") — and the Gamma method is applied
    independently within each group.  Cross-covariances between different
    observable types are set to zero, which slightly overestimates errors on
    combined observables but keeps each BLAS call at O(N·M_ft²).
    """
    if not data:
        return {}

    groups: dict[str, list[str]] = {}
    for k in data:
        obs_type = k.split("_")[1]
        groups.setdefault(obs_type, []).append(k)

    result: dict[str, object] = {}
    for obs_type, group_keys in groups.items():
        sorted_keys = sorted(group_keys)
        n_configs = len(data[sorted_keys[0]])

        X = _numpy.empty((n_configs, len(sorted_keys)), dtype=float)
        for col, k in enumerate(sorted_keys):
            X[:, col] = data[k]

        means, cov_of_means, _W = gamma_method_covariance(X, window_factor=window_factor)
        gvars = _gvar.gvar(means, cov_of_means)
        for i, k in enumerate(sorted_keys):
            result[k] = gvars[i]

    return result


def integrated_autocorrelation_time(series, max_lag: int | None = None) -> float:
    """Estimate the integrated autocorrelation time of a Monte Carlo series.

    Uses the standard sum of normalised autocorrelations with an automatic
    windowing cutoff at the first non-positive coefficient.
    """
    values = _numpy.asarray(series, dtype=float)
    n = len(values)
    if n < 4:
        return 0.5
    centered = values - values.mean()
    variance = float(centered @ centered) / n
    if variance == 0.0:
        return 0.5
    max_lag = n // 2 if max_lag is None else min(max_lag, n - 1)
    tau = 0.5
    for lag in range(1, max_lag):
        rho = float(centered[:-lag] @ centered[lag:]) / ((n - lag) * variance)
        if rho <= 0.0:
            break
        tau += rho
    return tau
