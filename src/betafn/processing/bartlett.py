"""Multivariate Bartlett and lugsail spectral estimation for MC histories.

The estimator in this module acts on the *configuration/trajectory* index.
Gradient-flow time and discretisation label are components of the vector
measured on each configuration; they are not the time index of the spectral
estimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gvar as _gvar
import numpy as _numpy


@dataclass(frozen=True)
class BartlettDiagnostics:
    """Diagnostics accompanying one multivariate spectral estimate."""

    n_configurations: int
    n_observables: int
    estimator: str
    bandwidth_method: str
    bandwidth: int
    bandwidth_fraction: float
    plugin_unbounded_bandwidth: float
    plugin_max_ar_order: int
    plugin_median_ar_order: float
    plugin_standardized_marginals: bool
    window_factor: float
    max_bandwidth_fraction: float
    tau_int_median: float
    tau_int_max: float
    min_eigenvalue: float
    max_eigenvalue: float
    numerical_rank: int
    lugsail_r: float
    lugsail_c: float
    lugsail_inner_bandwidth: int
    raw_min_eigenvalue: float
    psd_adjusted: bool
    psd_adjusted_eigenvalues: int
    psd_relative_frobenius_change: float
    fallback_reason: str
    vanilla_median_se_ratio: float
    obm_batch_size: int
    obm_median_se_ratio: float
    obm_max_abs_log_se_ratio: float
    scan_lower_bandwidth: int
    scan_upper_bandwidth: int
    scan_lower_median_se_ratio: float
    scan_upper_median_se_ratio: float
    scan_max_abs_log_se_ratio: float

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def _as_history_matrix(values) -> _numpy.ndarray:
    matrix = _numpy.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("MC history must have shape (configurations, observables)")
    if matrix.shape[0] < 4:
        raise ValueError("Bartlett estimation requires at least four configurations")
    if matrix.shape[1] < 1:
        raise ValueError("Bartlett estimation requires at least one observable")
    if not _numpy.isfinite(matrix).all():
        raise ValueError("MC history contains non-finite values")
    return matrix


def _autocorrelation_diagnostics(
    centered: _numpy.ndarray,
    *,
    window_factor: float,
    maximum_lag: int,
) -> tuple[_numpy.ndarray, _numpy.ndarray]:
    """Return per-component self-consistent tau estimates and their windows."""
    n_configurations, n_observables = centered.shape
    variance_sum = _numpy.einsum("ij,ij->j", centered, centered)
    active_variance = variance_sum > 0.0
    safe_variance_sum = _numpy.where(active_variance, variance_sum, 1.0)

    tau = _numpy.full(n_observables, 0.5, dtype=float)
    windows = _numpy.full(n_observables, maximum_lag, dtype=int)
    finished = ~active_variance
    windows[finished] = 0

    for lag in range(1, maximum_lag + 1):
        rho = _numpy.einsum(
            "ij,ij->j", centered[:-lag], centered[lag:]
        ) / safe_variance_sum
        tau = _numpy.where(finished, tau, _numpy.maximum(0.5, tau + rho))
        newly_finished = (~finished) & (lag >= window_factor * tau)
        windows[newly_finished] = lag
        finished |= newly_finished
        if finished.all():
            break
    return tau, windows


def automatic_bartlett_bandwidth(
    values,
    *,
    window_factor: float = 5.0,
    max_fraction: float = 0.25,
    minimum: int = 2,
) -> tuple[int, _numpy.ndarray]:
    """Choose one conservative bandwidth for a multivariate MC history.

    A self-consistent Madras--Sokal window is found separately for every
    non-constant component.  The largest component window is then used for
    every covariance entry.  A common window is essential: choosing a
    different lag cutoff for each entry need not produce a positive
    semidefinite covariance matrix.
    """
    matrix = _as_history_matrix(values)
    n_configurations = matrix.shape[0]
    if window_factor <= 0.0:
        raise ValueError("window_factor must be positive")
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("max_fraction must lie strictly between zero and one")
    maximum_lag = max(1, min(n_configurations - 2, int(n_configurations * max_fraction)))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    tau, windows = _autocorrelation_diagnostics(
        centered, window_factor=window_factor, maximum_lag=maximum_lag,
    )
    bandwidth = max(int(minimum), int(windows.max(initial=0)))
    bandwidth = min(bandwidth, maximum_lag)
    return bandwidth, tau


def _autocovariances(values: _numpy.ndarray, maximum_lag: int) -> _numpy.ndarray:
    """Biased marginal autocovariances, matching the spectral estimator."""
    centered = values - values.mean(axis=0, keepdims=True)
    n_configurations = len(centered)
    result = _numpy.empty((maximum_lag + 1, centered.shape[1]), dtype=float)
    for lag in range(maximum_lag + 1):
        result[lag] = _numpy.einsum(
            "ij,ij->j", centered[: n_configurations - lag], centered[lag:]
        ) / n_configurations
    return result


def _ar_spectral_quantities(
    autocovariance: _numpy.ndarray,
    *,
    n_configurations: int,
    threshold: float,
) -> tuple[float, float, int]:
    """Estimate Liu--Vats--Flegal Gamma and Sigma from a marginal AR fit.

    Orders are examined sequentially with Yule--Walker fits.  The first order
    whose final (partial-autorrelation) coefficient is statistically
    negligible is excluded, reproducing the thresholded-AR rule used by
    ``mcmcse::batchSize`` without adding an R dependency.
    """
    variance = float(autocovariance[0])
    if not _numpy.isfinite(variance) or variance <= 0.0:
        return 0.0, 0.0, 0

    selected = _numpy.empty(0, dtype=float)
    maximum_order = len(autocovariance) - 1
    for order in range(1, maximum_order + 1):
        toeplitz = autocovariance[_numpy.abs(
            _numpy.arange(order)[:, None] - _numpy.arange(order)[None, :]
        )]
        try:
            coefficients = _numpy.linalg.solve(toeplitz, autocovariance[1 : order + 1])
        except _numpy.linalg.LinAlgError:
            coefficients = _numpy.linalg.lstsq(
                toeplitz, autocovariance[1 : order + 1], rcond=None
            )[0]
        if not _numpy.isfinite(coefficients).all() or abs(coefficients[-1]) <= threshold:
            break
        selected = coefficients

    order = len(selected)
    if order == 0:
        return 0.0, variance, 0

    innovation = variance - float(selected @ autocovariance[1 : order + 1])
    degrees_of_freedom = n_configurations - (order + 1)
    if innovation <= 0.0 or degrees_of_freedom <= 0:
        return 0.0, variance, 0
    innovation *= n_configurations / degrees_of_freedom
    denominator = 1.0 - float(selected.sum())
    if abs(denominator) <= _numpy.sqrt(_numpy.finfo(float).eps):
        return 0.0, variance, 0
    sigma = innovation / denominator**2

    weighted_sum = 0.0
    for index, coefficient in enumerate(selected, start=1):
        weighted_sum += coefficient * sum(
            lag * autocovariance[abs(lag - index)]
            for lag in range(1, index + 1)
        )
    gamma = 2.0 * (
        weighted_sum
        + 0.5 * (sigma - variance)
        * float(_numpy.arange(1, order + 1) @ selected)
    ) / denominator
    return float(gamma), float(sigma), order


def plugin_bartlett_bandwidth(
    values,
    *,
    max_fraction: float = 0.25,
) -> tuple[int, dict[str, float | int | bool]]:
    """Select the Bartlett bandwidth by Liu--Vats--Flegal's plug-in rule.

    A thresholded stationary AR model is fitted to every nonconstant marginal.
    Their estimated first-order bias and variance constants are combined into
    the multivariate AMSE-optimal Bartlett pilot bandwidth.  The AR rule and
    constants follow ``mcmcse::batchSize(method='bartlett')``.  Unlike that
    general-purpose implementation, the marginals are standardized because
    this application mixes physical units, and its ``N/(p+1)`` full-rank cap
    is not applied because the dense flow-time vector intentionally has p>N.
    """
    matrix = _as_history_matrix(values)
    n_configurations = len(matrix)
    active = _numpy.var(matrix, axis=0) > 0.0
    matrix = matrix[:, active]
    n_observables = matrix.shape[1]
    maximum_lag = max(1, min(n_configurations - 2, int(n_configurations * max_fraction)))
    if n_observables == 0:
        return 1, {
            "plugin_unbounded_bandwidth": 1.0,
            "plugin_max_ar_order": 0,
            "plugin_median_ar_order": 0.0,
            "plugin_standardized_marginals": True,
        }

    # The unit-weight multivariate criterion is scale-dependent.  This analysis
    # mixes energy densities and topology and spans many flow-time magnitudes,
    # so select M from unit-variance marginals.  Scaling does not affect the
    # covariance estimate itself; it only makes the scalar pilot bandwidth
    # invariant to units.
    matrix = (matrix - matrix.mean(axis=0, keepdims=True)) / matrix.std(
        axis=0, keepdims=True
    )
    maximum_order = min(
        n_observables,
        n_configurations - 2,
        max(1, int(_numpy.floor(10.0 * _numpy.log10(n_configurations)))),
    )
    autocovariance = _autocovariances(matrix, maximum_order)
    # qnorm(0.975) / sqrt(N), as in mcmcse::batchSize.
    threshold = 1.959963984540054 / _numpy.sqrt(n_configurations)
    gammas, sigmas, orders = [], [], []
    for column in range(n_observables):
        gamma, sigma, order = _ar_spectral_quantities(
            autocovariance[:, column],
            n_configurations=n_configurations,
            threshold=threshold,
        )
        if _numpy.isfinite(sigma) and sigma > 0.0:
            gammas.append(gamma)
            sigmas.append(sigma)
            orders.append(order)

    numerator = float(_numpy.square(gammas).sum())
    denominator = float(_numpy.square(sigmas).sum())
    if denominator <= 0.0 or not _numpy.isfinite(numerator / denominator):
        unbounded = 1.0
    else:
        unbounded = ((1.5 * n_configurations) * numerator / denominator) ** (1.0 / 3.0)
    # mcmcse caps at floor(N/(p+1)) to ensure a full-rank p-dimensional output.
    # Here p>N by design because a dense flow-time vector is retained, so that
    # cap would force M=1 and erase the serial-correlation estimate.  The
    # scalar marginal plug-in calculation itself requires no covariance
    # inversion; rank is handled and reported separately below.
    sample_bound = max(1, n_configurations // 10) if n_configurations > 10 else maximum_lag
    bandwidth = max(1, int(_numpy.floor(unbounded)))
    bandwidth = min(bandwidth, sample_bound, maximum_lag)
    return bandwidth, {
        "plugin_unbounded_bandwidth": float(unbounded),
        "plugin_max_ar_order": int(max(orders, default=0)),
        "plugin_median_ar_order": float(_numpy.median(orders)) if orders else 0.0,
        "plugin_standardized_marginals": True,
    }


def _bartlett_long_run_covariance(centered: _numpy.ndarray, bandwidth: int) -> _numpy.ndarray:
    """Evaluate the Bartlett lag-window sum by an exact moving-sum identity.

    This equals Gamma(0) + sum_{k=1}^M (1-k/(M+1))
    [Gamma(k)+Gamma(k).T], with every Gamma(k) normalised by N.  The moving-
    sum form requires only one matrix product and is positive semidefinite up
    to floating-point roundoff.
    """
    n_configurations = centered.shape[0]
    block_length = bandwidth + 1
    padded = _numpy.pad(centered, ((bandwidth, bandwidth), (0, 0)))
    cumulative = _numpy.vstack([
        _numpy.zeros((1, centered.shape[1]), dtype=float),
        _numpy.cumsum(padded, axis=0),
    ])
    n_blocks = n_configurations + bandwidth
    starts = _numpy.arange(n_blocks)
    block_sums = cumulative[starts + block_length] - cumulative[starts]
    covariance = block_sums.T @ block_sums / (n_configurations * block_length)
    return 0.5 * (covariance + covariance.T)


def _bartlett_long_run_variance(centered: _numpy.ndarray, bandwidth: int) -> _numpy.ndarray:
    """Diagonal of the Bartlett long-run covariance without a matrix product."""
    n_configurations = centered.shape[0]
    block_length = bandwidth + 1
    padded = _numpy.pad(centered, ((bandwidth, bandwidth), (0, 0)))
    cumulative = _numpy.vstack([
        _numpy.zeros((1, centered.shape[1]), dtype=float),
        _numpy.cumsum(padded, axis=0),
    ])
    starts = _numpy.arange(n_configurations + bandwidth)
    block_sums = cumulative[starts + block_length] - cumulative[starts]
    return _numpy.einsum("ij,ij->j", block_sums, block_sums) / (
        n_configurations * block_length
    )


def _lugsail_long_run_covariance(
    centered: _numpy.ndarray,
    bandwidth: int,
    *,
    r: float,
    c: float,
) -> tuple[_numpy.ndarray, _numpy.ndarray, int]:
    """Return lugsail and vanilla Bartlett estimates on the same history."""
    vanilla = _bartlett_long_run_covariance(centered, bandwidth)
    if r == 1.0 or c == 0.0:
        return vanilla, vanilla, bandwidth
    inner_bandwidth = max(1, int(_numpy.floor(bandwidth / r)))
    inner = _bartlett_long_run_covariance(centered, inner_bandwidth)
    lugsail = (vanilla - c * inner) / (1.0 - c)
    return 0.5 * (lugsail + lugsail.T), vanilla, inner_bandwidth


def _lugsail_long_run_variance(
    centered: _numpy.ndarray,
    bandwidth: int,
    *,
    r: float,
    c: float,
) -> _numpy.ndarray:
    vanilla = _bartlett_long_run_variance(centered, bandwidth)
    if r == 1.0 or c == 0.0:
        return vanilla
    inner = _bartlett_long_run_variance(
        centered, max(1, int(_numpy.floor(bandwidth / r)))
    )
    return (vanilla - c * inner) / (1.0 - c)


def _project_covariance_to_psd(
    covariance: _numpy.ndarray,
    *,
    relative_floor: float = 1e-12,
) -> tuple[_numpy.ndarray, int, float]:
    """Eigen-project a covariance to PSD while restoring its diagonal.

    The projection is performed on the correlation matrix so component units
    cannot determine the correction.  It is used only when a lugsail linear
    combination is indefinite, and its size is reported in diagnostics.
    """
    covariance = 0.5 * (covariance + covariance.T)
    variances = _numpy.diag(covariance).copy()
    if _numpy.any(variances <= 0.0):
        raise ValueError("cannot PSD-adjust a covariance with nonpositive diagonal")
    scales = _numpy.sqrt(variances)
    correlation = covariance / _numpy.outer(scales, scales)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = _numpy.linalg.eigh(correlation)
    floor = max(float(eigenvalues[-1]) * relative_floor, _numpy.finfo(float).eps)
    adjusted_count = int(_numpy.count_nonzero(eigenvalues < floor))
    eigenvalues = _numpy.maximum(eigenvalues, floor)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    projected_scales = _numpy.sqrt(_numpy.clip(_numpy.diag(projected), floor, None))
    projected /= _numpy.outer(projected_scales, projected_scales)
    result = projected * _numpy.outer(scales, scales)
    result = 0.5 * (result + result.T)
    relative_change = float(
        _numpy.linalg.norm(result - covariance, ord="fro")
        / max(_numpy.linalg.norm(covariance, ord="fro"), _numpy.finfo(float).tiny)
    )
    return result, adjusted_count, relative_change


def overlapping_batch_means_covariance(values, batch_size: int) -> _numpy.ndarray:
    """Estimate the covariance of the sample mean using overlapping batches.

    This is an independent, asymptotically equivalent cross-check for the
    Bartlett standard errors; it is not used to weight the production fits.
    """
    matrix = _as_history_matrix(values)
    n_configurations = matrix.shape[0]
    batch_size = int(batch_size)
    if not 2 <= batch_size < n_configurations:
        raise ValueError("batch_size must satisfy 2 <= batch_size < N")
    cumulative = _numpy.vstack([
        _numpy.zeros((1, matrix.shape[1]), dtype=float),
        _numpy.cumsum(matrix, axis=0),
    ])
    batch_means = (
        cumulative[batch_size:] - cumulative[:-batch_size]
    ) / batch_size
    centered_batches = batch_means - matrix.mean(axis=0, keepdims=True)
    n_batches = len(batch_means)
    long_run = (
        n_configurations * batch_size
        / ((n_configurations - batch_size) * n_batches)
        * (centered_batches.T @ centered_batches)
    )
    return 0.5 * (long_run + long_run.T) / n_configurations


def _overlapping_batch_means_variance(values, batch_size: int) -> _numpy.ndarray:
    """Diagonal-only overlapping-batch-means covariance of the sample mean."""
    matrix = _as_history_matrix(values)
    n_configurations = matrix.shape[0]
    if not 2 <= batch_size < n_configurations:
        raise ValueError("batch_size must satisfy 2 <= batch_size < N")
    cumulative = _numpy.vstack([
        _numpy.zeros((1, matrix.shape[1]), dtype=float),
        _numpy.cumsum(matrix, axis=0),
    ])
    batch_means = (
        cumulative[batch_size:] - cumulative[:-batch_size]
    ) / batch_size
    centered_batches = batch_means - matrix.mean(axis=0, keepdims=True)
    n_batches = len(batch_means)
    return (
        batch_size
        / ((n_configurations - batch_size) * n_batches)
        * _numpy.einsum("ij,ij->j", centered_batches, centered_batches)
    )


def bartlett_spectral_covariance(
    values,
    *,
    bandwidth: int | None = None,
    bandwidth_method: str = "self-consistent",
    window_factor: float = 5.0,
    max_fraction: float = 0.25,
    lugsail_r: float = 1.0,
    lugsail_c: float = 0.5,
    psd_adjust: bool = True,
) -> tuple[_numpy.ndarray, _numpy.ndarray, BartlettDiagnostics]:
    """Return means and their multivariate (lugsail-)Bartlett covariance.

    The lag-covariance convention is the biased ``1/N`` convention used in
    Bartlett's lag-window estimator.  It gives the finite-sample PSD form;
    replacing it by ``1/(N-k)`` destroys that guarantee.
    """
    matrix = _as_history_matrix(values)
    n_configurations, n_observables = matrix.shape
    automatic, tau = automatic_bartlett_bandwidth(
        matrix,
        window_factor=window_factor,
        max_fraction=max_fraction,
    )
    bandwidth_method = bandwidth_method.lower().replace("_", "-")
    if bandwidth_method not in ("self-consistent", "plugin"):
        raise ValueError("bandwidth_method must be 'self-consistent' or 'plugin'")
    plugin_metadata = {
        "plugin_unbounded_bandwidth": float("nan"),
        "plugin_max_ar_order": 0,
        "plugin_median_ar_order": float("nan"),
        "plugin_standardized_marginals": True,
    }
    if bandwidth_method == "plugin":
        selected_bandwidth, plugin_metadata = plugin_bartlett_bandwidth(
            matrix, max_fraction=max_fraction
        )
    else:
        selected_bandwidth = automatic
    if bandwidth is None:
        bandwidth = selected_bandwidth
    else:
        bandwidth_method = "fixed"
    bandwidth = int(bandwidth)
    maximum_lag = max(1, min(n_configurations - 2, int(n_configurations * max_fraction)))
    if not 1 <= bandwidth <= maximum_lag:
        raise ValueError(
            f"bandwidth must satisfy 1 <= bandwidth <= {maximum_lag} for N={n_configurations}"
        )
    lugsail_r = float(lugsail_r)
    lugsail_c = float(lugsail_c)
    if lugsail_r < 1.0:
        raise ValueError("lugsail_r must be at least one")
    if not 0.0 <= lugsail_c < 1.0:
        raise ValueError("lugsail_c must lie in [0, 1)")

    means = matrix.mean(axis=0)
    centered = matrix - means
    long_run, vanilla_long_run, inner_bandwidth = _lugsail_long_run_covariance(
        centered, bandwidth, r=lugsail_r, c=lugsail_c,
    )
    estimator = "bartlett" if lugsail_r == 1.0 or lugsail_c == 0.0 else "lugsail-bartlett"
    fallback_reason = ""
    psd_adjusted = False
    psd_adjusted_eigenvalues = 0
    psd_relative_change = 0.0
    raw_covariance = long_run / n_configurations
    raw_eigenvalues = _numpy.linalg.eigvalsh(raw_covariance)
    if _numpy.any(_numpy.diag(raw_covariance) <= 0.0):
        long_run = vanilla_long_run
        fallback_reason = "nonpositive lugsail diagonal; reverted to vanilla Bartlett"
        estimator = "bartlett-fallback"
    elif raw_eigenvalues[0] < -max(float(raw_eigenvalues[-1]), 1.0) * 1e-12:
        if not psd_adjust:
            raise ValueError(
                "lugsail covariance is indefinite; enable psd_adjust or use vanilla Bartlett"
            )
        adjusted, psd_adjusted_eigenvalues, psd_relative_change = (
            _project_covariance_to_psd(raw_covariance)
        )
        long_run = adjusted * n_configurations
        psd_adjusted = True
    covariance_of_means = long_run / n_configurations

    if not fallback_reason and not psd_adjusted:
        eigenvalues = raw_eigenvalues
    else:
        eigenvalues = _numpy.linalg.eigvalsh(covariance_of_means)
    scale = max(float(eigenvalues[-1]), _numpy.finfo(float).tiny)
    rank = int(_numpy.count_nonzero(eigenvalues > scale * 1e-12))

    obm_batch_size = bandwidth + 1
    obm_variance = _overlapping_batch_means_variance(matrix, obm_batch_size)
    bartlett_se = _numpy.sqrt(_numpy.clip(_numpy.diag(covariance_of_means), 0.0, None))
    vanilla_se = _numpy.sqrt(_numpy.clip(
        _numpy.diag(vanilla_long_run) / n_configurations, 0.0, None
    ))
    vanilla_valid = (bartlett_se > 0.0) & (vanilla_se > 0.0)
    vanilla_ratios = vanilla_se[vanilla_valid] / bartlett_se[vanilla_valid]
    obm_se = _numpy.sqrt(_numpy.clip(obm_variance, 0.0, None))
    valid = (bartlett_se > 0.0) & (obm_se > 0.0)
    ratios = obm_se[valid] / bartlett_se[valid]
    median_ratio = float(_numpy.median(ratios)) if len(ratios) else float("nan")
    max_log_ratio = (
        float(_numpy.max(_numpy.abs(_numpy.log(ratios))))
        if len(ratios) else float("nan")
    )
    lower_bandwidth = max(1, bandwidth // 2)
    upper_bandwidth = min(maximum_lag, max(bandwidth + 1, 2 * bandwidth))
    lower_se = _numpy.sqrt(_numpy.clip(
        _lugsail_long_run_variance(
            centered, lower_bandwidth, r=lugsail_r, c=lugsail_c,
        )
        / n_configurations,
        0.0, None,
    ))
    upper_se = _numpy.sqrt(_numpy.clip(
        _lugsail_long_run_variance(
            centered, upper_bandwidth, r=lugsail_r, c=lugsail_c,
        )
        / n_configurations,
        0.0, None,
    ))
    scan_valid = (bartlett_se > 0.0) & (lower_se > 0.0) & (upper_se > 0.0)
    lower_ratios = lower_se[scan_valid] / bartlett_se[scan_valid]
    upper_ratios = upper_se[scan_valid] / bartlett_se[scan_valid]
    scan_logs = _numpy.concatenate([
        _numpy.abs(_numpy.log(lower_ratios)),
        _numpy.abs(_numpy.log(upper_ratios)),
    ])
    nonconstant = tau[_numpy.var(matrix, axis=0) > 0.0]
    diagnostics = BartlettDiagnostics(
        n_configurations=n_configurations,
        n_observables=n_observables,
        estimator=estimator,
        bandwidth_method=bandwidth_method,
        bandwidth=bandwidth,
        bandwidth_fraction=bandwidth / n_configurations,
        plugin_unbounded_bandwidth=float(plugin_metadata["plugin_unbounded_bandwidth"]),
        plugin_max_ar_order=int(plugin_metadata["plugin_max_ar_order"]),
        plugin_median_ar_order=float(plugin_metadata["plugin_median_ar_order"]),
        plugin_standardized_marginals=bool(
            plugin_metadata["plugin_standardized_marginals"]
        ),
        window_factor=float(window_factor),
        max_bandwidth_fraction=float(max_fraction),
        tau_int_median=float(_numpy.median(nonconstant)) if len(nonconstant) else 0.5,
        tau_int_max=float(_numpy.max(nonconstant)) if len(nonconstant) else 0.5,
        min_eigenvalue=float(eigenvalues[0]),
        max_eigenvalue=float(eigenvalues[-1]),
        numerical_rank=rank,
        lugsail_r=lugsail_r,
        lugsail_c=lugsail_c,
        lugsail_inner_bandwidth=inner_bandwidth,
        raw_min_eigenvalue=float(raw_eigenvalues[0]),
        psd_adjusted=psd_adjusted,
        psd_adjusted_eigenvalues=psd_adjusted_eigenvalues,
        psd_relative_frobenius_change=psd_relative_change,
        fallback_reason=fallback_reason,
        vanilla_median_se_ratio=(
            float(_numpy.median(vanilla_ratios)) if len(vanilla_ratios) else float("nan")
        ),
        obm_batch_size=obm_batch_size,
        obm_median_se_ratio=median_ratio,
        obm_max_abs_log_se_ratio=max_log_ratio,
        scan_lower_bandwidth=lower_bandwidth,
        scan_upper_bandwidth=upper_bandwidth,
        scan_lower_median_se_ratio=(
            float(_numpy.median(lower_ratios)) if len(lower_ratios) else float("nan")
        ),
        scan_upper_median_se_ratio=(
            float(_numpy.median(upper_ratios)) if len(upper_ratios) else float("nan")
        ),
        scan_max_abs_log_se_ratio=(
            float(_numpy.max(scan_logs)) if len(scan_logs) else float("nan")
        ),
    )
    return means, covariance_of_means, diagnostics


def bartlett_spectral_average(
    data: dict,
    process=None,
    *,
    bandwidth: int | None = None,
    bandwidth_method: str = "self-consistent",
    window_factor: float = 5.0,
    max_fraction: float = 0.25,
    lugsail_r: float = 1.0,
    lugsail_c: float = 0.5,
    psd_adjust: bool = True,
    return_diagnostics: bool = False,
):
    """Average a complete ensemble using one joint Bartlett covariance.

    ``process`` is accepted for ``ProcessHooks.average`` compatibility and is
    deliberately ignored.  Pre-binning before a spectral estimate discards
    short-lag information.  All keys are estimated jointly, preserving
    operator and flow-time cross-covariances.
    """
    if not data:
        return ({}, None) if return_diagnostics else {}
    keys = sorted(data)
    lengths = {len(data[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError(
            "Joint Bartlett estimation requires equal configuration counts for all observables"
        )
    matrix = _numpy.column_stack([
        _numpy.asarray(data[key], dtype=float) for key in keys
    ])
    means, covariance, diagnostics = bartlett_spectral_covariance(
        matrix,
        bandwidth=bandwidth,
        bandwidth_method=bandwidth_method,
        window_factor=window_factor,
        max_fraction=max_fraction,
        lugsail_r=lugsail_r,
        lugsail_c=lugsail_c,
        psd_adjust=psd_adjust,
    )
    estimates = _gvar.gvar(means, covariance)
    result = {key: estimates[index] for index, key in enumerate(keys)}
    if return_diagnostics:
        return result, diagnostics
    return result


def lugsail_bartlett_spectral_covariance(values, **kwargs):
    """Zero-lugsail Bartlett covariance with the modern plug-in bandwidth."""
    kwargs.setdefault("bandwidth_method", "plugin")
    kwargs.setdefault("lugsail_r", 2.0)
    kwargs.setdefault("lugsail_c", 0.5)
    kwargs.setdefault("psd_adjust", True)
    return bartlett_spectral_covariance(values, **kwargs)


def lugsail_bartlett_spectral_average(data: dict, process=None, **kwargs):
    """Dictionary adapter for :func:`lugsail_bartlett_spectral_covariance`."""
    kwargs.setdefault("bandwidth_method", "plugin")
    kwargs.setdefault("lugsail_r", 2.0)
    kwargs.setdefault("lugsail_c", 0.5)
    kwargs.setdefault("psd_adjust", True)
    return bartlett_spectral_average(data, process=process, **kwargs)


def original_bartlett_spectral_covariance(values, **kwargs):
    """Reproduce the pre-lugsail Bartlett estimator for comparisons.

    This named preset uses the vanilla Bartlett window and the former common
    self-consistent bandwidth selector.  Explicit keyword arguments may still
    override the bandwidth and safety bounds, but not the defining r=1 choice.
    """
    kwargs.setdefault("bandwidth_method", "self-consistent")
    kwargs["lugsail_r"] = 1.0
    kwargs["lugsail_c"] = 0.5
    return bartlett_spectral_covariance(values, **kwargs)


def original_bartlett_spectral_average(data: dict, process=None, **kwargs):
    """Dictionary adapter for the original-Bartlett comparison preset."""
    kwargs.setdefault("bandwidth_method", "self-consistent")
    kwargs["lugsail_r"] = 1.0
    kwargs["lugsail_c"] = 0.5
    return bartlett_spectral_average(data, process=process, **kwargs)
