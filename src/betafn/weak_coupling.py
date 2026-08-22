"""Weak-coupling continuations used by the NF4 analysis notebooks.

The integral-matching routine follows Eqs. (8)-(9) and Fig. 11 of
arXiv:2303.00704.  The direct routine is deliberately labelled a diagnostic:
it extrapolates the already-fitted intermediate curves outside their measured
coupling domains before taking the continuum limit.
"""
from __future__ import annotations

import numpy as np
import gvar as gv
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


def _shape_preserving_interpolant(x_data, y_data):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    order = np.argsort(x_data)
    x_data, y_data = x_data[order], y_data[order]

    return PchipInterpolator(x_data, y_data, extrapolate=False)


def _integral_of_inverse(beta, lower, upper):
    value, _ = quad(lambda x: 1.0 / beta(x), lower, upper,
                    epsabs=1e-11, epsrel=1e-10, limit=250)
    return value


def figure11_integral_match(
    g2,
    beta_over_g4,
    pt_over_g4,
    match_window=(1.4, 1.8),
    coefficient_bound=1.0,
):
    """Match a one-parameter 4-loop-like curve over a short coupling range.

    ``beta_over_g4`` may contain correlated gvars.  As in Fig. 11 of the
    paper, three deterministic matches are returned: the central curve and
    matches to the pointwise central values shifted by +/- one standard
    deviation.  No points outside ``match_window`` determine the coefficient.
    """
    g2 = np.asarray(g2, dtype=float)
    ratio = np.asarray(beta_over_g4, dtype=object)
    order = np.argsort(g2)
    g2, ratio = g2[order], ratio[order]
    lower, upper = map(float, match_window)
    if not lower < upper:
        raise ValueError("match_window must satisfy lower < upper")
    if g2[0] > lower or g2[-1] < upper:
        raise ValueError(
            f"continuum range [{g2[0]:g}, {g2[-1]:g}] does not cover "
            f"matching range [{lower:g}, {upper:g}]"
        )

    means, sdevs = gv.mean(ratio), gv.sdev(ratio)

    # Equation (8), with x = g_GF^2:
    # beta_4(x) = beta_PT^(3)(x) - b_p x^5.
    def matched_beta(x, b_p):
        return x * x * pt_over_g4(x, 3) - b_p * x**5

    def determine_coefficient(ratio_values):
        ratio_interp = _shape_preserving_interpolant(g2, ratio_values)
        numerical_integral = _integral_of_inverse(
            lambda x: x * x * ratio_interp(x), lower, upper
        )

        def residual(parameter):
            try:
                perturbative_integral = _integral_of_inverse(
                    lambda x: matched_beta(x, parameter[0]), lower, upper
                )
                return [(perturbative_integral - numerical_integral)
                        / max(abs(numerical_integral), 1e-14)]
            except (ZeroDivisionError, ValueError):
                return [1e12]

        result = least_squares(
            residual,
            x0=np.asarray([0.0]),
            bounds=(-abs(coefficient_bound), abs(coefficient_bound)),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=4000,
        )
        if not result.success or abs(result.x[0]) >= 0.999 * coefficient_bound:
            raise RuntimeError(
                "Fig. 11 integral match did not find an interior solution; "
                "inspect the matching range and continuum curve"
            )
        return float(result.x[0]), float(numerical_integral)

    coefficients = {}
    integrals = {}
    for label, values in (
        ("central", means),
        ("plus_sigma", means + sdevs),
        ("minus_sigma", means - sdevs),
    ):
        coefficients[label], integrals[label] = determine_coefficient(values)

    def ratio_curve(x, label="central"):
        x = np.asarray(x, dtype=float)
        return pt_over_g4(x, 3) - coefficients[label] * x**3

    return {
        "match_window": (lower, upper),
        "coefficients": coefficients,
        "inverse_beta_integrals": integrals,
        "ratio_curve": ratio_curve,
        "continuum_g2": g2,
        "continuum_ratio": ratio,
    }


def continuum_from_extended_interpolants(
    bf,
    flow,
    observable,
    window,
    g2_values,
    tau0=0.0,
    cov_mode="kernel",
    kernel="rbf",
    enforce_domains=False,
):
    """Continue intermediate fits to weak coupling, then take the continuum.

    By default this intentionally does not impose each intermediate fit's
    measured coupling domain. Results below that domain are therefore
    model-dependent extrapolations and must be presented as a diagnostic.
    Set ``enforce_domains=True`` to reproduce the ordinary domain restriction
    while evaluating a custom coupling grid.
    The returned quantity is beta/g^4, whose limit at g^2=0 is finite.
    """
    lower_t, upper_t = map(float, window)
    available_times = sorted(
        (
            time for time in bf.ntrp_fits[flow][observable]
            if float(time) - tau0 > 0.0
            and lower_t <= float(time) - tau0 <= upper_t
        ),
        key=float,
    )
    if len(available_times) < 2:
        raise ValueError(f"fewer than two flow times in window {window}")
    fit_fcn = bf._cnt_fcn
    p0 = {"beta": [0.0], "slope": [0.0]}
    continuum = []
    quality = []
    times_by_g2 = {}

    for requested_g2 in np.asarray(g2_values, dtype=float):
        # At exactly zero both numerator and denominator vanish.  The small
        # positive evaluation point gives the analytic beta/g^4 limit of the
        # PT-constrained interpolation without a 0/0 operation.
        evaluation_g2 = max(float(requested_g2), 1e-7)
        times = [
            time for time in available_times
            if not enforce_domains
            or (
                bf.ntrp_nf[flow][observable][time][0]
                <= evaluation_g2
                <= bf.ntrp_nf[flow][observable][time][-1]
            )
        ]
        if len(times) < 2:
            raise ValueError(
                f"fewer than two domain-supported flow times at "
                f"g^2={requested_g2:g} in window {window}"
            )
        times_by_g2[float(requested_g2)] = tuple(times)
        measured_times = np.asarray([float(time) for time in times])
        nominal_times = measured_times - tau0
        x_data = 1.0 / nominal_times
        jacobian = nominal_times / measured_times
        log_times = np.log(nominal_times)
        distances = [np.subtract.outer(log_times, log_times)]
        values = np.asarray([
            factor
            * bf.interpolation.model.evaluate(
                evaluation_g2, bf.ntrp_fits[flow][observable][time]
            )
            / evaluation_g2**2
            for factor, time in zip(jacobian, times)
        ], dtype=object)
        weight_cov = bf._weight_covariance(
            values,
            cov_mode=cov_mode,
            kernel=kernel,
            distances=distances,
        )
        params, fit = bf._correlated_weighted_fit(
            x_data,
            values,
            fit_fcn,
            weight_cov,
            prior=None,
            p0=p0,
        )
        continuum.append(params["beta"][0])
        quality.append({
            "g2": float(requested_g2),
            "chi2": float(fit.chi2),
            "dof": int(fit.dof),
            "Q": float(fit.Q),
        })

    return {
        "g2": np.asarray(g2_values, dtype=float),
        "beta_over_g4": np.asarray(continuum, dtype=object),
        "quality": quality,
        "times": tuple(available_times),
        "times_by_g2": times_by_g2,
        "enforce_domains": bool(enforce_domains),
        "measured_domains": {
            time: tuple(map(float, bf.ntrp_nf[flow][observable][time]))
            for time in available_times
        },
    }
