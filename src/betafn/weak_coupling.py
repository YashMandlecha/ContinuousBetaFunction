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


def lambda_parameter_from_matched_beta(
    g2,
    beta_over_g4,
    matched_result,
    perturbative_beta_function,
    reference_g2=None,
):
    """Evaluate Eq. (2) of Phys. Rev. D 108, 014502.

    Below the lower edge of the Figure-11 matching window this uses the
    matched four-loop-like curve.  Above that edge it uses a shape-preserving
    interpolation of the domain-supported continuum beta function.  The
    central and pointwise +/-1 sigma curves are evaluated separately, exactly
    as in the matching diagnostic.

    The returned GF and MSbar quantities are ``Lambda / mu_ref``.  With the
    gradient-flow convention ``mu_ref = 1/sqrt(8 t_ref)``, these are
    ``sqrt(8 t_ref) Lambda``.  A result can therefore be called a t0 result
    only when ``reference_g2`` equals the coupling defining t0.
    """
    g2 = np.asarray(g2, dtype=float)
    ratio = np.asarray(beta_over_g4, dtype=object)
    if len(g2) != len(ratio) or len(g2) < 2:
        raise ValueError("g2 and beta_over_g4 must have the same nontrivial length")
    order = np.argsort(g2)
    g2, ratio = g2[order], ratio[order]
    if np.any(np.diff(g2) <= 0.0):
        raise ValueError("g2 values must be distinct")

    switch_g2 = float(matched_result["match_window"][0])
    if reference_g2 is None:
        reference_g2 = float(g2[-1])
    reference_g2 = float(reference_g2)
    endpoint_tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(g2[-1]))
    if reference_g2 > g2[-1] and reference_g2 <= g2[-1] + endpoint_tolerance:
        reference_g2 = float(g2[-1])
    if not switch_g2 < reference_g2 <= g2[-1]:
        raise ValueError(
            f"reference_g2={reference_g2:g} must lie in "
            f"({switch_g2:g}, {g2[-1]:g}]"
        )

    match_g2 = np.asarray(matched_result["continuum_g2"], dtype=float)
    match_ratio = np.asarray(matched_result["continuum_ratio"], dtype=object)
    match_order = np.argsort(match_g2)
    match_g2, match_ratio = match_g2[match_order], match_ratio[match_order]
    if match_g2[0] > switch_g2 or match_g2[-1] < switch_g2:
        raise ValueError("matched continuum does not cover the switching coupling")

    # Retain the dedicated domain-supported matching segment through its upper
    # endpoint, then append the ordinary continuum points above it.
    match_upper = float(matched_result["match_window"][1])
    low_mask = match_g2 <= match_upper
    high_mask = (g2 > match_upper) & (g2 <= reference_g2)
    combined_g2 = np.concatenate((match_g2[low_mask], g2[high_mask]))
    if combined_g2[-1] < reference_g2:
        combined_g2 = np.append(combined_g2, reference_g2)

    pt = perturbative_beta_function
    b0 = float(pt.b[0] / pt.nrm)
    b1 = float(pt.b[1] / pt.nrm**2)
    b2 = float(pt.b[2] / pt.nrm**3)
    if min(b0, b1) <= 0.0:
        raise ValueError("the Lambda construction requires asymptotic freedom")

    def shifted(values, label):
        means, sdevs = gv.mean(values), gv.sdev(values)
        if label == "plus_sigma":
            return means + sdevs
        if label == "minus_sigma":
            return means - sdevs
        return means

    lambda_gf = {}
    integrals = {}
    for label in ("central", "plus_sigma", "minus_sigma"):
        low_values = shifted(match_ratio[low_mask], label)
        high_values = shifted(ratio[high_mask], label)
        combined_ratio = np.concatenate((low_values, high_values))
        if combined_g2[-1] == reference_g2 and len(combined_ratio) < len(combined_g2):
            continuum_interp = _shape_preserving_interpolant(
                g2, shifted(ratio, label)
            )
            combined_ratio = np.append(combined_ratio, continuum_interp(reference_g2))
        ratio_interp = _shape_preserving_interpolant(combined_g2, combined_ratio)

        b_p = float(matched_result["coefficients"][label])

        # Algebraically cancel the 1/x^2 and 1/x singularities in Eq. (2)
        # for the polynomial weak-coupling curve before numerical integration.
        def weak_regular_integrand(x):
            polynomial = b0 + b1*x + b2*x*x + b_p*x**3
            return (
                -b1 * (b1 + b2*x + b_p*x*x) / (b0*b0*polynomial)
                + (b2 + b_p*x) / (b0*polynomial)
            )

        def continuum_regular_integrand(x):
            beta = x*x*float(ratio_interp(x))
            return 1.0/beta + 1.0/(b0*x*x) - b1/(b0*b0*x)

        weak_integral, _ = quad(
            weak_regular_integrand, 0.0, switch_g2,
            epsabs=1e-10, epsrel=1e-9, limit=250,
        )
        integration_points = combined_g2[
            (combined_g2 > switch_g2) & (combined_g2 < reference_g2)
        ]
        continuum_integral, _ = quad(
            continuum_regular_integrand, switch_g2, reference_g2,
            points=integration_points,
            epsabs=1e-9, epsrel=1e-8,
            limit=max(250, len(integration_points) + 50),
        )
        regular_integral = weak_integral + continuum_integral
        log_lambda_over_mu = (
            -b1/(2.0*b0*b0) * np.log(b0*reference_g2)
            -1.0/(2.0*b0*reference_g2)
            -0.5*regular_integral
        )
        lambda_gf[label] = float(np.exp(log_lambda_over_mu))
        integrals[label] = float(regular_integral)

    conversion = float(pt.lambda_msbar_over_lambda_gf)
    lambda_msbar = {key: conversion*value for key, value in lambda_gf.items()}
    return {
        "reference_g2": reference_g2,
        "switch_g2": switch_g2,
        "lambda_gf_over_mu": lambda_gf,
        "lambda_msbar_over_mu": lambda_msbar,
        "lambda_msbar_over_lambda_gf": conversion,
        "regular_integrals": integrals,
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
