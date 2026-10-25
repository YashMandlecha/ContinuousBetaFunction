"""Weak-coupling continuations used by the NF4 analysis notebooks.

The integral-matching routine follows Eqs. (8)-(9) and Fig. 11 of
arXiv:2303.00704.  The direct routine is deliberately labelled a diagnostic:
it extrapolates the already-fitted intermediate curves outside their measured
coupling domains before taking the continuum limit.
"""
from __future__ import annotations

import numpy as np
import gvar as gv
import lsqfit
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator
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


def constrained_smooth_match(
    g2,
    beta_over_g4,
    pt_over_g4,
    match_window=(0.8, 1.2),
    correction_powers=(3, 4),
    svdcut=1e-10,
):
    """Correlated, perturbatively constrained smooth weak-coupling match.

    The fitted quantity is ``R(x) = beta(x) / x**2`` with ``x = g_GF**2``:

    ``R(x) = R_PT,3(x) + sum(c_n * x**n)``.

    Requiring every correction power to be at least three preserves the known
    one-, two-, and three-loop coefficients exactly as ``x -> 0``.  The
    coefficients are fitted to all correlated continuum values in the
    matching interval using generalized least squares.  The returned weak
    curve is intended to be joined to the continuum curve with
    :func:`smooth_blended_ratio`, which gives a C1 transition at both ends of
    the overlap interval.
    """
    g2 = np.asarray(g2, dtype=float)
    ratio = np.asarray(beta_over_g4, dtype=object)
    if g2.ndim != 1 or ratio.ndim != 1 or len(g2) != len(ratio):
        raise ValueError("g2 and beta_over_g4 must be one-dimensional and aligned")
    if len(g2) < 2:
        raise ValueError("at least two matching points are required")
    order = np.argsort(g2)
    g2, ratio = g2[order], ratio[order]
    if np.any(np.diff(g2) <= 0.0):
        raise ValueError("g2 values must be distinct")

    lower, upper = map(float, match_window)
    if not lower < upper:
        raise ValueError("match_window must satisfy lower < upper")
    powers = tuple(int(power) for power in correction_powers)
    if not powers or len(set(powers)) != len(powers) or min(powers) < 3:
        raise ValueError("correction_powers must be distinct integers >= 3")

    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(upper))
    selected = (g2 >= lower - tolerance) & (g2 <= upper + tolerance)
    fit_x, fit_y = g2[selected], ratio[selected]
    if len(fit_x) < len(powers):
        raise ValueError(
            f"matching interval contains {len(fit_x)} points for "
            f"{len(powers)} correction coefficients"
        )
    if fit_x[0] > lower + tolerance or fit_x[-1] < upper - tolerance:
        raise ValueError("continuum data must include both matching endpoints")

    parameter_names = tuple(f"c{power}" for power in powers)

    def fit_function(x, parameters):
        x = np.asarray(x, dtype=float)
        correction = sum(
            parameters[name] * x**power
            for name, power in zip(parameter_names, powers)
        )
        return pt_over_g4(x, 3) + correction

    fit = lsqfit.nonlinear_fit(
        data=(fit_x, fit_y),
        fcn=fit_function,
        p0={name: 0.0 for name in parameter_names},
        svdcut=svdcut,
        maxit=10000,
    )
    parameters = {name: fit.p[name] for name in parameter_names}

    def weak_ratio_curve(points):
        return np.asarray(fit_function(points, parameters), dtype=object)

    audit_x = np.linspace(0.0, upper, 401)
    if np.any(np.asarray(gv.mean(weak_ratio_curve(audit_x)), dtype=float) >= 0.0):
        raise RuntimeError("smooth weak-coupling central curve is not asymptotically free")

    return {
        "match_window": (lower, upper),
        "correction_powers": powers,
        "parameters": parameters,
        "fit": fit,
        "chi2": float(fit.chi2),
        "dof": int(fit.dof),
        "Q": float(fit.Q),
        "svdcut": float(svdcut),
        "continuum_g2": fit_x,
        "continuum_ratio": fit_y,
        "weak_ratio_curve": weak_ratio_curve,
    }


def _correlated_cubic_interpolate(points, coordinates, values):
    """Natural-cubic interpolation that remains linear in correlated gvars."""
    points = np.atleast_1d(np.asarray(points, dtype=float))
    coordinates = np.asarray(coordinates, dtype=float)
    values = np.asarray(values, dtype=object)
    order = np.argsort(coordinates)
    coordinates, values = coordinates[order], values[order]
    if len(coordinates) != len(values) or len(coordinates) < 2:
        raise ValueError("interpolation coordinates and values must be aligned")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("interpolation coordinates must be distinct")
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(coordinates[-1]))
    if (np.any(points < coordinates[0] - tolerance)
            or np.any(points > coordinates[-1] + tolerance)):
        raise ValueError("requested point lies outside continuum support")
    points = np.clip(points, coordinates[0], coordinates[-1])

    if len(coordinates) == 2:
        fraction = (points - coordinates[0]) / (coordinates[1] - coordinates[0])
        return np.asarray([
            (1.0 - weight) * values[0] + weight * values[1]
            for weight in fraction
        ], dtype=object)

    # A cubic spline at fixed knots is a linear map of its knot values.  Build
    # that map from the identity, then apply it to the original gvars so their
    # full covariance is preserved exactly.
    weights = CubicSpline(
        coordinates, np.eye(len(coordinates)), axis=0,
        bc_type="natural", extrapolate=False,
    )(points)
    return np.asarray([
        sum(weight * value for weight, value in zip(row, values))
        for row in np.asarray(weights, dtype=float)
    ], dtype=object)


def smooth_blended_ratio(points, smooth_result, continuum_g2, continuum_ratio):
    """Evaluate the C1 blend of the PT-constrained and continuum curves.

    The PT-constrained curve is used below the overlap, the correlated
    continuum curve above it, and ``3s**2 - 2s**3`` blends them inside it.
    The blend has zero derivative at each endpoint, so both the value and the
    first derivative join continuously there.
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    lower, upper = smooth_result["match_window"]
    weak = np.asarray(smooth_result["weak_ratio_curve"](points), dtype=object)
    result = weak.copy()
    continuum_mask = points >= lower
    if np.any(continuum_mask):
        continuum = _correlated_cubic_interpolate(
            points[continuum_mask], continuum_g2, continuum_ratio
        )
        selected_points = points[continuum_mask]
        transition = np.clip((selected_points - lower) / (upper - lower), 0.0, 1.0)
        smoothstep = 3.0 * transition**2 - 2.0 * transition**3
        weak_selected = weak[continuum_mask]
        result[continuum_mask] = np.asarray([
            (1.0 - weight) * weak_value + weight * continuum_value
            for weight, weak_value, continuum_value in zip(
                smoothstep, weak_selected, continuum
            )
        ], dtype=object)
    if np.any(np.asarray(gv.mean(result), dtype=float) >= 0.0):
        raise RuntimeError("smooth blended central beta/g^4 curve crosses zero")
    return result


def smooth_weak_regular_integrand(points, smooth_result, perturbative_beta_function):
    """Cancellation-safe Lambda integrand for the smooth weak curve."""
    points = np.asarray(points, dtype=float)
    pt = perturbative_beta_function
    b0 = float(pt.b[0] / pt.nrm)
    b1 = float(pt.b[1] / pt.nrm**2)
    b2 = float(pt.b[2] / pt.nrm**3)
    powers = smooth_result["correction_powers"]
    parameters = smooth_result["parameters"]

    # beta = -x^2 P(x), while the fitted ratio correction has the opposite
    # sign in P.  Keeping the algebraic cancellation explicit makes x=0 safe.
    higher_terms = {
        power: -parameters[f"c{power}"] for power in powers
    }
    polynomial = b0 + b1 * points + b2 * points**2
    numerator_tail = b2 + 0.0 * points
    derivative_tail = b1 + b2 * points
    for power, coefficient in higher_terms.items():
        polynomial = polynomial + coefficient * points**power
        numerator_tail = numerator_tail + coefficient * points**(power - 2)
        derivative_tail = derivative_tail + coefficient * points**(power - 1)
    return np.asarray(
        -b1 * derivative_tail / (b0**2 * polynomial)
        + numerator_tail / (b0 * polynomial),
        dtype=object,
    )


def lambda_parameter_from_smooth_beta(
    continuum_g2,
    continuum_beta_over_g4,
    smooth_result,
    perturbative_beta_function,
    reference_g2,
    integration_points=1601,
):
    """Propagate the smooth matched curve into a correlated Lambda estimate."""
    continuum_g2 = np.asarray(continuum_g2, dtype=float)
    continuum_ratio = np.asarray(continuum_beta_over_g4, dtype=object)
    order = np.argsort(continuum_g2)
    continuum_g2, continuum_ratio = continuum_g2[order], continuum_ratio[order]
    reference_g2 = float(reference_g2)
    lower, upper = smooth_result["match_window"]
    if continuum_g2[0] > lower or continuum_g2[-1] < reference_g2:
        raise ValueError("continuum curve does not cover the blend and reference range")
    if integration_points < 101:
        raise ValueError("integration_points must be at least 101")

    pt = perturbative_beta_function
    b0 = float(pt.b[0] / pt.nrm)
    b1 = float(pt.b[1] / pt.nrm**2)

    weak_x = np.linspace(0.0, lower, max(201, integration_points // 4))
    weak_integrand = smooth_weak_regular_integrand(weak_x, smooth_result, pt)
    weak_integral = sum(
        0.5 * (weak_integrand[index] + weak_integrand[index + 1])
        * (weak_x[index + 1] - weak_x[index])
        for index in range(len(weak_x) - 1)
    )

    supported_knots = continuum_g2[
        (continuum_g2 > lower) & (continuum_g2 < reference_g2)
    ]
    continuum_x = np.unique(np.concatenate((
        np.linspace(lower, reference_g2, integration_points),
        supported_knots,
        [upper],
    )))
    ratio = smooth_blended_ratio(
        continuum_x, smooth_result, continuum_g2, continuum_ratio
    )
    beta = continuum_x**2 * ratio
    regular_integrand = np.asarray(
        1.0 / beta + 1.0 / (b0 * continuum_x**2)
        - b1 / (b0**2 * continuum_x),
        dtype=object,
    )
    continuum_integral = sum(
        0.5 * (regular_integrand[index] + regular_integrand[index + 1])
        * (continuum_x[index + 1] - continuum_x[index])
        for index in range(len(continuum_x) - 1)
    )
    total_integral = weak_integral + continuum_integral
    log_lambda_over_mu = (
        -b1 / (2.0 * b0**2) * np.log(b0 * reference_g2)
        -1.0 / (2.0 * b0 * reference_g2)
        -0.5 * total_integral
    )
    lambda_gf = gv.exp(log_lambda_over_mu)
    conversion = float(pt.lambda_msbar_over_lambda_gf)
    return {
        "reference_g2": reference_g2,
        "blend_window": (lower, upper),
        "regular_integral": total_integral,
        "weak_regular_integral": weak_integral,
        "continuum_regular_integral": continuum_integral,
        "lambda_gf_over_mu": lambda_gf,
        "lambda_msbar_over_mu": conversion * lambda_gf,
        "lambda_msbar_over_lambda_gf": conversion,
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
    cov_mode="empirical",
    kernel="rbf",
    enforce_domains=False,
):
    """Continue intermediate fits to weak coupling, then take the continuum.

    By default this intentionally does not impose each intermediate fit's
    measured coupling domain. Results below that domain are therefore
    model-dependent extrapolations and must be presented as a diagnostic.
    Set ``enforce_domains=True`` to reproduce the ordinary domain restriction
    while evaluating a custom coupling grid.
    The returned quantity is beta/g^4.  Its limit at g^2=0 is finite only
    when the interpolation contains no beta terms below order g^4; callers
    must use a strictly positive grid for models that are singular there.
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
