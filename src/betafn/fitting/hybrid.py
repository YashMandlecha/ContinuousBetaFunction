"""Joint weak/strong interpolation and Bayesian model averaging.

The weak data are selected by *bare-coupling labels*, not by a fitted g^2
threshold.  The strong branch is a pole-free Pade-style deformation of a
Fit-4/Fit-11 weak branch.  All flow times and operators in one continuum
window are fitted jointly, so ``logGBF`` is a genuine joint Gaussian evidence
for that window rather than a product of correlated one-dimensional scores.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import gvar as _gvar
import lsqfit as _lsqfit
import numpy as _numpy

from .goodness import bruno_sommer_goodness_of_fit
from .model_average import (
    ModelAverageResult,
    model_average_predictions,
    posterior_model_weights,
    shapley_model_variance,
)


@dataclass(frozen=True)
class HybridInterpolationCandidate:
    weak_family: str
    weak_order: int
    numerator_order: int
    denominator_order: int
    match_fraction: float | None = None
    continuity: int = 1
    match_lower: float = 0.2
    match_upper: float = 0.8

    def __post_init__(self):
        if self.weak_family not in ("fit4", "fit11"):
            raise ValueError("weak_family must be 'fit4' or 'fit11'")
        if self.weak_order < 1:
            raise ValueError("weak_order must be positive")
        if self.numerator_order < 0 or self.denominator_order < 0:
            raise ValueError("Pade numerator and denominator orders must be nonnegative")
        if self.match_fraction is not None and not 0.0 < self.match_fraction < 1.0:
            raise ValueError("match_fraction must lie strictly between zero and one")
        if not 0.0 < self.match_lower < self.match_upper < 1.0:
            raise ValueError("matching bounds must satisfy 0 < lower < upper < 1")
        if self.continuity not in (1, 2):
            raise ValueError("continuity must be 1 (C1) or 2 (C2)")

    @property
    def name(self):
        fraction = (
            (
                f"fitted_{self.match_lower:.2f}_{self.match_upper:.2f}"
                .replace(".", "p")
            ) if self.match_fraction is None
            else f"{self.match_fraction:.2f}".replace(".", "p")
        )
        return (
            f"{self.weak_family}_o{self.weak_order}__pade_"
            f"{self.numerator_order}_{self.denominator_order}__"
            f"match_{fraction}__C{self.continuity}"
        )

    def metadata(self):
        return {
            "weak_family": self.weak_family,
            "weak_order": self.weak_order,
            "pade_order": f"[{self.numerator_order}/{self.denominator_order}]",
            "match_fraction": (
                "fitted" if self.match_fraction is None else self.match_fraction
            ),
            "match_lower": self.match_lower,
            "match_upper": self.match_upper,
            "continuity": f"C{self.continuity}",
        }


@dataclass(frozen=True)
class HybridTarget:
    key: tuple[str, str, str]
    couplings: tuple[str, ...]
    regions: tuple[str, ...]
    x: _numpy.ndarray
    y: _numpy.ndarray
    match_low: float
    match_high: float
    strong_scale: float


@dataclass
class HybridCandidateFit:
    candidate: HybridInterpolationCandidate
    fit: object
    targets: dict[tuple[str, str, str], HybridTarget]
    slices: dict[tuple[str, str, str], slice]
    log_evidence: float


@dataclass
class HybridFitSummary:
    """Serializable subset of an lsqfit result needed downstream."""

    p: dict
    chi2: float
    dof: int
    Q: float


@dataclass(frozen=True)
class _HybridTargetGeometry:
    key: tuple[str, str, str]
    regions: tuple[str, ...]
    match_low: float
    match_high: float
    strong_scale: float
    size: int


@dataclass(frozen=True)
class _HybridPreparedData:
    geometries: tuple[_HybridTargetGeometry, ...]
    joint_mean: _numpy.ndarray
    joint_covariance: _numpy.ndarray
    x_size: int
    perturbative_normalization: float
    perturbative_coefficients: tuple[float, ...]
    data_signature: str


@dataclass
class HybridInterpolationResult:
    window: tuple[float, float]
    flow_time_spacing: float | None
    candidate_fits: dict[str, HybridCandidateFit]
    weights: dict[str, float]


@dataclass
class HybridWindowResult:
    window: tuple[float, float]
    flow_time_spacing: float | None
    candidate_fits: dict[str, HybridCandidateFit]
    weights: dict[str, float]
    g2: _numpy.ndarray
    continuum_by_model: dict[str, dict[str, _numpy.ndarray]]
    continuum_parameters: dict[str, dict[str, list[dict | None]]]
    continuum_quality: dict[str, dict[str, list[dict[str, float]]]]
    averages: dict[str, ModelAverageResult]
    model_variance_by_choice: dict[str, dict[str, _numpy.ndarray]]


def default_hybrid_candidates(
    weak_orders=(4, 5),
    pade_orders=((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)),
    match_fractions=(None,),
    weak_families=("fit4", "fit11"),
    continuities=(1,),
    weak_orders_by_family=None,
    match_bounds=(0.2, 0.8),
):
    """Build a selectable, deliberately low-dimensional model ensemble.

    ``weak_orders_by_family`` may map each selected family to its own order
    list, for example ``{'fit4': (4, 5), 'fit11': (4,)}``.  When omitted, the
    common ``weak_orders`` list is used for every item in ``weak_families``.
    """
    def order_tuple(value):
        if isinstance(value, (int, _numpy.integer)):
            value = (value,)
        return tuple(int(order) for order in value)

    if weak_orders_by_family is None:
        family_orders = {
            str(family): order_tuple(weak_orders)
            for family in weak_families
        }
    else:
        family_orders = {
            str(family): order_tuple(orders)
            for family, orders in weak_orders_by_family.items()
        }
    if not family_orders or any(not orders for orders in family_orders.values()):
        raise ValueError("select at least one weak family and one order per family")
    candidates = tuple(
        HybridInterpolationCandidate(
            family, order, numerator, denominator, fraction, continuity,
            float(match_bounds[0]), float(match_bounds[1]),
        )
        for family, orders in family_orders.items()
        for order in orders
        for numerator, denominator in pade_orders
        for fraction in match_fractions
        for continuity in continuities
    )
    if not candidates:
        raise ValueError("the hybrid model selection produced no candidates")
    return candidates


def _iv_value(beta_function, coupling, quantity, flow, operator, flow_time):
    params = beta_function.iv_fits[coupling][quantity][flow][operator][flow_time]
    return beta_function.infinite_volume.model.evaluate(0.0, params)


def collect_hybrid_targets(
    beta_function,
    window,
    flow,
    observables,
    weak_couplings,
    strong_couplings,
    flow_time_spacing=None,
):
    """Collect aligned IV-limit x/y data with explicit bare-coupling labels."""
    weak = tuple(c for c in weak_couplings if c in beta_function.iv_fits)
    strong = tuple(c for c in strong_couplings if c in beta_function.iv_fits)
    if "8p00" not in weak or "7p50" not in strong:
        raise ValueError("hybrid matching requires beta_b=8.00 and 7.50 IV data")
    requested = weak + strong
    targets = {}
    for operator in observables:
        common_times = None
        for coupling in requested:
            by_time = beta_function.iv_fits[coupling]["g2"][flow][operator]
            times = set(by_time)
            common_times = times if common_times is None else common_times & times
        selected_times = sorted(common_times or (), key=float)
        selected_times = [
            flow_time for flow_time in selected_times
            if window[0] <= float(flow_time) <= window[1]
        ]
        if flow_time_spacing is not None:
            spacing = float(flow_time_spacing)
            if spacing <= 0.0:
                raise ValueError("flow_time_spacing must be positive")
            if not selected_times:
                raise ValueError(f"no common flow times for {operator} in {window}")
            targets_grid = _numpy.arange(
                float(window[0]), float(window[1]) + 0.5 * spacing, spacing
            )
            if len(targets_grid) == 0 or not _numpy.isclose(targets_grid[-1], window[1]):
                targets_grid = _numpy.append(targets_grid, float(window[1]))
            selected_times = sorted(set(
                min(selected_times, key=lambda time: abs(float(time) - target))
                for target in targets_grid
            ), key=float)
            if len(selected_times) < 3:
                raise ValueError(
                    f"thinning at spacing {spacing:g} leaves fewer than three "
                    f"flow times for {operator} in {window}"
                )
        for flow_time in selected_times:
            value = float(flow_time)
            x = _numpy.asarray([
                _iv_value(beta_function, coupling, "g2", flow, operator, flow_time)
                for coupling in requested
            ], dtype=object)
            y = _numpy.asarray([
                _iv_value(beta_function, coupling, "beta", flow, operator, flow_time)
                for coupling in requested
            ], dtype=object)
            mean_x = _gvar.mean(x)
            coupling_index = {coupling: index for index, coupling in enumerate(requested)}
            match_low = float(mean_x[coupling_index["8p00"]])
            match_high = float(mean_x[coupling_index["7p50"]])
            if not match_low < match_high:
                raise ValueError(
                    f"no g^2 gap between beta_b=8 and 7.5 for {operator}, t={flow_time}"
                )
            strong_max = float(max(mean_x[coupling_index[c]] for c in strong))
            targets[(flow, operator, flow_time)] = HybridTarget(
                key=(flow, operator, flow_time),
                couplings=requested,
                regions=tuple("weak" for _ in weak) + tuple("strong" for _ in strong),
                x=x,
                y=y,
                match_low=match_low,
                match_high=match_high,
                strong_scale=max(strong_max - match_low, match_high - match_low),
            )
    if not targets:
        raise ValueError(f"no common hybrid targets in flow-time window {window}")
    return targets


def _parameter_key(prefix, target_index, coefficient):
    return f"{prefix}_{target_index}_{coefficient}"


def _candidate_priors(candidate, target_count, weak_width, constant_width, strong_width, pole_log_width):
    prior, p0 = {}, {}
    if candidate.match_fraction is None:
        # Phi(N(0,1)) is exactly Uniform(0,1); the affine transformation in
        # _match_fraction therefore gives a proper Uniform(lower, upper)
        # prior shared by every flow time and operator.
        prior["match_latent"], p0["match_latent"] = [
            _gvar.gvar(0.0, 1.0)
        ], [0.0]
    for target_index in range(target_count):
        for order in range(1, candidate.weak_order + 1):
            key = _parameter_key("c", target_index, order)
            prior[key], p0[key] = [_gvar.gvar(0.0, weak_width)], [0.0]
        if candidate.weak_family == "fit11":
            key = _parameter_key("a0", target_index, 0)
            prior[key], p0[key] = [_gvar.gvar(0.0, constant_width)], [0.0]
        for order in range(candidate.numerator_order + 1):
            key = _parameter_key("a", target_index, order)
            prior[key], p0[key] = [_gvar.gvar(0.0, strong_width)], [0.0]
        for pole_index in range(candidate.denominator_order):
            prefix = "logrho" if pole_index == 0 else "logrho_step"
            key = _parameter_key(prefix, target_index, pole_index)
            prior[key], p0[key] = [_gvar.gvar(0.0, pole_log_width)], [0.0]
    return prior, p0


def _match_fraction(candidate, parameters):
    if candidate.match_fraction is not None:
        return float(candidate.match_fraction)
    latent = parameters["match_latent"][0]
    unit_fraction = 0.5 * (
        1.0 + _gvar.erf(latent / _numpy.sqrt(2.0))
    )
    return candidate.match_lower + (
        candidate.match_upper - candidate.match_lower
    ) * unit_fraction


def hybrid_match_fraction(candidate_fit):
    """Return the fixed or posterior global matching fraction for a fit."""
    return _match_fraction(candidate_fit.candidate, candidate_fit.fit.p)


def _weak_beta(beta_function, x, parameters, candidate, target_index):
    u = x / beta_function.perturbative_beta_function.nrm
    correction, term = 1.0, 1.0
    for order in range(1, candidate.weak_order + 1):
        term = term * u
        correction = correction + parameters[_parameter_key("c", target_index, order)][0] * term
    result = beta_function.perturbative_beta_function(x, loops=3) * correction
    if candidate.weak_family == "fit11":
        result = result + parameters[_parameter_key("a0", target_index, 0)][0]
    return result


def _hybrid_beta(beta_function, x, parameters, candidate, target, target_index, region=None):
    base = _weak_beta(beta_function, x, parameters, candidate, target_index)
    fraction = _match_fraction(candidate, parameters)
    match = target.match_low + fraction * (target.match_high - target.match_low)
    z = (x - match) / target.strong_scale
    numerator = 0.0
    term = 1.0
    for order in range(candidate.numerator_order + 1):
        numerator = numerator + parameters[_parameter_key("a", target_index, order)][0] * term
        term = term * z
    denominator = 1.0
    if candidate.denominator_order:
        # Parameterize the denominator as a product of linear factors with
        # ordered positive scales.  Its poles are therefore distinct and
        # strictly negative, outside the physical strong branch z >= 0.  The
        # ordered increments remove the pole-label permutation degeneracy
        # that would otherwise contaminate the Bayesian evidence.
        rho = _gvar.exp(
            parameters[_parameter_key("logrho", target_index, 0)][0]
        )
        denominator = denominator * (1.0 + z / rho)
        for pole_index in range(1, candidate.denominator_order):
            rho = rho + _gvar.exp(
                parameters[
                    _parameter_key("logrho_step", target_index, pole_index)
                ][0]
            )
            denominator = denominator * (1.0 + z / rho)
    power = candidate.continuity + 1
    correction = x * x * z**power * numerator / denominator
    if region is None:
        mask = _numpy.asarray(_gvar.mean(x), dtype=float) > match
    else:
        mask = _numpy.asarray([item == "strong" for item in region], dtype=bool)
    return _numpy.where(mask, base + correction, base)


def fit_hybrid_candidate(
    beta_function,
    targets: Mapping[tuple[str, str, str], HybridTarget],
    candidate: HybridInterpolationCandidate,
    weak_width=5.0,
    constant_width=0.25,
    strong_width=0.10,
    pole_log_width=0.7,
):
    """Fit one candidate jointly across targets, including correlated x errors."""
    target_items = tuple(sorted(targets.items()))
    slices, x_values, y_values = {}, [], []
    cursor = 0
    for key, target in target_items:
        next_cursor = cursor + len(target.x)
        slices[key] = slice(cursor, next_cursor)
        x_values.extend(target.x)
        y_values.extend(target.y)
        cursor = next_cursor
    x_values = _numpy.asarray(x_values, dtype=object)
    y_values = _numpy.asarray(y_values, dtype=object)
    prior, p0 = _candidate_priors(
        candidate, len(target_items), weak_width, constant_width,
        strong_width, pole_log_width,
    )
    prior["x"] = x_values

    def fit_function(parameters):
        latent_x = _numpy.asarray(parameters["x"], dtype=object)
        result = []
        for target_index, (key, target) in enumerate(target_items):
            block = latent_x[slices[key]]
            values = _hybrid_beta(
                beta_function, block, parameters, candidate, target,
                target_index, region=target.regions,
            )
            result.extend(values)
        return _numpy.asarray(result, dtype=object)

    fit = beta_function._run_fit_with_x_errors(
        y_values,
        lambda x, parameters: fit_function(parameters),
        prior,
        p0,
    )
    log_evidence = float(getattr(fit, "logGBF", _numpy.nan))
    if not _numpy.isfinite(log_evidence):
        raise RuntimeError(f"non-finite evidence for {candidate.name}")
    # The latent x posterior is not used after the interpolation fit.  Keeping
    # only the coefficient block preserves its complete marginal covariance
    # while substantially reducing memory and checkpoint size.
    posterior = {key: value for key, value in fit.p.items() if key != "x"}
    summary = HybridFitSummary(
        p=posterior,
        chi2=float(fit.chi2),
        dof=int(fit.dof),
        Q=float(fit.Q),
    )
    return HybridCandidateFit(
        candidate, summary, dict(target_items), slices, log_evidence,
    )


def _prepare_hybrid_data(beta_function, targets):
    """Freeze the common correlated data once for all candidate fits."""
    target_items = tuple(sorted(targets.items()))
    geometries, x_values, y_values = [], [], []
    for key, target in target_items:
        geometries.append(_HybridTargetGeometry(
            key=key,
            regions=tuple(target.regions),
            match_low=float(target.match_low),
            match_high=float(target.match_high),
            strong_scale=float(target.strong_scale),
            size=len(target.x),
        ))
        x_values.extend(target.x)
        y_values.extend(target.y)
    x_values = _numpy.asarray(x_values, dtype=object)
    y_values = _numpy.asarray(y_values, dtype=object)
    joint = _numpy.concatenate([x_values, y_values])
    joint_mean = _numpy.asarray(_gvar.mean(joint), dtype=float)
    joint_covariance = _numpy.asarray(_gvar.evalcov(joint), dtype=float)
    joint_covariance = 0.5 * (joint_covariance + joint_covariance.T)
    perturbative = beta_function.perturbative_beta_function
    normalization = float(perturbative.nrm)
    coefficients = tuple(float(value) for value in perturbative.b[:3])

    digest = hashlib.sha256()
    digest.update(joint_mean.tobytes())
    digest.update(joint_covariance.tobytes())
    digest.update(repr(geometries).encode())
    digest.update(repr((normalization, coefficients)).encode())
    return _HybridPreparedData(
        geometries=tuple(geometries),
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        x_size=len(x_values),
        perturbative_normalization=normalization,
        perturbative_coefficients=coefficients,
        data_signature=digest.hexdigest(),
    )


def _prepared_state(prepared):
    joint = _numpy.asarray(
        _gvar.gvar(prepared.joint_mean, prepared.joint_covariance),
        dtype=object,
    )
    return joint[:prepared.x_size], joint[prepared.x_size:]


def _perturbative_beta(x, normalization, coefficients):
    return -normalization * sum(
        coefficient * (x / normalization) ** (order + 2)
        for order, coefficient in enumerate(coefficients)
    )


def _prepared_hybrid_beta(
    x, parameters, candidate, geometry, target_index,
    normalization, perturbative_coefficients,
):
    u = x / normalization
    correction, term = 1.0, 1.0
    for order in range(1, candidate.weak_order + 1):
        term = term * u
        correction = correction + parameters[
            _parameter_key("c", target_index, order)
        ][0] * term
    result = _perturbative_beta(
        x, normalization, perturbative_coefficients,
    ) * correction
    if candidate.weak_family == "fit11":
        result = result + parameters[
            _parameter_key("a0", target_index, 0)
        ][0]

    fraction = _match_fraction(candidate, parameters)
    match = geometry.match_low + fraction * (
        geometry.match_high - geometry.match_low
    )
    z = (x - match) / geometry.strong_scale
    numerator, term = 0.0, 1.0
    for order in range(candidate.numerator_order + 1):
        numerator = numerator + parameters[
            _parameter_key("a", target_index, order)
        ][0] * term
        term = term * z
    denominator = 1.0
    if candidate.denominator_order:
        rho = _gvar.exp(parameters[
            _parameter_key("logrho", target_index, 0)
        ][0])
        denominator = denominator * (1.0 + z / rho)
        for pole_index in range(1, candidate.denominator_order):
            rho = rho + _gvar.exp(parameters[
                _parameter_key("logrho_step", target_index, pole_index)
            ][0])
            denominator = denominator * (1.0 + z / rho)
    strong_correction = (
        x * x * z ** (candidate.continuity + 1) * numerator / denominator
    )
    mask = _numpy.asarray(
        [region == "strong" for region in geometry.regions], dtype=bool,
    )
    return _numpy.where(mask, result + strong_correction, result)


def _fit_prepared_candidate(prepared, state, candidate, prior_widths):
    x_values, y_values = state
    prior, p0 = _candidate_priors(
        candidate,
        len(prepared.geometries),
        prior_widths["weak_width"],
        prior_widths["constant_width"],
        prior_widths["strong_width"],
        prior_widths["pole_log_width"],
    )
    prior["x"] = x_values
    slices, cursor = [], 0
    for geometry in prepared.geometries:
        next_cursor = cursor + geometry.size
        slices.append(slice(cursor, next_cursor))
        cursor = next_cursor

    def fit_function(parameters):
        latent_x = _numpy.asarray(parameters["x"], dtype=object)
        result = []
        for target_index, (geometry, selected) in enumerate(
            zip(prepared.geometries, slices)
        ):
            result.extend(_prepared_hybrid_beta(
                latent_x[selected], parameters, candidate, geometry,
                target_index, prepared.perturbative_normalization,
                prepared.perturbative_coefficients,
            ))
        return _numpy.asarray(result, dtype=object)

    fit = _lsqfit.nonlinear_fit(
        data=y_values,
        fcn=fit_function,
        prior=prior,
        p0=p0,
    )
    log_evidence = float(getattr(fit, "logGBF", _numpy.nan))
    if not _numpy.isfinite(log_evidence):
        raise RuntimeError(f"non-finite evidence for {candidate.name}")
    posterior = {key: value for key, value in fit.p.items() if key != "x"}
    return _gvar.dumps({
        "p": posterior,
        "chi2": float(fit.chi2),
        "dof": int(fit.dof),
        "Q": float(fit.Q),
        "log_evidence": log_evidence,
    })


_HYBRID_WORKER_PREPARED = None
_HYBRID_WORKER_STATE = None
_HYBRID_WORKER_PRIOR_WIDTHS = None
_HYBRID_THREADPOOL_LIMITER = None


def _initialize_hybrid_worker(prepared, prior_widths):
    global _HYBRID_WORKER_PREPARED, _HYBRID_WORKER_STATE
    global _HYBRID_WORKER_PRIOR_WIDTHS, _HYBRID_THREADPOOL_LIMITER
    _HYBRID_WORKER_PREPARED = prepared
    _HYBRID_WORKER_STATE = _prepared_state(prepared)
    _HYBRID_WORKER_PRIOR_WIDTHS = prior_widths
    try:
        from threadpoolctl import threadpool_limits
        _HYBRID_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _HYBRID_THREADPOOL_LIMITER = None


def _hybrid_worker_fit(candidate):
    started = time.monotonic()
    payload = _fit_prepared_candidate(
        _HYBRID_WORKER_PREPARED,
        _HYBRID_WORKER_STATE,
        candidate,
        _HYBRID_WORKER_PRIOR_WIDTHS,
    )
    return payload, time.monotonic() - started


def evaluate_hybrid_target(candidate_fit, beta_function, key, x):
    """Evaluate one target-specific posterior interpolation curve."""
    keys = tuple(sorted(candidate_fit.targets))
    target_index = keys.index(key)
    target = candidate_fit.targets[key]
    values = _hybrid_beta(
        beta_function, _numpy.asarray(x), candidate_fit.fit.p,
        candidate_fit.candidate, target, target_index,
    )
    return values


def hybrid_parameter_records(interpolation_result):
    """Return tidy coefficient, evidence, and model-probability records."""
    records = []
    for model, candidate_fit in interpolation_result.candidate_fits.items():
        candidate = candidate_fit.candidate
        target_keys = tuple(sorted(candidate_fit.targets))
        metadata = candidate.metadata()
        for parameter, values in candidate_fit.fit.p.items():
            try:
                family, target_text, coefficient_text = parameter.rsplit("_", 2)
                target_index = int(target_text)
                coefficient_index = int(coefficient_text)
            except (AttributeError, TypeError, ValueError):
                family = str(parameter)
                target_index = None
                coefficient_index = None
            target_key = (
                target_keys[target_index]
                if target_index is not None and 0 <= target_index < len(target_keys)
                else (None, None, None)
            )
            flattened = _numpy.asarray(values, dtype=object).reshape(-1)
            for component, value in enumerate(flattened):
                records.append({
                    "window": (
                        f"{interpolation_result.window[0]:g}_"
                        f"{interpolation_result.window[1]:g}"
                    ),
                    "model": model,
                    **metadata,
                    "log_evidence": float(candidate_fit.log_evidence),
                    "model_probability": float(interpolation_result.weights[model]),
                    "flow": target_key[0],
                    "operator": target_key[1],
                    "flow_time": (
                        float(target_key[2]) if target_key[2] is not None else _numpy.nan
                    ),
                    "parameter_family": family,
                    "coefficient_index": coefficient_index,
                    "component": component,
                    "parameter": str(parameter),
                    "mean": float(_gvar.mean(value)),
                    "sdev": float(_gvar.sdev(value)),
                })
        if candidate.match_fraction is None:
            fraction = hybrid_match_fraction(candidate_fit)
            records.append({
                "window": (
                    f"{interpolation_result.window[0]:g}_"
                    f"{interpolation_result.window[1]:g}"
                ),
                "model": model,
                **metadata,
                "log_evidence": float(candidate_fit.log_evidence),
                "model_probability": float(interpolation_result.weights[model]),
                "flow": None,
                "operator": None,
                "flow_time": _numpy.nan,
                "parameter_family": "match_fraction",
                "coefficient_index": None,
                "component": 0,
                "parameter": "match_fraction",
                "mean": float(_gvar.mean(fraction)),
                "sdev": float(_gvar.sdev(fraction)),
            })
    return records


def fit_hybrid_bma_candidates(
    beta_function,
    window,
    flow,
    observables,
    weak_couplings,
    strong_couplings,
    candidates=None,
    model_prior=None,
    flow_time_spacing=None,
    max_workers=1,
    checkpoint_dir=None,
    progress=True,
    **prior_widths,
):
    """Fit and weight hybrid interpolations without a continuum fit.

    Candidate models are statistically independent conditional on the shared
    input data, so they may be evaluated in separate processes without
    changing any likelihood, prior, covariance, evidence, or model weight.
    The parent process reconstructs every candidate from the same joint
    mean/covariance representation and restores its posterior with
    :func:`gvar.loads`, which preserves all within-model correlations.
    """
    candidates = default_hybrid_candidates() if candidates is None else tuple(candidates)
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    targets = collect_hybrid_targets(
        beta_function, window, flow, observables, weak_couplings,
        strong_couplings, flow_time_spacing=flow_time_spacing,
    )
    widths = {
        "weak_width": float(prior_widths.get("weak_width", 5.0)),
        "constant_width": float(prior_widths.get("constant_width", 0.25)),
        "strong_width": float(prior_widths.get("strong_width", 0.10)),
        "pole_log_width": float(prior_widths.get("pole_log_width", 0.70)),
    }
    prepared = _prepare_hybrid_data(beta_function, targets)
    target_items = tuple(sorted(targets.items()))
    slices, cursor = {}, 0
    for key, target in target_items:
        next_cursor = cursor + len(target.x)
        slices[key] = slice(cursor, next_cursor)
        cursor = next_cursor

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    def signature(candidate):
        digest = hashlib.sha256(prepared.data_signature.encode())
        digest.update(repr(candidate).encode())
        digest.update(repr(sorted(widths.items())).encode())
        return digest.hexdigest()

    def restore(candidate, payload):
        result = _gvar.loads(payload) if isinstance(payload, bytes) else payload
        summary = HybridFitSummary(
            p=result["p"], chi2=float(result["chi2"]),
            dof=int(result["dof"]), Q=float(result["Q"]),
        )
        return HybridCandidateFit(
            candidate=candidate,
            fit=summary,
            targets=dict(target_items),
            slices=dict(slices),
            log_evidence=float(result["log_evidence"]),
        )

    def checkpoint_file(candidate):
        return checkpoint_path / f"{candidate.name}.gvar" if checkpoint_path else None

    def load_checkpoint(candidate):
        path = checkpoint_file(candidate)
        if path is None or not path.is_file():
            return None
        try:
            payload = _gvar.load(str(path))
        except Exception as error:
            if progress:
                print(f"Ignoring unreadable checkpoint {path.name}: {error}", flush=True)
            return None
        if payload.get("signature") != signature(candidate):
            return None
        return restore(candidate, payload["result"])

    def save_checkpoint(candidate, payload):
        path = checkpoint_file(candidate)
        if path is None:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        _gvar.dump({
            "signature": signature(candidate),
            "result": _gvar.loads(payload),
        }, str(temporary))
        os.replace(temporary, path)

    completed = {}
    pending = []
    for candidate in candidates:
        result = load_checkpoint(candidate)
        if result is None:
            pending.append(candidate)
        else:
            completed[candidate.name] = result
    if progress and completed:
        print(
            f"Loaded {len(completed)}/{len(candidates)} valid hybrid checkpoints.",
            flush=True,
        )

    start_all = time.monotonic()
    durations = []
    worker_count = min(max_workers, max(1, len(pending)))

    def record(candidate, payload, elapsed):
        save_checkpoint(candidate, payload)
        completed[candidate.name] = restore(candidate, payload)
        durations.append(float(elapsed))
        if progress:
            done = len(completed)
            remaining = len(candidates) - done
            eta = remaining * (sum(durations) / len(durations)) / worker_count
            print(
                f"[{done}/{len(candidates)}] {candidate.name} completed "
                f"in {elapsed / 60.0:.1f} min; estimated remaining "
                f"{eta / 60.0:.1f} min",
                flush=True,
            )

    if pending and max_workers == 1:
        state = _prepared_state(prepared)
        for candidate in pending:
            started = time.monotonic()
            payload = _fit_prepared_candidate(prepared, state, candidate, widths)
            record(candidate, payload, time.monotonic() - started)
    elif pending:
        # Linux batch jobs can safely use fork here and benefit from
        # copy-on-write for the large read-only covariance. Interactive macOS
        # kernels use spawn, avoiding unsafe notebook-kernel forks.
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_hybrid_worker,
            initargs=(prepared, widths),
        ) as executor:
            submitted = {
                executor.submit(_hybrid_worker_fit, candidate): candidate
                for candidate in pending
            }
            for future in as_completed(submitted):
                candidate = submitted[future]
                payload, elapsed = future.result()
                record(candidate, payload, elapsed)

    if progress:
        print(
            f"Hybrid interpolation ensemble completed in "
            f"{(time.monotonic() - start_all) / 60.0:.1f} min.",
            flush=True,
        )
    fits = {candidate.name: completed[candidate.name] for candidate in candidates}
    weights = posterior_model_weights(
        {name: result.log_evidence for name, result in fits.items()},
        model_prior=model_prior,
    )
    return HybridInterpolationResult(
        window=tuple(map(float, window)),
        flow_time_spacing=flow_time_spacing,
        candidate_fits=fits,
        weights=weights,
    )


def _continuum_for_candidate(
    beta_function,
    candidate_fit,
    window,
    flow,
    observables,
    g2_grid,
    cov_mode,
    alpha,
    tau0,
):
    predictions, quality, parameters = {}, {}, {}
    fit_fcn = lambda x, p: p["beta"][0] + p["slope"][0] * x
    for operator in observables:
        operator_values, operator_quality, operator_parameters = [], [], []
        keys = [key for key in candidate_fit.targets if key[0] == flow and key[1] == operator]
        keys.sort(key=lambda key: float(key[2]))
        for g2 in g2_grid:
            supported = [
                key for key in keys
                if min(_gvar.mean(candidate_fit.targets[key].x)) <= g2
                <= max(_gvar.mean(candidate_fit.targets[key].x))
            ]
            if len(supported) < 3:
                operator_values.append(None)
                operator_quality.append(None)
                operator_parameters.append(None)
                continue
            measured = _numpy.asarray([float(key[2]) for key in supported])
            nominal = measured - tau0
            x_data = 1.0 / nominal
            jacobian = nominal / measured
            y = _numpy.asarray([
                factor * _numpy.asarray(
                    evaluate_hybrid_target(candidate_fit, beta_function, key, float(g2)),
                    dtype=object,
                ).item()
                for factor, key in zip(jacobian, supported)
            ], dtype=object).reshape(-1)
            distances = [_numpy.subtract.outer(_numpy.log(nominal), _numpy.log(nominal))]
            weight_covariance = beta_function._weight_covariance(
                y, cov_mode=cov_mode, alpha=alpha, distances=distances,
            )
            params, fit = beta_function._correlated_weighted_fit(
                x_data, y, fit_fcn, weight_covariance,
                prior=None, p0={"beta": [0.0], "slope": [0.0]},
            )
            operator_values.append(params["beta"][0])
            operator_parameters.append(params)
            actual_covariance = _gvar.evalcov(y)
            design = _numpy.column_stack([_numpy.ones(len(x_data)), x_data])
            residual = _gvar.mean(y) - _numpy.asarray(
                _gvar.mean(fit_fcn(x_data, fit.p)), dtype=float
            )
            generalized = bruno_sommer_goodness_of_fit(
                residual, design, actual_covariance, weight_covariance,
                model_hessian=_numpy.zeros((len(x_data), 2, 2)),
            )
            operator_quality.append({
                "chi2": float(fit.chi2), "dof": int(fit.dof),
                "p-value": float(fit.Q),
                "generalized_expectation": generalized.expectation,
                "generalized_reduced_chi2": (
                    generalized.observed / generalized.expectation
                    if generalized.expectation > 0.0 else None
                ),
                "generalized_effective_dof": generalized.effective_dof,
                "generalized_p_value": generalized.p_value,
                "generalized_expectation_parameter_sdev": (
                    generalized.expectation_parameter_sdev
                ),
            })
        predictions[operator] = operator_values
        quality[operator] = operator_quality
        parameters[operator] = operator_parameters
    return predictions, quality, parameters


def run_hybrid_bma_window(
    beta_function,
    window,
    flow,
    observables,
    weak_couplings,
    strong_couplings,
    g2_grid,
    candidates=None,
    model_prior=None,
    cov_mode="empirical",
    alpha=0.25,
    tau0=0.0,
    flow_time_spacing=None,
    prefit_candidate_fits=None,
    max_workers=1,
    checkpoint_dir=None,
    progress=True,
    **prior_widths,
):
    """Run joint hybrid fits, continuum limits, and posterior model averaging."""
    candidates = default_hybrid_candidates() if candidates is None else tuple(candidates)
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if prefit_candidate_fits is None:
        interpolation_result = fit_hybrid_bma_candidates(
            beta_function, window, flow, observables,
            weak_couplings=weak_couplings,
            strong_couplings=strong_couplings,
            candidates=candidates,
            model_prior=model_prior,
            flow_time_spacing=flow_time_spacing,
            max_workers=max_workers,
            checkpoint_dir=checkpoint_dir,
            progress=progress,
            **prior_widths,
        )
        fits = interpolation_result.candidate_fits
        weights = interpolation_result.weights
    else:
        fits = dict(prefit_candidate_fits)
        if set(fits) != {candidate.name for candidate in candidates}:
            raise ValueError("prefit_candidate_fits does not match the candidate ensemble")
        weights = posterior_model_weights(
            {name: result.log_evidence for name, result in fits.items()},
            model_prior=model_prior,
        )
    grid = _numpy.asarray(g2_grid, dtype=float)
    continuum_by_model, quality, continuum_parameters = {}, {}, {}
    continuum_started = time.monotonic()
    continuum_durations = []
    for candidate_index, (name, result) in enumerate(fits.items(), start=1):
        candidate_started = time.monotonic()
        (
            continuum_by_model[name],
            quality[name],
            continuum_parameters[name],
        ) = _continuum_for_candidate(
            beta_function, result, window, flow, observables, grid,
            cov_mode, alpha, tau0,
        )
        elapsed = time.monotonic() - candidate_started
        continuum_durations.append(elapsed)
        if progress:
            remaining = len(fits) - candidate_index
            eta = remaining * sum(continuum_durations) / len(continuum_durations)
            print(
                f"[continuum {candidate_index}/{len(fits)}] {name} completed "
                f"in {elapsed / 60.0:.1f} min; estimated remaining "
                f"{eta / 60.0:.1f} min",
                flush=True,
            )

    averages, variance_by_choice = {}, {}
    metadata = {name: result.candidate.metadata() for name, result in fits.items()}
    possible_axes = (
        "weak_family", "weak_order", "pade_order", "match_fraction", "continuity",
    )
    axes = tuple(
        axis for axis in possible_axes
        if len({values[axis] for values in metadata.values()}) > 1
    )
    for operator in observables:
        valid = _numpy.asarray([
            all(continuum_by_model[name][operator][index] is not None for name in fits)
            for index in range(len(grid))
        ], dtype=bool)
        if not _numpy.any(valid):
            continue
        model_predictions = {
            name: _numpy.asarray(continuum_by_model[name][operator], dtype=object)[valid]
            for name in fits
        }
        average = model_average_predictions(model_predictions, weights)
        averages[operator] = average
        variance_by_choice[operator] = shapley_model_variance(
            {name: _gvar.mean(values) for name, values in model_predictions.items()},
            weights, metadata, axes,
        )
        # Store the supported grid next to the result without changing the
        # ModelAverageResult's generic API.
        variance_by_choice[operator]["g2"] = grid[valid]
    if progress:
        print(
            f"Continuum candidate ensemble completed in "
            f"{(time.monotonic() - continuum_started) / 60.0:.1f} min.",
            flush=True,
        )
    return HybridWindowResult(
        window=tuple(map(float, window)), flow_time_spacing=flow_time_spacing,
        candidate_fits=fits, weights=weights,
        g2=grid, continuum_by_model=continuum_by_model,
        continuum_parameters=continuum_parameters,
        continuum_quality=quality, averages=averages,
        model_variance_by_choice=variance_by_choice,
    )
