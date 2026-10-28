"""Weak/strong interpolation and Bayesian model averaging.

The weak data are selected by *bare-coupling labels*, not by a fitted g^2
threshold. Weak branches include the original Fit-4/Fit-11 forms and
one-loop-anchored monotone polynomials with an optional constant.
The strong branch can use the ratio-of-polynomials form in
Eq. (7) of arXiv:2303.00704, matched to the selected weak branch, or the
older additive deformation retained for reproducibility.  Analyses fit every
discretization
operator and flow time independently, exactly as in the standard
interpolation stage.  Each such interpolation therefore has its own matching
fraction, evidence, and posterior model probabilities; different flow times
are combined only by the subsequent continuum extrapolation.
"""
from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait,
)
from dataclasses import dataclass, replace
import hashlib
from math import comb
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
    strong_form: str = "deformation"

    def __post_init__(self):
        if self.weak_family not in (
            "fit4", "fit5", "fit6", "fit7", "fit8", "fit9", "fit10",
            "fit11", "monotone", "monotone_constant",
        ):
            raise ValueError(
                "unsupported weak interpolation family"
            )
        if self.weak_family.startswith("monotone") and (
            self.strong_form != "eq7_linear" or self.continuity != 1
        ):
            raise ValueError("monotone weak families require the gap-only C1 connection")
        if self.weak_order < 1:
            raise ValueError("weak_order must be positive")
        if self.weak_family == "fit8" and self.weak_order < 2:
            raise ValueError("fit8 order must be at least 2")
        if self.weak_family in ("fit9", "fit10") and self.weak_order < 3:
            raise ValueError("fit9/fit10 order must be at least 3")
        if self.numerator_order < 0 or self.denominator_order < 0:
            raise ValueError("Pade numerator and denominator orders must be nonnegative")
        if self.match_fraction is not None and not 0.0 < self.match_fraction < 1.0:
            raise ValueError("match_fraction must lie strictly between zero and one")
        if not 0.0 < self.match_lower < self.match_upper < 1.0:
            raise ValueError("matching bounds must satisfy 0 < lower < upper < 1")
        if self.continuity not in (0, 1, 2):
            raise ValueError("continuity must be 0 (C0), 1 (C1), or 2 (C2)")
        if self.strong_form not in (
            "deformation", "eq7", "eq7_blend", "eq7_hermite",
            "eq7_linear",
        ):
            raise ValueError(
                "strong_form must be 'deformation', 'eq7', 'eq7_blend', "
                "'eq7_hermite', or 'eq7_linear'"
            )
        if self.continuity == 0 and self.strong_form != "eq7_linear":
            raise ValueError("C0 is reserved for the linear fixed-gap bridge")
        if self.strong_form == "eq7":
            if self.numerator_order < self.continuity:
                raise ValueError(
                    "Eq. (7) numerator order must be at least the requested "
                    "continuity order"
                )
            if self.denominator_order != self.numerator_order + 1:
                raise ValueError(
                    "Eq. (7) requires denominator order N+1 for numerator "
                    "order N so beta(g^2) is asymptotically linear"
                )
        if self.strong_form == "eq7_blend":
            if self.numerator_order < 1:
                raise ValueError("blended Eq. (7) numerator order must be positive")
            if self.denominator_order != self.numerator_order + 1:
                raise ValueError(
                    "Eq. (7) requires denominator order N+1 for numerator "
                    "order N so beta(g^2) is asymptotically linear"
                )
            if self.match_fraction is not None:
                raise ValueError(
                    "the C2 gap blend uses the beta_b=8 and 7.5 endpoints, "
                    "not a fitted match fraction"
                )
        if self.strong_form == "eq7_hermite":
            if self.numerator_order < 1:
                raise ValueError("Hermite-bridged Eq. (7) order must be positive")
            if self.denominator_order != self.numerator_order + 1:
                raise ValueError(
                    "Eq. (7) requires denominator order N+1 for numerator "
                    "order N so beta(g^2) is asymptotically linear"
                )
            if self.match_fraction is not None:
                raise ValueError(
                    "the Hermite gap bridge uses the beta_b=8 and 7.5 "
                    "endpoints, not a fitted match fraction"
                )
        if self.strong_form == "eq7_linear":
            if self.continuity not in (0, 1, 2):
                raise ValueError(
                    "the linear fixed-gap bridge supports either the legacy "
                    "C0 join, gap-only C1 rational bridge, or legacy C2 shoulders"
                )
            if self.numerator_order < 1:
                raise ValueError("linearly bridged Eq. (7) order must be positive")
            if self.denominator_order != self.numerator_order + 1:
                raise ValueError(
                    "Eq. (7) requires denominator order N+1 for numerator "
                    "order N so beta(g^2) is asymptotically linear"
                )
            if self.match_fraction is not None:
                raise ValueError(
                    "the linear gap bridge uses the beta_b=8 and 7.5 "
                    "endpoints, not a fitted match fraction"
                )

    @property
    def name(self):
        fraction = (
            (
                f"fitted_{self.match_lower:.2f}_{self.match_upper:.2f}"
                .replace(".", "p")
            ) if self.match_fraction is None
            else f"{self.match_fraction:.2f}".replace(".", "p")
        )
        # Preserve legacy candidate/checkpoint names until the batch script is
        # deliberately migrated to the Eq. (7) branch.
        strong_name = {
            "eq7": "eq7",
            "eq7_blend": "eq7blend",
            "eq7_hermite": "eq7hermite",
            "eq7_linear": "eq7linear",
            "deformation": "pade",
        }[self.strong_form]
        if self.strong_form in ("eq7_blend", "eq7_hermite", "eq7_linear"):
            fraction = "gap_8p00_7p50"
        return (
            f"{self.weak_family}_o{self.weak_order}__{strong_name}_"
            f"{self.numerator_order}_{self.denominator_order}__"
            f"match_{fraction}__C{self.continuity}"
        )

    def metadata(self):
        return {
            "gap_connection": (
                "C1 rational Hermite; weak extension = 0.05 of gap"
                if self.strong_form == "eq7_linear" and self.continuity == 1
                else None
            ),
            "weak_family": self.weak_family,
            "weak_parameterization": (
                "integrated positive Bernstein slope; L=4*pi; w0=1"
                if self.weak_family.startswith("monotone")
                else {
                    "fit4": "multiplicative three-loop prefactor",
                    "fit5": "fixed PT3 plus additive u^3 and higher terms",
                    "fit6": "free two-loop ratio coefficient plus additive higher terms",
                    "fit7": "free three-loop ratio coefficient plus additive higher terms",
                    "fit8": "multiplicative PT3 with c1 fixed to zero",
                    "fit9": "multiplicative PT3 with c2 fixed to zero",
                    "fit10": "multiplicative PT3 with free c0 and c1=c2=0",
                    "fit11": "additive beta constant plus multiplicative PT3",
                }[self.weak_family]
            ),
            "weak_order": self.weak_order,
            "strong_form": self.strong_form,
            "pade_order": f"[{self.numerator_order}/{self.denominator_order}]",
            "match_fraction": (
                "fixed beta_b=8--7.5 gap"
                if self.strong_form in ("eq7_blend", "eq7_hermite", "eq7_linear")
                else (
                    "fitted" if self.match_fraction is None
                    else self.match_fraction
                )
            ),
            "match_lower": (
                None if self.strong_form in ("eq7_blend", "eq7_hermite", "eq7_linear")
                else self.match_lower
            ),
            "match_upper": (
                None if self.strong_form in ("eq7_blend", "eq7_hermite", "eq7_linear")
                else self.match_upper
            ),
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


@dataclass
class HybridTargetAverage:
    """BMA prediction for one independently fitted operator/flow-time target."""

    key: tuple[str, str, str]
    g2: _numpy.ndarray
    average: ModelAverageResult
    model_variance_by_choice: dict[str, _numpy.ndarray]
    weighted_prediction: _numpy.ndarray
    within_prediction: _numpy.ndarray


@dataclass
class HybridIndependentWindowResult:
    """Continuum result built from independent flow-time interpolations."""

    window: tuple[float, float]
    flow_time_spacing: float | None
    target_interpolations: dict[tuple[str, str, str], HybridInterpolationResult]
    target_averages: dict[tuple[str, str, str], HybridTargetAverage]
    target_within_covariance: dict[
        tuple[tuple[str, str, str], tuple[str, str, str]], _numpy.ndarray
    ]
    g2: _numpy.ndarray
    continuum_parameters: dict[str, list[dict | None]]
    continuum_quality: dict[str, list[dict | None]]
    continuum_points: dict[str, list[dict | None]]
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
    strong_form="deformation",
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
            strong_form=str(strong_form),
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


def _weak_parameter_specs(candidate):
    """Return ``(family, power)`` keys for one weak-branch target.

    ``weak_order`` follows the established fit-specific convention: it is the
    correction order for Fits 4/8/9/10/11 and the number of additive higher-
    order terms for Fits 5/6/7.
    """
    family, order = candidate.weak_family, candidate.weak_order
    if family.startswith("monotone"):
        return tuple(("logw", power) for power in range(1, order + 1))
    if family in ("fit4", "fit11"):
        return tuple(("c", power) for power in range(1, order + 1))
    if family == "fit5":
        return tuple(("d", power) for power in range(3, 3 + order))
    if family == "fit6":
        return (("b", 1),) + tuple(
            ("d", power) for power in range(3, 3 + order)
        )
    if family == "fit7":
        return (("b", 2),) + tuple(
            ("d", power) for power in range(3, 3 + order)
        )
    if family == "fit8":
        return tuple(("c", power) for power in range(2, order + 1))
    if family == "fit9":
        return (("c", 1),) + tuple(
            ("c", power) for power in range(3, order + 1)
        )
    if family == "fit10":
        return (("c", 0),) + tuple(
            ("c", power) for power in range(3, order + 1)
        )
    raise ValueError(f"unsupported weak family {family}")


def _candidate_priors(candidate, target_count, weak_width, constant_width, strong_width, pole_log_width):
    prior, p0 = {}, {}
    if (
        candidate.match_fraction is None
        and candidate.strong_form not in (
            "eq7_blend", "eq7_hermite", "eq7_linear",
        )
    ):
        # Phi(N(0,1)) is exactly Uniform(0,1); the affine transformation in
        # _match_fraction therefore gives a proper Uniform(lower, upper)
        # prior.  Production calls pass exactly one operator/flow-time target,
        # so this matching fraction belongs only to that interpolation.
        prior["match_latent"], p0["match_latent"] = [
            _gvar.gvar(0.0, 1.0)
        ], [0.0]
    for target_index in range(target_count):
        for prefix, power in _weak_parameter_specs(candidate):
            key = _parameter_key(prefix, target_index, power)
            prior[key], p0[key] = [_gvar.gvar(0.0, weak_width)], [0.0]
        if candidate.weak_family in ("fit11", "monotone_constant"):
            key = _parameter_key("a0", target_index, 0)
            prior[key], p0[key] = [_gvar.gvar(0.0, constant_width)], [0.0]
        if candidate.strong_form in (
            "eq7_blend", "eq7_hermite", "eq7_linear",
        ):
            key = _parameter_key("logp0", target_index, 0)
            prior[key], p0[key] = [
                _gvar.gvar(_numpy.log(0.05277), 1.0)
            ], [_numpy.log(0.05277)]
            for order in range(1, candidate.numerator_order + 1):
                key = _parameter_key("p", target_index, order)
                prior[key], p0[key] = [
                    _gvar.gvar(0.0, strong_width)
                ], [0.0]
        elif candidate.strong_form == "eq7":
            # In the matched Eq. (7) representation, p_1 (and p_2 for C2)
            # are fixed analytically by continuity at the matching point.
            # Only the remaining numerator coefficients are sampled.
            for order in range(
                candidate.continuity + 1, candidate.numerator_order + 1
            ):
                key = _parameter_key("p", target_index, order)
                prior[key], p0[key] = [
                    _gvar.gvar(0.0, strong_width)
                ], [0.0]
        else:
            for order in range(candidate.numerator_order + 1):
                key = _parameter_key("a", target_index, order)
                prior[key], p0[key] = [
                    _gvar.gvar(0.0, strong_width)
                ], [0.0]
        for pole_index in range(candidate.denominator_order):
            prefix = "logrho" if pole_index == 0 else "logrho_step"
            key = _parameter_key(prefix, target_index, pole_index)
            prior[key], p0[key] = [_gvar.gvar(0.0, pole_log_width)], [0.0]
    return prior, p0


def _match_fraction(candidate, parameters):
    if candidate.strong_form in (
        "eq7_blend", "eq7_hermite", "eq7_linear",
    ):
        raise ValueError("the fixed gap bridge has no fitted matching fraction")
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
    """Return the fixed or posterior matching fraction for this fit."""
    return _match_fraction(candidate_fit.candidate, candidate_fit.fit.p)


def _weak_beta(beta_function, x, parameters, candidate, target_index):
    perturbative = beta_function.perturbative_beta_function
    return _weak_beta_derivatives(
        x, parameters, candidate, target_index,
        float(perturbative.nrm), tuple(float(value) for value in perturbative.b[:3]),
    )[0]


def _weak_beta_derivatives(
    x, parameters, candidate, target_index, normalization,
    perturbative_coefficients,
):
    """Return the weak branch and its first two x derivatives."""
    if candidate.weak_family.startswith("monotone"):
        return _monotone_weak_beta_derivatives(
            x, parameters, candidate, target_index, normalization,
            perturbative_coefficients[0] / normalization,
        )
    u = x / normalization
    perturbative = 0.0
    perturbative_first = 0.0
    perturbative_second = 0.0
    for index, coefficient in enumerate(perturbative_coefficients):
        power = index + 2
        perturbative = perturbative - normalization * coefficient * u**power
        perturbative_first = (
            perturbative_first - coefficient * power * u ** (power - 1)
        )
        perturbative_second = (
            perturbative_second
            - coefficient * power * (power - 1) * u ** (power - 2)
            / normalization
        )

    if candidate.weak_family in ("fit5", "fit6", "fit7"):
        ratio_coefficients = {
            power: -coefficient / normalization
            for power, coefficient in enumerate(perturbative_coefficients)
        }
        if candidate.weak_family == "fit6":
            ratio_coefficients[1] = parameters[
                _parameter_key("b", target_index, 1)
            ][0]
        elif candidate.weak_family == "fit7":
            ratio_coefficients[2] = parameters[
                _parameter_key("b", target_index, 2)
            ][0]
        for prefix, power in _weak_parameter_specs(candidate):
            if prefix == "d":
                ratio_coefficients[power] = parameters[
                    _parameter_key(prefix, target_index, power)
                ][0]
        ratio = sum(
            coefficient * u**power
            for power, coefficient in ratio_coefficients.items()
        )
        ratio_first = sum(
            coefficient * power * u**(power - 1) / normalization
            for power, coefficient in ratio_coefficients.items()
            if power >= 1
        )
        ratio_second = sum(
            coefficient * power * (power - 1) * u**(power - 2)
            / normalization**2
            for power, coefficient in ratio_coefficients.items()
            if power >= 2
        )
        value = x**2 * ratio
        first = 2.0*x*ratio + x**2*ratio_first
        second = 2.0*ratio + 4.0*x*ratio_first + x**2*ratio_second
        return value, first, second

    correction = 0.0 if candidate.weak_family == "fit10" else 1.0
    correction_first = 0.0
    correction_second = 0.0
    for prefix, order in _weak_parameter_specs(candidate):
        if prefix != "c":
            continue
        coefficient = parameters[
            _parameter_key(prefix, target_index, order)
        ][0]
        correction = correction + coefficient * u**order
        correction_first = (
            correction_first
            + coefficient * order * u ** (order - 1) / normalization
        )
        if order >= 2:
            correction_second = (
                correction_second
                + coefficient * order * (order - 1) * u ** (order - 2)
                / normalization**2
            )

    value = perturbative * correction
    first = perturbative_first * correction + perturbative * correction_first
    second = (
        perturbative_second * correction
        + 2.0 * perturbative_first * correction_first
        + perturbative * correction_second
    )
    if candidate.weak_family == "fit11":
        value = value + parameters[_parameter_key("a0", target_index, 0)][0]
    return value, first, second


def _bernstein_basis(degree, u):
    return [comb(degree, k) * u**k * (1.0-u)**(degree-k)
            for k in range(degree+1)]


def _monotone_weak_beta_derivatives(x, parameters, candidate, target_index, scale, b0):
    """An exact polynomial with a positive Bernstein representation of -beta'/x.

    beta'(x) = -2*b0*x*sum(w_k B_{k,N}(x/L)); w_0=1, w_k=exp(logw_k).
    The optional constant is beta(0). On 0<x<=L all basis terms are
    nonnegative, so beta decreases without inheriting the PT3 zero. The
    integral is evaluated analytically in Bernstein form, not by quadrature.
    This is a shape restriction on finite-time beta, not a theorem about
    arbitrary beta functions or about monotonicity of beta/x^2.
    """
    degree = candidate.weak_order
    u = x / scale
    weights = [1.0] + [
        _gvar.exp(parameters[_parameter_key("logw", target_index, k)][0])
        for k in range(1, degree+1)
    ]
    basis = _bernstein_basis(degree, u)
    lower_basis = _bernstein_basis(degree-1, u)
    integral_basis = _bernstein_basis(degree+2, u)
    slope_factor = sum(w*b for w, b in zip(weights, basis))
    slope_factor_derivative = degree/scale * sum(
        (weights[k+1]-weights[k])*lower_basis[k] for k in range(degree)
    )
    # Integral_0^u v B_{k,N}(v) dv =
    # (k+1)/[(N+1)(N+2)] * sum_{j=k+2}^{N+2} B_{j,N+2}(u).
    tails = [None]*(degree+3)
    tail = 0.0
    for j in range(degree+2, -1, -1):
        tail = tail + integral_basis[j]
        tails[j] = tail
    value = -2.0*b0*scale**2/((degree+1)*(degree+2)) * sum(
        (k+1)*weights[k]*tails[k+2] for k in range(degree+1)
    )
    if candidate.weak_family == "monotone_constant":
        value = value + parameters[_parameter_key("a0", target_index, 0)][0]
    first = -2.0*b0*x*slope_factor
    second = -2.0*b0*(slope_factor + x*slope_factor_derivative)
    return value, first, second


def _pole_free_denominator(z, parameters, candidate, target_index):
    """Return Q(z), Q'(0)/Q(0), and d2 log(Q)/dz2 at z=0."""
    denominator = 1.0
    inverse_scales = []
    if candidate.denominator_order:
        rho = _gvar.exp(
            parameters[_parameter_key("logrho", target_index, 0)][0]
        )
        inverse_scales.append(1.0 / rho)
        denominator = denominator * (1.0 + z / rho)
        for pole_index in range(1, candidate.denominator_order):
            rho = rho + _gvar.exp(
                parameters[
                    _parameter_key("logrho_step", target_index, pole_index)
                ][0]
            )
            inverse_scales.append(1.0 / rho)
            denominator = denominator * (1.0 + z / rho)
    log_first = sum(inverse_scales, 0.0)
    log_second = -sum((value**2 for value in inverse_scales), 0.0)
    return denominator, log_first, log_second


def _eq7_strong_beta(
    x, parameters, candidate, target_index, match, strong_scale,
    normalization, perturbative_coefficients, mask,
):
    """Matched Eq. (7) branch with beta(x) asymptotically proportional to x.

    With z=(x-x_match)/scale, P_N(z)/Q_{N+1}(z) is still a ratio
    of degree-N and degree-(N+1) polynomials in x.  Normalizing both at z=0
    fixes the value at the match.  The leading numerator coefficient(s) are
    derived analytically so the first (and optionally second) derivative also
    agrees with the weak branch.
    """
    weak_match, weak_first, weak_second = _weak_beta_derivatives(
        match, parameters, candidate, target_index, normalization,
        perturbative_coefficients,
    )
    # Do not evaluate the rational branch at negative-z weak points: the
    # pole-free construction is guaranteed only on the physical z >= 0 side.
    z = (x - match) / strong_scale
    z_eval = _numpy.where(mask, z, 0.0)
    denominator, q_log_first, q_log_second = _pole_free_denominator(
        z_eval, parameters, candidate, target_index,
    )

    weak_log_first = weak_first / weak_match
    p1 = (
        strong_scale * (weak_log_first - 2.0 / match)
        + q_log_first
    )
    numerator = 1.0 + p1 * z_eval
    if candidate.continuity == 2:
        weak_log_second = (
            weak_second / weak_match - weak_log_first**2
        )
        p_log_second = (
            strong_scale**2 * (weak_log_second + 2.0 / match**2)
            + q_log_second
        )
        p2 = 0.5 * (p_log_second + p1**2)
        numerator = numerator + p2 * z_eval**2

    for order in range(
        candidate.continuity + 1, candidate.numerator_order + 1
    ):
        numerator = numerator + parameters[
            _parameter_key("p", target_index, order)
        ][0] * z_eval**order

    return (
        weak_match * (x / match) ** 2 * numerator / denominator
    )


def _independent_eq7_strong_beta(
    x, parameters, candidate, target_index, normalization,
):
    """Direct Eq. (7) curve for the independently fitted strong branch."""
    return _independent_eq7_strong_beta_and_slope(
        x, parameters, candidate, target_index, normalization,
    )[0]


def _independent_eq7_strong_beta_and_slope(
    x, parameters, candidate, target_index, normalization,
):
    """Return the direct Eq. (7) strong branch and its x derivative."""
    scaled_x = x / normalization
    numerator = 1.0
    numerator_derivative = 0.0
    for order in range(1, candidate.numerator_order + 1):
        coefficient = parameters[
            _parameter_key("p", target_index, order)
        ][0]
        numerator = numerator + coefficient * scaled_x**order
        numerator_derivative = (
            numerator_derivative
            + order * coefficient * scaled_x ** (order - 1)
        )

    denominator = 1.0
    denominator_log_derivative = 0.0
    rho = _gvar.exp(
        parameters[_parameter_key("logrho", target_index, 0)][0]
    )
    denominator = denominator * (1.0 + scaled_x / rho)
    denominator_log_derivative = (
        denominator_log_derivative + 1.0 / (rho + scaled_x)
    )
    for pole_index in range(1, candidate.denominator_order):
        rho = rho + _gvar.exp(parameters[
            _parameter_key("logrho_step", target_index, pole_index)
        ][0])
        denominator = denominator * (1.0 + scaled_x / rho)
        denominator_log_derivative = (
            denominator_log_derivative + 1.0 / (rho + scaled_x)
        )
    p0 = _gvar.exp(parameters[
        _parameter_key("logp0", target_index, 0)
    ][0])
    value = -p0 * x**2 * numerator / denominator
    # Differentiate the product directly: the logarithmic form has removable
    # singularities at x=0 and at zeros of the numerator.
    slope = -p0 / denominator * (
        2.0 * x * numerator
        + x**2 / normalization * (
            numerator_derivative - numerator * denominator_log_derivative
        )
    )
    return value, slope


def _c2_gap_blend(x, weak, strong, lower, upper):
    """Blend two curves with exact C2 agreement at both gap endpoints."""
    if not lower < upper:
        raise ValueError("C2 blend requires lower < upper")
    coordinate = (x - lower) / (upper - lower)
    mean_x = _numpy.asarray(_gvar.mean(x), dtype=float)
    coordinate = _numpy.where(
        mean_x <= lower, 0.0,
        _numpy.where(mean_x >= upper, 1.0, coordinate),
    )
    weight = coordinate**3 * (
        10.0 - 15.0 * coordinate + 6.0 * coordinate**2
    )
    return (1.0 - weight) * weak + weight * strong


def _c1_hermite_bridge(
    x, weak, strong, weak_lower, weak_slope_lower,
    strong_upper, strong_slope_upper, lower, upper,
):
    """Minimum-curvature cubic joining endpoint values and slopes.

    Unlike a pointwise blend of extrapolated curves, this bridge uses only
    the two endpoint jets.  It therefore cannot inherit an uncontrolled
    backward extrapolation of the strong branch inside the data-free gap.
    """
    if not lower < upper:
        raise ValueError("Hermite bridge requires lower < upper")
    width = upper - lower
    coordinate = (x - lower) / width
    mean_x = _numpy.asarray(_gvar.mean(x), dtype=float)
    clipped = _numpy.where(
        mean_x <= lower, 0.0,
        _numpy.where(mean_x >= upper, 1.0, coordinate),
    )
    u2 = clipped**2
    u3 = u2 * clipped
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + clipped
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    bridge = (
        h00 * weak_lower
        + h10 * width * weak_slope_lower
        + h01 * strong_upper
        + h11 * width * strong_slope_upper
    )
    return _numpy.where(
        mean_x <= lower, weak,
        _numpy.where(mean_x >= upper, strong, bridge),
    )


def _linear_gap_bridge(
    x, weak, strong, weak_lower, strong_upper, lower, upper,
):
    """Join the fitted endpoint values by the unique straight segment.

    The beta_b=8--7.5 interval contains no observations.  A straight bridge
    is the minimum-assumption connection and, unlike an endpoint-slope
    Hermite polynomial, cannot overshoot when either fitted branch points in
    the wrong direction at the edge of the gap.  The value is continuous;
    no unsupported derivative-matching claim is made.
    """
    if not lower < upper:
        raise ValueError("linear bridge requires lower < upper")
    mean_x = _numpy.asarray(_gvar.mean(x), dtype=float)
    coordinate = (x - lower) / (upper - lower)
    bridge = weak_lower + coordinate * (strong_upper - weak_lower)
    return _numpy.where(
        mean_x <= lower, weak,
        _numpy.where(mean_x >= upper, strong, bridge),
    )


def _rational_gap_tension(value_left, slope_left, value_right, slope_right, width):
    """Smooth tension choice, retaining gvar dependence on both endpoint jets.

    For nonzero secant D, r=3+(d0/D-1)^2+(d1/D-1)^2 >= 3 and
    r >= (d0+d1)/D. Thus the Delbourgo--Gregory sufficient monotonicity
    condition holds whenever both slopes have the sign of the secant.
    Opposite-sign slopes are NOT clipped: they must be diagnosed, since
    exact C1 matching and monotonicity would then be incompatible.
    """
    secant = (value_right - value_left) / width
    scale = max(abs(float(_gvar.mean(value_left))),
                abs(float(_gvar.mean(value_right))), 1.0) / width
    if abs(float(_gvar.mean(secant))) <= 1e-12 * scale:
        # A nearly horizontal secant cannot define stable slope ratios.
        # Ordinary Hermite still matches both endpoint jets exactly.
        return 3.0
    return 3.0 + (slope_left / secant - 1.0)**2 + (slope_right / secant - 1.0)**2


def _c1_rational_gap_bridge(
    x, weak, strong, value_left, slope_left, value_right, slope_right,
    lower, upper,
):
    """Gap-only rational Hermite connection; never modifies either data region.

    Delbourgo & Gregory, TR/10/83, Eqs. (2.3)--(2.7), (3.8):
    https://bura.brunel.ac.uk/bitstream/2438/2200/1/TR_10_83.pdf
    This is a numerical gap prescription, not Eq. (7) of the physics paper.
    """
    if not lower < upper:
        raise ValueError("rational bridge requires lower < upper")
    width = upper - lower
    mean_x = _numpy.asarray(_gvar.mean(x), dtype=float)
    s = _numpy.where(mean_x <= lower, 0.0,
                    _numpy.where(mean_x >= upper, 1.0, (x - lower) / width))
    t = 1.0 - s
    r = _rational_gap_tension(value_left, slope_left, value_right, slope_right, width)
    numerator = (value_left * t**3
                 + (r * value_left + width * slope_left) * s * t**2
                 + (r * value_right - width * slope_right) * s**2 * t
                 + value_right * s**3)
    bridge = numerator / (1.0 + (r - 3.0) * s * t)
    return _numpy.where(mean_x <= lower, weak,
                        _numpy.where(mean_x >= upper, strong, bridge))


def _rational_gap_endpoint_data(beta_function, parameters, candidate, target, target_index):
    # Preserve W through beta_b=8 and a short, explicit extension into the
    # unmeasured gap. Five percent is a modelling choice, not a fitted datum.
    lower = target.match_low + 0.05 * (target.match_high - target.match_low)
    upper = target.match_high
    pt = beta_function.perturbative_beta_function
    if candidate.weak_family.startswith("monotone") and not 0.0 < lower < float(pt.nrm):
        raise ValueError(
            "monotone weak branch must join within 0 < g^2 < 4*pi; "
            "the present target requires a different basis scale"
        )
    value_left, slope_left, _ = _weak_beta_derivatives(
        lower, parameters, candidate, target_index, float(pt.nrm),
        tuple(float(value) for value in pt.b[:3]),
    )
    value_right, slope_right = _independent_eq7_strong_beta_and_slope(
        upper, parameters, candidate, target_index, float(pt.nrm),
    )
    return lower, upper, value_left, slope_left, value_right, slope_right


def _smooth_linear_gap_bridge(
    x, weak, strong, weak_lower, strong_upper,
    weak_outer, lower, upper, strong_outer,
):
    """Keep the gap affine while smoothing both endpoint joins to C2.

    The straight line between the beta_b=8 and beta_b=7.5 endpoint values is
    retained exactly.  On each side, a quintic smoothstep blends that line
    into the fitted branch over the adjacent measured interval.  Its first
    and second derivatives vanish at both ends, so the full curve has no
    derivative corner at either gap endpoint and does not use the unstable
    fitted endpoint slopes that caused the old Hermite overshoot.
    """
    if not weak_outer < lower < upper < strong_outer:
        raise ValueError(
            "smoothed linear bridge requires weak_outer < lower < upper "
            "< strong_outer"
        )

    mean_x = _numpy.asarray(_gvar.mean(x), dtype=float)
    line = weak_lower + (
        (x - lower) / (upper - lower)
    ) * (strong_upper - weak_lower)

    left_coordinate = (x - weak_outer) / (lower - weak_outer)
    left_coordinate = _numpy.clip(left_coordinate, 0.0, 1.0)
    left_weight = left_coordinate**3 * (
        10.0 - 15.0 * left_coordinate + 6.0 * left_coordinate**2
    )
    left_transition = (1.0 - left_weight) * weak + left_weight * line

    right_coordinate = (x - upper) / (strong_outer - upper)
    right_coordinate = _numpy.clip(right_coordinate, 0.0, 1.0)
    right_weight = right_coordinate**3 * (
        10.0 - 15.0 * right_coordinate + 6.0 * right_coordinate**2
    )
    right_transition = (1.0 - right_weight) * line + right_weight * strong

    return _numpy.where(
        mean_x <= weak_outer, weak,
        _numpy.where(
            mean_x < lower, left_transition,
            _numpy.where(
                mean_x <= upper, line,
                _numpy.where(mean_x < strong_outer, right_transition, strong),
            ),
        ),
    )


def _hybrid_beta(beta_function, x, parameters, candidate, target, target_index, region=None):
    base = _weak_beta(beta_function, x, parameters, candidate, target_index)
    if candidate.strong_form == "eq7_linear":
        normalization = float(
            beta_function.perturbative_beta_function.nrm
        )
        strong = _independent_eq7_strong_beta(
            x, parameters, candidate, target_index, normalization,
        )
        if region is not None:
            strong_mask = _numpy.asarray(
                [item == "strong" for item in region], dtype=bool,
            )
            return _numpy.where(strong_mask, strong, base)
        if candidate.continuity == 1:
            lower, upper, y0, d0, y1, d1 = _rational_gap_endpoint_data(
                beta_function, parameters, candidate, target, target_index,
            )
            return _c1_rational_gap_bridge(
                x, base, strong, y0, d0, y1, d1, lower, upper,
            )
        weak_lower = _weak_beta(
            beta_function, target.match_low, parameters,
            candidate, target_index,
        )
        strong_upper = _independent_eq7_strong_beta(
            target.match_high, parameters, candidate, target_index,
            normalization,
        )
        if candidate.continuity == 0:
            return _linear_gap_bridge(
                x, base, strong, weak_lower, strong_upper,
                target.match_low, target.match_high,
            )

        target_mean_x = _numpy.asarray(_gvar.mean(target.x), dtype=float)
        weak_x = _numpy.sort(target_mean_x[
            _numpy.asarray(target.regions) == "weak"
        ])
        strong_x = _numpy.sort(target_mean_x[
            _numpy.asarray(target.regions) == "strong"
        ])
        if weak_x.size < 2 or strong_x.size < 2:
            raise ValueError(
                "C2-smoothed linear bridge requires at least two data "
                "points on each side of the gap"
            )
        return _smooth_linear_gap_bridge(
            x, base, strong, weak_lower, strong_upper,
            weak_x[-2], target.match_low, target.match_high, strong_x[1],
        )

    if candidate.strong_form == "eq7_hermite":
        perturbative = beta_function.perturbative_beta_function
        normalization = float(perturbative.nrm)
        coefficients = tuple(float(value) for value in perturbative.b[:3])
        strong = _independent_eq7_strong_beta(
            x, parameters, candidate, target_index, normalization,
        )
        if region is not None:
            strong_mask = _numpy.asarray(
                [item == "strong" for item in region], dtype=bool,
            )
            return _numpy.where(strong_mask, strong, base)
        weak_lower, weak_slope_lower, _ = _weak_beta_derivatives(
            target.match_low, parameters, candidate, target_index,
            normalization, coefficients,
        )
        strong_upper, strong_slope_upper = (
            _independent_eq7_strong_beta_and_slope(
                target.match_high, parameters, candidate, target_index,
                normalization,
            )
        )
        return _c1_hermite_bridge(
            x, base, strong, weak_lower, weak_slope_lower,
            strong_upper, strong_slope_upper,
            target.match_low, target.match_high,
        )

    if candidate.strong_form == "eq7_blend":
        normalization = float(
            beta_function.perturbative_beta_function.nrm
        )
        strong = _independent_eq7_strong_beta(
            x, parameters, candidate, target_index, normalization,
        )
        return _c2_gap_blend(
            x, base, strong, target.match_low, target.match_high,
        )

    fraction = _match_fraction(candidate, parameters)
    match = target.match_low + fraction * (target.match_high - target.match_low)
    if region is None:
        mask = _numpy.asarray(_gvar.mean(x), dtype=float) > _gvar.mean(match)
    else:
        mask = _numpy.asarray([item == "strong" for item in region], dtype=bool)
    if candidate.strong_form == "eq7":
        perturbative = beta_function.perturbative_beta_function
        strong = _eq7_strong_beta(
            x, parameters, candidate, target_index, match,
            target.strong_scale, float(perturbative.nrm),
            tuple(float(value) for value in perturbative.b[:3]), mask,
        )
        return _numpy.where(mask, strong, base)

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


def _psd_clip(matrix, relative_floor=1e-13):
    """Symmetrize and remove only numerically negative covariance modes."""
    matrix = _numpy.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = _numpy.linalg.eigh(matrix)
    scale = max(float(_numpy.max(_numpy.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) < -relative_floor * scale:
        raise RuntimeError(
            "linear-response covariance has a materially negative eigenvalue: "
            f"{float(eigenvalues.min()):.6g}"
        )
    return (eigenvectors * _numpy.clip(eigenvalues, 0.0, None)) @ eigenvectors.T


def _flatten_parameter_dict(parameters):
    values, specification = [], []
    for key, value in parameters.items():
        array = _numpy.asarray(value, dtype=object)
        specification.append((key, tuple(array.shape), int(array.size)))
        values.extend(array.reshape(-1))
    return _numpy.asarray(values, dtype=object), tuple(specification)


def _unflatten_parameter_dict(values, specification):
    result, cursor = {}, 0
    for key, shape, size in specification:
        block = _numpy.asarray(values[cursor:cursor + size], dtype=object)
        result[key] = block.reshape(shape)
        cursor += size
    if cursor != len(values):
        raise ValueError("parameter specification does not consume the flat vector")
    return result


def _perturbative_beta(x, normalization, coefficients):
    return -normalization * sum(
        coefficient * (x / normalization) ** (order + 2)
        for order, coefficient in enumerate(coefficients)
    )


def _prepared_hybrid_beta(
    x, parameters, candidate, geometry, target_index,
    normalization, perturbative_coefficients,
):
    result = _weak_beta_derivatives(
        x, parameters, candidate, target_index, normalization,
        perturbative_coefficients,
    )[0]

    if candidate.strong_form in ("eq7_linear", "eq7_hermite"):
        strong = _independent_eq7_strong_beta(
            x, parameters, candidate, target_index, normalization,
        )
        strong_mask = _numpy.asarray(
            [region == "strong" for region in geometry.regions], dtype=bool,
        )
        return _numpy.where(strong_mask, strong, result)

    if candidate.strong_form == "eq7_blend":
        strong = _independent_eq7_strong_beta(
            x, parameters, candidate, target_index, normalization,
        )
        return _c2_gap_blend(
            x, result, strong, geometry.match_low, geometry.match_high,
        )

    fraction = _match_fraction(candidate, parameters)
    match = geometry.match_low + fraction * (
        geometry.match_high - geometry.match_low
    )
    mask = _numpy.asarray(
        [region == "strong" for region in geometry.regions], dtype=bool,
    )
    if candidate.strong_form == "eq7":
        strong = _eq7_strong_beta(
            x, parameters, candidate, target_index, match,
            geometry.strong_scale, normalization,
            perturbative_coefficients, mask,
        )
        return _numpy.where(mask, strong, result)

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
    flat_posterior, parameter_specification = _flatten_parameter_dict(posterior)
    local_data = _numpy.concatenate([x_values, y_values])
    joint_covariance = _numpy.asarray(
        _gvar.evalcov(_numpy.concatenate([flat_posterior, local_data])),
        dtype=float,
    )
    parameter_count = len(flat_posterior)
    posterior_covariance = joint_covariance[:parameter_count, :parameter_count]
    parameter_data_covariance = joint_covariance[:parameter_count, parameter_count:]
    data_jacobian = parameter_data_covariance @ _numpy.linalg.pinv(
        prepared.joint_covariance
    )
    propagated_covariance = (
        data_jacobian @ prepared.joint_covariance @ data_jacobian.T
    )
    residual_covariance = _psd_clip(
        posterior_covariance - propagated_covariance
    )
    return _gvar.dumps({
        "p": posterior,
        "p_mean": _numpy.asarray(_gvar.mean(flat_posterior), dtype=float),
        "p_specification": parameter_specification,
        "p_data_jacobian": data_jacobian,
        "p_residual_covariance": residual_covariance,
        "chi2": float(fit.chi2),
        "dof": int(fit.dof),
        "Q": float(fit.Q),
        "log_evidence": log_evidence,
    })


_HYBRID_WORKER_PREPARED = None
_HYBRID_WORKER_STATE = None
_HYBRID_WORKER_PRIOR_WIDTHS = None
_HYBRID_THREADPOOL_LIMITER = None
_HYBRID_OPERATOR_PREPARED = None
_HYBRID_OPERATOR_STATES = None
_HYBRID_TARGET_PREPARED = None
_HYBRID_TARGET_STATES = None


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


def _initialize_hybrid_operator_worker(prepared_by_operator, prior_widths):
    """Initialize one process for operator/candidate task parallelism."""
    global _HYBRID_OPERATOR_PREPARED, _HYBRID_OPERATOR_STATES
    global _HYBRID_WORKER_PRIOR_WIDTHS, _HYBRID_THREADPOOL_LIMITER
    _HYBRID_OPERATOR_PREPARED = prepared_by_operator
    _HYBRID_OPERATOR_STATES = {
        operator: _prepared_state(prepared)
        for operator, prepared in prepared_by_operator.items()
    }
    _HYBRID_WORKER_PRIOR_WIDTHS = prior_widths
    try:
        from threadpoolctl import threadpool_limits
        _HYBRID_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _HYBRID_THREADPOOL_LIMITER = None


def _hybrid_operator_worker_fit(operator, candidate):
    started = time.monotonic()
    payload = _fit_prepared_candidate(
        _HYBRID_OPERATOR_PREPARED[operator],
        _HYBRID_OPERATOR_STATES[operator],
        candidate,
        _HYBRID_WORKER_PRIOR_WIDTHS,
    )
    return operator, payload, time.monotonic() - started


def _initialize_hybrid_target_worker(prepared_by_target, prior_widths):
    """Initialize one process for independent target/candidate tasks."""
    global _HYBRID_TARGET_PREPARED, _HYBRID_TARGET_STATES
    global _HYBRID_WORKER_PRIOR_WIDTHS, _HYBRID_THREADPOOL_LIMITER
    _HYBRID_TARGET_PREPARED = prepared_by_target
    _HYBRID_TARGET_STATES = {
        key: _prepared_state(prepared)
        for key, prepared in prepared_by_target.items()
    }
    _HYBRID_WORKER_PRIOR_WIDTHS = prior_widths
    try:
        from threadpoolctl import threadpool_limits
        _HYBRID_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _HYBRID_THREADPOOL_LIMITER = None


def _hybrid_target_worker_fit(key, candidate):
    started = time.monotonic()
    payload = _fit_prepared_candidate(
        _HYBRID_TARGET_PREPARED[key],
        _HYBRID_TARGET_STATES[key],
        candidate,
        _HYBRID_WORKER_PRIOR_WIDTHS,
    )
    return key, payload, time.monotonic() - started


def _initialize_continuum_worker():
    """Prevent nested BLAS threading inside continuum worker processes."""
    global _HYBRID_THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _HYBRID_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _HYBRID_THREADPOOL_LIMITER = None


def _continuum_numeric_fit(task):
    """Fit one fixed-coupling continuum problem using numeric arrays only.

    The worker deliberately returns the GLS response matrix rather than
    serialized fitted gvars.  The parent process applies that response to the
    original shared gvars, preserving correlations between different g^2
    points in the final continuum curve.
    """
    key = task["key"]
    x_data = _numpy.asarray(task["x"], dtype=float)
    y_mean = _numpy.asarray(task["mean"], dtype=float)
    weight_covariance = _numpy.asarray(task["weight_covariance"], dtype=float)
    actual_covariance = _numpy.asarray(task["actual_covariance"], dtype=float)

    def fit_fcn(x, p):
        return p["beta"][0] + p["slope"][0] * x

    fit = _lsqfit.nonlinear_fit(
        data=(x_data, _gvar.gvar(y_mean, weight_covariance)),
        fcn=fit_fcn,
        prior=None,
        p0={"beta": [0.0], "slope": [0.0]},
    )
    parameter_mean = _numpy.asarray([
        _gvar.mean(fit.p["beta"][0]),
        _gvar.mean(fit.p["slope"][0]),
    ], dtype=float)
    design = _numpy.column_stack([_numpy.ones(len(x_data)), x_data])
    weight = _numpy.linalg.pinv(weight_covariance)
    gain = (
        _numpy.linalg.pinv(design.T @ weight @ design)
        @ design.T @ weight
    )
    residual = y_mean - design @ parameter_mean
    generalized = bruno_sommer_goodness_of_fit(
        residual, design, actual_covariance, weight_covariance,
        model_hessian=_numpy.zeros((len(x_data), 2, 2)),
    )
    quality = {
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
    }
    return key, parameter_mean, gain, quality


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
        target_operators = {key[1] for key in target_keys}
        sole_operator = (
            next(iter(target_operators)) if len(target_operators) == 1 else None
        )
        sole_target = target_keys[0] if len(target_keys) == 1 else (None, sole_operator, None)
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
        if (
            candidate.match_fraction is None
            and candidate.strong_form not in (
                "eq7_blend", "eq7_hermite", "eq7_linear",
            )
        ):
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
                "flow": sole_target[0],
                "operator": sole_target[1],
                "flow_time": (
                    float(sole_target[2])
                    if sole_target[2] is not None else _numpy.nan
                ),
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


def fit_hybrid_bma_by_operator(
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
    """Fit operator-specific BMAs with one shared process pool.

    Every operator retains its full cross-flow-time and cross-coupling
    covariance, including correlated x errors. Operator/candidate pairs are
    independent likelihoods and are scheduled together, allowing one useful
    worker per selected operator/candidate combination.
    """
    observables = tuple(observables)
    if not observables:
        raise ValueError("select at least one observable")
    candidates = (
        default_hybrid_candidates() if candidates is None
        else tuple(candidates)
    )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    widths = {
        "weak_width": float(prior_widths.get("weak_width", 5.0)),
        "constant_width": float(prior_widths.get("constant_width", 0.25)),
        "strong_width": float(prior_widths.get("strong_width", 0.10)),
        "pole_log_width": float(prior_widths.get("pole_log_width", 0.70)),
    }
    checkpoint_root = (
        None if checkpoint_dir is None
        else Path(checkpoint_dir).expanduser().resolve()
    )

    prepared_by_operator = {}
    target_items_by_operator = {}
    slices_by_operator = {}
    for operator in observables:
        targets = collect_hybrid_targets(
            beta_function, window, flow, (operator,), weak_couplings,
            strong_couplings, flow_time_spacing=flow_time_spacing,
        )
        prepared_by_operator[operator] = _prepare_hybrid_data(
            beta_function, targets
        )
        target_items = tuple(sorted(targets.items()))
        target_items_by_operator[operator] = target_items
        slices, cursor = {}, 0
        for key, target in target_items:
            next_cursor = cursor + len(target.x)
            slices[key] = slice(cursor, next_cursor)
            cursor = next_cursor
        slices_by_operator[operator] = slices
        if checkpoint_root is not None:
            (checkpoint_root / operator).mkdir(parents=True, exist_ok=True)

    def checkpoint_file(operator, candidate):
        if checkpoint_root is None:
            return None
        return checkpoint_root / operator / f"{candidate.name}.gvar"

    def signature(operator, candidate):
        prepared = prepared_by_operator[operator]
        digest = hashlib.sha256(prepared.data_signature.encode())
        digest.update(operator.encode())
        digest.update(repr(candidate).encode())
        digest.update(repr(sorted(widths.items())).encode())
        return digest.hexdigest()

    def restore(operator, candidate, payload):
        result = _gvar.loads(payload) if isinstance(payload, bytes) else payload
        summary = HybridFitSummary(
            p=result["p"], chi2=float(result["chi2"]),
            dof=int(result["dof"]), Q=float(result["Q"]),
        )
        return HybridCandidateFit(
            candidate=candidate,
            fit=summary,
            targets=dict(target_items_by_operator[operator]),
            slices=dict(slices_by_operator[operator]),
            log_evidence=float(result["log_evidence"]),
        )

    def load_checkpoint(operator, candidate):
        path = checkpoint_file(operator, candidate)
        if path is None or not path.is_file():
            return None
        try:
            payload = _gvar.load(str(path))
        except Exception as error:
            if progress:
                print(
                    f"Ignoring unreadable checkpoint {path.name}: {error}",
                    flush=True,
                )
            return None
        if payload.get("signature") != signature(operator, candidate):
            return None
        return restore(operator, candidate, payload["result"])

    def save_checkpoint(operator, candidate, payload):
        path = checkpoint_file(operator, candidate)
        if path is None:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        _gvar.dump({
            "signature": signature(operator, candidate),
            "result": _gvar.loads(payload),
        }, str(temporary))
        os.replace(temporary, path)

    completed = {operator: {} for operator in observables}
    pending = []
    for operator in observables:
        for candidate in candidates:
            result = load_checkpoint(operator, candidate)
            if result is None:
                pending.append((operator, candidate))
            else:
                completed[operator][candidate.name] = result
    completed_count = sum(len(fits) for fits in completed.values())
    total_count = len(observables) * len(candidates)
    if progress and completed_count:
        print(
            f"Loaded {completed_count}/{total_count} valid operator-model "
            "checkpoints.",
            flush=True,
        )

    start_all = time.monotonic()
    durations = []
    worker_count = min(max_workers, max(1, len(pending)))

    def record(operator, candidate, payload, elapsed):
        save_checkpoint(operator, candidate, payload)
        completed[operator][candidate.name] = restore(
            operator, candidate, payload
        )
        durations.append(float(elapsed))
        if progress:
            done = sum(len(fits) for fits in completed.values())
            remaining = total_count - done
            eta = remaining * (sum(durations) / len(durations)) / worker_count
            print(
                f"[{done}/{total_count}] {operator}:{candidate.name} "
                f"completed in {elapsed / 60.0:.1f} min; estimated remaining "
                f"{eta / 60.0:.1f} min",
                flush=True,
            )

    if pending and max_workers == 1:
        states = {
            operator: _prepared_state(prepared)
            for operator, prepared in prepared_by_operator.items()
        }
        for operator, candidate in pending:
            started = time.monotonic()
            payload = _fit_prepared_candidate(
                prepared_by_operator[operator], states[operator],
                candidate, widths,
            )
            record(
                operator, candidate, payload,
                time.monotonic() - started,
            )
    elif pending:
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_hybrid_operator_worker,
            initargs=(prepared_by_operator, widths),
        ) as executor:
            submitted = {
                executor.submit(
                    _hybrid_operator_worker_fit, operator, candidate
                ): (operator, candidate)
                for operator, candidate in pending
            }
            for future in as_completed(submitted):
                operator, candidate = submitted[future]
                returned_operator, payload, elapsed = future.result()
                if returned_operator != operator:
                    raise RuntimeError(
                        f"worker returned {returned_operator}, expected {operator}"
                    )
                record(operator, candidate, payload, elapsed)

    if progress:
        print(
            "Operator-specific hybrid interpolation ensemble completed in "
            f"{(time.monotonic() - start_all) / 60.0:.1f} min.",
            flush=True,
        )
    results = {}
    for operator in observables:
        fits = {
            candidate.name: completed[operator][candidate.name]
            for candidate in candidates
        }
        weights = posterior_model_weights(
            {name: result.log_evidence for name, result in fits.items()},
            model_prior=model_prior,
        )
        results[operator] = HybridInterpolationResult(
            window=tuple(map(float, window)),
            flow_time_spacing=flow_time_spacing,
            candidate_fits=fits,
            weights=weights,
        )
    return results


def fit_hybrid_bma_by_target(
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
    """Fit one BMA independently for every operator and flow time.

    A task contains only the coupling points belonging to a single
    ``(flow, operator, flow_time)`` target.  In particular, neither distinct
    operators nor distinct flow times are stacked into one likelihood.  The
    process pool parallelizes the resulting target/candidate tasks without
    changing any individual fit.
    """
    observables = tuple(observables)
    if not observables:
        raise ValueError("select at least one observable")
    candidates = (
        default_hybrid_candidates() if candidates is None
        else tuple(candidates)
    )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")

    weak_couplings = tuple(weak_couplings)
    strong_couplings = tuple(strong_couplings)
    targets = collect_hybrid_targets(
        beta_function, window, flow, observables, weak_couplings,
        strong_couplings, flow_time_spacing=flow_time_spacing,
    )
    target_items = tuple(sorted(targets.items()))
    expected_points = len(weak_couplings) + len(strong_couplings)
    for key, target in target_items:
        if len(target.x) != expected_points or len(target.y) != expected_points:
            raise ValueError(
                f"target {key} has {len(target.x)} x and {len(target.y)} beta "
                f"values; expected {expected_points} paired coupling points"
            )
        if any(c.weak_family.startswith("monotone") for c in candidates):
            join = target.match_low + 0.05*(target.match_high-target.match_low)
            scale = float(beta_function.perturbative_beta_function.nrm)
            weak_x = _numpy.asarray(_gvar.mean(target.x), dtype=float)[
                _numpy.asarray(target.regions) == "weak"
            ]
            if (not 0.0 < join < scale or _numpy.any(weak_x <= 0.0)
                    or _numpy.any(weak_x > join)):
                raise ValueError(
                    f"target {key}: monotone weak data and join must lie "
                    "inside 0 < g^2 < 4*pi; revise the basis scale before fitting"
                )
    widths = {
        "weak_width": float(prior_widths.get("weak_width", 5.0)),
        "constant_width": float(prior_widths.get("constant_width", 0.25)),
        "strong_width": float(prior_widths.get("strong_width", 0.10)),
        "pole_log_width": float(prior_widths.get("pole_log_width", 0.70)),
    }
    prepared_by_target = {
        key: _prepare_hybrid_data(beta_function, {key: target})
        for key, target in target_items
    }
    checkpoint_root = (
        None if checkpoint_dir is None
        else Path(checkpoint_dir).expanduser().resolve()
    )

    def target_directory(key):
        if checkpoint_root is None:
            return None
        _, operator, flow_time = key
        time_tag = f"t_{float(flow_time):.8g}".replace(".", "p")
        path = checkpoint_root / operator / time_tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint_file(key, candidate):
        directory = target_directory(key)
        return None if directory is None else directory / f"{candidate.name}.gvar"

    def signature(key, candidate):
        digest = hashlib.sha256(prepared_by_target[key].data_signature.encode())
        digest.update(repr(key).encode())
        digest.update(repr(candidate).encode())
        digest.update(repr(sorted(widths.items())).encode())
        digest.update(b"independent-operator-flow-time-v2-correlated-response")
        return digest.hexdigest()

    def restore(key, candidate, payload):
        result = _gvar.loads(payload) if isinstance(payload, bytes) else payload
        target = targets[key]
        original_data = _numpy.concatenate([target.x, target.y])
        data_shift = original_data - _numpy.asarray(
            _gvar.mean(original_data), dtype=float,
        )
        correlated_parameters = (
            _numpy.asarray(result["p_mean"], dtype=float)
            + _numpy.asarray(result["p_data_jacobian"], dtype=float).dot(data_shift)
        )
        residual_covariance = _numpy.asarray(
            result["p_residual_covariance"], dtype=float,
        )
        if residual_covariance.size:
            correlated_parameters = correlated_parameters + _numpy.asarray(
                _gvar.gvar(
                    _numpy.zeros(len(correlated_parameters)),
                    residual_covariance,
                ),
                dtype=object,
            )
        posterior = _unflatten_parameter_dict(
            correlated_parameters, result["p_specification"],
        )
        summary = HybridFitSummary(
            p=posterior, chi2=float(result["chi2"]),
            dof=int(result["dof"]), Q=float(result["Q"]),
        )
        return HybridCandidateFit(
            candidate=candidate,
            fit=summary,
            targets={key: target},
            slices={key: slice(0, len(target.x))},
            log_evidence=float(result["log_evidence"]),
        )

    def load_checkpoint(key, candidate):
        equivalent_candidates = [candidate]
        if (candidate.strong_form == "eq7_linear" and candidate.continuity == 1
                and candidate.weak_family in ("fit4", "fit11")):
            # All three connections use IDENTICAL labelled likelihoods and
            # priors. The bridge never enters _prepared_hybrid_beta. Reuse
            # only checkpoints whose complete data/prior signature matches;
            # downstream curves must still be regenerated with the C1 rule.
            equivalent_candidates.extend(
                replace(candidate, continuity=value) for value in (2, 0)
            )
        for stored_candidate in equivalent_candidates:
            path = checkpoint_file(key, stored_candidate)
            if path is None or not path.is_file():
                continue
            try:
                payload = _gvar.load(str(path))
            except Exception as error:
                if progress:
                    print(f"Ignoring unreadable checkpoint {path}: {error}", flush=True)
                continue
            if payload.get("signature") == signature(key, stored_candidate):
                return restore(key, candidate, payload["result"])
        return None

    def save_checkpoint(key, candidate, payload):
        path = checkpoint_file(key, candidate)
        if path is None:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        _gvar.dump({
            "signature": signature(key, candidate),
            "result": _gvar.loads(payload),
        }, str(temporary))
        os.replace(temporary, path)

    completed = {key: {} for key, _ in target_items}
    pending = []
    for key, _ in target_items:
        for candidate in candidates:
            result = load_checkpoint(key, candidate)
            if result is None:
                pending.append((key, candidate))
            else:
                completed[key][candidate.name] = result
    completed_count = sum(len(fits) for fits in completed.values())
    total_count = len(target_items) * len(candidates)
    if progress:
        print(
            f"Independent hybrid interpolation: {len(target_items)} targets x "
            f"{len(candidates)} models = {total_count} fits; "
            f"{completed_count} restored.",
            flush=True,
        )

    start_all = time.monotonic()
    durations = []
    worker_count = min(max_workers, max(1, len(pending)))

    def record(key, candidate, payload, elapsed):
        save_checkpoint(key, candidate, payload)
        completed[key][candidate.name] = restore(key, candidate, payload)
        durations.append(float(elapsed))
        if progress:
            done = sum(len(fits) for fits in completed.values())
            remaining = total_count - done
            eta = remaining * (sum(durations) / len(durations)) / worker_count
            print(
                f"[{done}/{total_count}] {key[1]} t={float(key[2]):g} "
                f"{candidate.name}; {elapsed:.1f} s; ETA {eta / 60.0:.1f} min",
                flush=True,
            )

    if pending and max_workers == 1:
        states = {
            key: _prepared_state(prepared)
            for key, prepared in prepared_by_target.items()
        }
        for key, candidate in pending:
            started = time.monotonic()
            payload = _fit_prepared_candidate(
                prepared_by_target[key], states[key], candidate, widths,
            )
            record(key, candidate, payload, time.monotonic() - started)
    elif pending:
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_hybrid_target_worker,
            initargs=(prepared_by_target, widths),
        ) as executor:
            # Bound the submitted queue so thousands of candidate models do
            # not create thousands of resident Future/task payload objects.
            pending_iterator = iter(pending)
            submitted = {}

            def submit_next():
                try:
                    key, candidate = next(pending_iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    _hybrid_target_worker_fit, key, candidate,
                )
                submitted[future] = (key, candidate)
                return True

            for _ in range(min(len(pending), 2 * worker_count)):
                submit_next()
            while submitted:
                finished, _ = wait(
                    tuple(submitted), return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    key, candidate = submitted.pop(future)
                    returned_key, payload, elapsed = future.result()
                    if returned_key != key:
                        raise RuntimeError(
                            f"worker returned {returned_key}, expected {key}"
                        )
                    record(key, candidate, payload, elapsed)
                    submit_next()

    if progress:
        print(
            "Independent operator/flow-time interpolation ensemble completed "
            f"in {(time.monotonic() - start_all) / 60.0:.1f} min.",
            flush=True,
        )
    results = {}
    for key, _ in target_items:
        fits = {
            candidate.name: completed[key][candidate.name]
            for candidate in candidates
        }
        weights = posterior_model_weights(
            {name: result.log_evidence for name, result in fits.items()},
            model_prior=model_prior,
        )
        results[key] = HybridInterpolationResult(
            window=tuple(map(float, window)),
            flow_time_spacing=flow_time_spacing,
            candidate_fits=fits,
            weights=weights,
        )
    return results


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


def run_hybrid_bma_by_operator(
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
    prefit_interpolations=None,
    max_workers=1,
    checkpoint_dir=None,
    progress=True,
    **prior_widths,
):
    """Run continuum limits and model averaging separately by operator."""
    observables = tuple(observables)
    if not observables:
        raise ValueError("select at least one observable")
    if prefit_interpolations is None:
        prefit_interpolations = fit_hybrid_bma_by_operator(
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
    missing = set(observables) - set(prefit_interpolations)
    if missing:
        raise ValueError(
            "prefit_interpolations is missing operators: "
            + ", ".join(sorted(missing))
        )
    results = {}
    for index, operator in enumerate(observables, start=1):
        if progress:
            print(
                f"[continuum operator {index}/{len(observables)}] {operator}",
                flush=True,
            )
        results[operator] = run_hybrid_bma_window(
            beta_function, window, flow, (operator,),
            weak_couplings=weak_couplings,
            strong_couplings=strong_couplings,
            g2_grid=g2_grid,
            candidates=candidates,
            model_prior=model_prior,
            cov_mode=cov_mode,
            alpha=alpha,
            tau0=tau0,
            flow_time_spacing=flow_time_spacing,
            prefit_candidate_fits=(
                prefit_interpolations[operator].candidate_fits
            ),
            progress=progress,
            **prior_widths,
        )
    return results


def _hybrid_choice_axes(candidate_fits):
    metadata = {
        name: result.candidate.metadata()
        for name, result in candidate_fits.items()
    }
    possible = (
        "weak_family", "weak_order", "pade_order",
        "match_fraction", "continuity",
    )
    return metadata, tuple(
        axis for axis in possible
        if len({values[axis] for values in metadata.values()}) > 1
    )


def _average_independent_target(
    beta_function, key, interpolation_result, g2_grid,
):
    reference_fit = next(iter(interpolation_result.candidate_fits.values()))
    target = reference_fit.targets[key]
    grid = _numpy.asarray(g2_grid, dtype=float)
    mean_x = _numpy.asarray(_gvar.mean(target.x), dtype=float)
    valid = (grid >= mean_x.min()) & (grid <= mean_x.max())
    supported_grid = grid[valid]
    if not len(supported_grid):
        raise ValueError(f"no requested g^2 values are supported by target {key}")
    # Stream complete model predictions into the exact first and second BMA
    # moments.  Keeping every model's gvar vector becomes prohibitive for a
    # large analysis multiverse and is unnecessary once these moments and the
    # weighted, data-correlated prediction have been accumulated.
    names = tuple(interpolation_result.candidate_fits)
    size = len(supported_grid)
    model_means = {}
    within_covariance = _numpy.zeros((size, size), dtype=float)
    weighted_prediction = _numpy.zeros(size, dtype=object)
    for name in names:
        prediction = _numpy.asarray(
            evaluate_hybrid_target(
                interpolation_result.candidate_fits[name],
                beta_function, key, supported_grid,
            ),
            dtype=object,
        ).reshape(-1)
        weight = float(interpolation_result.weights[name])
        model_means[name] = _numpy.asarray(_gvar.mean(prediction), dtype=float)
        within_covariance += weight * _numpy.asarray(
            _gvar.evalcov(prediction), dtype=float,
        )
        weighted_prediction = weighted_prediction + weight * prediction

    mean = sum(
        float(interpolation_result.weights[name]) * model_means[name]
        for name in names
    )
    between_covariance = _numpy.zeros((size, size), dtype=float)
    for name in names:
        displacement = model_means[name] - mean
        between_covariance += (
            float(interpolation_result.weights[name])
            * _numpy.outer(displacement, displacement)
        )
    within_covariance = 0.5 * (
        within_covariance + within_covariance.T
    )
    between_covariance = 0.5 * (
        between_covariance + between_covariance.T
    )
    total_covariance = within_covariance + between_covariance

    # The weighted prediction carries correlations caused by shared upstream
    # Monte Carlo data across targets.  Independent correction vectors restore
    # the exact within-model and total local BMA covariances without creating
    # one enormous global covariance matrix.
    weighted_covariance = _numpy.asarray(
        _gvar.evalcov(weighted_prediction), dtype=float,
    )
    within_correction = _psd_clip(
        within_covariance - weighted_covariance,
    )
    total_correction = _psd_clip(
        total_covariance - weighted_covariance,
    )
    within_prediction = weighted_prediction + _numpy.asarray(
        _gvar.gvar(_numpy.zeros(size), within_correction), dtype=object,
    )
    prediction = weighted_prediction + _numpy.asarray(
        _gvar.gvar(_numpy.zeros(size), total_correction), dtype=object,
    )
    average = ModelAverageResult(
        prediction=prediction,
        weights={
            name: float(interpolation_result.weights[name]) for name in names
        },
        mean=_numpy.asarray(mean, dtype=float),
        covariance=total_covariance,
        within_covariance=within_covariance,
        between_covariance=between_covariance,
    )
    metadata, axes = _hybrid_choice_axes(interpolation_result.candidate_fits)
    variance = shapley_model_variance(
        model_means,
        interpolation_result.weights,
        metadata,
        axes,
    )
    return HybridTargetAverage(
        key=key,
        g2=supported_grid,
        average=average,
        model_variance_by_choice=variance,
        weighted_prediction=weighted_prediction,
        within_prediction=within_prediction,
    )


def run_hybrid_bma_by_target(
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
    prefit_interpolations=None,
    max_workers=1,
    checkpoint_dir=None,
    progress=True,
    **prior_widths,
):
    """BMA each flow-time interpolation, then perform the continuum fit.

    Model evidences are compared only for models fitted to the same data: one
    operator at one flow time.  The moment-matched BMA predictions from those
    independent interpolations are subsequently supplied to the ordinary
    continuum extrapolation.  Bartlett/lugsail input uncertainties and the
    Bruno--Sommer goodness-of-fit calibration are therefore retained.
    """
    observables = tuple(observables)
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    candidates = (
        default_hybrid_candidates() if candidates is None
        else tuple(candidates)
    )
    grid = _numpy.asarray(g2_grid, dtype=float)
    if prefit_interpolations is None:
        prefit_interpolations = fit_hybrid_bma_by_target(
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
    target_interpolations = dict(prefit_interpolations)
    if not target_interpolations:
        raise ValueError("prefit_interpolations is empty")
    for key, result in target_interpolations.items():
        if key[0] != flow or key[1] not in observables:
            raise ValueError(f"unexpected interpolation target {key}")
        if set(result.candidate_fits) != {candidate.name for candidate in candidates}:
            raise ValueError(f"candidate ensemble mismatch for target {key}")
        for candidate_fit in result.candidate_fits.values():
            if set(candidate_fit.targets) != {key}:
                raise ValueError(
                    f"candidate {candidate_fit.candidate.name} for {key} "
                    "contains a joint multi-target likelihood"
                )

    target_items = tuple(sorted(target_interpolations.items()))
    total_interpolation_models = sum(
        len(result.candidate_fits) for _, result in target_items
    )
    if progress:
        print(
            f"[continuum setup] {len(target_items)} operator/flow-time targets; "
            f"{total_interpolation_models} completed interpolation models available",
            flush=True,
        )
    target_averages = {}
    completed_interpolation_models = 0
    averaging_started = time.monotonic()
    target_report_stride = max(1, len(target_items) // 20)
    for target_index, (key, result) in enumerate(target_items, start=1):
        target_averages[key] = _average_independent_target(
            beta_function, key, result, grid,
        )
        completed_interpolation_models += len(result.candidate_fits)
        if progress and (
            target_index == 1
            or target_index % target_report_stride == 0
            or target_index == len(target_items)
        ):
            elapsed = time.monotonic() - averaging_started
            eta = elapsed / target_index * (len(target_items) - target_index)
            print(
                f"[continuum preparation {target_index}/{len(target_items)}; "
                f"{100.0*target_index/len(target_items):.1f}%] "
                f"{completed_interpolation_models}/{total_interpolation_models} "
                f"interpolation models averaged; elapsed {elapsed/60.0:.1f} min; "
                f"ETA {eta/60.0:.1f} min",
                flush=True,
            )
    # Every compact target prediction already contains its exact local BMA
    # covariance and retains shared-data correlations through its weighted
    # component.  No global target-by-grid covariance matrix is constructed.
    target_within_covariance = {}
    if progress:
        print(
            "[continuum covariance] compact shared-gvar representation ready; "
            "no global covariance allocation",
            flush=True,
        )
    continuum_parameters = {operator: [] for operator in observables}
    continuum_quality = {operator: [] for operator in observables}
    continuum_points = {operator: [] for operator in observables}
    averages = {}
    variance_by_choice = {}

    # Prepare all operator/g^2 fits in the parent, retaining their original
    # shared gvars here.  Workers receive numeric arrays only.
    prepared_points = {operator: [None] * len(grid) for operator in observables}
    numeric_tasks = []
    for operator in observables:
        entries = sorted(
            (
                item for key, item in target_averages.items()
                if key[0] == flow and key[1] == operator
            ),
            key=lambda item: float(item.key[2]),
        )
        lookup = {
            item.key: {
                round(float(g2), 12): index
                for index, g2 in enumerate(item.g2)
            }
            for item in entries
        }
        for grid_index, g2 in enumerate(grid):
            grid_key = round(float(g2), 12)
            supported = [item for item in entries if grid_key in lookup[item.key]]
            if len(supported) < 3:
                continue
            measured = _numpy.asarray(
                [float(item.key[2]) for item in supported], dtype=float,
            )
            nominal = measured - float(tau0)
            x_data = 1.0 / nominal
            jacobian = nominal / measured
            local_indices = [lookup[item.key][grid_key] for item in supported]
            y = _numpy.asarray([
                factor * item.average.prediction[local_index]
                for factor, item, local_index in zip(
                    jacobian, supported, local_indices
                )
            ], dtype=object)
            distances = [
                _numpy.subtract.outer(_numpy.log(nominal), _numpy.log(nominal))
            ]
            weight_covariance = beta_function._weight_covariance(
                y, cov_mode=cov_mode, alpha=alpha, distances=distances,
            )
            actual_covariance = _numpy.asarray(_gvar.evalcov(y), dtype=float)
            point = {
                "g2": float(g2), "supported": supported,
                "local_indices": local_indices, "jacobian": jacobian,
                "x": x_data, "y": y,
            }
            prepared_points[operator][grid_index] = point
            numeric_tasks.append({
                "key": (operator, grid_index),
                "x": x_data,
                "mean": _numpy.asarray(_gvar.mean(y), dtype=float),
                "weight_covariance": weight_covariance,
                "actual_covariance": actual_covariance,
            })

    numeric_results = {}
    continuum_started = time.monotonic()
    worker_count = min(max_workers, max(1, len(numeric_tasks)))
    report_stride = max(1, len(numeric_tasks) // 20)

    def record_numeric_result(result):
        key, parameter_mean, gain, quality = result
        numeric_results[key] = (parameter_mean, gain, quality)
        done = len(numeric_results)
        if progress and (
            done == 1 or done % report_stride == 0 or done == len(numeric_tasks)
        ):
            elapsed = time.monotonic() - continuum_started
            eta = elapsed / done * (len(numeric_tasks) - done)
            print(
                f"[continuum fits {done}/{len(numeric_tasks)}; "
                f"{100.0*done/len(numeric_tasks):.1f}%] "
                f"{worker_count} worker(s); elapsed {elapsed/60.0:.1f} min; "
                f"ETA {eta/60.0:.1f} min",
                flush=True,
            )

    if progress:
        print(
            f"[continuum parallel] {len(numeric_tasks)} fixed-g^2 fits on "
            f"{worker_count} worker(s)",
            flush=True,
        )
    if numeric_tasks and worker_count == 1:
        _initialize_continuum_worker()
        for task in numeric_tasks:
            record_numeric_result(_continuum_numeric_fit(task))
    elif numeric_tasks:
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_continuum_worker,
        ) as executor:
            task_iterator = iter(numeric_tasks)
            submitted = set()

            def submit_next_continuum_task():
                try:
                    task = next(task_iterator)
                except StopIteration:
                    return False
                submitted.add(executor.submit(_continuum_numeric_fit, task))
                return True

            for _ in range(min(len(numeric_tasks), 2 * worker_count)):
                submit_next_continuum_task()
            while submitted:
                finished, _ = wait(
                    tuple(submitted), return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    submitted.remove(future)
                    record_numeric_result(future.result())
                    submit_next_continuum_task()

    # Reconstruct parameters from the original shared gvars in the parent.
    # This preserves cross-g^2 covariance even though the numerical GLS work
    # was distributed among independent processes.
    for operator_index, operator in enumerate(observables, start=1):
        intercepts = []
        within_intercepts = []
        valid_grid = []
        axis_records = []
        for grid_index, point in enumerate(prepared_points[operator]):
            if point is None:
                continuum_parameters[operator].append(None)
                continuum_quality[operator].append(None)
                continuum_points[operator].append(None)
                continue
            parameter_mean, gain, quality = numeric_results[(operator, grid_index)]
            y = point["y"]
            y_mean = _numpy.asarray(_gvar.mean(y), dtype=float)
            corrected = parameter_mean + gain.dot(y - y_mean)
            params = {
                "beta": [corrected[0]],
                "slope": [corrected[1]],
            }
            continuum_parameters[operator].append(params)
            continuum_quality[operator].append(quality)
            continuum_points[operator].append({
                "keys": tuple(item.key for item in point["supported"]),
                "x": point["x"],
                "y": y,
            })
            intercepts.append(params["beta"][0])
            within_intercepts.append(sum(
                float(gain[0, index] * point["jacobian"][index])
                * item.within_prediction[local_index]
                for index, (item, local_index) in enumerate(
                    zip(point["supported"], point["local_indices"])
                )
            ))
            valid_grid.append(point["g2"])
            axis_records.append({
                axis: sum(
                    (gain[0, index] * point["jacobian"][index]) ** 2
                    * float(item.model_variance_by_choice.get(
                        axis, _numpy.zeros(len(item.g2))
                    )[local_index])
                    for index, (item, local_index) in enumerate(
                        zip(point["supported"], point["local_indices"])
                    )
                )
                for axis in {
                    name
                    for item in point["supported"]
                    for name in item.model_variance_by_choice
                }
            })

        if not intercepts:
            continue
        prediction = _numpy.asarray(intercepts, dtype=object)
        total_covariance = _numpy.asarray(_gvar.evalcov(prediction), dtype=float)
        within_covariance = _numpy.asarray(
            _gvar.evalcov(_numpy.asarray(within_intercepts, dtype=object)),
            dtype=float,
        )
        within_covariance = 0.5 * (
            within_covariance + within_covariance.T
        )
        between_covariance = 0.5 * (
            total_covariance - within_covariance
            + (total_covariance - within_covariance).T
        )
        averages[operator] = ModelAverageResult(
            prediction=prediction,
            weights={},
            mean=_numpy.asarray(_gvar.mean(prediction), dtype=float),
            covariance=total_covariance,
            within_covariance=within_covariance,
            between_covariance=between_covariance,
        )
        axes = sorted({axis for row in axis_records for axis in row})
        variance_by_choice[operator] = {
            axis: _numpy.asarray([row.get(axis, 0.0) for row in axis_records])
            for axis in axes
        }
        variance_by_choice[operator]["g2"] = _numpy.asarray(valid_grid, dtype=float)
        if progress:
            print(
                f"[continuum operator {operator_index}/{len(observables)}] "
                f"{operator}: {len(valid_grid)} supported g^2 values",
                flush=True,
            )

    return HybridIndependentWindowResult(
        window=tuple(map(float, window)),
        flow_time_spacing=flow_time_spacing,
        target_interpolations=target_interpolations,
        target_averages=target_averages,
        target_within_covariance=target_within_covariance,
        g2=grid,
        continuum_parameters=continuum_parameters,
        continuum_quality=continuum_quality,
        continuum_points=continuum_points,
        averages=averages,
        model_variance_by_choice=variance_by_choice,
    )
