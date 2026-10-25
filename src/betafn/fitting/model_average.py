"""Bayesian model averaging utilities for correlated beta-function analyses.

The routines in this module deliberately average *complete model predictions*.
They do not average fitted coefficients, which generally have different
meanings and dimensions across models.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Iterable, Mapping, Sequence

import gvar as _gvar
import numpy as _numpy
from scipy.special import logsumexp as _logsumexp


@dataclass(frozen=True)
class ModelAverageResult:
    """Moment-matched representation of a finite posterior model mixture."""

    prediction: _numpy.ndarray
    weights: dict[str, float]
    mean: _numpy.ndarray
    covariance: _numpy.ndarray
    within_covariance: _numpy.ndarray
    between_covariance: _numpy.ndarray


def posterior_model_weights(
    log_evidence: Mapping[str, float],
    model_prior: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return normalized posterior model probabilities.

    ``model_prior`` contains prior probabilities, not log probabilities.  All
    evidences must use the same data, likelihood normalization, and proper
    parameter priors.  These conditions are checked by the caller because they
    cannot be inferred from scalar log-evidence values.
    """
    if not log_evidence:
        raise ValueError("at least one log evidence is required")
    names = tuple(log_evidence)
    values = _numpy.asarray([log_evidence[name] for name in names], dtype=float)
    if not _numpy.all(_numpy.isfinite(values)):
        raise ValueError("all log evidences must be finite")
    if model_prior is None:
        priors = _numpy.full(len(names), 1.0 / len(names))
    else:
        missing = set(names) - set(model_prior)
        if missing:
            raise ValueError(f"model_prior is missing {sorted(missing)}")
        priors = _numpy.asarray([model_prior[name] for name in names], dtype=float)
        if _numpy.any(priors <= 0.0) or not _numpy.all(_numpy.isfinite(priors)):
            raise ValueError("model prior probabilities must be finite and positive")
        priors = priors / priors.sum()
    log_weights = values + _numpy.log(priors)
    log_weights -= _logsumexp(log_weights)
    return {name: float(weight) for name, weight in zip(names, _numpy.exp(log_weights))}


def model_average_predictions(
    predictions: Mapping[str, Sequence],
    weights: Mapping[str, float],
) -> ModelAverageResult:
    """Moment-match correlated gvar predictions from a discrete model mixture.

    The total covariance obeys the law of total covariance,

    ``Cov(Y) = E_M[Cov(Y|M)] + Cov_M(E[Y|M])``.

    The returned gvars encode that covariance exactly through second order.
    The underlying posterior remains a non-Gaussian mixture, so nonlinear tail
    probabilities should instead be evaluated model by model.
    """
    names = tuple(predictions)
    if not names:
        raise ValueError("at least one model prediction is required")
    if set(names) != set(weights):
        raise ValueError("predictions and weights must contain the same models")
    arrays = {name: _numpy.asarray(predictions[name], dtype=object).reshape(-1) for name in names}
    size = len(arrays[names[0]])
    if any(len(values) != size for values in arrays.values()):
        raise ValueError("all model predictions must have the same flattened size")
    weight_values = _numpy.asarray([weights[name] for name in names], dtype=float)
    if _numpy.any(weight_values < 0.0) or not _numpy.isclose(weight_values.sum(), 1.0):
        raise ValueError("model weights must be nonnegative and sum to one")

    model_means = _numpy.vstack([_gvar.mean(arrays[name]) for name in names])
    mean = weight_values @ model_means
    within = _numpy.zeros((size, size), dtype=float)
    between = _numpy.zeros((size, size), dtype=float)
    for index, name in enumerate(names):
        within += weight_values[index] * _gvar.evalcov(arrays[name])
        displacement = model_means[index] - mean
        between += weight_values[index] * _numpy.outer(displacement, displacement)
    covariance = 0.5 * (within + between + (within + between).T)
    prediction = _numpy.asarray(_gvar.gvar(mean, covariance), dtype=object)
    return ModelAverageResult(
        prediction=prediction,
        weights={name: float(weights[name]) for name in names},
        mean=mean,
        covariance=covariance,
        within_covariance=within,
        between_covariance=between,
    )


def _weighted_conditional_variance(values, weights, metadata, axes):
    """Var(E[value | axes]) for a finite weighted model distribution."""
    if not axes:
        return _numpy.zeros(values.shape[1], dtype=float)
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(metadata):
        key = tuple(row[axis] for axis in axes)
        groups.setdefault(key, []).append(index)
    grand_mean = weights @ values
    result = _numpy.zeros(values.shape[1], dtype=float)
    for indices in groups.values():
        group_weight = float(weights[indices].sum())
        if group_weight <= 0.0:
            continue
        conditional_mean = weights[indices] @ values[indices] / group_weight
        result += group_weight * (conditional_mean - grand_mean) ** 2
    return result


def shapley_model_variance(
    model_means: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    metadata: Mapping[str, Mapping[str, object]],
    axes: Iterable[str],
) -> dict[str, _numpy.ndarray]:
    """Allocate between-model variance to analysis choices using Shapley effects.

    This is suitable for a crossed analysis multiverse and accounts for
    interactions without assigning them arbitrarily to whichever step happens
    to be varied first.  Contributions sum to the between-model variance (up
    to floating-point roundoff).  Missing/non-crossed model combinations are
    allowed, but the result then describes the supplied discrete model prior,
    not an independent-factor design.
    """
    names = tuple(model_means)
    if set(names) != set(weights) or set(names) != set(metadata):
        raise ValueError("means, weights, and metadata must contain the same models")
    axis_names = tuple(axes)
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("axes must be unique")
    for name in names:
        missing = set(axis_names) - set(metadata[name])
        if missing:
            raise ValueError(f"metadata for {name!r} is missing {sorted(missing)}")
    values = _numpy.vstack([_numpy.asarray(model_means[name], dtype=float).reshape(-1) for name in names])
    probability = _numpy.asarray([weights[name] for name in names], dtype=float)
    probability = probability / probability.sum()
    rows = [metadata[name] for name in names]
    count = len(axis_names)
    if count == 0:
        return {}

    cache: dict[tuple[str, ...], _numpy.ndarray] = {}
    def explained(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = _weighted_conditional_variance(values, probability, rows, key)
        return cache[key]

    result = {axis: _numpy.zeros(values.shape[1], dtype=float) for axis in axis_names}
    for axis in axis_names:
        others = tuple(item for item in axis_names if item != axis)
        for subset_size in range(len(others) + 1):
            coefficient = (
                factorial(subset_size)
                * factorial(count - subset_size - 1)
                / factorial(count)
            )
            for subset in combinations(others, subset_size):
                result[axis] += coefficient * (
                    explained((*subset, axis)) - explained(subset)
                )
    return result


def relative_partial_uncertainty_percent(variance, mean, floor=1e-15):
    """Return 100 sqrt(variance)/|mean| without implying variance fractions."""
    variance = _numpy.clip(_numpy.asarray(variance, dtype=float), 0.0, None)
    denominator = _numpy.maximum(_numpy.abs(_numpy.asarray(mean, dtype=float)), floor)
    return 100.0 * _numpy.sqrt(variance) / denominator
