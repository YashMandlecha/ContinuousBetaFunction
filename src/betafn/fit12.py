"""Fit 12 models with the perturbative continuum limit imposed exactly.

Fit 12 allows an additive beta-function constant at nonzero lattice spacing.
Its lattice-spacing dependence is constrained so that the additive term
vanishes in the continuum limit, while the multiplicative intercept is fixed
to one at every lattice spacing.
"""

from __future__ import annotations

import numpy as np


def interpolation_parameter_names(order: int) -> tuple[str, ...]:
    """Parameter names for one finite-flow-time interpolation."""
    _validate_order(order)
    return ("beta_const",) + tuple(
        f"pt_c{power}" for power in range(1, order + 1)
    )


def interpolation_p0(order: int) -> dict[str, list[float]]:
    """Prior-free starting point for a finite-flow-time interpolation."""
    return {name: [0.0] for name in interpolation_parameter_names(order)}


def interpolation_beta(x, parameters, perturbative, order: int):
    """Evaluate beta = a0 + beta_PT3 * (1 + sum(c_n u^n))."""
    x = np.asarray(x)
    u = x / perturbative.nrm
    correction = 1.0 + sum(
        parameters[f"pt_c{power}"][0] * u**power
        for power in range(1, order + 1)
    )
    return parameters["beta_const"][0] + perturbative(
        x, loops=3
    ) * correction


def joint_parameter_names(order: int) -> tuple[str, ...]:
    """Parameters in the joint (g^2, a^2/t) continuum model."""
    _validate_order(order)
    return ("A0",) + tuple(
        f"c{power}" for power in range(1, order + 1)
    )


def joint_design_matrix(g2, z, perturbative, order: int) -> np.ndarray:
    """Return the design matrix for beta(g2,z) minus beta_PT3(g2).

    The column ordering is ``joint_parameter_names(order)`` and the full
    model is

      beta(g2,z) = z*A0 + beta_PT3(g2)
                   * (1 + sum(c_n*u^n)).
    """
    _validate_order(order)
    g2 = np.asarray(g2, dtype=float)
    z = np.asarray(z, dtype=float)
    g2, z = np.broadcast_arrays(g2, z)
    u = g2 / perturbative.nrm
    pt = np.asarray(perturbative(g2, loops=3), dtype=float)
    columns = [z]
    columns.extend(pt * u**power for power in range(1, order + 1))
    return np.column_stack(columns)


def joint_beta(g2, z, parameters, perturbative, order: int):
    """Evaluate the constrained finite-spacing Fit 12 ansatz."""
    names = joint_parameter_names(order)
    coefficients = np.asarray(
        [parameters[name][0] for name in names], dtype=object
    )
    design = joint_design_matrix(g2, z, perturbative, order)
    return np.asarray(perturbative(g2, loops=3), dtype=object) + design.dot(
        coefficients
    )


def continuum_beta(g2, parameters, perturbative, order: int):
    """Evaluate Fit 12 at z=0, where a0=0 and c0=1 exactly."""
    return joint_beta(g2, np.zeros_like(np.asarray(g2, dtype=float)), parameters,
                      perturbative, order)


def lattice_slope(g2, parameters, perturbative, order: int):
    """Return d beta / dz for the Fit 12 continuum-panel line."""
    g2 = np.asarray(g2, dtype=float)
    return np.ones_like(g2, dtype=float) * parameters["A0"][0]


def solve_linear_gls(designs, targets, covariances):
    """Solve a block-weighted linear model and return response matrices.

    ``targets`` are central values after subtracting the fixed three-loop
    perturbative contribution.  The returned gains map each target block to
    the fitted coefficients and are also used to propagate the original gvar
    correlations in the batch runner.
    """
    if not (len(designs) == len(targets) == len(covariances)):
        raise ValueError("designs, targets, and covariances must have equal length")
    if not designs:
        raise ValueError("at least one GLS block is required")

    designs = [np.asarray(design, dtype=float) for design in designs]
    targets = [np.asarray(target, dtype=float) for target in targets]
    covariances = [np.asarray(covariance, dtype=float) for covariance in covariances]
    n_parameters = designs[0].shape[1]
    information = np.zeros((n_parameters, n_parameters))
    rhs = np.zeros(n_parameters)
    weights = []
    n_data = 0
    for design, target, covariance in zip(designs, targets, covariances):
        if design.ndim != 2 or design.shape[1] != n_parameters:
            raise ValueError("all design matrices must have the same column count")
        if design.shape[0] != len(target):
            raise ValueError("each design and target block must have equal rows")
        if covariance.shape != (len(target), len(target)):
            raise ValueError("each covariance must be square with one row per target")
        covariance = 0.5 * (covariance + covariance.T)
        weight = np.linalg.pinv(covariance, hermitian=True)
        weights.append(weight)
        information += design.T @ weight @ design
        rhs += design.T @ weight @ target
        n_data += len(target)

    # Normalize columns before inversion.  Powers of u=g^2/(4*pi) differ by
    # several orders of magnitude at order 4; solving the raw normal equations
    # would therefore be unnecessarily ill-conditioned even when the design is
    # full rank.  Transforming back leaves the physical coefficients unchanged.
    column_scales = np.sqrt(np.clip(np.diag(information), 0.0, None))
    if np.any(column_scales == 0.0):
        raise ValueError("GLS design contains an unconstrained parameter column")
    normalized_information = information / np.outer(
        column_scales, column_scales
    )
    normalized_rhs = rhs / column_scales
    rank = int(np.linalg.matrix_rank(normalized_information))
    normalized_inverse = np.linalg.pinv(
        normalized_information, hermitian=True
    )
    information_inverse = normalized_inverse / np.outer(
        column_scales, column_scales
    )
    information_inverse = 0.5 * (
        information_inverse + information_inverse.T
    )
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
        information_inverse
    )
    information_inverse = (
        covariance_eigenvectors
        * np.clip(covariance_eigenvalues, 0.0, None)
    ) @ covariance_eigenvectors.T
    information_inverse = 0.5 * (
        information_inverse + information_inverse.T
    )
    coefficients = information_inverse @ rhs
    gains = [
        information_inverse @ design.T @ weight
        for design, weight in zip(designs, weights)
    ]
    chi2 = 0.0
    for design, target, weight in zip(designs, targets, weights):
        residual = target - design @ coefficients
        chi2 += float(residual @ weight @ residual)
    return {
        "coefficients": coefficients,
        "information_inverse": information_inverse,
        "gains": gains,
        "weights": weights,
        "rank": rank,
        "normalized_condition": float(np.linalg.cond(normalized_information)),
        "n_data": n_data,
        "chi2": chi2,
    }


def _validate_order(order: int) -> None:
    if not isinstance(order, int) or order < 1:
        raise ValueError("Fit 12 order must be a positive integer")
