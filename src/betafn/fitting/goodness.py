"""Generalized goodness-of-fit diagnostics of Bruno and Sommer.

Reference: M. Bruno and R. Sommer, Comput. Phys. Commun. 285 (2023)
108643, doi:10.1016/j.cpc.2022.108643.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as _numpy
from scipy.integrate import quad as _quad
from scipy.stats import chi2 as _chi2


@dataclass(frozen=True)
class GeneralizedChiSquare:
    observed: float
    expectation: float
    p_value: float
    eigenvalues: _numpy.ndarray
    effective_dof: float
    rank: int
    expectation_parameter_sdev: float
    expectation_parameter_gradient: _numpy.ndarray
    parameter_covariance: _numpy.ndarray


def _symmetric_root(matrix, inverse=False, rtol=1e-12):
    matrix = 0.5 * (_numpy.asarray(matrix, dtype=float) + _numpy.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = _numpy.linalg.eigh(matrix)
    scale = max(float(_numpy.max(_numpy.abs(eigenvalues))), 1.0)
    keep = eigenvalues > rtol * scale
    transformed = _numpy.zeros_like(eigenvalues)
    if inverse:
        transformed[keep] = 1.0 / _numpy.sqrt(eigenvalues[keep])
    else:
        transformed[keep] = _numpy.sqrt(eigenvalues[keep])
    return (eigenvectors * transformed) @ eigenvectors.T, int(keep.sum())


def generalized_chi_square_survival(value, eigenvalues, epsabs=2e-8, epsrel=2e-7):
    """Survival probability for ``sum(lambda_i * Z_i**2)``.

    A Gil-Pelaez inversion of the characteristic function is used.  The exact
    ordinary chi-square survival function is selected when all nonzero
    eigenvalues are numerically equal, which also provides a stable limiting
    case for correctly weighted least-squares fits.
    """
    lambdas = _numpy.asarray(eigenvalues, dtype=float)
    tolerance = 1e-12 * max(float(_numpy.max(_numpy.abs(lambdas), initial=0.0)), 1.0)
    lambdas = lambdas[lambdas > tolerance]
    if value <= 0.0:
        return 1.0
    if len(lambdas) == 0:
        return 0.0
    if _numpy.allclose(lambdas, lambdas[0], rtol=1e-10, atol=1e-13):
        return float(_chi2.sf(value / lambdas[0], len(lambdas)))

    # Map t in [0, infinity) to s in [0, 1).  Evaluating log(phi) avoids
    # overflow when many continuum points enter the quadratic form.
    def integrand(s):
        if s <= 0.0:
            return -value + float(lambdas.sum())
        if s >= 1.0:
            return 0.0
        t = s / (1.0 - s)
        log_phi = -0.5 * _numpy.log(1.0 - 2.0j * t * lambdas).sum()
        numerator = _numpy.imag(_numpy.exp(-1.0j * t * value + log_phi))
        return float(numerator / t / (1.0 - s) ** 2)

    integral, _ = _quad(
        integrand, 0.0, 1.0,
        epsabs=epsabs, epsrel=epsrel, limit=800,
        points=(0.5, 0.9, 0.99, 0.999),
    )
    cdf = 0.5 - integral / _numpy.pi
    return float(_numpy.clip(1.0 - cdf, 0.0, 1.0))


def bruno_sommer_goodness_of_fit(
    residual,
    design,
    data_covariance,
    weight_covariance,
    model_hessian=None,
    rtol=1e-12,
):
    """Calibrate a weighted least-squares statistic for its actual covariance.

    ``design`` is the model Jacobian at the fitted parameters and
    ``model_hessian[i, alpha, beta]`` is the second parameter derivative of
    model component ``i``.  The latter supplies the Appendix-A contribution
    from the parameter dependence of the fit projector.  It is identically
    zero for a model linear in its parameters.

    The formula is for an unconstrained local least-squares fit. Informative
    parameter priors must instead be represented as additional Gaussian
    observations before calling this routine.
    """
    residual = _numpy.asarray(residual, dtype=float).reshape(-1)
    design = _numpy.asarray(design, dtype=float)
    covariance = _numpy.asarray(data_covariance, dtype=float)
    weight_covariance = _numpy.asarray(weight_covariance, dtype=float)
    size = len(residual)
    if design.ndim != 2 or design.shape[0] != size:
        raise ValueError("design must have one row per residual")
    if covariance.shape != (size, size) or weight_covariance.shape != (size, size):
        raise ValueError("covariance matrices must be square and match the residual")
    parameter_count = design.shape[1]
    if model_hessian is None:
        model_hessian = _numpy.zeros((size, parameter_count, parameter_count))
    model_hessian = _numpy.asarray(model_hessian, dtype=float)
    if model_hessian.shape != (size, parameter_count, parameter_count):
        raise ValueError(
            "model_hessian must have shape (ndata, nparameters, nparameters)"
        )

    weight, _ = _symmetric_root(weight_covariance, inverse=True, rtol=rtol)
    weighted_design = weight @ design
    curvature_inverse = _numpy.linalg.pinv(
        weighted_design.T @ weighted_design, rcond=rtol
    )
    projection = weighted_design @ curvature_inverse @ weighted_design.T
    complement = _numpy.eye(size) - projection
    data_root, _ = _symmetric_root(covariance, inverse=False, rtol=rtol)
    nu = data_root @ weight @ complement @ weight @ data_root
    nu = 0.5 * (nu + nu.T)
    eigenvalues = _numpy.linalg.eigvalsh(nu)
    scale = max(float(_numpy.max(_numpy.abs(eigenvalues), initial=0.0)), 1.0)
    eigenvalues = _numpy.clip(eigenvalues, 0.0, None)
    eigenvalues[eigenvalues < rtol * scale] = 0.0

    observed = float(residual @ _numpy.linalg.pinv(weight_covariance, rcond=rtol) @ residual)
    expectation = float(eigenvalues.sum())
    second_moment_variance = float(2.0 * _numpy.square(eigenvalues).sum())
    effective_dof = (
        float(2.0 * expectation**2 / second_moment_variance)
        if second_moment_variance > 0.0 else 0.0
    )

    # Appendix A, Eqs. (A.3), (A.5), and (A.6).  C_w = W C W and
    # Cov(a) is the sandwich covariance induced by the actual data covariance.
    weighted_covariance = weight @ covariance @ weight
    parameter_covariance = (
        curvature_inverse
        @ weighted_design.T
        @ weighted_covariance
        @ weighted_design
        @ curvature_inverse
    )
    expectation_gradient = _numpy.zeros(parameter_count, dtype=float)
    for alpha in range(parameter_count):
        second_derivatives = model_hessian[:, :, alpha]
        k_alpha = (
            weight
            @ design
            @ curvature_inverse
            @ second_derivatives.T
            @ weight
        )
        expectation_gradient[alpha] = 2.0 * _numpy.trace(
            weighted_covariance @ k_alpha @ complement
        )
    expectation_parameter_variance = float(
        expectation_gradient @ parameter_covariance @ expectation_gradient
    )
    # Roundoff can make a theoretically zero quadratic form slightly negative.
    expectation_parameter_sdev = float(
        _numpy.sqrt(max(expectation_parameter_variance, 0.0))
    )
    return GeneralizedChiSquare(
        observed=observed,
        expectation=expectation,
        p_value=generalized_chi_square_survival(observed, eigenvalues),
        eigenvalues=eigenvalues,
        effective_dof=effective_dof,
        rank=int(_numpy.count_nonzero(eigenvalues)),
        expectation_parameter_sdev=expectation_parameter_sdev,
        expectation_parameter_gradient=expectation_gradient,
        parameter_covariance=parameter_covariance,
    )
