"""Standard interpolation model factories that return InterpolationSpec objects."""
from __future__ import annotations

import gvar as _gvar
import numpy as _numpy

from ..base.exceptions import BetaFunctionException
from ..perturbative import PerturbativeBetaFunction


def polynomial_interpolation(order: int, width: float = 10.0) -> tuple:
    """Build (fcn, prior, p0) for a polynomial interpolation in g^2.

    Returns a ready-to-use triple for ``iv_ntrp`` or ``InterpolationSpec``,
    avoiding hand-written boilerplate for the most common interpolation family.
    Wrap with ``InterpolationSpec.polynomial(order)`` for the declarative form.
    """
    if order < 1:
        raise BetaFunctionException("Polynomial interpolation order must be at least 1.")

    names = [f"c{power}" for power in range(order + 1)]

    def fcn(x, p):
        return sum(p[name][0] * x**power for power, name in enumerate(names))

    prior = {name: [_gvar.gvar(0.0, width)] for name in names}
    p0 = {name: 0.0 for name in names}
    return fcn, prior, p0


def perturbative_interpolation(
    nf: float | int,
    nc: float | int = 3,
    loops: int = 2,
    correction_order: int = 2,
    free_intercept: bool = False,
    intercept_width: float = 0.2,
    width: float = 5.0,
    xerrors: bool = False,
):
    """PT-constrained interpolation model in g^2.

    Returns an ``InterpolationSpec`` parametrizing

        beta(g^2) = beta_PT^(loops)(g^2) × (c0 + Σ_n c_n u^n),   u = g²/(4π).

    When *free_intercept* is False (default) c0 is fixed to 1, so the
    universal perturbative coefficients are recovered exactly at weak coupling.
    When *free_intercept* is True, c0 carries prior gvar(1, intercept_width),
    absorbing a constant fractional offset without breaking asymptotic freedom.

    Parameters
    ----------
    nf, nc : physics parameters used to instantiate PerturbativeBetaFunction.
    loops : int
        Number of perturbative loops in the base function (1, 2, or 3).
    correction_order : int
        Number of polynomial correction terms beyond c0.
    free_intercept : bool
        Allow c0 to float with prior width *intercept_width*.
    intercept_width : float
        Prior width on c0 when *free_intercept* is True.
    width : float
        Prior width on correction coefficients c_n.
    xerrors : bool
        Whether to treat g^2 values as uncertain in the fit.
    """
    from ..base.specs import InterpolationSpec  # lazy — avoids circular import

    if loops not in (1, 2, 3):
        raise BetaFunctionException("loops must be 1, 2, or 3")
    if correction_order < 1:
        raise BetaFunctionException("correction_order must be at least 1")

    perturbative = PerturbativeBetaFunction(nf=nf, nc=nc)
    names = [f"pt_c{n}" for n in range(1, correction_order + 1)]

    def fcn(x, p):
        u = x / perturbative.nrm
        correction = p["pt_c0"][0] if free_intercept else 1.0
        term = 1.0
        for name in names:
            term = term * u
            correction = correction + p[name][0] * term
        return perturbative(x, loops=loops) * correction

    prior = {name: [_gvar.gvar(0.0, width)] for name in names}
    p0 = {name: 0.0 for name in names}
    if free_intercept:
        prior["pt_c0"] = [_gvar.gvar(1.0, intercept_width)]
        p0["pt_c0"] = 1.0

    return InterpolationSpec(fcn=fcn, prior=prior, p0=p0, xerrors=xerrors)
