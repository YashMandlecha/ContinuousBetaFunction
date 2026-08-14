"""Perturbative GF-scheme beta function from JHEP06(2019)121."""
from __future__ import annotations

import numpy as _numpy
from scipy.special import zeta as _zeta


class PerturbativeBetaFunction:
    """Perturbative gradient-flow beta function from JHEP06(2019)121."""

    def __init__(self, nf: float | int, nc: float | int):
        tr = 0.5
        tf = tr * nf
        cf = 0.5 * (nc * nc - 1.0) / nc
        ca = nc

        b0 = 11.0 * ca / 3.0 - 4.0 * tf / 3.0
        b1 = 34.0 * ca * ca / 3.0 - (4.0 * cf + 20.0 * ca / 3.0) * tf
        b2 = 2857.0 * ca * ca * ca / 54.0 - 1415.0 * ca * ca * tf / 27.0
        b2 += (-205.0 * cf * ca / 9.0 + 2.0 * cf * cf) * tf
        b2 += (44.0 * cf / 9.0 + 158.0 * ca / 27.0) * tf * tf

        rho = 1.0 / 8.0
        e10 = (52.0 / 9.0 + 22.0 * _numpy.log(2.0) / 3.0 - 3.0 * _numpy.log(3.0)) * ca
        e10 -= 8.0 * tf / 9.0
        e20 = 27.9786 * ca * ca - 31.5652 * tf * ca + (16.0 * _zeta(3.0) - 43.0 / 3.0) * tf * cf
        e20 += (8.0 * _numpy.pi * _numpy.pi / 27.0 - 80.0 / 81.0) * tf * tf

        log_factor = _numpy.log(2.0 * rho) + _numpy.euler_gamma
        e1 = e10 + b0 * log_factor
        e2 = e20 + (2.0 * b0 * e10 + b1) * log_factor + (b0 * log_factor) * (b0 * log_factor)
        b2 = b2 - e1 * b1 + (e2 - e1 * e1) * b0

        self.nrm = 4.0 * _numpy.pi
        self.b = [coef / self.nrm ** (order + 1) for order, coef in enumerate([b0, b1, b2])]

    def __call__(self, x, loops: int = 3):
        beta_function = sum(coef * (x / self.nrm) ** (order + 2) for order, coef in enumerate(self.b) if order < loops)
        return -self.nrm * beta_function
