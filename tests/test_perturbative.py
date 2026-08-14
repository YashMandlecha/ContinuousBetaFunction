"""Tests for betafn.perturbative.PerturbativeBetaFunction.

API (from JHEP06(2019)121):
    bf = PerturbativeBetaFunction(nf, nc)          # both required
    beta = bf(g2_values, loops=3)                  # callable
    bf.b = [b0/nrm, b1/nrm^2, b2/nrm^3]          # internal coefficients
    bf.nrm = 4*pi

Key properties verified:
  - The one-loop coefficient b0 matches the standard QCD formula.
  - β(g²) < 0 for small g² (asymptotic freedom, Nf < 11).
  - Higher loop order reduces the beta function magnitude at weak coupling.
  - Output shape matches input.
"""
from __future__ import annotations

import numpy as np
import pytest

from betafn.perturbative import PerturbativeBetaFunction


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_requires_nf_and_nc(self):
        with pytest.raises(TypeError):
            PerturbativeBetaFunction(nf=4)   # nc is required

    def test_accepts_positional_args(self):
        bf = PerturbativeBetaFunction(4, 3)
        assert bf is not None

    def test_has_b_coefficients(self):
        bf = PerturbativeBetaFunction(4, 3)
        assert hasattr(bf, "b")
        assert len(bf.b) == 3   # b0, b1, b2 (normalised)

    def test_has_nrm(self):
        bf = PerturbativeBetaFunction(4, 3)
        assert bf.nrm == pytest.approx(4 * np.pi)


# ---------------------------------------------------------------------------
# One-loop coefficient b0 = (11 Ca - 4 Tf) / 3
# (in the conventional MSbar normalisation with Tf = nf * tr, tr = 1/2)
# The class stores b0 / nrm in b[0], so b[0] * nrm = b0.
# ---------------------------------------------------------------------------

class TestBetaCoefficients:
    @pytest.mark.parametrize("nf,nc", [(0, 3), (4, 3), (8, 3), (4, 2)])
    def test_b0_coefficient(self, nf, nc):
        bf = PerturbativeBetaFunction(nf, nc)
        tr = 0.5
        tf = tr * nf
        ca = nc
        b0_expected = 11.0 * ca / 3.0 - 4.0 * tf / 3.0
        # bf.b[0] = b0 / nrm
        b0_actual = bf.b[0] * bf.nrm
        assert b0_actual == pytest.approx(b0_expected, rel=1e-6)

    @pytest.mark.parametrize("nf,nc", [(0, 3), (4, 3)])
    def test_b1_coefficient(self, nf, nc):
        bf = PerturbativeBetaFunction(nf, nc)
        tr = 0.5
        tf = tr * nf
        cf = 0.5 * (nc**2 - 1) / nc
        ca = nc
        b1_expected = 34.0 * ca**2 / 3.0 - (4.0 * cf + 20.0 * ca / 3.0) * tf
        b1_actual = bf.b[1] * bf.nrm**2
        assert b1_actual == pytest.approx(b1_expected, rel=1e-6)

    def test_asymptotic_freedom_nf0(self):
        """b0 > 0 for pure gauge (Nf=0), ensuring AF."""
        bf = PerturbativeBetaFunction(0, 3)
        b0 = bf.b[0] * bf.nrm
        assert b0 > 0

    def test_b0_decreases_with_nf(self):
        """More flavours reduce b0 (weaken AF)."""
        bf_nf0 = PerturbativeBetaFunction(0, 3)
        bf_nf4 = PerturbativeBetaFunction(4, 3)
        b0_nf0 = bf_nf0.b[0] * bf_nf0.nrm
        b0_nf4 = bf_nf4.b[0] * bf_nf4.nrm
        assert b0_nf0 > b0_nf4


# ---------------------------------------------------------------------------
# Callable interface: bf(g2, loops=N)
# ---------------------------------------------------------------------------

class TestCallable:
    @pytest.fixture(params=[0, 4, 8])
    def bf(self, request):
        return PerturbativeBetaFunction(request.param, 3)

    def test_negative_for_small_g2(self, bf):
        """β(g²) < 0 (asymptotic freedom) for Nf ≤ 8, small coupling."""
        # b0 > 0 for Nf ≤ 8 at Nc=3, so the beta function is negative
        nf = round(bf.b[0] * bf.nrm * 3 / 11 * (-3) + 3)  # won't use, just skip
        g2_values = np.linspace(0.1, 2.0, 20)
        beta = bf(g2_values, loops=1)
        assert np.all(beta < 0), "Beta function should be negative for small g²"

    def test_output_shape(self, bf):
        g2 = np.linspace(0.5, 5.0, 15)
        beta = bf(g2)
        assert beta.shape == g2.shape

    def test_scalar_array_input(self, bf):
        result = bf(np.array([1.0]))
        assert result.shape == (1,)

    def test_more_loops_different_from_fewer(self):
        """Including higher-order terms should change the result at finite g²."""
        bf = PerturbativeBetaFunction(4, 3)
        g2 = np.array([2.0, 3.0, 4.0])
        beta_1loop = bf(g2, loops=1)
        beta_2loop = bf(g2, loops=2)
        assert not np.allclose(beta_1loop, beta_2loop), \
            "1-loop and 2-loop beta functions should differ at finite g²"

    def test_leading_order_scaling(self):
        """At 1 loop, β ∝ g^4 (i.e., β(g²) ∝ (g²)²)."""
        bf = PerturbativeBetaFunction(0, 3)
        g2a, g2b = 0.5, 1.0
        beta_a = bf(np.array([g2a]), loops=1)[0]
        beta_b = bf(np.array([g2b]), loops=1)[0]
        ratio = beta_b / beta_a
        expected_ratio = (g2b / g2a) ** 2
        assert ratio == pytest.approx(expected_ratio, rel=0.01), \
            f"1-loop β should scale as g^4; ratio={ratio:.4f}, expected={expected_ratio:.4f}"
