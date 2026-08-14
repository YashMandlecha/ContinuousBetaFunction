"""Tests for betafn.fitting.families: polynomial_interpolation, perturbative_interpolation."""
from __future__ import annotations

import numpy as np
import pytest

from betafn.fitting.families import perturbative_interpolation, polynomial_interpolation
from betafn.base.specs import InterpolationSpec


# ---------------------------------------------------------------------------
# polynomial_interpolation
# ---------------------------------------------------------------------------

class TestPolynomialInterpolation:
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_returns_three_tuple(self, order):
        result = polynomial_interpolation(order)
        assert len(result) == 3, "Should return (fcn, prior, p0)"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_prior_has_correct_parameter_count(self, order):
        _, prior, p0 = polynomial_interpolation(order)
        assert len(prior) == order + 1
        assert len(p0) == order + 1

    def test_function_is_callable(self):
        fcn, _, p0 = polynomial_interpolation(2)
        # p0 values are plain floats; lsqfit expects lists
        p = {k: [v] for k, v in p0.items()}
        x = np.array([1.0, 2.0, 3.0])
        result = fcn(x, p)
        assert hasattr(result, "__len__") or np.isscalar(result)

    def test_wider_prior_with_larger_width(self):
        _, prior_narrow, _ = polynomial_interpolation(2, width=1.0)
        _, prior_wide, _ = polynomial_interpolation(2, width=10.0)
        import gvar as gv
        # Prior widths for non-intercept coefficients should be larger for larger width
        for key in prior_narrow:
            sdev_narrow = gv.sdev(prior_narrow[key][0])
            sdev_wide = gv.sdev(prior_wide[key][0])
            assert sdev_wide >= sdev_narrow


# ---------------------------------------------------------------------------
# perturbative_interpolation
# ---------------------------------------------------------------------------

class TestPerturbativeInterpolation:
    def test_returns_interpolation_spec(self):
        spec = perturbative_interpolation(nf=4)
        assert isinstance(spec, InterpolationSpec)

    def test_spec_is_callable(self):
        spec = perturbative_interpolation(nf=4)
        assert callable(spec.fcn)

    def test_prior_is_dict(self):
        spec = perturbative_interpolation(nf=4)
        assert isinstance(spec.prior, dict)
        assert len(spec.prior) > 0

    def test_different_nf_produces_different_objects(self):
        spec4 = perturbative_interpolation(nf=4)
        spec8 = perturbative_interpolation(nf=8)
        # At minimum they should be distinct objects
        assert spec4 is not spec8

    @pytest.mark.parametrize("loops", [2, 3])
    def test_loops_parameter_accepted(self, loops):
        spec = perturbative_interpolation(nf=4, loops=loops)
        assert isinstance(spec, InterpolationSpec)

    def test_free_intercept_adds_prior_entry(self):
        spec_fixed = perturbative_interpolation(nf=4, free_intercept=False)
        spec_free = perturbative_interpolation(nf=4, free_intercept=True)
        # Free intercept version should have at least as many prior entries
        assert len(spec_free.prior) >= len(spec_fixed.prior)

    def test_xerrors_flag(self):
        spec = perturbative_interpolation(nf=4, xerrors=True)
        assert spec.xerrors is True


# ---------------------------------------------------------------------------
# InterpolationSpec factory methods (mirror of test_base_specs but via families)
# ---------------------------------------------------------------------------

class TestInterpolationSpecRoundtrip:
    def test_polynomial_via_spec_classmethod(self):
        spec_direct = polynomial_interpolation(3)
        spec_via = InterpolationSpec.polynomial(order=3)
        # Both should produce functions with the same number of parameters
        assert len(spec_via.p0) == len(spec_direct[2])

    def test_perturbative_via_spec_classmethod(self):
        spec = InterpolationSpec.perturbative(nf=4)
        assert isinstance(spec, InterpolationSpec)
        assert callable(spec.fcn)
