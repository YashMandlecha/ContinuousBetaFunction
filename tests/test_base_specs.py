"""Tests for betafn.base.specs: InterpolationSpec, AnalysisConfig."""
from __future__ import annotations

import numpy as np
import pytest

from betafn.base.specs import AnalysisConfig, InterpolationSpec


# ---------------------------------------------------------------------------
# InterpolationSpec — polynomial
# ---------------------------------------------------------------------------

class TestInterpolationSpecPolynomial:
    def test_construction(self):
        spec = InterpolationSpec.polynomial(order=3)
        assert callable(spec.fcn)
        assert spec.prior is not None
        assert spec.p0 is not None

    def test_parameter_count_matches_order(self):
        for order in (2, 3, 4, 5):
            spec = InterpolationSpec.polynomial(order=order)
            n_params = len(spec.p0)
            # polynomial of order N has N+1 coefficients
            assert n_params == order + 1, f"order={order} gave {n_params} params"

    def test_xerrors_flag_passes_through(self):
        spec = InterpolationSpec.polynomial(order=2, xerrors=True)
        assert spec.xerrors is True

    def test_fcn_evaluates_at_scalar(self):
        spec = InterpolationSpec.polynomial(order=2)
        # p0 values are plain floats; wrap them in lists as lsqfit expects
        p = {k: [v] for k, v in spec.p0.items()}
        # Should not raise; we just want it callable
        _ = spec.fcn(np.array([1.0, 2.0]), p)


# ---------------------------------------------------------------------------
# InterpolationSpec — perturbative
# ---------------------------------------------------------------------------

class TestInterpolationSpecPerturbative:
    def test_construction_returns_spec(self):
        spec = InterpolationSpec.perturbative(nf=4, loops=2, correction_order=2)
        assert isinstance(spec, InterpolationSpec)
        assert callable(spec.fcn)

    def test_prior_contains_intercept_when_free(self):
        spec = InterpolationSpec.perturbative(nf=4, free_intercept=True)
        assert any("c0" in k or "intercept" in k for k in spec.prior)

    def test_different_nf_gives_different_fcns(self):
        spec4 = InterpolationSpec.perturbative(nf=4)
        spec8 = InterpolationSpec.perturbative(nf=8)
        # The two specs should differ (they encode nf in the closure)
        assert spec4.fcn is not spec8.fcn


# ---------------------------------------------------------------------------
# AnalysisConfig — validation
# ---------------------------------------------------------------------------

class TestAnalysisConfig:
    @pytest.fixture
    def minimal_spec(self):
        return InterpolationSpec.polynomial(order=3)

    def test_valid_construction(self, minimal_spec):
        cfg = AnalysisConfig(
            data_path=".",
            interpolation=minimal_spec,
            continuum_window=(2.0, 6.0),
            g2_grid=(0.0, 15.0, 0.2),
        )
        assert cfg.correction == "finite-volume"
        assert cfg.binsize == 1

    def test_invalid_continuum_window_raises(self, minimal_spec):
        with pytest.raises(Exception):
            AnalysisConfig(
                data_path=".",
                interpolation=minimal_spec,
                continuum_window=(6.0, 2.0),   # mnt > mxt
                g2_grid=(0.0, 15.0, 0.2),
            )

    def test_equal_continuum_window_raises(self, minimal_spec):
        with pytest.raises(Exception):
            AnalysisConfig(
                data_path=".",
                interpolation=minimal_spec,
                continuum_window=(4.0, 4.0),
                g2_grid=(0.0, 15.0, 0.2),
            )

    def test_invalid_g2_grid_raises(self, minimal_spec):
        with pytest.raises(Exception):
            AnalysisConfig(
                data_path=".",
                interpolation=minimal_spec,
                continuum_window=(2.0, 6.0),
                g2_grid=(15.0, 0.0, 0.2),   # mng2 > mxg2
            )

    def test_zero_binsize_raises(self, minimal_spec):
        with pytest.raises(Exception):
            AnalysisConfig(
                data_path=".",
                interpolation=minimal_spec,
                continuum_window=(2.0, 6.0),
                g2_grid=(0.0, 15.0, 0.2),
                binsize=0,
            )

    def test_invalid_error_mode_raises(self, minimal_spec):
        with pytest.raises(Exception):
            AnalysisConfig(
                data_path=".",
                interpolation=minimal_spec,
                continuum_window=(2.0, 6.0),
                g2_grid=(0.0, 15.0, 0.2),
                error_mode="banana",
            )

    def test_describe_returns_dict_with_expected_keys(self, minimal_spec):
        cfg = AnalysisConfig(
            data_path=".",
            interpolation=minimal_spec,
            continuum_window=(2.0, 6.0),
            g2_grid=(0.0, 15.0, 0.2),
        )
        desc = cfg.describe()
        for key in ("data_path", "flows", "correction", "binsize",
                    "continuum_window", "g2_grid"):
            assert key in desc, f"Missing key '{key}' in describe()"

    def test_describe_no_tln_path(self, minimal_spec):
        """tln_path was removed; describe() should not include it."""
        cfg = AnalysisConfig(
            data_path=".",
            interpolation=minimal_spec,
            continuum_window=(2.0, 6.0),
            g2_grid=(0.0, 15.0, 0.2),
        )
        assert "tln_path" not in cfg.describe()
