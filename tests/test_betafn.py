"""Tests for betafn.BetaFunction: instantiation, stage infrastructure,
pipeline helpers, and the perturbative_interpolation convenience wrapper.
"""
from __future__ import annotations

import numpy as np
import pytest

from betafn.betafn import BetaFunction
from betafn.base.specs import InterpolationSpec
from betafn.fitting.containers import StageStore


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_requires_nf(self):
        with pytest.raises(Exception):
            BetaFunction()   # nf is mandatory

    def test_default_nc(self):
        bf = BetaFunction(nf=4)
        assert bf.nc == 3

    def test_custom_nc(self):
        bf = BetaFunction(nf=4, nc=2)
        assert bf.nc == 2

    def test_default_gauge_action(self):
        bf = BetaFunction(nf=4)
        assert bf.gauge_action == "s"

    def test_stores_nf(self):
        bf = BetaFunction(nf=8)
        assert bf.nf == 8

    def test_repr_string(self):
        bf = BetaFunction(nf=4)
        r = repr(bf)
        assert "BetaFunction" in r
        assert "nf=4" in r


# ---------------------------------------------------------------------------
# Stage stores
# ---------------------------------------------------------------------------

class TestStageStores:
    @pytest.fixture
    def bf(self):
        return BetaFunction(nf=4)

    def test_all_stages_present(self, bf):
        for attr in ("chiral", "infinite_volume", "interpolation", "continuum"):
            assert hasattr(bf, attr)
            assert isinstance(getattr(bf, attr), StageStore)

    def test_stage_aliases_consistent(self, bf):
        assert bf.ch_fits is bf.chiral.fits
        assert bf.iv_fits is bf.infinite_volume.fits
        assert bf.ntrp_fits is bf.interpolation.fits
        assert bf.cnt_fits is bf.continuum.fits

    def test_available_stage_names(self, bf):
        names = bf.available_stage_names()
        for alias in ("chiral", "iv", "infinite_volume", "ntrp",
                      "interpolation", "cnt", "continuum"):
            assert alias in names

    def test_stage_store_lookup(self, bf):
        assert bf.stage_store("iv").name == "infinite_volume"
        assert bf.stage_store("ntrp").name == "interpolation"
        assert bf.stage_store("cnt").name == "continuum"

    def test_invalid_stage_name_raises(self, bf):
        from betafn.base.exceptions import BetaFunctionException
        with pytest.raises(BetaFunctionException):
            bf.stage_store("nonexistent")


# ---------------------------------------------------------------------------
# Analysis summary and quality table
# ---------------------------------------------------------------------------

class TestAnalysisSummary:
    @pytest.fixture
    def bf(self):
        return BetaFunction(nf=4)

    def test_empty_summary_schema(self, bf):
        summary = bf.analysis_summary()
        assert set(summary.keys()) == {"chiral", "infinite_volume", "interpolation", "continuum"}
        for stage, payload in summary.items():
            assert payload["n_fits"] == 0
            assert payload["mean_chi2_per_dof"] is None

    def test_quality_table_empty_before_fits(self, bf):
        for stage in ("chiral", "iv", "ntrp", "cnt"):
            assert bf.quality_table(stage) == []

    def test_stage_summary_with_injected_quality(self, bf):
        bf.chiral.quality = {
            "b7p00": {"vol": {"g2": {"wilson": {
                "p": {"2.0": {"chi2": 4.0, "dof": 2, "p-value": 0.1}}
            }}}}
        }
        summary = bf.stage_summary("chiral")
        assert summary["n_fits"] == 1
        assert summary["mean_chi2_per_dof"] == pytest.approx(2.0)
        assert summary["mean_pvalue"] == pytest.approx(0.1)

    def test_report_returns_string(self, bf):
        report = bf.report()
        assert isinstance(report, str)
        assert "BetaFunction" in report


# ---------------------------------------------------------------------------
# perturbative_interpolation convenience wrapper
# ---------------------------------------------------------------------------

class TestPerturbativeInterpolationWrapper:
    def test_returns_interpolation_spec(self):
        bf = BetaFunction(nf=4)
        spec = bf.perturbative_interpolation()
        assert isinstance(spec, InterpolationSpec)

    def test_uses_instance_nf(self):
        bf4 = BetaFunction(nf=4)
        bf8 = BetaFunction(nf=8)
        spec4 = bf4.perturbative_interpolation()
        spec8 = bf8.perturbative_interpolation()
        assert spec4.fcn is not spec8.fcn, \
            "Different nf should produce different model functions"

    def test_accepts_all_kwargs(self):
        bf = BetaFunction(nf=4)
        spec = bf.perturbative_interpolation(
            loops=3,
            correction_order=4,
            free_intercept=True,
            intercept_width=0.3,
            width=7.0,
            xerrors=True,
        )
        assert spec.xerrors is True


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_schema(self):
        bf = BetaFunction(nf=4)
        prov = bf.provenance()
        for key in ("timestamp", "python", "platform", "packages", "nc", "nf"):
            assert key in prov

    def test_provenance_nf_nc_match(self):
        bf = BetaFunction(nf=8, nc=2)
        prov = bf.provenance()
        assert prov["nf"] == 8
        assert prov["nc"] == 2
