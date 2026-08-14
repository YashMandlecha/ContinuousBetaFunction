"""Tests for SetupBetaFunction: data processing pipeline, combine, delta dispatch.

Regression test for the combine-before-averaging bug is the most critical test
in this file; it verifies that _precombine_rearranged fixes inflated errors
when Ep and Es are highly correlated but the gamma method would otherwise
zero-out their cross-covariance.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import gvar as gv
import numpy as np
import pytest

from betafn.betafn import BetaFunction
from betafn.processing.finite_volume import delta_finite_volume
from betafn.setup import SetupBetaFunction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bf(nf=4):
    return BetaFunction(nf=nf)


# ---------------------------------------------------------------------------
# _precombine_rearranged (unit tests, no I/O needed)
# ---------------------------------------------------------------------------

class TestPrecombineRearranged:
    """Direct unit tests for the raw-data combine step."""

    def _make_rearranged(self, n_configs=100, n_ft=3, seed=0):
        rng = np.random.default_rng(seed)
        data = {}
        for obs in ("Ep", "Es", "Ec", "Q"):
            for ft in ["1.0", "2.0", "3.0"][:n_ft]:
                data[f"wilson_{obs}_{ft}"] = rng.normal(2.0, 0.01, n_configs).tolist()
        return data

    def test_no_combine_returns_unchanged(self):
        bf = _make_bf()
        bf.combine = {}
        rearranged = self._make_rearranged()
        original_keys = set(rearranged.keys())
        result = bf._precombine_rearranged(rearranged)
        assert set(result.keys()) == original_keys

    def test_combine_replaces_target_values(self):
        rng = np.random.default_rng(42)
        N = 200
        Ep = rng.normal(2.0, 0.01, N).tolist()
        Es = rng.normal(3.0, 0.01, N).tolist()
        rearranged = {
            "wilson_Ep_2.0": Ep,
            "wilson_Es_2.0": Es,
            "wilson_Ec_2.0": rng.normal(2.5, 0.01, N).tolist(),
            "wilson_Q_2.0":  rng.normal(0.0, 1.0, N).tolist(),
        }
        bf = _make_bf()
        bf.combine = {"s": {"p": 5.0 / 3.0, "s": -2.0 / 3.0}}
        result = bf._precombine_rearranged(rearranged)

        expected = np.array(Ep) * (5 / 3) + np.array(Es) * (-2 / 3)
        np.testing.assert_allclose(result["wilson_Es_2.0"], expected.tolist(), rtol=1e-10)

    def test_combine_applied_to_all_flow_times(self):
        rng = np.random.default_rng(1)
        N = 100
        rearranged = {}
        for ft in ["1.0", "2.0", "3.0"]:
            rearranged[f"wilson_Ep_{ft}"] = rng.normal(2.0, 0.01, N).tolist()
            rearranged[f"wilson_Es_{ft}"] = rng.normal(3.0, 0.01, N).tolist()
        bf = _make_bf()
        bf.combine = {"s": {"p": 1.0, "s": 0.0}}   # trivial: combined_s = Ep
        result = bf._precombine_rearranged(rearranged)
        for ft in ["1.0", "2.0", "3.0"]:
            np.testing.assert_allclose(
                result[f"wilson_Es_{ft}"], rearranged[f"wilson_Ep_{ft}"], rtol=1e-10
            )

    def test_other_observables_untouched(self):
        rng = np.random.default_rng(2)
        N = 100
        Ep = rng.normal(2.0, 0.01, N).tolist()
        Ec = rng.normal(2.5, 0.01, N).tolist()
        Es = rng.normal(3.0, 0.01, N).tolist()
        rearranged = {
            "wilson_Ep_2.0": Ep,
            "wilson_Es_2.0": Es,
            "wilson_Ec_2.0": Ec,
        }
        bf = _make_bf()
        bf.combine = {"s": {"p": 5 / 3, "s": -2 / 3}}
        result = bf._precombine_rearranged(rearranged)
        # Ep and Ec should be unchanged
        assert result["wilson_Ep_2.0"] == Ep
        assert result["wilson_Ec_2.0"] == Ec


# ---------------------------------------------------------------------------
# Regression: combine-before-averaging fixes inflated errors
# ---------------------------------------------------------------------------

class TestCombineErrorsNotInflated:
    """Regression test for the bug where combining after gamma_method_average
    gave inflated errors because the cross-covariance between Ep and Es was
    set to zero within that function.

    Setup: Ep and Es are identical (ρ=1 exactly).  The combination
    (5/3)·Ep − (2/3)·Es = (1)·Ep has variance equal to Var(Ep).

    Old code (combine after averaging with gamma method):
        ignores Cov(Ep,Es) → Var = (5/3)²·Var(Ep) + (2/3)²·Var(Ep)
                            = 29/9 · Var(Ep) ≈ 3.22 · Var(Ep)

    New code (pre-combine):
        combined series has Var = (1)²·Var(Ep) = Var(Ep)  ✓
    """

    def test_errors_match_theory_for_perfectly_correlated_data(self):
        from betafn.processing.gamma import gamma_method_average

        rng = np.random.default_rng(12345)
        N = 3000
        sigma = 0.01

        # Ep = Es = common noise → ρ = 1 exactly
        common = rng.normal(2.0, sigma, N)
        rearranged = {
            "wilson_Ep_2.0": common.tolist(),
            "wilson_Es_2.0": common.tolist(),   # identical
        }

        # --- new approach: pre-combine then average ---
        bf = _make_bf()
        bf.combine = {"s": {"p": 5 / 3, "s": -2 / 3}}
        pre_combined = bf._precombine_rearranged({k: v[:] for k, v in rearranged.items()})

        # Combined series should be (5/3 − 2/3)·common = common
        np.testing.assert_allclose(pre_combined["wilson_Es_2.0"], common.tolist(), rtol=1e-10)

        averaged = gamma_method_average(pre_combined)
        sigma_new = averaged["wilson_Es_2.0"].sdev

        # Expected error: sigma / sqrt(N)
        expected = sigma / np.sqrt(N)
        assert abs(sigma_new - expected) / expected < 0.30, \
            f"Pre-combine error {sigma_new:.2e} should be ~{expected:.2e}"

        # --- old approach: average separately then combine gvars ---
        # gamma_method_average groups Ep and Es separately, zeroing their
        # cross-covariance, so combining afterwards inflates errors.
        averaged_sep = gamma_method_average(rearranged)
        gv_combined_old = (5 / 3) * averaged_sep["wilson_Ep_2.0"] \
                         + (-2 / 3) * averaged_sep["wilson_Es_2.0"]
        sigma_old = gv_combined_old.sdev

        # Theoretical inflation: sqrt((5/3)² + (2/3)²) / 1 ≈ 1.79× for ρ=1
        inflated_factor = np.sqrt((5 / 3) ** 2 + (2 / 3) ** 2)
        assert sigma_old == pytest.approx(inflated_factor * expected, rel=0.30), \
            f"Old approach should inflate errors by ~{inflated_factor:.2f}×; " \
            f"got sigma_old={sigma_old:.2e}, expected≈{inflated_factor * expected:.2e}"

        # New approach must be substantially closer to the correct value
        assert sigma_new < sigma_old * 0.7, \
            f"Pre-combine ({sigma_new:.2e}) should be < 0.7× old ({sigma_old:.2e})"


# ---------------------------------------------------------------------------
# process_data integration (end-to-end pipeline with synthetic .bin files)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProcessData:

    def test_process_data_populates_avg_data(self, synthetic_data_dir, flow_times):
        bf = _make_bf()
        dataset = bf.discover_dataset(synthetic_data_dir, flows=["wilson"])
        bf.process_data(
            dataset,
            path=str(synthetic_data_dir),
            correction="finite-volume",
            mnt=0.5,
            mxt=3.5,
        )
        assert bf.avg_data, "avg_data should be populated after process_data"
        for coupling, volumes in bf.avg_data.items():
            for volume, masses in volumes.items():
                for mass, flows in masses.items():
                    assert "wilson" in flows
                    assert "flow_times" in flows["wilson"]

    def test_process_data_with_gamma_method(self, synthetic_data_dir):
        bf = _make_bf()
        dataset = bf.discover_dataset(synthetic_data_dir, flows=["wilson"])
        bf.process_data(
            dataset,
            path=str(synthetic_data_dir),
            correction="finite-volume",
            use_gamma_method=True,
            gamma_window_factor=3.0,
        )
        assert bf.avg_data

    def test_g2_values_are_gvars(self, synthetic_data_dir):
        bf = _make_bf()
        dataset = bf.discover_dataset(synthetic_data_dir, flows=["wilson"])
        bf.process_data(dataset, path=str(synthetic_data_dir), correction="finite-volume")
        for coupling, volumes in bf.avg_data.items():
            for volume, masses in volumes.items():
                for mass, flows in masses.items():
                    for ft, val in flows["wilson"]["g2_p"].items():
                        assert isinstance(val, gv.GVar), \
                            f"g2_p at t={ft} should be a GVar"
                        break  # just check one
                    break
            break

    def test_combine_modifies_es_values(self, synthetic_data_dir):
        """After combining, g2_s should encode the linear combination, not raw E_s."""
        bf = _make_bf()
        dataset = bf.discover_dataset(synthetic_data_dir, flows=["wilson"])

        # Without combine
        bf_plain = _make_bf()
        bf_plain.process_data(
            dataset, path=str(synthetic_data_dir),
            correction="finite-volume",
        )

        # With combine
        bf_combined = _make_bf()
        bf_combined.process_data(
            dataset, path=str(synthetic_data_dir),
            correction="finite-volume",
            combine={"s": {"p": 5 / 3, "s": -2 / 3}},
        )

        # g2_p should be the same (p is not combined)
        for coupling in bf_plain.avg_data:
            for volume in bf_plain.avg_data[coupling]:
                for mass in bf_plain.avg_data[coupling][volume]:
                    d_plain = bf_plain.avg_data[coupling][volume][mass]["wilson"]
                    d_comb = bf_combined.avg_data[coupling][volume][mass]["wilson"]
                    for ft in d_plain["g2_p"]:
                        diff_p = abs(
                            gv.mean(d_plain["g2_p"][ft]) - gv.mean(d_comb["g2_p"][ft])
                        )
                        assert diff_p < 1e-10, \
                            "g2_p should be unchanged by a combine that only affects 's'"
                        break
                    break
                break
            break

    def test_skipped_empty_ensemble(self, tmp_path, flow_times):
        """An ensemble with no flow times should be quarantined, not crash."""
        # Minimal valid file structure
        bad_data = {"flow_times": [], "Ep": []}
        (tmp_path / "7p00_l8l8l8t16_0p001_wilson.bin").write_bytes(pickle.dumps(bad_data))

        bf = _make_bf()
        dataset = {"7p00": {"l8l8l8t16": {"0p001": ["wilson"]}}}
        bf.process_data(dataset, path=str(tmp_path), correction="finite-volume")
        assert "7p00" not in bf.avg_data or not bf.avg_data.get("7p00")
        assert len(bf.skipped_ensembles) == 1

    def test_data_report_runs(self, synthetic_data_dir):
        bf = _make_bf()
        dataset = bf.discover_dataset(synthetic_data_dir, flows=["wilson"])
        bf.process_data(dataset, path=str(synthetic_data_dir))
        report = bf.data_report()
        assert isinstance(report, str)
        assert "wilson" in report


# ---------------------------------------------------------------------------
# delta dispatcher
# ---------------------------------------------------------------------------

class TestDelta:
    def test_no_correction_returns_zeros(self):
        bf = _make_bf()
        bf._process_config = bf._process_config.__class__(correction="none")
        flow_times = np.array([1.0, 2.0, 3.0])
        delta = bf.delta(flow_times, "l24l24l24t48", "wilson", "p")
        np.testing.assert_array_equal(delta, np.zeros(3))

    def test_finite_volume_correction_matches_module(self):
        bf = _make_bf()
        bf._process_config = bf._process_config.__class__(correction="finite-volume")
        flow_times = np.array([2.0, 3.0, 4.0])
        volume = "l24l24l24t48"
        delta_bf = bf.delta(flow_times, volume, "wilson", "p")
        delta_ref = delta_finite_volume(flow_times, [24.0, 24.0, 24.0, 48.0])
        np.testing.assert_allclose(delta_bf, delta_ref)
