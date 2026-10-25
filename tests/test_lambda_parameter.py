from __future__ import annotations

import unittest

import gvar as gv
import numpy as np
from scipy.integrate import quad

from betafn.blinding import blind_lambda_estimate
from betafn.perturbative import PerturbativeBetaFunction
from betafn.weak_coupling import (
    constrained_smooth_match,
    figure11_integral_match,
    lambda_parameter_from_matched_beta,
    lambda_parameter_from_smooth_beta,
    smooth_blended_ratio,
)


class TestLambdaParameter(unittest.TestCase):
    @staticmethod
    def _pt_ratio(pt, x, loops=3):
        x = np.asarray(x, dtype=float)
        return sum(
            -coefficient * x**power / pt.nrm**(power + 1)
            for power, coefficient in enumerate(pt.b[:loops])
        )

    def test_blinding_scales_both_schemes_without_exposing_factor(self):
        estimate = {
            "lambda_gf_over_mu": {
                "central": 0.2,
                "plus_sigma": 0.22,
                "minus_sigma": 0.18,
            },
            "lambda_msbar_over_mu": {
                "central": 0.1,
                "plus_sigma": 0.11,
                "minus_sigma": 0.09,
            },
            "lambda_msbar_over_lambda_gf": 0.5,
        }
        from unittest.mock import patch

        with patch("betafn.blinding.load_lambda_blinding_factor", return_value=1.3):
            blinded = blind_lambda_estimate(estimate)

        self.assertEqual(blinded["lambda_gf_over_mu"]["central"], 0.26)
        self.assertEqual(blinded["lambda_msbar_over_mu"]["central"], 0.13)
        self.assertEqual(blinded["lambda_msbar_over_lambda_gf"], 0.5)
        self.assertTrue(blinded["blinded"])
        self.assertNotIn("blinding_factor", blinded)

    def test_pure_gauge_msbar_conversion_matches_published_value(self):
        pt = PerturbativeBetaFunction(nf=0, nc=3)
        published_ratio = 0.622 / 1.164
        self.assertAlmostEqual(
            pt.lambda_msbar_over_lambda_gf, published_ratio, places=3
        )

    def test_reference_scale_change_obeys_integrated_beta_function(self):
        pt = PerturbativeBetaFunction(nf=4, nc=3)

        def pt_over_g4(x, loops=3):
            x = np.asarray(x, dtype=float)
            safe = np.where(x == 0.0, 1.0, x)
            ratio = pt(safe, loops=loops) / safe**2
            if np.ndim(x) == 0:
                return float(-pt.b[0] / pt.nrm) if x == 0.0 else float(ratio)
            return np.where(x == 0.0, -pt.b[0] / pt.nrm, ratio)

        match_g2 = np.linspace(0.8, 1.2, 9)
        match_ratio = pt_over_g4(match_g2)
        matched = figure11_integral_match(
            match_g2, match_ratio, pt_over_g4, match_window=(0.8, 1.2)
        )
        # A dense exact curve isolates the RG identity from interpolation
        # discretization, which is intentionally present for real data.
        continuum_g2 = np.arange(0.9, 5.0, 0.01)
        continuum_ratio = pt_over_g4(continuum_g2)
        first = lambda_parameter_from_matched_beta(
            continuum_g2, continuum_ratio, matched, pt, reference_g2=4.0
        )
        second = lambda_parameter_from_matched_beta(
            continuum_g2, continuum_ratio, matched, pt, reference_g2=4.9
        )

        running, _ = quad(lambda x: 1.0 / pt(x, loops=3), 4.0, 4.9)
        observed = np.log(
            second["lambda_gf_over_mu"]["central"]
            / first["lambda_gf_over_mu"]["central"]
        )
        self.assertAlmostEqual(observed, -0.5 * running, places=6)

        for label in ("central", "plus_sigma", "minus_sigma"):
            self.assertGreater(first["lambda_gf_over_mu"][label], 0.0)
            self.assertGreater(first["lambda_msbar_over_mu"][label], 0.0)

    def test_smooth_match_recovers_pt_preserving_corrections(self):
        pt = PerturbativeBetaFunction(nf=4, nc=3)
        match_x = np.linspace(0.8, 1.2, 9)
        means = (
            self._pt_ratio(pt, match_x)
            + 0.002 * match_x**3 - 0.0004 * match_x**4
        )
        covariance = (2e-5) ** 2 * (
            0.35 + 0.65 * np.eye(len(match_x))
        )
        match_y = np.asarray(gv.gvar(means, covariance), dtype=object)

        result = constrained_smooth_match(
            match_x, match_y,
            lambda x, loops=3: self._pt_ratio(pt, x, loops),
            match_window=(0.8, 1.2),
        )
        self.assertAlmostEqual(gv.mean(result["parameters"]["c3"]), 0.002, places=8)
        self.assertAlmostEqual(gv.mean(result["parameters"]["c4"]), -0.0004, places=8)

        continuum_x = np.linspace(0.8, 4.0, 33)
        continuum_y = np.asarray([
            gv.gvar(value, 2e-5)
            for value in (
                self._pt_ratio(pt, continuum_x)
                + 0.002 * continuum_x**3 - 0.0004 * continuum_x**4
            )
        ], dtype=object)
        endpoint_values = smooth_blended_ratio(
            np.asarray([0.8, 1.2]), result, continuum_x, continuum_y
        )
        self.assertAlmostEqual(
            gv.mean(endpoint_values[0]),
            gv.mean(result["weak_ratio_curve"]([0.8])[0]),
            places=12,
        )
        self.assertAlmostEqual(
            gv.mean(endpoint_values[1]),
            gv.mean(continuum_y[4]),
            places=12,
        )

        step = 1e-5
        around_lower = gv.mean(smooth_blended_ratio(
            np.asarray([0.8 - step, 0.8, 0.8 + step]),
            result, continuum_x, continuum_y,
        ))
        around_upper = gv.mean(smooth_blended_ratio(
            np.asarray([1.2 - step, 1.2, 1.2 + step]),
            result, continuum_x, continuum_y,
        ))
        self.assertAlmostEqual(
            (around_lower[1] - around_lower[0]) / step,
            (around_lower[2] - around_lower[1]) / step,
            places=5,
        )
        self.assertAlmostEqual(
            (around_upper[1] - around_upper[0]) / step,
            (around_upper[2] - around_upper[1]) / step,
            places=5,
        )

        estimate = lambda_parameter_from_smooth_beta(
            continuum_x, continuum_y, result, pt,
            reference_g2=4.0, integration_points=401,
        )
        self.assertGreater(gv.mean(estimate["lambda_gf_over_mu"]), 0.0)
        self.assertGreater(gv.sdev(estimate["lambda_gf_over_mu"]), 0.0)


if __name__ == "__main__":
    unittest.main()
