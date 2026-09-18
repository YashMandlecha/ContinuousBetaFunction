from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import quad

from betafn.perturbative import PerturbativeBetaFunction
from betafn.weak_coupling import (
    figure11_integral_match,
    lambda_parameter_from_matched_beta,
)


class TestLambdaParameter(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
