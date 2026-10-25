from __future__ import annotations

import unittest
from unittest import mock
import tempfile

import gvar as gv
import lsqfit
import numpy as np
from scipy.stats import chi2

from betafn.fitting.goodness import (
    bruno_sommer_goodness_of_fit,
    generalized_chi_square_survival,
)
from betafn.fitting.model_average import (
    model_average_predictions,
    posterior_model_weights,
    shapley_model_variance,
)
from betafn.fitting.hybrid import (
    HybridInterpolationCandidate,
    HybridTarget,
    _candidate_priors,
    _match_fraction,
    default_hybrid_candidates,
    fit_hybrid_candidate,
    fit_hybrid_bma_candidates,
    hybrid_parameter_records,
)


class _TestPerturbativeBetaFunction:
    nrm = 4.0 * np.pi
    b = (0.05, 0.02, -0.01)

    def __call__(self, x, loops=3):
        return -self.nrm * sum(
            coefficient * (x / self.nrm) ** (order + 2)
            for order, coefficient in enumerate(self.b[:loops])
        )


class _TestBetaFunction:
    perturbative_beta_function = _TestPerturbativeBetaFunction()

    @staticmethod
    def _run_fit_with_x_errors(y, fit_fcn, prior, p0):
        def wrapper(parameters):
            return fit_fcn(parameters["x"], parameters)
        return lsqfit.nonlinear_fit(
            data=y, fcn=wrapper, prior=prior, p0=p0,
        )


class TestModelAverage(unittest.TestCase):
    def test_fitted_match_fraction_has_uniform_prior_parameterization(self):
        candidate = HybridInterpolationCandidate("fit4", 4, 2, 1, None, 1)
        prior, initial = _candidate_priors(
            candidate, target_count=1, weak_width=5.0,
            constant_width=0.25, strong_width=0.10, pole_log_width=0.70,
        )
        self.assertEqual(gv.mean(prior["match_latent"][0]), 0.0)
        self.assertEqual(gv.sdev(prior["match_latent"][0]), 1.0)
        self.assertEqual(initial["match_latent"], [0.0])
        fraction = _match_fraction(candidate, {"match_latent": [gv.gvar(0.0, 1.0)]})
        self.assertAlmostEqual(gv.mean(fraction), 0.5)
        self.assertAlmostEqual(
            gv.sdev(fraction), 0.6 / np.sqrt(2.0 * np.pi)
        )

        default = default_hybrid_candidates(
            weak_orders=(4,), weak_families=("fit4",),
            pade_orders=((2, 1),), continuities=(1,),
        )
        self.assertEqual(len(default), 1)
        self.assertIsNone(default[0].match_fraction)
        self.assertIn("match_fitted_0p20_0p80", default[0].name)

    def test_single_hybrid_order_may_be_given_as_an_integer(self):
        candidates = default_hybrid_candidates(
            weak_orders_by_family={"fit4": 4, "fit11": 5},
            pade_orders=((2, 1),),
            match_fractions=(0.5,),
            continuities=(1,),
        )
        selected = {(item.weak_family, item.weak_order) for item in candidates}
        self.assertEqual(selected, {("fit4", 4), ("fit11", 5)})

    def test_family_specific_hybrid_order_selection(self):
        candidates = default_hybrid_candidates(
            weak_orders_by_family={"fit4": (4, 5), "fit11": (5,)},
            pade_orders=((0, 0), (1, 1)),
            match_fractions=(0.5,),
            continuities=(1,),
        )
        selected = {(item.weak_family, item.weak_order) for item in candidates}
        self.assertEqual(selected, {("fit4", 4), ("fit4", 5), ("fit11", 5)})
        self.assertEqual(len(candidates), 6)

    def test_quadratic_pade_denominator_has_two_ordered_pole_parameters(self):
        candidate = HybridInterpolationCandidate(
            weak_family="fit4",
            weak_order=4,
            numerator_order=3,
            denominator_order=2,
        )
        prior, initial = _candidate_priors(
            candidate,
            target_count=1,
            weak_width=5.0,
            constant_width=0.25,
            strong_width=0.10,
            pole_log_width=0.70,
        )
        self.assertIn("logrho_0_0", prior)
        self.assertIn("logrho_step_0_1", prior)
        self.assertEqual(initial["logrho_0_0"], [0.0])
        self.assertEqual(initial["logrho_step_0_1"], [0.0])

    def test_parallel_hybrid_candidates_match_serial_results(self):
        mean = np.array([1.0, 2.0, 3.0, 4.0, -0.004, -0.016, -0.035, -0.061])
        covariance = np.diag(
            [0.01, 0.01, 0.015, 0.02, 2e-6, 2e-6, 3e-6, 4e-6]
        )
        covariance[0, 4] = covariance[4, 0] = -2e-5
        covariance[1, 5] = covariance[5, 1] = -2e-5
        values = gv.gvar(mean, covariance)
        target = HybridTarget(
            key=("wilson", "p", "5.0"),
            couplings=("20p0", "8p00", "7p50", "7p00"),
            regions=("weak", "weak", "strong", "strong"),
            x=np.asarray(values[:4], dtype=object),
            y=np.asarray(values[4:], dtype=object),
            match_low=2.0,
            match_high=3.0,
            strong_scale=2.0,
        )
        targets = {target.key: target}
        candidates = (
            HybridInterpolationCandidate("fit4", 1, 0, 0, None, 1),
            HybridInterpolationCandidate("fit4", 1, 0, 0, None, 2),
        )
        arguments = dict(
            beta_function=_TestBetaFunction(),
            window=(5.0, 6.0),
            flow="wilson",
            observables=("p",),
            weak_couplings=("20p0", "8p00"),
            strong_couplings=("7p50", "7p00"),
            candidates=candidates,
            progress=False,
        )
        with mock.patch(
            "betafn.fitting.hybrid.collect_hybrid_targets",
            return_value=targets,
        ):
            serial = fit_hybrid_bma_candidates(max_workers=1, **arguments)
        with mock.patch(
            "betafn.fitting.hybrid.collect_hybrid_targets",
            return_value=targets,
        ):
            parallel = fit_hybrid_bma_candidates(max_workers=2, **arguments)

        for candidate in candidates:
            name = candidate.name
            direct = fit_hybrid_candidate(
                arguments["beta_function"], targets, candidate,
            )
            self.assertAlmostEqual(
                direct.log_evidence,
                serial.candidate_fits[name].log_evidence,
                places=8,
            )
            self.assertAlmostEqual(
                serial.candidate_fits[name].log_evidence,
                parallel.candidate_fits[name].log_evidence,
                places=8,
            )
            serial_parameters = serial.candidate_fits[name].fit.p
            parallel_parameters = parallel.candidate_fits[name].fit.p
            self.assertEqual(set(serial_parameters), set(parallel_parameters))
            for key in serial_parameters:
                np.testing.assert_allclose(
                    gv.mean(serial_parameters[key]),
                    gv.mean(parallel_parameters[key]),
                    rtol=1e-8,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    gv.evalcov(serial_parameters[key]),
                    gv.evalcov(parallel_parameters[key]),
                    rtol=1e-8,
                    atol=1e-10,
                )
        np.testing.assert_allclose(
            list(serial.weights.values()),
            list(parallel.weights.values()),
            rtol=1e-10,
            atol=1e-12,
        )
        records = hybrid_parameter_records(serial)
        self.assertTrue(records)
        self.assertEqual({row["model"] for row in records}, {
            candidate.name for candidate in candidates
        })
        targeted_records = [row for row in records if row["operator"] is not None]
        self.assertEqual({row["operator"] for row in targeted_records}, {"p"})
        self.assertEqual({row["flow_time"] for row in targeted_records}, {5.0})
        self.assertTrue(all(np.isfinite(row["mean"]) for row in records))
        self.assertTrue(all(row["sdev"] >= 0.0 for row in records))
        self.assertTrue(all(np.isfinite(row["log_evidence"]) for row in records))
        self.assertTrue(all(0.0 < row["model_probability"] < 1.0 for row in records))
        match_rows = [row for row in records if row["parameter"] == "match_fraction"]
        self.assertEqual(len(match_rows), len(candidates))
        self.assertTrue(all(0.2 < row["mean"] < 0.8 for row in match_rows))

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_arguments = {
                **arguments,
                "candidates": candidates[:1],
                "checkpoint_dir": checkpoint_dir,
                "max_workers": 1,
            }
            with mock.patch(
                "betafn.fitting.hybrid.collect_hybrid_targets",
                return_value=targets,
            ):
                saved = fit_hybrid_bma_candidates(**checkpoint_arguments)
            with (
                mock.patch(
                    "betafn.fitting.hybrid.collect_hybrid_targets",
                    return_value=targets,
                ),
                mock.patch(
                    "betafn.fitting.hybrid._fit_prepared_candidate",
                    side_effect=AssertionError("valid checkpoint was not reused"),
                ),
            ):
                restored = fit_hybrid_bma_candidates(**checkpoint_arguments)
            name = candidates[0].name
            self.assertAlmostEqual(
                saved.candidate_fits[name].log_evidence,
                restored.candidate_fits[name].log_evidence,
                places=12,
            )

    def test_weights_are_stable_and_normalized(self):
        weights = posterior_model_weights({"a": -1001.0, "b": -1000.0})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["b"], weights["a"])

    def test_law_of_total_covariance(self):
        first = gv.gvar([0.0, 1.0], [[1.0, 0.2], [0.2, 4.0]])
        second = gv.gvar([2.0, 3.0], [[9.0, 0.1], [0.1, 1.0]])
        result = model_average_predictions(
            {"a": first, "b": second}, {"a": 0.25, "b": 0.75}
        )
        expected_mean = np.array([1.5, 2.5])
        expected_within = 0.25 * gv.evalcov(first) + 0.75 * gv.evalcov(second)
        displacement_a = gv.mean(first) - expected_mean
        displacement_b = gv.mean(second) - expected_mean
        expected_between = (
            0.25 * np.outer(displacement_a, displacement_a)
            + 0.75 * np.outer(displacement_b, displacement_b)
        )
        np.testing.assert_allclose(result.mean, expected_mean)
        np.testing.assert_allclose(result.within_covariance, expected_within)
        np.testing.assert_allclose(result.between_covariance, expected_between)
        np.testing.assert_allclose(gv.evalcov(result.prediction), expected_within + expected_between)

    def test_shapley_effects_sum_to_between_variance(self):
        means = {
            "00": [0.0], "01": [1.0], "10": [2.0], "11": [4.0],
        }
        weights = {name: 0.25 for name in means}
        metadata = {
            "00": {"a": 0, "b": 0}, "01": {"a": 0, "b": 1},
            "10": {"a": 1, "b": 0}, "11": {"a": 1, "b": 1},
        }
        effects = shapley_model_variance(means, weights, metadata, ("a", "b"))
        values = np.array([0.0, 1.0, 2.0, 4.0])
        np.testing.assert_allclose(effects["a"] + effects["b"], [np.var(values)])


class TestGeneralizedGoodness(unittest.TestCase):
    def test_equal_eigenvalues_reduce_to_chi_square(self):
        value = 5.2
        eigenvalues = np.ones(4)
        self.assertAlmostEqual(
            generalized_chi_square_survival(value, eigenvalues),
            chi2.sf(value, 4), places=12,
        )

    def test_correct_weighting_recovers_ordinary_dof(self):
        covariance = np.array([
            [2.0, 0.3, 0.1, 0.0],
            [0.3, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.5, 0.4],
            [0.0, 0.1, 0.4, 2.0],
        ])
        design = np.column_stack([np.ones(4), np.arange(4.0)])
        residual = np.array([0.2, -0.1, 0.05, -0.15])
        result = bruno_sommer_goodness_of_fit(
            residual, design, covariance, covariance
        )
        self.assertAlmostEqual(result.expectation, 2.0, places=10)
        self.assertAlmostEqual(result.effective_dof, 2.0, places=10)
        self.assertEqual(result.rank, 2)
        self.assertAlmostEqual(result.expectation_parameter_sdev, 0.0, places=14)

    def test_appendix_a_projector_uncertainty_for_nonlinear_model(self):
        rng = np.random.default_rng(7)
        data_count, parameter_count = 6, 2
        design = rng.normal(size=(data_count, parameter_count))
        hessian = rng.normal(size=(data_count, parameter_count, parameter_count))
        hessian = 0.5 * (hessian + hessian.swapaxes(1, 2))
        data_factor = rng.normal(size=(data_count, data_count))
        weight_factor = rng.normal(size=(data_count, data_count))
        covariance = data_factor @ data_factor.T
        weight_covariance = weight_factor @ weight_factor.T
        result = bruno_sommer_goodness_of_fit(
            rng.normal(size=data_count), design, covariance,
            weight_covariance, model_hessian=hessian,
        )
        self.assertGreater(result.expectation_parameter_sdev, 0.0)
        self.assertEqual(result.expectation_parameter_gradient.shape, (parameter_count,))
        self.assertEqual(result.parameter_covariance.shape, (parameter_count, parameter_count))


if __name__ == "__main__":
    unittest.main()
