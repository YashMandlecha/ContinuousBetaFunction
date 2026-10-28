from __future__ import annotations

import unittest
from unittest import mock
import tempfile
from pathlib import Path

import gvar as gv
import lsqfit
import numpy as np
from scipy.stats import chi2

from betafn import BetaFunction

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
    _hybrid_beta,
    _match_fraction,
    default_hybrid_candidates,
    fit_hybrid_candidate,
    fit_hybrid_bma_by_operator,
    fit_hybrid_bma_by_target,
    fit_hybrid_bma_candidates,
    hybrid_parameter_records,
    run_hybrid_bma_by_target,
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
    def test_eq7_gap_blend_is_c2_at_both_endpoints(self):
        candidate = HybridInterpolationCandidate(
            "fit4", 1, 1, 2, None, 2, strong_form="eq7_blend",
        )
        parameters = {
            "c_0_1": [0.15],
            "logp0_0_0": [np.log(0.05277)],
            "p_0_1": [0.30],
            "logrho_0_0": [0.0],
            "logrho_step_0_1": [0.0],
        }
        target = HybridTarget(
            key=("wilson", "p", "5.0"),
            couplings=("8p00", "7p50"),
            regions=("weak", "strong"),
            x=np.asarray([5.0, 8.0]),
            y=np.asarray([-0.1, -0.2]),
            match_low=5.0,
            match_high=8.0,
            strong_scale=3.0,
        )
        beta_function = _TestBetaFunction()

        def evaluate(value):
            return float(np.asarray(_hybrid_beta(
                beta_function, np.asarray([value]), parameters,
                candidate, target, 0,
            )).item())

        step = 3e-4
        for endpoint in (target.match_low, target.match_high):
            left_first = (
                3.0 * evaluate(endpoint)
                - 4.0 * evaluate(endpoint - step)
                + evaluate(endpoint - 2.0 * step)
            ) / (2.0 * step)
            right_first = (
                -3.0 * evaluate(endpoint)
                + 4.0 * evaluate(endpoint + step)
                - evaluate(endpoint + 2.0 * step)
            ) / (2.0 * step)
            left_second = (
                2.0 * evaluate(endpoint)
                - 5.0 * evaluate(endpoint - step)
                + 4.0 * evaluate(endpoint - 2.0 * step)
                - evaluate(endpoint - 3.0 * step)
            ) / step**2
            right_second = (
                2.0 * evaluate(endpoint)
                - 5.0 * evaluate(endpoint + step)
                + 4.0 * evaluate(endpoint + 2.0 * step)
                - evaluate(endpoint + 3.0 * step)
            ) / step**2
            self.assertAlmostEqual(left_first, right_first, places=6)
            self.assertAlmostEqual(left_second, right_second, places=5)

        ratio_one = evaluate(1e6) / 1e6
        ratio_two = evaluate(2e6) / 2e6
        self.assertAlmostEqual(ratio_one, ratio_two, places=5)

    def test_eq7_hermite_bridge_is_c1_and_uses_endpoint_jets(self):
        candidate = HybridInterpolationCandidate(
            "fit4", 1, 1, 2, None, 1, strong_form="eq7_hermite",
        )
        parameters = {
            "c_0_1": [0.15],
            "logp0_0_0": [np.log(0.05277)],
            "p_0_1": [0.30],
            "logrho_0_0": [0.0],
            "logrho_step_0_1": [0.0],
        }
        target = HybridTarget(
            key=("wilson", "p", "5.0"),
            couplings=("8p00", "7p50"),
            regions=("weak", "strong"),
            x=np.asarray([5.0, 8.0]),
            y=np.asarray([-0.1, -0.2]),
            match_low=5.0,
            match_high=8.0,
            strong_scale=3.0,
        )
        beta_function = _TestBetaFunction()

        def evaluate(value):
            return float(np.asarray(_hybrid_beta(
                beta_function, np.asarray([value]), parameters,
                candidate, target, 0,
            )).item())

        step = 1e-5
        for endpoint in (target.match_low, target.match_high):
            left_slope = (evaluate(endpoint) - evaluate(endpoint - step)) / step
            right_slope = (evaluate(endpoint + step) - evaluate(endpoint)) / step
            self.assertAlmostEqual(left_slope, right_slope, places=5)

        # A cubic Hermite bridge is determined completely by its endpoint
        # values and slopes.  Verify the midpoint against that construction.
        lower, upper = target.match_low, target.match_high
        width = upper - lower
        weak_slope = (evaluate(lower) - evaluate(lower - step)) / step
        strong_slope = (evaluate(upper + step) - evaluate(upper)) / step
        expected_midpoint = (
            0.5 * evaluate(lower)
            + 0.125 * width * weak_slope
            + 0.5 * evaluate(upper)
            - 0.125 * width * strong_slope
        )
        self.assertAlmostEqual(
            evaluate(0.5 * (lower + upper)), expected_midpoint, places=5,
        )

        ratio_one = evaluate(1e6) / 1e6
        ratio_two = evaluate(2e6) / 2e6
        self.assertAlmostEqual(ratio_one, ratio_two, places=5)

    def test_eq7_linear_bridge_is_affine_and_has_no_overshoot(self):
        candidate = HybridInterpolationCandidate(
            "fit4", 1, 1, 2, None, 0, strong_form="eq7_linear",
        )
        parameters = {
            "c_0_1": [0.15],
            "logp0_0_0": [np.log(0.05277)],
            "p_0_1": [0.30],
            "logrho_0_0": [0.0],
            "logrho_step_0_1": [0.0],
        }
        target = HybridTarget(
            key=("wilson", "p", "5.0"),
            couplings=("8p00", "7p50"),
            regions=("weak", "strong"),
            x=np.asarray([5.0, 8.0]),
            y=np.asarray([-0.1, -0.2]),
            match_low=5.0,
            match_high=8.0,
            strong_scale=3.0,
        )
        beta_function = _TestBetaFunction()

        def evaluate(value):
            return float(np.asarray(_hybrid_beta(
                beta_function, np.asarray([value]), parameters,
                candidate, target, 0,
            )).item())

        lower_value = evaluate(target.match_low)
        upper_value = evaluate(target.match_high)
        for fraction in np.linspace(0.0, 1.0, 11):
            x = target.match_low + fraction * (
                target.match_high - target.match_low
            )
            expected = lower_value + fraction * (upper_value - lower_value)
            self.assertAlmostEqual(evaluate(x), expected, places=12)
            self.assertGreaterEqual(
                evaluate(x), min(lower_value, upper_value) - 1e-14,
            )
            self.assertLessEqual(
                evaluate(x), max(lower_value, upper_value) + 1e-14,
            )

        ratio_one = evaluate(1e6) / 1e6
        ratio_two = evaluate(2e6) / 2e6
        self.assertAlmostEqual(ratio_one, ratio_two, places=5)

    def test_eq7_branch_is_c1_and_asymptotically_linear(self):
        candidate = HybridInterpolationCandidate(
            "fit4", 1, 1, 2, 0.5, 1, strong_form="eq7",
        )
        parameters = {
            "c_0_1": [0.0],
            "logrho_0_0": [0.0],
            "logrho_step_0_1": [0.0],
        }
        target = HybridTarget(
            key=("wilson", "p", "5.0"),
            couplings=("8p00", "7p50"),
            regions=("weak", "strong"),
            x=np.asarray([4.0, 6.0]),
            y=np.asarray([-0.1, -0.2]),
            match_low=4.0,
            match_high=6.0,
            strong_scale=3.0,
        )
        beta_function = _TestBetaFunction()

        def evaluate(value):
            return float(np.asarray(_hybrid_beta(
                beta_function, np.asarray([value]), parameters,
                candidate, target, 0,
            )).item())

        match = 5.0
        step = 1e-5
        center = evaluate(match)
        left_slope = (center - evaluate(match - step)) / step
        right_slope = (evaluate(match + step) - center) / step
        self.assertAlmostEqual(left_slope, right_slope, places=6)

        ratio_one = evaluate(1e6) / 1e6
        ratio_two = evaluate(2e6) / 2e6
        self.assertAlmostEqual(ratio_one, ratio_two, places=6)

    def test_eq7_requires_one_more_denominator_power(self):
        with self.assertRaisesRegex(ValueError, "denominator order N\\+1"):
            HybridInterpolationCandidate(
                "fit4", 4, 2, 1, None, 1, strong_form="eq7",
            )

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
        targeted_records = [
            row for row in records
            if row["operator"] is not None
            and row["parameter"] != "match_fraction"
        ]
        self.assertEqual({row["operator"] for row in targeted_records}, {"p"})
        self.assertEqual({row["flow_time"] for row in targeted_records}, {5.0})
        self.assertTrue(all(np.isfinite(row["mean"]) for row in records))
        self.assertTrue(all(row["sdev"] >= 0.0 for row in records))
        self.assertTrue(all(np.isfinite(row["log_evidence"]) for row in records))
        self.assertTrue(all(0.0 < row["model_probability"] < 1.0 for row in records))
        match_rows = [row for row in records if row["parameter"] == "match_fraction"]
        self.assertEqual(len(match_rows), len(candidates))
        self.assertEqual({row["operator"] for row in match_rows}, {"p"})
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

    def test_operator_specific_hybrid_fits_have_independent_evidences(self):
        values = gv.gvar(
            [1.0, 2.0, 3.0, 4.0, -0.004, -0.016, -0.035, -0.061],
            np.diag([0.01, 0.01, 0.015, 0.02, 2e-6, 2e-6, 3e-6, 4e-6]),
        )
        targets = {
            operator: {
                ("wilson", operator, "5.0"): HybridTarget(
                    key=("wilson", operator, "5.0"),
                    couplings=("20p0", "8p00", "7p50", "7p00"),
                    regions=("weak", "weak", "strong", "strong"),
                    x=np.asarray(values[:4], dtype=object),
                    y=np.asarray(values[4:], dtype=object),
                    match_low=2.0,
                    match_high=3.0,
                    strong_scale=2.0,
                )
            }
            for operator in ("p", "c")
        }
        candidates = (
            HybridInterpolationCandidate("fit4", 1, 0, 0, None, 1),
        )

        def collect(*args, **kwargs):
            operator = tuple(args[3])[0]
            return targets[operator]

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with mock.patch(
                "betafn.fitting.hybrid.collect_hybrid_targets",
                side_effect=collect,
            ):
                results = fit_hybrid_bma_by_operator(
                    _TestBetaFunction(), (5.0, 6.0), "wilson", ("p", "c"),
                    weak_couplings=("20p0", "8p00"),
                    strong_couplings=("7p50", "7p00"),
                    candidates=candidates,
                    max_workers=2,
                    checkpoint_dir=checkpoint_dir,
                    progress=False,
                )
            self.assertEqual(set(results), {"p", "c"})
            for operator, result in results.items():
                self.assertEqual(set(result.weights), {candidates[0].name})
                self.assertAlmostEqual(result.weights[candidates[0].name], 1.0)
                fitted_operators = {
                    key[1]
                    for fit in result.candidate_fits.values()
                    for key in fit.targets
                }
                self.assertEqual(fitted_operators, {operator})
                checkpoint = (
                    Path(checkpoint_dir) / operator
                    / f"{candidates[0].name}.gvar"
                )
                self.assertTrue(checkpoint.is_file())

    def test_flow_times_are_independent_interpolation_likelihoods(self):
        candidates = (
            HybridInterpolationCandidate("fit4", 1, 0, 0, None, 1),
            HybridInterpolationCandidate("fit4", 1, 0, 0, None, 2),
        )
        targets = {}
        shared_x = gv.gvar(0.0, 0.04)
        shared_beta = gv.gvar(0.0, 8e-4)
        for flow_time, scale in (("5.0", 1.0), ("5.5", 1.02), ("6.0", 1.04)):
            local_values = gv.gvar(
                [1.0, 2.0, 3.0, 4.0,
                 -0.004 * scale, -0.016 * scale,
                 -0.035 * scale, -0.061 * scale],
                np.diag([
                    0.01, 0.01, 0.015, 0.02,
                    2e-6, 2e-6, 3e-6, 4e-6,
                ]),
            )
            x_values = np.asarray(local_values[:4], dtype=object) + shared_x
            y_values = (
                np.asarray(local_values[4:], dtype=object)
                + scale * shared_beta
            )
            key = ("wilson", "p", flow_time)
            targets[key] = HybridTarget(
                key=key,
                couplings=("20p0", "8p00", "7p50", "7p00"),
                regions=("weak", "weak", "strong", "strong"),
                x=x_values,
                y=y_values,
                match_low=2.0,
                match_high=3.0,
                strong_scale=2.0,
            )
        arguments = dict(
            beta_function=BetaFunction(nf=4),
            window=(5.0, 6.0),
            flow="wilson",
            observables=("p",),
            weak_couplings=("20p0", "8p00"),
            strong_couplings=("7p50", "7p00"),
            candidates=candidates,
            max_workers=2,
            progress=False,
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with mock.patch(
                "betafn.fitting.hybrid.collect_hybrid_targets",
                return_value=targets,
            ):
                interpolations = fit_hybrid_bma_by_target(
                    checkpoint_dir=checkpoint_dir, **arguments,
                )
            with (
                mock.patch(
                    "betafn.fitting.hybrid.collect_hybrid_targets",
                    return_value=targets,
                ),
                mock.patch(
                    "betafn.fitting.hybrid._fit_prepared_candidate",
                    side_effect=AssertionError("valid target checkpoint not reused"),
                ),
            ):
                restored_interpolations = fit_hybrid_bma_by_target(
                    checkpoint_dir=checkpoint_dir, **arguments,
                )
            self.assertEqual(set(interpolations), set(targets))
            self.assertEqual(set(restored_interpolations), set(targets))
            for key in targets:
                for candidate in candidates:
                    original_parameters = interpolations[key].candidate_fits[
                        candidate.name
                    ].fit.p
                    restored_parameters = restored_interpolations[key].candidate_fits[
                        candidate.name
                    ].fit.p
                    self.assertEqual(
                        set(original_parameters), set(restored_parameters)
                    )
                    for parameter in original_parameters:
                        np.testing.assert_allclose(
                            gv.mean(original_parameters[parameter]),
                            gv.mean(restored_parameters[parameter]),
                            rtol=1e-10, atol=1e-12,
                        )
                        np.testing.assert_allclose(
                            gv.evalcov(original_parameters[parameter]),
                            gv.evalcov(restored_parameters[parameter]),
                            rtol=1e-8, atol=1e-12,
                        )
            for key, result in interpolations.items():
                for candidate_fit in result.candidate_fits.values():
                    self.assertEqual(set(candidate_fit.targets), {key})
                time_tag = f"t_{float(key[2]):.8g}".replace(".", "p")
                for candidate in candidates:
                    self.assertTrue((
                        Path(checkpoint_dir) / key[1] / time_tag
                        / f"{candidate.name}.gvar"
                    ).is_file())

            continuum = run_hybrid_bma_by_target(
                g2_grid=(2.0,),
                prefit_interpolations=restored_interpolations,
                **arguments,
            )
            self.assertEqual(set(continuum.averages), {"p"})
            continuum_average = continuum.averages["p"]
            self.assertEqual(len(continuum_average.prediction), 1)
            np.testing.assert_allclose(
                continuum_average.covariance,
                continuum_average.within_covariance
                + continuum_average.between_covariance,
                rtol=1e-10,
                atol=1e-14,
            )
            self.assertIn(
                "continuity", continuum.model_variance_by_choice["p"]
            )
            np.testing.assert_allclose(
                continuum.model_variance_by_choice["p"]["continuity"],
                np.diag(continuum_average.between_covariance),
                rtol=1e-7,
                atol=1e-12,
            )
            self.assertEqual(
                len(continuum.continuum_points["p"][0]["keys"]), 3
            )
            continuum_point_covariance = gv.evalcov(
                continuum.continuum_points["p"][0]["y"]
            )
            self.assertGreater(abs(continuum_point_covariance[0, 1]), 0.0)
            local_bma_values = [
                continuum.target_averages[key].average.prediction[0]
                for key in sorted(continuum.target_averages)
            ]
            local_bma_covariance = gv.evalcov(local_bma_values)
            self.assertGreater(abs(local_bma_covariance[0, 1]), 0.0)

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
