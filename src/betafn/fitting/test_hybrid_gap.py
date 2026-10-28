"""Regression checks for the gap-only hybrid interpolation (no ensemble runs)."""
import unittest
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from unittest import mock
from dataclasses import replace
from types import SimpleNamespace

import gvar as gv
import lsqfit
import numpy as np

from betafn.fitting import hybrid as h
from betafn.perturbative import PerturbativeBetaFunction
from betafn.stages.continuum import ContinuumMixin


class TestHybridGap(unittest.TestCase):
    def setUp(self):
        self.bf = SimpleNamespace(
            perturbative_beta_function=PerturbativeBetaFunction(4, 3),
        )
        self.candidate = h.HybridInterpolationCandidate(
            'fit11', 2, 1, 2, continuity=1, strong_form='eq7_linear',
        )
        self.p = {
            'c_0_1': [gv.gvar(.1, .02)], 'c_0_2': [gv.gvar(-.1, .03)],
            'a0_0_0': [gv.gvar(.001, .0001)],
            'logp0_0_0': [gv.gvar(np.log(.05277), .1)],
            'p_0_1': [gv.gvar(.2, .02)],
            'logrho_0_0': [gv.gvar(0., .1)],
            'logrho_step_0_1': [gv.gvar(0., .1)],
        }
        self.target = SimpleNamespace(match_low=4., match_high=7.,
                                      regions=('weak', 'weak', 'strong', 'strong'))

    def beta(self, x, region=None):
        return h._hybrid_beta(self.bf, np.asarray(x), self.p,
                              self.candidate, self.target, 0, region=region)

    def test_measured_regions_and_short_extension_preserve_full_gvar(self):
        # Verify identical random variables, not merely identical means.
        weak_x = np.linspace(.01, 4.15, 100)
        strong_x = np.linspace(7., 15., 100)
        for x, regions in [(weak_x, ['weak']*100), (strong_x, ['strong']*100)]:
            difference = self.beta(x) - self.beta(x, regions)
            np.testing.assert_allclose(gv.mean(difference), 0., atol=1e-14)
            np.testing.assert_allclose(gv.sdev(difference), 0., atol=1e-14)

    def test_fast_and_reference_likelihood_match_evaluator_on_data(self):
        x = np.array([1., 4., 7., 9.])
        pt = self.bf.perturbative_beta_function
        fast = h._prepared_hybrid_beta(x, self.p, self.candidate, self.target,
                                       0, pt.nrm, pt.b)
        difference = self.beta(x) - fast
        np.testing.assert_allclose(gv.mean(difference), 0., atol=1e-14)
        np.testing.assert_allclose(gv.sdev(difference), 0., atol=1e-14)

    def test_both_beta_and_ratio_are_c1_with_correlated_errors(self):
        for point in (4.15, 7.):
            eps = 1e-5
            for ratio in (False, True):
                x = point + eps*np.array([-2., -1., 0., 1., 2.])
                y = self.beta(x)
                if ratio:
                    y = y/x**2
                left = (3*y[2]-4*y[1]+y[0])/(2*eps)
                right = (-3*y[2]+4*y[3]-y[4])/(2*eps)
                self.assertLess(abs(gv.mean(left-right)), 1e-7)
                self.assertLess(gv.sdev(left-right), 1e-7)

    def test_monotonic_gap_for_extreme_compatible_endpoint_slopes(self):
        x = np.linspace(0., 1., 10001)
        for d0, d1 in [(0., 0.), (-100., -.001), (-.001, -100.), (-1., -1.)]:
            y = h._c1_rational_gap_bridge(
                x, np.full_like(x, -1.), np.full_like(x, -2.),
                -1., d0, -2., d1, 0., 1.,
            )
            self.assertTrue(np.isfinite(y).all())
            self.assertLessEqual(np.max(np.diff(y)), 1e-12)
            self.assertGreaterEqual(np.min(y), -2.-1e-12)
            self.assertLessEqual(np.max(y), -1.+1e-12)

    def test_gap_covariance_matches_numerical_jacobian(self):
        means = np.array([-1., -.7, -3., -1.2])
        covariance = .0001*(np.eye(4)+np.ones((4, 4)))
        endpoints = gv.gvar(means, covariance)
        x = np.array([.2, .5, .8])
        def evaluate(p):
            return h._c1_rational_gap_bridge(x, 0., 0., *p, 0., 1.)
        jac = np.column_stack([
            (evaluate(means+np.eye(4)[i]*1e-5)
             - evaluate(means-np.eye(4)[i]*1e-5))/(2e-5)
            for i in range(4)
        ])
        np.testing.assert_allclose(gv.evalcov(evaluate(endpoints)),
                                   jac@covariance@jac.T, rtol=1e-8, atol=1e-12)

    def test_eq7_value_and_derivative_finite_at_zero_and_numerator_zero(self):
        p = {key: gv.mean(value) for key, value in self.p.items()}
        p['p_0_1'] = [-1.]
        x = np.array([0., 4*np.pi])
        y, d = h._independent_eq7_strong_beta_and_slope(
            x, p, self.candidate, 0, 4*np.pi,
        )
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(d).all())

    def test_c1_reuses_only_matching_likelihood_checkpoint(self):
        target = h.HybridTarget(
            key=('wilson', 'p', '5.0'),
            couplings=('20p0', '8p00', '7p50', '7p00'),
            regions=('weak', 'weak', 'strong', 'strong'),
            x=gv.gvar([1., 4., 7., 9.], [.01]*4),
            y=gv.gvar([-.055, -.6, -1.7, -2.4], [.01]*4),
            match_low=4., match_high=7., strong_scale=5.,
        )
        candidate = replace(self.candidate, weak_order=1)
        arguments = dict(
            beta_function=self.bf, window=(5., 6.), flow='wilson',
            observables=('p',), weak_couplings=('20p0', '8p00'),
            strong_couplings=('7p50', '7p00'), max_workers=1, progress=False,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            h, 'collect_hybrid_targets', return_value={target.key: target},
        ):
            legacy = h.fit_hybrid_bma_by_target(
                **arguments, candidates=(replace(candidate, continuity=2),),
                checkpoint_dir=directory,
            )
            with mock.patch.object(h, '_fit_prepared_candidate',
                                   side_effect=AssertionError('unnecessary refit')):
                updated = h.fit_hybrid_bma_by_target(
                    **arguments, candidates=(candidate,), checkpoint_dir=directory,
                )
                old_fit = next(iter(legacy[target.key].candidate_fits.values()))
                new_fit = next(iter(updated[target.key].candidate_fits.values()))
                self.assertEqual(new_fit.candidate.continuity, 1)
                self.assertEqual(old_fit.log_evidence, new_fit.log_evidence)
                self.assertEqual(old_fit.fit.chi2, new_fit.fit.chi2)
                # A changed prior invalidates the old likelihood checkpoint.
                with self.assertRaisesRegex(AssertionError, 'unnecessary refit'):
                    h.fit_hybrid_bma_by_target(
                        **arguments, candidates=(candidate,),
                        checkpoint_dir=directory, weak_width=123.,
                    )


class TestMonotoneWeak(unittest.TestCase):
    def setUp(self):
        self.bf = SimpleNamespace(perturbative_beta_function=PerturbativeBetaFunction(4, 3))
        self.pt = self.bf.perturbative_beta_function
        self.candidate = h.HybridInterpolationCandidate(
            'monotone_constant', 4, 1, 2, continuity=1, strong_form='eq7_linear',
        )
        _, self.p = h._candidate_priors(self.candidate, 1, 1.5, .25, 2., 1.)
        self.p.update({f'logw_0_{k}': [v] for k, v in enumerate([.3, -.2, .1, -.5], 1)})
        self.p['a0_0_0'] = [.002]

    def weak(self, x, parameters=None):
        return h._weak_beta_derivatives(
            np.asarray(x), self.p if parameters is None else parameters,
            self.candidate, 0, self.pt.nrm, self.pt.b,
        )

    def test_analytic_integral_matches_quadrature_and_derivatives(self):
        from scipy.integrate import quad
        b0 = self.pt.b[0]/self.pt.nrm
        for x in (.01, 1., 5., 8.5, 12.):
            integral = quad(lambda t: float(self.weak(t)[1]), 0., x, epsabs=1e-12)[0]
            y, d, d2 = self.weak(x)
            self.assertAlmostEqual(float(y), .002+integral, places=11)
            eps = 1e-4
            num_d = (self.weak(x+eps)[0]-self.weak(x-eps)[0])/(2*eps)
            num_d2 = (self.weak(x+eps)[1]-self.weak(x-eps)[1])/(2*eps)
            self.assertAlmostEqual(float(d), float(num_d), places=7)
            self.assertAlmostEqual(float(d2), float(num_d2), places=7)
        self.assertAlmostEqual(float((self.weak(1e-6)[0]-.002)/1e-12), -b0, places=6)

    def test_no_inherited_pt3_zero_and_negative_slope(self):
        x = np.linspace(.001, 4*np.pi, 1000)
        y, d, _ = self.weak(x)
        self.assertTrue((d < 0).all())
        xzero = float(max(np.roots(self.pt.b[::-1]))*self.pt.nrm)
        self.assertLess(float(self.weak(xzero)[0]), -1.)
        self.assertGreater(float(self.weak(1.)[0]), float(self.weak(8.)[0]))

    def test_fast_path_correlated_gvar_and_c1_gap(self):
        parameters = {k: gv.gvar(v, [.01]*len(v)) for k,v in self.p.items()}
        target = SimpleNamespace(match_low=7.8, match_high=11.5,
                                 regions=('weak','weak','strong','strong'))
        x = np.array([1., 7.8, 11.5, 16.])
        fast = h._prepared_hybrid_beta(x, parameters, self.candidate, target,
                                       0, self.pt.nrm, self.pt.b)
        def evaluate(x):
            return h._hybrid_beta(self.bf, np.asarray(x), parameters,
                                  self.candidate, target, 0)
        delta = evaluate(x)-fast
        np.testing.assert_allclose(gv.mean(delta),0.,atol=1e-13)
        np.testing.assert_allclose(gv.sdev(delta),0.,atol=1e-13)
        for a in (7.8+.05*(11.5-7.8),11.5):
            dx=1e-5
            xx=a+dx*np.array([-2.,-1.,0.,1.,2.])
            for y in (evaluate(xx),evaluate(xx)/xx**2):
                delta=(3*y[2]-4*y[1]+y[0]+3*y[2]-4*y[3]+y[4])/(2*dx)
                self.assertLess(abs(gv.mean(delta)),1e-7)
                self.assertLess(gv.sdev(delta),1e-7)

    def test_prior_parameters_and_old_families_remain_selectable(self):
        candidates = h.default_hybrid_candidates(
            weak_orders_by_family={'monotone':(4,), 'monotone_constant':(4,),
                                   'fit4':(4,), 'fit11':(4,)},
            pade_orders=((1,2),), continuities=(1,), strong_form='eq7_linear',
        )
        self.assertEqual(len(candidates),4)
        for c in candidates:
            prior, _ = h._candidate_priors(c,1,1.5,.25,2.,1.)
            self.assertEqual('a0_0_0' in prior,c.weak_family in ('fit11','monotone_constant'))
            self.assertEqual('logw_0_1' in prior,c.weak_family.startswith('monotone'))
            self.assertNotIn('match_latent',prior)

    def test_small_errors_in_variables_fit_and_new_checkpoint(self):
        # One synthetic 15-point target exercises the production fitter and
        # checkpoint loader. This does not read or process ensemble data.
        p = dict(self.p)
        p['logp0_0_0'] = [np.log(.12)]
        xmean = np.r_[np.linspace(.8, 7.8, 11), [11.5, 13., 15., 17.]]
        geometry = SimpleNamespace(match_low=7.8, match_high=11.5)
        ymean = h._hybrid_beta(self.bf, xmean, p, self.candidate, geometry, 0)
        n = len(xmean)
        cov = np.diag(np.r_[np.full(n, .01**2), np.full(n, .015**2)])
        for j in range(n):
            cov[j, j+n] = cov[j+n, j] = -.3*.01*.015
        xy = gv.gvar(np.r_[xmean, ymean], cov)
        target = h.HybridTarget(
            key=('wilson','p','5.0'),
            couplings=tuple(f'b{i}' for i in range(n)),
            regions=('weak',)*11+('strong',)*4,
            x=xy[:n], y=xy[n:], match_low=7.8, match_high=11.5, strong_scale=9.2,
        )
        arguments = dict(
            beta_function=self.bf, window=(5.,6.), flow='wilson',
            observables=('p',), weak_couplings=target.couplings[:11],
            strong_couplings=target.couplings[11:], candidates=(self.candidate,),
            max_workers=1, progress=False, weak_width=1.5, strong_width=2.,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            h, 'collect_hybrid_targets', return_value={target.key:target},
        ):
            fitted = h.fit_hybrid_bma_by_target(**arguments, checkpoint_dir=directory)
            result = fitted[target.key].candidate_fits[self.candidate.name]
            self.assertLess(result.fit.chi2/result.fit.dof, 1.)
            self.assertTrue(np.isfinite(result.log_evidence))
            prediction = h.evaluate_hybrid_target(result, self.bf, target.key, xmean)
            np.testing.assert_allclose(gv.mean(prediction), ymean, atol=.03)
            with mock.patch.object(h, '_fit_prepared_candidate',
                                   side_effect=AssertionError('unnecessary refit')):
                restored = h.fit_hybrid_bma_by_target(**arguments, checkpoint_dir=directory)
            self.assertEqual(restored[target.key].weights, fitted[target.key].weights)


class TestEstablishedWeakFamilies(unittest.TestCase):
    def setUp(self):
        self.pt = PerturbativeBetaFunction(4, 3)
        self.x = np.array([.8, 2.5, 6.0, 8.0])
        self.u = self.x / self.pt.nrm

    def candidate_and_parameters(self, family, order, values):
        candidate = h.HybridInterpolationCandidate(
            family, order, 1, 2, continuity=1, strong_form='eq7_linear',
        )
        _, parameters = h._candidate_priors(candidate, 1, 1.5, .25, 2., 1.)
        for (prefix, power), value in zip(h._weak_parameter_specs(candidate), values):
            parameters[h._parameter_key(prefix, 0, power)] = [value]
        return candidate, parameters

    def check_value_and_derivatives(self, candidate, parameters, expected):
        value, first, second = h._weak_beta_derivatives(
            self.x, parameters, candidate, 0, self.pt.nrm, self.pt.b[:3],
        )
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
        step = 1e-4
        plus = h._weak_beta_derivatives(
            self.x+step, parameters, candidate, 0, self.pt.nrm, self.pt.b[:3],
        )[0]
        minus = h._weak_beta_derivatives(
            self.x-step, parameters, candidate, 0, self.pt.nrm, self.pt.b[:3],
        )[0]
        numerical_first = (plus-minus)/(2*step)
        numerical_second = (plus-2*value+minus)/step**2
        np.testing.assert_allclose(first, numerical_first, rtol=2e-8, atol=1e-9)
        np.testing.assert_allclose(second, numerical_second, rtol=2e-5, atol=1e-7)

    def test_fit5_fit6_fit7_match_production_definitions(self):
        pt_ratio = sum(
            -coefficient/self.pt.nrm*self.u**power
            for power, coefficient in enumerate(self.pt.b[:3])
        )
        candidate, p = self.candidate_and_parameters('fit5', 2, (.2, -.1))
        self.check_value_and_derivatives(
            candidate, p, self.x**2*(pt_ratio+.2*self.u**3-.1*self.u**4),
        )
        candidate, p = self.candidate_and_parameters(
            'fit6', 2, (-.03, .2, -.1),
        )
        ratio = (
            -self.pt.b[0]/self.pt.nrm - self.pt.b[2]/self.pt.nrm*self.u**2
            -.03*self.u + .2*self.u**3 - .1*self.u**4
        )
        self.check_value_and_derivatives(candidate, p, self.x**2*ratio)
        candidate, p = self.candidate_and_parameters(
            'fit7', 2, (.04, .2, -.1),
        )
        ratio = (
            -self.pt.b[0]/self.pt.nrm - self.pt.b[1]/self.pt.nrm*self.u
            + .04*self.u**2 + .2*self.u**3 - .1*self.u**4
        )
        self.check_value_and_derivatives(candidate, p, self.x**2*ratio)

    def test_fit8_fit9_fit10_match_production_definitions(self):
        perturbative = self.pt(self.x, loops=3)
        candidate, p = self.candidate_and_parameters(
            'fit8', 4, (.2, -.1, .05),
        )
        self.check_value_and_derivatives(
            candidate, p,
            perturbative*(1+.2*self.u**2-.1*self.u**3+.05*self.u**4),
        )
        candidate, p = self.candidate_and_parameters(
            'fit9', 5, (.1, -.2, .08, -.03),
        )
        self.check_value_and_derivatives(
            candidate, p,
            perturbative*(
                1+.1*self.u-.2*self.u**3+.08*self.u**4-.03*self.u**5
            ),
        )
        candidate, p = self.candidate_and_parameters(
            'fit10', 5, (.9, -.2, .08, -.03),
        )
        self.check_value_and_derivatives(
            candidate, p,
            perturbative*(.9-.2*self.u**3+.08*self.u**4-.03*self.u**5),
        )

    def test_family_specific_order_constraints(self):
        with self.assertRaisesRegex(ValueError, 'fit8 order'):
            h.HybridInterpolationCandidate(
                'fit8', 1, 1, 2, continuity=1, strong_form='eq7_linear',
            )
        for family in ('fit9', 'fit10'):
            with self.assertRaisesRegex(ValueError, 'fit9/fit10 order'):
                h.HybridInterpolationCandidate(
                    family, 2, 1, 2, continuity=1,
                    strong_form='eq7_linear',
                )


class TestParallelContinuumWorker(unittest.TestCase):
    def test_streaming_target_average_matches_materialized_bma(self):
        key = ('wilson', 'p', '5.0')
        candidates = (
            h.HybridInterpolationCandidate('fit4', 1, 1, 0, 0.5, 1),
            h.HybridInterpolationCandidate('fit11', 1, 1, 0, 0.5, 1),
        )
        target = SimpleNamespace(x=gv.gvar([1., 3.], [.01, .01]))
        fits = {
            candidate.name: SimpleNamespace(
                candidate=candidate, targets={key: target},
            )
            for candidate in candidates
        }
        weights = {candidates[0].name: 0.35, candidates[1].name: 0.65}
        primary = gv.gvar(np.zeros(7), np.eye(7))
        predictions = {
            candidates[0].name: np.array([
                1.+primary[0], 2.+primary[1], 3.+primary[2],
            ]),
            candidates[1].name: np.array([
                1.2+primary[0]+.2*primary[4],
                1.8+primary[1]+.3*primary[5],
                3.1+primary[3]+.1*primary[6],
            ]),
        }
        interpolation = h.HybridInterpolationResult(
            window=(5., 6.), flow_time_spacing=None,
            candidate_fits=fits, weights=weights,
        )
        materialized = h.model_average_predictions(predictions, weights)
        with mock.patch.object(
            h, 'evaluate_hybrid_target',
            side_effect=lambda fit, beta_function, requested_key, grid:
            predictions[fit.candidate.name],
        ):
            streamed = h._average_independent_target(
                SimpleNamespace(), key, interpolation, np.array([1., 2., 3.]),
            )
        np.testing.assert_allclose(streamed.average.mean, materialized.mean)
        np.testing.assert_allclose(
            streamed.average.within_covariance,
            materialized.within_covariance,
        )
        np.testing.assert_allclose(
            streamed.average.between_covariance,
            materialized.between_covariance,
        )
        np.testing.assert_allclose(
            gv.evalcov(streamed.average.prediction),
            materialized.covariance,
            rtol=1e-12, atol=1e-12,
        )
        expected_weighted = sum(
            weights[name]*prediction for name, prediction in predictions.items()
        )
        joint = np.concatenate([
            streamed.average.prediction, primary[:1],
        ])
        expected_joint = np.concatenate([expected_weighted, primary[:1]])
        np.testing.assert_allclose(
            gv.evalcov(joint)[:3, 3:],
            gv.evalcov(expected_joint)[:3, 3:],
            rtol=1e-12, atol=1e-12,
        )

    def test_weighted_cross_covariance_fast_path_is_exact(self):
        primary = gv.gvar(np.zeros(8), np.eye(8))
        row_predictions = {
            'a': np.array([primary[0]+primary[4], primary[1]-primary[5]]),
            'b': np.array([primary[2]+2*primary[4], primary[3]-primary[6]]),
        }
        column_predictions = {
            'a': np.array([primary[4]+primary[0], primary[6]-primary[2]]),
            'b': np.array([primary[5]+primary[1], primary[7]-primary[3]]),
        }
        weights = {'a': 0.3, 'b': 0.7}
        explicit = np.zeros((2, 2))
        for row_name, row in row_predictions.items():
            for column_name, column in column_predictions.items():
                covariance = gv.evalcov(np.concatenate([row, column]))
                explicit += (
                    weights[row_name]*weights[column_name]*covariance[:2, 2:]
                )
        row_weighted = sum(weights[name]*row for name, row in row_predictions.items())
        column_weighted = sum(
            weights[name]*column for name, column in column_predictions.items()
        )
        covariance = gv.evalcov(np.concatenate([row_weighted, column_weighted]))
        np.testing.assert_allclose(covariance[:2, 2:], explicit, atol=1e-15)

    def test_parallel_numeric_worker_matches_serial_gvar_gls(self):
        x = np.array([1/5., 1/5.2, 1/5.4, 1/5.6, 1/5.8, 1/6.])
        mean = -0.05 + 0.02*x
        indices = np.arange(len(x))
        actual_covariance = 2e-7*np.exp(-abs(indices[:, None]-indices[None, :])/2)
        weight_covariance = 0.8*actual_covariance + 0.2*np.diag(
            np.diag(actual_covariance)
        )
        y = gv.gvar(mean, actual_covariance)
        task = {
            'key': ('p', 0), 'x': x, 'mean': mean,
            'weight_covariance': weight_covariance,
            'actual_covariance': actual_covariance,
        }

        def model(coordinate, parameters):
            return parameters['beta'][0] + parameters['slope'][0]*coordinate

        class ContinuumHarness(ContinuumMixin):
            @staticmethod
            def _run_fit(coordinate, values, function, prior, p0):
                return lsqfit.nonlinear_fit(
                    data=(coordinate, values), fcn=function,
                    prior=prior, p0=p0,
                )

        serial_parameters, serial_fit = ContinuumHarness()._correlated_weighted_fit(
            x, y, model, weight_covariance, prior=None,
            p0={'beta': [0.], 'slope': [0.]},
        )
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=2, mp_context=context,
            initializer=h._initialize_continuum_worker,
        ) as executor:
            key, parameter_mean, gain, quality = executor.submit(
                h._continuum_numeric_fit, task,
            ).result()
        corrected = parameter_mean + gain.dot(y-mean)
        parallel_parameters = {
            'beta': [corrected[0]], 'slope': [corrected[1]],
        }
        self.assertEqual(key, ('p', 0))
        np.testing.assert_allclose(
            gv.mean([parallel_parameters['beta'][0], parallel_parameters['slope'][0]]),
            gv.mean([serial_parameters['beta'][0], serial_parameters['slope'][0]]),
            rtol=1e-12, atol=1e-14,
        )
        np.testing.assert_allclose(
            gv.evalcov([parallel_parameters['beta'][0], parallel_parameters['slope'][0]]),
            gv.evalcov([serial_parameters['beta'][0], serial_parameters['slope'][0]]),
            rtol=1e-12, atol=1e-14,
        )
        self.assertAlmostEqual(quality['chi2'], serial_fit.chi2, places=12)
        self.assertEqual(quality['dof'], serial_fit.dof)


if __name__ == '__main__':
    unittest.main()
