"""ContinuumMixin — continuum-limit extrapolation stage."""
from __future__ import annotations

import decimal as _decimal
import gc as _gc
import itertools as _itertools

import gvar as _gvar
import numpy as _numpy
from tqdm import tqdm as _tqdm

from ..base.exceptions import BetaFunctionException
from ..fitting.containers import FitInput, StageStore


# ---------------------------------------------------------------------------
# Private kernel and covariance helpers (module-level)
# ---------------------------------------------------------------------------

def _kernel_correlation(distances, length: float, kernel: str = "rbf"):
    """Stationary unit-variance correlation function of |distance|."""
    if length <= 0.0:
        raise BetaFunctionException("kernel correlation length must be positive")
    scaled = _numpy.abs(distances) / length
    if kernel == "rbf":
        return _numpy.exp(-0.5 * scaled * scaled)
    if kernel == "matern32":
        factor = _numpy.sqrt(3.0) * scaled
        return (1.0 + factor) * _numpy.exp(-factor)
    raise BetaFunctionException("kernel must be 'rbf' or 'matern32'")


def _fit_correlation_model(empirical_corr, distance_matrices, kernel: str = "rbf", n_scan: int = 25, nugget_floor: float = 1e-3):
    """Fit a product-kernel + nugget model to an empirical correlation matrix."""
    empirical_corr = _numpy.asarray(empirical_corr)
    off_diagonal = ~_numpy.eye(len(empirical_corr), dtype=bool)
    targets = empirical_corr[off_diagonal]

    grids = []
    for distances in distance_matrices:
        off_values = _numpy.abs(_numpy.asarray(distances))[off_diagonal]
        positive = off_values[off_values > 0.0]
        if len(positive) == 0:
            grids.append(_numpy.array([1.0]))
            continue
        grids.append(_numpy.geomspace(0.5 * positive.min(), 4.0 * positive.max(), n_scan))

    best = None
    for lengths in _itertools.product(*grids):
        model_off = _numpy.ones(int(off_diagonal.sum()))
        for distances, length in zip(distance_matrices, lengths):
            model_off = model_off * _kernel_correlation(_numpy.asarray(distances)[off_diagonal], length, kernel)
        denominator = float(model_off @ model_off)
        amplitude = float(model_off @ targets) / denominator if denominator > 0.0 else 0.0
        amplitude = min(max(amplitude, 0.0), 1.0 - nugget_floor)
        loss = float(((amplitude * model_off - targets) ** 2).sum())
        if best is None or loss < best[0]:
            best = (loss, lengths, amplitude)

    _, lengths, amplitude = best
    model_corr = _numpy.ones_like(empirical_corr)
    for distances, length in zip(distance_matrices, lengths):
        model_corr = model_corr * _kernel_correlation(_numpy.asarray(distances), length, kernel)
    model_corr = amplitude * model_corr
    _numpy.fill_diagonal(model_corr, 1.0)
    return model_corr, [float(length) for length in lengths], 1.0 - amplitude


def _psd_clip(matrix):
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = _numpy.linalg.eigh(symmetric)
    return (eigenvectors * _numpy.clip(eigenvalues, 0.0, None)) @ eigenvectors.T


def _flatten_pdict(params) -> tuple[list, list[tuple[str, int]]]:
    """Flatten a dict of scalars/sequences into (flat list, structure spec)."""
    flat, spec = [], []
    for key in params:
        values = _numpy.atleast_1d(params[key])
        spec.append((key, len(values)))
        flat.extend(values)
    return flat, spec


def _unflatten_pdict(flat, spec) -> dict:
    """Inverse of _flatten_pdict."""
    result, index = {}, 0
    for key, count in spec:
        result[key] = [flat[index + offset] for offset in range(count)]
        index += count
    return result


# ---------------------------------------------------------------------------
# ContinuumMixin
# ---------------------------------------------------------------------------

class ContinuumMixin:
    """Continuum-limit extrapolation at fixed renormalized coupling g^2."""

    def _cnt_fcn(self, x, p):
        return p["beta"][0] + p["slope"][0] * x

    def _record_continuum_point(self, flow, obs, g2, fit, times, y_data, params=None, beta_value=None, x_data=None):
        params = fit.p if params is None else params
        self.cnt_fits[flow][obs].append(params)
        self.continuum.quality[flow][obs].append(self._quality_of_fit(fit))
        self.g2s[flow][obs].append(g2)
        self.betas[flow][obs].append(
            self.continuum.model.evaluate(0.0, params) if beta_value is None else beta_value
        )
        if x_data is None:
            x_data = 1.0 / _numpy.array([float(t) for t in times])
        self.continuum.store("domains", (flow, obs, str(g2)), times)
        self.continuum.store(
            "inputs", (flow, obs, str(g2)),
            FitInput(x=_numpy.asarray(x_data), y=y_data, labels=times),
        )

    def _weight_covariance(self, y, cov_mode: str = "empirical", alpha: float = 0.25, kernel: str = "rbf", distances=None):
        """Return the covariance used to weight a fit of gvar data y.

        Modes: 'empirical' (full gvar covariance), 'diagonal',
        'shrink' (linear blend toward the diagonal), and 'kernel'
        (smooth stationary product-kernel + nugget model).
        """
        covariance = _gvar.evalcov(_numpy.asarray(y))
        if cov_mode in (None, "empirical"):
            return covariance
        if cov_mode == "diagonal":
            return _numpy.diag(_numpy.diag(covariance))
        if cov_mode == "shrink":
            if not 0.0 <= alpha <= 1.0:
                raise BetaFunctionException("alpha must lie in [0, 1] when cov_mode='shrink'")
            diagonal_covariance = _numpy.diag(_numpy.diag(covariance))
            return (1.0 - alpha) * covariance + alpha * diagonal_covariance
        if cov_mode == "kernel":
            if distances is None:
                raise BetaFunctionException("cov_mode='kernel' requires distance matrices")
            sdev = _numpy.sqrt(_numpy.diag(covariance))
            scale = _numpy.outer(sdev, sdev)
            correlation = covariance / scale
            model_corr, lengths, nugget = _fit_correlation_model(correlation, distances, kernel=kernel)
            self._last_covariance_model = {"kernel": kernel, "lengths": lengths, "nugget": nugget}
            return scale * model_corr
        raise BetaFunctionException("cov_mode must be one of: empirical, diagonal, shrink, kernel")

    def _model_jacobian(self, fcn, x, params: dict):
        flat, spec = _flatten_pdict(params)
        primaries = _gvar.gvar(_gvar.mean(flat), _numpy.ones(len(flat)))
        values = _numpy.atleast_1d(fcn(x, _unflatten_pdict(list(primaries), spec)))
        jacobian = _numpy.zeros((len(values), len(primaries)))
        for row, value in enumerate(values):
            if isinstance(value, _gvar.GVar):
                jacobian[row] = _gvar.deriv(value, primaries)
        return jacobian

    def _correlated_weighted_fit(self, x, y, fcn, weight_cov, prior=None, p0=None, extra_cov=None):
        """Weighted least-squares fit that keeps output parameters correlated with input gvars."""
        y = _numpy.asarray(y)
        y_mean = _gvar.mean(y)
        central = self._run_fit(x, _gvar.gvar(y_mean, weight_cov), fcn, prior, p0)
        flat_central, spec = _flatten_pdict(central.p)
        p_mean = _gvar.mean(flat_central)
        jacobian = self._model_jacobian(fcn, x, _unflatten_pdict(list(p_mean), spec))
        weight = _numpy.linalg.pinv(weight_cov)
        curvature = jacobian.T @ weight @ jacobian
        prior_precision = _numpy.zeros(len(p_mean))
        if prior:
            normalized = self._normalize_param_dict(prior)
            index = 0
            for key, count in spec:
                for entry in range(count):
                    if key in normalized:
                        value = normalized[key][entry]
                        sdev = value.sdev if isinstance(value, _gvar.GVar) else 0.0
                        if sdev > 0.0:
                            prior_precision[index] = 1.0 / (sdev * sdev)
                    index += 1
            curvature = curvature + _numpy.diag(prior_precision)
        posterior = _numpy.linalg.inv(curvature)
        gain = posterior @ jacobian.T @ weight
        corrected = p_mean + gain.dot(y - y_mean)
        noise_cov = None
        if prior_precision.any():
            noise_cov = posterior @ _numpy.diag(prior_precision) @ posterior
        if extra_cov is not None:
            discrepancy = gain @ extra_cov @ gain.T
            noise_cov = discrepancy if noise_cov is None else noise_cov + discrepancy
        if noise_cov is not None:
            corrected = corrected + _gvar.gvar(_numpy.zeros(len(corrected)), _psd_clip(noise_cov))
        return _unflatten_pdict(list(corrected), spec), central

    def _gp_log_marginal(self, residuals, covariance) -> float:
        sign, logdet = _numpy.linalg.slogdet(covariance)
        if sign <= 0.0:
            return -_numpy.inf
        return -0.5 * (residuals @ _numpy.linalg.solve(covariance, residuals) + logdet)

    def _tune_gp_amplitude(self, residuals, base_cov, unit_kernel, n_scan: int = 20) -> float:
        scale = float(_numpy.mean(residuals * residuals))
        if scale <= 0.0:
            return 0.0
        best_amp, best_ll = 0.0, self._gp_log_marginal(residuals, base_cov)
        for amplitude in scale * _numpy.geomspace(1e-3, 30.0, n_scan):
            log_likelihood = self._gp_log_marginal(residuals, base_cov + amplitude * unit_kernel)
            if log_likelihood > best_ll:
                best_amp, best_ll = float(amplitude), log_likelihood
        return best_amp

    def _shifted_fit_beta(self, x_data, means, sdevs, times):
        intercepts = []
        for sign in (1.0, -1.0):
            shifted = _gvar.gvar(means + sign * sdevs, sdevs)
            fit = self.continuum.model.fit(
                self._run_fit, self._run_fit_with_x_errors,
                FitInput(x=x_data, y=shifted, labels=times),
            )
            intercepts.append(_gvar.mean(self.continuum.model.evaluate(0.0, fit.p)))
        return 0.5 * abs(intercepts[0] - intercepts[1])

    def _default_artifact_model(self):
        nrm = self.perturbative_beta_function.nrm

        def artifact(g2, x, p):
            u = g2 / nrm
            return (p["s0"][0] + p["s1"][0] * u) * u * u * x

        prior = {"s0": [_gvar.gvar(0.0, 10.0)], "s1": [_gvar.gvar(0.0, 10.0)]}
        p0 = {"s0": 0.0, "s1": 0.0}
        return artifact, prior, p0

    def cnt_xtrp(
        self,
        mnt: float,
        mxt: float,
        mng2: float,
        mxg2: float,
        dg2: float = 0.1,
        fcn=None,
        ntrp_fcn=None,
        prior: dict | None = None,
        p0: dict | None = None,
        v: int = 1,
        diagonal: bool = False,
        g2_round_precis: int | None = None,
        alpha: float = 0.25,
        shrink: bool = False,
        error_mode: str = "fit",
        tau0: float = 0.0,
        cov_mode: str | None = None,
        kernel: str = "rbf",
        correlated: bool = True,
        gp_length: float | None = None,
    ):
        """Take the continuum limit at fixed renormalized coupling g^2."""
        if error_mode not in ("fit", "shifted", "gp"):
            raise BetaFunctionException("error_mode must be 'fit', 'shifted', or 'gp'")
        if not self.ntrp_fits:
            raise BetaFunctionException("Must run iv_ntrp before cnt_xtrp")
        if cov_mode is None:
            if diagonal:
                cov_mode = "diagonal"
            elif shrink:
                cov_mode = "shrink"
            else:
                cov_mode = "kernel" if error_mode == "gp" else "empirical"

        self.continuum = self._reset_stage(self.continuum)
        self._assign_stage_aliases()
        self.continuum.model = self._configured_model(
            self._cnt_fcn if fcn is None else fcn,
            prior,
            {"beta": [0.0], "slope": [0.0]} if p0 is None else p0,
        )
        self.cnt_fcn = self.continuum.model.evaluate
        ntrp_eval = self._wrap_model(self.ntrp_fcn if ntrp_fcn is None else ntrp_fcn)
        fit_fcn = self.continuum.model.fcn
        fit_prior = self.continuum.model.prior
        fit_p0 = self.continuum.model.p0

        if g2_round_precis is None:
            decimal_step = _decimal.Decimal(str(dg2))
            g2_round_precis = -decimal_step.as_tuple().exponent
        g2_values = _numpy.round(_numpy.arange(mng2, mxg2 + dg2, dg2), g2_round_precis)

        self.mnt, self.mxt = mnt, mxt
        self.continuum.metadata = {
            "window": (mnt, mxt), "g2_range": (mng2, mxg2, dg2),
            "cov_mode": cov_mode, "kernel": kernel, "alpha": alpha,
            "error_mode": error_mode, "tau0": tau0, "correlated": correlated,
            "gp_amplitudes": {},
        }
        flow_obs_pairs = [(flow, obs) for flow in self.ntrp_fits for obs in self.ntrp_fits[flow]]
        for flow, obs in flow_obs_pairs:
            self.g2s.setdefault(flow, {}).setdefault(obs, [])
            self.betas.setdefault(flow, {}).setdefault(obs, [])
            self.cnt_fits.setdefault(flow, {}).setdefault(obs, [])
            self.continuum.quality.setdefault(flow, {}).setdefault(obs, [])

            if v >= 1:
                print("Working on flow,discr. = " + ",".join([flow, obs]))

            iterator = g2_values if v == 0 else _tqdm(g2_values)
            for g2 in iterator:
                times = [
                    ft for ft in self.ntrp_fits[flow][obs]
                    if float(ft) - tau0 > 0.0
                    and mnt <= float(ft) - tau0 <= mxt
                    and self.ntrp_nf[flow][obs][ft][0] <= g2 <= self.ntrp_nf[flow][obs][ft][-1]
                ]
                if len(times) < 2:
                    continue
                times.sort(key=float)

                measured_times = _numpy.array([float(ft) for ft in times])
                nominal_times = measured_times - tau0
                x_data = 1.0 / nominal_times
                jacobian = nominal_times / measured_times
                beta_values = _numpy.array([
                    factor * ntrp_eval(g2, self.ntrp_fits[flow][obs][ft])
                    for factor, ft in zip(jacobian, times)
                ])
                log_times = _numpy.log(nominal_times)
                distances = [_numpy.subtract.outer(log_times, log_times)]
                means = _gvar.mean(beta_values)

                if error_mode == "shifted":
                    sdevs = _gvar.sdev(beta_values)
                    y_data = _gvar.gvar(means, sdevs)
                    fit = self.continuum.model.fit(
                        self._run_fit, self._run_fit_with_x_errors,
                        FitInput(x=x_data, y=y_data, labels=times),
                    )
                    half_diff = self._shifted_fit_beta(x_data, means, sdevs, times)
                    central = _gvar.mean(self.continuum.model.evaluate(0.0, fit.p))
                    beta_value = _gvar.gvar(central, half_diff)
                    self._record_continuum_point(flow, obs, g2, fit, times, y_data, beta_value=beta_value, x_data=x_data)
                    continue

                weight_cov = self._weight_covariance(beta_values, cov_mode=cov_mode, alpha=alpha, kernel=kernel, distances=distances)

                extra_cov = None
                if error_mode == "gp":
                    central_fit = self._run_fit(x_data, _gvar.gvar(means, weight_cov), fit_fcn, fit_prior, fit_p0)
                    residuals = means - _gvar.mean(_numpy.atleast_1d(fit_fcn(x_data, central_fit.p)))
                    length = gp_length
                    if length is None:
                        spread = float(log_times.max() - log_times.min())
                        length = 0.5 * spread if spread > 0.0 else 1.0
                    unit_kernel = _numpy.outer(x_data, x_data) * _kernel_correlation(distances[0], length, kernel)
                    amplitude = self._tune_gp_amplitude(residuals, weight_cov, unit_kernel)
                    self.continuum.metadata["gp_amplitudes"][(flow, obs, float(g2))] = amplitude
                    extra_cov = amplitude * unit_kernel
                    weight_cov = weight_cov + extra_cov

                if correlated:
                    params, fit = self._correlated_weighted_fit(
                        x_data, beta_values, fit_fcn, weight_cov,
                        prior=fit_prior, p0=fit_p0, extra_cov=extra_cov,
                    )
                    self._record_continuum_point(flow, obs, g2, fit, times, beta_values, params=params, x_data=x_data)
                else:
                    y_data = _gvar.gvar(means, weight_cov)
                    fit = self.continuum.model.fit(
                        self._run_fit, self._run_fit_with_x_errors,
                        FitInput(x=x_data, y=y_data, labels=times),
                    )
                    self._record_continuum_point(flow, obs, g2, fit, times, y_data, x_data=x_data)

                if v >= 2:
                    print(self.betas[flow][obs][-1])

    def tune_tshift(self, taus, mnt, mxt, mng2, mxg2, dg2=0.2, v=1, **cnt_kwargs):
        """Scan tau0 and pick the flattest continuum extrapolation."""
        snapshot = self.continuum
        metrics: dict[float, float] = {}
        for tau0 in taus:
            options = {"cov_mode": "diagonal", "error_mode": "fit", "correlated": False}
            options.update(cnt_kwargs)
            self.cnt_xtrp(mnt=mnt, mxt=mxt, mng2=mng2, mxg2=mxg2, dg2=dg2, v=0, tau0=tau0, **options)
            pulls = []
            for flow in self.cnt_fits:
                for obs in self.cnt_fits[flow]:
                    for params in self.cnt_fits[flow][obs]:
                        slope = params["slope"][0]
                        if slope.sdev > 0.0:
                            pulls.append((slope.mean / slope.sdev) ** 2)
            metrics[float(tau0)] = float(_numpy.mean(pulls)) if pulls else _numpy.inf
            if v >= 1:
                print(f"tau0 = {float(tau0):+.3f}:  <(slope/err)^2> = {metrics[float(tau0)]:.3f}  ({len(pulls)} fits)")
        self.continuum = snapshot
        self._assign_stage_aliases()
        best = min(metrics, key=metrics.get)
        if v >= 1:
            print(f"flattest extrapolation at tau0 = {best:+.3f}")
        return best, metrics

    def cnt_xtrp_global(
        self,
        mnt: float,
        mxt: float,
        mng2: float,
        mxg2: float,
        dg2: float = 0.1,
        intercepts: str = "free",
        loops: int = 2,
        correction_order: int = 2,
        correction_width: float = 5.0,
        artifact_fcn=None,
        artifact_prior: dict | None = None,
        artifact_p0: dict | None = None,
        ntrp_fcn=None,
        cov_mode: str = "kernel",
        kernel: str = "rbf",
        alpha: float = 0.25,
        gp: bool = False,
        gp_lengths: tuple[float, float] | None = None,
        tau0: float = 0.0,
        g2_round_precis: int | None = None,
        v: int = 1,
    ):
        """One joint continuum fit over the whole (g^2, a^2/t) plane."""
        if intercepts not in ("free", "perturbative"):
            raise BetaFunctionException("intercepts must be 'free' or 'perturbative'")
        if not self.ntrp_fits:
            raise BetaFunctionException("Must run iv_ntrp before cnt_xtrp_global")

        ntrp_eval = self._wrap_model(self.ntrp_fcn if ntrp_fcn is None else ntrp_fcn)
        self.continuum = self._reset_stage(self.continuum)
        self._assign_stage_aliases()
        self.continuum.model = None
        self.cnt_fcn = None

        if g2_round_precis is None:
            decimal_step = _decimal.Decimal(str(dg2))
            g2_round_precis = -decimal_step.as_tuple().exponent
        g2_values = _numpy.round(_numpy.arange(mng2, mxg2 + dg2, dg2), g2_round_precis)

        if artifact_fcn is None:
            artifact_fcn, default_prior, default_p0 = self._default_artifact_model()
            artifact_prior = default_prior if artifact_prior is None else artifact_prior
            artifact_p0 = default_p0 if artifact_p0 is None else artifact_p0
        elif artifact_prior is None or artifact_p0 is None:
            raise BetaFunctionException("Custom artifact_fcn requires artifact_prior and artifact_p0")

        perturbative = self.perturbative_beta_function
        correction_names = [f"pt_c{n}" for n in range(1, correction_order + 1)]

        self.mnt, self.mxt = mnt, mxt
        self.continuum.metadata = {
            "method": "global", "window": (mnt, mxt), "g2_range": (mng2, mxg2, dg2),
            "intercepts": intercepts, "loops": loops, "cov_mode": cov_mode,
            "kernel": kernel, "alpha": alpha, "gp": gp, "tau0": tau0, "gp_amplitudes": {},
        }

        flow_obs_pairs = [(flow, obs) for flow in self.ntrp_fits for obs in self.ntrp_fits[flow]]
        for flow, obs in flow_obs_pairs:
            if v >= 1:
                print("Global continuum fit for flow,discr. = " + ",".join([flow, obs]))

            kept_g2, y_values, g2_index = [], [], []
            g2_coord, x_coord, logt_coord, labels = [], [], [], []
            for g2 in g2_values:
                times = [
                    ft for ft in self.ntrp_fits[flow][obs]
                    if float(ft) - tau0 > 0.0
                    and mnt <= float(ft) - tau0 <= mxt
                    and self.ntrp_nf[flow][obs][ft][0] <= g2 <= self.ntrp_nf[flow][obs][ft][-1]
                ]
                if len(times) < 2:
                    continue
                times.sort(key=float)
                index = len(kept_g2)
                kept_g2.append(float(g2))
                for ft in times:
                    measured = float(ft)
                    nominal = measured - tau0
                    y_values.append((nominal / measured) * ntrp_eval(g2, self.ntrp_fits[flow][obs][ft]))
                    g2_index.append(index)
                    g2_coord.append(float(g2))
                    x_coord.append(1.0 / nominal)
                    logt_coord.append(_numpy.log(nominal))
                    labels.append(f"{g2}/{ft}")

            if len(kept_g2) < 2:
                continue

            y_values = _numpy.array(y_values)
            g2_index = _numpy.array(g2_index)
            g2_coord = _numpy.array(g2_coord)
            x_coord = _numpy.array(x_coord)
            logt_coord = _numpy.array(logt_coord)
            u_coord = g2_coord / perturbative.nrm
            means = _gvar.mean(y_values)

            prior = dict(self._normalize_param_dict(artifact_prior))
            p0 = dict(self._normalize_param_dict(artifact_p0))
            if intercepts == "free":
                guesses = _numpy.array([_numpy.mean(means[g2_index == i]) for i in range(len(kept_g2))])
                scale = 100.0 * max(float(_numpy.max(_numpy.abs(guesses))), 1.0)
                prior["beta"] = [_gvar.gvar(float(g), scale) for g in guesses]
                p0["beta"] = list(map(float, guesses))

                def mean_fcn(x, p):
                    base = _numpy.array(p["beta"], dtype=object)[g2_index]
                    return base + artifact_fcn(g2_coord, x_coord, p)
            else:
                pt_values = perturbative(g2_coord, loops=loops)
                for name in correction_names:
                    prior[name] = [_gvar.gvar(0.0, correction_width)]
                    p0[name] = [0.0]

                def mean_fcn(x, p):
                    correction, term = 1.0, 1.0
                    for name in correction_names:
                        term = term * u_coord
                        correction = correction + p[name][0] * term
                    return pt_values * correction + artifact_fcn(g2_coord, x_coord, p)

            distances = [
                _numpy.subtract.outer(logt_coord, logt_coord),
                _numpy.subtract.outer(g2_coord, g2_coord),
            ]
            weight_cov = self._weight_covariance(y_values, cov_mode=cov_mode, alpha=alpha, kernel=kernel, distances=distances)

            extra_cov = None
            if gp:
                dummy_x = _numpy.arange(len(y_values))
                central_fit = self._run_fit(dummy_x, _gvar.gvar(means, weight_cov), mean_fcn, prior, p0)
                residuals = means - _gvar.mean(_numpy.atleast_1d(mean_fcn(dummy_x, central_fit.p)))
                lengths_pair = gp_lengths
                if lengths_pair is None:
                    logt_spread = float(logt_coord.max() - logt_coord.min())
                    g2_spread = float(g2_coord.max() - g2_coord.min())
                    lengths_pair = (
                        0.5 * logt_spread if logt_spread > 0.0 else 1.0,
                        0.5 * g2_spread if g2_spread > 0.0 else 1.0,
                    )
                unit_kernel = (
                    _numpy.outer(x_coord, x_coord)
                    * _kernel_correlation(distances[0], lengths_pair[0], kernel)
                    * _kernel_correlation(distances[1], lengths_pair[1], kernel)
                )
                amplitude = self._tune_gp_amplitude(residuals, weight_cov, unit_kernel)
                self.continuum.metadata["gp_amplitudes"][(flow, obs)] = amplitude
                extra_cov = amplitude * unit_kernel
                weight_cov = weight_cov + extra_cov

            params, fit = self._correlated_weighted_fit(
                _numpy.arange(len(y_values)), y_values, mean_fcn, weight_cov,
                prior=prior, p0=p0, extra_cov=extra_cov,
            )

            if intercepts == "free":
                betas = list(params["beta"])
            else:
                kept = _numpy.array(kept_g2)
                u_kept = kept / perturbative.nrm
                correction, term = 1.0, 1.0
                for name in correction_names:
                    term = term * u_kept
                    correction = correction + params[name][0] * term
                betas = list(perturbative(kept, loops=loops) * correction)

            self.g2s.setdefault(flow, {})[obs] = kept_g2
            self.betas.setdefault(flow, {})[obs] = betas
            self.cnt_fits.setdefault(flow, {})[obs] = [params]
            self.continuum.quality.setdefault(flow, {})[obs] = [self._quality_of_fit(fit)]
            self.continuum.store(
                "inputs", (flow, obs, "global"),
                FitInput(
                    x=_numpy.column_stack([g2_coord, x_coord]),
                    y=y_values, labels=labels,
                    meta={"g2_index": g2_index, "kept_g2": kept_g2},
                ),
            )
            self.continuum.store("domains", (flow, obs, "global"), kept_g2)

            if v >= 1:
                quality = self.continuum.quality[flow][obs][0]
                print(
                    f"  {len(y_values)} points, {len(kept_g2)} g^2 values, "
                    f"chi2/dof = {quality['chi2'] / max(quality['dof'], 1):.2f}"
                )
            if v >= 2:
                print(fit)

        _gc.collect()

    def continuum_window_scan(self, windows, mng2, mxg2, dg2=0.1, v=0, **cnt_kwargs):
        """Rerun the continuum limit over several flow-time windows for systematics."""
        if len(windows) < 2:
            raise BetaFunctionException("continuum_window_scan needs at least two windows")
        scan: dict = {}
        for mnt, mxt in windows:
            self.cnt_xtrp(mnt=mnt, mxt=mxt, mng2=mng2, mxg2=mxg2, dg2=dg2, v=v, **cnt_kwargs)
            scan[(mnt, mxt)] = self.continuum
        self._window_scan = scan
        central = windows[0]
        self.continuum = scan[central]
        self._assign_stage_aliases()
        return scan

    def continuum_systematics(self, flow: str, obs: str) -> list[dict[str, float]]:
        """Combine a window scan into a per-g^2 error budget."""
        scan = getattr(self, "_window_scan", None)
        if not scan:
            raise BetaFunctionException("Must run continuum_window_scan before continuum_systematics")
        windows = list(scan.keys())
        central_store = scan[windows[0]]
        central_g2s = central_store.inputs["g2s"][flow][obs]
        central_betas = central_store.inputs["betas"][flow][obs]
        budget = []
        for index, g2 in enumerate(central_g2s):
            central = central_betas[index]
            window_means = []
            for window in windows:
                store = scan[window]
                g2_list = store.inputs["g2s"][flow][obs]
                if g2 in g2_list:
                    window_means.append(_gvar.mean(store.inputs["betas"][flow][obs][g2_list.index(g2)]))
            if len(window_means) < 2:
                continue
            statistical = _gvar.sdev(central)
            systematic = 0.5 * (max(window_means) - min(window_means))
            budget.append({
                "g2": float(g2), "beta": float(_gvar.mean(central)),
                "stat": float(statistical), "syst": float(systematic),
                "total": float(_numpy.hypot(statistical, systematic)),
            })
        return budget

    def continuum_curve(self, flow: str, obs: str):
        """Return final continuum beta-function points for one flow/observable pair."""
        return _numpy.array(self.g2s[flow][obs]), self.betas[flow][obs]
