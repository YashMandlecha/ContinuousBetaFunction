"""BetaFunction: the main staged analysis engine."""
from __future__ import annotations

from collections.abc import Callable
import datetime as _datetime
import decimal as _decimal
import gc as _gc
import importlib.metadata as _importlib_metadata
import inspect as _inspect
import itertools as _itertools
import platform as _platform
import time as _time

import gvar as _gvar
import lsqfit as _lsqfit
import numpy as _numpy
from tqdm import tqdm as _tqdm

from .exceptions import BetaFunctionException
from .processing import SetupBetaFunction
from .fitting import FitInput, FitModel, StageStore, polynomial_interpolation  # noqa: F401
from .config import InterpolationSpec, AnalysisConfig, AnalysisResult


# ---------------------------------------------------------------------------
# Private kernel and covariance helpers
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
    """Fit a product-kernel + nugget model to an empirical correlation matrix.

    Action-continuous data have empirical correlation matrices that are
    genuinely (near-)rank-deficient, so the model correlation is
    amp * prod_k k(d_k/ell_k) off the diagonal with exact ones on the
    diagonal; the implied nugget 1 - amp >= nugget_floor keeps the model
    full rank by construction.  Correlation lengths are chosen by a
    log-spaced grid scan minimizing the squared off-diagonal mismatch, with
    the amplitude solved analytically at each grid point.  Returns
    (model_corr, lengths, nugget).
    """
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
# BetaFunction
# ---------------------------------------------------------------------------

class BetaFunction(SetupBetaFunction):
    """Method-centric continuous beta-function analysis built around staged fits."""

    STAGE_NAME_MAP = {
        "chiral": "chiral",
        "iv": "infinite_volume",
        "infinite_volume": "infinite_volume",
        "ntrp": "interpolation",
        "interpolation": "interpolation",
        "cnt": "continuum",
        "continuum": "continuum",
    }

    def __init__(
        self,
        nc: float | int = 3.0,
        nf: float | int | None = None,
        gauge_action: str = "s",
        logfn: str | None = None,
    ):
        # Explicit base-class call for IPython-autoreload safety.
        SetupBetaFunction.__init__(self, nc=nc, nf=nf, gauge_action=gauge_action, logfn=logfn)
        self._thin = None
        self._reset_analysis_state()

    def _reset_analysis_state(self) -> None:
        self.chiral = StageStore(name="chiral")
        self.infinite_volume = StageStore(name="infinite_volume")
        self.interpolation = StageStore(name="interpolation")
        self.continuum = StageStore(name="continuum")

        self.ch_fits = self.chiral.fits
        self.ch_qof = self.chiral.quality
        self.iv_fits = self.infinite_volume.fits
        self.iv_qof = self.infinite_volume.quality
        self.ntrp_fits = self.interpolation.fits
        self.ntrp_qof = self.interpolation.quality
        self.ntrp_nf = self.interpolation.domains
        self.cnt_fits = self.continuum.fits
        self.g2s = self.continuum.inputs.setdefault("g2s", {})
        self.betas = self.continuum.inputs.setdefault("betas", {})

    def _require_avg_data(self, stage: str) -> None:
        if not self.avg_data:
            raise BetaFunctionException(f"Must run process_data before {stage}")

    def _normalize_param_dict(self, params: dict | None) -> dict | None:
        if params is None:
            return None
        normalized = {}
        for key, value in params.items():
            if key == "x":
                normalized[key] = value
            elif isinstance(value, (list, tuple, _numpy.ndarray)):
                normalized[key] = list(value)
            else:
                normalized[key] = [value]
        return normalized

    def _new_nested_fit_entry(self, flows: list[str], xtrp: list[str]) -> dict:
        return {x: {flow: {obs: {} for obs in self.os} for flow in flows} for x in xtrp}

    def _reset_stage(self, stage: StageStore) -> StageStore:
        return StageStore(name=stage.name)

    def _mass_value(self, mass: str) -> float:
        return float(mass.replace("p", "."))

    def _volume_value(self, volume: str) -> float:
        return _numpy.prod([*map(float, volume.replace("t", "l").split("l")[1:])])

    def _wrap_model(self, fcn: Callable) -> Callable:
        signature = _inspect.signature(fcn)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (_inspect.Parameter.POSITIONAL_ONLY, _inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) == 1:
            return lambda x, p: fcn(p)
        return fcn

    def _quality_of_fit(self, fit) -> dict:
        return {
            "chi2": fit.chi2,
            "dof": fit.dof,
            "p-value": fit.Q,
            "logGBF": getattr(fit, "logGBF", None),
        }

    def _run_fit(self, x, y, fit_fcn: Callable, prior: dict | None, p0: dict | None):
        return _lsqfit.nonlinear_fit(data=(x, y), fcn=fit_fcn, prior=prior, p0=p0)

    def _run_fit_with_x_errors(self, y, fit_fcn: Callable, prior: dict, p0: dict | None):
        def wrapper(parameters):
            return fit_fcn(parameters["x"], parameters)

        return _lsqfit.nonlinear_fit(data=y, fcn=wrapper, prior=prior, p0=p0)

    def _flatten(self, dct: dict, prefix: tuple = tuple(), stop: int = 5) -> dict:
        flattened = []
        for key, value in dct.items():
            next_prefix = prefix + (key,) if prefix else (key,)
            if isinstance(value, dict) and len(next_prefix) < stop:
                flattened.extend(self._flatten(value, prefix=next_prefix, stop=stop).items())
            else:
                flattened.append((next_prefix, value))
        return dict(flattened)

    def _gather_ch_info(self, coupling: str, volume: str, xtrp: list[str], mnt: float, mxt: float):
        masses = [mass for mass in self.avg_data[coupling][volume] if mass != "0p00"]
        flows = sorted({flow for mass in masses for flow in self.avg_data[coupling][volume][mass]})
        times = sorted(
            {
                flow_time
                for mass in masses
                for flow in flows
                for x in xtrp
                for obs in self.os
                for flow_time in self.avg_data[coupling][volume][mass][flow]["_".join([x, obs])]
                if mnt <= float(flow_time) <= mxt
            },
            key=float,
        )
        targets = [
            (x, flow, obs, flow_time)
            for x in xtrp
            for flow in flows
            for obs in self.os
            for flow_time in times[:: self._thin]
        ]
        return masses, flows, targets

    def _gather_iv_info(self, coupling: str, xtrp: list[str], mnt: float, mxt: float):
        mass = "0p00"
        volumes = sorted(self.avg_data[coupling].keys())
        flows = sorted({flow for volume in volumes for flow in self.avg_data[coupling][volume][mass]})
        times = sorted(
            {
                flow_time
                for volume in volumes
                for flow in flows
                for x in xtrp
                for obs in self.os
                for flow_time in self.avg_data[coupling][volume][mass][flow]["_".join([x, obs])]
                if mnt <= float(flow_time) <= mxt
            },
            key=float,
        )
        targets = [
            (x, flow, obs, flow_time)
            for x in xtrp
            for flow in flows
            for obs in self.os
            for flow_time in times[:: self._thin]
        ]
        return volumes, flows, targets

    def _clear_gvar_state(self) -> None:
        """Also drop stage fits and window scans, which hold gvars."""
        # Explicit base-class call: zero-arg super() breaks under IPython
        # autoreload, which re-patches methods and severs the __class__ cell.
        SetupBetaFunction._clear_gvar_state(self)
        self._reset_analysis_state()
        self._window_scan = {}
        self._thin = None

    def _rehome_prior(self, prior: dict | None) -> dict | None:
        """Recreate prior gvars inside the active covariance environment.

        Priors may have been built (e.g. via InterpolationSpec) before a
        reprocess retired their gvar environment; gvars from different
        environments cannot be combined, so they are recreated here from
        their means and standard deviations.  Cross-correlations between
        prior entries are not preserved (priors are normally independent).
        """
        if prior is None:
            return None
        rehomed = {}
        for key, values in prior.items():
            rehomed[key] = [
                _gvar.gvar(value.mean, value.sdev) if isinstance(value, _gvar.GVar) else value
                for value in values
            ]
        return rehomed

    def _configured_model(
        self,
        fcn: Callable,
        prior: dict | None,
        p0: dict | None,
        xerrors: bool = False,
    ) -> FitModel:
        return FitModel(
            fcn=self._wrap_model(fcn),
            prior=self._rehome_prior(self._normalize_param_dict(prior)),
            p0=self._normalize_param_dict(p0),
            xerrors=xerrors,
        )

    def _assign_stage_aliases(self) -> None:
        self.ch_fits = self.chiral.fits
        self.ch_qof = self.chiral.quality
        self.iv_fits = self.infinite_volume.fits
        self.iv_qof = self.infinite_volume.quality
        self.ntrp_fits = self.interpolation.fits
        self.ntrp_qof = self.interpolation.quality
        self.ntrp_nf = self.interpolation.domains
        self.cnt_fits = self.continuum.fits
        self.g2s = self.continuum.inputs.setdefault("g2s", {})
        self.betas = self.continuum.inputs.setdefault("betas", {})

    def _resolve_stage(self, stage_name: str) -> StageStore:
        """Resolve user-facing stage aliases to a concrete StageStore instance."""
        if stage_name not in self.STAGE_NAME_MAP:
            raise BetaFunctionException(
                "Unknown stage name.",
                "Allowed names:",
                ", ".join(sorted(self.STAGE_NAME_MAP.keys())),
            )
        return getattr(self, self.STAGE_NAME_MAP[stage_name])

    def available_stage_names(self) -> list[str]:
        """Return all accepted stage names and aliases."""
        return sorted(self.STAGE_NAME_MAP.keys())

    def stage_store(self, stage: str) -> StageStore:
        """Return the StageStore object for a stage alias."""
        return self._resolve_stage(stage)

    def _flatten_quality(self, quality_dict: dict) -> list[tuple[tuple[str, ...], dict]]:
        """Flatten a nested quality tree, treating metric dicts (with chi2/dof) as leaves."""

        def walk(node, prefix: tuple):
            if isinstance(node, dict) and "chi2" in node and "dof" in node:
                yield prefix, node
            elif isinstance(node, dict):
                for key, value in node.items():
                    yield from walk(value, prefix + (key,))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    yield from walk(value, prefix + (str(index),))

        return sorted(walk(quality_dict, tuple()), key=lambda item: item[0])

    def quality_table(self, stage: str) -> list[dict[str, object]]:
        """Return flattened quality-of-fit entries for a stage."""
        store = self._resolve_stage(stage)
        rows = []
        for keys, metrics in self._flatten_quality(store.quality):
            row = {
                "stage": store.name,
                "target": "/".join(map(str, keys)),
            }
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
        return rows

    def stage_summary(self, stage: str) -> dict[str, object]:
        """Return aggregate diagnostics for a stage."""
        rows = self.quality_table(stage)
        if not rows:
            return {
                "stage": self._resolve_stage(stage).name,
                "n_fits": 0,
                "mean_chi2_per_dof": None,
                "mean_pvalue": None,
            }

        chi2_over_dof = []
        p_values = []
        for row in rows:
            chi2 = row.get("chi2")
            dof = row.get("dof")
            p_value = row.get("p-value")
            if dof not in (None, 0):
                chi2_over_dof.append(float(chi2) / float(dof))
            if p_value is not None:
                p_values.append(float(p_value))

        return {
            "stage": self._resolve_stage(stage).name,
            "n_fits": len(rows),
            "mean_chi2_per_dof": float(_numpy.mean(chi2_over_dof)) if chi2_over_dof else None,
            "mean_pvalue": float(_numpy.mean(p_values)) if p_values else None,
        }

    def analysis_summary(self) -> dict[str, dict[str, object]]:
        """Return summaries for all standard stages."""
        return {
            "chiral": self.stage_summary("chiral"),
            "infinite_volume": self.stage_summary("infinite_volume"),
            "interpolation": self.stage_summary("interpolation"),
            "continuum": self.stage_summary("continuum"),
        }

    def provenance(self) -> dict[str, object]:
        """Capture reproducibility metadata: versions, platform, physics parameters."""
        package_versions = {}
        for package in ("numpy", "scipy", "gvar", "lsqfit"):
            try:
                package_versions[package] = _importlib_metadata.version(package)
            except _importlib_metadata.PackageNotFoundError:
                package_versions[package] = "unknown"
        return {
            "timestamp": _datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "python": _platform.python_version(),
            "platform": _platform.platform(),
            "packages": package_versions,
            "nc": self.nc,
            "nf": self.nf,
            "gauge_action": self.gauge_action,
            "binsize": self._binsize,
        }

    def report(self) -> str:
        """Return a human-readable status report of every analysis stage."""
        lines = [repr(self), 50 * "-"]
        for stage, payload in self.analysis_summary().items():
            chi2 = payload["mean_chi2_per_dof"]
            chi2_text = "n/a" if chi2 is None else f"{chi2:.2f}"
            lines.append(f"{stage:<18} fits: {payload['n_fits']:>5}   <chi2/dof>: {chi2_text}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_ensembles = sum(
            len(masses)
            for volumes in self.avg_data.values()
            for masses in volumes.values()
        )
        return (
            f"BetaFunction(nc={self.nc}, nf={self.nf}, "
            f"couplings={len(self.avg_data)}, ensembles={n_ensembles})"
        )

    def get_ch_data(self, coupling: str, volume: str, flow: str, x: str, obs: str, t: str, masses: list[str]):
        """Build chiral extrapolation fit input for a specific (beta, volume, flow, obs, t)."""
        xo = "_".join([x, obs])
        valid_masses = [mass for mass in masses if mass != "0p00" and mass not in self._ch_exclude[coupling][volume]]
        return FitInput(
            x=[self._mass_value(mass) for mass in valid_masses],
            y=[self.avg_data[coupling][volume][mass][flow][xo][t] for mass in valid_masses],
            labels=valid_masses,
        )

    def get_fv_data(self, coupling: str, flow: str, x: str, obs: str, t: str, volumes: list[str]):
        """Build infinite-volume extrapolation fit input for a specific (beta, flow, obs, t)."""
        xo = "_".join([x, obs])
        mass = "0p00"
        valid_volumes = [volume for volume in volumes if volume not in self._iv_exclude[coupling]]
        return FitInput(
            x=[1.0 / self._volume_value(volume) for volume in valid_volumes],
            y=[self.avg_data[coupling][volume][mass][flow][xo][t] for volume in valid_volumes],
            labels=valid_volumes,
        )

    def _fit_inputs_from_iv(self, fake_iv_data: bool) -> dict:
        iv_data = {}
        if not fake_iv_data:
            if not self.iv_fits:
                raise BetaFunctionException("Must run iv_xtrp (or load iv fits) before iv_ntrp")
            for (_, x, flow, obs, flow_time), fit_params in self._flatten(self.iv_fits, stop=5).items():
                iv_data.setdefault(flow, {}).setdefault(obs, {}).setdefault(flow_time, {}).setdefault(x, [])
                iv_data[flow][obs][flow_time][x].append(self.infinite_volume.model.evaluate(0.0, fit_params))
            return iv_data

        self._require_avg_data("iv_ntrp with fake_iv_data")
        self._iv_exclude = {coupling: [] for coupling in self.avg_data}
        for coupling in self.data:
            volumes, _, targets = self._gather_iv_info(coupling, ["g2", "beta"], self._min_fv_flt, self._max_fv_flt)
            for x, flow, obs, flow_time in targets:
                iv_data.setdefault(flow, {}).setdefault(obs, {}).setdefault(flow_time, {}).setdefault(x, [])
                iv_data[flow][obs][flow_time][x].append(self.get_fv_data(coupling, flow, x, obs, flow_time, volumes).y[-1])
        return iv_data

    def _record_continuum_point(self, flow: str, obs: str, g2: float, fit, times: list[str], y_data, params=None, beta_value=None, x_data=None) -> None:
        params = fit.p if params is None else params
        self.cnt_fits[flow][obs].append(params)
        self.continuum.quality[flow][obs].append(self._quality_of_fit(fit))
        self.g2s[flow][obs].append(g2)
        self.betas[flow][obs].append(
            self.continuum.model.evaluate(0.0, params) if beta_value is None else beta_value
        )
        if x_data is None:
            x_data = 1.0 / _numpy.array([float(flow_time) for flow_time in times])
        self.continuum.store(
            "domains",
            (flow, obs, str(g2)),
            times,
        )
        self.continuum.store(
            "inputs",
            (flow, obs, str(g2)),
            FitInput(x=_numpy.asarray(x_data), y=y_data, labels=times),
        )

    def ch_xtrp(
        self,
        mnt: float | None = None,
        mxt: float | None = None,
        exclude: dict[str, list[str]] | None = None,
        fcn: Callable | None = None,
        prior: dict[str, object] | None = None,
        p0: dict[str, object] | None = None,
        xtrp: list[str] = ["g2", "beta"],
        v: int = 1,
        thin: int = 1,
        postprocess: bool = True,
    ):
        """Perform chiral extrapolation to m=0 over selected flow-time windows."""
        self._require_avg_data("ch_xtrp")
        self.chiral = self._reset_stage(self.chiral)
        self._assign_stage_aliases()
        mnt = self._min_fv_flt if mnt is None else mnt
        mxt = self._max_fv_flt if mxt is None else mxt
        self.chiral.model = self._configured_model(
            self._ch_xtrp_fcn if fcn is None else fcn,
            prior,
            {"k1(t;beta,L)": [0.0], "k2(t;beta,L)": [0.0]} if p0 is None else p0,
        )
        self.ch_fcn = self.chiral.model.evaluate
        self._thin = thin

        self._ch_exclude = (
            {coupling: {volume: [] for volume in self.avg_data[coupling]} for coupling in self.avg_data}
            if exclude is None
            else exclude
        )

        if v >= 1:
            print("Chiral extrapolation\n" + 50 * "~")

        for coupling in self.data:
            self.ch_fits[coupling], self.ch_qof[coupling] = {}, {}
            for volume in self.avg_data[coupling]:
                masses, flows, targets = self._gather_ch_info(coupling, volume, xtrp, mnt, mxt)
                self.ch_fits[coupling][volume] = self._new_nested_fit_entry(flows, xtrp)
                self.ch_qof[coupling][volume] = self._new_nested_fit_entry(flows, xtrp)

                if len(masses) <= 1:
                    continue

                if postprocess:
                    self.avg_data[coupling][volume]["0p00"] = {
                        flow: {"_".join([x_key, obs]): {} for x_key in xtrp for obs in self.os} for flow in flows
                    }
                    for flow in flows:
                        self.avg_data[coupling][volume]["0p00"][flow]["flow_times"] = []

                if v >= 1:
                    self._start_timer()

                iterator = targets if v == 0 else _tqdm(targets)
                for x, flow, obs, flow_time in iterator:
                    xo = "_".join([x, obs])
                    try:
                        data = self.get_ch_data(coupling, volume, flow, x, obs, flow_time, masses)
                        if data.point_count < 2:
                            continue
                        fit = self.chiral.model.fit(self._run_fit, self._run_fit_with_x_errors, data)
                        self.chiral.record_fit(
                            (coupling, volume, x, flow, obs, flow_time),
                            fit.p,
                            self._quality_of_fit(fit),
                            fit_input=data,
                        )
                        if postprocess:
                            self.avg_data[coupling][volume]["0p00"][flow][xo][flow_time] = self.chiral.model.evaluate(0.0, fit.p)
                        if postprocess and obs == self.os[0] and x == xtrp[0]:
                            self.avg_data[coupling][volume]["0p00"][flow]["flow_times"].append(flow_time)
                    except KeyError as err:
                        self.log.write("ERROR: " + " ".join([coupling, x, flow, obs, flow_time]), repr(err))

                if v >= 1:
                    message = "Finished beta_b = " + coupling.replace("p", ".")
                    message += ", L^{Nd-1}xT = " + volume[1:].replace("l", "x").replace("t", "x")
                    message += " in " + str(self._stop_timer()) + " secs"
                    print(message)

    def iv_xtrp(
        self,
        mnt: float | None = None,
        mxt: float | None = None,
        exclude: dict[str, list[str]] | None = None,
        fcn: Callable | None = None,
        prior: dict[str, object] | None = None,
        p0: dict[str, object] | None = None,
        xtrp: list[str] = ["g2", "beta"],
        model_average: bool = False,
        v: int = 1,
        thin: int = 1,
    ):
        """Perform infinite-volume extrapolation at fixed bare coupling and flow time."""
        self._require_avg_data("iv_xtrp")
        if model_average:
            raise BetaFunctionException("Bayesian model averaging for iv_xtrp has been disabled.")

        self.infinite_volume = self._reset_stage(self.infinite_volume)
        self._assign_stage_aliases()

        mnt = self._min_fv_flt if mnt is None else mnt
        mxt = self._max_fv_flt if mxt is None else mxt
        self.infinite_volume.model = self._configured_model(
            self._iv_xtrp_fcn if fcn is None else fcn,
            prior,
            {"k1(t;beta)": [0.0], "k2(t;beta)": [0.0]} if p0 is None else p0,
        )
        self.iv_fcn = self.infinite_volume.model.evaluate

        if self._thin is not None and self._thin != thin:
            raise BetaFunctionException("thin already set and not equal to input")
        self._thin = thin

        self._iv_exclude = {coupling: [] for coupling in self.avg_data} if exclude is None else exclude

        if v >= 1:
            print("Infinite volume extrapolation\n" + 50 * "~")

        for coupling in self.data:
            if v >= 1:
                self._start_timer()
            volumes, flows, targets = self._gather_iv_info(coupling, xtrp, mnt, mxt)
            self.iv_fits[coupling] = self._new_nested_fit_entry(flows, xtrp)
            self.iv_qof[coupling] = self._new_nested_fit_entry(flows, xtrp)

            iterator = targets if v == 0 else _tqdm(targets)
            for x, flow, obs, flow_time in iterator:
                try:
                    data = self.get_fv_data(coupling, flow, x, obs, flow_time, volumes)
                    if data.point_count < 2:
                        continue
                    fit = self.infinite_volume.model.fit(self._run_fit, self._run_fit_with_x_errors, data)
                    self.infinite_volume.record_fit(
                        (coupling, x, flow, obs, flow_time),
                        fit.p,
                        self._quality_of_fit(fit),
                        fit_input=data,
                    )
                except KeyError as err:
                    self.log.write("ERROR: " + " ".join([coupling, x, flow, obs, flow_time]), repr(err))

            if v >= 1:
                message = "Finished beta_b = " + coupling.replace("p", ".")
                message += " in " + str(self._stop_timer()) + " secs"
                print(message)

        _gc.collect()

    def iv_ntrp(
        self,
        fcn: Callable | None = None,
        prior: dict[str, object] | None = None,
        p0: dict[str, object] | None = None,
        v: int = 1,
        xerrors: bool = False,
        fake_iv_data: bool = False,
        emp_bayes_fcn: Callable | None = None,
        eblb: float | None = None,
        ebub: float | None = None,
        ebnpt: int | None = None,
        ebprms: list[str] | None = None,
        ebalg: str = "steffen",
    ):
        """Interpolate infinite-volume data in g^2 using a user-specified model."""
        _ = eblb, ebub, ebnpt, ebprms, ebalg
        if fcn is None:
            raise BetaFunctionException("Must provide interpolating function as callable fcn(x,p) or fcn(p)")
        if emp_bayes_fcn is not None:
            raise BetaFunctionException("Empirical-Bayes shrinkage support has been removed from iv_ntrp.")

        self.interpolation = self._reset_stage(self.interpolation)
        self._assign_stage_aliases()
        if p0 is None:
            raise BetaFunctionException("Must provide starting values for parameters")

        self.interpolation.model = self._configured_model(
            fcn,
            {} if xerrors and prior is None else prior,
            p0,
            xerrors=xerrors,
        )
        self.ntrp_fcn = self.interpolation.model.evaluate
        iv_data = self._fit_inputs_from_iv(fake_iv_data=fake_iv_data)

        if v >= 1:
            print("Intermediate interpolation\n" + 50 * "~")

        iterator = self._flatten(iv_data, stop=3).items()
        iterator = iterator if v == 0 else _tqdm(iterator)
        for (flow, obs, flow_time), datum in iterator:
            if obs not in self.os:
                continue
            fit_input = FitInput(x=datum["g2"], y=datum["beta"])
            if fit_input.point_count < 2:
                continue
            fit = self.interpolation.model.fit(self._run_fit, self._run_fit_with_x_errors, fit_input)
            self.interpolation.record_fit(
                (flow, obs, flow_time),
                fit.p,
                self._quality_of_fit(fit),
                fit_input=fit_input,
                domain=[min(_gvar.mean(fit_input.x)), max(_gvar.mean(fit_input.x))],
            )

            if v >= 2:
                print(fit)

    def _cnt_fcn(self, x, p):
        return p["beta"][0] + p["slope"][0] * x

    def _covariance_mode(self, beta_values, diagonal: bool, shrink: bool, alpha: float):
        means = _gvar.mean(beta_values)
        covariance = _gvar.evalcov(beta_values)
        if diagonal:
            covariance = _numpy.diag(_numpy.diag(covariance))
        elif shrink:
            if not 0.0 <= alpha <= 1.0:
                raise BetaFunctionException("alpha must lie in [0, 1] when shrink=True")
            diagonal_covariance = _numpy.diag(_numpy.diag(covariance))
            covariance = (1.0 - alpha) * covariance + alpha * diagonal_covariance
        return _gvar.gvar(means, covariance)

    def _weight_covariance(self, y, cov_mode: str = "empirical", alpha: float = 0.25, kernel: str = "rbf", distances=None):
        """Return the covariance used to *weight* a fit of gvar data y.

        Modes: 'empirical' (full gvar covariance), 'diagonal',
        'shrink' (linear blend toward the diagonal), and 'kernel'
        (smooth stationary product-kernel + nugget model fit to the
        empirical correlation matrix; requires `distances`, a list of
        distance matrices, one per kernel factor).  The kernel mode is
        built for action-continuous data whose empirical correlation
        matrix is near-singular: it keeps the measured per-point errors
        and replaces only the correlation structure with a smooth,
        full-rank model.
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
        """Jacobian of fcn(x, p) w.r.t. flattened parameters via gvar autodiff."""
        flat, spec = _flatten_pdict(params)
        primaries = _gvar.gvar(_gvar.mean(flat), _numpy.ones(len(flat)))
        values = _numpy.atleast_1d(fcn(x, _unflatten_pdict(list(primaries), spec)))
        jacobian = _numpy.zeros((len(values), len(primaries)))
        for row, value in enumerate(values):
            if isinstance(value, _gvar.GVar):
                jacobian[row] = _gvar.deriv(value, primaries)
        return jacobian

    def _correlated_weighted_fit(self, x, y, fcn, weight_cov, prior=None, p0=None, extra_cov=None):
        """Weighted least-squares fit of gvar data that keeps outputs correlated.

        This is the trick lsqfit itself plays, generalized to an arbitrary
        weight matrix: solve the (prior-regulated) weighted least-squares
        problem for central values, then linearize the estimator around the
        solution so the fitted parameters become an explicit linear map of the
        original data gvars,

            p = p_hat + (J^T W J + P)^{-1} J^T W (y - <y>).

        Because the map is applied to the *original* gvars, the returned
        parameters carry exact correlations with the input data (and hence
        with every other quantity in the analysis), while their quoted errors
        are the honest sandwich errors of the true data covariance under the
        chosen weight.  `extra_cov` adds independent variance M K M^T for a
        GP-discrepancy kernel K.  Returns (params_dict, central_fit).
        """
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
        """Empirical-Bayes GP amplitude: maximize the Gaussian marginal likelihood."""
        scale = float(_numpy.mean(residuals * residuals))
        if scale <= 0.0:
            return 0.0
        best_amp, best_ll = 0.0, self._gp_log_marginal(residuals, base_cov)
        for amplitude in scale * _numpy.geomspace(1e-3, 30.0, n_scan):
            log_likelihood = self._gp_log_marginal(residuals, base_cov + amplitude * unit_kernel)
            if log_likelihood > best_ll:
                best_amp, best_ll = float(amplitude), log_likelihood
        return best_amp

    def _shifted_fit_beta(self, x_data, means, sdevs, times: list[str]):
        """Estimate the continuum error by refitting data shifted by +/-1 sigma."""
        intercepts = []
        for sign in (1.0, -1.0):
            shifted = _gvar.gvar(means + sign * sdevs, sdevs)
            fit = self.continuum.model.fit(
                self._run_fit,
                self._run_fit_with_x_errors,
                FitInput(x=x_data, y=shifted, labels=times),
            )
            intercepts.append(_gvar.mean(self.continuum.model.evaluate(0.0, fit.p)))
        return 0.5 * abs(intercepts[0] - intercepts[1])

    def cnt_xtrp(
        self,
        mnt: float,
        mxt: float,
        mng2: float,
        mxg2: float,
        dg2: float = 0.1,
        fcn: Callable | None = None,
        ntrp_fcn: Callable | None = None,
        prior: dict[str, object] | None = None,
        p0: dict[str, object] | None = None,
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
        """Take the continuum limit at fixed renormalized coupling g^2.

        error_mode='fit' propagates errors through the weighted fit;
        error_mode='shifted' performs an uncorrelated fit of the central
        values and quotes half the difference between fits of the data shifted
        up and down by one standard deviation as the error;
        error_mode='gp' augments the fit with a Gaussian-process discrepancy
        term k(x,x') = amp * x x' rho(|log t - log t'|) whose amplitude is set
        by maximizing the marginal likelihood of the fit residuals.  The x x'
        prefactor forces the discrepancy to vanish at a^2/t = 0, so it can
        absorb correlated fluctuations and Symanzik-truncation error without
        contaminating the continuum intercept.

        cov_mode selects the weight covariance: 'empirical', 'diagonal',
        'shrink' (blend controlled by alpha), or 'kernel' (smooth stationary
        correlation model in log t + nugget, fit to the empirical correlation
        matrix — built for action-continuous data whose empirical covariance
        is near-singular).  The legacy diagonal/shrink flags map onto
        cov_mode when it is not given.

        tau0 applies the t-shift improvement (Cheng-Hasenfratz-Liu-
        Petropoulos-Schaich, arXiv:1404.0984): data measured at lattice flow
        time t_m are assigned nominal flow time t = t_m - tau0, so
        g^2(t) -> g^2(t + tau0 a^2) with the exact Jacobian t/t_m applied to
        the beta-function values.  Use tune_tshift to pick tau0.

        With correlated=True (default) the fitted parameters are rebuilt as an
        explicit linear map of the original data gvars, so the continuum betas
        remain correlated with the interpolation stage and *with each other
        across the g^2 grid* — as required when the resulting curve is
        integrated.  correlated=False reproduces the legacy behavior of
        refitting recreated gvars (correlations severed).
        """
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
            "window": (mnt, mxt),
            "g2_range": (mng2, mxg2, dg2),
            "cov_mode": cov_mode,
            "kernel": kernel,
            "alpha": alpha,
            "error_mode": error_mode,
            "tau0": tau0,
            "correlated": correlated,
            "gp_amplitudes": {},
        }
        flow_observable_pairs = [(flow, obs) for flow in self.ntrp_fits for obs in self.ntrp_fits[flow]]
        for flow, obs in flow_observable_pairs:
            self.g2s.setdefault(flow, {}).setdefault(obs, [])
            self.betas.setdefault(flow, {}).setdefault(obs, [])
            self.cnt_fits.setdefault(flow, {}).setdefault(obs, [])
            self.continuum.quality.setdefault(flow, {}).setdefault(obs, [])

            if v >= 1:
                print("Working on flow,discr. = " + ",".join([flow, obs]))

            iterator = g2_values if v == 0 else _tqdm(g2_values)
            for g2 in iterator:
                times = [
                    flow_time
                    for flow_time in self.ntrp_fits[flow][obs]
                    if float(flow_time) - tau0 > 0.0
                    and mnt <= float(flow_time) - tau0 <= mxt
                    and self.ntrp_nf[flow][obs][flow_time][0] <= g2 <= self.ntrp_nf[flow][obs][flow_time][-1]
                ]
                if len(times) < 2:
                    continue
                times.sort(key=float)

                measured_times = _numpy.array([float(flow_time) for flow_time in times])
                nominal_times = measured_times - tau0
                x_data = 1.0 / nominal_times
                jacobian = nominal_times / measured_times
                beta_values = _numpy.array(
                    [factor * ntrp_eval(g2, self.ntrp_fits[flow][obs][flow_time]) for factor, flow_time in zip(jacobian, times)]
                )
                log_times = _numpy.log(nominal_times)
                distances = [_numpy.subtract.outer(log_times, log_times)]
                means = _gvar.mean(beta_values)

                if error_mode == "shifted":
                    sdevs = _gvar.sdev(beta_values)
                    y_data = _gvar.gvar(means, sdevs)
                    fit = self.continuum.model.fit(
                        self._run_fit,
                        self._run_fit_with_x_errors,
                        FitInput(x=x_data, y=y_data, labels=times),
                    )
                    half_difference = self._shifted_fit_beta(x_data, means, sdevs, times)
                    central = _gvar.mean(self.continuum.model.evaluate(0.0, fit.p))
                    beta_value = _gvar.gvar(central, half_difference)
                    self._record_continuum_point(
                        flow, obs, g2, fit, times, y_data, beta_value=beta_value, x_data=x_data
                    )
                    continue

                weight_cov = self._weight_covariance(
                    beta_values, cov_mode=cov_mode, alpha=alpha, kernel=kernel, distances=distances
                )

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
                    self._record_continuum_point(
                        flow, obs, g2, fit, times, beta_values, params=params, x_data=x_data
                    )
                else:
                    y_data = _gvar.gvar(means, weight_cov)
                    fit = self.continuum.model.fit(
                        self._run_fit,
                        self._run_fit_with_x_errors,
                        FitInput(x=x_data, y=y_data, labels=times),
                    )
                    self._record_continuum_point(flow, obs, g2, fit, times, y_data, x_data=x_data)

                if v >= 2:
                    print(self.betas[flow][obs][-1])

    def tune_tshift(
        self,
        taus,
        mnt: float,
        mxt: float,
        mng2: float,
        mxg2: float,
        dg2: float = 0.2,
        v: int = 1,
        **cnt_kwargs,
    ) -> tuple[float, dict[float, float]]:
        """Scan the t-shift parameter tau0 and pick the flattest extrapolation.

        For each candidate tau0 a fast uncorrelated continuum fit is run over
        the g^2 grid and the mean squared slope pull <(slope/sigma_slope)^2>
        is recorded across all flows, observables, and couplings; the
        nonperturbatively tuned tau0 is the one that minimizes it, i.e. the
        shift that best removes the leading a^2/t dependence (arXiv:1404.0984).
        Requires the default linear continuum model (parameter 'slope').
        Leaves the continuum stage as it was before the scan.  Returns
        (best_tau0, {tau0: metric}).
        """
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

    def perturbative_interpolation(
        self,
        loops: int = 2,
        correction_order: int = 2,
        free_intercept: bool = False,
        intercept_width: float = 0.2,
        width: float = 5.0,
        xerrors: bool = False,
    ) -> InterpolationSpec:
        """PT-constrained interpolation model in g^2 for iv_ntrp.

        Parametrizes

            beta(g^2) = beta_PT^(loops)(g^2) * (c0 + sum_n c_n u^n)

        with u = g^2/(4 pi).  When free_intercept=False (default) c0 is fixed
        to 1, recovering the standard form beta_PT * (1 + sum_n c_n u^n) where
        the universal perturbative coefficients are reproduced exactly at weak
        coupling.

        When free_intercept=True the leading coefficient c0 carries a prior
        gvar(1.0, intercept_width) and is allowed to drift.  This absorbs an
        O(intercept_width) constant fractional offset between beta_data and
        beta_PT — common at finite a^2/t as a tree-level lattice artefact —
        without breaking asymptotic freedom (beta -> c0 * beta_PT -> 0 as
        g^2 -> 0).  A typical choice is intercept_width ~ 0.2.

        NOTE — loops=3 GF scheme: b2_GF ~ -1.82 (code units) causes
        beta_PT(3L) to change sign near g^2 ~ 9.  Within the typical
        interpolation range g^2 in [0.5, 5] this is not a problem, but the
        3-loop term can already be 30-40% of beta_2L at g^2 ~ 4.  Use
        loops=2 (default) unless the dataset clearly favours the 3-loop base.
        """
        if loops not in (1, 2, 3):
            raise BetaFunctionException("loops must be 1, 2, or 3")
        if correction_order < 1:
            raise BetaFunctionException("correction_order must be at least 1")
        perturbative = self.perturbative_beta_function
        names = [f"pt_c{order}" for order in range(1, correction_order + 1)]

        def fcn(x, p):
            u = x / perturbative.nrm
            correction = p['pt_c0'][0] if free_intercept else 1.0
            term = 1.0
            for name in names:
                term = term * u
                correction = correction + p[name][0] * term
            return perturbative(x, loops=loops) * correction

        prior = {name: [_gvar.gvar(0.0, width)] for name in names}
        p0 = {name: 0.0 for name in names}
        if free_intercept:
            prior['pt_c0'] = [_gvar.gvar(1.0, intercept_width)]
            p0['pt_c0'] = 1.0
        return InterpolationSpec(fcn=fcn, prior=prior, p0=p0, xerrors=xerrors)

    def _default_artifact_model(self) -> tuple[Callable, dict, dict]:
        """Leading coupling-dependent cutoff-effect model for the global fit."""
        nrm = self.perturbative_beta_function.nrm

        def artifact(g2, x, p):
            u = g2 / nrm
            return (p["s0"][0] + p["s1"][0] * u) * u * u * x

        prior = {"s0": [_gvar.gvar(0.0, 10.0)], "s1": [_gvar.gvar(0.0, 10.0)]}
        p0 = {"s0": 0.0, "s1": 0.0}
        return artifact, prior, p0

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
        artifact_fcn: Callable | None = None,
        artifact_prior: dict[str, object] | None = None,
        artifact_p0: dict[str, object] | None = None,
        ntrp_fcn: Callable | None = None,
        cov_mode: str = "kernel",
        kernel: str = "rbf",
        alpha: float = 0.25,
        gp: bool = False,
        gp_lengths: tuple[float, float] | None = None,
        tau0: float = 0.0,
        g2_round_precis: int | None = None,
        v: int = 1,
    ):
        """One joint continuum fit over the whole (g^2, a^2/t) plane.

        Replaces the per-g^2 extrapolations with a single 2D fit per
        flow/observable:

            beta(g^2, x) = beta_0(g^2) + s(g^2) * x,   x = a^2/t,

        where the artifact slope s(g^2) is shared across the entire g^2 grid
        (default: one-loop-suppressed u^2 (s0 + s1 u), see
        _default_artifact_model) instead of being refit incoherently at each
        coupling.  intercepts='free' keeps one unconstrained beta_0 per grid
        point; intercepts='perturbative' parametrizes the continuum curve as
        beta_PT^(loops) * (1 + sum c_n u^n) with fixed universal coefficients
        and naturalness priors on c_n, injecting perturbation theory directly
        into the continuum stage.
        """
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
        correction_names = [f"pt_c{order}" for order in range(1, correction_order + 1)]

        self.mnt, self.mxt = mnt, mxt
        self.continuum.metadata = {
            "method": "global",
            "window": (mnt, mxt),
            "g2_range": (mng2, mxg2, dg2),
            "intercepts": intercepts,
            "loops": loops,
            "cov_mode": cov_mode,
            "kernel": kernel,
            "alpha": alpha,
            "gp": gp,
            "tau0": tau0,
            "gp_amplitudes": {},
        }

        flow_observable_pairs = [(flow, obs) for flow in self.ntrp_fits for obs in self.ntrp_fits[flow]]
        for flow, obs in flow_observable_pairs:
            if v >= 1:
                print("Global continuum fit for flow,discr. = " + ",".join([flow, obs]))

            kept_g2, y_values, g2_index = [], [], []
            g2_coord, x_coord, logt_coord, labels = [], [], [], []
            for g2 in g2_values:
                times = [
                    flow_time
                    for flow_time in self.ntrp_fits[flow][obs]
                    if float(flow_time) - tau0 > 0.0
                    and mnt <= float(flow_time) - tau0 <= mxt
                    and self.ntrp_nf[flow][obs][flow_time][0] <= g2 <= self.ntrp_nf[flow][obs][flow_time][-1]
                ]
                if len(times) < 2:
                    continue
                times.sort(key=float)
                index = len(kept_g2)
                kept_g2.append(float(g2))
                for flow_time in times:
                    measured = float(flow_time)
                    nominal = measured - tau0
                    y_values.append((nominal / measured) * ntrp_eval(g2, self.ntrp_fits[flow][obs][flow_time]))
                    g2_index.append(index)
                    g2_coord.append(float(g2))
                    x_coord.append(1.0 / nominal)
                    logt_coord.append(_numpy.log(nominal))
                    labels.append(f"{g2}/{flow_time}")

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
                guesses = _numpy.array([_numpy.mean(means[g2_index == index]) for index in range(len(kept_g2))])
                scale = 100.0 * max(float(_numpy.max(_numpy.abs(guesses))), 1.0)
                prior["beta"] = [_gvar.gvar(float(guess), scale) for guess in guesses]
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
            weight_cov = self._weight_covariance(
                y_values, cov_mode=cov_mode, alpha=alpha, kernel=kernel, distances=distances
            )

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
                "inputs",
                (flow, obs, "global"),
                FitInput(
                    x=_numpy.column_stack([g2_coord, x_coord]),
                    y=y_values,
                    labels=labels,
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

    def processed_series(self, coupling: str, volume: str, mass: str, flow: str, observable: str, kind: str, mnt=None, mxt=None):
        """Extract processed (flow-time, observable) series for plotting or diagnostics."""
        mnt = self._min_fv_flt if mnt is None else mnt
        mxt = self._max_fv_flt if mxt is None else mxt
        flow_times = [
            flow_time
            for flow_time in self.avg_data[coupling][volume][mass][flow]["flow_times"]
            if mnt <= float(flow_time) <= mxt
        ]
        key = f"{kind}_{observable}"
        values = [self.avg_data[coupling][volume][mass][flow][key][flow_time] for flow_time in flow_times]
        return _numpy.array([float(flow_time) for flow_time in flow_times]), values

    def chiral_curve(self, coupling: str, volume: str, flow: str, obs: str, flow_time: str, x: str = "g2", points: int = 100):
        """Return curve points and source data for a stored chiral fit."""
        data = self.chiral.fetch("inputs", (coupling, volume, x, flow, obs, flow_time))
        fit_params = self.chiral.fetch("fits", (coupling, volume, x, flow, obs, flow_time))
        x_grid = _numpy.linspace(min(data.x), max(data.x), points)
        y_grid = [self.chiral.model.evaluate(x_value, fit_params) for x_value in x_grid]
        return x_grid, y_grid, data

    def infinite_volume_curve(self, coupling: str, flow: str, obs: str, flow_time: str, x: str = "g2", points: int = 100):
        """Return curve points and source data for a stored infinite-volume fit."""
        data = self.infinite_volume.fetch("inputs", (coupling, x, flow, obs, flow_time))
        fit_params = self.infinite_volume.fetch("fits", (coupling, x, flow, obs, flow_time))
        x_grid = _numpy.linspace(min(data.x), max(data.x), points)
        y_grid = [self.infinite_volume.model.evaluate(x_value, fit_params) for x_value in x_grid]
        return x_grid, y_grid, data

    def interpolation_curve(
        self,
        flow: str,
        obs: str,
        flow_time: str,
        points: int = 200,
        g2_min: float | None = None,
        g2_max: float | None = None,
    ):
        """Return curve points and source data for a stored interpolation fit.

        By default the curve spans the data range ``[x_min_data, x_max_data]``
        determined by ``data.bounds()``.  Pass *g2_min* and/or *g2_max* to
        override either endpoint — useful for extrapolating into the
        weak-coupling regime below the smallest measured :math:`g^2` or
        extending beyond the largest.

        At :math:`g^2 \\to 0` the PT-constrained model is anchored by
        perturbation theory, so the error band grows only mildly; the
        extrapolation is meaningful as long as :math:`g^2` stays within the
        radius of convergence of the polynomial correction series (typically
        safe for :math:`g^2 \\gtrsim 0.05`).

        Parameters
        ----------
        flow, obs, flow_time :
            Keys identifying the stored fit.
        points :
            Number of equally spaced grid points.
        g2_min :
            Lower :math:`g^2` bound of the returned grid.  Defaults to the
            minimum :math:`g^2` in the fit data.
        g2_max :
            Upper :math:`g^2` bound.  Defaults to the maximum in the fit data.

        Returns
        -------
        x_grid : numpy.ndarray
            :math:`g^2` values (length *points*).
        y_grid : list[gvar]
            Model evaluations (beta function values, as gvars).
        data :
            The stored fit input object (contains the actual data points).
        """
        data = self.interpolation.fetch("inputs", (flow, obs, flow_time))
        fit_params = self.interpolation.fetch("fits", (flow, obs, flow_time))
        x_min_data, x_max_data = data.bounds()
        x_min = x_min_data if g2_min is None else float(g2_min)
        x_max = x_max_data if g2_max is None else float(g2_max)
        x_grid = _numpy.linspace(x_min, x_max, points)
        y_grid = [self.interpolation.model.evaluate(x_value, fit_params) for x_value in x_grid]
        return x_grid, y_grid, data

    def continuum_curve(self, flow: str, obs: str):
        """Return final continuum beta-function points for one flow/observable pair."""
        return _numpy.array(self.g2s[flow][obs]), self.betas[flow][obs]

    def continuum_window_scan(
        self,
        windows: list[tuple[float, float]],
        mng2: float,
        mxg2: float,
        dg2: float = 0.1,
        v: int = 0,
        **cnt_kwargs,
    ) -> dict[tuple[float, float], StageStore]:
        """Rerun the continuum limit over several flow-time windows for systematics.

        Each window produces an independent continuum StageStore; the scan
        results are kept on the instance for `continuum_systematics`.  After
        the scan the instance is left holding the *first* window's result, so
        pass the central/preferred window first.
        """
        if len(windows) < 2:
            raise BetaFunctionException("continuum_window_scan needs at least two windows")

        scan: dict[tuple[float, float], StageStore] = {}
        for mnt, mxt in windows:
            self.cnt_xtrp(mnt=mnt, mxt=mxt, mng2=mng2, mxg2=mxg2, dg2=dg2, v=v, **cnt_kwargs)
            scan[(mnt, mxt)] = self.continuum

        self._window_scan = scan

        # Leave the instance holding the central (first) window's result.
        central = windows[0]
        self.continuum = scan[central]
        self._assign_stage_aliases()
        return scan

    def continuum_systematics(self, flow: str, obs: str) -> list[dict[str, float]]:
        """Combine a window scan into a per-g^2 error budget.

        For each g^2 present in the central window, returns the central value,
        its statistical error, the half-spread across scan windows as the
        window systematic, and the quadrature-combined total.
        """
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
            budget.append(
                {
                    "g2": float(g2),
                    "beta": float(_gvar.mean(central)),
                    "stat": float(statistical),
                    "syst": float(systematic),
                    "total": float(_numpy.hypot(statistical, systematic)),
                }
            )
        return budget

    def run_processing(self, config: AnalysisConfig) -> None:
        """Stage 1: discover the configured dataset slice and process raw data."""
        self.set_binsize(config.binsize)
        self.os = list(config.observables)

        catalog = self.catalog(config.data_path, flows=list(config.flows))
        if config.couplings is not None:
            catalog = catalog.filter(couplings=list(config.couplings))
        if config.volumes is not None:
            allowed = config.volumes
            catalog = catalog.filter(
                predicate=lambda entry: entry.key.coupling in allowed
                and entry.key.volume in allowed[entry.key.coupling]
            )
        dataset = catalog.to_mapping()
        if not dataset:
            raise BetaFunctionException(f"No ensembles discovered in {config.data_path}")

        self.process_data(
            dataset,
            path=str(config.data_path),
            correction=config.correction,
            tree_level_normalization_data_path=str(config.data_path) if config.tln_path is None else config.tln_path,
            combine=config.combine,
            mnt=config.process_window[0],
            mxt=config.process_window[1],
            verbosity=config.verbosity,
            use_gamma_method=config.use_gamma_method,
            gamma_window_factor=config.gamma_window_factor,
        )

    def run_chiral(self, config: AnalysisConfig) -> None:
        """Stage 2: chiral extrapolation over the configured fit window."""
        fit_window = config.process_window if config.fit_window is None else config.fit_window
        self.ch_xtrp(mnt=fit_window[0], mxt=fit_window[1], v=config.verbosity)

    def run_infinite_volume(self, config: AnalysisConfig) -> None:
        """Stage 3: infinite-volume extrapolation over the configured fit window."""
        fit_window = config.process_window if config.fit_window is None else config.fit_window
        self.iv_xtrp(mnt=fit_window[0], mxt=fit_window[1], v=config.verbosity)

    def run_interpolation(self, config: AnalysisConfig) -> None:
        """Stage 4: interpolation in g^2 with the configured model."""
        self.iv_ntrp(
            fcn=config.interpolation.fcn,
            prior=config.interpolation.prior,
            p0=config.interpolation.p0,
            xerrors=config.interpolation.xerrors,
            v=config.verbosity,
        )

    def run_continuum(self, config: AnalysisConfig) -> None:
        """Stage 5: continuum limit, with a window scan when config.scan_windows is set."""
        mng2, mxg2, dg2 = config.g2_grid
        if config.scan_windows:
            windows = [config.continuum_window, *config.scan_windows]
            self.continuum_window_scan(
                windows,
                mng2=mng2,
                mxg2=mxg2,
                dg2=dg2,
                v=config.verbosity,
                diagonal=config.diagonal,
                shrink=config.shrink,
                alpha=config.alpha,
                error_mode=config.error_mode,
                tau0=config.tau0,
                cov_mode=config.cov_mode,
                correlated=config.correlated,
            )
        else:
            self.cnt_xtrp(
                mnt=config.continuum_window[0],
                mxt=config.continuum_window[1],
                mng2=mng2,
                mxg2=mxg2,
                dg2=dg2,
                v=config.verbosity,
                diagonal=config.diagonal,
                shrink=config.shrink,
                alpha=config.alpha,
                error_mode=config.error_mode,
                tau0=config.tau0,
                cov_mode=config.cov_mode,
                correlated=config.correlated,
            )

    def collect_result(self, config: AnalysisConfig, wall_time: float = 0.0) -> AnalysisResult:
        """Bundle current stage state into an AnalysisResult with provenance."""
        continuum: dict[tuple[str, str], tuple[_numpy.ndarray, list]] = {}
        systematics: dict[tuple[str, str], list[dict[str, float]]] = {}
        for flow in config.flows:
            for obs in config.observables:
                if flow in self.g2s and obs in self.g2s[flow]:
                    continuum[(flow, obs)] = self.continuum_curve(flow, obs)
                    if config.scan_windows:
                        systematics[(flow, obs)] = self.continuum_systematics(flow, obs)

        provenance = self.provenance()
        provenance["config"] = config.describe()

        return AnalysisResult(
            config=config,
            provenance=provenance,
            summaries=self.analysis_summary(),
            continuum=continuum,
            systematics=systematics,
            wall_time=wall_time,
        )

    def run_analysis(self, config: AnalysisConfig) -> AnalysisResult:
        """Execute the full pipeline from raw data to continuum limit.

        One declarative `AnalysisConfig` in, one `AnalysisResult` out — with
        provenance, per-stage diagnostics, continuum curves, and (if
        `config.scan_windows` is set) a window-scan systematic error budget.

        Each stage is also available individually: `run_processing`,
        `run_chiral`, `run_infinite_volume`, `run_interpolation`,
        `run_continuum`, then `collect_result`.
        """
        start_time = _time.time()

        self.run_processing(config)
        self.run_chiral(config)
        self.run_infinite_volume(config)
        self.run_interpolation(config)
        self.run_continuum(config)

        return self.collect_result(config, wall_time=round(_time.time() - start_time, 2))

    def save_analysis(self, file_name: str, **kwargs) -> None:
        """Persist processed data and all stage fits/diagnostics to one file."""
        payload = {
            "avg_data": self.avg_data,
            "provenance": self.provenance(),
            "stages": {
                stage.name: {
                    "fits": stage.fits,
                    "quality": stage.quality,
                    "domains": stage.domains,
                    "metadata": stage.metadata,
                }
                for stage in (self.chiral, self.infinite_volume, self.interpolation, self.continuum)
            },
            "continuum_inputs": {"g2s": self.g2s, "betas": self.betas},
        }
        _gvar.dump(payload, file_name, **kwargs)

    def load_analysis(self, file_name: str, **kwargs) -> dict[str, object]:
        """Restore processed data and stage fits/diagnostics saved by save_analysis."""
        payload = _gvar.load(file_name, **kwargs)
        self.avg_data = payload["avg_data"]
        for stage_name, contents in payload["stages"].items():
            store = StageStore(name=stage_name)
            store.fits = contents["fits"]
            store.quality = contents["quality"]
            store.domains = contents["domains"]
            store.metadata = contents["metadata"]
            setattr(self, stage_name, store)
        self.continuum.inputs["g2s"] = payload["continuum_inputs"]["g2s"]
        self.continuum.inputs["betas"] = payload["continuum_inputs"]["betas"]
        self._assign_stage_aliases()
