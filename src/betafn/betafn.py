"""BetaFunction: the main staged analysis engine."""
from __future__ import annotations

from collections.abc import Callable
import inspect as _inspect

import gvar as _gvar
import lsqfit as _lsqfit
import numpy as _numpy

from .base.exceptions import BetaFunctionException
from .fitting.containers import FitInput, FitModel, StageStore
from .setup import SetupBetaFunction
from .stages.chiral import ChiralMixin
from .stages.iv import InfiniteVolumeMixin
from .stages.interpolation import InterpolationMixin
from .stages.continuum import ContinuumMixin
from .stages.pipeline import PipelineMixin


class BetaFunction(PipelineMixin, ContinuumMixin, InterpolationMixin, InfiniteVolumeMixin, ChiralMixin, SetupBetaFunction):
    """Method-centric continuous beta-function analysis built around staged fits.

    The analysis proceeds through five stages, each driven by a dedicated mixin:
      1. Processing    — SetupBetaFunction.process_data
      2. Chiral        — ChiralMixin.ch_xtrp
      3. Infinite vol. — InfiniteVolumeMixin.iv_xtrp
      4. Interpolation — InterpolationMixin.iv_ntrp
      5. Continuum     — ContinuumMixin.cnt_xtrp / cnt_xtrp_global

    The full pipeline can be driven declaratively via PipelineMixin.run_analysis.
    """

    def __init__(
        self,
        nc: float | int = 3.0,
        nf: float | int | None = None,
        gauge_action: str = "s",
        logfn: str | None = None,
    ):
        SetupBetaFunction.__init__(self, nc=nc, nf=nf, gauge_action=gauge_action, logfn=logfn)
        self._thin = None
        self._window_scan: dict = {}
        self._reset_analysis_state()

    # ------------------------------------------------------------------
    # Stage state initialisation
    # ------------------------------------------------------------------

    def _reset_analysis_state(self) -> None:
        self.chiral = StageStore(name="chiral")
        self.infinite_volume = StageStore(name="infinite_volume")
        self.interpolation = StageStore(name="interpolation")
        self.continuum = StageStore(name="continuum")
        self._assign_stage_aliases()

    def _assign_stage_aliases(self) -> None:
        """Keep the legacy short-name aliases in sync with the StageStore objects."""
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

    def _clear_gvar_state(self) -> None:
        """Drop every gvar-holding container (overrides base class)."""
        SetupBetaFunction._clear_gvar_state(self)
        self._reset_analysis_state()
        self._window_scan = {}
        self._thin = None

    # ------------------------------------------------------------------
    # Shared private helpers used by multiple stage mixins
    # ------------------------------------------------------------------

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
        """Wrap a 1- or 2-argument function into the standard (x, p) → y form."""
        signature = _inspect.signature(fcn)
        positional = [
            param for param in signature.parameters.values()
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

    def _run_fit(self, x, y, fit_fcn: Callable, prior, p0):
        return _lsqfit.nonlinear_fit(data=(x, y), fcn=fit_fcn, prior=prior, p0=p0)

    def _run_fit_with_x_errors(self, y, fit_fcn: Callable, prior, p0):
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

    def _rehome_prior(self, prior: dict | None) -> dict | None:
        """Recreate prior gvars inside the active covariance environment."""
        if prior is None:
            return None
        return {
            key: [
                _gvar.gvar(v.mean, v.sdev) if isinstance(v, _gvar.GVar) else v
                for v in values
            ]
            for key, values in prior.items()
        }

    def _configured_model(self, fcn: Callable, prior, p0, xerrors: bool = False) -> FitModel:
        return FitModel(
            fcn=self._wrap_model(fcn),
            prior=self._rehome_prior(self._normalize_param_dict(prior)),
            p0=self._normalize_param_dict(p0),
            xerrors=xerrors,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers that supply nf/nc from self
    # ------------------------------------------------------------------

    def perturbative_interpolation(
        self,
        loops: int = 2,
        correction_order: int = 2,
        free_intercept: bool = False,
        intercept_width: float = 0.2,
        width: float = 5.0,
        xerrors: bool = False,
    ):
        """PT-constrained interpolation spec using this instance's nf and nc.

        Delegates to ``betafn.fitting.families.perturbative_interpolation``
        with ``nf=self.nf`` and ``nc=self.nc`` pre-filled.  All other
        parameters are forwarded unchanged; see that function for full docs.
        """
        from .fitting.families import perturbative_interpolation as _pt
        return _pt(
            nf=self.nf, nc=self.nc,
            loops=loops, correction_order=correction_order,
            free_intercept=free_intercept, intercept_width=intercept_width,
            width=width, xerrors=xerrors,
        )

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def processed_series(self, coupling, volume, mass, flow, observable, kind, mnt=None, mxt=None):
        """Extract (flow_times, values) for plotting or diagnostics."""
        mnt = self._min_fv_flt if mnt is None else mnt
        mxt = self._max_fv_flt if mxt is None else mxt
        flow_times = [
            ft for ft in self.avg_data[coupling][volume][mass][flow]["flow_times"]
            if mnt <= float(ft) <= mxt
        ]
        key = f"{kind}_{observable}"
        values = [self.avg_data[coupling][volume][mass][flow][key][ft] for ft in flow_times]
        return _numpy.array([float(ft) for ft in flow_times]), values

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

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
