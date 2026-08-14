"""InterpolationMixin — g^2 interpolation stage."""
from __future__ import annotations

import gvar as _gvar
import numpy as _numpy
from tqdm import tqdm as _tqdm

from ..base.exceptions import BetaFunctionException
from ..fitting.containers import FitInput


class InterpolationMixin:
    """Interpolate the infinite-volume beta function in g^2."""

    def iv_ntrp(
        self,
        fcn=None,
        prior: dict | None = None,
        p0: dict | None = None,
        v: int = 1,
        xerrors: bool = False,
        fake_iv_data: bool = False,
        emp_bayes_fcn=None,
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

    def interpolation_curve(
        self,
        flow: str,
        obs: str,
        flow_time: str,
        points: int = 200,
        g2_min: float | None = None,
        g2_max: float | None = None,
    ):
        """Return grid, model values, and source data for a stored interpolation fit.

        By default the curve spans the data range determined by the fit input.
        Pass *g2_min* / *g2_max* to override either endpoint — useful for
        extrapolating into the weak-coupling regime below the smallest measured
        g^2, or extending beyond the largest.
        """
        data = self.interpolation.fetch("inputs", (flow, obs, flow_time))
        fit_params = self.interpolation.fetch("fits", (flow, obs, flow_time))
        x_min_data, x_max_data = data.bounds()
        x_min = x_min_data if g2_min is None else float(g2_min)
        x_max = x_max_data if g2_max is None else float(g2_max)
        import numpy as _np
        x_grid = _np.linspace(x_min, x_max, points)
        y_grid = [self.interpolation.model.evaluate(xv, fit_params) for xv in x_grid]
        return x_grid, y_grid, data
