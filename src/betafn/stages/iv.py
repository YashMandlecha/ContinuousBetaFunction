"""InfiniteVolumeMixin — infinite-volume extrapolation stage."""
from __future__ import annotations

import gc as _gc

import gvar as _gvar
import numpy as _numpy
from tqdm import tqdm as _tqdm

from ..base.exceptions import BetaFunctionException
from ..fitting.containers import FitInput


class InfiniteVolumeMixin:
    """Infinite-volume extrapolation at fixed bare coupling and flow time."""

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

    def get_fv_data(
        self,
        coupling: str,
        flow: str,
        x: str,
        obs: str,
        t: str,
        volumes: list[str],
    ) -> FitInput:
        """Build infinite-volume fit input for a specific (beta, flow, obs, t)."""
        xo = "_".join([x, obs])
        mass = "0p00"
        valid_volumes = [volume for volume in volumes if volume not in self._iv_exclude[coupling]]
        return FitInput(
            x=[1.0 / self._volume_value(volume) for volume in valid_volumes],
            y=[self.avg_data[coupling][volume][mass][flow][xo][t] for volume in valid_volumes],
            labels=valid_volumes,
        )

    def _fit_inputs_from_iv(self, fake_iv_data: bool) -> dict:
        iv_data: dict = {}
        if not fake_iv_data:
            if not self.iv_fits:
                raise BetaFunctionException("Must run iv_xtrp (or load iv fits) before iv_ntrp")
            for (_, x, flow, obs, flow_time), fit_params in self._flatten(self.iv_fits, stop=5).items():
                iv_data.setdefault(flow, {}).setdefault(obs, {}).setdefault(flow_time, {}).setdefault(x, [])
                iv_data[flow][obs][flow_time][x].append(
                    self.infinite_volume.model.evaluate(0.0, fit_params)
                )
            return iv_data

        self._require_avg_data("iv_ntrp with fake_iv_data")
        self._iv_exclude = {coupling: [] for coupling in self.avg_data}
        for coupling in self.data:
            volumes, _, targets = self._gather_iv_info(
                coupling, ["g2", "beta"], self._min_fv_flt, self._max_fv_flt
            )
            for x, flow, obs, flow_time in targets:
                iv_data.setdefault(flow, {}).setdefault(obs, {}).setdefault(flow_time, {}).setdefault(x, [])
                iv_data[flow][obs][flow_time][x].append(
                    self.get_fv_data(coupling, flow, x, obs, flow_time, volumes).y[-1]
                )
        return iv_data

    def iv_xtrp(
        self,
        mnt: float | None = None,
        mxt: float | None = None,
        exclude: dict[str, list[str]] | None = None,
        fcn=None,
        prior: dict | None = None,
        p0: dict | None = None,
        xtrp: list[str] = ["g2", "beta"],
        model_average: bool = False,
        v: int = 1,
        thin: int = 1,
    ):
        """Extrapolate to infinite volume at fixed bare coupling and flow time."""
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

        self._iv_exclude = (
            {coupling: [] for coupling in self.avg_data} if exclude is None else exclude
        )

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
                msg = "Finished beta_b = " + coupling.replace("p", ".")
                msg += " in " + str(self._stop_timer()) + " secs"
                print(msg)

        _gc.collect()

    def infinite_volume_curve(
        self,
        coupling: str,
        flow: str,
        obs: str,
        flow_time: str,
        x: str = "g2",
        points: int = 100,
    ):
        """Return grid, model values, and source data for a stored IV fit."""
        data = self.infinite_volume.fetch("inputs", (coupling, x, flow, obs, flow_time))
        fit_params = self.infinite_volume.fetch("fits", (coupling, x, flow, obs, flow_time))
        import numpy as _np
        x_grid = _np.linspace(min(data.x), max(data.x), points)
        y_grid = [self.infinite_volume.model.evaluate(xv, fit_params) for xv in x_grid]
        return x_grid, y_grid, data
