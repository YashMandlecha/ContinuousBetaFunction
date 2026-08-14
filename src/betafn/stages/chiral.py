"""ChiralMixin — chiral extrapolation stage of the BetaFunction analysis."""
from __future__ import annotations

from tqdm import tqdm as _tqdm

from ..base.exceptions import BetaFunctionException
from ..fitting.containers import FitInput


class ChiralMixin:
    """Chiral extrapolation to m=0 at fixed bare coupling, volume, and flow time."""

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

    def get_ch_data(
        self,
        coupling: str,
        volume: str,
        flow: str,
        x: str,
        obs: str,
        t: str,
        masses: list[str],
    ) -> FitInput:
        """Build chiral fit input for a specific (beta, volume, flow, obs, t)."""
        xo = "_".join([x, obs])
        valid_masses = [
            mass for mass in masses
            if mass != "0p00" and mass not in self._ch_exclude[coupling][volume]
        ]
        return FitInput(
            x=[self._mass_value(mass) for mass in valid_masses],
            y=[self.avg_data[coupling][volume][mass][flow][xo][t] for mass in valid_masses],
            labels=valid_masses,
        )

    def ch_xtrp(
        self,
        mnt: float | None = None,
        mxt: float | None = None,
        exclude: dict[str, list[str]] | None = None,
        fcn=None,
        prior: dict | None = None,
        p0: dict | None = None,
        xtrp: list[str] = ["g2", "beta"],
        v: int = 1,
        thin: int = 1,
        postprocess: bool = True,
    ):
        """Extrapolate to zero quark mass at fixed bare coupling, volume, and flow time."""
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
                        flow: {"_".join([x_key, obs]): {} for x_key in xtrp for obs in self.os}
                        for flow in flows
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
                            self.avg_data[coupling][volume]["0p00"][flow][xo][flow_time] = (
                                self.chiral.model.evaluate(0.0, fit.p)
                            )
                        if postprocess and obs == self.os[0] and x == xtrp[0]:
                            self.avg_data[coupling][volume]["0p00"][flow]["flow_times"].append(flow_time)
                    except KeyError as err:
                        self.log.write("ERROR: " + " ".join([coupling, x, flow, obs, flow_time]), repr(err))

                if v >= 1:
                    msg = "Finished beta_b = " + coupling.replace("p", ".")
                    msg += ", L^{Nd-1}xT = " + volume[1:].replace("l", "x").replace("t", "x")
                    msg += " in " + str(self._stop_timer()) + " secs"
                    print(msg)

    def chiral_curve(self, coupling: str, volume: str, flow: str, obs: str, flow_time: str, x: str = "g2", points: int = 100):
        """Return grid, model values, and source data for a stored chiral fit."""
        data = self.chiral.fetch("inputs", (coupling, volume, x, flow, obs, flow_time))
        fit_params = self.chiral.fetch("fits", (coupling, volume, x, flow, obs, flow_time))
        import numpy as _np
        x_grid = _np.linspace(min(data.x), max(data.x), points)
        y_grid = [self.chiral.model.evaluate(xv, fit_params) for xv in x_grid]
        return x_grid, y_grid, data
