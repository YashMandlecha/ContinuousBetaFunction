"""SetupBetaFunction: data loading, preprocessing, and the processing base class."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import gc as _gc
import pickle as _pickle
from pathlib import Path
import time as _time
import warnings as _warnings

import gvar as _gvar
import numpy as _numpy

from .base.exceptions import BetaFunctionException, EmptyEnsembleError, BetaFunctionLog
from .base.catalog import (
    FlowWindow, ProcessConfig, ProcessHooks,
    EnsembleKey, EnsembleFile, DatasetCatalog,
)
from .perturbative import PerturbativeBetaFunction
from .processing.gamma import (
    gamma_method_covariance,
    gamma_method_average,
    integrated_autocorrelation_time,
)
from .processing.finite_volume import delta_finite_volume
from .processing.tln import delta_tln


class SetupBetaFunction:
    """Base class responsible for loading, filtering, and processing raw flow data."""

    DATA_FILE_PATTERN = DatasetCatalog.FILE_PATTERN

    def __init__(
        self,
        nc: float | int = 3.0,
        nf: float | int | None = None,
        gauge_action: str = "s",
        logfn: str | None = None,
    ):
        if nf is None:
            raise BetaFunctionException("must specify nf")

        self.nf = nf
        self.nc = nc
        self.gauge_action = gauge_action

        self.perturbative_beta_function = PerturbativeBetaFunction(self.nf, self.nc)
        self._coupling_norm = 128.0 * _numpy.pi * _numpy.pi / 3.0 / (self.nc * self.nc - 1.0)

        self.os = ["p", "s", "c"]
        self.mc_observables = ["E" + obs for obs in self.os] + ["Q"]

        self._binsize = 1
        self._window = FlowWindow()
        self._process_config = ProcessConfig()

        self.data: dict = {}
        self.avg_data: dict = {}
        self.combine: dict[str, dict[str, float]] = {}
        self._ensemble_stats: dict[EnsembleKey, dict[str, object]] = {}
        self.skipped_ensembles: dict[EnsembleKey, str] = {}

        self.log = BetaFunctionLog(logfn)
        if logfn is not None:
            _warnings.showwarning = self.log

    def _start_timer(self) -> None:
        self._ti = _time.time()

    def _stop_timer(self) -> float:
        return round(_time.time() - self._ti, 2)

    @property
    def _min_fv_flt(self) -> float:
        return self._window.minimum_t

    @property
    def _max_fv_flt(self) -> float:
        return self._window.maximum_t

    @property
    def _min_Q(self) -> float:
        return self._window.minimum_q

    @property
    def _max_Q(self) -> float:
        return self._window.maximum_q

    @property
    def _correction(self) -> str:
        return self._process_config.correction

    def _dxdlogt(self, values, flow_times):
        values_array = _numpy.array(values)
        flow_time_array = _numpy.array(flow_times)
        delta_values = -values_array[4:] + 8.0 * values_array[3:-1] - 8.0 * values_array[1:-3] + values_array[:-4]
        delta_log_t = 6.0 * (flow_time_array[4:] - flow_time_array[:-4]) / (flow_time_array[4:] + flow_time_array[:-4])
        return -delta_values / delta_log_t

    def _flow_time_sort_key(self, key: str) -> float:
        return float(key)

    def _flow_times(self, data: dict) -> list[str]:
        flow_times = list({observable.split("_")[-1] for observable in data})
        flow_times.sort(key=self._flow_time_sort_key)
        return [str(ft) for ft in flow_times]

    def _volume_dims(self, volume: str) -> list[float]:
        dims = volume.replace("t", "l").split("l")[1:]
        return [float(dim) for dim in dims if dim]

    def _apply_topological_filter(self, raw_data: dict) -> Callable[[list], list]:
        if "Q" not in raw_data or not self._window.uses_topological_filter:
            return lambda values: values
        charges = raw_data["Q"][-1]

        def apply(values):
            return [
                value for index, value in enumerate(values)
                if self._window.minimum_q <= charges[index] <= self._window.maximum_q
            ]
        return apply

    def _ensemble_file_path(self, path: str, coupling: str, volume: str, mass: str, flow: str) -> Path:
        file_name = "_".join([coupling, volume, mass, flow]) + ".bin"
        return Path(path) / file_name if path else Path(file_name)

    def _get(self, coupling: str, volume: str, mass: str, flow: str, path: str):
        file_path = self._ensemble_file_path(path, coupling, volume, mass, flow)
        with open(file_path, "rb") as input_file:
            raw_data = _pickle.load(input_file)
        if "flow_times" not in raw_data or not raw_data["flow_times"]:
            raise EmptyEnsembleError(f"{file_path.name} contains no flow-time grid (empty measurement file)")
        apply_filter = self._apply_topological_filter(raw_data)
        return {
            "_".join([observable, flow_time]): apply_filter(raw_data[observable][index])
            for index, flow_time in enumerate(raw_data["flow_times"])
            for observable in raw_data
            if observable != "flow_times" and self._window.includes_time(flow_time)
        }

    def _rearrange(self, data: dict):
        return {
            "_".join([flow, obs_with_time]): data[flow][obs_with_time]
            for flow in data
            for obs_with_time in data[flow]
        }

    def _preprocess(self, data: dict):
        return _gvar.dataset.bin_data(data, binsize=self._binsize)

    def _average(self, data: dict, process: Callable | None = None):
        payload = data if process is None else process(data)
        return _gvar.dataset.avg_data(payload)

    def _gamma_method_average(self, data: dict, process=None, window_factor: float = 3.0) -> dict:
        """Delegate to the module-level gamma_method_average function."""
        return gamma_method_average(data, process=process, window_factor=window_factor)

    def _undorearrange(self, data: dict, flows: list[str], data_ref: dict):
        return {
            flow: {
                observable: [
                    data["_".join([flow, observable, ft])]
                    for ft in self._flow_times(data_ref[flow])
                ]
                for observable in self.mc_observables
            }
            for flow in flows
        }

    def _precombine_rearranged(self, rearranged: dict) -> dict:
        """Apply ``combine`` weights to the raw MC series before averaging.

        The combination is done at the raw-data level so that whatever averager
        is used (gamma method or ``gvar.dataset.avg_data``) sees the combined
        observable as a single time series and produces statistically correct
        errors.

        **Why this matters for the gamma method.**  ``gamma_method_average``
        groups keys by observable type (``Ep``, ``Es``, …) and applies the
        gamma estimator independently within each group, setting all
        *between-group* covariances to zero.  When a combined observable such
        as ``(5/3)·E_p − (2/3)·E_s`` is formed *after* averaging, gvar
        propagates errors through the combination but uses zero for
        ``Cov(E_p, E_s)``.  Because the two energy densities are measured on
        the same gauge configurations they are highly correlated; discarding
        that correlation overestimates the error on the combination —
        potentially by a factor of ~2 or more.

        **Why it is also correct for ``gvar.avg_data``.**  For a linear
        combination, ``avg(a·X + b·Y) = a·avg(X) + b·avg(Y)``, so pre-
        combining and post-combining give the same central value.  The
        difference is only in the covariance structure: pre-combining always
        produces the exact sample covariance of the combined series.

        Keys in ``rearranged`` have the format ``"{flow}_{obs}_{flowtime}"``,
        e.g. ``"wilson_Es_0.5"``.  The combined values replace the original
        entries in-place.
        """
        if not self.combine:
            return rearranged

        for combined_obs, weights in self.combine.items():
            target_obs = "E" + combined_obs          # e.g. "Es"
            target_keys = {}
            for key in list(rearranged.keys()):
                flow, obs, *rest = key.split("_", 2)
                if obs != target_obs:
                    continue
                ft = rest[0] if rest else ""
                target_keys[key] = [
                    (w, f"{flow}_E{sub}_{ft}")
                    for sub, w in weights.items()
                ]

            for target_key, src_list in target_keys.items():
                n = len(rearranged[target_key])
                combined = _numpy.zeros(n)
                for w, src_key in src_list:
                    combined += w * _numpy.asarray(rearranged[src_key], dtype=float)
                rearranged[target_key] = combined.tolist()

        return rearranged

    def delta(self, flow_times, volume: str, flow: str, observable: str):
        """Compute the correction delta-factor for a given ensemble and correction mode.

        The ``flow`` parameter is accepted for API consistency but is not used in
        the TLN calculation: the gradient flow is always Wilson (cf = 0) regardless
        of the discretisation used in the actual Runge-Kutta integration.
        """
        match self._process_config.correction:
            case "finite-volume":
                return delta_finite_volume(flow_times, self._volume_dims(volume))
            case "tree-level-normalization" | "tln":
                return delta_tln(
                    flow_times, observable, volume,
                    gauge_action=self.gauge_action,
                )
            case _:
                return _numpy.array([0.0 for _ in flow_times])

    def _norm(self, flow_times, volume: str, flow: str, observable: str):
        correction = 1.0 + self.delta(flow_times, volume, flow, observable)
        return self._coupling_norm * flow_times * flow_times / correction

    def get_g2GF_betaGF_and_Q(self, data, data_ref, flow, volume, coupling, mass):
        _ = coupling, mass
        flow_times = _numpy.array([float(ft) for ft in self._flow_times(data_ref[flow])])
        processed = {key: value[:] if isinstance(value, list) else value for key, value in data.items()}
        # combine was already applied to raw MC data in _precombine_rearranged before
        # averaging, so the combined observable is already in processed with correct
        # covariances.  The TLN correction below is therefore applied after combining.
        result = {
            "_".join(["g2", obs[-1]]): self._norm(flow_times, volume, flow, obs[-1]) * processed[flow][obs]
            for obs in processed[flow]
            if obs.startswith("E") and obs[-1] in self.os
        }
        for obs_key in list(result.keys()):
            result["_".join(["beta", obs_key[-1]])] = self._dxdlogt(result[obs_key], flow_times)
            result[obs_key] = result[obs_key][2:-2]
        result["Q"] = processed[flow]["Q"][2:-2]
        result["flow_times"] = [str(ft) for ft in flow_times][2:-2]
        for key in list(result.keys()):
            if key == "flow_times":
                continue
            result[key] = {ft: result[key][i] for i, ft in enumerate(result["flow_times"])}
        return result

    def _reset_processed_data(self) -> None:
        self._clear_gvar_state()
        self._renew_gvar_environment()

    def _clear_gvar_state(self) -> None:
        """Drop every container that holds gvars. Subclasses extend this."""
        del self.data, self.avg_data
        self.data, self.avg_data = {}, {}
        self._ensemble_stats = {}
        self.skipped_ensembles = {}

    def _renew_gvar_environment(self) -> None:
        """Retire the previous gvar covariance environment and open a fresh one."""
        _gc.collect()
        if getattr(self, "_owns_gvar_env", False):
            _gvar.restore_gvar()
        _gvar.switch_gvar()
        self._owns_gvar_env = True

    def release_memory(self) -> None:
        """Free all processed data, fits, and gvar covariance buffers in place."""
        self._reset_processed_data()

    def _build_process_hooks(self, get_data=None, average_data=None, preprocess_data=None) -> ProcessHooks:
        return ProcessHooks(
            get=self._get if get_data is None else get_data,
            average=self._average if average_data is None else average_data,
            preprocess=self._preprocess if preprocess_data is None else preprocess_data,
        )

    def _configure_processing(self, correction, combine, mnt, mxt, mnQ, mxQ):
        self._window = FlowWindow(minimum_t=mnt, maximum_t=mxt, minimum_q=mnQ, maximum_q=mxQ)
        self._process_config = ProcessConfig(correction=correction, combine=combine)
        self.combine = combine

    def _initialize_output_branches(self, coupling: str, volume: str) -> None:
        self.data.setdefault(coupling, {})[volume] = {}
        self.avg_data.setdefault(coupling, {})[volume] = {}

    def _load_ensemble_raw_data(self, ensemble: EnsembleKey, flows: list[str], hooks: ProcessHooks, path: str) -> dict:
        return {
            flow: hooks.get(ensemble.coupling, ensemble.volume, ensemble.mass, flow, path)
            for flow in flows
        }

    def _record_ensemble_statistics(self, ensemble: EnsembleKey, raw_by_flow: dict) -> None:
        config_counts = {}
        for flow, observables in raw_by_flow.items():
            first_key = next(iter(observables), None)
            config_counts[flow] = 0 if first_key is None else len(observables[first_key])
        self._ensemble_stats[ensemble] = {"configs": config_counts, "binsize": self._binsize}

    @staticmethod
    def integrated_autocorrelation_time(series, max_lag: int | None = None) -> float:
        """Estimate the integrated autocorrelation time of a Monte Carlo series."""
        return integrated_autocorrelation_time(series, max_lag=max_lag)

    def montecarlo_series(self, coupling: str, volume: str, mass: str, flow: str, observable: str, flow_time: str):
        """Return the raw Monte Carlo history of one observable at one flow time."""
        flow_times = self.data[coupling][volume][mass][flow]["flow_times"]
        if flow_time not in flow_times:
            raise BetaFunctionException(
                f"flow time {flow_time} not stored for this ensemble;",
                "available: " + ", ".join(flow_times),
            )
        return _numpy.asarray(self.data[coupling][volume][mass][flow][observable][flow_times.index(flow_time)])

    def data_report(self, flow_time: str | None = None) -> str:
        """Return a per-ensemble table of statistics: configs, bins, and tau_int."""
        if not self._ensemble_stats:
            raise BetaFunctionException("Must run process_data before data_report")
        header = f"{'ensemble':<38} {'flow':<10} {'configs':>7} {'bins':>6}"
        if flow_time is not None:
            header += f" {'tau_int':>8}"
        lines = [header, len(header) * "-"]
        for ensemble in sorted(self._ensemble_stats, key=lambda k: k.sort_key):
            stats = self._ensemble_stats[ensemble]
            for flow, n_configs in stats["configs"].items():
                row = f"{str(ensemble):<38} {flow:<10} {n_configs:>7} {n_configs // stats['binsize']:>6}"
                if flow_time is not None:
                    try:
                        series = self.montecarlo_series(
                            ensemble.coupling, ensemble.volume, ensemble.mass,
                            flow, "E" + self.os[0], flow_time,
                        )
                        row += f" {self.integrated_autocorrelation_time(series):>8.2f}"
                    except (KeyError, BetaFunctionException):
                        row += f" {'n/a':>8}"
                lines.append(row)
        if self.skipped_ensembles:
            lines.append("")
            lines.append(f"Quarantined ({len(self.skipped_ensembles)} empty file(s)):")
            for ensemble in sorted(self.skipped_ensembles, key=lambda k: k.sort_key):
                lines.append(f"  {ensemble}")
        return "\n".join(lines)

    def _build_unaveraged_output(self, flows, raw_by_flow, rearranged):
        output = self._undorearrange(rearranged, flows, raw_by_flow)
        for flow in flows:
            output[flow]["flow_times"] = self._flow_times(raw_by_flow[flow])
        return output

    def _build_averaged_output(self, ensemble, flows, raw_by_flow, hooks, rearranged):
        averaged = hooks.average(rearranged, process=hooks.preprocess)
        averaged_by_flow = self._undorearrange(averaged, flows, raw_by_flow)
        return {
            flow: self.get_g2GF_betaGF_and_Q(
                averaged_by_flow, raw_by_flow, flow,
                ensemble.volume, ensemble.coupling, ensemble.mass,
            )
            for flow in flows
        }

    def _process_single_ensemble(self, ensemble, flows, hooks, path, verbosity):
        if verbosity >= 1:
            self._start_timer()
            print(ensemble.log_message())
        try:
            raw_by_flow = self._load_ensemble_raw_data(ensemble, flows, hooks, path)
        except EmptyEnsembleError as err:
            reason = str(err).strip("-.\n")
            self.skipped_ensembles[ensemble] = reason
            self.log.write(f"SKIPPED {ensemble}:", reason)
            if verbosity >= 1:
                print("skipped (empty data file)\n" + 25 * "-.")
            return
        self._record_ensemble_statistics(ensemble, raw_by_flow)
        rearranged = self._rearrange(raw_by_flow)
        rearranged = self._precombine_rearranged(rearranged)   # combine before averaging
        self.data[ensemble.coupling][ensemble.volume][ensemble.mass] = self._build_unaveraged_output(flows, raw_by_flow, rearranged)
        self.avg_data[ensemble.coupling][ensemble.volume][ensemble.mass] = self._build_averaged_output(
            ensemble, flows, raw_by_flow, hooks, rearranged,
        )
        if verbosity >= 1:
            print("dt =", self._stop_timer(), "(secs)\n" + 25 * "-.")

    @classmethod
    def discover_dataset(cls, data_path: str | Path, flows: list[str] | None = None) -> dict:
        """Discover .bin ensembles and return the process_data input mapping."""
        return DatasetCatalog.scan(data_path, flows=flows).to_mapping()

    @classmethod
    def catalog(cls, data_path: str | Path, flows: list[str] | None = None) -> DatasetCatalog:
        """Return a queryable DatasetCatalog for the given data directory."""
        return DatasetCatalog.scan(data_path, flows=flows)

    def process_data(
        self,
        data: dict,
        path: str = "",
        get_data=None,
        average_data=None,
        preprocess_data=None,
        correction: str = "finite-volume",
        combine: dict | None = None,
        mnt: float = 0.0,
        mxt: float = _numpy.inf,
        mnQ: float = -_numpy.inf,
        mxQ: float = _numpy.inf,
        verbosity: int = 0,
        use_gamma_method: bool = False,
        gamma_window_factor: float = 3.0,
    ):
        """Load and preprocess all ensembles into the analysis-ready avg_data structure.

        When ``correction='tln'`` (or ``'tree-level-normalization'``), the TLN
        correction factors are computed on-the-fly via spectral decomposition for
        each unique (Ns, Nt, gauge_action, observable) combination encountered.
        Results are cached within the Python session so repeated calls with the
        same geometry are fast.  No pre-computed ``.tln`` files are required.
        """
        if use_gamma_method and average_data is None:
            _wf = gamma_window_factor

            def _gm_average(d, process=None):
                return self._gamma_method_average(d, process=process, window_factor=_wf)

            average_data = _gm_average

        hooks = self._build_process_hooks(get_data=get_data, average_data=average_data, preprocess_data=preprocess_data)
        self._configure_processing(
            correction=correction,
            combine=combine,
            mnt=mnt, mxt=mxt, mnQ=mnQ, mxQ=mxQ,
        )
        self._reset_processed_data()

        for coupling, volumes in data.items():
            self.data[coupling], self.avg_data[coupling] = {}, {}
            for volume, masses in volumes.items():
                self._initialize_output_branches(coupling, volume)
                for mass, flows in masses.items():
                    ensemble = EnsembleKey(coupling=coupling, volume=volume, mass=mass)
                    self._process_single_ensemble(ensemble, flows, hooks, path, verbosity)

        self._prune_empty_branches()
        if self.skipped_ensembles and verbosity >= 1:
            print(f"Quarantined {len(self.skipped_ensembles)} empty ensemble file(s); see data_report().")

    def _prune_empty_branches(self) -> None:
        for container in (self.data, self.avg_data):
            for coupling in list(container):
                for volume in list(container[coupling]):
                    if not container[coupling][volume]:
                        del container[coupling][volume]
                if not container[coupling]:
                    del container[coupling]

    def _iv_xtrp_fcn(self, x, p: Mapping[str, object]):
        return p["k1(t;beta)"][0] + p["k2(t;beta)"][0] * x

    def _ch_xtrp_fcn(self, x, p: Mapping[str, object]):
        return p["k1(t;beta,L)"][0] + p["k2(t;beta,L)"][0] * x

    def set_binsize(self, binsize: int):
        """Set bootstrap bin size used by the default preprocessing hook."""
        if binsize <= 0:
            raise BetaFunctionException("binsize must be positive")
        self._binsize = binsize

    def load(self, info: str, fn: str, **kwargs):
        """Load serialized finite-volume or infinite-volume artifacts."""
        match info:
            case "fv":
                self.avg_data = _gvar.load(fn, **kwargs)
            case "iv":
                self.iv_fits = _gvar.load(fn, **kwargs)
            case _:
                raise BetaFunctionException(info + " is not a valid option.")

    def load_iv(self, fn: str, fcn: Callable | None = None, exclude: dict | None = None, **kwargs):
        """Load infinite-volume fits and configure IV model/exclusion metadata."""
        if hasattr(self, "iv_fits"):
            del self.iv_fits
        _gc.collect()
        self.load("iv", fn, **kwargs)
        self.iv_fcn = self._iv_xtrp_fcn if fcn is None else fcn
        if not self.avg_data:
            self._iv_exclude = {}
        elif exclude is None:
            self._iv_exclude = {coupling: [] for coupling in self.avg_data}
        else:
            self._iv_exclude = exclude

    def save(self, info: str, fn: str, **kwargs):
        """Persist finite-volume processed data or infinite-volume fit artifacts."""
        match info:
            case "fv":
                _gvar.dump(self.avg_data, fn, **kwargs)
            case "iv":
                _gvar.dump(self.iv_fits, fn, **kwargs)
            case _:
                raise BetaFunctionException(info + " is not a valid option.")

    def get(self, info: str):
        """Return in-memory finite-volume or infinite-volume artifacts."""
        match info:
            case "fv":
                return self.avg_data
            case "iv":
                return self.iv_fits
            case _:
                raise BetaFunctionException(info + " is not a valid option.")
