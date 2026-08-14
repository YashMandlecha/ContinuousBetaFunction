"""Data loading, preprocessing, and the SetupBetaFunction base class."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import gc as _gc
import os as _os
import pickle as _pickle
from pathlib import Path
import time as _time
import warnings as _warnings

import gvar as _gvar
import numpy as _numpy

from .exceptions import BetaFunctionException, EmptyEnsembleError, BetaFunctionLog
from .catalog import (
    FlowWindow, 
    ProcessConfig, 
    ProcessHooks,
    EnsembleKey, 
    EnsembleFile, 
    DatasetCatalog,
)
from .perturbative import PerturbativeBetaFunction


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

    @property
    def _tln_path(self) -> str:
        return self._process_config.tree_level_normalization_data_path

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
        return [str(flow_time) for flow_time in flow_times]

    def _volume_dims(self, volume: str) -> list[float]:
        dims = volume.replace("t", "l").split("l")[1:]
        return [float(dim) for dim in dims if dim]

    def _apply_topological_filter(self, raw_data: dict) -> Callable[[list], list]:
        if "Q" not in raw_data or not self._window.uses_topological_filter:
            return lambda values: values

        charges = raw_data["Q"][-1]

        def apply(values):
            return [
                value
                for index, value in enumerate(values)
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
            "_".join([flow, observable_with_time]): data[flow][observable_with_time]
            for flow in data
            for observable_with_time in data[flow]
        }

    def _preprocess(self, data: dict):
        return _gvar.dataset.bin_data(data, binsize=self._binsize)

    def _average(self, data: dict, process: Callable | None = None):
        payload = data if process is None else process(data)
        return _gvar.dataset.avg_data(payload)

    # ------------------------------------------------------------------
    # Gamma method (Madras-Sokal / Wolff UWerr) covariance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _gamma_method_covariance(
        X: _numpy.ndarray,
        window_factor: float = 3.0,
    ) -> tuple[_numpy.ndarray, _numpy.ndarray, int]:
        """Compute the mean vector and covariance-of-means using the Gamma method.

        Standard Markov-chain averages underestimate statistical errors whenever
        successive configurations are autocorrelated (τ_int > 0.5).  For a single
        observable the fix is well-known—bin or multiply σ² by 2τ_int—but the
        off-diagonal entries of the covariance matrix between two observables α, β
        measured on the *same* MC history each receive a *different* correction that
        depends on the full cross-spectral density Γ_αβ(k).  No single τ_int can
        handle both diagonal and off-diagonal entries simultaneously; the only
        rigorous solution is the Gamma method.

        The Gamma method integrates the exact cross-correlation functions::

            Cov(ā_α, ā_β) = (1/N) * [Γ_αβ(0) + Σ_{k=1}^W (Γ_αβ(k) + Γ_βα(k))]

        where Γ_αβ(k) = (1/N) Σ_s δX_α(s) δX_β(s+k) is the lag-k cross-correlator
        of the centered series, and W is the Sokal automatic window.

        **Window selection.**  The window W is chosen to satisfy
        W ≥ window_factor × max_α τ_int(α), using the diagonal per-observable
        integrated autocorrelation times.  Because the same W is used for every
        pair (α, β), the resulting covariance matrix is guaranteed to be positive
        semi-definite.  The default window_factor=3.0 is intentionally conservative
        (Sokal recommends 1.5); this ensures that the estimate converges from below
        toward the true covariance rather than cutting off the window too early.

        **Computational cost.**  The dominant step is W+1 matrix multiplications of
        shape (M, N) × (N, M), each an O(N·M²) BLAS DGEMM.  For typical ensembles
        (N≈200–700, M≈651 flow times, W≈5–50), this takes 0.1–5 seconds per
        observable-type group; see _gamma_method_average for the grouping strategy.

        References: Madras & Sokal (1988) J.Stat.Phys. 50, 109;
        Wolff (2004) Comput.Phys.Commun. 156, 143.

        Parameters
        ----------
        X : ndarray, shape (N, M)
            Raw MC time series in trajectory order.  N = number of configurations,
            M = number of primary observables (e.g., E(t) at every flow time in
            the analysis window).
        window_factor : float
            Safety factor S for the Sokal window: W = ceil(S × max_α τ_int(α)).
            Default 3.0 is conservative; Sokal's original recommendation is 1.5.

        Returns
        -------
        means : ndarray, shape (M,)
        cov_of_means : ndarray, shape (M, M)
            Covariance matrix of the sample means, including all autocorrelations
            up to lag W.  Pass directly to ``gvar.gvar(means, cov_of_means)``.
        W : int
            The window width used (useful for diagnostics and data_report).
        """
        N, M = X.shape
        means = X.mean(axis=0)                           # (M,)
        delta = X - means[_numpy.newaxis, :]             # (N, M) centred series

        # ---- Step 1: estimate W from per-observable diagonal τ_int ------------
        var_diag = (delta * delta).sum(axis=0) / N      # Γ_αα(0), shape (M,)
        # Guard against zero-variance observables (e.g. Q frozen at β_b = 8.50).
        safe_var = _numpy.where(var_diag > 0.0, var_diag, 1.0)

        tau_diag = _numpy.full(M, 0.5)
        max_probe = min(N // 4, 1000)
        for k in range(1, max_probe):
            lag_corr = (delta[:N - k] * delta[k:]).sum(axis=0) / (N * safe_var)
            still_active = (lag_corr > 0.0) & (k < window_factor * tau_diag)
            tau_diag = _numpy.where(still_active, tau_diag + lag_corr, tau_diag)
            if not still_active.any():
                break

        W = max(4, int(_numpy.ceil(window_factor * float(_numpy.max(tau_diag)))))
        W = min(W, N // 4)   # cap at N/4: don't estimate from fewer than 4 samples

        # ---- Step 2: accumulate the symmetrised cross-correlation sum ----------
        cov_sum = delta.T @ delta                        # N·Γ(0), shape (M, M)
        for k in range(1, W + 1):
            Gk = delta[: N - k].T @ delta[k:]           # N·Γ_αβ(k), shape (M, M)
            cov_sum += Gk + Gk.T                        # symmetrise: add Γ_βα(k)

        cov_of_means = cov_sum / (N * N)                # Cov(ā_α, ā_β)

        return means, cov_of_means, W

    def _gamma_method_average(self, data: dict, process=None, window_factor: float = 3.0) -> dict:
        """Compute averaged gvars using the Gamma method instead of binning.

        This replaces the standard _preprocess (bin_data) + _average (avg_data)
        pipeline.  The ``process`` argument is accepted for API compatibility with
        ``ProcessHooks.average`` but is ignored; the Gamma method supersedes all
        binning.

        **Grouping strategy.**  All observables in ``data`` originate from the same
        MC trajectory sequence, so their cross-correlations are physically real.
        Rather than computing one enormous covariance matrix for every key at once
        (which would be O(M_total²·N·W) in time and memory), observables are grouped
        by *type* — the second underscore-delimited token in each key (e.g. "Ep",
        "Es", "Ec", "Q") — and the Gamma method is applied independently within
        each group.  This is exact for any analysis that uses a single observable
        type at a time (the typical beta-function workflow).  The only approximation
        is that cross-covariances *between* different observable types (e.g. Ep vs Es
        at the same flow time) are set to zero, which slightly overestimates the
        error on combined observables.  The block-diagonal structure keeps each DGEMM
        at O(N·M_ft²) where M_ft is the number of flow times, rather than the full
        O(N·(N_types·M_ft)²).

        Parameters
        ----------
        data : dict
            Rearranged observable dict ``{flow_obs_flowtime: [config values]}``,
            as produced by ``_rearrange``.
        process : callable or None
            Ignored (accepted for compatibility with ``ProcessHooks``).
        window_factor : float
            Passed to ``_gamma_method_covariance``.  Default 3.0 (conservative).

        Returns
        -------
        dict
            Same key set as ``data``, values replaced by gvar primary variables
            carrying the full within-type cross-covariance structure.
        """
        if not data:
            return {}

        # Group keys by observable type (second "_"-delimited token).
        # Example: "wilson_Ep_1.50" → type "Ep".
        groups: dict[str, list[str]] = {}
        for k in data:
            obs_type = k.split("_")[1]
            groups.setdefault(obs_type, []).append(k)

        result: dict[str, object] = {}
        for obs_type, group_keys in groups.items():
            sorted_keys = sorted(group_keys)
            n_configs = len(data[sorted_keys[0]])

            # Build (N_configs, M_group) matrix and apply Gamma method.
            X = _numpy.empty((n_configs, len(sorted_keys)), dtype=float)
            for col, k in enumerate(sorted_keys):
                X[:, col] = data[k]

            means, cov_of_means, _W = self._gamma_method_covariance(
                X, window_factor=window_factor
            )

            # gvar.gvar(means, cov) creates M primary gvars sharing the given
            # covariance matrix.  All arithmetic on these gvars (e.g. norm*Ep)
            # correctly propagates the full cross-covariance via gvar's engine.
            gvars = _gvar.gvar(means, cov_of_means)
            for i, k in enumerate(sorted_keys):
                result[k] = gvars[i]

        return result

    def _undorearrange(self, data: dict, flows: list[str], data_ref: dict):
        return {
            flow: {
                observable: [data["_".join([flow, observable, flow_time])] for flow_time in self._flow_times(data_ref[flow])]
                for observable in self.mc_observables
            }
            for flow in flows
        }

    def _combined_energies(self, processed: dict, flow: str) -> None:
        if self.combine is None:
            return
        for observable, weights in self.combine.items():
            processed[flow]["E" + observable] = sum(
                weights[sub_observable] * _numpy.array(processed[flow]["E" + sub_observable])
                for sub_observable in weights
            )

    def delta(self, flow_times, volume: str, flow: str, observable: str):
        match self._process_config.correction:
            case "finite-volume":
                dims = self._volume_dims(volume)
                delta_a = -64.0 * _numpy.pi * _numpy.pi / 3.0
                delta_e = 1.0
                for dim in dims:
                    ratio = dim * dim / flow_times
                    delta_a /= _numpy.sqrt(ratio)
                    delta_e *= 1.0 + 2.0 * _numpy.exp(-0.125 * ratio) + 2.0 * _numpy.exp(-0.5 * ratio)
                return delta_a + delta_e - 1.0

            case "tree-level-normalization" | "tln":
                flow_translation = {
                    "symanzik": "s",
                    "C0p0": "p",
                    "wilson": "p",
                    "C13": "a",
                }
                if flow not in flow_translation:
                    raise BetaFunctionException(flow + " not known flow for tln")

                data_path = _os.path.join(
                    self._process_config.tree_level_normalization_data_path,
                    self.gauge_action + flow_translation[flow] + observable + volume + ".tln",
                )
                t_values, d_values = [], []
                with open(data_path, "r", encoding="utf-8") as input_file:
                    for line in input_file:
                        flow_time, delta_value, _, _ = [*map(float, line.split())]
                        t_values.append(flow_time)
                        d_values.append(delta_value)
                spline = _gvar.cspline.CSpline(t_values, d_values)
                return _numpy.array([spline(flow_time) for flow_time in flow_times]) - 1.0

            case _:
                return _numpy.array([0.0 for _ in flow_times])

    def _norm(self, flow_times, volume: str, flow: str, observable: str):
        correction = 1.0 + self.delta(flow_times, volume, flow, observable)
        return self._coupling_norm * flow_times * flow_times / correction

    def get_g2GF_betaGF_and_Q(self, data, data_ref, flow, volume, coupling, mass):
        _ = coupling, mass
        flow_times = _numpy.array([float(flow_time) for flow_time in self._flow_times(data_ref[flow])])

        processed = {key: value[:] if isinstance(value, list) else value for key, value in data.items()}
        self._combined_energies(processed, flow)

        result = {
            "_".join(["g2", observable[-1]]): self._norm(flow_times, volume, flow, observable[-1]) * processed[flow][observable]
            for observable in processed[flow]
            if observable.startswith("E") and observable[-1] in self.os
        }

        for observable_key in list(result.keys()):
            result["_".join(["beta", observable_key[-1]])] = self._dxdlogt(result[observable_key], flow_times)
            result[observable_key] = result[observable_key][2:-2]

        result["Q"] = processed[flow]["Q"][2:-2]
        result["flow_times"] = [str(flow_time) for flow_time in flow_times][2:-2]

        for key in list(result.keys()):
            if key == "flow_times":
                continue
            result[key] = {flow_time: result[key][index] for index, flow_time in enumerate(result["flow_times"])}

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
        """Retire the previous gvar covariance environment and open a fresh one.

        gvar keeps one append-only covariance matrix per environment; primary
        gvars created by averaging and fitting stay in it forever, even after
        all Python references are gone.  Reprocessing therefore leaks an
        entire analysis worth of memory each time — unless the old
        environment is retired, which is what this does.  Old gvars kept by
        the caller (e.g. an earlier AnalysisResult) remain readable but hold
        their memory until deleted, and cannot be combined with new gvars.
        """
        _gc.collect()
        if getattr(self, "_owns_gvar_env", False):
            _gvar.restore_gvar()
        _gvar.switch_gvar()
        self._owns_gvar_env = True

    def release_memory(self) -> None:
        """Free all processed data, fits, and gvar covariance buffers in place.

        Use this instead of restarting the kernel when memory piles up:
        everything on this instance is cleared and gvar's covariance matrix
        is replaced with an empty one.  Rerun the analysis stages afterwards.
        """
        self._reset_processed_data()

    def _build_process_hooks(self, get_data=None, average_data=None, preprocess_data=None) -> ProcessHooks:
        """Build pluggable processing hooks used by process_data."""
        return ProcessHooks(
            get=self._get if get_data is None else get_data,
            average=self._average if average_data is None else average_data,
            preprocess=self._preprocess if preprocess_data is None else preprocess_data,
        )

    def _configure_processing(
        self,
        correction: str,
        tree_level_normalization_data_path: str,
        combine,
        mnt: float,
        mxt: float,
        mnQ: float,
        mxQ: float,
    ) -> None:
        """Set all process-time controls (windows, correction type, and observable combinations)."""
        self._window = FlowWindow(minimum_t=mnt, maximum_t=mxt, minimum_q=mnQ, maximum_q=mxQ)
        self._process_config = ProcessConfig(
            correction=correction,
            tree_level_normalization_data_path=tree_level_normalization_data_path,
            combine=combine,
        )
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
        """Record configuration counts per flow for the data quality report."""
        config_counts = {}
        for flow, observables in raw_by_flow.items():
            first_key = next(iter(observables), None)
            config_counts[flow] = 0 if first_key is None else len(observables[first_key])
        self._ensemble_stats[ensemble] = {
            "configs": config_counts,
            "binsize": self._binsize,
        }

    @staticmethod
    def integrated_autocorrelation_time(series, max_lag: int | None = None) -> float:
        """Estimate the integrated autocorrelation time of a Monte Carlo series.

        Uses the standard sum of normalized autocorrelations with an automatic
        windowing cutoff at the first non-positive coefficient.
        """
        values = _numpy.asarray(series, dtype=float)
        n = len(values)
        if n < 4:
            return 0.5
        centered = values - values.mean()
        variance = float(centered @ centered) / n
        if variance == 0.0:
            return 0.5
        max_lag = n // 2 if max_lag is None else min(max_lag, n - 1)
        tau = 0.5
        for lag in range(1, max_lag):
            rho = float(centered[:-lag] @ centered[lag:]) / ((n - lag) * variance)
            if rho <= 0.0:
                break
            tau += rho
        return tau

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
        """Return a per-ensemble table of statistics: configs, bins, and tau_int.

        If flow_time is given, the integrated autocorrelation time of the energy
        density at that flow time is estimated for each ensemble.
        """
        if not self._ensemble_stats:
            raise BetaFunctionException("Must run process_data before data_report")

        header = f"{'ensemble':<38} {'flow':<10} {'configs':>7} {'bins':>6}"
        if flow_time is not None:
            header += f" {'tau_int':>8}"
        lines = [header, len(header) * "-"]

        for ensemble in sorted(self._ensemble_stats, key=lambda key: key.sort_key):
            stats = self._ensemble_stats[ensemble]
            for flow, n_configs in stats["configs"].items():
                row = f"{str(ensemble):<38} {flow:<10} {n_configs:>7} {n_configs // stats['binsize']:>6}"
                if flow_time is not None:
                    try:
                        observable = "E" + self.os[0]
                        series = self.montecarlo_series(
                            ensemble.coupling, ensemble.volume, ensemble.mass, flow, observable, flow_time
                        )
                        row += f" {self.integrated_autocorrelation_time(series):>8.2f}"
                    except (KeyError, BetaFunctionException):
                        row += f" {'n/a':>8}"
                lines.append(row)

        if self.skipped_ensembles:
            lines.append("")
            lines.append(f"Quarantined ({len(self.skipped_ensembles)} empty file(s)):")
            for ensemble in sorted(self.skipped_ensembles, key=lambda key: key.sort_key):
                lines.append(f"  {ensemble}")
        return "\n".join(lines)

    def _build_unaveraged_output(self, flows: list[str], raw_by_flow: dict, rearranged: dict) -> dict:
        output = self._undorearrange(rearranged, flows, raw_by_flow)
        for flow in flows:
            output[flow]["flow_times"] = self._flow_times(raw_by_flow[flow])
        return output

    def _build_averaged_output(self, ensemble: EnsembleKey, flows: list[str], raw_by_flow: dict, hooks: ProcessHooks, rearranged: dict) -> dict:
        averaged = hooks.average(rearranged, process=hooks.preprocess)
        averaged_by_flow = self._undorearrange(averaged, flows, raw_by_flow)
        return {
            flow: self.get_g2GF_betaGF_and_Q(
                averaged_by_flow,
                raw_by_flow,
                flow,
                ensemble.volume,
                ensemble.coupling,
                ensemble.mass,
            )
            for flow in flows
        }

    def _process_single_ensemble(
        self,
        ensemble: EnsembleKey,
        flows: list[str],
        hooks: ProcessHooks,
        path: str,
        verbosity: int,
    ) -> None:
        """Run load, transform, average, and writeback for one ensemble key."""
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
                print(f"skipped (empty data file)\n" + 25 * "-.")
            return

        self._record_ensemble_statistics(ensemble, raw_by_flow)
        rearranged = self._rearrange(raw_by_flow)
        self.data[ensemble.coupling][ensemble.volume][ensemble.mass] = self._build_unaveraged_output(flows, raw_by_flow, rearranged)
        self.avg_data[ensemble.coupling][ensemble.volume][ensemble.mass] = self._build_averaged_output(
            ensemble,
            flows,
            raw_by_flow,
            hooks,
            rearranged,
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
        data: dict[str, dict[str, dict[str, list[str]]]],
        path: str = "",
        get_data=None,
        average_data=None,
        preprocess_data=None,
        correction: str = "finite-volume",
        tree_level_normalization_data_path: str = "./",
        combine: dict[str, dict[str, float]] = None,
        mnt: float = 0.0,
        mxt: float = _numpy.inf,
        mnQ: float = -_numpy.inf,
        mxQ: float = _numpy.inf,
        verbosity: int = 0,
        use_gamma_method: bool = False,
        gamma_window_factor: float = 3.0,
    ):
        """Load and preprocess all ensembles into the analysis-ready avg_data structure.

        Parameters
        ----------
        use_gamma_method : bool
            When True, replace the default bin-data + avg-data pipeline with the
            Gamma method (Madras-Sokal / Wolff UWerr).  This computes the exact
            covariance of the sample means across all flow times simultaneously,
            correctly accounting for autocorrelations and cross-correlations between
            different flow times on the same MC history.  The binsize set by
            set_binsize() is ignored when this flag is True.
        gamma_window_factor : float
            Safety factor for the Sokal automatic window; see
            _gamma_method_covariance.  Default 3.0 (conservative).  Only used when
            use_gamma_method=True.
        """
        if use_gamma_method and average_data is None:
            _wf = gamma_window_factor

            def _gm_average(d, process=None):
                return self._gamma_method_average(d, process=process, window_factor=_wf)

            average_data = _gm_average

        hooks = self._build_process_hooks(get_data=get_data, average_data=average_data, preprocess_data=preprocess_data)
        self._configure_processing(
            correction=correction,
            tree_level_normalization_data_path=tree_level_normalization_data_path,
            combine=combine,
            mnt=mnt,
            mxt=mxt,
            mnQ=mnQ,
            mxQ=mxQ,
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
        """Remove volumes/couplings that ended up with no processed ensembles."""
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

    def load_iv(self, fn: str, fcn: Callable | None = None, exclude: dict[str, list[str]] | None = None, **kwargs):
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
