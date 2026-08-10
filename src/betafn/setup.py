from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import gc as _gc
import os as _os
import pickle as _pickle
from pathlib import Path
import re as _re
import time as _time
import warnings as _warnings

import gvar as _gvar
import numpy as _numpy
from scipy.special import zeta as _zeta


class BetaFunctionLog:
	"""Minimal file-backed logger used by the analysis classes."""

	def __init__(self, file_name: str | None):
		self.fn = file_name
		self.file = None
		if file_name is not None:
			self.file = open(file_name, "w", encoding="utf-8")

	def _emit(self, text: str) -> None:
		if self.file is None:
			return
		self.file.write(text)
		self.file.flush()

	def __call__(self, message, category, filename, lineno, file=None, line=None):
		formatted = 25 * "-." + "\nWARNING:\n"
		formatted += _warnings.formatwarning(message, category, filename, lineno, line=line)
		formatted += 25 * "-." + "\n"
		self._emit(formatted)

	def write(self, *lines: object) -> None:
		formatted = 25 * "-." + "\n"
		for line in lines:
			formatted += str(line) + "\n"
		formatted += 25 * "-." + "\n"
		self._emit(formatted)

	def close(self) -> None:
		if self.file is not None:
			self.file.close()
			self.file = None


class BetaFunctionException(Exception):
	def __init__(self, *lines: object):
		message = 25 * "-." + "\n"
		for line in lines:
			message += str(line) + "\n"
		message += 25 * "-." + "\n"
		Exception.__init__(self, message)


class EmptyEnsembleError(BetaFunctionException):
	"""Raised when a data file contains no measurements or no flow-time grid."""


@dataclass(frozen=True)
class FlowWindow:
	minimum_t: float = 0.0
	maximum_t: float = _numpy.inf
	minimum_q: float = -_numpy.inf
	maximum_q: float = _numpy.inf

	def includes_time(self, flow_time: str | float) -> bool:
		return self.minimum_t <= float(flow_time) <= self.maximum_t

	@property
	def uses_topological_filter(self) -> bool:
		return not (_numpy.isinf(self.minimum_q) and _numpy.isinf(self.maximum_q))


@dataclass(frozen=True)
class ProcessConfig:
	correction: str = "finite-volume"
	tree_level_normalization_data_path: str = "./"
	combine: Mapping[str, Mapping[str, float]] | None = None


@dataclass(frozen=True)
class ProcessHooks:
	get: Callable
	average: Callable
	preprocess: Callable


@dataclass(frozen=True)
class EnsembleKey:
	"""Physics-aware identity of one ensemble: bare coupling, volume, and mass.

	Beyond acting as a dictionary key, it decodes the lattice geometry and
	physical parameters from the string tokens so downstream code never has
	to re-parse `"7p00"`-style labels by hand.
	"""

	coupling: str
	volume: str
	mass: str

	@property
	def beta_value(self) -> float:
		return float(self.coupling.replace("p", "."))

	@property
	def mass_value(self) -> float:
		return float(self.mass.replace("p", "."))

	@property
	def dimensions(self) -> tuple[int, ...]:
		return tuple(int(dim) for dim in self.volume.replace("t", "l").split("l")[1:] if dim)

	@property
	def spatial_extent(self) -> int:
		return self.dimensions[0]

	@property
	def temporal_extent(self) -> int:
		return self.dimensions[-1]

	@property
	def aspect_ratio(self) -> float:
		return self.temporal_extent / self.spatial_extent

	@property
	def lattice_volume(self) -> float:
		return float(_numpy.prod(self.dimensions))

	@property
	def sort_key(self) -> tuple[float, float, float]:
		return (self.beta_value, self.lattice_volume, self.mass_value)

	def __str__(self) -> str:
		geometry = "x".join(map(str, self.dimensions))
		return f"beta={self.beta_value:g} V={geometry} m={self.mass_value:g}"

	def log_message(self) -> str:
		message = 25 * "-." + "\n"
		message += "beta_b = " + self.coupling.replace("p", ".")
		message += ", vol = " + self.volume
		message += ", mass = " + self.mass
		return message


@dataclass(frozen=True)
class EnsembleFile:
	"""One data file on disk: an ensemble measured with one flow discretization."""

	key: EnsembleKey
	flow: str
	file_path: Path

	def __str__(self) -> str:
		return f"{self.key} flow={self.flow}"


@dataclass(frozen=True)
class DatasetCatalog:
	"""Immutable, queryable inventory of ensemble data files.

	Scan a directory once, then slice the catalog with `filter`, inspect it
	with `summary`, and hand it straight to `process_data`.  Filtering
	returns new catalogs, so exploratory dataset selection composes cleanly:

		catalog = DatasetCatalog.scan("data", flows=["wilson"])
		massless = catalog.filter(predicate=lambda e: e.key.mass_value == 0.0)
	"""

	FILE_PATTERN = _re.compile(
		r"^(?P<beta>[^_]+)_(?P<volume>l\d+l\d+l\d+t\d+)_(?P<mass>[^_]+)_(?P<flow>[^.]+)\.bin$"
	)

	root: Path
	entries: tuple[EnsembleFile, ...]

	@classmethod
	def scan(cls, data_path: str | Path, flows: list[str] | None = None) -> "DatasetCatalog":
		"""Build a catalog from the .bin files under data_path."""
		root = Path(data_path)
		allowed_flows = None if flows is None else set(flows)
		entries = []
		for file_path in sorted(root.glob("*.bin")):
			match = cls.FILE_PATTERN.match(file_path.name)
			if match is None:
				continue
			if allowed_flows is not None and match.group("flow") not in allowed_flows:
				continue
			key = EnsembleKey(match.group("beta"), match.group("volume"), match.group("mass"))
			entries.append(EnsembleFile(key=key, flow=match.group("flow"), file_path=file_path))
		entries.sort(key=lambda entry: (*entry.key.sort_key, entry.flow))
		return cls(root=root, entries=tuple(entries))

	def __len__(self) -> int:
		return len(self.entries)

	def __iter__(self):
		return iter(self.entries)

	def __repr__(self) -> str:
		return (
			f"DatasetCatalog(root={str(self.root)!r}, files={len(self.entries)}, "
			f"couplings={len(self.couplings)}, flows={sorted(self.flows)})"
		)

	@property
	def couplings(self) -> list[str]:
		return sorted({entry.key.coupling for entry in self.entries}, key=lambda c: float(c.replace("p", ".")))

	@property
	def flows(self) -> set[str]:
		return {entry.flow for entry in self.entries}

	def filter(
		self,
		couplings: list[str] | None = None,
		volumes: list[str] | None = None,
		masses: list[str] | None = None,
		flows: list[str] | None = None,
		predicate: Callable[[EnsembleFile], bool] | None = None,
	) -> "DatasetCatalog":
		"""Return a new catalog restricted to the requested slice."""
		selected = []
		for entry in self.entries:
			if couplings is not None and entry.key.coupling not in couplings:
				continue
			if volumes is not None and entry.key.volume not in volumes:
				continue
			if masses is not None and entry.key.mass not in masses:
				continue
			if flows is not None and entry.flow not in flows:
				continue
			if predicate is not None and not predicate(entry):
				continue
			selected.append(entry)
		return DatasetCatalog(root=self.root, entries=tuple(selected))

	def to_mapping(self) -> dict:
		"""Return the nested mapping consumed by process_data."""
		mapping: dict = {}
		for entry in self.entries:
			flows = mapping.setdefault(entry.key.coupling, {}).setdefault(entry.key.volume, {}).setdefault(entry.key.mass, [])
			if entry.flow not in flows:
				flows.append(entry.flow)
		return mapping

	def summary(self) -> str:
		"""Return a per-coupling inventory table."""
		header = f"{'beta':>7}  {'volumes':>7}  {'masses':>6}  {'files':>5}  flows"
		lines = [f"Dataset catalog: {self.root}  ({len(self.entries)} files)", header, len(header) * "-"]
		for coupling in self.couplings:
			slice_entries = [entry for entry in self.entries if entry.key.coupling == coupling]
			n_volumes = len({entry.key.volume for entry in slice_entries})
			n_masses = len({entry.key.mass for entry in slice_entries})
			flow_names = ",".join(sorted({entry.flow for entry in slice_entries}))
			lines.append(
				f"{coupling.replace('p', '.'):>7}  {n_volumes:>7}  {n_masses:>6}  {len(slice_entries):>5}  {flow_names}"
			)
		return "\n".join(lines)


class PerturbativeBetaFunction:
	"""Perturbative gradient-flow beta function from JHEP06(2019)121."""

	def __init__(self, nf: float | int, nc: float | int):
		tr = 0.5
		tf = tr * nf
		cf = 0.5 * (nc * nc - 1.0) / nc
		ca = nc

		b0 = 11.0 * ca / 3.0 - 4.0 * tf / 3.0
		b1 = 34.0 * ca * ca / 3.0 - (4.0 * cf + 20.0 * ca / 3.0) * tf
		b2 = 2857.0 * ca * ca * ca / 54.0 - 1415.0 * ca * ca * tf / 27.0
		b2 += (-205.0 * cf * ca / 9.0 + 2.0 * cf * cf) * tf
		b2 += (44.0 * cf / 9.0 + 158.0 * ca / 27.0) * tf * tf

		rho = 1.0 / 8.0
		e10 = (52.0 / 9.0 + 22.0 * _numpy.log(2.0) / 3.0 - 3.0 * _numpy.log(3.0)) * ca
		e10 -= 8.0 * tf / 9.0
		e20 = 27.9786 * ca * ca - 31.5652 * tf * ca + (16.0 * _zeta(3.0) - 43.0 / 3.0) * tf * cf
		e20 += (8.0 * _numpy.pi * _numpy.pi / 27.0 - 80.0 / 81.0) * tf * tf

		log_factor = _numpy.log(2.0 * rho) + _numpy.euler_gamma
		e1 = e10 + b0 * log_factor
		e2 = e20 + (2.0 * b0 * e10 + b1) * log_factor + (b0 * log_factor) * (b0 * log_factor)
		b2 = b2 - e1 * b1 + (e2 - e1 * e1) * b0

		self.nrm = 4.0 * _numpy.pi
		self.b = [coef / self.nrm ** (order + 1) for order, coef in enumerate([b0, b1, b2])]

	def __call__(self, x, loops: int = 3):
		beta_function = sum(coef * (x / self.nrm) ** (order + 2) for order, coef in enumerate(self.b) if order < loops)
		return -self.nrm * beta_function


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
		self.combine = None
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
		combine=None,
		mnt: float = 0.0,
		mxt: float = _numpy.inf,
		mnQ: float = -_numpy.inf,
		mxQ: float = _numpy.inf,
		verbosity: int = 0,
	):
		"""Load and preprocess all ensembles into the analysis-ready avg_data structure."""
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


if __name__ == "__main__":
	pass
