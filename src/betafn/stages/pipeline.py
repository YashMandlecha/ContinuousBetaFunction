"""PipelineMixin — high-level orchestration, diagnostics, and I/O."""
from __future__ import annotations

import datetime as _datetime
import importlib.metadata as _importlib_metadata
import platform as _platform
import time as _time

import gvar as _gvar
import numpy as _numpy

from ..base.exceptions import BetaFunctionException
from ..base.specs import AnalysisConfig, AnalysisResult
from ..fitting.containers import FitInput, StageStore


class PipelineMixin:
    """High-level pipeline orchestration, quality diagnostics, and I/O."""

    STAGE_NAME_MAP = {
        "chiral": "chiral",
        "iv": "infinite_volume",
        "infinite_volume": "infinite_volume",
        "ntrp": "interpolation",
        "interpolation": "interpolation",
        "cnt": "continuum",
        "continuum": "continuum",
    }

    def _resolve_stage(self, stage_name: str) -> StageStore:
        if stage_name not in self.STAGE_NAME_MAP:
            raise BetaFunctionException(
                "Unknown stage name.",
                "Allowed names: " + ", ".join(sorted(self.STAGE_NAME_MAP.keys())),
            )
        return getattr(self, self.STAGE_NAME_MAP[stage_name])

    def available_stage_names(self) -> list[str]:
        """Return all accepted stage names and aliases."""
        return sorted(self.STAGE_NAME_MAP.keys())

    def stage_store(self, stage: str) -> StageStore:
        """Return the StageStore for a stage alias."""
        return self._resolve_stage(stage)

    def _flatten_quality(self, quality_dict: dict):
        def walk(node, prefix):
            if isinstance(node, dict) and "chi2" in node and "dof" in node:
                yield prefix, node
            elif isinstance(node, dict):
                for key, value in node.items():
                    yield from walk(value, prefix + (key,))
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    yield from walk(value, prefix + (str(i),))
        return sorted(walk(quality_dict, tuple()), key=lambda item: item[0])

    def quality_table(self, stage: str) -> list[dict[str, object]]:
        """Return flattened quality-of-fit entries for a stage."""
        store = self._resolve_stage(stage)
        rows = []
        for keys, metrics in self._flatten_quality(store.quality):
            row = {"stage": store.name, "target": "/".join(map(str, keys))}
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
        return rows

    def stage_summary(self, stage: str) -> dict[str, object]:
        """Return aggregate diagnostics for a stage."""
        rows = self.quality_table(stage)
        store = self._resolve_stage(stage)
        if not rows:
            return {"stage": store.name, "n_fits": 0, "mean_chi2_per_dof": None, "mean_pvalue": None}
        chi2_over_dof, p_values = [], []
        for row in rows:
            chi2, dof, p_value = row.get("chi2"), row.get("dof"), row.get("p-value")
            if dof not in (None, 0):
                chi2_over_dof.append(float(chi2) / float(dof))
            if p_value is not None:
                p_values.append(float(p_value))
        return {
            "stage": store.name,
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
        versions = {}
        for pkg in ("numpy", "scipy", "gvar", "lsqfit"):
            try:
                versions[pkg] = _importlib_metadata.version(pkg)
            except _importlib_metadata.PackageNotFoundError:
                versions[pkg] = "unknown"
        return {
            "timestamp": _datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "python": _platform.python_version(),
            "platform": _platform.platform(),
            "packages": versions,
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

    # ------------------------------------------------------------------
    # High-level pipeline methods (take AnalysisConfig, delegate to stages)
    # ------------------------------------------------------------------

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
            combine=config.combine,
            mnt=config.process_window[0],
            mxt=config.process_window[1],
            verbosity=config.verbosity,
            use_gamma_method=config.use_gamma_method,
            gamma_window_factor=config.gamma_window_factor,
        )

    def run_chiral(self, config: AnalysisConfig) -> None:
        """Stage 2: chiral extrapolation."""
        fit_window = config.process_window if config.fit_window is None else config.fit_window
        self.ch_xtrp(mnt=fit_window[0], mxt=fit_window[1], v=config.verbosity)

    def run_infinite_volume(self, config: AnalysisConfig) -> None:
        """Stage 3: infinite-volume extrapolation."""
        fit_window = config.process_window if config.fit_window is None else config.fit_window
        self.iv_xtrp(mnt=fit_window[0], mxt=fit_window[1], v=config.verbosity)

    def run_interpolation(self, config: AnalysisConfig) -> None:
        """Stage 4: g^2 interpolation."""
        self.iv_ntrp(
            fcn=config.interpolation.fcn,
            prior=config.interpolation.prior,
            p0=config.interpolation.p0,
            xerrors=config.interpolation.xerrors,
            v=config.verbosity,
        )

    def run_continuum(self, config: AnalysisConfig) -> None:
        """Stage 5: continuum limit, with optional window scan."""
        mng2, mxg2, dg2 = config.g2_grid
        kwargs = dict(
            mng2=mng2, mxg2=mxg2, dg2=dg2, v=config.verbosity,
            diagonal=config.diagonal, shrink=config.shrink, alpha=config.alpha,
            error_mode=config.error_mode, tau0=config.tau0,
            cov_mode=config.cov_mode, correlated=config.correlated,
        )
        if config.scan_windows:
            windows = [config.continuum_window, *config.scan_windows]
            self.continuum_window_scan(windows, **kwargs)
        else:
            self.cnt_xtrp(mnt=config.continuum_window[0], mxt=config.continuum_window[1], **kwargs)

    def collect_result(self, config: AnalysisConfig, wall_time: float = 0.0) -> AnalysisResult:
        """Bundle current stage state into an AnalysisResult with provenance."""
        continuum: dict = {}
        systematics: dict = {}
        for flow in config.flows:
            for obs in config.observables:
                if flow in self.g2s and obs in self.g2s[flow]:
                    continuum[(flow, obs)] = self.continuum_curve(flow, obs)
                    if config.scan_windows:
                        systematics[(flow, obs)] = self.continuum_systematics(flow, obs)
        prov = self.provenance()
        prov["config"] = config.describe()
        return AnalysisResult(
            config=config, provenance=prov, summaries=self.analysis_summary(),
            continuum=continuum, systematics=systematics, wall_time=wall_time,
        )

    def run_analysis(self, config: AnalysisConfig) -> AnalysisResult:
        """Execute the full pipeline from raw data to continuum limit."""
        start_time = _time.time()
        self.run_processing(config)
        self.run_chiral(config)
        self.run_infinite_volume(config)
        self.run_interpolation(config)
        self.run_continuum(config)
        return self.collect_result(config, wall_time=round(_time.time() - start_time, 2))

    def save_analysis(self, file_name: str, **kwargs) -> None:
        """Persist processed data and all stage fits/diagnostics."""
        payload = {
            "avg_data": self.avg_data,
            "provenance": self.provenance(),
            "stages": {
                stage.name: {
                    "fits": stage.fits, "quality": stage.quality,
                    "domains": stage.domains, "metadata": stage.metadata,
                }
                for stage in (self.chiral, self.infinite_volume, self.interpolation, self.continuum)
            },
            "continuum_inputs": {"g2s": self.g2s, "betas": self.betas},
        }
        _gvar.dump(payload, file_name, **kwargs)

    def load_analysis(self, file_name: str, **kwargs) -> None:
        """Restore processed data and stage fits saved by save_analysis."""
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
