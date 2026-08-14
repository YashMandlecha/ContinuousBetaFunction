"""Declarative analysis configuration, interpolation specs, and result containers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as _numpy
import gvar as _gvar

from .exceptions import BetaFunctionException


@dataclass(frozen=True)
class InterpolationSpec:
    """Declarative specification of the g^2 interpolation model."""

    fcn: Callable
    p0: dict
    prior: dict | None = None
    xerrors: bool = False

    @classmethod
    def polynomial(cls, order: int, width: float = 10.0, xerrors: bool = False) -> "InterpolationSpec":
        from ..fitting.families import polynomial_interpolation  # lazy — avoids circular import
        fcn, prior, p0 = polynomial_interpolation(order, width=width)
        return cls(fcn=fcn, prior=prior, p0=p0, xerrors=xerrors)

    @classmethod
    def perturbative(
        cls,
        nf: float | int,
        nc: float | int = 3,
        loops: int = 2,
        correction_order: int = 2,
        free_intercept: bool = False,
        intercept_width: float = 0.2,
        width: float = 5.0,
        xerrors: bool = False,
    ) -> "InterpolationSpec":
        from ..fitting.families import perturbative_interpolation  # lazy — avoids circular import
        return perturbative_interpolation(
            nf=nf,
            nc=nc,
            loops=loops,
            correction_order=correction_order,
            free_intercept=free_intercept,
            intercept_width=intercept_width,
            width=width,
            xerrors=xerrors,
        )


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete declarative specification of one beta-function analysis.

    One config object equals one reproducible analysis: it fixes the dataset,
    processing windows, fit windows, interpolation model, and continuum
    extraction in a single immutable value that is stored in the result's
    provenance.
    """

    data_path: str
    interpolation: InterpolationSpec
    continuum_window: tuple[float, float]
    g2_grid: tuple[float, float, float]
    flows: tuple[str, ...] = ("wilson",)
    observables: tuple[str, ...] = ("p", "s", "c")
    combine: dict[str, dict[str, float]] = field(default_factory=dict)
    couplings: tuple[str, ...] | None = None
    volumes: dict[str, tuple[str, ...]] | None = None
    correction: str = "finite-volume"
    binsize: int = 1
    use_gamma_method: bool = False
    gamma_window_factor: float = 3.0
    process_window: tuple[float, float] = (0.0, _numpy.inf)
    fit_window: tuple[float, float] | None = None
    diagonal: bool = False
    shrink: bool = False
    alpha: float = 0.25
    error_mode: str = "fit"
    tau0: float = 0.0
    cov_mode: str | None = None
    correlated: bool = True
    scan_windows: tuple[tuple[float, float], ...] = tuple()
    verbosity: int = 1

    def __post_init__(self):
        mnt, mxt = self.continuum_window
        if mnt >= mxt:
            raise BetaFunctionException("continuum_window must satisfy mnt < mxt")
        mng2, mxg2, dg2 = self.g2_grid
        if not (mng2 < mxg2 and dg2 > 0.0):
            raise BetaFunctionException("g2_grid must satisfy mng2 < mxg2 and dg2 > 0")
        if self.binsize <= 0:
            raise BetaFunctionException("binsize must be positive")
        if self.error_mode not in ("fit", "shifted", "gp"):
            raise BetaFunctionException("error_mode must be 'fit', 'shifted', or 'gp'")

    def describe(self) -> dict[str, object]:
        """Return a JSON-friendly description of this configuration."""
        return {
            "data_path": str(self.data_path),
            "flows": list(self.flows),
            "observables": list(self.observables),
            "couplings": None if self.couplings is None else list(self.couplings),
            "volumes": None if self.volumes is None else {key: list(value) for key, value in self.volumes.items()},
            "correction": self.correction,
            "binsize": self.binsize,
            "use_gamma_method": self.use_gamma_method,
            "gamma_window_factor": self.gamma_window_factor,
            "process_window": list(self.process_window),
            "fit_window": None if self.fit_window is None else list(self.fit_window),
            "continuum_window": list(self.continuum_window),
            "g2_grid": list(self.g2_grid),
            "diagonal": self.diagonal,
            "shrink": self.shrink,
            "alpha": self.alpha,
            "error_mode": self.error_mode,
            "tau0": self.tau0,
            "cov_mode": self.cov_mode,
            "correlated": self.correlated,
            "scan_windows": [list(window) for window in self.scan_windows],
            "interpolation_xerrors": self.interpolation.xerrors,
            "interpolation_parameters": sorted(self.interpolation.p0.keys()),
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of a full pipeline run: provenance, diagnostics, and final curves."""

    config: AnalysisConfig
    provenance: dict[str, object]
    summaries: dict[str, dict[str, object]]
    continuum: dict[tuple[str, str], tuple[_numpy.ndarray, list]]
    systematics: dict[tuple[str, str], list[dict[str, float]]]
    wall_time: float

    def curve(self, flow: str, obs: str) -> tuple[_numpy.ndarray, list]:
        return self.continuum[(flow, obs)]

    def report(self) -> str:
        lines = ["Beta-function analysis result", 50 * "="]
        lines.append(f"completed: {self.provenance['timestamp']}  (wall time {self.wall_time:.1f} s)")
        lines.append(f"dataset:   {self.config.data_path}")
        lines.append(f"continuum window: t/a^2 in {list(self.config.continuum_window)}")
        lines.append("")
        lines.append("Stage diagnostics")
        lines.append(50 * "-")
        for stage, payload in self.summaries.items():
            chi2 = payload["mean_chi2_per_dof"]
            pval = payload["mean_pvalue"]
            chi2_text = "n/a" if chi2 is None else f"{chi2:.2f}"
            pval_text = "n/a" if pval is None else f"{pval:.2f}"
            lines.append(
                f"{stage:<18} fits: {payload['n_fits']:>5}   <chi2/dof>: {chi2_text:>6}   <p>: {pval_text:>5}"
            )
        lines.append("")
        lines.append("Continuum curves")
        lines.append(50 * "-")
        for (flow, obs), (g2_values, betas) in self.continuum.items():
            if len(g2_values) == 0:
                continue
            lines.append(
                f"{flow}/{obs}: {len(g2_values)} points, g^2 in [{g2_values[0]:.2f}, {g2_values[-1]:.2f}]"
            )
        if self.systematics:
            lines.append("")
            lines.append("Window-scan systematics available for: " + ", ".join(f"{flow}/{obs}" for flow, obs in self.systematics))
        return "\n".join(lines)
