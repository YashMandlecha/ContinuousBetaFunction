"""Fit data containers, model wrappers, and stage storage."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import gvar as _gvar
import numpy as _numpy

from ..base.exceptions import BetaFunctionException


@dataclass
class FitInput:
    """Container for x/y fit data and optional labels/metadata."""

    x: object
    y: object
    labels: list[str] | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return len(self.x)

    def mean_x(self):
        return _gvar.mean(self.x)

    def bounds(self) -> tuple[float, float]:
        mean_x = self.mean_x()
        return min(mean_x), max(mean_x)


@dataclass
class FitModel:
    """Configurable wrapper around a fit function, prior, and initialization."""

    fcn: Callable
    prior: dict | None
    p0: dict | None
    xerrors: bool = False

    def fit(self, run_fit: Callable, run_fit_with_x_errors: Callable, fit_input: FitInput):
        if self.xerrors:
            if self.prior is None:
                fit_prior = {"x": fit_input.x}
            else:
                fit_prior = {
                    key: (value.copy() if isinstance(value, list) else value)
                    for key, value in self.prior.items()
                }
                fit_prior["x"] = fit_input.x
            return run_fit_with_x_errors(fit_input.y, self.fcn, fit_prior, self.p0)
        return run_fit(_gvar.mean(fit_input.x), fit_input.y, self.fcn, self.prior, self.p0)

    def evaluate(self, x, params):
        return self.fcn(x, params)


@dataclass
class StageStore:
    """Hierarchical storage for one analysis stage."""

    name: str
    fits: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    domains: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    model: FitModel | None = None

    def _ensure_branch(self, container: dict, keys: tuple) -> dict:
        target = container
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        return target

    def store(self, container_name: str, keys: tuple, value) -> None:
        container = getattr(self, container_name)
        self._ensure_branch(container, keys)[keys[-1]] = value

    def fetch(self, container_name: str, keys: tuple):
        target = getattr(self, container_name)
        for key in keys:
            target = target[key]
        return target

    def record_fit(self, keys: tuple, params, quality: dict, fit_input: FitInput | None = None, domain=None) -> None:
        self.store("fits", keys, params)
        self.store("quality", keys, quality)
        if fit_input is not None:
            self.store("inputs", keys, fit_input)
        if domain is not None:
            self.store("domains", keys, domain)
