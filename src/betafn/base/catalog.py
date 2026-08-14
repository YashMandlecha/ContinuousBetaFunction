"""Ensemble file catalog: keys, file references, and directory scanning."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re as _re

import numpy as _numpy

from .exceptions import BetaFunctionException  # noqa: F401 (re-exported for convenience)


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
    combine: object = None  # Mapping[str, Mapping[str, float]] | None


@dataclass(frozen=True)
class ProcessHooks:
    get: Callable
    average: Callable
    preprocess: Callable


@dataclass(frozen=True)
class EnsembleKey:
    """Physics-aware identity of one ensemble: bare coupling, volume, and mass."""

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
    """Immutable, queryable inventory of ensemble data files."""

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
