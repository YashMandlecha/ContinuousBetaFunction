"""Tests for betafn.base.catalog: FlowWindow, ProcessConfig, EnsembleKey, DatasetCatalog."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from betafn.base.catalog import (
    DatasetCatalog,
    EnsembleKey,
    FlowWindow,
    ProcessConfig,
)


# ---------------------------------------------------------------------------
# FlowWindow
# ---------------------------------------------------------------------------

class TestFlowWindow:
    def test_default_includes_all_times(self):
        w = FlowWindow()
        assert w.includes_time(0.0)
        assert w.includes_time(1e6)

    def test_minimum_excludes_earlier(self):
        w = FlowWindow(minimum_t=2.0)
        assert not w.includes_time(1.9)
        assert w.includes_time(2.0)

    def test_maximum_excludes_later(self):
        w = FlowWindow(maximum_t=5.0)
        assert w.includes_time(5.0)
        assert not w.includes_time(5.01)

    def test_accepts_string_float(self):
        w = FlowWindow(minimum_t=1.0, maximum_t=3.0)
        assert w.includes_time("2.5")
        assert not w.includes_time("0.5")

    def test_no_topological_filter_by_default(self):
        assert not FlowWindow().uses_topological_filter

    def test_topological_filter_detected(self):
        assert FlowWindow(minimum_q=-1.0, maximum_q=1.0).uses_topological_filter


# ---------------------------------------------------------------------------
# ProcessConfig
# ---------------------------------------------------------------------------

class TestProcessConfig:
    def test_default_correction(self):
        cfg = ProcessConfig()
        assert cfg.correction == "finite-volume"

    def test_immutable(self):
        cfg = ProcessConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.correction = "tln"  # type: ignore[misc]

    def test_custom_combine(self):
        combo = {"s": {"p": 5 / 3, "s": -2 / 3}}
        cfg = ProcessConfig(correction="tln", combine=combo)
        assert cfg.correction == "tln"
        assert cfg.combine is combo


# ---------------------------------------------------------------------------
# EnsembleKey
# ---------------------------------------------------------------------------

class TestEnsembleKey:
    @pytest.fixture
    def key(self):
        return EnsembleKey(coupling="7p25", volume="l24l24l24t48", mass="0p001")

    def test_beta_value(self, key):
        assert key.beta_value == pytest.approx(7.25)

    def test_mass_value(self, key):
        assert key.mass_value == pytest.approx(0.001)

    def test_dimensions(self, key):
        assert key.dimensions == (24, 24, 24, 48)

    def test_spatial_extent(self, key):
        assert key.spatial_extent == 24

    def test_temporal_extent(self, key):
        assert key.temporal_extent == 48

    def test_aspect_ratio(self, key):
        assert key.aspect_ratio == pytest.approx(2.0)

    def test_lattice_volume(self, key):
        assert key.lattice_volume == pytest.approx(24 ** 3 * 48)

    def test_sort_key_type(self, key):
        sk = key.sort_key
        assert len(sk) == 3 and all(isinstance(v, float) for v in sk)

    def test_str_repr(self, key):
        s = str(key)
        assert "7.25" in s
        assert "24" in s

    def test_ordering_by_beta(self):
        k1 = EnsembleKey("7p00", "l24l24l24t48", "0p001")
        k2 = EnsembleKey("7p25", "l24l24l24t48", "0p001")
        assert k1.sort_key < k2.sort_key


# ---------------------------------------------------------------------------
# DatasetCatalog
# ---------------------------------------------------------------------------

class TestDatasetCatalog:
    def test_scan_finds_correctly_named_files(self, tmp_path):
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        (tmp_path / "7p25_l24l24l24t48_0p0025_symanzik.bin").touch()
        (tmp_path / "not_a_dataset.bin").touch()

        cat = DatasetCatalog.scan(tmp_path)
        assert len(cat) == 2

    def test_flow_filter(self, tmp_path):
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        (tmp_path / "7p00_l20l20l20t40_0p001_symanzik.bin").touch()

        cat = DatasetCatalog.scan(tmp_path, flows=["wilson"])
        assert len(cat) == 1
        assert list(cat.flows) == ["wilson"]

    def test_couplings_property(self, tmp_path):
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        (tmp_path / "7p25_l24l24l24t48_0p001_wilson.bin").touch()

        cat = DatasetCatalog.scan(tmp_path)
        assert cat.couplings == ["7p00", "7p25"]

    def test_filter_by_coupling(self, tmp_path):
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        (tmp_path / "7p25_l24l24l24t48_0p001_wilson.bin").touch()

        cat = DatasetCatalog.scan(tmp_path).filter(couplings=["7p00"])
        assert len(cat) == 1
        assert cat.couplings == ["7p00"]

    def test_to_mapping_structure(self, tmp_path):
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        (tmp_path / "7p00_l20l20l20t40_0p001_symanzik.bin").touch()

        mapping = DatasetCatalog.scan(tmp_path).to_mapping()
        flows = mapping["7p00"]["l20l20l20t40"]["0p001"]
        assert set(flows) == {"wilson", "symanzik"}

    def test_no_duplicate_flows_in_mapping(self, tmp_path):
        # Two files for the same ensemble+flow would be a user error, but
        # to_mapping should still not produce duplicates.
        f = tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin"
        f.touch()
        mapping = DatasetCatalog.scan(tmp_path).to_mapping()
        flows = mapping["7p00"]["l20l20l20t40"]["0p001"]
        assert len(flows) == len(set(flows))

    def test_empty_directory(self, tmp_path):
        cat = DatasetCatalog.scan(tmp_path)
        assert len(cat) == 0

    def test_discover_dataset_classmethod(self, tmp_path):
        from betafn.setup import SetupBetaFunction
        (tmp_path / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
        mapping = SetupBetaFunction.discover_dataset(tmp_path, flows=["wilson"])
        assert "7p00" in mapping
