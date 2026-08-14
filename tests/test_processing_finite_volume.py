"""Tests for betafn.processing.finite_volume.delta_finite_volume."""
from __future__ import annotations

import numpy as np
import pytest

from betafn.processing.finite_volume import delta_finite_volume


class TestDeltaFiniteVolume:
    def test_output_shape_matches_flow_times(self):
        flow_times = np.array([1.0, 2.0, 3.0, 4.0])
        delta = delta_finite_volume(flow_times, [24.0, 24.0, 24.0, 48.0])
        assert delta.shape == flow_times.shape

    def test_scalar_flow_time_returns_array(self):
        delta = delta_finite_volume(np.array([2.0]), [24.0, 24.0, 24.0, 48.0])
        assert delta.shape == (1,)

    def test_correction_is_small_for_large_volume(self):
        """For L >> sqrt(t), the FV correction should be tiny."""
        flow_times = np.linspace(2.0, 6.0, 20)
        delta = delta_finite_volume(flow_times, [1000.0, 1000.0, 1000.0, 2000.0])
        np.testing.assert_allclose(delta, 0.0, atol=1e-4)

    def test_larger_volume_gives_smaller_correction(self):
        """Increasing the volume should reduce |delta|."""
        flow_times = np.array([3.0])
        delta_small = delta_finite_volume(flow_times, [8.0, 8.0, 8.0, 16.0])
        delta_large = delta_finite_volume(flow_times, [24.0, 24.0, 24.0, 48.0])
        assert abs(delta_large[0]) < abs(delta_small[0]), \
            "Larger volume should give smaller FV correction"

    def test_returns_numpy_array(self):
        delta = delta_finite_volume([1.0, 2.0], [24.0, 24.0, 24.0, 48.0])
        assert isinstance(delta, np.ndarray)

    def test_accepts_list_input(self):
        delta = delta_finite_volume([2.0, 3.0], [24.0, 24.0, 24.0, 48.0])
        assert len(delta) == 2

    def test_standard_lattice_correction_magnitude(self):
        """For a 24^3×48 lattice at t=3, FV correction should be sub-percent."""
        flow_times = np.array([3.0])
        delta = delta_finite_volume(flow_times, [24.0, 24.0, 24.0, 48.0])
        assert abs(delta[0]) < 0.01, \
            f"Expected sub-percent FV correction, got {delta[0]:.4f}"
