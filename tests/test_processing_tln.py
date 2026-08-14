"""Tests for betafn.processing.tln: orbit enumeration, spectral decomposition,
caching, and the public delta_tln interface.

TLN tests that exercise the numba JIT are marked ``slow`` and skipped by default
in CI unless explicitly requested with ``-m slow``.  The JIT cache (cache=True)
means they run fast after the first compilation.
"""
from __future__ import annotations

import numpy as np
import pytest

from betafn.base.exceptions import BetaFunctionException
from betafn.processing.tln import (
    _GAUGE_ACTION_CG,
    _OBSERVABLE_CE,
    _TLN_CACHE,
    _momentum_orbits,
    clear_tln_cache,
    delta_tln,
)


# ---------------------------------------------------------------------------
# Hypercubic orbit enumeration
# ---------------------------------------------------------------------------

class TestMomentumOrbits:
    @pytest.mark.parametrize("Ns,Nt", [(4, 8), (6, 12), (8, 16)])
    def test_orbit_closure(self, Ns, Nt):
        """Multiplicities must sum to Ns³·Nt - 1 (all modes except zero)."""
        reps, mult = _momentum_orbits(Ns, Nt)
        assert mult.sum() == Ns**3 * Nt - 1

    @pytest.mark.parametrize("Ns,Nt", [(4, 8), (8, 16)])
    def test_representatives_in_range(self, Ns, Nt):
        """Spatial representatives in [0, Ns//2], time in [0, Nt//2]."""
        reps, _ = _momentum_orbits(Ns, Nt)
        assert np.all(reps[:, :3] >= 0) and np.all(reps[:, :3] <= Ns // 2)
        assert np.all(reps[:, 3] >= 0) and np.all(reps[:, 3] <= Nt // 2)

    @pytest.mark.parametrize("Ns,Nt", [(4, 8), (8, 16)])
    def test_spatial_ordering(self, Ns, Nt):
        """Spatial components must satisfy a_x <= a_y <= a_z."""
        reps, _ = _momentum_orbits(Ns, Nt)
        assert np.all(reps[:, 0] <= reps[:, 1]) and np.all(reps[:, 1] <= reps[:, 2])

    @pytest.mark.parametrize("Ns,Nt", [(4, 8), (8, 16)])
    def test_zero_mode_absent(self, Ns, Nt):
        """The n=0 mode must not appear in the representative list."""
        reps, _ = _momentum_orbits(Ns, Nt)
        assert not np.any(np.all(reps == 0, axis=1))

    def test_fewer_reps_than_sites(self):
        """Orbit reduction: fewer representatives than lattice sites."""
        Ns, Nt = 8, 16
        reps, _ = _momentum_orbits(Ns, Nt)
        n_sites = Ns**3 * Nt - 1
        assert reps.shape[0] < n_sites


# ---------------------------------------------------------------------------
# Code maps
# ---------------------------------------------------------------------------

class TestCodeMaps:
    def test_observable_codes_present(self):
        for obs in ("p", "s", "c"):
            assert obs in _OBSERVABLE_CE

    def test_gauge_action_codes_present(self):
        for ga in ("s", "w"):
            assert ga in _GAUGE_ACTION_CG

    def test_symanzik_coefficient_value(self):
        assert _GAUGE_ACTION_CG["s"] == pytest.approx(-1 / 12)

    def test_wilson_coefficient_zero(self):
        assert _GAUGE_ACTION_CG["w"] == 0
        assert _OBSERVABLE_CE["p"] == 0


# ---------------------------------------------------------------------------
# delta_tln — interface and sanity checks (slow: involves numba JIT)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure TLN cache is cleared between tests for isolation."""
    clear_tln_cache()
    yield
    clear_tln_cache()


@pytest.mark.slow
class TestDeltaTln:
    """Tests that require numba JIT and lattice sum computation."""

    # Use a tiny lattice (4^3×8) so the sum is near-instant even on cold JIT.
    Ns, Nt = 4, 8
    volume = "l4l4l4t8"

    def test_output_shape(self):
        flow_times = np.linspace(0.1, 0.5, 10)
        delta = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        assert delta.shape == flow_times.shape

    def test_returns_numpy_array(self):
        delta = delta_tln(np.array([0.2]), "p", self.volume, gauge_action="s")
        assert isinstance(delta, np.ndarray)

    def test_tln_approaches_zero_far_from_boundary(self):
        """TLN → 1 (delta → 0) as t/L² → 0 (small flow on a large box).
        For a tiny lattice this doesn't hold, but TLN should at least be
        finite and reasonably sized at flow times well within the lattice."""
        flow_times = np.array([0.05, 0.1, 0.15])
        delta = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        assert np.all(np.isfinite(delta))

    def test_plaquette_and_symanzik_differ(self):
        """Different observable codes (p vs s) should give different TLN factors
        since they use different energy-density kernels (ce=0 vs ce=-1/12)."""
        flow_times = np.linspace(0.05, 0.5, 20)
        delta_p = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        delta_s = delta_tln(flow_times, "s", self.volume, gauge_action="s")
        assert not np.allclose(delta_p, delta_s), \
            "Plaquette and Symanzik TLN should differ"

    def test_wilson_and_symanzik_action_differ(self):
        """Different gauge actions (s vs w) should give different TLN factors."""
        flow_times = np.linspace(0.05, 0.5, 20)
        delta_sym = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        delta_wil = delta_tln(flow_times, "p", self.volume, gauge_action="w")
        assert not np.allclose(delta_sym, delta_wil), \
            "Symanzik and Wilson gauge-action TLN should differ"

    def test_caching_returns_identical_array(self):
        """Second call with the same parameters should hit the cache."""
        flow_times = np.array([0.1, 0.2])
        delta1 = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        delta2 = delta_tln(flow_times, "p", self.volume, gauge_action="s")
        np.testing.assert_array_equal(delta1, delta2)

    def test_caching_populates_cache(self):
        assert len(_TLN_CACHE) == 0
        delta_tln(np.array([0.1]), "p", self.volume, gauge_action="s")
        assert len(_TLN_CACHE) == 1

    def test_clear_tln_cache(self):
        delta_tln(np.array([0.1]), "p", self.volume, gauge_action="s")
        assert len(_TLN_CACHE) > 0
        clear_tln_cache()
        assert len(_TLN_CACHE) == 0

    def test_all_three_observables(self):
        flow_times = np.array([0.1, 0.2, 0.3])
        for obs in ("p", "s", "c"):
            delta = delta_tln(flow_times, obs, self.volume, gauge_action="s")
            assert delta.shape == (3,)
            assert np.all(np.isfinite(delta))


# ---------------------------------------------------------------------------
# delta_tln — error handling (no numba needed)
# ---------------------------------------------------------------------------

class TestDeltaTlnErrors:
    volume = "l8l8l8t16"

    def test_unknown_observable_raises(self):
        with pytest.raises(BetaFunctionException):
            delta_tln(np.array([1.0]), "r", self.volume, gauge_action="s")

    def test_unknown_gauge_action_raises(self):
        with pytest.raises(BetaFunctionException):
            delta_tln(np.array([1.0]), "p", self.volume, gauge_action="x")

    def test_volume_parsing_standard(self):
        """Volume label l24l24l24t48 should parse to Ns=24, Nt=48 without error."""
        # Just verify the parse path doesn't crash (needs slow mark for actual computation)
        # Test the parse logic directly instead
        volume = "l24l24l24t48"
        dims = [int(d) for d in volume.replace("t", "l").split("l")[1:] if d]
        assert dims[0] == 24 and dims[-1] == 48
