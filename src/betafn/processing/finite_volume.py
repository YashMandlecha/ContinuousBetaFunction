"""Finite-volume delta-factor correction for the gradient-flow coupling."""
from __future__ import annotations

import numpy as _numpy


def delta_finite_volume(flow_times: _numpy.ndarray, volume_dims: list[float]) -> _numpy.ndarray:
    """Finite-volume correction Δ(t, L) such that g²_GF,∞ = g²_GF / (1 + Δ).

    Implements the semi-analytical formula of Fodor et al. / Ramos-Sint, which
    decomposes the correction into an analytic piece Δ_a and an exponentially
    suppressed finite-volume artefact Δ_e.  The result is returned as an array
    of the same length as *flow_times*, ready to be passed to _norm().

    Parameters
    ----------
    flow_times : array_like
        Flow times t at which the coupling is measured (physical units, t = a²·n).
    volume_dims : list[float]
        Lattice extents [L₁, L₂, …, Lₙ] in the same units as flow_times.

    Returns
    -------
    delta : ndarray
        Correction Δ(t, L) = g²_lat / g²_∞ − 1 at each flow time.
    """
    flow_times = _numpy.asarray(flow_times, dtype=float)
    delta_a = -64.0 * _numpy.pi * _numpy.pi / 3.0
    delta_e = _numpy.ones_like(flow_times)
    for dim in volume_dims:
        ratio = dim * dim / flow_times
        delta_a /= _numpy.sqrt(ratio)
        delta_e *= 1.0 + 2.0 * _numpy.exp(-0.125 * ratio) + 2.0 * _numpy.exp(-0.5 * ratio)
    return delta_a + delta_e - 1.0
