"""On-the-fly tree-level normalization (TLN) delta-factor via spectral decomposition.

Author: Akhil Chauhan (University of Illinois Urbana-Champaign)
Reference: https://link.springer.com/article/10.1007/JHEP09(2014)018

What is tree-level normalization?
----------------------------------
The tree-level normalization (TLN) factor (JHEP09(2014)018) corrects for gauge 
gauge zero modes due to periodic boundary conditions in the gauge fields and
tree-level discretization effects in the gradient flow coupling; it is calculated as:

    TLN(t) = 2 f(t) + f(t) · Σ_{n≠0} tr[exp(-t Sf) Inv(t) exp(-t Sf) Se]

where f(t) = 64π²t² / (3 Ns³ Nt), Sf is the flow-kernel matrix, Inv is the
inverse of the gauge-action matrix (both evaluated on the finite Fourier grid),
and Se is the energy-density kernel matrix.

Algorithm
---------
Two structural reductions make the naive O(Ns³ Nt) lattice sum practical:

  1. **Hypercubic symmetry.**  Every kernel (action, clover, gauge fixing)
     is invariant under reflections nμ → −nμ and permutations of the three
     spatial axes.  Summing over orbit representatives (unique sorted triples
     aₓ ≤ aᵧ ≤ a_z plus a temporal component a_t) weighted by their
     multiplicity reduces the number of sites processed by ~50–80×.

  2. **Spectral decomposition.**  Because Sf + G is real symmetric for each
     momentum site, it admits the eigendecomposition
     Sf + G = V diag(λ) Vᵀ.  A single eigendecomposition per site serves
     every flow time.  Writing A = Vᵀ Inv V and B = Vᵀ Se V:

         tr[exp(-t(Sf+G)) Inv exp(-t(Sf+G)) Se]
             = Σᵢⱼ Aᵢⱼ Bⱼᵢ exp(-t(λᵢ + λⱼ))

     so the entire lattice sum collapses to a pure sum of exponentials
     Σₖ cₖ exp(-t μₖ).  Because the grid is uniform with spacing ε,
     exp(-tⱼ μₖ) = exp(-j ε μₖ) = rₖʲ, meaning only one exp() call is
     needed *per term* (not per term per flow time).  A Numba-parallel
     loop over chunks of terms finishes in O(10⁻¹ – 10⁰) seconds for
     production lattice sizes after a one-time JIT warmup.

Observable / action codes
-------------------------
Observable (betafn code → energy kernel ce):
    'p'  Wilson plaquette    →  ce = 0
    's'  Symanzik-improved   →  ce = -1/12
    'c'  clover              →  ce = 999  (tags the clover kernel branch)

Gauge action (betafn gauge_action char → gauge kernel cg):
    's'  Symanzik  →  cg = -1/12
    'w'  Wilson    →  cg = 0

The gradient flow is always Wilson (cf = 0); this is the standard for all
current discretisations.  The TLN is independent of the flow discretisation
(C0p0, C13, etc.) — only the energy density observable and gauge action matter.

Validation
----------
The algorithm has been validated at machine precision (relative error ~10⁻¹⁴)
against the reference implementation

    qex/src/examples/tln.nim  (Curtis Taylor Peterson et al.)
    https://github.com/jcosborn/qex/blob/devel/src/examples/tln.nim

which was used to generate the pre-computed ``.tln`` tables previously
shipped with betafn.  Agreement was confirmed for 24³×48 lattices at the
plaquette, clover, and Symanzik-improved observables with Symanzik gauge
action.

Caching
-------
Results are cached in the module-level ``_TLN_CACHE`` dict keyed by
(Ns, Nt, cg, ce, ε).  The same geometry+discretisation appearing in multiple
ensembles within one Python session is therefore computed only once.
"""
from __future__ import annotations

import numpy as _numpy
from scipy.interpolate import CubicSpline as _CubicSpline
from numba import njit as _njit, prange as _prange, get_num_threads as _get_num_threads

from ..base.exceptions import BetaFunctionException


# ---------------------------------------------------------------------------
# Observable and gauge-action code maps
# ---------------------------------------------------------------------------

_OBSERVABLE_CE: dict[str, float | int] = {
    "p": 0,         # Wilson plaquette
    "s": -1 / 12,   # Symanzik-improved
    "c": 999,       # clover (special tag)
}

_GAUGE_ACTION_CG: dict[str, float | int] = {
    "s": -1 / 12,   # Symanzik
    "w": 0,         # Wilson
}

_D = _numpy.arange(4)


# ---------------------------------------------------------------------------
# Hypercubic orbit enumeration
# ---------------------------------------------------------------------------

def _momentum_orbits(Ns: int, Nt: int):
    """Representatives aₓ ≤ aᵧ ≤ a_z in [0, Ns//2], a_t in [0, Nt//2],
    zero mode dropped.  Returns (reps, multiplicities)."""
    sa = _numpy.arange(Ns // 2 + 1)
    ta = _numpy.arange(Nt // 2 + 1)
    ws = _numpy.where((sa == 0) | (2 * sa == Ns), 1, 2)
    wt = _numpy.where((ta == 0) | (2 * ta == Nt), 1, 2)

    A, B, C = _numpy.meshgrid(sa, sa, sa, indexing="ij")
    keep = (A <= B) & (B <= C)
    a, b, c = A[keep], B[keep], C[keep]

    eq_ab, eq_bc = a == b, b == c
    n_perm = _numpy.where(eq_ab & eq_bc, 1, _numpy.where(eq_ab | eq_bc, 3, 6))
    w_spatial = n_perm * ws[a] * ws[b] * ws[c]

    n_t = ta.size
    reps = _numpy.empty((a.size * n_t, 4), dtype=_numpy.int64)
    reps[:, 0] = _numpy.repeat(a, n_t)
    reps[:, 1] = _numpy.repeat(b, n_t)
    reps[:, 2] = _numpy.repeat(c, n_t)
    reps[:, 3] = _numpy.tile(ta, a.size)
    mult = (_numpy.repeat(w_spatial, n_t) * _numpy.tile(wt, a.size)).astype(_numpy.float64)

    reps, mult = reps[1:], mult[1:]   # drop zero mode
    assert mult.sum() == Ns**3 * Nt - 1, "orbit multiplicities do not close"
    return reps, mult


# ---------------------------------------------------------------------------
# Kernel matrices, vectorised over momentum sites
# ---------------------------------------------------------------------------

def _action_kernel(c, ph, ph2, ph_sq):
    M = -ph[:, :, None] * ph[:, None, :] * (1.0 - c * (ph2[:, :, None] + ph2[:, None, :]))
    M[:, _D, _D] += ph_sq[:, None] - c * _numpy.sum(ph2**2, axis=1)[:, None] - c * ph2 * ph_sq[:, None]
    return M


def _clover_kernel(pt, pt_sq, cos_half):
    M = -pt[:, :, None] * pt[:, None, :]
    M[:, _D, _D] += pt_sq[:, None]
    return M * cos_half[:, :, None] * cos_half[:, None, :]


def _kernel(code, ph, ph2, ph_sq, pt, pt_sq, cos_half):
    if code == 999:
        return _clover_kernel(pt, pt_sq, cos_half)
    return _action_kernel(code, ph, ph2, ph_sq)


# ---------------------------------------------------------------------------
# Spectral decomposition → (coefficient, exponent) pairs
# ---------------------------------------------------------------------------

def _spectral_terms(reps, mult, Ns, Nt, cf, cg, ce, alpha=1.0, block=200_000):
    """Reduce the lattice sum to (coef, mu) pairs with sum_k coef_k exp(-t μₖ)."""
    N = _numpy.array([Ns, Ns, Ns, Nt], dtype=_numpy.float64)
    coefs, mus = [], []

    for lo in range(0, reps.shape[0], block):
        idx = reps[lo:lo + block]
        w = mult[lo:lo + block]

        p = 2.0 * _numpy.pi * idx / N
        ph = 2.0 * _numpy.sin(p / 2.0)
        ph2 = ph**2
        ph_sq = ph2.sum(axis=1)
        pt = _numpy.sin(p)
        pt_sq = (pt**2).sum(axis=1)
        cos_half = _numpy.cos(p / 2.0)

        G = ph[:, :, None] * ph[:, None, :] / alpha        # gauge-fixing term
        Sf_G = _kernel(cf, ph, ph2, ph_sq, pt, pt_sq, cos_half) + G
        Sg_G = _kernel(cg, ph, ph2, ph_sq, pt, pt_sq, cos_half) + G
        Se   = _kernel(ce, ph, ph2, ph_sq, pt, pt_sq, cos_half)

        lam, V = _numpy.linalg.eigh(Sf_G)
        Vt = _numpy.swapaxes(V, 1, 2)
        A = Vt @ _numpy.linalg.solve(Sg_G, V)   # Vᵀ (Sg+G)⁻¹ V
        B = Vt @ (Se @ V)                        # Vᵀ Se V

        coefs.append((A * _numpy.swapaxes(B, 1, 2) * w[:, None, None]).ravel())
        mus.append((lam[:, :, None] + lam[:, None, :]).ravel())

    return _numpy.concatenate(coefs), _numpy.concatenate(mus)


def _collapse(mu, coef, tol=1e-11):
    """Merge terms sharing an exponent (shrinks the list >16× for Wilson flow)."""
    order = _numpy.argsort(mu, kind="stable")
    mu_s, c_s = mu[order], coef[order]
    is_new = _numpy.empty(mu_s.size, dtype=bool)
    is_new[0] = True
    _numpy.greater(_numpy.diff(mu_s), tol, out=is_new[1:])
    starts = _numpy.flatnonzero(is_new)
    return mu_s[starts], _numpy.add.reduceat(c_s, starts)


# ---------------------------------------------------------------------------
# Numba-parallel exponential sum
# ---------------------------------------------------------------------------

@_njit(parallel=True, fastmath=True, cache=True)
def _sum_exponentials(coef, mu, eps, n_t, n_chunks):
    acc = _numpy.zeros((n_chunks, n_t))
    K = coef.shape[0]
    for c in _prange(n_chunks):
        lo = K * c // n_chunks
        hi = K * (c + 1) // n_chunks
        row = acc[c]
        for k in range(lo, hi):
            r = _numpy.exp(-eps * mu[k])
            v = coef[k]
            for j in range(n_t):
                row[j] += v
                v *= r
                if v == 0.0:
                    break
    return acc


# ---------------------------------------------------------------------------
# Main computation (cached by geometry + discretisation)
# ---------------------------------------------------------------------------

_TLN_CACHE: dict[tuple, tuple] = {}


def _compute_tln(Ns: int, Nt: int, cg: float, ce: float, eps: float):
    """Compute the TLN factor array on a uniform flow-time grid.

    Always uses Wilson gradient flow (cf = 0).  Returns (flowtimes, tln_values).
    """
    n_t = int(_numpy.ceil(Ns**2 / 32 / eps) + 1)
    flowtimes = _numpy.arange(n_t) * eps

    reps, mult = _momentum_orbits(Ns, Nt)
    coef, mu = _spectral_terms(reps, mult, Ns, Nt, cf=0, cg=cg, ce=ce)
    mu, coef = _collapse(mu, coef)

    lattice_sum = _sum_exponentials(coef, mu, eps, n_t,
                                    max(1, _get_num_threads())).sum(axis=0)
    otherfactor = 64.0 * _numpy.pi**2 * flowtimes**2 / (3.0 * Ns**3 * Nt)
    tln = 2.0 * otherfactor + otherfactor * lattice_sum
    return flowtimes, tln


def _get_tln_cached(Ns: int, Nt: int, cg: float, ce: float, eps: float = 0.01):
    """Return (flowtimes, tln_values), computing only on the first call per key."""
    key = (Ns, Nt, cg, ce, eps)
    if key not in _TLN_CACHE:
        _TLN_CACHE[key] = _compute_tln(Ns, Nt, cg=cg, ce=ce, eps=eps)
    return _TLN_CACHE[key]


# ---------------------------------------------------------------------------
# Public interface called by setup.py
# ---------------------------------------------------------------------------

def delta_tln(
    flow_times: _numpy.ndarray,
    observable: str,
    volume: str,
    gauge_action: str,
    eps: float = 0.01,
) -> _numpy.ndarray:
    """Compute the TLN correction Δ(t) = TLN(t) − 1 on-the-fly.

    Parameters
    ----------
    flow_times : array_like
        Flow times at which to evaluate the correction.
    observable : str
        Single-character betafn observable code: 'p' (plaquette), 's'
        (Symanzik-improved), or 'c' (clover).
    volume : str
        Volume label, e.g. 'l24l24l24t48'.  Ns and Nt are parsed from it.
    gauge_action : str
        Single-character gauge-action code: 's' (Symanzik) or 'w' (Wilson).
    eps : float
        Flow-time grid spacing for the internal computation.  Default 0.01 is
        fine enough for cubic-spline interpolation at any analysis window.

    Returns
    -------
    delta : ndarray
        Δ(t) = TLN(t) − 1 at each requested flow time.
    """
    if observable not in _OBSERVABLE_CE:
        raise BetaFunctionException(
            f"Observable '{observable}' not supported for TLN correction.",
            "Supported: " + ", ".join(sorted(_OBSERVABLE_CE)),
        )
    if gauge_action not in _GAUGE_ACTION_CG:
        raise BetaFunctionException(
            f"Gauge action '{gauge_action}' not supported for TLN correction.",
            "Supported: " + ", ".join(sorted(_GAUGE_ACTION_CG)),
        )

    dims = [int(d) for d in volume.replace("t", "l").split("l")[1:] if d]
    Ns, Nt = dims[0], dims[-1]

    cg = _GAUGE_ACTION_CG[gauge_action]
    ce = _OBSERVABLE_CE[observable]

    t_grid, tln_values = _get_tln_cached(Ns, Nt, cg=cg, ce=ce, eps=eps)

    spline = _CubicSpline(t_grid, tln_values)
    return _numpy.array([spline(float(t)) for t in flow_times]) - 1.0


def clear_tln_cache() -> None:
    """Evict all cached TLN computations (useful in testing or memory-constrained runs)."""
    _TLN_CACHE.clear()
