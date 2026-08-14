"""Shared fixtures and helpers for the betafn test suite.

All tests import betafn from src/ via the pythonpath setting in pyproject.toml.
No nf4 (or any other real-data) files are required; every test that needs
processed data generates it from synthetic Gaussian samples.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic .bin file builder
# ---------------------------------------------------------------------------

def make_bin_file(
    path: Path,
    flow_times: Sequence[float],
    n_configs: int = 200,
    *,
    ep_mean: float = 2.0,
    es_mean: float = 3.0,
    ec_mean: float = 2.5,
    sigma: float = 0.01,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> dict:
    """Write a synthetic ensemble .bin file and return the raw-data dict.

    The file contains white-noise Gaussian observables with no autocorrelation,
    so binning and the gamma method both give correct errors.

    Format expected by SetupBetaFunction._get:
        {
            "flow_times": ["0.5", "1.0", ...],   # strings
            "Ep": [[cfg1, cfg2, ...], [...], ...], # indexed [flow_time][config]
            "Es": ..., "Ec": ..., "Q": ...,
        }
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_ft = len(flow_times)

    data: dict = {
        "flow_times": [str(t) for t in flow_times],
        "Ep": [rng.normal(ep_mean, sigma, n_configs).tolist() for _ in range(n_ft)],
        "Es": [rng.normal(es_mean, sigma * 1.2, n_configs).tolist() for _ in range(n_ft)],
        "Ec": [rng.normal(ec_mean, sigma, n_configs).tolist() for _ in range(n_ft)],
        "Q":  [rng.integers(-3, 4, n_configs).astype(float).tolist() for _ in range(n_ft)],
    }
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    return data


def make_correlated_bin_file(
    path: Path,
    flow_times: Sequence[float],
    n_configs: int = 500,
    *,
    sigma_common: float = 0.05,
    sigma_indep: float = 0.001,
    rng: np.random.Generator | None = None,
    seed: int = 99,
) -> dict:
    """Write a .bin file where Ep and Es share a common fluctuation.

    Used to test that the combine-before-averaging fix gives correct (small)
    errors for combined observables; ignoring the cross-covariance would
    overestimate errors by ~ sqrt((5/3)^2 + (2/3)^2) / 1 ≈ 1.79×.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_ft = len(flow_times)

    Ep_data, Es_data, Ec_data, Q_data = [], [], [], []
    for _ in range(n_ft):
        common = rng.normal(0.0, sigma_common, n_configs)
        Ep_data.append((2.0 + common + rng.normal(0.0, sigma_indep, n_configs)).tolist())
        Es_data.append((3.0 + common + rng.normal(0.0, sigma_indep, n_configs)).tolist())
        Ec_data.append(rng.normal(2.5, sigma_common, n_configs).tolist())
        Q_data.append(rng.integers(-3, 4, n_configs).astype(float).tolist())

    data = {
        "flow_times": [str(t) for t in flow_times],
        "Ep": Ep_data,
        "Es": Es_data,
        "Ec": Ec_data,
        "Q":  Q_data,
    }
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    return data


# ---------------------------------------------------------------------------
# Shared dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_times():
    return [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


@pytest.fixture
def synthetic_data_dir(tmp_path, flow_times):
    """Directory with two synthetic ensembles at different bare couplings."""
    for beta, seed in [("7p00", 1), ("7p25", 2)]:
        fname = tmp_path / f"{beta}_l8l8l8t16_0p001_wilson.bin"
        make_bin_file(fname, flow_times, n_configs=200, seed=seed)
    return tmp_path


@pytest.fixture
def correlated_data_dir(tmp_path, flow_times):
    """Directory with one ensemble whose Ep and Es share common fluctuations."""
    fname = tmp_path / "7p00_l8l8l8t16_0p001_wilson.bin"
    make_correlated_bin_file(fname, flow_times)
    return tmp_path
