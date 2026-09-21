"""Prospective blinding helpers for reported Lambda-parameter results.

The blinding factor is deliberately stored outside the repository.  Analysis
code must fail closed when the secret is missing so that an apparently final,
unblinded Lambda plot cannot be produced by accident.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


BLINDING_FILE_ENV = "BETAFN_LAMBDA_BLINDING_FILE"
DEFAULT_BLINDING_FILE = (
    Path.home() / ".config" / "continuous-betafn" / "lambda_blinding_factor"
)
LOWER_FACTOR_BOUND = 0.6
UPPER_FACTOR_BOUND = 1.4


def lambda_blinding_file() -> Path:
    """Return the repo-external path containing the Lambda blinding factor."""
    configured = os.environ.get(BLINDING_FILE_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_BLINDING_FILE


def load_lambda_blinding_factor() -> float:
    """Load and validate the secret factor without displaying it."""
    path = lambda_blinding_file()
    try:
        factor = float(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Lambda blinding secret is missing. Have the blinding custodian "
            f"install it at {path} or set {BLINDING_FILE_ENV}. Refusing to "
            "produce unblinded Lambda output."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Lambda blinding secret is not a valid number") from exc

    if not LOWER_FACTOR_BOUND <= factor <= UPPER_FACTOR_BOUND:
        raise RuntimeError("Lambda blinding factor lies outside its allowed range")
    if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Lambda blinding factor must not equal one")
    return factor


def blind_lambda_estimate(estimate: dict) -> dict:
    """Return an estimate whose reported GF and MSbar values are blinded.

    A single positive factor multiplies both schemes and all three uncertainty
    curves.  Consequently the MSbar/GF conversion ratio, relative errors, and
    comparisons among fit choices are preserved.  The factor itself is never
    added to the returned object.
    """
    factor = load_lambda_blinding_factor()
    blinded = dict(estimate)
    for key in ("lambda_gf_over_mu", "lambda_msbar_over_mu"):
        blinded[key] = {
            label: factor * float(value)
            for label, value in estimate[key].items()
        }
    blinded["blinded"] = True
    return blinded


def add_blinded_watermark(fig) -> None:
    """Place a prominent diagonal blinding mark on a Matplotlib figure."""
    fig.text(
        0.5,
        0.5,
        "BLINDED",
        ha="center",
        va="center",
        rotation=30,
        fontsize=64,
        fontweight="bold",
        color="crimson",
        alpha=0.16,
        zorder=1000,
    )
