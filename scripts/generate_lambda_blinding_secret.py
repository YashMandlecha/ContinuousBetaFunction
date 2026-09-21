#!/usr/bin/env python3
"""Create a repo-external Lambda blinding secret without displaying it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets


DEFAULT_OUTPUT = (
    Path.home() / ".config" / "continuous-betafn" / "lambda_blinding_factor"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Repo-external secret path (must not already exist).",
    )
    return parser.parse_args()


def generate_factor() -> float:
    # Draw uniformly from millionth-spaced values strictly inside (0.6, 1.4),
    # excluding exactly one.  SystemRandom is supplied by the operating system.
    while True:
        numerator = 600_001 + secrets.randbelow(799_999)
        if numerator != 1_000_000:
            return numerator / 1_000_000.0


def main() -> int:
    output = parse_args().output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{generate_factor():.6f}\n")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(f"Created Lambda blinding secret at {output}; value not displayed.")
    print("Give custody of this file to a non-analyst before inspecting results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
