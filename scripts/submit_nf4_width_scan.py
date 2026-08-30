#!/usr/bin/env python3
"""Submit the fit4 prior-width/order scan to a SLURM HPCC cluster."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "nf4_fit4_single.sbatch"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def width_text(width: float) -> str:
    return f"{width:g}"


def width_tag(width: float) -> str:
    return width_text(width).replace(".", "p")


def required_environment() -> dict[str, str]:
    names = ("BETAFN_DATA_DIR", "BETAFN_OUTPUT_BASE", "BETAFN_VENV")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required environment variable(s): " + ", ".join(missing))

    values = {name: str(Path(os.environ[name]).expanduser().resolve()) for name in names}
    data_dir = Path(values["BETAFN_DATA_DIR"])
    venv_python = Path(values["BETAFN_VENV"]) / "bin" / "python"
    if not data_dir.is_dir():
        raise SystemExit(f"BETAFN_DATA_DIR is not a directory: {data_dir}")
    if not venv_python.is_file():
        raise SystemExit(f"Python virtual environment is invalid: {venv_python}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit independent fit4 jobs for each correction order and prior width."
    )
    parser.add_argument("--orders", type=positive_int, nargs="+", default=(3, 4))
    parser.add_argument(
        "--widths", type=positive_float, nargs="+", default=(1.0, 3.0, 10.0, 30.0)
    )
    parser.add_argument("--time", default="24:00:00", help="SLURM wall time")
    parser.add_argument("--cpus", type=positive_int, default=8)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--account", help="Optional SLURM account")
    parser.add_argument("--partition", help="Optional SLURM partition")
    parser.add_argument("--qos", help="Optional SLURM QoS")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = required_environment()

    if not SBATCH_SCRIPT.is_file():
        raise SystemExit(f"Missing SLURM job script: {SBATCH_SCRIPT}")
    if not args.dry_run and shutil.which("sbatch") is None:
        raise SystemExit("sbatch is not available; run this script on the HPCC login node")

    log_dir = REPO_ROOT / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(environment["BETAFN_OUTPUT_BASE"]).mkdir(parents=True, exist_ok=True)

    submitted: list[tuple[int, float, str]] = []
    for order in args.orders:
        for width in args.widths:
            job_name = f"nf4-o{order}-w{width_tag(width)}"
            exported = ",".join(
                (
                    "ALL",
                    f"FIT4_ORDER={order}",
                    f"FIT4_WIDTH={width_text(width)}",
                    f"BETAFN_REPO_DIR={REPO_ROOT}",
                    f"BETAFN_DATA_DIR={environment['BETAFN_DATA_DIR']}",
                    f"BETAFN_OUTPUT_BASE={environment['BETAFN_OUTPUT_BASE']}",
                    f"BETAFN_VENV={environment['BETAFN_VENV']}",
                )
            )
            command = [
                "sbatch",
                "--parsable",
                f"--job-name={job_name}",
                f"--time={args.time}",
                f"--cpus-per-task={args.cpus}",
                f"--mem={args.memory}",
                f"--export={exported}",
            ]
            if args.account:
                command.append(f"--account={args.account}")
            if args.partition:
                command.append(f"--partition={args.partition}")
            if args.qos:
                command.append(f"--qos={args.qos}")
            command.append(str(SBATCH_SCRIPT))

            if args.dry_run:
                print(" ".join(command))
                submitted.append((order, width, "dry-run"))
                continue

            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            job_id = result.stdout.strip().split(";", 1)[0]
            submitted.append((order, width, job_id))
            print(f"submitted order={order}, width={width_text(width)}: job {job_id}")

    print(f"Prepared {len(submitted)} independent fit4 job(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
