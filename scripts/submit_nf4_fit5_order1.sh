#!/bin/bash
set -euo pipefail

: "${BETAFN_DATA_DIR:?Export BETAFN_DATA_DIR first.}"
: "${BETAFN_OUTPUT_BASE:?Export BETAFN_OUTPUT_BASE first.}"
: "${BETAFN_VENV:?Export BETAFN_VENV first.}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${BETAFN_REPO_DIR:-$(cd "${script_dir}/.." && pwd)}"

sbatch \
  --job-name=nf4-fit5-order1 \
  --export="ALL,NF4_MODEL=fit5,NF4_ORDER=1,BETAFN_REPO_DIR=${repo_dir}" \
  "${repo_dir}/scripts/slurm/nf4_pt_preserving_single.sbatch"
