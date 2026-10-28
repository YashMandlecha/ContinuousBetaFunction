#!/bin/bash
set -euo pipefail

: "${BETAFN_DATA_DIR:?Export BETAFN_DATA_DIR first.}"
: "${BETAFN_OUTPUT_BASE:?Export BETAFN_OUTPUT_BASE first.}"
: "${BETAFN_VENV:?Export BETAFN_VENV first.}"
: "${BETAFN_HYBRID_THINNING:=0}"
: "${BETAFN_HYBRID_WORKERS:=12}"
: "${BETAFN_CONTINUUM_WORKERS:=12}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${BETAFN_REPO_DIR:-$(cd "${script_dir}/.." && pwd)}"

sbatch \
  --job-name=nf4-hybrid-bma \
  --export="ALL,BETAFN_REPO_DIR=${repo_dir},BETAFN_HYBRID_THINNING=${BETAFN_HYBRID_THINNING},BETAFN_HYBRID_WORKERS=${BETAFN_HYBRID_WORKERS},BETAFN_CONTINUUM_WORKERS=${BETAFN_CONTINUUM_WORKERS}" \
  "${repo_dir}/scripts/slurm/nf4_hybrid_bma.sbatch"
