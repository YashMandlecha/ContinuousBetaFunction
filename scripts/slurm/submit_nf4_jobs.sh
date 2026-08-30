#!/bin/bash
set -euo pipefail

: "${BETAFN_DATA_DIR:?Export BETAFN_DATA_DIR first.}"
: "${BETAFN_OUTPUT_BASE:?Export BETAFN_OUTPUT_BASE first.}"
: "${BETAFN_VENV:?Export BETAFN_VENV first.}"

BETAFN_REPO_DIR="${BETAFN_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BETAFN_REPO_DIR
mkdir -p "${BETAFN_REPO_DIR}/slurm_logs" "${BETAFN_OUTPUT_BASE}"
cd "${BETAFN_REPO_DIR}"

# Each fit4 correction order is submitted as an independent SLURM job.
# Override the defaults, for example: FIT4_ORDERS="2 3 4 5" ./scripts/slurm/submit_nf4_jobs.sh
read -r -a fit4_orders <<< "${FIT4_ORDERS:-1 2 3 4}"
read -r -a fit4_widths <<< "${FIT4_WIDTHS:-10}"
for order in "${fit4_orders[@]}"; do
  for width in "${fit4_widths[@]}"; do
    width_tag="${width/./p}"
    sbatch \
      --job-name="nf4-fit4-o${order}-w${width_tag}" \
      --export="ALL,FIT4_ORDER=${order},FIT4_WIDTH=${width},BETAFN_REPO_DIR=${BETAFN_REPO_DIR}" \
      "${BETAFN_REPO_DIR}/scripts/slurm/nf4_fit4_single.sbatch"
  done
done

# Submit the two PT-preserving notebook equivalents as separate jobs too.
for model in fit5 fit6; do
  sbatch \
    --job-name="nf4-${model}" \
    --export="ALL,NF4_MODEL=${model},BETAFN_REPO_DIR=${BETAFN_REPO_DIR}" \
    "${BETAFN_REPO_DIR}/scripts/slurm/nf4_pt_preserving_single.sbatch"
done
