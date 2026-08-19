#!/bin/bash
#SBATCH --job-name=nf4-fit4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${FIT4_ORDER:?Submit with FIT4_ORDER set; use submit_nf4_jobs.sh.}"
: "${BETAFN_DATA_DIR:?Set BETAFN_DATA_DIR to the HPCC dataset directory.}"
: "${BETAFN_OUTPUT_BASE:?Set BETAFN_OUTPUT_BASE to an HPCC output directory.}"
: "${BETAFN_VENV:?Set BETAFN_VENV to the Python virtual-environment directory.}"

BETAFN_REPO_DIR="${BETAFN_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${BETAFN_REPO_DIR}"
mkdir -p slurm_logs "${BETAFN_OUTPUT_BASE}"

source "${BETAFN_VENV}/bin/activate"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/betafn-numba-${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

srun python -u scripts/run_nf4_analysis.py \
  --model fit4 \
  --fit4-order "${FIT4_ORDER}" \
  --data-dir "${BETAFN_DATA_DIR}" \
  --output-base "${BETAFN_OUTPUT_BASE}" \
  --binsize 15 \
  --use-gamma-method
