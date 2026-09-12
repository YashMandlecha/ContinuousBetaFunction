#!/bin/bash
#SBATCH --job-name=nf4-pt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${NF4_MODEL:?Set NF4_MODEL to fit5, fit6, or fit7.}"
: "${NF4_ORDER:=2}"
: "${BETAFN_DATA_DIR:?Set BETAFN_DATA_DIR to the HPCC dataset directory.}"
: "${BETAFN_OUTPUT_BASE:?Set BETAFN_OUTPUT_BASE to an HPCC output directory.}"
: "${BETAFN_VENV:?Set BETAFN_VENV to the Python virtual-environment directory.}"

if [[ "${NF4_MODEL}" != "fit5" && "${NF4_MODEL}" != "fit6" && "${NF4_MODEL}" != "fit7" ]]; then
  echo "NF4_MODEL must be fit5, fit6, or fit7; got ${NF4_MODEL}" >&2
  exit 2
fi
if [[ "${NF4_ORDER}" != "1" && "${NF4_ORDER}" != "2" ]]; then
  echo "NF4_ORDER must be 1 or 2; got ${NF4_ORDER}" >&2
  exit 2
fi

BETAFN_REPO_DIR="${BETAFN_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${BETAFN_REPO_DIR}"
mkdir -p slurm_logs "${BETAFN_OUTPUT_BASE}"

source "${BETAFN_VENV}/bin/activate"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/betafn-numba-${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

model_order_option="--${NF4_MODEL}-order"
srun python -u scripts/run_nf4_analysis.py \
  --model "${NF4_MODEL}" "${model_order_option}" "${NF4_ORDER}" \
  --data-dir "${BETAFN_DATA_DIR}" \
  --output-base "${BETAFN_OUTPUT_BASE}" \
  --binsize 15 \
  --use-gamma-method \
  --reviewed-weak-coupling
