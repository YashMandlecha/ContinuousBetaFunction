#!/bin/bash
#SBATCH --job-name=nf4-pt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${NF4_MODEL:?Set NF4_MODEL to a supported fit model.}"
: "${NF4_ORDER:=2}"
: "${BETAFN_DATA_DIR:?Set BETAFN_DATA_DIR to the HPCC dataset directory.}"
: "${BETAFN_OUTPUT_BASE:?Set BETAFN_OUTPUT_BASE to an HPCC output directory.}"
: "${BETAFN_VENV:?Set BETAFN_VENV to the Python virtual-environment directory.}"
: "${BETAFN_CONTINUUM_THINNING:=0}"
: "${BETAFN_THINNING_ONLY:=0}"

analysis_mode_args=()
if [[ "${BETAFN_THINNING_ONLY}" == "1" ]]; then
  thinning_arg="--continuum-thinning"
  analysis_mode_args+=(--thinning-only)
elif [[ "${BETAFN_THINNING_ONLY}" != "0" ]]; then
  echo "BETAFN_THINNING_ONLY must be 0 or 1; got ${BETAFN_THINNING_ONLY}" >&2
  exit 2
elif [[ "${BETAFN_CONTINUUM_THINNING}" == "1" ]]; then
  thinning_arg="--continuum-thinning"
elif [[ "${BETAFN_CONTINUUM_THINNING}" == "0" ]]; then
  thinning_arg="--no-continuum-thinning"
else
  echo "BETAFN_CONTINUUM_THINNING must be 0 or 1; got ${BETAFN_CONTINUUM_THINNING}" >&2
  exit 2
fi

if [[ ! "${NF4_MODEL}" =~ ^fit(1|2|3|5|6|7|8|9|10|11|12)$ ]]; then
  echo "NF4_MODEL must be fit1--fit3 or fit5--fit12; got ${NF4_MODEL}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" =~ ^fit(1|2|3)$ && "${NF4_ORDER}" != "3" && "${NF4_ORDER}" != "4" && "${NF4_ORDER}" != "5" ]]; then
  echo "NF4_ORDER must be 3, 4, or 5 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" =~ ^fit(5|7)$ && "${NF4_ORDER}" != "1" && "${NF4_ORDER}" != "2" ]]; then
  echo "NF4_ORDER must be 1 or 2 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" == "fit6" && "${NF4_ORDER}" != "1" && "${NF4_ORDER}" != "2" && "${NF4_ORDER}" != "3" ]]; then
  echo "NF4_ORDER must be 1, 2, or 3 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" == "fit8" && "${NF4_ORDER}" != "3" && "${NF4_ORDER}" != "4" ]]; then
  echo "NF4_ORDER must be 3 or 4 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" =~ ^fit(9|10)$ && "${NF4_ORDER}" != "3" && "${NF4_ORDER}" != "4" && "${NF4_ORDER}" != "5" ]]; then
  echo "NF4_ORDER must be 3, 4, or 5 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
  exit 2
fi
if [[ "${NF4_MODEL}" =~ ^fit(11|12)$ && "${NF4_ORDER}" != "3" && "${NF4_ORDER}" != "4" && "${NF4_ORDER}" != "5" ]]; then
  echo "NF4_ORDER must be 3, 4, or 5 for ${NF4_MODEL}; got ${NF4_ORDER}" >&2
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

# Fail before the expensive analysis if the repo-external Lambda blinding
# secret is absent or invalid.  The factor itself is never printed.
python -c 'import sys; sys.path.insert(0, "src"); from betafn.blinding import load_lambda_blinding_factor; load_lambda_blinding_factor()'

model_order_option="--${NF4_MODEL}-order"
srun python -u scripts/run_nf4_analysis.py \
  --model "${NF4_MODEL}" "${model_order_option}" "${NF4_ORDER}" \
  --data-dir "${BETAFN_DATA_DIR}" \
  --output-base "${BETAFN_OUTPUT_BASE}" \
  --binsize 15 \
  --use-gamma-method \
  --reviewed-weak-coupling \
  "${thinning_arg}" \
  "${analysis_mode_args[@]}"
