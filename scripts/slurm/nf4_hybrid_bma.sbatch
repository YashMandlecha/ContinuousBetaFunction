#!/bin/bash
#SBATCH --job-name=nf4-hybrid-bma
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${BETAFN_DATA_DIR:?Set BETAFN_DATA_DIR to the HPCC dataset directory.}"
: "${BETAFN_OUTPUT_BASE:?Set BETAFN_OUTPUT_BASE to the HPCC output directory.}"
: "${BETAFN_VENV:?Set BETAFN_VENV to the Python virtual-environment directory.}"
: "${BETAFN_HYBRID_THINNING:=0}"
: "${BETAFN_HYBRID_WORKERS:=${SLURM_CPUS_PER_TASK}}"
: "${BETAFN_CONTINUUM_WORKERS:=${SLURM_CPUS_PER_TASK}}"

if [[ "${BETAFN_HYBRID_THINNING}" != "0" && "${BETAFN_HYBRID_THINNING}" != "1" ]]; then
  echo "BETAFN_HYBRID_THINNING must be 0 or 1; got ${BETAFN_HYBRID_THINNING}" >&2
  exit 2
fi
if ! [[ "${BETAFN_HYBRID_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BETAFN_HYBRID_WORKERS must be a positive integer; got ${BETAFN_HYBRID_WORKERS}" >&2
  exit 2
fi
if (( BETAFN_HYBRID_WORKERS > SLURM_CPUS_PER_TASK )); then
  echo "BETAFN_HYBRID_WORKERS cannot exceed SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}" >&2
  exit 2
fi
if ! [[ "${BETAFN_CONTINUUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BETAFN_CONTINUUM_WORKERS must be a positive integer; got ${BETAFN_CONTINUUM_WORKERS}" >&2
  exit 2
fi
if (( BETAFN_CONTINUUM_WORKERS > SLURM_CPUS_PER_TASK )); then
  echo "BETAFN_CONTINUUM_WORKERS cannot exceed SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}" >&2
  exit 2
fi

BETAFN_REPO_DIR="${BETAFN_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${BETAFN_REPO_DIR}"
mkdir -p slurm_logs "${BETAFN_OUTPUT_BASE}"

source "${BETAFN_VENV}/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
# Operator/flow-time/model tasks use process-level parallelism. Keep every worker's BLAS,
# OpenMP and numexpr pools single-threaded to prevent multiplicative CPU
# oversubscription. More candidate models are queued automatically.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/betafn-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${NUMBA_CACHE_DIR}"

# Fail before allocating the expensive fit if the code, selected model
# ensemble, data path, or repository-external blinding secret is unavailable.
python -u scripts/run_nf4_hybrid_bma.py \
  --validate-only \
  --data-dir "${BETAFN_DATA_DIR}" \
  --hybrid-workers "${BETAFN_HYBRID_WORKERS}" \
  --continuum-workers "${BETAFN_CONTINUUM_WORKERS}" \
  --output-base "${BETAFN_OUTPUT_BASE}"
python -c 'import sys; sys.path.insert(0, "src"); from betafn.blinding import load_lambda_blinding_factor; load_lambda_blinding_factor()'
test -d "${BETAFN_DATA_DIR}"

analysis_args=(
  --data-dir "${BETAFN_DATA_DIR}"
  --output-base "${BETAFN_OUTPUT_BASE}"
  --hybrid-workers "${BETAFN_HYBRID_WORKERS}"
  --continuum-workers "${BETAFN_CONTINUUM_WORKERS}"
)
if [[ "${BETAFN_HYBRID_THINNING}" == "1" ]]; then
  analysis_args+=(--hybrid-thinning)
else
  analysis_args+=(--no-hybrid-thinning)
fi

srun python -u scripts/run_nf4_hybrid_bma.py "${analysis_args[@]}"
