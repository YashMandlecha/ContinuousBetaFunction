#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BETAFN_VENV:?Export BETAFN_VENV first.}"
exec "${BETAFN_VENV}/bin/python" "${script_dir}/submit_nf4_width_scan.py" \
  --orders 5 6 --no-priors "$@"
