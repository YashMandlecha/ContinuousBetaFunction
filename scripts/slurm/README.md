# NF4 HPCC jobs

The runner reproduces the fit4 through fit12 notebook analysis and figures,
except for the intentionally omitted flow-time correlation-matrix figures.
It uses Matplotlib's non-interactive `Agg` backend and writes every result under
the selected output base.

The submitted jobs explicitly enable the reviewed weak-coupling plot families:
the narrow Figure-11-style integral match and the diagnostic that extends the
intermediate PT-preserving interpolants before taking the correlated continuum
limit. These plots are produced for all applicable fit variants. For a shorter debugging
run they can be disabled with `--no-reviewed-weak-coupling` when invoking the
Python runner directly.

Before submission on MSU HPCC:

```bash
export BETAFN_REPO_DIR=/path/to/ContinuousBetaFunction
export BETAFN_DATA_DIR=/path/to/New\ Four
export BETAFN_OUTPUT_BASE=/path/to/nf4_outputs
export BETAFN_VENV=/path/to/python/venv
cd "$BETAFN_REPO_DIR"
FIT4_ORDERS="1 2 3 4" ./scripts/slurm/submit_nf4_jobs.sh
```

Add the appropriate MSU account and partition directives to the `.sbatch`
templates if required by your allocation. The environment must contain the
packages from `requirements.txt`. Submitted jobs use Matplotlib's internal math
renderer and DejaVu Serif, so they do not depend on a TeX installation or the
Computer Modern fonts on HPCC. External LaTeX is only enabled when `--latex`
is passed explicitly to the Python runner.

Outputs are isolated as follows:

```text
<output-base>/fit4/order_<N>/...
<output-base>/fit5/pt_preserving_u3/...
<output-base>/fit6/pt_preserving_u3_u4/...
<output-base>/fit12/order_4_joint_a0_continuum_nopriors/...
```

Each completed job writes `run_configuration.json`, all PNG/PDF figures, the
serialized diagonal/correlated continuum cases, the scan-summary CSV, the five
flow-time thinning studies under `continuum_thinning_scan/`, and a final
`RUN_COMPLETE.json`. Absence of `RUN_COMPLETE.json` means the job did not finish
successfully.

Fit12 uses
`beta(x,z) = z*A0 + beta_PT3(x) * [1 + sum(c_n*u^n)]`,
where `z=a^2/t`. Its finite-spacing additive term is free but vanishes in the
continuum, while the multiplicative intercept `c0=1` and the coefficients
`c_n` are shared at every `z`. There are no additional `d_n` cutoff terms.

Both shared Slurm templates request 48 hours of wall time per submitted order
because every order runs 15 flow-time windows and the five-spacing thinning
study by default.
