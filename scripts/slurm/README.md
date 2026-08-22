# NF4 HPCC jobs

The runner reproduces the fit4, fit5, and fit6 notebook analysis and figures,
except for the intentionally omitted flow-time correlation-matrix figures.
It uses Matplotlib's non-interactive `Agg` backend and writes every result under
the selected output base.

The submitted jobs explicitly enable the reviewed weak-coupling plot families:
the narrow Figure-11-style integral match and the diagnostic that extends the
intermediate PT-preserving interpolants before taking the correlated continuum
limit. These plots are produced for fit4, fit5, and fit6. For a shorter debugging
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
packages from `requirements.txt`. External LaTeX is disabled by default for
cluster portability; pass `--latex` manually only when a complete TeX setup is
available.

Outputs are isolated as follows:

```text
<output-base>/fit4/order_<N>/...
<output-base>/fit5/pt_preserving_u3/...
<output-base>/fit6/pt_preserving_u3_u4/...
```

Each completed job writes `run_configuration.json`, all PNG/PDF figures, the
serialized diagonal/correlated continuum cases, the scan-summary CSV, and a
final `RUN_COMPLETE.json`. Absence of `RUN_COMPLETE.json` means the job did not
finish successfully.
