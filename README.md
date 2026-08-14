# ContinuousBetaFunction

Python analysis code for extracting a continuous renormalization group beta-function from gradient-flow lattice data.

Example under `examples/nf12` (based on [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.114507](Phys. Rev. D 109, 114507))

References:
- [https://doi.org/10.1051/epjconf/201817508027](EPJ Web of Conferences 175, 08027 (2018))
- [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034514](Phys. Rev. D 101, 034514)
- [https://pos.sissa.it/363/094](PoS(LATTICE2019)094)
- [https://pos.sissa.it/396/321](PoS(LATTICE2021)321)
- [https://pos.sissa.it/430/043](PoSLATTICE2022(2023)043)
- [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.114504](Phys. Rev. D 107, 114504)
- [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.L071503](Phys. Rev. D 108, L071503)
- [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.014502](Phys. Rev. D 108, 014502)
- [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.114507](Phys. Rev. D 109, 114507)

## Analysis Workflow

First start off by preparing the data that is to be analyzed (hardest part):

```python
import betafn

# Put together catalog of dataset to be analyzed
CVS = {
    '20p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '18p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '16p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '14p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '12p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '10p0': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '9p00': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
    '8p50': ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96'),
}
catalog = betafn.BetaFunction.catalog('data', flows=['wilson']).filter(
    predicate=lambda entry: entry.key.coupling in CVS and entry.key.volume in CVS[entry.key.coupling]
)
print(catalog.summary())

# Construct beta-function object
bf = betafn.BetaFunction(nf = 4)

# Prepare analysis configuration
config = betafn.AnalysisConfig(
    data_path = 'data',           # <-+- data catalog: tells BetaFunction object
    couplings = tuple(CVS),       #   |  where to grab the data that it needs
    volumes = CVS,                # <-+
    use_gamma_method = True,      # Apply Gamma-method for expectations/errors
    gamma_window_factor = 3.0,    # Madras-Sokal windowing factor
    process_window = (2.0, 10.0), # t/a^2 over which to save processed data
    interpolation = bf.perturbative_interpolation(
        loops            = 3,
        correction_order = 4,
        free_intercept   = True,
        intercept_width  = 0.2,
    ), # PT-based interpolation beta_{3-loop GF}(g^2)*(c0 + sum_i c_i g^{2i})
    continuum_window = (4.0, 6.0), # Flow time range for continuumm extrapolation
    g2_grid = (0.0, 15.8, 0.2),    # Fixed g^2 over which extrapolation to be done
    flows = ('wilson',),           # Flow type
    observables = ('p', 's', 'c'), # Operator types
    combine = {'s': {'p': 5./3., 's': -2./3.}}, # Observable combination
    correction = 'tree-level-normalization',    # zero-mode & tree-level disc. correction
    verbosity = 1,
)
```

Run the analysis for each stage (`config` mutable):

```python
# Process dataset & create report of processed data
bf.run_processing(config)
reference_time = bf.data['12p0']['l32l32l32t64']['0p00']['wilson']['flow_times'][-1]
print(bf.data_report(flow_time=reference_time))

# Run chiral (amf -> 0) extrapolation (not relevant to this example)
bf.run_chiral(config)
print(bf.stage_summary('chiral'))

# Infinite volume (1/V -> 0) extrapolation 
bf.run_infinite_volume(config)  # V -> infinity
print(bf.stage_summary('infinite_volume'))

# Intermediate interpolation (needed for continuum extrapolation)
bf.run_interpolation(config)
print(bf.stage_summary('interpolation'))

# Continuum (a^2/t -> 0) extrapolation (fixed g^2)
bf.run_continuum(config)
print(bf.stage_summary('continuum'))

# Collect results and print report
result = bf.collect_result(config)
print(result.report())
```

## Core Abstractions

- `AnalysisConfig` / `AnalysisResult`: one immutable config in, one result (provenance + diagnostics + curves + error budget) out.
- `InterpolationSpec`: declarative interpolation model; `InterpolationSpec.polynomial(order)` covers the common case.
- `DatasetCatalog` / `EnsembleFile` / `EnsembleKey`: queryable, physics-aware inventory of data files with composable filtering.
- `FitInput`: typed fit payload (`x`, `y`, labels, metadata).
- `FitModel`: model function + prior + p0 + execution behavior.
- `StageStore`: structured per-stage storage for fits, quality metrics, domains, and inputs.

## Fit Diagnostics

Use stage diagnostics in `BetaFunction`:

- `quality_table(stage)` / `stage_summary(stage)` / `analysis_summary()`
- `report()` — one-screen status of every stage
- `data_report(flow_time=...)` — per-ensemble configs, bins, and integrated autocorrelation times (with quarantined empty files listed)
- `continuum_systematics(flow, obs)` — per-g^2 stat/syst/total error budget after `continuum_window_scan`
- `provenance()` — timestamp, package versions, and physics parameters for reproducibility
- `save_analysis(fn)` / `load_analysis(fn)` — persist and restore the full analysis state

## Memory Management

gvar keeps one append-only covariance matrix per environment, so reprocessing normally leaks a full analysis worth of memory each time. `process_data` (and therefore `run_processing`/`run_analysis`) retires the previous gvar environment automatically — rerun stages as often as you like without restarting the kernel. `bf.release_memory()` frees everything on demand. Results from before a reprocess stay readable but hold their memory until deleted, and cannot be combined with new gvars.

Valid stage aliases include: `chiral`, `iv`, `infinite_volume`, `ntrp`, `interpolation`, `cnt`, `continuum`. 
