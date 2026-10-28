#!/usr/bin/env python3
# coding: utf-8
"""Run the NF4 weak/strong hybrid Bayesian model average.

This is the batch counterpart of
``nf4_hybrid_weak_fit4_fit11_pade_bma.ipynb``.  User-editable model choices
are intentionally kept identical to the notebook configuration.
"""



from pathlib import Path
import argparse, copy, gc, json, os, sys
import time as time_module

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import betafn
import gvar as gv
import lsqfit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


print('betafn package:', Path(betafn.__file__).resolve())
print('repository:', REPO_ROOT)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': False, 'font.family': 'serif',
    'font.serif': ['DejaVu Serif'], 'font.size': 15,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.labelsize': 20, 'axes.titlesize': 18,
    'legend.fontsize': 11, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'grid.alpha': .35, 'grid.linestyle': '-', 'axes.axisbelow': True,
})


parser = argparse.ArgumentParser(
    description='Run independent-flow-time NF4 weak/strong hybrid BMAs.'
)
parser.add_argument('--data-dir', type=Path, default=os.environ.get('BETAFN_DATA_DIR'))
parser.add_argument(
    '--output-base', type=Path,
    default=os.environ.get('BETAFN_OUTPUT_BASE', REPO_ROOT / 'hpcc_outputs'),
)
parser.add_argument(
    '--hybrid-thinning', action=argparse.BooleanOptionalAction, default=False,
    help='Run the optional dt=0.10, 0.20, 0.30 hybrid thinning study.',
)
parser.add_argument(
    '--hybrid-workers', type=int,
    default=int(os.environ.get(
        'BETAFN_HYBRID_WORKERS',
        int(os.environ.get('SLURM_CPUS_PER_TASK', '2')),
    )),
    help='Independent target/model processes (default: 2 locally or all allocated SLURM CPUs).',
)
parser.add_argument(
    '--continuum-workers', type=int,
    default=int(os.environ.get(
        'BETAFN_CONTINUUM_WORKERS',
        int(os.environ.get('SLURM_CPUS_PER_TASK', '2')),
    )),
    help='Independent continuum-GLS processes (default: 2 locally or all allocated SLURM CPUs).',
)
parser.add_argument(
    '--validate-only', action='store_true',
    help='Validate the exact candidate ensemble and exit before reading data.',
)
args = parser.parse_args()
if args.hybrid_workers < 1:
    parser.error('--hybrid-workers must be at least one')
if args.continuum_workers < 1:
    parser.error('--continuum-workers must be at least one')
RUN_STARTED = time_module.time()
if args.data_dir is None and not args.validate_only:
    parser.error('Set --data-dir or BETAFN_DATA_DIR.')


ANALYSIS_ID = 'hybrid_fit4_fit11_eq7_linear_bma_independent_flow_times'
ANALYSIS_FOOTER = 'hybridBMA'
CORRECTION = 'tln'
DATA_DIR = None if args.data_dir is None else args.data_dir.expanduser().resolve()
HYBRID_OUTPUT_ROOT = (
    args.output_base.expanduser().resolve() / ANALYSIS_ID
)
OUTPUT_ROOT = HYBRID_OUTPUT_ROOT
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Strong-coupling ensembles use three finite masses on each of L/a=20, 24, 32.
# This includes the newly available beta_b=7.125 ensembles.
STRONG_COUPLINGS = ('7p00', '7p125', '7p25', '7p50')
WEAK_COUPLINGS = (
    '8p00', '8p50', '9p00', '9p50', '10p0', '11p0',
    '12p0', '14p0', '16p0', '18p0', '20p0',
)
COUPLINGS = STRONG_COUPLINGS + WEAK_COUPLINGS
STRONG_VOLUMES = ('l20l20l20t40', 'l24l24l24t48', 'l32l32l32t64')
VOLUMES = {
    **{coupling: STRONG_VOLUMES for coupling in STRONG_COUPLINGS},
    '8p00': (
        'l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96',
    ),
    **{
        coupling: ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96')
        for coupling in WEAK_COUPLINGS if coupling != '8p00'
    },
}
FLOW = 'wilson'
OBSERVABLES = ('p', 'c', 's')
OP_LABELS = {'p': 'Wilson', 'c': 'clover', 's': 'Symanzik'}
# Match the operator convention used in the reference scan notebook.
OP_COLORS = {
    'p': plt.cm.YlOrRd(0.75),
    'c': plt.cm.YlGn(0.75),
    's': plt.cm.Greys(0.75),
}

# Flow-time windows remain an explicit analysis choice. Every selected
# window is analyzed independently; CENTRAL_WINDOW is the primary result.
TMIN_VALUES = tuple(range(5, 6))
TMAX_LIMIT = 6
WINDOWS = tuple(
    (float(tmin), float(tmax))
    for tmin in TMIN_VALUES
    for tmax in range(tmin + 1, TMAX_LIMIT + 1)
)
CENTRAL_WINDOW = (5.0, 6.0)
# Extend through the t0-defining coupling, g_GF^2(t0) ~= 15.79.
# Domain checks in cnt_xtrp still prevent unsupported extrapolation.
G2_GRID = (0.9, 16.0, 0.1)
TARGET_G2 = (1.0, 1.5, 3.0, 5.0, 8.0, 10.0, 12.0, 15.8)
# t0^2 <E(t0)> = 0.3 and g_GF^2 = (16 pi^2 / 3) t^2 <E>.
G2_T0 = (8.0 / 5.0) * np.pi**2

# -------------------------------------------------------------------------
# BMA MODEL SELECTION -- edit only this block to choose included models.
# -------------------------------------------------------------------------
# Each weak family can have its own selected correction orders. Delete a
# family or an order here to exclude it from the BMA.
HYBRID_WEAK_ORDERS_BY_FAMILY = {
    'monotone': (6, 7, 8),
    'monotone_constant': (6, 7, 8),
}

# Strong-coupling Pade orders are (numerator order, denominator order).
# Denominator poles are constrained to the negative-z half-line.
HYBRID_PADE_ORDERS = (
    (1, 2),
    (2, 3),
    (3, 4),
)
# The Equation-7 strong branch is connected with the fixed C1 gap bridge;
# no free matching-location parameter is used in this implementation.
HYBRID_MATCH_FRACTIONS = (None,)
HYBRID_MATCH_BOUNDS = (0.20, 0.80)
HYBRID_CONTINUITIES = (1,)
HYBRID_STRONG_FORM = 'eq7_linear'

# Leave this as None to use the Cartesian selection above. Alternatively,
# provide an exact list of models, for example:
# HYBRID_EXPLICIT_MODELS = (
#     dict(weak_family='monotone', weak_order=7,
#          numerator_order=2, denominator_order=3,
#          match_fraction=None, continuity=1,
#          strong_form='eq7_linear'),
# )
HYBRID_EXPLICIT_MODELS = None
HYBRID_MAX_WORKERS = args.hybrid_workers
HYBRID_CONTINUUM_WORKERS = args.continuum_workers
HYBRID_CHECKPOINT_ROOT = HYBRID_OUTPUT_ROOT / 'interpolation_checkpoints'

RUN_HYBRID_THINNING = args.hybrid_thinning
HYBRID_THINNING_SPACINGS = (0.10, 0.20, 0.30)
HYBRID_PRIOR_WIDTHS = {
    'weak_width': 1.5,
    'constant_width': 0.25,
    'strong_width': 2.0,
    'pole_log_width': 1.0,
}

def window_tag(window):
    return f't_{window[0]:g}_{window[1]:g}'.replace('.', 'p')

def case_dir(window, mode=None):
    path = OUTPUT_ROOT / window_tag(window)
    if mode is not None:
        path = path / mode
    path.mkdir(parents=True, exist_ok=True)
    return path

def add_fit_footer(fig):
    fig.text(0.995, 0.005, ANALYSIS_FOOTER, ha='right', va='bottom',
             fontsize=8, color='gray', alpha=.75)

def save_figure(fig, window, name, mode=None):
    base = case_dir(window, mode) / name
    add_fit_footer(fig)
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def beta_value(coupling):
    return float(coupling.replace('p', '.'))

def volume_value(volume):
    return np.prod([float(x) for x in volume.replace('t', 'l').split('l')[1:] if x])


def coupling_figure(couplings, ncols=5):
    nrows = max(1, int(np.ceil(len(couplings) / ncols)))
    return plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 4.2 * nrows),
        squeeze=False,
    )

def pt_over_g4(x, loops):
    x = np.asarray(x, dtype=float)
    pt = bf.perturbative_beta_function
    return sum(-coefficient * x**power / pt.nrm**(power + 1)
               for power, coefficient in enumerate(pt.b[:loops]))

print(f'{len(WINDOWS)} flow-time windows:', WINDOWS)
print('output root:', OUTPUT_ROOT)

_PROGRESS_START_TIMES = {}
def progress_start(label):
    _PROGRESS_START_TIMES[label] = time_module.monotonic()
    print(f'[START] {label}', flush=True)

def progress_done(label):
    started = _PROGRESS_START_TIMES.pop(label, None)
    elapsed = time_module.monotonic() - started if started is not None else 0.0
    print(f'[DONE] {label} ({elapsed / 60.0:.1f} min)', flush=True)


from betafn import (
    HybridInterpolationCandidate,
    default_hybrid_candidates,
    fit_hybrid_bma_by_target,
    hybrid_match_fraction,
    hybrid_parameter_records,
    model_average_predictions,
    relative_partial_uncertainty_percent,
    run_hybrid_bma_by_target,
)


def build_hybrid_candidates():
    if HYBRID_EXPLICIT_MODELS is None:
        candidates = default_hybrid_candidates(
            weak_orders_by_family=HYBRID_WEAK_ORDERS_BY_FAMILY,
            pade_orders=HYBRID_PADE_ORDERS,
            match_fractions=HYBRID_MATCH_FRACTIONS,
            match_bounds=HYBRID_MATCH_BOUNDS,
            continuities=HYBRID_CONTINUITIES,
            strong_form=HYBRID_STRONG_FORM,
        )
    else:
        candidates = tuple(
            HybridInterpolationCandidate(**selection)
            for selection in HYBRID_EXPLICIT_MODELS
        )
    if not candidates:
        raise ValueError('Select at least one hybrid BMA candidate')
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError('The selected hybrid BMA candidates contain duplicates')
    return candidates


HYBRID_CANDIDATES = build_hybrid_candidates()
expected_candidate_count = (
    sum(len(orders) for orders in HYBRID_WEAK_ORDERS_BY_FAMILY.values())
    * len(HYBRID_PADE_ORDERS)
    * len(HYBRID_MATCH_FRACTIONS)
    * len(HYBRID_CONTINUITIES)
    if HYBRID_EXPLICIT_MODELS is None else len(HYBRID_EXPLICIT_MODELS)
)
if len(HYBRID_CANDIDATES) != expected_candidate_count:
    raise RuntimeError(
        f'candidate count {len(HYBRID_CANDIDATES)} != expected '
        f'{expected_candidate_count}'
    )

candidate_configuration = [
    {'model': candidate.name, **candidate.metadata()}
    for candidate in HYBRID_CANDIDATES
]


def validate_dataset_layout(data_dir):
    if data_dir is None:
        return None
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    discovered = betafn.SetupBetaFunction.discover_dataset(
        data_dir, flows=[FLOW]
    )
    missing = []
    for coupling in COUPLINGS:
        expected_masses = (
            ('0p001', '0p0025', '0p005')
            if coupling in STRONG_COUPLINGS else ('0p00',)
        )
        for volume in VOLUMES[coupling]:
            for mass in expected_masses:
                flows = (
                    discovered.get(coupling, {})
                    .get(volume, {})
                    .get(mass, ())
                )
                if FLOW not in flows:
                    missing.append(f'{coupling}_{volume}_{mass}_{FLOW}.bin')
    if missing:
        preview = ', '.join(missing[:12])
        suffix = '' if len(missing) <= 12 else f', ... ({len(missing)} missing)'
        raise FileNotFoundError(
            f'dataset is missing required ensembles: {preview}{suffix}'
        )
    return {
        'couplings': len(discovered),
        'required_ensemble_mass_points': sum(
            3 * len(VOLUMES[coupling])
            if coupling in STRONG_COUPLINGS else len(VOLUMES[coupling])
            for coupling in COUPLINGS
        ),
    }


dataset_validation = validate_dataset_layout(DATA_DIR)
if args.validate_only:
    print(json.dumps({
        'analysis_id': ANALYSIS_ID,
        'candidate_count': len(HYBRID_CANDIDATES),
        'fits_per_operator_flow_time': len(HYBRID_CANDIDATES),
        'interpolation_treatment': 'independent_by_operator_and_flow_time',
        'cross_flow_time_covariance': 'linear_response_to_shared_gvar_data',
        'candidates': candidate_configuration,
        'windows': WINDOWS,
        'g2_grid': G2_GRID,
        'hybrid_thinning': RUN_HYBRID_THINNING,
        'hybrid_workers': HYBRID_MAX_WORKERS,
        'hybrid_continuum_workers': HYBRID_CONTINUUM_WORKERS,
        'hybrid_strong_form': HYBRID_STRONG_FORM,
        'hybrid_match_bounds': list(HYBRID_MATCH_BOUNDS),
        'dataset_validation': dataset_validation,
    }, indent=2))
    raise SystemExit(0)

(OUTPUT_ROOT / 'RUN_COMPLETE.json').unlink(missing_ok=True)
(OUTPUT_ROOT / 'RUN_COMPLETE.json.tmp').unlink(missing_ok=True)

from betafn.blinding import load_lambda_blinding_factor
load_lambda_blinding_factor()  # validate without printing the hidden factor

run_configuration = {
    'analysis_id': ANALYSIS_ID,
    'candidate_count': len(HYBRID_CANDIDATES),
    'fits_per_operator_flow_time': len(HYBRID_CANDIDATES),
    'candidates': candidate_configuration,
    'model_prior_probability': 1.0 / len(HYBRID_CANDIDATES),
    'prior_widths': HYBRID_PRIOR_WIDTHS,
    'windows': [list(window) for window in WINDOWS],
    'central_window': list(CENTRAL_WINDOW),
    'g2_grid': list(G2_GRID),
    'target_g2': list(TARGET_G2),
    'data_dir': str(DATA_DIR),
    'output_root': str(HYBRID_OUTPUT_ROOT),
    'covariance_estimator': 'lugsail',
    'bartlett_bandwidth_method': 'plugin',
    'bartlett_window_factor': 5.0,
    'bartlett_max_fraction': 0.25,
    'lugsail_r': 2.0,
    'lugsail_c': 0.5,
    'hybrid_thinning': RUN_HYBRID_THINNING,
    'hybrid_workers': HYBRID_MAX_WORKERS,
    'hybrid_continuum_workers': HYBRID_CONTINUUM_WORKERS,
    'hybrid_strong_form': HYBRID_STRONG_FORM,
    'hybrid_transition': 'C1 rational Hermite inside gap; weak extension 0.05',
    'hybrid_match_bounds': list(HYBRID_MATCH_BOUNDS),
    'interpolation_treatment': 'independent_by_operator_and_flow_time',
    'cross_flow_time_covariance': 'linear_response_to_shared_gvar_data',
    'hybrid_thinning_spacings': list(HYBRID_THINNING_SPACINGS),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
}
(HYBRID_OUTPUT_ROOT / 'run_configuration.json').write_text(
    json.dumps(run_configuration, indent=2) + '\n'
)


bf = betafn.BetaFunction(nf=4)

# AnalysisConfig requires an interpolation field for shared stages 1--3.
# The hybrid candidate fits below bypass this placeholder completely.
def _unused_interpolation(x, parameters):
    return np.zeros_like(np.asarray(x, dtype=float))

interpolation = betafn.InterpolationSpec(
    fcn=_unused_interpolation, prior=None, p0={}, xerrors=False,
)

config = betafn.AnalysisConfig(
    data_path=str(DATA_DIR),
    interpolation=interpolation,
    continuum_window=CENTRAL_WINDOW,
    g2_grid=G2_GRID,
    flows=(FLOW,),
    observables=OBSERVABLES,
    combine={'s': {'p': 5.0 / 3.0, 's': -2.0 / 3.0}},
    couplings=COUPLINGS,
    volumes=VOLUMES,
    correction=CORRECTION,
    binsize=1,  # spectral estimator uses the ordered, unbinned history
    covariance_estimator='lugsail',
    bartlett_bandwidth_method='plugin',
    bartlett_window_factor=5.0,
    bartlett_max_fraction=0.25,
    lugsail_r=2.0, lugsail_c=0.5, spectral_psd_adjust=True,
    process_window=(1.5, 10.5),
    fit_window=(2.0, 10.0),
    error_mode='fit',
    cov_mode='empirical',
    correlated=True,
    verbosity=1,  # progress only; does not alter any fit
)
config_summary = config.describe()
config_summary.pop('interpolation_xerrors')
config_summary.pop('interpolation_parameters')
print(pd.DataFrame([config_summary]).T.rename(columns={0: 'hybrid analysis setting'}).to_string())


progress_start('processing and covariance estimation')
bf.run_processing(config)
print('correction:', bf._process_config.correction)
print('processed couplings:', sorted(bf.avg_data, key=beta_value))
print('pre-binning: not applied; lugsail Bartlett uses the ordered raw history')
print(bf.data_report())
bartlett_rows = [
    {'coupling': ensemble.coupling, 'volume': ensemble.volume,
     'mass': ensemble.mass, **diagnostics}
    for ensemble, diagnostics in sorted(
        bf.covariance_estimator_diagnostics.items(),
        key=lambda item: item[0].sort_key,
    )
]
BARTLETT_DIAGNOSTICS = pd.DataFrame(bartlett_rows)
BARTLETT_DIAGNOSTICS.to_csv(
    OUTPUT_ROOT / 'bartlett_covariance_diagnostics.csv', index=False
)
print(BARTLETT_DIAGNOSTICS)
progress_done('processing and covariance estimation')


fig, ax = plt.subplots(figsize=(8, 6))
for operator in OBSERVABLES:
    first = True
    for coupling in sorted(bf.avg_data, key=beta_value):
        massless_volumes = [
            volume for volume in bf.avg_data[coupling]
            if '0p00' in bf.avg_data[coupling][volume]
        ]
        if not massless_volumes:
            continue
        volume = max(massless_volumes, key=volume_value)
        _, g2 = bf.processed_series(coupling, volume, '0p00', FLOW, operator, 'g2')
        _, beta = bf.processed_series(coupling, volume, '0p00', FLOW, operator, 'beta')
        ratio = np.asarray([b / x**2 for x, b in zip(g2, beta)], dtype=object)
        ax.errorbar(gv.mean(g2), gv.mean(ratio), yerr=gv.sdev(ratio), fmt='o', ms=2.5,
                    alpha=.6, color=OP_COLORS[operator],
                    label=OP_LABELS[operator] if first else None)
        first = False
ax.set(xlabel=r'$g^2_{GF}$', ylabel=r'$\beta_{GF}/g_{GF}^4$', title='Directly simulated massless largest-volume data (TLN)')
ax.legend(frameon=False)
base = OUTPUT_ROOT / f'processed_largest_volume_{ANALYSIS_ID}'
add_fit_footer(fig)
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)


progress_start('chiral extrapolation')
bf.run_chiral(config)
progress_done('chiral extrapolation')
progress_start('infinite-volume extrapolation')
bf.run_infinite_volume(config)
progress_done('infinite-volume extrapolation')
print(bf.stage_summary('chiral'))
print(bf.stage_summary('infinite_volume'))


ACTIVE_COUPLINGS = tuple(sorted(bf.avg_data, key=beta_value))
ACTIVE_STRONG_COUPLINGS = tuple(
    coupling for coupling in STRONG_COUPLINGS if coupling in bf.avg_data
)
MISSING_STRONG_COUPLINGS = tuple(
    coupling for coupling in STRONG_COUPLINGS if coupling not in bf.avg_data
)
IV_COUPLINGS = tuple(
    coupling for coupling in ACTIVE_COUPLINGS
    if coupling in bf.iv_fits
    and any(bf.iv_fits[coupling]['g2'][FLOW][operator]
            for operator in OBSERVABLES)
)

print('active couplings:', ACTIVE_COUPLINGS)
print('configured strong couplings without usable data:', MISSING_STRONG_COUPLINGS or 'none')

EXPECTED_CHIRAL_MASSES = ('0p001', '0p0025', '0p005')
inventory_rows = []
for coupling in STRONG_COUPLINGS:
    if coupling not in bf.avg_data:
        inventory_rows.append({
            'coupling': coupling,
            'beta_b': beta_value(coupling),
            'volume': 'all configured volumes',
            'finite_masses': '',
            'massless_limit_available': False,
            'status': 'awaiting files',
        })
        continue
    for volume in VOLUMES[coupling]:
        volume_data = bf.avg_data[coupling].get(volume, {})
        finite_masses = sorted(
            (mass for mass in volume_data if mass != '0p00'),
            key=lambda mass: float(mass.replace('p', '.')),
        )
        inventory_rows.append({
            'coupling': coupling,
            'beta_b': beta_value(coupling),
            'volume': volume,
            'finite_masses': ', '.join(finite_masses),
            'massless_limit_available': '0p00' in volume_data,
            'status': (
                'ready'
                if tuple(finite_masses) == EXPECTED_CHIRAL_MASSES
                else 'incomplete mass set'
            ),
        })

chiral_inventory = pd.DataFrame(inventory_rows)
chiral_inventory.to_csv(OUTPUT_ROOT / 'chiral_ensemble_inventory.csv', index=False)
print(chiral_inventory)

chiral_quality_rows = []
for coupling in ACTIVE_STRONG_COUPLINGS:
    for volume in VOLUMES[coupling]:
        if volume not in bf.ch_fits.get(coupling, {}):
            continue
        for quantity in ('g2', 'beta'):
            for operator in OBSERVABLES:
                fits = bf.ch_fits[coupling][volume][quantity][FLOW][operator]
                qualities = bf.ch_qof[coupling][volume][quantity][FLOW][operator]
                for time, params in fits.items():
                    qof = qualities[time]
                    intercept = bf.chiral.model.evaluate(0.0, params)
                    chiral_quality_rows.append({
                        'coupling': coupling,
                        'beta_b': beta_value(coupling),
                        'volume': volume,
                        'quantity': quantity,
                        'operator': operator,
                        'flow_time': float(time),
                        'intercept_mean': float(gv.mean(intercept)),
                        'intercept_sdev': float(gv.sdev(intercept)),
                        'chi2': float(qof['chi2']),
                        'dof': int(qof['dof']),
                        'p_value': float(qof['p-value']),
                    })

chiral_fit_quality = pd.DataFrame(chiral_quality_rows)
chiral_fit_quality.to_csv(
    OUTPUT_ROOT / 'chiral_fit_quality_summary.csv', index=False
)
print(chiral_fit_quality)

CHIRAL_PLOT_TIMES = (4.0, 6.0, 8.0)
CHIRAL_TIME_COLORS = dict(zip(
    CHIRAL_PLOT_TIMES,
    plt.cm.viridis(np.linspace(0.15, 0.85, len(CHIRAL_PLOT_TIMES))),
))

def plot_chiral_extrapolations(quantity):
    ylabel = (
        r'$g^2_{\mathrm{GF}}$'
        if quantity == 'g2'
        else r'$\beta_{\mathrm{GF}}$'
    )
    for coupling in ACTIVE_STRONG_COUPLINGS:
        volumes = [
            volume for volume in VOLUMES[coupling]
            if volume in bf.ch_fits.get(coupling, {})
            and any(
                bf.ch_fits[coupling][volume]['g2'][FLOW][operator]
                for operator in OBSERVABLES
            )
        ]
        if not volumes:
            continue
        fig, axes = plt.subplots(
            len(volumes), len(OBSERVABLES),
            figsize=(15, 4.2 * len(volumes)),
            squeeze=False,
        )
        for row, volume in enumerate(volumes):
            for column, operator in enumerate(OBSERVABLES):
                ax = axes[row, column]
                available = sorted(
                    bf.ch_fits[coupling][volume][quantity][FLOW][operator],
                    key=float,
                )
                selected = []
                selected_times = set()
                for target in CHIRAL_PLOT_TIMES:
                    if not available:
                        continue
                    time = min(available, key=lambda value: abs(float(value) - target))
                    if time not in selected_times:
                        selected.append((target, time))
                        selected_times.add(time)

                pvalue_lines = []
                for target, time in selected:
                    data = bf.chiral.fetch(
                        'inputs', (coupling, volume, quantity, FLOW, operator, time)
                    )
                    params = bf.chiral.fetch(
                        'fits', (coupling, volume, quantity, FLOW, operator, time)
                    )
                    qof = bf.chiral.fetch(
                        'quality', (coupling, volume, quantity, FLOW, operator, time)
                    )
                    masses = np.asarray(data.x, dtype=float)
                    values = np.asarray(data.y, dtype=object)
                    mass_grid = np.linspace(0.0, 1.05 * masses.max(), 160)
                    curve = np.asarray([
                        bf.chiral.model.evaluate(mass, params)
                        for mass in mass_grid
                    ], dtype=object)
                    intercept = bf.chiral.model.evaluate(0.0, params)
                    color = CHIRAL_TIME_COLORS[target]
                    label = rf'$t/a^2={float(time):g}$'
                    ax.errorbar(
                        masses, gv.mean(values), yerr=gv.sdev(values),
                        fmt='o', ms=4, capsize=3, color=color, label=label,
                    )
                    ax.plot(mass_grid, gv.mean(curve), color=color, lw=1.5)
                    ax.fill_between(
                        mass_grid,
                        gv.mean(curve) - gv.sdev(curve),
                        gv.mean(curve) + gv.sdev(curve),
                        color=color, alpha=.14, linewidth=0,
                    )
                    ax.errorbar(
                        [0.0], [gv.mean(intercept)], yerr=[gv.sdev(intercept)],
                        fmt='s', ms=5, capsize=3, color=color,
                        markeredgecolor='black', markeredgewidth=.4,
                    )
                    pvalue_lines.append(
                        rf'$t={float(time):g}$: $p={100.0 * float(qof["p-value"]):.1f}\%$'
                    )

                spatial_extent = int(volume.split('l')[1])
                ax.set_xlabel(r'$am_f$')
                ax.text(
                    .97, .97,
                    rf'$\beta_b={beta_value(coupling):g}$, ' + OP_LABELS[operator]
                    + '\n' + rf'$L/a={spatial_extent}$',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10.5,
                )
                ax.text(
                    .03, .97, '\n'.join(pvalue_lines), transform=ax.transAxes,
                    ha='left', va='top', fontsize=8.5,
                )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=max(1, len(handles)), loc='upper center', frameon=False)
        fig.supylabel(ylabel)
        fig.tight_layout(rect=(.03, .02, 1, .93))
        base = OUTPUT_ROOT / f'chiral_extrapolation_{quantity}_{coupling}_{ANALYSIS_ID}'
        add_fit_footer(fig)
        fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
        fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

plot_chiral_extrapolations('g2')
plot_chiral_extrapolations('beta')

# Confirm that the massless data used downstream are mutually consistent
# after the infinite-volume extrapolation. Keep each flow-time window
# separate so the plotted sample matches that window.
def plot_processed_massless_infinite_volume(window):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for operator in OBSERVABLES:
        first = True
        for coupling in sorted(IV_COUPLINGS, key=beta_value, reverse=True):
            g2_fits = bf.iv_fits[coupling]['g2'][FLOW][operator]
            beta_fits = bf.iv_fits[coupling]['beta'][FLOW][operator]
            common_times = [
                flow_time
                for flow_time in sorted(set(g2_fits) & set(beta_fits), key=float)
                if window[0] <= float(flow_time) <= window[1]
            ]
            for flow_time in common_times:
                g2_iv = bf.infinite_volume.model.evaluate(
                    0.0, g2_fits[flow_time]
                )
                beta_iv = bf.infinite_volume.model.evaluate(
                    0.0, beta_fits[flow_time]
                )
                ratio_iv = beta_iv / g2_iv**2
                ax.errorbar(
                    gv.mean(g2_iv), gv.mean(ratio_iv),
                    xerr=gv.sdev(g2_iv), yerr=gv.sdev(ratio_iv),
                    fmt='o', ms=2.5, alpha=.6,
                    color=OP_COLORS[operator],
                    label=OP_LABELS[operator] if first else None,
                )
                first = False
    ax.set(
        xlabel=r'$g^2_{\mathrm{GF}}$',
        ylabel=r'$\beta_{\mathrm{GF}}/g_{\mathrm{GF}}^4$',
        title=(
            'Massless-limit infinite-volume data after chiral extrapolation, '
            rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
        ),
    )
    ax.legend(frameon=False)
    plt.show()
    save_figure(
        fig, window,
        f'processed_massless_limit_infinite_volume_{ANALYSIS_ID}',
    )


for window in WINDOWS:
    plot_processed_massless_infinite_volume(window)


# Raw massless infinite-volume intercepts before interpolation.
ordered_iv_couplings = tuple(
    sorted(IV_COUPLINGS, key=beta_value, reverse=True)
)
qualitative_colors = tuple(plt.get_cmap('tab20').colors)
beta_colors = {
    coupling: qualitative_colors[index % len(qualitative_colors)]
    for index, coupling in enumerate(ordered_iv_couplings)
}
operator_markers = {'p': 'o', 'c': 's', 's': '^'}
plot_xmax = max(G2_GRID[1], G2_T0) + 0.25
xp_pt = np.linspace(0.001, plot_xmax, 900)
pt_styles = (
    (1, ':', '1-loop'),
    (2, '-.', '2-loop'),
    (3, '-', '3-loop'),
)
operator_handles = [
    Line2D(
        [0], [0], marker=operator_markers[operator], linestyle='none',
        color='gray', markeredgecolor='black', markeredgewidth=.4,
        label=OP_LABELS[operator],
    )
    for operator in OBSERVABLES
]
coupling_handles = [
    Line2D(
        [0], [0], marker='o', linestyle='none',
        color=beta_colors[coupling], markersize=6,
        label=rf'$\beta_b={beta_value(coupling):g}$',
    )
    for coupling in ordered_iv_couplings
]
pt_handles = [
    Line2D(
        [0], [0], color='gray', linestyle=linestyle,
        lw=1.5, label=label,
    )
    for _, linestyle, label in pt_styles
]
t0_handle = Line2D(
    [0], [0], color='black', linestyle='--', lw=1.5,
    label=rf'$g^2_{{\mathrm{{GF}}}}(t_0)=\frac{{8}}{{5}}\pi^2={G2_T0:.9f}$',
)


def plot_massless_infinite_volume(window, divide_by_g4=False):
    """Plot massless IV data using only flow times inside one window."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for coupling in ordered_iv_couplings:
        color = beta_colors[coupling]
        for operator in OBSERVABLES:
            g2_fits = bf.iv_fits[coupling]['g2'][FLOW][operator]
            beta_fits = bf.iv_fits[coupling]['beta'][FLOW][operator]
            common_times = [
                flow_time
                for flow_time in sorted(set(g2_fits) & set(beta_fits), key=float)
                if window[0] <= float(flow_time) <= window[1]
            ]
            for flow_time in common_times:
                g2_iv = bf.infinite_volume.model.evaluate(
                    0.0, g2_fits[flow_time]
                )
                beta_iv = bf.infinite_volume.model.evaluate(
                    0.0, beta_fits[flow_time]
                )
                value = beta_iv / g2_iv**2 if divide_by_g4 else beta_iv
                ax.errorbar(
                    gv.mean(g2_iv), gv.mean(value),
                    xerr=gv.sdev(g2_iv), yerr=gv.sdev(value),
                    fmt=operator_markers[operator], ms=3.8, capsize=2,
                    color=color, alpha=.58, linestyle='none',
                )

    for loops, linestyle, _ in pt_styles:
        ratio_pt = pt_over_g4(xp_pt, loops)
        perturbative_value = ratio_pt if divide_by_g4 else xp_pt**2 * ratio_pt
        ax.plot(
            xp_pt, perturbative_value,
            color='gray', ls=linestyle, lw=1.5, zorder=1,
        )

    quantity = (
        r'$\beta_{\mathrm{GF}}/g_{\mathrm{GF}}^4$'
        if divide_by_g4 else r'$\beta_{\mathrm{GF}}$'
    )
    ax.set(
        xlim=(0.0, plot_xmax),
        xlabel=r'$g^2_{\mathrm{GF}}$',
        ylabel=quantity,
        title=(
            rf'Massless infinite-volume data: {quantity}, '
            rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
        ),
    )
    if not divide_by_g4:
        ax.set_ylim(-13.0, 0.5)
    ax.axvline(G2_T0, color='black', linestyle='--', lw=1.5, zorder=1)
    ax.legend(
        handles=(
            operator_handles + coupling_handles + pt_handles + [t0_handle]
        ),
        frameon=False, fontsize=8, ncol=2,
    )
    fig.tight_layout()
    name = (
        f'massless_infinite_volume_beta_over_g4_vs_g2_{ANALYSIS_ID}'
        if divide_by_g4
        else f'massless_infinite_volume_beta_vs_g2_{ANALYSIS_ID}'
    )
    plt.show()
    save_figure(fig, window, name)


for window in WINDOWS:
    plot_massless_infinite_volume(window, divide_by_g4=False)
    plot_massless_infinite_volume(window, divide_by_g4=True)


fig, axes = coupling_figure(IV_COUPLINGS)
for ax, coupling in zip(axes.ravel(), IV_COUPLINGS):
    times = sorted(bf.iv_fits[coupling]['g2'][FLOW][OBSERVABLES[0]], key=float)
    time = min(times, key=lambda t: abs(float(t) - CENTRAL_WINDOW[0]))
    for operator in OBSERVABLES:
        xfit, yfit, data = bf.infinite_volume_curve(coupling, FLOW, operator, time, x='g2')
        intercept = bf.infinite_volume.model.evaluate(0.0, bf.iv_fits[coupling]['g2'][FLOW][operator][time])
        color = OP_COLORS[operator]
        ax.errorbar(gv.mean(data.x), gv.mean(data.y), xerr=gv.sdev(data.x), yerr=gv.sdev(data.y),
                    fmt='o', capsize=3, color=color, label=OP_LABELS[operator])
        ax.plot(xfit, gv.mean(yfit), color=color)
        ax.fill_between(xfit, gv.mean(yfit)-gv.sdev(yfit), gv.mean(yfit)+gv.sdev(yfit), color=color, alpha=.15)
        ax.errorbar([0], [gv.mean(intercept)], yerr=[gv.sdev(intercept)], fmt='s', capsize=3, color=color)
    ax.set(title=rf'$\beta_b={beta_value(coupling):g}$, $t/a^2={time}$', xlabel=r'$1/V$', ylabel=r'$g^2_{GF}$')
for ax in axes.ravel()[len(IV_COUPLINGS):]:
    ax.set_visible(False)

handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc='upper center', frameon=False)
fig.tight_layout(rect=(0, 0, 1, .96))
base = OUTPUT_ROOT / f'infinite_volume_all_beta_{ANALYSIS_ID}'
add_fit_footer(fig)
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)


# Infinite-volume extrapolation: beta_GF versus 1/V

fig, axes = coupling_figure(IV_COUPLINGS)

for ax, coupling in zip(axes.ravel(), IV_COUPLINGS):
  available_times = sorted(
      bf.iv_fits[coupling]['beta'][FLOW][OBSERVABLES[0]],
      key=float,
  )
  time = min(
      available_times,
      key=lambda value: abs(float(value) - CENTRAL_WINDOW[0]),
  )

  for operator in OBSERVABLES:
      x_fit, beta_fit, beta_data = bf.infinite_volume_curve(
          coupling,
          FLOW,
          operator,
          time,
          x='beta',
      )

      x_fit = np.asarray(x_fit, dtype=float)
      beta_fit = np.asarray(beta_fit, dtype=object)
      beta_data_y = np.asarray(beta_data.y, dtype=object)

      color = OP_COLORS[operator]

      # Finite-volume beta_GF data.
      ax.errorbar(
          beta_data.x,
          gv.mean(beta_data_y),
          yerr=gv.sdev(beta_data_y),
          fmt='o',
          ms=5,
          capsize=3,
          elinewidth=1.0,
          color=color,
          label=OP_LABELS[operator],
          zorder=3,
      )

      # Infinite-volume extrapolation curve.
      ax.plot(
          x_fit,
          gv.mean(beta_fit),
          color=color,
          lw=1.6,
          zorder=2,
      )

      ax.fill_between(
          x_fit,
          gv.mean(beta_fit) - gv.sdev(beta_fit),
          gv.mean(beta_fit) + gv.sdev(beta_fit),
          color=color,
          alpha=0.16,
          linewidth=0,
          zorder=1,
      )

      # Infinite-volume intercept at 1/V = 0.
      beta_params = bf.iv_fits[coupling]['beta'][FLOW][operator][time]
      beta_infinite = bf.infinite_volume.model.evaluate(
          0.0,
          beta_params,
      )

      ax.errorbar(
          [0.0],
          [gv.mean(beta_infinite)],
          yerr=[gv.sdev(beta_infinite)],
          fmt='s',
          ms=6,
          capsize=3,
          color=color,
          markeredgecolor='black',
          markeredgewidth=0.4,
          zorder=4,
      )

  ax.set_title(
      rf'$\beta_b={beta_value(coupling):g}$, '
      rf'$t/a^2={float(time):g}$'
  )
  ax.set_xlabel(r'$1/V$')
  ax.set_ylabel(r'$\beta_{\mathrm{GF}}$')
  ax.minorticks_on()
  ax.grid(which='major', linestyle='-', alpha=0.35)
  ax.grid(which='minor', linestyle='--', alpha=0.20)

for ax in axes.ravel()[len(IV_COUPLINGS):]:
    ax.set_visible(False)

# Shared legend without duplicate entries.
handles, labels = axes.ravel()[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(
  unique.values(),
  unique.keys(),
  ncol=len(OBSERVABLES),
  loc='upper center',
  frameon=False,
)

fig.suptitle(
  r'Infinite-volume extrapolation of $\beta_{\mathrm{GF}}$',
  y=0.995,
)

fig.tight_layout(rect=(0, 0, 1, 0.95))

output_base = OUTPUT_ROOT / f'infinite_volume_beta_vs_invV_{ANALYSIS_ID}'
add_fit_footer(fig)
fig.savefig(
  output_base.with_suffix('.png'),
  dpi=300,
  bbox_inches='tight',
)
fig.savefig(
  output_base.with_suffix('.pdf'),
  dpi=300,
  bbox_inches='tight',
)

plt.show()
plt.close(fig)


# Infinite-volume extrapolation:
# beta_GF / g_GF^4 versus 1/V

fig, axes = coupling_figure(IV_COUPLINGS)

for ax, coupling in zip(axes.ravel(), IV_COUPLINGS):
  available_times = sorted(
      bf.iv_fits[coupling]['beta'][FLOW][OBSERVABLES[0]],
      key=float,
  )
  time = min(
      available_times,
      key=lambda value: abs(float(value) - CENTRAL_WINDOW[0]),
  )

  for operator in OBSERVABLES:
      # Retrieve the beta_GF and g_GF^2 data used in the IV fits.
      _, _, beta_data = bf.infinite_volume_curve(
          coupling,
          FLOW,
          operator,
          time,
          x='beta',
      )
      _, _, g2_data = bf.infinite_volume_curve(
          coupling,
          FLOW,
          operator,
          time,
          x='g2',
      )

      # Confirm that beta and g^2 use the same volumes in the same order.
      if beta_data.labels != g2_data.labels:
          raise RuntimeError(
              f'Volume ordering mismatch for {coupling}, '
              f'{operator}, t/a^2={time}'
          )

      inv_volume = np.asarray(beta_data.x, dtype=float)
      beta_values = np.asarray(beta_data.y, dtype=object)
      g2_values = np.asarray(g2_data.y, dtype=object)

      # Since g2_values = g_GF^2, g_GF^4 = g2_values**2.
      ratio_data = beta_values / g2_values**2

      beta_params = bf.iv_fits[coupling]['beta'][FLOW][operator][time]
      g2_params = bf.iv_fits[coupling]['g2'][FLOW][operator][time]

      x_fit = np.linspace(
          float(np.min(inv_volume)),
          float(np.max(inv_volume)),
          300,
      )

      beta_fit = np.asarray(
          [
              bf.infinite_volume.model.evaluate(x, beta_params)
              for x in x_fit
          ],
          dtype=object,
      )
      g2_fit = np.asarray(
          [
              bf.infinite_volume.model.evaluate(x, g2_params)
              for x in x_fit
          ],
          dtype=object,
      )

      ratio_fit = beta_fit / g2_fit**2
      color = OP_COLORS[operator]

      # Finite-volume beta_GF/g_GF^4 data.
      ax.errorbar(
          inv_volume,
          gv.mean(ratio_data),
          yerr=gv.sdev(ratio_data),
          fmt='o',
          ms=5,
          capsize=3,
          elinewidth=1.0,
          color=color,
          label=OP_LABELS[operator],
          zorder=3,
      )

      # Ratio of the fitted beta_GF and g_GF^2 IV curves.
      ax.plot(
          x_fit,
          gv.mean(ratio_fit),
          color=color,
          lw=1.6,
          zorder=2,
      )

      ax.fill_between(
          x_fit,
          gv.mean(ratio_fit) - gv.sdev(ratio_fit),
          gv.mean(ratio_fit) + gv.sdev(ratio_fit),
          color=color,
          alpha=0.16,
          linewidth=0,
          zorder=1,
      )

      # Infinite-volume ratio at 1/V = 0.
      beta_infinite = bf.infinite_volume.model.evaluate(
          0.0,
          beta_params,
      )
      g2_infinite = bf.infinite_volume.model.evaluate(
          0.0,
          g2_params,
      )
      ratio_infinite = beta_infinite / g2_infinite**2

      ax.errorbar(
          [0.0],
          [gv.mean(ratio_infinite)],
          yerr=[gv.sdev(ratio_infinite)],
          fmt='s',
          ms=6,
          capsize=3,
          color=color,
          markeredgecolor='black',
          markeredgewidth=0.4,
          zorder=4,
      )

  ax.set_title(
      rf'$\beta_b={beta_value(coupling):g}$, '
      rf'$t/a^2={float(time):g}$'
  )
  ax.set_xlabel(r'$1/V$')
  ax.set_ylabel(r'$\beta_{\mathrm{GF}}/g_{\mathrm{GF}}^4$')
  ax.minorticks_on()
  ax.grid(which='major', linestyle='-', alpha=0.35)
  ax.grid(which='minor', linestyle='--', alpha=0.20)

for ax in axes.ravel()[len(IV_COUPLINGS):]:
    ax.set_visible(False)

# Shared legend without duplicate entries.
handles, labels = axes.ravel()[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(
  unique.values(),
  unique.keys(),
  ncol=len(OBSERVABLES),
  loc='upper center',
  frameon=False,
)

fig.suptitle(
  r'Infinite-volume extrapolation of '
  r'$\beta_{\mathrm{GF}}/g_{\mathrm{GF}}^4$',
  y=0.995,
)

fig.tight_layout(rect=(0, 0, 1, 0.95))

output_base = (
  OUTPUT_ROOT
  / f'infinite_volume_beta_over_g4_vs_invV_{ANALYSIS_ID}'
)

add_fit_footer(fig)
fig.savefig(
  output_base.with_suffix('.png'),
  dpi=300,
  bbox_inches='tight',
)
fig.savefig(
  output_base.with_suffix('.pdf'),
  dpi=300,
  bbox_inches='tight',
)

plt.show()
plt.close(fig)



progress_start('independent flow-time hybrid interpolation and evidence')
hybrid_model_selection = pd.DataFrame(candidate_configuration)
hybrid_model_selection['model_prior_probability'] = 1.0 / len(HYBRID_CANDIDATES)
hybrid_model_selection.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_model_selection.csv', index=False
)
print(hybrid_model_selection.to_string(index=False))
print(f'Selected {len(HYBRID_CANDIDATES)} hybrid models.')

hybrid_interpolations = {}
for window in WINDOWS:
    print('hybrid interpolation:', window, 'independent operator/flow-time fits')
    hybrid_interpolations[window] = fit_hybrid_bma_by_target(
        bf, window, FLOW, OBSERVABLES,
        weak_couplings=WEAK_COUPLINGS,
        strong_couplings=STRONG_COUPLINGS,
        candidates=HYBRID_CANDIDATES,
        max_workers=HYBRID_MAX_WORKERS,
        checkpoint_dir=HYBRID_CHECKPOINT_ROOT / window_tag(window),
        **HYBRID_PRIOR_WIDTHS,
    )

evidence_rows = []
for window, target_results in hybrid_interpolations.items():
    for key, result in target_results.items():
        _, operator, flow_time = key
        for name, candidate_fit in result.candidate_fits.items():
            evidence_rows.append({
                'window': window_tag(window), 'operator': operator,
                'flow_time': float(flow_time),
                'model': name,
                'log_evidence': candidate_fit.log_evidence,
                'posterior_model_probability': result.weights[name],
                **candidate_fit.candidate.metadata(),
            })
hybrid_evidence = pd.DataFrame(evidence_rows)
hybrid_evidence.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_model_evidence.csv', index=False
)
print(
    hybrid_evidence.sort_values(
        'posterior_model_probability', ascending=False
    ).head(20).to_string(index=False)
)

hybrid_coefficient_table = pd.DataFrame([
    row
    for target_results in hybrid_interpolations.values()
    for result in target_results.values()
    for row in hybrid_parameter_records(result)
])
hybrid_coefficient_table.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_fitted_coefficients.csv', index=False
)
coefficient_text_path = HYBRID_OUTPUT_ROOT / 'hybrid_fitted_coefficients.txt'
with coefficient_text_path.open('w') as stream:
    for (window, operator, flow_time, model), rows in hybrid_coefficient_table.groupby(
        ['window', 'operator', 'flow_time', 'model'], sort=False
    ):
        first = rows.iloc[0]
        stream.write(
            f'Window {window}\nOperator {operator}\nFlow time {flow_time:g}\nModel {model}\n'
            f'log evidence = {first.log_evidence:.10g}\n'
            f'model probability = {first.model_probability:.10g}\n'
        )
        stream.write(rows[[
            'operator', 'flow_time', 'parameter', 'mean', 'sdev'
        ]].to_string(
            index=False,
            formatters={
                'flow_time': lambda value: f'{value:g}',
                'mean': lambda value: f'{value:.10g}',
                'sdev': lambda value: f'{value:.6g}',
            },
        ))
        stream.write('\n\n')
print(
    f'Saved {len(hybrid_coefficient_table)} coefficient rows to',
    coefficient_text_path,
)
# Diagnose endpoint compatibility without altering any fitted coefficient.
gap_rows = []
for window, target_results in hybrid_interpolations.items():
    for key, result in target_results.items():
        for name, candidate_fit in result.candidate_fits.items():
            candidate = candidate_fit.candidate
            if candidate.strong_form != 'eq7_linear' or candidate.continuity != 1:
                continue
            a, b, y0, d0, y1, d1 = (
                betafn.fitting.hybrid._rational_gap_endpoint_data(
                    bf, candidate_fit.fit.p, candidate,
                    candidate_fit.targets[key], 0,
                )
            )
            secant = float(gv.mean((y1-y0)/(b-a)))
            gap_rows.append({
                'window': window_tag(window), 'operator': key[1],
                'flow_time': float(key[2]), 'model': name,
                'bridge_start_g2': a, 'bridge_end_g2': b,
                'weak_slope': float(gv.mean(d0)),
                'strong_slope': float(gv.mean(d1)),
                'secant': secant,
                'monotone_endpoint_compatible': (
                    secant != 0.0 and gv.mean(d0)*secant >= 0.0
                    and gv.mean(d1)*secant >= 0.0
                ),
            })
hybrid_gap_diagnostics = pd.DataFrame(gap_rows)
hybrid_gap_diagnostics.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_gap_diagnostics.csv', index=False,
)
if not hybrid_gap_diagnostics.empty:
    conflicts = hybrid_gap_diagnostics[
        ~hybrid_gap_diagnostics.monotone_endpoint_compatible
    ]
    print(f'Gap endpoint slope conflicts: {len(conflicts)}/{len(gap_rows)}')
    if not conflicts.empty:
        print(conflicts.to_string(index=False))
progress_done('independent flow-time hybrid interpolation and evidence')


progress_start('candidate and model-averaged interpolation plots')
central_interpolations = hybrid_interpolations[CENTRAL_WINDOW]

# Evidence weights are specific to one operator and one flow time.
fig, axes = plt.subplots(
    1, len(OBSERVABLES), figsize=(7 * len(OBSERVABLES), 6), squeeze=False,
)
for ax, operator in zip(axes.ravel(), OBSERVABLES):
    keys = sorted(
        [key for key in central_interpolations if key[1] == operator],
        key=lambda key: float(key[2]),
    )
    for model in central_interpolations[keys[0]].weights:
        ax.plot(
            [float(key[2]) for key in keys],
            [central_interpolations[key].weights[model] for key in keys],
            lw=1.4, label=model,
        )
    ax.set_xlabel(r'$t/a^2$')
    ax.set_ylabel('posterior model probability')
    ax.set_title(OP_LABELS[operator])
axes.ravel()[-1].legend(frameon=False, fontsize=7)
fig.tight_layout()
base = HYBRID_OUTPUT_ROOT / 'hybrid_model_weights'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

continuity_colors = {1: 'tab:blue', 2: 'tab:orange'}
continuity_styles = {1: '-', 2: '--'}
candidate_plot_jobs = []
candidate_group_keys = set()
for operator in OBSERVABLES:
    operator_keys = sorted(
        [key for key in central_interpolations if key[1] == operator],
        key=lambda key: float(key[2]),
    )
    selected_indices = sorted(set((0, len(operator_keys) // 2, len(operator_keys) - 1)))
    selected_keys = tuple(operator_keys[index] for index in selected_indices)
    interpolation_result = central_interpolations[selected_keys[0]]
    candidate_groups = {}
    for name, candidate_fit in interpolation_result.candidate_fits.items():
        candidate = candidate_fit.candidate
        group_key = (
            candidate.weak_family, candidate.weak_order,
            candidate.strong_form,
            candidate.numerator_order, candidate.denominator_order,
            candidate.match_fraction,
            candidate.match_lower, candidate.match_upper,
        )
        candidate_groups.setdefault(group_key, []).append(name)
        candidate_group_keys.add(group_key)
    candidate_plot_jobs.extend(
        (operator, selected_keys, group_key, names)
        for group_key, names in candidate_groups.items()
    )
for plot_index, (
    operator, selected_keys, group_key, variant_names
) in enumerate(
    candidate_plot_jobs, start=1
):
    fig, axes = plt.subplots(
        1, len(selected_keys), figsize=(6 * len(selected_keys), 5),
        squeeze=False, sharey=True,
    )
    for ax, key in zip(axes.ravel(), selected_keys):
        interpolation_result = central_interpolations[key]
        variants = sorted(
            [
                (name, interpolation_result.candidate_fits[name])
                for name in variant_names
            ],
            key=lambda item: item[1].candidate.continuity,
        )
        reference_candidate = variants[0][1]
        target = reference_candidate.targets[key]
        xmin, xmax = min(gv.mean(target.x)), max(gv.mean(target.x))
        xgrid = np.unique(np.r_[
            np.linspace(xmin, xmax, 240),
            target.match_low, target.match_high,
            target.match_low + .05*(target.match_high-target.match_low),
        ])
        join_low = float(target.match_low)
        join_high = float(target.match_high)
        ax.axvspan(join_low, join_high, color='0.5', alpha=.055, zorder=0)
        ax.axvline(join_low, color='0.35', ls=':', lw=1.1, zorder=1)
        ax.axvline(join_high, color='0.35', ls='--', lw=1.1, zorder=1)
        axis_transform = ax.get_xaxis_transform()
        for position, region_label in (
            (.5*(xmin+join_low), 'weak branch'),
            (.5*(join_low+join_high), 'smooth gap'),
            (.5*(join_high+xmax), 'Eq. (7)'),
        ):
            ax.text(
                position, .025, region_label, transform=axis_transform,
                ha='center', va='bottom', fontsize=8.5, color='0.25',
            )
        ax.errorbar(
            gv.mean(target.x), gv.mean(target.y),
            xerr=gv.sdev(target.x), yerr=gv.sdev(target.y),
            fmt='o', ms=4, capsize=2, color='black', alpha=.70,
            label='infinite-volume data',
        )
        for name, candidate_fit in variants:
            continuity = candidate_fit.candidate.continuity
            prediction = np.asarray(
                betafn.fitting.hybrid.evaluate_hybrid_target(
                    candidate_fit, bf, key, xgrid
                ), dtype=object,
            )
            mean, sdev = gv.mean(prediction), gv.sdev(prediction)
            color = continuity_colors.get(continuity, f'C{continuity % 10}')
            if candidate_fit.candidate.strong_form == 'eq7_linear':
                label = (
                    'C1 rational gap bridge; '
                    f'P={interpolation_result.weights[name]:.3f}'
                )
            elif candidate_fit.candidate.strong_form == 'eq7_hermite':
                label = (
                    'C1 Hermite bridge; '
                    f'P={interpolation_result.weights[name]:.3f}'
                )
            elif candidate_fit.candidate.strong_form == 'eq7_blend':
                label = (
                    'C2 gap blend; '
                    f'P={interpolation_result.weights[name]:.3f}'
                )
            else:
                fitted_fraction = hybrid_match_fraction(candidate_fit)
                label = (
                    f'C{continuity}; '
                    f'f={gv.mean(fitted_fraction):.3f}'
                    f'({gv.sdev(fitted_fraction):.3f}); '
                    f'P={interpolation_result.weights[name]:.3f}'
                )
            ax.plot(
                xgrid, mean, color=color,
                ls=continuity_styles.get(continuity, '-'), label=label,
            )
            ax.fill_between(
                xgrid, mean-sdev, mean+sdev, color=color, alpha=.14,
            )
        ax.set_title(rf'$t/a^2={float(key[2]):g}$')
        ax.set_xlabel(r'$g^2_{\rm GF}$')
    axes.ravel()[0].set_ylabel(r'$\beta_{\rm GF}$')
    axes.ravel()[0].legend(frameon=False, fontsize=10)
    (family, order, strong_form, numerator, denominator, fraction,
     match_lower, match_upper) = group_key
    if strong_form == 'eq7_linear':
        transition_description = r'C1 gap bridge after $\beta_b=8$'
    elif strong_form == 'eq7_hermite':
        transition_description = r'C1 Hermite bridge across $\beta_b=8\to7.5$ gap'
    elif strong_form == 'eq7_blend':
        transition_description = r'C2 blend across $\beta_b=8\to7.5$ gap'
    elif fraction is None:
        transition_description = (
            f'match fitted in [{match_lower:.2f}, {match_upper:.2f}]'
        )
    else:
        transition_description = f'match={fraction:.2f}'
    fig.suptitle(
        f'{OP_LABELS[operator]}: {family} order {order}, '
        f'{strong_form} [{numerator}/{denominator}], '
        + transition_description
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    fraction_tag = (
        'gap_8p00_7p50'
        if strong_form in ('eq7_blend', 'eq7_hermite', 'eq7_linear')
        else (
            f'fitted_{match_lower:.2f}_{match_upper:.2f}'.replace('.', 'p')
            if fraction is None else f'{fraction:.2f}'.replace('.', 'p')
        )
    )
    base = HYBRID_OUTPUT_ROOT / (
        f'hybrid_candidate_interpolation_{family}_o{order}_'
        f'{strong_form}_{numerator}_{denominator}_match_{fraction_tag}_{operator}'
    )
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(
        f'[candidate plot {plot_index}/{len(candidate_plot_jobs)}] '
        f'{base.name}',
        flush=True,
    )

for operator in OBSERVABLES:
    keys = sorted(
        [key for key in central_interpolations if key[1] == operator],
        key=lambda key: float(key[2]),
    )
    selected_indices = sorted(set((0, len(keys) // 2, len(keys) - 1)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(.12, .88, len(selected_indices)))
    for color, selected_index in zip(colors, selected_indices):
        key = keys[selected_index]
        interpolation_result = central_interpolations[key]
        reference_fit = next(iter(interpolation_result.candidate_fits.values()))
        target = reference_fit.targets[key]
        xmin, xmax = min(gv.mean(target.x)), max(gv.mean(target.x))
        xgrid = np.unique(np.r_[
            np.linspace(xmin, xmax, 240),
            target.match_low, target.match_high,
            target.match_low + .05*(target.match_high-target.match_low),
        ])
        predictions = {
            name: np.asarray(
                betafn.fitting.hybrid.evaluate_hybrid_target(
                    fit, bf, key, xgrid
                ), dtype=object,
            )
            for name, fit in interpolation_result.candidate_fits.items()
        }
        average = model_average_predictions(
            predictions, interpolation_result.weights
        )
        ax.plot(xgrid, average.mean, color=color, label=rf'$t/a^2={float(key[2]):g}$')
        sigma = np.sqrt(np.clip(np.diag(average.covariance), 0.0, None))
        ax.fill_between(xgrid, average.mean - sigma, average.mean + sigma, color=color, alpha=.18)
        ax.errorbar(
            gv.mean(target.x), gv.mean(target.y),
            xerr=gv.sdev(target.x), yerr=gv.sdev(target.y),
            fmt='o', ms=4, capsize=2, color=color, alpha=.75,
        )
    ax.set_xlabel(r'$g^2_{\rm GF}$'); ax.set_ylabel(r'$\beta_{\rm GF}$')
    ax.set_title(f'{OP_LABELS[operator]}: weak/strong BMA interpolation')
    ax.legend(frameon=False)
    fig.tight_layout()
    base = HYBRID_OUTPUT_ROOT / f'hybrid_bma_interpolation_{operator}'
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

progress_done('candidate and model-averaged interpolation plots')

progress_start('candidate and model-averaged beta/g^4 interpolation plots')

def hybrid_variant_label(candidate_fit, interpolation_result, name):
    candidate = candidate_fit.candidate
    if candidate.strong_form == 'eq7_linear':
        prefix = 'C1 rational gap bridge'
    elif candidate.strong_form == 'eq7_hermite':
        prefix = 'C1 Hermite bridge'
    elif candidate.strong_form == 'eq7_blend':
        prefix = 'C2 gap blend'
    else:
        fraction = hybrid_match_fraction(candidate_fit)
        prefix = (
            f'C{candidate.continuity}; '
            f'f={gv.mean(fraction):.3f}({gv.sdev(fraction):.3f})'
        )
    return prefix + f'; P={interpolation_result.weights[name]:.3f}'


for plot_index, (
    operator, selected_keys, group_key, variant_names
) in enumerate(candidate_plot_jobs, start=1):
    fig, axes = plt.subplots(
        1, len(selected_keys), figsize=(6*len(selected_keys), 5),
        squeeze=False, sharey=True,
    )
    for ax, key in zip(axes.ravel(), selected_keys):
        interpolation_result = central_interpolations[key]
        variants = sorted(
            [(name, interpolation_result.candidate_fits[name])
             for name in variant_names],
            key=lambda item: item[1].candidate.continuity,
        )
        reference_fit = variants[0][1]
        target = reference_fit.targets[key]
        target_x = np.asarray(target.x, dtype=object)
        target_ratio = np.asarray(target.y, dtype=object) / target_x**2
        xmin, xmax = min(gv.mean(target_x)), max(gv.mean(target_x))
        xgrid = np.unique(np.r_[
            np.linspace(xmin, xmax, 240),
            target.match_low, target.match_high,
            target.match_low + .05*(target.match_high-target.match_low),
        ])
        join_low, join_high = float(target.match_low), float(target.match_high)
        ax.axvspan(join_low, join_high, color='0.5', alpha=.055, zorder=0)
        ax.axvline(join_low, color='0.35', ls=':', lw=1.1, zorder=1)
        ax.axvline(join_high, color='0.35', ls='--', lw=1.1, zorder=1)
        ax.errorbar(
            gv.mean(target_x), gv.mean(target_ratio),
            xerr=gv.sdev(target_x), yerr=gv.sdev(target_ratio),
            fmt='o', ms=4, capsize=2, color='black', alpha=.70,
            label='infinite-volume data',
        )
        for name, candidate_fit in variants:
            continuity = candidate_fit.candidate.continuity
            prediction = np.asarray(
                betafn.fitting.hybrid.evaluate_hybrid_target(
                    candidate_fit, bf, key, xgrid
                ), dtype=object,
            ) / xgrid**2
            mean, sdev = gv.mean(prediction), gv.sdev(prediction)
            color = continuity_colors.get(continuity, f'C{continuity % 10}')
            ax.plot(
                xgrid, mean, color=color,
                ls=continuity_styles.get(continuity, '-'),
                label=hybrid_variant_label(
                    candidate_fit, interpolation_result, name
                ),
            )
            ax.fill_between(
                xgrid, mean-sdev, mean+sdev, color=color, alpha=.14,
            )
        ax.set_title(rf'$t/a^2={float(key[2]):g}$')
        ax.set_xlabel(r'$g^2_{\rm GF}$')
    axes.ravel()[0].set_ylabel(r'$\beta_{\rm GF}/g_{\rm GF}^4$')
    axes.ravel()[0].legend(frameon=False, fontsize=10)
    (family, order, strong_form, numerator, denominator, fraction,
     match_lower, match_upper) = group_key
    fraction_tag = (
        'gap_8p00_7p50'
        if strong_form in ('eq7_blend', 'eq7_hermite', 'eq7_linear')
        else (
            f'fitted_{match_lower:.2f}_{match_upper:.2f}'.replace('.', 'p')
            if fraction is None else f'{fraction:.2f}'.replace('.', 'p')
        )
    )
    fig.suptitle(
        f'{OP_LABELS[operator]}: {family} order {order}, '
        f'{strong_form} [{numerator}/{denominator}]'
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    base = HYBRID_OUTPUT_ROOT / (
        f'hybrid_candidate_interpolation_beta_over_g4_{family}_o{order}_'
        f'{strong_form}_{numerator}_{denominator}_match_{fraction_tag}_{operator}'
    )
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(
        f'[beta/g^4 candidate plot {plot_index}/{len(candidate_plot_jobs)}] '
        f'{base.name}', flush=True,
    )

for operator in OBSERVABLES:
    keys = sorted(
        [key for key in central_interpolations if key[1] == operator],
        key=lambda key: float(key[2]),
    )
    selected_indices = sorted(set((0, len(keys)//2, len(keys)-1)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(.12, .88, len(selected_indices)))
    for color, selected_index in zip(colors, selected_indices):
        key = keys[selected_index]
        interpolation_result = central_interpolations[key]
        reference_fit = next(iter(interpolation_result.candidate_fits.values()))
        target = reference_fit.targets[key]
        target_x = np.asarray(target.x, dtype=object)
        target_ratio = np.asarray(target.y, dtype=object) / target_x**2
        xmin, xmax = min(gv.mean(target_x)), max(gv.mean(target_x))
        xgrid = np.unique(np.r_[
            np.linspace(xmin, xmax, 240),
            target.match_low, target.match_high,
            target.match_low + .05*(target.match_high-target.match_low),
        ])
        predictions = {
            name: np.asarray(
                betafn.fitting.hybrid.evaluate_hybrid_target(
                    fit, bf, key, xgrid
                ), dtype=object,
            ) / xgrid**2
            for name, fit in interpolation_result.candidate_fits.items()
        }
        average = model_average_predictions(
            predictions, interpolation_result.weights
        )
        ax.plot(
            xgrid, average.mean, color=color,
            label=rf'$t/a^2={float(key[2]):g}$',
        )
        sigma = np.sqrt(np.clip(np.diag(average.covariance), 0.0, None))
        ax.fill_between(
            xgrid, average.mean-sigma, average.mean+sigma,
            color=color, alpha=.18,
        )
        ax.errorbar(
            gv.mean(target_x), gv.mean(target_ratio),
            xerr=gv.sdev(target_x), yerr=gv.sdev(target_ratio),
            fmt='o', ms=4, capsize=2, color=color, alpha=.75,
        )
    ax.set_xlabel(r'$g^2_{\rm GF}$')
    ax.set_ylabel(r'$\beta_{\rm GF}/g_{\rm GF}^4$')
    ax.set_title(f'{OP_LABELS[operator]}: weak/strong BMA interpolation')
    ax.legend(frameon=False)
    fig.tight_layout()
    base = HYBRID_OUTPUT_ROOT / f'hybrid_bma_interpolation_beta_over_g4_{operator}'
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

progress_done('candidate and model-averaged beta/g^4 interpolation plots')

progress_start('continuum extrapolation of flow-time-local BMAs')
hybrid_results = {}
hybrid_grid = np.round(
    np.arange(G2_GRID[0], G2_GRID[1] + 0.5 * G2_GRID[2], G2_GRID[2]),
    10,
)
for window in WINDOWS:
    interpolation_results = hybrid_interpolations[window]
    print('continuum extrapolation:', window, 'correlated')
    hybrid_results[(window, 'correlated')] = run_hybrid_bma_by_target(
        bf, window, FLOW, OBSERVABLES,
        weak_couplings=WEAK_COUPLINGS,
        strong_couplings=STRONG_COUPLINGS,
        g2_grid=hybrid_grid,
        candidates=HYBRID_CANDIDATES,
        cov_mode='empirical', tau0=config.tau0,
        prefit_interpolations=interpolation_results,
        max_workers=HYBRID_CONTINUUM_WORKERS,
        **HYBRID_PRIOR_WIDTHS,
    )
    print('continuum extrapolation:', window, 'diagonal')
    hybrid_results[(window, 'diagonal')] = run_hybrid_bma_by_target(
        bf, window, FLOW, OBSERVABLES,
        weak_couplings=WEAK_COUPLINGS,
        strong_couplings=STRONG_COUPLINGS,
        g2_grid=hybrid_grid,
        candidates=HYBRID_CANDIDATES,
        cov_mode='diagonal', tau0=config.tau0,
        prefit_interpolations=interpolation_results,
        max_workers=HYBRID_CONTINUUM_WORKERS,
        **HYBRID_PRIOR_WIDTHS,
    )

progress_done('continuum extrapolation of flow-time-local BMAs')

progress_start('model-averaged continuum summary plots')
correction_label = (
    'TLN' if config.correction in ('tln', 'tree-level-normalization')
    else 'No TLN'
)
for window in WINDOWS:
    for mode in ('correlated', 'diagonal'):
        result = hybrid_results[(window, mode)]
        mode_label = (
            'Correlated continuum' if mode == 'correlated'
            else 'Diagonal continuum'
        )
        for divide_by_g4, stem, ylabel in (
            (False, 'hybrid_bma_continuum_beta_vs_g2',
             r'$\beta_{\rm GF}$'),
            (True, 'hybrid_bma_continuum_beta_over_g4_vs_g2',
             r'$\beta_{\rm GF}/g_{\rm GF}^4$'),
        ):
            fig, ax = plt.subplots(figsize=(8, 6))
            for operator in OBSERVABLES:
                average = result.averages[operator]
                x = result.model_variance_by_choice[operator]['g2']
                y = average.prediction / x**2 if divide_by_g4 else average.prediction
                order = np.argsort(x)
                x, y = x[order], y[order]
                ax.plot(
                    x, gv.mean(y), color=OP_COLORS[operator], lw=1.8,
                    label=OP_LABELS[operator], zorder=3,
                )
                ax.fill_between(
                    x, gv.mean(y)-gv.sdev(y), gv.mean(y)+gv.sdev(y),
                    color=OP_COLORS[operator], alpha=.18, linewidth=0,
                    zorder=2,
                )
            plot_xmax = max(G2_GRID[1], G2_T0) + .25
            xp = np.linspace(.001, plot_xmax, 900)
            for loops, linestyle, label in (
                (1, '-', '1-loop universal'),
                (2, '--', '2-loop universal'),
                (3, '-.', '3-loop gradient flow'),
            ):
                perturbative_ratio = pt_over_g4(xp, loops)
                perturbative_curve = (
                    perturbative_ratio if divide_by_g4
                    else xp**2 * perturbative_ratio
                )
                ax.plot(
                    xp, perturbative_curve, color='gray', ls=linestyle,
                    lw=1.5, label=label, zorder=1,
                )
            ax.text(
                .50, .52, 'Preliminary', transform=ax.transAxes,
                fontsize=42, color='gray', alpha=.25, ha='center',
                va='center', rotation=30, zorder=20, clip_on=True,
            )
            ax.text(
                .58, .15,
                'hybridBMA\n' + mode_label + '\n'
                + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
                + '\n' + correction_label + '\n' + ylabel,
                transform=ax.transAxes, fontsize=10.5, color='gray',
                alpha=.82, ha='center', va='center',
                fontweight='bold', zorder=20, clip_on=True,
            )
            if not divide_by_g4:
                ax.axvline(
                    G2_T0, color='black', ls=':', lw=1.5,
                    label=(
                        rf'$g^2_{{\mathrm{{GF}}}}(t_0)'
                        rf'=\frac{{8}}{{5}}\pi^2={G2_T0:.9f}$'
                    ), zorder=1,
                )
                ax.set_ylim(-13.0, .5)
            else:
                ax.set_ylim(-.065, -.035)
            ax.set(xlim=(0, plot_xmax), xlabel=r'$g^2_{\rm GF}$', ylabel=ylabel)
            ax.legend(ncol=2, frameon=False, fontsize=10)
            if window == CENTRAL_WINDOW and mode == 'correlated':
                central_base = HYBRID_OUTPUT_ROOT / stem
                fig.savefig(central_base.with_suffix('.pdf'), bbox_inches='tight')
                fig.savefig(
                    central_base.with_suffix('.png'), dpi=300,
                    bbox_inches='tight',
                )
            save_figure(fig, window, stem, mode)

progress_done('model-averaged continuum summary plots')

progress_start('hybrid beta-function error budget')
central = hybrid_results[(CENTRAL_WINDOW, 'correlated')]
diagonal = hybrid_results[(CENTRAL_WINDOW, 'diagonal')]
error_rows = []
for operator in OBSERVABLES:
    average = central.averages[operator]
    x = central.model_variance_by_choice[operator]['g2']
    diagonal_average = diagonal.averages[operator]
    statistical_variance = np.diag(average.within_covariance)
    components = central.model_variance_by_choice[operator]
    covariance_variance = (diagonal_average.mean - average.mean) ** 2

    window_means = []
    for window in WINDOWS:
        result = hybrid_results[(window, 'correlated')]
        candidate_x = result.model_variance_by_choice[operator]['g2']
        mapping = {round(float(g), 12): value for g, value in zip(candidate_x, result.averages[operator].mean)}
        window_means.append(mapping)

    for index, g2 in enumerate(x):
        key = round(float(g2), 12)
        available_window_values = [values[key] for values in window_means if key in values]
        window_variance = (
            float(np.var(available_window_values, ddof=0))
            if len(available_window_values) > 1 else np.nan
        )
        model_component_variances = {
            axis: float(values[index])
            for axis, values in components.items() if axis != 'g2'
        }
        total_variance = float(statistical_variance[index]) + sum(model_component_variances.values())
        total_variance += float(covariance_variance[index])
        if np.isfinite(window_variance):
            total_variance += window_variance
        mean = float(average.mean[index])
        row = {
            'operator': operator, 'g2': float(g2), 'beta_mean': mean,
            'statistical_percent': float(relative_partial_uncertainty_percent(statistical_variance[index], mean)),
            'continuum_covariance_percent': float(relative_partial_uncertainty_percent(covariance_variance[index], mean)),
            'continuum_window_percent': (
                float(relative_partial_uncertainty_percent(window_variance, mean))
                if np.isfinite(window_variance) else np.nan
            ),
            'chiral_model_percent': np.nan,
            'infinite_volume_model_percent': np.nan,
            'total_available_percent': float(relative_partial_uncertainty_percent(total_variance, mean)),
        }
        for axis, variance in model_component_variances.items():
            row[f'interpolation_{axis}_percent'] = float(
                relative_partial_uncertainty_percent(variance, mean)
            )
        error_rows.append(row)

hybrid_beta_error_budget = pd.DataFrame(error_rows)
hybrid_beta_error_budget.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_beta_error_budget.csv', index=False
)
component_columns = [
    column for column in hybrid_beta_error_budget
    if column.endswith('_percent')
    and column not in ('total_available_percent',)
    and hybrid_beta_error_budget[column].notna().any()
]
fig, axes = plt.subplots(1, len(OBSERVABLES), figsize=(17, 5), sharey=True)
for ax, operator in zip(np.atleast_1d(axes), OBSERVABLES):
    rows = hybrid_beta_error_budget.query('operator == @operator')
    for column in component_columns:
        ax.plot(rows.g2, rows[column], lw=1.2, label=column.replace('_percent', '').replace('_', ' '))
    ax.plot(rows.g2, rows.total_available_percent, color='black', lw=2.2, label='total available')
    ax.set_title(OP_LABELS[operator]); ax.set_xlabel(r'$g^2_{\rm GF}$'); ax.set_yscale('log')
axes[0].set_ylabel(r'partial relative uncertainty [\%]')
axes[-1].legend(frameon=False, fontsize=8)
fig.tight_layout()
base = HYBRID_OUTPUT_ROOT / 'hybrid_beta_error_budget'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)
print(hybrid_beta_error_budget)

progress_done('hybrid beta-function error budget')

progress_start('generalized continuum goodness of fit')
gof_rows = []
for (window, mode), result in hybrid_results.items():
    for operator in OBSERVABLES:
        for g2, entry in zip(result.g2, result.continuum_quality[operator]):
            if entry is None:
                continue
            gof_rows.append({
                'window': window_tag(window), 'mode': mode,
                'interpolation': 'flow_time_local_bma',
                'operator': operator, 'g2': float(g2), **entry,
            })
hybrid_continuum_gof = pd.DataFrame(gof_rows)
hybrid_continuum_gof.to_csv(
    HYBRID_OUTPUT_ROOT / 'hybrid_continuum_generalized_gof.csv', index=False
)
print(hybrid_continuum_gof.head())

progress_done('generalized continuum goodness of fit')

progress_start('continuum extrapolation diagnostic plots')
correction_label = (
    'TLN' if config.correction in ('tln', 'tree-level-normalization')
    else 'No TLN'
)
for window in WINDOWS:
    for mode in ('correlated', 'diagonal'):
        result = hybrid_results[(window, mode)]
        mode_label = (
            'Correlated continuum' if mode == 'correlated'
            else 'Diagonal continuum'
        )
        targets_to_plot = tuple(
            g for g in TARGET_G2 if result.g2[0] <= g <= result.g2[-1]
        )
        for divide_by_g4, stem, ylabel in (
            (False, 'hybrid_bma_continuum_a2_over_t_beta',
             r'$\beta_{\rm GF}$'),
            (True, 'hybrid_bma_continuum_a2_over_t_beta_over_g4',
             r'$\beta_{\rm GF}/g_{\rm GF}^4$'),
        ):
            fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
            for ax, requested_g2 in zip(axes.ravel(), targets_to_plot):
                panel_g2 = None
                for operator in OBSERVABLES:
                    grid_index = int(np.argmin(
                        np.abs(result.g2-requested_g2)
                    ))
                    g2 = float(result.g2[grid_index])
                    params = result.continuum_parameters[operator][grid_index]
                    points = result.continuum_points[operator][grid_index]
                    if params is None or points is None:
                        continue
                    panel_g2 = g2
                    normalization = g2**2 if divide_by_g4 else 1.0
                    xdata = points['x']
                    ydata = points['y'] / normalization
                    xline = np.linspace(0.0, 1.05*max(xdata), 200)
                    line = (
                        params['beta'][0] + params['slope'][0]*xline
                    ) / normalization
                    ax.plot(
                        xline, gv.mean(line), color=OP_COLORS[operator],
                        lw=1.5, zorder=2,
                    )
                    ax.fill_between(
                        xline, gv.mean(line)-gv.sdev(line),
                        gv.mean(line)+gv.sdev(line),
                        color=OP_COLORS[operator], alpha=.15, zorder=1,
                    )
                    ax.errorbar(
                        xdata, gv.mean(ydata), yerr=gv.sdev(ydata),
                        fmt='o', ms=5, capsize=3,
                        color=OP_COLORS[operator],
                        markerfacecolor=OP_COLORS[operator],
                        markeredgecolor=OP_COLORS[operator],
                        label=OP_LABELS[operator], zorder=4,
                    )
                if panel_g2 is not None:
                    ax.text(
                        .04, .94, rf'$g^2_{{\rm GF}}={panel_g2:g}$',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=13, zorder=20,
                    )
                ax.set_xlabel(r'$a^2/t$')
            for ax in axes.ravel()[len(targets_to_plot):]:
                ax.set_visible(False)
            axes.ravel()[0].legend(frameon=False, fontsize=11)
            fig.supylabel(ylabel)
            fig.text(
                .50, .52, 'Preliminary', fontsize=58, color='gray',
                alpha=.18, ha='center', va='center', rotation=30,
                zorder=100,
            )
            fig.text(
                .50, .025,
                'hybridBMA; ' + mode_label + '; '
                + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
                + '; ' + correction_label
                + '; filled points are included in the fit',
                fontsize=11, color='gray', alpha=.9, ha='center',
                va='bottom', fontweight='bold', zorder=100,
            )
            fig.tight_layout(rect=(.025, .075, 1, 1))
            if window == CENTRAL_WINDOW and mode == 'correlated':
                central_base = HYBRID_OUTPUT_ROOT / stem
                fig.savefig(central_base.with_suffix('.pdf'), bbox_inches='tight')
                fig.savefig(
                    central_base.with_suffix('.png'), dpi=300,
                    bbox_inches='tight',
                )
            save_figure(fig, window, stem, mode)

progress_done('continuum extrapolation diagnostic plots')

progress_start('blinded Lambda analysis')
from betafn.blinding import add_blinded_watermark, blind_lambda_estimate
from betafn.weak_coupling import constrained_smooth_match, lambda_parameter_from_smooth_beta

central = hybrid_results[(CENTRAL_WINDOW, 'correlated')]
hybrid_lambda_rows = []
for operator in OBSERVABLES:
    continuum_average = central.averages[operator]
    x = central.model_variance_by_choice[operator]['g2']
    beta = continuum_average.prediction
    if x[-1] < G2_T0 - 1e-10:
        raise RuntimeError(
            f'{operator}: continuum curve does not reach g^2(t0)'
        )
    ratio = beta / x**2
    smooth = constrained_smooth_match(
        x, ratio, pt_over_g4,
        match_window=(max(float(x[0]), .9), 1.2),
        correction_powers=(3, 4),
    )
    estimate = lambda_parameter_from_smooth_beta(
        x, ratio, smooth, bf.perturbative_beta_function,
        reference_g2=G2_T0,
    )
    lambda_gf = estimate['lambda_gf_over_mu']

    within_beta = np.asarray(gv.gvar(
        continuum_average.mean,
        continuum_average.within_covariance,
    ), dtype=object)
    within_smooth = constrained_smooth_match(
        x, within_beta / x**2, pt_over_g4,
        match_window=(max(float(x[0]), .9), 1.2),
        correction_powers=(3, 4),
    )
    within_estimate = lambda_parameter_from_smooth_beta(
        x, within_beta / x**2, within_smooth,
        bf.perturbative_beta_function, reference_g2=G2_T0,
    )['lambda_gf_over_mu']
    within_lambda_variance = float(gv.sdev(within_estimate) ** 2)
    total_lambda_variance = float(gv.sdev(lambda_gf) ** 2)
    between_lambda_variance = max(
        total_lambda_variance - within_lambda_variance, 0.0
    )
    report = {
        'lambda_gf_over_mu': {
            'central': float(gv.mean(lambda_gf)),
            'plus_sigma': float(gv.mean(lambda_gf) + gv.sdev(lambda_gf)),
            'minus_sigma': float(gv.mean(lambda_gf) - gv.sdev(lambda_gf)),
        },
        'lambda_msbar_over_mu': {},
        'lambda_msbar_over_lambda_gf': float(bf.perturbative_beta_function.lambda_msbar_over_lambda_gf),
    }
    report['lambda_msbar_over_mu'] = {
        label: report['lambda_msbar_over_lambda_gf'] * value
        for label, value in report['lambda_gf_over_mu'].items()
    }
    blinded = blind_lambda_estimate(report)
    values = blinded['lambda_gf_over_mu']
    hybrid_lambda_rows.append({
        'operator': operator, 'operator_label': OP_LABELS[operator],
        'g2_reference': G2_T0, 'lambda_values_blinded': True,
        'sqrt8t0_lambda_GF_central': values['central'],
        'sqrt8t0_lambda_GF_low': min(values['plus_sigma'], values['minus_sigma']),
        'sqrt8t0_lambda_GF_high': max(values['plus_sigma'], values['minus_sigma']),
        'within_model_variance': within_lambda_variance,
        'between_model_variance': between_lambda_variance,
    })

hybrid_lambda_table = pd.DataFrame(hybrid_lambda_rows)
hybrid_lambda_table.to_csv(HYBRID_OUTPUT_ROOT / 'hybrid_bma_lambda_blinded.csv', index=False)
fig, ax = plt.subplots(figsize=(8, 5.5))
for index, row in hybrid_lambda_table.iterrows():
    center = row.sqrt8t0_lambda_GF_central
    ax.errorbar(
        index, center,
        yerr=[[center-row.sqrt8t0_lambda_GF_low], [row.sqrt8t0_lambda_GF_high-center]],
        fmt='o', capsize=4, color=OP_COLORS[row.operator], label=row.operator_label,
    )
ax.set_xticks(range(len(hybrid_lambda_table)), hybrid_lambda_table.operator_label)
ax.set_ylabel(r'$\sqrt{8t_0}\,\Lambda_{\rm GF}$')
ax.set_title('BLINDED hybrid-BMA result')
add_blinded_watermark(fig)
fig.tight_layout()
base = HYBRID_OUTPUT_ROOT / 'hybrid_bma_lambda_blinded'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)
print(hybrid_lambda_table)

progress_done('blinded Lambda analysis')

progress_start('optional hybrid thinning study')
hybrid_thinning_results = {}
if RUN_HYBRID_THINNING:
    hybrid_grid = np.round(
        np.arange(G2_GRID[0], G2_GRID[1] + 0.5 * G2_GRID[2], G2_GRID[2]), 10
    )
    for spacing in HYBRID_THINNING_SPACINGS:
        for window in WINDOWS:
            key = (float(spacing), window)
            print('hybrid thinning:', spacing, window)
            hybrid_thinning_results[key] = run_hybrid_bma_by_target(
                bf, window, FLOW, OBSERVABLES,
                weak_couplings=WEAK_COUPLINGS,
                strong_couplings=STRONG_COUPLINGS,
                g2_grid=hybrid_grid,
                candidates=HYBRID_CANDIDATES,
                cov_mode='empirical', tau0=config.tau0,
                flow_time_spacing=spacing,
                max_workers=HYBRID_MAX_WORKERS,
                checkpoint_dir=(
                    HYBRID_CHECKPOINT_ROOT
                    / f'dt_{float(spacing):.2f}'.replace('.', 'p')
                    / window_tag(window)
                ),
                **HYBRID_PRIOR_WIDTHS,
            )
    thinning_rows = []
    for (spacing, window), result in hybrid_thinning_results.items():
        for operator in OBSERVABLES:
            average = result.averages[operator]
            x = result.model_variance_by_choice[operator]['g2']
            for g2, mean, variance in zip(x, average.mean, np.diag(average.covariance)):
                thinning_rows.append({
                    'spacing': spacing, 'window': window_tag(window),
                    'operator': operator, 'g2': float(g2),
                    'beta_mean': float(mean), 'beta_sdev': float(np.sqrt(max(variance, 0.0))),
                })
    pd.DataFrame(thinning_rows).to_csv(
        HYBRID_OUTPUT_ROOT / 'hybrid_thinning_continuum.csv', index=False
    )
else:
    print('Hybrid thinning disabled.')

progress_done('optional hybrid thinning study')


required_outputs = (
    'run_configuration.json',
    'hybrid_model_selection.csv',
    'hybrid_model_evidence.csv',
    'hybrid_fitted_coefficients.csv',
    'hybrid_fitted_coefficients.txt',
    'hybrid_gap_diagnostics.csv',
    'hybrid_model_weights.pdf',
    'hybrid_bma_interpolation_p.pdf',
    'hybrid_bma_interpolation_c.pdf',
    'hybrid_bma_interpolation_s.pdf',
    'hybrid_bma_interpolation_beta_over_g4_p.pdf',
    'hybrid_bma_interpolation_beta_over_g4_c.pdf',
    'hybrid_bma_interpolation_beta_over_g4_s.pdf',
    'hybrid_bma_continuum_beta_vs_g2.pdf',
    'hybrid_bma_continuum_beta_over_g4_vs_g2.pdf',
    'hybrid_beta_error_budget.csv',
    'hybrid_continuum_generalized_gof.csv',
    'hybrid_bma_continuum_a2_over_t_beta.pdf',
    'hybrid_bma_continuum_a2_over_t_beta_over_g4.pdf',
    'hybrid_bma_lambda_blinded.csv',
    'hybrid_bma_lambda_blinded.pdf',
)
missing_outputs = [
    name for name in required_outputs
    if not (HYBRID_OUTPUT_ROOT / name).is_file()
]
for group_key in candidate_group_keys:
    (family, order, strong_form, numerator, denominator, fraction,
     match_lower, match_upper) = group_key
    fraction_tag = (
        'gap_8p00_7p50'
        if strong_form in ('eq7_blend', 'eq7_hermite', 'eq7_linear')
        else (
            f'fitted_{match_lower:.2f}_{match_upper:.2f}'.replace('.', 'p')
            if fraction is None else f'{fraction:.2f}'.replace('.', 'p')
        )
    )
    for operator in OBSERVABLES:
        for quantity_tag in ('', 'beta_over_g4_'):
            name = (
                f'hybrid_candidate_interpolation_{quantity_tag}{family}_o{order}_'
                f'{strong_form}_{numerator}_{denominator}_match_{fraction_tag}_'
                f'{operator}.pdf'
            )
            if not (HYBRID_OUTPUT_ROOT / name).is_file():
                missing_outputs.append(name)
if missing_outputs:
    raise RuntimeError(f'missing required hybrid outputs: {missing_outputs}')

completion_payload = {
    'analysis_id': ANALYSIS_ID,
    'candidate_count': len(HYBRID_CANDIDATES),
    'interpolation_target_count': len(
        hybrid_interpolations[CENTRAL_WINDOW]
    ),
    'interpolation_fit_count': (
        len(hybrid_interpolations[CENTRAL_WINDOW])
        * len(HYBRID_CANDIDATES)
    ),
    'interpolation_treatment': 'independent_by_operator_and_flow_time',
    'cross_flow_time_covariance': 'linear_response_to_shared_gvar_data',
    'hybrid_workers': HYBRID_MAX_WORKERS,
    'hybrid_continuum_workers': HYBRID_CONTINUUM_WORKERS,
    'evidence_rows': int(len(hybrid_evidence)),
    'coefficient_rows': int(len(hybrid_coefficient_table)),
    'candidate_interpolation_plots': int(len(candidate_plot_jobs)),
    'continuum_operators': sorted(central.averages),
    'lambda_rows': int(len(hybrid_lambda_table)),
    'hybrid_thinning': RUN_HYBRID_THINNING,
    'elapsed_seconds': round(time_module.time() - RUN_STARTED, 3),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
}
temporary_completion = HYBRID_OUTPUT_ROOT / 'RUN_COMPLETE.json.tmp'
temporary_completion.write_text(json.dumps(completion_payload, indent=2) + '\n')
temporary_completion.replace(HYBRID_OUTPUT_ROOT / 'RUN_COMPLETE.json')
print(json.dumps(completion_payload, indent=2))
print('hybrid analysis complete:', HYBRID_OUTPUT_ROOT)
