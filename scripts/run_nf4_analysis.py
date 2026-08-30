#!/usr/bin/env python3
# coding: utf-8

# # NF4 continuous-beta-function cluster analysis
# 
# Updated upstream analysis (`2af131f`), Wilson flow, on-the-fly TLN, three-volume infinite-volume limit, and exhaustive integer flow-time-window scan.  The interpolation ansatz is
# $$
# \beta(x)=\beta_{\rm PT}^{(3)}(x)\left[1+\sum_{n=1}^{4}c_n(x/4\pi)^n\right].
# $$
# 
# Every plot family has its own cell. Outputs are organized as `fit4/order_4/t_<min>_<max>/<mode>/`, where mode is `diagonal` or `correlated`. The correlated result uses the upstream kernel covariance. Rectangle measurements are combined at the raw-history level into the Symanzik observable before its dedicated TLN correction.

# In[1]:


from pathlib import Path
import argparse, copy, gc, json, os, shutil, sys
import time as time_module

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import betafn
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

print('betafn package:', Path(betafn.__file__).resolve())
print('repository:', REPO_ROOT)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': False, 'font.family': 'serif',
    'font.serif': ['Computer Modern'], 'font.size': 15,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.labelsize': 20, 'axes.titlesize': 18,
    'legend.fontsize': 11, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'grid.alpha': .35, 'grid.linestyle': '-', 'axes.axisbelow': True,
})


# ## Configuration and complete window catalogue
# 
# Change physics/statistical choices only here. Every downstream cell reads these values.

# In[2]:


parser = argparse.ArgumentParser(description='Run the complete NF4 fit4/fit5/fit6 analysis without Jupyter.')
parser.add_argument('--model', choices=('fit4', 'fit5', 'fit6'), default='fit4')
parser.add_argument('--fit4-order', type=int, default=4, help='Correction order for fit4 only.')
parser.add_argument('--fit4-width', type=float, default=10.0,
                    help='Zero-centered prior width for fit4 correction coefficients.')
parser.add_argument('--data-dir', type=Path, default=os.environ.get('BETAFN_DATA_DIR'))
parser.add_argument('--output-base', type=Path, default=os.environ.get('BETAFN_OUTPUT_BASE', REPO_ROOT / 'hpcc_outputs'))
parser.add_argument('--correction', choices=('tln', 'tree-level-normalization', 'finite-volume', 'none'), default='tln')
parser.add_argument('--binsize', type=int, default=15)
parser.add_argument('--use-gamma-method', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--latex', action='store_true', help='Use external LaTeX for plot text.')
parser.add_argument(
    '--reviewed-weak-coupling',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Generate the Figure-11 matching and extended-interpolant diagnostics.',
)
parser.add_argument('--validate-only', action='store_true', help='Validate configuration/model construction, then exit.')
args = parser.parse_args()
RUN_STARTED = time_module.time()

if args.data_dir is None:
    parser.error('Set --data-dir or BETAFN_DATA_DIR.')
if args.fit4_order < 1:
    parser.error('--fit4-order must be positive.')
if args.fit4_width <= 0:
    parser.error('--fit4-width must be positive.')
if args.latex:
    missing_tex_tools = [name for name in ('latex', 'dvipng') if shutil.which(name) is None]
    if missing_tex_tools:
        parser.error(
            '--latex requires the following executables on PATH: '
            + ', '.join(missing_tex_tools)
            + '. Load the HPCC TeX/TeX-Live module before submitting the job.'
        )

FIT_ID = args.model
CORRECTION = args.correction
DATA_DIR = args.data_dir.expanduser().resolve()
plt.rcParams['text.usetex'] = args.latex
if args.latex:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',
    })

if FIT_ID == 'fit4':
    ORDER = args.fit4_order
    FIT_WIDTH = args.fit4_width
    WIDTH_TAG = f'{FIT_WIDTH:g}'.replace('.', 'p')
    PT_POWERS = tuple(range(1, ORDER + 1))
    MODEL_TAG = f'order_{ORDER}'
    OUTPUT_FAMILY = f'fit4_width{WIDTH_TAG}'
    FIT_WATERMARK = rf'fit4, correction order {ORDER}, prior width {FIT_WIDTH:g}'
elif FIT_ID == 'fit5':
    ORDER = None
    FIT_WIDTH = None
    PT_POWERS = (3,)
    MODEL_TAG = 'pt_preserving_u3'
    OUTPUT_FAMILY = FIT_ID
    FIT_WATERMARK = r'fit5, PT-preserving order $u^3$'
else:
    ORDER = None
    FIT_WIDTH = None
    PT_POWERS = (3, 4)
    MODEL_TAG = 'pt_preserving_u3_u4'
    OUTPUT_FAMILY = FIT_ID
    FIT_WATERMARK = r'fit6, PT-preserving orders $u^3+u^4$'

OUTPUT_ROOT = args.output_base.expanduser().resolve() / OUTPUT_FAMILY / MODEL_TAG
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / 'run_configuration.json').write_text(json.dumps({
    'model': FIT_ID,
    'model_tag': MODEL_TAG,
    'fit4_order': ORDER,
    'fit4_width': FIT_WIDTH,
    'pt_powers': PT_POWERS,
    'data_dir': str(DATA_DIR),
    'output_root': str(OUTPUT_ROOT),
    'correction': CORRECTION,
    'binsize': args.binsize,
    'use_gamma_method': args.use_gamma_method,
    'latex': args.latex,
    'reviewed_weak_coupling': args.reviewed_weak_coupling,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
}, indent=2) + '\n')

COUPLINGS = ('20p0', '18p0', '16p0', '14p0', '12p0', '11p0', '10p0', '9p50', '9p00', '8p50')
VOLUMES = {
    coupling: ('l32l32l32t64', 'l40l40l40t80', 'l48l48l48t96')
    for coupling in COUPLINGS
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

# Exhaustive integer windows used by the scan reference notebook.
TMIN_VALUES = tuple(range(4, 8))
TMAX_LIMIT = 8
WINDOWS = tuple(
    (float(tmin), float(tmax))
    for tmin in TMIN_VALUES
    for tmax in range(tmin + 1, TMAX_LIMIT + 1)
)
CENTRAL_WINDOW = (4.0, 6.0)
G2_GRID = (0.9, 4.9, 0.2)
TARGET_G2 = (1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 4.0)

def window_tag(window):
    return f't_{window[0]:g}_{window[1]:g}'.replace('.', 'p')

def case_dir(window, mode=None):
    path = OUTPUT_ROOT / window_tag(window)
    if mode is not None:
        path = path / mode
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_figure(fig, window, name, mode=None):
    base = case_dir(window, mode) / name
    if FIT_WIDTH is not None:
        fig.text(0.995, 0.005, rf'prior width $={FIT_WIDTH:g}$', ha='right', va='bottom',
                 fontsize=8, color='gray', alpha=.75)
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def beta_value(coupling):
    return float(coupling.replace('p', '.'))

def volume_value(volume):
    return np.prod([float(x) for x in volume.replace('t', 'l').split('l')[1:] if x])

def flow_times(window):
    return [t for t in sorted(bf.ntrp_fits[FLOW][OBSERVABLES[0]], key=float)
            if window[0] <= float(t) <= window[1]]

def pt_over_g4(x, loops):
    x = np.asarray(x, dtype=float)
    pt = bf.perturbative_beta_function
    return sum(-coefficient * x**power / pt.nrm**(power + 1)
               for power, coefficient in enumerate(pt.b[:loops]))

print(f'{len(WINDOWS)} windows:', WINDOWS)
print('output root:', OUTPUT_ROOT)


# ## Analysis construction
# 
# The pulled upstream package computes TLN on demand. No external `.tln` path is supplied. The raw `Es` history is the rectangle measurement and is replaced by `(5/3)Ep-(2/3)Es` before averaging and TLN.

# In[3]:


if not DATA_DIR.is_dir():
    raise FileNotFoundError(DATA_DIR)

bf = betafn.BetaFunction(nf=4)
if FIT_ID == 'fit4':
    interpolation = bf.perturbative_interpolation(
        loops=3, correction_order=ORDER, free_intercept=False,
        width=FIT_WIDTH, xerrors=True,
    )
else:
    def pt_preserving_interpolation(x, p):
        u = np.asarray(x) / bf.perturbative_beta_function.nrm
        correction = 1.0 + sum(p[f'd{power}'][0] * u**power for power in PT_POWERS)
        return bf.perturbative_beta_function(x, loops=3) * correction

    interpolation = betafn.InterpolationSpec(
        fcn=pt_preserving_interpolation,
        prior={f'd{power}': [gv.gvar(0, 10)] for power in PT_POWERS},
        p0={f'd{power}': 0.0 for power in PT_POWERS},
        xerrors=True,
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
    binsize=args.binsize,
    use_gamma_method=args.use_gamma_method,
    process_window=(1.5, 10.5),
    fit_window=(2.0, 10.0),
    error_mode='fit',
    cov_mode='kernel',
    correlated=True,
    verbosity=0,
)
print(pd.DataFrame([config.describe()]).T.rename(columns={0: f'{FIT_ID} setting'}).to_string())
if args.validate_only:
    parameter_names = sorted(interpolation.p0)
    print(f'validation successful: model={FIT_ID}, tag={MODEL_TAG}, parameters={parameter_names}')
    raise SystemExit(0)


# ## Stage 1 — processing with TLN

# In[4]:


bf.run_processing(config)
print('correction:', bf._process_config.correction)
print('processed couplings:', sorted(bf.avg_data, key=beta_value))
print('binsize:', bf._binsize)
print(bf.data_report())


# ## Plot 1 — processed largest-volume data

# In[5]:


fig, ax = plt.subplots(figsize=(8, 6))
for operator in OBSERVABLES:
    first = True
    for coupling in sorted(bf.avg_data, key=beta_value):
        volume = max(bf.avg_data[coupling], key=volume_value)
        _, g2 = bf.processed_series(coupling, volume, '0p00', FLOW, operator, 'g2')
        _, beta = bf.processed_series(coupling, volume, '0p00', FLOW, operator, 'beta')
        ratio = np.asarray([b / x**2 for x, b in zip(g2, beta)], dtype=object)
        ax.errorbar(gv.mean(g2), gv.mean(ratio), yerr=gv.sdev(ratio), fmt='o', ms=2.5,
                    alpha=.6, color=OP_COLORS[operator],
                    label=OP_LABELS[operator] if first else None)
        first = False
ax.set(xlabel=r'$g^2_{GF}$', ylabel=r'$\beta_{GF}/g_{GF}^4$', title='Processed largest-volume data (TLN)')
ax.legend(frameon=False)
base = OUTPUT_ROOT / f'processed_largest_volume_{FIT_ID}'
if FIT_WIDTH is not None:
    fig.text(0.995, 0.005, rf'prior width $={FIT_WIDTH:g}$', ha='right', va='bottom',
             fontsize=8, color='gray', alpha=.75)
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig)


# ## Stages 2–4 — chiral, infinite volume, and fixed fit4 interpolation

# In[6]:


bf.run_chiral(config)
bf.run_infinite_volume(config)
bf.run_interpolation(config)
print(bf.stage_summary('chiral'))
print(bf.stage_summary('infinite_volume'))
print(bf.stage_summary('interpolation'))


def save_interpolation_diagnostics():
    """Save per-fit quality and per-coupling interpolation residuals."""
    detail_rows = []
    for window in WINDOWS:
        for operator in OBSERVABLES:
            times = [
                time for time in sorted(bf.ntrp_fits[FLOW][operator], key=float)
                if window[0] <= float(time) <= window[1]
            ]
            for time in times:
                data = bf.interpolation.fetch('inputs', (FLOW, operator, time))
                params = bf.interpolation.fetch('fits', (FLOW, operator, time))
                qof = bf.interpolation.fetch('quality', (FLOW, operator, time))
                data_x = np.asarray(data.x, dtype=object)
                data_y = np.asarray(data.y, dtype=object)
                fitted_x = np.asarray(params['x'], dtype=object) if 'x' in params else data_x
                prediction = np.asarray([
                    bf.interpolation.model.evaluate(value, params) for value in fitted_x
                ], dtype=object)
                residual = data_y - prediction
                residual_sdev = np.asarray(gv.sdev(residual), dtype=float)
                pull = np.divide(
                    gv.mean(residual), residual_sdev,
                    out=np.full(len(residual), np.nan), where=residual_sdev > 0,
                )
                fit_couplings = [
                    coupling for coupling in COUPLINGS
                    if time in bf.iv_fits[coupling]['g2'][FLOW][operator]
                    and time in bf.iv_fits[coupling]['beta'][FLOW][operator]
                ]
                if len(fit_couplings) != len(data_y):
                    raise RuntimeError(
                        f'Coupling/data mismatch for {operator}, t={time}: '
                        f'{len(fit_couplings)} versus {len(data_y)}'
                    )
                chi2 = float(qof['chi2'])
                dof = int(qof['dof'])
                parameter_summary = '; '.join(
                    f'{name}={params[name][0]}' for name in sorted(params) if name != 'x'
                )
                for index, coupling in enumerate(fit_couplings):
                    detail_rows.append({
                        'prior_width': FIT_WIDTH,
                        'window': window_tag(window),
                        'operator': operator,
                        'operator_label': OP_LABELS[operator],
                        'flow_time': float(time),
                        'coupling': coupling,
                        'beta_b': beta_value(coupling),
                        'g2_mean': float(gv.mean(data_x[index])),
                        'g2_sdev': float(gv.sdev(data_x[index])),
                        'fitted_g2_mean': float(gv.mean(fitted_x[index])),
                        'beta_mean': float(gv.mean(data_y[index])),
                        'beta_sdev': float(gv.sdev(data_y[index])),
                        'prediction_mean': float(gv.mean(prediction[index])),
                        'prediction_sdev': float(gv.sdev(prediction[index])),
                        'residual_mean': float(gv.mean(residual[index])),
                        'residual_sdev': float(residual_sdev[index]),
                        'signed_sigma_deviation': float(pull[index]),
                        'abs_sigma_deviation': float(abs(pull[index])),
                        'chi2_including_priors': chi2,
                        'dof_reported': dof,
                        'chi2_per_dof': chi2 / dof if dof else np.nan,
                        'p_value': float(qof['p-value']),
                        'logGBF': qof.get('logGBF'),
                        'n_data_points': len(data_y),
                        'n_coefficient_priors': ORDER,
                        'xerrors': config.interpolation.xerrors,
                        'posterior_parameters': parameter_summary,
                    })

    details = pd.DataFrame(detail_rows)
    fit_columns = [
        'prior_width', 'window', 'operator', 'operator_label', 'flow_time',
        'chi2_including_priors', 'dof_reported', 'chi2_per_dof', 'p_value',
        'logGBF', 'n_data_points', 'n_coefficient_priors', 'xerrors',
        'posterior_parameters',
    ]
    summary = (
        details[fit_columns]
        .drop_duplicates(['window', 'operator', 'flow_time'])
        .sort_values(['window', 'operator', 'flow_time'])
        .reset_index(drop=True)
    )
    details.to_csv(OUTPUT_ROOT / 'interpolation_diagnostics_by_coupling.csv', index=False)
    summary.to_csv(OUTPUT_ROOT / 'interpolation_fit_quality_summary.csv', index=False)
    (OUTPUT_ROOT / 'interpolation_fit_quality_summary.txt').write_text(
        summary.to_string(index=False) + '\n'
    )
    for window in WINDOWS:
        tag = window_tag(window)
        window_details = details.query('window == @tag')
        window_summary = summary.query('window == @tag')
        window_details.to_csv(
            case_dir(window) / 'interpolation_diagnostics_by_coupling.csv', index=False
        )
        window_summary.to_csv(
            case_dir(window) / 'interpolation_fit_quality_summary.csv', index=False
        )
        (case_dir(window) / 'interpolation_fit_quality_summary.txt').write_text(
            window_summary.to_string(index=False) + '\n'
        )


if FIT_ID == 'fit4':
    save_interpolation_diagnostics()


# ## Plot 2 — infinite-volume extrapolations for every bare coupling

# In[7]:


fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, coupling in zip(axes.ravel(), COUPLINGS):
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
handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc='upper center', frameon=False)
fig.tight_layout(rect=(0, 0, 1, .96))
base = OUTPUT_ROOT / f'infinite_volume_all_beta_{FIT_ID}'
if FIT_WIDTH is not None:
    fig.text(0.995, 0.005, rf'prior width $={FIT_WIDTH:g}$', ha='right', va='bottom',
             fontsize=8, color='gray', alpha=.75)
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig)


# ## Plot 2b — infinite-volume beta_GF versus 1/V


# Infinite-volume extrapolation: beta_GF versus 1/V

fig, axes = plt.subplots(2, 5, figsize=(22, 9))

for ax, coupling in zip(axes.ravel(), COUPLINGS):
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

output_base = OUTPUT_ROOT / f'infinite_volume_beta_vs_invV_{FIT_ID}'
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


# ## Plot 2c — infinite-volume beta_GF/g_GF^4 versus 1/V


# Infinite-volume extrapolation:
# beta_GF / g_GF^4 versus 1/V

fig, axes = plt.subplots(2, 5, figsize=(22, 9))

for ax, coupling in zip(axes.ravel(), COUPLINGS):
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
  / f'infinite_volume_beta_over_g4_vs_invV_{FIT_ID}'
)

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


# ## Continuum scan — diagonal and kernel-correlated
# 
# Each case is serialized immediately to its own range/mode folder, matching the reference scan organization and preventing all fit covariance objects from accumulating in memory.

# In[ ]:


def snapshot_case(window, mode):
    payload = {
        'window': window, 'mode': mode, 'fit_id': FIT_ID,
        'model_tag': MODEL_TAG, 'fit4_order': ORDER, 'pt_powers': PT_POWERS,
        'g2s': copy.deepcopy(bf.g2s), 'betas': copy.deepcopy(bf.betas),
        'cnt_fits': copy.deepcopy(bf.cnt_fits), 'cnt_qof': copy.deepcopy(bf.continuum.quality),
        'metadata': copy.deepcopy(bf.continuum.metadata),
    }
    path = case_dir(window, mode) / 'continuum_case.gvar'
    with path.open('wb') as stream:
        gv.dump(payload, stream)

def load_case(window, mode):
    path = case_dir(window, mode) / 'continuum_case.gvar'
    with path.open('rb') as stream:
        return gv.load(stream)

mng2, mxg2, dg2 = G2_GRID
for window in WINDOWS:
    bf.cnt_xtrp(mnt=window[0], mxt=window[1], mng2=mng2, mxg2=mxg2, dg2=dg2,
                 cov_mode='diagonal', diagonal=True, correlated=False,
                 error_mode='fit', tau0=config.tau0, v=0)
    snapshot_case(window, 'diagonal')
    bf.cnt_xtrp(mnt=window[0], mxt=window[1], mng2=mng2, mxg2=mxg2, dg2=dg2,
                 cov_mode='kernel', diagonal=False, correlated=True,
                 error_mode='fit', tau0=config.tau0, v=0)
    snapshot_case(window, 'correlated')
    print('saved', window)
    gc.collect()

print(f'saved {2 * len(WINDOWS)} continuum cases')


# ## Plot 3 — interpolation curves together with the fitted IV data points

# In[ ]:


PLOT_TIME_STEP = 0.25
PLOT_OPERATORS = OBSERVABLES


def select_flow_times(available_times, window, step):
    """Select available flow times nearest to a regular grid with spacing step."""
    available_times = sorted(available_times, key=float)
    if not available_times:
        return []
    targets = np.arange(window[0], window[1] + 0.5 * step, step)
    selected = []
    for target in targets:
        nearest = min(available_times, key=lambda time: abs(float(time) - target))
        if nearest not in selected:
            selected.append(nearest)
    return selected


for window in WINDOWS:
    fig, ax = plt.subplots(figsize=(8, 6))

    for operator in PLOT_OPERATORS:
        times = select_flow_times(flow_times(window), window, PLOT_TIME_STEP)
        for time in times:
            x_grid, beta_grid, data = bf.interpolation_curve(FLOW, operator, time)
            x_grid = np.asarray(x_grid, dtype=float)
            beta_grid = np.asarray(beta_grid, dtype=object)
            curve_ratio = beta_grid / x_grid**2

            data_x = np.asarray(data.x, dtype=object)
            data_beta = np.asarray(data.y, dtype=object)
            data_ratio = data_beta / data_x**2

            alpha = 0.25 + 0.65 * (
                (float(time) - window[0]) / max(window[1] - window[0], 1e-12)
            )
            color = OP_COLORS[operator]

            ax.plot(x_grid, gv.mean(curve_ratio), color=color, alpha=alpha, lw=1.5)
            ax.fill_between(
                x_grid,
                gv.mean(curve_ratio) - gv.sdev(curve_ratio),
                gv.mean(curve_ratio) + gv.sdev(curve_ratio),
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            ax.errorbar(
                gv.mean(data_x),
                gv.mean(data_ratio),
                xerr=gv.sdev(data_x),
                yerr=gv.sdev(data_ratio),
                fmt="o",
                ms=4.5,
                capsize=2.5,
                elinewidth=0.9,
                color=color,
                alpha=alpha,
                markeredgecolor="black",
                markeredgewidth=0.35,
                linestyle="none",
                zorder=3,
            )

    x_pt = np.linspace(0.01, 5.0, 300)
    pt_handles = []
    for loops, linestyle, label in (
        (1, "-", "1-loop universal"),
        (2, "--", "2-loop universal"),
        (3, "-.", "3-loop gradient flow"),
    ):
        line, = ax.plot(
            x_pt,
            pt_over_g4(x_pt, loops),
            color="gray",
            linestyle=linestyle,
            lw=1.4,
            alpha=0.7,
            label=label,
        )
        pt_handles.append(line)

    operator_handles = [
        Line2D(
            [0], [0], color=OP_COLORS[operator], lw=1.5, marker="o",
            markersize=5, markeredgecolor="black", markeredgewidth=0.35,
            label=OP_LABELS[operator],
        )
        for operator in PLOT_OPERATORS
    ]
    ax.legend(handles=operator_handles + pt_handles, ncol=2, frameon=False)
    ax.set(
        xlim=(0, 5),
        xlabel=r"$g^2_{GF}$",
        ylabel=r"$\beta_{GF}/g_{GF}^4$",
    )
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", alpha=0.35)
    ax.grid(which="minor", linestyle="--", alpha=0.20)
    ax.text(
        0.50, 0.52, r"\textbf{Preliminary}", transform=ax.transAxes,
        fontsize=42, color="gray", alpha=0.28, ha="center", va="center",
        rotation=30, zorder=0,
    )
    ax.text(
        0.58,
        0.15,
        FIT_WATERMARK + "\n"
        + "Interpolation with IV data, "
        + rf"$t/a^2\in[{window[0]:g},{window[1]:g}]$" + "\n"
        + rf"$\Delta(t/a^2)\simeq {PLOT_TIME_STEP:g}$" + "\n"
        + r"TLN",
        transform=ax.transAxes,
        fontsize=10.5,
        color="gray",
        alpha=0.52,
        ha="center",
        va="center",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, window, f"interpolation_beta_over_g4_{FIT_ID}_with_data")


# ## Plot 4 — interpolation quality for every flow-time range

# In[ ]:


for window in WINDOWS:
      fig, (ax1, ax2) = plt.subplots(
          2,
          1,
          figsize=(8, 8),
          sharex=True,
      )

      for operator in OBSERVABLES:
          rows = []
          for time, qof in bf.ntrp_qof[FLOW][operator].items():
              if window[0] <= float(time) <= window[1]:
                  rows.append(
                      (
                          float(time),
                          qof['chi2'] / qof['dof']
                          if qof['dof']
                          else np.nan,
                          qof['p-value'],
                      )
                  )

          if rows:
              rows = np.asarray(sorted(rows), dtype=float)
              ax1.plot(
                  rows[:, 0],
                  rows[:, 1],
                  'o-',
                  color=OP_COLORS[operator],
                  label=OP_LABELS[operator],
              )
              ax2.plot(
                  rows[:, 0],
                  rows[:, 2],
                  'o-',
                  color=OP_COLORS[operator],
              )

      # Draw watermark above all plot elements.
      ax1.text(
          0.50,
          0.52,
          r'\textbf{Preliminary}',
          transform=ax1.transAxes,
          fontsize=42,
          color='gray',
          alpha=.28,
          ha='center',
          va='center',
          rotation=30,
          zorder=20,
          clip_on=False,
      )
      ax1.text(
          0.58,
          0.16,
          (
              FIT_WATERMARK
              + '\n'
              rf'Interpolation QoF, $t/a^2\in[{window[0]:g},{window[1]:g}]$'
              '\n'
              r'TLN'
          ),
          transform=ax1.transAxes,
          fontsize=10.5,
          color='gray',
          alpha=.52,
          ha='center',
          va='center',
          fontweight='bold',
          zorder=20,
          clip_on=False,
      )

      ax1.axhline(1, color='gray', ls='--')
      ax2.axhline(.05, color='gray', ls='--')

      ax1.set_ylabel(r'$\chi^2/{\rm dof}$')
      ax2.set_ylabel(r'$p$-value')
      ax2.set_xlabel(r'$t/a^2$')

      ax1.legend(frameon=False)

      save_figure(
          fig,
          window,
          f'interpolation_quality_{FIT_ID}',
      )


# ## Plot helper — continuum extrapolations versus $a^2/t$

# In[ ]:


def plot_continuum_panels(window, mode, divide_by_g4):
      case = load_case(window, mode)
      fig, axes = plt.subplots(2, 4, figsize=(18, 9))

      tau0 = config.tau0
      correction_label = (
          'TLN'
          if config.correction in ('tln', 'tree-level-normalization')
          else 'No TLN'
      )
      mode_label = (
          'Correlated continuum'
          if mode == 'correlated'
          else 'Diagonal continuum'
      )

      for ax, target in zip(axes.ravel(), TARGET_G2):
          displayed_g2 = []

          for operator in OBSERVABLES:
              grid = np.asarray(
                  case['g2s'][FLOW][operator],
                  dtype=float,
              )

              if len(grid) == 0:
                  continue

              # Use the continuum point nearest to the requested target.
              idx = int(np.argmin(np.abs(grid - target)))
              g2 = float(grid[idx])
              displayed_g2.append(g2)

              params = case['cnt_fits'][FLOW][operator][idx]

              all_times = sorted(
                  bf.ntrp_fits[FLOW][operator],
                  key=float,
              )

              # Reproduce the exact flow-time and interpolation-domain selection
              # used by cnt_xtrp.
              fit_times = [
                  time
                  for time in all_times
                  if float(time) - tau0 > 0.0
                  and window[0] <= float(time) - tau0 <= window[1]
                  and bf.ntrp_nf[FLOW][operator][time][0]
                  <= g2
                  <= bf.ntrp_nf[FLOW][operator][time][-1]
              ]

              # Show one additional flow-time unit on either side as hollow
              # points. These points are displayed but were not used in the fit.
              outside_times = [
                  time
                  for time in all_times
                  if float(time) - tau0 > 0.0
                  and window[0] - 1.0
                  <= float(time) - tau0
                  <= window[1] + 1.0
                  and not (
                      window[0]
                      <= float(time) - tau0
                      <= window[1]
                  )
                  and bf.ntrp_nf[FLOW][operator][time][0]
                  <= g2
                  <= bf.ntrp_nf[FLOW][operator][time][-1]
              ]

              def continuum_plot_values(times):
                  nominal_times = np.asarray(
                      [float(time) - tau0 for time in times],
                      dtype=float,
                  )
                  measured_times = np.asarray(
                      [float(time) for time in times],
                      dtype=float,
                  )

                  xvalues = 1.0 / nominal_times
                  jacobian = nominal_times / measured_times

                  yvalues = np.asarray(
                      [
                          factor
                          * bf.interpolation.model.evaluate(
                              g2,
                              bf.ntrp_fits[FLOW][operator][time],
                          )
                          for factor, time in zip(jacobian, times)
                      ],
                      dtype=object,
                  )

                  norm = g2**2 if divide_by_g4 else 1.0
                  return xvalues, yvalues / norm

              norm = g2**2 if divide_by_g4 else 1.0

              # Filled points: data included in the continuum fit.
              if fit_times:
                  xfit_data, yfit_data = continuum_plot_values(fit_times)

                  ax.errorbar(
                      xfit_data,
                      gv.mean(yfit_data),
                      yerr=gv.sdev(yfit_data),
                      fmt='o',
                      ms=5,
                      capsize=3,
                      color=OP_COLORS[operator],
                      markerfacecolor=OP_COLORS[operator],
                      markeredgecolor=OP_COLORS[operator],
                      label=OP_LABELS[operator],
                      zorder=4,
                  )

              # Hollow points: valid neighboring points outside the fit window.
              if outside_times:
                  xoutside, youtside = continuum_plot_values(outside_times)

                  ax.errorbar(
                      xoutside,
                      gv.mean(youtside),
                      yerr=gv.sdev(youtside),
                      fmt='o',
                      ms=5,
                      capsize=3,
                      color=OP_COLORS[operator],
                      markerfacecolor='none',
                      markeredgecolor=OP_COLORS[operator],
                      alpha=.75,
                      zorder=3,
                  )

              # Draw the fitted extrapolation over the displayed a²/t range.
              displayed_times = fit_times + outside_times
              if displayed_times:
                  displayed_x = np.asarray(
                      [
                          1.0 / (float(time) - tau0)
                          for time in displayed_times
                      ],
                      dtype=float,
                  )
                  xmax = 1.05 * np.max(displayed_x)
              else:
                  xmax = 1.0 / max(window[0] - tau0, 1e-12)

              xline = np.linspace(0.0, xmax, 200)
              yline = (
                  params['beta'][0]
                  + params['slope'][0] * xline
              ) / norm

              ax.plot(
                  xline,
                  gv.mean(yline),
                  color=OP_COLORS[operator],
                  lw=1.5,
                  zorder=2,
              )
              ax.fill_between(
                  xline,
                  gv.mean(yline) - gv.sdev(yline),
                  gv.mean(yline) + gv.sdev(yline),
                  color=OP_COLORS[operator],
                  alpha=.15,
                  zorder=1,
              )

          if displayed_g2:
              panel_g2 = displayed_g2[0]
              ax.text(
                  .04,
                  .94,
                  rf'$g^2_{{GF}}={panel_g2:g}$',
                  transform=ax.transAxes,
                  ha='left',
                  va='top',
                  fontsize=13,
                  zorder=20,
              )

          ax.set_xlabel(r'$a^2/t$')

      axes.ravel()[0].legend(frameon=False)

      ylabel = (
          r'$\beta_{GF}/g_{GF}^4$'
          if divide_by_g4
          else r'$\beta_{GF}$'
      )
      fig.supylabel(ylabel)

      # Figure-level watermark and analysis information.
      fig.text(
          .50,
          .52,
          r'\textbf{Preliminary}',
          fontsize=58,
          color='gray',
          alpha=.18,
          ha='center',
          va='center',
          rotation=30,
          zorder=100,
      )
      fig.text(
          .50,
          .025,
          (
              FIT_WATERMARK
              + r'; '
              + mode_label
              + r'; '
              + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
              + r'; '
              + correction_label
              + '\n'
              + 'Filled points: included in fit; '
              + 'hollow points: neighboring flow times not included in fit'
          ),
          fontsize=11,
          color='gray',
          alpha=.9,
          ha='center',
          va='bottom',
          fontweight='bold',
          zorder=100,
      )

      fig.tight_layout(rect=(.025, .075, 1, 1))

      name = (
          f'continuum_a2_over_t_beta_over_g4_{FIT_ID}'
          if divide_by_g4
          else f'continuum_a2_over_t_beta_{FIT_ID}'
      )
      save_figure(fig, window, name, mode)


# ## Plot 5 — raw $eta_{GF}$ continuum panels for all ranges and modes

# In[ ]:


for window in WINDOWS:
    for mode in ('diagonal', 'correlated'):
        plot_continuum_panels(window, mode, divide_by_g4=False)


# ## Plot 6 — $eta_{GF}/g_{GF}^4$ continuum panels for all ranges and modes

# In[ ]:


for window in WINDOWS:
    for mode in ('diagonal', 'correlated'):
        plot_continuum_panels(window, mode, divide_by_g4=True)


# ## Plot 7 — continuum fit quality for all ranges and modes

# In[ ]:


for window in WINDOWS:
      for mode in ('diagonal', 'correlated'):
          case = load_case(window, mode)
          fig, (ax1, ax2) = plt.subplots(
              2,
              1,
              figsize=(8, 8),
              sharex=True,
          )

          for operator in OBSERVABLES:
              grid = np.asarray(
                  case['g2s'][FLOW][operator],
                  dtype=float,
              )
              q = case['cnt_qof'][FLOW][operator]

              chi = np.asarray(
                  [
                      row['chi2'] / row['dof']
                      if row['dof']
                      else np.nan
                      for row in q
                  ],
                  dtype=float,
              )
              pv = np.asarray(
                  [row['p-value'] for row in q],
                  dtype=float,
              )

              # Ensure that every available continuum-fit point is plotted.
              npoints = min(len(grid), len(chi), len(pv))
              grid_plot = grid[:npoints]
              chi_plot = chi[:npoints]
              pv_plot = pv[:npoints]

              ax1.plot(
                  grid_plot,
                  chi_plot,
                  '-',
                  color=OP_COLORS[operator],
                  lw=1.2,
                  alpha=.8,
              )
              ax1.scatter(
                  grid_plot,
                  chi_plot,
                  color=OP_COLORS[operator],
                  s=28,
                  zorder=3,
                  label=OP_LABELS[operator],
              )

              ax2.plot(
                  grid_plot,
                  pv_plot,
                  '-',
                  color=OP_COLORS[operator],
                  lw=1.2,
                  alpha=.8,
              )
              ax2.scatter(
                  grid_plot,
                  pv_plot,
                  color=OP_COLORS[operator],
                  s=28,
                  zorder=3,
              )

          ax1.axhline(1, color='gray', ls='--')
          ax2.axhline(.05, color='gray', ls='--')

          ax1.set_ylabel(r'$\chi^2/{\rm dof}$')
          ax2.set_ylabel(r'$p$-value')
          ax2.set_xlabel(r'$g^2_{GF}$')
          ax1.legend(frameon=False)

          correction_label = (
              'TLN'
              if config.correction in ('tln', 'tree-level-normalization')
              else 'No TLN'
          )
          mode_label = (
              'Correlated continuum QoF'
              if mode == 'correlated'
              else 'Diagonal continuum QoF'
          )

          # Watermark.
          for ax in (ax1, ax2):
              ax.text(
                  .50,
                  .52,
                  r'\textbf{Preliminary}',
                  transform=ax.transAxes,
                  fontsize=36,
                  color='gray',
                  alpha=.25,
                  ha='center',
                  va='center',
                  rotation=30,
                  zorder=0,
              )
              ax.text(
                  .58,
                  .15,
                  (
                      FIT_WATERMARK
                      + '\n'
                      + mode_label
                      + '\n'
                      + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
                      + '\n'
                      + correction_label
                  ),
                  transform=ax.transAxes,
                  fontsize=10.5,
                  color='gray',
                  alpha=.92,
                  ha='center',
                  va='center',
                  fontweight='bold',
              )

          save_figure(
              fig,
              window,
              f'continuum_quality_{FIT_ID}',
              mode,
          )


# ## Plot helper — final continuum curves

# In[ ]:


def plot_final_curve(window, mode, divide_by_g4):
      case = load_case(window, mode)
      fig, ax = plt.subplots(figsize=(8, 6))

      correction_label = (
          'TLN'
          if config.correction in ('tln', 'tree-level-normalization')
          else 'No TLN'
      )
      mode_label = (
          'Correlated continuum'
          if mode == 'correlated'
          else 'Diagonal continuum'
      )
      quantity_label = (
          r'$\beta_{GF}/g_{GF}^4$'
          if divide_by_g4
          else r'$\beta_{GF}$'
      )

      # Continuum central curves and uncertainty bands.
      for operator in OBSERVABLES:
          x = np.asarray(
              case['g2s'][FLOW][operator],
              dtype=float,
          )
          y = np.asarray(
              case['betas'][FLOW][operator],
              dtype=object,
          )

          if divide_by_g4:
              y = y / x**2

          # Sort explicitly so the lines and bands cannot cross because of
          # an unordered coupling grid.
          order = np.argsort(x)
          x = x[order]
          y = y[order]

          ymean = gv.mean(y)
          ysdev = gv.sdev(y)

          ax.plot(
              x,
              ymean,
              color=OP_COLORS[operator],
              lw=1.8,
              label=OP_LABELS[operator],
              zorder=3,
          )
          ax.fill_between(
              x,
              ymean - ysdev,
              ymean + ysdev,
              color=OP_COLORS[operator],
              alpha=.18,
              linewidth=0,
              zorder=2,
          )

      # Perturbative comparison curves.
      xp = np.linspace(.001, 5.0, 500)
      pt_curves = (
          (1, '-', '1-loop universal'),
          (2, '--', '2-loop universal'),
          (3, '-.', '3-loop gradient flow'),
      )

      for loops, linestyle, label in pt_curves:
          perturbative_ratio = pt_over_g4(xp, loops)

          # pt_over_g4 returns beta_PT/g^4. Restore beta_PT for the
          # unscaled beta plot.
          perturbative_curve = (
              perturbative_ratio
              if divide_by_g4
              else xp**2 * perturbative_ratio
          )

          ax.plot(
              xp,
              perturbative_curve,
              color='gray',
              ls=linestyle,
              lw=1.5,
              label=label,
              zorder=1,
          )

      # Watermark drawn inside the axes and above the plotted curves.
      ax.text(
          .50,
          .52,
          r'\textbf{Preliminary}',
          transform=ax.transAxes,
          fontsize=42,
          color='gray',
          alpha=.25,
          ha='center',
          va='center',
          rotation=30,
          zorder=20,
          clip_on=True,
      )
      ax.text(
          .58,
          .15,
          (
              FIT_WATERMARK
              + '\n'
              + mode_label
              + '\n'
              + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
              + '\n'
              + correction_label
              + '\n'
              + quantity_label
          ),
          transform=ax.transAxes,
          fontsize=10.5,
          color='gray',
          alpha=.82,
          ha='center',
          va='center',
          fontweight='bold',
          zorder=20,
          clip_on=True,
      )

      ax.set(
          xlim=(0, 5),
          xlabel=r'$g^2_{GF}$',
          ylabel=quantity_label,
      )
      ax.legend(
          ncol=2,
          frameon=False,
      )

      name = (
          f'continuum_beta_over_g4_vs_g2_{FIT_ID}'
          if divide_by_g4
          else f'continuum_beta_vs_g2_{FIT_ID}'
      )
      save_figure(
          fig,
          window,
          name,
          mode,
      )


# ## Plot 8 — final raw continuum curves for all ranges and modes

# In[ ]:


for window in WINDOWS:
      for mode in ('diagonal', 'correlated'):
          plot_final_curve(
              window,
              mode,
              divide_by_g4=False,
          )


# ## Plot 9 — final scaled continuum curves for all ranges and modes

# In[ ]:


for window in WINDOWS:
      for mode in ('diagonal', 'correlated'):
          plot_final_curve(
              window,
              mode,
              divide_by_g4=True,
          )


# ## Plot 10 — diagonal versus correlated continuum for every range

# In[ ]:


for window in WINDOWS:
      diag = load_case(window, 'diagonal')
      corr = load_case(window, 'correlated')

      fig, ax = plt.subplots(figsize=(8, 6))

      correction_label = (
          'TLN'
          if config.correction in ('tln', 'tree-level-normalization')
          else 'No TLN'
      )

      for operator in OBSERVABLES:
          for case, linestyle, mode_label in (
              (diag, '--', 'diagonal'),
              (corr, '-', 'correlated'),
          ):
              x = np.asarray(
                  case['g2s'][FLOW][operator],
                  dtype=float,
              )
              y = (
                  np.asarray(
                      case['betas'][FLOW][operator],
                      dtype=object,
                  )
                  / x**2
              )

              # Sort the continuum grid before drawing the curve and band.
              order = np.argsort(x)
              x = x[order]
              y = y[order]

              ymean = gv.mean(y)
              ysdev = gv.sdev(y)

              # Dashed: diagonal continuum.
              # Solid: correlated continuum.
              ax.plot(
                  x,
                  ymean,
                  color=OP_COLORS[operator],
                  ls=linestyle,
                  lw=1.8,
                  label=f'{OP_LABELS[operator]} {mode_label}',
                  zorder=4 if mode_label == 'correlated' else 3,
              )

              # Uncertainty band associated with the corresponding curve.
              ax.fill_between(
                  x,
                  ymean - ysdev,
                  ymean + ysdev,
                  color=OP_COLORS[operator],
                  alpha=.18 if mode_label == 'correlated' else .07,
                  linewidth=0,
                  zorder=2 if mode_label == 'correlated' else 1,
              )

      # Perturbative beta/g^4 comparison curves.
      xp = np.linspace(.001, 5.0, 500)
      pt_curves = (
          (1, '-', '1-loop universal'),
          (2, '--', '2-loop universal'),
          (3, '-.', '3-loop gradient flow'),
      )

      for loops, linestyle, label in pt_curves:
          ax.plot(
              xp,
              pt_over_g4(xp, loops),
              color='gray',
              ls=linestyle,
              lw=1.5,
              label=label,
              zorder=1,
          )

      # Watermark and analysis information inside the plotting area.
      ax.text(
          .50,
          .52,
          r'\textbf{Preliminary}',
          transform=ax.transAxes,
          fontsize=42,
          color='gray',
          alpha=.25,
          ha='center',
          va='center',
          rotation=30,
          zorder=20,
          clip_on=True,
      )
      ax.text(
          .58,
          .15,
          (
              FIT_WATERMARK
              + '\n'
              + 'Diagonal vs correlated continuum'
              + '\n'
              + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
              + '\n'
              + correction_label
              + '\n'
              + r'Solid: correlated; dashed: diagonal'
          ),
          transform=ax.transAxes,
          fontsize=10.5,
          color='gray',
          alpha=.82,
          ha='center',
          va='center',
          fontweight='bold',
          zorder=20,
          clip_on=True,
      )

      ax.set(
          xlim=(0, 4.8),
          xlabel=r'$g^2_{GF}$',
          ylabel=r'$\beta_{GF}/g_{GF}^4$',
      )
      ax.legend(
          ncol=2,
          frameon=False,
      )

      save_figure(
          fig,
          window,
          f'diagonal_vs_correlated_{FIT_ID}',
      )


# ## Plot 11 — correlated weak-coupling extrapolation for every range
# 
# Point error bars are continuum-point uncertainties; colored bands are posterior uncertainties of the shared fitted curve and are not expected to coincide.

# In[ ]:


def ratio_model(x, p):
  u = np.asarray(x) / bf.perturbative_beta_function.nrm
  prefix = 'c' if FIT_ID == 'fit4' else 'd'
  correction = 1.0 + sum(
      p[f'{prefix}{power}'][0] * u**power
      for power in PT_POWERS
  )

  return pt_over_g4(x, 3) * correction


for window in WINDOWS:
  case = load_case(window, 'correlated')
  fig, ax = plt.subplots(figsize=(8, 6))

  correction_label = (
      'TLN'
      if config.correction in ('tln', 'tree-level-normalization')
      else 'No TLN'
  )

  fits = {}
  minimum_couplings = []

  for operator in OBSERVABLES:
      x = np.asarray(
          case['g2s'][FLOW][operator],
          dtype=float,
      )
      y = (
          np.asarray(
              case['betas'][FLOW][operator],
              dtype=object,
          )
          / x**2
      )

      if len(x) == 0:
          continue

      # Sort the correlated continuum results.
      order = np.argsort(x)
      x = x[order]
      y = y[order]
      minimum_couplings.append(float(x[0]))

      continuum_mean = gv.mean(y)
      continuum_sdev = gv.sdev(y)

      prefix = 'c' if FIT_ID == 'fit4' else 'd'
      prior = {
          f'{prefix}{power}': [gv.gvar(0, 10)]
          for power in PT_POWERS
      }

      # The gvars in y retain the covariance of the correlated continuum
      # calculation. lsqfit therefore uses their full covariance matrix.
      fit = lsqfit.nonlinear_fit(
          data=(x, y),
          fcn=ratio_model,
          prior=prior,
      )
      fits[operator] = fit

      # Evaluate the fitted curve and its posterior uncertainty from
      # g²=0 through the complete correlated-continuum data range.
      xp = np.linspace(0.0, float(x[-1]), 600)
      yp = np.asarray(
          ratio_model(xp, fit.p),
          dtype=object,
      )

      fit_mean = gv.mean(yp)
      fit_sdev = gv.sdev(yp)

      # Original correlated-continuum uncertainty envelope. This is shown
      # only over the region where continuum results actually exist.
      ax.plot(
          x,
          continuum_mean,
          color=OP_COLORS[operator],
          ls=':',
          lw=1.4,
          alpha=.9,
          label=(
              f'{OP_LABELS[operator]} correlated continuum '
              r'($\pm1\sigma$)'
          ),
          zorder=4,
      )
      ax.fill_between(
          x,
          continuum_mean - continuum_sdev,
          continuum_mean + continuum_sdev,
          color=OP_COLORS[operator],
          alpha=.10,
          linewidth=0,
          zorder=2,
      )

      # PT-constrained weak-coupling fit and posterior uncertainty,
      # including the extrapolated interval down to g²=0.
      ax.plot(
          xp,
          fit_mean,
          color=OP_COLORS[operator],
          ls='-',
          lw=2.0,
          label=(
              f'{OP_LABELS[operator]} correlated '
              'weak-coupling fit'
          ),
          zorder=5,
      )
      ax.fill_between(
          xp,
          fit_mean - fit_sdev,
          fit_mean + fit_sdev,
          color=OP_COLORS[operator],
          alpha=.22,
          linewidth=0,
          zorder=3,
      )

      origin = np.asarray(
          ratio_model(np.asarray([0.0]), fit.p),
          dtype=object,
      )[0]

      print(
          window,
          OP_LABELS[operator],
          'correlated',
          'origin=',
          origin,
          'Q=',
          fit.Q,
      )

  # The shaded interval has no direct correlated-continuum points.
  if minimum_couplings:
      threshold = min(minimum_couplings)
      ax.axvspan(
          0.0,
          threshold,
          color='gray',
          alpha=.08,
          label=(
              rf'extrapolated region: '
              rf'$0\leq g^2_{{GF}}<{threshold:g}$'
          ),
          zorder=0,
      )

  # Perturbative reference curves.
  xp_pt = np.linspace(0.0, 5.0, 600)
  pt_curves = (
      (1, '-', '1-loop universal'),
      (2, '--', '2-loop universal'),
      (3, '-.', '3-loop gradient flow'),
  )

  for loops, linestyle, label in pt_curves:
      ax.plot(
          xp_pt,
          pt_over_g4(xp_pt, loops),
          color='black',
          ls=linestyle,
          lw=1.4,
          alpha=.65,
          label=label,
          zorder=1,
      )

  # Watermark and analysis information inside the axes.
  ax.text(
      .50,
      .52,
      r'\textbf{Preliminary}',
      transform=ax.transAxes,
      fontsize=42,
      color='gray',
      alpha=.25,
      ha='center',
      va='center',
      rotation=30,
      zorder=20,
      clip_on=True,
  )
  ax.text(
      .58,
      .15,
      (
          FIT_WATERMARK
          + '\n'
          + 'Correlated continuum weak-coupling extrapolation'
          + '\n'
          + rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'
          + '\n'
          + correction_label
          + '\n'
          + r'Bands show posterior $\pm1\sigma$'
          + '\n'
          + r'Fixed $c_0=1$ perturbative limit'
      ),
      transform=ax.transAxes,
      fontsize=10.5,
      color='gray',
      alpha=.82,
      ha='center',
      va='center',
      fontweight='bold',
      zorder=20,
      clip_on=True,
  )

  ax.set(
      xlim=(0, 5),
      xlabel=r'$g^2_{GF}$',
      ylabel=r'$\beta_{GF}/g_{GF}^4$',
  )
  ax.legend(
      ncol=2,
      frameon=False,
  )

  save_figure(
      fig,
      window,
      f'weak_coupling_to_zero_{FIT_ID}',
      'correlated',
  )


# ## Reviewed weak-coupling diagnostics used by the fit4 reference notebooks

from betafn.weak_coupling_plots import run_reviewed_weak_coupling_plots

reviewed_weak_results = None
if args.reviewed_weak_coupling:
    reviewed_weak_results = run_reviewed_weak_coupling_plots(
        bf=bf,
        flow=FLOW,
        observables=OBSERVABLES,
        windows=WINDOWS,
        config=config,
        fit_id=FIT_ID,
        fit_watermark=FIT_WATERMARK,
        op_colors=OP_COLORS,
        op_labels=OP_LABELS,
        flow_times=flow_times,
        pt_over_g4=pt_over_g4,
        load_case=load_case,
        save_figure=save_figure,
    )


# ## Scan manifest and validation summary

# In[ ]:


rows=[]
for window in WINDOWS:
    for mode in ('diagonal','correlated'):
        case=load_case(window,mode)
        for operator in OBSERVABLES:
            q=case['cnt_qof'][FLOW][operator]
            rows.append({'tmin':window[0],'tmax':window[1],'mode':mode,'operator':operator,
                         'nfits':len(q),'mean_chi2_dof':np.mean([r['chi2']/r['dof'] for r in q if r['dof']]),
                         'mean_pvalue':np.mean([r['p-value'] for r in q])})
summary=pd.DataFrame(rows)
summary.to_csv(OUTPUT_ROOT/f'{FIT_ID}_scan_summary.csv',index=False)
print(summary.to_string(index=False))

expected={(window,mode) for window in WINDOWS for mode in ('diagonal','correlated')}
present={(window,mode) for window in WINDOWS for mode in ('diagonal','correlated')
         if (case_dir(window,mode)/'continuum_case.gvar').is_file()}
assert present==expected, f'missing cases: {expected-present}'
assert config.binsize==args.binsize and config.correction==CORRECTION
assert config.combine=={'s': {'p':5/3,'s':-2/3}}
print(f'{FIT_ID} validation complete:',len(present),'continuum cases')
(OUTPUT_ROOT / 'RUN_COMPLETE.json').write_text(json.dumps({
    'model': FIT_ID,
    'model_tag': MODEL_TAG,
    'continuum_cases': len(present),
    'reviewed_weak_coupling': args.reviewed_weak_coupling,
    'elapsed_seconds': round(time_module.time() - RUN_STARTED, 3),
}, indent=2) + '\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
