"""Build the upstream-API fit4 TLN/correlated flow-window scan notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/betafn/nf4_original_multiplicative_pt3_order4_fit4.ipynb"


def src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src(text)}


cells = [
md(r'''# fit4 — nf4 multiplicative three-loop-PT, order 4

Updated upstream analysis (`2af131f`), Wilson flow, on-the-fly TLN, three-volume infinite-volume limit, and exhaustive integer flow-time-window scan.  The interpolation ansatz is
\[
\beta(x)=\beta_{\rm PT}^{(3)}(x)\left[1+\sum_{n=1}^{4}c_n(x/4\pi)^n\right].
\]

Every plot family has its own cell. Outputs are organized as `fit4/order_4/t_<min>_<max>/<mode>/`, where mode is `diagonal` or `correlated`. The correlated result uses the upstream kernel covariance. Rectangle measurements are combined at the raw-history level into the Symanzik observable before its dedicated TLN correction.'''),
code(r'''%load_ext autoreload
%autoreload 2
%matplotlib inline

from pathlib import Path
import copy, gc, json, sys

REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / 'pyproject.toml').is_file():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / 'pyproject.toml').is_file():
    raise RuntimeError('Run this notebook from inside the ContinuousBetaFunction repository')
sys.path.insert(0, str(REPO_ROOT / 'src'))

import betafn
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from IPython.display import display

print('betafn package:', Path(betafn.__file__).resolve())
print('repository:', REPO_ROOT)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'figure.dpi': 150,
    'savefig.dpi': 300, 'axes.labelsize': 18, 'axes.titlesize': 16,
    'legend.fontsize': 10, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
})'''),
md('''## Configuration and complete window catalogue

Change physics/statistical choices only here. Every downstream cell reads these values.'''),
code(r'''FIT_ID = 'fit4'
ORDER = 4
FIT_WATERMARK = rf'{FIT_ID}, correction order {ORDER}'
CORRECTION = 'tln'
DATA_DIR = Path('/Users/yaman/Contbetafn/data/New Four')
OUTPUT_ROOT = REPO_ROOT / 'src/betafn' / FIT_ID / f'order_{ORDER}'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

COUPLINGS = ('20p0', '18p0', '16p0', '14p0', '12p0', '10p0', '9p00', '8p50')
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
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
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
print('output root:', OUTPUT_ROOT)'''),
md('''## Analysis construction

The pulled upstream package computes TLN on demand. No external `.tln` path is supplied. The raw `Es` history is the rectangle measurement and is replaced by `(5/3)Ep-(2/3)Es` before averaging and TLN.'''),
code(r'''if not DATA_DIR.is_dir():
    raise FileNotFoundError(DATA_DIR)

bf = betafn.BetaFunction(nf=4)
interpolation = bf.perturbative_interpolation(
    loops=3, correction_order=ORDER, free_intercept=False,
    width=10.0, xerrors=True,
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
    binsize=15,
    use_gamma_method=False,
    process_window=(1.5, 10.5),
    fit_window=(2.0, 10.0),
    error_mode='fit',
    cov_mode='kernel',
    correlated=True,
    verbosity=0,
)
display(pd.DataFrame([config.describe()]).T.rename(columns={0: 'fit4 setting'}))'''),
md('## Stage 1 — processing with TLN'),
code(r'''bf.run_processing(config)
print('correction:', bf._process_config.correction)
print('processed couplings:', sorted(bf.avg_data, key=beta_value))
print('binsize:', bf._binsize)
print(bf.data_report())'''),
md('## Plot 1 — processed largest-volume data'),
code(r'''fig, ax = plt.subplots(figsize=(8, 6))
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
base = OUTPUT_ROOT / 'processed_largest_volume_fit4'
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)'''),
md('## Stages 2–4 — chiral, infinite volume, and fixed fit4 interpolation'),
code(r'''bf.run_chiral(config)
bf.run_infinite_volume(config)
bf.run_interpolation(config)
print(bf.stage_summary('chiral'))
print(bf.stage_summary('infinite_volume'))
print(bf.stage_summary('interpolation'))'''),
md('## Plot 2 — infinite-volume extrapolations for every bare coupling'),
code(r'''fig, axes = plt.subplots(2, 4, figsize=(18, 9))
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
base = OUTPUT_ROOT / 'infinite_volume_all_beta_fit4'
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)'''),
md('## Plot 2b — infinite-volume beta_GF versus 1/V'),
code(r'''# Infinite-volume extrapolation: beta_GF versus 1/V

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

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
plt.close(fig)'''),
md('## Plot 2c — infinite-volume beta_GF/g_GF^4 versus 1/V'),
code(r'''# Infinite-volume extrapolation:
# beta_GF / g_GF^4 versus 1/V

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

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
plt.close(fig)'''),
md('''## Continuum scan — diagonal and kernel-correlated

Each case is serialized immediately to its own range/mode folder, matching the reference scan organization and preventing all fit covariance objects from accumulating in memory.'''),
code(r'''def snapshot_case(window, mode):
    payload = {
        'window': window, 'mode': mode, 'fit_id': FIT_ID, 'order': ORDER,
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

print(f'saved {2 * len(WINDOWS)} continuum cases')'''),
md('## Plot 3 — interpolation curves together with the fitted IV data points'),
code(r'''PLOT_TIME_STEP = 0.25
PLOT_OPERATORS = OBSERVABLES

def select_flow_times(available_times, window, step):
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
        for time in select_flow_times(flow_times(window), window, PLOT_TIME_STEP):
            x, y, data = bf.interpolation_curve(FLOW, operator, time)
            x = np.asarray(x, dtype=float)
            ratio = np.asarray(y, dtype=object) / x**2
            data_x = np.asarray(data.x, dtype=object)
            data_ratio = np.asarray(data.y, dtype=object) / data_x**2
            alpha = .25 + .65 * (float(time)-window[0]) / max(window[1]-window[0], 1e-12)
            color = OP_COLORS[operator]
            ax.plot(x, gv.mean(ratio), color=color, alpha=alpha, lw=1.5)
            ax.fill_between(x, gv.mean(ratio)-gv.sdev(ratio), gv.mean(ratio)+gv.sdev(ratio),
                            color=color, alpha=.08, linewidth=0)
            ax.errorbar(gv.mean(data_x), gv.mean(data_ratio), xerr=gv.sdev(data_x),
                        yerr=gv.sdev(data_ratio), fmt='o', ms=4.5, capsize=2.5,
                        elinewidth=.9, color=color, alpha=alpha,
                        markeredgecolor='black', markeredgewidth=.35,
                        linestyle='none', zorder=3)
    xpt = np.linspace(.01, 5, 300)
    pt_handles=[]
    for loops, ls, label in ((1,'-','1-loop universal'),
                             (2,'--','2-loop universal'),
                             (3,'-.','3-loop gradient flow')):
        line,=ax.plot(xpt,pt_over_g4(xpt,loops),color='gray',ls=ls,lw=1.4,
                      alpha=.7,label=label)
        pt_handles.append(line)
    op_handles=[Line2D([0],[0],color=OP_COLORS[o],lw=1.5,marker='o',markersize=5,
                       markeredgecolor='black',markeredgewidth=.35,label=OP_LABELS[o])
                for o in PLOT_OPERATORS]
    ax.legend(handles=op_handles+pt_handles,ncol=2,frameon=False)
    ax.set(xlim=(0,5),xlabel=r'$g^2_{GF}$',ylabel=r'$\beta_{GF}/g_{GF}^4$')
    ax.minorticks_on(); ax.grid(which='major',linestyle='-',alpha=.35)
    ax.grid(which='minor',linestyle='--',alpha=.20)
    ax.text(.50,.52,r'\textbf{Preliminary}',transform=ax.transAxes,fontsize=42,
            color='gray',alpha=.28,ha='center',va='center',rotation=30,zorder=0)
    ax.text(.58,.15,FIT_WATERMARK+'\nInterpolation with IV data, '
            +rf'$t/a^2\in[{window[0]:g},{window[1]:g}]$'+'\n'
            +rf'$\Delta(t/a^2)\simeq {PLOT_TIME_STEP:g}$'+'\nTLN',
            transform=ax.transAxes,fontsize=10.5,color='gray',alpha=.52,
            ha='center',va='center',fontweight='bold')
    fig.tight_layout()
    save_figure(fig,window,'interpolation_beta_over_g4_fit4_with_data')'''),
md('## Plot 4 — interpolation quality for every flow-time range'),
code(r'''for window in WINDOWS:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for operator in OBSERVABLES:
        rows=[]
        for time, qof in bf.ntrp_qof[FLOW][operator].items():
            if window[0] <= float(time) <= window[1]:
                rows.append((float(time), qof['chi2']/qof['dof'] if qof['dof'] else np.nan, qof['p-value']))
        if rows:
            rows=np.asarray(sorted(rows), float)
            ax1.plot(rows[:,0], rows[:,1], 'o-', color=OP_COLORS[operator], label=OP_LABELS[operator])
            ax2.plot(rows[:,0], rows[:,2], 'o-', color=OP_COLORS[operator])
    ax1.axhline(1, color='gray', ls='--'); ax2.axhline(.05, color='gray', ls='--')
    ax1.set_ylabel(r'$\chi^2/{\rm dof}$'); ax2.set_ylabel('$p$-value'); ax2.set_xlabel(r'$t/a^2$')
    ax1.legend(frameon=False); fig.suptitle(rf'Interpolation quality, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    save_figure(fig, window, 'interpolation_quality_fit4')'''),
md('## Plot helper — continuum extrapolations versus $a^2/t$'),
code(r'''def plot_continuum_panels(window, mode, divide_by_g4):
    case = load_case(window, mode)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    times = flow_times(window)
    xpts = np.asarray([1/float(t) for t in times])
    for ax, target in zip(axes.ravel(), TARGET_G2):
        for operator in OBSERVABLES:
            grid=np.asarray(case['g2s'][FLOW][operator], float)
            idx=int(np.argmin(abs(grid-target))); g2=float(grid[idx])
            values=np.asarray([bf.interpolation.model.evaluate(g2, bf.ntrp_fits[FLOW][operator][t]) for t in times], dtype=object)
            params=case['cnt_fits'][FLOW][operator][idx]
            xline=np.linspace(0, max(xpts)*1.05, 200)
            yline=params['beta'][0] + params['slope'][0]*xline
            norm=g2**2 if divide_by_g4 else 1.0
            ax.errorbar(xpts, gv.mean(values/norm), yerr=gv.sdev(values/norm), fmt='o', capsize=3,
                        color=OP_COLORS[operator], label=OP_LABELS[operator])
            ax.plot(xline, gv.mean(yline/norm), color=OP_COLORS[operator])
            ax.fill_between(xline, gv.mean(yline/norm)-gv.sdev(yline/norm),
                            gv.mean(yline/norm)+gv.sdev(yline/norm), color=OP_COLORS[operator], alpha=.15)
        ax.set(title=rf'$g^2_{{GF}}={g2:g}$', xlabel=r'$a^2/t$')
    axes.ravel()[0].legend(frameon=False)
    ylabel=r'$\beta_{GF}/g_{GF}^4$' if divide_by_g4 else r'$\beta_{GF}$'
    fig.supylabel(ylabel); fig.suptitle(rf'{mode} continuum, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    name='continuum_a2_over_t_beta_over_g4_fit4' if divide_by_g4 else 'continuum_a2_over_t_beta_fit4'
    save_figure(fig, window, name, mode)
'''),
md('## Plot 5 — raw $\beta_{GF}$ continuum panels for all ranges and modes'),
code(r'''for window in WINDOWS:
    for mode in ('diagonal', 'correlated'):
        plot_continuum_panels(window, mode, divide_by_g4=False)'''),
md('## Plot 6 — $\beta_{GF}/g_{GF}^4$ continuum panels for all ranges and modes'),
code(r'''for window in WINDOWS:
    for mode in ('diagonal', 'correlated'):
        plot_continuum_panels(window, mode, divide_by_g4=True)'''),
md('## Plot 7 — continuum fit quality for all ranges and modes'),
code(r'''for window in WINDOWS:
    for mode in ('diagonal', 'correlated'):
        case=load_case(window, mode); fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,8),sharex=True)
        for operator in OBSERVABLES:
            grid=np.asarray(case['g2s'][FLOW][operator],float)
            q=case['cnt_qof'][FLOW][operator]
            chi=np.asarray([row['chi2']/row['dof'] if row['dof'] else np.nan for row in q])
            pv=np.asarray([row['p-value'] for row in q])
            ax1.plot(grid,chi,'o-',color=OP_COLORS[operator],label=OP_LABELS[operator]); ax2.plot(grid,pv,'o-',color=OP_COLORS[operator])
        ax1.axhline(1,color='gray',ls='--'); ax2.axhline(.05,color='gray',ls='--')
        ax1.set_ylabel(r'$\chi^2/{\rm dof}$'); ax2.set_ylabel('$p$-value'); ax2.set_xlabel(r'$g^2_{GF}$')
        ax1.legend(frameon=False); fig.suptitle(rf'{mode} continuum quality, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
        save_figure(fig,window,'continuum_quality_fit4',mode)'''),
md('## Plot helper — final continuum curves'),
code(r'''def plot_final_curve(window, mode, divide_by_g4):
    case=load_case(window,mode); fig,ax=plt.subplots(figsize=(8,6))
    for operator in OBSERVABLES:
        x=np.asarray(case['g2s'][FLOW][operator],float); y=np.asarray(case['betas'][FLOW][operator],dtype=object)
        if divide_by_g4: y=y/x**2
        ax.errorbar(x,gv.mean(y),yerr=gv.sdev(y),fmt='o-',capsize=3,color=OP_COLORS[operator],label=OP_LABELS[operator])
        ax.fill_between(x,gv.mean(y)-gv.sdev(y),gv.mean(y)+gv.sdev(y),color=OP_COLORS[operator],alpha=.12)
    xp=np.linspace(.001,5,400)
    if divide_by_g4:
        for loops,ls in ((1,'-'),(2,'--'),(3,'-.')): ax.plot(xp,pt_over_g4(xp,loops),color='gray',ls=ls,label=f'{loops}-loop PT')
    ax.set(xlabel=r'$g^2_{GF}$',ylabel=r'$\beta_{GF}/g_{GF}^4$' if divide_by_g4 else r'$\beta_{GF}$',
           title=rf'{mode} continuum, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    ax.legend(ncol=2,frameon=False)
    name='continuum_beta_over_g4_vs_g2_fit4' if divide_by_g4 else 'continuum_beta_vs_g2_fit4'
    save_figure(fig,window,name,mode)'''),
md('## Plot 8 — final raw continuum curves for all ranges and modes'),
code(r'''for window in WINDOWS:
    for mode in ('diagonal','correlated'): plot_final_curve(window,mode,False)'''),
md('## Plot 9 — final scaled continuum curves for all ranges and modes'),
code(r'''for window in WINDOWS:
    for mode in ('diagonal','correlated'): plot_final_curve(window,mode,True)'''),
md('## Plot 10 — diagonal versus correlated continuum for every range'),
code(r'''for window in WINDOWS:
    diag=load_case(window,'diagonal'); corr=load_case(window,'correlated'); fig,ax=plt.subplots(figsize=(8,6))
    for operator in OBSERVABLES:
        for case,ls,label in ((diag,'--','diagonal'),(corr,'-','correlated')):
            x=np.asarray(case['g2s'][FLOW][operator],float); y=np.asarray(case['betas'][FLOW][operator],dtype=object)/x**2
            marker='o' if label=='correlated' else 's'
            ax.errorbar(x,gv.mean(y),yerr=gv.sdev(y),fmt=marker,ls=ls,ms=3,capsize=2,
                        color=OP_COLORS[operator],label=f'{OP_LABELS[operator]} {label}')
            ax.fill_between(x,gv.mean(y)-gv.sdev(y),gv.mean(y)+gv.sdev(y),
                            color=OP_COLORS[operator],alpha=.12 if label=='correlated' else .05)
    ax.set(xlabel=r'$g^2_{GF}$',ylabel=r'$\beta_{GF}/g_{GF}^4$',title=rf'Diagonal vs correlated, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    ax.legend(ncol=2,frameon=False); save_figure(fig,window,'diagonal_vs_correlated_fit4')'''),
md('## Plot 11 — flow-time correlation matrices for every range'),
code(r'''for window in WINDOWS:
    times=flow_times(window); values=np.asarray([
        bf.interpolation.model.evaluate(2.0,bf.ntrp_fits[FLOW]['p'][t]) for t in times
    ],dtype=object)
    corr=gv.evalcorr(values); fig,ax=plt.subplots(figsize=(6.5,5.8)); im=ax.imshow(corr,vmin=-1,vmax=1,origin='lower')
    fig.colorbar(im,ax=ax,label='Correlation'); labels=[f'{float(t):g}' for t in times]
    ax.set_xticks(range(len(times)),labels,rotation=45,ha='right'); ax.set_yticks(range(len(times)),labels)
    ax.set(xlabel=r'$t/a^2$',ylabel=r'$t/a^2$',title=rf'Wilson correlation, $g^2=2$, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    save_figure(fig,window,'flow_time_correlation_wilson_g2_2_fit4','correlated')'''),
md('''## Plot 12 — correlated weak-coupling extrapolation for every range

Point error bars are continuum-point uncertainties; colored bands are posterior uncertainties of the shared fitted curve and are not expected to coincide.'''),
code(r'''def ratio_model(x,p):
    u=np.asarray(x)/bf.perturbative_beta_function.nrm; correction=1.0; term=1.0
    for n in range(1,ORDER+1): term*=u; correction+=p[f'c{n}'][0]*term
    return pt_over_g4(x,3)*correction

for window in WINDOWS:
    case=load_case(window,'correlated'); fig,ax=plt.subplots(figsize=(8,6))
    for operator in OBSERVABLES:
        x=np.asarray(case['g2s'][FLOW][operator],float); y=np.asarray(case['betas'][FLOW][operator],dtype=object)/x**2
        prior={f'c{n}':[gv.gvar(0,10)] for n in range(1,ORDER+1)}
        fit=lsqfit.nonlinear_fit(data=(x,y),fcn=ratio_model,prior=prior)
        xp=np.linspace(0,max(x),400); yp=np.asarray(ratio_model(xp,fit.p),dtype=object)
        ax.errorbar(x,gv.mean(y),yerr=gv.sdev(y),fmt='o',capsize=3,color=OP_COLORS[operator])
        ax.plot(xp,gv.mean(yp),color=OP_COLORS[operator],label=OP_LABELS[operator])
        ax.fill_between(xp,gv.mean(yp)-gv.sdev(yp),gv.mean(yp)+gv.sdev(yp),color=OP_COLORS[operator],alpha=.15)
        print(window,OP_LABELS[operator],'origin=',ratio_model(np.array([0.]),fit.p)[0],'Q=',fit.Q)
    threshold=min(min(case['g2s'][FLOW][o]) for o in OBSERVABLES); ax.axvspan(0,threshold,color='gray',alpha=.1,label='extrapolated')
    xp=np.linspace(0,5,400)
    for loops,ls in ((1,'-'),(2,'--'),(3,'-.')): ax.plot(xp,pt_over_g4(xp,loops),color='black',ls=ls,alpha=.6,label=f'{loops}-loop PT')
    ax.set(xlim=(0,5),xlabel=r'$g^2_{GF}$',ylabel=r'$\beta_{GF}/g_{GF}^4$',title=rf'Weak-coupling fit, $t/a^2\in[{window[0]:g},{window[1]:g}]$')
    ax.legend(ncol=2,frameon=False); save_figure(fig,window,'weak_coupling_to_zero_fit4','correlated')'''),
md('''## Reviewed weak-coupling diagnostics

Generate the narrow Figure-11-style integral match and the separate diagnostic
that extends the PT-constrained intermediate interpolants before taking the
kernel-correlated continuum limit.'''),
code(r'''from betafn.weak_coupling_plots import run_reviewed_weak_coupling_plots

reviewed_weak_results = run_reviewed_weak_coupling_plots(
    bf=bf, flow=FLOW, observables=OBSERVABLES, windows=WINDOWS,
    config=config, fit_id=FIT_ID, fit_watermark=FIT_WATERMARK,
    op_colors=OP_COLORS, op_labels=OP_LABELS, flow_times=flow_times,
    pt_over_g4=pt_over_g4, load_case=load_case, save_figure=save_figure,
)'''),
md('## Scan manifest and validation summary'),
code(r'''rows=[]
for window in WINDOWS:
    for mode in ('diagonal','correlated'):
        case=load_case(window,mode)
        for operator in OBSERVABLES:
            q=case['cnt_qof'][FLOW][operator]
            rows.append({'tmin':window[0],'tmax':window[1],'mode':mode,'operator':operator,
                         'nfits':len(q),'mean_chi2_dof':np.mean([r['chi2']/r['dof'] for r in q if r['dof']]),
                         'mean_pvalue':np.mean([r['p-value'] for r in q])})
summary=pd.DataFrame(rows)
summary.to_csv(OUTPUT_ROOT/'fit4_scan_summary.csv',index=False)
display(summary)

expected={(window,mode) for window in WINDOWS for mode in ('diagonal','correlated')}
present={(window,mode) for window in WINDOWS for mode in ('diagonal','correlated')
         if (case_dir(window,mode)/'continuum_case.gvar').is_file()}
assert present==expected, f'missing cases: {expected-present}'
assert config.binsize==15 and config.correction=='tln'
assert config.combine=={'s': {'p':5/3,'s':-2/3}}
print('fit4 validation complete:',len(present),'continuum cases')'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUT)
