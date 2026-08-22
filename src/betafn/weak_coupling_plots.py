"""Reviewed weak-coupling plot families shared by notebooks and batch runs."""

from __future__ import annotations

import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

from .weak_coupling import continuum_from_extended_interpolants, figure11_integral_match


def run_reviewed_weak_coupling_plots(
    *, bf, flow, observables, windows, config, fit_id, fit_watermark,
    op_colors, op_labels, flow_times, pt_over_g4, load_case, save_figure,
):
    """Generate integral-matching and extended-interpolant diagnostics."""
    match_window = (0.8, 1.2)
    match_g2 = np.linspace(*match_window, 9)
    xmax = 3.0
    figure11_results = {}

    for window in windows:
        case = load_case(window, "correlated")
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharex=True, sharey=True)
        figure11_results[window] = {}
        for ax, operator in zip(axes, observables):
            matched = continuum_from_extended_interpolants(
                bf, flow, operator, window, match_g2, tau0=config.tau0,
                cov_mode="kernel", kernel="rbf", enforce_domains=True,
            )
            x = np.asarray(case["g2s"][flow][operator], dtype=float)
            y = np.asarray(case["betas"][flow][operator], dtype=object) / x**2
            result = figure11_integral_match(
                matched["g2"], matched["beta_over_g4"], pt_over_g4,
                match_window=match_window,
            )
            result["matching_continuum"] = matched
            figure11_results[window][operator] = result

            order = np.argsort(x)
            x, y = x[order], y[order]
            shown = x <= xmax
            ax.plot(x[shown], gv.mean(y[shown]), color=op_colors[operator], lw=1.8,
                    label=f"{op_labels[operator]} correlated continuum")
            ax.fill_between(x[shown], gv.mean(y[shown])-gv.sdev(y[shown]),
                            gv.mean(y[shown])+gv.sdev(y[shown]),
                            color=op_colors[operator], alpha=.18, linewidth=0)
            mx, my = matched["g2"], matched["beta_over_g4"]
            ax.plot(mx, gv.mean(my), color=op_colors[operator], lw=2.2)
            ax.fill_between(mx, gv.mean(my)-gv.sdev(my), gv.mean(my)+gv.sdev(my),
                            color=op_colors[operator], alpha=.24, linewidth=0,
                            label="domain-restricted matching segment")

            xp = np.linspace(0.0, match_window[1], 500)
            central = result["ratio_curve"](xp, "central")
            plus = result["ratio_curve"](xp, "plus_sigma")
            minus = result["ratio_curve"](xp, "minus_sigma")
            ax.fill_between(xp, np.minimum(plus, minus), np.maximum(plus, minus),
                            color="purple", alpha=.24, linewidth=0,
                            label=r"matched $\pm1\sigma$ limits")
            ax.plot(xp, central, color="royalblue", lw=2.1,
                    label=r"integral-matched $\beta_4$")
            ax.axvspan(*match_window, facecolor="none", edgecolor="gray",
                       hatch="///", lw=0, alpha=.45, label="matching interval")
            _add_pt_curves(ax, np.linspace(0.0, xmax, 500), pt_over_g4)
            _decorate(ax, fit_watermark,
                      rf"Fig. 11 integral match: $g^2\in[{match_window[0]:g},{match_window[1]:g}]$",
                      window, "TLN")
            ax.set(xlim=(0, xmax), xlabel=r"$g^2_{GF}$")
            ax.legend(fontsize=7.5, frameon=False, loc="best")
            print(window, op_labels[operator], "Fig.11 b_p:", result["coefficients"])
        axes[0].set_ylabel(r"$\beta_{GF}/g_{GF}^4$")
        save_figure(fig, window, f"figure11_integral_matching_{fit_id}", "correlated")

    direct_grid = np.linspace(0.0, 2.0, 21)
    display_x = np.linspace(1e-7, direct_grid[-1], 350)
    direct_results = {}
    correction_label = (
        "TLN" if config.correction in ("tln", "tree-level-normalization") else "No TLN"
    )
    for window in windows:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharex=True, sharey=True)
        for ax, operator in zip(axes, observables):
            times = flow_times(window)
            for index, time in enumerate(times):
                curve = np.asarray(
                    bf.interpolation.model.evaluate(
                        display_x, bf.ntrp_fits[flow][operator][time]
                    ), dtype=object,
                ) / display_x**2
                domain = tuple(map(float, bf.ntrp_nf[flow][operator][time]))
                alpha = .22 + .65 * index / max(len(times)-1, 1)
                ax.plot(display_x, gv.mean(curve), color=op_colors[operator],
                        alpha=alpha, lw=1.1)
                ax.axvline(domain[0], color=op_colors[operator], alpha=.10, lw=.7)
            _add_pt_curves(ax, display_x, pt_over_g4)
            _decorate(ax, fit_watermark, "Extended intermediate interpolation",
                      window, correction_label)
            ax.set(xlim=(0, direct_grid[-1]), xlabel=r"$g^2_{GF}$")
            ax.legend(fontsize=8, frameon=False)
        axes[0].set_ylabel(r"$\beta_{GF}/g_{GF}^4$")
        save_figure(fig, window, f"extended_intermediate_to_zero_{fit_id}",
                    "correlated_diagnostic")

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharex=True, sharey=True)
        direct_results[window] = {}
        for ax, operator in zip(axes, observables):
            result = continuum_from_extended_interpolants(
                bf, flow, operator, window, direct_grid, tau0=config.tau0,
                cov_mode="kernel", kernel="rbf",
            )
            direct_results[window][operator] = result
            x, y = result["g2"], result["beta_over_g4"]
            supported_from = max(d[0] for d in result["measured_domains"].values())
            ax.axvspan(0, min(supported_from, x[-1]), color="gray", alpha=.10,
                       label="intermediate-fit extrapolation")
            ax.plot(x, gv.mean(y), color=op_colors[operator], lw=2,
                    label=f"{op_labels[operator]} extended continuum")
            ax.fill_between(x, gv.mean(y)-gv.sdev(y), gv.mean(y)+gv.sdev(y),
                            color=op_colors[operator], alpha=.22, linewidth=0,
                            label=r"continuum $\pm1\sigma$")
            _add_pt_curves(ax, np.linspace(0, direct_grid[-1], 400), pt_over_g4)
            _decorate(ax, fit_watermark, "Extended-interpolation continuum diagnostic",
                      window, correction_label)
            ax.set(xlim=(0, direct_grid[-1]), xlabel=r"$g^2_{GF}$")
            ax.legend(fontsize=8, frameon=False)
            print(window, op_labels[operator], "direct weak continuum QoF:",
                  result["quality"])
        axes[0].set_ylabel(r"$\beta_{GF}/g_{GF}^4$")
        save_figure(fig, window,
                    f"extended_interpolation_continuum_to_zero_{fit_id}",
                    "correlated_diagnostic")
    return {"figure11": figure11_results, "direct": direct_results}


def _add_pt_curves(ax, x, pt_over_g4):
    for loops, linestyle, label in (
        (1, "-", "1-loop universal"),
        (2, "--", "2-loop universal"),
        (3, "-.", "3-loop gradient flow"),
    ):
        ax.plot(x, pt_over_g4(x, loops), color="gray", ls=linestyle,
                lw=1.2, alpha=.75, label=label)


def _decorate(ax, watermark, description, window, correction):
    ax.text(.50, .53, "Preliminary", transform=ax.transAxes, fontsize=29,
            color="gray", alpha=.23, ha="center", va="center", rotation=30,
            zorder=20, clip_on=True)
    ax.text(.55, .13, watermark + "\n" + description + "\n"
            + rf"$t/a^2\in[{window[0]:g},{window[1]:g}]$, {correction}",
            transform=ax.transAxes, fontsize=8.8, color="gray", alpha=.82,
            ha="center", va="center", fontweight="bold", zorder=20)
