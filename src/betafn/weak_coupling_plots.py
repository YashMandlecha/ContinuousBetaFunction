"""Reviewed weak-coupling plot families shared by notebooks and batch runs."""

from __future__ import annotations

import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import pandas as pd

from .blinding import add_blinded_watermark, blind_lambda_estimate
from .weak_coupling import (
    continuum_from_extended_interpolants,
    figure11_integral_match,
    lambda_parameter_from_matched_beta,
)


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
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        figure11_results[window] = {}
        bp_lines = [r"$b_p$ (central; shifted-data range)"]
        for operator in observables:
            matched = continuum_from_extended_interpolants(
                bf, flow, operator, window, match_g2, tau0=config.tau0,
                cov_mode="empirical", enforce_domains=True,
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
                            color=op_colors[operator], alpha=.24, linewidth=0)

            xp = np.linspace(0.0, match_window[1], 500)
            central = result["ratio_curve"](xp, "central")
            plus = result["ratio_curve"](xp, "plus_sigma")
            minus = result["ratio_curve"](xp, "minus_sigma")
            ax.fill_between(xp, np.minimum(plus, minus), np.maximum(plus, minus),
                            color=op_colors[operator], alpha=.12, linewidth=0)
            ax.plot(xp, central, color=op_colors[operator], ls="--", lw=2.1,
                    label=rf"{op_labels[operator]} integral-matched $\beta_4$")

            coefficients = result["coefficients"]
            coefficient_low = min(
                coefficients["plus_sigma"], coefficients["minus_sigma"]
            )
            coefficient_high = max(
                coefficients["plus_sigma"], coefficients["minus_sigma"]
            )
            bp_lines.append(
                f"{op_labels[operator]}: {coefficients['central']:.4g} "
                f"[{coefficient_low:.4g}, {coefficient_high:.4g}]"
            )
            print(
                window, op_labels[operator], "Integral-match b_p:", coefficients
            )

        ax.axvspan(*match_window, facecolor="none", edgecolor="gray",
                   hatch="///", lw=0, alpha=.45, label="matching interval")
        _add_pt_curves(ax, np.linspace(0.0, xmax, 500), pt_over_g4)
        _decorate(
            ax,
            fit_watermark,
            rf"Integral match: $g^2\in[{match_window[0]:g},{match_window[1]:g}]$",
            window,
            "TLN",
            detail_fontsize=9.2,
            watermark_fontsize=34,
        )
        ax.text(
            .025, .035, "\n".join(bp_lines), transform=ax.transAxes,
            fontsize=9.2, ha="left", va="bottom", color="black",
            bbox={
                "boxstyle": "round,pad=.35", "facecolor": "white",
                "edgecolor": "0.65", "alpha": .88,
            },
            zorder=30,
        )
        ax.set(
            xlim=(0, xmax), xlabel=r"$g^2_{GF}$",
            ylabel=r"$\beta_{GF}/g_{GF}^4$",
        )
        ax.legend(
            fontsize=8.2, frameon=False, loc="upper right", ncol=2,
            columnspacing=.8, handlelength=2.2, labelspacing=.35,
        )
        fig.tight_layout()
        save_figure(fig, window, f"figure11_integral_matching_{fit_id}", "correlated")

    # beta/g^4 is singular at the origin when the interpolation contains
    # beta terms below x^2 (x = g_GF^2).  Those models still admit the same
    # extended-interpolation/continuum diagnostic at strictly positive weak
    # coupling.  Starting them at g^2=0.1 avoids presenting a regulator-sized
    # value as an origin limit.  Models with a finite origin retain the
    # original grid, including g^2=0.
    origin_singular = fit_id in {"fit1", "fit2", "fit11", "fit12"}
    direct_min = 0.1 if origin_singular else 0.0
    direct_max = 2.0
    direct_grid = np.linspace(
        direct_min,
        direct_max,
        int(round((direct_max - direct_min) / 0.1)) + 1,
    )
    display_x = np.linspace(max(direct_min, 1e-7), direct_max, 350)
    if origin_singular:
        diagnostic_detail = (
            rf"Extended toward weak coupling; $g^2_{{GF}}\geq{direct_min:g}$ "
            r"(origin singular)"
        )
    else:
        diagnostic_detail = "Extended to zero coupling"
    plot_watermark = fit_watermark.replace(", $x$ errors off", "")
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
            if origin_singular:
                ax.axvspan(
                    0.0, direct_min, color="tab:red", alpha=.06,
                    label=r"excluded: $\beta/g^4$ singular",
                )
            _add_pt_curves(ax, np.linspace(0.0, direct_max, 400), pt_over_g4)
            _decorate(
                ax, plot_watermark, diagnostic_detail, window, correction_label,
                detail_fontsize=9.6,
            )
            ax.set(xlim=(0, direct_grid[-1]), xlabel=r"$g^2_{GF}$")
            ax.legend(fontsize=8.8, frameon=False)
        axes[0].set_ylabel(r"$\beta_{GF}/g_{GF}^4$")
        save_figure(fig, window, f"extended_intermediate_to_zero_{fit_id}",
                    "correlated_diagnostic")

        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        direct_results[window] = {}
        supported_from_by_operator = []
        for operator in observables:
            result = continuum_from_extended_interpolants(
                bf, flow, operator, window, direct_grid, tau0=config.tau0,
                cov_mode="empirical",
            )
            direct_results[window][operator] = result
            x, y = result["g2"], result["beta_over_g4"]
            supported_from_by_operator.append(
                max(d[0] for d in result["measured_domains"].values())
            )
            ax.plot(x, gv.mean(y), color=op_colors[operator], lw=2,
                    label=f"{op_labels[operator]} extended continuum")
            ax.fill_between(x, gv.mean(y)-gv.sdev(y), gv.mean(y)+gv.sdev(y),
                            color=op_colors[operator], alpha=.22, linewidth=0)
            print(window, op_labels[operator], "direct weak continuum QoF:",
                  result["quality"])

        if origin_singular:
            ax.axvspan(
                0.0, direct_min, color="tab:red", alpha=.06,
                label=r"excluded: $\beta/g^4$ singular",
            )
        fully_supported_from = max(supported_from_by_operator)
        ax.axvspan(
            direct_min, min(fully_supported_from, direct_grid[-1]),
            color="gray", alpha=.10,
            label="intermediate-fit extrapolation for at least one operator",
        )
        _add_pt_curves(ax, np.linspace(0, direct_grid[-1], 400), pt_over_g4)
        _decorate(
            ax, plot_watermark, diagnostic_detail, window, correction_label,
            detail_fontsize=10.2, watermark_fontsize=34,
        )
        ax.set(
            xlim=(0, direct_grid[-1]), xlabel=r"$g^2_{GF}$",
            ylim=(-0.065, -0.035), ylabel=r"$\beta_{GF}/g_{GF}^4$",
        )
        ax.legend(
            fontsize=9.2, frameon=False, loc="upper right", ncol=2,
            columnspacing=.8, handlelength=2.2, labelspacing=.35,
        )
        fig.tight_layout()
        save_figure(fig, window,
                    f"extended_interpolation_continuum_to_zero_{fit_id}",
                    "correlated_diagnostic")
    return {"figure11": figure11_results, "direct": direct_results}


def run_lambda_parameter_plots(
    *, bf, flow, observables, windows, fit_id, fit_footer, op_colors,
    op_labels, window_tag, load_case, figure11_results, output_root,
    plot_msbar=False,
):
    """Write the per-window Lambda table and the blinded t0 summary figure."""
    nc = 3
    g2_t0 = 0.3 * 128.0 * np.pi**2 / (3.0 * nc**2 - 3.0)
    lambda_results = {}
    rows = []

    for window in windows:
        case = load_case(window, "correlated")
        lambda_results[window] = {}
        for operator in observables:
            x = np.asarray(case["g2s"][flow][operator], dtype=float)
            y = np.asarray(case["betas"][flow][operator], dtype=object) / x**2
            order = np.argsort(x)
            x, y = x[order], y[order]

            available_g2_max = float(x[-1])
            if available_g2_max < g2_t0 - 1e-10:
                raise RuntimeError(
                    f"{window}, {operator}: continuum support ends at "
                    f"g^2={available_g2_max:g}, below g^2(t0)={g2_t0:g}"
                )
            reference_g2 = g2_t0
            reference_scale = "t0"

            estimate = blind_lambda_estimate(
                lambda_parameter_from_matched_beta(
                    x,
                    y,
                    figure11_results[window][operator],
                    bf.perturbative_beta_function,
                    reference_g2=reference_g2,
                )
            )
            lambda_results[window][operator] = estimate

            conversion = estimate["lambda_msbar_over_lambda_gf"]
            for label in ("central", "plus_sigma", "minus_sigma"):
                np.testing.assert_allclose(
                    estimate["lambda_msbar_over_mu"][label],
                    conversion * estimate["lambda_gf_over_mu"][label],
                    rtol=1e-13,
                    atol=0.0,
                )

            row = {
                "window": window_tag(window),
                "operator": operator,
                "operator_label": op_labels[operator],
                "reference_scale": reference_scale,
                "g2_reference": reference_g2,
                "g2_supported_max": available_g2_max,
                "g2_t0": g2_t0,
                "reaches_t0": True,
                "lambda_values_blinded": True,
                "lambda_msbar_over_lambda_gf": conversion,
            }
            for scheme, values in (
                ("GF", estimate["lambda_gf_over_mu"]),
                ("MSbar", estimate["lambda_msbar_over_mu"]),
            ):
                central = values["central"]
                envelope_low = min(
                    central, values["plus_sigma"], values["minus_sigma"]
                )
                envelope_high = max(
                    central, values["plus_sigma"], values["minus_sigma"]
                )
                row[f"sqrt8t0_lambda_{scheme}_central"] = central
                row[f"sqrt8t0_lambda_{scheme}_envelope_low"] = envelope_low
                row[f"sqrt8t0_lambda_{scheme}_envelope_high"] = envelope_high
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(
        output_root / f"lambda_parameter_by_window_{fit_id}.csv", index=False
    )
    print(table.to_string(index=False))

    window_keys = [window_tag(window) for window in windows]
    window_labels = [key.replace("t_", "").replace("_", "-") for key in window_keys]
    schemes = [("GF", r"$\sqrt{8t_0}\,\Lambda_{\rm GF}$")]
    if plot_msbar:
        schemes.append(
            ("MSbar", r"$\sqrt{8t_0}\,\Lambda_{\overline{\rm MS}}$")
        )
    fig, axes = plt.subplots(
        len(schemes), 1, figsize=(12, 5.8 * len(schemes)),
        sharex=True, squeeze=False,
    )
    axes = axes.ravel()
    for ax, (scheme, ylabel) in zip(axes, schemes):
        for offset, operator in zip((-0.18, 0.0, 0.18), observables):
            subset = table.query("operator == @operator").set_index("window").loc[
                window_keys
            ]
            central = subset[f"sqrt8t0_lambda_{scheme}_central"].to_numpy()
            lower = subset[f"sqrt8t0_lambda_{scheme}_envelope_low"].to_numpy()
            upper = subset[f"sqrt8t0_lambda_{scheme}_envelope_high"].to_numpy()
            ax.errorbar(
                np.arange(len(windows)) + offset,
                central,
                yerr=np.vstack((central - lower, upper - central)),
                fmt="o",
                capsize=3,
                color=op_colors[operator],
                label=op_labels[operator],
            )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=3)

    axes[-1].set_xticks(
        np.arange(len(windows)), window_labels, rotation=45, ha="right"
    )
    axes[-1].set_xlabel(r"flow-time window $t/a^2$")
    fig.suptitle(
        rf"{fit_id}: BLINDED $\Lambda$ estimates at "
        rf"$g^2_{{\rm GF}}(t_0)={g2_t0:.9f}$"
    )
    fig.tight_layout()
    if fit_footer is not None:
        fig.text(
            0.995, 0.005, fit_footer, ha="right", va="bottom",
            fontsize=8, color="gray", alpha=.75,
        )
    add_blinded_watermark(fig)
    base = output_root / f"lambda_parameter_by_window_{fit_id}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"table": table, "estimates": lambda_results, "g2_t0": g2_t0}


def _add_pt_curves(ax, x, pt_over_g4):
    for loops, linestyle, label in (
        (1, "-", "1-loop universal"),
        (2, "--", "2-loop universal"),
        (3, "-.", "3-loop gradient flow"),
    ):
        ax.plot(x, pt_over_g4(x, loops), color="gray", ls=linestyle,
                lw=1.2, alpha=.75, label=label)


def _decorate(
    ax, watermark, description, window, correction,
    detail_fontsize=8.8, watermark_fontsize=29,
):
    ax.text(.50, .53, "Preliminary", transform=ax.transAxes,
            fontsize=watermark_fontsize,
            color="gray", alpha=.23, ha="center", va="center", rotation=30,
            zorder=20, clip_on=True)
    ax.text(.55, .13, watermark + "\n" + description + "\n"
            + rf"$t/a^2\in[{window[0]:g},{window[1]:g}]$, {correction}",
            transform=ax.transAxes, fontsize=detail_fontsize,
            color="gray", alpha=.82,
            ha="center", va="center", fontweight="bold", zorder=20)
