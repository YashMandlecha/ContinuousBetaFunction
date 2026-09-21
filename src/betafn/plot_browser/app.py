"""Streamlit browser for the NF4 beta-function analysis outputs."""

from __future__ import annotations

import base64
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SEARCH_ROOTS = (ROOT, ROOT.parent)
DATA_ROOT = Path("/Users/yaman/Contbetafn/data/New Four")

st.set_page_config(page_title="Gradient-flow beta-function plot browser", layout="wide")
st.markdown(
    """
    <style>
    h1, h2, h3 { font-family: "Computer Modern Serif", "CMU Serif",
        "Latin Modern Roman", "KaTeX_Main", "Times New Roman", serif; }
    .subtitle { font-size: 1.05rem; color: #888; margin-top: -0.5rem;
                margin-bottom: 1.2rem; }
    .small-muted { color: #666; font-size: 0.92rem; }
    iframe { border: 1px solid #ddd; border-radius: 8px; }
    .ensemble-table-wrap { display: flex; justify-content: center; }
    table.ensemble-table { width: auto; border-collapse: collapse; font-size: 0.95rem; }
    table.ensemble-table th, table.ensemble-table td {
        padding: 0.28rem 0.8rem; text-align: center; white-space: nowrap;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }
    table.ensemble-table th { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(r"# $\alpha_s$ gradient-flow $\beta$-function plot browser")
st.markdown(
    '<div class="subtitle">Fermilab Lattice and MILC collaboration</div>',
    unsafe_allow_html=True,
)
st.markdown(
    r"Define $x=g_{\rm GF}^2$, $u=x/(4\pi)$, and "
    r"$R(x)=\beta_{\rm GF}(x)/x^2$. The models below differ only in the "
    "interpolation ansatz; data processing, infinite-volume extrapolation, "
    "continuum analysis, and diagnostic procedures are otherwise shared."
)

with st.expander("Interpolation models and priors", expanded=True):
    (fit4_tab, fit5_tab, fit6_tab, fit7_tab, fit8_tab, fit9_tab,
     fit10_tab, fit11_tab) = st.tabs(
        ("fit4", "fit5", "fit6", "fit7", "fit8", "fit9", "fit10", "fit11")
    )
    with fit4_tab:
        st.markdown("**Multiplicative correction to fixed three-loop perturbation theory**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_nu^n\right]"
        )
        st.markdown(
            "Runs use correction orders $N=3,4,5,6$. For `fit4_width1000`, "
            "each coefficient has the independent prior "
            r"$c_n\sim\mathcal N(0,1000^2)$ and `xerrors=True`. For "
            "`fit4_nopriors`, all $c_n$ are unconstrained fit parameters and "
            "`xerrors=False`. The perturbative prefactor is fixed exactly."
        )
    with fit5_tab:
        st.markdown("**All three known perturbative coefficients fixed**")
        st.latex(
            r"R(x)=R_{\rm PT}^{(3)}(x)+"
            r"\sum_{k=3}^{2+N}d_ku^k,\qquad N\in\{1,2\}"
        )
        st.markdown(
            "Order 1 fits $d_3$; order 2 fits $d_3,d_4$. Every fitted "
            "coefficient is unconstrained: there are no coefficient priors, "
            "and both variants use `xerrors=False`."
        )
    with fit6_tab:
        st.markdown("**One- and three-loop coefficients fixed; two-loop coefficient free**")
        st.latex(
            r"R(x)=A_0+b_1u+A_2u^2+"
            r"\sum_{k=3}^{2+N}d_ku^k,\qquad N\in\{1,2\}"
        )
        st.markdown(
            "$A_0$ and $A_2$ are fixed at their perturbative values. The "
            "normalized two-loop coefficient $b_1$ and each included $d_k$ "
            "are unconstrained. Order 1 fits $b_1,d_3$; order 2 fits "
            "$b_1,d_3,d_4$. There are no coefficient priors and "
            "`xerrors=False`."
        )
    with fit7_tab:
        st.markdown("**One- and two-loop coefficients fixed; three-loop coefficient free**")
        st.latex(
            r"R(x)=A_0+A_1u+b_2u^2+"
            r"\sum_{k=3}^{2+N}d_ku^k,\qquad N\in\{1,2\}"
        )
        st.markdown(
            "$A_0$ and $A_1$ are fixed at their perturbative values. The "
            "normalized three-loop coefficient $b_2$ and each included $d_k$ "
            "are unconstrained. Order 1 fits $b_2,d_3$; order 2 fits "
            "$b_2,d_3,d_4$. There are no coefficient priors and "
            "`xerrors=False`."
        )
    with fit8_tab:
        st.markdown("**Multiplicative fit with $c_1$ fixed to zero**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=2}^{N}c_nu^n\right],\qquad c_1=0,\quad N\in\{3,4\}"
        )
        st.markdown(
            "Order 3 fits $c_2,c_3$; order 4 fits $c_2,c_3,c_4$. "
            "All fitted coefficients are unconstrained: there are no priors "
            "and `xerrors=False`."
        )
    with fit9_tab:
        st.markdown("**Multiplicative fit with $c_2$ fixed to zero**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+c_1u+\sum_{n=3}^{N}c_nu^n\right],"
            r"\qquad c_2=0,\quad N\in\{3,4\}"
        )
        st.markdown(
            "Order 3 fits $c_1,c_3$; order 4 fits $c_1,c_3,c_4$. "
            "All fitted coefficients are unconstrained: there are no priors "
            "and `xerrors=False`."
        )
    with fit10_tab:
        st.markdown("**Free multiplicative intercept with $c_1=c_2=0$**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[c_0+\sum_{n=3}^{N}c_nu^n\right],"
            r"\qquad c_1=c_2=0,\quad N\in\{3,4\}"
        )
        st.markdown(
            "Order 3 fits $c_0,c_3$; order 4 fits $c_0,c_3,c_4$. "
            "All fitted coefficients are unconstrained: there are no priors "
            "and `xerrors=False`."
        )
    with fit11_tab:
        st.markdown("**Free additive constant in beta plus the Fit 4 correction**")
        st.latex(
            r"\beta(x)=a_0+\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_nu^n\right],"
            r"\qquad N\in\{3,4\}"
        )
        st.latex(
            r"R(x)=\frac{a_0}{x^2}+R_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_nu^n\right]"
        )
        st.markdown(
            "The additive beta-function constant $a_0$ is fitted independently "
            "for each operator and finite flow time. In this initial variant, "
            "$a_0$ and all included $c_n$ are unconstrained: there are no "
            "coefficient priors and `xerrors=False`. The $a_0/x^2$ behavior "
            "makes this a discretization diagnostic rather than a valid "
            "standalone weak-coupling continuum ansatz. Consequently, the "
            "finite-spacing extended-to-zero diagnostics are intentionally "
            "not generated for Fit 11."
        )

    st.markdown("**Fixed $N_f=4$, SU(3), gradient-flow perturbative reference**")
    st.latex(
        r"R_{\rm PT}^{(3)}(x)="
        r"-0.0527714498137-0.00205854331669x+0.000919267536173x^2"
    )
    st.caption(
        "In the normalized-u notation, A₀ = −0.0527714498137, "
        "A₁ = −0.0258684182433, and A₂ = 0.145164910733. "
        "The symbols b₁ and b₂ above denote fitted normalized-u coefficients, "
        "not additional fixed perturbative constants. Fits 5–11 use no "
        "coefficient priors, so their free coefficients may take either sign."
    )
    st.markdown(
        "**Common analysis settings:** continuum points use the 41-value grid "
        r"$x=0.9,1.0,\ldots,4.9$; all ten integer flow-time windows from "
        "$[4,5]$ through $[7,8]$ are analyzed with both diagonal and "
        "kernel-correlated continuum fits."
    )
    st.markdown(
        "| Variant | Output model tag | Fitted parameters |\n"
        "|---|---|---|\n"
        "| fit5 order 1 | `pt_fixed_additive_u3_nopriors` | $d_3$ |\n"
        "| fit5 order 2 | `pt_fixed_additive_u3_u4_nopriors` | $d_3,d_4$ |\n"
        "| fit6 order 1 | `two_loop_free_additive_u3_nopriors` | $b_1,d_3$ |\n"
        "| fit6 order 2 | `two_loop_free_additive_u3_u4_nopriors` | $b_1,d_3,d_4$ |\n"
        "| fit7 order 1 | `three_loop_free_additive_u3_nopriors` | $b_2,d_3$ |\n"
        "| fit7 order 2 | `three_loop_free_additive_u3_u4_nopriors` | $b_2,d_3,d_4$ |\n"
        "| fit8 order 3 | `order_3_c1_fixed_0` | $c_2,c_3$ |\n"
        "| fit8 order 4 | `order_4_c1_fixed_0` | $c_2,c_3,c_4$ |\n"
        "| fit9 order 3 | `order_3_c2_fixed_0` | $c_1,c_3$ |\n"
        "| fit9 order 4 | `order_4_c2_fixed_0` | $c_1,c_3,c_4$ |\n"
        "| fit10 order 3 | `order_3_c1_c2_fixed_0_free_c0` | $c_0,c_3$ |\n"
        "| fit10 order 4 | `order_4_c1_c2_fixed_0_free_c0` | $c_0,c_3,c_4$ |\n"
        "| fit11 order 3 | `order_3_additive_beta_constant_nopriors` | $a_0,c_1,c_2,c_3$ |\n"
        "| fit11 order 4 | `order_4_additive_beta_constant_nopriors` | $a_0,c_1,c_2,c_3,c_4$ |"
    )


def make_widget_key(prefix: str, value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value))
    return f"{prefix}_{cleaned[-150:]}"


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def embed_pdf(path: Path, height: int = 780) -> None:
    pdf_bytes = path.read_bytes()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" '
        f'height="{height}" type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download selected file", data=pdf_bytes, file_name=path.name,
        mime="application/pdf", key=make_widget_key("download_pdf", path),
    )


def display_plot(path: Path, height: int = 780) -> None:
    if path.suffix.lower() == ".pdf":
        embed_pdf(path, height=height)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        st.download_button(
            "Download selected file", data=path.read_bytes(), file_name=path.name,
            mime="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
            key=make_widget_key("download_image", path),
        )
        st.image(str(path), width="stretch")
    else:
        st.warning("Unsupported file type.")


def configuration_sort_key(name: str) -> tuple[int, float]:
    if name == "fit4_nopriors":
        return 0, 0.0
    match = re.fullmatch(r"fit4_width(.+)", name)
    if not match:
        return 2, float("inf")
    try:
        return 1, float(match.group(1).replace("p", "."))
    except ValueError:
        return 2, float("inf")


def configuration_label(path: Path) -> str:
    if path.name == "fit4_nopriors":
        return "No coefficient priors — xerrors=False"
    match = re.fullmatch(r"fit4_width(.+)", path.name)
    if match:
        return f"Prior width {match.group(1).replace('p', '.')} — xerrors=True"
    return path.name


def order_sort_key(path: Path) -> int:
    match = re.fullmatch(r"order_(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def window_sort_key(path: Path) -> tuple[float, float]:
    match = re.fullmatch(r"t_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)", path.name)
    if not match:
        return float("inf"), float("inf")
    return tuple(float(value.replace("p", ".")) for value in match.groups())


def pretty_order(path: Path) -> str:
    return f"Order {path.name.removeprefix('order_')}"


def pretty_window(path: Path) -> str:
    match = re.fullmatch(r"t_(.+)_(.+)", path.name)
    if not match:
        return path.name
    return ("t/a² ∈ [" + match.group(1).replace("p", ".") + ", "
            + match.group(2).replace("p", ".") + "]")


def pretty_mode(mode: str) -> str:
    return {
        "window": "Window-level / interpolation",
        "diagonal": "Continuum — diagonal covariance",
        "correlated": "Continuum — correlated covariance",
        "correlated_diagnostic": "Correlated weak-coupling diagnostics",
    }.get(mode, mode)


@st.cache_data(show_spinner=False)
def discover_configurations() -> list[str]:
    found: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        for candidate in root.glob("fit4_*"):
            if not candidate.is_dir():
                continue
            valid = candidate.name == "fit4_nopriors" or candidate.name.startswith("fit4_width")
            if valid:
                found.setdefault(candidate.name, candidate.resolve())
    return [str(found[name]) for name in sorted(found, key=configuration_sort_key)]


def list_orders(configuration: Path) -> list[Path]:
    if not configuration.is_dir():
        return []
    return sorted((path for path in configuration.iterdir()
                   if path.is_dir() and re.fullmatch(r"order_\d+", path.name)),
                  key=order_sort_key)


def list_windows(order: Path) -> list[Path]:
    if not order.is_dir():
        return []
    return sorted((path for path in order.iterdir()
                   if path.is_dir() and re.fullmatch(r"t_.+_.+", path.name)),
                  key=window_sort_key)


def mode_folder(window: Path, mode: str) -> Path:
    return window if mode == "window" else window / mode


def available_modes(window: Path) -> list[str]:
    modes = ["window"]
    modes.extend(mode for mode in ("diagonal", "correlated", "correlated_diagnostic")
                 if (window / mode).is_dir())
    return modes


def classify_plot(filename: str) -> str:
    stem = Path(filename.lower()).stem
    exact = {
        "continuum_a2_over_t_beta_fit4": "infinite-volume beta_GF extrapolation",
        "continuum_a2_over_t_beta_over_g4_fit4": "infinite-volume beta_GF/g_GF^4 extrapolation",
        "continuum_beta_vs_g2_fit4": "final continuum beta_GF",
        "continuum_beta_over_g4_vs_g2_fit4": "final continuum beta_GF/g_GF^4",
        "interpolation_beta_over_g4_fit4_with_data": "interpolation fit with data and coefficients",
        "interpolation_beta_over_g4_fit4": "interpolation fit",
        "interpolation_residuals_fit4": "interpolation residuals",
        "interpolation_quality_fit4": "interpolation quality",
        "continuum_quality_fit4": "continuum quality",
        "diagonal_vs_correlated_fit4": "diagonal vs correlated comparison",
        "figure11_integral_matching_fit4": "Figure 11 integral matching",
        "weak_coupling_to_zero_fit4": "weak-coupling extrapolation",
        "extended_intermediate_to_zero_fit4": "extended intermediate interpolations",
        "extended_interpolation_continuum_to_zero_fit4": "continuum limit of extended interpolations",
    }
    if stem in exact:
        return exact[stem]
    if "correlation" in stem:
        return "correlation diagnostic"
    if "continuum" in stem or "a2_over_t" in stem:
        return "continuum extrapolation"
    if "interpolation" in stem:
        return "interpolation"
    return stem.replace("_", " ")


@st.cache_data(show_spinner=False)
def scan_plot_folder(folder_string: str) -> pd.DataFrame:
    folder = Path(folder_string)
    if not folder.is_dir():
        return pd.DataFrame()
    files = sorted(folder.glob("*.pdf"))
    pdf_stems = {path.stem for path in files}
    files.extend(path for suffix in ("*.png", "*.jpg", "*.jpeg")
                 for path in sorted(folder.glob(suffix)) if path.stem not in pdf_stems)
    return pd.DataFrame({
        "category": [classify_plot(path.name) for path in files],
        "filename": [path.name for path in files],
        "absolute_path": [str(path) for path in files],
    })


def fit4_selection(prefix: str) -> tuple[Path, Path, Path] | None:
    order_selection = fit4_order_selection(prefix)
    if order_selection is None:
        return None
    configuration, order = order_selection
    windows = list_windows(order)
    if not windows:
        st.warning(f"No flow-time ranges were found in `{order}`.")
        return None
    window = st.selectbox("Flow-time range", windows, format_func=pretty_window,
                          key=f"{prefix}_window")
    return configuration, order, window


def fit4_order_selection(prefix: str) -> tuple[Path, Path] | None:
    configurations = [Path(value) for value in discover_configurations()]
    if not configurations:
        st.error("No fit4 result folders were found. Paste `fit4_nopriors` and/or "
                 "`fit4_width*` beside `app.py`, preserving their internal structure.")
        return None
    configuration = st.selectbox("Fit4 prior configuration", configurations,
                                 format_func=configuration_label,
                                 key=f"{prefix}_configuration")
    orders = list_orders(configuration)
    if not orders:
        st.warning(f"No order folders were found in `{configuration}`.")
        return None
    order = st.selectbox("Correction order", orders, format_func=pretty_order,
                         key=f"{prefix}_order")
    return configuration, order


def render_analysis_tab() -> None:
    selection_col, main_col = st.columns([0.95, 3.05], gap="large")
    with selection_col:
        st.markdown("### Analysis selection")
        st.markdown(f'<div class="small-muted">Root:<br><code>{ROOT}</code></div>',
                    unsafe_allow_html=True)
        selected = fit4_selection("analysis")
        if selected is None:
            return
        configuration, order, window = selected
        if st.button("Rescan", width="stretch", key="analysis_rescan"):
            discover_configurations.clear()
            scan_plot_folder.clear()
            rerun_app()

    requested = (
        (window, "interpolation_beta_over_g4_fit4_with_data", "Interpolation fit with data and coefficients"),
        (window, "interpolation_quality_fit4", "Interpolation fit quality"),
        (window / "correlated", "continuum_beta_over_g4_vs_g2_fit4", r"Correlated continuum $\beta_{\rm GF}/g_{\rm GF}^4$ versus $g_{\rm GF}^2$"),
        (window / "correlated", "continuum_a2_over_t_beta_fit4", r"Correlated $\beta_{\rm GF}$ versus $a^2/t$"),
        (window / "correlated", "continuum_a2_over_t_beta_over_g4_fit4", r"Correlated $\beta_{\rm GF}/g_{\rm GF}^4$ versus $a^2/t$"),
        (window / "correlated", "figure11_integral_matching_fit4", "Figure 11 integral matching"),
        (window / "correlated_diagnostic", "extended_intermediate_to_zero_fit4", "Extended intermediate interpolations to zero"),
        (window / "correlated_diagnostic", "extended_interpolation_continuum_to_zero_fit4", "Continuum limit of extended interpolations"),
    )
    rows = []
    for folder, stem, label in requested:
        path = order_level_plot(folder, stem)
        if path:
            rows.append({"category": label, "filename": path.name,
                         "absolute_path": str(path)})
    plots = pd.DataFrame(rows)
    with main_col:
        if plots.empty:
            st.info("No plot files were found in the selected section.")
            st.code(str(selected_folder))
            return
        paths = plots["absolute_path"].tolist()
        state_key = "selected_analysis_plot"
        if st.session_state.get(state_key) not in paths:
            st.session_state[state_key] = paths[0]
        current = paths.index(st.session_state[state_key])
        previous, selector, following = st.columns([1, 6, 1])
        with previous:
            if st.button("← Previous", width="stretch", key="analysis_previous"):
                st.session_state[state_key] = paths[(current - 1) % len(paths)]
                rerun_app()
        with following:
            if st.button("Next →", width="stretch", key="analysis_next"):
                st.session_state[state_key] = paths[(current + 1) % len(paths)]
                rerun_app()
        labels = {row.absolute_path: f"{row.category} — {row.filename}"
                  for row in plots.itertuples()}
        with selector:
            chosen = st.selectbox("Choose a plot", paths, index=current,
                                  format_func=lambda value: labels[value],
                                  label_visibility="collapsed",
                                  key="analysis_plot_selector")
        st.session_state[state_key] = chosen
        row = plots.loc[plots["absolute_path"] == chosen].iloc[0]
        st.markdown(
            f"**Prior:** `{configuration_label(configuration)}` &nbsp; | &nbsp; "
            f"**Order:** `{pretty_order(order)}` &nbsp; | &nbsp; "
            f"**Flow time:** `{pretty_window(window)}` &nbsp; | &nbsp; "
            f"**Plot:** `{row['category']}`"
        )
        display_plot(Path(chosen))


INFINITE_VOLUME_PLOTS = (
    (r"$g_{\rm GF}^{2}$ versus inverse volume", "infinite_volume_all_beta_fit4"),
    (r"$\beta_{\rm GF}$ versus inverse volume", "infinite_volume_beta_vs_invV_fit4"),
    (r"$\beta_{\rm GF}/g_{\rm GF}^{4}$ versus inverse volume",
     "infinite_volume_beta_over_g4_vs_invV_fit4"),
)


def order_level_plot(order: Path, stem: str) -> Path | None:
    # Prefer the vector PDF so users can zoom; use PNG only as a fallback.
    for suffix in (".pdf", ".png"):
        candidate = order / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def render_iv_tab() -> None:
    st.markdown("### Infinite-volume extrapolation plots")
    controls, plots_col = st.columns([0.95, 3.05], gap="large")
    with controls:
        configurations = [Path(value) for value in discover_configurations()]
        configuration = next(
            (path for path in configurations if path.name == "fit4_width1000"), None
        )
        if configuration is None:
            st.error("The `fit4_width1000` result folder was not found.")
            return
        st.markdown("**Fit4 prior configuration**")
        st.markdown("Prior width 1000 — xerrors=True")
        orders = list_orders(configuration)
        if not orders:
            st.warning(f"No order folders were found in `{configuration}`.")
            return
        order = st.selectbox("Correction order", orders, format_func=pretty_order,
                             key="iv_order")
        if st.button("Rescan IV plots", width="stretch", key="iv_rescan"):
            discover_configurations.clear()
            rerun_app()
    with plots_col:
        st.markdown(
            f"**Prior:** `{configuration_label(configuration)}` &nbsp; | &nbsp; "
            f"**Order:** `{pretty_order(order)}`"
        )
        for plot_label, stem in INFINITE_VOLUME_PLOTS:
            st.markdown(f"#### {plot_label}")
            plot_path = order_level_plot(order, stem)
            if plot_path:
                display_plot(plot_path, height=720)
            else:
                st.warning(f"The selected order does not contain `{stem}.png` or `.pdf`.")


def parse_diagnostic(path: Path) -> dict[str, str]:
    text = str(path).lower()
    ensemble = re.search(r"(?<![a-z0-9])([1-9]\d*p\d+)(?![a-z0-9])", text)
    volume = re.search(r"(l\d+l\d+l\d+t\d+)", text)
    flow = re.search(r"(?:flowtime|tau|t)_([0-9]+p[0-9]+)", text)
    return {
        "plot type": "autocorrelation plot" if "autocorr" in text else "binsize plot",
        "ensemble": ensemble.group(1) if ensemble else "unknown",
        "volume": volume.group(1) if volume else "unknown",
        "flow time": flow.group(1) if flow else "unknown",
        "filename": path.name, "absolute_path": str(path),
    }


@st.cache_data(show_spinner=False)
def scan_diagnostics() -> pd.DataFrame:
    rows, seen = [], set()
    for root in SEARCH_ROOTS:
        for path in root.rglob("*"):
            lowered = str(path).lower()
            if (not path.is_file() or path.suffix.lower() not in {".pdf", ".png"}
                    or ("autocorr" not in lowered and "binsize" not in lowered)):
                continue
            if path.suffix.lower() == ".png" and path.with_suffix(".pdf").is_file():
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                rows.append(parse_diagnostic(path))
    return pd.DataFrame(rows)


def filter_values(frame: pd.DataFrame, column: str, label: str, key: str) -> str:
    values = sorted(str(value) for value in frame[column].dropna().unique())
    return st.selectbox(label, ["all", *values], key=key)


def render_diagnostics_tab() -> None:
    st.markdown("### Autocorrelation / binsize diagnostics")
    if st.button("Rescan diagnostics", key="diagnostics_rescan"):
        scan_diagnostics.clear()
        rerun_app()
    frame = scan_diagnostics()
    if frame.empty:
        st.info("No autocorrelation or binsize diagnostic plots were found.")
        return
    filters, viewer = st.columns([1.05, 3], gap="large")
    with filters:
        st.markdown("#### Filters")
        choices = [(column, filter_values(frame, column, label, f"diag_{column}"))
                   for column, label in (("plot type", "Plot type"),
                                         ("ensemble", "Ensemble"),
                                         ("volume", "Volume"),
                                         ("flow time", "Flow time"))]
    filtered = frame.copy()
    for column, value in choices:
        if value != "all":
            filtered = filtered.loc[filtered[column] == value]
    with viewer:
        if filtered.empty:
            st.info("No diagnostic plots match the filters.")
            return
        paths = filtered["absolute_path"].tolist()
        chosen = st.selectbox("Choose a diagnostic plot", paths,
                              format_func=lambda value: Path(value).name,
                              key="diagnostic_plot_selector")
        display_plot(Path(chosen))


@st.cache_data(show_spinner=False)
def find_summary_csvs() -> list[str]:
    found, seen = [], set()
    for root in SEARCH_ROOTS:
        for path in root.rglob("autocorrelation_classification_summary.csv"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(str(resolved))
    return sorted(found)


DISPLAYED_BETAS = {7.0, 7.25, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                   11.0, 12.0, 14.0, 16.0, 18.0, 20.0}
DISPLAYED_SPATIAL_EXTENTS = {20, 24, 32, 40, 48}
ENSEMBLE_FILE_PATTERN = re.compile(
    r"^(?P<beta>[^_]+)_l(?P<L>\d+)l\d+l\d+t(?P<T>\d+)_"
    r"(?P<mass>[^_]+)_wilson\.bin$"
)


def data_inventory_fingerprint() -> tuple[tuple[str, int, int], ...]:
    if not DATA_ROOT.is_dir():
        return ()
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                 for path in sorted(DATA_ROOT.glob("*_wilson.bin")))


@st.cache_data(show_spinner="Scanning updated ensemble data…")
def ensemble_statistics_frame(
        fingerprint: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    del fingerprint  # Invalidates the cache whenever a data file changes.
    rows = []
    for path in sorted(DATA_ROOT.glob("*_wilson.bin")):
        match = ENSEMBLE_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        beta = float(match.group("beta").replace("p", "."))
        spatial_extent = int(match.group("L"))
        if beta not in DISPLAYED_BETAS or spatial_extent not in DISPLAYED_SPATIAL_EXTENTS:
            continue
        with path.open("rb") as input_file:
            raw_data = pickle.load(input_file)
        histories = [slices[0] for observable, slices in raw_data.items()
                     if observable != "flow_times" and slices]
        configurations = len(histories[0]) if histories else 0
        if configurations == 0:
            continue
        if any(len(history) != configurations for history in histories):
            raise ValueError(f"Inconsistent observable histories in {path.name}")
        rows.append({
            "beta_b": beta,
            "volume": f"{spatial_extent}³×{int(match.group('T'))}",
            "mass": float(match.group("mass").replace("p", ".")),
            "flow": "wilson",
            "configurations": configurations,
            "bins": configurations // 15,
        })
    return pd.DataFrame(rows).sort_values(
        ["beta_b", "volume", "mass"], kind="stable", ignore_index=True
    )


def render_summary_tab() -> None:
    st.markdown("### Data summary table")
    st.markdown(
        "Available Wilson-flow ensembles and their configuration counts. The "
        "reported bin count uses **binsize = 15**."
    )
    ensemble_frame = ensemble_statistics_frame(data_inventory_fingerprint())
    display_frame = ensemble_frame.rename(columns={
        "beta_b": "β_b", "volume": "Volume", "mass": "Mass",
        "flow": "Flow", "configurations": "Configurations", "bins": "Bins",
    })
    table_html = display_frame.to_html(
        index=False, classes="ensemble-table", border=0,
        formatters={"β_b": lambda value: f"{value:g}",
                    "Mass": lambda value: f"{value:g}"},
    )
    st.markdown(f'<div class="ensemble-table-wrap">{table_html}</div>',
                unsafe_allow_html=True)
    st.caption(
        f"{len(ensemble_frame)} ensembles and "
        f"{int(ensemble_frame['configurations'].sum()):,} configurations."
    )


tab_analysis, tab_iv, tab_diagnostics, tab_summary = st.tabs((
    "Analysis plots", "Infinite-volume extrapolation",
    "Autocorr / binsize diagnostics", "Data summary table",
))
with tab_analysis:
    render_analysis_tab()
with tab_iv:
    render_iv_tab()
with tab_diagnostics:
    render_diagnostics_tab()
with tab_summary:
    render_summary_tab()
