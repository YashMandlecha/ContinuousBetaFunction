"""Streamlit browser for fit4 continuum and diagnostic plots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULT_ROOTS = (ROOT, ROOT.parent)


@dataclass(frozen=True)
class PlotSpec:
    label: str
    folder: str
    stem: str
    group: str


PLOT_SPECS = (
    PlotSpec(
        "Infinite-volume extrapolation of beta_GF",
        "correlated",
        "continuum_a2_over_t_beta_fit4",
        "Infinite-volume extrapolation",
    ),
    PlotSpec(
        "Infinite-volume extrapolation of beta_GF / g_GF^4",
        "correlated",
        "continuum_a2_over_t_beta_over_g4_fit4",
        "Infinite-volume extrapolation",
    ),
    PlotSpec(
        "Infinite-volume extrapolation of beta_GF (diagonal covariance)",
        "diagonal",
        "continuum_a2_over_t_beta_fit4",
        "Infinite-volume extrapolation",
    ),
    PlotSpec(
        "Infinite-volume extrapolation of beta_GF / g_GF^4 (diagonal covariance)",
        "diagonal",
        "continuum_a2_over_t_beta_over_g4_fit4",
        "Infinite-volume extrapolation",
    ),
    PlotSpec(
        "Final continuum beta_GF versus g_GF^2",
        "correlated",
        "continuum_beta_vs_g2_fit4",
        "Continuum results",
    ),
    PlotSpec(
        "Final continuum beta_GF / g_GF^4 versus g_GF^2",
        "correlated",
        "continuum_beta_over_g4_vs_g2_fit4",
        "Continuum results",
    ),
    PlotSpec(
        "Diagonal versus correlated continuum result",
        "",
        "diagonal_vs_correlated_fit4",
        "Continuum results",
    ),
    PlotSpec(
        "Interpolation fit with data and coefficients",
        "",
        "interpolation_beta_over_g4_fit4_with_data",
        "Interpolation",
    ),
    PlotSpec(
        "Interpolation fit",
        "",
        "interpolation_beta_over_g4_fit4",
        "Interpolation",
    ),
    PlotSpec(
        "Interpolation quality",
        "",
        "interpolation_quality_fit4",
        "Interpolation",
    ),
    PlotSpec(
        "Extended intermediate interpolations",
        "correlated_diagnostic",
        "extended_intermediate_to_zero_fit4",
        "Weak-coupling diagnostics",
    ),
    PlotSpec(
        "Continuum limit of extended interpolations",
        "correlated_diagnostic",
        "extended_interpolation_continuum_to_zero_fit4",
        "Weak-coupling diagnostics",
    ),
    PlotSpec(
        "Figure 11 integral matching",
        "correlated",
        "figure11_integral_matching_fit4",
        "Weak-coupling diagnostics",
    ),
    PlotSpec(
        "Weak-coupling extrapolation to zero",
        "correlated",
        "weak_coupling_to_zero_fit4",
        "Weak-coupling diagnostics",
    ),
    PlotSpec(
        "Continuum fit quality",
        "correlated",
        "continuum_quality_fit4",
        "Quality diagnostics",
    ),
)


st.set_page_config(page_title="Fit4 plot browser", layout="wide")
st.markdown(
    """
    <style>
    h1, h2, h3 { font-family: "Computer Modern Serif", "CMU Serif",
        "Latin Modern Roman", "Times New Roman", serif; }
    .subtitle { color: #777; margin-top: -0.65rem; margin-bottom: 1.2rem; }
    .path { color: #777; font-size: 0.86rem; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(r"# Fit4 gradient-flow $\beta$-function plots")
st.markdown(
    '<div class="subtitle">Infinite-volume results and fit4 diagnostics, '
    'organized by prior, correction order, and flow-time range.</div>',
    unsafe_allow_html=True,
)


def order_key(path: Path) -> int:
    match = re.fullmatch(r"order_(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def window_key(path: Path) -> tuple[float, float]:
    match = re.fullmatch(r"t_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)", path.name)
    if not match:
        return 10**9, 10**9
    return tuple(float(value.replace("p", ".")) for value in match.groups())


def display_number(text: str) -> str:
    return text.replace("p", ".")


def configuration_label(path: Path) -> str:
    if path.name == "fit4_nopriors":
        return "No coefficient priors (x errors off)"
    match = re.fullmatch(r"fit4_width(.+)", path.name)
    if match:
        return f"Gaussian coefficient prior, width {display_number(match.group(1))}"
    return path.name


def pretty_order(path: Path) -> str:
    return f"Order {path.name.removeprefix('order_')}"


def pretty_window(path: Path) -> str:
    match = re.fullmatch(r"t_(.+)_(.+)", path.name)
    if not match:
        return path.name
    return f"t/a² ∈ [{display_number(match.group(1))}, {display_number(match.group(2))}]"


@st.cache_data(show_spinner=False)
def discover_configurations() -> list[str]:
    found: dict[str, Path] = {}
    for result_root in RESULT_ROOTS:
        for candidate in result_root.glob("fit4_*"):
            if not candidate.is_dir():
                continue
            if candidate.name != "fit4_nopriors" and not candidate.name.startswith("fit4_width"):
                continue
            # ROOT is visited first, so pasted browser-local results take precedence.
            found.setdefault(candidate.name, candidate.resolve())
    return [str(found[name]) for name in sorted(found, key=configuration_sort_key)]


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


def list_directories(parent: Path, pattern: str, key) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(
        (item for item in parent.iterdir() if item.is_dir() and re.fullmatch(pattern, item.name)),
        key=key,
    )


def plot_path(window: Path, spec: PlotSpec) -> Path | None:
    folder = window / spec.folder if spec.folder else window
    png = folder / f"{spec.stem}.png"
    pdf = folder / f"{spec.stem}.pdf"
    if png.is_file():
        return png
    if pdf.is_file():
        return pdf
    return None


def show_plot(path: Path) -> None:
    st.markdown(f'<div class="path">{path}</div>', unsafe_allow_html=True)
    if path.suffix.lower() == ".png":
        st.image(str(path), width="stretch")
    else:
        st.info("Only the PDF is present. Download it to view the full-resolution plot.")
    st.download_button(
        "Download selected plot",
        data=path.read_bytes(),
        file_name=path.name,
        mime="image/png" if path.suffix.lower() == ".png" else "application/pdf",
        key=f"download_{hash(str(path))}",
    )
    companion_suffix = ".pdf" if path.suffix.lower() == ".png" else ".png"
    companion = path.with_suffix(companion_suffix)
    if companion.is_file():
        st.download_button(
            f"Download {companion_suffix[1:].upper()} version",
            data=companion.read_bytes(),
            file_name=companion.name,
            mime="application/pdf" if companion_suffix == ".pdf" else "image/png",
            key=f"download_companion_{hash(str(companion))}",
        )


def selections(key_prefix: str) -> tuple[Path, Path, Path] | None:
    configurations = [Path(value) for value in discover_configurations()]
    if not configurations:
        st.warning(
            "No fit4 result folders were found. Paste `fit4_nopriors` or any "
            "`fit4_width*` folder beside `app.py`, preserving its internal structure."
        )
        return None
    configuration = st.selectbox(
        "Prior configuration",
        configurations,
        format_func=configuration_label,
        key=f"{key_prefix}_configuration",
    )
    orders = list_directories(configuration, r"order_\d+", order_key)
    if not orders:
        st.warning(f"No order folders found in `{configuration}`.")
        return None
    order = st.selectbox(
        "Fit4 correction order", orders, format_func=pretty_order, key=f"{key_prefix}_order"
    )
    windows = list_directories(order, r"t_.+_.+", window_key)
    if not windows:
        st.warning(f"No flow-time ranges found in `{order}`.")
        return None
    window = st.selectbox(
        "Flow-time range", windows, format_func=pretty_window, key=f"{key_prefix}_window"
    )
    return configuration, order, window


def render_infinite_volume() -> None:
    controls, viewer = st.columns([1, 3], gap="large")
    with controls:
        st.markdown("### Selection")
        selected = selections("iv")
        if selected is None:
            return
        _, _, window = selected
        covariance = st.radio("Continuum covariance", ("Correlated", "Diagonal"))
        quantity = st.radio(
            "Quantity", (r"$\beta_{GF}/g_{GF}^4$", r"$\beta_{GF}$")
        )
        if st.button("Rescan folders", key="iv_rescan", width="stretch"):
            discover_configurations.clear()
            st.rerun()
    folder = "correlated" if covariance == "Correlated" else "diagonal"
    stem = (
        "continuum_a2_over_t_beta_over_g4_fit4"
        if quantity == r"$\beta_{GF}/g_{GF}^4$"
        else "continuum_a2_over_t_beta_fit4"
    )
    path = plot_path(window, PlotSpec(quantity, folder, stem, ""))
    with viewer:
        st.markdown("### Infinite-volume continuum extrapolation")
        if path is None:
            st.info("This plot is not present in the selected result folder.")
        else:
            show_plot(path)


def render_all_fit4() -> None:
    controls, viewer = st.columns([1, 3], gap="large")
    with controls:
        st.markdown("### Selection")
        selected = selections("all")
        if selected is None:
            return
        _, _, window = selected
        available = [(spec, plot_path(window, spec)) for spec in PLOT_SPECS]
        available = [(spec, path) for spec, path in available if path is not None]
        if not available:
            st.warning("No recognized fit4 plots are present for this flow-time range.")
            return
        groups = list(dict.fromkeys(spec.group for spec, _ in available))
        group = st.selectbox("Plot group", groups, key="all_group")
        group_plots = [(spec, path) for spec, path in available if spec.group == group]
        selected_index = st.selectbox(
            "Plot",
            range(len(group_plots)),
            format_func=lambda index: group_plots[index][0].label,
            key="all_plot",
        )
        spec, path = group_plots[selected_index]
        if st.button("Rescan folders", key="all_rescan", width="stretch"):
            discover_configurations.clear()
            st.rerun()
    with viewer:
        st.markdown(f"### {spec.label}")
        show_plot(path)


tab_iv, tab_all = st.tabs(("Fit4 infinite-volume plots", "Other fit4 plots"))
with tab_iv:
    render_infinite_volume()
with tab_all:
    render_all_fit4()
