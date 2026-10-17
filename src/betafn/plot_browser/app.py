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
ANALYSIS_ROOT = Path("/Volumes/T7/ContinuousBetaFunction/hpcc_outputs")
ANALYSIS_FIT_NAMES = tuple(f"fit{number}" for number in range(1, 13))
ANALYSIS_ORDER_DIRECTORIES = {
    "fit1": {
        "order_3_nopriors": 3,
        "order_4_nopriors": 4,
        "order_5_nopriors": 5,
    },
    "fit2": {
        "order_3_nopriors": 3,
        "order_4_nopriors": 4,
        "order_5_nopriors": 5,
    },
    "fit3": {
        "order_3_nopriors": 3,
        "order_4_nopriors": 4,
        "order_5_nopriors": 5,
    },
    "fit5": {
        "pt_fixed_additive_u3_nopriors": 1,
        "pt_fixed_additive_u3_u4_nopriors": 2,
    },
    "fit6": {
        "two_loop_free_additive_u3_nopriors": 1,
        "two_loop_free_additive_u3_u4_nopriors": 2,
        "two_loop_free_additive_u3_u4_u5_nopriors": 3,
    },
    "fit7": {
        "three_loop_free_additive_u3_nopriors": 1,
        "three_loop_free_additive_u3_u4_nopriors": 2,
    },
    "fit8": {
        "order_3_c1_fixed_0": 3,
        "order_4_c1_fixed_0": 4,
    },
    "fit9": {
        "order_3_c2_fixed_0": 3,
        "order_4_c2_fixed_0": 4,
        "order_5_c2_fixed_0": 5,
    },
    "fit10": {
        "order_3_c1_c2_fixed_0_free_c0": 3,
        "order_4_c1_c2_fixed_0_free_c0": 4,
        "order_5_c1_c2_fixed_0_free_c0": 5,
    },
    "fit11": {
        "order_3_additive_beta_constant_nopriors": 3,
        "order_4_additive_beta_constant_nopriors": 4,
        "order_5_additive_beta_constant_nopriors": 5,
    },
    "fit12": {
        "order_3_rescaled_a0_standard_continuum_nopriors": 3,
        "order_4_rescaled_a0_standard_continuum_nopriors": 4,
        "order_5_rescaled_a0_standard_continuum_nopriors": 5,
    },
}

st.set_page_config(page_title="Gradient-flow beta-function plot browser", layout="wide")
st.markdown(
    """
    <style>
    h1, h2, h3 { font-family: "Computer Modern Serif", "CMU Serif",
        "Latin Modern Roman", "KaTeX_Main", "Times New Roman", serif; }
    .subtitle { font-size: 1.05rem; color: #888; margin-top: -0.5rem;
                margin-bottom: 1.2rem; }
    .small-muted { color: #666; font-size: 0.92rem; }
    code, [data-testid="stMarkdownContainer"] code {
        background: transparent !important;
        color: inherit !important;
        border: 0 !important;
        border-radius: 0 !important;
        padding: 0 !important;
        font-family: inherit !important;
        font-size: inherit !important;
    }
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
    r"Define $x=g_{\rm GF}^2$ and $u=x/(4\pi)$. Plots of "
    r"$\beta_{\rm GF}/g_{\rm GF}^4$ show $\beta_{\rm GF}(x)/x^2$. "
    "The models below differ only in the "
    "interpolation ansatz; data processing, infinite-volume extrapolation, "
    "continuum analysis, and diagnostic procedures are otherwise shared."
)

with st.expander("Interpolation models and priors", expanded=True):
    (fit1_tab, fit2_tab, fit3_tab, fit4_tab, fit5_tab, fit6_tab, fit7_tab, fit8_tab, fit9_tab,
     fit10_tab, fit11_tab, fit12_tab) = st.tabs(
        (
            "fit1", "fit2", "fit3", "fit4", "fit5", "fit6", "fit7",
            "fit8", "fit9", "fit10", "fit11", "fit12",
        )
    )
    with fit1_tab:
        st.markdown("**Unconstrained polynomial in the gradient-flow coupling**")
        st.latex(
            r"\beta(x)=\sum_{n=0}^{N}p_nx^n,"
            r"\qquad N\in\{3,4,5\}"
        )
        st.markdown("All polynomial coefficients are free and have no priors.")
    with fit2_tab:
        st.markdown("**Free constant with running terms beginning at $x^2$**")
        st.latex(
            r"\beta(x)=p_0+x^2\sum_{n=1}^{N}p_nx^{n-1},"
            r"\qquad N\in\{3,4,5\}"
        )
        st.markdown("All fitted coefficients are free and have no priors.")
    with fit3_tab:
        st.markdown("**Origin-constrained polynomial with running terms beginning at $x^2$**")
        st.latex(
            r"\beta(x)=x^2\sum_{n=1}^{N}p_nx^{n-1},"
            r"\qquad p_0=0,\quad N\in\{3,4,5\}"
        )
        st.markdown(r"The coefficients $p_1,\ldots,p_N$ are free and have no priors.")
    with fit4_tab:
        st.markdown("**Multiplicative correction to fixed three-loop perturbation theory**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_nu^n\right],"
            r"\qquad N\in\{3,4,5,6\}"
        )
    with fit5_tab:
        st.markdown("**All three known perturbative coefficients fixed**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)+x^2"
            r"\sum_{k=3}^{2+N}d_ku^k,\qquad N\in\{1,2\}"
        )
    with fit6_tab:
        st.markdown("**One- and three-loop coefficients fixed; two-loop coefficient free**")
        st.latex(
            r"\beta(x)=x^2\left[A_0+b_1u+A_2u^2+"
            r"\sum_{k=3}^{2+N}d_ku^k\right],\qquad N\in\{1,2,3\}"
        )
    with fit7_tab:
        st.markdown("**One- and two-loop coefficients fixed; three-loop coefficient free**")
        st.latex(
            r"\beta(x)=x^2\left[A_0+A_1u+b_2u^2+"
            r"\sum_{k=3}^{2+N}d_ku^k\right],\qquad N\in\{1,2\}"
        )
    with fit8_tab:
        st.markdown("**Multiplicative fit with $c_1$ fixed to zero**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=2}^{N}c_nu^n\right],\qquad c_1=0,\quad N\in\{3,4\}"
        )
    with fit9_tab:
        st.markdown("**Multiplicative fit with $c_2$ fixed to zero**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+c_1u+\sum_{n=3}^{N}c_nu^n\right],"
            r"\qquad c_2=0,\quad N\in\{3,4,5\}"
        )
    with fit10_tab:
        st.markdown("**Free multiplicative intercept with $c_1=c_2=0$**")
        st.latex(
            r"\beta(x)=\beta_{\rm PT}^{(3)}(x)"
            r"\left[c_0+\sum_{n=3}^{N}c_nu^n\right],"
            r"\qquad c_1=c_2=0,\quad N\in\{3,4,5\}"
        )
    with fit11_tab:
        st.markdown("**Free additive constant in beta plus the Fit 4 correction**")
        st.latex(
            r"\beta(x)=a_0+\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_nu^n\right],"
            r"\qquad N\in\{3,4,5\}"
        )
        st.markdown(
            "$a_0$ is a free additive constant at each operator and flow time; "
            "$a_0$ and the $c_n$ have no priors."
        )
    with fit12_tab:
        st.markdown(
            "**Fit 11 constant reparameterized at each fixed flow time**"
        )
        st.latex(
            r"\beta_j(x)=z_jA_{0,j}+\beta_{\rm PT}^{(3)}(x)"
            r"\left[1+\sum_{n=1}^{N}c_{n,j}u^n\right],"
            r"\qquad z_j=(a^2/t)_j,\quad N\in\{3,4,5\}"
        )
        st.markdown(
            "$z_j=(a^2/t)_j$ is known and $a_{0,j}=z_jA_{0,j}$; "
            "$A_{0,j}$ and the $c_{n,j}$ have no priors."
        )

    st.markdown("**Fixed $N_f=4$, SU(3), gradient-flow perturbative reference**")
    st.latex(
        r"\beta_{\rm PT}^{(3)}(x)=x^2\left["
        r"-0.0527714498137-0.00205854331669x"
        r"+0.000919267536173x^2\right]"
    )
    st.markdown(
        "| Variant | Fitted parameters |\n"
        "|---|---|\n"
        "| fit1 order 3 | $p_0,p_1,p_2,p_3$ |\n"
        "| fit1 order 4 | $p_0,p_1,p_2,p_3,p_4$ |\n"
        "| fit1 order 5 | $p_0,p_1,p_2,p_3,p_4,p_5$ |\n"
        "| fit2 order 3 | $p_0,p_1,p_2,p_3$ |\n"
        "| fit2 order 4 | $p_0,p_1,p_2,p_3,p_4$ |\n"
        "| fit2 order 5 | $p_0,p_1,p_2,p_3,p_4,p_5$ |\n"
        "| fit3 order 3 | $p_1,p_2,p_3$ |\n"
        "| fit3 order 4 | $p_1,p_2,p_3,p_4$ |\n"
        "| fit3 order 5 | $p_1,p_2,p_3,p_4,p_5$ |\n"
        "| fit5 order 1 | $d_3$ |\n"
        "| fit5 order 2 | $d_3,d_4$ |\n"
        "| fit6 order 1 | $b_1,d_3$ |\n"
        "| fit6 order 2 | $b_1,d_3,d_4$ |\n"
        "| fit6 order 3 | $b_1,d_3,d_4,d_5$ |\n"
        "| fit7 order 1 | $b_2,d_3$ |\n"
        "| fit7 order 2 | $b_2,d_3,d_4$ |\n"
        "| fit8 order 3 | $c_2,c_3$ |\n"
        "| fit8 order 4 | $c_2,c_3,c_4$ |\n"
        "| fit9 order 3 | $c_1,c_3$ |\n"
        "| fit9 order 4 | $c_1,c_3,c_4$ |\n"
        "| fit9 order 5 | $c_1,c_3,c_4,c_5$ |\n"
        "| fit10 order 3 | $c_0,c_3$ |\n"
        "| fit10 order 4 | $c_0,c_3,c_4$ |\n"
        "| fit10 order 5 | $c_0,c_3,c_4,c_5$ |\n"
        "| fit11 order 3 | $a_0,c_1,c_2,c_3$ |\n"
        "| fit11 order 4 | $a_0,c_1,c_2,c_3,c_4$ |\n"
        "| fit11 order 5 | $a_0,c_1,c_2,c_3,c_4,c_5$ |\n"
        "| fit12 order 3 | $A_{0,j},c_1,c_2,c_3$ |\n"
        "| fit12 order 4 | $A_{0,j},c_1,c_2,c_3,c_4$ |\n"
        "| fit12 order 5 | $A_{0,j},c_1,c_2,c_3,c_4,c_5$ |"
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
    bitmap = path.with_suffix(".png")
    if len(pdf_bytes) <= 4 * 1024 * 1024:
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" '
            f'height="{height}" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
    elif bitmap.is_file():
        # Large PDFs can exceed browser data-URL limits. Display the matching
        # bitmap reliably and retain the vector PDF download below.
        st.image(str(bitmap), width="stretch")
    else:
        st.info("Use the download button below to open this PDF.")
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
        return "No coefficient priors"
    match = re.fullmatch(r"fit4_width(.+)", path.name)
    if match:
        return f"Prior width {match.group(1).replace('p', '.')} — xerrors=True"
    return path.name


def analysis_order_number(path: Path) -> int:
    for directory_orders in ANALYSIS_ORDER_DIRECTORIES.values():
        if path.name in directory_orders:
            return directory_orders[path.name]
    match = re.match(r"order_(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else 10**9


def order_sort_key(path: Path) -> int:
    return analysis_order_number(path)


def window_sort_key(path: Path) -> tuple[float, float]:
    match = re.fullmatch(r"t_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)", path.name)
    if not match:
        return float("inf"), float("inf")
    return tuple(float(value.replace("p", ".")) for value in match.groups())


def pretty_order(path: Path) -> str:
    order = analysis_order_number(path)
    return f"Order {order}" if order < 10**9 else path.name


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


def fit_family_label(path: Path) -> str:
    match = re.fullmatch(r"fit(\d+)", path.name)
    return f"Fit {match.group(1)}" if match else path.name


def thinning_sort_key(path: Path) -> float:
    match = re.fullmatch(r"dt_(\d+(?:p\d+)?)", path.name)
    if match is None:
        return float("inf")
    return float(match.group(1).replace("p", "."))


def sampling_label(path: Path, order: Path) -> str:
    if path == order:
        return "Unthinned (standard continuum analysis)"
    spacing = thinning_sort_key(path)
    if spacing != float("inf"):
        return f"Thinned: Δ(t/a²) = {spacing:g}"
    return path.name


@st.cache_data(show_spinner=False)
def discover_analysis_fits() -> list[str]:
    """Find completed analysis families from Fit 1 through Fit 12 on T7."""
    if not ANALYSIS_ROOT.is_dir():
        return []
    fits = []
    for name in ANALYSIS_FIT_NAMES:
        path = ANALYSIS_ROOT / name
        if path.is_dir():
            fits.append(str(path))
        elif name == "fit4" and any(
                candidate.is_dir()
                for candidate in ANALYSIS_ROOT.glob("fit4_*")
        ):
            # Fit 4 configurations live in fit4_nopriors / fit4_width*, so
            # keep a virtual Fit 4 family even when no literal fit4 exists.
            fits.append(str(path))
    return fits


@st.cache_data(show_spinner=False)
def discover_fit4_analysis_configurations() -> list[str]:
    if not ANALYSIS_ROOT.is_dir():
        return []
    configurations = [
        path for path in ANALYSIS_ROOT.glob("fit4_*")
        if path.is_dir()
        and (path.name == "fit4_nopriors" or path.name.startswith("fit4_width"))
    ]
    return [str(path) for path in sorted(
        configurations, key=lambda path: configuration_sort_key(path.name)
    )]


def list_analysis_orders(result_root: Path, fit_name: str) -> list[Path]:
    if not result_root.is_dir():
        return []
    directory_orders = ANALYSIS_ORDER_DIRECTORIES.get(fit_name)
    if directory_orders is None:
        candidates = [
            path for path in result_root.iterdir()
            if path.is_dir() and re.fullmatch(r"order_\d+", path.name)
        ]
    else:
        candidates = [result_root / name for name in directory_orders
                      if (result_root / name).is_dir()]
    completed = [path for path in candidates
                 if (path / "RUN_COMPLETE.json").is_file()]
    return sorted(completed, key=order_sort_key)


def list_sampling_directories(order: Path) -> list[Path]:
    thinning_root = order / "continuum_thinning_scan"
    thinned = []
    if thinning_root.is_dir():
        thinned = sorted(
            (path for path in thinning_root.iterdir()
             if path.is_dir() and re.fullmatch(r"dt_.+", path.name)),
            key=thinning_sort_key,
        )
    return [order, *thinned]


def result_plot(folder: Path, stem: str, fit_tag: str) -> Path | None:
    """Return a vector result when available, with a bitmap fallback."""
    tagged_stems = [f"{stem}_{fit_tag}"]
    if stem.endswith("_with_data"):
        tagged_stems.insert(
            0, f"{stem.removesuffix('_with_data')}_{fit_tag}_with_data"
        )
    tagged_stems.append(stem)
    for suffix in (".pdf", ".png"):
        for candidate_stem in tagged_stems:
            candidate = folder / f"{candidate_stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


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
        "figure11_integral_matching_fit4": (
            "Integral matching across operators with fitted b_p"
        ),
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
        if st.button("Rescan results", width="stretch", key="analysis_rescan"):
            discover_analysis_fits.clear()
            discover_fit4_analysis_configurations.clear()
            rerun_app()
        fits = [Path(value) for value in discover_analysis_fits()]
        if not fits:
            st.error(
                "No Fit 1–12 analysis folders were found. "
                "Connect the T7 drive and rescan."
            )
            return
        fit_directory = st.selectbox(
            "Fit family", fits, format_func=fit_family_label,
            key="analysis_fit_family",
        )
        result_root = fit_directory
        fit4_configuration = None
        if fit_directory.name == "fit4":
            configurations = [
                Path(value) for value in discover_fit4_analysis_configurations()
            ]
            if not configurations:
                st.warning("No completed Fit 4 configurations were found.")
                return
            fit4_configuration = st.selectbox(
                "Fit 4 prior configuration", configurations,
                format_func=configuration_label,
                key="analysis_fit4_configuration",
            )
            result_root = fit4_configuration
        orders = list_analysis_orders(result_root, fit_directory.name)
        if not orders:
            st.warning(f"No completed order folders were found in `{result_root}`.")
            return
        order = st.selectbox(
            "Interpolation order", orders, format_func=pretty_order,
            key="analysis_order",
        )
        sampling_directories = list_sampling_directories(order)
        sampling = st.selectbox(
            "Continuum flow-time sampling", sampling_directories,
            format_func=lambda path: sampling_label(path, order),
            key="analysis_sampling",
        )
        windows = list_windows(sampling)
        if not windows:
            st.warning(f"No flow-time ranges were found in `{sampling}`.")
            return
        window = st.selectbox(
            "Flow-time range", windows, format_func=pretty_window,
            key="analysis_window",
        )
        covariance_modes = [mode for mode in ("diagonal", "correlated")
                            if (window / mode).is_dir()]
        if not covariance_modes:
            st.warning(f"No continuum covariance folders were found in `{window}`.")
            return
        default_mode = (covariance_modes.index("correlated")
                        if "correlated" in covariance_modes else 0)
        covariance_mode = st.selectbox(
            "Continuum covariance", covariance_modes, index=default_mode,
            format_func=lambda value: value.capitalize(),
            key="analysis_covariance_mode",
        )
    fit_tag = fit_directory.name
    is_thinned = sampling != order
    covariance_label = covariance_mode.capitalize()
    requested: list[tuple[Path, str, str]] = []
    if not is_thinned:
        requested.extend((
            (window, "interpolation_beta_over_g4_with_data",
             "Interpolation fit with data and coefficients"),
            (window, "interpolation_quality", "Interpolation fit quality"),
        ))
    requested.extend((
        (order, "processed_massless_limit_largest_volume",
         "Massless-limit largest-volume data"),
        (order, "massless_infinite_volume_beta_and_ratio_vs_g2",
         "Massless infinite-volume beta and beta/g⁴ versus g²"),
    ))
    requested.extend((
        (window / covariance_mode, "continuum_beta_over_g4_vs_g2",
         rf"{covariance_label} continuum $\beta_{{\rm GF}}/g_{{\rm GF}}^4$ versus $g_{{\rm GF}}^2$"),
        (window / covariance_mode, "continuum_a2_over_t_beta",
         rf"{covariance_label} $\beta_{{\rm GF}}$ versus $a^2/t$"),
        (window / covariance_mode, "continuum_a2_over_t_beta_over_g4",
         rf"{covariance_label} $\beta_{{\rm GF}}/g_{{\rm GF}}^4$ versus $a^2/t$"),
        (window / covariance_mode, "continuum_beta_vs_g2",
         rf"{covariance_label} continuum $\beta_{{\rm GF}}$ versus $g_{{\rm GF}}^2$"),
        (window / covariance_mode, "continuum_quality",
         f"{covariance_label} continuum fit quality"),
        (window, "diagonal_vs_correlated",
         "Diagonal versus correlated continuum comparison"),
    ))
    if not is_thinned:
        singular_origin = fit_tag in {"fit1", "fit2", "fit11", "fit12"}
        extended_limit = (
            "toward weak coupling (g² ≥ 0.1; singular origin excluded)"
            if singular_origin else "to zero coupling"
        )
        requested.extend((
            (window / "correlated", "figure11_integral_matching",
             "Integral matching across operators with fitted b_p (correlated)"),
            (order, "lambda_parameter_by_window",
             "Interim blinded Lambda parameter across flow-time windows"),
        ))
        if fit_tag not in {"fit1", "fit2", "fit3"}:
            requested.extend((
                (window / "correlated_diagnostic",
                 "extended_intermediate_to_zero",
                 f"Extended intermediate interpolations {extended_limit}"),
                (window / "correlated_diagnostic",
                 "extended_interpolation_continuum_to_zero",
                 f"Continuum limit of interpolations extended {extended_limit}"),
            ))
    rows = []
    for folder, stem, label in requested:
        path = result_plot(folder, stem, fit_tag)
        if path:
            rows.append({"category": label, "filename": path.name,
                         "absolute_path": str(path)})
    plots = pd.DataFrame(rows)
    with main_col:
        if plots.empty:
            st.info("No plot files were found in the selected section.")
            st.code(str(window))
            return
        paths = plots["absolute_path"].tolist()
        selection_context = "__".join((
            fit_tag, order.name, sampling.name, window.name, covariance_mode,
        ))
        state_key = make_widget_key(
            "analysis_plot_selector", selection_context
        )
        stored_path = st.session_state.get(state_key)
        if stored_path not in paths:
            if state_key in st.session_state:
                del st.session_state[state_key]
            stored_path = None
        current = paths.index(stored_path) if stored_path is not None else 0
        previous, selector, following = st.columns([1, 6, 1])
        with previous:
            if st.button("← Previous", width="stretch", key="analysis_previous"):
                st.session_state[state_key] = paths[(current - 1) % len(paths)]
                rerun_app()
        with following:
            if st.button("Next →", width="stretch", key="analysis_next"):
                st.session_state[state_key] = paths[(current + 1) % len(paths)]
                rerun_app()
        labels = {row.absolute_path: row.category for row in plots.itertuples()}
        with selector:
            selector_index = (None if state_key in st.session_state else current)
            chosen = st.selectbox(
                "Choose a plot", paths, index=selector_index,
                format_func=lambda value: labels[value],
                label_visibility="collapsed", key=state_key,
            )
        if chosen not in labels:
            chosen = paths[0]
        matching_rows = plots.loc[plots["absolute_path"] == chosen]
        if matching_rows.empty:
            st.warning("The selected plot changed while results were rescanned.")
            return
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
        configuration = ANALYSIS_ROOT / "fit4_nopriors"
        if not configuration.is_dir():
            st.error("The Fit 4 no-prior result folder was not found on T7.")
            return
        st.markdown("**Fit4 prior configuration**")
        st.markdown("No coefficient priors")
        orders = list_analysis_orders(configuration, "fit4")
        if not orders:
            st.warning("No completed Fit 4 no-prior order folders were found.")
            return
        order = st.selectbox("Correction order", orders, format_func=pretty_order,
                             key="iv_order")
        if st.button("Rescan IV plots", width="stretch", key="iv_rescan"):
            rerun_app()
    with plots_col:
        st.markdown(
            f"**Configuration:** No coefficient priors &nbsp; | &nbsp; "
            f"**Order:** `{pretty_order(order)}`"
        )
        for plot_label, stem in INFINITE_VOLUME_PLOTS:
            st.markdown(f"#### {plot_label}")
            plot_path = order_level_plot(order, stem)
            if plot_path:
                display_plot(plot_path, height=720)
            else:
                st.warning(f"The selected order does not contain `{stem}.png` or `.pdf`.")


CHIRAL_BETAS = (
    ("7p00", "7"),
    ("7p25", "7.25"),
    ("7p50", "7.5"),
)
CHIRAL_OBSERVABLES = (
    ("g2", r"$g_{\rm GF}^{2}$ chiral extrapolation"),
    ("beta", r"$\beta_{\rm GF}$ chiral extrapolation"),
)


def render_chiral_tab() -> None:
    st.markdown("### Chiral extrapolation plots")
    controls, plots_col = st.columns([0.95, 3.05], gap="large")
    with controls:
        if st.button("Rescan results", width="stretch", key="chiral_rescan"):
            discover_analysis_fits.clear()
            discover_fit4_analysis_configurations.clear()
            rerun_app()
        fits = [Path(value) for value in discover_analysis_fits()]
        if not fits:
            st.error(
                "No Fit 1–12 analysis folders were found. "
                "Connect the T7 drive and rescan."
            )
            return
        fit_directory = st.selectbox(
            "Fit family", fits, format_func=fit_family_label,
            key="chiral_fit_family",
        )
        result_root = fit_directory
        if fit_directory.name == "fit4":
            configurations = [
                Path(value) for value in discover_fit4_analysis_configurations()
            ]
            if not configurations:
                st.warning("No completed Fit 4 configurations were found.")
                return
            default_configuration = next(
                (index for index, path in enumerate(configurations)
                 if path.name == "fit4_nopriors"),
                0,
            )
            result_root = st.selectbox(
                "Fit 4 prior configuration", configurations,
                index=default_configuration, format_func=configuration_label,
                key="chiral_fit4_configuration",
            )
        orders = list_analysis_orders(result_root, fit_directory.name)
        if not orders:
            st.warning(f"No completed order folders were found in `{result_root}`.")
            return
        order = st.selectbox(
            "Interpolation order", orders, format_func=pretty_order,
            key="chiral_order",
        )

    fit_tag = fit_directory.name
    with plots_col:
        found = 0
        for beta_tag, beta_label in CHIRAL_BETAS:
            st.markdown(rf"#### $\beta_b={beta_label}$")
            for observable, plot_label in CHIRAL_OBSERVABLES:
                stem = f"chiral_extrapolation_{observable}_{beta_tag}"
                plot_path = result_plot(order, stem, fit_tag)
                if plot_path is None:
                    st.warning(
                        f"No {plot_label} plot was found for "
                        rf"$\beta_b={beta_label}$."
                    )
                    continue
                found += 1
                st.markdown(f"**{plot_label}**")
                display_plot(plot_path, height=720)
        if found == 0:
            st.info(
                "This completed order predates the chiral-analysis outputs. "
                "Rescan after its updated results are copied to T7."
            )


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


DISPLAYED_BETAS = {7.0, 7.125, 7.25, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
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
        })
    columns = ("beta_b", "volume", "mass", "flow", "configurations")
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["beta_b", "volume", "mass"], kind="stable", ignore_index=True
    )


def render_summary_tab() -> None:
    st.markdown("### Data summary table")
    st.markdown("Available Wilson-flow ensembles and their configuration counts.")
    if st.button("Rescan ensemble data", key="summary_rescan"):
        ensemble_statistics_frame.clear()
        rerun_app()
    ensemble_frame = ensemble_statistics_frame(data_inventory_fingerprint())
    if ensemble_frame.empty:
        st.info("No nonempty Wilson-flow ensemble files were found.")
        return
    display_frame = ensemble_frame.rename(columns={
        "beta_b": "β_b", "volume": "Volume", "mass": "Mass",
        "flow": "Flow", "configurations": "Configurations",
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


tab_analysis, tab_iv, tab_chiral, tab_diagnostics, tab_summary = st.tabs((
    "Analysis plots", "Infinite-volume extrapolation", "Chiral extrapolation",
    "Autocorr / binsize diagnostics", "Data summary table",
))
with tab_analysis:
    render_analysis_tab()
with tab_iv:
    render_iv_tab()
with tab_chiral:
    render_chiral_tab()
with tab_diagnostics:
    render_diagnostics_tab()
with tab_summary:
    render_summary_tab()
