"""betafn — continuous beta-function analysis toolkit."""
# subpackages (importable as betafn.base, betafn.fitting, etc.)
from . import base, processing, fitting, stages

# base
from .base.exceptions import BetaFunctionLog, BetaFunctionException, EmptyEnsembleError
from .base.catalog import FlowWindow, ProcessConfig, ProcessHooks, EnsembleKey, EnsembleFile, DatasetCatalog
from .base.specs import InterpolationSpec, AnalysisConfig, AnalysisResult
# physics
from .perturbative import PerturbativeBetaFunction
# processing
from .processing.gamma import gamma_method_covariance, gamma_method_average, integrated_autocorrelation_time
from .processing.bartlett import (
    BartlettDiagnostics,
    automatic_bartlett_bandwidth,
    plugin_bartlett_bandwidth,
    bartlett_spectral_covariance,
    bartlett_spectral_average,
    lugsail_bartlett_spectral_covariance,
    lugsail_bartlett_spectral_average,
    original_bartlett_spectral_covariance,
    original_bartlett_spectral_average,
    overlapping_batch_means_covariance,
)
from .processing.finite_volume import delta_finite_volume
from .processing.tln import delta_tln
# fitting
from .fitting.containers import FitInput, FitModel, StageStore
from .fitting.families import polynomial_interpolation, perturbative_interpolation
from .fitting.goodness import GeneralizedChiSquare, bruno_sommer_goodness_of_fit
from .fitting.hybrid import (
    HybridInterpolationCandidate,
    HybridInterpolationResult,
    HybridWindowResult,
    default_hybrid_candidates,
    fit_hybrid_bma_candidates,
    hybrid_match_fraction,
    hybrid_parameter_records,
    run_hybrid_bma_window,
)
from .fitting.model_average import (
    ModelAverageResult,
    model_average_predictions,
    posterior_model_weights,
    relative_partial_uncertainty_percent,
    shapley_model_variance,
)
# classes
from .setup import SetupBetaFunction
from .betafn import BetaFunction

__all__ = [
    # base
    "BetaFunctionLog", "BetaFunctionException", "EmptyEnsembleError",
    "FlowWindow", "ProcessConfig", "ProcessHooks",
    "EnsembleKey", "EnsembleFile", "DatasetCatalog",
    "InterpolationSpec", "AnalysisConfig", "AnalysisResult",
    # physics
    "PerturbativeBetaFunction",
    # processing
    "gamma_method_covariance", "gamma_method_average", "integrated_autocorrelation_time",
    "BartlettDiagnostics", "automatic_bartlett_bandwidth", "plugin_bartlett_bandwidth",
    "bartlett_spectral_covariance", "bartlett_spectral_average",
    "lugsail_bartlett_spectral_covariance", "lugsail_bartlett_spectral_average",
    "original_bartlett_spectral_covariance", "original_bartlett_spectral_average",
    "overlapping_batch_means_covariance",
    "delta_finite_volume", "delta_tln",
    # fitting
    "FitInput", "FitModel", "StageStore",
    "polynomial_interpolation", "perturbative_interpolation",
    "GeneralizedChiSquare", "bruno_sommer_goodness_of_fit",
    "HybridInterpolationCandidate", "HybridInterpolationResult", "HybridWindowResult",
    "default_hybrid_candidates", "fit_hybrid_bma_candidates", "hybrid_match_fraction",
    "hybrid_parameter_records",
    "run_hybrid_bma_window",
    "ModelAverageResult", "model_average_predictions", "posterior_model_weights",
    "relative_partial_uncertainty_percent", "shapley_model_variance",
    # classes
    "SetupBetaFunction",
    "BetaFunction",
]
