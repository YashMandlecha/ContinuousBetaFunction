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
from .processing.finite_volume import delta_finite_volume
from .processing.tln import delta_tln
# fitting
from .fitting.containers import FitInput, FitModel, StageStore
from .fitting.families import polynomial_interpolation, perturbative_interpolation
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
    "delta_finite_volume", "delta_tln",
    # fitting
    "FitInput", "FitModel", "StageStore",
    "polynomial_interpolation", "perturbative_interpolation",
    # classes
    "SetupBetaFunction",
    "BetaFunction",
]
