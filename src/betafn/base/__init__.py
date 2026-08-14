"""betafn.base — foundational types shared across the whole package."""
from .exceptions import BetaFunctionLog, BetaFunctionException, EmptyEnsembleError
from .catalog import (
    FlowWindow, ProcessConfig, ProcessHooks,
    EnsembleKey, EnsembleFile, DatasetCatalog,
)
from .specs import InterpolationSpec, AnalysisConfig, AnalysisResult
