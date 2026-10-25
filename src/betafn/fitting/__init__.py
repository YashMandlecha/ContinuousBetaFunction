"""betafn.fitting — fit infrastructure and standard interpolation families."""
from .containers import FitInput, FitModel, StageStore
from .families import polynomial_interpolation, perturbative_interpolation
from .goodness import GeneralizedChiSquare, bruno_sommer_goodness_of_fit
from .hybrid import (
    HybridInterpolationCandidate,
    HybridInterpolationResult,
    HybridWindowResult,
    default_hybrid_candidates,
    fit_hybrid_bma_candidates,
    hybrid_match_fraction,
    hybrid_parameter_records,
    run_hybrid_bma_window,
)
from .model_average import (
    ModelAverageResult,
    model_average_predictions,
    posterior_model_weights,
    relative_partial_uncertainty_percent,
    shapley_model_variance,
)
