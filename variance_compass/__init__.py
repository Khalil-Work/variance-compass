"""Variance Compass: Monte Carlo uncertainty propagation and variance-based
(Sobol') sensitivity analysis with reproducible sampling and analytic
ground truth.

Public API
----------
* :func:`ishigami` / :func:`ishigami_analytic` / :func:`ishigami_bounds`
  -- Ishigami test function and its closed-form indices (a = 7, b = 0.1).
* :func:`additive_model` / :func:`additive_analytic` / :func:`additive_bounds`
  -- exactly decomposable anchor model with analytic indices.
* :func:`saltelli_matrices` -- original Saltelli (k + 2)N sampling design.
* :func:`sample_mean` / :func:`sample_variance` -- moment estimators.
* :func:`first_order_indices` / :func:`total_effect_indices` /
  :func:`saltelli_indices` -- Saltelli S_i / T_i estimators.
* :func:`analyze_sensitivity` / :func:`convergence_sweep` /
  :func:`index_errors` -- analysis driver with bootstrap uncertainty.
"""

from .additive import additive_analytic, additive_bounds, additive_model
from .estimators import (
    first_order_indices,
    sample_mean,
    sample_variance,
    saltelli_indices,
    total_effect_indices,
)
from .ishigami import (
    ISHIGAMI_A,
    ISHIGAMI_B,
    IshigamiAnalytic,
    ishigami,
    ishigami_analytic,
    ishigami_bounds,
)
from .sampling import saltelli_matrices, to_physical
from .sensitivity import (
    AnalyticIndices,
    SensitivityResult,
    SweepResult,
    analyze_sensitivity,
    convergence_sweep,
    index_errors,
)

__version__ = "0.1.0"

__all__ = [
    "ISHIGAMI_A",
    "ISHIGAMI_B",
    "AnalyticIndices",
    "IshigamiAnalytic",
    "SensitivityResult",
    "SweepResult",
    "additive_analytic",
    "additive_bounds",
    "additive_model",
    "analyze_sensitivity",
    "convergence_sweep",
    "first_order_indices",
    "index_errors",
    "ishigami",
    "ishigami_analytic",
    "ishigami_bounds",
    "saltelli_indices",
    "saltelli_matrices",
    "sample_mean",
    "sample_variance",
    "to_physical",
    "total_effect_indices",
]
