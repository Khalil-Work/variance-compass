"""Exactly decomposable additive test model (exact-regression anchor).

For ``f(x) = sum_j c_j * x_j`` with independent ``x_j ~ U(-1, 1)``:

* ``Var(x_j) = 1/3`` so the first-order variances are ``V_j = c_j^2 / 3``
  and the total variance is ``V = sum_j c_j^2 / 3``;
* the Sobol' (HDMR) decomposition terminates at first order because the
  model is strictly additive with independent inputs, hence analytically

      S_j = T_j = c_j^2 / sum_m c_m^2 .

This provides an exact regression target with no interaction terms: any
deviation of the Monte Carlo estimates from ``c_j^2 / sum(c^2)`` is pure
estimator error, which is what the verification anchor measures.
"""

from __future__ import annotations

import numpy as np

from .sensitivity import AnalyticIndices
from .validation import validate_positive_int

__all__ = ["additive_model", "additive_analytic", "additive_bounds"]


def additive_bounds(dim) -> np.ndarray:
    """Return the ``(dim, 2)`` bounds array ``[(-1, 1)] * dim``."""
    dim = validate_positive_int(dim, "dim")
    return np.tile(np.array([-1.0, 1.0], dtype=float), (dim, 1))


def _coefficients(coeffs):
    c = np.asarray(coeffs, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("coeffs must be a non-empty 1-D array-like of real numbers")
    if not np.all(np.isfinite(c)):
        raise ValueError("coeffs must contain only finite values")
    return c


def additive_model(coeffs):
    """Build the vectorised additive model ``f(X) = X @ coeffs``."""

    c = _coefficients(coeffs)

    def model(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != c.size:
            raise ValueError(
                f"additive model expects an (n, {c.size}) array, got shape {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("additive model input contains non-finite values")
        return X @ c

    model.coefficients = c
    return model


def additive_analytic(coeffs) -> AnalyticIndices:
    """Analytic Sobol' indices of the additive model (``S == T``)."""
    c = _coefficients(coeffs)
    variances = c * c / 3.0
    total = float(variances.sum())
    if total <= 0.0:
        raise ValueError("additive_analytic requires at least one non-zero coefficient")
    S = variances / total
    S_tuple = tuple(float(s) for s in S)
    return AnalyticIndices(variance=total, S=S_tuple, T=S_tuple)
