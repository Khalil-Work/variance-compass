"""Moment estimators and Sobol' index estimators for the Saltelli design.

Everything here is derived from the mathematics of the variance
decomposition (the full derivation is in ``docs/model.md``); no
third-party sensitivity-analysis library is used anywhere.

Estimator definitions
---------------------
Given the Saltelli design outputs ``f_a = f(A)``, ``f_b = f(B)`` and
``f_ab[i] = f(A_B^(i))`` (each of length ``n``), where the crossed
matrix ``A_B^(i)`` is ``A`` with its ``i``-th column replaced by the
``i``-th column of ``B``, define the symmetric moment estimates

    f0   = (mean(f_a) + mean(f_b)) / 2
    Vhat = (mean(f_a**2) + mean(f_b**2)) / 2 - f0**2

Then, with ``.*`` the row-wise dot product over rows ``j = 1..n``,

    S_hat_i = ( f_b .* f_ab[i] / n - f0**2 ) / Vhat
    T_hat_i = 1 - ( f_a .* f_ab[i] / n - f0**2 ) / Vhat

These follow from the expectation identities

    E[f(B_j) * f(A_B^(i)_j)] = f0**2 + V_i
    E[f(A_j) * f(A_B^(i)_j)] = f0**2 + V - V_i_total

because ``f(B_j)`` and ``f(A_B^(i)_j)`` share exactly the input ``X_i``
(the transplanted ``B`` column), while ``f(A_j)`` and ``f(A_B^(i)_j)``
share every input *except* ``X_i``; all remaining inputs are
independent copies.  ``Vhat`` carries an ``O(1/n)`` bias from the
estimated mean; the bootstrap in :mod:`variance_compass.sensitivity`
captures the resulting estimator spread empirically.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_mean",
    "sample_variance",
    "first_order_indices",
    "total_effect_indices",
    "saltelli_indices",
]


def _as_1d_float(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of model outputs, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def sample_mean(x):
    """Unbiased sample mean, written out explicitly (tested against numpy)."""
    arr = _as_1d_float(x, "x")
    return float(arr.sum() / arr.size)


def sample_variance(x, ddof=1):
    """Sample variance ``sum((x - mean)^2) / (n - ddof)`` (default ``ddof=1``)."""
    arr = _as_1d_float(x, "x")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be an integer")
    ddof = int(ddof)
    if arr.size - ddof <= 0:
        raise ValueError(f"sample size {arr.size} is too small for ddof={ddof}")
    centered = arr - arr.sum() / arr.size
    return float(centered @ centered / (arr.size - ddof))


def _design_outputs(f_a, f_b, f_ab):
    fa = _as_1d_float(f_a, "f_a")
    fb = _as_1d_float(f_b, "f_b")
    fab = np.asarray(f_ab, dtype=float)
    if fab.ndim != 2:
        raise ValueError(
            f"f_ab must be a 2-D array of shape (dim, n), got shape {fab.shape}"
        )
    if fab.shape[1] != fa.size or fab.shape[1] != fb.size:
        raise ValueError("f_a, f_b and f_ab must share the same sample size n")
    if not np.all(np.isfinite(fab)):
        raise ValueError("f_ab contains non-finite values")
    return fa, fb, fab


def _design_moments(fa, fb):
    f0 = 0.5 * (fa.mean() + fb.mean())
    variance = 0.5 * (np.mean(fa * fa) + np.mean(fb * fb)) - f0 * f0
    if variance <= 0.0:
        raise ValueError(
            "estimated output variance is not positive; the model output looks constant"
        )
    return f0, variance


def first_order_indices(f_a, f_b, f_ab):
    """Saltelli first-order estimates ``S_hat`` as an array of length ``dim``."""
    fa, fb, fab = _design_outputs(f_a, f_b, f_ab)
    f0, variance = _design_moments(fa, fb)
    n = fa.size
    return (fab @ fb / n - f0 * f0) / variance


def total_effect_indices(f_a, f_b, f_ab):
    """Saltelli total-effect estimates ``T_hat`` as an array of length ``dim``."""
    fa, fb, fab = _design_outputs(f_a, f_b, f_ab)
    f0, variance = _design_moments(fa, fb)
    n = fa.size
    return 1.0 - (fab @ fa / n - f0 * f0) / variance


def saltelli_indices(f_a, f_b, f_ab):
    """Compute both index families plus the design moments in one pass.

    Returns
    -------
    S : numpy.ndarray, shape ``(dim,)``
        First-order index estimates.
    T : numpy.ndarray, shape ``(dim,)``
        Total-effect index estimates.
    mean : float
        Pooled mean estimate ``f0``.
    variance : float
        Symmetric variance estimate ``Vhat``.
    """
    fa, fb, fab = _design_outputs(f_a, f_b, f_ab)
    f0, variance = _design_moments(fa, fb)
    n = fa.size
    S = (fab @ fb / n - f0 * f0) / variance
    T = 1.0 - (fab @ fa / n - f0 * f0) / variance
    return S, T, float(f0), float(variance)
