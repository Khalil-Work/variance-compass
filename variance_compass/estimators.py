"""Moment estimators and Sobol' index estimators for the Saltelli design.

Everything here is derived from the mathematics of the variance
decomposition (the full derivation is in ``docs/model.md``); no
third-party sensitivity-analysis library is used anywhere.

Estimator definitions
---------------------
For the Saltelli crossed design, center EVERY output family at the same
pooled A/B mean m. Let a=f(A)-m, b=f(B)-m, c_i=f(A_B^(i))-m.

    Vhat = (mean(a*a) + mean(b*b))/2
    S_i  = mean(b*c_i)/Vhat
    T_i  = 1 - mean(a*c_i)/Vhat

At the population mean these covariance identities equal V_i/V and
1-(V-V_total_i)/V. Using an estimated common mean gives a consistent,
finite-sample biased estimator; estimates are not clipped to [0,1].
A shifted-origin calculation avoids subtracting two large raw moments.
Every bootstrap replicate calls this same estimator with its own pooled
mean. See docs/model.md for the derivation and translation invariance.
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
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(arr[0] + np.mean(arr-arr[0]))
    if not np.isfinite(mean):
        raise FloatingPointError("sample mean is not representable; rescale outputs")
    return mean


def sample_variance(x, ddof=1):
    """Sample variance ``sum((x - mean)^2) / (n - ddof)`` (default ``ddof=1``)."""
    arr = _as_1d_float(x, "x")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be an integer")
    ddof = int(ddof)
    if arr.size - ddof <= 0:
        raise ValueError(f"sample size {arr.size} is too small for ddof={ddof}")
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = arr-arr[0]
        centered = shifted-shifted.mean()
        variance = float(centered @ centered / (arr.size-ddof))
    if not np.isfinite(variance):
        raise FloatingPointError("sample variance is not representable; rescale outputs")
    return variance


def _design_outputs(f_a, f_b, f_ab):
    fa = _as_1d_float(f_a, "f_a")
    fb = _as_1d_float(f_b, "f_b")
    fab = np.asarray(f_ab, dtype=float)
    if fab.ndim != 2:
        raise ValueError(
            f"f_ab must be a 2-D array of shape (dim, n), got shape {fab.shape}"
        )
    if fab.shape[0] == 0:
        raise ValueError("f_ab must include at least one input dimension")
    if fab.shape[1] != fa.size or fab.shape[1] != fb.size:
        raise ValueError("f_a, f_b and f_ab must share the same sample size n")
    if not np.all(np.isfinite(fab)):
        raise ValueError("f_ab contains non-finite values")
    return fa, fb, fab


class _ZeroVarianceError(ValueError):
    """A constant/under-resolved replicate, distinguishable from malformed input."""


def _centered_design(fa, fb, fab):
    # Subtract a reference before estimating the pooled mean. This keeps
    # small variations resolved when a large common offset is present.
    with np.errstate(over="ignore", invalid="ignore"):
        origin = fa[0]
        da, db = fa-origin, fb-origin
        shift = 0.5*(da.mean()+db.mean())
        a, b, c = da-shift, db-shift, (fab-origin)-shift
        variance = 0.5*(np.mean(a*a)+np.mean(b*b))
        mean = float(origin+shift)
    if not np.isfinite(variance) or not np.isfinite(mean) or not np.all(np.isfinite(c)):
        raise FloatingPointError("design moments are not representable; rescale outputs")
    if variance <= 0:
        raise _ZeroVarianceError("estimated output variance is not positive; the model output looks constant")
    return a, b, c, mean, float(variance)


def first_order_indices(f_a, f_b, f_ab):
    """Common-centered first-order covariance estimates, length dim."""
    return saltelli_indices(f_a, f_b, f_ab)[0]


def total_effect_indices(f_a, f_b, f_ab):
    """Common-centered total-effect covariance estimates, length dim."""
    return saltelli_indices(f_a, f_b, f_ab)[1]


def saltelli_indices(f_a, f_b, f_ab):
    """Return (S, T, pooled mean, pooled population variance).

    Uses one common A/B center for all three output families. Additive
    offsets cancel; a nonzero output scaling cancels between covariance
    and variance, up to rounding already present in the supplied outputs.
    """
    fa, fb, fab = _design_outputs(f_a, f_b, f_ab)
    a, b, c, mean, variance = _centered_design(fa, fb, fab)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        S = (c @ b / fa.size) / variance
        T = 1-(c @ a / fa.size) / variance
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(T)):
        raise FloatingPointError("indices are not representable; rescale outputs")
    return S, T, mean, variance
