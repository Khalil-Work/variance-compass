"""Ishigami test function and its analytic Sobol' indices.

The Ishigami function (Ishigami & Homma, 1990, "An importance
quantification technique in uncertainty analysis for computer models";
popularised by Sobol' and by Saltelli et al., 2008, "Global Sensitivity
Analysis: The Primer") is

    f(x1, x2, x3) = sin(x1) + a * sin(x2)**2 + b * x3**4 * sin(x1)

with independent inputs ``x_i ~ Uniform(-pi, pi)`` and the standard
published parameters ``a = 7`` and ``b = 0.1``.

Derivation sketch (full derivation in ``docs/model.md``)
--------------------------------------------------------
For ``x ~ U(-pi, pi)``: ``E[sin(x)] = 0``, ``E[sin(x)^2] = 1/2``,
``E[sin(x)^4] = 3/8``, ``E[x^4] = pi^4 / 5``, ``E[x^8] = pi^8 / 9``.
Splitting ``f - E[f] = sin(x1)(1 + b*x3^4) + a(sin(x2)^2 - 1/2)`` into
independent zero-mean parts and using the orthogonality of the HDMR
terms gives

    E[f]     = a / 2
    V        = 1/2 + a^2/8 + b*pi^4/5 + b^2*pi^8/18        (total variance)
    V1       = 1/2 * (1 + b*pi^4/5)^2                       (first order, x1)
    V2       = a^2 / 8                                       (first order, x2)
    V3       = 0                                             (first order, x3)
    V13      = 8 * b^2 * pi^8 / 225                          (x1-x3 interaction)

with the exact decomposition identity ``V1 + V2 + V3 + V13 = V``.
Hence ``S_i = V_i / V`` and ``T_i = (V_i + V13 * 1[i in {1, 3}]) / V``;
numerically, for a = 7 and b = 0.1:

    S = (0.3139, 0.4424, 0.0000)      T = (0.5576, 0.4424, 0.2437)

The first-order deficit ``1 - (S1 + S2 + S3)`` equals ``V13 / V = T3``,
which is exactly the known x1-x3 interaction structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import validate_finite_scalar

__all__ = [
    "ISHIGAMI_A",
    "ISHIGAMI_B",
    "IshigamiAnalytic",
    "ishigami",
    "ishigami_analytic",
    "ishigami_bounds",
]

#: Standard published Ishigami parameters.
ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1


def ishigami_bounds() -> np.ndarray:
    """Return the ``(3, 2)`` bounds array ``[(-pi, pi)] * 3``."""
    return np.tile(np.array([-np.pi, np.pi], dtype=float), (3, 1))


def ishigami(X, a=ISHIGAMI_A, b=ISHIGAMI_B):
    """Evaluate the Ishigami function on an ``(n, 3)`` array of inputs.

    Parameters
    ----------
    X : array_like, shape ``(n, 3)``
        Rows are ``(x1, x2, x3)``.
    a, b : float, optional
        Function parameters (defaults are the standard published values).

    Returns
    -------
    numpy.ndarray, shape ``(n,)``
        ``sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)`` per row.
    """
    a = validate_finite_scalar(a, "a")
    b = validate_finite_scalar(b, "b")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"ishigami expects an (n, 3) array, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("ishigami input contains non-finite values")
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)


@dataclass(frozen=True)
class IshigamiAnalytic:
    """Analytic ground truth for the Ishigami function."""

    a: float
    b: float
    variance: float
    first_order_variances: tuple  # (V1, V2, V3)
    interaction_variance: float  # V13, the x1-x3 interaction
    S: tuple  # analytic first-order indices
    T: tuple  # analytic total-effect indices

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "variance": self.variance,
            "first_order_variances": list(self.first_order_variances),
            "interaction_variance": self.interaction_variance,
            "S": list(self.S),
            "T": list(self.T),
        }


def ishigami_analytic(a=ISHIGAMI_A, b=ISHIGAMI_B) -> IshigamiAnalytic:
    """Closed-form variance decomposition of the Ishigami function.

    See the module docstring for the derivation sketch.  The exact
    identity ``V1 + V2 + V3 + V13 = V`` is asserted internally to guard
    against transcription errors in the formulas.
    """
    a = validate_finite_scalar(a, "a")
    b = validate_finite_scalar(b, "b")
    pi4 = np.pi**4
    pi8 = np.pi**8

    variance = 0.5 + a * a / 8.0 + b * pi4 / 5.0 + b * b * pi8 / 18.0
    v1 = 0.5 * (1.0 + b * pi4 / 5.0) ** 2
    v2 = a * a / 8.0
    v3 = 0.0
    v13 = 8.0 * b * b * pi8 / 225.0

    total = v1 + v2 + v3 + v13
    if not np.isclose(total, variance, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "internal inconsistency in the Ishigami variance decomposition"
        )
    if variance <= 0.0:
        raise ValueError("analytic variance must be positive for the given a, b")

    S = (v1 / variance, v2 / variance, v3 / variance)
    T = ((v1 + v13) / variance, v2 / variance, (v3 + v13) / variance)
    return IshigamiAnalytic(
        a=a,
        b=b,
        variance=float(variance),
        first_order_variances=(float(v1), float(v2), float(v3)),
        interaction_variance=float(v13),
        S=tuple(float(s) for s in S),
        T=tuple(float(t) for t in T),
    )
