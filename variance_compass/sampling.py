"""Saltelli sampling scheme, implemented from the mathematics.

Design (Saltelli et al., 2010; described in Saltelli et al., 2008,
"Global Sensitivity Analysis: The Primer", Chapter 4): draw two
independent base matrices ``A`` and ``B`` of shape ``(n, dim)`` with
i.i.d. ``U(0, 1)`` entries, then for each input ``i`` form the crossed
matrix ``A_B^(i)``, which is ``A`` with its ``i``-th column replaced by
the ``i``-th column of ``B``.

The full design costs ``n * (dim + 2)`` model evaluations and provides
both the first-order estimator (pairing ``f(B)`` with ``f(A_B^(i))``,
which share exactly the transplanted input ``X_i``) and the
total-effect estimator (pairing ``f(A)`` with ``f(A_B^(i))``, which
share every input except ``X_i``); see
:mod:`variance_compass.estimators` for the expectation algebra.

The sampler below is original code: it uses a ``numpy.random.Generator``
(PCG64 by default) directly and is reproducible for a documented seed.
"""

from __future__ import annotations

import numpy as np

from .validation import validate_bounds, validate_positive_int

__all__ = ["saltelli_matrices", "to_physical"]


def saltelli_matrices(n, dim, rng):
    """Build the Saltelli design matrices for ``dim`` inputs.

    Parameters
    ----------
    n : int
        Base sample size (positive). The total design cost is
        ``n * (dim + 2)`` rows of the model.
    dim : int
        Number of independent inputs (positive).
    rng : numpy.random.Generator
        Source of ``U(0, 1)`` draws. Pass a Generator seeded with a
        documented seed for reproducibility.

    Returns
    -------
    A, B : numpy.ndarray, shape ``(n, dim)``
        Independent base matrices with i.i.d. ``U(0, 1)`` entries.
    crossed : list of numpy.ndarray
        ``crossed[i]`` equals ``A`` except column ``i``, which is taken
        from ``B`` (a fresh copy; ``A`` is never mutated).
    """
    n = validate_positive_int(n, "n")
    dim = validate_positive_int(dim, "dim")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    A = rng.random((n, dim))
    B = rng.random((n, dim))
    crossed = []
    for i in range(dim):
        A_B_i = A.copy()
        A_B_i[:, i] = B[:, i]
        crossed.append(A_B_i)
    return A, B, crossed


def to_physical(u, bounds):
    """Inverse-transform ``U(0, 1)`` draws to uniform marginals on bounds.

    ``u`` is an ``(n, dim)`` array in ``[0, 1)``; ``bounds`` is a
    ``(dim, 2)`` array of ``(low, high)`` pairs.  Returns
    ``low + (high - low) * u``.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"u must be a 2-D array, got shape {u.shape}")
    bounds_arr = validate_bounds(bounds, dim=u.shape[1])
    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    return low + (high - low) * u
