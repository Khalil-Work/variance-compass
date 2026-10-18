"""Sensitivity analysis driver: Saltelli estimates with bootstrap
uncertainty and convergence diagnostics versus sample size.

A single ``numpy.random.Generator`` drives sampling and bootstrap for a
run, so every result is exactly reproducible from its documented seed
(and, for sweeps, from the position in the sweep).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .estimators import saltelli_indices, _ZeroVarianceError
from .sampling import saltelli_matrices, to_physical
from .validation import (
    validate_bounds,
    validate_model,
    validate_positive_int,
    validate_seed,
)

__all__ = [
    "AnalyticIndices",
    "SensitivityResult",
    "SweepResult",
    "analyze_sensitivity",
    "convergence_sweep",
    "index_errors",
]


@dataclass(frozen=True)
class AnalyticIndices:
    """Analytic ground truth for a user-supplied test function."""

    variance: float
    S: tuple
    T: tuple


@dataclass(eq=False)
class SensitivityResult:
    """Saltelli estimates for one sample size, with bootstrap uncertainty.

    Attributes
    ----------
    n, dim : int
        Base sample size and number of inputs.
    seed : int or None
        Documented seed of the run (``None`` if a shared Generator was
        passed in directly; see :func:`convergence_sweep`).
    S, T : numpy.ndarray, shape ``(dim,)``
        First-order and total-effect index estimates.
    S_se, T_se : numpy.ndarray, shape ``(dim,)``
        Bootstrap standard errors of the estimates.
    S_ci, T_ci : numpy.ndarray, shape ``(dim, 2)``
        Bootstrap percentile confidence intervals (2.5%, 97.5%).
    mean, variance : float
        Pooled mean and symmetric variance estimates of the output.
    n_bootstrap : int
        Number of bootstrap resamples used for the uncertainty.
    runtime_s : float
        Wall-clock runtime of the analysis in seconds.
    """

    n: int
    dim: int
    seed: int | None
    S: np.ndarray
    T: np.ndarray
    S_se: np.ndarray
    T_se: np.ndarray
    S_ci: np.ndarray
    T_ci: np.ndarray
    mean: float
    variance: float
    n_bootstrap: int
    runtime_s: float

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "dim": int(self.dim),
            "seed": None if self.seed is None else int(self.seed),
            "S": [float(v) for v in self.S],
            "T": [float(v) for v in self.T],
            "S_se": [float(v) for v in self.S_se],
            "T_se": [float(v) for v in self.T_se],
            "S_ci": [[float(a), float(b)] for a, b in self.S_ci],
            "T_ci": [[float(a), float(b)] for a, b in self.T_ci],
            "mean": float(self.mean),
            "variance": float(self.variance),
            "n_bootstrap": int(self.n_bootstrap),
            "runtime_s": float(self.runtime_s),
        }


@dataclass(eq=False)
class SweepResult:
    """A convergence sweep: one :class:`SensitivityResult` per sample size."""

    n_values: np.ndarray
    results: list
    seed: int

    @property
    def S_matrix(self) -> np.ndarray:
        """Shape ``(len(n_values), dim)`` matrix of first-order estimates."""
        return np.vstack([r.S for r in self.results])

    @property
    def T_matrix(self) -> np.ndarray:
        """Shape ``(len(n_values), dim)`` matrix of total-effect estimates."""
        return np.vstack([r.T for r in self.results])

    @property
    def S_se_matrix(self) -> np.ndarray:
        return np.vstack([r.S_se for r in self.results])

    @property
    def T_se_matrix(self) -> np.ndarray:
        return np.vstack([r.T_se for r in self.results])


def _evaluate(model, X):
    y = np.asarray(model(X), dtype=float)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            "model must return exactly one finite scalar per input row "
            f"(got output shape {y.shape} for {X.shape[0]} rows)"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("model returned non-finite values")
    return y


def _bootstrap_indices(f_a, f_b, f_ab, n_boot, rng):
    """Nonparametric bootstrap of the Saltelli estimators.

    Resamples the ``n`` design rows with replacement (jointly for all
    ``f_a``, ``f_b``, ``f_ab`` outputs) and recomputes the indices for
    each replicate.  Degenerate replicates with non-positive variance
    are recorded as NaN and skipped by the nan-aware statistics. At least
    two valid replicates are required for a finite ddof=1 standard error.
    Each replicate reuses the same common-centered public estimator.
    """
    n = f_a.size
    k = f_ab.shape[0]
    S_b = np.full((n_boot, k), np.nan)
    T_b = np.full((n_boot, k), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f_a[idx]
        fb = f_b[idx]
        fab = f_ab[:, idx]
        try:
            S_b[b], T_b[b], _, _ = saltelli_indices(fa, fb, fab)
        except _ZeroVarianceError:
            continue
    if np.count_nonzero(np.all(np.isfinite(S_b), axis=1)) < 2:
        raise ValueError("at least two nondegenerate bootstrap replicates are required")
    S_se = np.nanstd(S_b, axis=0, ddof=1)
    T_se = np.nanstd(T_b, axis=0, ddof=1)
    S_ci = np.nanpercentile(S_b, [2.5, 97.5], axis=0).T
    T_ci = np.nanpercentile(T_b, [2.5, 97.5], axis=0).T
    return S_se, T_se, S_ci, T_ci


def analyze_sensitivity(model, n, bounds, *, seed=None, rng=None, n_bootstrap=100):
    """Estimate first-order and total-effect Sobol' indices via Saltelli.

    Parameters
    ----------
    model : callable
        Vectorised map ``(n, dim) -> (n,)`` returning one scalar per row.
    n : int
        Base sample size; the design costs ``n * (dim + 2)`` evaluations.
    bounds : array_like, shape ``(dim, 2)``
        Uniform marginal ``(low, high)`` per input.
    seed : int, optional
        Seed for the ``numpy.random.Generator``.  Either ``seed`` or
        ``rng`` must be given.
    rng : numpy.random.Generator, optional
        Pre-built generator (e.g. threaded through a sweep); sampling
        and bootstrap draws are taken from it in order, so results are
        reproducible from the seed that created the stream.
    n_bootstrap : int, optional
        Number of bootstrap resamples for the uncertainty (>= 2).

    Returns
    -------
    SensitivityResult
    """
    validate_model(model)
    n = validate_positive_int(n, "n")
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap", minimum=2)
    bounds_arr = validate_bounds(bounds)
    dim = bounds_arr.shape[0]
    if rng is None:
        if seed is None:
            raise ValueError(
                "provide either an integer seed or a numpy Generator via rng"
            )
        seed_used = validate_seed(seed)
        rng = np.random.default_rng(seed_used)
    else:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
            )
        seed_used = validate_seed(seed) if seed is not None else None

    t0 = time.perf_counter()
    A, B, crossed = saltelli_matrices(n, dim, rng)
    f_a = _evaluate(model, to_physical(A, bounds_arr))
    f_b = _evaluate(model, to_physical(B, bounds_arr))
    f_ab = np.vstack(
        [_evaluate(model, to_physical(X_i, bounds_arr)) for X_i in crossed]
    )
    S, T, mean, variance = saltelli_indices(f_a, f_b, f_ab)
    S_se, T_se, S_ci, T_ci = _bootstrap_indices(f_a, f_b, f_ab, n_bootstrap, rng)
    runtime = time.perf_counter() - t0
    return SensitivityResult(
        n=n,
        dim=dim,
        seed=seed_used,
        S=S,
        T=T,
        S_se=S_se,
        T_se=T_se,
        S_ci=S_ci,
        T_ci=T_ci,
        mean=mean,
        variance=variance,
        n_bootstrap=n_bootstrap,
        runtime_s=runtime,
    )


def convergence_sweep(model, bounds, n_values, *, seed, n_bootstrap=100):
    """Run :func:`analyze_sensitivity` over increasing sample sizes.

    A single ``Generator`` seeded with ``seed`` is threaded through the
    whole sweep in order of increasing ``n`` (duplicates removed), so
    the entire sweep is deterministic for a documented seed.
    """
    validate_model(model)
    bounds_arr = validate_bounds(bounds)
    seed = validate_seed(seed)
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap", minimum=2)
    try:
        entries = list(n_values)
    except TypeError as exc:
        raise ValueError("n_values must be a non-empty sequence of sample sizes") from exc
    if not entries:
        raise ValueError("n_values must be a non-empty sequence of sample sizes")
    sizes = np.unique(
        np.asarray(
            [validate_positive_int(v, "n_values entry") for v in entries],
            dtype=np.int64,
        )
    )
    rng = np.random.default_rng(seed)
    results = []
    for n in sizes:
        results.append(
            analyze_sensitivity(
                model,
                int(n),
                bounds_arr,
                seed=seed,
                rng=rng,
                n_bootstrap=n_bootstrap,
            )
        )
    return SweepResult(n_values=sizes, results=results, seed=seed)


def index_errors(S, T, S_ref, T_ref) -> dict:
    """Absolute estimation errors against analytic reference indices."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    S_ref = np.asarray(S_ref, dtype=float)
    T_ref = np.asarray(T_ref, dtype=float)
    if S.shape != S_ref.shape or T.shape != T_ref.shape:
        raise ValueError("estimates and references must have matching shapes")
    S_abs = np.abs(S - S_ref)
    T_abs = np.abs(T - T_ref)
    return {
        "S_abs": S_abs,
        "T_abs": T_abs,
        "max_S_abs": float(S_abs.max()),
        "max_T_abs": float(T_abs.max()),
    }
