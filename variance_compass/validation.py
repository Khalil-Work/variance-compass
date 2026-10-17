"""Centralised input validation for :mod:`variance_compass`.

Every public entry point validates its arguments through the helpers in
this module so that failure modes are explicit, uniform and testable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_model",
    "validate_positive_int",
    "validate_seed",
    "validate_bounds",
    "validate_finite_scalar",
]


def validate_model(model):
    """Return ``model`` if it is callable, otherwise raise ``TypeError``."""
    if not callable(model):
        raise TypeError(f"model must be callable, got {type(model).__name__}")
    return model


def validate_positive_int(value, name, *, minimum=1):
    """Validate that ``value`` is an integer with ``value >= minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def validate_seed(seed):
    """Validate that ``seed`` is an integer accepted by ``default_rng``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be in [0, 2**63 - 1]")
    return seed


def validate_bounds(bounds, dim=None):
    """Validate per-input uniform bounds ``[(low, high), ...]``.

    Returns a ``float`` ndarray of shape ``(dim, 2)`` with ``low < high``.
    """
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "bounds must be an array-like of shape (dim, 2) whose rows are (low, high) pairs"
        )
    if arr.shape[0] == 0:
        raise ValueError("bounds must contain at least one (low, high) pair")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds must contain only finite values")
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError("each bounds pair requires low < high")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"bounds define {arr.shape[0]} inputs but {dim} were expected")
    return arr


def validate_finite_scalar(value, name):
    """Validate that ``value`` is a finite real number and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value
