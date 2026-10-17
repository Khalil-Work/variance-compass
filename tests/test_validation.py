"""Input validation across the public API."""

import numpy as np
import pytest

from variance_compass import (
    additive_analytic,
    additive_bounds,
    additive_model,
    analyze_sensitivity,
    convergence_sweep,
    ishigami,
    ishigami_bounds,
)


def test_non_callable_model_raises():
    with pytest.raises(TypeError, match="callable"):
        analyze_sensitivity(3.0, 8, ishigami_bounds(), seed=1)


@pytest.mark.parametrize("bad_n", [0, -5, 2.5, "16", True])
def test_bad_base_sample_size_raises(bad_n):
    with pytest.raises((TypeError, ValueError)):
        analyze_sensitivity(ishigami, bad_n, ishigami_bounds(), seed=1)


@pytest.mark.parametrize(
    "bad_bounds",
    [
        [(1.0, 2.0)],  # wrong number of inputs for ishigami
        [(2.0, 1.0)] * 3,  # low >= high
        [(1.0, 2.0), (1.0, np.nan), (0.0, 1.0)],  # non-finite
        [(0.0, 1.0, 2.0)] * 3,  # not (low, high) pairs
        [],
    ],
)
def test_bad_bounds_raise(bad_bounds):
    with pytest.raises(ValueError):
        analyze_sensitivity(ishigami, 8, bad_bounds, seed=1)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "seed"])
def test_bad_seed_raises(bad_seed):
    with pytest.raises((TypeError, ValueError)):
        analyze_sensitivity(ishigami, 8, ishigami_bounds(), seed=bad_seed)


def test_missing_seed_and_rng_raises():
    with pytest.raises(ValueError, match="seed"):
        analyze_sensitivity(ishigami, 8, ishigami_bounds())


def test_bad_rng_raises():
    with pytest.raises(TypeError, match="Generator"):
        analyze_sensitivity(
            ishigami, 8, ishigami_bounds(), rng=np.random.RandomState(0)
        )


@pytest.mark.parametrize("bad_b", [1, 0])
def test_bad_bootstrap_raises(bad_b):
    with pytest.raises((ValueError, TypeError)):
        analyze_sensitivity(
            ishigami, 8, ishigami_bounds(), seed=1, n_bootstrap=bad_b
        )


def test_model_with_wrong_output_shape_raises():
    def bad_model(X):
        return np.zeros(X.shape[0] + 1)

    with pytest.raises(ValueError, match="one finite scalar per input row"):
        analyze_sensitivity(bad_model, 8, ishigami_bounds(), seed=1)


def test_model_with_non_finite_output_raises():
    def bad_model(X):
        y = np.zeros(X.shape[0])
        y[0] = np.nan
        return y

    with pytest.raises(ValueError, match="non-finite"):
        analyze_sensitivity(bad_model, 8, ishigami_bounds(), seed=1)


def test_sweep_rejects_empty_and_bad_sizes():
    with pytest.raises(ValueError):
        convergence_sweep(ishigami, ishigami_bounds(), [], seed=1)
    with pytest.raises((TypeError, ValueError)):
        convergence_sweep(ishigami, ishigami_bounds(), [8, -1], seed=1)


def test_additive_model_validation():
    with pytest.raises(ValueError, match="non-empty"):
        additive_model([])
    with pytest.raises(ValueError, match="finite"):
        additive_model([1.0, np.inf])
    model = additive_model((1.0, 2.0))
    with pytest.raises(ValueError, match="\\(n, 2\\)"):
        model(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        model(np.array([[np.nan, 0.0]]))


def test_additive_analytic_requires_nonzero_coefficients():
    with pytest.raises(ValueError, match="non-zero"):
        additive_analytic((0.0, 0.0))


def test_additive_bounds_shape():
    b = additive_bounds(4)
    assert b.shape == (4, 2)
    assert np.all(b[:, 0] == -1.0) and np.all(b[:, 1] == 1.0)
    with pytest.raises((TypeError, ValueError)):
        additive_bounds(0)
