"""Moment and index estimators: exact agreement with numpy on identical
samples, structural sanity, and documented failure modes."""

import numpy as np
import pytest

from variance_compass.estimators import (
    first_order_indices,
    sample_mean,
    sample_variance,
    saltelli_indices,
    total_effect_indices,
)

RNG = np.random.default_rng(7)


@pytest.mark.parametrize("size", [1, 2, 17, 1000])
def test_sample_mean_matches_numpy(size):
    x = RNG.normal(size=size)
    assert sample_mean(x) == pytest.approx(np.mean(x), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_sample_variance_matches_numpy(ddof):
    x = RNG.normal(size=500)
    assert sample_variance(x, ddof=ddof) == pytest.approx(
        np.var(x, ddof=ddof), rel=1e-12, abs=1e-15
    )


def test_sample_mean_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sample_mean(np.array([]))


def test_sample_variance_bad_shape_raises():
    with pytest.raises(ValueError, match="1-D"):
        sample_variance(np.zeros((3, 3)))


def test_sample_variance_ddof_too_large_raises():
    with pytest.raises(ValueError, match="too small"):
        sample_variance(np.array([1.0, 2.0]), ddof=2)


def test_saltelli_indices_shapes_and_sanity():
    n, k = 500, 3
    fa = RNG.normal(size=n)
    fb = RNG.normal(size=n)
    fab = RNG.normal(size=(k, n))
    S, T, mean, variance = saltelli_indices(fa, fb, fab)
    assert S.shape == T.shape == (k,)
    assert isinstance(mean, float) and isinstance(variance, float)
    assert variance > 0.0


def test_split_estimators_match_combined_call():
    n, k = 300, 4
    fa = RNG.normal(size=n)
    fb = RNG.normal(size=n)
    fab = RNG.normal(size=(k, n))
    S, T, _, _ = saltelli_indices(fa, fb, fab)
    assert np.array_equal(first_order_indices(fa, fb, fab), S)
    assert np.array_equal(total_effect_indices(fa, fb, fab), T)


def test_saltelli_indices_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same sample size"):
        saltelli_indices(np.ones(10), np.ones(11), np.ones((3, 10)))


def test_saltelli_indices_constant_output_raises():
    with pytest.raises(ValueError, match="constant"):
        saltelli_indices(np.ones(10), np.ones(10), np.ones((2, 10)))


def test_non_finite_outputs_raise():
    fa = np.ones(8)
    fa[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sample_mean(fa)
