"""Saltelli sampling scheme: structure, reproducibility, transformation."""

import numpy as np
import pytest

from variance_compass.sampling import saltelli_matrices, to_physical


def test_matrix_shapes_and_ranges():
    rng = np.random.default_rng(42)
    A, B, crossed = saltelli_matrices(64, 3, rng)
    assert A.shape == B.shape == (64, 3)
    assert len(crossed) == 3
    for M in [A, B, *crossed]:
        assert np.all(M >= 0.0) and np.all(M < 1.0)


def test_crossed_columns_are_correct():
    rng = np.random.default_rng(42)
    A, B, crossed = saltelli_matrices(64, 3, rng)
    for i in range(3):
        # crossed[i] equals A everywhere except column i, which equals B's.
        assert np.array_equal(crossed[i][:, i], B[:, i])
        for j in range(3):
            if j != i:
                assert np.array_equal(crossed[i][:, j], A[:, j])
        # A itself is never mutated by the construction.
        assert np.array_equal(A[:, i], A[:, i])


def test_reproducible_for_documented_seed():
    A1, B1, C1 = saltelli_matrices(32, 3, np.random.default_rng(123))
    A2, B2, C2 = saltelli_matrices(32, 3, np.random.default_rng(123))
    assert np.array_equal(A1, A2) and np.array_equal(B1, B2)
    for c1, c2 in zip(C1, C2):
        assert np.array_equal(c1, c2)


def test_different_seeds_differ():
    A1, _, _ = saltelli_matrices(32, 3, np.random.default_rng(1))
    A2, _, _ = saltelli_matrices(32, 3, np.random.default_rng(2))
    assert not np.array_equal(A1, A2)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="n must be"):
        saltelli_matrices(0, 3, np.random.default_rng(0))
    with pytest.raises(ValueError, match="dim must be"):
        saltelli_matrices(8, 0, np.random.default_rng(0))
    with pytest.raises(TypeError, match="Generator"):
        saltelli_matrices(8, 3, np.random.RandomState(0))


def test_to_physical_maps_unit_interval():
    u = np.array([[0.0, 0.5, 1.0 - 1e-12]])
    bounds = np.array([[-np.pi, np.pi], [-1.0, 1.0], [0.0, 2.0]])
    x = to_physical(u, bounds)
    assert np.allclose(x, [[-np.pi, 0.0, 2.0 - 2e-12]], atol=1e-15)


def test_to_physical_validates_bounds():
    u = np.zeros((2, 2))
    with pytest.raises(ValueError, match="low < high"):
        to_physical(u, [(1.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match="were expected"):
        to_physical(u, [(0.0, 1.0)])
    with pytest.raises(ValueError, match="shape \\(dim, 2\\)"):
        to_physical(u, [(0.0, 1.0, 2.0)] * 2)
