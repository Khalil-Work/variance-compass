"""Ishigami function: pointwise evaluation, analytic indices vs the
published literature values, and an independent numerical verification of
the closed-form decomposition via trapezoidal quadrature."""

import math

import numpy as np
import pytest

from variance_compass import ishigami, ishigami_analytic, ishigami_bounds
from variance_compass.ishigami import ISHIGAMI_A, ISHIGAMI_B


def test_reference_parameters_are_the_standard_published_ones():
    assert ISHIGAMI_A == 7.0
    assert ISHIGAMI_B == 0.1


def test_pointwise_values_match_formula():
    pts = np.array([[0.3, -1.2, 2.0], [1.0, 0.5, -0.7], [0.0, 0.0, 0.0]])
    expected = np.array(
        [
            math.sin(0.3) + 7.0 * math.sin(-1.2) ** 2
            + 0.1 * 2.0**4 * math.sin(0.3),
            math.sin(1.0) + 7.0 * math.sin(0.5) ** 2
            + 0.1 * (-0.7) ** 4 * math.sin(1.0),
            0.0,
        ]
    )
    assert np.allclose(ishigami(pts), expected, rtol=0, atol=1e-12)
    # Non-default parameters are honoured.
    other = ishigami(pts, a=2.0, b=0.05)
    expected_other = np.array(
        [
            math.sin(0.3) + 2.0 * math.sin(-1.2) ** 2 + 0.05 * 2.0**4 * math.sin(0.3),
            math.sin(1.0) + 2.0 * math.sin(0.5) ** 2 + 0.05 * (-0.7) ** 4 * math.sin(1.0),
            0.0,
        ]
    )
    assert np.allclose(other, expected_other, rtol=0, atol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        ishigami(np.zeros(3))
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        ishigami(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="non-finite"):
        ishigami(np.array([[0.0, np.nan, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        ishigami(np.zeros((1, 3)), a=math.inf)


def test_bounds_shape():
    b = ishigami_bounds()
    assert b.shape == (3, 2)
    assert np.allclose(b, -np.pi) or np.allclose(b[:, 0], -np.pi)
    assert np.allclose(b[:, 0], -np.pi) and np.allclose(b[:, 1], np.pi)


def test_analytic_indices_match_published_values():
    """Values widely published for a = 7, b = 0.1 (e.g. Saltelli et al.,
    "Global Sensitivity Analysis: The Primer", 2008, Table 1... Chapter 4)."""
    an = ishigami_analytic()
    assert an.S == pytest.approx((0.3139, 0.4424, 0.0), abs=5e-5)
    assert an.T == pytest.approx((0.5576, 0.4424, 0.2437), abs=5e-5)
    assert an.variance == pytest.approx(13.8446, abs=5e-4)


def test_exact_decomposition_identity():
    an = ishigami_analytic()
    v1, v2, v3 = an.first_order_variances
    total = v1 + v2 + v3 + an.interaction_variance
    assert total == pytest.approx(an.variance, rel=1e-12)


def test_first_order_deficit_equals_interaction_total_index():
    an = ishigami_analytic()
    assert 1 - sum(an.S) == pytest.approx(an.T[2], rel=1e-12)
    assert 1 - sum(an.S) == pytest.approx(
        an.interaction_variance / an.variance, rel=1e-12
    )


def test_analytic_index_ranges_and_ordering():
    an = ishigami_analytic()
    for s, t in zip(an.S, an.T):
        assert 0.0 <= s <= 1.0
        assert 0.0 <= t <= 1.0 + 1e-12
        assert t >= s  # total-effect always dominates first order


def _cond_mean_curve(ishigami_fn, axis, m, q):
    """Numerical E[f | x_axis] over a trapezoidal grid of the other inputs.

    Integrals are normalised by the domain volume (2*pi)^2 so the curve is
    a conditional *mean*, then variance over x_axis uses the 1/(2*pi)
    normalisation of the outer integral.
    """
    others = [ax for ax in (0, 1, 2) if ax != axis]
    g1 = np.linspace(-np.pi, np.pi, q)
    g2 = np.linspace(-np.pi, np.pi, q)
    G1, G2 = np.meshgrid(g1, g2, indexing="ij")
    x_grid = np.linspace(-np.pi, np.pi, m)
    curve = np.empty(m)
    for j, xj in enumerate(x_grid):
        pts = np.zeros((q * q, 3))
        pts[:, others[0]] = G1.ravel()
        pts[:, others[1]] = G2.ravel()
        pts[:, axis] = xj
        vals = ishigami_fn(pts).reshape(q, q)
        inner = np.trapezoid(vals, x=g2, axis=1)
        curve[j] = np.trapezoid(inner, x=g1) / (2 * np.pi) ** 2
    return x_grid, curve


@pytest.mark.parametrize(
    "axis, expected",
    [
        (0, lambda an: an.first_order_variances[0]),
        (1, lambda an: an.first_order_variances[1]),
        (2, lambda an: an.first_order_variances[2]),
    ],
)
def test_first_order_variances_by_quadrature(axis, expected):
    """Independently re-derive V_i = Var(E[f | x_i]) by numerical
    integration; agreement rules out transcription errors in the closed
    forms."""
    an = ishigami_analytic()
    x, g = _cond_mean_curve(ishigami, axis, m=161, q=81)
    mean_g = np.trapezoid(g, x=x) / (2 * np.pi)
    v_i = np.trapezoid((g - mean_g) ** 2, x=x) / (2 * np.pi)
    assert v_i == pytest.approx(expected(an), rel=2e-3)


def test_total_variance_by_quadrature():
    q = 101  # fine enough that the O(h^2) trapezoid error stays below 2e-3
    g1 = np.linspace(-np.pi, np.pi, q)
    G1, G2, G3 = np.meshgrid(g1, g1, g1, indexing="ij")
    pts = np.column_stack([G1.ravel(), G2.ravel(), G3.ravel()])
    F = ishigami(pts).reshape(q, q, q)
    volume = (2 * np.pi) ** 3
    mean = (
        np.trapezoid(
            np.trapezoid(np.trapezoid(F, x=g1, axis=2), x=g1, axis=1), x=g1
        )
        / volume
    )
    second = (
        np.trapezoid(
            np.trapezoid(np.trapezoid(F * F, x=g1, axis=2), x=g1, axis=1), x=g1
        )
        / volume
    )
    assert second - mean**2 == pytest.approx(
        ishigami_analytic().variance, rel=2e-3
    )


def test_interaction_variance_by_subtraction():
    """V13 = V - V1 - V2 - V3, using the quadrature-verified pieces."""
    an = ishigami_analytic()
    v1, v2, v3 = an.first_order_variances
    v13 = an.variance - v1 - v2 - v3
    assert v13 == pytest.approx(an.interaction_variance, rel=1e-12)
    assert an.interaction_variance == pytest.approx(
        8 * 0.1**2 * np.pi**8 / 225.0, rel=1e-12
    )
