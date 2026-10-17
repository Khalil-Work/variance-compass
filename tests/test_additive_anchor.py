"""Exact-regression anchor: the additive model has S_i = T_i = V_i / V
analytically, so every deviation of the Saltelli estimates is pure
estimator error.

At N = 2**17 (seed 424242, 100 bootstrap replicates) the bootstrap SE of
each index is ~0.003, so the required ~1e-2 agreement is a >3-sigma
statement; observed absolute errors are <= 0.0026 (deterministic for the
seed).
"""

import numpy as np
import pytest

from variance_compass import (
    additive_analytic,
    additive_bounds,
    additive_model,
    analyze_sensitivity,
)

COEFFS = (2.0, 1.5, 0.5, -1.0)
SEED = 424242
N = 2**17
TOL_S = 0.01  # required ~1e-2 agreement for the anchor
TOL_T = 0.01  # T_i = S_i analytically; same estimator noise applies
TOL_GAP = 0.01  # |T_i - S_i| has SE ~ 0.003; this is a > 3-sigma bound
TOL_VARIANCE_REL = 0.02


@pytest.fixture(scope="module")
def anchor():
    model = additive_model(COEFFS)
    return analyze_sensitivity(
        model, N, additive_bounds(len(COEFFS)), seed=SEED, n_bootstrap=100
    )


def test_additive_analytic_values():
    an = additive_analytic(COEFFS)
    c2 = np.array(COEFFS) ** 2
    assert np.allclose(an.S, c2 / c2.sum(), rtol=1e-15)
    assert an.S == an.T  # strictly additive: total == first order
    assert an.variance == pytest.approx(c2.sum() / 3.0, rel=1e-15)


def test_anchor_first_order_within_1e_minus_2(anchor):
    an = additive_analytic(COEFFS)
    errors = np.abs(anchor.S - np.array(an.S))
    assert np.all(errors <= TOL_S)
    assert errors.max() < 0.002  # observed: 0.0017 (deterministic for the seed)


def test_anchor_total_effect_matches_first_order(anchor):
    an = additive_analytic(COEFFS)
    errors = np.abs(anchor.T - np.array(an.T))
    assert np.all(errors <= TOL_T)
    assert np.all(np.abs(anchor.T - anchor.S) <= TOL_GAP)


def test_anchor_total_ge_first_order(anchor):
    assert np.all(anchor.T >= anchor.S - TOL_GAP)


def test_anchor_variance_matches_sum_of_variances(anchor):
    an = additive_analytic(COEFFS)
    assert anchor.variance == pytest.approx(an.variance, rel=TOL_VARIANCE_REL)


def test_anchor_bootstrap_se_is_small(anchor):
    assert np.all(anchor.S_se < 0.005)
    assert np.all(anchor.T_se < 0.005)


def test_anchor_reproducible_for_documented_seed():
    model = additive_model(COEFFS)
    r1 = analyze_sensitivity(
        model, 2**14, additive_bounds(len(COEFFS)), seed=SEED, n_bootstrap=20
    )
    r2 = analyze_sensitivity(
        model, 2**14, additive_bounds(len(COEFFS)), seed=SEED, n_bootstrap=20
    )
    assert np.array_equal(r1.S, r2.S) and np.array_equal(r1.T, r2.T)
