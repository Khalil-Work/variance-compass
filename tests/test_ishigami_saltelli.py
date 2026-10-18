"""Fixed-seed covariance estimates vs analytic Ishigami indices.

The 0.02 allowances exceed three bootstrap standard errors on the
documented N=2**17 fixture. Exact current measurements are reproduced by
examples/run_verification.py and saved in run_summary.json; they are not
universal coverage or unbiasedness claims.
"""

import numpy as np
import pytest

from variance_compass import (
    analyze_sensitivity,
    index_errors,
    ishigami,
    ishigami_analytic,
    ishigami_bounds,
)

SEED = 20261018
N = 2**17
N_BOOTSTRAP = 200
TOL_INDEX = 0.02  # > 3 x bootstrap SE (~0.013) of any index at this N
TOL_GAP = 0.02  # >= 3 x SE of the (T_i - S_i) difference; x2's true gap is 0
TOL_DEFICIT = 0.03  # ~3 x SE of the sum of the three S_i estimates
TOL_VARIANCE_REL = 0.02  # variance estimate is much better determined


@pytest.fixture(scope="module")
def result():
    return analyze_sensitivity(
        ishigami, N, ishigami_bounds(), seed=SEED, n_bootstrap=N_BOOTSTRAP
    )


@pytest.fixture(scope="module")
def analytic():
    return ishigami_analytic()


def test_first_order_indices_match_analytic(result, analytic):
    errors = np.abs(result.S - np.array(analytic.S))
    assert np.all(errors <= TOL_INDEX)
    assert errors.max() < 0.006  # observed centered max: 0.0031


def test_total_effect_indices_match_analytic(result, analytic):
    errors = np.abs(result.T - np.array(analytic.T))
    assert np.all(errors <= TOL_INDEX)
    assert errors.max() < 0.007  # observed centered max: 0.0035


def test_bootstrap_se_supports_the_stated_tolerance(result):
    """3 x bootstrap SE must sit inside the tolerance, i.e. the tolerance
    really is a ~3-sigma statistical bound rather than a loose cap."""
    assert np.all(3 * result.S_se <= TOL_INDEX)
    assert np.all(3 * result.T_se <= TOL_INDEX)
    assert np.all(result.S_se > 0) and np.all(result.T_se > 0)


def test_confidence_intervals_bracket_point_estimates(result):
    assert np.all(result.S_ci[:, 0] <= result.S + 1e-12)
    assert np.all(result.S_ci[:, 1] >= result.S - 1e-12)
    assert np.all(result.T_ci[:, 0] <= result.T + 1e-12)
    assert np.all(result.T_ci[:, 1] >= result.T - 1e-12)


def test_total_ge_first_order(result):
    """Necessary property: T_i >= S_i in expectation; finite-sample noise
    may dent it by a few SE (the x2 gap is exactly zero analytically)."""
    assert np.all(result.T >= result.S - TOL_GAP)


def test_first_order_sum_leq_one_and_deficit_is_the_known_interaction(
    result, analytic
):
    total_first_order = result.S.sum()
    assert total_first_order <= 1.0 + 1e-9
    deficit = 1.0 - total_first_order
    interaction_fraction = 1.0 - sum(analytic.S)  # = V13 / V = T3
    assert deficit == pytest.approx(interaction_fraction, abs=TOL_DEFICIT)


def test_variance_estimate_matches_analytic(result, analytic):
    assert result.variance == pytest.approx(
        analytic.variance, rel=TOL_VARIANCE_REL
    )


def test_mean_estimate_is_a_over_two(result, analytic):
    assert result.mean == pytest.approx(analytic.a / 2.0, abs=0.05)


def test_exact_reproducibility_for_documented_seed():
    r1 = analyze_sensitivity(
        ishigami, 2**14, ishigami_bounds(), seed=SEED, n_bootstrap=50
    )
    r2 = analyze_sensitivity(
        ishigami, 2**14, ishigami_bounds(), seed=SEED, n_bootstrap=50
    )
    assert np.array_equal(r1.S, r2.S)
    assert np.array_equal(r1.T, r2.T)
    assert np.array_equal(r1.S_se, r2.S_se)


def test_different_seeds_give_different_but_close_estimates():
    r1 = analyze_sensitivity(
        ishigami, 2**14, ishigami_bounds(), seed=1, n_bootstrap=20
    )
    r2 = analyze_sensitivity(
        ishigami, 2**14, ishigami_bounds(), seed=2, n_bootstrap=20
    )
    assert not np.array_equal(r1.S, r2.S)
    assert np.all(np.abs(r1.S - r2.S) < 0.1)


def test_total_ge_first_order_holds_across_seeds():
    """Across four independent seeds the property T_i >= S_i must hold to
    within estimator noise (3-sigma margin at N = 2**15 is ~0.04)."""
    for seed in (11, 22, 33, 44):
        r = analyze_sensitivity(
            ishigami, 2**15, ishigami_bounds(), seed=seed, n_bootstrap=10
        )
        assert np.all(r.T >= r.S - 0.04), f"violated for seed {seed}"


def test_index_errors_helper(result, analytic):
    errs = index_errors(result.S, result.T, analytic.S, analytic.T)
    assert errs["max_S_abs"] == float(np.max(np.abs(result.S - np.array(analytic.S))))
    assert errs["max_T_abs"] <= TOL_INDEX
    with pytest.raises(ValueError, match="matching shapes"):
        index_errors(result.S, result.T, [0.1, 0.2], analytic.T)
