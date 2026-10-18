"""A model's sensitivity must not depend on its output zero or unit scale."""
import numpy as np
import pytest
from variance_compass import analyze_sensitivity, additive_model, additive_bounds
from variance_compass.estimators import (saltelli_indices, first_order_indices,
                                        total_effect_indices, sample_variance)
from variance_compass.sampling import saltelli_matrices


def outputs():
    a,b,crossed = saltelli_matrices(8192, 2, np.random.default_rng(10))
    model = additive_model((1.,1.))
    fa,fb = model(2*a-1),model(2*b-1)
    fab = np.vstack([model(2*x-1) for x in crossed])
    return fa,fb,fab


@pytest.mark.parametrize("scale,offset,tol", [(1.,1000.,2e-12), (1.,1e8,2e-8),
    (-3.,1000.,2e-12), (1e-6,0.,2e-12), (1e6,1e8,2e-12)])
def test_all_index_apis_are_invariant_to_affine_output_changes(scale, offset, tol):
    fa,fb,fab = outputs()
    base_s,base_t,mean,var = saltelli_indices(fa,fb,fab)
    shifted = tuple(scale*x+offset for x in (fa,fb,fab))
    s,t,m,v = saltelli_indices(*shifted)
    np.testing.assert_allclose(s,base_s,rtol=0,atol=tol)
    np.testing.assert_allclose(t,base_t,rtol=0,atol=tol)
    np.testing.assert_array_equal(first_order_indices(*shifted),s)
    np.testing.assert_array_equal(total_effect_indices(*shifted),t)
    assert m == pytest.approx(scale*mean+offset, rel=0, abs=max(tol,abs(offset)*2e-16))
    assert v == pytest.approx(var*scale**2,rel=max(tol,2e-14))
    np.testing.assert_allclose(s,[.5,.5],rtol=0,atol=.03)
    np.testing.assert_allclose(t,[.5,.5],rtol=0,atol=.03)


@pytest.mark.parametrize("offset,tolerance", [(1000.,2e-12), (1e8,2e-8)])
def test_bootstrap_standard_errors_and_intervals_share_translation_invariance(offset,tolerance):
    model = additive_model((1.,1.))
    base = analyze_sensitivity(model,2048,additive_bounds(2),seed=71,n_bootstrap=30)
    moved = analyze_sensitivity(lambda x:model(x)+offset,2048,additive_bounds(2),seed=71,n_bootstrap=30)
    for field in ("S","T","S_se","T_se","S_ci","T_ci"):
        np.testing.assert_allclose(getattr(moved,field),getattr(base,field),rtol=0,atol=tolerance)


def test_centering_uses_hand_covariances_without_raw_large_moments():
    # Small integer variations remain exactly represented after adding 2**40.
    fa = np.array([-1.,1.,-1.,1.])
    fb = np.array([-1.,-1.,1.,1.])
    fab = np.array([fb,fa])
    s,t,_,v = saltelli_indices(fa+2**40,fb+2**40,fab+2**40)
    np.testing.assert_array_equal(s,[1.,0.])
    np.testing.assert_array_equal(t,[1.,0.])
    assert v == 1.
    assert sample_variance(fa+2**40,ddof=0) == 1.


def test_a_single_nondegenerate_bootstrap_replicate_cannot_produce_an_se():
    from variance_compass.sensitivity import _bootstrap_indices
    with pytest.raises(ValueError,match="two nondegenerate"):
        _bootstrap_indices(np.array([0.,1.]),np.array([0.,1.]),np.array([[0.,1.]]),2,np.random.default_rng(0))


def test_no_input_dimension_and_overflowing_moments_raise():
    with pytest.raises(ValueError,match="dimension"):
        saltelli_indices([0.,1.],[0.,1.],np.empty((0,2)))
    with pytest.raises(FloatingPointError,match="rescale"):
        saltelli_indices([-1e308,1e308],[-1e308,1e308],[[-1e308,1e308]])
