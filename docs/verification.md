# Verification evidence

Run `python -m pytest tests -q`: **88 collected cases pass**. Regenerate
all data and figures with `python examples/run_verification.py`. The
largest test designs use N=2^17; default result figures extend to N=2^18.
Local versions: Python 3.13.5, NumPy 2.1.3, pytest 8.3.4. CI uses Ubuntu
and Python 3.12. Seeds reproduce sampling streams; last-bit agreement
across different libraries/hardware is not guaranteed.

## What is checked

| Test module | Independent evidence |
| --- | --- |
| `test_ishigami_analytic.py` | Hand function values, published rounded indices, decomposition identities and trapezoidal conditional-moment integration within 0.2% |
| `test_ishigami_saltelli.py` | Fixed-seed estimates vs analytic values, bootstrap scale, interactions, variance/mean, repeated seeds and cross-seed spread |
| `test_additive_anchor.py` | Exact S=T=c²/sum(c²), no interaction, variance=sum(c²)/3 and Monte Carlo comparisons |
| `test_estimators.py` | NumPy mean/variance anchors, split/combined API equivalence, shape and constant-output failures |
| `test_sampling.py` | Crossed columns, independent seeded draws, bounds, shapes and unchanged input matrices |
| `test_validation.py` | Documented driver/model/seed/bounds/size/output failures |
| `test_affine_invariance.py` | Ten offset/scale/bootstrap/covariance/degeneracy and numeric-range regressions |

## Statistical thresholds and actual test measurements

At N=131,072, seed 20261018 and B=200, max absolute index errors are
**0.003144 (S)** and **0.003532 (T)**. The largest bootstrap SE is
0.003755; three SEs remain
below the suite's 0.02 allowance. Estimated mean is 3.488547
versus 3.5 and variance 13.805713 versus 13.844588.

The additive test fixture uses N=131,072, seed 424242 and B=100.
Its maximum S error is
0.001706, maximum T
error 0.002515; both
remain inside the required 0.01. Standard errors and percentile intervals
for these exact fixtures are saved in `run_summary.json`.

These deterministic fixed-seed checks do not establish universal
coverage. Finite estimates may violate S<=T or [0,1] within sampling
spread. Tolerances tied to SEs are useful diagnostics, not proof that
all errors are normally distributed or that the estimator is unbiased.

## The reproduced offset failure

The same two-input additive model and N=8,192 design (seed 10) has exact
indices (0.5,0.5). Adding a constant must leave them unchanged. With an
offset of 1,000, the previous raw-moment path returned S1=-25.3214; at
1e8 it returned -854,719 and variance 2 instead of about 0.6618755.
The new common-centered path's maximum index drift over the six offsets
is **1.83e-10**; the offset-1,000 drift is about 1e-15. The remaining
high-offset drift is consistent with quantization of the provided arrays.

Tests exercise every public index function and the bootstrap's S/T,
standard errors and percentile intervals. A small integer design shifted
by 2^40 checks exact hand covariance values. Positive and negative output
scale changes are covered. Estimates are never clipped to hide errors.
The old formula is retained only in the example benchmark comparison.

## Published results and reproducibility

The default 44-size sweep uses N=128..262,144 and seed 20261018 with B=100;
one generator drives independent designs in sequence. This is not a
nested sample sequence, so errors need not decrease at every frame.

- At final N, maximum errors are 0.004348 (S) and
  0.003190 (T); variance 13.842700.
- All six errors first fall below 0.02 at N=7,558.
  They meet it at **every later recorded point** only from
  N=15,361. No monotonic claim is made.
- `convergence.gif`: 44 distinct computed frames, 1,100×600 px, white/no
  grids; first-order and total effects in separate panels, both with ±2 SE
  whiskers. GIF timing is 140 ms/frame (nominal 7 fps).
- `convergence_sweep.csv`: all 44 rows with estimates, SEs, separate
  percentile intervals, pooled mean/variance, model cost and runtime.
- `affine_invariance.png/.csv`: prior and centered results under identical
  output offsets; the right plot's display floor is 1e-16, raw zeros saved.
- `ishigami_surface.png`: response slice at x3=0; not a sensitivity map.
- `run_summary.json`: exact result/anchor/test-fixture/API values, seeds,
  numerical versions and timing (46.9 s locally).

Bootstrap reuse costs no additional calls to the user model. The analytic
models are cheap; a large CFD simulation may make (k+2)N calls impractical.
The library does not address correlated inputs, physical model error,
surrogate acceleration or guaranteed interval coverage. At least two
nondegenerate bootstrap replicates are required for an SE; skipped
zero-variance replicates do not count as successful observations.
