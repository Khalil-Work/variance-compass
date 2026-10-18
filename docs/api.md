# API reference

All public functions are importable from the package root:

```python
import variance_compass as vc
```

The theory behind every object below is in [model.md](model.md);
per-test evidence is in [verification.md](verification.md).

---

## Test models

### `ishigami(X, a=7.0, b=0.1)`

Evaluate the Ishigami function
`sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)` on an `(n, 3)` array.
Returns shape `(n,)`. Raises `ValueError` for non-`(n, 3)` shapes,
non-finite values, or non-finite `a`/`b`.

### `ishigami_analytic(a=7.0, b=0.1) -> IshigamiAnalytic`

Exact decomposition: fields `variance`, `first_order_variances`
`(V1, V2, V3)`, `interaction_variance` (`V13`), `S`, `T` (tuples of
three floats), plus `to_dict()`.

### `ishigami_bounds()`

The `(3, 2)` bounds array `[(-pi, pi)] * 3`.

### `additive_model(coeffs)`

Factory returning the vectorised model `f(X) = X @ coeffs` on
`U(-1, 1)^k` inputs, with shape/finiteness validation.

### `additive_analytic(coeffs) -> AnalyticIndices`

Exact `S = T = c_j^2 / sum(c^2)` and `variance = sum(c^2)/3`.

### `additive_bounds(dim)`

The `(dim, 2)` bounds array `[(-1, 1)] * dim`.

---

## Sampling

### `saltelli_matrices(n, dim, rng)`

Build the Saltelli design: returns `(A, B, crossed)` where `A`, `B`
are independent `(n, dim)` matrices of i.i.d. `U(0, 1)` draws and
`crossed[i]` is `A` with column `i` taken from `B` (fresh copies; `A`
is never mutated). `rng` must be a `numpy.random.Generator` (e.g.
`np.random.default_rng(seed)`).

### `to_physical(u, bounds)`

Inverse-transform `U(0,1)` draws `(n, dim)` to uniform marginals on
`bounds` `(dim, 2)`: returns `low + (high - low) * u`.

---

## Estimators

### `sample_mean(x)`

Arithmetic mean of a finite nonempty 1-D array. Uses a shifted origin
before summation; tested against `numpy.mean` on identical samples.

### `sample_variance(x, ddof=1)`

`sum((x - mean)^2) / (n - ddof)`; tested to agree with
`numpy.var(x, ddof=ddof)` on identical samples.

### `first_order_indices(f_a, f_b, f_ab)` / `total_effect_indices(f_a, f_b, f_ab)`

Saltelli estimates from design outputs: `f_a`, `f_b` are length-`n`
arrays, `f_ab` is the `(dim, n)` stack of crossed outputs. Return
`(dim,)` arrays.

### `saltelli_indices(f_a, f_b, f_ab)`

Both families in one pass; returns `(S, T, mean, variance)`. All output
families use one common pooled A/B center. Variance is the centered pooled
population estimate (denominator 2n). Results are not clipped. Nonfinite,
wrong-shape or empty-dimension data raises ValueError; zero/under-resolved
variance also raises ValueError. Nonrepresentable centered moments/indices
raises FloatingPointError. Offsets and nonzero scales cancel up to rounding
already present in the supplied data.

---

## Analysis driver

### `analyze_sensitivity(model, n, bounds, *, seed=None, rng=None, n_bootstrap=100) -> SensitivityResult`

Full analysis at base sample size `n`:

* draws the `(k+2)n` Saltelli design (`k = number of bounds rows`),
* evaluates `model` (a vectorised `(n, k) -> (n,)` callable),
* computes `S`, `T` plus bootstrap standard errors and 95% percentile
  intervals. Every replicate reuses `saltelli_indices`; fewer than two
  nondegenerate bootstrap replicates raises ValueError.

Pass either an integer `seed` (recommended; everything is then exactly
reproducible) or a pre-built Generator `rng` (used internally to thread
one stream through a sweep). The result carries:

| Field | Meaning |
|---|---|
| `S`, `T` | index estimates, shape `(k,)` |
| `S_se`, `T_se` | bootstrap standard errors |
| `S_ci`, `T_ci` | bootstrap 2.5%/97.5% intervals, shape `(k, 2)` |
| `mean`, `variance` | pooled output mean and variance estimates |
| `n`, `dim`, `seed`, `n_bootstrap` | run parameters |
| `runtime_s` | wall-clock seconds |
| `to_dict()` | JSON-friendly serialisation |

### `convergence_sweep(model, bounds, n_values, *, seed, n_bootstrap=100) -> SweepResult`

Runs `analyze_sensitivity` over increasing sample sizes (duplicates in
`n_values` are removed and sorted). A single Generator seeded with
`seed` is threaded through the whole sweep, so the entire sweep is
deterministic. The result exposes `n_values`, per-size `results` and
the convenience matrices `S_matrix`, `T_matrix`, `S_se_matrix`,
`T_se_matrix` (rows follow `n_values`).

### `index_errors(S, T, S_ref, T_ref) -> dict`

Absolute errors against reference indices: `S_abs`, `T_abs` arrays and
`max_S_abs`, `max_T_abs` scalars. Raises `ValueError` on shape
mismatch.

---

## Validation guarantees

Drivers validate model callability, integer sample/bootstrap sizes,
finite ordered bounds, generator/seed types and model-output shape/finiteness.
Estimators validate their documented arrays and positive variance. Model
functions check their shapes and parameters. `index_errors` checks matching
shapes. See tests for each supported failure contract; these checks do not
establish scientific correctness of user-supplied models.
