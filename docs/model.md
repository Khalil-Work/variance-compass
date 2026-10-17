# Model: variance decomposition, the Saltelli scheme, and analytic ground truth

This document states the mathematics implemented by `variance_compass`:
the variance-based (Sobol') sensitivity framework, the Saltelli Monte
Carlo estimators as implemented here, and the two analytic test models
(the Ishigami function and the additive anchor) with their closed-form
indices. Everything below is derived from first principles; no
sensitivity-analysis library is used anywhere in the package.

References (by title) are listed at the end.

## Symbols

| Symbol | Meaning |
|---|---|
| `k` | number of independent inputs |
| `X = (X_1, ..., X_k)` | vector of independent random inputs |
| `f(X)` | scalar model output |
| `f_0 = E[f]` | output mean |
| `V = Var(f)` | total output variance |
| `V_i = Var( E[f \| X_i] )` | first-order (main-effect) variance of input `i` |
| `V_{~i} = E[ Var(f \| X_{~i}) ]` | total contribution of all terms involving `X_i` |
| `S_i = V_i / V` | first-order Sobol' index |
| `T_i = V_{~i} / V` | total-effect Sobol' index |
| `A, B` | two independent `n x k` matrices of i.i.d. `U(0,1)` draws |
| `A_B^(i)` | matrix `A` with column `i` replaced by column `i` of `B` |
| `f_a, f_b, f_ab[i]` | model outputs evaluated on `A`, `B`, `A_B^(i)` |
| `n` | base sample size; a full design costs `n * (k + 2)` model evaluations |
| `N` | used interchangeably with `n` in the plots for the base sample size |

## Variance decomposition (ANOVA / HDMR)

For independent inputs, the function admits a unique orthogonal
decomposition (the HDMR / ANOVA expansion, see Sobol'):

```
f(X) = f_0 + sum_i f_i(X_i) + sum_{i<j} f_{ij}(X_i, X_j) + ...,
```

where each term has zero mean and is orthogonal to all others, e.g.
`f_i(X_i) = E[f | X_i] - f_0` and
`f_{ij}(X_i, X_j) = E[f | X_i, X_j] - f_i - f_j - f_0`. Orthogonality
makes the variances additive:

```
V = sum_i V_i + sum_{i<j} V_{ij} + ...   (over all interaction orders),
```

with `V_i = Var(E[f | X_i])` and, for any subset `u` of inputs,
`V_u = Var( E[f | X_u] - sum_{v proper subset of u} f_v ] )`. Two
conditional-variance identities (the law of total variance) define the
two index families used in this package:

```
S_i = Var( E[f | X_i] ) / V                       (first-order index)
T_i = E[ Var(f | X_{~i} ] ) / V = 1 - Var( E[f | X_{~i}] ) / V
                                                    (total-effect index)
```

where `X_{~i}` denotes all inputs except `X_i`. Consequences used by
the verification suite:

* `0 <= S_i, T_i <= 1`;
* `S_i <= T_i` for every input (in expectation; Monte Carlo noise can
  dent this by a few standard errors for inputs with no interactions);
* `sum_i S_i <= 1`, with the deficit `1 - sum_i S_i` equal to the total
  fraction of variance carried by interactions of order >= 2;
* `sum_i T_i >= 1`, again with equality only for additive models.

## The Saltelli sampling scheme, as implemented

The design (Saltelli et al., 2010) draws two independent base matrices
`A` and `B` of shape `(n, k)` with i.i.d. `U(0, 1)` entries, then forms
`k` crossed matrices `A_B^(i)`. Uniform marginals on the physical
domain `[low_j, high_j]` are produced by inverse transform,
`x = low + (high - low) * u` (`variance_compass.sampling.to_physical`).
Total cost: `n * (k + 2)` model evaluations, and both index families
come out of the same design.

The estimators are the correlation ("matrix") form derived directly
from the design structure. Write

```
f0_hat  = ( mean(f_a) + mean(f_b) ) / 2
V_hat   = ( mean(f_a^2) + mean(f_b^2) ) / 2 - f0_hat^2
```

(the symmetric pooled variance estimate). Then

```
S_hat_i = ( mean_j( f_b[j] * f_ab[i][j] ) - f0_hat^2 ) / V_hat
T_hat_i = 1 - ( mean_j( f_a[j] * f_ab[i][j] ) - f0_hat^2 ) / V_hat
```

**Why these work.** Row `j` of `B` and row `j` of `A_B^(i)` share
exactly one input: the transplanted `X_i = B[j, i]`. All other inputs
are independent copies, so

```
E[ f(B_j) * f(A_B^(i)_j) ] = f_0^2 + V_i,
```

because conditioning on the shared `X_i` gives
`E[f | X_i] * E[f | X_i]` plus zero cross terms; expanding
`E[ E[f|X_i]^2 ] = f_0^2 + V_i`. Similarly row `j` of `A` and row `j`
of `A_B^(i)` share every input *except* `X_i` (all of `A`'s columns
other than `i`), so

```
E[ f(A_j) * f(A_B^(i)_j) ] = f_0^2 + V - V_{~i},
```

since the two outputs are i.i.d. conditionally on `X_{~i}` and the
conditional variance given `X_{~i}` averages to `V_{~i}`. Subtracting
`f0_hat^2` and dividing by `V_hat` turns these into the estimators
above. `V_hat` carries an `O(1/n)` bias from the estimated mean; the
bootstrap described next captures the resulting estimator spread
empirically.

**Bootstrap uncertainty.** The `n` design rows are resampled with
replacement (jointly for `f_a`, `f_b`, `f_ab[i]`), and the *same*
estimators (including recomputation of `f0_hat` and `V_hat` on the
resampled rows) are applied to each replicate. The bootstrap standard
error is the replicate standard deviation (ddof = 1) and the 95%
confidence interval the 2.5% / 97.5% percentiles. Monte Carlo standard
errors shrink like `n^(-1/2)`: doubling the base sample size shrinks
the index standard error by roughly `sqrt(2)`.

## Analytic ground truth 1: the Ishigami function

```
f(x1, x2, x3) = sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1),
x_i ~ U(-pi, pi) i.i.d.,   a = 7,   b = 0.1     (standard published values).
```

### Derivation of the exact indices

Uniform moments on `(-pi, pi)` (density `1/(2pi)`, all integrals
symmetric about 0):

```
E[sin(x)] = 0        E[sin(x)^2] = 1/2        E[sin(x)^4] = 3/8
E[x^4] = pi^4 / 5    E[x^8] = pi^8 / 9        (by symmetry: (1/pi) * int_0^pi x^m dx)
```

**Mean.** Only the `a * sin(x2)^2` term has non-zero mean:

```
f_0 = E[f] = a / 2.
```

**Second moment.** Expanding the square gives three squared terms and
three cross terms. Every cross term contains `sin(x1)` (mean zero,
independent of the other factors) or `sin(x2)^2` against an independent
zero-mean factor, so all cross terms vanish:

```
E[f^2] = E[sin^2(x1)] + a^2 E[sin^4(x2)] + b^2 E[x3^8] E[sin^2(x1)]
         + 2 b E[x3^4] E[sin^2(x1)]
       = 1/2 + 3 a^2 / 8 + b^2 pi^8 / 18 + b pi^4 / 5.
```

```
V = E[f^2] - f_0^2 = 1/2 + a^2 / 8 + b pi^4 / 5 + b^2 pi^8 / 18.
```

**First-order variances** (from the conditional means):

```
E[f | x1] = (1 + b E[x3^4]) sin(x1) + a/2 = (1 + b pi^4/5) sin(x1) + a/2
    =>  V1 = (1 + b pi^4 / 5)^2 * Var(sin(x1)) = (1/2) (1 + b pi^4/5)^2

E[f | x2] = a sin^2(x2) + ... mean-zero x1 terms ... = a (sin^2(x2))
    =>  V2 = a^2 Var(sin^2(x2)) = a^2 (3/8 - 1/4) = a^2 / 8

E[f | x3] = a/2 + E[sin(x1)] (1 + b x3^4) = a/2
    =>  V3 = 0.
```

**The interaction.** `x1` and `x3` only appear through the product
`x3^4 sin(x1)`, so the only non-zero second-order HDMR term is

```
f_13(x1, x3) = b (x3^4 - pi^4/5) sin(x1)
    =>  V13 = b^2 Var(x3^4) * E[sin^2(x1)]
            = b^2 (pi^8/9 - pi^8/25) / 2
            = 8 b^2 pi^8 / 225.
```

(Algebraic check of the decomposition identity, exact in floating
point: `(1/2)(1+c)^2 + a^2/8 + 8 b^2 pi^8 / 225` with `c = b pi^4/5`
equals `1/2 + c + c^2/2 + a^2/8 + 8 b^2 pi^8/225`, and
`c^2/2 + 8 b^2 pi^8/225 = b^2 pi^8 (1/50 + 8/225) = b^2 pi^8 / 18`,
recovering `V`.)

**Indices.** `S_i = V_i / V`, `T_i = (V_i + V13 * 1[i in {1,3}]) / V`:

```
S = ( V1/V,          V2/V,   0 )
T = ( (V1+V13)/V,    V2/V,   V13/V )
```

Numeric values for `a = 7`, `b = 0.1`:

| Quantity | Value |
|---|---|
| `V` | 13.8445879407 |
| `V1` | 4.3458880239 |
| `V2` | 6.125 |
| `V3` | 0 |
| `V13` | 3.3736999168 |
| `S` | (0.313905, 0.442411, 0.000000) |
| `T` | (0.557589, 0.442411, 0.243684) |

The published rounded values `S = (0.3139, 0.4424, 0)`,
`T = (0.5576, 0.4424, 0.2437)` (e.g. Saltelli et al., "Global
Sensitivity Analysis: The Primer", and the 2010 total-sensitivity
paper) are matched to within 5e-5. Note the structural fingerprints the
tests exploit: `S3 = 0` while `T3 > 0` (a pure interaction input), and
`1 - (S1+S2+S3) = T3 = V13/V` (the first-order deficit is exactly the
x1-x3 interaction).

## Analytic ground truth 2: the additive anchor

```
f(x) = sum_j c_j x_j ,     x_j ~ U(-1, 1) i.i.d.
```

Because the model is strictly additive with independent inputs, the
HDMR expansion terminates at first order (`V_u = 0` for `|u| >= 2`), so

```
V_j = Var(c_j x_j) = c_j^2 * Var(U(-1,1)) = c_j^2 / 3
V   = sum_j V_j
S_j = T_j = c_j^2 / sum_m c_m^2            (exact, no interactions).
```

This is the exact-regression anchor of the verification suite: any
deviation of the Monte Carlo estimates from `c_j^2 / sum(c^2)` is pure
estimator error, with no analytic ambiguity to blame.

## What the estimators must reproduce

Combining the sections above, the verification targets are:

1. `S_hat, T_hat -> S, T` (Ishigami) as `n` grows, with bootstrap
   standard errors shrinking like `n^(-1/2)` (convergence plot and GIF
   in `docs/results/`);
2. `T_i >= S_i` for all inputs, up to a few standard errors of noise;
3. `sum_i S_hat_i <= 1` with the deficit consistent with `V13 / V`;
4. `S_hat = T_hat = c_j^2 / sum(c^2)` for the additive anchor;
5. exact reproducibility for a documented seed.

## References (by title)

* "Global Sensitivity Analysis: The Primer" -- A. Saltelli, M. Ratto,
  T. Andres, F. Campolongo, J. Cariboni, D. Gatelli, M. Saisana,
  S. Tarantola, Wiley, 2008.
* "Variance based sensitivity analysis of model output. The design and
  estimation of total sensitivity index" -- A. Saltelli, P. Annoni,
  I. Azzini, F. Campolongo, M. Ratto, S. Tarantola, Computer Physics
  Communications 181(2), 2010 (the `(k+2)N` design and the estimators).
* "Sensitivity estimates for nonlinear mathematical models" --
  I. M. Sobol', Mathematical Modelling and Computational Experiments,
  1993; and "Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates" -- I. M. Sobol', Mathematics
  and Computers in Simulation, 2001 (the HDMR/ANOVA decomposition).
* "An importance quantification technique in uncertainty analysis for
  computer models" -- T. Ishigami and T. Homma, Proceedings of the
  First International Symposium on Uncertainty Modelling and Analysis
  (ISUMA), 1990 (the test function).
