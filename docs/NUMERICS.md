# Numerical analysis

This document expands the error study summarised in `results/RESULTS.md`.
Every number quoted here is read from a file in `results/`; the tables there are
the authoritative source.

## 1. Basis convention and the PySCF reconciliation

cuERI writes a primitive Cartesian Gaussian as

```
g_p(r) = N(L, alpha_p) (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-alpha_p |r-A|^2)
N(L, alpha) = (2 alpha/pi)^(3/4) (4 alpha)^(L/2) / sqrt((2L-1)!!)
```

and folds `N(L, alpha_p)` into the contraction coefficient.  Each Cartesian
component of each contraction is then rescaled so that its self-overlap is
exactly one.  This is the usual "unit-normalised Cartesian Gaussian" basis.

PySCF with `cart=True` uses a different convention for d shells: the contraction
is normalised for the `(l,0,0)` component and the other five components inherit
that scale, so `diag(S_pyscf)` is `(2.5133, 0.8378, 0.8378, 2.5133, 0.8378,
2.5133)` for a d shell.  The two bases differ only by a diagonal matrix,

```
D_mu = sqrt(S_pyscf[mu,mu])      (our diagonal is 1),
S_pyscf = D S_ours D
( mu nu | la si )_pyscf = D_mu D_nu D_la D_si ( mu nu | la si )_ours
```

so `scripts/validate_vs_pyscf.py` applies this diagonal similarity transform
before every comparison.  The RHF closed loop needs no reconciliation: it uses
our own `S`, `T`, `V` and ERI exclusively, and the energy is invariant under the
similarity transform.

## 2. Boys function

`F_n(T) = int_0^1 t^(2n) exp(-T t^2) dt` is evaluated in three branches.

| branch | range | method |
|---|---|---|
| Taylor | `T < 1e-8` | alternating series in T; harmless because the leading term is 1/(2n+1) |
| downward recursion | `1e-8 <= T < 30` | seed `F_N`, `N = nmax + 8`, from the positive incomplete-gamma series, then `F_n = (2T F_{n+1} + e^-T)/(2n+1)` |
| erf / asymptotic | `T >= 30` | `F_0 = sqrt(pi/(4T)) erf(sqrt(T))`, then upward recursion |

The dangerous branch is the middle one.  Naive **upward** recursion

```
F_{n+1} = ((2n+1) F_n - exp(-T)) / (2T)
```

divides by `2T`.  For small T the subtraction `(2n+1)F_n - exp(-T)` cancels
catastrophically (both terms approach 1) and the `1/(2T)` factor amplifies the
lost digits by about `(1/(2T))^n`, so the result for l = 2 shell quartets (n up
to 8) is destroyed.  Downward recursion multiplies by `2T` instead: it is
contractive and damps seed errors.  The positive-term seed series

```
F_N(T) = 1/2 exp(-T) sum_k T^k / prod_{i=0..k} (N + i + 1/2)
```

has no cancellation at all, which is why the downward branch is accurate for the
whole `1e-8 <= T < 30` range.

`scripts/boys_corner_cases.py` sweeps T = 1e-16 .. 1e3 and n = 0..12.  The
measured worst relative error is 4.94e-14 at T = 8.06e-15, n = 10 (small-T
Taylor branch); the downward and erf branches are at or below 1e-14.  The
SciPy reference (`scipy.special.gammainc` and `gamma`) is exact to double
precision for this range.

## 3. McMurchie-Davidson versus Rys

The MD path expands each Gaussian product in Hermite Gaussians and contracts the
result with the Hermite Coulomb auxiliary `R_t`.  The Rys path rewrites the same
sum as a one-dimensional integral over `t in [0,1]` with weight `exp(-T t^2)`
and integrates it with a Gauss rule whose moments are the Boys values
`F_0..F_{2n-1}`; the roots and weights are generated on the fly by a
Gram-Schmidt/Stieltjes step followed by a Jacobi diagonalisation of the resulting
tridiagonal matrix (Golub-Welsch).  For l <= 2 the rule needs at most five
roots.

The two algorithms share only the Hermite E coefficients and the Boys function;
the summation and quadrature are independent.  `tests/test_eri.cpp` reports the
maximum absolute difference per shell class for H2O/STO-3G, H2O/6-31G,
H2O/6-31G\* and CH4/6-31G.  The measured maximum is 4.4e-16, i.e. round-off.

The Rys rule is built from moments normalised by `S = max(T,1)`:
`mu_k = S^k F_k(T)/F_0(T)`.  Without this scaling the Hankel system becomes
catastrophically ill-conditioned for large T (the moments span hundreds of
orders of magnitude) even though the final rule is well conditioned.

## 4. Screening bound

The Schwarz bound is `Q_pq = sqrt(max_{mu in p, nu in q} |(mu nu|mu nu)|)`.
By the Cauchy-Schwarz inequality for the positive-semidefinite four-index
supermatrix,

```
|(mu nu | la si)| <= Q_pq Q_rs   for mu,nu in p,q and la,si in r,s.
```

Every tensor element belongs to exactly one shell quartet, so skipping a quartet
with `Q_pq Q_rs < tol` changes each of its elements by less than `tol`.  The
screening study (`results/screening.csv`) verifies this: for benzene/6-31G the
maximum deviation is 9.50e-9 at tol = 1e-8 with 11.0% of canonical quartets
skipped.  For the compact H2O/6-31G\* system no quartet is below any of the
tested thresholds, and the deviation is exactly zero.

## 5. Precision: fp32 versus fp64

The fp32 path evaluates the Boys function and the exponential prefactors in
double precision (they are scalar setup), then performs every contraction,
Hermite coefficient, Coulomb auxiliary and accumulation in single precision.
That is what makes the fp32 path a genuine single-precision study rather than a
double-precision kernel with a float output.  Switching the contraction
accumulation to double would move the fp32 results toward the fp64 ones roughly
in proportion to the number of terms in the contraction sum, because the error
is dominated by the accumulated rounding of `sum_p c_p c_q (...)`, not by the
Boys values or the exponentials.

Measured against PySCF, the fp32 ERI error scales with the tensor magnitude:
the largest relative error is `7.92e-4` (`pd-sd`, benzene/6-31G\*) and the
largest scaled error is `1.91e-6`; for the `dd-dd` class, where the largest tensor
elements live, the relative error is `3.07e-4`.  fp64 stays at or below
`5.41e-14` scaled.

`scripts/precision_error_growth.py` pushes two conditioning knobs:

* near-degenerate exponents (ratio `1 + delta`, delta down to 1e-10) on s, p and
  d shells of a two-centre H/He system;
* near-coincident centres (separation down to 1e-10 bohr) for H2/STO-3G.

The measured fp32-vs-fp64 error changes by at most a factor of about 2 across
ten decades of delta, and not at all across ten decades of separation.  This is a
property of the Gaussian product formulation: the product centre and the Boys
argument stay bounded as the centres coincide, so there is no cancellation
catastrophe to amplify.  The amplification column in
`results/precision_error_growth.csv` records this per class.

## 6. Validation method

Three independent references are used:

1. `tests/test_boys.cpp` against a SciPy incomplete-gamma table
   (`data/ref/boys_reference.csv`).
2. `tests/test_one_electron.cpp` against `src/reference/ref_one_electron.cpp`,
   which uses the Obara-Saika overlap recursion, the analytic gradient kinetic
   formula and an incomplete-gamma Boys function, sharing no integral code with
   the production MD path.
3. `scripts/validate_vs_pyscf.py` against PySCF/libcint for nine
   molecule/basis combinations, per shell class and per precision.

The closed loop is `scripts/rhf_closed_loop.py`: a NumPy RHF SCF built only from
cuERI integrals.  The converged energies agree with PySCF RHF to 1.24e-11 Ha
(H2O/STO-3G) and 1.46e-11 Ha (H2O/6-31G).
