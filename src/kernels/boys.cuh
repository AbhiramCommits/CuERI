/**
 * @file boys.cuh
 * @brief Boys function F_n(T) = int_0^1 t^(2n) exp(-T t^2) dt.
 *
 * Three numerical branches are used, with named crossover constants:
 *
 *  1. T < CUEST_BOYS_SMALL_T  -- Taylor series
 *
 *         F_n(T) = sum_{k>=0} (-T)^k / (k! (2n + 2k + 1)).
 *
 *     Every term is a small correction to 1/(2n+1); the alternating series
 *     loses no significant digits for T < 1e-8.
 *
 *  2. CUEST_BOYS_SMALL_T <= T < CUEST_BOYS_MID_MAX -- downward recursion
 *
 *         F_n(T) = [2T F_{n+1}(T) + exp(-T)] / (2n+1),
 *
 *     seeded by an accurate F_N with N = nmax + CUEST_BOYS_NMARGIN.  The seed
 *     is evaluated with the positive-term (incomplete-gamma) series
 *
 *         F_N(T) = 1/2 exp(-T) sum_k T^k / prod_{i=0..k} (N + i + 1/2),
 *
 *     which has no cancellation at all.  WHY NOT UPWARD RECURSION HERE: the
 *     upward form F_{n+1} = [(2n+1) F_n - exp(-T)] / (2T) divides by 2T.  As
 *     T -> 0 the factor 1/(2T) amplifies the (unavoidable) cancellation
 *     between (2n+1)F_n and exp(-T) at every step, and for large n it amplifies
 *     the error by roughly (1/(2T))^(n), destroying the result.  Downward
 *     recursion multiplies by 2T instead and damps errors, so it is safe in
 *     exactly the region where upward recursion is not.
 *
 *  3. T >= CUEST_BOYS_MID_MAX -- erf / asymptotic branch
 *
 *         F_0(T) = sqrt(pi/(4T)) erf(sqrt(T)),
 *
 *     followed by upward recursion, which is strongly contractive for T >= 30
 *     (amplification 1/(2T) <= 1/60 per step).  In this regime the result is
 *     identical to the large-T asymptotic series
 *
 *         F_n(T) ~ (2n-1)!! / (2T)^n * sqrt(pi/(4T))
 *
 *     to better than one part in 10^13, and the erf form avoids the alternating
 *     cancellation that a direct asymptotic evaluation suffers at n = 12.
 *
 * T = 0 is returned exactly as 1/(2n+1).
 */
#ifndef CUEST_BOYS_CUH
#define CUEST_BOYS_CUH

#include <math.h>

#include "../cuest_macros.h"

/** @brief Below this T the Taylor series branch is used. */
constexpr double CUEST_BOYS_SMALL_T = 1e-8;
/** @brief At and above this T the erf/asymptotic branch is used. */
constexpr double CUEST_BOYS_MID_MAX = 30.0;
/** @brief Pi, duplicated locally so boys.cuh has no include cycle. */
constexpr double CUEST_BOYS_PI = 3.14159265358979323846264338327950288;

/** @brief Extra orders used to seed the downward recursion. */
constexpr int CUEST_BOYS_NMARGIN = 8;

/**
 * @brief Boys function of order @p n for a single argument, double precision.
 */
CUEST_HD inline double cuest_boys(int n, double Tval) {
  if (Tval <= 0.0) return 1.0 / (2.0 * n + 1.0);

  if (Tval < CUEST_BOYS_SMALL_T) {
    double s = 1.0 / (2.0 * n + 1.0);
    double term = 1.0;
    for (int k = 1; k < 64; ++k) {
      term *= -Tval / static_cast<double>(k);
      const double add = term / (2.0 * n + 2.0 * k + 1.0);
      s += add;
      if (fabs(add) <= 1e-20 * fabs(s)) break;
    }
    return s;
  }

  if (Tval < CUEST_BOYS_MID_MAX) {
    const int N = n + CUEST_BOYS_NMARGIN;
    double val = 0.0;
    double prod = static_cast<double>(N) + 0.5;
    double tk = 1.0;
    for (int k = 0; k < 1024; ++k) {
      if (k > 0) {
        prod *= static_cast<double>(N + k) + 0.5;
        tk *= Tval;
      }
      const double add = tk / prod;
      val += add;
      if (k > 2 && add <= 1e-18 * val) break;
    }
    const double e = exp(-Tval);
    double f[CUEST_MAX_BOYS + CUEST_BOYS_NMARGIN + 2];
    f[N] = 0.5 * e * val;
    for (int m = N - 1; m >= 0; --m) f[m] = (2.0 * Tval * f[m + 1] + e) / (2.0 * m + 1.0);
    return f[n];
  }

  /* Large-T branch: erf seed, upward recursion. */
  {
    const double e = exp(-Tval);
    double f[CUEST_MAX_BOYS + 2];
    f[0] = 0.5 * sqrt(CUEST_BOYS_PI / Tval) * erf(sqrt(Tval));
    const int nmax = n;
    for (int m = 0; m < nmax; ++m) f[m + 1] = ((2.0 * m + 1.0) * f[m] - e) / (2.0 * Tval);
    return f[n];
  }
}

/**
 * @brief Boys function vector F_0..F_nmax at one argument.
 *
 * @param[in]  nmax  Highest order (<= CUEST_MAX_BOYS).
 * @param[in]  Tval  Argument T >= 0.
 * @param[out] out   Array of nmax+1 values.
 */
CUEST_HD inline void cuest_boys_vec(int nmax, double Tval, double *out) {
  if (nmax < 0) return;
  if (Tval <= 0.0) {
    for (int n = 0; n <= nmax; ++n) out[n] = 1.0 / (2.0 * n + 1.0);
    return;
  }
  if (Tval < CUEST_BOYS_SMALL_T) {
    for (int n = 0; n <= nmax; ++n) out[n] = cuest_boys(n, Tval);
    return;
  }
  if (Tval < CUEST_BOYS_MID_MAX) {
    const int N = nmax + CUEST_BOYS_NMARGIN;
    double val = 0.0;
    double prod = static_cast<double>(N) + 0.5;
    double tk = 1.0;
    for (int k = 0; k < 1024; ++k) {
      if (k > 0) {
        prod *= static_cast<double>(N + k) + 0.5;
        tk *= Tval;
      }
      const double add = tk / prod;
      val += add;
      if (k > 2 && add <= 1e-18 * val) break;
    }
    const double e = exp(-Tval);
    double f[CUEST_MAX_BOYS + CUEST_BOYS_NMARGIN + 2];
    f[N] = 0.5 * e * val;
    for (int m = N - 1; m >= 0; --m) f[m] = (2.0 * Tval * f[m + 1] + e) / (2.0 * m + 1.0);
    for (int n = 0; n <= nmax; ++n) out[n] = f[n];
    return;
  }
  const double e = exp(-Tval);
  out[0] = 0.5 * sqrt(CUEST_BOYS_PI / Tval) * erf(sqrt(Tval));
  for (int m = 0; m < nmax; ++m) out[m + 1] = ((2.0 * m + 1.0) * out[m] - e) / (2.0 * Tval);
}

#endif /* CUEST_BOYS_CUH */
