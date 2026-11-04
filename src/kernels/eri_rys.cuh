/**
 * @file eri_rys.cuh
 * @brief Primitive two-electron integrals by Rys quadrature.
 *
 * The McMurchie-Davidson sum over Hermite Coulomb functions can be written as a
 * one-dimensional integral over t in [0,1]:
 *
 *   (ab|cd) = C int_0^1 exp(-T t^2) prod_d P_d(t^2) dt,
 *   C = 2 pi^(5/2)/(p q sqrt(p+q)) K1 K2,
 *   T = alpha |P-Q|^2,  alpha = pq/(p+q),
 *
 * where P_d(u) is a polynomial of degree at most the total angular momentum.
 * Rys quadrature is Gauss quadrature for the weight exp(-T t^2) on [0,1]; its
 * moments are exactly the Boys function values F_k(T).  We therefore build the
 * Gauss rule on the fly from the Boys moments (no committed root tables are
 * required, which keeps the fp32/fp64 paths identical) by
 *
 *   1. forming the normalised moments mu_k = S^k F_k(T)/F_0(T), S=max(T,1);
 *   2. orthogonalising the monomials with a Gram-Schmidt/Stieltjes procedure;
 *   3. diagonalising the resulting Jacobi matrix with cyclic Jacobi rotations.
 *
 * The eigenvalues are the Rys roots in the scaled variable w = S u, and the
 * weights follow from the first eigenvector component (Golub-Welsch).  The
 * number of roots is floor(L/2)+1, at most 5 for L <= 8.
 */
#ifndef CUEST_ERI_RYS_CUH
#define CUEST_ERI_RYS_CUH

#include "../cuest_macros.h"
#include "boys.cuh"
#include "md_recurrence.cuh"
#include "shell.cuh"

/** @brief Maximum number of Rys roots for l <= 2 shell quartets. */
constexpr int CUEST_RYS_MAX_ROOTS = 5;

/**
 * @brief Probabilists' Hermite polynomial He_m(x).
 */
CUEST_HD inline double cuest_hermite_he(int m, double x) {
  if (m <= 0) return 1.0;
  double hm2 = 1.0;
  double hm1 = x;
  for (int k = 2; k <= m; ++k) {
    const double h = x * hm1 - static_cast<double>(k - 1) * hm2;
    hm2 = hm1;
    hm1 = h;
  }
  return hm1;
}

/**
 * @brief Cyclic Jacobi diagonalisation of a symmetric n x n matrix.
 *
 * @param[in,out] A     Input matrix; overwritten with the diagonal.
 * @param[in]     n     Matrix order (<= 5).
 * @param[out]    evals Eigenvalues.
 * @param[out]    V     Eigenvector matrix (columns are eigenvectors).
 */
CUEST_HD inline void cuest_jacobi_eigen(double A[CUEST_RYS_MAX_ROOTS][CUEST_RYS_MAX_ROOTS], int n,
                                        double *evals,
                                        double V[CUEST_RYS_MAX_ROOTS][CUEST_RYS_MAX_ROOTS]) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;

  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
    if (off < 1e-30) break;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        if (fabs(A[p][q]) < 1e-300) continue;
        const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        const double sgn = (theta >= 0.0) ? 1.0 : -1.0;
        const double t = sgn / (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < n; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < n; ++i) evals[i] = A[i][i];
}

/**
 * @brief Rys roots and weights from the Boys moments.
 *
 * On output @p u and @p w satisfy sum_k w_k u_k^j = F_j(T)/F_0(T) for
 * j = 0..2n-1, and the integral of a polynomial P(u) against
 * exp(-T t^2) dt (u = t^2) is F_0(T) sum_k w_k P(u_k).
 */
CUEST_HD inline void cuest_rys_rule(double Tval, int n, const double *F, double *u, double *w) {
  const double S = (Tval > 1.0) ? Tval : 1.0;
  double mu[2 * CUEST_RYS_MAX_ROOTS];
  for (int j = 0; j < 2 * n; ++j) mu[j] = cuest_pow_int(S, j) * F[j] / F[0];

  double Q[CUEST_RYS_MAX_ROOTS][CUEST_RYS_MAX_ROOTS];
  for (int k = 0; k < n; ++k) {
    double v[CUEST_RYS_MAX_ROOTS];
    for (int i = 0; i < n; ++i) v[i] = (i == k) ? 1.0 : 0.0;
    for (int q = 0; q < k; ++q) {
      double ip = 0.0;
      for (int i = 0; i < n; ++i) ip += Q[q][i] * mu[k + i];
      for (int i = 0; i < n; ++i) v[i] -= ip * Q[q][i];
    }
    double nn = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) nn += v[i] * v[j] * mu[i + j];
    const double inv = 1.0 / sqrt(nn);
    for (int i = 0; i < n; ++i) Q[k][i] = v[i] * inv;
  }

  double J[CUEST_RYS_MAX_ROOTS][CUEST_RYS_MAX_ROOTS];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) s += Q[i][a] * Q[j][b] * mu[a + b + 1];
      J[i][j] = 0.5 * (s + s);
    }
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double s = 0.5 * (J[i][j] + J[j][i]);
      J[i][j] = s;
      J[j][i] = s;
    }

  double evals[CUEST_RYS_MAX_ROOTS];
  double V[CUEST_RYS_MAX_ROOTS][CUEST_RYS_MAX_ROOTS];
  cuest_jacobi_eigen(J, n, evals, V);

  for (int k = 0; k < n; ++k) {
    double node = evals[k] / S;
    if (node < 0.0) node = 0.0;
    if (node > 1.0) node = 1.0;
    u[k] = node;
    w[k] = V[0][k] * V[0][k];
    /* Weights must be non-negative; numerical noise can make them slightly
       negative only for degenerate rules, clamp to zero. */
    if (w[k] < 0.0) w[k] = 0.0;
  }
}

/**
 * @brief Primitive ERI (ab|cd) by Rys quadrature.
 */
template <typename T>
CUEST_HD T cuest_prim_eri_rys(const int la[3], const int lb[3], const int lc[3], const int ld[3],
                              T a, T b, T c, T d, const T A[3], const T B[3], const T Cc[3],
                              const T Dd[3]) {
  const T p = a + b;
  const T q = c + d;
  T P[3], Q[3];
  for (int k = 0; k < 3; ++k) {
    P[k] = (a * A[k] + b * B[k]) / p;
    Q[k] = (c * Cc[k] + d * Dd[k]) / q;
  }
  const T AB[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
  const T CD[3] = {Cc[0] - Dd[0], Cc[1] - Dd[1], Cc[2] - Dd[2]};
  const T K1 = T(exp(-static_cast<double>(a) * static_cast<double>(b) / static_cast<double>(p) *
                     static_cast<double>(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2])));
  const T K2 = T(exp(-static_cast<double>(c) * static_cast<double>(d) / static_cast<double>(q) *
                     static_cast<double>(CD[0] * CD[0] + CD[1] * CD[1] + CD[2] * CD[2])));

  T PA[3], PB[3], QC[3], QD[3];
  for (int k = 0; k < 3; ++k) {
    PA[k] = P[k] - A[k];
    PB[k] = P[k] - B[k];
    QC[k] = Q[k] - Cc[k];
    QD[k] = Q[k] - Dd[k];
  }

  T Ex1[CUEST_E1D_TMAX], Ey1[CUEST_E1D_TMAX], Ez1[CUEST_E1D_TMAX];
  T Ex2[CUEST_E1D_TMAX], Ey2[CUEST_E1D_TMAX], Ez2[CUEST_E1D_TMAX];
  cuest_md_E1d(la[0], lb[0], p, PA[0], PB[0], T(1), Ex1);
  cuest_md_E1d(la[1], lb[1], p, PA[1], PB[1], T(1), Ey1);
  cuest_md_E1d(la[2], lb[2], p, PA[2], PB[2], T(1), Ez1);
  cuest_md_E1d(lc[0], ld[0], q, QC[0], QD[0], T(1), Ex2);
  cuest_md_E1d(lc[1], ld[1], q, QC[1], QD[1], T(1), Ey2);
  cuest_md_E1d(lc[2], ld[2], q, QC[2], QD[2], T(1), Ez2);

  const T alpha = p * q / (p + q);
  T PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
  const T Tval = alpha * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
  const int ltot = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2] + lc[0] + lc[1] + lc[2] + ld[0] +
                   ld[1] + ld[2];
  const int nroots = ltot / 2 + 1;

  double F[2 * CUEST_RYS_MAX_ROOTS];
  cuest_boys_vec(2 * nroots - 1, static_cast<double>(Tval), F);

  double u[CUEST_RYS_MAX_ROOTS], w[CUEST_RYS_MAX_ROOTS];
  cuest_rys_rule(static_cast<double>(Tval), nroots, F, u, w);

  const T sq2a = T(sqrt(2.0 * static_cast<double>(alpha)));
  T total = T(0);
  for (int k = 0; k < nroots; ++k) {
    const T t = T(sqrt(u[k]));
    T prod = T(1);
    for (int dd = 0; dd < 3; ++dd) {
      const T *E1 = (dd == 0) ? Ex1 : ((dd == 1) ? Ey1 : Ez1);
      const T *E2 = (dd == 0) ? Ex2 : ((dd == 1) ? Ey2 : Ez2);
      const T z = sq2a * PQ[dd] * t;
      T s = T(0);
      for (int tx = 0; tx <= la[dd] + lb[dd]; ++tx) {
        for (int ux = 0; ux <= lc[dd] + ld[dd]; ++ux) {
          const int m = tx + ux;
          T p2a = T(1);
          T tm = T(1);
          for (int e = 0; e < m; ++e) {
            p2a *= sq2a;
            tm *= t;
          }
          const T he = T(cuest_hermite_he(m, static_cast<double>(z)));
          T contrib = p2a * tm * he;
          if (tx & 1) contrib = -contrib;
          s += contrib * E1[tx] * E2[ux];
        }
      }
      prod *= s;
    }
    total += T(w[k]) * prod;
  }

  const T pre = T(CUEST_TWO_PI_5_2) / (p * q * T(sqrt(static_cast<double>(p + q))));
  return pre * K1 * K2 * T(F[0]) * total;
}

#endif /* CUEST_ERI_RYS_CUH */
