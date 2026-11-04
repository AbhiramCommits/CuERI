/**
 * @file eri_md.cuh
 * @brief Primitive two-electron repulsion integrals via McMurchie-Davidson.
 *
 * For shell quartet (ab|cd) with primitive exponents a,b,c,d and centres
 * A,B,C,D, define p=a+b, q=c+d, P=(aA+bB)/p, Q=(cC+dD)/q and
 * alpha = pq/(p+q).  Expanding both Gaussian products in Hermite Gaussians and
 * using the Hermite Coulomb auxiliary R_t gives
 *
 *   (ab|cd) = 2 pi^(5/2) / (p q sqrt(p+q))
 *             sum_{t,u} E_t^{ab} E_u^{cd} (-1)^|u| R_{t+u}(alpha, P-Q).
 *
 * The sign (-1)^|u| follows from J_tu = d_P^t d_Q^u J_00 with d_Q = -d_P:
 * J_tu = (-1)^|u| d_P^(t+u) J_00, and R_t(0) = d_P^t F_0(alpha |P-Q|^2).
 * The bra multi-index is t and the ket multi-index is u.
 */
#ifndef CUEST_ERI_MD_CUH
#define CUEST_ERI_MD_CUH

#include "../cuest_macros.h"
#include "md_recurrence.cuh"

/** @brief 2 pi^(5/2), the McMurchie-Davidson two-electron prefactor. */
constexpr double CUEST_TWO_PI_5_2 = 34.986836655249726;

/**
 * @brief Primitive ERI (ab|cd) by McMurchie-Davidson.
 */
template <typename T>
CUEST_HD T cuest_prim_eri_md(const int la[3], const int lb[3], const int lc[3], const int ld[3],
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
  const int lmax = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2] + lc[0] + lc[1] + lc[2] + ld[0] +
                   ld[1] + ld[2];
  T R[CUEST_RDIM * CUEST_RDIM * CUEST_RDIM];
  cuest_md_Rtable<T>(lmax, alpha, PQ, Tval, R);

  T sum = T(0);
  for (int tx = 0; tx <= la[0] + lb[0]; ++tx) {
    for (int ty = 0; ty <= la[1] + lb[1]; ++ty) {
      for (int tz = 0; tz <= la[2] + lb[2]; ++tz) {
        const T e1 = Ex1[tx] * Ey1[ty] * Ez1[tz];
        for (int ux = 0; ux <= lc[0] + ld[0]; ++ux) {
          for (int uy = 0; uy <= lc[1] + ld[1]; ++uy) {
            for (int uz = 0; uz <= lc[2] + ld[2]; ++uz) {
              const T e2 = Ex2[ux] * Ey2[uy] * Ez2[uz];
              const int parity = (ux + uy + uz) & 1;
              const T r = R[cuest_rindex(tx + ux, ty + uy, tz + uz)];
              sum += parity ? -(e1 * e2 * r) : (e1 * e2 * r);
            }
          }
        }
      }
    }
  }

  const T pre = T(CUEST_TWO_PI_5_2) /
                (p * q * T(sqrt(static_cast<double>(p + q))));
  return pre * K1 * K2 * sum;
}

#endif /* CUEST_ERI_MD_CUH */
