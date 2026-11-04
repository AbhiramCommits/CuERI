/**
 * @file one_electron.cpp
 * @brief CPU backend for overlap, kinetic and nuclear-attraction matrices.
 *
 * Work is distributed over unordered shell pairs with OpenMP.  Each pair is
 * written exactly once (the mirrored block is filled from the same thread), so
 * no atomics or reductions are required.
 */
#include <cmath>
#include <utility>
#include <vector>

#include "cuest_internal.h"
#include "kernels/md_recurrence.cuh"

namespace cuest {
namespace {

constexpr int CUEST_MAX_SHELL_FUNCS = 8 * CUEST_MAX_CART;

/**
 * @brief Contracted shell-pair block for one operator.
 *
 * @param which 0 = overlap, 1 = kinetic, 2 = nuclear attraction.
 */
template <typename T>
void one_electron_block(const PrimShell &sa, const PrimShell &sb, const double *A,
                        const double *B, const BasisData &basis, int which, T *block) {
  const int na = sa.ncomp();
  const int nb = sb.ncomp();
  const int ma = sa.nctr * na;
  const int mb = sb.nctr * nb;
  for (int k = 0; k < ma * mb; ++k) block[k] = T(0);

  T aT[3], bT[3];
  for (int k = 0; k < 3; ++k) {
    aT[k] = T(A[k]);
    bT[k] = T(B[k]);
  }

  int ca_l[CUEST_MAX_CART][3], cb_l[CUEST_MAX_CART][3];
  for (int k = 0; k < na; ++k) cuest_cart_component(sa.L, k, &ca_l[k][0], &ca_l[k][1], &ca_l[k][2]);
  for (int k = 0; k < nb; ++k) cuest_cart_component(sb.L, k, &cb_l[k][0], &cb_l[k][1], &cb_l[k][2]);

  for (int ca = 0; ca < sa.nctr; ++ca) {
    for (int cb = 0; cb < sb.nctr; ++cb) {
      for (int pa = 0; pa < sa.nprim; ++pa) {
        const T a = T(sa.exps[pa]);
        for (int pb = 0; pb < sb.nprim; ++pb) {
          const T b = T(sb.exps[pb]);
          const T p = a + b;
          T P[3], PA[3], PB[3], AB[3];
          for (int k = 0; k < 3; ++k) {
            P[k] = (a * aT[k] + b * bT[k]) / p;
            PA[k] = P[k] - aT[k];
            PB[k] = P[k] - bT[k];
            AB[k] = aT[k] - bT[k];
          }
          const T K = T(exp(-static_cast<double>(a) * static_cast<double>(b) /
                            static_cast<double>(p) *
                            static_cast<double>(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2])));

          for (int ka = 0; ka < na; ++ka) {
            const T coef_a = T(sa.coeff(ca, ka)[pa]);
            for (int kb = 0; kb < nb; ++kb) {
              const T coef_b = T(sb.coeff(cb, kb)[pb]);
              const T coef = coef_a * coef_b;
              if (coef == T(0)) continue;
              T val = T(0);
              if (which == 0) {
                val = cuest_prim_overlap<T>(ca_l[ka], cb_l[kb], p, PA, PB, K);
              } else if (which == 1) {
                val = cuest_prim_kinetic<T>(ca_l[ka], cb_l[kb], a, b, p, PA, PB, K);
              } else {
                for (int at = 0; at < basis.natoms; ++at) {
                  const double *C = &basis.atom_xyz[3 * at];
                  T PC[3];
                  for (int k = 0; k < 3; ++k) PC[k] = P[k] - T(C[k]);
                  val += cuest_prim_nuclear<T>(ca_l[ka], cb_l[kb], p, PA, PB, K, PC,
                                               T(basis.atom_Z[at]));
                }
              }
              block[(ca * na + ka) * mb + (cb * nb + kb)] += coef * val;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void run_one_electron(const BasisData &basis, int which, T *out, int ld) {
  std::vector<std::pair<int, int>> pairs;
  pairs.reserve(static_cast<size_t>(basis.nshells) * (basis.nshells + 1) / 2);
  for (int i = 0; i < basis.nshells; ++i)
    for (int j = 0; j <= i; ++j) pairs.emplace_back(i, j);

  const long npairs = static_cast<long>(pairs.size());
#if defined(CUEST_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (long ip = 0; ip < npairs; ++ip) {
    T block[CUEST_MAX_SHELL_FUNCS * CUEST_MAX_SHELL_FUNCS];
    const int i = pairs[static_cast<size_t>(ip)].first;
    const int j = pairs[static_cast<size_t>(ip)].second;
    const PrimShell &si = basis.shells[static_cast<size_t>(i)];
    const PrimShell &sj = basis.shells[static_cast<size_t>(j)];
    const double *Ai = &basis.atom_xyz[3 * si.atom];
    const double *Aj = &basis.atom_xyz[3 * sj.atom];
    one_electron_block<T>(si, sj, Ai, Aj, basis, which, block);
    const int ma = si.nctr * si.ncomp();
    const int mb = sj.nctr * sj.ncomp();
    for (int m = 0; m < ma; ++m) {
      for (int n = 0; n < mb; ++n) {
        const int row = si.func_offset + m;
        const int col = sj.func_offset + n;
        const T v = block[m * mb + n];
        out[static_cast<size_t>(row) * ld + col] = v;
        if (i != j) out[static_cast<size_t>(col) * ld + row] = v;
      }
    }
  }
}

}  // namespace

cuestStatus_t cpu_one_electron(const BasisData &basis, int which, cuestPrecision_t precision,
                               double *out, int ld) {
  if (!out || ld < basis.nbf) return CUEST_INVALID_VALUE;
  const size_t n = static_cast<size_t>(basis.nbf);
  std::vector<double> tmp(n * n, 0.0);
  if (precision == CUEST_FP32) {
    std::vector<float> t32(n * n, 0.0f);
    run_one_electron<float>(basis, which, t32.data(), basis.nbf);
    for (size_t k = 0; k < n * n; ++k) tmp[k] = static_cast<double>(t32[k]);
  } else if (precision == CUEST_FP64) {
    run_one_electron<double>(basis, which, tmp.data(), basis.nbf);
  } else {
    return CUEST_INVALID_VALUE;
  }
  for (int i = 0; i < basis.nbf; ++i)
    for (int j = 0; j < basis.nbf; ++j)
      out[static_cast<size_t>(i) * ld + j] = tmp[static_cast<size_t>(i) * basis.nbf + j];
  return CUEST_SUCCESS;
}

}  // namespace cuest
