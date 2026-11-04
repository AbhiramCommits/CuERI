/**
 * @file eri_cpu.cpp
 * @brief CPU backend for two-electron repulsion integrals.
 *
 * Shell pairs are formed with Schwarz bounds Q_pq = sqrt(max |(pq|pq)|); shell
 * quartets are enumerated in canonical form (i>=j, k>=l, pair(i,j)>=pair(k,l))
 * and the 8-fold permutational symmetry is filled by scattering each canonical
 * block, which avoids write races without atomics.
 *
 * The McMurchie-Davidson kernel hoists the geometry-independent work out of the
 * component loops: for every primitive quartet the 1-D Hermite tables and the
 * 3-D Hermite Coulomb table are built once, then the bra/ket contractions are
 * evaluated with a small dense tensor contraction.  The Rys kernel builds the
 * roots and weights once per primitive quartet and evaluates the polynomial
 * integrand per component quartet.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cuest_internal.h"
#include "kernels/eri_md.cuh"
#include "kernels/eri_rys.cuh"

#if defined(CUEST_USE_OPENMP)
#include <omp.h>
#endif

namespace cuest {
namespace {

constexpr int CUEST_MAX_T = 35;       /* number of |t| <= 4 multi-indices */
constexpr int CUEST_MAX_COMP_PAIRS = 36; /* 6 x 6 */

/** @brief Flat index of a multi-index with |t| <= 4. */
inline int tindex(int tx, int ty, int tz) {
  const int n = tx + ty + tz;
  return (n * (n + 1) * (n + 2)) / 6 + tx * (n + 1) - (tx * (tx - 1)) / 2 + ty;
}

template <typename T>
inline void build_E3(const T tab[3][9 * CUEST_E1D_TMAX], const int la[3], const int lb[3], int Lt,
                     T *E) {
  for (int tx = 0; tx <= Lt; ++tx)
    for (int ty = 0; ty <= Lt - tx; ++ty)
      for (int tz = 0; tz <= Lt - tx - ty; ++tz)
        E[tindex(tx, ty, tz)] = tab[0][(la[0] * 3 + lb[0]) * CUEST_E1D_TMAX + tx] *
                                tab[1][(la[1] * 3 + lb[1]) * CUEST_E1D_TMAX + ty] *
                                tab[2][(la[2] * 3 + lb[2]) * CUEST_E1D_TMAX + tz];
}

/**
 * @brief Contracted block for one shell quartet.
 *
 * @param algo        ERI algorithm.
 * @param block       Output, size ma*mb*mc*md.
 * @param prim_count  Incremented by the number of primitive quartets evaluated.
 */
template <typename T>
void eri_shell_quartet(const BasisData &basis, const PrimShell &sa, const PrimShell &sb,
                       const PrimShell &sc, const PrimShell &sd, cuestEriAlgo_t algo, T *block,
                       size_t *prim_count) {
  const int na = sa.ncomp(), nb = sb.ncomp(), nc = sc.ncomp(), nd = sd.ncomp();
  const int ma = sa.nctr * na, mb = sb.nctr * nb, mc = sc.nctr * nc, md = sd.nctr * nd;
  const size_t nblk = static_cast<size_t>(ma) * mb * mc * md;
  for (size_t k = 0; k < nblk; ++k) block[k] = T(0);

  int La[CUEST_MAX_CART][3], Lb[CUEST_MAX_CART][3], Lc[CUEST_MAX_CART][3], Ld[CUEST_MAX_CART][3];
  for (int k = 0; k < na; ++k) cuest_cart_component(sa.L, k, &La[k][0], &La[k][1], &La[k][2]);
  for (int k = 0; k < nb; ++k) cuest_cart_component(sb.L, k, &Lb[k][0], &Lb[k][1], &Lb[k][2]);
  for (int k = 0; k < nc; ++k) cuest_cart_component(sc.L, k, &Lc[k][0], &Lc[k][1], &Lc[k][2]);
  for (int k = 0; k < nd; ++k) cuest_cart_component(sd.L, k, &Ld[k][0], &Ld[k][1], &Ld[k][2]);

  T A[3], B[3], C[3], D[3];
  for (int k = 0; k < 3; ++k) {
    A[k] = T(basis.atom_xyz[3 * sa.atom + k]);
    B[k] = T(basis.atom_xyz[3 * sb.atom + k]);
    C[k] = T(basis.atom_xyz[3 * sc.atom + k]);
    D[k] = T(basis.atom_xyz[3 * sd.atom + k]);
  }

  const int Lt = sa.L + sb.L;
  const int Lu = sc.L + sd.L;

  for (int ica = 0; ica < sa.nctr; ++ica) {
    for (int icb = 0; icb < sb.nctr; ++icb) {
      for (int icc = 0; icc < sc.nctr; ++icc) {
        for (int icd = 0; icd < sd.nctr; ++icd) {
          for (int pa = 0; pa < sa.nprim; ++pa) {
            const T a = T(sa.exps[pa]);
            for (int pb = 0; pb < sb.nprim; ++pb) {
              const T b = T(sb.exps[pb]);
              const T p = a + b;
              for (int pc = 0; pc < sc.nprim; ++pc) {
                const T c = T(sc.exps[pc]);
                for (int pd = 0; pd < sd.nprim; ++pd) {
                  ++(*prim_count);
                  const T d = T(sd.exps[pd]);
                  const T q = c + d;
                  T P[3], Q[3];
                  for (int k = 0; k < 3; ++k) {
                    P[k] = (a * A[k] + b * B[k]) / p;
                    Q[k] = (c * C[k] + d * D[k]) / q;
                  }
                  const T AB[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
                  const T CD[3] = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};
                  const T K1 = T(exp(-static_cast<double>(a) * static_cast<double>(b) /
                                    static_cast<double>(p) *
                                    static_cast<double>(AB[0] * AB[0] + AB[1] * AB[1] +
                                                        AB[2] * AB[2])));
                  const T K2 = T(exp(-static_cast<double>(c) * static_cast<double>(d) /
                                    static_cast<double>(q) *
                                    static_cast<double>(CD[0] * CD[0] + CD[1] * CD[1] +
                                                        CD[2] * CD[2])));

                  T tabA[3][9 * CUEST_E1D_TMAX];
                  T tabC[3][9 * CUEST_E1D_TMAX];
                  for (int k = 0; k < 3; ++k) {
                    cuest_md_E1d_table(2, 2, p, P[k] - A[k], P[k] - B[k], T(1), tabA[k]);
                    cuest_md_E1d_table(2, 2, q, Q[k] - C[k], Q[k] - D[k], T(1), tabC[k]);
                  }

                  const T alpha = p * q / (p + q);
                  T PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
                  const T Tval = alpha * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
                  const T pre = T(CUEST_TWO_PI_5_2) / (p * q * T(sqrt(static_cast<double>(p + q))));
                  const T outer = pre * K1 * K2;
                  const int Ltot = Lt + Lu;
                  T R[CUEST_RDIM * CUEST_RDIM * CUEST_RDIM];
                  cuest_md_Rtable<T>(Ltot, alpha, PQ, Tval, R);

                  if (algo == CUEST_ERI_ALGO_MD) {
                    const int nt = (Lt + 1) * (Lt + 2) * (Lt + 3) / 6;
                    T E1[CUEST_MAX_COMP_PAIRS * CUEST_MAX_T];
                    T H[CUEST_MAX_COMP_PAIRS * CUEST_MAX_T];
                    T E2[CUEST_MAX_T];
                    for (int ka = 0; ka < na; ++ka)
                      for (int kb = 0; kb < nb; ++kb)
                        build_E3(tabA, La[ka], Lb[kb], Lt, &E1[(ka * nb + kb) * nt]);
                    for (int kc = 0; kc < nc; ++kc) {
                      for (int kd = 0; kd < nd; ++kd) {
                        build_E3(tabC, Lc[kc], Ld[kd], Lu, E2);
                        T *h = &H[(kc * nd + kd) * nt];
                        for (int it = 0; it < nt; ++it) h[it] = T(0);
                        for (int tx = 0; tx <= Lt; ++tx)
                          for (int ty = 0; ty <= Lt - tx; ++ty)
                            for (int tz = 0; tz <= Lt - tx - ty; ++tz) {
                              const int it = tindex(tx, ty, tz);
                              T acc = T(0);
                              for (int ux = 0; ux <= Lu; ++ux)
                                for (int uy = 0; uy <= Lu - ux; ++uy)
                                  for (int uz = 0; uz <= Lu - ux - uy; ++uz) {
                                    const int iu = tindex(ux, uy, uz);
                                    const T r = R[cuest_rindex(tx + ux, ty + uy, tz + uz)];
                                    if ((ux + uy + uz) & 1)
                                      acc -= E2[iu] * r;
                                    else
                                      acc += E2[iu] * r;
                                  }
                              h[it] = acc;
                            }
                      }
                    }
                    for (int ka = 0; ka < na; ++ka) {
                      const T ca = T(sa.coeff(ica, ka)[pa]);
                      for (int kb = 0; kb < nb; ++kb) {
                        const T cab = ca * T(sb.coeff(icb, kb)[pb]);
                        for (int kc = 0; kc < nc; ++kc) {
                          const T cabc = cab * T(sc.coeff(icc, kc)[pc]);
                          for (int kd = 0; kd < nd; ++kd) {
                            const T coef = cabc * T(sd.coeff(icd, kd)[pd]);
                            const T *e1 = &E1[(ka * nb + kb) * nt];
                            const T *h = &H[(kc * nd + kd) * nt];
                            T s = T(0);
                            for (int it = 0; it < nt; ++it) s += e1[it] * h[it];
                            const int fa = ica * na + ka;
                            const int fb = icb * nb + kb;
                            const int fc = icc * nc + kc;
                            const int fd = icd * nd + kd;
                            block[((static_cast<size_t>(fa) * mb + fb) * mc + fc) * md + fd] +=
                                (coef * outer) * s;
                          }
                        }
                      }
                    }
                  } else {
                    /* Rys quadrature: roots and weights once per primitive quartet. */
                    const int nroots = Ltot / 2 + 1;
                    double F[2 * CUEST_RYS_MAX_ROOTS];
                    cuest_boys_vec(2 * nroots - 1, static_cast<double>(Tval), F);
                    double u[CUEST_RYS_MAX_ROOTS], w[CUEST_RYS_MAX_ROOTS];
                    cuest_rys_rule(static_cast<double>(Tval), nroots, F, u, w);
                    const T sq2a = T(sqrt(2.0 * static_cast<double>(alpha)));
                    T F0 = T(F[0]);
                    for (int ka = 0; ka < na; ++ka) {
                      const T ca = T(sa.coeff(ica, ka)[pa]);
                      for (int kb = 0; kb < nb; ++kb) {
                        const T cab = ca * T(sb.coeff(icb, kb)[pb]);
                        for (int kc = 0; kc < nc; ++kc) {
                          const T cabc = cab * T(sc.coeff(icc, kc)[pc]);
                          for (int kd = 0; kd < nd; ++kd) {
                            const T coef = cabc * T(sd.coeff(icd, kd)[pd]);
                            T total = T(0);
                            for (int ir = 0; ir < nroots; ++ir) {
                              const T t = T(sqrt(u[ir]));
                              T prod = T(1);
                              for (int dd = 0; dd < 3; ++dd) {
                                const int *la = La[ka];
                                const int *lb = Lb[kb];
                                const int *lc = Lc[kc];
                                const int *ld = Ld[kd];
                                const T z = sq2a * PQ[dd] * t;
                                T s = T(0);
                                for (int tx = 0; tx <= la[dd] + lb[dd]; ++tx) {
                                  const T e1 = tabA[dd][(la[dd] * 3 + lb[dd]) * CUEST_E1D_TMAX + tx];
                                  for (int ux = 0; ux <= lc[dd] + ld[dd]; ++ux) {
                                    const int m = tx + ux;
                                    T p2a = T(1);
                                    T tm = T(1);
                                    for (int e = 0; e < m; ++e) {
                                      p2a *= sq2a;
                                      tm *= t;
                                    }
                                    const T he = T(cuest_hermite_he(m, static_cast<double>(z)));
                                    T contrib =
                                        p2a * tm * he *
                                        tabC[dd][(lc[dd] * 3 + ld[dd]) * CUEST_E1D_TMAX + ux] * e1;
                                    if (tx & 1) contrib = -contrib;
                                    s += contrib;
                                  }
                                }
                                prod *= s;
                              }
                              total += T(w[ir]) * prod;
                            }
                            const int fa = ica * na + ka;
                            const int fb = icb * nb + kb;
                            const int fc = icc * nc + kc;
                            const int fd = icd * nd + kd;
                            block[((static_cast<size_t>(fa) * mb + fb) * mc + fc) * md + fd] +=
                                (coef * outer * F0) * total;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Schwarz bounds and shell pairs                                             */
/* ------------------------------------------------------------------------- */

template <typename T>
double schwarz_bound(const BasisData &basis, const PrimShell &sa, const PrimShell &sb) {
  const int ma = sa.nctr * sa.ncomp();
  const int mb = sb.nctr * sb.ncomp();
  std::vector<T> block(static_cast<size_t>(ma) * mb * ma * mb);
  size_t prim = 0;
  eri_shell_quartet<T>(basis, sa, sb, sa, sb, CUEST_ERI_ALGO_MD, block.data(), &prim);
  double best = 0.0;
  for (const T v : block) {
    const double a = std::fabs(static_cast<double>(v));
    if (a > best) best = a;
  }
  return std::sqrt(best);
}

}  // namespace

int am_class(int la, int lb, int lc, int ld) {
  /* Ordered class id from the two pair angular momenta. */
  const int p = la + lb;
  const int q = lc + ld;
  const int lo = (p < q) ? p : q;
  const int hi = (p < q) ? q : p;
  return lo * 5 + hi;
}

void build_shell_pairs(const BasisData &basis, std::vector<ShellPair> &pairs, cuestEriAlgo_t algo) {
  (void)algo;
  pairs.clear();
  pairs.reserve(static_cast<size_t>(basis.nshells) * (basis.nshells + 1) / 2);
  const long n = basis.nshells;
  std::vector<double> bounds(static_cast<size_t>(n) * (n + 1) / 2, 0.0);
  std::vector<std::pair<int, int>> index;
  index.reserve(bounds.size());
  for (int i = 0; i < basis.nshells; ++i)
    for (int j = 0; j <= i; ++j) index.emplace_back(i, j);

  const long np = static_cast<long>(index.size());
#if defined(CUEST_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (long k = 0; k < np; ++k) {
    const int i = index[static_cast<size_t>(k)].first;
    const int j = index[static_cast<size_t>(k)].second;
    bounds[static_cast<size_t>(k)] = schwarz_bound<double>(
        basis, basis.shells[static_cast<size_t>(i)], basis.shells[static_cast<size_t>(j)]);
  }
  for (size_t k = 0; k < index.size(); ++k) {
    ShellPair sp;
    sp.i = index[k].first;
    sp.j = index[k].second;
    sp.q = bounds[k];
    pairs.push_back(sp);
  }
}

cuestStatus_t cpu_schwarz(const BasisData &basis, double *q_pair) {
  if (!q_pair) return CUEST_INVALID_VALUE;
  std::vector<ShellPair> pairs;
  build_shell_pairs(basis, pairs, CUEST_ERI_ALGO_MD);
  const size_t n = static_cast<size_t>(basis.nshells);
  for (size_t k = 0; k < n * n; ++k) q_pair[k] = 0.0;
  for (const ShellPair &p : pairs) {
    q_pair[static_cast<size_t>(p.i) * n + p.j] = p.q;
    q_pair[static_cast<size_t>(p.j) * n + p.i] = p.q;
  }
  return CUEST_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Batch and tensor drivers                                                   */
/* ------------------------------------------------------------------------- */

namespace {

struct QuartetDesc {
  int i, j, k, l;
};

template <typename T>
void scatter_symmetries(const BasisData &basis, const QuartetDesc &q, const T *block,
                        double *eri) {
  const int nbf = basis.nbf;
  const PrimShell *sh[4] = {&basis.shells[static_cast<size_t>(q.i)],
                            &basis.shells[static_cast<size_t>(q.j)],
                            &basis.shells[static_cast<size_t>(q.k)],
                            &basis.shells[static_cast<size_t>(q.l)]};
  const int m[4] = {sh[0]->nctr * sh[0]->ncomp(), sh[1]->nctr * sh[1]->ncomp(),
                    sh[2]->nctr * sh[2]->ncomp(), sh[3]->nctr * sh[3]->ncomp()};
  const int perms[8][4] = {{0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
                           {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};
  for (int ip = 0; ip < 8; ++ip) {
    const int *pm = perms[ip];
    for (int fa = 0; fa < m[0]; ++fa) {
      for (int fb = 0; fb < m[1]; ++fb) {
        for (int fc = 0; fc < m[2]; ++fc) {
          for (int fd = 0; fd < m[3]; ++fd) {
            const int f[4] = {fa, fb, fc, fd};
            const size_t src = ((static_cast<size_t>(fa) * m[1] + fb) * m[2] + fc) * m[3] + fd;
            const size_t g0 = static_cast<size_t>(sh[pm[0]]->func_offset + f[pm[0]]);
            const size_t g1 = static_cast<size_t>(sh[pm[1]]->func_offset + f[pm[1]]);
            const size_t g2 = static_cast<size_t>(sh[pm[2]]->func_offset + f[pm[2]]);
            const size_t g3 = static_cast<size_t>(sh[pm[3]]->func_offset + f[pm[3]]);
            const size_t dst = ((g0 * nbf + g1) * nbf + g2) * nbf + g3;
            eri[dst] = static_cast<double>(block[src]);
          }
        }
      }
    }
  }
}

}  // namespace

cuestStatus_t cpu_eri_batch(const BasisData &basis, const cuestQuartet_t *quartets, int nquartets,
                            cuestPrecision_t precision, cuestEriAlgo_t algo, double *out,
                            const size_t *offsets) {
  if (nquartets < 0 || (nquartets > 0 && (!quartets || !out))) return CUEST_INVALID_VALUE;
  if (precision != CUEST_FP32 && precision != CUEST_FP64) return CUEST_INVALID_VALUE;
  size_t cursor = 0;
  for (int iq = 0; iq < nquartets; ++iq) {
    const cuestQuartet_t &q = quartets[iq];
    const int ids[4] = {q.i, q.j, q.k, q.l};
    for (int k = 0; k < 4; ++k) {
      if (ids[k] < 0 || ids[k] >= basis.nshells) return CUEST_INVALID_VALUE;
    }
    const PrimShell &sa = basis.shells[static_cast<size_t>(q.i)];
    const PrimShell &sb = basis.shells[static_cast<size_t>(q.j)];
    const PrimShell &sc = basis.shells[static_cast<size_t>(q.k)];
    const PrimShell &sd = basis.shells[static_cast<size_t>(q.l)];
    const size_t nblk = static_cast<size_t>(sa.nctr * sa.ncomp()) * (sb.nctr * sb.ncomp()) *
                        (sc.nctr * sc.ncomp()) * (sd.nctr * sd.ncomp());
    double *dst = out + (offsets ? offsets[iq] : cursor);
    if (precision == CUEST_FP32) {
      std::vector<float> block(nblk);
      size_t prim = 0;
      eri_shell_quartet<float>(basis, sa, sb, sc, sd, algo, block.data(), &prim);
      for (size_t k = 0; k < nblk; ++k) dst[k] = static_cast<double>(block[k]);
    } else {
      std::vector<double> block(nblk);
      size_t prim = 0;
      eri_shell_quartet<double>(basis, sa, sb, sc, sd, algo, block.data(), &prim);
      for (size_t k = 0; k < nblk; ++k) dst[k] = block[k];
    }
    cursor += nblk;
  }
  return CUEST_SUCCESS;
}

namespace {

template <typename T>
cuestStatus_t run_eri_tensor(const BasisData &basis, cuestEriAlgo_t algo, double screen_tol,
                             double *eri, cuestEriStats_t *stats) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<ShellPair> pairs;
  build_shell_pairs(basis, pairs, algo);

  std::vector<QuartetDesc> quartets;
  quartets.reserve(pairs.size() * (pairs.size() + 1) / 2);
  size_t total = 0;
  size_t screened = 0;
  for (size_t a = 0; a < pairs.size(); ++a) {
    for (size_t b = 0; b <= a; ++b) {
      ++total;
      if (screen_tol > 0.0 && pairs[a].q * pairs[b].q < screen_tol) {
        ++screened;
        continue;
      }
      QuartetDesc q;
      q.i = pairs[a].i;
      q.j = pairs[a].j;
      q.k = pairs[b].i;
      q.l = pairs[b].j;
      quartets.push_back(q);
    }
  }

  const long nq = static_cast<long>(quartets.size());
  long long prim_total = 0;
#if defined(CUEST_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : prim_total)
#endif
  for (long iq = 0; iq < nq; ++iq) {
    const QuartetDesc &q = quartets[static_cast<size_t>(iq)];
    const PrimShell &sa = basis.shells[static_cast<size_t>(q.i)];
    const PrimShell &sb = basis.shells[static_cast<size_t>(q.j)];
    const PrimShell &sc = basis.shells[static_cast<size_t>(q.k)];
    const PrimShell &sd = basis.shells[static_cast<size_t>(q.l)];
    const size_t nblk = static_cast<size_t>(sa.nctr * sa.ncomp()) * (sb.nctr * sb.ncomp()) *
                        (sc.nctr * sc.ncomp()) * (sd.nctr * sd.ncomp());
    std::vector<T> block(nblk);
    size_t prim = 0;
    eri_shell_quartet<T>(basis, sa, sb, sc, sd, algo, block.data(), &prim);
    prim_total += static_cast<long long>(prim);
    scatter_symmetries<T>(basis, q, block.data(), eri);
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (stats) {
    stats->quartets_total = total;
    stats->quartets_screened = screened;
    stats->primitive_pairs = static_cast<size_t>(prim_total);
    stats->kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats->max_shell_am = basis.max_am;
  }
  return CUEST_SUCCESS;
}

}  // namespace

cuestStatus_t cpu_eri_tensor(const BasisData &basis, cuestPrecision_t precision,
                             cuestEriAlgo_t algo, double screen_tol, double *eri,
                             cuestEriStats_t *stats) {
  if (!eri) return CUEST_INVALID_VALUE;
  if (precision != CUEST_FP32 && precision != CUEST_FP64) return CUEST_INVALID_VALUE;
  if (stats) {
    stats->quartets_total = 0;
    stats->quartets_screened = 0;
    stats->primitive_pairs = 0;
    stats->kernel_ms = 0.0;
    stats->max_shell_am = basis.max_am;
  }
  if (precision == CUEST_FP32)
    return run_eri_tensor<float>(basis, algo, screen_tol, eri, stats);
  return run_eri_tensor<double>(basis, algo, screen_tol, eri, stats);
}

}  // namespace cuest
