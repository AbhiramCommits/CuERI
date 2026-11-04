/**
 * @file ref_one_electron.cpp
 * @brief Independent one-electron reference (test-only).
 *
 * Overlap:  Obara-Saika recursion in one dimension, multiplied over x,y,z.
 *
 *   S(0,0)   = sqrt(pi/p) exp(-mu (A-B)^2)
 *   S(i+1,j) = (P-A) S(i,j) + [ i S(i-1,j) + j S(i,j-1) ] / (2p)
 *
 * Kinetic:  T = 1/2 int grad(phi_a).grad(phi_b) expanded into overlap-like
 * terms with shifted angular momenta, all evaluated with the OS overlap above.
 *
 * Nuclear:  Hermite expansion with the Coulomb auxiliary R_t, but the Boys
 * function is computed from the incomplete gamma function (series for T <= a+1,
 * continued fraction otherwise) instead of the production branch structure.
 */
#include "reference/ref_one_electron.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/shell.cuh"

namespace cuest_ref {
namespace {

using LD = long double;
constexpr LD PI_L = 3.14159265358979323846264338327950288L;

[[maybe_unused]] LD dfact(int n) {
  LD r = 1.0L;
  for (int k = n; k > 0; k -= 2) r *= static_cast<LD>(k);
  return r;
}

/* --- incomplete gamma (Numerical Recipes gser/gcf), independent of boys.cuh -- */
LD gamma_series(LD a, LD x) {
  const int itmax = 500;
  const LD eps = 1e-19L;
  LD ap = a;
  LD sum = 1.0L / a;
  LD del = sum;
  for (int n = 1; n <= itmax; ++n) {
    ap += 1.0L;
    del *= x / ap;
    sum += del;
    if (fabsl(del) < fabsl(sum) * eps) break;
  }
  return sum * expl(-x + a * logl(x) - lgammal(a));
}

LD gamma_cf(LD a, LD x) {
  const int itmax = 500;
  const LD eps = 1e-19L;
  const LD fpmin = 1e-300L;
  LD b = x + 1.0L - a;
  LD c = 1.0L / fpmin;
  LD d = 1.0L / b;
  LD h = d;
  for (int i = 1; i <= itmax; ++i) {
    const LD an = -static_cast<LD>(i) * (static_cast<LD>(i) - a);
    b += 2.0L;
    d = an * d + b;
    if (fabsl(d) < fpmin) d = fpmin;
    c = b + an / c;
    if (fabsl(c) < fpmin) c = fpmin;
    d = 1.0L / d;
    const LD del = d * c;
    h *= del;
    if (fabsl(del - 1.0L) < eps) break;
  }
  return expl(-x + a * logl(x) - lgammal(a)) * h;
}

LD gammp(LD a, LD x) {
  if (x <= 0.0L) return 0.0L;
  if (x < a + 1.0L) return gamma_series(a, x);
  return 1.0L - gamma_cf(a, x);
}

}  // namespace

LD boys_ref(int n, LD T) {
  if (T <= 0.0L) return 1.0L / (2.0L * n + 1.0L);
  const LD s = static_cast<LD>(n) + 0.5L;
  return 0.5L * gammp(s, T) * tgammal(s) / powl(T, s);
}

namespace {

/** @brief 1-D Obara-Saika overlap. */
LD os_overlap_1d(int i, int j, LD p, LD PA, LD PB, LD K) {
  const int ni = i, nj = j;
  std::vector<LD> tab(static_cast<size_t>(ni + 1) * (nj + 1), 0.0L);
  auto at = [&](int ii, int jj) -> LD & { return tab[static_cast<size_t>(ii) * (nj + 1) + jj]; };
  at(0, 0) = sqrtl(PI_L / p) * K;
  for (int ii = 0; ii <= ni; ++ii) {
    for (int jj = 0; jj <= nj; ++jj) {
      if (ii == 0 && jj == 0) continue;
      LD v = 0.0L;
      if (ii > 0) {
        v = PA * at(ii - 1, jj);
        if (ii - 1 > 0) v += (ii - 1) * at(ii - 2, jj) / (2.0L * p);
        if (jj > 0) v += jj * at(ii - 1, jj - 1) / (2.0L * p);
      } else {
        v = PB * at(ii, jj - 1);
        if (jj - 1 > 0) v += (jj - 1) * at(ii, jj - 2) / (2.0L * p);
      }
      at(ii, jj) = v;
    }
  }
  return at(ni, nj);
}

struct Geom {
  LD a, b, p;
  LD A[3], B[3], P[3], PA[3], PB[3], Kd[3];
};

Geom make_geom(LD a, LD b, const double *A, const double *B) {
  Geom g;
  g.a = a;
  g.b = b;
  g.p = a + b;
  for (int k = 0; k < 3; ++k) {
    g.A[k] = A[k];
    g.B[k] = B[k];
    g.P[k] = (a * g.A[k] + b * g.B[k]) / g.p;
    g.PA[k] = g.P[k] - g.A[k];
    g.PB[k] = g.P[k] - g.B[k];
    const LD d = g.A[k] - g.B[k];
    g.Kd[k] = expl(-a * b / g.p * d * d);
  }
  return g;
}

LD prim_overlap_ref(const int la[3], const int lb[3], const Geom &g) {
  LD v = 1.0L;
  for (int d = 0; d < 3; ++d) v *= os_overlap_1d(la[d], lb[d], g.p, g.PA[d], g.PB[d], g.Kd[d]);
  return v;
}

/* S3 with one angular momentum component modified. */
LD overlap_mod(const int la[3], const int lb[3], const Geom &g, int dim, int di, int dj) {
  int a2[3] = {la[0], la[1], la[2]};
  int b2[3] = {lb[0], lb[1], lb[2]};
  a2[dim] += di;
  b2[dim] += dj;
  if (a2[dim] < 0 || b2[dim] < 0) return 0.0L;
  return prim_overlap_ref(a2, b2, g);
}

LD prim_kinetic_ref(const int la[3], const int lb[3], const Geom &g) {
  LD tot = 0.0L;
  for (int d = 0; d < 3; ++d) {
    const LD la_d = la[d], lb_d = lb[d];
    tot += la_d * lb_d * overlap_mod(la, lb, g, d, -1, -1);
    tot += -2.0L * g.b * la_d * overlap_mod(la, lb, g, d, -1, +1);
    tot += -2.0L * g.a * lb_d * overlap_mod(la, lb, g, d, +1, -1);
    tot += 4.0L * g.a * g.b * overlap_mod(la, lb, g, d, +1, +1);
  }
  return 0.5L * tot;
}

/** @brief Independent Hermite E table (1-D). */
void hermite_E(int i, int j, LD p, LD PA, LD PB, LD K, std::vector<LD> &out) {
  const int n = i + j + 1;
  std::vector<LD> tab(static_cast<size_t>(i + 1) * (j + 1) * n, 0.0L);
  auto at = [&](int ii, int jj, int t) -> LD & {
    return tab[(static_cast<size_t>(ii) * (j + 1) + jj) * n + t];
  };
  at(0, 0, 0) = K;
  for (int ii = 0; ii <= i; ++ii)
    for (int jj = 0; jj <= j; ++jj) {
      if (ii == 0 && jj == 0) continue;
      for (int t = 0; t <= ii + jj; ++t) {
        const int pi = (ii > 0) ? ii - 1 : ii;
        const int pj = (ii > 0) ? jj : jj - 1;
        const LD x = (ii > 0) ? PA : PB;
        LD v = x * at(pi, pj, t);
        if (t > 0) v += at(pi, pj, t - 1) / (2.0L * p);
        if (t + 1 < n) v += (t + 1) * at(pi, pj, t + 1);
        at(ii, jj, t) = v;
      }
    }
  out.resize(static_cast<size_t>(i + j + 1));
  for (int t = 0; t <= i + j; ++t) out[t] = at(i, j, t);
}

LD prim_nuclear_ref(const int la[3], const int lb[3], const Geom &g, const double *C, LD Z) {
  std::vector<LD> Ex, Ey, Ez;
  hermite_E(la[0], lb[0], g.p, g.PA[0], g.PB[0], 1.0L, Ex);
  hermite_E(la[1], lb[1], g.p, g.PA[1], g.PB[1], 1.0L, Ey);
  hermite_E(la[2], lb[2], g.p, g.PA[2], g.PB[2], 1.0L, Ez);

  const int lmax = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2];
  LD D[3], T = 0.0L;
  for (int k = 0; k < 3; ++k) {
    D[k] = g.P[k] - C[k];
    T += D[k] * D[k];
  }
  T *= g.p;

  std::vector<LD> F(lmax + 1);
  for (int n = 0; n <= lmax; ++n) F[n] = boys_ref(n, T);

  /* R(tx,ty,tz) generated downward in n. */
  std::vector<LD> next, cur;
  auto ridx = [](int tx, int ty, int tz) { return (tx * 9 + ty) * 9 + tz; };
  next.assign(729, 0.0L);
  cur.assign(729, 0.0L);
  next[0] = powl(-2.0L * g.p, lmax) * F[lmax];
  for (int n = lmax - 1; n >= 0; --n) {
    std::fill(cur.begin(), cur.end(), 0.0L);
    const int lim = lmax - n;
    for (int tx = 0; tx <= lim; ++tx)
      for (int ty = 0; ty <= lim - tx; ++ty)
        for (int tz = 0; tz <= lim - tx - ty; ++tz) {
          if (tx == 0 && ty == 0 && tz == 0) {
            cur[0] = powl(-2.0L * g.p, n) * F[n];
          } else if (tx > 0) {
            cur[ridx(tx, ty, tz)] = D[0] * next[ridx(tx - 1, ty, tz)] +
                                    ((tx >= 2) ? (tx - 1) * next[ridx(tx - 2, ty, tz)] : 0.0L);
          } else if (ty > 0) {
            cur[ridx(tx, ty, tz)] = D[1] * next[ridx(tx, ty - 1, tz)] +
                                    ((ty >= 2) ? (ty - 1) * next[ridx(tx, ty - 2, tz)] : 0.0L);
          } else {
            cur[ridx(tx, ty, tz)] = D[2] * next[ridx(tx, ty, tz - 1)] +
                                    ((tz >= 2) ? (tz - 1) * next[ridx(tx, ty, tz - 2)] : 0.0L);
          }
        }
    next.swap(cur);
  }

  LD sum = 0.0L;
  for (int tx = 0; tx <= la[0] + lb[0]; ++tx)
    for (int ty = 0; ty <= la[1] + lb[1]; ++ty)
      for (int tz = 0; tz <= la[2] + lb[2]; ++tz)
        sum += Ex[tx] * Ey[ty] * Ez[tz] * next[ridx(tx, ty, tz)];

  LD K = 1.0L;
  for (int k = 0; k < 3; ++k) K *= g.Kd[k];
  return -Z * (2.0L * PI_L / g.p) * K * sum;
}

}  // namespace

void compute_one_electron(const cuest::BasisData &basis, int which, double *out, int ld) {
  const size_t nbf = static_cast<size_t>(basis.nbf);
  for (size_t k = 0; k < nbf * nbf; ++k) out[k] = 0.0;
  for (size_t ia = 0; ia < basis.shells.size(); ++ia) {
    const cuest::PrimShell &sa = basis.shells[ia];
    for (size_t ib = 0; ib < basis.shells.size(); ++ib) {
      const cuest::PrimShell &sb = basis.shells[ib];
      const double *A = &basis.atom_xyz[3 * sa.atom];
      const double *B = &basis.atom_xyz[3 * sb.atom];
      const int na = sa.ncomp(), nb = sb.ncomp();
      for (int ca = 0; ca < sa.nctr; ++ca) {
        for (int cb = 0; cb < sb.nctr; ++cb) {
          for (int p = 0; p < sa.nprim; ++p) {
            for (int q = 0; q < sb.nprim; ++q) {
              const Geom g = make_geom(sa.exps[p], sb.exps[q], A, B);
              for (int ka = 0; ka < na; ++ka) {
                int la[3], lb[3];
                cuest_cart_component(sa.L, ka, &la[0], &la[1], &la[2]);
                for (int kb = 0; kb < nb; ++kb) {
                  cuest_cart_component(sb.L, kb, &lb[0], &lb[1], &lb[2]);
                  LD v = 0.0L;
                  if (which == 0) {
                    v = prim_overlap_ref(la, lb, g);
                  } else if (which == 1) {
                    v = prim_kinetic_ref(la, lb, g);
                  } else {
                    for (int at = 0; at < basis.natoms; ++at) {
                      v += prim_nuclear_ref(la, lb, g, &basis.atom_xyz[3 * at],
                                            static_cast<LD>(basis.atom_Z[at]));
                    }
                  }
                  const LD coef = static_cast<LD>(sa.coeff(ca, ka)[p]) *
                                  static_cast<LD>(sb.coeff(cb, kb)[q]);
                  const int row = sa.func_offset + ca * na + ka;
                  const int col = sb.func_offset + cb * nb + kb;
                  out[static_cast<size_t>(row) * ld + col] += static_cast<double>(coef * v);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace cuest_ref
