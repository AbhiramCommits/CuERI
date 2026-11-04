/**
 * @file test_eri.cpp
 * @brief Two-electron integrals: symmetry, MD vs Rys, screening, analytic H2.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

namespace {

const char *class_name(int la, int lb) {
  static const char *names[3][3] = {{"ss", "sp", "sd"}, {"sp", "pp", "pd"}, {"sd", "pd", "dd"}};
  const int a = (la < lb) ? la : lb;
  const int b = (la < lb) ? lb : la;
  return names[a][b];
}

bool load(const char *molecule, const char *basis_name, cuest::BasisData &basis) {
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/data/molecules/" + molecule + ".json";
  cuest::Molecule mol;
  std::string err;
  if (!cuest::load_molecule(path, mol, err)) {
    std::printf("FAIL: %s\n", err.c_str());
    return false;
  }
  if (!cuest::build_basis(mol, basis_name, basis, err)) {
    std::printf("FAIL: %s\n", err.c_str());
    return false;
  }
  return true;
}

int test_h2_analytic() {
  cuest::BasisData basis;
  if (!load("h2", "sto-3g", basis)) return 1;
  std::vector<double> eri(static_cast<size_t>(basis.nbf) * basis.nbf * basis.nbf * basis.nbf);
  cuestEriStats_t stats{};
  if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 0.0, eri.data(), &stats) !=
      CUEST_SUCCESS)
    return 1;
  auto at = [&](int i, int j, int k, int l) {
    const size_t n = basis.nbf;
    return eri[((static_cast<size_t>(i) * n + j) * n + k) * n + l];
  };
  /* Closed-form (ss|ss) values for H2/STO-3G at R = 1.4010429488 bohr, obtained
     from the analytic Gaussian formula
       (ss|ss) = 2 pi^(5/2)/(p q sqrt(p+q)) exp(-mu1 R_AB^2 - mu2 R_CD^2) F0(...)
     summed over the STO-3G contraction with unit-normalised components. */
  const double refs[4] = {0.7746059439198977, 0.5694684067443392, 0.2966631722823799,
                          0.4437931535453381};
  const int idx[4][4] = {{0, 0, 0, 0}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 1}};
  int failures = 0;
  for (int t = 0; t < 4; ++t) {
    const double got = at(idx[t][0], idx[t][1], idx[t][2], idx[t][3]);
    const double err = std::fabs(got - refs[t]);
    std::printf("  H2 analytic (%d%d|%d%d) = %.16f  ref %.16f  err %.2e\n", idx[t][0], idx[t][1],
                idx[t][2], idx[t][3], got, refs[t], err);
    if (err > 1e-12) ++failures;
  }
  return failures;
}

int test_symmetry() {
  int failures = 0;
  const char *cases[2][2] = {{"h2o", "sto-3g"}, {"ch4", "6-31g"}};
  for (auto &c : cases) {
    cuest::BasisData basis;
    if (!load(c[0], c[1], basis)) return 1;
    const size_t n = basis.nbf;
    std::vector<double> eri(n * n * n * n);
    cuestEriStats_t stats{};
    if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 1e-12, eri.data(), &stats) !=
        CUEST_SUCCESS)
      return 1;
    auto at = [&](int i, int j, int k, int l) {
      return eri[((static_cast<size_t>(i) * n + j) * n + k) * n + l];
    };
    double worst = 0.0;
    for (int i = 0; i < static_cast<int>(n); ++i)
      for (int j = 0; j < static_cast<int>(n); ++j)
        for (int k = 0; k < static_cast<int>(n); ++k)
          for (int l = 0; l < static_cast<int>(n); ++l) {
            const double v = at(i, j, k, l);
            worst = std::max(worst, std::fabs(v - at(j, i, k, l)));
            worst = std::max(worst, std::fabs(v - at(i, j, l, k)));
            worst = std::max(worst, std::fabs(v - at(k, l, i, j)));
            worst = std::max(worst, std::fabs(v - at(j, i, l, k)));
          }
    std::printf("  symmetry %s/%s: max deviation over 8 permutations %.3e\n", c[0], c[1], worst);
    if (worst > 1e-13) ++failures;
  }
  return failures;
}

int test_md_vs_rys() {
  int failures = 0;
  const char *cases[4][2] = {{"h2o", "sto-3g"}, {"h2o", "6-31g"}, {"h2o", "6-31g*"},
                             {"ch4", "6-31g"}};
  for (auto &c : cases) {
    cuest::BasisData basis;
    if (!load(c[0], c[1], basis)) return 1;
    const size_t n = basis.nbf;
    std::vector<double> emd(n * n * n * n), erys(n * n * n * n);
    cuestEriStats_t s1{}, s2{};
    if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 0.0, emd.data(), &s1) !=
        CUEST_SUCCESS)
      return 1;
    if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_RYS, 0.0, erys.data(), &s2) !=
        CUEST_SUCCESS)
      return 1;
    std::map<std::string, double> worst;
    for (size_t a = 0; a < basis.shells.size(); ++a)
      for (size_t b = 0; b < basis.shells.size(); ++b)
        for (size_t cc = 0; cc < basis.shells.size(); ++cc)
          for (size_t d = 0; d < basis.shells.size(); ++d) {
            const cuest::PrimShell &sa = basis.shells[a];
            const cuest::PrimShell &sb = basis.shells[b];
            const cuest::PrimShell &sc = basis.shells[cc];
            const cuest::PrimShell &sd = basis.shells[d];
            const std::string cls = std::string(class_name(sa.L, sb.L)) + "/" +
                                    class_name(sc.L, sd.L);
            const int ma = sa.nctr * sa.ncomp(), mb = sb.nctr * sb.ncomp();
            const int mc = sc.nctr * sc.ncomp(), md = sd.nctr * sd.ncomp();
            for (int i = 0; i < ma; ++i)
              for (int j = 0; j < mb; ++j)
                for (int k = 0; k < mc; ++k)
                  for (int l = 0; l < md; ++l) {
                    const size_t fi = ((static_cast<size_t>(sa.func_offset + i) * n +
                                        (sb.func_offset + j)) * n + (sc.func_offset + k)) * n +
                                      (sd.func_offset + l);
                    const double diff = std::fabs(emd[fi] - erys[fi]);
                    worst[cls] = std::max(worst[cls], diff);
                  }
          }
    double all = 0.0;
    for (auto &kv : worst) {
      all = std::max(all, kv.second);
      std::printf("  MD-vs-Rys %s/%s %-9s max abs diff %.3e\n", c[0], c[1], kv.first.c_str(),
                  kv.second);
    }
    if (all > 1e-10) {
      std::printf("FAIL: MD vs Rys max diff %.3e exceeds 1e-10\n", all);
      ++failures;
    }
  }
  return failures;
}

int test_screening() {
  int failures = 0;
  const char *cases[2][2] = {{"h2o", "6-31g*"}, {"benzene", "sto-3g"}};
  const double tols[4] = {0.0, 1e-14, 1e-10, 1e-8};
  for (auto &c : cases) {
    cuest::BasisData basis;
    if (!load(c[0], c[1], basis)) return 1;
    const size_t n = basis.nbf;
    std::vector<double> ref(n * n * n * n);
    cuestEriStats_t sref{};
    if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 0.0, ref.data(), &sref) !=
        CUEST_SUCCESS)
      return 1;
    std::printf("  screening %s/%s: %zu canonical quartets\n", c[0], c[1], sref.quartets_total);
    for (double tol : tols) {
      std::vector<double> got(n * n * n * n);
      cuestEriStats_t st{};
      if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, tol, got.data(), &st) !=
          CUEST_SUCCESS)
        return 1;
      double dev = 0.0;
      for (size_t k = 0; k < ref.size(); ++k) dev = std::max(dev, std::fabs(ref[k] - got[k]));
      const double frac = st.quartets_total > 0
                              ? static_cast<double>(st.quartets_screened) / st.quartets_total
                              : 0.0;
      std::printf("    tol=%.0e  screened %zu/%zu (%.1f%%)  max dev %.3e\n", tol,
                  st.quartets_screened, st.quartets_total, 100.0 * frac, dev);
      const double bound = (tol > 0.0) ? tol : 1e-13;
      if (dev > bound) {
        std::printf("FAIL: screening deviation %.3e exceeds bound %.3e\n", dev, bound);
        ++failures;
      }
    }
  }
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  std::printf("test_eri: analytic H2/STO-3G ssss\n");
  failures += test_h2_analytic();
  std::printf("test_eri: 8-fold permutational symmetry\n");
  failures += test_symmetry();
  std::printf("test_eri: MD vs Rys per shell class\n");
  failures += test_md_vs_rys();
  std::printf("test_eri: Schwarz screening soundness\n");
  failures += test_screening();
  std::printf("test_eri: %s\n", failures == 0 ? "all checks passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
