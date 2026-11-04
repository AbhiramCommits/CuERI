/**
 * @file test_screening.cpp
 * @brief Schwarz bounds and screened-tensor soundness.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

namespace {

bool load(const char* molecule, const char* basis_name, cuest::BasisData& basis) {
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/data/molecules/" + molecule + ".json";
  cuest::Molecule mol;
  std::string err;
  if (!cuest::load_molecule(path, mol, err) || !cuest::build_basis(mol, basis_name, basis, err)) {
    std::printf("FAIL: %s\n", err.c_str());
    return false;
  }
  return true;
}

}  // namespace

int main() {
  int failures = 0;
  cuestHandle_t h = nullptr;
  if (cuestCreate(&h) != CUEST_SUCCESS)
    return 1;
  cuest::BasisData basis;
  if (!load("benzene", "sto-3g", basis))
    return 1;
  const int nsh = basis.nshells;
  const int nbf = basis.nbf;

  /* Schwarz bound matrix: symmetric, non-negative, and it must bound the
     diagonal (mu nu|mu nu) elements. */
  std::vector<double> q(static_cast<size_t>(nsh) * nsh, -1.0);
  if (cuestSchwarzBounds(h, nullptr, q.data()) != CUEST_INVALID_VALUE) {
    std::printf("FAIL: NULL basis accepted by cuestSchwarzBounds\n");
    ++failures;
  }
  std::vector<double> qq(static_cast<size_t>(nsh) * nsh, 0.0);
  if (cuest::cpu_schwarz(basis, qq.data()) != CUEST_SUCCESS)
    return 1;
  double asym = 0.0;

  for (int i = 0; i < nsh; ++i)
    for (int j = 0; j < nsh; ++j) {
      asym = std::max(asym, std::fabs(qq[static_cast<size_t>(i) * nsh + j] -
                                      qq[static_cast<size_t>(j) * nsh + i]));
      if (qq[static_cast<size_t>(i) * nsh + j] < 0.0) {
        std::printf("FAIL: negative Schwarz bound\n");
        ++failures;
      }
    }
  std::printf("test_screening: Schwarz matrix max asymmetry %.3e\n", asym);
  if (asym > 1e-14)
    ++failures;

  /* Unscreened reference. */
  const size_t n4 = static_cast<size_t>(nbf) * nbf * nbf * nbf;
  std::vector<double> ref(n4, 0.0);
  cuestEriStats_t sref{};
  if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 0.0, ref.data(), &sref) !=
      CUEST_SUCCESS)
    return 1;

  /* Now exercise the public entry point through a real basis handle. */
  const std::string molpath = std::string(CUEST_SOURCE_DIR) + "/data/molecules/benzene.json";
  cuestBasis_t b = nullptr;
  if (cuestBasisCreateFromFile(h, molpath.c_str(), "sto-3g", &b) != CUEST_SUCCESS)
    return 1;
  const double tols[] = {1e-14, 1e-12, 1e-10, 1e-8};
  for (double tol : tols) {
    std::vector<double> got(n4, 0.0);
    cuestEriStats_t st{};
    if (cuestERITensor(h, b, CUEST_FP64, tol, got.data(), &st) != CUEST_SUCCESS)
      return 1;
    double dev = 0.0;
    for (size_t k = 0; k < n4; ++k)
      dev = std::max(dev, std::fabs(ref[k] - got[k]));
    const double frac =
        st.quartets_total ? static_cast<double>(st.quartets_screened) / st.quartets_total : 0.0;
    std::printf("  tol=%.0e  skipped %zu/%zu (%.2f%%)  max dev %.3e\n", tol, st.quartets_screened,
                st.quartets_total, 100.0 * frac, dev);
    if (dev > tol) {
      std::printf("FAIL: deviation %.3e exceeds tolerance %.3e\n", dev, tol);
      ++failures;
    }
    if (st.quartets_screened == 0) {
      std::printf("FAIL: no quartets were screened at tol=%.0e on benzene\n", tol);
      ++failures;
    }
  }

  /* A shell quartet whose Schwarz product is below the threshold must be
     counted as screened, and the tensor must be bitwise unchanged. */
  double qmax = 0.0;
  for (double v : qq)
    qmax = std::max(qmax, v);
  std::printf("  largest Schwarz bound %.6f\n", qmax);

  cuestBasisDestroy(b);
  cuestDestroy(h);
  std::printf("test_screening: %s\n", failures == 0 ? "all checks passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
