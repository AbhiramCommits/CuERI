/**
 * @file test_one_electron.cpp
 * @brief McMurchie-Davidson one-electron matrices vs the independent reference.
 *
 * The reference (src/reference/ref_one_electron.cpp) uses the Obara-Saika
 * overlap recursion, the analytic gradient kinetic formula and an
 * incomplete-gamma Boys function; it shares no integral code with the
 * production MD path.  Errors are reported per shell class.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"
#include "reference/ref_one_electron.h"

namespace {

constexpr double kTolRel = 1e-9;
constexpr double kTolAbs = 1e-11;

const char *class_name(int la, int lb) {
  static const char *names[3][3] = {{"ss", "sp", "sd"}, {"sp", "pp", "pd"}, {"sd", "pd", "dd"}};
  const int a = (la < lb) ? la : lb;
  const int b = (la < lb) ? lb : la;
  return names[a][b];
}

int run_case(const char *molecule, const char *basis_name) {
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/data/molecules/" + molecule + ".json";
  cuest::Molecule mol;
  std::string err;
  if (!cuest::load_molecule(path, mol, err)) {
    std::printf("FAIL: %s\n", err.c_str());
    return 1;
  }
  cuest::BasisData basis;
  if (!cuest::build_basis(mol, basis_name, basis, err)) {
    std::printf("FAIL: %s\n", err.c_str());
    return 1;
  }
  const int nbf = basis.nbf;
  cuestHandle_t h = nullptr;
  if (cuestCreate(&h) != CUEST_SUCCESS) return 1;

  int failures = 0;
  const char *matrix_names[3] = {"S", "T", "V"};
  for (int which = 0; which < 3; ++which) {
    std::vector<double> lib(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> ref(static_cast<size_t>(nbf) * nbf, 0.0);
    cuestStatus_t st;
    if (which == 0) {
      st = cuestOverlapMatrix(h, nullptr, CUEST_FP64, lib.data(), nbf);
    } else if (which == 1) {
      st = cuestKineticMatrix(h, nullptr, CUEST_FP64, lib.data(), nbf);
    } else {
      st = cuestNuclearMatrix(h, nullptr, CUEST_FP64, lib.data(), nbf);
    }
    if (st != CUEST_INVALID_VALUE) {
      std::printf("FAIL: expected basis==NULL to be rejected\n");
      ++failures;
    }
    /* Use the C API with a real basis handle created through the loader is not
       possible here (the loader owns the basis), so call the CPU backend. */
    st = cuest::cpu_one_electron(basis, which, CUEST_FP64, lib.data(), nbf);
    if (st != CUEST_SUCCESS) {
      std::printf("FAIL: cpu_one_electron which=%d status=%d\n", which, st);
      ++failures;
      continue;
    }
    cuest_ref::compute_one_electron(basis, which, ref.data(), nbf);

    std::map<std::string, std::pair<double, double>> worst;  // class -> (abs, rel)
    for (size_t a = 0; a < basis.shells.size(); ++a) {
      for (size_t b = 0; b < basis.shells.size(); ++b) {
        const cuest::PrimShell &sa = basis.shells[a];
        const cuest::PrimShell &sb = basis.shells[b];
        const std::string cls = class_name(sa.L, sb.L);
        const int ma = sa.nctr * sa.ncomp();
        const int mb = sb.nctr * sb.ncomp();
        for (int m = 0; m < ma; ++m) {
          for (int n = 0; n < mb; ++n) {
            const int row = sa.func_offset + m;
            const int col = sb.func_offset + n;
            const double l = lib[static_cast<size_t>(row) * nbf + col];
            const double r = ref[static_cast<size_t>(row) * nbf + col];
            const double ad = std::fabs(l - r);
            const double rd = ad / std::max(std::fabs(r), 1.0);
            auto &w = worst[cls];
            if (ad > w.first) w.first = ad;
            if (rd > w.second) w.second = rd;
            if (!(ad <= kTolAbs || rd <= kTolRel)) {
              std::printf("FAIL: %s %s[%d,%d] lib=%.17g ref=%.17g\n", matrix_names[which],
                          cls.c_str(), row, col, l, r);
              ++failures;
            }
          }
        }
      }
    }
    for (const auto &kv : worst) {
      std::printf("  %s/%s %-2s  max abs %.3e  max rel %.3e\n", molecule, basis_name,
                  kv.first.c_str(), kv.second.first, kv.second.second);
    }
  }
  cuestDestroy(h);
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  failures += run_case("h2o", "sto-3g");
  failures += run_case("ch4", "6-31g");
  failures += run_case("h2o", "6-31g*");
  std::printf("test_one_electron: %s\n", failures == 0 ? "all checks passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
