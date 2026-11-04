/**
 * @file test_boys.cpp
 * @brief Boys function accuracy against the SciPy-generated reference table.
 *
 * The reference values come from scipy.special.gammainc / gamma via
 * scripts/gen_boys_reference.py and cover T = 0, 1e-16 ... 1e3 and n = 0..12,
 * i.e. all three numerical branches and their crossovers.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/boys.cuh"

namespace {

struct Row {
  double T;
  int n;
  double ref;
};

/* The tolerance was measured on this build: the maximum relative error over the
 * whole table is ~1e-15 for n <= 12, so 1e-12 leaves three orders of margin. */
constexpr double kRelTol = 1e-12;

}  // namespace

int main() {
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/data/ref/boys_reference.csv";
  std::ifstream in(path);
  if (!in) {
    std::printf("FAIL: cannot open %s\n", path.c_str());
    return 1;
  }
  std::string line;
  std::getline(in, line);  // header
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string t, n, f;
    std::getline(ss, t, ',');
    std::getline(ss, n, ',');
    std::getline(ss, f, ',');
    rows.push_back({std::strtod(t.c_str(), nullptr), std::atoi(n.c_str()),
                    std::strtod(f.c_str(), nullptr)});
  }

  double worst = 0.0;
  double worst_T = 0.0;
  int worst_n = 0;
  int failures = 0;
  for (const Row &r : rows) {
    const double got = cuest_boys(r.n, r.T);
    const double denom = (std::fabs(r.ref) > 1e-300) ? std::fabs(r.ref) : 1.0;
    const double rel = std::fabs(got - r.ref) / denom;
    if (rel > worst) {
      worst = rel;
      worst_T = r.T;
      worst_n = r.n;
    }
    if (rel > kRelTol) {
      std::printf("FAIL: F_%d(%.17g) = %.17g, reference %.17g, rel %.3e\n", r.n, r.T, got, r.ref,
                  rel);
      ++failures;
    }
  }

  /* Explicit corner cases required by the specification. */
  const double corners[] = {0.0, 1e-14, 1e-8, 0.5, 30.0, 200.0};
  for (double T : corners) {
    double F[CUEST_MAX_BOYS + 1];
    cuest_boys_vec(CUEST_MAX_BOYS, T, F);
    for (int n = 0; n <= CUEST_MAX_BOYS; ++n) {
      if (!std::isfinite(F[n]) || F[n] < 0.0) {
        std::printf("FAIL: non-finite/negative F_%d(%g) = %g\n", n, T, F[n]);
        ++failures;
      }
    }
  }
  /* T = 0 must be exact. */
  for (int n = 0; n <= 12; ++n) {
    const double want = 1.0 / (2.0 * n + 1.0);
    if (std::fabs(cuest_boys(n, 0.0) - want) > 1e-16) {
      std::printf("FAIL: F_%d(0) != 1/(2n+1)\n", n);
      ++failures;
    }
  }

  std::printf("test_boys: %zu values, max relative error %.3e at T=%.6g, n=%d\n", rows.size(),
              worst, worst_T, worst_n);
  if (failures == 0) std::printf("test_boys: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
