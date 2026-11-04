/**
 * @file bench_integrals.cpp
 * @brief Wall-clock benchmark of the CPU integral drivers.
 *
 * Uses the CPU backend entry points directly (never the runtime dispatch) so
 * that the "cpu-1thread" and "cpu-openmp" rows are genuine CPU measurements on
 * any tier.  Emits CSV rows on stdout; scripts/benchmark.py adds the CUDA rows
 * and the speed-up columns.
 *
 * Each configuration runs one warm-up pass followed by N timed passes; the
 * reported value is the median.
 */
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

namespace {

struct Case {
  const char* molecule;
  const char* basis;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void emit(const char* backend, const char* algo, const Case& c, const char* op,
          const char* precision, double ms, double rate, const char* rate_unit, int reps,
          const cuestEriStats_t* stats, int nbf) {
  std::printf("%s,%s,%s,%s,%s,%s,%.6f,%.6f,%s,%d,%.6f,%zu,%zu,%d\n", backend, algo, c.molecule,
              c.basis, op, precision, ms, rate, rate_unit, reps,
              stats && stats->quartets_total > 0
                  ? static_cast<double>(stats->quartets_screened) / stats->quartets_total
                  : 0.0,
              stats ? stats->quartets_screened : 0, stats ? stats->quartets_total : 0, nbf);
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--quick") == 0)
      quick = true;
  const int reps = quick ? 1 : 3;

  const Case cases[] = {{"h2", "sto-3g"},     {"h2o", "sto-3g"},    {"h2o", "6-31g"},
                        {"h2o", "6-31g*"},    {"ch4", "6-31g"},     {"benzene", "sto-3g"},
                        {"benzene", "6-31g"}, {"benzene", "6-31g*"}};

  std::printf(
      "backend,algorithm,molecule,basis,operation,precision,median_ms,rate,rate_unit,repeats,"
      "screened_fraction,quartets_screened,quartets_total,nbf\n");

  const int max_threads = omp_get_max_threads();
  const int thread_settings[2] = {1, max_threads};
  const char* backend_names[2] = {"cpu-1thread", "cpu-openmp"};

  for (const Case& c : cases) {
    const std::string molpath =
        std::string(CUEST_SOURCE_DIR) + "/data/molecules/" + c.molecule + ".json";
    cuest::Molecule mol;
    cuest::BasisData basis;
    std::string err;
    if (!cuest::load_molecule(molpath, mol, err) || !cuest::build_basis(mol, c.basis, basis, err)) {
      std::fprintf(stderr, "cannot load %s/%s: %s\n", c.molecule, c.basis, err.c_str());
      continue;
    }
    const int nbf = basis.nbf;
    const size_t n2 = static_cast<size_t>(nbf) * nbf;
    const size_t n4 = n2 * n2;

    for (int ts = 0; ts < 2; ++ts) {
      omp_set_num_threads(thread_settings[ts]);
      std::vector<double> mat(n2, 0.0);
      for (int which = 0; which < 3; ++which) {
        const char* names[3] = {"overlap", "kinetic", "nuclear"};
        std::vector<double> times;
        for (int r = 0; r < reps + 1; ++r) {
          const auto t0 = std::chrono::steady_clock::now();
          cuest::cpu_one_electron(basis, which, CUEST_FP64, mat.data(), nbf);
          const auto t1 = std::chrono::steady_clock::now();
          if (r > 0)
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double ms = median(times);
        emit(backend_names[ts], "md", c, names[which], "fp64", ms,
             static_cast<double>(n2) / (ms * 1e-3), "matrix-elements/s", reps, nullptr, nbf);
      }

      std::vector<double> eri(n4, 0.0);
      const struct {
        cuestEriAlgo_t algo;
        cuestPrecision_t prec;
        const char* aname;
        const char* pname;
      } configs[4] = {{CUEST_ERI_ALGO_MD, CUEST_FP64, "md", "fp64"},
                      {CUEST_ERI_ALGO_RYS, CUEST_FP64, "rys", "fp64"},
                      {CUEST_ERI_ALGO_MD, CUEST_FP32, "md", "fp32"},
                      {CUEST_ERI_ALGO_RYS, CUEST_FP32, "rys", "fp32"}};
      for (int ic = 0; ic < 4; ++ic) {
        std::vector<double> times;
        cuestEriStats_t stats{};
        for (int r = 0; r < reps + 1; ++r) {
          const auto t0 = std::chrono::steady_clock::now();
          cuest::cpu_eri_tensor(basis, configs[ic].prec, configs[ic].algo, 1e-10, eri.data(),
                                &stats);
          const auto t1 = std::chrono::steady_clock::now();
          if (r > 0)
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double ms = median(times);
        emit(backend_names[ts], configs[ic].aname, c, "eri", configs[ic].pname, ms,
             static_cast<double>(n4) / (ms * 1e-3), "integrals/s", reps, &stats, nbf);
      }
    }
    std::fflush(stdout);
  }
  return 0;
}
