/**
 * @file test_backend_parity.cpp
 * @brief CUDA fp64 output vs CPU fp64 output, elementwise.
 *
 * Tier A: runs the comparison for every (molecule, basis) pair.
 * Tier B: self-skips because no GPU is present, and records the reason.
 * Tier C: self-skips because the CUDA backend was not compiled in.
 *
 * The test never silently passes: the skip reason is printed and written to
 * results/backend_parity_skip.txt.
 */
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

#if CUEST_HAS_CUDA
#include "cuda/cuda_backend.h"
#endif

namespace {

void record_skip(const char* reason) {
  std::printf("SKIP test_backend_parity: %s\n", reason);
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/results/backend_parity_skip.txt";
  std::ofstream out(path);
  if (out)
    out << "test_backend_parity skipped: " << reason << "\n";
}

#if CUEST_HAS_CUDA
int compare_case(const char* molecule, const char* basis_name, double& worst) {
  const std::string path = std::string(CUEST_SOURCE_DIR) + "/data/molecules/" + molecule + ".json";
  cuest::Molecule mol;
  std::string err;
  if (!cuest::load_molecule(path, mol, err))
    return 1;
  cuest::BasisData basis;
  if (!cuest::build_basis(mol, basis_name, basis, err))
    return 1;
  const size_t n = static_cast<size_t>(basis.nbf);
  std::vector<double> cpu(n * n * n * n), gpu(n * n * n * n);
  cuestEriStats_t st{};
  if (cuest::cpu_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 1e-12, cpu.data(), &st) !=
      CUEST_SUCCESS)
    return 1;
  if (cuest::cuda_eri_tensor(basis, CUEST_FP64, CUEST_ERI_ALGO_MD, 1e-12, gpu.data(), &st) !=
      CUEST_SUCCESS)
    return 1;
  double dev = 0.0;
  for (size_t k = 0; k < cpu.size(); ++k)
    dev = std::max(dev, std::fabs(cpu[k] - gpu[k]));
  worst = std::max(worst, dev);
  std::printf("  parity %s/%s: max |CUDA-CPU| = %.3e\n", molecule, basis_name, dev);
  return dev > 1e-10 ? 1 : 0;
}

#endif

}  // namespace

int main() {
#if !CUEST_HAS_CUDA
  record_skip("Tier C build: no CUDA toolkit, the CUDA backend was not compiled");
  return 0;
#else
  cuestHandle_t h = nullptr;
  if (cuestCreate(&h) != CUEST_SUCCESS)
    return 1;
  cuestBackend_t backend = CUEST_BACKEND_CPU;
  cuestGetBackend(h, &backend);
  if (backend != CUEST_BACKEND_CUDA) {
    record_skip("Tier B build: CUDA toolkit present but no usable GPU device");
    cuestDestroy(h);
    return 0;
  }
  double worst = 0.0;
  int failures = 0;
  failures += compare_case("h2o", "sto-3g", worst);
  failures += compare_case("ch4", "6-31g", worst);
  failures += compare_case("h2o", "6-31g*", worst);
  cuestDestroy(h);
  std::printf("test_backend_parity: %s (worst %.3e)\n", failures == 0 ? "passed" : "FAILED", worst);
  return failures == 0 ? 0 : 1;
#endif
}
