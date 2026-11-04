/**
 * @file test_error_handling.cpp
 * @brief Argument validation, unsupported features and lifecycle stress.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

static int g_failures = 0;

#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                               \
    }                                                             \
  } while (0)

int main() {
  /* NULL handle / basis everywhere. */
  cuestBackend_t backend = CUEST_BACKEND_CPU;
  CHECK(cuestGetBackend(nullptr, &backend) == CUEST_INVALID_VALUE);
  CHECK(cuestSetEriAlgo(nullptr, CUEST_ERI_ALGO_MD) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisDestroy(nullptr) == CUEST_SUCCESS);
  CHECK(cuestDestroy(nullptr) == CUEST_SUCCESS);

  const std::string dir = CUEST_SOURCE_DIR;
  const std::string h2 = dir + "/data/molecules/h2.json";
  const std::string empty = dir + "/data/molecules/empty.json";
  const std::string unsupported = dir + "/data/basis/unsupported_l.json";

  cuestHandle_t h = nullptr;
  CHECK(cuestCreate(&h) == CUEST_SUCCESS);
  CHECK(h != nullptr);

  /* Bad basis name, missing file, empty molecule. */
  cuestBasis_t b = nullptr;
  CHECK(cuestBasisCreateFromFile(h, h2.c_str(), "does-not-exist", &b) == CUEST_INVALID_VALUE);
  CHECK(b == nullptr);
  CHECK(cuestBasisCreateFromFile(h, "/no/such/file.json", "sto-3g", &b) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisCreateFromFile(h, empty.c_str(), "sto-3g", &b) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisCreateFromFile(h, h2.c_str(), nullptr, &b) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisCreateFromFile(h, nullptr, "sto-3g", &b) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisCreateFromFile(h, h2.c_str(), "sto-3g", nullptr) == CUEST_INVALID_VALUE);

  /* Angular momentum beyond d must be reported as NOT_SUPPORTED. */
  CHECK(cuestBasisCreateFromFile(h, h2.c_str(), "unsupported_l", &b) == CUEST_NOT_SUPPORTED);
  CHECK(b == nullptr);
  (void)unsupported;

  CHECK(cuestBasisCreateFromFile(h, h2.c_str(), "sto-3g", &b) == CUEST_SUCCESS);
  int nbf = 0;
  CHECK(cuestBasisNumFunctions(b, &nbf) == CUEST_SUCCESS);
  CHECK(nbf == 2);

  /* Bad precision, null outputs, too-small leading dimension. */
  std::vector<double> out(64, 0.0);
  CHECK(cuestOverlapMatrix(h, b, static_cast<cuestPrecision_t>(7), out.data(), nbf) ==
        CUEST_INVALID_VALUE);
  CHECK(cuestOverlapMatrix(h, b, CUEST_FP64, nullptr, nbf) == CUEST_INVALID_VALUE);
  CHECK(cuestOverlapMatrix(h, b, CUEST_FP64, out.data(), 1) == CUEST_INVALID_VALUE);
  CHECK(cuestKineticMatrix(nullptr, b, CUEST_FP64, out.data(), nbf) == CUEST_INVALID_VALUE);
  CHECK(cuestNuclearMatrix(h, nullptr, CUEST_FP64, out.data(), nbf) == CUEST_INVALID_VALUE);
  CHECK(cuestOverlapMatrix(h, b, CUEST_FP64, out.data(), nbf + 3) == CUEST_SUCCESS);

  /* Schwarz / ERI argument checks. */
  CHECK(cuestSchwarzBounds(h, b, nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestSchwarzBounds(nullptr, b, out.data()) == CUEST_INVALID_VALUE);
  std::vector<double> eri(static_cast<size_t>(nbf) * nbf * nbf * nbf, 0.0);
  CHECK(cuestERITensor(h, b, CUEST_FP64, -1.0, eri.data(), nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestERITensor(h, b, static_cast<cuestPrecision_t>(9), 0.0, eri.data(), nullptr) ==
        CUEST_INVALID_VALUE);
  CHECK(cuestERITensor(h, b, CUEST_FP64, 0.0, nullptr, nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestERITensor(h, b, CUEST_FP64, 0.0, eri.data(), nullptr) == CUEST_SUCCESS);

  /* Batch argument checks. */
  cuestQuartet_t q{0, 0, 0, 0};
  cuestQuartet_t bad{0, 0, 0, 99};
  CHECK(cuestERIBatch(h, b, &bad, 1, CUEST_FP64, out.data(), nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestERIBatch(h, b, nullptr, 1, CUEST_FP64, out.data(), nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestERIBatch(h, b, &q, -1, CUEST_FP64, out.data(), nullptr) == CUEST_INVALID_VALUE);
  CHECK(cuestERIBatch(h, b, &q, 1, static_cast<cuestPrecision_t>(5), out.data(), nullptr) ==
        CUEST_INVALID_VALUE);
  CHECK(cuestERIBatch(h, b, &q, 0, CUEST_FP64, nullptr, nullptr) == CUEST_SUCCESS);

  /* Repeated create/destroy: the sanitizer build checks that nothing leaks. */
  for (int i = 0; i < 200; ++i) {
    cuestHandle_t hh = nullptr;
    CHECK(cuestCreate(&hh) == CUEST_SUCCESS);
    cuestBasis_t bb = nullptr;
    CHECK(cuestBasisCreateFromFile(hh, h2.c_str(), "sto-3g", &bb) == CUEST_SUCCESS);
    CHECK(cuestBasisDestroy(bb) == CUEST_SUCCESS);
    CHECK(cuestDestroy(hh) == CUEST_SUCCESS);
  }

  CHECK(cuestBasisDestroy(b) == CUEST_SUCCESS);
  CHECK(cuestDestroy(h) == CUEST_SUCCESS);
  std::printf("test_error_handling: %s\n", g_failures == 0 ? "all checks passed" : "FAILURES");
  return g_failures == 0 ? 0 : 1;
}
