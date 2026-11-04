/**
 * @file test_api.cpp
 * @brief C API contract: handle lifecycle, error strings and tier reporting.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "cuest/cuest_eri.h"

static int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

int main() {
  int major = -1, minor = -1, patch = -1;
  CHECK(cuestGetVersion(&major, &minor, &patch) == CUEST_SUCCESS);
  CHECK(major == CUEST_VERSION_MAJOR && minor == CUEST_VERSION_MINOR && patch == CUEST_VERSION_PATCH);
  CHECK(cuestGetVersion(nullptr, nullptr, nullptr) == CUEST_SUCCESS);

  /* Error-string coverage for every declared status code. */
  const cuestStatus_t all[] = {CUEST_SUCCESS,       CUEST_INVALID_VALUE, CUEST_NOT_SUPPORTED,
                               CUEST_OUT_OF_MEMORY, CUEST_NUMERICAL_ERROR, CUEST_INTERNAL_ERROR};
  for (cuestStatus_t s : all) {
    const char *msg = nullptr;
    CHECK(cuestGetErrorString(s, &msg) == CUEST_SUCCESS);
    CHECK(msg != nullptr && std::strlen(msg) > 0);
  }
  const char *unknown = nullptr;
  CHECK(cuestGetErrorString(static_cast<cuestStatus_t>(99), &unknown) == CUEST_SUCCESS);
  CHECK(unknown != nullptr && std::strlen(unknown) > 0);
  CHECK(cuestGetErrorString(CUEST_SUCCESS, nullptr) == CUEST_INVALID_VALUE);

  /* Handle lifecycle. */
  CHECK(cuestCreate(nullptr) == CUEST_INVALID_VALUE);
  cuestHandle_t handle = nullptr;
  CHECK(cuestCreate(&handle) == CUEST_SUCCESS);
  CHECK(handle != nullptr);

  cuestBackend_t backend = CUEST_BACKEND_CUDA;
  CHECK(cuestGetBackend(handle, &backend) == CUEST_SUCCESS);
  CHECK(cuestGetBackend(nullptr, &backend) == CUEST_INVALID_VALUE);
  CHECK(cuestGetBackend(handle, nullptr) == CUEST_INVALID_VALUE);
#if CUEST_EXPECT_CPU_ONLY
  CHECK(backend == CUEST_BACKEND_CPU);
  std::printf("test_api: backend = CPU (Tier C build, no CUDA toolkit)\n");
#else
  std::printf("test_api: backend = %s\n", backend == CUEST_BACKEND_CUDA ? "CUDA" : "CPU");
#endif

  CHECK(cuestSetEriAlgo(handle, CUEST_ERI_ALGO_RYS) == CUEST_SUCCESS);
  CHECK(cuestSetEriAlgo(handle, CUEST_ERI_ALGO_MD) == CUEST_SUCCESS);
  CHECK(cuestSetEriAlgo(handle, static_cast<cuestEriAlgo_t>(7)) == CUEST_INVALID_VALUE);
  CHECK(cuestSetEriAlgo(nullptr, CUEST_ERI_ALGO_MD) == CUEST_INVALID_VALUE);

  /* Basis construction from the committed molecule JSON. */
  const std::string mol = std::string(CUEST_SOURCE_DIR) + "/data/molecules/h2.json";
  cuestBasis_t basis = nullptr;
  CHECK(cuestBasisCreateFromFile(handle, mol.c_str(), "sto-3g", &basis) == CUEST_SUCCESS);
  CHECK(basis != nullptr);
  int nbf = -1, nshells = -1;
  CHECK(cuestBasisNumFunctions(basis, &nbf) == CUEST_SUCCESS);
  CHECK(cuestBasisNumShells(basis, &nshells) == CUEST_SUCCESS);
  CHECK(nbf == 2);
  CHECK(nshells == 2);

  int atom = -1, am = -1, nprim = -1, nctr = -1, off = -1;
  CHECK(cuestBasisShellInfo(basis, 0, &atom, &am, &nprim, &nctr, &off) == CUEST_SUCCESS);
  CHECK(atom == 0 && am == 0 && nprim == 3 && nctr == 1 && off == 0);
  CHECK(cuestBasisShellInfo(basis, 1, &atom, &am, nullptr, nullptr, &off) == CUEST_SUCCESS);
  CHECK(atom == 1 && am == 0 && off == 1);
  CHECK(cuestBasisShellInfo(basis, 2, &atom, &am, &nprim, &nctr, &off) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisShellInfo(basis, -1, &atom, &am, &nprim, &nctr, &off) == CUEST_INVALID_VALUE);

  CHECK(cuestBasisNumFunctions(nullptr, &nbf) == CUEST_INVALID_VALUE);
  CHECK(cuestBasisNumFunctions(basis, nullptr) == CUEST_INVALID_VALUE);

  /* Invalid basis names and missing files must fail cleanly. */
  cuestBasis_t bad = nullptr;
  CHECK(cuestBasisCreateFromFile(handle, mol.c_str(), "no-such-basis", &bad) == CUEST_INVALID_VALUE);
  CHECK(bad == nullptr);
  CHECK(cuestBasisCreateFromFile(handle, "/nonexistent/molecule.json", "sto-3g", &bad) ==
        CUEST_INVALID_VALUE);
  CHECK(cuestBasisCreateFromFile(nullptr, mol.c_str(), "sto-3g", &bad) == CUEST_INVALID_VALUE);

  CHECK(cuestBasisDestroy(basis) == CUEST_SUCCESS);
  CHECK(cuestBasisDestroy(nullptr) == CUEST_SUCCESS);
  CHECK(cuestDestroy(handle) == CUEST_SUCCESS);
  CHECK(cuestDestroy(nullptr) == CUEST_SUCCESS);

  if (g_failures == 0) std::printf("test_api: all checks passed\n");
  return g_failures == 0 ? 0 : 1;
}
