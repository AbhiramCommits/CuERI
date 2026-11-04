/**
 * @file cuest_api.cpp
 * @brief C API implementation: handles, basis objects, validation and dispatch.
 *
 * No exception is allowed to cross the C boundary; every entry point catches
 * allocation and unexpected failures and maps them to a cuestStatus_t.
 */
#include <cmath>
#include <cstring>
#include <new>
#include <utility>

#include "cuest_internal.h"

#if defined(CUEST_WITH_CUDA)
#include <cuda_runtime.h>

#include "cuda/cuda_backend.h"
#endif

namespace {

cuestStatus_t check_handle(cuestHandle_t handle) { return handle ? CUEST_SUCCESS : CUEST_INVALID_VALUE; }

cuestStatus_t dispatch_one_electron(cuestHandle_t handle, cuestBasis_t basis, int which,
                                    cuestPrecision_t precision, double *out, int ld) {
  if (!handle || !basis) return CUEST_INVALID_VALUE;
  if (precision != CUEST_FP32 && precision != CUEST_FP64) return CUEST_INVALID_VALUE;
  if (!out || ld < basis->data.nbf) return CUEST_INVALID_VALUE;
  if (handle->backend == CUEST_BACKEND_CUDA) {
#if defined(CUEST_WITH_CUDA)
    return cuest::cuda_one_electron(basis->data, which, precision, out, ld);
#else
    return CUEST_NOT_SUPPORTED;
#endif
  }
  return cuest::cpu_one_electron(basis->data, which, precision, out, ld);
}

}  // namespace

extern "C" {

cuestStatus_t cuestCreate(cuestHandle_t *handle) {
  if (!handle) return CUEST_INVALID_VALUE;
  try {
    cuest_handle_t *h = new cuest_handle_t();
    h->algo = CUEST_ERI_ALGO_MD;
    h->num_threads = 1;
#if defined(CUEST_WITH_CUDA)
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    h->backend = (err == cudaSuccess && count > 0) ? CUEST_BACKEND_CUDA : CUEST_BACKEND_CPU;
#else
    h->backend = CUEST_BACKEND_CPU;
#endif
    *handle = h;
    return CUEST_SUCCESS;
  } catch (const std::bad_alloc &) {
    return CUEST_OUT_OF_MEMORY;
  } catch (...) {
    return CUEST_INTERNAL_ERROR;
  }
}

cuestStatus_t cuestDestroy(cuestHandle_t handle) {
  if (!handle) return CUEST_SUCCESS;
  delete handle;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestGetBackend(cuestHandle_t handle, cuestBackend_t *backend) {
  if (!handle || !backend) return CUEST_INVALID_VALUE;
  *backend = handle->backend;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestSetEriAlgo(cuestHandle_t handle, cuestEriAlgo_t algo) {
  if (!handle) return CUEST_INVALID_VALUE;
  if (algo != CUEST_ERI_ALGO_MD && algo != CUEST_ERI_ALGO_RYS) return CUEST_INVALID_VALUE;
  handle->algo = algo;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestGetVersion(int *major, int *minor, int *patch) {
  if (major) *major = CUEST_VERSION_MAJOR;
  if (minor) *minor = CUEST_VERSION_MINOR;
  if (patch) *patch = CUEST_VERSION_PATCH;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestGetErrorString(cuestStatus_t status, const char **msg) {
  if (!msg) return CUEST_INVALID_VALUE;
  switch (status) {
    case CUEST_SUCCESS: *msg = "success"; break;
    case CUEST_INVALID_VALUE: *msg = "invalid value"; break;
    case CUEST_NOT_SUPPORTED: *msg = "not supported"; break;
    case CUEST_OUT_OF_MEMORY: *msg = "out of memory"; break;
    case CUEST_NUMERICAL_ERROR: *msg = "numerical error"; break;
    case CUEST_INTERNAL_ERROR: *msg = "internal error"; break;
    default: *msg = "unknown status"; break;
  }
  return CUEST_SUCCESS;
}

cuestStatus_t cuestBasisCreateFromFile(cuestHandle_t handle, const char *molecule_json,
                                       const char *basis_name, cuestBasis_t *basis) {
  if (!handle || !molecule_json || !basis_name || !basis) return CUEST_INVALID_VALUE;
  try {
    cuest::Molecule mol;
    std::string err;
    if (!cuest::load_molecule(molecule_json, mol, err)) return CUEST_INVALID_VALUE;
    cuest::BasisData data;
    if (!cuest::build_basis(mol, basis_name, data, err)) {
      if (err.find("angular momentum") != std::string::npos) return CUEST_NOT_SUPPORTED;
      if (err.find("cannot open") != std::string::npos) return CUEST_INVALID_VALUE;
      return CUEST_INVALID_VALUE;
    }
    cuest_basis_t *b = new cuest_basis_t();
    b->data = std::move(data);
    *basis = b;
    return CUEST_SUCCESS;
  } catch (const std::bad_alloc &) {
    return CUEST_OUT_OF_MEMORY;
  } catch (...) {
    return CUEST_INTERNAL_ERROR;
  }
}

cuestStatus_t cuestBasisDestroy(cuestBasis_t basis) {
  if (!basis) return CUEST_SUCCESS;
  delete basis;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestBasisNumFunctions(cuestBasis_t basis, int *nbf) {
  if (!basis || !nbf) return CUEST_INVALID_VALUE;
  *nbf = basis->data.nbf;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestBasisNumShells(cuestBasis_t basis, int *nshells) {
  if (!basis || !nshells) return CUEST_INVALID_VALUE;
  *nshells = basis->data.nshells;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestBasisShellInfo(cuestBasis_t basis, int ishell, int *atom, int *am, int *nprim,
                                 int *nctr, int *func_offset) {
  if (!basis) return CUEST_INVALID_VALUE;
  if (ishell < 0 || ishell >= basis->data.nshells) return CUEST_INVALID_VALUE;
  const cuest::PrimShell &s = basis->data.shells[static_cast<size_t>(ishell)];
  if (atom) *atom = s.atom;
  if (am) *am = s.L;
  if (nprim) *nprim = s.nprim;
  if (nctr) *nctr = s.nctr;
  if (func_offset) *func_offset = s.func_offset;
  return CUEST_SUCCESS;
}

cuestStatus_t cuestOverlapMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                 cuestPrecision_t precision, double *out, int ld) {
  return dispatch_one_electron(handle, basis, 0, precision, out, ld);
}

cuestStatus_t cuestKineticMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                 cuestPrecision_t precision, double *out, int ld) {
  return dispatch_one_electron(handle, basis, 1, precision, out, ld);
}

cuestStatus_t cuestNuclearMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                 cuestPrecision_t precision, double *out, int ld) {
  return dispatch_one_electron(handle, basis, 2, precision, out, ld);
}

cuestStatus_t cuestSchwarzBounds(cuestHandle_t handle, cuestBasis_t basis, double *q_pair) {
  if (check_handle(handle) != CUEST_SUCCESS || !basis) return CUEST_INVALID_VALUE;
  if (!q_pair) return CUEST_INVALID_VALUE;
  if (handle->backend == CUEST_BACKEND_CUDA) {
#if defined(CUEST_WITH_CUDA)
    return cuest::cuda_schwarz(basis->data, q_pair);
#else
    return CUEST_NOT_SUPPORTED;
#endif
  }
  return cuest::cpu_schwarz(basis->data, q_pair);
}

cuestStatus_t cuestERIBatch(cuestHandle_t handle, cuestBasis_t basis,
                            const cuestQuartet_t *quartets, int nquartets,
                            cuestPrecision_t precision, double *out, const size_t *offsets) {
  if (!handle || !basis) return CUEST_INVALID_VALUE;
  if (precision != CUEST_FP32 && precision != CUEST_FP64) return CUEST_INVALID_VALUE;
  if (handle->backend == CUEST_BACKEND_CUDA) {
#if defined(CUEST_WITH_CUDA)
    return cuest::cuda_eri_batch(basis->data, quartets, nquartets, precision, handle->algo, out,
                                 offsets);
#else
    return CUEST_NOT_SUPPORTED;
#endif
  }
  return cuest::cpu_eri_batch(basis->data, quartets, nquartets, precision, handle->algo, out,
                              offsets);
}

cuestStatus_t cuestERITensor(cuestHandle_t handle, cuestBasis_t basis, cuestPrecision_t precision,
                             double screen_tol, double *eri_nbf4, cuestEriStats_t *stats) {
  if (!handle || !basis) return CUEST_INVALID_VALUE;
  if (precision != CUEST_FP32 && precision != CUEST_FP64) return CUEST_INVALID_VALUE;
  if (screen_tol < 0.0) return CUEST_INVALID_VALUE;
  if (handle->backend == CUEST_BACKEND_CUDA) {
#if defined(CUEST_WITH_CUDA)
    return cuest::cuda_eri_tensor(basis->data, precision, handle->algo, screen_tol, eri_nbf4, stats);
#else
    return CUEST_NOT_SUPPORTED;
#endif
  }
  return cuest::cpu_eri_tensor(basis->data, precision, handle->algo, screen_tol, eri_nbf4, stats);
}

}  // extern "C"
