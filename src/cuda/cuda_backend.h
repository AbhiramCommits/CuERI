/**
 * @file cuda_backend.h
 * @brief CUDA backend interface (compiled only when a CUDA toolkit is present).
 */
#ifndef CUEST_CUDA_BACKEND_H
#define CUEST_CUDA_BACKEND_H

#include "cuest_internal.h"

namespace cuest {

cuestStatus_t cuda_one_electron(const BasisData &basis, int which, cuestPrecision_t precision,
                                double *out, int ld);
cuestStatus_t cuda_schwarz(const BasisData &basis, double *q_pair);
cuestStatus_t cuda_eri_batch(const BasisData &basis, const cuestQuartet_t *quartets, int nquartets,
                             cuestPrecision_t precision, cuestEriAlgo_t algo, double *out,
                             const size_t *offsets);
cuestStatus_t cuda_eri_tensor(const BasisData &basis, cuestPrecision_t precision,
                              cuestEriAlgo_t algo, double screen_tol, double *eri,
                              cuestEriStats_t *stats);

/**
 * @brief Device description of a basis set in a flat SoA layout.
 *
 * All pointers refer to device memory.  Coefficients are kept in double
 * precision and cast to the kernel real type at use, so one device copy serves
 * both the fp32 and fp64 instantiations.
 */
struct DeviceBasis {
  int nshells = 0;
  int nbf = 0;
  int natoms = 0;
  const int *shell_L = nullptr;
  const int *shell_atom = nullptr;
  const int *shell_nprim = nullptr;
  const int *shell_nctr = nullptr;
  const int *shell_func_off = nullptr;
  const int *prim_off = nullptr;
  const int *coeff_off = nullptr;
  const double *exps = nullptr;
  const double *coeffs = nullptr;
  const double *centers = nullptr;
  const double *charges = nullptr;
};

struct QuartetDev {
  int i, j, k, l;
};

/* Launchers implemented in cuest_kernels.cu (plain C++ entry points so the
   kernel launch lives in the same translation unit as the kernel). */
int cuest_cuda_launch_one_electron(const DeviceBasis &db, int which, bool fp32, double *out, int ld,
                                   int nwork);
int cuest_cuda_launch_schwarz(const DeviceBasis &db, double *q_pair);
int cuest_cuda_launch_eri(const DeviceBasis &db, const QuartetDev *host_quartets, int nq, bool fp32,
                          int algo, double *eri, size_t *screened_out);

}  // namespace cuest

#endif /* CUEST_CUDA_BACKEND_H */
