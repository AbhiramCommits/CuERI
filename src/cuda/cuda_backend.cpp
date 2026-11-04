/**
 * @file cuda_backend.cpp
 * @brief CUDA backend host driver: device buffers, async transfers, event timing.
 */
#include "cuda/cuda_backend.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace cuest {
namespace {

struct DeviceBuffers {
  int *shell_L = nullptr;
  int *shell_atom = nullptr;
  int *shell_nprim = nullptr;
  int *shell_nctr = nullptr;
  int *shell_func_off = nullptr;
  int *prim_off = nullptr;
  int *coeff_off = nullptr;
  double *exps = nullptr;
  double *coeffs = nullptr;
  double *centers = nullptr;
  double *charges = nullptr;
  size_t nexps = 0, ncoeff = 0;

  ~DeviceBuffers() { release(); }

  void release() {
    cudaFree(shell_L);
    cudaFree(shell_atom);
    cudaFree(shell_nprim);
    cudaFree(shell_nctr);
    cudaFree(shell_func_off);
    cudaFree(prim_off);
    cudaFree(coeff_off);
    cudaFree(exps);
    cudaFree(coeffs);
    cudaFree(centers);
    cudaFree(charges);
    shell_L = shell_atom = shell_nprim = shell_nctr = shell_func_off = nullptr;
    prim_off = coeff_off = nullptr;
    exps = coeffs = centers = charges = nullptr;
  }
};

template <typename T>
bool upload(T *&dst, const std::vector<T> &src) {
  if (src.empty()) {
    dst = nullptr;
    return true;
  }
  if (cudaMalloc(&dst, sizeof(T) * src.size()) != cudaSuccess) return false;
  return cudaMemcpy(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool build_device_basis(const BasisData &basis, DeviceBuffers &buf, DeviceBasis &db) {
  std::vector<int> shell_L, shell_atom, shell_nprim, shell_nctr, shell_func_off, prim_off,
      coeff_off;
  std::vector<double> exps, coeffs;
  int eoff = 0, coff = 0;
  for (const PrimShell &s : basis.shells) {
    shell_L.push_back(s.L);
    shell_atom.push_back(s.atom);
    shell_nprim.push_back(s.nprim);
    shell_nctr.push_back(s.nctr);
    shell_func_off.push_back(s.func_offset);
    prim_off.push_back(eoff);
    coeff_off.push_back(coff);
    for (double e : s.exps) exps.push_back(e);
    for (double c : s.cn) coeffs.push_back(c);
    eoff += s.nprim;
    coff += static_cast<int>(s.cn.size());
  }
  db.nshells = basis.nshells;
  db.nbf = basis.nbf;
  db.natoms = basis.natoms;
  db.shell_L = nullptr;
  if (!upload(buf.shell_L, shell_L) || !upload(buf.shell_atom, shell_atom) ||
      !upload(buf.shell_nprim, shell_nprim) || !upload(buf.shell_nctr, shell_nctr) ||
      !upload(buf.shell_func_off, shell_func_off) || !upload(buf.prim_off, prim_off) ||
      !upload(buf.coeff_off, coeff_off) || !upload(buf.exps, exps) ||
      !upload(buf.coeffs, coeffs) || !upload(buf.centers, basis.atom_xyz) ||
      !upload(buf.charges, basis.atom_Z))
    return false;
  db.shell_L = buf.shell_L;
  db.shell_atom = buf.shell_atom;
  db.shell_nprim = buf.shell_nprim;
  db.shell_nctr = buf.shell_nctr;
  db.shell_func_off = buf.shell_func_off;
  db.prim_off = buf.prim_off;
  db.coeff_off = buf.coeff_off;
  db.exps = buf.exps;
  db.coeffs = buf.coeffs;
  db.centers = buf.centers;
  db.charges = buf.charges;
  return true;
}

}  // namespace

cuestStatus_t cuda_one_electron(const BasisData &basis, int which, cuestPrecision_t precision,
                                double *out, int ld) {
  DeviceBuffers buf;
  DeviceBasis db;
  if (!build_device_basis(basis, buf, db)) return CUEST_OUT_OF_MEMORY;
  const size_t n = static_cast<size_t>(basis.nbf) * ld;
  double *d_out = nullptr;
  if (cudaMalloc(&d_out, sizeof(double) * n) != cudaSuccess) return CUEST_OUT_OF_MEMORY;
  cudaMemset(d_out, 0, sizeof(double) * n);
  const int rc = cuest_cuda_launch_one_electron(db, which, precision == CUEST_FP32, d_out, ld,
                                                basis.nbf);
  if (rc != 0) {
    cudaFree(d_out);
    return CUEST_INTERNAL_ERROR;
  }
  if (cudaMemcpy2D(out, sizeof(double) * ld, d_out, sizeof(double) * ld,
                   sizeof(double) * basis.nbf, basis.nbf, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_out);
    return CUEST_INTERNAL_ERROR;
  }
  cudaFree(d_out);
  return CUEST_SUCCESS;
}

cuestStatus_t cuda_schwarz(const BasisData &basis, double *q_pair) {
  DeviceBuffers buf;
  DeviceBasis db;
  if (!build_device_basis(basis, buf, db)) return CUEST_OUT_OF_MEMORY;
  const size_t n = static_cast<size_t>(basis.nshells) * basis.nshells;
  double *d_q = nullptr;
  if (cudaMalloc(&d_q, sizeof(double) * n) != cudaSuccess) return CUEST_OUT_OF_MEMORY;
  cudaMemset(d_q, 0, sizeof(double) * n);
  const int rc = cuest_cuda_launch_schwarz(db, d_q);
  if (rc != 0) {
    cudaFree(d_q);
    return CUEST_INTERNAL_ERROR;
  }
  cudaMemcpy(q_pair, d_q, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  return CUEST_SUCCESS;
}

cuestStatus_t cuda_eri_tensor(const BasisData &basis, cuestPrecision_t precision,
                              cuestEriAlgo_t algo, double screen_tol, double *eri,
                              cuestEriStats_t *stats) {
  DeviceBuffers buf;
  DeviceBasis db;
  if (!build_device_basis(basis, buf, db)) return CUEST_OUT_OF_MEMORY;

  std::vector<ShellPair> pairs;
  build_shell_pairs(basis, pairs, algo);
  std::vector<QuartetDev> quartets;
  size_t total = 0, screened = 0;
  for (size_t a = 0; a < pairs.size(); ++a) {
    for (size_t b = 0; b <= a; ++b) {
      ++total;
      if (screen_tol > 0.0 && pairs[a].q * pairs[b].q < screen_tol) {
        ++screened;
        continue;
      }
      quartets.push_back({pairs[a].i, pairs[a].j, pairs[b].i, pairs[b].j});
    }
  }

  const size_t n = static_cast<size_t>(basis.nbf);
  double *d_eri = nullptr;
  if (cudaMalloc(&d_eri, sizeof(double) * n * n * n * n) != cudaSuccess) return CUEST_OUT_OF_MEMORY;
  cudaMemset(d_eri, 0, sizeof(double) * n * n * n * n);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  size_t ignored = 0;
  const int rc = cuest_cuda_launch_eri(db, quartets.data(), static_cast<int>(quartets.size()),
                                       precision == CUEST_FP32, static_cast<int>(algo), d_eri,
                                       &ignored);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  if (rc != 0) {
    cudaFree(d_eri);
    return CUEST_INTERNAL_ERROR;
  }
  cudaMemcpy(eri, d_eri, sizeof(double) * n * n * n * n, cudaMemcpyDeviceToHost);
  cudaFree(d_eri);
  if (stats) {
    stats->quartets_total = total;
    stats->quartets_screened = screened;
    stats->primitive_pairs = 0;
    stats->kernel_ms = static_cast<double>(ms);
    stats->max_shell_am = basis.max_am;
  }
  return CUEST_SUCCESS;
}

cuestStatus_t cuda_eri_batch(const BasisData &basis, const cuestQuartet_t *quartets, int nquartets,
                             cuestPrecision_t precision, cuestEriAlgo_t algo, double *out,
                             const size_t *offsets) {
  (void)basis;
  (void)quartets;
  (void)nquartets;
  (void)precision;
  (void)algo;
  (void)out;
  (void)offsets;
  /* The batch entry point packs variable-size blocks contiguously, which the
     scatter-based device kernel does not produce.  The CPU backend implements
     it; the CUDA backend reports the gap instead of returning wrong data. */
  return CUEST_NOT_SUPPORTED;
}

}  // namespace cuest
