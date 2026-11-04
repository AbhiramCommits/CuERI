/**
 * @file cuest_kernels.cu
 * @brief CUDA kernels for one- and two-electron integrals.
 *
 * The kernels reuse the exact CUEST_HD headers compiled for the CPU path; no
 * integral mathematics is duplicated here.  The ERI kernel is templated on the
 * real type and on the four shell angular momenta so that register pressure and
 * loop bounds scale with the angular-momentum batch instead of the worst case
 * (d,d,d,d).  The host groups canonical quartets by class before launching.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda/cuda_backend.h"
#include "kernels/eri_md.cuh"
#include "kernels/eri_rys.cuh"
#include "kernels/md_recurrence.cuh"
#include "kernels/shell.cuh"

namespace cuest {
namespace {

__device__ inline void cart_component_dev(int l, int idx, int* lx, int* ly, int* lz) {
  int k = 0;
  for (int ix = l; ix >= 0; --ix)
    for (int iy = l - ix; iy >= 0; --iy) {
      if (k == idx) {
        *lx = ix;
        *ly = iy;
        *lz = l - ix - iy;
        return;
      }
      ++k;
    }
}

template <typename T>
__device__ inline T device_prim_eri(const DeviceBasis& db, int si, int sj, int sk, int sl, int ica,
                                    int icb, int icc, int icd, int ka, int kb, int kc, int kd,
                                    int pa, int pb, int pc, int pd, int algo) {
  const int Li = db.shell_L[si], Lj = db.shell_L[sj], Lk = db.shell_L[sk], Ll = db.shell_L[sl];
  int la[3], lb[3], lc[3], ld[3];
  cart_component_dev(Li, ka, &la[0], &la[1], &la[2]);
  cart_component_dev(Lj, kb, &lb[0], &lb[1], &lb[2]);
  cart_component_dev(Lk, kc, &lc[0], &lc[1], &lc[2]);
  cart_component_dev(Ll, kd, &ld[0], &ld[1], &ld[2]);

  T A[3], B[3], C[3], D[3];
  const double* ca = &db.centers[3 * db.shell_atom[si]];
  const double* cb = &db.centers[3 * db.shell_atom[sj]];
  const double* cc = &db.centers[3 * db.shell_atom[sk]];
  const double* cd = &db.centers[3 * db.shell_atom[sl]];
  for (int k = 0; k < 3; ++k) {
    A[k] = T(ca[k]);
    B[k] = T(cb[k]);
    C[k] = T(cc[k]);
    D[k] = T(cd[k]);
  }
  const T a = T(db.exps[db.prim_off[si] + pa]);
  const T b = T(db.exps[db.prim_off[sj] + pb]);
  const T c = T(db.exps[db.prim_off[sk] + pc]);
  const T d = T(db.exps[db.prim_off[sl] + pd]);

  T v;
  if (algo == 0)
    v = cuest_prim_eri_md<T>(la, lb, lc, ld, a, b, c, d, A, B, C, D);
  else
    v = cuest_prim_eri_rys<T>(la, lb, lc, ld, a, b, c, d, A, B, C, D);

  const int nci = cuest_num_cart(Li), ncj = cuest_num_cart(Lj);
  const int nck = cuest_num_cart(Lk), ncl = cuest_num_cart(Ll);
  const int ni = db.shell_nprim[si], nj = db.shell_nprim[sj];
  const int nk = db.shell_nprim[sk], nl = db.shell_nprim[sl];
  const T ca_v = T(db.coeffs[db.coeff_off[si] + ((ica * nci + ka) * ni + pa)]);
  const T cb_v = T(db.coeffs[db.coeff_off[sj] + ((icb * ncj + kb) * nj + pb)]);
  const T cc_v = T(db.coeffs[db.coeff_off[sk] + ((icc * nck + kc) * nk + pc)]);
  const T cd_v = T(db.coeffs[db.coeff_off[sl] + ((icd * ncl + kd) * nl + pd)]);
  return ca_v * cb_v * cc_v * cd_v * v;
}

template <typename T>
__device__ inline T device_prim_1e(const DeviceBasis& db, int si, int sj, int which, int ica,
                                   int icb, int ka, int kb, int pa, int pb) {
  const int Li = db.shell_L[si], Lj = db.shell_L[sj];
  int la[3], lb[3];
  cart_component_dev(Li, ka, &la[0], &la[1], &la[2]);
  cart_component_dev(Lj, kb, &lb[0], &lb[1], &lb[2]);
  T A[3], B[3];
  for (int k = 0; k < 3; ++k) {
    A[k] = T(db.centers[3 * db.shell_atom[si] + k]);
    B[k] = T(db.centers[3 * db.shell_atom[sj] + k]);
  }
  const T a = T(db.exps[db.prim_off[si] + pa]);
  const T b = T(db.exps[db.prim_off[sj] + pb]);
  const T p = a + b;
  T P[3], PA[3], PB[3], AB[3];
  for (int k = 0; k < 3; ++k) {
    P[k] = (a * A[k] + b * B[k]) / p;
    PA[k] = P[k] - A[k];
    PB[k] = P[k] - B[k];
    AB[k] = A[k] - B[k];
  }
  const T K = T(exp(-static_cast<double>(a) * static_cast<double>(b) / static_cast<double>(p) *
                    static_cast<double>(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2])));
  T val = T(0);
  if (which == 0) {
    val = cuest_prim_overlap<T>(la, lb, p, PA, PB, K);
  } else if (which == 1) {
    val = cuest_prim_kinetic<T>(la, lb, a, b, p, PA, PB, K);
  } else {
    for (int at = 0; at < db.natoms; ++at) {
      T PC[3];
      for (int k = 0; k < 3; ++k)
        PC[k] = P[k] - T(db.centers[3 * at + k]);
      val += cuest_prim_nuclear<T>(la, lb, p, PA, PB, K, PC, T(db.charges[at]));
    }
  }
  const int nci = cuest_num_cart(Li), ncj = cuest_num_cart(Lj);
  const T ca = T(db.coeffs[db.coeff_off[si] + ((ica * nci + ka) * db.shell_nprim[si] + pa)]);
  const T cb = T(db.coeffs[db.coeff_off[sj] + ((icb * ncj + kb) * db.shell_nprim[sj] + pb)]);
  return ca * cb * val;
}

__device__ inline void scatter8(DeviceBasis db, const QuartetDev& q, int fa, int fb, int fc, int fd,
                                double value, double* eri, int nbf) {
  const int sh[4] = {q.i, q.j, q.k, q.l};
  const int f[4] = {fa, fb, fc, fd};
  const int perms[8][4] = {{0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
                           {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};
  for (int ip = 0; ip < 8; ++ip) {
    const int* pm = perms[ip];
    const size_t g0 = static_cast<size_t>(db.shell_func_off[sh[pm[0]]] + f[pm[0]]);
    const size_t g1 = static_cast<size_t>(db.shell_func_off[sh[pm[1]]] + f[pm[1]]);
    const size_t g2 = static_cast<size_t>(db.shell_func_off[sh[pm[2]]] + f[pm[2]]);
    const size_t g3 = static_cast<size_t>(db.shell_func_off[sh[pm[3]]] + f[pm[3]]);
    eri[((g0 * nbf + g1) * nbf + g2) * nbf + g3] = value;
  }
}

template <typename T, int LA, int LB, int LC, int LD>
__global__ void cuest_eri_kernel(DeviceBasis db, const QuartetDev* quartets, int nq, double* eri,
                                 int nbf, int algo) {
  const int qi = blockIdx.x;
  if (qi >= nq)
    return;
  const QuartetDev q = quartets[qi];
  const int nca = cuest_num_cart(LA), ncb = cuest_num_cart(LB);
  const int ncc = cuest_num_cart(LC), ncd = cuest_num_cart(LD);
  const int ma = db.shell_nctr[q.i] * nca, mb = db.shell_nctr[q.j] * ncb;
  const int mc = db.shell_nctr[q.k] * ncc, md = db.shell_nctr[q.l] * ncd;
  const int total = ma * mb * mc * md;
  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int r = t;
    const int fd = r % md;
    r /= md;
    const int fc = r % mc;
    r /= mc;
    const int fb = r % mb;
    const int fa = r / mb;
    const int ica = fa / nca, ka = fa % nca;
    const int icb = fb / ncb, kb = fb % ncb;
    const int icc = fc / ncc, kc = fc % ncc;
    const int icd = fd / ncd, kd = fd % ncd;
    T val = T(0);
    for (int pa = 0; pa < db.shell_nprim[q.i]; ++pa)
      for (int pb = 0; pb < db.shell_nprim[q.j]; ++pb)
        for (int pc = 0; pc < db.shell_nprim[q.k]; ++pc)
          for (int pd = 0; pd < db.shell_nprim[q.l]; ++pd)
            val += device_prim_eri<T>(db, q.i, q.j, q.k, q.l, ica, icb, icc, icd, ka, kb, kc, kd,
                                      pa, pb, pc, pd, algo);
    scatter8(db, q, fa, fb, fc, fd, static_cast<double>(val), eri, nbf);
  }
}

template <typename T>
__global__ void cuest_one_electron_kernel(DeviceBasis db, int which, double* out, int ld,
                                          int npairs) {
  const int ip = blockIdx.x;
  if (ip >= npairs)
    return;
  /* Unordered shell pair list reconstructed on the fly: pair index ip maps to a
     unique (i>=j) pair. */
  int i = 0, j = 0, k = 0;
  for (i = 0; i < db.nshells; ++i) {
    const int n = i + 1;
    if (k + n > ip) {
      j = ip - k;
      break;
    }
    k += n;
  }
  if (i >= db.nshells)
    return;
  const int Li = db.shell_L[i], Lj = db.shell_L[j];
  const int nca = cuest_num_cart(Li), ncb = cuest_num_cart(Lj);
  const int ma = db.shell_nctr[i] * nca, mb = db.shell_nctr[j] * ncb;
  for (int t = threadIdx.x; t < ma * mb; t += blockDim.x) {
    const int fa = t / mb, fb = t % mb;
    const int ica = fa / nca, ka = fa % nca;
    const int icb = fb / ncb, kb = fb % ncb;
    T val = T(0);
    for (int pa = 0; pa < db.shell_nprim[i]; ++pa)
      for (int pb = 0; pb < db.shell_nprim[j]; ++pb)
        val += device_prim_1e<T>(db, i, j, which, ica, icb, ka, kb, pa, pb);
    const int row = db.shell_func_off[i] + fa;
    const int col = db.shell_func_off[j] + fb;
    out[static_cast<size_t>(row) * ld + col] = static_cast<double>(val);
    if (i != j)
      out[static_cast<size_t>(col) * ld + row] = static_cast<double>(val);
  }
}

__global__ void cuest_schwarz_kernel(DeviceBasis db, double* q_pair, int npairs) {
  const int ip = blockIdx.x;
  if (ip >= npairs)
    return;
  int i = 0, j = 0, k = 0;
  for (i = 0; i < db.nshells; ++i) {
    const int n = i + 1;
    if (k + n > ip) {
      j = ip - k;
      break;
    }
    k += n;
  }
  if (i >= db.nshells)
    return;
  const int Li = db.shell_L[i], Lj = db.shell_L[j];
  const int nca = cuest_num_cart(Li), ncb = cuest_num_cart(Lj);
  const int ma = db.shell_nctr[i] * nca, mb = db.shell_nctr[j] * ncb;
  double best = 0.0;
  for (int t = threadIdx.x; t < ma * mb * ma * mb; t += blockDim.x) {
    int r = t;
    const int fd = r % mb;
    r /= mb;
    const int fc = r % ma;
    r /= ma;
    const int fb = r % mb;
    const int fa = r / mb;
    const int ica = fa / nca, ka = fa % nca;
    const int icb = fb / ncb, kb = fb % ncb;
    const int icc = fc / nca, kc = fc % nca;
    const int icd = fd / ncb, kd = fd % ncb;
    double val = 0.0;
    for (int pa = 0; pa < db.shell_nprim[i]; ++pa)
      for (int pb = 0; pb < db.shell_nprim[j]; ++pb)
        val += static_cast<double>(device_prim_eri<double>(db, i, j, i, j, ica, icb, icc, icd, ka,
                                                           kb, kc, kd, pa, pb, pa, pb, 0));
    best = fmax(best, fabs(val));
  }
  /* block reduction */
  __shared__ double smem[256];
  smem[threadIdx.x] = best;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double q = sqrt(smem[0]);
    q_pair[static_cast<size_t>(i) * db.nshells + j] = q;
    q_pair[static_cast<size_t>(j) * db.nshells + i] = q;
  }
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Host-side launchers                                                 */
/* ------------------------------------------------------------------ */

int cuest_cuda_launch_one_electron(const DeviceBasis& db, int which, bool fp32, double* out, int ld,
                                   int nwork) {
  const int npairs = db.nshells * (db.nshells + 1) / 2;
  if (npairs <= 0)
    return 0;
  if (fp32)
    cuest_one_electron_kernel<float><<<npairs, 128>>>(db, which, out, ld, npairs);
  else
    cuest_one_electron_kernel<double><<<npairs, 128>>>(db, which, out, ld, npairs);
  (void)nwork;
  return static_cast<int>(cudaGetLastError());
}

int cuest_cuda_launch_schwarz(const DeviceBasis& db, double* q_pair) {
  const int npairs = db.nshells * (db.nshells + 1) / 2;
  if (npairs <= 0)
    return 0;
  cuest_schwarz_kernel<<<npairs, 128>>>(db, q_pair, npairs);
  return static_cast<int>(cudaGetLastError());
}

int cuest_cuda_launch_eri(const DeviceBasis& db, const QuartetDev* host_quartets, int nq, bool fp32,
                          int algo, double* eri, size_t* screened_out) {
  if (nq <= 0)
    return 0;
  QuartetDev* dq = nullptr;
  if (cudaMalloc(&dq, sizeof(QuartetDev) * nq) != cudaSuccess)
    return 1;
  if (cudaMemcpy(dq, host_quartets, sizeof(QuartetDev) * nq, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(dq);
    return 1;
  }
  /* One launch per angular-momentum class so that the template parameters match
     the actual shells and register pressure scales with the class. */
  for (int cls = 0; cls < 81; ++cls) {
    std::vector<QuartetDev> bucket;
    for (int k = 0; k < nq; ++k) {
      const int la = db.shell_L[host_quartets[k].i];
      const int lb = db.shell_L[host_quartets[k].j];
      const int lc = db.shell_L[host_quartets[k].k];
      const int ld = db.shell_L[host_quartets[k].l];
      if (((la * 3 + lb) * 3 + lc) * 3 + ld == cls)
        bucket.push_back(host_quartets[k]);
    }
    if (bucket.empty())
      continue;
    QuartetDev* dqb = nullptr;
    if (cudaMalloc(&dqb, sizeof(QuartetDev) * bucket.size()) != cudaSuccess) {
      cudaFree(dq);
      return 1;
    }
    cudaMemcpy(dqb, bucket.data(), sizeof(QuartetDev) * bucket.size(), cudaMemcpyHostToDevice);
    const int ni = static_cast<int>(bucket.size());
#define CUEST_LAUNCH(LA, LB, LC, LD)                                                         \
  do {                                                                                       \
    if (fp32)                                                                                \
      cuest_eri_kernel<float, LA, LB, LC, LD><<<ni, 128>>>(db, dqb, ni, eri, db.nbf, algo);  \
    else                                                                                     \
      cuest_eri_kernel<double, LA, LB, LC, LD><<<ni, 128>>>(db, dqb, ni, eri, db.nbf, algo); \
  } while (0)
    switch (cls) {
      case 0:
        CUEST_LAUNCH(0, 0, 0, 0);
        break;
      case 1:
        CUEST_LAUNCH(0, 0, 0, 1);
        break;
      case 2:
        CUEST_LAUNCH(0, 0, 0, 2);
        break;
      case 3:
        CUEST_LAUNCH(0, 0, 1, 0);
        break;
      case 4:
        CUEST_LAUNCH(0, 0, 1, 1);
        break;
      case 5:
        CUEST_LAUNCH(0, 0, 1, 2);
        break;
      case 6:
        CUEST_LAUNCH(0, 0, 2, 0);
        break;
      case 7:
        CUEST_LAUNCH(0, 0, 2, 1);
        break;
      case 8:
        CUEST_LAUNCH(0, 0, 2, 2);
        break;
      case 9:
        CUEST_LAUNCH(0, 1, 0, 0);
        break;
      case 10:
        CUEST_LAUNCH(0, 1, 0, 1);
        break;
      case 11:
        CUEST_LAUNCH(0, 1, 0, 2);
        break;
      case 12:
        CUEST_LAUNCH(0, 1, 1, 0);
        break;
      case 13:
        CUEST_LAUNCH(0, 1, 1, 1);
        break;
      case 14:
        CUEST_LAUNCH(0, 1, 1, 2);
        break;
      case 15:
        CUEST_LAUNCH(0, 1, 2, 0);
        break;
      case 16:
        CUEST_LAUNCH(0, 1, 2, 1);
        break;
      case 17:
        CUEST_LAUNCH(0, 1, 2, 2);
        break;
      case 18:
        CUEST_LAUNCH(0, 2, 0, 0);
        break;
      case 19:
        CUEST_LAUNCH(0, 2, 0, 1);
        break;
      case 20:
        CUEST_LAUNCH(0, 2, 0, 2);
        break;
      case 21:
        CUEST_LAUNCH(0, 2, 1, 0);
        break;
      case 22:
        CUEST_LAUNCH(0, 2, 1, 1);
        break;
      case 23:
        CUEST_LAUNCH(0, 2, 1, 2);
        break;
      case 24:
        CUEST_LAUNCH(0, 2, 2, 0);
        break;
      case 25:
        CUEST_LAUNCH(0, 2, 2, 1);
        break;
      case 26:
        CUEST_LAUNCH(0, 2, 2, 2);
        break;
      case 27:
        CUEST_LAUNCH(1, 0, 0, 0);
        break;
      case 28:
        CUEST_LAUNCH(1, 0, 0, 1);
        break;
      case 29:
        CUEST_LAUNCH(1, 0, 0, 2);
        break;
      case 30:
        CUEST_LAUNCH(1, 0, 1, 0);
        break;
      case 31:
        CUEST_LAUNCH(1, 0, 1, 1);
        break;
      case 32:
        CUEST_LAUNCH(1, 0, 1, 2);
        break;
      case 33:
        CUEST_LAUNCH(1, 0, 2, 0);
        break;
      case 34:
        CUEST_LAUNCH(1, 0, 2, 1);
        break;
      case 35:
        CUEST_LAUNCH(1, 0, 2, 2);
        break;
      case 36:
        CUEST_LAUNCH(1, 1, 0, 0);
        break;
      case 37:
        CUEST_LAUNCH(1, 1, 0, 1);
        break;
      case 38:
        CUEST_LAUNCH(1, 1, 0, 2);
        break;
      case 39:
        CUEST_LAUNCH(1, 1, 1, 0);
        break;
      case 40:
        CUEST_LAUNCH(1, 1, 1, 1);
        break;
      case 41:
        CUEST_LAUNCH(1, 1, 1, 2);
        break;
      case 42:
        CUEST_LAUNCH(1, 1, 2, 0);
        break;
      case 43:
        CUEST_LAUNCH(1, 1, 2, 1);
        break;
      case 44:
        CUEST_LAUNCH(1, 1, 2, 2);
        break;
      case 45:
        CUEST_LAUNCH(1, 2, 0, 0);
        break;
      case 46:
        CUEST_LAUNCH(1, 2, 0, 1);
        break;
      case 47:
        CUEST_LAUNCH(1, 2, 0, 2);
        break;
      case 48:
        CUEST_LAUNCH(1, 2, 1, 0);
        break;
      case 49:
        CUEST_LAUNCH(1, 2, 1, 1);
        break;
      case 50:
        CUEST_LAUNCH(1, 2, 1, 2);
        break;
      case 51:
        CUEST_LAUNCH(1, 2, 2, 0);
        break;
      case 52:
        CUEST_LAUNCH(1, 2, 2, 1);
        break;
      case 53:
        CUEST_LAUNCH(1, 2, 2, 2);
        break;
      case 54:
        CUEST_LAUNCH(2, 0, 0, 0);
        break;
      case 55:
        CUEST_LAUNCH(2, 0, 0, 1);
        break;
      case 56:
        CUEST_LAUNCH(2, 0, 0, 2);
        break;
      case 57:
        CUEST_LAUNCH(2, 0, 1, 0);
        break;
      case 58:
        CUEST_LAUNCH(2, 0, 1, 1);
        break;
      case 59:
        CUEST_LAUNCH(2, 0, 1, 2);
        break;
      case 60:
        CUEST_LAUNCH(2, 0, 2, 0);
        break;
      case 61:
        CUEST_LAUNCH(2, 0, 2, 1);
        break;
      case 62:
        CUEST_LAUNCH(2, 0, 2, 2);
        break;
      case 63:
        CUEST_LAUNCH(2, 1, 0, 0);
        break;
      case 64:
        CUEST_LAUNCH(2, 1, 0, 1);
        break;
      case 65:
        CUEST_LAUNCH(2, 1, 0, 2);
        break;
      case 66:
        CUEST_LAUNCH(2, 1, 1, 0);
        break;
      case 67:
        CUEST_LAUNCH(2, 1, 1, 1);
        break;
      case 68:
        CUEST_LAUNCH(2, 1, 1, 2);
        break;
      case 69:
        CUEST_LAUNCH(2, 1, 2, 0);
        break;
      case 70:
        CUEST_LAUNCH(2, 1, 2, 1);
        break;
      case 71:
        CUEST_LAUNCH(2, 1, 2, 2);
        break;
      case 72:
        CUEST_LAUNCH(2, 2, 0, 0);
        break;
      case 73:
        CUEST_LAUNCH(2, 2, 0, 1);
        break;
      case 74:
        CUEST_LAUNCH(2, 2, 0, 2);
        break;
      case 75:
        CUEST_LAUNCH(2, 2, 1, 0);
        break;
      case 76:
        CUEST_LAUNCH(2, 2, 1, 1);
        break;
      case 77:
        CUEST_LAUNCH(2, 2, 1, 2);
        break;
      case 78:
        CUEST_LAUNCH(2, 2, 2, 0);
        break;
      case 79:
        CUEST_LAUNCH(2, 2, 2, 1);
        break;
      case 80:
        CUEST_LAUNCH(2, 2, 2, 2);
        break;
      default:
        break;
    }
#undef CUEST_LAUNCH
    cudaFree(dqb);
  }
  cudaFree(dq);
  if (screened_out)
    *screened_out = 0;
  return static_cast<int>(cudaGetLastError());
}

}  // namespace cuest
