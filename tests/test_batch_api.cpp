/**
 * @file test_batch_api.cpp
 * @brief cuestERIBatch offsets and batch-vs-tensor cross-check.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_internal.h"

int main() {
  int failures = 0;
  cuestHandle_t h = nullptr;
  if (cuestCreate(&h) != CUEST_SUCCESS)
    return 1;
  const std::string molpath = std::string(CUEST_SOURCE_DIR) + "/data/molecules/h2o.json";
  cuestBasis_t b = nullptr;
  if (cuestBasisCreateFromFile(h, molpath.c_str(), "6-31g*", &b) != CUEST_SUCCESS)
    return 1;
  cuest::BasisData basis;
  {
    cuest::Molecule mol;
    std::string err;
    if (!cuest::load_molecule(molpath, mol, err) || !cuest::build_basis(mol, "6-31g*", basis, err))
      return 1;
  }
  const int nbf = basis.nbf;
  const size_t n4 = static_cast<size_t>(nbf) * nbf * nbf * nbf;
  std::vector<double> tensor(n4, 0.0);
  if (cuestERITensor(h, b, CUEST_FP64, 0.0, tensor.data(), nullptr) != CUEST_SUCCESS)
    return 1;

  /* Every shell quartet, including permutations, as a batch. */
  std::vector<cuestQuartet_t> quartets;
  for (int i = 0; i < basis.nshells; ++i)
    for (int j = 0; j < basis.nshells; ++j)
      for (int k = 0; k < basis.nshells; ++k)
        for (int l = 0; l < basis.nshells; ++l)
          quartets.push_back({i, j, k, l});

  std::vector<size_t> offsets(quartets.size(), 0);
  size_t total = 0;
  for (size_t iq = 0; iq < quartets.size(); ++iq) {
    offsets[iq] = total;
    const cuest::PrimShell& sa = basis.shells[static_cast<size_t>(quartets[iq].i)];
    const cuest::PrimShell& sb = basis.shells[static_cast<size_t>(quartets[iq].j)];
    const cuest::PrimShell& sc = basis.shells[static_cast<size_t>(quartets[iq].k)];
    const cuest::PrimShell& sd = basis.shells[static_cast<size_t>(quartets[iq].l)];
    total += static_cast<size_t>(sa.nctr * sa.ncomp()) * (sb.nctr * sb.ncomp()) *
             (sc.nctr * sc.ncomp()) * (sd.nctr * sd.ncomp());
  }
  std::vector<double> batch(total, 0.0);
  if (cuestERIBatch(h, b, quartets.data(), static_cast<int>(quartets.size()), CUEST_FP64,
                    batch.data(), offsets.data()) != CUEST_SUCCESS)
    return 1;

  double worst = 0.0;
  for (size_t iq = 0; iq < quartets.size(); ++iq) {
    const cuestQuartet_t& q = quartets[iq];
    const cuest::PrimShell& sa = basis.shells[static_cast<size_t>(q.i)];
    const cuest::PrimShell& sb = basis.shells[static_cast<size_t>(q.j)];
    const cuest::PrimShell& sc = basis.shells[static_cast<size_t>(q.k)];
    const cuest::PrimShell& sd = basis.shells[static_cast<size_t>(q.l)];
    const int ma = sa.nctr * sa.ncomp(), mb = sb.nctr * sb.ncomp();
    const int mc = sc.nctr * sc.ncomp(), md = sd.nctr * sd.ncomp();
    size_t pos = offsets[iq];
    for (int fa = 0; fa < ma; ++fa)
      for (int fb = 0; fb < mb; ++fb)
        for (int fc = 0; fc < mc; ++fc)
          for (int fd = 0; fd < md; ++fd) {
            const size_t t =
                ((static_cast<size_t>(sa.func_offset + fa) * nbf + (sb.func_offset + fb)) * nbf +
                 (sc.func_offset + fc)) *
                    nbf +
                (sd.func_offset + fd);
            worst = std::max(worst, std::fabs(batch[pos++] - tensor[t]));
          }
  }
  std::printf("test_batch_api: %zu quartets, %zu values, max |batch - tensor| = %.3e\n",
              quartets.size(), total, worst);
  if (worst > 1e-14)
    ++failures;

  /* Contiguous packing (offsets == NULL) must agree with the explicit offsets. */
  std::vector<double> packed(total, 0.0);
  if (cuestERIBatch(h, b, quartets.data(), 3, CUEST_FP64, packed.data(), nullptr) != CUEST_SUCCESS)
    return 1;
  for (size_t k = 0; k < offsets[3]; ++k) {
    if (std::fabs(packed[k] - batch[k]) > 0.0) {
      std::printf("FAIL: contiguous packing differs at %zu\n", k);
      ++failures;
      break;
    }
  }

  /* Single-quartet convenience: (0 1|2 3) in the batch equals the tensor slice. */
  cuestQuartet_t one{0, 1, 2, 3};
  double single[6 * 6 * 6 * 6];
  if (cuestERIBatch(h, b, &one, 1, CUEST_FP64, single, nullptr) != CUEST_SUCCESS)
    return 1;
  double sgl_worst = 0.0;
  const cuest::PrimShell& sa = basis.shells[0];
  const cuest::PrimShell& sb = basis.shells[1];
  const cuest::PrimShell& sc = basis.shells[2];
  const cuest::PrimShell& sd = basis.shells[3];
  const int ma = sa.nctr * sa.ncomp(), mb = sb.nctr * sb.ncomp();
  const int mc = sc.nctr * sc.ncomp(), md = sd.nctr * sd.ncomp();
  size_t pos = 0;
  for (int fa = 0; fa < ma; ++fa)
    for (int fb = 0; fb < mb; ++fb)
      for (int fc = 0; fc < mc; ++fc)
        for (int fd = 0; fd < md; ++fd) {
          const size_t t =
              ((static_cast<size_t>(sa.func_offset + fa) * nbf + (sb.func_offset + fb)) * nbf +
               (sc.func_offset + fc)) *
                  nbf +
              (sd.func_offset + fd);
          sgl_worst = std::max(sgl_worst, std::fabs(single[pos++] - tensor[t]));
        }
  if (sgl_worst > 1e-14) {
    std::printf("FAIL: single quartet block differs by %.3e\n", sgl_worst);
    ++failures;
  }

  cuestBasisDestroy(b);
  cuestDestroy(h);
  std::printf("test_batch_api: %s\n", failures == 0 ? "all checks passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
