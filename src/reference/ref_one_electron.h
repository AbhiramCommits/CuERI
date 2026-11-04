/**
 * @file ref_one_electron.h
 * @brief Independent (non-McMurchie-Davidson) one-electron reference.
 *
 * Test-only.  Overlap and kinetic integrals are evaluated with the
 * Obara-Saika recursion and the analytic gradient formula; the Boys function is
 * obtained from the incomplete gamma function (series / continued fraction)
 * rather than the production branch structure.  Nothing in this file is shared
 * with src/kernels/.
 */
#ifndef CUEST_REF_ONE_ELECTRON_H
#define CUEST_REF_ONE_ELECTRON_H

#include "cuest_internal.h"

namespace cuest_ref {

/**
 * @brief Compute a one-electron matrix by the independent reference route.
 *
 * @param which 0 = overlap, 1 = kinetic, 2 = nuclear attraction.
 */
void compute_one_electron(const cuest::BasisData& basis, int which, double* out, int ld);

/** @brief Boys function from the incomplete gamma function (long double). */
long double boys_ref(int n, long double T);

}  // namespace cuest_ref

#endif /* CUEST_REF_ONE_ELECTRON_H */
