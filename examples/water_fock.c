/*
 * water_fock.c - minimal C consumer of the cuERI public API.
 *
 * Builds the overlap, kinetic and nuclear-attraction matrices and the full
 * two-electron tensor for a molecule, then prints the core-Hamiltonian trace
 * and the ERI statistics.  This file is compiled as C99 and is the usage
 * snippet reproduced in README.md.
 *
 *   water_fock [molecule.json] [basis]
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuest/cuest_eri.h>

static int fail(cuestStatus_t status) {
  const char* msg = "?";
  cuestGetErrorString(status, &msg);
  fprintf(stderr, "cuERI error: %s (%d)\n", msg, (int)status);
  return 1;
}

int main(int argc, char** argv) {
  const char* molecule = (argc > 1) ? argv[1] : "data/molecules/h2o.json";
  const char* basis_name = (argc > 2) ? argv[2] : "sto-3g";

  cuestHandle_t handle = NULL;
  cuestStatus_t status = cuestCreate(&handle);
  if (status != CUEST_SUCCESS)
    return fail(status);

  cuestBasis_t basis = NULL;
  status = cuestBasisCreateFromFile(handle, molecule, basis_name, &basis);
  if (status != CUEST_SUCCESS)
    return fail(status);

  int nbf = 0, nshells = 0;
  cuestBasisNumFunctions(basis, &nbf);
  cuestBasisNumShells(basis, &nshells);

  double* S = (double*)calloc((size_t)nbf * nbf, sizeof(double));
  double* T = (double*)calloc((size_t)nbf * nbf, sizeof(double));
  double* V = (double*)calloc((size_t)nbf * nbf, sizeof(double));
  double* eri = (double*)calloc((size_t)nbf * nbf * nbf * nbf, sizeof(double));
  if (!S || !T || !V || !eri)
    return fail(CUEST_OUT_OF_MEMORY);

  if ((status = cuestOverlapMatrix(handle, basis, CUEST_FP64, S, nbf)) != CUEST_SUCCESS)
    return fail(status);
  if ((status = cuestKineticMatrix(handle, basis, CUEST_FP64, T, nbf)) != CUEST_SUCCESS)
    return fail(status);
  if ((status = cuestNuclearMatrix(handle, basis, CUEST_FP64, V, nbf)) != CUEST_SUCCESS)
    return fail(status);

  cuestEriStats_t stats;
  if ((status = cuestERITensor(handle, basis, CUEST_FP64, 1e-10, eri, &stats)) != CUEST_SUCCESS)
    return fail(status);

  double trace_h = 0.0;
  for (int i = 0; i < nbf; ++i)
    trace_h += T[(size_t)i * nbf + i] + V[(size_t)i * nbf + i];

  printf("molecule: %s\n", molecule);
  printf("basis:    %s\n", basis_name);
  printf("shells:   %d\n", nshells);
  printf("basis functions: %d\n", nbf);
  printf("core Hamiltonian trace: %.12f Hartree\n", trace_h);
  printf("(00|00) = %.12f\n", eri[0]);
  printf("ERI quartets: %zu canonical, %zu screened, %zu primitive quartets\n",
         stats.quartets_total, stats.quartets_screened, stats.primitive_pairs);
  printf("ERI kernel time: %.3f ms\n", stats.kernel_ms);

  free(eri);
  free(V);
  free(T);
  free(S);
  cuestBasisDestroy(basis);
  cuestDestroy(handle);
  return 0;
}
