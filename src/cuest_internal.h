/**
 * @file cuest_internal.h
 * @brief Internal data structures shared by the host backends.
 */
#ifndef CUEST_INTERNAL_H
#define CUEST_INTERNAL_H

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "cuest/cuest_eri.h"
#include "cuest_macros.h"

namespace cuest {

/** @brief One contracted Cartesian shell. */
struct PrimShell {
  int atom = 0;         /**< Zero-based atom index. */
  int L = 0;            /**< Angular momentum (0, 1, 2). */
  int nprim = 0;        /**< Number of primitive Gaussians. */
  int nctr = 0;         /**< Number of contraction columns. */
  int func_offset = 0;  /**< Index of the first basis function. */
  std::vector<double> exps; /**< nprim exponents. */
  /** Normalised coefficients, layout [ictr][icomp][iprim]. */
  std::vector<double> cn;

  int ncomp() const { return (L + 1) * (L + 2) / 2; }
  const double *coeff(int ictr, int icomp) const {
    return &cn[(static_cast<size_t>(ictr) * ncomp() + icomp) * static_cast<size_t>(nprim)];
  }
};

/** @brief A complete basis set plus molecular geometry. */
struct BasisData {
  int nbf = 0;
  int nshells = 0;
  int natoms = 0;
  int max_am = 0;
  std::vector<PrimShell> shells;
  std::vector<double> atom_xyz;  /**< 3*natoms, bohr. */
  std::vector<double> atom_Z;    /**< ncharges. */
};

/** @brief Result of parsing a molecule file. */
struct Molecule {
  std::vector<double> xyz; /**< 3*natoms, bohr. */
  std::vector<int> Z;
};

/** @brief One contracted shell as exported in data/basis/<name>.json. */
struct RawShell {
  int am = 0;
  std::vector<double> exps;
  std::vector<std::vector<double>> coeffs; /**< [ictr][iprim] */
};

/** @brief Load a molecule JSON file. */
bool load_molecule(const std::string &path, Molecule &mol, std::string &err);

/** @brief Load one element from a basis JSON file. */
bool load_basis_element(const std::string &path, const std::string &element,
                        std::vector<RawShell> &shells, std::string &err);

/** @brief Resolve the data directory (env CUEST_DATA_DIR, then compile default). */
std::string data_directory();

/** @brief Assemble a BasisData from a molecule and basis name. */
bool build_basis(const Molecule &mol, const std::string &basis_name, BasisData &out,
                 std::string &err);

/** @brief Element symbol for atomic number Z. */
const char *element_symbol(int Z);

}  // namespace cuest

/* Public opaque types are defined here so the C API layer can allocate them. */
struct cuest_basis_t {
  cuest::BasisData data;
};

struct cuest_handle_t {
  cuestBackend_t backend = CUEST_BACKEND_CPU;
  cuestEriAlgo_t algo = CUEST_ERI_ALGO_MD;
  int num_threads = 1;
};

namespace cuest {

/* CPU backend entry points (implemented in one_electron.cpp / eri_cpu.cpp). */
cuestStatus_t cpu_one_electron(const BasisData &basis, int which, cuestPrecision_t precision,
                               double *out, int ld);
cuestStatus_t cpu_schwarz(const BasisData &basis, double *q_pair);
cuestStatus_t cpu_eri_batch(const BasisData &basis, const cuestQuartet_t *quartets, int nquartets,
                            cuestPrecision_t precision, cuestEriAlgo_t algo, double *out,
                            const size_t *offsets);
cuestStatus_t cpu_eri_tensor(const BasisData &basis, cuestPrecision_t precision,
                             cuestEriAlgo_t algo, double screen_tol, double *eri,
                             cuestEriStats_t *stats);

/** @brief Shell pair (i >= j) used by the screening and batching machinery. */
struct ShellPair {
  int i = 0;
  int j = 0;
  double q = 0.0; /**< Schwarz bound sqrt(max |(ij|ij)|). */
};

/** @brief Build the sorted shell-pair list with Schwarz bounds. */
void build_shell_pairs(const BasisData &basis, std::vector<ShellPair> &pairs, cuestEriAlgo_t algo);

/** @brief Angular-momentum class id 0..9 for a shell quartet (upper triangle of L_i+L_j). */
int am_class(int la, int lb, int lc, int ld);

}  // namespace cuest

#endif /* CUEST_INTERNAL_H */
