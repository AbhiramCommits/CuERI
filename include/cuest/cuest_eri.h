/**
 * @file cuest_eri.h
 * @brief Public C API for cuERI, a Gaussian-basis quantum-chemistry integral library.
 *
 * cuERI computes the one-electron (overlap, kinetic, nuclear-attraction) and
 * two-electron repulsion integrals over Cartesian s, p and d Gaussian shells.
 * The public header is C99-clean and may be included from C or C++.
 *
 * Every entry point returns a #cuestStatus_t.  No C++ exception is allowed to
 * escape the C boundary; on any internal failure a status code is returned and
 * the output buffers are left untouched or zero-filled as documented.
 */
#ifndef CUEST_ERI_H
#define CUEST_ERI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(CUEST_BUILD_SHARED)
#define CUEST_API __declspec(dllexport)
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define CUEST_API __attribute__((visibility("default")))
#else
#define CUEST_API
#endif

/** @brief Library version, major component. */
#define CUEST_VERSION_MAJOR 1
/** @brief Library version, minor component. */
#define CUEST_VERSION_MINOR 0
/** @brief Library version, patch component. */
#define CUEST_VERSION_PATCH 0

/** @brief Opaque library handle. */
typedef struct cuest_handle_t* cuestHandle_t;

/** @brief Opaque basis-set handle. */
typedef struct cuest_basis_t* cuestBasis_t;

/**
 * @brief Status codes returned by every cuERI entry point.
 */
typedef enum cuestStatus_t_ {
  CUEST_SUCCESS = 0,         /**< Operation completed successfully. */
  CUEST_INVALID_VALUE = 1,   /**< A caller-supplied argument is invalid. */
  CUEST_NOT_SUPPORTED = 2,   /**< The requested feature/backend is unavailable. */
  CUEST_OUT_OF_MEMORY = 3,   /**< An allocation failed. */
  CUEST_NUMERICAL_ERROR = 4, /**< A numerical failure (NaN/Inf) was detected. */
  CUEST_INTERNAL_ERROR = 5   /**< An unexpected internal failure. */
} cuestStatus_t;

/**
 * @brief Arithmetic precision used for a computation.
 *
 * CUEST_FP32 accumulates the primitive contractions in single precision and
 * writes the result as doubles; CUEST_FP64 accumulates in double precision.
 */
typedef enum cuestPrecision_t_ {
  CUEST_FP32 = 0, /**< Single-precision accumulation. */
  CUEST_FP64 = 1  /**< Double-precision accumulation. */
} cuestPrecision_t;

/**
 * @brief Compute backend selected when the handle was created.
 */
typedef enum cuestBackend_t_ {
  CUEST_BACKEND_CPU = 0, /**< Host CPU backend (optionally OpenMP). */
  CUEST_BACKEND_CUDA = 1 /**< NVIDIA CUDA backend. */
} cuestBackend_t;

/**
 * @brief Algorithm used for two-electron repulsion integrals.
 */
typedef enum cuestEriAlgo_t_ {
  CUEST_ERI_ALGO_MD = 0, /**< McMurchie-Davidson Hermite recurrences. */
  CUEST_ERI_ALGO_RYS = 1 /**< Rys quadrature. */
} cuestEriAlgo_t;

/**
 * @brief A shell quartet (i j | k l) addressed by shell index.
 *
 * Coulomb indices are chemists' notation: index i,j belong to the bra pair and
 * k,l to the ket pair.
 */
typedef struct cuestQuartet_t_ {
  int i; /**< Bra shell 1. */
  int j; /**< Bra shell 2. */
  int k; /**< Ket shell 1. */
  int l; /**< Ket shell 2. */
} cuestQuartet_t;

/**
 * @brief Statistics reported by cuestERITensor().
 */
typedef struct cuestEriStats_t_ {
  size_t quartets_total;    /**< Canonical shell quartets enumerated. */
  size_t quartets_screened; /**< Quartets skipped by Schwarz screening. */
  size_t primitive_pairs;   /**< Primitive quartet products actually evaluated. */
  double kernel_ms;         /**< Wall-clock time of the integral kernel only. */
  int max_shell_am;         /**< Largest angular momentum present in the basis. */
} cuestEriStats_t;

/**
 * @brief Create a cuERI library handle.
 *
 * The backend is fixed at build time: when the library was compiled with CUDA
 * support and a usable device is present the handle selects
 * #CUEST_BACKEND_CUDA, otherwise #CUEST_BACKEND_CPU.
 *
 * @param[out] handle Receives the new handle.  Must not be NULL.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE / #CUEST_OUT_OF_MEMORY.
 */
CUEST_API cuestStatus_t cuestCreate(cuestHandle_t* handle);

/**
 * @brief Destroy a handle created by cuestCreate().
 *
 * Passing NULL is a no-op and returns #CUEST_SUCCESS.
 *
 * @param[in] handle Handle to destroy.
 * @return #CUEST_SUCCESS.
 */
CUEST_API cuestStatus_t cuestDestroy(cuestHandle_t handle);

/**
 * @brief Report the compute backend selected by the handle.
 *
 * @param[in]  handle  Library handle.
 * @param[out] backend Receives the backend identifier.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestGetBackend(cuestHandle_t handle, cuestBackend_t* backend);

/**
 * @brief Select the two-electron integral algorithm.
 *
 * @param[in] handle Library handle.
 * @param[in] algo   #CUEST_ERI_ALGO_MD or #CUEST_ERI_ALGO_RYS.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestSetEriAlgo(cuestHandle_t handle, cuestEriAlgo_t algo);

/**
 * @brief Return the library version.
 *
 * @param[out] major Receives the major version (may be NULL).
 * @param[out] minor Receives the minor version (may be NULL).
 * @param[out] patch Receives the patch version (may be NULL).
 * @return #CUEST_SUCCESS.
 */
CUEST_API cuestStatus_t cuestGetVersion(int* major, int* minor, int* patch);

/**
 * @brief Map a status code to a human-readable, static string.
 *
 * @param[in]  status Status code.
 * @param[out] msg    Receives a pointer to a static string owned by the library.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestGetErrorString(cuestStatus_t status, const char** msg);

/**
 * @brief Build a basis set from a molecule JSON file and a named basis.
 *
 * The molecule file contains atomic numbers and Cartesian coordinates in bohr
 * (an "angstrom" unit tag is also accepted).  The basis is loaded from the
 * library data directory (see data/basis/<name>.json).
 *
 * @param[in]  handle        Library handle.
 * @param[in]  molecule_json Path to the molecule JSON file.
 * @param[in]  basis_name    Basis name, e.g. "sto-3g", "6-31g", "6-31g*".
 * @param[out] basis         Receives the new basis handle.
 * @return #CUEST_SUCCESS, #CUEST_INVALID_VALUE, #CUEST_NOT_SUPPORTED,
 *         #CUEST_OUT_OF_MEMORY or #CUEST_INTERNAL_ERROR.
 */
CUEST_API cuestStatus_t cuestBasisCreateFromFile(cuestHandle_t handle, const char* molecule_json,
                                                 const char* basis_name, cuestBasis_t* basis);

/**
 * @brief Destroy a basis set.
 *
 * Passing NULL is a no-op and returns #CUEST_SUCCESS.
 *
 * @param[in] basis Basis to destroy.
 * @return #CUEST_SUCCESS.
 */
CUEST_API cuestStatus_t cuestBasisDestroy(cuestBasis_t basis);

/**
 * @brief Number of Cartesian basis functions in a basis set.
 *
 * @param[in]  basis Basis set.
 * @param[out] nbf   Receives the function count.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestBasisNumFunctions(cuestBasis_t basis, int* nbf);

/**
 * @brief Number of shells in a basis set.
 *
 * @param[in]  basis    Basis set.
 * @param[out] nshells  Receives the shell count.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestBasisNumShells(cuestBasis_t basis, int* nshells);

/**
 * @brief Query one shell's metadata.
 *
 * Any output pointer may be NULL.  The angular momentum is the total
 * \f$l\f$ of the shell (0, 1 or 2).  The shell spans nctr blocks of
 * (l+1)(l+2)/2 Cartesian functions starting at func_offset.
 *
 * @param[in]  basis       Basis set.
 * @param[in]  ishell      Zero-based shell index.
 * @param[out] atom        Receives the zero-based atom index.
 * @param[out] am          Receives the angular momentum.
 * @param[out] nprim       Receives the number of primitives.
 * @param[out] nctr        Receives the number of contractions.
 * @param[out] func_offset Receives the first basis-function index.
 * @return #CUEST_SUCCESS or #CUEST_INVALID_VALUE.
 */
CUEST_API cuestStatus_t cuestBasisShellInfo(cuestBasis_t basis, int ishell, int* atom, int* am,
                                            int* nprim, int* nctr, int* func_offset);

/**
 * @brief Compute the overlap matrix \f$S_{\mu\nu}=\langle\mu|\nu\rangle\f$.
 *
 * @param[in]  handle    Library handle.
 * @param[in]  basis     Basis set.
 * @param[in]  precision Accumulation precision.
 * @param[out] out       Row-major matrix with leading dimension @p ld
 *                       (at least nbf).
 * @param[in]  ld        Leading dimension.
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestOverlapMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                           cuestPrecision_t precision, double* out, int ld);

/**
 * @brief Compute the kinetic-energy matrix \f$T_{\mu\nu}\f$.
 *
 * @param[in]  handle    Library handle.
 * @param[in]  basis     Basis set.
 * @param[in]  precision Accumulation precision.
 * @param[out] out       Row-major matrix with leading dimension @p ld.
 * @param[in]  ld        Leading dimension.
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestKineticMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                           cuestPrecision_t precision, double* out, int ld);

/**
 * @brief Compute the nuclear-attraction matrix \f$V_{\mu\nu}\f$.
 *
 * @param[in]  handle    Library handle.
 * @param[in]  basis     Basis set.
 * @param[in]  precision Accumulation precision.
 * @param[out] out       Row-major matrix with leading dimension @p ld.
 * @param[in]  ld        Leading dimension.
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestNuclearMatrix(cuestHandle_t handle, cuestBasis_t basis,
                                           cuestPrecision_t precision, double* out, int ld);

/**
 * @brief Compute Schwarz shell-pair bounds.
 *
 * @p q_pair receives nshell*nshell values, row-major, with
 * \f$Q_{pq}=\sqrt{\max_{\mu\in p,\nu\in q}|(\mu\nu|\mu\nu)|}\f$.  The matrix
 * is symmetric.  A shell quartet (pq|rs) may be skipped when
 * \f$Q_{pq}Q_{rs}<\varepsilon\f$.
 *
 * @param[in]  handle Library handle.
 * @param[in]  basis  Basis set.
 * @param[out] q_pair Array of nshell*nshell doubles.
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestSchwarzBounds(cuestHandle_t handle, cuestBasis_t basis,
                                           double* q_pair);

/**
 * @brief Compute a caller-selected batch of shell quartets.
 *
 * For quartet q with shells (i,j,k,l) the output block is stored contiguously
 * in chemists' order with dimensions
 * na*nb*nc*nd (na = functions in shell i, ...).  When @p offsets is NULL the
 * blocks are packed back-to-back in input order; otherwise
 * \p out + offsets[q] is the first element of block q.
 *
 * @param[in]  handle    Library handle.
 * @param[in]  basis     Basis set.
 * @param[in]  quartets  Array of @p nquartets shell quartets.
 * @param[in]  nquartets Number of quartets (>= 0).
 * @param[in]  precision Accumulation precision.
 * @param[out] out       Output buffer (see above).
 * @param[in]  offsets   Optional cumulative offsets, length nquartets.
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestERIBatch(cuestHandle_t handle, cuestBasis_t basis,
                                      const cuestQuartet_t* quartets, int nquartets,
                                      cuestPrecision_t precision, double* out,
                                      const size_t* offsets);

/**
 * @brief Compute the full two-electron tensor.
 *
 * The @p eri_nbf4 buffer receives nbf^4 doubles in chemists' notation with
 * indices (mu,nu,lambda,sigma) packed as
 * ((mu*nbf + nu)*nbf + lambda)*nbf + sigma.  All eight permutational
 * symmetries are filled in.
 *
 * @param[in]  handle     Library handle.
 * @param[in]  basis      Basis set.
 * @param[in]  precision  Accumulation precision.
 * @param[in]  screen_tol Schwarz threshold; 0 disables screening.
 * @param[out] eri_nbf4   Output tensor of nbf^4 doubles.
 * @param[out] stats      Optional statistics (may be NULL).
 * @return #CUEST_SUCCESS or an error status.
 */
CUEST_API cuestStatus_t cuestERITensor(cuestHandle_t handle, cuestBasis_t basis,
                                       cuestPrecision_t precision, double screen_tol,
                                       double* eri_nbf4, cuestEriStats_t* stats);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUEST_ERI_H */
