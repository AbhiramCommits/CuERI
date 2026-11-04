# Design notes

## Opaque handles

`cuestHandle_t` and `cuestBasis_t` are pointers to incomplete structs.  Callers
never see the layout, which lets the library change its internal basis
representation (flat SoA device buffers, added contraction columns, different
normalisation bookkeeping) without breaking the ABI.  Handles are created and
destroyed by the library; `cuestDestroy(NULL)` and `cuestBasisDestroy(NULL)` are
documented no-ops so cleanup code does not need special cases.

The handle owns build-time configuration that is not per-call: the compute
backend (selected when the library is compiled and, at runtime, when a CUDA
device is present) and the ERI algorithm.  The algorithm is a handle property
because it is a performance/accuracy choice that applies to a whole computation,
not to a single tensor.

## Status codes instead of exceptions

Every entry point returns `cuestStatus_t` and the header is C99-clean.  C++
exceptions are caught at the boundary and mapped:

* `std::bad_alloc` -> `CUEST_OUT_OF_MEMORY`
* malformed arguments -> `CUEST_INVALID_VALUE`
* angular momentum above d -> `CUEST_NOT_SUPPORTED`
* a missing CUDA backend -> `CUEST_NOT_SUPPORTED`
* anything else -> `CUEST_INTERNAL_ERROR`

`cuestGetErrorString` returns a pointer to a static string, so it is safe to call
from any thread and never allocates.  The current API deliberately does not
expose a per-thread "last error" buffer: all the information a caller needs is
the status code, and adding a message buffer would make the ABI harder to keep
stable.

## Precision as a per-call parameter

`cuestPrecision_t` is passed to every numerical entry point rather than being
fixed on the handle.  A caller may want fp64 for the SCF and fp32 for a
throughput study in the same process, and the fp32 path is a distinct template
instantiation, not a post-hoc cast.  The output buffer is always `double*`: the
precision parameter controls the accumulation, which is what the error study is
about, and keeping one output type avoids a second ABI surface.

## Tensor versus batch entry points

`cuestERITensor` is the convenience path: it enumerates canonical shell
quartets, applies Schwarz screening, fills the eight permutational symmetries
and reports statistics.  It is what the RHF script and the benchmark use.

`cuestERIBatch` exists for callers that already know which quartets they need
(for example a screened list produced elsewhere, or a distributed computation
that owns a slice of the tensor).  It takes explicit shell indices and optional
byte offsets, computes exactly those blocks and does not assume symmetry.  The
two entry points share the same per-quartet kernel, so a batch result and the
corresponding tensor block are bit-for-bit identical
(`tests/test_batch_api.cpp`).

The batch block layout is `(fa, fb, fc, fd)` with `fa` running over the
functions of shell `i` first, matching the tensor's `(mu, nu, la, si)` indexing.

## One copy of the mathematics

`src/cuest_macros.h` defines `CUEST_HD` as `__host__ __device__` under nvcc and
as nothing otherwise.  Every integral routine lives in a header under
`src/kernels/`, uses only plain arithmetic and the global math functions, and is
compiled twice: once into the CPU backend and once into the CUDA kernels.  There
is no second implementation of the recurrences to keep in sync; the only
backend-specific code is the launch/traversal layer.

The reference implementation under `src/reference/` is deliberately separate and
is linked only so the tests can call it.  It shares the data structures but not
the algorithms.

## Data files

Molecules and bases are plain JSON committed to the repository.  The basis files
are produced by `scripts/dump_basis.py` from PySCF and contain only exponents and
raw contraction coefficients; normalisation is applied by the C++ loader on
every run, so the interpretation of the coefficients is in one place.  This
keeps the build offline and the data reviewable in a diff.

The loader resolves the basis directory from `CUEST_DATA_DIR`, then from the
compile-time `CUEST_DATA_DIR_DEFAULT` (the source `data/` directory).  The
environment override is what makes the precision stress study able to generate
temporary bases without touching the repository.

## CUDA layout

The device kernel receives a `DeviceBasis` struct of flat arrays (one offset per
shell into concatenated exponent and coefficient arrays) rather than an array of
structs.  Coefficients stay in double precision on the device and are cast to
the kernel real type at use, so one H2D copy serves both the fp32 and fp64
instantiations.

The ERI kernel is templated on `<typename T, int LA, int LB, int LC, int LD>` and
the host launches it once per angular-momentum class, so the compiler sees
constant loop bounds and the register footprint scales with the class instead of
always being sized for dddd.

At Tier C (this build) the CUDA sources are not compiled; the parity test
records that fact instead of silently passing.
