# Changelog

All notable changes to cuERI are recorded here.  The project uses semantic
versioning for the public C API.

## [1.0.0] - 2026-11-03

### Added

- Public C99 API (`include/cuest/cuest_eri.h`) with opaque handles, six status
  codes, fp32/fp64 precision selection, MD/Rys algorithm selection and both
  tensor and batch ERI entry points.
- Boys function F_0..F_12 with Taylor, downward-recursion and erf/asymptotic
  branches, plus the vector form used by the Hermite recurrences.
- McMurchie-Davidson one-electron integrals (overlap, kinetic,
  nuclear-attraction) over contracted Cartesian s, p and d shells.
- Two-electron repulsion integrals via McMurchie-Davidson Hermite recurrences
  and via on-the-fly Rys quadrature built from the Boys moments.
- Schwartz shell-pair screening, canonical-quartet enumeration with all eight
  permutational symmetries, and angular-momentum batching.
- CPU backend with OpenMP (one thread and all cores), fp32 and fp64
  accumulation.
- CUDA backend: templated one-electron, Schwarz and ERI kernels sharing the
  same `CUEST_HD` headers, SoA device basis, event timing.  Compiled and
  measured only when a toolkit and GPU are present; this build is Tier C and
  records the CUDA cells as not measured.
- Independent test reference using the Obara-Saika recursion and an
  incomplete-gamma Boys function.
- Validation harness against PySCF/libcint for nine molecule/basis
  combinations, a NumPy RHF closed loop over cuERI integrals, Boys corner-case
  and fp32/fp64 conditioning sweeps, a Schwarz screening study, benchmarks and
  occupancy reporting.
- CTest suite (8 tests), pytest suite (9 tests), sanitizer build and a GitHub
  Actions matrix (gcc/clang, Debug/Release) plus a CUDA compile-only job.
- CMake install rules and `find_package(cuERI)` package configuration.

### Known limitations

- Angular momentum above d is rejected with `CUEST_NOT_SUPPORTED`.
- No derivative integrals, no spherical harmonics, no general contractions with
  more than one coefficient column in the committed data (the loader supports
  them; `dump_basis.py` emits one column per shell).
- The CUDA batch entry point returns `CUEST_NOT_SUPPORTED`; the CPU backend
  implements it.
