/**
 * @file cuest_macros.h
 * @brief Host/device backend shim.
 *
 * All integral mathematics lives in headers under src/kernels/ and is written
 * against CUEST_HD plus plain arithmetic.  Under nvcc the same function body is
 * compiled as __host__ __device__; under a C++ host compiler CUEST_HD expands
 * to nothing and the identical code runs on the CPU.  This is what keeps the
 * CPU reference and the CUDA kernels from drifting apart.
 */
#ifndef CUEST_MACROS_H
#define CUEST_MACROS_H

#ifdef __CUDACC__
#define CUEST_HD __host__ __device__
#define CUEST_INLINE __host__ __device__ __forceinline__
#else
#define CUEST_HD
#define CUEST_INLINE inline
#endif

/* Largest angular momentum supported by the kernels (s, p, d). */
#define CUEST_MAX_AM 2
/* Largest number of Cartesian components in a shell: (2+1)(2+2)/2 = 6. */
#define CUEST_MAX_CART 6
/* Largest total Hermite order for an l<=2 shell quartet: 2+2+2+2 = 8. */
#define CUEST_MAX_LTOT 8
/* Largest Boys order needed anywhere in the library. */
#define CUEST_MAX_BOYS 32

#endif /* CUEST_MACROS_H */
