/**
 * @file boys_probe.cpp
 * @brief Dump the library's Boys function on a requested grid (used by
 *        scripts/boys_corner_cases.py so the Python analysis exercises the same
 *        compiled kernel code as the library).
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernels/boys.cuh"

static const char* branch_of(double t) {
  if (t <= 0.0)
    return "exact_zero";
  if (t < CUEST_BOYS_SMALL_T)
    return "taylor_small_T";
  if (t < CUEST_BOYS_MID_MAX)
    return "downward_recursion";
  return "erf_asymptotic";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: boys_probe grid <Tmin> <Tmax> <nT> <nmax>\n");
    return 2;
  }
  if (std::strcmp(argv[1], "grid") == 0 && argc >= 6) {
    const double tmin = std::atof(argv[2]);
    const double tmax = std::atof(argv[3]);
    const int nt = std::atoi(argv[4]);
    const int nmax = std::atoi(argv[5]);
    std::printf("T,n,F,branch\n");
    for (int i = 0; i < nt; ++i) {
      double t;
      if (nt == 1) {
        t = tmin;
      } else {
        const double l0 = std::log(tmin);
        const double l1 = std::log(tmax);
        t = std::exp(l0 + (l1 - l0) * i / (nt - 1));
      }
      for (int n = 0; n <= nmax; ++n) {
        std::printf("%.17g,%d,%.17g,%s\n", t, n, cuest_boys(n, t), branch_of(t));
      }
    }
    return 0;
  }
  if (std::strcmp(argv[1], "point") == 0 && argc >= 4) {
    const double t = std::atof(argv[2]);
    const int nmax = std::atoi(argv[3]);
    for (int n = 0; n <= nmax; ++n)
      std::printf("%.17g,%d,%.17g,%s\n", t, n, cuest_boys(n, t), branch_of(t));
    return 0;
  }
  std::fprintf(stderr, "unknown mode\n");
  return 2;
}
