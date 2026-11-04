#!/usr/bin/env python3
"""Generate the committed Boys-function reference table.

Uses SciPy's regularized incomplete gamma function:
    F_n(T) = 1/2 T^{-(n+1/2)} Gamma(n+1/2) P(n+1/2, T),  F_n(0) = 1/(2n+1).
"""
from __future__ import annotations

import csv
import os

from scipy.special import gamma, gammainc

TS = [
    0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1,
    0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 29.999, 30.0, 30.001, 50.0,
    100.0, 200.0, 500.0, 1000.0,
]
NMAX = 12


def boys(n: int, t: float) -> float:
    if t <= 0.0:
        return 1.0 / (2 * n + 1)
    s = n + 0.5
    return 0.5 * t ** (-s) * gamma(s) * gammainc(s, t)


def main() -> int:
    out = os.path.join("data", "ref", "boys_reference.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["T", "n", "F_n"])
        for t in TS:
            for n in range(NMAX + 1):
                w.writerow([repr(t), n, repr(float(boys(n, t)))])
    print(f"wrote {out} ({len(TS) * (NMAX + 1)} rows, T = {len(TS)} values, n = 0..{NMAX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
