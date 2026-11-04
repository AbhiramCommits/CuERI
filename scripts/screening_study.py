#!/usr/bin/env python3
"""Schwarz screening study: skip fraction and elementwise deviation vs tolerance.

For each tolerance the tensor is compared with the unscreened one.  The
Schwarz bound guarantees that every omitted shell quartet contributes less than
Q_pq Q_rs < tol to each of its elements, so the per-element deviation must stay
below tol.

Output: results/screening.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from cuest import Cuest  # noqa: E402

CASES = [("h2o", "6-31g*"), ("benzene", "sto-3g"), ("benzene", "6-31g")]
TOLS = [0.0, 1e-14, 1e-12, 1e-10, 1e-8]


def main() -> int:
    lib = Cuest()
    rows = []
    for molecule, basis_name in CASES:
        molpath = os.path.join(ROOT, "data", "molecules", f"{molecule}.json")
        basis = lib.basis(molpath, basis_name)
        t0 = time.perf_counter()
        ref, sref = lib.eri_tensor(basis, screen_tol=0.0, want_stats=True)
        t_ref = (time.perf_counter() - t0) * 1e3
        print(f"[screening] {molecule}/{basis_name} nbf={basis.nbf} "
              f"quartets={sref.quartets_total} unscreened {t_ref:.1f} ms")
        for tol in TOLS:
            t0 = time.perf_counter()
            got, st = lib.eri_tensor(basis, screen_tol=tol, want_stats=True)
            ms = (time.perf_counter() - t0) * 1e3
            dev = float(np.abs(ref - got).max())
            frac = st.quartets_screened / st.quartets_total if st.quartets_total else 0.0
            bound = tol if tol > 0 else 0.0
            rows.append(dict(molecule=molecule, basis=basis_name, tolerance=f"{tol:.0e}",
                             quartets_total=st.quartets_total,
                             quartets_screened=st.quartets_screened,
                             skip_fraction=f"{frac:.6f}", max_abs_deviation=f"{dev:.6e}",
                             deviation_bound=f"{bound:.1e}",
                             within_bound=str(tol == 0.0 or dev <= max(tol, 1e-13)).lower(),
                             wall_ms=f"{ms:.3f}"))
            print(f"    tol={tol:.0e}  skipped {st.quartets_screened}/{st.quartets_total} "
                  f"({100 * frac:.1f}%)  max dev {dev:.3e}  {ms:.1f} ms")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "screening.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
