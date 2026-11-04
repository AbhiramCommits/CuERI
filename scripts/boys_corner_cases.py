#!/usr/bin/env python3
"""Boys-function corner-case sweep.

Runs the compiled kernel (via bench/boys_probe) over T = 1e-16 .. 1e3 and
n = 0..12, compares with SciPy's incomplete gamma, flags the numerical branch
and writes results/boys_corner_cases.csv plus a figure.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np
from scipy.special import gamma, gammainc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def boys_ref(n: int, t: float) -> float:
    if t <= 0.0:
        return 1.0 / (2 * n + 1)
    s = n + 0.5
    return 0.5 * t ** (-s) * gamma(s) * gammainc(s, t)


def main() -> int:
    probe = os.path.join(ROOT, "build", "bench", "boys_probe")
    if not os.path.exists(probe):
        raise SystemExit(f"missing {probe}; build the project first")
    out = subprocess.run([probe, "grid", "1e-16", "1e3", "300", "12"],
                         capture_output=True, text=True, check=True)
    rows = []
    worst = (0.0, 0.0, 0, "")
    for line in out.stdout.splitlines()[1:]:
        t, n, f, branch = line.split(",")
        t = float(t)
        n = int(n)
        f = float(f)
        ref = boys_ref(n, t)
        if ref == 0.0:
            continue
        rel = abs(f - ref) / abs(ref)
        rows.append(dict(T=f"{t:.17g}", n=n, F_ours=f"{f:.17g}", F_ref=f"{ref:.17g}",
                         rel_error=f"{rel:.6e}", branch=branch,
                         reference_source="SciPy gammainc/gamma"))
        if rel > worst[0]:
            worst = (rel, t, n, branch)

    os.makedirs("results/figures", exist_ok=True)
    with open("results/boys_corner_cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for n in (0, 2, 4, 6, 8, 10, 12):
        sel = [(float(r["T"]), float(r["rel_error"])) for r in rows if r["n"] == n]
        sel.sort()
        ax.loglog([s[0] for s in sel], [max(s[1], 1e-17) for s in sel], label=f"n={n}", lw=1.1)
    ax.axvline(1e-8, color="0.5", ls="--", lw=0.8)
    ax.axvline(30.0, color="0.5", ls="--", lw=0.8)
    ax.text(1.2e-8, 3e-14, "series -> downward", fontsize=8, color="0.35")
    ax.text(35, 3e-14, "erf/asymptotic", fontsize=8, color="0.35")
    ax.set_xlabel("T")
    ax.set_ylabel("relative error vs SciPy")
    ax.set_title("cuERI Boys function: relative error across all branches")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("results/figures/boys_corner_cases.png", dpi=140)
    plt.close(fig)

    print(f"[boys] {len(rows)} values; worst relative error {worst[0]:.3e} "
          f"at T={worst[1]:.6g}, n={worst[2]} ({worst[3]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
