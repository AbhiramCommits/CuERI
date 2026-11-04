#!/usr/bin/env python3
"""Generate a reference table for the on-the-fly Rys roots and weights.

cuERI builds the Rys rule from the Boys moments at run time, so no committed
header is needed.  This script independently recomputes the same rules with
NumPy (Golub-Welsch) and commits them for documentation and cross-checking:

    data/ref/rys_roots_reference.csv   T, nroots, node, weight, sum(weights)
"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy.special import gamma, gammainc


def boys(n: int, t: float) -> float:
    if t <= 0.0:
        return 1.0 / (2 * n + 1)
    s = n + 0.5
    return 0.5 * t ** (-s) * gamma(s) * gammainc(s, t)


def rule(t: float, nroots: int):
    S = max(t, 1.0)
    mu = np.array([S ** j * boys(j, t) / boys(0, t) for j in range(2 * nroots)])
    # orthonormal polynomials by Gram-Schmidt on the monomial basis
    Q = []
    for k in range(nroots):
        v = np.zeros(nroots)
        v[k] = 1.0
        for q in Q:
            ip = sum(q[i] * mu[k + i] for i in range(nroots))
            v = v - ip * q
        nn = sum(v[i] * v[j] * mu[i + j] for i in range(nroots) for j in range(nroots))
        Q.append(v / np.sqrt(nn))
    J = np.zeros((nroots, nroots))
    for i in range(nroots):
        for j in range(nroots):
            J[i, j] = sum(Q[i][a] * Q[j][b] * mu[a + b + 1]
                          for a in range(nroots) for b in range(nroots))
    ev, evec = np.linalg.eigh(0.5 * (J + J.T))
    nodes = ev / S
    weights = evec[0, :] ** 2
    return nodes, weights


def main() -> int:
    ts = [0.0, 1e-6, 0.1, 1.0, 5.0, 20.0, 30.0, 100.0, 500.0]
    out = os.path.join("data", "ref", "rys_roots_reference.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["T", "nroots", "node", "weight", "weight_sum"])
        for t in ts:
            for nroots in range(1, 6):
                nodes, weights = rule(t, nroots)
                for node, weight in zip(nodes, weights):
                    w.writerow([repr(t), nroots, repr(float(node)), repr(float(weight)),
                                repr(float(weights.sum()))])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
