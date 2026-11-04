#!/usr/bin/env python3
"""Pin a small set of cuERI integrals as committed golden values.

Writes data/ref/golden_integrals.npz, used by tests/python/test_golden.py as a
fast regression check (tolerance: max abs difference <= 1e-12).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from cuest import Cuest  # noqa: E402

CASES = [("h2o", "sto-3g"), ("ch4", "6-31g")]


def main() -> int:
    lib = Cuest()
    out = {}
    for molecule, basis_name in CASES:
        basis = lib.basis(os.path.join(ROOT, "data", "molecules", f"{molecule}.json"), basis_name)
        tag = f"{molecule}_{basis_name}".replace("*", "s")
        out[f"{tag}_S"] = lib.overlap(basis)
        out[f"{tag}_T"] = lib.kinetic(basis)
        out[f"{tag}_V"] = lib.nuclear(basis)
        out[f"{tag}_ERI"] = lib.eri_tensor(basis, screen_tol=1e-12)
        print(f"[golden] {molecule}/{basis_name}: {basis.nbf} functions")
    path = os.path.join(ROOT, "data", "ref", "golden_integrals.npz")
    np.savez_compressed(path, **out)
    print(f"[golden] wrote {path} ({os.path.getsize(path) / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
