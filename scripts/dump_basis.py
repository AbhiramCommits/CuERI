#!/usr/bin/env python3
"""Export contracted Cartesian basis data from PySCF to committed JSON.

The C++ library is offline and reproducible: it reads only the JSON files
written here.  Each shell is stored with its angular momentum, the primitive
exponents and one or more contraction coefficient columns; the C++ loader folds
in the primitive normalisation and unit-normalises every Cartesian component.

Usage:
    python scripts/dump_basis.py [--out data/basis]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ELEMENTS = ["H", "C", "N", "O"]
BASES = ["sto-3g", "6-31g", "6-31g*"]


def export_basis(name: str) -> dict:
    from pyscf import gto
    import pyscf

    data = {
        "basis": name,
        "generator": "scripts/dump_basis.py",
        "pyscf_version": pyscf.__version__,
        "cartesian": True,
        "note": (
            "Coefficients are the raw PySCF contraction coefficients. The C++ "
            "loader multiplies each primitive by N(l,alpha) = "
            "(2 alpha/pi)^(3/4) (4 alpha)^(l/2)/sqrt((2l-1)!!) and then rescales "
            "each Cartesian component to unit self-overlap."
        ),
        "elements": {},
    }
    for el in ELEMENTS:
        shells = gto.basis.load(name, el)
        out_shells = []
        for entry in shells:
            am = int(entry[0])
            exps = []
            coeffs = []
            for prim in entry[1:]:
                exps.append(float(prim[0]))
                if not coeffs:
                    coeffs = [[] for _ in range(len(prim) - 1)]
                for ic in range(len(prim) - 1):
                    coeffs[ic].append(float(prim[ic + 1]))
            out_shells.append(
                {
                    "am": am,
                    "exponents": exps,
                    "coefficients": coeffs,
                }
            )
        data["elements"][el] = {"shells": out_shells}
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "basis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in BASES:
        data = export_basis(name)
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")
        nsh = sum(len(v["shells"]) for v in data["elements"].values())
        print(f"wrote {path} ({nsh} shells for {', '.join(ELEMENTS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
