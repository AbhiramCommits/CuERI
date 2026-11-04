#!/usr/bin/env python3
"""Generate molecule JSON files (atomic numbers + Cartesian coordinates in bohr).

Geometries are the standard experimental/optimised structures, converted from
angstrom to bohr with the CODATA 2018 conversion factor.
"""
from __future__ import annotations

import json
import os

BOHR_PER_ANGSTROM = 1.8897261246257702


def bohr(xyz_ang):
    return [[round(c * BOHR_PER_ANGSTROM, 10) for c in row] for row in xyz_ang]


MOLECULES = {
    "h2": {
        "description": "H2, r(H-H) = 0.7414 Angstrom",
        "atoms": [("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.7414])],
    },
    "h2o": {
        "description": "H2O, O-H = 0.9572 Angstrom, angle 104.52 deg",
        "atoms": [
            ("O", [0.0, 0.0, 0.1173]),
            ("H", [0.0, 0.7572, -0.4692]),
            ("H", [0.0, -0.7572, -0.4692]),
        ],
    },
    "ch4": {
        "description": "CH4, C-H = 1.089 Angstrom, tetrahedral",
        "atoms": [
            ("C", [0.0, 0.0, 0.0]),
            ("H", [0.0, 0.0, 1.089]),
            ("H", [1.0267, 0.0, -0.3633]),
            ("H", [-0.5133, 0.8892, -0.3633]),
            ("H", [-0.5133, -0.8892, -0.3633]),
        ],
    },
    "benzene": {
        "description": "C6H6, D6h, C-C = 1.39 Angstrom, C-H = 1.09 Angstrom",
        "atoms": [
            ("C", [1.39, 0.0, 0.0]),
            ("C", [0.695, 1.2038, 0.0]),
            ("C", [-0.695, 1.2038, 0.0]),
            ("C", [-1.39, 0.0, 0.0]),
            ("C", [-0.695, -1.2038, 0.0]),
            ("C", [0.695, -1.2038, 0.0]),
            ("H", [2.48, 0.0, 0.0]),
            ("H", [1.24, 2.1478, 0.0]),
            ("H", [-1.24, 2.1478, 0.0]),
            ("H", [-2.48, 0.0, 0.0]),
            ("H", [-1.24, -2.1478, 0.0]),
            ("H", [1.24, -2.1478, 0.0]),
        ],
    },
    # Two pairs of nearly coincident centres.  The pairs are 1e-7 bohr apart,
    # which drives the Gaussian product centre and the Boys argument into the
    # 1e-14 regime where naive upward recursion would fail.
    "stress": {
        "description": "stress: two nearly coincident H2 fragments (separation 1e-7 bohr)",
        "atoms": [
            ("H", [0.0, 0.0, 0.0]),
            ("H", [1e-7, 0.0, 0.0]),
            ("H", [0.0, 0.0, 1.4]),
            ("H", [0.0, 1e-7, 1.4]),
        ],
    },
}

Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def main():
    outdir = os.path.join("data", "molecules")
    os.makedirs(outdir, exist_ok=True)
    for name, mol in MOLECULES.items():
        xyz = bohr([a[1] for a in mol["atoms"]])
        doc = {
            "name": name,
            "description": mol["description"],
            "units": "bohr",
            "atoms": [
                {"element": el, "Z": Z[el], "xyz": xyz[i]} for i, (el, _) in enumerate(mol["atoms"])
            ],
        }
        path = os.path.join(outdir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
