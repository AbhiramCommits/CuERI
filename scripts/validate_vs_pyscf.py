#!/usr/bin/env python3
"""Validate cuERI's S, T, V and full ERI tensor against PySCF/libcint.

The two codes use different Cartesian-component normalization: cuERI
unit-normalizes every Cartesian function, while PySCF (cart=True) normalizes
the (l,0,0) component and lets the other components inherit its scale.  The
difference is exactly diagonal, so we compare after applying
D = diag(sqrt(diag(S_pyscf))) to our matrices:

    S_pyscf = D S_ours D,   (mu nu|lambda sigma)_pyscf = D_mu D_nu D_l D_s (...)ours.

Outputs (both include the reference source label):
    results/accuracy_one_electron.csv
    results/accuracy_eri.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from cuest import Cuest  # noqa: E402

CASES = [
    ("h2o", "sto-3g"),
    ("h2o", "6-31g"),
    ("h2o", "6-31g*"),
    ("ch4", "sto-3g"),
    ("ch4", "6-31g"),
    ("ch4", "6-31g*"),
    ("benzene", "sto-3g"),
    ("benzene", "6-31g"),
    ("benzene", "6-31g*"),
]

PAIR = {(0, 0): "ss", (0, 1): "sp", (0, 2): "sd", (1, 1): "pp", (1, 2): "pd", (2, 2): "dd"}


def pair_name(a: int, b: int) -> str:
    return PAIR[(min(a, b), max(a, b))]


def load_pyscf(path: str, basis: str):
    from pyscf import gto

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    atoms = [(a["Z"], tuple(a["xyz"])) for a in doc["atoms"]]
    return gto.M(atom=atoms, basis=basis, unit="Bohr", verbose=0, cart=True)


def shell_indices(shells):
    """(L, [function indices]) for every shell."""
    out = []
    for s in shells:
        ncomp = (s["am"] + 1) * (s["am"] + 2) // 2
        idx = list(range(s["func_offset"], s["func_offset"] + s["nctr"] * ncomp))
        out.append((s["am"], idx))
    return out


def accumulate(store, key, diff, ref):
    ad = np.abs(diff)
    mx = float(ad.max()) if ad.size else 0.0
    floor = 1e-6 * float(np.abs(ref).max()) if ref.size else 0.0
    if floor <= 0.0:
        floor = 1e-300
    mask = np.abs(ref) >= floor
    rel = float((ad[mask] / np.abs(ref[mask])).max()) if np.any(mask) else 0.0
    ss = float(np.sum(diff * diff))
    rmax = float(np.abs(ref).max()) if ref.size else 0.0
    prev = store.get(key)
    if prev is None:
        store[key] = dict(max_abs=mx, max_rel=rel, sumsq=ss, count=int(diff.size), refmax=rmax)
    else:
        prev["max_abs"] = max(prev["max_abs"], mx)
        prev["max_rel"] = max(prev["max_rel"], rel)
        prev["sumsq"] += ss
        prev["count"] += int(diff.size)
        prev["refmax"] = max(prev["refmax"], rmax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="only H2O/STO-3G (used by the CI smoke check)")
    args = ap.parse_args()
    cases = [("h2o", "sto-3g")] if args.smoke else CASES
    import pyscf

    lib = Cuest()
    ref_source = f"PySCF/libcint {pyscf.__version__}"
    os.makedirs("results", exist_ok=True)
    rows_1e = []
    rows_eri = []
    for molecule, basis_name in cases:
        molpath = os.path.join("data", "molecules", f"{molecule}.json")
        mol = load_pyscf(molpath, basis_name)
        basis = lib.basis(molpath, basis_name)
        shells = shell_indices(basis.shells())
        Sp = mol.intor("int1e_ovlp")
        Tp = mol.intor("int1e_kin")
        Vp = mol.intor("int1e_nuc")
        D = np.sqrt(np.diag(Sp))
        Ep = mol.intor("int2e")
        print(f"[validate] {molecule}/{basis_name}: nbf={basis.nbf} nbas={mol.nbas}")

        for precision in ("fp64", "fp32"):
            S = lib.overlap(basis, precision)
            T = lib.kinetic(basis, precision)
            V = lib.nuclear(basis, precision)
            for name, X, Xp in (("overlap", S, Sp), ("kinetic", T, Tp), ("nuclear", V, Vp)):
                store = {}
                for la, ia in shells:
                    for lb, ib in shells:
                        block = X[np.ix_(ia, ib)]
                        ref = Xp[np.ix_(ia, ib)]
                        scaled = block * D[ia][:, None] * D[ib][None, :]
                        accumulate(store, pair_name(la, lb), scaled - ref, ref)
                for cls, v in sorted(store.items()):
                    rows_1e.append(dict(
                        molecule=molecule, basis=basis_name, precision=precision, matrix=name,
                        shell_class=cls, n_elements=v["count"], max_abs_error=v["max_abs"],
                        max_rel_error=v["max_rel"],
                        rms_error=float(np.sqrt(v["sumsq"] / max(v["count"], 1))),
                        max_scaled_error=(v["max_abs"] / v["refmax"] if v["refmax"] > 0 else 0.0),
                        rel_floor="1e-6 * max|ref| in class", reference_source=ref_source))
            E = lib.eri_tensor(basis, precision)
            by_l = {0: [], 1: [], 2: []}
            for la, ia in shells:
                by_l[la].extend(ia)
            store = {}
            for la, lb, lc, ld in product(range(3), repeat=4):
                ia, ib, ic, id_ = by_l[la], by_l[lb], by_l[lc], by_l[ld]
                if not ia or not ib or not ic or not id_:
                    continue
                block = E[np.ix_(ia, ib, ic, id_)]
                ref = Ep[np.ix_(ia, ib, ic, id_)]
                scaled = (block * D[ia][:, None, None, None] * D[ib][None, :, None, None] *
                          D[ic][None, None, :, None] * D[id_][None, None, None, :])
                accumulate(store, f"{pair_name(la, lb)}-{pair_name(lc, ld)}", scaled - ref, ref)
            del E
            for cls, v in sorted(store.items()):
                rows_eri.append(dict(
                    molecule=molecule, basis=basis_name, precision=precision, shell_class=cls,
                    n_elements=v["count"], max_abs_error=v["max_abs"],
                    max_rel_error=v["max_rel"],
                    rms_error=float(np.sqrt(v["sumsq"] / max(v["count"], 1))),
                    max_scaled_error=(v["max_abs"] / v["refmax"] if v["refmax"] > 0 else 0.0),
                    rel_floor="1e-6 * max|ref| in class", reference_source=ref_source))
            print(f"           {precision}: one-electron and ERI classes done")

    with open("results/accuracy_one_electron.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_1e[0].keys()))
        w.writeheader()
        w.writerows(rows_1e)
    with open("results/accuracy_eri.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_eri[0].keys()))
        w.writeheader()
        w.writerows(rows_eri)

    for precision in ("fp64", "fp32"):
        print(f"[validate] {precision} worst per shell class (ERI):")
        for cls in sorted({r["shell_class"] for r in rows_eri}):
            vals = [r for r in rows_eri if r["precision"] == precision and r["shell_class"] == cls]
            if not vals:
                continue
            w = max(vals, key=lambda r: r["max_scaled_error"])
            print(f"   {cls:8s} max rel {w['max_rel_error']:.3e}  scaled {w['max_scaled_error']:.3e}  ({w['molecule']}/{w['basis']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
