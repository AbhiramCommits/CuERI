#!/usr/bin/env python3
"""Closed-loop restricted Hartree-Fock using only cuERI integrals.

Builds S, T, V and the full ERI tensor with cuERI, runs a NumPy RHF SCF and
compares the converged total energy with PySCF's RHF on the same Cartesian
basis.  The absolute energy difference is the headline correctness number.

Output: results/rhf_closed_loop.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from cuest import Cuest  # noqa: E402

CASES = [("h2o", "sto-3g"), ("h2o", "6-31g")]


def nuclear_repulsion(charges, xyz):
    e = 0.0
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            r = np.linalg.norm(np.array(xyz[i]) - np.array(xyz[j]))
            e += charges[i] * charges[j] / r
    return e


def rhf(S, T, V, eri, nelec, enuc, tol=1e-10, maxit=400, mixing=0.5):
    """Plain RHF SCF with linear mixing.  Returns (energy, n_iter, converged)."""
    n = S.shape[0]
    nocc = nelec // 2
    evals, evecs = np.linalg.eigh(S)
    X = evecs @ np.diag(evals ** -0.5) @ evecs.T
    H = T + V
    D = np.zeros((n, n))
    energy = 0.0
    converged = False
    it = 0
    for it in range(1, maxit + 1):
        J = np.einsum("ls,mnls->mn", D, eri, optimize=True)
        K = np.einsum("ls,mlns->mn", D, eri, optimize=True)
        G = J - 0.5 * K
        F = H + G
        e, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        Cocc = C[:, :nocc]
        Dnew = 2.0 * Cocc @ Cocc.T
        energy = 0.5 * np.sum(Dnew * (H + F)) + enuc
        delta = np.max(np.abs(Dnew - D))
        D = (1.0 - mixing) * D + mixing * Dnew
        if delta < tol:
            converged = True
            break
    return energy, it, converged


def main() -> int:
    import pyscf
    from pyscf import gto, scf

    lib = Cuest()
    rows = []
    for molecule, basis_name in CASES:
        molpath = os.path.join("data", "molecules", f"{molecule}.json")
        with open(molpath, encoding="utf-8") as fh:
            doc = json.load(fh)
        charges = [a["Z"] for a in doc["atoms"]]
        xyz = [a["xyz"] for a in doc["atoms"]]
        nelec = sum(charges)
        enuc = nuclear_repulsion(charges, xyz)
        basis = lib.basis(molpath, basis_name)
        S = lib.overlap(basis)
        T = lib.kinetic(basis)
        V = lib.nuclear(basis)
        eri = lib.eri_tensor(basis, screen_tol=1e-12)
        e_cueri, nit, conv = rhf(S, T, V, eri, nelec, enuc)
        atoms = [(z, tuple(x)) for z, x in zip(charges, xyz)]
        mol = gto.M(atom=atoms, basis=basis_name, unit="Bohr", verbose=0, cart=True)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.verbose = 0
        e_pyscf = mf.kernel()
        delta = abs(e_cueri - e_pyscf)
        rows.append(dict(molecule=molecule, basis=basis_name, nbf=basis.nbf, n_elec=nelec,
                         n_iter=nit, e_cueri=e_cueri, e_pyscf=e_pyscf,
                         abs_delta_hartree=delta, converged=conv,
                         reference_source=f"PySCF RHF {pyscf.__version__}"))
        print(f"[rhf] {molecule}/{basis_name}: E(cuERI)={e_cueri:.12f}  "
              f"E(PySCF)={e_pyscf:.12f}  |dE|={delta:.3e}  ({nit} it, conv={conv})")
    os.makedirs("results", exist_ok=True)
    with open("results/rhf_closed_loop.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
