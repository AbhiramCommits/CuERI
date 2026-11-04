#!/usr/bin/env python3
"""fp32 vs fp64 error growth under near-degenerate exponents and near-coincident centres.

Two studies:
  * exponent_ratio: a two-centre H/He system whose exponents differ by a factor
    1 + delta, delta = 1e-1 ... 1e-10, on s, p and d shells;
  * separation: H2/STO-3G with the two centres 1e-1 ... 1e-10 bohr apart.

For every class we record the fp64 and fp32 errors against PySCF and the
fp32-vs-fp64 difference, plus the amplification relative to the mildest point.

Output: results/precision_error_growth.csv and
        results/figures/precision_error_growth.png
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from cuest import Cuest  # noqa: E402

PAIR = {(0, 0): "ss", (0, 1): "sp", (0, 2): "sd", (1, 1): "pp", (1, 2): "pd", (2, 2): "dd"}


def pair_name(a, b):
    return PAIR[(min(a, b), max(a, b))]


def pyscf_basis_from_elements(elements):
    from pyscf import gto

    out = {}
    labels = {0: "S", 1: "P", 2: "D"}
    for el, shells in elements.items():
        lines = []
        for s in shells:
            lines.append(labels[s["am"]])
            for i, e in enumerate(s["exponents"]):
                cs = " ".join("%.17g" % col[i] for col in s["coefficients"])
                lines.append(f"{e:.17g} {cs}")
        out[el] = gto.basis.parse("\n".join(lines))
    return out


def class_indices(shells):
    by_l = {0: [], 1: [], 2: []}
    pairs = []
    for s in shells:
        ncomp = (s["am"] + 1) * (s["am"] + 2) // 2
        idx = list(range(s["func_offset"], s["func_offset"] + s["nctr"] * ncomp))
        by_l[s["am"]].extend(idx)
        pairs.append((s["am"], idx))
    return by_l, pairs


def mat_errors(X64, X32, ref, D, pairs):
    acc = {}
    for la, ia in pairs:
        for lb, ib in pairs:
            if not ia or not ib:
                continue
            r = ref[np.ix_(ia, ib)]
            s64 = X64[np.ix_(ia, ib)] * D[ia][:, None] * D[ib][None, :]
            s32 = X32[np.ix_(ia, ib)] * D[ia][:, None] * D[ib][None, :]
            key = pair_name(la, lb)
            a64 = float(np.abs(s64 - r).max()) if r.size else 0.0
            a32 = float(np.abs(s32 - r).max()) if r.size else 0.0
            d32 = float(np.abs(s32 - s64).max()) if r.size else 0.0
            rmax = float(np.abs(r).max()) if r.size else 0.0
            prev = acc.get(key, (0.0, 0.0, 0.0, 0.0))
            acc[key] = (max(prev[0], a64), max(prev[1], a32), max(prev[2], d32),
                        max(prev[3], rmax))
    return {k: (v[0] / max(v[3], 1e-30), v[1] / max(v[3], 1e-30), v[2] / max(v[3], 1e-30))
            for k, v in acc.items()}


def eri_errors(E64, E32, ref, D, by_l):
    acc = {}
    for la, lb, lc, ld in product(range(3), repeat=4):
        ia, ib, ic, id_ = by_l[la], by_l[lb], by_l[lc], by_l[ld]
        if not ia or not ib or not ic or not id_:
            continue
        r = ref[np.ix_(ia, ib, ic, id_)]
        sc = D[ia][:, None, None, None] * D[ib][None, :, None, None] * \
            D[ic][None, None, :, None] * D[id_][None, None, None, :]
        s64 = E64[np.ix_(ia, ib, ic, id_)] * sc
        s32 = E32[np.ix_(ia, ib, ic, id_)] * sc
        key = f"{pair_name(la, lb)}-{pair_name(lc, ld)}"
        a64 = float(np.abs(s64 - r).max()) if r.size else 0.0
        a32 = float(np.abs(s32 - r).max()) if r.size else 0.0
        d32 = float(np.abs(s32 - s64).max()) if r.size else 0.0
        rmax = float(np.abs(r).max()) if r.size else 0.0
        prev = acc.get(key, (0.0, 0.0, 0.0, 0.0))
        acc[key] = (max(prev[0], a64), max(prev[1], a32), max(prev[2], d32), max(prev[3], rmax))
    return {k: (v[0] / max(v[3], 1e-30), v[1] / max(v[3], 1e-30), v[2] / max(v[3], 1e-30))
            for k, v in acc.items()}


def evaluate(lib, molpath, basis_name, mol, label, param, rows, ref_label):
    """Compute fp64/fp32 integrals and PySCF references for one configuration."""
    basis = lib.basis(molpath, basis_name)
    shells = basis.shells()
    by_l, pairs = class_indices(shells)
    Sp = mol.intor("int1e_ovlp")
    Tp = mol.intor("int1e_kin")
    Vp = mol.intor("int1e_nuc")
    Ep = mol.intor("int2e")
    D = np.sqrt(np.diag(Sp))
    res = {}
    for precision in ("fp64", "fp32"):
        res[precision] = dict(S=lib.overlap(basis, precision), T=lib.kinetic(basis, precision),
                              V=lib.nuclear(basis, precision),
                              E=lib.eri_tensor(basis, precision))
    for obs, ref in (("overlap", Sp), ("kinetic", Tp), ("nuclear", Vp)):
        key = obs[0].upper() if obs != "kinetic" else "T"
        key = {"overlap": "S", "kinetic": "T", "nuclear": "V"}[obs]
        errs = mat_errors(res["fp64"][key], res["fp32"][key], ref, D, pairs)
        for cls, (e64, e32, d32) in errs.items():
            rows.append(dict(study=label, parameter="value", value=f"{param:.3e}",
                             observable=obs, shell_class=cls,
                             max_err_fp64_vs_ref=f"{e64:.6e}",
                             max_err_fp32_vs_ref=f"{e32:.6e}",
                             max_err_fp32_vs_fp64=f"{d32:.6e}",
                             reference_source=ref_label))
    errs = eri_errors(res["fp64"]["E"], res["fp32"]["E"], Ep, D, by_l)
    for cls, (e64, e32, d32) in errs.items():
        rows.append(dict(study=label, parameter="value", value=f"{param:.3e}",
                         observable="eri", shell_class=cls,
                         max_err_fp64_vs_ref=f"{e64:.6e}",
                         max_err_fp32_vs_ref=f"{e32:.6e}",
                         max_err_fp32_vs_fp64=f"{d32:.6e}",
                         reference_source=ref_label))
    return res


def main() -> int:
    import pyscf
    from pyscf import gto

    lib = Cuest()
    ref_label = f"PySCF/libcint {pyscf.__version__}"
    rows = []

    # ---- Study 1: near-degenerate exponents ------------------------------
    deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    tmp = tempfile.mkdtemp(prefix="cuest_stress_")
    try:
        os.makedirs(os.path.join(tmp, "basis"), exist_ok=True)
        molpath = os.path.join(tmp, "hhe.json")
        with open(molpath, "w", encoding="utf-8") as fh:
            json.dump({"units": "bohr",
                       "atoms": [{"Z": 1, "xyz": [0.0, 0.0, 0.0]},
                                 {"Z": 2, "xyz": [0.0, 0.0, 1.4]}]}, fh)
        for delta in deltas:
            ratio = 1.0 + delta
            elements = {
                "H": [{"am": 0, "exponents": [1.0], "coefficients": [[1.0]]},
                      {"am": 1, "exponents": [1.0], "coefficients": [[1.0]]},
                      {"am": 2, "exponents": [0.8], "coefficients": [[1.0]]}],
                "He": [{"am": 0, "exponents": [1.0 * ratio], "coefficients": [[1.0]]},
                       {"am": 1, "exponents": [1.0 * ratio], "coefficients": [[1.0]]},
                       {"am": 2, "exponents": [0.8 * ratio], "coefficients": [[1.0]]}],
            }
            with open(os.path.join(tmp, "basis", "stressbasis.json"), "w", encoding="utf-8") as fh:
                json.dump({"basis": "stressbasis",
                           "elements": {el: {"shells": sh} for el, sh in elements.items()}}, fh)
            old = os.environ.get("CUEST_DATA_DIR")
            os.environ["CUEST_DATA_DIR"] = tmp
            mol = gto.M(atom=[(1, (0, 0, 0)), (2, (0, 0, 1.4))],
                        basis=pyscf_basis_from_elements(elements), unit="Bohr", verbose=0,
                        cart=True, spin=1)
            evaluate(lib, molpath, "stressbasis", mol, "exponent_ratio", delta, rows, ref_label)
            if old is None:
                os.environ.pop("CUEST_DATA_DIR", None)
            else:
                os.environ["CUEST_DATA_DIR"] = old
            print(f"[precision] exponent_ratio delta={delta:.0e} done")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- Study 2: near-coincident centres -------------------------------
    seps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    tmp = tempfile.mkdtemp(prefix="cuest_sep_")
    try:
        for sep in seps:
            molpath = os.path.join(tmp, f"h2_{sep:.0e}.json")
            with open(molpath, "w", encoding="utf-8") as fh:
                json.dump({"units": "bohr",
                           "atoms": [{"Z": 1, "xyz": [0.0, 0.0, 0.0]},
                                     {"Z": 1, "xyz": [0.0, 0.0, sep]}]}, fh)
            mol = gto.M(atom=[(1, (0, 0, 0)), (1, (0, 0, sep))], basis="sto-3g", unit="Bohr",
                        verbose=0, cart=True)
            evaluate(lib, molpath, "sto-3g", mol, "separation", sep, rows, ref_label)
            print(f"[precision] separation={sep:.0e} bohr done")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- amplification relative to the mildest point ---------------------
    baseline = {}
    for r in rows:
        k = (r["study"], r["observable"], r["shell_class"])
        v = float(r["max_err_fp32_vs_fp64"])
        if k not in baseline or v > baseline[k]:
            # the mildest point is the first row per key; store it explicitly below
            pass
    first_value = {}
    for r in rows:
        k = (r["study"], r["observable"], r["shell_class"])
        if k not in first_value:
            first_value[k] = max(float(r["max_err_fp32_vs_fp64"]), 1e-30)
    for r in rows:
        k = (r["study"], r["observable"], r["shell_class"])
        r["amplification_fp32_vs_baseline"] = f"{float(r['max_err_fp32_vs_fp64']) / first_value[k]:.4e}"

    os.makedirs("results", exist_ok=True)
    fields = ["study", "parameter", "value", "observable", "shell_class",
              "max_err_fp64_vs_ref", "max_err_fp32_vs_ref", "max_err_fp32_vs_fp64",
              "amplification_fp32_vs_baseline", "reference_source"]
    with open("results/precision_error_growth.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, study, xlabel in ((axes[0], "exponent_ratio", "exponent ratio 1+delta"),
                              (axes[1], "separation", "centre separation (bohr)")):
        vals = sorted({float(r["value"]) for r in rows if r["study"] == study}, reverse=True)
        e32 = []
        e64 = []
        for v in vals:
            rr = [r for r in rows if r["study"] == study and float(r["value"]) == v]
            e32.append(max(float(r["max_err_fp32_vs_fp64"]) for r in rr))
            e64.append(max(float(r["max_err_fp64_vs_ref"]) for r in rr))
        ax.loglog(vals, [max(x, 1e-17) for x in e32], "o-", label="fp32 vs fp64", lw=1.4)
        ax.loglog(vals, [max(x, 1e-17) for x in e64], "s--", label="fp64 vs PySCF", lw=1.2)
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("max scaled error")
        ax.set_title(study)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/precision_error_growth.png", dpi=140)
    plt.close(fig)

    worst = max(rows, key=lambda r: float(r["max_err_fp32_vs_fp64"]))
    print(f"[precision] {len(rows)} rows; worst fp32-vs-fp64 scaled error "
          f"{worst['max_err_fp32_vs_fp64']} ({worst['study']}, {worst['observable']}, "
          f"{worst['shell_class']}, value {worst['value']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
