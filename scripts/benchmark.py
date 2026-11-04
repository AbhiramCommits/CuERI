#!/usr/bin/env python3
"""Run the integral benchmark and assemble results/benchmark.csv.

The C++ driver measures the CPU backends (1 thread and all OpenMP threads) for
every molecule/basis pair, both ERI algorithms and both precisions.  CUDA rows
are measured here through the ctypes layer when the library reports a CUDA
backend; otherwise they carry the literal string
"not measured (no GPU on build machine)".
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from cuest import Cuest  # noqa: E402

NOT_MEASURED = "not measured (no GPU on build machine)"


def main() -> int:
    exe = os.path.join(ROOT, "build", "bench", "bench_integrals")
    if not os.path.exists(exe):
        raise SystemExit(f"missing {exe}; build the project first")
    args = [exe]
    if os.environ.get("CUEST_BENCH_QUICK"):
        args.append("--quick")
    print("[bench] running", " ".join(args), flush=True)
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith("backend,")]
    raw = []
    for line in lines:
        parts = line.split(",")
        raw.append(dict(backend=parts[0], algorithm=parts[1], molecule=parts[2], basis=parts[3],
                        operation=parts[4], precision=parts[5], median_ms=float(parts[6]),
                        rate=float(parts[7]), rate_unit=parts[8], repeats=int(parts[9]),
                        screened_fraction=float(parts[10]), quartets_screened=int(parts[11]),
                        quartets_total=int(parts[12]), nbf=int(parts[13]),
                        speedup_vs_cpu1="", status="measured"))

    # speed-up vs single-thread CPU for the matching configuration
    lookup = {(r["molecule"], r["basis"], r["operation"], r["precision"], r["algorithm"]): r
              for r in raw if r["backend"] == "cpu-1thread"}
    for r in raw:
        if r["backend"] == "cpu-1thread":
            r["speedup_vs_cpu1"] = "1.000"
        else:
            base = lookup.get((r["molecule"], r["basis"], r["operation"], r["precision"],
                               r["algorithm"]))
            r["speedup_vs_cpu1"] = f"{base['median_ms'] / r['median_ms']:.3f}" if base else ""

    # CUDA rows
    tier = "C"
    try:
        with open(os.path.join(ROOT, "results", "environment.json"), encoding="utf-8") as fh:
            tier = json.load(fh)["tier"]
    except OSError:
        pass
    lib = Cuest()
    has_cuda = lib.backend() == "cuda"
    cases = sorted({(r["molecule"], r["basis"]) for r in raw})
    if has_cuda:
        print("[bench] measuring CUDA backends", flush=True)
    for molecule, basis_name in cases:
        molpath = os.path.join(ROOT, "data", "molecules", f"{molecule}.json")
        basis = lib.basis(molpath, basis_name)
        for precision, pname in (("fp64", "fp64"), ("fp32", "fp32")):
            for algo in ("md", "rys"):
                if has_cuda:
                    lib.set_eri_algo(algo)
                    for _ in range(1):
                        lib.eri_tensor(basis, precision, 1e-10)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        lib.eri_tensor(basis, precision, 1e-10)
                        times.append((time.perf_counter() - t0) * 1e3)
                    ms = statistics.median(times)
                    base = lookup.get((molecule, basis_name, "eri", pname, algo))
                    raw.append(dict(backend=f"cuda-{precision}", algorithm=algo, molecule=molecule,
                                    basis=basis_name, operation="eri", precision=pname,
                                    median_ms=ms, rate=basis.nbf ** 4 / (ms * 1e-3),
                                    rate_unit="integrals/s", repeats=3, screened_fraction="",
                                    quartets_screened="", quartets_total="", nbf=basis.nbf,
                                    speedup_vs_cpu1=(
                                        f"{base['median_ms'] / ms:.3f}" if base else ""),
                                    status="measured"))
                else:
                    raw.append(dict(backend=f"cuda-{precision}", algorithm=algo,
                                    molecule=molecule, basis=basis_name, operation="eri",
                                    precision=pname, median_ms=NOT_MEASURED, rate=NOT_MEASURED,
                                    rate_unit="integrals/s", repeats=0, screened_fraction="",
                                    quartets_screened="", quartets_total="", nbf=basis.nbf,
                                    speedup_vs_cpu1=NOT_MEASURED, status="not measured"))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "benchmark.csv")
    fields = ["backend", "algorithm", "molecule", "basis", "operation", "precision", "median_ms",
              "rate", "rate_unit", "repeats", "speedup_vs_cpu1", "screened_fraction",
              "quartets_screened", "quartets_total", "nbf", "status"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in raw:
            w.writerow({k: r.get(k, "") for k in fields})

    # headline table
    def find(mol, bas, prec="fp64", algo="md", op="eri", backend="cpu-openmp"):
        for r in raw:
            if (r["molecule"], r["basis"], r["operation"], r["precision"], r["algorithm"],
                    r["backend"]) == (mol, bas, op, prec, algo, backend):
                return r
        return None

    print("[bench] headline ERI timings (cpu-openmp, md, fp64):")
    for mol, bas in cases:
        r = find(mol, bas)
        if r:
            print(f"   {mol:8s} {bas:7s} nbf={r['nbf']:3d} {r['median_ms']:9.2f} ms  "
                  f"{r['rate']:.3e} integrals/s  speedup {r['speedup_vs_cpu1']}")
    print(f"[bench] tier {tier}; CUDA measured: {has_cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
