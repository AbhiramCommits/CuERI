#!/usr/bin/env python3
"""Occupancy / register-pressure reporting.

Tier A: runs Nsight Compute (ncu) per angular-momentum kernel and parses the
        achieved occupancy, registers per thread, memory and SM throughput.
Tier B: parses results/ptxas_verbose.txt for registers/thread, spill stores and
        loads and computes the analytic occupancy for the target architecture.
Tier C: writes only "not measured" rows with the reason.

Output: results/occupancy.csv (and, at Tier C, results/ptxas_verbose.txt with
the same explanation).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NOT_MEASURED = "not measured (no GPU on build machine)"

CLASSES = ["ss-ss", "ss-sp", "ss-pp", "sp-sp", "sp-pp", "pp-pp", "ss-sd", "sp-sd", "pp-sd",
           "sd-sd", "ss-pd", "sp-pd", "pp-pd", "pd-pd", "ss-dd", "sp-dd", "pp-dd", "sd-dd",
           "pd-dd", "dd-dd"]

# Conservative analytic-occupancy constants for sm_80 (Ampere GA100/GA102):
# 65536 registers per SM, 2048 threads per SM, 64 warps per SM, 101376 bytes of
# shared memory per SM.  Used only at Tier B.
SM80 = dict(regs_per_sm=65536, threads_per_sm=2048, warps_per_sm=64, smem_per_sm=101376,
            reg_alloc_granularity=256, smem_alloc_granularity=256)


def tier_of() -> str:
    try:
        with open(os.path.join(ROOT, "results", "environment.json"), encoding="utf-8") as fh:
            return json.load(fh)["tier"]
    except OSError:
        return "C"


def analytic_occupancy(regs_per_thread: int, smem_bytes: int = 0, block: int = 128) -> dict:
    regs_block = ((regs_per_thread * block + SM80["reg_alloc_granularity"] - 1)
                  // SM80["reg_alloc_granularity"]) * SM80["reg_alloc_granularity"]
    blocks_reg = SM80["regs_per_sm"] // regs_block
    blocks_thread = SM80["threads_per_sm"] // block
    smem_block = ((smem_bytes + SM80["smem_alloc_granularity"] - 1)
                  // SM80["smem_alloc_granularity"]) * SM80["smem_alloc_granularity"]
    blocks_smem = SM80["smem_per_sm"] // smem_block if smem_block > 0 else blocks_thread
    blocks = max(0, min(blocks_reg, blocks_thread, blocks_smem))
    return dict(registers_per_thread=regs_per_thread, shared_memory_bytes=smem_bytes,
                blocks_per_sm=blocks, threads_per_sm=blocks * block,
                occupancy=(blocks * block) / SM80["threads_per_sm"])


def parse_ptxas(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"Function properties for (\S+)\n(.*?)(?=\n\s*\n|\Z)", text, re.S):
        name, body = m.group(1), m.group(2)
        regs = re.search(r"Used (\d+) registers", body)
        spill_st = re.search(r"(\d+) bytes spill stores", body)
        spill_ld = re.search(r"(\d+) bytes spill loads", body)
        smem = re.search(r"(\d+) bytes smem", body)
        out.append(dict(kernel=name, registers=int(regs.group(1)) if regs else 0,
                        spill_stores=int(spill_st.group(1)) if spill_st else 0,
                        spill_loads=int(spill_ld.group(1)) if spill_ld else 0,
                        shared_memory=int(smem.group(1)) if smem else 0))
    return out


def main() -> int:
    tier = tier_of()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    rows = []
    if tier == "A":
        if shutil.which("ncu") is None:
            tier = "B"
        else:
            # Measure the per-class ERI kernel with Nsight Compute.
            exe = os.path.join(ROOT, "build", "bench", "bench_integrals")
            metrics = ("sm__warps_active.avg.pct_of_peak_sustained_active,"
                       "launch__registers_per_thread,"
                       "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
                       "sm__throughput.avg.pct_of_peak_sustained_elapsed")
            for cls in CLASSES:
                out_csv = os.path.join(ROOT, "results", f"ncu_{cls}.csv")
                cmd = ["ncu", "--csv", "--metrics", metrics, "--kernel-name",
                       f"cuest_eri_kernel.*", "-c", "1", "-o", out_csv, exe, "--quick"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                rows.append(dict(angular_class=cls, status="measured" if proc.returncode == 0
                                 else "not measured (ncu run failed)",
                                 registers_per_thread="", occupancy_pct="",
                                 memory_throughput_pct="", sm_throughput_pct="",
                                 spill_stores="", spill_loads="", blocks_per_sm="",
                                 shared_memory_bytes="", note="ncu CSV: " + out_csv))
    if tier == "B":
        ptx = os.path.join(ROOT, "results", "ptxas_verbose.txt")
        kernels = parse_ptx(ptx)
        for k in kernels:
            occ = analytic_occupancy(k["registers"], k["shared_memory"], 128)
            rows.append(dict(angular_class=k["kernel"], status="measured (ptxas)",
                             registers_per_thread=k["registers"],
                             occupancy_pct=f"{100 * occ['occupancy']:.2f}",
                             memory_throughput_pct=NOT_MEASURED, sm_throughput_pct=NOT_MEASURED,
                             spill_stores=k["spill_stores"], spill_loads=k["spill_loads"],
                             blocks_per_sm=occ["blocks_per_sm"],
                             shared_memory_bytes=k["shared_memory"],
                             note="analytic occupancy for sm_80"))
        if not kernels:
            tier = "C"
    if tier == "C":
        reason = ("Tier C build: no CUDA toolkit and no NVIDIA GPU on the build machine; "
                  "no kernel was compiled or run, so registers, spills and occupancy do not exist")
        for cls in CLASSES:
            rows.append(dict(angular_class=cls, status="not measured",
                             registers_per_thread=NOT_MEASURED, occupancy_pct=NOT_MEASURED,
                             memory_throughput_pct=NOT_MEASURED, sm_throughput_pct=NOT_MEASURED,
                             spill_stores=NOT_MEASURED, spill_loads=NOT_MEASURED,
                             blocks_per_sm=NOT_MEASURED, shared_memory_bytes=NOT_MEASURED,
                             note=reason))
        with open(os.path.join(ROOT, "results", "ptxas_verbose.txt"), "w", encoding="utf-8") as fh:
            fh.write("results/ptxas_verbose.txt\n")
            fh.write("=======================\n")
            fh.write(reason + "\n\n")
            fh.write("The CMake build adds -lineinfo -Xptxas -v and runs a ptxas capture "
                     "target only when a CUDA toolkit is found. This build found no nvcc.\n")

    with open(os.path.join(ROOT, "results", "occupancy.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[occupancy] tier {tier}: wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
