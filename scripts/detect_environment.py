#!/usr/bin/env python3
"""Detect the machine, choose the hardware tier, and write results/environment.json.

Tier A: CUDA toolkit + a working NVIDIA GPU.
Tier B: CUDA toolkit present, no GPU.
Tier C: no CUDA toolkit.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip()
    except Exception:  # noqa: BLE001 - detection must never fail the build
        return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        v = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return v or platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def nvcc_version() -> str:
    exe = shutil.which("nvcc")
    if not exe:
        return "none"
    txt = run([exe, "--version"])
    for line in txt.splitlines():
        if "release" in line:
            return line.strip()
    return txt.splitlines()[-1] if txt else "present"


def gpu_model() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "none"
    txt = run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return txt.splitlines()[0].strip() if txt else "none"


def compiler_version(exe: str) -> str:
    if not shutil.which(exe):
        return "none"
    txt = run([exe, "--version"])
    return txt.splitlines()[0].strip() if txt else "unknown"


def main() -> int:
    nvcc = nvcc_version()
    gpu = gpu_model()
    has_toolkit = nvcc != "none"
    has_gpu = gpu != "none"
    if has_toolkit and has_gpu:
        tier = "A"
        tier_text = "Tier A: CUDA toolkit + working NVIDIA GPU"
    elif has_toolkit:
        tier = "B"
        tier_text = "Tier B: CUDA toolkit present, no GPU"
    else:
        tier = "C"
        tier_text = "Tier C: no CUDA toolkit; CPU backend only"

    doc = {
        "timestamp_utc": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os": platform.system(),
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "memory_bytes": None,
        "compiler_cxx": compiler_version(os.environ.get("CXX", "c++")),
        "compiler_cc": compiler_version(os.environ.get("CC", "cc")),
        "cmake_version": run(["cmake", "--version"]).splitlines()[0]
        if shutil.which("cmake")
        else "none",
        "python_version": platform.python_version(),
        "nvcc_version": nvcc,
        "gpu_model": gpu,
        "cuda_toolkit_present": has_toolkit,
        "gpu_present": has_gpu,
        "tier": tier,
        "tier_text": tier_text,
        "notes": (
            "GPU kernel timings and Nsight Compute metrics are only measured at Tier A. "
            "At Tier B the CUDA sources compile and ptxas register/spill counts are real; "
            "all numerics and timings come from the CPU backend. At Tier C the CUDA sources "
            "are not compiled at all."
        ),
    }
    if sys.platform == "darwin":
        doc["memory_bytes"] = int(run(["sysctl", "-n", "hw.memsize"]) or 0) or None
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        doc["memory_bytes"] = int(line.split()[1]) * 1024
        except OSError:
            pass

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: doc[k] for k in
                      ("platform", "cpu_model", "cpu_logical_cores", "compiler_cxx",
                       "nvcc_version", "gpu_model", "tier", "tier_text")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
