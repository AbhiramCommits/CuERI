#!/usr/bin/env bash
# Reproduce every measurement end to end.
#
#   ./scripts/run_all.sh            full run (benchmark: 3 timed passes)
#   CUEST_BENCH_QUICK=1 ./scripts/run_all.sh   fast benchmark (1 pass)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "error: .venv/bin/python not found; create the virtualenv first (see README)" >&2
  exit 1
fi

echo "== environment =="
$PY scripts/detect_environment.py

echo "== reference tables =="
$PY scripts/gen_boys_reference.py
$PY scripts/gen_rys_tables.py

echo "== configure / build / test =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
{
  echo "cuERI test summary ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo
  echo "CTest:"
  ctest --test-dir build --output-on-failure 2>&1 | tail -4
  echo
  echo "pytest:"
  $PY -m pytest -q tests/python 2>&1 | tail -1
} > results/test_summary.txt

echo "== validation vs PySCF =="
$PY scripts/validate_vs_pyscf.py

echo "== RHF closed loop =="
$PY scripts/rhf_closed_loop.py

echo "== Boys corner cases =="
$PY scripts/boys_corner_cases.py

echo "== precision error growth =="
$PY scripts/precision_error_growth.py

echo "== Schwarz screening =="
$PY scripts/screening_study.py

echo "== benchmark =="
$PY scripts/benchmark.py

echo "== occupancy =="
$PY scripts/occupancy.py

echo "== assemble results =="
$PY scripts/make_results.py

echo "== pytest =="
$PY -m pytest -q tests/python

echo "done: results/RESULTS.md"
