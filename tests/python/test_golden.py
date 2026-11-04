"""Golden-integral regression test (kept separate so pytest can select it)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from cuest import Cuest  # noqa: E402


@pytest.fixture(scope="module")
def lib():
    return Cuest()


def test_golden_files_are_reproducible(lib):
    golden = np.load(os.path.join(ROOT, "data/ref/golden_integrals.npz"))
    for molecule, basis_name in (("h2o", "sto-3g"), ("ch4", "6-31g")):
        tag = f"{molecule}_{basis_name}".replace("*", "s")
        b = lib.basis(os.path.join(ROOT, "data", "molecules", f"{molecule}.json"), basis_name)
        got = {"S": lib.overlap(b), "T": lib.kinetic(b), "V": lib.nuclear(b),
               "ERI": lib.eri_tensor(b, screen_tol=1e-12)}
        for name, arr in got.items():
            ref = golden[f"{tag}_{name}"]
            assert np.abs(arr - ref).max() < 1e-12
