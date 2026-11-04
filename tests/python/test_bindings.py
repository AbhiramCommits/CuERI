"""pytest checks for the ctypes bindings and the committed golden integrals."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import cuest  # noqa: E402
from cuest import Cuest, CuestError  # noqa: E402


@pytest.fixture(scope="module")
def lib():
    return Cuest()


def test_handle_and_backend(lib):
    assert lib.backend() in ("cpu", "cuda")


def test_basis_metadata(lib):
    b = lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "sto-3g")
    assert b.nbf == 7
    assert b.nshells == 5
    shells = b.shells()
    assert shells[0]["am"] == 0 and shells[0]["func_offset"] == 0
    assert [s["am"] for s in shells] == [0, 0, 1, 0, 0]


def test_one_electron_shapes_and_normalization(lib):
    b = lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "sto-3g")
    S = lib.overlap(b)
    T = lib.kinetic(b)
    V = lib.nuclear(b)
    assert S.shape == (7, 7) and T.shape == (7, 7) and V.shape == (7, 7)
    assert np.allclose(S, S.T, atol=0, rtol=0)
    assert np.allclose(np.diag(S), 1.0, atol=1e-13)
    # kinetic is positive on the diagonal, nuclear attraction negative
    assert np.all(np.diag(T) > 0)
    assert np.all(np.diag(V) < 0)


def test_eri_symmetry_and_dtypes(lib):
    b = lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "sto-3g")
    E = lib.eri_tensor(b)
    assert E.shape == (7, 7, 7, 7)
    assert np.allclose(E, E.transpose(1, 0, 2, 3), atol=1e-14)
    assert np.allclose(E, E.transpose(0, 1, 3, 2), atol=1e-14)
    assert np.allclose(E, E.transpose(2, 3, 0, 1), atol=1e-14)
    # (00|00) for water/STO-3G is a known positive number
    assert E[0, 0, 0, 0] == pytest.approx(4.785065404705, abs=1e-9)


def test_algorithm_switch_and_stats(lib):
    b = lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "6-31g")
    lib.set_eri_algo("md")
    Emd, smd = lib.eri_tensor(b, want_stats=True)
    lib.set_eri_algo("rys")
    Erys, srys = lib.eri_tensor(b, want_stats=True)
    lib.set_eri_algo("md")
    assert np.abs(Emd - Erys).max() < 1e-12
    assert smd.quartets_total == srys.quartets_total
    assert smd.max_shell_am == 1
    assert smd.kernel_ms >= 0.0


def test_fp32_is_close(lib):
    b = lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "sto-3g")
    S64 = lib.overlap(b, "fp64")
    S32 = lib.overlap(b, "fp32")
    assert np.abs(S64 - S32).max() < 1e-5


def test_errors(lib):
    with pytest.raises(CuestError):
        lib.basis(os.path.join(ROOT, "data/molecules/h2o.json"), "no-such-basis")
    with pytest.raises(CuestError):
        lib.basis("/nonexistent.json", "sto-3g")


def test_golden_integrals(lib):
    """Regression against committed golden values (tolerance 1e-12, measured)."""
    golden = np.load(os.path.join(ROOT, "data/ref/golden_integrals.npz"))
    cases = [("h2o", "sto-3g"), ("ch4", "6-31g")]
    for molecule, basis_name in cases:
        tag = f"{molecule}_{basis_name}".replace("*", "s")
        b = lib.basis(os.path.join(ROOT, "data", "molecules", f"{molecule}.json"), basis_name)
        checks = [("S", lib.overlap(b)), ("T", lib.kinetic(b)), ("V", lib.nuclear(b)),
                  ("ERI", lib.eri_tensor(b, screen_tol=1e-12))]
        for name, got in checks:
            ref = golden[f"{tag}_{name}"]
            assert got.shape == ref.shape
            assert np.abs(got - ref).max() < 1e-12, f"{tag}_{name}"
