"""ctypes bindings for the cuERI integral library.

Example
-------
>>> from cuest import Cuest
>>> lib = Cuest()
>>> basis = lib.basis("data/molecules/h2o.json", "sto-3g")
>>> S = lib.overlap(basis)
>>> S.shape
(7, 7)
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_double, c_int, c_size_t, c_void_p

import numpy as np

CUEST_SUCCESS = 0
CUEST_FP32 = 0
CUEST_FP64 = 1
CUEST_BACKEND_CPU = 0
CUEST_BACKEND_CUDA = 1
CUEST_ERI_ALGO_MD = 0
CUEST_ERI_ALGO_RYS = 1


class CuestError(RuntimeError):
    """Raised when a cuERI entry point returns a non-success status."""


def find_library() -> str:
    """Locate the shared library (CUEST_LIBRARY overrides)."""
    env = os.environ.get("CUEST_LIBRARY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    names = ["libcuest_eri.dylib", "libcuest_eri.so", "cuest_eri.dll"]
    for base in (os.path.join(root, "build"), root, os.path.join(root, "install", "lib")):
        for name in names:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
    raise CuestError(
        "libcuest_eri not found; build the project or set CUEST_LIBRARY to the shared library path"
    )


class _Status(ctypes.Structure):
    _fields_ = [("quartets_total", c_size_t), ("quartets_screened", c_size_t),
                ("primitive_pairs", c_size_t), ("kernel_ms", c_double),
                ("max_shell_am", c_int)]


class EriStats:
    def __init__(self, quartets_total=0, quartets_screened=0, primitive_pairs=0,
                 kernel_ms=0.0, max_shell_am=0):
        self.quartets_total = quartets_total
        self.quartets_screened = quartets_screened
        self.primitive_pairs = primitive_pairs
        self.kernel_ms = kernel_ms
        self.max_shell_am = max_shell_am

    def __repr__(self) -> str:
        return (f"EriStats(total={self.quartets_total}, screened={self.quartets_screened}, "
                f"primitive_pairs={self.primitive_pairs}, kernel_ms={self.kernel_ms:.3f}, "
                f"max_am={self.max_shell_am})")


class Cuest:
    """A thin, typed wrapper around the C API."""

    def __init__(self, library: str | None = None):
        self._lib = ctypes.CDLL(library or find_library())
        self._configure()
        self._algo = CUEST_ERI_ALGO_MD
        self._handle = c_void_p()
        self._check(self._lib.cuestCreate(byref(self._handle)))

    # -- C API plumbing -----------------------------------------------------
    def _configure(self) -> None:
        L = self._lib
        L.cuestCreate.argtypes = [POINTER(c_void_p)]
        L.cuestCreate.restype = c_int
        L.cuestDestroy.argtypes = [c_void_p]
        L.cuestDestroy.restype = c_int
        L.cuestGetBackend.argtypes = [c_void_p, POINTER(c_int)]
        L.cuestGetBackend.restype = c_int
        L.cuestSetEriAlgo.argtypes = [c_void_p, c_int]
        L.cuestSetEriAlgo.restype = c_int
        L.cuestGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
        L.cuestGetErrorString.restype = c_int
        L.cuestBasisCreateFromFile.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(c_void_p)]
        L.cuestBasisCreateFromFile.restype = c_int
        L.cuestBasisDestroy.argtypes = [c_void_p]
        L.cuestBasisDestroy.restype = c_int
        L.cuestBasisNumFunctions.argtypes = [c_void_p, POINTER(c_int)]
        L.cuestBasisNumFunctions.restype = c_int
        L.cuestBasisNumShells.argtypes = [c_void_p, POINTER(c_int)]
        L.cuestBasisNumShells.restype = c_int
        L.cuestBasisShellInfo.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_int),
                                          POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        L.cuestBasisShellInfo.restype = c_int
        L.cuestSchwarzBounds.argtypes = [c_void_p, c_void_p, POINTER(c_double)]
        L.cuestSchwarzBounds.restype = c_int
        for fn in ("cuestOverlapMatrix", "cuestKineticMatrix", "cuestNuclearMatrix"):
            f = getattr(L, fn)
            f.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_double), c_int]
            f.restype = c_int
        L.cuestERITensor.argtypes = [c_void_p, c_void_p, c_int, c_double, POINTER(c_double),
                                     POINTER(_Status)]
        L.cuestERITensor.restype = c_int

    def _check(self, status: int) -> None:
        if status == CUEST_SUCCESS:
            return
        msg = c_char_p()
        self._lib.cuestGetErrorString(status, byref(msg))
        text = msg.value.decode() if msg.value else f"status {status}"
        raise CuestError(f"cuERI error: {text} ({status})")

    # -- public API ---------------------------------------------------------
    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.cuestDestroy(handle)

    def close(self) -> None:
        if self._handle:
            self._lib.cuestDestroy(self._handle)
            self._handle = c_void_p()

    def backend(self) -> str:
        b = c_int()
        self._check(self._lib.cuestGetBackend(self._handle, byref(b)))
        return "cuda" if b.value == CUEST_BACKEND_CUDA else "cpu"

    def set_eri_algo(self, algo: str) -> None:
        value = CUEST_ERI_ALGO_MD if algo == "md" else CUEST_ERI_ALGO_RYS
        self._check(self._lib.cuestSetEriAlgo(self._handle, value))
        self._algo = value

    def basis(self, molecule_path: str, basis_name: str) -> "Basis":
        handle = c_void_p()
        self._check(self._lib.cuestBasisCreateFromFile(
            self._handle, str(molecule_path).encode(), basis_name.encode(), byref(handle)))
        nbf = c_int()
        self._check(self._lib.cuestBasisNumFunctions(handle, byref(nbf)))
        return Basis(self, handle, nbf.value)

    def _matrix(self, fn_name: str, basis: "Basis", precision: str) -> np.ndarray:
        out = np.zeros((basis.nbf, basis.nbf), dtype=np.float64)
        prec = CUEST_FP32 if precision == "fp32" else CUEST_FP64
        fn = getattr(self._lib, fn_name)
        self._check(fn(self._handle, basis._handle, prec,
                       out.ctypes.data_as(POINTER(c_double)), basis.nbf))
        return out

    def overlap(self, basis: "Basis", precision: str = "fp64") -> np.ndarray:
        return self._matrix("cuestOverlapMatrix", basis, precision)

    def kinetic(self, basis: "Basis", precision: str = "fp64") -> np.ndarray:
        return self._matrix("cuestKineticMatrix", basis, precision)

    def nuclear(self, basis: "Basis", precision: str = "fp64") -> np.ndarray:
        return self._matrix("cuestNuclearMatrix", basis, precision)

    def schwarz(self, basis: "Basis") -> np.ndarray:
        ns = basis.nshells
        out = np.zeros((ns, ns), dtype=np.float64)
        self._check(self._lib.cuestSchwarzBounds(
            self._handle, basis._handle, out.ctypes.data_as(POINTER(c_double))))
        return out

    def eri_tensor(self, basis: "Basis", precision: str = "fp64", screen_tol: float = 0.0,
                   want_stats: bool = False):
        nbf = basis.nbf
        out = np.zeros((nbf, nbf, nbf, nbf), dtype=np.float64)
        stats = _Status()
        prec = CUEST_FP32 if precision == "fp32" else CUEST_FP64
        self._check(self._lib.cuestERITensor(
            self._handle, basis._handle, prec, float(screen_tol),
            out.ctypes.data_as(POINTER(c_double)), byref(stats)))
        if want_stats:
            return out, EriStats(stats.quartets_total, stats.quartets_screened,
                                 stats.primitive_pairs, stats.kernel_ms,
                                 stats.max_shell_am)
        return out


class Basis:
    """Opaque basis handle."""

    def __init__(self, owner: Cuest, handle, nbf: int):
        self._owner = owner
        self._handle = handle
        self.nbf = nbf
        ns = c_int()
        owner._check(owner._lib.cuestBasisNumShells(handle, byref(ns)))
        self.nshells = ns.value

    def shells(self):
        """Return shell metadata as a list of dicts."""
        out = []
        for i in range(self.nshells):
            atom = c_int()
            am = c_int()
            nprim = c_int()
            nctr = c_int()
            off = c_int()
            self._owner._check(self._owner._lib.cuestBasisShellInfo(
                self._handle, i, byref(atom), byref(am), byref(nprim), byref(nctr), byref(off)))
            out.append(dict(atom=atom.value, am=am.value, nprim=nprim.value,
                            nctr=nctr.value, func_offset=off.value))
        return out

    def __del__(self):
        if getattr(self, "_handle", None):
            self._owner._lib.cuestBasisDestroy(self._handle)
            self._handle = c_void_p()


__all__ = ["Cuest", "Basis", "CuestError", "EriStats", "find_library"]
