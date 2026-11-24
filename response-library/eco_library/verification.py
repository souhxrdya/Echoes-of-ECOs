"""Release-verification policy across runtimes (``library-verification/1.0``).

The stored catalogue is a deterministic function of immutable archives, but some of its numbers are
recomputed at assembly time with library mathematics that is not correctly rounded and differs between
builds.  Reproduced on Linux: disabling numpy's AVX512 dispatch (the SVML transcendental kernels that
numpy enables only in Linux builds) changes exactly 41 derived numbers of the 16-entry catalogue by at
most 7.9e-16 relative, the same count and size an independent Windows rebuild reported; no other field
changes.  The policy therefore has two tiers.

* ``SAME_RUNTIME``: the verifying runtime equals the reference runtime recorded beside the catalogue
  (``RUNTIME_REFERENCE``); the rebuild must be byte identical, as before.
* ``CROSS_RUNTIME``: any other runtime.  Every key, list length, string, integer, boolean and null must be
  identical, and so must every float that is copied rather than recomputed (parameters, experiment,
  coordinates, metadata, mode values, transfer gates, hashes).  Only the floats in ``DERIVED_PATHS``
  (assembly-time norms, observed orders, estimates and budget fractions) may differ, each by at most
  ``DERIVED_RTOL`` relative to max(|stored|, |rebuilt|, floor), with floor 1 for an observed order.
  Every scientific state is a string and so is compared exactly; in addition, a stored decision value
  lying within that tolerance of its threshold fails, because its state would not be portable.

Tolerance basis: the derived numbers are sums of squares (pairwise summation, deterministic across
platforms), square roots and quotients (correctly rounded), and one log and one power per estimate.
numpy documents at most 4 ulp for its SIMD transcendental kernels; propagating 5 ulp per call through
p = log(d_cm/d_mf)/log r and e = 1.25 d_mf/(r^p - 1) bounds the change by about 1e-14 relative for
p >= 0.1.  DERIVED_RTOL = 1e-12 keeps a factor 100 above that bound and about 1e3 above the observed
7.9e-16, while any change of a number by more than one part in 1e12 fails.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from pathlib import Path

POLICY_VERSION = "library-verification/1.0"
DERIVED_RTOL = 1e-12
DERIVED_PATHS = (
    re.compile(r"^/entries\[\d+\]/observable_resolution/[a-z]+/"
               r"(norm_L2|refinement_budget_L2|non_subdominant_controls_L2|budget_fraction_of_norm)$"),
    re.compile(r"^/entries\[\d+\]/observable_resolution/[a-z]+/non_subdominant_controls_relative/[A-Za-z0-9_]+$"),
    re.compile(r"^/entries\[\d+\]/refinement/[A-Z_]+/[a-z]+/(fine_relative_error_estimate|p_observed)$"),
)
ORDER_FLOOR_PATH = re.compile(r"/p_observed$")
DECLARED_ORDER_BY_SEQUENCE = {"BH": 4, "BHGS": 4, "GS": 2, "WH": 4, "BS": 4, "QH": 4, "QH_CORRECTION": 4}
ORDER_BAND = .5
RESOLUTION_FRACTION = .1


def runtime_fingerprint():
    """What can change a recomputed float: platform, C library, interpreter, numerical packages and
    the CPU features numpy dispatches to (its SIMD kernels)."""
    import h5py
    import numpy
    import scipy
    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    except ImportError:  # numpy < 2
        from numpy.core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    return {"system": platform.system(), "machine": platform.machine(), "libc": "-".join(platform.libc_ver()),
            "python_implementation": platform.python_implementation(), "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
            "numpy_dispatched_cpu_features": sorted(k for k in __cpu_dispatch__ if __cpu_features__.get(k))}


def is_derived(path):
    return any(p.match(path) for p in DERIVED_PATHS)


def _tolerance(path, a, b):
    floor = 1. if ORDER_FLOOR_PATH.search(path) else 0.
    return DERIVED_RTOL*max(abs(a), abs(b), floor)


def _walk(a, b, path, out):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a is not b and a != b or type(a) is not type(b):
            out["exact"].append({"path": path, "stored": a, "rebuilt": b})
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["structure"].append({"path": path, "only_stored": sorted(set(a)-set(b)),
                                     "only_rebuilt": sorted(set(b)-set(a))})
        for k in sorted(set(a) & set(b)):
            _walk(a[k], b[k], f"{path}/{k}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structure"].append({"path": path, "stored_length": len(a), "rebuilt_length": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
        return
    if isinstance(a, float) and isinstance(b, float) and is_derived(path):
        if a == b:
            return
        if not (math.isfinite(a) and math.isfinite(b)):
            out["derived_beyond_tolerance"].append({"path": path, "stored": a, "rebuilt": b})
            return
        tol = _tolerance(path, a, b)
        rel = abs(a-b)/max(abs(a), abs(b))
        row = {"path": path, "stored": a, "rebuilt": b, "relative": rel, "tolerance": tol}
        out["derived_differing"].append(row)
        if abs(a-b) > tol:
            out["derived_beyond_tolerance"].append(row)
        return
    if type(a) is not type(b) or a != b:
        out["exact"].append({"path": path, "stored": a, "rebuilt": b})


def decision_margins(catalogue):
    """Stored decision values closer to their threshold than the cross-runtime tolerance."""
    close = []
    for i, entry in enumerate(catalogue.get("entries", [])):
        for w, row in entry.get("observable_resolution", {}).items():
            x = row.get("budget_fraction_of_norm")
            if isinstance(x, float) and abs(x-RESOLUTION_FRACTION) <= DERIVED_RTOL*max(abs(x), RESOLUTION_FRACTION):
                close.append({"path": f"/entries[{i}]/observable_resolution/{w}/budget_fraction_of_norm", "value": x,
                              "threshold": RESOLUTION_FRACTION})
        for seq, windows in entry.get("refinement", {}).items():
            order = DECLARED_ORDER_BY_SEQUENCE.get(seq)
            for w, row in windows.items():
                p = row.get("p_observed")
                if order is None or not isinstance(p, float):
                    continue
                for edge in (order-ORDER_BAND, order+ORDER_BAND):
                    if abs(p-edge) <= DERIVED_RTOL*max(abs(p), 1.):
                        close.append({"path": f"/entries[{i}]/refinement/{seq}/{w}/p_observed", "value": p,
                                      "threshold": edge})
    return close


def compare(stored, rebuilt, *, same_runtime):
    """Policy comparison of a stored catalogue with a rebuild (both as parsed JSON)."""
    byte_identical = json.dumps(stored, sort_keys=True) == json.dumps(rebuilt, sort_keys=True)
    out = {"policy": POLICY_VERSION, "tier": "SAME_RUNTIME" if same_runtime else "CROSS_RUNTIME",
           "byte_identical": byte_identical, "structure": [], "exact": [], "derived_differing": [],
           "derived_beyond_tolerance": [], "derived_rtol": DERIVED_RTOL}
    _walk(stored, rebuilt, "", out)
    out["decision_margin_below_tolerance"] = decision_margins(stored)
    out["derived_differing_count"] = len(out["derived_differing"])
    out["derived_max_relative"] = max((r["relative"] for r in out["derived_differing"]), default=0.)
    if same_runtime:
        ok = byte_identical
    else:
        ok = not (out["structure"] or out["exact"] or out["derived_beyond_tolerance"]
                  or out["decision_margin_below_tolerance"])
    out["state"] = "PASS" if ok else "FAIL"
    return out


def reference_runtime(path):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    return data


def rebuild_policy(stored, rebuilt, reference, catalogue_sha256, fingerprint):
    """The tier is SAME_RUNTIME only if the verifying runtime equals the recorded reference runtime, and
    the record must describe this very catalogue file."""
    policy = compare(stored, rebuilt, same_runtime=fingerprint == reference.get("reference_runtime"))
    if reference.get("catalogue_sha256") != catalogue_sha256:
        policy.update(state="FAIL", reference_problem="the runtime record describes a different catalogue file")
    return policy


def same_runtime(reference, fingerprint=None):
    return (fingerprint or runtime_fingerprint()) == reference["reference_runtime"]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
