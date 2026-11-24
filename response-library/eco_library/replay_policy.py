"""Scope of exact replay and the roundoff bound for portable reproduction checks.

Exact (bitwise) replay of a stored record is a property of an environment, not of
the algorithm: the frozen controlled BS/QH records replay bit-for-bit under the
Windows runtime below (Codex verification, 2026-11-22, 118/118 library tests),
while a Linux x86_64 runtime with identical numpy/scipy/h5py versions differs by
at most 7.4e-15 (different C math library and SIMD code paths).  Exact tests are
therefore asserted only in their reference environment.  Everywhere else a
portable check bounds the deviation by an a-priori forward roundoff estimate for
the same algorithm, derived from the number of time steps; it is not a fitted or
observed tolerance.
"""
from __future__ import annotations

import math
import platform
from numbers import Integral, Real

import numpy as np
import scipy

ENVIRONMENT_KEYS = ("system", "machine", "python", "numpy", "scipy")
REFERENCE_ENVIRONMENTS = {
    "frozen-controlled-records": {
        "system": "Windows", "machine": "AMD64", "python": "3.11.9", "numpy": "1.26.4", "scipy": "1.15.3",
        "basis": ("bitwise replay of baseline/eco_outputs_final controlled BS and QH records and "
                  "qh_measured_source.npz verified by Codex on this runtime (2026-11-22, 118/118 "
                  "library tests; producer provenance platform Windows-10-10.0.26300-SP0)"),
    },
}
UNIT_ROUNDOFF = 2.**-53
# Roundings that can reach one updated grid value per RK4 step: four stages, each a
# fourth-order five-point second-derivative stencil (5 multiplies, 4 adds), the potential
# term (2) and the stage update (2), i.e. 4*13 = 52, rounded up to 64.
OPS_PER_STEP = 64


def environment(**override):
    env = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
           "numpy": np.__version__, "scipy": scipy.__version__}
    env.update(override)
    return env


def matches_reference(name, env=None):
    """(matched, {key: (reference, actual)}) for a named reference environment."""
    reference = REFERENCE_ENVIRONMENTS[name]
    env = environment() if env is None else env
    differences = {k: (reference[k], env.get(k)) for k in ENVIRONMENT_KEYS if env.get(k) != reference[k]}
    return not differences, differences


def roundoff_bound(steps, scale):
    """Forward roundoff bound OPS_PER_STEP * u * steps * scale for a stable linear evolution.

    ``steps`` is the number of time steps actually taken and ``scale`` the largest
    magnitude of the evolved state (the unit initial Gaussian gives 1).  The bound
    grows linearly with the step count (no cancellation assumed), so it is a
    conservative worst case for the same algorithm under different rounding, not
    an error bound on the continuum solution.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return OPS_PER_STEP*UNIT_ROUNDOFF*int(steps)*float(scale)
