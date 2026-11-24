"""Controlled direct TD records from the immutable BH/GS/WH baseline equations.

BS equilibrium and QH transfer records require different production adapters;
they are explicitly unsupported here, not approximated by a reflective wall.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .baseline_runtime import FrozenRuntime, load_frozen_runtime

SUPPORTED_TD_MODELS = ("BH", "GS", "WH")
SOLVER_ID = "frozen_axial_fd4_rk4_td_v1"


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _parameters(model: str, params: Mapping[str, float]) -> dict[str, float]:
    if model not in SUPPORTED_TD_MODELS:
        raise NotImplementedError(f"TD adapter supports {SUPPORTED_TD_MODELS}; {model!r} has no implemented production adapter")
    allowed = {"BH": set(), "GS": {"compactness"}, "WH": {"r0_over_M"}}[model]
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown {model} parameters: {sorted(unknown)}")
    result = {k: float(v) for k, v in params.items()}
    if model == "GS":
        c = result.get("compactness", .47)
        if not math.isfinite(c) or not 0 < c < .5:
            raise ValueError("GS compactness must lie in (0, 0.5)")
        result["compactness"] = c
    elif model == "WH":
        r0 = result.get("r0_over_M", 2.0005)
        if not math.isfinite(r0) or r0 <= 2:
            raise ValueError("WH r0_over_M must exceed 2")
        result["r0_over_M"] = r0
    return result


def refinement_spacings(model: str, params: Mapping[str, float], *,
                        coarse_dx: float = .2, levels: int = 3,
                        notebook_path: str | Path | None = None) -> tuple[float, ...]:
    """Return nested *actual* spacings, including shell alignment for GS.

    At compactness .47 the baseline grid rounds requested (.2,.1,.05) to
    (23,46,91) interior cells.  This helper instead requests (23,46,92), so
    Richardson comparisons really use ratio two.  Actual spacing is verified
    again by the grid construction; the caller must retain actual_dx as well.
    """
    model = model.upper()
    p = _parameters(model, params)
    h = _positive(coarse_dx, "coarse_dx")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    if model != "GS":
        return tuple(h / 2**i for i in range(levels))
    rt = load_frozen_runtime(notebook_path)
    cfg = rt["Config"](dx=h)
    x, _, _, meta = rt["gravastar_grid"](p["compactness"], cfg=cfg, dx=h)
    interior_length = meta["shell_x"] - meta["center_x"]
    n0 = int(meta["shell_index"])
    spacings = tuple(interior_length / (n0 * 2**i) for i in range(levels))
    actual = []
    for i, requested in enumerate(spacings):
        xx, _, _, mm = rt["gravastar_grid"](p["compactness"], cfg=cfg, dx=requested)
        if mm["shell_index"] != n0 * 2**i:
            raise ArithmeticError("Unable to construct nested GS interior cells")
        actual.append(float(np.mean(np.diff(xx))))
    if len(actual) > 1 and not np.allclose(np.array(actual[:-1]) / actual[1:], 2., rtol=1e-12, atol=0.):
        raise ArithmeticError("Actual GS grid is not a ratio-two refinement")
    return spacings


def run_td(model: str, params: Mapping[str, float], *, dx: float,
           t_end: float = 520., cfl: float = .35, source_global: float = 55.,
           observer_global: float = 95., sigma: float = 4., x_right: float = 400.,
           x_left: float | None = None, sample_every: int = 2,
           notebook_path: str | Path | None = None) -> dict[str, Any]:
    """Run one controlled, M=1, ell=2, unit-left-going Gaussian experiment.

    x_right and x_left are in the solver's native coordinate, explicitly saved
    alongside global source/observer positions.  BH defaults to a horizon-side
    boundary at -250M (or farther for long records), WH to -400M.  An explicitly
    short domain is permitted for controlled robustness/replay tests, but its
    boundary_causal check fails and the record must not be accepted as a
    boundary-cleared library point.  No time or amplitude fit is performed.
    """
    model = str(model).upper()
    p = _parameters(model, params)
    dx, t_end, sigma = (_positive(dx, "dx"), _positive(t_end, "t_end"),
                        _positive(sigma, "sigma"))
    cfl = float(cfl)
    if not math.isfinite(cfl) or not 0 < cfl < .55:
        raise ValueError("cfl must lie in (0, 0.55), the frozen solver's accepted range")
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    source_global, observer_global, x_right = map(float, (source_global, observer_global, x_right))
    if not all(math.isfinite(v) for v in (source_global, observer_global, x_right)):
        raise ValueError("Source, observer and boundary coordinates must be finite")
    if x_left is not None and not math.isfinite(float(x_left)):
        raise ValueError("x_left must be finite")
    rt = load_frozen_runtime(notebook_path)
    offset = float(rt["schwarzschild_rstar"](p["r0_over_M"])) if model == "WH" else 0.
    source_native, observer_native = source_global - offset, observer_global - offset
    if model == "WH" and min(source_native, observer_native) <= 0:
        raise ValueError("Controlled WH source and observer must lie in the right exterior")
    if model == "GS":
        surface_radius = 1. / p["compactness"]
        if not math.isfinite(surface_radius):
            raise ValueError("GS surface radius is not representable at this compactness")
        shell = float(rt["schwarzschild_rstar"](surface_radius))
        if min(source_global, observer_global) <= shell:
            raise ValueError("Controlled source and observer must lie in the Schwarzschild exterior")
        if x_right <= shell:
            raise ValueError("GS right boundary must lie beyond its surface")
    cfg = rt["Config"](M=1., ell=2, dx=dx, cfl=cfl, t_max=t_end,
                       pulse_center=source_native, extraction_x=observer_native,
                       pulse_width=sigma, sample_every=sample_every)
    if model == "BH":
        # Conservative 8sigma front, plus 20M guard; never shorten the -250M default.
        required = .5 * (source_native + observer_native - 8*sigma - t_end - 20.)
        # Choose physical boundary placement independently of refinement h.
        left = -max(250., 10.*math.ceil(-required / 10.)) if x_left is None else float(x_left)
        if left >= x_right:
            raise ValueError("BH left boundary must precede the right boundary")
        x, V, r, grid_meta = rt["black_hole_grid"](cfg=cfg, dx=dx, x_left=left, x_right=x_right)
        left_bc = "outgoing"
        interface = None
        scope = "Schwarzschild axial Regge-Wheeler reference"
        x_interface = None
    elif model == "GS":
        if x_left is not None:
            raise ValueError("GS left boundary is the physical regular centre; x_left is not adjustable")
        x, V, r, grid_meta = rt["gravastar_grid"](p["compactness"], cfg=cfg, dx=dx, x_right=x_right)
        left_bc = "regular_center"
        interface = None
        scope = "Zero-surface-energy thin-shell gravastar, axial subclass"
        x_interface = float(grid_meta["shell_x"])
    else:
        left = -400. if x_left is None else float(x_left)
        if left >= 0 or x_right <= 0:
            raise ValueError("WH domain must straddle the throat")
        x, V, r, grid_meta = rt["wormhole_grid"](p["r0_over_M"], cfg=cfg, dx=dx,
                                                x_left=left, x_right=x_right)
        left_bc = "outgoing"
        interface = {"x": 0., "Lambda": grid_meta["throat_Lambda"]}
        scope = "Yang-Xia-Miao thin-shell Schwarzschild wormhole with axial delta throat; not Damour-Solodukhin"
        x_interface = offset
    if len(x) < 12 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(V)):
        raise ValueError("Invalid or underresolved spatial grid")
    if not x[4] < observer_native < x[-5]:
        raise ValueError("Observer outside safe cubic-extraction interior")
    if not x[4] < source_native - 8*sigma or not source_native + 8*sigma < x[-5]:
        raise ValueError("Initial Gaussian 8sigma region must lie inside the numerical domain")
    actual_dx = float(np.mean(np.diff(x)))
    dt = cfl * actual_dx
    if t_end < 8*dt:
        raise ValueError("At least eight time steps are required for the inherited stability monitor")
    t, records, numerical = rt["evolve_wave"](
        x, V, t_max=t_end, cfl=cfl, pulse_center=source_native, pulse_width=sigma,
        extraction_points={"observer": observer_native}, left_bc=left_bc,
        sample_every=sample_every, ell=2, interface_jump=interface,
    )
    psi = records["observer"]
    # Conservative domain-of-dependence diagnostic, not a rigorous compact front:
    # Gaussian support is infinite.  The 8sigma amplitude is exp(-32).
    right_return = float(2*x[-1] - source_native - observer_native - 8*sigma)
    left_return = float(source_native + observer_native - 2*x[0] - 8*sigma) if left_bc == "outgoing" else None
    earliest = min(v for v in (left_return, right_return) if v is not None)
    margin = earliest - float(t[-1])
    actual_radii = rt["r_from_schwarzschild_rstar"](np.array([source_global, observer_global]))
    finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(psi)) and np.all(np.diff(t) > 0))
    checks = {
        "finite": {"passed": finite},
        "geometry": {"passed": True, "source_areal_radius_over_M": float(actual_radii[0]),
                     "observer_areal_radius_over_M": float(actual_radii[1]),
                     "coordinate_mapping": "global = native + rstar0 on WH right exterior; global = native for BH/GS"},
        "stability": {"passed": bool(numerical["stability_pass"]),
                      "max_amp_ratio": numerical["max_amp_ratio"]},
        "boundary_causal": {"passed": bool(margin > 0), "margin_M": margin,
                            "earliest_return_M": earliest, "left_return_M": left_return,
                            "right_return_M": right_return, "gaussian_threshold": math.exp(-32),
                            "scope": "8sigma practical causal bound; not a replacement for domain-variation checks"},
    }
    if model == "WH":
        jumps = numerical["internal_interface"]
        max_jump = max(abs(jumps["final_Psi_jump_residual"]), abs(jumps["final_Pi_jump_residual"]))
        checks["interface"] = {"passed": bool(max_jump < 1e-10),
                               "max_final_jump_residual": max_jump,
                               "tolerance": 1e-10}
    elif model == "GS":
        checks["interface"] = {"passed": bool(grid_meta["shell_aligned_error"] < 1e-10),
                               "shell_alignment_error": grid_meta["shell_aligned_error"]}
    status = "OK" if finite and numerical["stability_pass"] else "STABILITY_FAILED"
    metadata = {**grid_meta, **numerical, "model_family": model, "params": p,
                "M_ADM": 1., "ell": 2, "polarisation": "axial", "units": "G=c=M_ADM=1",
                "time_convention": "exp(-i omega t)", "field": "Q_RW",
                "requested_dx": dx, "actual_dx": actual_dx,
                "source_global": source_global, "observer_global": observer_global,
                "source_native": source_native, "observer_native": observer_native,
                "source_areal_radius_over_M": float(actual_radii[0]),
                "observer_areal_radius_over_M": float(actual_radii[1]),
                "native_to_global_offset_right_exterior": offset,
                "x_interface_global": x_interface, "pulse_shape": "exp(-(x-x_source)^2/(2 sigma^2))",
                "sigma": sigma, "initial_time": 0., "initial_amplitude": 1.,
                "sample_every": sample_every, "record_end": float(t[-1]),
                "solver_id": SOLVER_ID, "solver_status": status,
                "numerics_evidence": "UNQUANTIFIED", "model_scope": scope,
                "baseline_provenance": rt.provenance,
                "boundary_causal_pass": checks["boundary_causal"]["passed"],
                "boundary_margin_M": margin}
    return {"t": t, "psi": psi, "x": x, "V": V, "r": r,
            "metadata": metadata, "checks": checks}
