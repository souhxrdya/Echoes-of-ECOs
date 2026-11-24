"""Mini-boson-star equilibrium and controlled *axial* TD migration adapter.

The background is solved afresh, never loaded from saved profiles.  The reviewed
notebook declarations are pinned by their whole-notebook and declaration hashes.
Macedo et al., arXiv:1307.4812v3 (Phys. Rev. D 88, 064046), supply the selected model:
background Eqs. (6)--(11), mini potential Eq. (42), axial master equation and potential
Eqs. (26)--(27) and its source-free form Eq. (52) in the arXiv v3 PDF numbering; the
arXiv v3 HTML (LaTeXML) rendering numbers the same formulas (6)--(11), (52), (36)--(37)
and (62).  With sigma=sqrt(8*pi)*phi and mu=1, (kappa/2)(rho-p_rad)=sigma**2.
The scalar/polar dynamics and radial stability are NOT computed here.

Public API: ``run_boson_star(params, *, dx, ...)`` returns the usual
t/psi/x/V/r/metadata/checks record, plus ``equilibrium`` and ``initial_data``
array dictionaries. ``solve_equilibrium`` returns only the independently solved
background and diagnostics. No function writes files or uses a persistent cache.

The finite-radius Dirichlet scalar condition and subsequent Schwarzschild
extension are inherited approximations, not an exact finite stellar surface.
Background sampling (5000 points by default), quadrature, centre cutoff, and finite-tail
errors are separate from a TD mesh refinement and remain explicitly unquantified.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
import types
import uuid
from collections.abc import Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from .baseline_runtime import (
    FrozenRuntime, NOTEBOOK_SHA256, default_notebook_path, load_frozen_runtime,
)

FIDUCIAL_SIGMA_C = 0.270963318550685
SOLVER_ID = "frozen_mini_bs_ekg_axial_fd4_rk4_v1"
# This is a deliberately bounded implementation/pilot domain, NOT a proved
# stable branch or a claim that all intermediate points have been validated.
SIGMA_C_DOMAIN = (.18, .30)
DECLARATIONS = {
    15: ("BosonBackground", "solve_mini_boson_star", "boson_photon_spheres"),
    16: ("boson_axial_grid",),
    31: ("boson_native_tortoise_at_radius",),
}
# Equation numbers depend on the format.  Verified against the formulas on 2026-11-23:
# PDF https://arxiv.org/pdf/1307.4812v3 and HTML https://arxiv.org/html/1307.4812v3.
# Eq. 26 (PDF) / 36 (HTML): [d^2/dr*^2 + sigma^2 - V_RW] Psi_RW = S_RW;
# Eq. 27 / 37: V_RW = e^v [l(l+1)/r^2 - 6m/r^3 - (kappa/2)(p_rad - rho)];
# Eq. 42 / 52: V(|Phi|^2) = mu^2 |Phi|^2;  Eq. 52 / 62: the source-free RW equation.
LITERATURE = {
    "reference": "Macedo, Pani, Cardoso, Crispino, arXiv:1307.4812v3, Phys. Rev. D 88, 064046 (2013)",
    "url_pdf": "https://arxiv.org/pdf/1307.4812v3",
    "url_html": "https://arxiv.org/html/1307.4812v3",
    "equation_numbers": {
        "arXiv:1307.4812v3 PDF": {"background": "6-11", "axial_master_with_source": "26",
                                  "axial_potential": "27", "mini_potential": "42", "axial_source_free": "52"},
        "arXiv:1307.4812v3 HTML (LaTeXML)": {"background": "6-11", "axial_master_with_source": "36",
                                             "axial_potential": "37", "mini_potential": "52",
                                             "axial_source_free": "62"},
    },
    "equation_numbering_note": ("the two formats agree for Eqs. 6-11 and differ by ten from the perturbation "
                                "equations onward (checked for the entries above only); journal numbering not "
                                "checked"),
    "fiducial_benchmark": "Table I, mini BS I",
    "field_conversion": "sigma=sqrt(8*pi*G)*phi=sqrt(2)*phi_tilde",
}


def _sample_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 500 <= value <= 200000:
        raise ValueError("background_samples must be an integer in [500, 200000]")
    return value


def _parameterize_profile(node, samples):
    """Change only the audited post-BVP dense-output sample-count literal.

    The notebook hash and exact assignment shape must both match. No equation,
    BVP collocation mesh, boundary condition, or solver tolerance is transformed.
    """
    expected = ast.parse("rr=np.linspace(1e-5,rmax,5000)").body[0]
    matches = [statement for statement in node.body
               if ast.dump(statement, include_attributes=False)
               == ast.dump(expected, include_attributes=False)]
    if len(matches) != 1:
        raise ValueError("Reviewed BS dense-profile assignment changed")
    matches[0].value.args[2] = ast.copy_location(ast.Constant(samples), matches[0].value.args[2])


def load_boson_runtime(notebook_path: str | Path | None = None, *, background_samples=5000) -> FrozenRuntime:
    """Load reviewed declarations, optionally changing only dense sampling.

    At the default no AST transformation is made. Nondefault sampling is an
    explicit numerical variant, with original and executed AST provenance.
    """
    background_samples = _sample_count(background_samples)
    rt = load_frozen_runtime(notebook_path)
    path = default_notebook_path() if notebook_path is None else Path(notebook_path)
    raw = path.read_bytes()
    # A second guard also protects against the file changing after the TD load.
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Frozen notebook SHA-256 mismatch before BS declaration load")
    document = json.loads(raw)
    module_name = "_eco_frozen_bs_" + uuid.uuid4().hex
    module = types.ModuleType(module_name)
    ns = module.__dict__
    ns.update(rt.namespace)
    ns.update(__name__=module_name, solve_bvp=solve_bvp,
              cumulative_trapezoid=cumulative_trapezoid,
              CubicSpline=CubicSpline, PchipInterpolator=PchipInterpolator,
              brentq=brentq)
    manifest = list(rt.provenance["declarations"])
    sys.modules[module_name] = module
    try:
        for index, allowed in DECLARATIONS.items():
            source = "".join(document["cells"][index]["source"])
            nodes = [node for node in ast.parse(source).body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name in allowed]
            if len(nodes) != len(allowed) or {node.name for node in nodes} != set(allowed):
                raise ValueError(f"Reviewed BS declaration set changed in cell {index}")
            for node in nodes:
                snippet = ast.get_source_segment(source, node)
                original_ast = ast.dump(node, include_attributes=False)
                if node.name == "solve_mini_boson_star" and background_samples != 5000:
                    _parameterize_profile(node, background_samples)
                manifest.append({"name": node.name, "cell_index": index,
                                 "first_line": node.lineno, "last_line": node.end_lineno,
                                 "source_sha256": hashlib.sha256(snippet.encode()).hexdigest(),
                                 "original_ast_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
                                 "executed_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
            exec(compile(ast.Module(body=nodes, type_ignores=[]),
                         f"{path.name}:cell{index}", "exec"), ns)
    finally:
        del sys.modules[module_name]
    return FrozenRuntime(ns, {**rt.provenance, "loader": "reviewed_bs_ast_declarations_v2",
                              "numerical_transform": {"kind": "dense_profile_sample_count_only",
                                  "default_samples": 5000, "background_samples": background_samples,
                                  "applied": background_samples != 5000},
                              "declarations": manifest})


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _settings(params, rmax, n, tol):
    if not isinstance(params, Mapping) or set(params) - {"sigma_c"}:
        raise ValueError("Mini BS params accept only sigma_c (not phi_tilde_c or self-interaction)")
    sc = _finite(params.get("sigma_c", FIDUCIAL_SIGMA_C), "sigma_c", positive=True)
    if not SIGMA_C_DOMAIN[0] <= sc <= SIGMA_C_DOMAIN[1]:
        raise ValueError(f"sigma_c outside bounded continuation domain {SIGMA_C_DOMAIN}")
    rmax = _finite(rmax, "equilibrium_rmax", positive=True)
    tol = _finite(tol, "equilibrium_tol", positive=True)
    if not 30 <= rmax <= 200:
        raise ValueError("equilibrium_rmax must be in [30, 200] mu^-1 for this adapter")
    if isinstance(n, bool) or not isinstance(n, int) or not 100 <= n <= 50000:
        raise ValueError("equilibrium_n must be an integer in [100, 50000]")
    if not 1e-10 <= tol <= 1e-4:
        raise ValueError("equilibrium_tol must lie in [1e-10, 1e-4]")
    return sc, rmax, n, tol


def _background_diagnostics(bg, tol):
    """Fail closed on resolved nonphysical/incorrect-node background solutions."""
    arrays = {name: np.asarray(getattr(bg, name))
              for name in ("r", "sigma", "dsigma", "a", "alpha")}
    finite = all(np.isfinite(value).all() for value in arrays.values())
    if not finite or not (np.all(np.diff(bg.r) > 0) and np.all(bg.a > 0)
                          and np.all(bg.alpha > 0) and 0 < bg.omega_s < 1
                          and math.isfinite(bg.mass) and bg.mass > 0):
        raise ArithmeticError("Unphysical/nonfinite mini-BS equilibrium")
    mass = .5 * bg.r * (1 - 1 / bg.a**2)
    node_floor = 1e-8 * bg.sigma_c
    significant = bg.sigma[np.abs(bg.sigma) > node_floor]
    node_count = int(np.count_nonzero(np.diff(np.sign(significant))))
    if len(significant) == 0 or np.any(significant < 0) or node_count:
        raise ArithmeticError("Resolved scalar nodes/sign reversal: not the selected nodeless branch")
    residual_limit = max(2 * tol, 1e-10)
    monotonic_tolerance = 1e-8 * bg.mass
    checks = {
        "equilibrium": {"passed": bool(bg.residual <= residual_limit),
                        "BVP_residual": float(bg.residual), "tolerance": residual_limit,
                        "scope": "collocation/boundary residual, not total background error"},
        "nodeless_background": {"passed": True, "resolved_nodes": node_count,
                                "scalar_amplitude_floor": node_floor,
                                "scope": "resolved radial profile only; no radial-stability claim"},
        "mass_monotonicity": {"passed": bool(np.min(np.diff(mass)) >= -monotonic_tolerance),
                              "min_mass_increment": float(np.min(np.diff(mass))),
                              "roundoff_tolerance": monotonic_tolerance},
    }
    if not all(item["passed"] for item in checks.values()):
        raise ArithmeticError("Mini-BS equilibrium failed residual/mass monotonicity checks")
    x_mu = cumulative_trapezoid(bg.a / bg.alpha, bg.r, initial=0.)
    axial_V_mu = bg.alpha**2 * (6/bg.r**2 - 6*mass/bg.r**3 + bg.sigma**2)
    arrays.update(mass_mu=mass, x_mu=x_mu, axial_V_mu2=axial_V_mu,
                  r_over_M=bg.r/bg.mass, x_over_M=x_mu/bg.mass,
                  axial_V_M2=axial_V_mu*bg.mass**2)
    tail_start = int(np.searchsorted(bg.r, .8*bg.r[-1]))
    metadata = {
        "raw_mass_mu_units": float(bg.mass), "omega_s_over_mu": float(bg.omega_s),
        "omega_s_M": float(bg.omega_s*bg.mass), "radius_99_mu_units": float(bg.radius_99),
        "R99_over_M": float(bg.radius_99/bg.mass), "sigma_c": float(bg.sigma_c),
        "phi_tilde_c": float(bg.phi_tilde_c), "EKG_residual": float(bg.residual),
        "mass_rescaling": "r/M=r_mu/M_mu; x/M=x_mu/M_mu; t/M=t_mu/M_mu; V_M2=V_mu2*M_mu^2",
        "radial_stability": "NOT_COMPUTED", "branch_identity": "NODELESS_CONTINUATION_NOT_SPECTRAL_BRANCH_PROOF",
        "finite_tail": {
            "state": "UNQUANTIFIED", "is_error_bound": False,
            "scalar_boundary": "sigma(rmax)=0 imposed; not evidence of a physically vanishing tail",
            "metric_boundary": "a(rmax)*alpha(rmax)=1; subsequent Schwarzschild extension",
            "last_20_percent_mass_fraction": float((bg.mass-mass[tail_start])/bg.mass),
            "scalar_at_0p8_rmax": float(bg.sigma[tail_start]),
            "scalar_derivative_at_rmax": float(bg.dsigma[-1]),
            "required_control": "independent rmax variation, separated from background quadrature/sampling error",
        },
        "background_reconstruction": {
            "state": "UNQUANTIFIED", "samples": len(bg.r), "r_min_mu": float(bg.r[0]),
            "r_max_mu": float(bg.r[-1]), "native_origin": "x(r_min)=0; finite centre cutoff inherited",
            "radial_spacing_mu": float((bg.r[-1]-bg.r[0])/(len(bg.r)-1)),
            "tortoise_quadrature": "cumulative trapezoid on the declared uniform background samples",
            "axial_grid_interpolation": "PCHIP of V_M2(x/M) and r/M(x/M)",
            "radius_99_convention": "first background sample with m(r)>=0.99*M; not interpolated",
            "required_control": "background sampling/quadrature/centre-cutoff variation, not TD dx alone",
        },
    }
    return arrays, metadata, checks


def _solve(rt, sc, rmax, n, tol):
    bg = rt["solve_mini_boson_star"](sigma_c=sc, rmax=rmax, n=n, tol=tol)
    arrays, metadata, checks = _background_diagnostics(bg, tol)
    metadata.update(equilibrium_rmax=rmax, equilibrium_n=n, equilibrium_tol=tol,
                    background_samples=len(bg.r),
                    equilibrium_solve="FRESH_BVP_CONTINUATION_NO_SAVED_PROFILE",
                    literature=LITERATURE, baseline_provenance=rt.provenance)
    return bg, arrays, metadata, checks


def solve_equilibrium(params: Mapping[str, float], *, equilibrium_rmax=50.,
                      equilibrium_n=1200, equilibrium_tol=2e-7, background_samples=5000,
                      notebook_path=None):
    """Return a newly solved, serializable background (mu units and M units)."""
    settings = _settings(params, equilibrium_rmax, equilibrium_n, equilibrium_tol)
    background_samples = _sample_count(background_samples)
    rt = load_boson_runtime(notebook_path, background_samples=background_samples)
    _, arrays, metadata, checks = _solve(rt, *settings)
    return {"equilibrium": arrays, "metadata": metadata, "checks": checks}


def run_boson_star(params: Mapping[str, float], *, dx, t_end=520., cfl=.35,
                   source_global=55., observer_global=95., sigma=4., x_right=400.,
                   sample_every=2, equilibrium_rmax=50., equilibrium_n=1200,
                   equilibrium_tol=2e-7, background_samples=5000, notebook_path=None):
    """Controlled unit-Psi Gaussian on a freshly solved nodeless mini BS.

    ``source_global``/``observer_global`` are Schwarzschild reference r* values
    used ONLY to select common areal radii. In the BS they are mapped through the
    actual equilibrium tortoise map, not treated as global BS coordinates.
    The physical left endpoint is the regular centre, not a tunable boundary.
    """
    settings = _settings(params, equilibrium_rmax, equilibrium_n, equilibrium_tol)
    background_samples = _sample_count(background_samples)
    dx, t_end, sigma, x_right = [_finite(v, name, positive=True) for v, name in
        ((dx, "dx"), (t_end, "t_end"), (sigma, "sigma"), (x_right, "x_right"))]
    source_global = _finite(source_global, "source_global")
    observer_global = _finite(observer_global, "observer_global")
    cfl = _finite(cfl, "cfl", positive=True)
    if not cfl < .55:
        raise ValueError("cfl must be in (0, 0.55)")
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if t_end < 8*cfl*dx or sample_every > math.ceil(t_end/(cfl*dx)):
        raise ValueError("Insufficient evolution steps or fewer than two recorded samples")
    if math.ceil(x_right/dx)+1 > 2_000_000:
        raise ValueError("Spatial allocation exceeds the bounded adapter's 2,000,000-point limit")
    rt = load_boson_runtime(notebook_path, background_samples=background_samples)
    bg, equilibrium, metadata, checks = _solve(rt, *settings)
    radii = rt["r_from_schwarzschild_rstar"](np.array([source_global, observer_global]), 1.)
    native = rt["boson_native_tortoise_at_radius"](bg, radii)
    source_native, observer_native = map(float, native)
    cfg = rt["Config"](M=1., ell=2, dx=dx, cfl=cfl, t_max=t_end,
                       pulse_center=source_native, extraction_x=observer_native,
                       pulse_width=sigma, sample_every=sample_every)
    x, V, r = rt["boson_axial_grid"](bg, cfg=cfg, dx=dx, x_right_M=x_right)
    if len(x) < 12 or not all(np.isfinite(a).all() for a in (x, V, r)):
        raise ValueError("Invalid/underresolved BS axial grid")
    if not x[4] < observer_native < x[-5]:
        raise ValueError("Observer outside safe cubic-extraction interior")
    if not (x[4] < source_native-8*sigma and source_native+8*sigma < x[-5]):
        raise ValueError("Initial Gaussian 8sigma region must lie inside the BS numerical domain")
    # Root-solved inverse consistency of the same equilibrium map. This checks
    # coordinate bookkeeping, not the independent accuracy of its quadrature.
    rM, xM = equilibrium["r_over_M"], equilibrium["x_over_M"]
    recovered = []
    for xx in native:
        if xx <= xM[-1]:
            hi = min(len(xM)-1, int(np.searchsorted(xM, xx)))
            lo = max(0, hi-1)
            recovered.append(brentq(lambda rr: float(rt["boson_native_tortoise_at_radius"](bg, rr))-xx,
                                     float(rM[lo]), float(rM[hi]), xtol=1e-12))
        else:
            recovered.append(float(rt["r_from_schwarzschild_rstar"](
                rt["schwarzschild_rstar"](rM[-1])+xx-xM[-1])))
    radius_error = float(np.max(np.abs(np.asarray(recovered)-radii)))
    checks["geometry"] = {"passed": bool(radius_error < 1e-9),
                          "max_areal_radius_error_M": radius_error, "tolerance": 1e-9,
                          "scope": "inverse consistency of the same equilibrium map; not a quadrature error bound"}
    if not checks["geometry"]["passed"]:
        raise ArithmeticError("BS controlled areal-radius mapping failed")
    initial_psi, initial_pi, char = rt["initial_characteristics"](x, source_native, sigma)
    # This is exactly the regular-centre projection performed by evolve_wave.
    initial_psi[0] = initial_pi[0] = 0.
    initial_dx = rt["d1_4"](initial_psi, float(np.mean(np.diff(x))))
    initial_data = {"psi": initial_psi, "Pi": initial_pi,
                    "u_plus": initial_pi+initial_dx, "u_minus": initial_pi-initial_dx}
    t, records, numerical = rt["evolve_wave"](
        x, V, t_max=t_end, cfl=cfl, pulse_center=source_native, pulse_width=sigma,
        extraction_points={"observer": observer_native}, left_bc="regular_center",
        sample_every=sample_every, ell=2)
    psi = records["observer"]
    actual_dx = float(np.mean(np.diff(x)))
    earliest = float(2*x[-1]-source_native-observer_native-8*sigma)
    margin = earliest-float(t[-1])
    finite = bool(np.isfinite(t).all() and np.isfinite(psi).all() and np.all(np.diff(t) > 0))
    checks.update({
        "finite": {"passed": finite},
        "stability": {"passed": bool(numerical["stability_pass"]),
                      "max_amp_ratio": float(numerical["max_amp_ratio"]),
                      "scope": "TD evolution monitor, not background radial stability"},
        "boundary_causal": {"passed": bool(margin > 0), "margin_M": margin,
                            "earliest_return_M": earliest, "left_return_M": None,
                            "right_return_M": earliest, "gaussian_threshold": math.exp(-32),
                            "scope": "8sigma practical exterior-boundary screen; not a strict bound"},
    })
    roots, photon_margin, _ = rt["boson_photon_spheres"](bg)
    metadata.update({**numerical, "model_family": "BS", "params": {"sigma_c": settings[0]},
        "M_ADM": 1., "ell": 2, "polarisation": "axial", "field": "Q_RW", "units": "G=c=M_ADM=1",
        "time_convention": "exp(-i omega t)", "requested_dx": dx, "actual_dx": actual_dx,
        "source_global": source_global, "observer_global": observer_global,
        "global_coordinate_scope": "Schwarzschild reference r* specifying target areal radii, not a BS coordinate",
        "source_native": source_native, "observer_native": observer_native,
        "source_areal_radius_over_M": float(recovered[0]),
        "observer_areal_radius_over_M": float(recovered[1]),
        "source_coordinate_mapping": "actual equilibrium r/M -> x/M, including finite-radius vacuum continuation",
        "native_to_global_offset_right_exterior": None,
        "vacuum_extension_native_to_global_offset": float(rt["schwarzschild_rstar"](rM[-1])-xM[-1]),
        "vacuum_match_native_M": float(xM[-1]), "vacuum_match_radius_over_M": float(rM[-1]),
        "x_interface_global": None, "surface_status": "NO_FINITE_PHYSICAL_SURFACE",
        "sigma": sigma, "pulse_shape": "Psi=exp(-(x-x_source)^2/(2*sigma^2)); Pi=D1_4(Psi)",
        "initial_data_scope": "Gaussian is Psi, not u_plus; u_plus=Pi+D1_4(Psi)",
        "initial_regular_center_projection": "Psi[0]=Pi[0]=0 after characteristic construction",
        "initial_characteristic_F_in_before_center_projection": float(char["F_in"]),
        "initial_time": 0., "initial_amplitude": 1., "sample_every": sample_every,
        "record_end": float(t[-1]), "solver_id": SOLVER_ID,
        "background_samples": background_samples,
        "numerical_variant": "baseline_dense_profile" if background_samples == 5000 else "explicit_dense_profile_sampling",
        "solver_status": "OK" if finite and numerical["stability_pass"] else "STABILITY_FAILED",
        "numerics_evidence": "UNQUANTIFIED", "model_scope": "mini BS, decoupled linear axial metric sector; no polar or scalar perturbations",
        "boundary_causal_pass": checks["boundary_causal"]["passed"], "boundary_margin_M": margin,
        "photon_spheres_mu_r": roots.tolist(), "photon_sphere_margin": float(photon_margin),
        "photon_sphere_search_scope": "sampled equilibrium interior only; not a certified root-completeness result",
    })
    return {"t": t, "psi": psi, "x": x, "V": V, "r": r,
            "metadata": metadata, "checks": checks,
            "equilibrium": equilibrium, "initial_data": initial_data}
