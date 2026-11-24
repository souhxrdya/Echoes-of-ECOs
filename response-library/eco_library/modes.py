"""Complex axial modes of GS, WH and BS with verified continuation (plan modes/0.1).

Search region (predeclared, docs/METHOD.md): 0.05 <= Re(M omega) <= 1.0,
-0.6 <= Im(M omega) < 0, ell=2 axial.  Seeds are the frozen baseline's selected modes at
the fiducial parameters (baseline/eco_outputs_final/selected_complex_modes_all_models.csv),
used only as starting points: every root is solved anew from the exact outgoing
condition (complex-contour exterior shooting) and the model's interior matching, then
continued in the model parameter to the target point.  A step is accepted only if the
re-evaluated normalised residual is below the model tolerance and the root moved less
than one third of its distance to the nearest other tracked root; a rejected step is
retried at half size and logged.  Branch identity is within the ECO family along the
logged path; no ECO branch is mapped to a BH overtone.  Numerical evidence: the spread
of the root under numerical-setting variants (VERIFIED_BY_VARIATION with that spread as
sigma_num, or UNQUANTIFIED when the variants agree to roundoff); for BS the matching-radius
(scalar-tail) dependence is a separate systematic, never mixed into sigma_num.  Lists are
not exhaustive.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from .frozen_declarations import load_declarations

REGION = {"re_min": .05, "re_max": 1., "im_min": -.6, "im_max": 0.}
TOLERANCE = {"GS": 2e-8, "WH": 2e-8, "BS": 1e-7}
from .paths import BASELINE_OUTPUTS
SEEDS_CSV = BASELINE_OUTPUTS/"selected_complex_modes_all_models.csv"
CELLS = {22: ("_WH_POLE_TRACK_CACHE", "_WH_POLE_TRACK_LOG", "wh_outgoing_cauchy", "wh_pole_residual", "wh_complex_pole"),
         27: ("gs_regular_shell", "gs_shell_reflectivity")}
CONTOUR_VARIANTS = ({"angle": .80}, {"length": 120.}, {"order": 20}, {"rtol": 2e-12})
_RT = {}


def runtime():
    if "rt" not in _RT:
        _RT["rt"] = load_declarations(CELLS, {"solve_ivp": solve_ivp, "root": root}, label="modes")
    return _RT["rt"]


def boson_runtime():
    if "bs" not in _RT:
        from .boson_star import load_boson_runtime
        base = load_boson_runtime()
        _RT["bs"] = load_declarations({22: ("wh_outgoing_cauchy",), 24: ("BosonAxialModeProblem",)},
                                      {"solve_ivp": solve_ivp, "root": root, "CubicSpline": CubicSpline},
                                      label="bsmodes", base=base)
    return _RT["bs"]


def frozen_seeds(model):
    rows = []
    with SEEDS_CSV.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                rows.append({"family": row["family"], "index": int(row["selected_index"]),
                             "omega": complex(float(row["Momega_R"]), float(row["Momega_I"]))})
    return rows


def in_region(w):
    return REGION["re_min"] <= w.real <= REGION["re_max"] and REGION["im_min"] <= w.imag < REGION["im_max"]


# --------------------------------------------------------------------------- residuals

def gs_residual(w, compactness, **numeric):
    """Scale-invariant Wronskian of the regular de Sitter interior and the outgoing exterior at the shell."""
    rt = runtime()
    q, p = rt["gs_regular_shell"](np.asarray([w], complex), compactness)
    u, P = rt["wh_outgoing_cauchy"](complex(w), 1./compactness, 2, 1., **numeric)
    q, p = complex(q[0]), complex(p[0])
    return (q*P-p*u)/(abs(q)*abs(P)+abs(p)*abs(u)+1e-300)


def wh_residual(w, r0, parity, **numeric):
    return runtime()["wh_pole_residual"](complex(w), r0, parity, 2, 1., **numeric)


def solve(residual, guess, tol):
    def fun(v):
        try:
            r = residual(complex(v[0], v[1]))
        except ValueError:
            return [10+abs(v[0]), 10+abs(v[1])]
        return [r.real, r.imag]
    sol = root(fun, [complex(guess).real, complex(guess).imag], tol=tol*1e-2)
    w = complex(*sol.x)
    try:
        res = abs(residual(w))
    except ValueError:
        res = math.inf
    return w, res, bool(np.isfinite(res) and res <= tol and in_region(w))


# ------------------------------------------------------------------------ continuation

def continue_modes(residual_at, path, seeds, tol, max_halvings=4):
    """Track every seed along the parameter path together; returns per-mode logged steps."""
    tracks = {k: [] for k in seeds}
    current = dict(seeds)
    previous = path[0]
    queue = list(path)
    while queue:
        p = queue.pop(0)
        trial = {}
        for k, w in current.items():
            trial[k] = solve(lambda z: residual_at(p, z), w, tol)
        ok = all(t[2] for t in trial.values())
        moved = {k: abs(trial[k][0]-current[k]) for k in trial}
        nearest = {k: min((abs(trial[k][0]-trial[j][0]) for j in trial if j != k), default=math.inf) for k in trial}
        ok = ok and all(moved[k] < nearest[k]/3. for k in trial)
        if not ok and max_halvings > 0 and p != previous:
            mid = .5*(p+previous)
            for k in tracks:
                tracks[k].append({"parameter": p, "accepted": False, "reason": "retry at half step",
                                  "omega": [trial[k][0].real, trial[k][0].imag], "residual": trial[k][1]})
            queue = [mid, p]+queue
            max_halvings -= 1
            continue
        for k in trial:
            tracks[k].append({"parameter": p, "omega": [trial[k][0].real, trial[k][0].imag], "residual": trial[k][1],
                              "accepted": bool(trial[k][2] and moved[k] < nearest[k]/3.),
                              "step_displacement": moved[k],
                              "nearest_root_distance": nearest[k] if math.isfinite(nearest[k]) else None,
                              **({} if math.isfinite(nearest[k]) else
                                 {"crossing_check": "vacuous: no other root tracked on this path"})})
            current[k] = trial[k][0]
        if not ok:
            break
        previous = p
    return tracks, current


def variation(residual, w, tol, variants):
    """Re-solve one root under numerical-setting variants; spread is sigma_num (never zero)."""
    rows, roots = [], []
    for v in variants:
        ww, res, ok = solve(lambda z: residual(z, **v), w, tol)
        rows.append({"setting": v, "omega": [ww.real, ww.imag], "residual": res, "converged": ok})
        if ok:
            roots.append(ww)
    if len(roots) < len(variants):
        return {"evidence": "NOT_CONVERGED", "sigma_num": None, "variants": rows}
    spread = max(abs(r-w) for r in roots)
    if spread <= 1e-13*abs(w):
        return {"evidence": "UNQUANTIFIED", "sigma_num": None, "variants": rows,
                "reason": "variants agree to roundoff; no zero uncertainty is inferred"}
    return {"evidence": "VERIFIED_BY_VARIATION", "sigma_num": float(spread), "variants": rows}


def mode_record(model, family, index, w, method, scope, var, branch_id, track=None, **extra):
    return {"omega": [w.real, w.imag], "family": family, "selected_index": index, "method": method, "scope": scope,
            "sigma_num": var["sigma_num"], "evidence": var["evidence"], "branch_id": branch_id,
            "bh_counterpart": None, "bh_counterpart_reason": "no justified ECO-to-BH branch mapping",
            "exhaustive": False, "ell": 2, "polarisation": "axial", "variation": var["variants"],
            "continuation": track or [], **extra}


def gs_modes(target=.44, fiducial=.47, step=.005):
    seeds = {f"GS-axial-{s['index']}": s["omega"] for s in frozen_seeds("GS")}
    n = int(round(abs(fiducial-target)/step))
    path = list(np.linspace(fiducial, target, n+1))
    tracks, final = continue_modes(lambda c, z: gs_residual(z, c), path, seeds, TOLERANCE["GS"])
    out = []
    for i, (k, w) in enumerate(final.items()):
        accepted = all(s.get("accepted") for s in tracks[k] if "step_displacement" in s)
        var = variation(lambda z, **v: gs_residual(z, target, **v), w, TOLERANCE["GS"], CONTOUR_VARIANTS)
        out.append(mode_record("GS", "axial", i, w, "regular de Sitter interior / outgoing complex-contour exterior "
                               "Wronskian at the Sigma=0 shell", "zero-surface-energy thin-shell gravastar",
                               var, f"{k}@M/a={fiducial}->{target}", tracks[k], continuation_accepted=accepted))
    return out


def wh_modes(target=2.005, fiducial=2.0005, max_log_step=.2):
    seeds = {f"WH-{s['family']}-{s['index']}": s["omega"] for s in frozen_seeds("WH")}
    parity = {k: k.split("-")[1] for k in seeds}
    a, b = math.log(fiducial-2.), math.log(target-2.)
    n = max(1, int(math.ceil(abs(b-a)/max_log_step)))
    path = list(np.linspace(a, b, n+1))
    out, tracks_all, finals = [], {}, {}
    for k, w0 in seeds.items():  # parities never mix, so each family is tracked against its own parity
        tr, fin = continue_modes(lambda s, z, k=k: wh_residual(z, 2.+math.exp(s), parity[k]), path,
                                 {k: w0, **{j: v for j, v in seeds.items() if j != k and parity[j] == parity[k]}},
                                 TOLERANCE["WH"])
        tracks_all[k], finals[k] = tr[k], fin[k]
    for i, (k, w) in enumerate(finals.items()):
        accepted = all(s.get("accepted") for s in tracks_all[k] if "step_displacement" in s)
        var = variation(lambda z, **v: wh_residual(z, target, parity[k], **v), w, TOLERANCE["WH"], CONTOUR_VARIANTS)
        out.append(mode_record("WH", parity[k], i, w, "outgoing complex-contour exterior / exact delta-throat "
                               "matching; spatial-reflection parity", "thin-shell Schwarzschild wormhole, axial",
                               var, f"{k}@r0={fiducial}->{target} (log(r0/M-2) path)", tracks_all[k],
                               continuation_accepted=accepted,
                               continuation_parameter="log(r0/M-2)"))
    return out


def wh_literature_check():
    """Yang-Xia-Miao EPJC 85, 742 Table 6 (Prony estimates, 2M omega) versus fresh complex poles."""
    rows = []
    for abar, published in ((1.06, .782781-.039926j), (1.1, .825691-.072068j)):
        r0 = 2.*abar
        target = published/2.
        best = None
        for parity in ("even", "odd"):
            w, res, ok = solve(lambda z: wh_residual(z, r0, parity), target, TOLERANCE["WH"])
            if ok and (best is None or abs(w-target) < abs(best[0]-target)):
                best = (w, res, parity)
        rows.append({"a_bar": abar, "r0_over_M": r0, "published_2M_omega": [published.real, published.imag],
                     "computed_M_omega": None if best is None else [best[0].real, best[0].imag],
                     "parity": None if best is None else best[2],
                     "relative_difference": None if best is None else abs(best[0]-target)/abs(target),
                     "scope": "published values are Prony fits to time-domain data, not exact poles"})
    return rows


# --------------------------------------------------------------------------------- BS

BS_NUMERIC_VARIANTS = ({"length": 75.}, {"length": 130.}, {"angle": 1.30}, {"order": 22}, {"rtol": 5e-12},
                       {"center_radius": 1e-4})
BS_MATCH_FACTORS = (1.2, 1.4, 1.8)      # matching radius / R99: scalar-tail approximation, reported separately
BS_MATCH_SCAN = tuple(np.round(np.arange(1.2, 2.01, .1), 2))   # continuation in the matching radius
BS_APPROXIMATION_RULE = .01             # |omega(2.0 R99)-omega(1.8 R99)| <= 1% |omega| for a converged approximation
BS_TABLE_IV = (.277-.388j, .456-.374j)  # Macedo et al. Table IV, mini BS I, ell=2 axial, omega/mu (3 digits)


def bs_background(sigma_c, *, rmax=50., n=1200, tol=2e-7):
    from .boson_star import _solve, load_boson_runtime
    if "bs_base" not in _RT:
        _RT["bs_base"] = load_boson_runtime()
    bg, _, metadata, checks = _solve(_RT["bs_base"], float(sigma_c), rmax, n, tol)
    return bg, metadata, checks


def bs_residual(bg, w_mu, match_factor=1.4, **settings):
    rt = boson_runtime()
    problem = rt["BosonAxialModeProblem"](bg, 2, match_radius=match_factor*bg.radius_99, **settings)
    return problem.residual(complex(w_mu))


def _bs_in_region(bg):
    def ok(w_mu):
        return in_region(w_mu*float(bg.mass))
    return ok


def bs_solve(bg, guess_mu, **settings):
    def residual(z):
        return bs_residual(bg, z, **settings)
    def fun(v):
        try:
            r = residual(complex(v[0], v[1]))
        except ValueError:
            return [1+abs(v[0]), 1+abs(v[1])]
        return [r.real, r.imag]
    sol = root(fun, [complex(guess_mu).real, complex(guess_mu).imag], tol=2e-9)
    w = complex(*sol.x)
    try:
        res = abs(residual(w))
    except ValueError:
        res = math.inf
    return w, res, bool(np.isfinite(res) and res <= TOLERANCE["BS"] and _bs_in_region(bg)(w))


def bs_match_scan(bg, w):
    """Continue one root (omega/mu at 1.4 R99) in the matching radius over BS_MATCH_SCAN; M omega per radius."""
    scan = {}
    for direction in ([f for f in BS_MATCH_SCAN if f >= 1.4], sorted([f for f in BS_MATCH_SCAN if f < 1.4], reverse=True)):
        guess = w
        for factor in direction:
            guess, res, ok = bs_solve(bg, guess, match_factor=float(factor))
            scan[float(factor)] = {"omega": [guess.real*float(bg.mass), guess.imag*float(bg.mass)], "converged": ok,
                                   "residual": res if math.isfinite(res) else None}
    scan = dict(sorted(scan.items()))
    values = [complex(*v["omega"]) for v in scan.values()]
    last = [complex(*scan[f]["omega"]) for f in (1.8, 2.0)]
    change = abs(last[1]-last[0])/abs(last[1])
    state = ("APPROXIMATION_CONVERGED" if all(v["converged"] for v in scan.values()) and change <= BS_APPROXIMATION_RULE
             else "APPROXIMATION_NOT_CONVERGED")
    return {"match_radius_scan": {str(k): v for k, v in scan.items()}, "relative_change_1.8_to_2.0_R99": change,
            "scan_spread_M_omega": float(max(abs(a-b) for a in values for b in values)), "rule": BS_APPROXIMATION_RULE,
            "state": state, "scope": "Schwarzschild matching outside a finite radius (scalar-tail approximation); "
                                     "separate from the numerical sigma_num"}


def bs_table_iv_benchmark(sigma_c=None):
    from .boson_star import FIDUCIAL_SIGMA_C
    bg, metadata, _ = bs_background(FIDUCIAL_SIGMA_C if sigma_c is None else sigma_c)
    rows = []
    for published in BS_TABLE_IV:
        w, res, ok = bs_solve(bg, published)
        rows.append({"published_omega_over_mu": [published.real, published.imag],
                     "computed_omega_over_mu": [w.real, w.imag], "residual": res, "converged": ok,
                     "relative_difference": abs(w-published)/abs(published),
                     "criterion": "within 5%: three printed digits and the paper's few-percent matching-radius "
                                  "systematic (Sec. V.C); not a numerical error bound",
                     "passed": bool(ok and abs(w-published)/abs(published) <= .05),
                     "matching_radius_dependence": bs_match_scan(bg, w)})
    return {"phi_tilde_c": float(metadata.get("phi_tilde_c", bg.sigma_c/math.sqrt(2.))), "mass_mu": float(bg.mass),
            "rows": rows}


def bs_modes(target=.22, step=.01):
    """Table-IV-anchored BS modes continued in sigma_c from mini BS I to the target with fresh equilibria."""
    from .boson_star import FIDUCIAL_SIGMA_C
    n = int(math.ceil(abs(FIDUCIAL_SIGMA_C-target)/step))
    path = list(np.linspace(FIDUCIAL_SIGMA_C, target, n+1))
    bg0, _, _ = bs_background(path[0])
    seeds = {f"BS-axial-{s['index']}": s["omega"]/float(bg0.mass) for s in frozen_seeds("BS")}  # to omega/mu
    tracks = {k: [] for k in seeds}
    current = dict(seeds)
    for sc in path:
        bg, meta, checks = bs_background(sc)
        trial = {k: bs_solve(bg, w) for k, w in current.items()}
        for k, (w, res, ok) in trial.items():
            moved = abs(w-current[k])
            nearest = min((abs(w-trial[j][0]) for j in trial if j != k), default=math.inf)
            tracks[k].append({"parameter": float(sc), "omega_over_mu": [w.real, w.imag], "mass_mu": float(bg.mass),
                              "omega": [w.real*float(bg.mass), w.imag*float(bg.mass)], "residual": res,
                              "accepted": bool(ok and moved < nearest/3.), "step_displacement": moved,
                              "nearest_root_distance": nearest if math.isfinite(nearest) else None,
                              "background_checks_passed": all(c["passed"] for c in checks.values())})
            current[k] = w
    bg, meta, checks = bs_background(target)
    out = []
    for i, (k, w) in enumerate(current.items()):
        rows, roots = [], []
        for v in BS_NUMERIC_VARIANTS:
            ww, res, ok = bs_solve(bg, w, **v)
            rows.append({"setting": v, "omega_over_mu": [ww.real, ww.imag], "residual": res, "converged": ok})
            if ok:
                roots.append(ww)
        if len(roots) < len(BS_NUMERIC_VARIANTS):
            var = {"evidence": "NOT_CONVERGED", "sigma_num": None, "variants": rows}
        else:
            spread = max(abs(r-w) for r in roots)*float(bg.mass)
            var = ({"evidence": "VERIFIED_BY_VARIATION", "sigma_num": float(spread), "variants": rows}
                   if spread > 1e-13*abs(w)*float(bg.mass) else
                   {"evidence": "UNQUANTIFIED", "sigma_num": None, "variants": rows,
                    "reason": "variants agree to roundoff"})
        tail = [{"match_radius_over_R99": factor,
                 **boson_runtime()["BosonAxialModeProblem"](bg, 2, match_radius=factor*bg.radius_99).tail_metadata()}
                for factor in BS_MATCH_FACTORS]
        scan = bs_match_scan(bg, w)
        approximation_state = scan["state"]
        accepted = all(s["accepted"] for s in tracks[k])
        Mw = w*float(bg.mass)
        out.append(mode_record("BS", "axial", i, Mw, "regular-centre interior integration / outgoing complex-contour "
                               "exterior matched at 1.4 R99 (Macedo et al. Sec. V.A)",
                               "mini boson star, decoupled axial sector; Schwarzschild exterior beyond the match radius",
                               var, f"{k}@sigma_c={path[0]:.6f}->{target} (Table IV mini BS I anchor)", tracks[k],
                               continuation_accepted=accepted, omega_over_mu=[w.real, w.imag], mass_mu=float(bg.mass),
                               tail_systematic={"tail_at_match_radii": tail, **scan},
                               export_evidence=("NOT_CONVERGED" if approximation_state != "APPROXIMATION_CONVERGED"
                                                else var["evidence"])))
    return out


def compute_all(output):
    """Mode records for the off-fiducial points and their fiducials, with literature checks; writes JSON."""
    from .boson_star import FIDUCIAL_SIGMA_C
    import time
    started = time.perf_counter()
    result = {"version": "modes/0.1", "region": REGION, "tolerance": TOLERANCE,
              "seed_source": str(SEEDS_CSV.relative_to(SEEDS_CSV.parents[2])),
              "GS": {"M/a=0.44": gs_modes()},
              "WH": {"r0/M=2.005": wh_modes()},
              "BS": {"sigma_c=0.22": bs_modes()},
              "literature": {"WH_EPJC_85_742_Table_6": wh_literature_check(),
                             "BS_Macedo_Table_IV": bs_table_iv_benchmark()},
              "QH": {"state": "PRESCRIPTION_DEPENDENT_NOT_COMPUTED",
                     "reason": "see docs/METHOD.md Section 4: the OWA law is defined on the real axis "
                               "through |omega|; poles require choosing the analytic continuation of the omega>0 "
                               "branch, which the library states but does not adopt as a model output"},
              "bh_counterpart": None, "exhaustive": False}
    result["wall_seconds"] = time.perf_counter()-started
    Path(output).write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    return result


def compute_point(model, value, output):
    """Modes of one catalogue entry, continued from the frozen fiducial seeds (docs/METHOD.md)."""
    import time
    started = time.perf_counter()
    if model == "GS":
        rows = gs_modes(target=float(value))
    elif model == "WH":
        rows = wh_modes(target=float(value))
    elif model == "BS":
        rows = bs_modes(target=float(value))
    else:
        raise ValueError("modes are computed for GS, WH and BS; QH is PRESCRIPTION_DEPENDENT_NOT_COMPUTED")
    result = {"version": "modes/0.1", "model": model, "value": float(value), "region": REGION,
              "tolerance": TOLERANCE, "modes": rows, "wall_seconds": time.perf_counter()-started}
    Path(output).write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    return result


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "point":
        out = compute_point(sys.argv[2], sys.argv[3], sys.argv[4])
        for row in out["modes"]:
            print(sys.argv[2], sys.argv[3], row["branch_id"][:24], [round(x, 7) for x in row["omega"]], row["evidence"],
                  row.get("continuation_accepted"), (row.get("tail_systematic") or {}).get("state"))
        print("wall", round(out["wall_seconds"], 1))
        raise SystemExit(0)
    r = compute_all(sys.argv[1])
    for model in ("GS", "WH", "BS"):
        for point, rows in r[model].items():
            for row in rows:
                print(model, point, row["branch_id"][:24], [round(x, 8) for x in row["omega"]], row["evidence"],
                      row["sigma_num"], row.get("continuation_accepted"),
                      (row.get("tail_systematic") or {}).get("state"))
    print("wall", round(r["wall_seconds"], 1))
