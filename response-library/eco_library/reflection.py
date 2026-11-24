"""GS/WH reflection coefficients, version 0.2 (docs/METHOD.md, predeclared at 67b106c).

The outgoing Jost solution starts from the optimally truncated outgoing asymptotic series of the s = 2
Regge-Wheeler equation at band-wise radii instead of a plane wave at r* = 400, and is carried to the seam
by the frozen fourth-order seam integrator.  In the seam basis U = A e^{i w X} + B e^{-i w X},
R_barrier = B/A and T_inf = 1/A.  GS R_interface is the frozen closed form; WH R_interface is the
throat-plus-opposite-exterior coefficient of the frozen ``wh_seam_transfer`` with the version-0.2
barrier coefficient.  Variants (step 0.00625; start radii x1.5) and an independent adaptive solve in
areal r decide the evidence label by the unchanged 1e-6 rule; group delays carry grid and variant
uncertainties and are RESOLVED only where the declared rule holds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "eco-reflection/0.2"
PLAN = "docs/METHOD.md"
GRID = .01 + .0025*np.arange(597)
BANDS = ((0., .05, 1300.), (.05, .15, 600.), (.15, np.inf, 400.))
VARIANT_RADII = {(0., .05): 1000., (.05, .15): 900., (.15, np.inf): 600.}
FROZEN_RSTAR_LIMIT = 1400.   # frozen r_from_schwarzschild_rstar saturates beyond r* = 1402 M (exp(700) clip)
STEP = .0125
VARIANTS = {"step_half": {"step": .00625}, "start_radii_changed": {"radii": VARIANT_RADII}}
AGREEMENT = 1e-6
SERIES_REL = 1e-17
INDEPENDENT_RTOL, INDEPENDENT_ATOL = 1e-12, 1e-14
INDEPENDENT_FREQUENCIES = 25
GD_REL, GD_FLOOR_M, R_MIN = 1e-6, 1., 1e-3
DELTA = 5e-6                 # amendment 2: local derivative step for the group delay
PACKET_BAND = (.15, .9)


def outgoing_series(z, r, M=1., ell=2, nmax=2000, rel=SERIES_REL):
    """S(r) and dS/dr of psi = e^{i z r*} S(r), with S = sum_k t_k, t_k = a_k r^-k from the s = 2 recurrence
    a_k = {[(k-1)k - l(l+1)] a_{k-1} + M[6 - 2(k-2)k] a_{k-2}}/(2 i z k).  Returns (S, dS, error estimate,
    truncation index, not_converged mask); the error of truncating after t_k is max(|t_{k+1}|, |t_{k+2}|)."""
    z = np.atleast_1d(np.asarray(z, complex))
    L = ell*(ell+1)
    t = [np.ones_like(z)]
    S_parts, dS_parts = [np.ones_like(z)], [np.zeros_like(z)]
    best_S, best_dS = np.ones_like(z), np.zeros_like(z)
    best_err, best_k = np.full(z.shape, np.inf), np.zeros(z.shape, int)
    active = np.ones(z.shape, bool)
    for m in range(1, nmax+1):
        t2 = t[-2] if m >= 2 else np.zeros_like(z)
        tm = (((m-1)*m - L)*t[-1]/r + (M*(6 - 2*(m-2)*m)*t2/r**2 if m >= 2 else 0))/(2j*z*m)
        tm = np.where(active, tm, 0)
        t.append(tm)
        S_parts.append(S_parts[-1] + tm)
        dS_parts.append(dS_parts[-1] - m*tm/r)
        if m >= 2:
            k = m-2
            err = np.maximum(np.abs(t[k+1]), np.abs(t[k+2]))
            better = active & (err < best_err)
            best_S = np.where(better, S_parts[k], best_S)
            best_dS = np.where(better, dS_parts[k], best_dS)
            best_err = np.where(better, err, best_err)
            best_k = np.where(better, k, best_k)
            active &= ~((err < rel*np.abs(S_parts[k])) | (err > 1e6*best_err))
            S_parts[k] = dS_parts[k] = None
        if len(t) > 3:
            t[-4] = None
        if not active.any():
            break
    return best_S, best_dS, best_err, best_k, active


def _rt():
    from . import transfer
    return transfer.runtime()


def barrier(z, seam, *, step=STEP, radii=None, band_z=None, M=1., ell=2):
    """(R_barrier, T_inf, info) at the seam (global r*) for real frequencies z; ``radii`` maps a band
    (lo, hi) to its start X_out (default BANDS); ``band_z`` assigns bands (default z), so that the shifted
    frequencies of a local derivative use the start of their central frequency."""
    rt = _rt()
    z = np.asarray(z, float)
    bz = z if band_z is None else np.asarray(band_z, float)
    R, T = np.empty(z.size, complex), np.empty(z.size, complex)
    err, terms, radii_used = np.empty(z.size), np.empty(z.size, int), {}
    for lo, hi, X in BANDS:
        m = (bz >= lo) & (bz < hi)
        if not m.any():
            continue
        X_out = (radii or {}).get((lo, hi), X)
        if seam + X_out > FROZEN_RSTAR_LIMIT:
            raise ValueError("start beyond the validity of the frozen r(r*) inverse (r* <= 1400 M)")
        r_out = float(rt["r_from_schwarzschild_rstar"](seam+X_out, M))
        S, dS, e, k, unconverged = outgoing_series(z[m], r_out, M, ell)
        if unconverged.any():
            raise RuntimeError("outgoing series did not reach its optimal truncation")
        zz = z[m]+0j
        phase = np.exp(1j*zz*X_out)
        u, du = phase*S, phase*(1j*zz*S + (1-2*M/r_out)*dS)
        u0, du0 = rt["_seam_advance"](zz, u, du, X_out, 0., seam, ell, M, step)
        A, B = .5*(u0 + du0/(1j*zz)), .5*(u0 - du0/(1j*zz))
        R[m], T[m], err[m], terms[m] = B/A, 1/A, e, k
        radii_used[f"{lo:g}-{hi:g}"] = {"X_out": X_out, "r_out": r_out, "series_error_max": float(e.max())}
    return R, T, {"series_error_max": float(err.max()), "series_terms": [int(terms.min()), int(terms.max())],
                  "start": radii_used, "step": step}


def throat_coefficients(z, r0_over_M, M=1.):
    """Delta-throat r, t of the frozen wh_seam_transfer: Lambda = 2(r0 - M)/r0^2 (chain() in transfer.py)."""
    r0 = r0_over_M*M
    lam = 2*(r0-M)/r0**2
    z = np.asarray(z, complex)
    return lam/(2j*z-lam), 2j*z/(2j*z-lam), lam


def interface(model, params, z, R_far):
    """GS: closed form (R_far unused); WH: throat with the opposite exterior."""
    if model == "GS":
        return _rt()["gs_shell_reflectivity"](np.asarray(z, float), float(params["compactness"]))
    rd, td, _ = throat_coefficients(z, float(params["r0_over_M"]))
    return rd + td**2*R_far/(1-rd*R_far)


def independent_barrier(z, seam, M=1.):
    """Adaptive DOP853 in areal r from a far radius with a two-term outgoing start, per frequency."""
    from scipy.integrate import solve_ivp
    rt = _rt()
    r_seam = float(rt["r_from_schwarzschild_rstar"](seam, M))
    out = []
    for w in np.atleast_1d(z):
        r_far = 20000. if w < .05 else 5000. if w < .15 else 2000.
        a1 = -6/(2j*w)
        a2 = (-4*a1 + 6*M)/(4j*w)
        S, dS = 1 + a1/r_far + a2/r_far**2, -a1/r_far**2 - 2*a2/r_far**3
        rs = r_far + 2*M*np.log(r_far/(2*M) - 1)

        def rhs(r, y, w=w):
            f, fp = 1-2*M/r, 2*M/r**2
            V = f*(6/r**2 - 6*M/r**3)
            return [y[1], -(fp/f)*y[1] - (w*w - V)/(f*f)*y[0]]
        y0 = np.exp(1j*w*(rs-seam))*np.array([S, 1j*w/(1-2*M/r_far)*S + dS])
        sol = solve_ivp(rhs, (r_far, r_seam), y0, method="DOP853", rtol=INDEPENDENT_RTOL, atol=INDEPENDENT_ATOL)
        if not sol.success:
            raise RuntimeError(sol.message)
        u, dur = sol.y[:, -1]
        du = (1-2*M/r_seam)*dur
        out.append((.5*(u - du/(1j*w)))/(.5*(u + du/(1j*w))))
    return np.asarray(out)


def evidence(production, variants, independent_diff):
    differences = {n: float(np.max(np.abs(v-production))) for n, v in variants.items()}
    if len(variants) < 2 or not all(np.all(np.isfinite(v)) for v in variants.values()):
        return "UNQUANTIFIED", differences
    if all(d == 0. for d in differences.values()):
        return "UNQUANTIFIED", differences
    ok = max(differences.values()) <= AGREEMENT and (independent_diff is None or independent_diff <= AGREEMENT)
    return ("VERIFIED_BY_VARIATION" if ok else "NOT_CONVERGED"), differences


def group_delay(omega, R):
    """Fourth-order centred derivative of the unwrapped phase; NaN at the two end points per side."""
    phase = np.unwrap(np.angle(R))
    h = float(omega[1]-omega[0])
    tau = np.full(omega.shape, np.nan)
    tau[2:-2] = (-phase[4:] + 8*phase[3:-1] - 8*phase[1:-3] + phase[:-4])/(12*h)
    return tau, float(np.max(np.abs(np.diff(phase))))


def delay_states(R, tau, sigma):
    state = np.full(R.shape, "RESOLVED", dtype=object)
    state[np.abs(R) < R_MIN] = "PHASE_ILL_CONDITIONED"
    bad = (np.abs(R) >= R_MIN) & ~(sigma <= GD_REL*np.maximum(np.abs(tau), GD_FLOOR_M))
    state[bad] = "UNRESOLVED"
    state[~np.isfinite(tau)] = "UNDEFINED"
    return state


def _settings_json(cfg):
    out = dict(cfg)
    if "radii" in out:
        out["radii"] = {f"{lo:g}-{hi:g}": x for (lo, hi), x in out["radii"].items()}
    return out


def _local_tau(R0, Rp, Rm, d):
    return np.imag((Rp-Rm)/(2*d*R0))


def compute(model, params, seam):
    """All version-0.2 products for one GS or WH point (plan sections 2-3, amendments 1-2)."""
    from .interface_export import GS_INTERFACE_VARIANTS, gs_interface_by_radial_ode
    n = GRID.size
    shifts = (0., DELTA, -DELTA, 2*DELTA, -2*DELTA)
    zz = np.concatenate([GRID+s for s in shifts])
    bz = np.tile(GRID, len(shifts))
    split = lambda a: [a[i*n:(i+1)*n] for i in range(len(a)//n)]
    Rb_all, T_all, info = barrier(zz, seam, band_z=bz)
    Rb = split(Rb_all)
    T = T_all[:n]
    var_b = {}
    for name, cfg in VARIANTS.items():
        zv, bv = np.concatenate([GRID+s for s in shifts[:3]]), np.tile(GRID, 3)
        var_b[name] = split(barrier(zv, seam, band_z=bv, **cfg)[0])
    sample = np.unique(np.linspace(0, n-1, INDEPENDENT_FREQUENCIES).round().astype(int))
    indep_b = independent_barrier(GRID[sample], seam)
    fam = {"R_barrier": (Rb, {k: v for k, v in var_b.items()}, indep_b,
                         "outgoing asymptotic series start (plan 0.2) and frozen RK4 seam integrator")}
    meta = {"version": VERSION, "plan": PLAN, "model": model, "params": dict(params), "plane_rstar": seam,
            "sector": "single", "grid": "0.01 + 0.0025 k, k = 0..596", "delta": DELTA}
    if model == "GS":
        c = float(params["compactness"])
        Ri = [interface(model, params, GRID+s, None) for s in shifts]
        var_i = {name: [gs_interface_by_radial_ode(GRID+s, c, **cfg) for s in shifts[:3]]
                 for name, cfg in GS_INTERFACE_VARIANTS.items()}
        fam["R_interface"] = (Ri, var_i, None, "frozen closed form (gs_shell_reflectivity); variants: independent "
                                               "radial ODE at two settings")
    else:
        Ri = [interface(model, params, GRID+s, rb) for s, rb in zip(shifts, Rb)]
        var_i = {name: [interface(model, params, GRID+s, rb) for s, rb in zip(shifts[:3], v)] for name, v in var_b.items()}
        indep_i = interface(model, params, GRID[sample], indep_b)
        fam["R_interface"] = (Ri, var_i, indep_i, "frozen wh_seam_transfer formula r + t^2 R_far/(1 - r R_far), "
                                                  "R_far = version-0.2 R_barrier")
        rd, td, lam = throat_coefficients(GRID, params["r0_over_M"])
        meta["throat_Lambda"] = lam
        meta["min_abs_one_minus_r_Rfar"] = float(np.min(np.abs(1-rd*Rb[0])))
    out = {"omega": GRID}
    meta["arrays"] = {}
    band = (GRID >= PACKET_BAND[0]) & (GRID <= PACKET_BAND[1])
    names = ["RESOLVED", "UNRESOLVED", "PHASE_ILL_CONDITIONED", "UNDEFINED"]
    for key, (sets, variants, indep, method) in fam.items():
        R = sets[0]
        tau = _local_tau(R, sets[1], sets[2], DELTA)
        tau2 = _local_tau(R, sets[3], sets[4], 2*DELTA)
        sigma_trunc = np.abs(tau-tau2)
        tau_v = {name: _local_tau(v[0], v[1], v[2], DELTA) for name, v in variants.items()}
        sigma_var = np.max([np.abs(t-tau) for t in tau_v.values()], axis=0)
        sigma = sigma_trunc + sigma_var
        states = delay_states(R, tau, sigma)
        tau_grid, max_step = group_delay(GRID, R)
        cross = np.abs(tau_grid-tau)[2:-2]/np.maximum(np.abs(tau[2:-2]), GD_FLOOR_M)
        indep_diff = None if indep is None else float(np.max(np.abs(indep-R[sample])))
        state, diffs = evidence(R, {name: v[0] for name, v in variants.items()}, indep_diff)
        out[key] = {"R": R, "variants": {name: v[0] for name, v in variants.items()}, "tau": tau,
                    "sigma_tau": sigma, "sigma_tau_truncation": sigma_trunc, "sigma_tau_variants": sigma_var,
                    "tau_state_code": np.array([names.index(s) for s in states], dtype=np.int8)}
        counts = {s: int(np.sum(states == s)) for s in names}
        meta["arrays"][key] = {
            "method": method, "evidence": state, "variant_max_abs_difference": diffs, "variant_order": list(variants),
            "variant_settings": {name: _settings_json(VARIANTS.get(name) or GS_INTERFACE_VARIANTS.get(name))
                                 for name in variants},
            "independent_frequencies": GRID[sample].tolist(), "independent_max_abs_difference": indep_diff,
            "min_abs_R": float(np.min(np.abs(R))), "max_phase_step_rad": max_step,
            "tau_states": counts, "tau_state_codes": names,
            "tau_states_packet_band": {s: int(np.sum((states == s) & band)) for s in names},
            "max_sigma_tau_over_scale": float(np.max(sigma/np.maximum(np.abs(tau), GD_FLOOR_M))),
            "grid_stencil_cross_check_max_relative": float(np.max(cross)),
            "grid_stencil_cross_check_max_relative_resolved": float(np.max(cross[states[2:-2] == "RESOLVED"]))
                if np.any(states[2:-2] == "RESOLVED") else None,
            "group_delay_claimable": state == "VERIFIED_BY_VARIATION"}
    meta["barrier_solve"] = info
    meta["flux_identity_max_abs_by_band"] = {
        f"{lo:g}-{hi:g}": float(np.max(np.abs(np.abs(Rb[0][m])**2 + np.abs(T[m])**2 - 1)))
        for lo, hi, _ in BANDS for m in [(GRID >= lo) & (GRID < hi)] if m.any()}
    meta["rules"] = {"agreement": AGREEMENT, "group_delay_relative": GD_REL, "group_delay_floor_M": GD_FLOOR_M,
                     "R_min": R_MIN, "packet_band": PACKET_BAND}
    return out, meta


def write(path, arrays, meta):
    from .adapter_archive import _write_tree
    from .schema import canonical_json
    import h5py
    path = Path(path)
    if path.exists():
        raise ValueError(f"exists, not overwritten: {path}")
    text = canonical_json(meta)
    with h5py.File(path, "x") as handle:
        manifest = {}
        _write_tree(handle, "arrays", arrays, manifest)
        encoded = canonical_json(manifest)
        handle.attrs.update(schema_version=VERSION, header=text, header_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            manifest=encoded, manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest())
    return path


def read(path):
    from .adapter_archive import _read_tree
    import h5py
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != VERSION:
            raise ValueError("not a reflection 0.2 archive")
        for key in ("header", "manifest"):
            if hashlib.sha256(handle.attrs[key].encode()).hexdigest() != handle.attrs[key+"_sha256"]:
                raise ValueError(f"{key} checksum mismatch")
        manifest = json.loads(handle.attrs["manifest"])
        arrays = _read_tree(handle["arrays"], manifest)
        if manifest:
            raise ValueError("manifest lists missing nodes")
        return json.loads(handle.attrs["header"]), arrays


def summarize(output_dir):
    """The summary the catalogue assembly reads: per entry the archive checksum, the plane and, per
    array, the evidence fields of its header together with the frequencies whose delay state is
    UNRESOLVED and the frequency range whose phase is ill conditioned (both from the stored arrays)."""
    import hashlib
    output_dir = Path(output_dir)
    points, version = {}, None
    for path in sorted(output_dir.glob("*.h5")):
        header, arrays = read(path)
        version = version or header["version"]
        if header["version"] != version:
            raise ValueError("reflection archives of different versions")
        codes = None
        rows = {}
        for key, meta in header["arrays"].items():
            codes = meta["tau_state_codes"]
            state = np.asarray(arrays[key]["tau_state_code"])
            omega = np.asarray(arrays["omega"])
            unresolved = omega[state == codes.index("UNRESOLVED")]
            ill = omega[state == codes.index("PHASE_ILL_CONDITIONED")]
            rows[key] = {**{k: meta[k] for k in (
                "evidence", "variant_max_abs_difference", "independent_max_abs_difference", "min_abs_R",
                "tau_states", "tau_states_packet_band", "max_sigma_tau_over_scale", "group_delay_claimable")},
                "unresolved_omega": [round(float(w), 6) for w in unresolved],
                "phase_ill_conditioned_omega_range": [float(ill.min()), float(ill.max())] if ill.size else None}
        points[header["entry_id"]] = {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "plane_rstar": header["plane_rstar"], "arrays": rows}
    return {"version": version, "plan": "docs/METHOD.md (reflection coefficients)",
            "archives": "runs/reflection/<entry>.h5 (+ .json header copy)", "points": points}


def main(argv=None):
    import argparse
    import time
    from .catalogue_base import BLOCK_ROOTS
    from .transfer import seam_of
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=Path("runs/summaries/base-catalogue.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/reflection"))
    parser.add_argument("--entries", nargs="*")
    args = parser.parse_args(argv)
    catalogue = json.loads(args.catalogue.read_text(encoding="utf8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in catalogue["entries"]:
        if entry["model"] not in ("GS", "WH") or (args.entries and entry["entry_id"] not in args.entries):
            continue
        t0 = time.time()
        seam = seam_of(entry["model"], entry["params"])
        arrays, meta = compute(entry["model"], entry["params"], seam)
        meta.update(entry_id=entry["entry_id"], wall_seconds=time.time()-t0)
        write(args.output/f"{entry['entry_id']}.h5", arrays, meta)
        (args.output/f"{entry['entry_id']}.json").write_text(json.dumps(meta, indent=1, sort_keys=True)+"\n",
                                                            encoding="utf8")
        print(entry["entry_id"], {k: (v["evidence"], "%.1e" % max(v["variant_max_abs_difference"].values()), "%.1e" % v["max_sigma_tau_over_scale"],
                                      v["independent_max_abs_difference"], v["tau_states_packet_band"])
                                  for k, v in meta["arrays"].items()}, "%.0fs" % meta["wall_seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
