"""Production GS and WH frequency-domain transfer outputs (plan transfer/0.1, docs/METHOD.md).

Conventions, stated in every output:

* time dependence exp(-i omega t); the physical spectrum of a real record is
  X(z) = int x(t) exp(+i z t) dt, i.e. conj(rfft(x e^{-eps t})) dt at z = omega + i eps;
* K(z) multiplies the BH-reference *ingoing characteristic recorded at the near-horizon
  monitor plane* x_mon (global Schwarzschild r*, where |V| < 1e-11) and returns the
  ECO-minus-BH correction at the detector plane x_det, including every propagation
  factor between them (monitor to seam, seam to detector).  Moving the monitor by d
  multiplies K by exp(i z d); the reconstruction is irfft(conj(K) * I_numpy);
* seam X=0 at the GS shell r*(a) or the WH throat r*(r0); local characteristics
  U = A e^{izX} + B e^{-izX} define R_interface (GS: regular de Sitter interior with the
  Sigma=0 junction; WH: delta throat plus the opposite exterior), R_barrier = B/A (the
  exterior barrier seen from the seam, round-trip phase included), T_inf = 1/A (plane-wave
  coefficient at the outer truncation) and T_det = U(X_det)/A (to the detector);
  loop = R_barrier R_interface, K = T_det (R_interface I_seam - O_seam)/(1 - loop);
* group delay tau(omega) = + d arg K / d omega on the real axis.

Seam scattering, the GS shell reflectivity and the WH throat algebra are the frozen
notebook's reviewed declarations (cells 26-27), loaded by hash.  The inverse transform is
evaluated on z = omega + i eps (eps = ln(1e8)/T_fft) and multiplied by e^{eps t}, so
long-lived GS trapped modes do not wrap around the FFT period.  No output is fitted,
aligned or normalised.  Transfer outputs carry their variants; a variant spread is a
sensitivity, not an error bound.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid

from .adapter_archive import _read_tree, _write_tree
from .convergence import WINDOWS, common_series
from .frozen_declarations import load_declarations
from .schema import canonical_json

VERSION = "eco-transfer-archive/0.1"
CELLS = {26: ("ScatteringReference", "SeamScattering", "_seam_advance", "make_seam_scattering",
              "_scattering_return", "gs_seam_transfer", "wh_seam_transfer"),
         27: ("gs_regular_shell", "gs_shell_reflectivity")}
MONITORS = (-65., -80.)   # both recorded in one BH run; -80 serves the monitor-plane variant
DEFAULTS = {"monitor": -65., "step": .025, "outer_X": 400., "omega_max": 2.5, "wrap_suppression": 1e-8,
            "fft_factor": 4, "monitor_x_left": -400., "monitor_x_right": 400.}
VARIANTS = {"jost_step_half": {"step": .0125}, "outer_X_600": {"outer_X": 600.},
            "fft_resolution_x2": {"fft_factor": 8}, "laplace_shift_x2": {"wrap_suppression": 1e-16},
            "omega_max_3p5": {"omega_max": 3.5}, "monitor_minus_80": {"monitor": -80.}}
_RT = {}


def runtime():
    if "rt" not in _RT:
        _RT["rt"] = load_declarations(CELLS, label="transfer")
    return _RT["rt"]


def seam_of(model, params):
    rt = runtime()
    if model == "GS":
        c = float(params["compactness"])
        if not 0 < c < .5:
            raise ValueError("GS compactness must lie in (0, 1/2)")
        return float(rt["schwarzschild_rstar"](1./c))
    if model == "WH":
        r0 = float(params["r0_over_M"])
        if not r0 > 2:
            raise ValueError("WH throat must exceed 2M")
        return float(rt["schwarzschild_rstar"](r0))
    raise ValueError("transfer outputs are defined for the sharp-interface GS and WH models")


def chain(model, params, z, observer, *, monitor=DEFAULTS["monitor"], step=DEFAULTS["step"],
          outer_X=DEFAULTS["outer_X"]):
    """Seam scattering and the model's return chain at complex frequencies z (Re z>=0, Im z>=0)."""
    rt = runtime()
    seam = seam_of(model, params)
    sc = rt["make_seam_scattering"](np.asarray(z, complex), seam, float(observer)-seam, float(monitor),
                                    step=step, outer_X=outer_X)
    if model == "GS":
        R_i = rt["gs_shell_reflectivity"](sc.omega, float(params["compactness"]))
        out = rt["gs_seam_transfer"](sc, R_i)
        out["R_interface"] = R_i
    else:
        r0 = float(params["r0_over_M"])
        lam = 2*(r0-1.)/r0**2
        out = rt["wh_seam_transfer"](sc, sc, lam)
        out["R_interface"] = out["R_eff"]
        out["throat_Lambda"] = lam
    out.update(R_barrier=sc.Rbar, T_inf=sc.Tinf, T_det=sc.Tdet, incoming_BH_at_seam=sc.incoming_BH,
               outgoing_BH_at_seam=sc.outgoing_BH, z=sc.omega)
    planes = {"seam_rstar": seam, "monitor_rstar": float(monitor), "detector_rstar": float(observer),
              "outer_truncation_rstar": seam+float(outer_X), "jost_step": float(step)}
    return out, planes


def monitor_run(experiment, *, dx, t_end, cfl=.35, monitors=MONITORS,
                x_left=DEFAULTS["monitor_x_left"], x_right=DEFAULTS["monitor_x_right"], sample_every=2):
    """BH evolution recording Psi and Psi_x at each monitor and Psi at the observer (frozen evolve_wave)."""
    rt = runtime()
    e = experiment
    cfg = rt["Config"](M=1., ell=2, dx=dx, cfl=cfl, t_max=t_end, pulse_center=e.source_global_rstar,
                       extraction_x=e.observer_global_rstar, pulse_width=e.sigma, sample_every=sample_every)
    x, V, r, meta = rt["black_hole_grid"](cfg=cfg, dx=dx, x_left=x_left, x_right=x_right)
    names = {f"monitor{abs(m):g}": float(m) for m in monitors}
    for m in names.values():
        if not (x[4] < m and m < e.source_global_rstar-8*e.sigma):
            raise ValueError("monitor must lie inside the grid and outside the initial pulse")
        if abs(float(rt["rw_potential"](rt["r_from_schwarzschild_rstar"](m), 2, 1.))) > 1e-11:
            raise ValueError("monitor is not in the free near-horizon region")
    t, rec, num = rt["evolve_wave"](x, V, t_max=t_end, cfl=cfl, pulse_center=e.source_global_rstar,
                                    pulse_width=e.sigma, extraction_points={"observer": e.observer_global_rstar,
                                                                           **names},
                                    left_bc="outgoing", sample_every=sample_every, derivative_points=tuple(names))
    return {"t": t, "observer": rec["observer"],
            "monitors": {m: {"field": rec[k], "derivative": rec[k+"_dx"]} for k, m in names.items()},
            "x": x, "metadata": {"dx": float(np.mean(np.diff(x))), "cfl": cfl, "x_left": float(x[0]),
                                 "x_right": float(x[-1]), "monitors": list(names.values()), "steps": int(num["steps"]),
                                 "stability_pass": bool(num["stability_pass"]),
                                 "monitor_left_return_M": {k: float(e.source_global_rstar+m-2*x[0]-8*e.sigma)
                                                           for k, m in names.items()}}}


def reconstruct(model, params, source, *, observer, monitor=DEFAULTS["monitor"], step=DEFAULTS["step"],
                outer_X=DEFAULTS["outer_X"], omega_max=DEFAULTS["omega_max"],
                wrap_suppression=DEFAULTS["wrap_suppression"], fft_factor=DEFAULTS["fft_factor"]):
    """Correction at the detector from the measured monitor characteristic (damped Laplace inversion)."""
    t = np.asarray(source["t"], float)
    dt = float(np.mean(np.diff(t)))
    if abs(t[0]) > 1e-12 or not np.allclose(np.diff(t), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("uniform t=0 monitor record required")
    nfft = 1 << (fft_factor*len(t)-1).bit_length()
    T = nfft*dt
    eps = math.log(1./wrap_suppression)/T
    damp = np.exp(-eps*t)
    omega = 2*np.pi*np.fft.rfftfreq(nfft, dt)
    z = omega+1j*eps
    record = source["monitors"][float(monitor)]   # the field measured at the same plane as K
    q = np.fft.rfft(np.asarray(record["field"], float)*damp, nfft)
    qx = np.fft.rfft(np.asarray(record["derivative"], float)*damp, nfft)
    ingoing_np = .5*(q+qx/(1j*np.conj(z)))       # numpy convention of the physical I(z)
    keep = omega <= omega_max
    out, planes = chain(model, params, z[keep], observer, monitor=monitor, step=step, outer_X=outer_X)
    K = np.zeros(len(omega), complex)
    K[keep] = out["K"]
    correction = np.fft.irfft(np.conj(K)*ingoing_np, nfft)[:len(t)]*np.exp(eps*t)
    return {"t": t, "correction": correction, "omega": omega, "eps": eps, "nfft": int(nfft), "T_fft": T,
            "K_complex_contour": K, "planes": planes, "omega_max": float(omega_max)}


def real_axis_products(model, params, observer, omega, **settings):
    """Real-frequency products with identities, conditioning and group delays."""
    out, planes = chain(model, params, np.asarray(omega, float)+0j, observer, **settings)
    w = np.asarray(omega, float)
    flux = np.abs(np.abs(out["R_barrier"])**2+np.abs(out["T_inf"])**2-1)
    # RK4 shooting is not norm-conserving: its amplitude error grows like (omega*step)^6 per
    # step, so the identity is reported per band (the Jost-step variant tests the same error).
    bands = {"omega_lt_0.05": w < .05, "omega_0.05_to_1.5": (w >= .05) & (w <= 1.5), "omega_gt_1.5": w > 1.5}
    identities = {f"barrier_flux_max_abs_{k}": (float(np.max(flux[m])) if np.any(m) else None)
                  for k, m in bands.items()}
    if model == "GS":
        identities["shell_unitarity_max_abs"] = float(np.max(np.abs(np.abs(out["R_interface"])-1)))
    else:
        rd, td = out["r_delta"], out["t_delta"]
        identities["throat_flux_max_abs"] = float(np.max(np.abs(np.abs(rd)**2+np.abs(td)**2-1)))
        identities["odd_parity_eigenvalue_max_abs"] = float(np.max(np.abs(rd-td+1)))
    denominator = np.abs(1-out["loop"])
    phase = np.unwrap(np.angle(out["K"]))
    products = {"omega": w, "K": out["K"], "R_interface": out["R_interface"], "R_barrier": out["R_barrier"],
                "T_inf": out["T_inf"], "T_det": out["T_det"], "loop": out["loop"],
                "group_delay_K": np.gradient(phase, w),
                "group_delay_R_barrier": np.gradient(np.unwrap(np.angle(out["R_barrier"])), w),
                "group_delay_R_interface": np.gradient(np.unwrap(np.angle(out["R_interface"])), w)}
    if model == "WH":
        products.update(K_even=out["K_even"], K_odd=out["K_odd"], r_delta=out["r_delta"], t_delta=out["t_delta"])
    conditioning = {"min_abs_one_minus_loop": float(np.min(denominator)),
                    "omega_at_min": float(w[int(np.argmin(denominator))]),
                    "max_abs_K": float(np.max(np.abs(out["K"]))),
                    "ill_conditioned_below_1e-6": int(np.count_nonzero(denominator < 1e-6))}
    return products, planes, identities, conditioning


def _window_rows(t, a, b, windows=WINDOWS):
    rows = {}
    for w, (lo, hi) in windows.items():
        m = (t >= lo) & (t <= hi)
        if t[0] > lo or t[-1] < hi or np.count_nonzero(m) < 3:
            rows[w] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
            continue
        d = float(np.sqrt(trapezoid((b[m]-a[m])**2, t[m])))
        n = float(np.sqrt(trapezoid(a[m]**2, t[m])))
        rows[w] = {"state": "MEASURED", "difference_L2": d, "reference_L2": n, "relative_difference": d/n if n else None}
    return rows


def write_archive(path, header, arrays):
    """Checksummed HDF5 (same node manifest as the adapter archives); refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise ValueError("transfer archive exists; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(header)
    with h5py.File(path, "x") as handle:
        handle.attrs.update(schema_version=VERSION, header=text, header_sha256=hashlib.sha256(text.encode()).hexdigest())
        manifest = {}
        _write_tree(handle, "arrays", arrays, manifest)
        encoded = canonical_json(manifest)
        handle.attrs.update(manifest=encoded, manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest())
    return path


def read_archive(path):
    with h5py.File(path, "r") as handle:
        if handle.attrs["schema_version"] != VERSION:
            raise ValueError("not a transfer archive")
        for key in ("header", "manifest"):
            if hashlib.sha256(handle.attrs[key].encode()).hexdigest() != handle.attrs[key+"_sha256"]:
                raise ValueError("transfer archive checksum mismatch")
        manifest = json.loads(handle.attrs["manifest"])
        arrays = _read_tree(handle["arrays"], manifest)
        if manifest:
            raise ValueError("transfer archive manifest lists missing nodes")
        return {"header": json.loads(handle.attrs["header"]), "arrays": arrays}


# --------------------------------------------------------------- validation against TD

AGREEMENT_FACTOR = 10.
GATE_WINDOWS = ("late", "full")


def validate(root, point, output_dir):
    """FD transfer prediction versus the library's TD residual for one point of a plan-0.3 root.

    The TD residual is the stored fine ECO record minus the stored fine matched BH record of
    the same experiment.  The FD prediction uses a fresh BH monitor run at the BH record's
    spacing.  Gate (predeclared, docs/METHOD.md): in the late and full windows
    the FD-TD relative difference must not exceed AGREEMENT_FACTOR times the sum of the TD
    paired residual budget, the largest FD variant spread and the monitor run's BH
    discrepancy, all relative to the residual norm.  Otherwise DISAGREES (reported).
    """
    from . import offfiducial as off
    root = Path(root)
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    report = off.make_report(plan, root, None, [], mode="historical")
    model = point
    seq = plan["points"][point]["sequence"]
    bh_seq = plan["points"][point]["bh_sequence"]
    params = plan["points"][point]["params"]
    experiment = off.as_experiment(plan["experiment"])
    windows = off._windows(plan["sequences"][seq])
    eco = off.load_job(root, plan["jobs"][f"{seq}_fine"], None, mode="historical")
    bh = off.load_job(root, plan["jobs"][f"{bh_seq}_fine"], None, mode="historical")
    spec = off.RunSpec.from_dict(plan["jobs"][f"{bh_seq}_fine"]["spec"])
    t_end = float(bh["t"][-1])+1.
    source = monitor_run(experiment, dx=spec.dx, t_end=t_end, cfl=spec.cfl, sample_every=spec.sample_every)
    observer = experiment.observer_global_rstar
    base = reconstruct(model, params, source, observer=observer)
    variants = {name: reconstruct(model, params, source, observer=observer, **{**{}, **change})
                for name, change in VARIANTS.items()}
    # TD residual and BH consistency on one common grid (cubic spline, no extrapolation).
    t, (y_eco, y_bh, y_pred, y_mon_obs) = common_series(
        [{"t": eco["t"], "psi": eco["psi"]}, {"t": bh["t"], "psi": bh["psi"]},
         {"t": base["t"], "psi": base["correction"]}, {"t": source["t"], "psi": source["observer"]}],
        dt=off.POLICY["common_grid_dt"])
    residual = y_eco-y_bh
    fd_td = _window_rows(t, residual, y_pred, windows)
    bh_consistency = _window_rows(t, y_bh, y_mon_obs, windows)
    spreads = {}
    for name, v in variants.items():
        tv, (a, b) = common_series([{"t": base["t"], "psi": base["correction"]}, {"t": v["t"], "psi": v["correction"]}],
                                   dt=off.POLICY["common_grid_dt"])
        spreads[name] = _window_rows(tv, a, b, windows)
    residual_budget = report["residuals"][seq]["paired_budget"]
    gate = {}
    for w in windows:
        row = fd_td[w]
        if row["state"] != "MEASURED":
            gate[w] = {"state": "UNQUANTIFIED"}
            continue
        n = row["reference_L2"]
        budget_row = residual_budget.get(w, {})
        td_budget = budget_row.get("error_estimate_L2") if budget_row.get("state") == "CONDITIONAL_ESTIMATE" else None
        fd_spread = max(s[w]["difference_L2"] for s in spreads.values() if s[w]["state"] == "MEASURED")
        bh_diff = bh_consistency[w]["difference_L2"]
        if td_budget is None:
            gate[w] = {"state": "UNQUANTIFIED", "reason": "TD residual budget unquantified"}
            continue
        allowance = AGREEMENT_FACTOR*(td_budget+fd_spread+bh_diff)
        gate[w] = {"state": "AGREES" if row["difference_L2"] <= allowance else "DISAGREES",
                   "fd_td_relative": row["relative_difference"], "td_budget_relative": td_budget/n,
                   "fd_variant_spread_relative": fd_spread/n, "bh_monitor_discrepancy_relative": bh_diff/n,
                   "allowance_relative": allowance/n}
    passed = all(gate[w]["state"] == "AGREES" for w in GATE_WINDOWS)
    grid = np.arange(.005, 2.5, .0025)
    products, planes, identities, conditioning = real_axis_products(model, params, observer, grid)
    moved, _, _, _ = real_axis_products(model, params, observer, grid, monitor=-80.)
    plane_error = (np.abs(moved["K"]-products["K"]*np.exp(1j*grid*(-80.+65.)))/(1+np.abs(products["K"])))
    plane_identity = {"max_scaled_all": float(np.max(plane_error)),
                      "max_scaled_omega_le_1.5": float(np.max(plane_error[grid <= 1.5])),
                      "rule": "K(x_m2) = K(x_m1) exp(i omega (x_m2 - x_m1)) for free monitor planes"}
    header = {"model": model, "params": params, "experiment": plan["experiment"],
              "root_name": root.name, "plan_sha256": off.plan_sha256(plan),
              "eco_record": plan["jobs"][f"{seq}_fine"]["identity"], "bh_record": plan["jobs"][f"{bh_seq}_fine"]["identity"],
              "planes": planes, "settings": DEFAULTS, "variants": VARIANTS,
              "conventions": __doc__.split("Seam scattering")[0].strip()}
    arrays = {"real_axis": {k: np.asarray(v) for k, v in products.items()},
              "real_axis_monitor_minus_80": {"K": moved["K"]},
              "comparison": {"t": t, "td_residual": residual, "fd_prediction": y_pred},
              "reconstruction": {"t": base["t"], "correction": base["correction"],
                                 "omega": base["omega"], "K_on_contour": base["K_complex_contour"]},
              "variants": {name: {"correction": v["correction"]} for name, v in variants.items()},
              "monitor_run": {"t": source["t"], "observer": source["observer"],
                              **{f"monitor{abs(m):g}_field": v["field"] for m, v in source["monitors"].items()},
                              **{f"monitor{abs(m):g}_derivative": v["derivative"] for m, v in source["monitors"].items()}}}
    summary = {"model": model, "params": params, "experiment": plan["experiment"], "root_name": root.name,
               "planes": planes, "laplace_shift_eps": base["eps"], "nfft": base["nfft"], "T_fft": base["T_fft"],
               "identities": identities, "plane_transform_identity_max_scaled": plane_identity,
               "conditioning": conditioning, "fd_minus_td": fd_td, "bh_monitor_consistency": bh_consistency,
               "variant_spreads": spreads, "gate": gate,
               "state": "VALIDATED_AGAINST_TD" if passed else "NOT_VALIDATED",
               "monitor_run": source["metadata"], "is_error_bound": False}
    output_dir = Path(output_dir)
    stamp = hashlib.sha256(canonical_json(header).encode()).hexdigest()[:16]
    archive = write_archive(output_dir/f"transfer-{model.lower()}-{stamp}.h5", header, arrays)
    summary["archive"] = archive.name
    summary["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output_dir/f"transfer-{model.lower()}-{stamp}.json").write_text(canonical_json(summary)+"\n", encoding="utf8")
    return summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--point", choices=("GS", "WH"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = validate(args.root, args.point, args.output)
    print(json.dumps({"state": summary["state"], "gate": {w: (g.get("state"), g.get("fd_td_relative"),
                                                             g.get("allowance_relative"))
                                                         for w, g in summary["gate"].items()},
                      "identities": summary["identities"], "plane": summary["plane_transform_identity_max_scaled"],
                      "conditioning": summary["conditioning"], "archive": summary["archive"]}, indent=None))
    return 0 if summary["state"] == "VALIDATED_AGAINST_TD" else 2


if __name__ == "__main__":
    raise SystemExit(main())
