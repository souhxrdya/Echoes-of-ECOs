"""Bounded BS/QH adapter audit, not a parameter sweep or an analysis atlas."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import h5py
import numpy as np
from scipy.integrate import trapezoid

from .adapter_archive import read_archive, request_id, write_archive
from .boson_star import FIDUCIAL_SIGMA_C, run_boson_star
from .quantum_horizon import run_quantum_horizon
from .convergence import common_series, assess, WINDOWS
from .run import BASELINE_H5_SHA256, gate_refinement
from .schema import canonical_json
from .storage import write_json, runtime_provenance


def plan():
    jobs = {}
    for model, params in (("BS", {"sigma_c": FIDUCIAL_SIGMA_C}), ("QH", {"EplM": 1e8, "gamma": 1.})):
        for level, dx in zip(("coarse", "medium", "fine"), (.2, .1, .05)):
            settings = {"dx": dx, "t_end": 520., "cfl": .35, "sigma": 4.,
                        "source_global": 55., "observer_global": 95., "x_right": 400., "sample_every": 2}
            if model == "QH":
                settings.update(surface=-40., x_left=-250., source_t_end=650., source_x_left=-400.,
                                scattering_step=.02, scattering_x_right=200.)
                if level == "medium":
                    settings["independent_dx"] = .05
            else:
                settings.update(equilibrium_rmax=50., equilibrium_n=1200, equilibrium_tol=2e-7)
            jobs[model+"_"+level] = {"model": model, "params": params.copy(), "settings": settings}
    def variant(label, base, **settings):
        req = jobs[base]
        jobs[label] = {**req, "settings": {**req["settings"], **settings}}
    variant("BS_tighter_BVP", "BS_medium", equilibrium_tol=2e-8)
    variant("BS_larger_background", "BS_medium", equilibrium_rmax=60.)
    variant("QH_frozen_replay", "QH_medium", x_left=-140., independent_dx=None)
    variant("QH_finer_scattering", "QH_medium", scattering_step=.01, independent_dx=None)
    return {"version": "bs-qh-adapter-pilot/0.1", "jobs": jobs,
            "scope": "fiducial models only; no mode/classification claims",
            "declared_TD_order": {"BS": 4, "QH": 4},
            "order_basis": "smooth FD4/RK4 expectation, tested rather than presumed; fixed background/scattering settings",
            "independent_QH_comparison_gate": .01,
            "independent_QH_gate_basis": "inherited 1% raw late reconstruction comparison, not total error bound"}


def execute(label, request, root, provenance):
    root = Path(root)
    filename = root/"records"/(request_id(request)+".h5")
    try:
        reused = filename.exists()
        if reused:
            result = read_archive(filename, request, provenance)["result"]
        else:
            start = time.perf_counter()
            solver = run_boson_star if request["model"] == "BS" else run_quantum_horizon
            result = solver(request["params"], **request["settings"])
            result["metadata"]["wall_seconds"] = time.perf_counter()-start
            write_archive(filename, request, result, provenance, root)
            result = read_archive(filename, request, provenance)["result"]
        return {"label": label, "record_id": request_id(request), "state": "COMPLETE", "reused": reused,
                "wall_seconds": result["metadata"]["wall_seconds"],
                "failed_checks": [k for k, v in result["checks"].items() if not v["passed"]]}
    except Exception as exc:
        failure = {"label": label, "state": "FAILED", "request": request, "reason": str(exc),
                   "traceback": traceback.format_exc(), "provenance": provenance}
        write_json(root/"failures"/(label+"-"+uuid.uuid4().hex+".json"), failure, root)
        return {k: failure[k] for k in ("label", "state", "reason")}


def difference(a, b, field="psi"):
    t, y = common_series([{"t": r["t"], "psi": r[field]} for r in (a, b)])
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        if t[0] > lo or t[-1] < hi:
            out[name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT",
                         "difference_L2": None, "relative_difference": None,
                         "meaning": "Full declared window is not covered by both records"}
            continue
        mask = (t >= lo) & (t <= hi)
        if np.count_nonzero(mask) < 3:
            out[name] = {"state": "UNQUANTIFIED", "reason": "TOO_FEW_SAMPLES",
                         "difference_L2": None, "relative_difference": None,
                         "meaning": "Fewer than three samples in the declared window"}
            continue
        delta = float(np.sqrt(trapezoid((y[1][mask]-y[0][mask])**2, t[mask])))
        norm = float(np.sqrt(trapezoid(y[0][mask]**2, t[mask])))
        out[name] = {"state": "MEASURED_SENSITIVITY", "difference_L2": delta,
                     "relative_difference": delta/norm if norm else None,
                     "meaning": "measured sensitivity, not a total error estimate"}
    return out


def report(configuration, root, provenance, execution):
    root = Path(root)
    records, missing = {}, {}
    for label, request in configuration["jobs"].items():
        try:
            records[label] = read_archive(root/"records"/(request_id(request)+".h5"), request, provenance)["result"]
        except (OSError, ValueError, KeyError) as exc:
            missing[label] = str(exc)
    out = {"version": configuration["version"], "provenance": provenance, "execution": execution,
           "missing": missing, "refinement": {}, "controls": {}, "replay": {},
           "complete_library": False, "numerical_estimates_are_bounds": False}
    for model in ("BS", "QH"):
        labels = [model+"_"+level for level in ("coarse", "medium", "fine")]
        if any(k not in records for k in labels):
            continue
        rs = [records[k] for k in labels]
        t, ys = common_series(rs)
        hs = [r["metadata"]["actual_dx"] for r in rs]
        valid = all(v["passed"] for r in rs for v in r["checks"].values())
        out["refinement"][model] = gate_refinement(assess(t, ys, hs, expected_order=4), valid)
        if model == "QH":
            correction = [{"t": r["t"], "psi": r["response"]["correction"]} for r in rs]
            tc, yc = common_series(correction)
            out["refinement"]["QH_correction"] = gate_refinement(assess(tc, yc, hs, expected_order=4), valid)
    for label, base in (("BS_tighter_BVP", "BS_medium"), ("BS_larger_background", "BS_medium"),
                        ("QH_finer_scattering", "QH_medium")):
        if label in records and base in records:
            out["controls"][label] = difference(records[base], records[label])
            if label.startswith("QH"):
                out["controls"][label+"_correction"] = difference(
                    {"t": records[base]["t"], "psi": records[base]["response"]["correction"]},
                    {"t": records[label]["t"], "psi": records[label]["response"]["correction"]})
    from .paths import BASELINE_OUTPUTS
    frozen = BASELINE_OUTPUTS/"ECO_waveforms.h5"
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != BASELINE_H5_SHA256:
        raise ValueError("Frozen HDF5 changed; regression refused")
    with h5py.File(frozen, "r") as f:
        for model, label in (("BS", "BS_medium"), ("QH", "QH_frozen_replay")):
            if label not in records:
                continue
            saved = f["controlled_comparison"][model]
            r = records[label]
            ts, ps = saved["time"][:], saved["waveform_real"][:]
            exact_time = np.array_equal(ts, r["t"])
            error = float(np.max(abs(ps-r["psi"]))) if exact_time else None
            out["replay"][model] = {"exact_time_grid": exact_time, "max_abs_error": error,
                "samples": len(ts), "passed": bool(exact_time and error <= 1e-10*np.max(abs(ps))),
                "purpose": "adapter fidelity only"}
    if "QH_medium" in records:
        r = records["QH_medium"]
        direct = r["independent"]
        value = direct.get("raw_late_relative_L2")
        out["independent_QH"] = {"raw_late_relative_L2": value, "dx": direct.get("dx"),
            "gate": .01, "passed": value is not None and value < .01,
            "scope": "independent scheme comparison, not a continuum error bound",
            "source_omega_c": r["metadata"]["source_omega_c"]}
    out["failed_checks"] = {k: [n for n, v in r["checks"].items() if not v["passed"]]
                            for k, r in records.items()}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        p.error("workers outside available CPU count")
    root = args.output.resolve()
    configuration = plan()
    path = root/"plan.json"
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError("Conflicting plan; choose a new output directory")
    else:
        write_json(path, configuration, root)
    print(f"Bounded adapter pilot: {len(configuration['jobs'])} runs, {args.workers} workers", flush=True)
    if args.plan_only:
        return 0
    provenance = runtime_provenance(Path(__file__).resolve().parents[1])
    execution = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, label, req, str(root), provenance)
                   for label, req in configuration["jobs"].items()]
        for future in as_completed(futures):
            result = future.result()
            execution.append(result)
            print(canonical_json(result), flush=True)
    summary = report(configuration, root, provenance, execution)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
    destination = root/"reports"/(stamp+".json")
    write_json(destination, summary, root)
    print(f"Report: {destination}", flush=True)
    production_failed = any(v for k, v in summary["failed_checks"].items() if k != "QH_frozen_replay")
    return int(bool(summary["missing"]) or production_failed or
               any(not v["passed"] for v in summary["replay"].values()) or
               not summary.get("independent_QH", {}).get("passed", False))


if __name__ == "__main__":
    raise SystemExit(main())
