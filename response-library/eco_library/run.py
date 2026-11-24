"""Bounded local pilot; spawn-safe, resumable, one immutable HDF5 per run.

Run from the repository root: python -m eco_library.run --output runs/pilot-v1
This produces numerical evidence, not an atlas or a universal acceptance label.
"""
from __future__ import annotations

import os

# Each process is a 1-D finite-difference solver, not a multithreaded BLAS job.
# Set before importing numpy/scipy, including in Windows spawned workers.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import h5py
import numpy as np
from scipy.integrate import trapezoid

from .baseline_runtime import NOTEBOOK_SHA256, load_frozen_runtime
from .convergence import WINDOWS, assess, common_series
from .schema import RunSpec, canonical_json
from .solvers import refinement_spacings, run_td
from .storage import assert_output_path, read_record, runtime_provenance, write_json, write_record

BASELINE_H5_SHA256 = "cfaba7f82193ed4a94a5258fe2c688be90c8c65a3e29831b3d1e4a46fb1a10fc"


def build_pilot(*, t_end=520., with_controls=False):
    """12 refinement runs + one original-domain BH adapter regression.

    The 3 extra BH runs match the actual GS spacings, not its nominal dx.
    Same spacing does not imply identical grid origin or error cancellation.
    """
    records, sequences = {}, []

    def sequence(label, model, params, hs, order):
        specs = [RunSpec(model, params, dx=h, t_end=t_end,
                         x_left=-250. if model == "BH" else None) for h in hs]
        for spec in specs:
            records[spec.record_id] = spec.to_dict()
        result = {"name": label, "model": model, "record_ids": [s.record_id for s in specs],
                  "expected_order": order, "bh_reference_ids": None}
        sequences.append(result)
        return result

    hs = refinement_spacings("BH", {})
    bh = sequence("BH", "BH", {}, hs, 4)
    gs_hs = refinement_spacings("GS", {"compactness": .47})
    bh_gs = sequence("BH_GS_SPACINGS", "BH", {}, gs_hs, 4)
    gs = sequence("GS", "GS", {"compactness": .47}, gs_hs, 2)
    wh = sequence("WH", "WH", {"r0_over_M": 2.0005}, hs, 4)
    gs["bh_reference_ids"] = bh_gs["record_ids"]
    wh["bh_reference_ids"] = bh["record_ids"]
    replay = RunSpec("BH", dx=.1, t_end=t_end, x_left=-140.)
    records[replay.record_id] = replay.to_dict()
    controls = []
    if with_controls:
        for seq in sequences:
            base = RunSpec.from_dict(records[seq["record_ids"][-1]])
            variants = [("RIGHT_DOMAIN", replace(base, x_right=450.)),
                        ("HALF_TIMESTEP", replace(base, cfl=base.cfl/2,
                                                   sample_every=base.sample_every*2))]
            if base.model == "BH":
                # Move by an integer requested-h cell count. np.arange's effective
                # representable step can still change: audit grid perturbation too.
                variants.append(("LEFT_DOMAIN", replace(base,
                    x_left=base.x_left-round(50./base.dx)*base.dx)))
            elif base.model == "WH":
                variants.append(("LEFT_DOMAIN", replace(base, x_left=-450.)))
            for kind, variant in variants:
                records[variant.record_id] = variant.to_dict()
                controls.append({"name": seq["name"]+"_"+kind, "sequence": seq["name"],
                                 "kind": kind, "base_id": base.record_id,
                                 "variant_id": variant.record_id})
    return {"plan_version": "axial-td-pilot/0.2", "records": records,
            "sequences": sequences, "baseline_replay_id": replay.record_id,
            "controls": controls, "control_budget_fraction": .1,
            "bh_reference_mode": "MATCHED_RESOLUTION_NOT_IDENTICAL_GRID",
            "common_grid_dt": .07, "windows_M": WINDOWS,
            "order_band": .5, "richardson_safety_factor": 1.25,
            "threshold_provenance": "PILOT_POLICY_NOT_CALIBRATED_PHYSICS_THRESHOLD",
            "scope": "BH/GS/WH fiducial TD refinement only; no modes, reflectivity or classification"}


def gate_refinement(evidence, checks_pass):
    """Keep raw differences for debugging, never publish a usable failed-run estimate."""
    if checks_pass:
        return evidence
    return {name: {**item, "state": "UNQUANTIFIED", "reason": "RECORD_CHECKS_FAILED",
                   "fine_error_estimate_L2": None, "fine_relative_error_estimate": None}
            for name, item in evidence.items()}


def report_exit_code(report):
    if report["unavailable_records"] or any(x["state"] in {"FAILED", "WORKER_FAILED"}
                                           for x in report["execution"]):
        return 1
    if (any(x["state"] == "FAIL" for x in report["baseline_regression"].values())
            or any(x.get("checks_pass") is False for x in report["sequences"].values())
            or any(x.get("checks_pass") is False for x in report.get("controls", {}).values())):
        return 2
    return 0


def _execute(spec_dict, root_string, provenance):
    root = Path(root_string)
    spec = RunSpec.from_dict(spec_dict)
    path = root / "records" / (spec.record_id + ".h5")
    start = time.perf_counter()
    try:
        if path.exists():
            record = read_record(path, expected_spec=spec, expected_provenance=provenance)
            reused = True
        else:
            e = spec.experiment
            record = run_td(spec.model, spec.params, dx=spec.dx, t_end=spec.t_end,
                            cfl=spec.cfl, source_global=e.source_global_rstar,
                            observer_global=e.observer_global_rstar, sigma=e.sigma,
                            x_right=spec.x_right, x_left=spec.x_left,
                            sample_every=spec.sample_every)
            record["metadata"]["wall_seconds"] = time.perf_counter() - start
            write_record(path, spec, record, provenance, root)
            # Always read back the published bytes, rather than trust the writer.
            record = read_record(path, expected_spec=spec, expected_provenance=provenance)
            reused = False
        return {"record_id": spec.record_id, "state": "COMPLETE", "reused": reused,
                "wall_seconds": record["metadata"]["wall_seconds"],
                "checks_pass": all(v["passed"] for v in record["checks"].values()),
                "failed_checks": [k for k, v in record["checks"].items() if not v["passed"]]}
    except Exception as exc:
        failure = {"record_id": spec.record_id, "state": "FAILED", "spec": spec.to_dict(),
                   "exception": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "provenance": provenance}
        write_json(root / "failures" / f"{spec.record_id}-{uuid.uuid4().hex}.json", failure, root)
        return {"record_id": spec.record_id, "state": "FAILED", "reason": str(exc)}


def _l2(t, y, lo, hi):
    mask = (t >= lo) & (t <= hi)
    return float(np.sqrt(trapezoid(y[mask]**2, t[mask])))


def _paired_budget(eco, bh, residual_t, residual_fine):
    """No cancellation assumption: triangle sum of *conditional* estimates."""
    result = {}
    for name, (lo, hi) in WINDOWS.items():
        a = eco[name].get("fine_error_estimate_L2")
        b = bh[name].get("fine_error_estimate_L2")
        if a is None or b is None or residual_t[0] > lo or residual_t[-1] < hi:
            result[name] = {"state": "UNQUANTIFIED", "reason": "BOTH_WAVEFORM_ESTIMATES_REQUIRED"}
            continue
        norm = _l2(residual_t, residual_fine, lo, hi)
        result[name] = {"state": "CONDITIONAL_ESTIMATE", "error_estimate_L2": a+b,
                        "fine_residual_L2": norm, "relative_error_estimate": (a+b)/norm if norm else None,
                        "is_error_bound": False, "includes_independent_systematics": False}
    return result


def baseline_replay(plan, records, repo):
    """Compare like experiments only; no alignment and no amplitude fitting."""
    from .paths import BASELINE_OUTPUTS
    path = BASELINE_OUTPUTS/"ECO_waveforms.h5"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_H5_SHA256:
        raise ValueError("Frozen HDF5 checksum changed; baseline regression refused")
    ids = {"BH": plan["baseline_replay_id"]}
    ids.update({s["model"]: s["record_ids"][1] for s in plan["sequences"] if s["model"] != "BH"})
    out = {}
    with h5py.File(path, "r") as f:
        for model, record_id in ids.items():
            if record_id not in records:
                out[model] = {"state": "UNAVAILABLE"}
                continue
            rec = records[record_id]
            saved = f["controlled_comparison"][model]
            ts, ys = saved["time"][:], saved["waveform_real"][:]
            n = min(len(ts), len(rec["t"]))
            # GS nominal .1 and exact shell-spacing requests have the same actual grid.
            exact_t = np.array_equal(ts[:n], rec["t"][:n])
            if exact_t:
                error = float(np.max(np.abs(ys[:n]-rec["psi"][:n])))
                scale = float(np.max(np.abs(ys[:n])))
                out[model] = {"state": "PASS" if error <= 1e-10*scale else "FAIL",
                              "max_abs_error": error, "relative_max_error": error/scale if scale else None,
                              "exact_time_grid": True, "samples": n, "tested_end_M": float(ts[n-1]),
                              "full_saved_record": n == len(ts),
                              "tolerance_relative_max": 1e-10,
                              "record_id": record_id, "purpose": "ADAPTER_REGRESSION_NOT_CONTINUUM_VALIDATION"}
            else:
                out[model] = {"state": "FAIL", "reason": "NATIVE_TIME_GRID_DIFFERS", "record_id": record_id}
    return out


def make_report(plan, root, provenance, execution, repo):
    records = {}
    unavailable = {}
    for rid, spec in plan["records"].items():
        try:
            records[rid] = read_record(Path(root)/"records"/(rid+".h5"),
                                       expected_spec=RunSpec.from_dict(spec), expected_provenance=provenance)
        except (OSError, ValueError, KeyError) as exc:
            unavailable[rid] = str(exc)
    sequences = {}
    for seq in plan["sequences"]:
        ids = seq["record_ids"]
        if any(rid not in records for rid in ids):
            sequences[seq["name"]] = {"state": "INCOMPLETE"}
            continue
        rs = [records[rid] for rid in ids]
        t, ys = common_series(rs, dt=plan["common_grid_dt"])
        hs = [r["metadata"]["actual_dx"] for r in rs]
        failed = {rid: [k for k, v in records[rid]["checks"].items() if not v["passed"]]
                  for rid in ids}
        passed = not any(failed.values())
        raw = gate_refinement(assess(t, ys, hs, expected_order=seq["expected_order"]), passed)
        sequences[seq["name"]] = {"state": "NUMERICALLY_ASSESSED" if passed else "CHECKS_FAILED",
                                    "checks_pass": passed, "failed_checks": failed,
                                    "record_ids": ids, "waveform_refinement": raw,
                                    "global_validation": "NOT_ASSIGNED"}
    for seq in plan["sequences"]:
        ref = seq["bh_reference_ids"]
        if not ref or any(rid not in records for rid in seq["record_ids"]+ref):
            continue
        item = sequences[seq["name"]]
        rs = [records[rid] for rid in seq["record_ids"]+ref]
        t, ys = common_series(rs, dt=plan["common_grid_dt"])
        residuals = [ys[i]-ys[i+3] for i in range(3)]
        hs = [r["metadata"]["actual_dx"] for r in rs[:3]]
        item["bh_reference_ids"] = ref
        ref_name = next(s["name"] for s in plan["sequences"] if s["record_ids"] == ref)
        all_checks = item["checks_pass"] and sequences[ref_name]["checks_pass"]
        item["residual_refinement"] = gate_refinement(
            assess(t, residuals, hs, expected_order=seq["expected_order"]), all_checks)
        item["paired_waveform_budget"] = (_paired_budget(item["waveform_refinement"],
            sequences[ref_name]["waveform_refinement"], t, residuals[-1]) if all_checks else
            {name: {"state": "UNQUANTIFIED", "reason": "RECORD_CHECKS_FAILED"} for name in WINDOWS})
    controls = {}
    for control in plan["controls"]:
        ids = (control["base_id"], control["variant_id"])
        if any(rid not in records for rid in ids):
            controls[control["name"]] = {"state": "INCOMPLETE"}
            continue
        rs = [records[rid] for rid in ids]
        t, ys = common_series(rs, dt=plan["common_grid_dt"])
        checks_pass = all(v["passed"] for r in rs for v in r["checks"].values())
        windows = {}
        for name, (lo, hi) in WINDOWS.items():
            if t[0] > lo or t[-1] < hi or not checks_pass:
                windows[name] = {"state": "UNQUANTIFIED"}
                continue
            delta = _l2(t, ys[1]-ys[0], lo, hi)
            norm = _l2(t, ys[0], lo, hi)
            budget = sequences[control["sequence"]]["waveform_refinement"][name].get("fine_error_estimate_L2")
            ratio = delta/budget if budget is not None and budget > 0 else None
            windows[name] = {"difference_L2": delta, "relative_difference_L2": delta/norm if norm else None,
                             "fraction_of_conditional_refinement_estimate": ratio,
                             "declared_budget_fraction": plan["control_budget_fraction"],
                             "state": ("UNQUANTIFIED" if ratio is None else
                                       "BELOW_DECLARED_REFINEMENT_BUDGET" if ratio <= plan["control_budget_fraction"]
                                       else "SYSTEMATIC_NOT_SUBDOMINANT"), "is_error_bound": False}
        # Expose any nominally pure-domain comparison's floating-point lattice drift.
        xb, xv = rs[0]["x"], rs[1]["x"]
        start = int(np.argmin(abs(xv-xb[0])))
        overlap = xv[start:start+len(xb)]
        drift = float(np.max(abs(overlap-xb))) if len(overlap) == len(xb) else None
        controls[control["name"]] = {**control, "checks_pass": checks_pass, "windows": windows,
                                     "max_overlap_grid_drift_M": drift,
                                     "actual_dx": [r["metadata"]["actual_dx"] for r in rs],
                                     "scope": "MEASURED_SENSITIVITY_NOT_SYSTEMATIC_ERROR_BOUND"}
    sample_grid_sensitivity = {}
    for seq in plan["sequences"]:
        if any(rid not in records for rid in seq["record_ids"]):
            continue
        rs = [records[rid] for rid in seq["record_ids"]]
        hs = [r["metadata"]["actual_dx"] for r in rs]
        sample_grid_sensitivity[seq["name"]] = {}
        for dt in (.035, .0175):
            t, ys = common_series(rs, dt=dt)
            sample_grid_sensitivity[seq["name"]][str(dt)] = gate_refinement(
                assess(t, ys, hs, expected_order=seq["expected_order"]),
                sequences[seq["name"]]["checks_pass"])
    return {"report_version": "axial-td-pilot/0.2", "provenance": provenance,
            "execution": execution, "unavailable_records": unavailable, "sequences": sequences,
            "controls": controls, "common_sample_grid_sensitivity": sample_grid_sensitivity,
            "baseline_regression": baseline_replay(plan, records, repo),
            "mode_status": "NOT_COMPUTED", "reflectivity_status": "NOT_COMPUTED",
            "classification_status": "NOT_COMPUTED", "library_complete": False,
            "systematics": {"boundary_variation": "SEE_CONTROLS" if controls else "NOT_COMPUTED",
                            "timestep_variation": "SEE_CONTROLS" if controls else "NOT_COMPUTED",
                            "source_width": "NOT_COMPUTED", "observer_interpolation": "NOT_COMPUTED",
                            "common_grid_sensitivity": "MEASURED_NOT_INTERPOLATION_ERROR_BOUND"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--t-end", type=float, default=520.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--with-controls", action="store_true",
                        help="Add 11 fine-grid domain/time-step controls (24 total runs)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        parser.error("workers must be between 1 and available logical CPU count")
    root = args.output.resolve()
    assert_output_path(root/"plan.json", root)
    repo = Path(__file__).resolve().parents[1]
    # Verify the immutable source even if all records are about to be resumed.
    load_frozen_runtime()
    plan = build_pilot(t_end=args.t_end, with_controls=args.with_controls)
    plan_path = root/"plan.json"
    if plan_path.exists():
        if canonical_json(json.loads(plan_path.read_text())) != canonical_json(plan):
            raise ValueError("Output directory contains a different plan; choose a new directory")
    else:
        write_json(plan_path, plan, root)
    print(f"Plan: {len(plan['records'])} independent runs; {args.workers} workers; {root}", flush=True)
    if args.plan_only:
        return 0
    provenance = runtime_provenance(repo)
    provenance["baseline_notebook_sha256"] = NOTEBOOK_SHA256
    execution = []
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_execute, s, str(root), provenance) for s in plan["records"].values()]
            for f in as_completed(futures):
                try:
                    result = f.result()
                except Exception as exc:
                    result = {"state": "WORKER_FAILED", "reason": repr(exc)}
                execution.append(result)
                print(canonical_json(result), flush=True)
    report = make_report(plan, root, provenance, execution, repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = root/"reports"/(stamp+".json")
    write_json(path, report, root)
    print(f"Report: {path}", flush=True)
    for name, item in report["sequences"].items():
        states = {k: v["state"] for k, v in item.get("waveform_refinement", {}).items()}
        print(f"{name}: {item['state']} {states}", flush=True)
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
