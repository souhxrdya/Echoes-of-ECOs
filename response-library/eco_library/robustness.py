"""Phase 2 robustness of the off-fiducial points (plan robustness/0.1, docs/VALIDATION.md).

Three experiments besides the production experiment E0 (sigma 4 M, r*_s 55, r*_o 95)
are executed as complete plan-0.3 blocks with ``offfiducial plan --experiment``:
S3 (sigma 3 M), S5 (sigma 5 M) and O120 (observer r* 120 M).  This module adds

* a supplement of near-threshold and two-factor controls at E0 (``plan``/``run``), and
* the predeclared analysis (``report``): physical dependence on the experiment against
  the numerical budget of both experiments, comparison-grid sensitivity of every
  refinement estimate, time-step convergence and plane dependence of the near-threshold
  factors, and additivity of two-factor controls.

Pulse amplitude and incident normalisation are the experiments' own (unit Gaussian,
F_in=1); no output is peak-normalised, aligned or fitted.  A record at another observer
is compared at retarded time t - [(r*_s-55)+(r*_o-95)], the declared propagation
convention.  Differences between two settings are sensitivities, not error bounds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import trapezoid

from . import offfiducial as off
from .convergence import WINDOWS, assess, common_series
from .schema import Experiment, RunSpec, canonical_json

PLAN_VERSION = "robustness-supplement/0.1"
REPORT_VERSION = "robustness-report/0.1"
EXPERIMENTS = {"E0": off.PRODUCTION_EXPERIMENT, "S3": Experiment(sigma=3.), "S5": Experiment(sigma=5.),
               "O120": Experiment(observer_global_rstar=120.)}
RESOLUTION_FRACTION = .1      # numerical budget must be at most 10% of a difference to resolve it
GRID_SENSITIVITY_FRACTION = .1
POINT_SEQUENCES = {"GS": ("GS", "BHGS"), "WH": ("WH", "BH"), "BS": ("BS", "BH"), "QH": ("QH", None)}


# ----------------------------------------------------------------------- supplement plan

def build_supplement(base_plan):
    """Near-threshold and two-factor controls on the E0 base plan's records; no measured result."""
    if base_plan.get("experiment") != asdict(off.PRODUCTION_EXPERIMENT):
        raise ValueError("the supplement is declared at the production experiment E0")
    base_jobs = base_plan["jobs"]
    jobs, comparisons = {}, {}

    def td(name, base_label, purpose, **changes):
        base = base_jobs[base_label]
        spec = replace(RunSpec.from_dict(base["spec"]), **changes)
        extra = {"expected_interior_cells": base["expected_interior_cells"]} if "expected_interior_cells" in base else {}
        jobs[name] = off._td_job(spec, "control", base["point"], purpose, **extra)

    def adapter(name, base_label, purpose, seconds, **changes):
        base = base_jobs[base_label]
        request = {**base["request"], "settings": {**base["request"]["settings"], **changes}}
        jobs[name] = off._adapter_job(base["kind"], request, "control", base["point"], purpose, seconds)

    half, quarter = {"cfl": .175, "sample_every": 4}, {"cfl": .0875, "sample_every": 8}
    td("GS_HALF_TIMESTEP_RIGHT_DOMAIN", "GS_fine", "two factors: CFL 0.175 and right boundary 450", x_right=450., **half)
    td("WH_HALF_TIMESTEP_LEFT_DOMAIN", "WH_fine", "two factors: CFL 0.175 and left boundary -450", x_left=-450., **half)
    adapter("BS_QUARTER_TIMESTEP", "BS_fine", "near threshold: CFL 0.0875 (time-step convergence)", 110., **quarter)
    adapter("BS_HALF_TIMESTEP_RIGHT_DOMAIN", "BS_fine", "two factors: CFL 0.175 and right boundary 450", 60.,
            x_right=450., **half)
    adapter("QH_QUARTER_TIMESTEP", "QH_medium", "near threshold: CFL 0.0875 at dx=0.1", 110., **quarter)
    adapter("QH_PLANE_45", "QH_medium", "near threshold: computational plane -45, source remeasured", 30.,
            surface=-45.)
    adapter("QH_PLANE_60", "QH_medium", "near threshold: computational plane -60, source remeasured", 30.,
            surface=-60.)
    adapter("QH_PLANE_50_HALF_TIMESTEP", "QH_medium", "two factors: plane -50 and CFL 0.175", 60.,
            surface=-50., **half)
    # Comparisons name records by (root, label): "base" is the E0 plan-0.3 root, "supplement" this plan.
    comparisons["time_step_convergence"] = {
        "BS": {"field": "psi", "estimate_sequence": "BS", "levels": [["base", "BS_fine"], ["base", "BS_HALF_TIMESTEP"],
                                                                    ["supplement", "BS_QUARTER_TIMESTEP"]]},
        "QH": {"field": "psi", "estimate_sequence": "QH", "levels": [["base", "QH_medium"], ["base", "QH_HALF_TIMESTEP"],
                                                                    ["supplement", "QH_QUARTER_TIMESTEP"]]},
        "QH_CORRECTION": {"field": "correction", "estimate_sequence": "QH_CORRECTION",
                          "levels": [["base", "QH_medium"], ["base", "QH_HALF_TIMESTEP"],
                                     ["supplement", "QH_QUARTER_TIMESTEP"]]}}
    comparisons["plane_dependence"] = {"base": ["base", "QH_medium"], "planes": {
        "-45": ["supplement", "QH_PLANE_45"], "-50": ["base", "QH_PLANE_50"], "-60": ["supplement", "QH_PLANE_60"]}}
    comparisons["interactions"] = {
        "GS": {"field": "psi", "base": ["base", "GS_fine"], "a": ["base", "GS_HALF_TIMESTEP"],
               "b": ["base", "GS_RIGHT_DOMAIN"], "ab": ["supplement", "GS_HALF_TIMESTEP_RIGHT_DOMAIN"]},
        "WH": {"field": "psi", "base": ["base", "WH_fine"], "a": ["base", "WH_HALF_TIMESTEP"],
               "b": ["base", "WH_LEFT_DOMAIN"], "ab": ["supplement", "WH_HALF_TIMESTEP_LEFT_DOMAIN"]},
        "BS": {"field": "psi", "base": ["base", "BS_fine"], "a": ["base", "BS_HALF_TIMESTEP"],
               "b": ["base", "BS_RIGHT_DOMAIN"], "ab": ["supplement", "BS_HALF_TIMESTEP_RIGHT_DOMAIN"]},
        "QH": {"field": "correction", "base": ["base", "QH_medium"], "a": ["base", "QH_HALF_TIMESTEP"],
               "b": ["base", "QH_PLANE_50"], "ab": ["supplement", "QH_PLANE_50_HALF_TIMESTEP"]}}
    return {"plan_version": PLAN_VERSION, "document": "docs/VALIDATION.md",
            "base_plan_sha256": off.plan_sha256(base_plan), "base_plan_version": base_plan["plan_version"],
            "experiment": base_plan["experiment"], "jobs": jobs, "comparisons": comparisons,
            "experiments": {k: asdict(v) for k, v in EXPERIMENTS.items()},
            "policy": {"resolution_fraction": RESOLUTION_FRACTION,
                       "grid_sensitivity_fraction": GRID_SENSITIVITY_FRACTION,
                       "control_budget_fraction": off.POLICY["control_budget_fraction"],
                       "retarded_time_rule": off.POLICY["experiment_window_rule"],
                       "normalisation": "none: unit initial Gaussian and F_in=1 in every experiment",
                       "differences_are_error_bounds": False}}


# --------------------------------------------------------------------------- helpers

def _load(root, plan, label):
    return off.load_job(root, plan["jobs"][label], None, mode="historical")


def _stored_plan(root):
    return json.loads((Path(root)/"plan.json").read_text(encoding="utf8"))


def _norm(t, y, lo, hi):
    mask = (t >= lo) & (t <= hi)
    return float(np.sqrt(trapezoid(y[mask]**2, t[mask]))) if np.count_nonzero(mask) >= 3 else None


def _on_common(records, dt=off.POLICY["common_grid_dt"]):
    return common_series([{"t": r["t"], "psi": r["psi"]} for r in records], dt=dt)


def _shifted(series_value, shift):
    return {"t": np.asarray(series_value["t"], float)-shift, "psi": np.asarray(series_value["psi"], float)}


def _components(report, sequence, window):
    """L2 differences of the non-subdominant controls on one sequence (a measured numerical budget part)."""
    total, names = 0., []
    for name, control in report["controls"].items():
        row = control["windows"].get(window, {})
        if control["estimate_sequence"] == sequence and row.get("state") == "SYSTEMATIC_NOT_SUBDOMINANT":
            total += row["difference_L2"]
            names.append(name)
    return total, names


# --------------------------------------------------------------- experiment dependence

def _budget(report, sequences, window, residual_of=None):
    """Numerical budget of one experiment for an observable: refinement (or paired residual) estimate
    plus the non-subdominant control differences of every sequence involved.  None if unquantified."""
    if residual_of is not None:
        row = report["residuals"].get(residual_of, {}).get("paired_budget", {}).get(window, {})
        estimate = row.get("error_estimate_L2") if row.get("state") == "CONDITIONAL_ESTIMATE" else None
    else:
        row = report["sequences"][sequences[0]]["refinement"][window]
        estimate = row.get("fine_error_estimate_L2") if row.get("state") == "OBSERVED_ORDER_CONSISTENT" else None
    if estimate is None:
        return None, []
    total, names = estimate, []
    for seq in sequences:
        extra, more = _components(report, seq, window)
        total += extra
        names += more
    return total, names


def experiment_dependence(roots, reports, plans, base="E0"):
    """Differences of each point's observables between E0 and each other experiment (Section 3)."""
    out = {}
    e0 = off.as_experiment(plans[base]["experiment"])
    for name in [k for k in roots if k != base]:
        shift = off.window_shift(off.as_experiment(plans[name]["experiment"]))-off.window_shift(e0)
        rows = {}
        for point, (seq, bh_seq) in POINT_SEQUENCES.items():
            if reports[base]["points"][point]["outcome"] != "ACCEPTED_PILOT_SCOPE" or \
                    reports[name]["points"][point]["outcome"] != "ACCEPTED_PILOT_SCOPE":
                rows[point] = {"state": "UNQUANTIFIED", "reason": "point not accepted in both experiments"}
                continue
            observables = {}
            fine = f"{seq}_fine"
            a, b = _load(roots[base], plans[base], fine), _load(roots[name], plans[name], fine)
            if point == "QH":
                pairs = {"waveform": (off.series(a, "psi"), off.series(b, "psi"), ("QH",), None),
                         "correction": (off.series(a, "correction"), off.series(b, "correction"),
                                        ("QH_CORRECTION",), None)}
            else:
                bh = f"{bh_seq}_fine"
                ra, rb = _load(roots[base], plans[base], bh), _load(roots[name], plans[name], bh)
                t_a, (ya, yra) = _on_common([a, ra])
                t_b, (yb, yrb) = _on_common([b, rb])
                pairs = {"waveform": (off.series(a, "psi"), off.series(b, "psi"), (seq,), None),
                         "residual": ({"t": t_a, "psi": ya-yra}, {"t": t_b, "psi": yb-yrb}, (seq, bh_seq), seq)}
            for observable, (sa, sb, seqs, residual_of) in pairs.items():
                t, (ya_, yb_) = common_series([sa, _shifted(sb, shift)], dt=off.POLICY["common_grid_dt"])
                windows = {}
                for w, (lo, hi) in WINDOWS.items():
                    d, n = _norm(t, yb_-ya_, lo, hi), _norm(t, ya_, lo, hi)
                    ba, na = _budget(reports[base], seqs, w, residual_of)
                    bb, nb = _budget(reports[name], seqs, w, residual_of)
                    row = {"difference_L2": d, "base_L2": n, "relative_difference": d/n if d is not None and n else None,
                           "budget_base_L2": ba, "budget_other_L2": bb, "budget_components": na+nb}
                    if d is None or ba is None or bb is None:
                        row["state"] = "UNQUANTIFIED"
                    else:
                        row["budget_fraction_of_difference"] = (ba+bb)/d if d > 0 else None
                        row["state"] = ("RESOLVED_DEPENDENCE" if d > 0 and ba+bb <= RESOLUTION_FRACTION*d
                                        else "NOT_RESOLVED")
                    windows[w] = row
                observables[observable] = windows
            rows[point] = {"state": "MEASURED", "observables": observables}
        out[name] = {"experiment": plans[name]["experiment"], "retarded_shift_M": shift, "points": rows,
                     "scope": "physical dependence on the experiment; a sensitivity, not an error bound"}
    return out


# ---------------------------------------------------------- comparison-grid sensitivity

def _native_coincident(records):
    """The coarsest record's native times (within the common support) and every level's values there,
    without interpolation."""
    t0 = np.asarray(records[0]["t"], float)
    lo = max(float(np.asarray(r["t"], float)[0]) for r in records)
    hi = min(float(np.asarray(r["t"], float)[-1]) for r in records)
    t0 = t0[(t0 >= lo-1e-9) & (t0 <= hi+1e-9)]
    values = []
    for r in records:
        t = np.asarray(r["t"], float)
        idx = np.searchsorted(t, t0)
        idx = np.clip(idx, 0, len(t)-1)
        lower = np.clip(idx-1, 0, len(t)-1)
        pick = np.where(np.abs(t[lower]-t0) < np.abs(t[idx]-t0), lower, idx)
        if np.max(np.abs(t[pick]-t0)) > 1e-9:
            raise ValueError("levels share no native sample times")
        values.append(np.asarray(r["psi"], float)[pick])
    return t0, values


def comparison_grid_sensitivity(root, plan):
    """Refinement estimates on the default spline grid (dt 0.07), a finer spline grid (dt 0.035) and the
    coarsest level's native samples (no interpolation), per sequence and window (Section 4)."""
    out = {}
    windows_all = {n: off._windows(s) for n, s in plan["sequences"].items()}
    for name, seq in plan["sequences"].items():
        records = [_load(root, plan, label) for label in seq["labels"]]
        series = [off.series(r, seq["field"]) for r in records]
        hs = [off.spacing(r, seq["h"]) for r in records]
        variants = {}
        for key, maker in (("spline_dt_0.07", lambda: common_series(series, dt=.07)),
                           ("spline_dt_0.035", lambda: common_series(series, dt=.035)),
                           ("native_coarse_samples", lambda: _native_coincident(series))):
            t, ys = maker()
            variants[key] = assess(t, ys, hs, expected_order=seq["expected_order"],
                                   order_tolerance=off.POLICY["order_band"], windows=windows_all[name])
        rows = {}
        for w in windows_all[name]:
            ref = variants["spline_dt_0.07"][w]
            row = {k: {"state": v[w]["state"], "fine_error_estimate_L2": v[w].get("fine_error_estimate_L2"),
                       "p_observed": v[w].get("p_observed")} for k, v in variants.items()}
            e0 = ref.get("fine_error_estimate_L2")
            ratios = {k: (v[w].get("fine_error_estimate_L2")/e0 if e0 and v[w].get("fine_error_estimate_L2") else None)
                      for k, v in variants.items() if k != "spline_dt_0.07"}
            same_state = all(v[w]["state"] == ref["state"] for v in variants.values())
            sensitive = (not same_state) or any(r is None or abs(r-1) > GRID_SENSITIVITY_FRACTION
                                                for r in ratios.values()) if e0 else not same_state
            row.update(estimate_ratio_to_default=ratios, same_state=same_state,
                       state="COMPARISON_GRID_SENSITIVE" if sensitive else "COMPARISON_GRID_INSENSITIVE")
            rows[w] = row
        out[name] = rows
    return out


# --------------------------------------------------- near-threshold and two-factor controls

def _record(ref, roots, plans):
    where, label = ref
    return _load(roots[where], plans[where], label)


def time_step_convergence(spec, roots, plans, report):
    levels = [off.series(_record(ref, roots, plans), spec["field"]) for ref in spec["levels"]]
    t, (y1, y2, y4) = common_series(levels, dt=off.POLICY["common_grid_dt"])
    rows = {}
    for w, (lo, hi) in WINDOWS.items():
        d12, d24 = _norm(t, y2-y1, lo, hi), _norm(t, y4-y2, lo, hi)
        est = report["sequences"][spec["estimate_sequence"]]["refinement"][w].get("fine_error_estimate_L2")
        ratio = d12/d24 if d12 and d24 else None
        rows[w] = {"full_to_half_L2": d12, "half_to_quarter_L2": d24, "ratio": ratio,
                   "observed_temporal_order": math.log2(ratio) if ratio and ratio > 0 else None,
                   "half_to_quarter_fraction_of_fine_estimate": d24/est if d24 is not None and est else None,
                   "full_to_half_fraction_of_fine_estimate": d12/est if d12 is not None and est else None,
                   "expected": "ratio near 16 if the RK4 temporal error dominates"}
    return rows


def plane_dependence(spec, roots, plans, report):
    base = _record(spec["base"], roots, plans)
    out = {}
    for plane, ref in spec["planes"].items():
        rec = _record(ref, roots, plans)
        out[plane] = {}
        for field, seq in (("psi", "QH"), ("correction", "QH_CORRECTION")):
            diff = off.pair_difference(base, rec, field)
            rows = {}
            for w, row in diff["windows"].items():
                est = report["sequences"][seq]["refinement"][w].get("fine_error_estimate_L2")
                rows[w] = {**row, "fraction_of_fine_estimate": row["difference_L2"]/est
                           if row.get("state") == "MEASURED" and est else None}
            out[plane][field] = rows
    return out


def interaction(spec, roots, plans):
    records = [off.series(_record(spec[k], roots, plans), spec["field"]) for k in ("base", "a", "b", "ab")]
    t, (y0, ya, yb, yab) = common_series(records, dt=off.POLICY["common_grid_dt"])
    rows = {}
    for w, (lo, hi) in WINDOWS.items():
        da, db, dab = _norm(t, ya-y0, lo, hi), _norm(t, yb-y0, lo, hi), _norm(t, yab-y0, lo, hi)
        residual = _norm(t, yab-ya-yb+y0, lo, hi)
        scale = max(v for v in (da, db) if v is not None) if (da or db) else None
        rows[w] = {"a_L2": da, "b_L2": db, "both_L2": dab, "interaction_L2": residual,
                   "interaction_fraction_of_largest_single": residual/scale if scale else None,
                   "scope": "|both - a - b + base|: departure from additive single-factor effects"}
    return rows


def make_report(roots, supplement_root=None):
    plans = {k: _stored_plan(v) for k, v in roots.items()}
    reports = {k: off.make_report(plans[k], Path(v), None, [], mode="historical") for k, v in roots.items()}
    out = {"report_version": REPORT_VERSION, "utc": datetime.now(timezone.utc).isoformat(),
           "roots": {k: Path(v).name for k, v in roots.items()},
           "plan_sha256": {k: off.plan_sha256(p) for k, p in plans.items()},
           "experiments": {k: p["experiment"] for k, p in plans.items()},
           "points": {k: {p: {"outcome": r["points"][p]["outcome"], "reasons": r["points"][p]["reasons"],
                              "windows": {w: v["state"] for w, v in r["points"][p].get("windows", {}).items()}}
                          for p in r["points"]} for k, r in reports.items()},
           "unavailable": {k: r["unavailable"] for k, r in reports.items()},
           "producer_provenance": {k: [{q: p["provenance"].get(q) for q in ("git_commit", "git_dirty", "python",
                                                                          "platform", "plan_sha256")}
                                       for p in r["producer_provenance"].values()] for k, r in reports.items()},
           "experiment_dependence": experiment_dependence(roots, reports, plans),
           "comparison_grid": {k: comparison_grid_sensitivity(Path(v), plans[k]) for k, v in roots.items()},
           "policy": {"resolution_fraction": RESOLUTION_FRACTION, "grid_sensitivity_fraction":
                      GRID_SENSITIVITY_FRACTION, "differences_are_error_bounds": False}}
    if supplement_root is not None:
        supplement = _stored_plan(supplement_root)
        if supplement["base_plan_sha256"] != out["plan_sha256"]["E0"]:
            raise ValueError("supplement was declared for a different E0 plan")
        all_roots = {"base": Path(roots["E0"]), "supplement": Path(supplement_root)}
        all_plans = {"base": plans["E0"], "supplement": supplement}
        comps = supplement["comparisons"]
        out["supplement"] = {
            "plan_sha256": off.plan_sha256(supplement),
            "record_checks": {label: off.failed_checks(_load(supplement_root, supplement, label))
                              for label in supplement["jobs"]},
            "time_step_convergence": {k: time_step_convergence(v, all_roots, all_plans, reports["E0"])
                                      for k, v in comps["time_step_convergence"].items()},
            "plane_dependence": plane_dependence(comps["plane_dependence"], all_roots, all_plans, reports["E0"]),
            "interactions": {k: interaction(v, all_roots, all_plans) for k, v in comps["interactions"].items()}}
    return out, reports


# ----------------------------------------------------------------------------- commands

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True, help="E0 plan-0.3 root")
    p = sub.add_parser("run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=float, default=140.)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("verify")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report")
    for name in ("E0", "S3", "S5", "O120"):
        p.add_argument("--"+name.lower(), type=Path, required=name == "E0")
    p.add_argument("--supplement", type=Path)
    p.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        root = off._root(args.output)
        plan = build_supplement(_stored_plan(args.base))
        path = root/"plan.json"
        if path.exists():
            if canonical_json(json.loads(path.read_text(encoding="utf8"))) != canonical_json(plan):
                raise ValueError("Output directory holds a different supplement plan")
        else:
            off.write_json(path, plan, root)
        print(f"Plan {PLAN_VERSION}: {len(plan['jobs'])} jobs; sha256 {off.plan_sha256(plan)}")
        return 0
    if args.command == "verify":
        root = Path(args.output).resolve()
        plan = _stored_plan(root)
        # Checked readback with the producer provenance reported as stored (the analyzer's code may
        # differ from the producer's); file hashes make the manifest usable for later integrity checks.
        checked = off.verify_archive(root, plan, analyzer_provenance=off.make_provenance(plan))
        readback = checked["readback"]
        files = {p.relative_to(root).as_posix(): off._sha256_file(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and not p.name.endswith(".tmp")}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        manifest = {"utc": stamp, "root_name": root.name, "plan_sha256": off.plan_sha256(plan),
                    "readback": readback, "files_sha256": files,
                    "producer_provenance": checked["producer_provenance"],
                    "analyzer_provenance": checked["analyzer_provenance"],
                    "resume_compatible": checked["resume_compatible"]}
        path = root/"manifests"/f"{stamp}-{off.uuid.uuid4().hex[:8]}.json"
        off.write_json(path, manifest, root)
        bad = [k for k, v in readback.items() if v["archive_state"] != "VERIFIED"]
        print(f"Manifest: {path}; verified {len(readback)-len(bad)}/{len(readback)}; files hashed {len(files)}")
        return 1 if bad else 0
    if args.command == "run":
        root = off._root(args.output)
        plan = _stored_plan(root)
        if plan.get("plan_version") != PLAN_VERSION:
            raise ValueError("not a robustness supplement root")
        return off.execute_plan(root, plan, off.make_provenance(plan), budget=args.budget, workers=args.workers)
    roots = {k: getattr(args, k.lower()) for k in ("E0", "S3", "S5", "O120") if getattr(args, k.lower())}
    report, _ = make_report(roots, args.supplement)
    out = Path(args.report_output).resolve()
    for root in list(roots.values())+([args.supplement] if args.supplement else []):
        if out == Path(root).resolve() or Path(root).resolve() in out.parents:
            raise ValueError("the report is written outside every analysed root")
    path = off._write_external(out, "robustness-report", report)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
