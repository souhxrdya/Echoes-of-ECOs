"""Declared blocks of the response library, and the base catalogue assembly and checks (docs/METHOD.md).

Blocks are plan-0.3 runs at declared points (one point per model per block); block B0 is the
executed pilot root.  The catalogue is assembled from immutable archives in historical mode
and never reruns a solver.  Entry states: ACCEPTED_V1, INCONCLUSIVE (with reasons) or
REJECTED.  An entry is ACCEPTED_V1 only if its plan-0.3 decision is ACCEPTED_PILOT_SCOPE and
its observable (BH-subtracted residual; QH correction) is RESOLVED in the late and full
windows: the residual's numerical budget plus every non-subdominant control difference on the
sequences involved is at most 10% of the observable's own norm in that window.
"""
from __future__ import annotations

from .boson_star import FIDUCIAL_SIGMA_C

CATALOGUE_VERSION = "eco-response-library/1.0"
RESOLUTION_FRACTION = .1
ACCEPTANCE_WINDOWS = ("late", "full")
BLOCKS = {
    "B0": {"GS": {"compactness": .44}, "WH": {"r0_over_M": 2.005}, "BS": {"sigma_c": .22},
           "QH": {"EplM": 1e10, "gamma": 1.}},
    "B1": {"GS": {"compactness": .47}, "WH": {"r0_over_M": 2.0005}, "BS": {"sigma_c": FIDUCIAL_SIGMA_C},
           "QH": {"EplM": 1e8, "gamma": 1.}},
    "B2": {"GS": {"compactness": .40}, "WH": {"r0_over_M": 2.05}, "BS": {"sigma_c": .18},
           "QH": {"EplM": 1e6, "gamma": 1.}},
    "B3": {"GS": {"compactness": .49}, "WH": {"r0_over_M": 2.02}, "BS": {"sigma_c": .25},
           "QH": {"EplM": 1e12, "gamma": 1.}},
}
DOMAINS = {
    "GS": {"parameter": "compactness", "interval": [.40, .49],
           "points": {.40: "lower edge", .44: "interior", .47: "frozen fiducial", .49: "upper edge"}},
    "WH": {"parameter": "r0_over_M", "interval": [2.0005, 2.05],
           "points": {2.0005: "lower edge, frozen fiducial", 2.005: "interior", 2.02: "interior", 2.05: "upper edge"}},
    "BS": {"parameter": "sigma_c", "interval": [.18, FIDUCIAL_SIGMA_C],
           "points": {.18: "lower edge", .22: "interior", .25: "interior",
                      FIDUCIAL_SIGMA_C: "upper edge: mini BS I, maximum mass, stability critical point"}},
    "QH": {"parameter": "EplM", "interval": [1e6, 1e12], "fixed": {"gamma": 1.},
           "points": {1e6: "lower edge", 1e8: "frozen fiducial", 1e10: "interior", 1e12: "upper edge"}},
}


# ------------------------------------------------------------------------- assembly

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

BLOCK_ROOTS = {"B0": "blocks/B0", "B1": "blocks/B1", "B2": "blocks/B2", "B3": "blocks/B3"}
BLOCK_SUMMARIES = {b: f"block-{b}.json" for b in ("B0", "B1", "B2", "B3")}
PHYSICS_DIRS = ("transfer-modes",)
ROBUSTNESS_SUMMARY = "robustness-report-summary.json"
# BH reference of an ECO sequence: the library's own mode label (as in the report's residual entries)
# and five match items, each decided level by level by exact equality of stored record metadata.
# Nominal agreement (equal requested spacing, equal Courant factor) is reported beside the booleans
# and is never counted as a match.
LIBRARY_REFERENCE_MODE = "MATCHED_RESOLUTION_NOT_IDENTICAL_GRID"
MATCH_RULES = {
    "solver": "metadata solver_id of the two records identical",
    "grid": "identical native node set: x_left, actual_dx and Nx identical",
    "dt": "time step dt identical",
    "domain": "identical computational interval: x_left and x_right identical",
    "extraction": "identical grid and identical global and native observer coordinates",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_match(eco, bh):
    """(match, detail) of one BH reference record against one ECO record by MATCH_RULES."""
    def same(*keys):
        return all(eco.get(k) is not None and eco.get(k) == bh.get(k) for k in keys)
    grid = same("x_left", "actual_dx", "Nx")
    match = {"solver": same("solver_id"), "grid": grid, "dt": same("dt"), "domain": same("x_left", "x_right"),
             "extraction": grid and same("observer_global", "observer_native")}
    keys = ("solver_id", "dt", "actual_dx", "Nx", "x_left", "x_right", "observer_global", "observer_native")
    detail = {k: [eco.get(k), bh.get(k)] for k in keys}
    detail["dt_relative_difference"] = abs(eco["dt"]-bh["dt"])/abs(bh["dt"])
    detail["actual_dx_relative_difference"] = abs(eco["actual_dx"]-bh["actual_dx"])/abs(bh["actual_dx"])
    return match, detail


def sequence_match(eco_metadata, bh_metadata):
    """A match item holds for a sequence only if it holds at every level (coarse to fine)."""
    if not eco_metadata or len(eco_metadata) != len(bh_metadata):
        raise ValueError("matched sequences need the same nonzero number of levels")
    rows = [reference_match(e, b) for e, b in zip(eco_metadata, bh_metadata)]
    return {k: all(m[k] for m, _ in rows) for k in MATCH_RULES}, [d for _, d in rows]


def bh_reference(root, plan, point):
    """The BH reference block of a catalogue entry, derived from the stored records."""
    from . import offfiducial as off
    metadata = {s: [off.load_job(root, plan["jobs"][label], None, mode="historical")["metadata"]
                    for label in plan["sequences"][s]["labels"]] for s in (point["sequence"], point["bh_sequence"])}
    match, levels = sequence_match(metadata[point["sequence"]], metadata[point["bh_sequence"]])
    return {"sequence": point["bh_sequence"], "mode": LIBRARY_REFERENCE_MODE, "match": match,
            "match_rules": MATCH_RULES, "levels": levels}


def _components(report, sequences, window):
    total, names = 0., {}
    for name, control in report["controls"].items():
        row = control["windows"].get(window, {})
        if control["estimate_sequence"] in sequences and row.get("state") == "SYSTEMATIC_NOT_SUBDOMINANT":
            total += row["difference_L2"]
            names[name] = row["relative_difference"]
    return total, names


def observable_resolution(report, model, point):
    """Section 3 criterion 2 per window: budget B (estimate + non-subdominant controls) versus norm N."""
    seq, bh = point["sequence"], point.get("bh_sequence")
    rows = {}
    for w in report["sequences"][seq]["refinement"]:
        if model == "QH":
            ref = report["sequences"]["QH_CORRECTION"]["refinement"][w]
            estimate = ref.get("fine_error_estimate_L2") if ref.get("state") == "OBSERVED_ORDER_CONSISTENT" else None
            norm = ref.get("fine_L2")
            extra, names = _components(report, ("QH_CORRECTION",), w)
            observable = "QH correction (ECO minus internal BH stage)"
        else:
            budget = report["residuals"][seq]["paired_budget"][w]
            estimate = budget.get("error_estimate_L2") if budget.get("state") == "CONDITIONAL_ESTIMATE" else None
            norm = budget.get("fine_residual_L2")
            extra, names = _components(report, (seq, bh), w)
            observable = "BH-subtracted residual (fine ECO minus fine matched BH)"
        if estimate is None or not norm:
            rows[w] = {"state": "UNQUANTIFIED", "observable": observable, "reason": "estimate or norm missing"}
            continue
        total = estimate+extra
        rows[w] = {"state": "RESOLVED" if total <= RESOLUTION_FRACTION*norm else "UNRESOLVED",
                   "observable": observable, "norm_L2": norm, "refinement_budget_L2": estimate,
                   "non_subdominant_controls_L2": extra, "budget_fraction_of_norm": total/norm,
                   "non_subdominant_controls_relative": names}
    return rows


def _physics_index(library):
    transfers, modes = {}, {}
    for d in PHYSICS_DIRS:
        for path in sorted((library/d).glob("transfer-*.json")):
            s = json.loads(path.read_text(encoding="utf8"))
            transfers[(s["root_name"], s["model"])] = {"file": f"{d}/{path.name}", "sha256": _sha(path), "summary": s}
        for path in sorted((library/d).glob("*.json")):
            if path.name.startswith("transfer-") or path.name == "modes-v1.json":
                continue
            m = json.loads(path.read_text(encoding="utf8"))
            modes[(m["model"], float(m["value"]))] = {"file": f"{d}/{path.name}", "sha256": _sha(path), "data": m}
    return transfers, modes


def _mode_rows(entry):
    out = []
    for row in entry["data"]["modes"]:
        evidence = row.get("export_evidence", row["evidence"])
        out.append({"omega": row["omega"], "family": row["family"], "branch_id": row["branch_id"],
                    "sigma_num": row["sigma_num"], "numerical_evidence": row["evidence"], "export_evidence": evidence,
                    "continuation_accepted": row.get("continuation_accepted"),
                    "approximation_state": (row.get("tail_systematic") or {}).get("state"),
                    "bh_counterpart": None, "exhaustive": False})
    return out


def assemble(library, results):
    """Build the catalogue dictionary from block archives (historical readback) and physics outputs."""
    from . import offfiducial as off
    library, results = Path(library), Path(results)
    transfers, modes = _physics_index(library)
    robustness = json.loads((results/ROBUSTNESS_SUMMARY).read_text(encoding="utf8"))
    entries, blocks = [], {}
    for block, root_name in BLOCK_ROOTS.items():
        root = library/root_name
        plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
        summary = json.loads((results/BLOCK_SUMMARIES[block]).read_text(encoding="utf8"))
        integrity = off.integrity_check(root, summary["archive_sha256"])
        report = off.make_report(plan, root, None, [], mode="historical")
        triggered = {n: [w for w in ACCEPTANCE_WINDOWS if s["refinement"][w]["state"] in
                         ("ORDER_NOT_CONFIRMED", "AGREEMENT_AT_FLOOR")]
                     for n, s in report["sequences"].items() if n not in ("BS_PROFILE", "QH_PDE")}
        blocks[block] = {"root": root_name, "plan_sha256": off.plan_sha256(plan), "plan_version": plan["plan_version"],
                         "integrity": {k: integrity[k] for k in ("state", "checked", "matched", "mismatched", "missing")},
                         "unavailable": report["unavailable"],
                         "adaptive_refinement_triggered": {n: w for n, w in triggered.items() if w},
                         "producer_provenance": [{k: p["provenance"].get(k) for k in (
                             "git_commit", "git_dirty", "python", "numpy", "scipy", "h5py", "platform")}
                             for p in report["producer_provenance"].values()],
                         "runtime_regression": report["runtime_regression"].get("state")}
        for model, point in plan["points"].items():
            decision = report["points"][model]
            resolution = observable_resolution(report, model, point)
            reasons = list(decision["reasons"])
            if decision["outcome"] == "REJECTED":
                state = "REJECTED"
            else:
                unresolved = [w for w in ACCEPTANCE_WINDOWS if resolution[w]["state"] != "RESOLVED"]
                if unresolved:
                    reasons.append("observable not resolved in: "+", ".join(unresolved))
                state = "ACCEPTED_V1" if decision["outcome"] == "ACCEPTED_PILOT_SCOPE" and not unresolved else "INCONCLUSIVE"
            param = DOMAINS[model]["parameter"]
            value = point["params"][param]
            seqs = [point["sequence"]]+[point[k] for k in ("correction_sequence", "bh_sequence") if point.get(k)]
            records = {s: [{"label": label, "identity": plan["jobs"][label]["identity"],
                            "file": f"{root_name}/{off.output_path(Path('.'), plan['jobs'][label]).as_posix()}",
                            "sha256": summary["archive_sha256"].get(off.output_path(Path('.'), plan['jobs'][label]).as_posix())}
                           for label in plan["sequences"][s]["labels"]] for s in seqs}
            transfer = transfers.get((Path(root_name).name, model))
            mode_entry = modes.get((model, float(value)))
            entry = {
                "entry_id": f"{model.lower()}-{param}-{value:.6g}-E0",
                "model": model, "params": point["params"], "domain_role": DOMAINS[model]["points"].get(value),
                "block": block, "experiment": plan["experiment"], "experiment_fingerprint": plan["experiment_fingerprint"],
                "windows_M": plan["common_experiment"]["windows_M"], "state": state, "reasons": reasons,
                "plan_decision": decision["outcome"],
                "window_states": {w: v["state"] for w, v in decision.get("windows", {}).items()},
                "observable_resolution": resolution,
                "refinement": {s: {w: {k: report["sequences"][s]["refinement"][w].get(k) for k in
                                       ("state", "p_observed", "fine_relative_error_estimate")}
                                   for w in ACCEPTANCE_WINDOWS} for s in seqs},
                "bh_reference": bh_reference(root, plan, point) if point.get("bh_sequence") else None,
                "control_coverage": {s: {f: r["status"] for f, r in fs.items()}
                                     for s, fs in decision.get("control_coverage", {}).items()},
                "records": records,
                "transfer": None if transfer is None else {"state": transfer["summary"]["state"], "file": transfer["file"],
                                                           "sha256": transfer["sha256"],
                                                           "archive": transfer["summary"]["archive"],
                                                           "archive_sha256": transfer["summary"]["archive_sha256"],
                                                           "gate": transfer["summary"]["gate"]},
                "transfer_status": ("NOT_APPLICABLE: no sharp interface" if model == "BS" else
                                    "QH transfer is the production adapter's own (K, planes in the QH archives)"
                                    if model == "QH" else None),
                "modes": None if mode_entry is None else {"file": mode_entry["file"], "sha256": mode_entry["sha256"],
                                                          "rows": _mode_rows(mode_entry)},
                "modes_status": "PRESCRIPTION_DEPENDENT_NOT_COMPUTED" if model == "QH" else None,
                "robustness": (robustness["experiment_dependence"] and
                               {x: {obs: {w: v["state"] for w, v in ws.items()}
                                    for obs, ws in b["points"][model].items()} if isinstance(b["points"][model], dict)
                                and "state" not in b["points"][model] else b["points"][model]
                                for x, b in robustness["experiment_dependence"].items()}) if block == "B0" else
                              "E0 only: source-width and observer dependence measured for block B0 points",
                "record_length_M": float(plan["common_experiment"]["t_end"]),
                "not_covered": (["decay beyond the record end (slowest mode decay time exceeds the record)"]
                                if model == "WH" else ["echoes after the record end"] if model == "QH" else []),
            }
            entries.append(entry)
    counts = {s: sum(e["state"] == s for e in entries) for s in ("ACCEPTED_V1", "INCONCLUSIVE", "REJECTED")}
    return {"catalogue_version": CATALOGUE_VERSION, "plan": "docs/METHOD.md", "domains": {
                m: {**d, "points": {f"{k:.12g}": v for k, v in d["points"].items()}} for m, d in DOMAINS.items()},
            "blocks": blocks, "counts": counts, "entries": entries,
            "policy": {"resolution_fraction": RESOLUTION_FRACTION, "acceptance_windows": list(ACCEPTANCE_WINDOWS),
                       "control_budget_fraction": .1, "estimates_are_error_bounds": False},
            "conventions": {"field": "Q_RW axial ell=2", "units": "G=c=M_ADM=1", "time": "exp(-i omega t)",
                            "spectrum": "X(omega)=int x(t) exp(+i omega t) dt", "norm": "sqrt(trapezoid(y^2, t)) on "
                            "inclusive window masks of the dt=0.07 cubic-spline common grid"},
            "statements": {"echo_classification": "NOT_COMPUTED", "detectability": "NOT_COMPUTED",
                           "total_error_bounds": "NONE", "bh_overtone_mapping": "NONE"}}


def runtime_reference_path(catalogue_path):
    """The runtime record beside a catalogue: <name>.runtime.json."""
    p = Path(catalogue_path)
    return p.with_name(p.stem+".runtime.json")


def check(library, results, catalogue_path, runtime_reference=None, rebuild=None):
    """Independent check: integrity, decisions re-derived, coarse reproduction, the rebuild compared with
    the stored catalogue under the verification policy (byte identical in the reference runtime,
    eco_library.verification rules elsewhere), and the observable norms recomputed by a separate
    implementation.  ``rebuild`` (default ``assemble``) builds the catalogue to compare with the stored
    file; a later catalogue version passes its own assembler."""
    from . import offfiducial as off
    from . import verification as ver
    library, results = Path(library), Path(results)
    stored = json.loads(Path(catalogue_path).read_text(encoding="utf8"))
    reference = ver.reference_runtime(runtime_reference or runtime_reference_path(catalogue_path))
    fingerprint = ver.runtime_fingerprint()
    # Integrity first and on its own: a damaged archive must be reported as such even if it also makes
    # the rebuild impossible.
    integrity = {}
    for block, root_name in BLOCK_ROOTS.items():
        try:
            summary = json.loads((results/BLOCK_SUMMARIES[block]).read_text(encoding="utf8"))
            integrity[block] = off.integrity_check(library/root_name, summary["archive_sha256"])["state"]
        except Exception as exc:  # unreadable summary or root
            integrity[block] = f"ERROR: {type(exc).__name__}: {exc}"
    try:
        rebuilt = (rebuild or assemble)(library, results)
        policy = ver.rebuild_policy(stored, rebuilt, reference, ver.file_sha256(catalogue_path), fingerprint)
    except Exception as exc:  # the rebuild itself failed: never a pass
        rebuilt = None
        policy = {"policy": ver.POLICY_VERSION, "tier": "SAME_RUNTIME" if fingerprint == reference.get(
                      "reference_runtime") else "CROSS_RUNTIME", "state": "FAIL", "byte_identical": False,
                  "derived_rtol": ver.DERIVED_RTOL, "derived_differing_count": 0, "derived_max_relative": 0.,
                  "structure": [], "exact": [], "derived_beyond_tolerance": [], "derived_differing": [],
                  "decision_margin_below_tolerance": [], "rebuild_error": f"{type(exc).__name__}: {exc}"}
    out = {"rebuild_identical": policy["byte_identical"],
           "rebuild_policy": {k: policy[k] for k in (
               "policy", "tier", "state", "byte_identical", "derived_rtol", "derived_differing_count",
               "derived_max_relative", "structure", "exact", "derived_beyond_tolerance",
               "decision_margin_below_tolerance")} | {k: policy[k] for k in ("reference_problem", "rebuild_error")
                                                     if k in policy},
           "rebuild_derived_differing": policy["derived_differing"],
           "runtime": fingerprint, "reference_runtime": reference["reference_runtime"],
           "integrity": integrity, "reproduction": {}, "independent_norms": {}}
    for block, root_name in BLOCK_ROOTS.items():
        root = library/root_name
        plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
        for label in ("BH_coarse", "GS_coarse", "WH_coarse", "BS_coarse", "QH_coarse"):
            try:
                out["reproduction"][f"{block}:{label}"] = off.reproduce_job(root, plan, label)["state"]
            except Exception as exc:  # an unreadable stored record cannot be reproduced
                out["reproduction"][f"{block}:{label}"] = f"ERROR: {type(exc).__name__}: {exc}"
        for model, point in plan["points"].items():
            if model == "QH":
                continue
            try:
                eco = off.load_job(root, plan["jobs"][f"{point['sequence']}_fine"], None, mode="historical")
                bh = off.load_job(root, plan["jobs"][f"{point['bh_sequence']}_fine"], None, mode="historical")
            except Exception as exc:
                out["independent_norms"][f"{block}:{model}"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            # Separate implementation: both records linearly interpolated (no spline, no common-grid helper)
            # onto the documented window samples t_k = 0.07 k with lo <= t_k <= hi; agreement is expected to
            # the linear-interpolation accuracy, about (omega dt_native)^2/8 < 2e-4 for omega <= 1.
            entry = next(e for e in stored["entries"] if e["block"] == block and e["model"] == model)
            rows = {}
            for w in ACCEPTANCE_WINDOWS:
                lo, hi = entry["windows_M"][w]
                k = np.arange(int(np.ceil(lo/.07-1e-9)), int(np.floor(hi/.07+1e-9))+1)
                t = .07*k
                t = t[(t >= lo) & (t <= hi)]
                r = np.interp(t, eco["t"], eco["psi"])-np.interp(t, bh["t"], bh["psi"])
                norm = float(np.sqrt(trapezoid(r**2, t)))
                cat = entry["observable_resolution"][w].get("norm_L2")
                rows[w] = {"independent_norm": norm, "catalogue_norm": cat,
                           "relative_difference": abs(norm-cat)/cat if cat else None}
            out["independent_norms"][f"{block}:{model}"] = rows
    rows = [r for v in out["independent_norms"].values() if "error" not in v for r in v.values()]
    errors = [k for k, v in out["independent_norms"].items() if "error" in v]
    worst = max((r["relative_difference"] for r in rows), default=float("inf")) if not errors else float("inf")
    out["independent_norm_max_relative_difference"] = worst if worst != float("inf") else None
    worst = worst if worst is not None else float("inf")
    out["state"] = ("PASS" if out["rebuild_policy"]["state"] == "PASS" and set(out["integrity"].values()) == {"PASS"}
                    and set(out["reproduction"].values()) <= {"BITWISE_IDENTICAL", "WITHIN_ROUNDOFF_BOUND"}
                    and worst < 1e-3 else "FAIL")
    return out


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("assemble", "check"))
    parser.add_argument("--library", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("runs/summaries"))
    parser.add_argument("--catalogue", type=Path, default=Path("runs/summaries/base-catalogue.json"))
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        catalogue = assemble(args.library, args.results)
        args.catalogue.write_text(json.dumps(catalogue, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
        print(json.dumps(catalogue["counts"]), {e["entry_id"]: e["state"] for e in catalogue["entries"]})
        return 0
    result = check(args.library, args.results, args.catalogue)
    text = json.dumps(result, indent=1, sort_keys=True, allow_nan=False)+"\n"
    if args.check_output:
        args.check_output.write_text(text, encoding="utf8")
    print(json.dumps({k: result[k] for k in ("state", "rebuild_identical", "integrity",
                                             "independent_norm_max_relative_difference")}),
          json.dumps({k: result["rebuild_policy"][k] for k in ("tier", "state", "derived_differing_count",
                                                                "derived_max_relative")}),
          sorted(set(result["reproduction"].values())), file=sys.stderr)
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
