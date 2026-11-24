"""Tracked summary of one executed off-fiducial plan (0.3 or later) for publication.

Usage: python scripts/summarize_offfiducial_run.py ROOT REPORT MANIFEST OUTPUT_JSON

The summary records the archive's file hashes (``archive_sha256``, usable as the
immutable manifest for ``verify --mode historical``), the plan identity and experiment,
point outcomes, control coverage, refinement and control numbers per window and the
producer provenance.  It adds no physics or acceptance logic of its own.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root, report_path, manifest_path, output):
    root = Path(root)
    report = json.loads(Path(report_path).read_text(encoding="utf8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf8"))
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    files = dict(manifest["files_sha256"])
    rel = lambda p: Path(p).resolve().relative_to(root.resolve()).as_posix()
    files[rel(manifest_path)] = sha(manifest_path)
    files[rel(report_path)] = sha(report_path)
    keep = ("state", "p_observed", "fine_error_estimate_L2", "fine_relative_error_estimate")
    sequences = {n: {"state": s["state"], "refinement_kind": s.get("refinement_kind"), "actual_h": s.get("actual_h"),
                     "windows": {w: {k: v.get(k) for k in keep} for w, v in s["refinement"].items()},
                     **({"fixed_background": s["fixed_background"]["identical"]} if "fixed_background" in s else {})}
                 for n, s in report["sequences"].items()}
    controls = {n: {"state": c["state"], "factor_key": c.get("factor_key"), "base": c["base"],
                    "variant": c["variant"], "field": c["field"], "estimate_sequence": c["estimate_sequence"],
                    "windows": {w: {k: v.get(k) for k in ("state", "difference_L2", "relative_difference",
                                                          "fraction_of_conditional_refinement_estimate")}
                                for w, v in c["windows"].items()}}
                for n, c in report["controls"].items()}
    residuals = {n: {w: {k: v.get(k) for k in ("state", "relative_error_estimate", "fine_residual_L2",
                                               "error_estimate_L2")} for w, v in r["paired_budget"].items()}
                 for n, r in report["residuals"].items()}
    points = {n: {k: p.get(k) for k in ("outcome", "reasons", "params", "fiducial", "model")} |
              {"windows": {w: {k: v.get(k) for k in ("state", "eco_relative_estimate", "bh_relative_estimate",
                                                      "residual_relative_budget", "correction_relative_estimate",
                                                      "non_subdominant_controls_relative_difference")}
                           for w, v in p.get("windows", {}).items()},
                 "missing_coverage": [f"{s}:{f}" for s, fs in p.get("control_coverage", {}).items()
                                      for f, row in fs.items() if row["status"] == "MISSING"]}
              for n, p in report["points"].items()}
    producers = [{"labels": len(p["labels"]), **{k: p["provenance"].get(k) for k in (
        "git_commit", "git_dirty", "python", "numpy", "scipy", "h5py", "platform", "plan_sha256", "cpu_count")}}
        for p in report["producer_provenance"].values()]
    summary = {"summary_version": "offfiducial-run-summary/0.1", "root_name": root.name,
               "plan_version": plan["plan_version"], "plan_sha256": report["plan_sha256"],
               "experiment": plan.get("experiment"), "experiment_fingerprint": plan.get("experiment_fingerprint"),
               "windows_M": plan.get("common_experiment", {}).get("windows_M"),
               "report_file": rel(report_path), "manifest_file": rel(manifest_path),
               "archive_sha256": dict(sorted(files.items())),
               "jobs": len(plan["jobs"]), "unavailable": report["unavailable"],
               "job_failed_checks": {k: v for k, v in report["job_failed_checks"].items() if v},
               "producer_provenance": producers, "points": points, "sequences": sequences,
               "residual_budgets": residuals, "controls": controls,
               "control_coverage": report["control_coverage"],
               "qh_independent_gate": report["qh_independent"]["gate"],
               "runtime_regression": report["runtime_regression"],
               "bs_background": {"turning_point": (report["bs_background"]["sequence"] or {}).get("turning_point"),
                                 "table_benchmark": {n: {q: {k: v.get(k) for k in ("passed", "table", "central")}
                                                         for q, v in c["comparison"].items()}
                                                     for n, c in ((report["bs_background"]["table_benchmark"] or {})
                                                                  .get("configurations", {}).items())}},
               "statements": report["statements"]}
    Path(output).write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps({n: p["outcome"] for n, p in points.items()}), len(files), "files")


if __name__ == "__main__":
    main(*sys.argv[1:5])
