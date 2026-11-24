"""Tracked summary of a robustness report (eco_library.robustness report).

Usage: python scripts/summarize_robustness.py REPORT_JSON SUPPLEMENT_ROOT OUTPUT_JSON
"""
import hashlib
import json
import sys
from pathlib import Path


def main(report_path, supplement_root, output):
    report = json.loads(Path(report_path).read_text(encoding="utf8"))
    supplement_root = Path(supplement_root)
    manifest = sorted((supplement_root/"manifests").glob("*.json"))[-1]
    files = dict(json.loads(manifest.read_text(encoding="utf8"))["files_sha256"])
    files[manifest.relative_to(supplement_root).as_posix()] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    keep = ("state", "relative_difference", "budget_fraction_of_difference", "difference_L2", "budget_base_L2",
            "budget_other_L2")
    dependence = {x: {"retarded_shift_M": block["retarded_shift_M"], "experiment": block["experiment"],
                      "points": {p: ({obs: {w: {k: v.get(k) for k in keep} for w, v in ws.items()}
                                      for obs, ws in row["observables"].items()} if row["state"] == "MEASURED"
                                     else row) for p, row in block["points"].items()}}
                  for x, block in report["experiment_dependence"].items()}
    grid = {x: {s: {w: {"state": v["state"], "estimate_ratio_to_default": v["estimate_ratio_to_default"]}
                    for w, v in rows.items()} for s, rows in seqs.items()}
            for x, seqs in report["comparison_grid"].items()}
    out = {"summary_version": "robustness-summary/0.1",
           "report_sha256": hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
           "report_version": report["report_version"], "roots": report["roots"], "plan_sha256": report["plan_sha256"],
           "experiments": report["experiments"], "points": report["points"], "unavailable": report["unavailable"],
           "producer_provenance": report["producer_provenance"], "experiment_dependence": dependence,
           "comparison_grid": grid, "supplement": report.get("supplement"),
           "supplement_archive_sha256": dict(sorted(files.items())), "policy": report["policy"]}
    Path(output).write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    print("written", output, len(files), "supplement files")


if __name__ == "__main__":
    main(*sys.argv[1:4])
