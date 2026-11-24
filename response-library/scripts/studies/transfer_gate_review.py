"""Transfer-gate review numbers (results/checks/transfer-gate-review.json, docs/VALIDATION.md) from
the stored gate entries of catalogue 1.0 (copied there from the transfer summaries in runs).

The review was first computed inline; this script recomputes every number of its ``points`` block from
the E0 gates the catalogue references and compares them with the published file (exact equality: only
divisions and sums of stored numbers).  Writing this script exposed a defect of the inline version: the
all-window block of WH 2.005 had been read from the O120 transfer summary (observer r* 120) instead of
the E0 one; ``--write`` regenerates the file with the error recorded beside the corrected numbers.

  python scripts/studies/transfer_gate_review.py [--published results/checks/transfer-gate-review.json] [--write]
"""
import argparse
import json
import sys
from pathlib import Path


def review(catalogue):
    points = {}
    for e in catalogue["entries"]:
        if e["model"] not in ("GS", "WH"):
            continue
        gate = e["transfer"]["gate"]
        late_full = {}
        for w in ("late", "full"):
            g = dict(gate[w])
            g["ratio_to_combined_budget"] = g["fd_td_relative"]/(g["td_budget_relative"]+g["fd_variant_spread_relative"]
                                                                 + g["bh_monitor_discrepancy_relative"])
            late_full[w] = g
        windows = {w: {"fd_td_relative": g["fd_td_relative"], "residual_budget_relative": g["td_budget_relative"],
                       "ratio": g["fd_td_relative"]/g["td_budget_relative"]} for w, g in gate.items()}
        points[e["entry_id"]] = {"gate_late_full": late_full, "windows": windows}
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue", type=Path, default=Path("runs/summaries/base-catalogue.json"))
    parser.add_argument("--published", type=Path, default=Path("results/checks/transfer-gate-review.json"))
    parser.add_argument("--write", action="store_true", help="rewrite the published file from the E0 gates")
    args = parser.parse_args(argv)
    points = review(json.loads(args.catalogue.read_text(encoding="utf8")))
    published = json.loads(args.published.read_text(encoding="utf8"))
    if args.write and points != published["points"]:
        ratios = {m: [w["ratio"] for k, p in points.items() if k.startswith(m) for w in p["windows"].values()]
                  for m in ("gs", "wh")}
        wrong = {k: {"windows": v["windows"]} for k, v in published["points"].items() if v != points[k]}
        corrected = {**{k: v for k, v in published.items() if k != "points"}, "points": points,
                     "gs_ratio_range": [min(ratios["gs"]), max(ratios["gs"])],
                     "wh_ratio_range": [min(ratios["wh"]), max(ratios["wh"])],
                     "max_ratio_to_residual_budget_all_windows": max(ratios["gs"]+ratios["wh"]),
                     "correction": {
                         "found_by": "scripts/studies/transfer_gate_review.py (recomputation from the catalogue's E0 gates)",
                         "defect": "the all-window block of the entries below had been read from a transfer summary of "
                                   "another experiment (WH 2.005: transfer-wh-6882ed09131ed802, robustness-o120, observer "
                                   "r* 120) instead of the E0 summary the catalogue references (transfer-wh-db07391e5fcfa9bb)",
                         "previous_values": wrong,
                         "late_full_gate_block_affected": False}}
        args.published.write_text(json.dumps(corrected, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
        published = corrected
    same = points == published["points"]
    ratios = [g["ratio_to_combined_budget"] for p in points.values() for g in p["gate_late_full"].values()]
    print(json.dumps({"points_identical_to_published": same, "max_ratio_late_full": max(ratios),
                      "max_ratio_all_windows_published": published["max_ratio_to_residual_budget_all_windows"]}))
    return 0 if same else 2


if __name__ == "__main__":
    raise SystemExit(main())
