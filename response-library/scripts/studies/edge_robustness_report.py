"""Edge-robustness/0.1 outcome from the stored roots (read-only): the script that produced
results/checks/edge-robustness-o120.json.  As run, it read the per-file SHA-256 manifest written when the
roots were copied; here the manifest is the published summary itself (--manifest), so a rerun checks the
roots against the published hashes and must reproduce the summary exactly.

  python scripts/studies/edge_robustness_report.py --manifest results/checks/edge-robustness-o120.json --output OUT.json
"""
import argparse, sys, json, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.integrate import trapezoid
from eco_library import offfiducial as off
from eco_library.convergence import common_series, assess
from eco_library import robustness as rb
LIB = Path("runs")
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
scratch_hashes = {name: r["archive_sha256"] for name, r in json.loads(args.manifest.read_text())["roots"].items()}
out = {"summary_version": "edge-robustness-summary/1", "plan_version": "edge-robustness/0.1",
       "plan_file": "docs/VALIDATION.md", "plan_commit": "629d7a6", "roots": {}, "points": {},
       "rule": {"MEASURED": "both O120 sequences OBSERVED_ORDER_CONSISTENT in the window (declared order, band 0.5) "
                            "and paired budget <= 0.1 x BH-subtracted residual norm", "otherwise": "UNQUANTIFIED"},
       "statements": {"catalogue_state_changes": "NONE", "observer_dependence_threshold": "NONE (physics, reported)",
                      "source_width_dependence": "measured at the four B0 points only (ROBUSTNESS_RESULTS.md)",
                      "source_position_dependence": "NOT_STUDIED"}}
for model, b in (("GS", "b3"), ("WH", "b1")):
    name = f"edge-o120-{b}"; root = LIB/"studies"/name
    plan = json.loads((root/"plan.json").read_text())
    point = plan["points"][model]; seq, bseq = point["sequence"], point["bh_sequence"]
    labels = plan["sequences"][seq]["labels"]+plan["sequences"][bseq]["labels"]
    ver = off.verify_archive(root, plan, manifest=scratch_hashes[name], labels=labels,
                             analyzer_provenance=off.analyzer_provenance() if hasattr(off, "analyzer_provenance") else None)
    out["roots"][name] = {"plan_sha256": off.plan_sha256(plan), "archive_sha256": scratch_hashes[name],
                          "integrity_of_mounted_copy": {k: ver["integrity"][k] for k in ("state", "checked", "matched", "mismatched", "missing", "unlisted")},
                          "readback": ver["readback"], "executed_labels": labels,
                          "not_executed": sorted(set(plan["jobs"])-set(labels)),
                          "experiment": plan["experiment"], "experiment_fingerprint": plan["experiment_fingerprint"]}
    windows = {w: tuple(v) for w, v in plan["sequences"][seq]["windows"].items()}
    E = [off.load_job(root, plan["jobs"][l], None, mode="historical") for l in plan["sequences"][seq]["labels"]]
    B = [off.load_job(root, plan["jobs"][l], None, mode="historical") for l in plan["sequences"][bseq]["labels"]]
    t, ys = common_series([off.series(p, "psi") for p in E+B], dt=.07)
    ae = assess(t, ys[:3], np.array([p["metadata"]["actual_dx"] for p in E]), expected_order=off.DECLARED_ORDER[model], windows=windows)
    ab = assess(t, ys[3:], np.array([p["metadata"]["actual_dx"] for p in B]), expected_order=4, windows=windows)
    r0 = LIB/f"blocks/{b.upper()}"; p0 = json.loads((r0/"plan.json").read_text())
    e0 = off.load_job(r0, p0["jobs"][f"{seq}_fine"], None, mode="historical"); b0 = off.load_job(r0, p0["jobs"][f"{bseq}_fine"], None, mode="historical")
    t0, (x0, y0) = common_series([off.series(e0, "psi"), off.series(b0, "psi")], dt=.07)
    w0 = {w: tuple(v) for w, v in p0["sequences"][seq]["windows"].items()}
    rows = {}
    for w in ("prompt", "ringdown", "late", "full"):
        lo, hi = windows[w]; m = (t >= lo) & (t <= hi)
        norm = float(np.sqrt(trapezoid((ys[2]-ys[5])[m]**2, t[m])))
        ea, eb = ae[w], ab[w]
        fe, fb = ea.get("fine_error_estimate_L2"), eb.get("fine_error_estimate_L2")
        budget = None if fe is None or fb is None else float(fe+fb)
        lo0, hi0 = w0[w]; m0 = (t0 >= lo0) & (t0 <= hi0)
        norm0 = float(np.sqrt(trapezoid((x0[m0]-y0[m0])**2, t0[m0])))
        ok = ea["state"] == eb["state"] == "OBSERVED_ORDER_CONSISTENT" and budget is not None and budget <= .1*norm
        rows[w] = {"window_O120": [lo, hi], "window_E0": [lo0, hi0], "eco_state": ea["state"], "eco_p_observed": ea.get("p_observed"),
                   "bh_state": eb["state"], "bh_p_observed": eb.get("p_observed"), "paired_budget_L2": budget,
                   "residual_norm_O120": norm, "budget_fraction": None if budget is None else budget/norm,
                   "residual_norm_E0_stored_block": norm0, "relative_change_E0_to_O120": (norm-norm0)/norm0,
                   "state": "MEASURED" if ok else "UNQUANTIFIED", "predeclared": w in ("late", "full")}
        print(model, b, w, rows[w]["state"], rows[w]["predeclared"], "p %.3f/%.3f frac %s change %+.5f" % (ea.get("p_observed") or -1, eb.get("p_observed") or -1, rows[w]["budget_fraction"], rows[w]["relative_change_E0_to_O120"]))
    # Supplementary (not predeclared): the interior study's Section-3 metric, residual difference at retarded time.
    shift = off.window_shift(off.as_experiment(plan["experiment"]))-off.window_shift(off.as_experiment(p0["experiment"]))
    rE0 = {"t": t0, "psi": x0-y0}; rO = rb._shifted({"t": t, "psi": ys[2]-ys[5]}, shift)
    tc, (ya, yb) = common_series([rE0, rO], dt=off.POLICY["common_grid_dt"])
    cat = {(e["model"], json.dumps(e["params"], sort_keys=True)): e for e in json.load(open("runs/summaries/base-catalogue.json"))["entries"]}
    entry = cat[(model, json.dumps(point["params"], sort_keys=True))]
    supp = {}
    for w in ("late", "full"):
        lo, hi = w0[w]; d, n = rb._norm(tc, yb-ya, lo, hi), rb._norm(tc, ya, lo, hi)
        be0 = entry["observable_resolution"][w]["refinement_budget_L2"]; bo = rows[w]["paired_budget_L2"]
        supp[w] = {"difference_L2": d, "E0_residual_L2": n, "relative_difference": d/n,
                   "E0_refinement_budget_L2_catalogue": be0, "O120_paired_budget_L2": bo,
                   "refinement_budgets_fraction_of_difference": (be0+bo)/d}
        print(model, w, "retarded-time relative difference %.4f budgets/diff %.2e" % (d/n, (be0+bo)/d))
    out["points"][f"{model} {list(point['params'])[0]}={list(point['params'].values())[0]}"] = {
        "root": name, "E0_reference_root": r0.name, "params": point["params"], "sequence": seq, "bh_sequence": bseq,
        "producer_provenance": {k: E[-1]["provenance"].get(k) for k in ("git_commit", "git_dirty", "python", "numpy", "scipy", "h5py", "platform")},
        "record_ends": [p["metadata"].get("record_end") or float(off.series(p, "psi")[0][-1]) for p in E+B], "windows": rows,
        "supplementary_retarded_time_difference": {"predeclared": False, "retarded_shift_M": shift, "definition": "ROBUSTNESS_RESULTS.md Section 3 (residual at t-25 vs E0 record, E0 windows)", "windows": supp}}
    print(name, ver["integrity"]["state"], ver["integrity"]["matched"], {k: v.get("record_check_state") for k, v in ver["readback"].items()})
args.output.write_text(json.dumps(out, indent=1, sort_keys=True, default=float)+"\n")
