"""Long-record feasibility check for WH r0/M=2.005 and its BH reference at E0 (Phase 3).

Record end T=1000 M with boundaries moved so the 8-sigma practical causal bound still
holds (x_right=620, x_left=-450, native coordinates): right return BH 2*620-55-95-32=1058 M,
WH 2*620-65-105-32=1038 M (WH native = global + 9.98), BH left return
55+95+900-32=1018 M, WH left return (native) 65+105+900-32=1038 M.  Three levels each;
refinement is assessed in the plan windows and in an extended window [500, 980] M.
Records are archived with the standard writer under the given root; usage:

  python scripts/long_record_check.py ROOT run LABEL      # one record per call
  python scripts/long_record_check.py ROOT report OUT.json
"""
import json
import sys
from pathlib import Path

from eco_library import offfiducial as off
from eco_library.convergence import WINDOWS, assess, common_series
from eco_library.schema import RunSpec

T_END, X_LEFT, X_RIGHT = 1000., -450., 620.
EXTENDED = {**WINDOWS, "extended_late": (500., 980.)}


def jobs():
    out = {}
    for model, params in (("BH", {}), ("WH", {"r0_over_M": 2.005})):
        for level, h in zip(off.LEVEL_NAMES, off.LEVELS):
            spec = RunSpec(model, params, dx=h, t_end=T_END, x_left=X_LEFT, x_right=X_RIGHT,
                           experiment=off.PRODUCTION_EXPERIMENT)
            out[f"{model}_{level}"] = off._td_job(spec, "long_record", model, "long-record feasibility")
    return out


def main(root, command, arg):
    root = Path(root)
    plan = {"plan_version": "long-record-check/0.1", "jobs": jobs()}
    if command == "run":
        job = plan["jobs"][arg]
        result = off.compute_job(arg, job, root, None, plan["jobs"])
        off.persist(root, job, result, off.make_provenance(plan))
        print(arg, "written", [k for k, v in result["checks"].items() if not v["passed"]])
        return
    report = {"plan": {k: v["spec"] for k, v in plan["jobs"].items()}, "sequences": {}}
    payloads = {k: off.load_job(root, v, None, mode="historical") for k, v in plan["jobs"].items()}
    for model, order in (("BH", 4), ("WH", 4)):
        labels = [f"{model}_{level}" for level in off.LEVEL_NAMES]
        t, ys = common_series([off.series(payloads[k], "psi") for k in labels], dt=.07)
        hs = [off.spacing(payloads[k], "actual_dx") for k in labels]
        report["sequences"][model] = {
            "failed_checks": {k: off.failed_checks(payloads[k]) for k in labels},
            "refinement": assess(t, ys, hs, expected_order=order, windows=EXTENDED)}
    t, (a, b, c, d, e, f) = common_series([off.series(payloads[k], "psi") for k in
                                           ("WH_coarse", "WH_medium", "WH_fine", "BH_coarse", "BH_medium", "BH_fine")],
                                          dt=.07)
    report["residual_refinement"] = assess(t, [a-d, b-e, c-f], [off.spacing(payloads[k], "actual_dx") for k in
                                                                ("WH_coarse", "WH_medium", "WH_fine")],
                                           expected_order=4, windows=EXTENDED)
    Path(arg).write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False)+"\n")
    for name, seq in report["sequences"].items():
        print(name, {w: (r["state"], r.get("fine_relative_error_estimate")) for w, r in seq["refinement"].items()})
    print("residual", {w: (r["state"], r.get("fine_relative_error_estimate")) for w, r in report["residual_refinement"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
