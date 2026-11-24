"""Record-length requirements per catalogue entry (results/checks/record-length-requirements.json):
t_arrival + max(5 round trips, 5 slowest decay times), TRANSFER_AND_MODES.md section 4.

  python scripts/studies/record_length_requirements.py --output OUT.json
"""
import argparse, json, numpy as np
from scipy.optimize import minimize_scalar
V = lambda r: (1-2/r)*(6/r**2-6/r**3)
rp = minimize_scalar(lambda r: -V(r), bounds=(2.5, 5), method="bounded").x
rs = lambda r: r+2*np.log(r/2-1)
t_arr = 55+95-2*rs(rp)
cat = json.load(open("runs/summaries/base-catalogue.json"))
rows = []
for e in cat["entries"]:
    m, p = e["model"], e["params"]; v = list(p.values())[0]
    tau = rt = None; basis = ""
    if m in ("GS", "WH"):
        im = min(abs(r["omega"][1]) for r in e["modes"]["rows"] if r["numerical_evidence"] == "VERIFIED_BY_VARIATION")
        tau = 1/im; basis = "slowest verified pole (catalogue v1.0 modes; not exhaustive)"
        if m == "WH": rt = 2*(rs(rp)-rs(v))
    elif m == "QH":
        rt = 8*np.log(p["EplM"]/.4); basis = "round trip 8 ln(E_Pl M/omega) at omega 0.4 (TRANSFER_AND_MODES.md); no decay time (modes not computed)"
    else:
        basis = "stellar response decays to ~1e-12 by 450 M (GATE_AUDIT.md); modes withdrawn"
    need = t_arr+max(5*rt if rt else 0, 5*tau if tau else 0) if (rt or tau) else None
    rows.append({"model": m, "params": p, "state_v1_0": e["state"], "slowest_decay_time_M": tau, "round_trip_M": rt,
                 "record_needed_M_declared_rule": need, "decay_times_in_520M": None if tau is None else (520-t_arr)/tau,
                 "echo_arrivals_M": None if m != "QH" else [t_arr+n*rt for n in (1, 2, 3)], "basis": basis})
    print(m, v, e["state"], "tau %s rt %s need %s cover %s" % (None if tau is None else round(tau, 1), None if rt is None else round(rt, 1), None if need is None else round(need), None if tau is None else round((520-t_arr)/tau, 2)), "" if m != "QH" else [round(x) for x in rows[-1]["echo_arrivals_M"]])
out = {"t_arrival_M": t_arr, "r_peak_M": rp, "rule": "record end >= t_arrival + max(5 round trips, 5 slowest decay times) (TRANSFER_AND_MODES.md Section 4)",
       "single_slice_reach_M": {"two_boundary_TD_dx_0.05": 1000, "QH_two_stage_TD": 600}, "rows": rows}
parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
out = {"summary_version": "record-length-requirements/1",
       "source": "runs/summaries/base-catalogue.json (v1.0 modes) and TRANSFER_AND_MODES.md Section 4", **out}
open(args.output, "w").write(json.dumps(json.loads(json.dumps(out, default=float)), indent=1, sort_keys=True, default=float)+"\n")
print("t_arrival", round(t_arr, 2), "r_peak", round(rp, 4))
