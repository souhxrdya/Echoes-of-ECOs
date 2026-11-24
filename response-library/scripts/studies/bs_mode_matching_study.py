"""BS axial pole versus matching radius (continuation in steps of 0.1 R99) with residual conditioning."""
# usage: python scripts/studies/bs_mode_matching_study.py 0.18,0.22,0.25,0.270963318550685 OUT.json
# (the points block of results/checks/bs-mode-matching-study.json; the other blocks were written by hand)
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eco_library import modes
from eco_library.boson_star import FIDUCIAL_SIGMA_C
points = [float(x) for x in sys.argv[1].split(",")]
out = sys.argv[2]
try:
    res = json.load(open(out))
except Exception:
    res = {}
catalogue = json.load(open("runs/summaries/base-catalogue.json"))
factors_up = [round(1.4 + .1*k, 2) for k in range(1, 13)]      # 1.5 .. 2.6
factors_down = [1.3, 1.2]
for sc in points:
    t0 = time.time()
    bg, meta, checks = modes.bs_background(sc)
    M = float(bg.mass)
    entry = next(e for e in catalogue["entries"] if e["model"] == "BS" and abs(e["params"]["sigma_c"]-sc) < 1e-12)
    rows = {}
    for j, row in enumerate(entry["modes"]["rows"], 1):
        w14 = complex(*row["omega"])/M          # catalogue omega is M omega
        scan = {}
        w, r, ok = modes.bs_solve(bg, w14, match_factor=1.4)
        scan["1.40"] = {"omega_mu": [w.real, w.imag], "residual": r, "ok": ok}
        for seq in (factors_up, factors_down):
            g = w
            for f in seq:
                ww, rr, ok = modes.bs_solve(bg, g, match_factor=f)
                # conditioning: |d residual / d omega| by a complex step of 1e-6 at the found root
                try:
                    d = abs(modes.bs_residual(bg, ww+1e-6, match_factor=f) - modes.bs_residual(bg, ww, match_factor=f))/1e-6
                except Exception:
                    d = None
                tail = modes.boson_runtime()["BosonAxialModeProblem"](bg, 2, match_radius=f*bg.radius_99).tail_metadata()
                scan[f"{f:.2f}"] = {"omega_mu": [ww.real, ww.imag], "M_omega": [ww.real*M, ww.imag*M], "residual": rr, "ok": ok,
                                    "residual_derivative_abs": d, "missing_mass_fraction": tail["missing_mass_fraction"],
                                    "relative_potential_jump": tail["relative_potential_jump"],
                                    "relative_light_speed_jump": tail["relative_light_speed_jump"]}
                if ok:
                    g = ww
                else:
                    break
        rows[str(j)] = dict(sorted(scan.items(), key=lambda kv: float(kv[0])))
    res[f"{sc:.12g}"] = {"mass_mu": M, "R99_mu": float(bg.radius_99), "modes": rows}
    json.dump(res, open(out, "w"), indent=1)
    print(sc, "%.0fs" % (time.time()-t0), {j: [(k, round(v["omega_mu"][0], 4), round(v["omega_mu"][1], 4), v["ok"]) for k, v in m.items()][::3] for j, m in rows.items()}, flush=True)
