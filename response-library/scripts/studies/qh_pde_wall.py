# usage: python scripts/studies/qh_pde_wall.py EplM h SURFACES BLOCK_ROOT OUT_DIR
#   e.g. 1e6 0.0125 -40,-35 blocks/B2 out/  (independent causal PDE with the wall at a near-horizon plane
#   against the stored FD correction; minutes per run at h = 0.0125)
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eco_library import quantum_horizon as qh, offfiducial as off
from eco_library.convergence import common_series
from scipy.integrate import trapezoid
E = float(sys.argv[1]); h = float(sys.argv[2]); surfaces = [float(s) for s in sys.argv[3].split(",")]
root = Path("runs")/sys.argv[4]
plan = json.loads((root/"plan.json").read_text())
fine = off.load_job(root, plan["jobs"]["QH_fine"], None, mode="historical")
fd = {"t": np.asarray(fine["t"]), "psi": np.asarray(fine["result"]["response"]["correction"])}
out = {}
for s in surfaces:
    t0 = time.time()
    p = qh.independent_td(dx=h, surface=s, EplM=E, t_end=520.)
    pde = {"t": p["time"], "psi": p["correction"]}
    t, (a, b) = common_series([pde, fd], dt=.07)
    row = {"neg": p["negative_lag_kernel_L2_fraction"], "seconds": time.time()-t0}
    for w, (lo, hi) in {"prompt": (100, 190), "late": (190, 500), "full": (100, 500)}.items():
        m = (t >= lo) & (t <= hi)
        row[w] = float(np.sqrt(trapezoid((b[m]-a[m])**2, t[m]))/np.sqrt(trapezoid(a[m]**2, t[m])))
    out[s] = row
    np.save(Path(sys.argv[5])/f"qhpde_{E:g}_{h:g}_{s:g}.npy", np.vstack([p["time"], p["correction"]]))
    print("EplM %.0e h %.4f surface %.0f: neg %.2e FD-PDE rel late %.3e full %.3e prompt %.3e (%.0fs)" % (E, h, s, row["neg"], row["late"], row["full"], row["prompt"], row["seconds"]), flush=True)
