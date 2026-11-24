"""Per-entry waveform audit from stored records: local refinement orders over time, boundary margins,
residual structure; figures per model."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# usage: python scripts/studies/waveform_audit.py GS,WH,BS,QH OUT_DIR  (as run: one model per call, merged)
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eco_library import offfiducial as off
from eco_library.catalogue_base import BLOCK_ROOTS
from eco_library.convergence import common_series
cat = json.load(open("runs/summaries/base-catalogue.json"))
out = {}
models = sys.argv[1].split(",")
for model in models:
    entries = [e for e in cat["entries"] if e["model"] == model]
    fig, axes = plt.subplots(len(entries), 1, figsize=(10, 3.2*len(entries)), sharex=True)
    for ax, e in zip(axes, entries):
        root = Path("runs")/BLOCK_ROOTS[e["block"]]
        plan = json.loads((root/"plan.json").read_text())
        point = plan["points"][model]
        seq = point["sequence"]
        labels = plan["sequences"][seq]["labels"]
        P = {l: off.load_job(root, plan["jobs"][l], None, mode="historical") for l in labels}
        field = "correction" if model == "QH" else "psi"
        S = [off.series(P[l], field) for l in labels]
        margins = {l: P[l]["metadata"].get("boundary_margin_M") for l in labels}
        bhS = None
        if point.get("bh_sequence"):
            bl = plan["sequences"][point["bh_sequence"]]["labels"]
            B = {l: off.load_job(root, plan["jobs"][l], None, mode="historical") for l in bl}
            bhS = [off.series(B[l], "psi") for l in bl]
            margins.update({l: B[l]["metadata"].get("boundary_margin_M") for l in bl})
        t, ys = common_series(S + (bhS or []), dt=.07)
        c, m, f = ys[:3]
        res = None if bhS is None else [ys[k]-ys[3+k] for k in range(3)]
        d1, d2 = np.abs(c-m), np.abs(m-f)
        # local order from window-sliding L2 norms (10 M windows) of the two differences
        win = int(round(10/.07)); ker = np.ones(win)
        n1 = np.sqrt(np.convolve(d1**2, ker, "same")); n2 = np.sqrt(np.convolve(d2**2, ker, "same"))
        ok = (n2 > 1e-13*np.sqrt(np.convolve(f**2, ker, "same")).max()) & (t >= 100) & (t <= 500)
        p_local = np.full(t.shape, np.nan); p_local[ok] = np.log2(n1[ok]/n2[ok])
        order = off.DECLARED_ORDER[model]
        late = ok & (t >= 190)
        weight = n2**2
        frac_out = float(np.sum(weight[late & (np.abs(p_local-order) > .5)])/np.sum(weight[late])) if late.any() else None
        row = {"declared_order": order, "boundary_margin_M_min": float(min(v for v in margins.values() if v is not None)),
               "local_order_late_weighted_median": float(np.nanmedian(p_local[late])) if late.any() else None,
               "local_order_late_p05_p95": [float(np.nanpercentile(p_local[late], 5)), float(np.nanpercentile(p_local[late], 95))] if late.any() else None,
               "late_weight_fraction_outside_band": frac_out,
               "peak_abs_fine": float(np.max(np.abs(f))), "t_end_common": float(t[-1])}
        if res is not None:
            rr = res[2]
            row["late_residual_over_total"] = float(np.sqrt(np.sum(rr[(t>=190)&(t<=500)]**2)/np.sum(f[(t>=190)&(t<=500)]**2)))
        out[e["entry_id"]] = row
        ax.semilogy(t, np.abs(f)+1e-18, lw=.8, label=f"{model} fine" + (" correction" if model == "QH" else ""))
        if bhS is not None:
            ax.semilogy(t, np.abs(ys[5])+1e-18, lw=.6, label="BH fine", alpha=.7)
            ax.semilogy(t, np.abs(res[2])+1e-18, lw=.8, label="ECO - BH (fine)")
        ax.semilogy(t, d1+1e-18, lw=.5, label="|coarse-medium|"); ax.semilogy(t, d2+1e-18, lw=.5, label="|medium-fine|")
        for x in (100, 150, 190, 500): ax.axvline(x, color="k", lw=.4, ls=":")
        ax.set_ylim(1e-14, 2); ax.set_title(f"{e['entry_id']} ({e['state']}); local order late median {row['local_order_late_weighted_median']:.2f}, margin {row['boundary_margin_M_min']:.0f} M", fontsize=9)
        ax.legend(fontsize=6, ncol=5, loc="lower left")
    axes[-1].set_xlabel("t / M")
    fig.tight_layout(); fig.savefig(OUT/f"{model}.png", dpi=80)
    print(model, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items() if k.startswith(model.lower())}))
prev = {}
try: prev = json.load(open(OUT/"waveform-audit.json"))
except Exception: pass
prev.update(out)
json.dump(prev, open(OUT/"waveform-audit.json", "w"), indent=1)
