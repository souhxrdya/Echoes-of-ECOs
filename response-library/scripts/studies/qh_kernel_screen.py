# usage: python scripts/studies/qh_kernel_screen.py OUT.json  (kernel negative-lag fractions, docs/VALIDATION.md)
import sys, numpy as np, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eco_library import quantum_horizon as qh
rt = qh._runtime(None)
def neg_fraction(EplM, surface=-40., h=.0125, nt=None, factor=1):
    nt = nt or int(np.ceil(520./h))+1
    nker = (1 << (max(32*nt, 32768)-1).bit_length())*factor
    w = 2*np.pi*np.fft.rfftfreq(nker, h)
    coeff = rt["quantum_local_reflection"](w, surface, kappa=.25, EplM=EplM, gamma=1.)
    kernel = np.fft.irfft(np.conj(coeff), nker)
    neg = float(np.linalg.norm(kernel[nker//2+1:])/np.linalg.norm(kernel))
    t = h*np.arange(nker//2+1)
    peak = float(t[np.argmax(np.abs(kernel[:nker//2+1]))])
    return neg, peak, nker
if __name__ == '__main__':
    rows = {}
    for E in (1e6, 3e6, 1e7, 3e7, 1e8, 1e10, 1e12):
        n1, p1, k1 = neg_fraction(E)
        n2, _, k2 = neg_fraction(E, factor=4)
        n3, _, _ = neg_fraction(E, h=.05)
        n4, p4, _ = neg_fraction(E, surface=-60.)
        rows[f"{E:g}"] = {"neg_h0.0125": n1, "neg_nfft_x4": n2, "neg_h0.05": n3, "kernel_peak_lag_M": p1, "neg_surface_-60": n4, "peak_surface_-60": p4}
        print("EplM %.0e: neg %.3e (nfft x4: %.3e, h=.05: %.3e) peak lag %.1f M | surface -60: neg %.3e peak %.1f" % (E, n1, n2, n3, p1, n4, p4), flush=True)
    v = rt["rw_potential"](rt["r_from_schwarzschild_rstar"](np.array([-60., -50., -40., -35., -30., -20.])), 2, 1.)
    print("V at x=-60,-50,-40,-35,-30,-20:", ["%.1e" % x for x in v])
    for s in (-40., -38., -36., -35.):   # plane scan at E_Pl M = 1e6 (as run in a second script)
        out = [neg_fraction(1e6, surface=s, h=h) for h in (.05, .025, .0125)]
        v = float(rt["rw_potential"](rt["r_from_schwarzschild_rstar"](np.array([s])), 2, 1.)[0])
        rows[f"1e6_surface_{s:g}"] = {"V": v, "neg_h0.05_0.025_0.0125": [o[0] for o in out], "peak_lag_M": out[-1][1]}
        print("surface %.0f: V=%.1e neg(h=.05,.025,.0125) = %s peak lag %.1f" % (s, v, ["%.3e" % o[0] for o in out], out[-1][1]), flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
