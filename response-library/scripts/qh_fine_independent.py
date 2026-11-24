"""Four-level fiducial QH diamond-PDE audit, retaining the actual wall kernel.

The FD reference is historical, read-only measured-source evidence. The paired
PDE never receives its K, barrier data or output waveform as an input. Norms
compare raw fields without time/phase/amplitude fits or mean subtraction.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import trapezoid

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.baseline_runtime import load_frozen_runtime
from eco_library.convergence import WINDOWS, common_series, assess
from eco_library.quantum_horizon import independent_td, _runtime
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance


def plan():
    return {"version": "qh-fine-independent/0.1", "levels": [.05, .025, .0125, .00625],
            "t_end": 520.2, "surface": -40., "observer": 95., "source_global": 55.,
            "sigma": 4., "x_right": 400., "params": {"EplM": 1e8, "gamma": 1.},
            "kernel_period_M": 26214.4, "comparison_dt": .07,
            "expected_order": 2, "order_tolerance": .5,
            "negative_lag_norm_screen": .001,
            "scope": "fixed fiducial causal-wall PDE; not a total uncertainty bound or parameter sweep",
            "cost": "four local single-thread runs; finest 83233 steps, under 1 GB estimated peak memory"}


def retained_kernel(direct):
    """Recompute the exact analytic-to-discrete kernel construction for storage.

    No fit or continuum interpolation is involved. The PDE implementation uses
    the same formula; its recorded negative-lag norm is cross-checked below.
    """
    n, h = direct["kernel_nfft"], direct["dt"]
    omega = 2*np.pi*np.fft.rfftfreq(n, h)
    rt = _runtime()
    coefficient = rt["quantum_local_reflection"](omega, direct["surface"], kappa=.25, **direct["params"])
    kernel = np.fft.irfft(np.conj(coefficient), n)
    norm_ratio = float(np.linalg.norm(kernel[n//2+1:])/np.linalg.norm(kernel))
    if norm_ratio != direct["negative_lag_kernel_L2_fraction"]:
        raise ArithmeticError("Retained kernel does not reproduce the PDE's negative-lag diagnostic")
    return {"omega": omega, "physical_local_reflectivity": coefficient,
            "full_periodic_impulse": kernel, "causal_impulse_used": kernel[:len(direct["time"])].copy(),
            "nfft": n, "sample_interval_M": h, "period_M": n*h,
            "impulse_action": "out[j]=sum(impulse[k]*incoming[j-k]); no extra dt factor",
            "construction": "irfft(conj(physical_local_reflectivity)); prefix through PDE duration",
            "array_origin": "exact recomputation of independent_td kernel formula; negative norm identity checked",
            "negative_lag_norm_is_error_bound": False}


def comparison(direct_record, fd_record, dt=.07):
    t, ys = common_series([direct_record, fd_record], dt=dt)
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        if t[0] > lo or t[-1] < hi:
            out[name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
            continue
        mask = (t >= lo) & (t <= hi)
        if np.count_nonzero(mask) < 3:
            out[name] = {"state": "UNQUANTIFIED", "reason": "TOO_FEW_SAMPLES"}
            continue
        delta, reference = (ys[1]-ys[0])[mask], ys[0][mask]
        dnorm = float(np.sqrt(trapezoid(delta**2, t[mask])))
        rnorm = float(np.sqrt(trapezoid(reference**2, t[mask])))
        sampled_norm = float(np.linalg.norm(reference))
        out[name] = {"state": "MEASURED_DISCREPANCY", "difference_L2": dnorm,
                     "PDE_L2": rnorm, "relative_L2": dnorm/rnorm if rnorm else None,
                     "sample_norm_relative_L2": float(np.linalg.norm(delta)/sampled_norm) if sampled_norm else None,
                     "samples": int(np.count_nonzero(mask)), "is_error_bound": False}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, repo = args.output.resolve(), Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError("Choose a new output directory; controls are immutable")
    archived = read_archive(args.record)
    original = archived["result"]
    expected = {"model": "QH", "params": {"EplM": 1e8, "gamma": 1.}}
    if any(archived["request"].get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the declared fiducial QH reference")
    settings = archived["request"]["settings"]
    required = {"dx": .1, "t_end": 520., "cfl": .35, "source_global": 55.,
                "observer_global": 95., "sigma": 4., "surface": -40., "x_left": -250.,
                "x_right": 400., "source_t_end": 650., "source_x_left": -400.,
                "scattering_step": .02, "scattering_x_right": 200., "sample_every": 2}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("Reference settings differ from the bounded medium-resolution experiment")
    if any(not value["passed"] for value in original["checks"].values()):
        raise ValueError("Reference has failed checks")
    config = plan()
    provenance = runtime_provenance(repo)
    provenance.update(diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reference_archive_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
                      reference_provenance=archived["provenance"], reference_request=archived["request"])
    write_json(root/"plan.json", {**config, "provenance": provenance}, root)
    rt = load_frozen_runtime()
    records, comparisons, execution, failures = [], {}, {}, {}
    for h in config["levels"]:
        label = f"h{h:g}"
        try:
            start = time.perf_counter()
            direct = independent_td(dx=h, t_end=config["t_end"], surface=config["surface"],
                observer=config["observer"], source_global=config["source_global"], sigma=config["sigma"],
                x_right=config["x_right"], **config["params"])
            kernel = retained_kernel(direct)
            x = config["surface"]+h*np.arange(int(round((config["x_right"]-config["surface"])/h))+1)
            r = rt["r_from_schwarzschild_rstar"](x)
            result = {"t": direct["time"], "psi": direct["correction"], "x": x, "r": r,
                      "V": rt["rw_potential"](r, 2, 1.), "independent": direct, "wall_kernel": kernel,
                      "metadata": {"model_family": "QH", "physical_params": config["params"],
                         "numerics_evidence": "UNQUANTIFIED", "wall_seconds": time.perf_counter()-start,
                         "purpose": "independent causal-wall correction; zero absorbing-wave baseline at readout",
                         "reference_role": "historical FD comparison only; not an evolution input"},
                      "checks": {
                         "finite": {"passed": bool(np.isfinite(direct["correction"]).all())},
                         "right_boundary": {"passed": bool(direct["right_boundary_return_M"] > direct["time"][-1]),
                            "earliest_practical_return_M": direct["right_boundary_return_M"],
                            "scope": "8sigma practical screen, not a strict Gaussian-support bound"},
                         "kernel_period": {"passed": bool(kernel["period_M"] == config["kernel_period_M"]),
                                           "period_M": kernel["period_M"]},
                         "negative_lag_norm": {"passed": bool(direct["negative_lag_kernel_L2_fraction"] < config["negative_lag_norm_screen"]),
                            "value": direct["negative_lag_kernel_L2_fraction"], "is_error_bound": False}}}
            request = {"model": "QH", "kind": "fine_independent_PDE", "dx": h, "configuration": config,
                       "reference_archive_sha256": provenance["reference_archive_sha256"]}
            path = root/"records"/(label+".h5")
            write_archive(path, request, result, provenance, root)
            checked = read_archive(path, request, provenance)["result"]
            records.append({"t": checked["t"], "psi": checked["psi"]})
            comparisons[label] = comparison(records[-1], {"t": original["t"], "psi": original["response"]["correction"]}, config["comparison_dt"])
            execution[label] = {"dx": h, "wall_seconds": result["metadata"]["wall_seconds"],
                               "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "failed_checks": [key for key, value in result["checks"].items() if not value["passed"]],
                               "kernel_nfft": kernel["nfft"], "kernel_period_M": kernel["period_M"],
                               "negative_lag_kernel_L2_fraction": direct["negative_lag_kernel_L2_fraction"]}
            print(label, "late", comparisons[label]["late"], "seconds", execution[label]["wall_seconds"], flush=True)
            del checked, kernel, result, direct
        except Exception as exc:
            failures[label] = {"exception": str(exc), "traceback": traceback.format_exc()}
            write_json(root/"failures"/(label+".json"), failures[label], root)
            print(label, "FAILED", str(exc), flush=True)
    refinement = {}
    if not failures and len(records) == 4:
        t, ys = common_series(records, dt=config["comparison_dt"])
        for first in (0, 1):
            refinement[f"levels_{first}_to_{first+2}"] = assess(t, ys[first:first+3], config["levels"][first:first+3],
                expected_order=config["expected_order"], order_tolerance=config["order_tolerance"])
    _check_resume_provenance(provenance, runtime_provenance(repo))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != provenance["diagnostic_script_sha256"]:
        raise ValueError("Diagnostic script changed during execution")
    if hashlib.sha256(args.record.read_bytes()).hexdigest() != provenance["reference_archive_sha256"]:
        raise ValueError("Reference archive changed during execution")
    report = {"configuration": config, "provenance": provenance, "execution": execution,
              "comparisons": comparisons, "refinement": refinement, "failures": failures,
              "total_error_status": "UNQUANTIFIED", "is_error_bound": False}
    write_json(root/"report.json", report, root)
    return int(bool(failures) or any(row["failed_checks"] for row in execution.values()))


if __name__ == "__main__":
    raise SystemExit(main())
