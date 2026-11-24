"""One-factor QH spectral/domain sensitivities; no fitting or error-bound claims."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import trapezoid

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.convergence import WINDOWS, _real_vector
from eco_library.quantum_horizon import _runtime, reconstruct, run_quantum_horizon
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance


def nested_grid(grid, factor):
    """Preserve every old node while refining all linear-frequency intervals."""
    grid = _real_vector(grid, "frequency grid")
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or grid[0] <= 0 or np.any(np.diff(grid) <= 0)
            or isinstance(factor, bool) or not isinstance(factor, int) or factor < 1):
        raise ValueError("Increasing positive grid and integer refinement factor required")
    return np.r_[np.concatenate([a+(b-a)*np.arange(factor)/factor
                                for a, b in zip(grid[:-1], grid[1:])]), grid[-1]]


def plan():
    return {
        "reference_identity": {"kind": "reconstruct"},
        "grid2": {"kind": "scattering", "grid_factor": 2},
        "grid4": {"kind": "scattering", "grid_factor": 4},
        "low_extend": {"kind": "scattering", "omega_min": 1e-5},
        "low_stress": {"kind": "subset", "omega_min": .005},
        "high_extend": {"kind": "scattering", "omega_max": 2.},
        "padding2": {"kind": "reconstruct", "padding_factor": 2},
        "padding4": {"kind": "reconstruct", "padding_factor": 4},
        "radius300": {"kind": "scattering", "scattering_x_right": 300.},
        "radius400": {"kind": "scattering", "scattering_x_right": 400.},
        "source750": {"kind": "fresh", "source_t_end": 750.},
        "source_left500": {"kind": "fresh", "source_x_left": -500.},
        "plane50": {"kind": "fresh", "surface": -50.},
    }


def compare(reference, changed):
    """Native-grid integral norms, with no alignment/interpolation/silent floor."""
    if not np.array_equal(reference["t"], changed["t"]):
        raise ValueError("Controls require exactly the same native output time grid")
    t = _real_vector(reference["t"], "reference time", minimum=3)
    if not np.all(np.diff(t) > 0):
        raise ValueError("Strictly increasing time required")
    result = {}
    for field in ("total", "correction"):
        a = reference["psi"] if field == "total" else reference["response"]["correction"]
        b = changed["psi"] if field == "total" else changed["response"]["correction"]
        a, b = _real_vector(a, "reference waveform"), _real_vector(b, "control waveform")
        if a.shape != t.shape or b.shape != t.shape or not np.isfinite([a, b]).all():
            raise ValueError("Finite waveforms on the reference time grid required")
        result[field] = {}
        for name, (lo, hi) in WINDOWS.items():
            if t[0] > lo or t[-1] < hi:
                result[field][name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
                continue
            mask = (t >= lo) & (t <= hi)
            if np.count_nonzero(mask) < 3:
                result[field][name] = {"state": "UNQUANTIFIED", "reason": "TOO_FEW_SAMPLES"}
                continue
            norm = float(np.sqrt(trapezoid(a[mask]**2, t[mask])))
            change = float(np.sqrt(trapezoid((b[mask]-a[mask])**2, t[mask])))
            result[field][name] = {"difference_L2": change, "reference_L2": norm,
                "relative_difference": change/norm if norm > 0 else None,
                "is_error_bound": False, "state": "MEASURED_SENSITIVITY"}
    return result


def make_control(original, original_request, control):
    """Reuse measured source only for spectral controls; never fit the result."""
    if control["kind"] == "fresh":
        settings = {**original_request["settings"], **{k: v for k, v in control.items() if k != "kind"}}
        settings["independent_dx"] = None
        return run_quantum_horizon(original_request["params"], **settings)
    out = copy.deepcopy(original)
    out["independent"] = {"status": "NOT_COMPUTED_FOR_CONTROL"}
    out["checks"].pop("independent_reconstruction", None)
    sc = out["scattering"]
    grid = sc["omega"]
    if "grid_factor" in control:
        grid = nested_grid(grid, control["grid_factor"])
    if control["kind"] == "subset":
        keep = grid >= control["omega_min"]
        sc = {**sc, **{k: sc[k][keep] for k in ("omega", "R_barrier", "T_infinity", "T_detector")}}
        grid = sc["omega"]
    elif "omega_min" in control:
        grid = np.r_[np.geomspace(control["omega_min"], grid[0], 36, endpoint=False), grid]
    if "omega_max" in control:
        n = int(np.ceil((control["omega_max"]-grid[-1])/np.max(np.diff(grid))))
        grid = np.r_[grid, np.linspace(grid[-1], control["omega_max"], n+1)[1:]]
    radius = control.get("scattering_x_right", original["metadata"]["scattering_x_right"])
    if control["kind"] == "scattering":
        rt = _runtime()
        rb, tb, te = rt["barrier_coefficients_vectorized"](grid, 2, 1.,
            xL=sc["surface"], xR=radius, step=original["metadata"]["scattering_step"],
            extract=sc["observer"])
        sc = {**sc, "omega": grid, "R_barrier": rb, "T_infinity": tb, "T_detector": te}
    nfft = original["response"]["nfft"]*control.get("padding_factor", 1)
    response = reconstruct(out["source"], out["t"], sc, **original_request["params"],
        surface=sc["surface"], observer=sc["observer"], nfft=nfft)
    out.update(scattering=sc, response=response,
               psi=out["bh_reference"]["psi"]+response["correction"])
    flux = float(np.max(abs(abs(sc["R_barrier"])**2+abs(sc["T_infinity"])**2-1)))
    out["checks"]["barrier_flux"].update(passed=bool(flux < 2e-6), max_abs=flux)
    out["checks"]["global_local_identity"].update(
        passed=bool(response["global_local_identity_error"] < 1e-12),
        max_scaled_abs=response["global_local_identity_error"])
    out["checks"]["finite"]["passed"] = bool(np.isfinite(out["psi"]).all()
                                               and np.isfinite(response["K"]).all())
    out["metadata"].update(scattering_x_right=radius, scattering_flux_residual=flux,
        omega_min_integrated=float(sc["omega"][0]), omega_max_integrated=float(sc["omega"][-1]),
        nfft=nfft, source_omega_c=response["source_spectrum"]["omega_c"],
        numerics_evidence="UNQUANTIFIED")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    archived = read_archive(args.record)
    request, original = archived["request"], archived["result"]
    # This script is deliberately a bounded audited experiment, not a general sweep.
    required = {"dx": .1, "t_end": 520., "cfl": .35, "sigma": 4., "source_global": 55.,
                "observer_global": 95., "x_right": 400., "sample_every": 2,
                "surface": -40., "x_left": -250., "source_t_end": 650.,
                "source_x_left": -400., "scattering_step": .02, "scattering_x_right": 200.}
    if (request.get("model") != "QH" or request["params"] != {"EplM": 1e8, "gamma": 1.}
            or any(request["settings"].get(k) != v for k, v in required.items())):
        raise ValueError("Expected the declared fiducial medium safe-domain QH reference")
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Choose a new output directory for immutable controls")
    provenance = runtime_provenance(Path(__file__).resolve().parents[1])
    provenance.update(diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_archive_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        reference_provenance=archived["provenance"], reference_request=request)
    configuration = {"version": "qh-spectral-controls/0.1", "controls": plan(), "provenance": provenance,
                     "scope": "one-factor measured sensitivity; not a bound, parameter atlas or total error estimate"}
    write_json(root/"plan.json", configuration, root)
    rows, failures = {}, {}
    for label, control in configuration["controls"].items():
        start = time.perf_counter()
        try:
            out = make_control(original, request, control)
            out["metadata"].update(diagnostic_control=control, wall_seconds=time.perf_counter()-start,
                reference_archive_sha256=provenance["reference_archive_sha256"],
                control_input_policy="historic measured source for spectral controls; fresh evolution for source controls")
            req = {"model": "QH", "kind": "spectral_control", "control": control,
                   "reference_archive_sha256": provenance["reference_archive_sha256"]}
            path = root/"records"/(label+".h5")
            write_archive(path, req, out, provenance, root)
            out = read_archive(path, req, provenance)["result"]
            w = out["response"]["omega"]
            rows[label] = {"control": control, "comparison": compare(original, out),
                "failed_checks": [k for k, v in out["checks"].items() if not v["passed"]],
                "positive_fft_bins_below_omega_min": int(np.sum((w > 0) & (w < out["scattering"]["omega"][0]))),
                "nfft": out["response"]["nfft"], "frequency_nodes": len(out["scattering"]["omega"]),
                "omega_min": float(out["scattering"]["omega"][0]),
                "omega_max": float(out["scattering"]["omega"][-1]),
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "wall_seconds": out["metadata"]["wall_seconds"]}
            print(label, rows[label]["comparison"]["correction"]["late"],
                  "failed_checks", rows[label]["failed_checks"], flush=True)
        except Exception as exc:
            failure = {"control": control, "exception": str(exc), "traceback": traceback.format_exc()}
            failures[label] = failure
            write_json(root/"failures"/(label+".json"), failure, root)
            print(label, "FAILED", str(exc), flush=True)
    if "reference_identity" in rows:
        for field in ("total", "correction"):
            if any(v["difference_L2"] != 0 for v in rows["reference_identity"]["comparison"][field].values()):
                failures["reference_identity"] = {"reason": "Reconstruction no longer reproduces archived source/scattering"}
    _check_resume_provenance(provenance, runtime_provenance(Path(__file__).resolve().parents[1]))
    if provenance["diagnostic_script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Diagnostic source changed during execution; refusing mixed-provenance report")
    summary = {**configuration, "results": rows, "failures": failures,
               "numerical_uncertainty": "UNQUANTIFIED_TOTAL", "is_error_bound": False}
    write_json(root/"report.json", summary, root)
    return int(bool(failures) or any(r["failed_checks"] for r in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
