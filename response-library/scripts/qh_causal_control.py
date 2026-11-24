"""Compare the actual causal discrete wall with the analytic OWA FD response.

This is a labelled numerical approximation audit, NOT a new physical model or
a change to the production reflection law. The wall polynomial exactly matches
independent_td's causal convolution; continuum Jost propagation is unchanged.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.signal import freqz

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.adapter_pilot import difference
from eco_library.convergence import _real_vector
from eco_library.quantum_horizon import reflection, reconstruct, _number
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance


def wall_kernel(h, *, t_end=520.2, surface=-40., EplM=1e8, gamma=1.):
    """Exactly the independent solver's wall, including full zero-lag weight."""
    h = _number(h, "kernel dt", positive=True)
    duration = _number(t_end, "duration", positive=True)
    surface = _number(surface, "surface")
    nt = int(np.ceil(duration/h))+1
    if not 3 <= nt <= 100000:
        raise ValueError("Wall history must contain 3..100000 samples")
    nfft = 1 << (max(32*nt, 32768)-1).bit_length()
    omega = 2*np.pi*np.fft.rfftfreq(nfft, h)
    local = reflection(omega, EplM=EplM, gamma=gamma, surface=surface)
    full = np.fft.irfft(np.conj(local), nfft)
    norm = float(np.linalg.norm(full))
    return {"impulse": full[:nt].copy(), "dt": h, "nfft": nfft,
            "period_M": nfft*h, "retained_end_M": (nt-1)*h,
            "negative_lag_L2_fraction": float(np.linalg.norm(full[nfft//2+1:])/norm),
            "omitted_positive_lag_L2_fraction": float(np.linalg.norm(full[nt:nfft//2+1])/norm),
            "full_kernel_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
            "surface": surface, "params": {"EplM": EplM, "gamma": gamma},
            "formula": "k=irfft(conj(R_local),N); retain k[0:nt], full weight at lag0",
            "scope": "finite causal discrete approximation; fractions are not waveform error bounds"}


def physical_wall_response(impulse, h, omega):
    """R_c(w)=sum_m k[m] exp(+i*w*m*h); no h factor or interpolation."""
    kernel = _real_vector(impulse, "wall impulse", minimum=1)
    omega = _real_vector(omega, "frequencies", minimum=1)
    h = _number(h, "kernel dt", positive=True)
    if np.any(omega < 0) or np.any(omega > np.pi/h):
        raise ValueError("Positive-real response must lie within kernel Nyquist band")
    # For an explicit frequency array freqz evaluates the FIR polynomial. Its
    # negative-exponent convention is conjugated to our exp(-i*w*t) convention.
    _, numpy_response = freqz(kernel, worN=omega*h)
    physical = np.conj(numpy_response)
    if not np.isfinite(physical).all():
        raise ArithmeticError("Nonfinite wall frequency response")
    return physical


def response_with_wall(source, time_grid, scattering, wall, *, nfft=None):
    """Only substitute the wall coefficient; all source/Jost data stay fixed."""
    if float(source["surface"]) != wall["surface"] or float(scattering["surface"]) != wall["surface"]:
        raise ValueError("Causal kernel/source/scattering reference planes must match")
    analytic = reconstruct(source, time_grid, scattering, nfft=nfft,
        surface=wall["surface"], observer=scattering["observer"], **wall["params"])
    omega = analytic["omega"]
    supported = omega <= scattering["omega"][-1]
    w = omega[supported]
    local = physical_wall_response(wall["impulse"], wall["dt"], w)
    rb, te = analytic["R_barrier"][supported], analytic["T_detector"][supported]
    loop = rb*local*np.exp(-2j*w*wall["surface"])
    transfer = np.exp(1j*w*(scattering["observer"]-wall["surface"]))*te*local/(1-loop)
    K = np.zeros_like(omega, dtype=complex)
    K[supported] = transfer
    correction = np.fft.irfft(np.conj(K)*analytic["source_spectrum"]["ingoing_fft"],
                              analytic["nfft"])[:len(time_grid)]
    if not np.isfinite(correction).all() or not np.isfinite(K).all():
        raise ArithmeticError("Nonfinite causal-wall transfer")
    return {"omega": omega, "K": K, "correction": correction,
            "wall_omega": w, "R_local_causal": local, "loop_function_on_support": loop,
            "nfft": analytic["nfft"], "analytic_OWA_correction": analytic["correction"],
            "source_spectrum": analytic["source_spectrum"],
            "coefficient_action": "physical R_c=sum(k[m]*exp(+i*w*m*h)); correction=irfft(conj(K)*I_numpy)",
            "frequency_support": "wall evaluated only through the Jost cutoff; K zero above cutoff",
            "DC_policy": "incoming DC prescribed zero exactly as production; no static response inferred"}


def plan():
    return {"version": "qh-causal-wall-controls/0.1", "t_end": 520.2,
            "levels": [.05, .025, .0125, .00625], "padding_factors_at_finest": [2, 4],
            "scope": "isolate causal wall approximation at fixed physical/source/Jost experiment",
            "expected_cost": "six local FIR-transfer controls; no new large PDE or cloud job",
            "comparison": "all four windows, raw integral L2, no fitted shift/scale/offset",
            "is_total_error_bound": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    archived = read_archive(args.record)
    original, request = archived["result"], archived["request"]
    if (request.get("model") != "QH" or request["params"] != {"EplM": 1e8, "gamma": 1.}
            or original["metadata"]["x_interface_global"] != -40.
            or original["metadata"]["observer_global"] != 95.
            or original["t"][-1] < 500.):
        raise ValueError("Expected fiducial full-duration QH archive at declared reference planes")
    root, repo = args.output.resolve(), Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError("Use a new immutable output directory")
    provenance = runtime_provenance(repo)
    provenance.update(diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_archive_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        reference_provenance=archived["provenance"], reference_request=request)
    configuration = plan()
    write_json(root/"plan.json", {**configuration, "provenance": provenance}, root)
    jobs = [(h, 1) for h in configuration["levels"]]
    jobs.extend((configuration["levels"][-1], factor) for factor in configuration["padding_factors_at_finest"])
    rows = {}
    for h, padding in jobs:
        started = time.perf_counter()
        wall = wall_kernel(h, t_end=configuration["t_end"], **request["params"])
        response = response_with_wall(original["source"], original["t"], original["scattering"],
            wall, nfft=padding*original["response"]["nfft"])
        result = {"t": original["t"], "psi": response["correction"],
            **{k: original[k] for k in ("x", "r", "V")}, "wall": wall, "response": response,
            "source": original["source"], "scattering": original["scattering"],
            "metadata": {"model_family": "QH", "role": "isolated_correction_not_total_field",
                "physical_law": "OWA unchanged in production; causal FIR substitution is a numerical control only",
                "numerics_evidence": "UNQUANTIFIED", "kernel_dt": h, "padding_factor": padding,
                "wall_seconds": time.perf_counter()-started},
            "checks": {"finite": {"passed": bool(np.isfinite(response["correction"]).all())}}}
        label = f"h{h:g}-padding{padding}"
        req = {"model": "QH", "kind": "causal_wall_FD_control", "kernel_dt": h,
               "t_end": configuration["t_end"], "padding_factor": padding,
               "reference_archive_sha256": provenance["reference_archive_sha256"]}
        path = root/"records"/(label+".h5")
        write_archive(path, req, result, provenance, root)
        result = read_archive(path, req, provenance)["result"]
        reference = {"t": original["t"], "psi": original["response"]["correction"]}
        same_padding_analytic = {"t": original["t"], "psi": response["analytic_OWA_correction"]}
        rows[label] = {"kernel_dt": h, "padding_factor": padding,
            "versus_original_FD": difference(reference, result),
            "versus_same_padding_analytic_FD": difference(same_padding_analytic, result),
            "negative_lag_L2_fraction": wall["negative_lag_L2_fraction"],
            "kernel_period_M": wall["period_M"],
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "wall_seconds": result["metadata"]["wall_seconds"]}
        print(label, rows[label]["versus_same_padding_analytic_FD"]["late"], flush=True)
    _check_resume_provenance(provenance, runtime_provenance(repo))
    if provenance["diagnostic_script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Diagnostic script changed during execution")
    write_json(root/"report.json", {**configuration, "provenance": provenance, "results": rows}, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
