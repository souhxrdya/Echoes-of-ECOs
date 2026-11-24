"""Separate finite kernel period from causal truncation in the QH diagnostic.

This does not rerun the PDE or replace the physical OWA prescription. Three
kernel periods at fixed h=.025 and common 4x FD padding test only the wall
construction. The full positive-time kernel need not be renormalized.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.adapter_pilot import difference
from eco_library.quantum_horizon import reflection
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance
from scripts.qh_causal_control import wall_kernel, response_with_wall


def period_wall(factor):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor not in (1, 2, 4):
        raise ValueError("Declared kernel period factors are 1, 2, 4")
    wall = wall_kernel(.025)
    if factor > 1:
        nfft = factor*wall["nfft"]
        omega = 2*np.pi*np.fft.rfftfreq(nfft, wall["dt"])
        full = np.fft.irfft(np.conj(reflection(omega, surface=wall["surface"], **wall["params"])), nfft)
        nt, norm = len(wall["impulse"]), float(np.linalg.norm(full))
        wall.update(impulse=full[:nt].copy(), nfft=nfft, period_M=nfft*wall["dt"],
            negative_lag_L2_fraction=float(np.linalg.norm(full[nfft//2+1:])/norm),
            omitted_positive_lag_L2_fraction=float(np.linalg.norm(full[nt:nfft//2+1])/norm),
            full_kernel_sha256=hashlib.sha256(full.tobytes()).hexdigest())
    wall["period_factor"] = factor
    return wall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    original = read_archive(args.record)
    data = original["result"]
    if (original["request"].get("model") != "QH"
            or original["request"]["params"] != {"EplM": 1e8, "gamma": 1.}
            or data["metadata"]["x_interface_global"] != -40.
            or data["metadata"]["observer_global"] != 95.):
        raise ValueError("Expected the declared fiducial QH experiment")
    root, repo = args.output.resolve(), Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError("Use a new output directory")
    dependency = Path(__file__).with_name("qh_causal_control.py")
    provenance = runtime_provenance(repo)
    provenance.update(diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        causal_control_dependency_sha256=hashlib.sha256(dependency.read_bytes()).hexdigest(),
        reference_archive_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        reference_provenance=original["provenance"], reference_request=original["request"])
    config = {"version": "qh-kernel-period/0.1", "kernel_dt": .025, "duration_M": 520.2,
              "period_factors": [1, 2, 4], "FD_padding_factor": 4,
              "cost": "three local FIR-transfer controls; no fresh PDE or physical parameter change",
              "scope": "finite-period sensitivity of diagnostic causal wall; not a total bound"}
    write_json(root/"plan.json", {**config, "provenance": provenance}, root)
    rows, base, previous = {}, None, None
    for factor in config["period_factors"]:
        wall = period_wall(factor)
        response = response_with_wall(data["source"], data["t"], data["scattering"], wall,
                                      nfft=4*data["response"]["nfft"])
        result = {"t": data["t"], "psi": response["correction"],
                  **{k: data[k] for k in ("x", "r", "V")}, "wall": wall, "response": response,
                  "source": data["source"], "scattering": data["scattering"],
                  "metadata": {"role": "diagnostic_causal_FD_correction_not_total", "kernel_period_factor": factor,
                               "numerics_evidence": "UNQUANTIFIED"},
                  "checks": {"finite": {"passed": bool(np.isfinite(response["correction"]).all())}}}
        request = {"model": "QH", "kind": "kernel_period_control", "factor": factor,
                   "configuration": config, "reference_archive_sha256": provenance["reference_archive_sha256"]}
        path = root/"records"/f"factor{factor}.h5"
        write_archive(path, request, result, provenance, root)
        read_archive(path, request, provenance)
        analytic = {"t": data["t"], "psi": response["analytic_OWA_correction"]}
        row = {"period_M": wall["period_M"], "nfft": wall["nfft"],
               "negative_lag_L2_fraction": wall["negative_lag_L2_fraction"],
               "retained_kernel_sum": float(np.sum(wall["impulse"])),
               "versus_same_padding_analytic_FD": difference(analytic, result),
               "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if base is None:
            base = result
        else:
            row["versus_base_period"] = difference(base, result)
            row["versus_previous_period"] = difference(previous, result)
        previous = result
        rows[f"factor{factor}"] = row
        print(factor, row["versus_same_padding_analytic_FD"]["late"], row["negative_lag_L2_fraction"], flush=True)
    _check_resume_provenance(provenance, runtime_provenance(repo))
    for path, key in ((Path(__file__), "diagnostic_script_sha256"), (dependency, "causal_control_dependency_sha256"),
                      (args.record, "reference_archive_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance[key]:
            raise ValueError("Diagnostic source/input changed during execution")
    write_json(root/"report.json", {**config, "provenance": provenance, "results": rows}, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
