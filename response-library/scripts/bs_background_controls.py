"""Separate dense-profile sampling from finite-radius BS background controls.

This is six fiducial 520 M runs, not a physical parameter sweep. Default 5000
sample replay is unchanged. Exactly nested 5001/10001/20001 samples at rmax=50
test profile/quadrature reconstruction at fixed BVP and TD settings. The domain
variants use rmax=59.999998 and 6001/24001 samples to preserve mu-radial spacing.
The remaining domain difference includes finite-tail, vacuum-matching and BVP
errors; no run certifies radial stability or a total error bound.
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
import h5py
import numpy as np

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.adapter_pilot import difference
from eco_library.boson_star import FIDUCIAL_SIGMA_C, run_boson_star
from eco_library.convergence import common_series, assess
from eco_library.run import BASELINE_H5_SHA256, gate_refinement
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance


def plan():
    base = {"dx": .1, "t_end": 520., "cfl": .35, "sigma": 4.,
            "source_global": 55., "observer_global": 95., "x_right": 400.,
            "sample_every": 2, "equilibrium_rmax": 50., "equilibrium_n": 1200,
            "equilibrium_tol": 2e-7}
    extended = 1e-5+6000*((50.-1e-5)/5000)
    settings = {
        "baseline5000": {**base, "background_samples": 5000},
        "profile5001": {**base, "background_samples": 5001},
        "profile10001": {**base, "background_samples": 10001},
        "profile20001": {**base, "background_samples": 20001},
        "domain6001": {**base, "equilibrium_rmax": extended, "background_samples": 6001},
        "domain24001": {**base, "equilibrium_rmax": extended, "background_samples": 24001},
    }
    return {"version": "bs-background-controls/0.1", "jobs": {
        key: {"model": "BS", "params": {"sigma_c": FIDUCIAL_SIGMA_C}, "settings": value}
        for key, value in settings.items()},
        "declared_sampling_order": 2, "order_basis": "trapezoidal tortoise quadrature; waveform order must be observed",
        "scope": "fiducial numerical-factor controls, not a physical sweep, stability proof or total bound",
        "expected_cost": "six local medium-grid runs, minutes and modest memory",
        "failure_policy": "retain explicit failure and do not substitute a waveform",
        "domain_control": "mu-radial spacing fixed; remaining BVP/tail/matching effects not individually bounded"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.output.resolve()
    configuration = plan()
    if args.plan_only:
        import json
        print(json.dumps(configuration, indent=2))
        return 0
    # Refuse reuse/overwriting: each diagnostic execution gets a new directory.
    if root.exists():
        raise FileExistsError("Choose a new output directory for these immutable controls")
    write_json(root/"plan.json", configuration, root)
    repo = Path(__file__).resolve().parents[1]
    provenance = runtime_provenance(repo)
    provenance["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records, execution = {}, {}
    for label, request in configuration["jobs"].items():
        try:
            start = time.perf_counter()
            result = run_boson_star(request["params"], **request["settings"])
            result["metadata"]["wall_seconds"] = time.perf_counter()-start
            path = root/"records"/(label+".h5")
            write_archive(path, request, result, provenance, root)
            records[label] = read_archive(path, request, provenance)["result"]
            execution[label] = {"state": "COMPLETE", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "wall_seconds": result["metadata"]["wall_seconds"],
                                "failed_checks": [k for k, v in result["checks"].items() if not v["passed"]]}
        except Exception as exc:
            execution[label] = {"state": "FAILED", "reason": str(exc)}
            write_json(root/"failures"/(label+".json"), {"request": request, "provenance": provenance,
                       "reason": str(exc), "traceback": traceback.format_exc()}, root)
        print(label, execution[label], flush=True)
    report = {"version": configuration["version"], "provenance": provenance,
              "execution": execution, "refinement": {}, "controls": {},
              "equilibrium": {}, "default_replay": {}, "estimates_are_total_bounds": False}
    names = ["profile5001", "profile10001", "profile20001"]
    if all(key in records for key in names):
        rs = [records[key] for key in names]
        t, ys = common_series(rs)
        hs = [r["metadata"]["background_reconstruction"]["radial_spacing_mu"] for r in rs]
        valid = all(v["passed"] for r in rs for v in r["checks"].values())
        report["refinement"] = gate_refinement(assess(t, ys, hs, expected_order=2), valid)
        report["same_BVP"] = {key: len({r["metadata"][key] for r in rs}) == 1
            for key in ("raw_mass_mu_units", "omega_s_over_mu", "EKG_residual")}
    for label, left, right in (("sampling5000_to20001", "baseline5000", "profile20001"),
                               ("domain_fixed_coarse_spacing", "profile5001", "domain6001"),
                               ("domain_fixed_fine_spacing", "profile20001", "domain24001")):
        if left in records and right in records:
            report["controls"][label] = difference(records[left], records[right])
            report["controls"][label]["radial_spacing_mu"] = [records[key]["metadata"]["background_reconstruction"]["radial_spacing_mu"] for key in (left, right)]
    for label, r in records.items():
        m = r["metadata"]
        report["equilibrium"][label] = {key: m[key] for key in (
            "raw_mass_mu_units", "omega_s_over_mu", "EKG_residual", "source_native", "observer_native",
            "background_samples", "equilibrium_rmax", "background_reconstruction", "finite_tail")}
    frozen = repo/"baseline/eco_outputs_final/ECO_waveforms.h5"
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != BASELINE_H5_SHA256:
        raise ValueError("Frozen HDF5 changed; replay refused")
    if "baseline5000" in records:
        with h5py.File(frozen, "r") as f:
            old = f["controlled_comparison"]["BS"]
            r = records["baseline5000"]
            report["default_replay"] = {"time_exact": np.array_equal(r["t"], old["time"][:]),
                                         "waveform_exact": np.array_equal(r["psi"], old["waveform_real"][:])}
    _check_resume_provenance(provenance, runtime_provenance(repo))
    if provenance["diagnostic_script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Diagnostic source changed during execution; refusing mixed-provenance report")
    write_json(root/"report.json", report, root)
    print("Report:", root/"report.json", flush=True)
    return int(any(v["state"] != "COMPLETE" or v.get("failed_checks") for v in execution.values())
               or report["default_replay"].get("time_exact") is not True
               or report["default_replay"].get("waveform_exact") is not True)


if __name__ == "__main__":
    raise SystemExit(main())
