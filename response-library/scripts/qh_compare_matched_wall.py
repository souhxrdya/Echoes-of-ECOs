"""Bind independent PDE and FD archives by their EXACT causal wall samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import trapezoid

from eco_library.adapter_archive import read_archive, write_archive
from eco_library.convergence import WINDOWS, common_series
from eco_library.storage import runtime_provenance, write_json, _check_resume_provenance


def validate_pair(pde, fd):
    """Matching filenames or nominal parameters alone are not sufficient."""
    a, b = pde["result"], fd["result"]
    if pde["request"].get("kind") != "fine_independent_PDE" or fd["request"].get("kind") != "causal_wall_FD_control":
        raise ValueError("Expected explicitly labelled independent PDE and causal-wall FD records")
    if pde["provenance"]["reference_archive_sha256"] != fd["provenance"]["reference_archive_sha256"]:
        raise ValueError("Different original reference experiments")
    if (a["independent"]["dt"] != b["wall"]["dt"]
            or a["independent"]["surface"] != b["wall"]["surface"]
            or a["independent"]["params"] != b["wall"]["params"]
            or a["wall_kernel"]["nfft"] != b["wall"]["nfft"]
            or not np.array_equal(a["wall_kernel"]["causal_impulse_used"], b["wall"]["impulse"])):
        raise ValueError("PDE and FD do not use the exact same discrete wall")
    if any(not value["passed"] for record in (a, b) for value in record["checks"].values()):
        raise ValueError("A contributing record has failed checks")


def paired_metrics(pde, causal, analytic):
    """Raw waveform identity and norms; no fit, projection, or cancellation claim."""
    t, y = common_series([pde, causal, analytic])
    direct, matched, reference = y
    raw = reference-direct
    wall = reference-matched
    residual = matched-direct
    identity = float(np.max(abs(raw-wall-residual)))
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        if t[0] > lo or t[-1] < hi:
            out[name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
            continue
        mask = (t >= lo) & (t <= hi)
        if np.count_nonzero(mask) < 3:
            out[name] = {"state": "UNQUANTIFIED", "reason": "TOO_FEW_SAMPLES"}
            continue
        norm = lambda array: float(np.sqrt(trapezoid(array[mask]**2, t[mask])))
        denominator = norm(direct)
        errors = {key: norm(value) for key, value in (
            ("analytic_FD_minus_PDE", raw), ("analytic_FD_minus_causal_FD", wall),
            ("causal_FD_minus_PDE", residual))}
        out[name] = {"state": "MEASURED_DISCREPANCY", "PDE_L2": denominator,
                     "absolute_L2": errors,
                     "relative_L2": {k: v/denominator if denominator else None for k, v in errors.items()},
                     "is_total_error_bound": False}
    return {"t": t, "PDE": direct, "FD_causal": matched, "FD_analytic": reference,
            "analytic_minus_PDE": raw, "wall_difference": wall, "matched_residual": residual}, out, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pde_directory", type=Path)
    parser.add_argument("causal_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, repo = args.output.resolve(), Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError("Choose a new output directory")
    provenance = runtime_provenance(repo)
    provenance["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, sources = {}, {}
    for h in (.05, .025, .0125, .00625):
        pde_path = args.pde_directory/"records"/f"h{h:g}.h5"
        fd_path = args.causal_directory/"records"/f"h{h:g}-padding1.h5"
        pde, fd = read_archive(pde_path), read_archive(fd_path)
        validate_pair(pde, fd)
        a, b = pde["result"], fd["result"]
        arrays, metrics, identity = paired_metrics(a, b,
            {"t": b["t"], "psi": b["response"]["analytic_OWA_correction"]})
        source_hashes = {"PDE": hashlib.sha256(pde_path.read_bytes()).hexdigest(),
                         "FD_causal": hashlib.sha256(fd_path.read_bytes()).hexdigest()}
        label = f"h{h:g}"
        sources[label] = source_hashes
        request = {"model": "QH", "kind": "matched_wall_difference", "h": h,
                   "source_archive_sha256": source_hashes}
        result = {"t": arrays["t"], "psi": arrays["matched_residual"],
                  **{key: a[key] for key in ("x", "r", "V")}, "comparison_arrays": arrays,
                  "metadata": {"role": "diagnostic_FD_minus_PDE_not_physical_waveform",
                      "exact_wall_samples_verified": True, "kernel_dt": h,
                      "numerics_evidence": "UNQUANTIFIED", "source_archive_sha256": source_hashes},
                  "checks": {"difference_identity": {"passed": identity < 1e-15,
                      "max_absolute": identity, "scope": "bookkeeping identity, not independent physical validation"}}}
        path = root/"records"/(label+".h5")
        write_archive(path, request, result, provenance, root)
        read_archive(path, request, provenance)
        rows[label] = {"h": h, "windows": metrics, "difference_identity_max_absolute": identity,
                       "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(label, metrics["late"]["relative_L2"], flush=True)
    scaling = {}
    for coarse, fine in zip((.05, .025, .0125), (.025, .0125, .00625)):
        per_window = {}
        for name in WINDOWS:
            a, b = (rows[f"h{h:g}"]["windows"][name]["absolute_L2"]["causal_FD_minus_PDE"]
                    for h in (coarse, fine))
            per_window[name] = float(np.log2(a/b)) if a > 0 and b > 0 else None
        scaling[f"{coarse:g}_to_{fine:g}"] = per_window
    _check_resume_provenance(provenance, runtime_provenance(repo))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != provenance["diagnostic_script_sha256"]:
        raise ValueError("Diagnostic script changed during execution")
    write_json(root/"report.json", {"version": "qh-matched-wall-comparison/0.1", "provenance": provenance,
        "source_archive_sha256": sources, "results": rows, "matched_error_norm_scaling": scaling,
        "scaling_scope": "error-norm scaling versus separately computed same-wall FD, not a total Richardson bound",
        "is_total_error_bound": False}, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
