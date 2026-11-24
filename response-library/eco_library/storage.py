"""Private per-record HDF5 files, atomically published without replacement."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import uuid

import h5py
import numpy as np
import scipy

from . import SCHEMA_VERSION
from .schema import canonical_json, RunSpec


# Resume is scientific reuse, not merely a matching filename.  Ignore the Git
# parent, dirty marker and host platform (reporting fields), but fail closed if
# executable sources, frozen physics source, interpreter or numerical libraries
# differ.  Cross-version reuse requires an explicit new compatibility policy.
RESUME_PROVENANCE_KEYS = (
    "source_sha256", "baseline_notebook_sha256", "python", "numpy", "scipy", "h5py",
)


def assert_output_path(path, output_root):
    root = Path(output_root).resolve()
    target = Path(path).resolve()
    # No path in any baseline tree, including a symlink into one, is writable.
    if any(p.casefold() == "baseline" for p in (*root.parts, *target.parts)):
        raise ValueError("Refusing write under baseline/")
    if target == root or root not in target.parents:
        raise ValueError("Output must be a file strictly inside the explicit output root")
    return target


def runtime_provenance(repo):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    source_files = sorted((Path(repo) / "eco_library").glob("*.py"))
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    from .paths import NOTEBOOK as notebook
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if (status := git("status", "--porcelain")) is None else bool(status), "source_sha256": sources,
            "baseline_notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__}


def _validate_arrays(result):
    t, psi = np.asarray(result["t"]), np.asarray(result["psi"])
    if t.ndim != 1 or len(t) < 2 or t.shape != psi.shape:
        raise ValueError("Invalid waveform dimensions")
    if (t.dtype.kind not in "iuf" or psi.dtype.kind not in "iuf"
            or not np.isfinite(t).all() or not np.isfinite(psi).all()
            or not np.all(t[1:] > t[:-1])):
        raise ValueError("Waveforms must be finite and real, with time strictly increasing")
    for name in ("x", "r", "V"):
        a = np.asarray(result[name])
        if (a.ndim != 1 or a.dtype.kind not in "iuf" or not np.isfinite(a).all()
                or len(a) < 2 or len(a) != len(result["x"])):
            raise ValueError(f"Invalid background {name}")
    x = np.asarray(result["x"])
    if not np.all(x[1:] > x[:-1]):
        raise ValueError("Background coordinate must be strictly increasing")
    return t, psi


def _check_resume_provenance(stored, expected):
    for key in RESUME_PROVENANCE_KEYS:
        if key not in stored or key not in expected:
            raise ValueError(f"Missing resume provenance: {key}")
        if stored[key] != expected[key]:
            raise ValueError(f"Resume provenance mismatch: {key}")


def _publish(temp, target):
    # Hard-link publication fails atomically if target exists, on Windows and Linux.
    # Both paths are in one directory/filesystem. No unlink of pre-existing targets.
    os.link(temp, target)


def write_json(path, value, output_root):
    target = assert_output_path(path, output_root)
    payload = canonical_json(value).encode("utf-8") + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def write_record(path, spec, result, provenance, output_root):
    target = assert_output_path(path, output_root)
    t, psi = _validate_arrays(result)
    # Validate metadata BEFORE opening a file; missing quantities must be explicit.
    metadata = canonical_json(result["metadata"])
    checks = canonical_json(result["checks"])
    provenance_json = canonical_json(provenance)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with h5py.File(temp, "x") as f:
            f.attrs.update(schema_version=SCHEMA_VERSION, record_id=spec.record_id,
                           point_id=spec.point_id, experiment_fingerprint=spec.experiment.fingerprint,
                           spec_json=canonical_json(spec.to_dict()), metadata_json=metadata,
                           checks_json=checks, provenance_json=provenance_json,
                           numerics_evidence="UNQUANTIFIED", record_complete=False)
            for group, arrays in (("waveform", {"time": t, "psi": psi}),
                                  ("background", {n: result[n] for n in ("x", "r", "V")})):
                g = f.create_group(group)
                for name, data in arrays.items():
                    d = g.create_dataset(name, data=data, compression="gzip", shuffle=True)
                    d.attrs["sha256"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
            for name in ("modes", "reflectivity", "classification"):
                f.create_group(name).attrs["status"] = "NOT_COMPUTED"
            f.attrs["record_complete"] = True
            f.flush()
        _publish(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target


def read_record(path, expected_spec=None, expected_provenance=None):
    """Read and validate a record; require provenance compatibility for resume.

    A plain analysis read may omit ``expected_provenance``.  A resumed job must
    supply the current ``runtime_provenance``; all RESUME_PROVENANCE_KEYS then
    match exactly.  Git commit/dirty state and host platform are not gates.
    """
    with h5py.File(path, "r") as f:
        if f.attrs["schema_version"] != SCHEMA_VERSION or not f.attrs["record_complete"]:
            raise ValueError("Incompatible/incomplete record")
        spec = RunSpec.from_dict(json.loads(f.attrs["spec_json"]))
        if f.attrs["record_id"] != spec.record_id or (expected_spec and spec != expected_spec):
            raise ValueError("Record identity mismatch")
        if f.attrs["point_id"] != spec.point_id:
            raise ValueError("Point identity mismatch")
        if f.attrs["experiment_fingerprint"] != spec.experiment.fingerprint:
            raise ValueError("Experiment fingerprint mismatch")
        out = {"spec": spec, "metadata": json.loads(f.attrs["metadata_json"]),
               "checks": json.loads(f.attrs["checks_json"]),
               "provenance": json.loads(f.attrs["provenance_json"])}
        # json.loads accepts NaN/Infinity by default, even though our writer does
        # not.  A foreign or damaged record must not silently reintroduce them.
        for key in ("metadata", "checks", "provenance"):
            canonical_json(out[key])
        if expected_provenance is not None:
            _check_resume_provenance(out["provenance"], expected_provenance)
        for group, mapping in (("waveform", {"time": "t", "psi": "psi"}),
                               ("background", {n: n for n in ("x", "r", "V")})):
            for key, name in mapping.items():
                a = f[group][key][:]
                if hashlib.sha256(a.tobytes()).hexdigest() != f[group][key].attrs["sha256"]:
                    raise ValueError("Array checksum mismatch")
                out[name] = a
        _validate_arrays(out)
        return out
