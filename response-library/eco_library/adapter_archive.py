"""Immutable experimental adapter archives, including complex scientific arrays.

This deliberately does not change eco-response-record/0.1 or impersonate the
pending analysis interface. It retains every BS/QH result field for the joint
checkpoint; no measured source, equilibrium or scattering array is discarded.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import numpy as np

from .schema import canonical_json, digest
from .storage import assert_output_path, _publish, _check_resume_provenance, _validate_arrays

VERSION = "eco-adapter-archive/0.1"


def request_id(request):
    if request.get("model") not in ("BS", "QH"):
        raise ValueError("Adapter archive requires explicit BS or QH model")
    return request["model"].lower()+"-"+digest(request)[:24]


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_scalar(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _scalar(v) for k, v in value.items()}
    return value


def _write_tree(parent, name, value, manifest):
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        raise ValueError("Archive dictionary keys must be simple nonempty strings")
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite/non-numeric array: {name}")
        array = np.array(value, copy=False, order="C")
        options = {"compression": "gzip", "shuffle": True} if array.ndim and array.size else {}
        node = parent.create_dataset(name, data=array, **options)
        checksum = hashlib.sha256(array.tobytes()).hexdigest()
        node.attrs.update(node_type="array", sha256=checksum)
        manifest[node.name] = {"sha256": checksum, "shape": list(array.shape), "dtype": array.dtype.str}
    elif isinstance(value, Mapping):
        node = parent.create_group(name)
        node.attrs["node_type"] = "mapping"
        manifest[node.name] = {"mapping_keys": sorted(value)}
        for key, item in value.items():
            _write_tree(node, key, item, manifest)
    else:
        node = parent.create_group(name)
        payload = canonical_json(_scalar(value))
        node.attrs.update(node_type="json", value=payload)
        manifest[node.name] = {"json_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def _read_tree(node, manifest):
    kind = node.attrs["node_type"]
    if kind == "array":
        # HDF5 returns a NumPy scalar for a scalar dataset. Restore the ndarray
        # node type, including ndim=0, so read -> write remains lossless.
        array = np.asarray(node[()])
        checksum = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        actual = {"sha256": checksum, "shape": list(array.shape), "dtype": array.dtype.str}
        if manifest.pop(node.name, None) != actual or node.attrs["sha256"] != checksum:
            raise ValueError("Archive array checksum/shape/type mismatch")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite archived array")
        return array
    if kind == "json":
        raw = node.attrs["value"]
        actual = {"json_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        if manifest.pop(node.name, None) != actual:
            raise ValueError("Archive scalar metadata checksum mismatch")
        value = json.loads(raw)
        canonical_json(value)  # Python's parser otherwise accepts NaN.
        return value
    if kind == "mapping":
        if manifest.pop(node.name, None) != {"mapping_keys": sorted(node.keys())}:
            raise ValueError("Archive mapping membership mismatch")
        return {key: _read_tree(value, manifest) for key, value in node.items()}
    raise ValueError("Unknown archive node type")


def write_archive(path, request, result, provenance, output_root):
    target = assert_output_path(path, output_root)
    rid = request_id(request)
    _validate_arrays(result)
    header = canonical_json({"request": request, "provenance": provenance, "record_id": rid})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs.update(schema_version=VERSION, complete=False, header=header,
                                header_sha256=hashlib.sha256(header.encode()).hexdigest())
            manifest = {}
            _write_tree(handle, "result", result, manifest)
            encoded = canonical_json(manifest)
            handle.attrs.update(manifest=encoded, manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(), complete=True)
            handle.flush()
        _publish(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def read_archive(path, expected_request=None, expected_provenance=None):
    with h5py.File(path, "r") as handle:
        if handle.attrs["schema_version"] != VERSION or not handle.attrs["complete"]:
            raise ValueError("Unsupported/incomplete adapter archive")
        for key in ("header", "manifest"):
            raw = handle.attrs[key]
            if hashlib.sha256(raw.encode()).hexdigest() != handle.attrs[key+"_sha256"]:
                raise ValueError("Archive manifest/header checksum mismatch")
        header = json.loads(handle.attrs["header"])
        canonical_json(header)
        if header["record_id"] != request_id(header["request"]):
            raise ValueError("Archive request identity mismatch")
        if expected_request is not None and header["request"] != expected_request:
            raise ValueError("Archive request mismatch")
        if expected_provenance is not None:
            _check_resume_provenance(header["provenance"], expected_provenance)
        manifest = json.loads(handle.attrs["manifest"])
        result = _read_tree(handle["result"], manifest)
        if manifest:
            raise ValueError("Archive manifest contains missing data nodes")
        _validate_arrays(result)
        return {**header, "result": result}
