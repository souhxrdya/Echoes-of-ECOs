"""Side-effect-free, hash-pinned declarations from the frozen TD notebook.

This is a migration adapter, not an execution of the notebook.  Only explicitly
listed declarations are compiled.  No experiment, cache, plot, file writer,
random seed, or notebook import is evaluated.  The exact source of every
declaration is included in the returned provenance manifest.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import time
import types
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import lambertw

NOTEBOOK_SHA256 = "64e873da4b7afbb0f2abec91e4aa077a0493e998854582374af0a7a46c6513b9"
DECLARATIONS = {
    4: ("Config",),
    6: (
        "schwarzschild_rstar", "r_from_schwarzschild_rstar", "rw_potential",
        "_D1_W0", "_D1_W1", "_D2_W0", "_D2_W1", "d1_4", "d2_4",
        "d2_4_regular_center", "initial_characteristics", "evolve_wave",
    ),
    10: ("black_hole_grid",),
    13: ("gravastar_grid",),
    18: ("rw_peak_radius", "wormhole_grid", "throat_delta_coefficients"),
}


class _NumpyCompatibility:
    """Notebook-compatible trapezoid spelling without mutating numpy itself."""

    def __getattr__(self, name: str) -> Any:
        if name == "trapezoid":
            return np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return getattr(np, name)


@dataclass
class FrozenRuntime:
    namespace: dict[str, Any]
    provenance: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.namespace[name]


def default_notebook_path() -> Path:
    from .paths import NOTEBOOK
    return NOTEBOOK


def load_frozen_runtime(notebook_path: str | Path | None = None) -> FrozenRuntime:
    """Verify immutable source and load only the reviewed TD dependency closure.

    A fresh namespace is built on every invocation.  Passing an explicit Config
    to grid builders prevents notebook defaults from leaking between records.
    The temporary module registration is needed by dataclasses during class
    creation, and is removed immediately afterwards; no solver cache survives.
    """
    path = default_notebook_path() if notebook_path is None else Path(notebook_path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != NOTEBOOK_SHA256:
        raise ValueError(f"Frozen notebook SHA-256 mismatch: {actual}; refusing execution")
    document = json.loads(raw)
    module_name = "_eco_frozen_td_" + uuid.uuid4().hex
    module = types.ModuleType(module_name)
    ns = module.__dict__
    ns.update(np=_NumpyCompatibility(), math=math, time=time, warnings=warnings,
              dataclass=dataclass, Any=Any, lambertw=lambertw)
    manifest = []
    sys.modules[module_name] = module
    try:
        for cell_index, allowed in DECLARATIONS.items():
            source = "".join(document["cells"][cell_index]["source"])
            nodes = []
            found = []
            for node in ast.parse(source).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    name = node.name
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                else:
                    continue
                if name not in allowed:
                    continue
                if name in found:
                    raise ValueError(f"Duplicate reviewed declaration {name}")
                found.append(name)
                nodes.append(node)
                snippet = ast.get_source_segment(source, node)
                manifest.append({"name": name, "cell_index": cell_index,
                                 "first_line": node.lineno, "last_line": node.end_lineno,
                                 "source_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest()})
            if set(found) != set(allowed):
                raise ValueError(f"Reviewed declaration set changed in cell {cell_index}")
            code = ast.Module(body=nodes, type_ignores=[])
            exec(compile(code, f"{path.name}:cell{cell_index}", "exec"), ns)
            if cell_index == 4:
                ns["CFG"] = ns["Config"]()
    finally:
        del sys.modules[module_name]
    return FrozenRuntime(ns, {"notebook_sha256": actual,
                              "notebook_file": path.name,
                              "loader": "reviewed_ast_declarations_v1",
                              "declarations": manifest})
