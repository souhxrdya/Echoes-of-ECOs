"""Hash-pinned loader for further reviewed declarations of the frozen notebook.

Same rules as ``baseline_runtime.load_frozen_runtime``: the whole-notebook SHA-256 must
match, only the listed top-level function/class/assignment declarations of the listed
cells are compiled (no experiment, plot or writer is executed), and every compiled
snippet's SHA-256 is returned in the provenance manifest.  The TD closure is loaded
first, so these declarations see the reviewed ``rw_potential``, tortoise maps and
evolution code.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
import uuid
from pathlib import Path

from .baseline_runtime import NOTEBOOK_SHA256, FrozenRuntime, default_notebook_path, load_frozen_runtime


def load_declarations(cells, extra_namespace=None, notebook_path=None, label="extra", base=None):
    """cells: {cell_index: (name, ...)}; returns a FrozenRuntime with ``base`` (default: the TD
    closure) plus these declarations."""
    rt = load_frozen_runtime(notebook_path) if base is None else base
    path = default_notebook_path() if notebook_path is None else Path(notebook_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Frozen notebook SHA-256 mismatch before declaration load")
    document = json.loads(raw)
    module_name = f"_eco_frozen_{label}_" + uuid.uuid4().hex
    module = types.ModuleType(module_name)
    ns = module.__dict__
    ns.update(rt.namespace)
    ns.update(extra_namespace or {})
    ns["__name__"] = module_name
    manifest = list(rt.provenance["declarations"])
    sys.modules[module_name] = module
    try:
        for index, allowed in cells.items():
            source = "".join(document["cells"][index]["source"])
            nodes, found = [], []
            for node in ast.parse(source).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    name = node.name
                elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name)):
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
                manifest.append({"name": name, "cell_index": index, "first_line": node.lineno,
                                 "last_line": node.end_lineno,
                                 "source_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest()})
            if set(found) != set(allowed):
                raise ValueError(f"Reviewed declaration set changed in cell {index}: missing "
                                 f"{sorted(set(allowed)-set(found))}")
            exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{path.name}:cell{index}", "exec"), ns)
    finally:
        del sys.modules[module_name]
    return FrozenRuntime(ns, {**rt.provenance, "loader": f"reviewed_ast_declarations_v1+{label}",
                              "declarations": manifest})
