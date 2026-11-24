"""ECO response library: ``python -m eco_library <command>``, run from the response-library folder.

  audit       check the committed results without running anything (seconds): the frozen notebook is
              unchanged, every entry state of results/catalogue.json follows from its stored evidence,
              and every file in results/entries matches the index and its catalogue entry
  reproduce   recompute the library from the frozen equations into runs/ (resumable; see
              eco_library.pipeline).  Options: --workers N (default 2), --budget SECONDS (stop after
              this long; run the same command again to continue)
  compare     compare the reproduction in runs/ with results/ (after reproduce)
  info        versions, pinned references and the entry states

Lower-level commands forward their arguments to one module (``--help`` for details):
  offfiducial  plan/run/report/verify/reproduce one block of time-domain jobs   (eco_library.offfiducial)
  robustness   source-width and observer experiments                          (eco_library.robustness)
  transfer     frequency-domain transfer validation of a GS or WH point        (eco_library.transfer)
  reflection   reflection coefficients of the GS and WH points                (eco_library.reflection)
  catalogue    assemble or audit a catalogue                                  (eco_library.catalogue)
  entries      write or index the per-entry result files                      (eco_library.interface_export)
Modes of one point: ``python -m eco_library.modes point MODEL VALUE OUT.json``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from .paths import RESULTS, RUNS

FORWARD = {"offfiducial": "offfiducial", "robustness": "robustness", "transfer": "transfer",
           "reflection": "reflection", "catalogue": "catalogue", "entries": "interface_export"}
CATALOGUE = RESULTS/"catalogue.json"
INDEX = RESULTS/"entries-index.json"
ENTRIES = RESULTS/"entries"


def environment():
    import h5py
    import numpy
    import scipy
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "h5py": h5py.__version__}


def info():
    from .baseline_runtime import NOTEBOOK_SHA256
    from .interface_export import VERSION
    catalogue = json.loads(CATALOGUE.read_text(encoding="utf8"))
    return {"catalogue_format": catalogue["catalogue_version"], "counts": catalogue["counts"],
            "entries": {e["entry_id"]: e["state"] for e in catalogue["entries"]},
            "entry_file_format": VERSION, "frozen_notebook_sha256": NOTEBOOK_SHA256, "environment": environment()}


def audit():
    """PASS/FAIL per step; reads only committed files."""
    from . import catalogue as cat
    from . import interface_export as ie
    from .baseline_runtime import NOTEBOOK_SHA256, default_notebook_path
    steps = {}
    actual = hashlib.sha256(default_notebook_path().read_bytes()).hexdigest()
    steps["frozen_notebook"] = {"state": "PASS" if actual == NOTEBOOK_SHA256 else "FAIL", "sha256": actual}
    catalogue = json.loads(CATALOGUE.read_text(encoding="utf8"))
    problems = cat.status_audit(catalogue)
    steps["catalogue_status_audit"] = {"state": "PASS" if not problems else "FAIL", "problems": problems}
    index = json.loads(INDEX.read_text(encoding="utf8"))
    entries = {e["entry_id"]: e for e in catalogue["entries"]}
    problems, files = [], {}
    if index["catalogue_sha256"] != hashlib.sha256(CATALOGUE.read_bytes()).hexdigest():
        problems.append("the index was built from a different catalogue file")
    if set(index["exports"]) != set(entries):
        problems.append("index and catalogue list different entries")
    for entry_id, row in index["exports"].items():
        path = ENTRIES/row["file"]
        try:
            header, arrays = ie.read_export(path)
            found = ie.audit_export(header, arrays, entries[entry_id])
            if ie.index_row(path, header) != row:
                found.append("file differs from its index row")
            files[entry_id] = found or "PASS"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            files[entry_id] = [f"{type(exc).__name__}: {exc}"]
    steps["entry_files"] = {"state": "PASS" if not problems and all(v == "PASS" for v in files.values()) else "FAIL",
                            "problems": problems, "files": files}
    return {"state": "PASS" if all(s["state"] == "PASS" for s in steps.values()) else "FAIL", "steps": steps}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command in FORWARD:
        import importlib
        return importlib.import_module(f"eco_library.{FORWARD[command]}").main(rest)
    if command == "info":
        print(json.dumps(info(), indent=1, sort_keys=True))
        return 0
    if command == "audit":
        result = audit()
        print(json.dumps({"state": result["state"], **{k: v["state"] for k, v in result["steps"].items()}}))
        for name, step in result["steps"].items():
            if step["state"] != "PASS":
                print(name, json.dumps(step, indent=1), file=sys.stderr)
        n = len(result["steps"]["entry_files"]["files"])
        print(f"{n} entry files checked against the catalogue and the index")
        return 0 if result["state"] == "PASS" else 2
    if command == "reproduce":
        import argparse
        from . import pipeline
        parser = argparse.ArgumentParser(prog="python -m eco_library reproduce")
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--budget", type=float, default=float("inf"), help="seconds")
        parser.add_argument("--runs", type=Path, default=RUNS)
        args = parser.parse_args(rest)
        return 0 if pipeline.reproduce(args.runs, args.budget, args.workers) else 3
    if command == "compare":
        import argparse
        from . import pipeline
        parser = argparse.ArgumentParser(prog="python -m eco_library compare")
        parser.add_argument("--runs", type=Path, default=RUNS)
        parser.add_argument("--report", type=Path)
        args = parser.parse_args(rest)
        result = pipeline.compare(args.runs)
        result["environment"] = environment()
        if args.report:
            args.report.write_text(json.dumps(result, indent=1, sort_keys=True)+"\n", encoding="utf8")
        n = sum(len(v) for v in result["entries"].values())
        print(json.dumps({"state": result["state"], "catalogue_differences": len(result["catalogue"]),
                          "entry_files_compared": len(result["entries"]), "entry_differences": n,
                          "status_audit_problems": len(result["status_audit"])}))
        for line in result["catalogue"][:20] + [f"{k}: {d}" for k, v in result["entries"].items() for d in v][:20]:
            print("  ", line)
        return 0 if result["state"] == "PASS" else 2
    print(f"unknown command {command!r}\n\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
