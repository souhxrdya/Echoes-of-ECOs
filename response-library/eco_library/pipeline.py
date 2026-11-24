"""Reproduce the response library from the frozen equations, and compare a reproduction with the
committed results.

``reproduce`` writes everything under ``runs/`` (not tracked by Git), in this order:

1. blocks B0-B3 (``runs/blocks/``): every time-domain job of the declared plan of the block (three
   resolutions of each object and of its black-hole reference, and the controls), then the block
   report, its verification manifest and its summary (``runs/summaries/block-B*.json``);
2. per block, the frequency-domain transfer validation of the GS and WH points and the complex modes of
   the GS, WH and BS points (``runs/transfer-modes/``);
3. the base catalogue (``runs/summaries/base-catalogue.json``);
4. the reflection coefficients of the GS and WH points and their summary (``runs/reflection/``);
5. the catalogue (``runs/catalogue.json``); the studies in ``results/checks/`` are its other inputs;
6. one result file per entry (``runs/entries/``) and their index (``runs/entries-index.json``).

Each completed step is recorded in ``runs/progress.json``; an interrupted reproduction continues where
it stopped when the same command is run again.  ``compare`` then checks the reproduction against
``results/``: every number and array must agree (exactly on the computer that produced the committed
results, within 1e-12 relative elsewhere); only provenance fields, which name files, checksums, times
and the producing environment, are excluded, and they are listed in ``PROVENANCE_KEYS``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from .paths import LIBRARY_ROOT, RESULTS, RUNS

BLOCK_IDS = ("B0", "B1", "B2", "B3")
RTOL = 1e-12
STUDY_INPUTS = ("robustness-report-summary.json", "transfer-gate-review.json", "bs-mode-matching-study.json",
                "qh-1e6-study.json", "waveform-audit.json", "edge-robustness-o120.json",
                "record-length-requirements.json")
# Keys whose values say where, when or by what a number was produced, not the number itself: file names,
# checksums of files that also contain such provenance, the producing commit, the root folder of a block
# and how many archive files its integrity check hashed (which depends on how many sessions ran it).
# Every other key is compared; references to documents inside strings are compared after replacing the
# document name by "docs/<document>".
PROVENANCE_KEYS = frozenset({"file", "file_in_1_0", "sha256", "archive", "archive_sha256", "file_sha256",
                             "study_sha256", "plan_sha256", "git_commit", "git_dirty", "root", "checked", "matched"})
_DOC = re.compile(r"docs/[A-Za-z0-9_.]+\.md(?: \([^)]*\))?")


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    tmp.replace(path)


def _script(name):
    path = LIBRARY_ROOT/"scripts"/name
    spec = importlib.util.spec_from_file_location("_script_"+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drop_unreadable(paths, reader):
    """Remove output files that an interrupted step left incomplete, so that they are computed again."""
    for path in list(paths):
        try:
            reader(path)
        except Exception:
            path.unlink()


def _block_points(block):
    from .catalogue_base import BLOCKS
    return BLOCKS[block]


def _outstanding(root):
    from . import offfiducial as off
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    return [label for label, job in plan["jobs"].items() if not off.output_path(root, job).exists()]


class _Progress:
    def __init__(self, runs):
        self.path = runs/"progress.json"
        self.done = json.loads(self.path.read_text(encoding="utf8"))["done"] if self.path.exists() else []

    def mark(self, step):
        self.done.append(step)
        _write_json(self.path, {"done": self.done})


def _steps(runs, workers):
    """(name, function); a function returns True when its step is complete."""
    from . import catalogue, catalogue_base, interface_export as ie, modes, offfiducial as off, reflection, transfer
    steps = []
    for block in BLOCK_IDS:
        root = runs/"blocks"/block

        def plan(root=root, block=block):
            off.main(["plan", "--output", str(root)] + ([] if block == "B0" else ["--block", block]))
            return True

        def run(budget, root=root):
            if _outstanding(root):
                off.main(["run", "--output", str(root), "--budget", str(budget), "--workers", str(workers)])
            return not _outstanding(root)

        def report(root=root, block=block):
            off.main(["report", "--output", str(root)])
            if off.main(["verify", "--output", str(root)]) != 0:
                raise RuntimeError(f"block {block}: the verification of its archive failed")
            latest = lambda d: sorted((root/d).glob("*.json"))[-1]
            _script("summarize_offfiducial_run.py").main(
                root, latest("reports"), latest("manifests"), runs/"summaries"/f"block-{block}.json")
            return True

        def transfer_point(model, root=root, block=block):
            out = runs/"transfer-modes"
            out.mkdir(parents=True, exist_ok=True)
            for h5 in out.glob("transfer-*.h5"):
                if not h5.with_suffix(".json").exists():
                    h5.unlink()                                   # interrupted while writing
            done = [json.loads(p.read_text(encoding="utf8")) for p in out.glob("transfer-*.json")]
            if not any(d["root_name"] == root.name and d["model"] == model for d in done):
                summary = transfer.validate(root, model, out)
                if summary["state"] != "VALIDATED_AGAINST_TD":
                    raise RuntimeError(f"block {block}: {model} transfer state {summary['state']}")
            return True

        def modes_point(model, block=block):
            out = runs/"transfer-modes"
            out.mkdir(parents=True, exist_ok=True)
            value = float(next(iter(_block_points(block)[model].values())))
            modes.compute_point(model, value, out/f"{model.lower()}-{value!r}.json")
            return True

        steps += [(f"{block} plan", plan), (f"{block} time-domain jobs", run), (f"{block} report and summary", report)]
        steps += [(f"{block} transfer {m}", lambda m=m, f=transfer_point: f(m)) for m in ("GS", "WH")]
        steps += [(f"{block} modes {m}", lambda m=m, f=modes_point: f(m)) for m in ("GS", "WH", "BS")]

    def base():
        for name in STUDY_INPUTS:
            shutil.copy2(RESULTS/"checks"/name, runs/"summaries"/name)
        _write_json(runs/"summaries"/catalogue.BASE_FILE, catalogue_base.assemble(runs, runs/"summaries"))
        return True

    def reflect():
        base_file = runs/"summaries"/catalogue.BASE_FILE
        out = runs/catalogue.REFLECTION_DIR
        _drop_unreadable(out.glob("*.h5"), reflection.read)
        entries = [e["entry_id"] for e in json.loads(base_file.read_text(encoding="utf8"))["entries"]
                   if e["model"] in ("GS", "WH") and not (out/f"{e['entry_id']}.h5").exists()]
        if entries:
            reflection.main(["--catalogue", str(base_file), "--output", str(out), "--entries", *entries])
        _write_json(runs/"summaries"/catalogue.INPUTS["reflection"], reflection.summarize(out))
        return True

    def final():
        from . import verification as ver
        path = runs/catalogue.FILE
        _write_json(path, catalogue.assemble(runs, runs/"summaries"))
        _write_json(catalogue_base.runtime_reference_path(path), {
            "basis": "the runtime that assembled this catalogue; a byte-identical rebuild in this runtime is the "
                     "SAME_RUNTIME tier of eco_library.verification",
            "catalogue": "catalogue.json", "catalogue_sha256": ver.file_sha256(path), "policy": ver.POLICY_VERSION,
            "reference_runtime": ver.runtime_fingerprint()})
        return True

    def entries():
        cat_path = runs/catalogue.FILE
        doc = json.loads(cat_path.read_text(encoding="utf8"))
        out = runs/"entries"
        reports, rows = ie._Reports(), {}
        _drop_unreadable(out.glob("*.h5"), ie.read_export)
        for entry in doc["entries"]:
            path = out/f"{entry['entry_id']}.h5"
            if not path.exists():
                ie.export_entry(entry, runs, out, reports)
            header, _ = ie.read_export(path)
            rows[entry["entry_id"]] = ie.index_row(path, header)
        _write_json(runs/"entries-index.json", {
            "version": ie.VERSION, "catalogue_sha256": hashlib.sha256(cat_path.read_bytes()).hexdigest(),
            "exports": rows, "regenerate": "python -m eco_library reproduce (entries are written by its last step); "
                                          "array checksums are in each file's manifest"})
        return True

    return steps + [("base catalogue", base), ("reflection coefficients", reflect), ("catalogue", final),
                    ("entry files", entries)]


def reproduce(runs=RUNS, budget=float("inf"), workers=2):
    """Run the outstanding steps until done or until ``budget`` seconds have passed; True when complete."""
    runs = Path(runs)
    (runs/"summaries").mkdir(parents=True, exist_ok=True)
    progress, start = _Progress(runs), time.perf_counter()
    for name, step in _steps(runs, workers):
        if name in progress.done:
            continue
        while True:
            left = budget-(time.perf_counter()-start)
            if left <= 0:
                print(f"time budget used; next step: {name}. Run the same command again to continue.", flush=True)
                return False
            print(f"== {name}", flush=True)
            complete = step(min(left, 1e9)) if name.endswith("time-domain jobs") else step()
            if complete:
                progress.mark(name)
                break
    print("reproduction complete: compare it with the committed results by `python -m eco_library compare`",
          flush=True)
    return True


# ------------------------------------------------------------------------------------- compare

def _compare(a, b, path, out, rtol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in PROVENANCE_KEYS:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                _compare(a[k], b[k], f"{path}/{k}", out, rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} and {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _compare(x, y, f"{path}[{i}]", out, rtol)
    elif isinstance(a, str) and isinstance(b, str):
        if _DOC.sub("docs/<document>", a) != _DOC.sub("docs/<document>", b):
            out.append(f"{path}: {a[:80]!r} != {b[:80]!r}")
    elif isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b or type(a) is not type(b):
            out.append(f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}")
    elif a != b and abs(a-b) > rtol*max(abs(a), abs(b)):
        out.append(f"{path}: {a!r} != {b!r}")


def _compare_arrays(a, b, path, out, rtol):
    if isinstance(a, dict):
        if set(a) != set(b):
            out.append(f"{path}: different arrays {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            _compare_arrays(a[k], b[k], f"{path}/{k}", out, rtol)
        return
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        out.append(f"{path}: shape/dtype {a.shape} {a.dtype} != {b.shape} {b.dtype}")
    elif not np.array_equal(a, b):
        scale = max(float(np.max(np.abs(a))) if a.size else 0., float(np.max(np.abs(b))) if b.size else 0.)
        worst = float(np.max(np.abs(a-b)))
        if a.dtype.kind not in "fc" or worst > rtol*scale:
            out.append(f"{path}: max difference {worst:.3e} at scale {scale:.3e}")


def compare(runs=RUNS, results=RESULTS, rtol=RTOL):
    """Differences between a reproduction and the committed results (empty lists: agreement)."""
    from . import catalogue, interface_export as ie
    runs, results = Path(runs), Path(results)
    out = {"catalogue": [], "entries": {}, "status_audit": [], "entry_audit": {}}
    mine = json.loads((runs/catalogue.FILE).read_text(encoding="utf8"))
    stored = json.loads((results/catalogue.FILE).read_text(encoding="utf8"))
    _compare(mine, stored, "catalogue", out["catalogue"], rtol)
    out["status_audit"] = catalogue.status_audit(mine)
    entries = {e["entry_id"]: e for e in mine["entries"]}
    for path in sorted((results/"entries").glob("*.h5")):
        diffs = []
        h1, a1 = ie.read_export(runs/"entries"/path.name)
        h2, a2 = ie.read_export(path)
        _compare(h1, h2, "header", diffs, rtol)
        _compare_arrays(a1, a2, "arrays", diffs, rtol)
        out["entries"][path.stem] = diffs
        out["entry_audit"][path.stem] = ie.audit_export(h1, a1, entries[path.stem])
    out["state"] = ("PASS" if not out["catalogue"] and not out["status_audit"] and len(out["entries"]) == len(entries)
                    and not any(out["entries"].values()) and not any(out["entry_audit"].values()) else "FAIL")
    return out
