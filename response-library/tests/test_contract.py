import tempfile
from pathlib import Path
import pickle
import unittest

import h5py
import numpy as np

from eco_library.schema import Experiment, RunSpec, canonical_json
from eco_library.storage import (write_record, read_record, write_json, assert_output_path,
                                 RESUME_PROVENANCE_KEYS)
from eco_library.convergence import assess


class ContractTests(unittest.TestCase):
    @staticmethod
    def result():
        return {"t": np.arange(10.), "psi": np.zeros(10), "x": np.arange(10.),
                "r": np.arange(10.), "V": np.zeros(10), "metadata": {}, "checks": {}}

    def test_identity(self):
        a = RunSpec("GS", {"compactness": .47})
        b = RunSpec("GS", {"compactness": .47}, dx=.05)
        self.assertNotEqual(a.record_id, b.record_id)
        self.assertEqual(a.point_id, b.point_id)
        self.assertEqual(a.experiment.fingerprint, b.experiment.fingerprint)
        self.assertNotEqual(Experiment(sigma=3).fingerprint, a.experiment.fingerprint)
        self.assertEqual(RunSpec.from_dict(a.to_dict()), a)

    def test_params_are_immutable_and_pickle_safe(self):
        params = {"compactness": .47}
        spec = RunSpec("GS", params)
        identities = spec.point_id, spec.record_id
        params["compactness"] = .48
        self.assertEqual(spec.params["compactness"], .47)
        with self.assertRaises(TypeError):
            spec.params["compactness"] = .49
        copied = spec.to_dict()
        copied["params"]["compactness"] = .48
        self.assertEqual((spec.point_id, spec.record_id), identities)
        self.assertEqual(pickle.loads(pickle.dumps(spec)), spec)
        self.assertEqual(RunSpec("WH", {"r0_over_M": 3}).record_id,
                         RunSpec("WH", {"r0_over_M": 3.0}).record_id)

    def test_invalid(self):
        for make in (lambda: RunSpec("BS"), lambda: RunSpec("GS", {"compactness": .5}),
                     lambda: RunSpec("WH", {"r0_over_M": 2}), lambda: RunSpec("BH", dx=float("nan")),
                     lambda: Experiment(amplitude=2)):
            with self.assertRaises(ValueError):
                make()
        with self.assertRaises(ValueError):
            canonical_json({"error": float("nan")})

    def test_boolean_numeric_inputs_rejected(self):
        for field in ("dx", "t_end", "cfl", "x_left", "x_right", "sample_every"):
            with self.subTest(run_field=field), self.assertRaises(ValueError):
                RunSpec("BH", **{field: True})
        for field in ("M_ADM", "ell", "source_global_rstar", "observer_global_rstar",
                      "sigma", "amplitude", "F_in", "t0"):
            with self.subTest(experiment_field=field), self.assertRaises(ValueError):
                Experiment(**{field: True})

    def test_storage_and_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"out"
            p = root/"bh.h5"
            spec = RunSpec("BH")
            result = self.result()
            write_record(p, spec, result, {}, root)
            self.assertEqual(read_record(p, spec)["spec"], spec)
            with self.assertRaises(FileExistsError):
                write_record(p, spec, result, {}, root)
            with self.assertRaises(ValueError):
                assert_output_path(root/"../bad.json", root)
            with self.assertRaises(ValueError):
                assert_output_path(root/"baseline"/"bad.h5", root)
            write_json(root/"report.json", {"ok": True}, root)
            with self.assertRaises(FileExistsError):
                write_json(root/"report.json", {}, root)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_stored_identity_and_checksums(self):
        for attr in ("record_id", "point_id", "experiment_fingerprint"):
            with self.subTest(attr=attr), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                p = root/"record.h5"
                spec = RunSpec("BH")
                write_record(p, spec, self.result(), {}, root)
                with h5py.File(p, "r+") as f:
                    f.attrs[attr] = "wrong"
                with self.assertRaises(ValueError):
                    read_record(p, spec)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root/"record.h5"
            write_record(p, RunSpec("BH"), self.result(), {}, root)
            with h5py.File(p, "r+") as f:
                f["waveform/psi"][0] = 1
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_record(p)

    def test_resume_provenance_policy(self):
        stored = {k: "v1" for k in RESUME_PROVENANCE_KEYS}
        stored["source_sha256"] = {"solvers.py": "abc"}
        stored.update(git_commit="old", git_dirty=True, platform="host-a")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root/"record.h5"
            spec = RunSpec("BH")
            write_record(p, spec, self.result(), stored, root)
            expected = dict(stored, git_commit="new", git_dirty=False, platform="host-b")
            read_record(p, spec, expected_provenance=expected)
            for key in RESUME_PROVENANCE_KEYS:
                with self.subTest(key=key):
                    changed = dict(expected)
                    changed[key] = "different"
                    with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                        read_record(p, spec, expected_provenance=changed)
            with self.assertRaisesRegex(ValueError, "Missing resume provenance"):
                read_record(p, spec, expected_provenance={})

    def test_invalid_arrays_do_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for key, value in (("psi", np.full(10, np.nan)),
                               ("psi", np.ones(10, dtype=complex)),
                               ("t", np.arange(10.)[::-1]),
                               ("x", np.arange(10.)[::-1]),
                               ("t", np.arange(10, dtype=np.uint64)[::-1]),
                               ("x", np.arange(10, dtype=np.uint64)[::-1]),
                               ("psi", np.ones(10, dtype=bool)),
                               ("r", np.arange(10).astype(str)),
                               ("V", np.arange(10).astype(object))):
                with self.subTest(key=key):
                    result = self.result()
                    result[key] = value
                    with self.assertRaises(ValueError):
                        write_record(root/"invalid.h5", RunSpec("BH"), result, {}, root)
                    self.assertFalse((root/"invalid.h5").exists())

    def test_nonfinite_metadata_rejected_on_write_and_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root/"record.h5"
            result = self.result()
            result["metadata"] = {"error": float("nan")}
            with self.assertRaises(ValueError):
                write_record(p, RunSpec("BH"), result, {}, root)
            self.assertFalse(p.exists())
            write_record(p, RunSpec("BH"), self.result(), {}, root)
            with h5py.File(p, "r+") as f:
                f.attrs["metadata_json"] = '{"error":NaN}'
            with self.assertRaises(ValueError):
                read_record(p)

    def test_refinement_semantics(self):
        t = np.linspace(0, 1, 101)
        h = [.2, .1, .05]
        y0 = np.sin(t)
        ys = [y0 + q*q*np.cos(t) for q in h]
        a = assess(t, ys, h, expected_order=2, windows={"w": (0, 1)})["w"]
        self.assertAlmostEqual(a["p_observed"], 2, places=10)
        self.assertEqual(a["state"], "OBSERVED_ORDER_CONSISTENT")
        self.assertFalse(a["is_error_bound"])
        b = assess(t, [y0]*3, h, expected_order=2, windows={"w": (0, 1)})["w"]
        self.assertEqual(b["state"], "AGREEMENT_AT_FLOOR")
        self.assertIsNone(b["fine_error_estimate_L2"])
        with self.assertRaises(ValueError):
            assess(t, ys, [.2, .1, .051], expected_order=2, windows={"w": (0, 1)})


if __name__ == "__main__":
    unittest.main()
