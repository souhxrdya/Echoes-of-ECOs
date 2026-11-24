"""Release-verification policy: the cross-runtime tier admits only roundoff-sized changes of derived numbers;
every adversarial change to identities, parameters, states, controls or material numbers fails."""
import copy
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from eco_library import verification as ver
from eco_library.offfiducial import integrity_check

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO/"results"/"catalogue.json"
STORED = json.loads(CATALOGUE.read_text(encoding="utf8"))


def derived_paths(doc):
    out = []
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, float) and ver.is_derived(path):
            out.append(path)
    walk(doc, "")
    return out


def get(doc, path):
    node = doc
    for part in path.strip("/").split("/"):
        name, *idx = part.replace("]", "").split("[")
        node = node[name]
        for i in idx:
            node = node[int(i)]
    return node


def put(doc, path, value):
    parts = path.strip("/").split("/")
    parent = get(doc, "/"+"/".join(parts[:-1])) if len(parts) > 1 else doc
    name, *idx = parts[-1].replace("]", "").split("[")
    if idx:
        node = parent[name]
        for i in idx[:-1]:
            node = node[int(i)]
        node[int(idx[-1])] = value
    else:
        parent[name] = value


def cross(rebuilt):
    return ver.compare(STORED, rebuilt, same_runtime=False)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.doc = copy.deepcopy(STORED)
        self.derived = derived_paths(STORED)

    def test_identical_rebuild_passes_both_tiers(self):
        for same in (True, False):
            r = ver.compare(STORED, self.doc, same_runtime=same)
            self.assertEqual((r["state"], r["byte_identical"]), ("PASS", True))

    def test_derived_paths_are_the_assembly_time_numbers(self):
        self.assertEqual(len(self.derived), 495)
        self.assertFalse(ver.is_derived("/entries[0]/params/sigma_c"))
        self.assertFalse(ver.is_derived("/entries[0]/modes/rows[0]/omega[0]"))
        self.assertFalse(ver.is_derived("/entries[0]/transfer/gate/late/fd_td_relative"))

    def test_roundoff_sized_changes_pass_cross_runtime_and_fail_same_runtime(self):
        rng = random.Random(1)
        for path in rng.sample(self.derived, 41):  # the observed count and size
            x = get(self.doc, path)
            put(self.doc, path, x*(1+rng.choice((-1, 1))*7.9e-16))
        r = cross(self.doc)
        self.assertEqual(r["state"], "PASS")
        self.assertGreater(r["derived_differing_count"], 0)
        self.assertEqual(ver.compare(STORED, self.doc, same_runtime=True)["state"], "FAIL")

    def test_every_derived_number_may_carry_a_bounded_perturbation(self):
        for path in self.derived:
            put(self.doc, path, get(self.doc, path)*(1+1e-14))
        self.assertEqual(cross(self.doc)["state"], "PASS")

    def test_materially_changed_number_fails(self):
        path = next(p for p in self.derived if p.endswith("/late/norm_L2"))
        put(self.doc, path, get(self.doc, path)*(1+1e-9))
        r = cross(self.doc)
        self.assertEqual(r["state"], "FAIL")
        self.assertEqual([x["path"] for x in r["derived_beyond_tolerance"]], [path])

    def test_number_crossing_an_acceptance_threshold_fails(self):
        path = next(p for p in self.derived if p.endswith("/late/budget_fraction_of_norm"))
        put(self.doc, path, .11)
        self.assertEqual(cross(self.doc)["state"], "FAIL")
        order = next(p for p in self.derived if "/refinement/WH/late/p_observed" in p)
        doc = copy.deepcopy(STORED)
        put(doc, order, 4.6)
        self.assertEqual(cross(doc)["state"], "FAIL")

    def test_decision_value_within_tolerance_of_its_threshold_is_not_portable(self):
        path = next(p for p in self.derived if p.endswith("/late/budget_fraction_of_norm"))
        stored = copy.deepcopy(STORED)
        put(stored, path, .1*(1+1e-14))
        r = ver.compare(stored, copy.deepcopy(stored), same_runtime=False)
        self.assertEqual(r["state"], "FAIL")
        self.assertEqual([x["path"] for x in r["decision_margin_below_tolerance"]], [path])

    def test_altered_parameter_or_copied_value_fails_even_by_one_ulp(self):
        for path in ("/entries[0]/params/sigma_c", "/entries[0]/experiment/sigma",
                     "/entries[1]/modes/rows[0]/omega[0]", "/entries[1]/transfer/gate/late/fd_td_relative",
                     "/entries[0]/bh_reference/levels[0]/dt_relative_difference"):
            with self.subTest(path=path):
                doc = copy.deepcopy(STORED)
                put(doc, path, math.nextafter(get(doc, path), math.inf))
                r = cross(doc)
                self.assertEqual(r["state"], "FAIL")
                self.assertEqual([x["path"] for x in r["exact"]], [path])

    def test_altered_identity_or_hash_fails(self):
        for path, value in (("/entries[0]/entry_id", "bs-sigma_c-0.23-E0"),
                            ("/entries[0]/records/BS[2]/identity", "bs-000000000000000000000000"),
                            ("/entries[0]/records/BS[2]/sha256", "0"*64),
                            ("/entries[0]/experiment_fingerprint", "1"*64),
                            ("/blocks/B0/plan_sha256", "2"*64)):
            with self.subTest(path=path):
                doc = copy.deepcopy(STORED)
                put(doc, path, value)
                self.assertEqual(cross(doc)["state"], "FAIL")

    def test_altered_classification_or_state_fails(self):
        for path, value in (("/entries[0]/state", "INCONCLUSIVE"), ("/entries[4]/state", "ACCEPTED_V1"),
                            ("/entries[0]/window_states/late", "ORDER_NOT_CONFIRMED"),
                            ("/entries[0]/refinement/BS/late/state", "AGREEMENT_AT_FLOOR"),
                            ("/entries[0]/observable_resolution/late/state", "UNRESOLVED"),
                            ("/entries[1]/bh_reference/match/solver", False),
                            ("/statements/echo_classification", "CLEAN_ECHO_TRAIN"),
                            ("/counts/ACCEPTED_V1", 15)):
            with self.subTest(path=path):
                doc = copy.deepcopy(STORED)
                put(doc, path, value)
                self.assertEqual(cross(doc)["state"], "FAIL")

    def test_missing_control_or_structure_change_fails(self):
        doc = copy.deepcopy(STORED)
        coverage = doc["entries"][0]["control_coverage"]
        seq = next(iter(coverage))
        coverage[seq].pop(next(iter(coverage[seq])))
        self.assertEqual(cross(doc)["state"], "FAIL")
        doc = copy.deepcopy(STORED)
        entry = next(e for e in doc["entries"] if any(r.get("non_subdominant_controls_relative")
                                                      for r in e["observable_resolution"].values()))
        row = next(r for r in entry["observable_resolution"].values() if r.get("non_subdominant_controls_relative"))
        row["non_subdominant_controls_relative"].pop(next(iter(row["non_subdominant_controls_relative"])))
        self.assertEqual(cross(doc)["state"], "FAIL")
        doc = copy.deepcopy(STORED)
        doc["entries"][1]["modes"]["rows"].pop()
        self.assertEqual(cross(doc)["state"], "FAIL")
        doc = copy.deepcopy(STORED)
        doc["entries"].pop()
        self.assertEqual(cross(doc)["state"], "FAIL")

    def test_type_changes_fail(self):
        doc = copy.deepcopy(STORED)
        doc["entries"][0]["record_length_M"] = int(doc["entries"][0]["record_length_M"])
        self.assertEqual(cross(doc)["state"], "FAIL")
        doc = copy.deepcopy(STORED)
        path = next(p for p in self.derived if p.endswith("/late/p_observed"))
        put(doc, path, None)
        self.assertEqual(cross(doc)["state"], "FAIL")

    def test_runtime_record_must_describe_the_catalogue(self):
        ref = {"reference_runtime": {"x": 1}, "catalogue_sha256": "0"*64}
        r = ver.rebuild_policy(STORED, copy.deepcopy(STORED), ref, "f"*64, {"x": 1})
        self.assertEqual((r["tier"], r["state"]), ("SAME_RUNTIME", "FAIL"))
        ref["catalogue_sha256"] = "f"*64
        self.assertEqual(ver.rebuild_policy(STORED, copy.deepcopy(STORED), ref, "f"*64, {"x": 2})["tier"],
                         "CROSS_RUNTIME")

    def test_stored_reference_describes_the_stored_catalogue(self):
        ref = ver.reference_runtime(REPO/"results"/"catalogue.runtime.json")
        self.assertEqual(ref["catalogue_sha256"], hashlib.sha256(CATALOGUE.read_bytes()).hexdigest())
        self.assertEqual(set(ref["reference_runtime"]), set(ver.runtime_fingerprint()))


class IntegrityTests(unittest.TestCase):
    def test_corrupted_or_missing_archive_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"records").mkdir()
            for name in ("a.h5", "b.h5"):
                (root/"records"/name).write_bytes(bytes(range(256))*8)
            manifest = {f"records/{n}": hashlib.sha256((root/"records"/n).read_bytes()).hexdigest()
                        for n in ("a.h5", "b.h5")}
            self.assertEqual(integrity_check(root, manifest)["state"], "PASS")
            data = bytearray((root/"records"/"a.h5").read_bytes())
            data[100] ^= 1
            (root/"records"/"a.h5").write_bytes(bytes(data))
            r = integrity_check(root, manifest)
            self.assertEqual((r["state"], r["mismatched"]), ("FAIL", ["records/a.h5"]))
            (root/"records"/"b.h5").unlink()
            self.assertEqual(integrity_check(root, manifest)["missing"], ["records/b.h5"])


if __name__ == "__main__":
    unittest.main()
