"""Result files: every status in a file is re-derived from its own arrays and its catalogue entry; each
status that its evidence does not support is rejected."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from eco_library import interface_export as ie
from eco_library import reflection

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = json.loads((REPO/"results"/"catalogue.json").read_text(encoding="utf8"))
ENTRIES = {e["entry_id"]: e for e in CATALOGUE["entries"]}
EXPORTS = REPO/"results"/"entries"
AVAILABLE = all((EXPORTS/f"{k}.h5").exists() for k in ENTRIES)


class DelayStates(unittest.TestCase):
    def test_states_follow_the_declared_rule(self):
        R = np.array([.5, .5, 1e-4, .5])
        tau = np.array([10., 10., 10., np.nan])
        sigma = np.array([5e-6, 2e-5, 0., 0.])
        self.assertEqual(list(reflection.delay_states(R, tau, sigma)),
                         ["RESOLVED", "UNRESOLVED", "PHASE_ILL_CONDITIONED", "UNDEFINED"])


@unittest.skipUnless(AVAILABLE, "result files missing")
class AuditOnStoredExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {k: ie.read_export(EXPORTS/f"{k}.h5") for k in
                     ("gs-compactness-0.47-E0", "wh-r0_over_M-2.0005-E0", "qh-EplM-1e+10-E0", "bs-sigma_c-0.22-E0")}

    def audit(self, entry_id, mutate=None):
        header, arrays = copy.deepcopy(self.files[entry_id])
        entry = copy.deepcopy(ENTRIES[entry_id])
        if mutate:
            mutate(header, arrays, entry)
        return ie.audit_export(header, arrays, entry)

    def assertRejected(self, entry_id, mutate, needle):
        problems = self.audit(entry_id, mutate)
        self.assertTrue(any(needle in p for p in problems), problems)

    def test_every_stored_export_passes(self):
        for k in ENTRIES:
            header, arrays = ie.read_export(EXPORTS/f"{k}.h5")
            self.assertEqual(header["version"], ie.VERSION)
            self.assertEqual(ie.audit_export(header, arrays, ENTRIES[k]), [], k)

    def test_state_and_identity(self):
        self.assertRejected("gs-compactness-0.47-E0", lambda h, a, e: h.update(catalogue_state="REJECTED"), "catalogue state")
        self.assertRejected("qh-EplM-1e+10-E0", lambda h, a, e: h["params"].update(gamma=2.), "do not recompute")

    def test_qh_residual_is_the_stored_correction(self):
        self.assertRejected("qh-EplM-1e+10-E0", lambda h, a, e: h["residual"].update(source="ECO_MINUS_MATCHED_BH"),
                            "stored correction")

        def m(h, a, e):
            a["common"]["correction"][2, 3000] += 1e-9
        self.assertRejected("qh-EplM-1e+10-E0", m, "not consistent")

    def test_reference_flags_and_mode(self):
        self.assertRejected("gs-compactness-0.47-E0",
                            lambda h, a, e: h["bh_reference"].update(analysis_mode="CO_COMPUTED_SAME_SOLVER"),
                            "without all five")
        self.assertRejected("qh-EplM-1e+10-E0", lambda h, a, e: h["bh_reference"]["match"].update(grid=False),
                            "derived flags")

    def test_reflectivity(self):
        def m(h, a, e):
            v = h["reflectivity"]["arrays"]["R_barrier"]["variant_order"][0]
            a["reflectivity"]["R_barrier"]["variants"][v] = a["reflectivity"]["R_barrier"]["R"]+2e-6
        self.assertRejected("wh-r0_over_M-2.0005-E0", m, "evidence contradicts")

        def n(h, a, e):
            codes = a["reflectivity"]["R_interface"]["tau_state_code"]
            k = int(np.flatnonzero(codes == 1)[0])  # an UNRESOLVED frequency relabelled RESOLVED
            codes[k] = 0
        self.assertRejected("wh-r0_over_M-2.0005-E0", n, "do not recompute")

        def k(h, a, e):
            a["reflectivity"]["R_interface"]["sigma_tau"][300] = 1.
        self.assertRejected("gs-compactness-0.47-E0", k, "do not recompute")

    def test_levels_and_modes(self):
        def m(h, a, e):
            r = h["levels"]["records"][0]
            r["checks"][next(iter(r["checks"]))] = False
        self.assertRejected("bs-sigma_c-0.22-E0", m, "usable flag")

        def n(h, a, e):
            h["modes"] = {"rows": []}
        self.assertRejected("bs-sigma_c-0.22-E0", n, "carries modes")

        def k(h, a, e):
            h["modes"]["rows"][0]["analysis_mode_arguments"]["sigma_num"] = None
        self.assertRejected("gs-compactness-0.47-E0", k, "overstated")


if __name__ == "__main__":
    unittest.main()
