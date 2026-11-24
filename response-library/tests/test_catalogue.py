"""The catalogue: identities come from stored metadata, gamma cannot collide silently, and the status audit
rejects every status that its stored evidence does not support."""
import copy
import json
import unittest
from pathlib import Path

from eco_library import catalogue as cat

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO/"results"/cat.FILE
STORED = json.loads(CATALOGUE.read_text(encoding="utf8"))


def entry(doc, entry_id):
    return next(e for e in doc["entries"] if e["entry_id"] == entry_id)


def audited(mutate):
    doc = copy.deepcopy(STORED)
    mutate(doc)
    return cat.status_audit(doc)


class StoredCatalogue(unittest.TestCase):
    def test_counts_and_states(self):
        self.assertEqual(STORED["counts"], {"ACCEPTED_V1": 14, "INCONCLUSIVE": 2, "REJECTED": 0})
        self.assertEqual({e["entry_id"] for e in STORED["entries"] if e["state"] == "INCONCLUSIVE"},
                         {"bs-sigma_c-0.270963-E0", "qh-EplM-1e+06-E0"})

    def test_stored_catalogue_passes_the_audit(self):
        self.assertEqual(cat.status_audit(STORED), [])

    def test_request_vectors_and_unique_analysis_hashes(self):
        self.assertTrue(entry(STORED, "gs-compactness-0.47-E0")["identity"]["analysis_param_hash"].startswith("472ff4ed1a335f13"))
        self.assertTrue(entry(STORED, "wh-r0_over_M-2.0005-E0")["identity"]["analysis_param_hash"].startswith("8440259ed72a2d08"))
        hashes = [e["identity"]["analysis_param_hash"] for e in STORED["entries"]]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_withdrawn_bs_modes_and_qh_domain(self):
        for e in STORED["entries"]:
            if e["model"] == "BS":
                self.assertIsNone(e["modes"])
                self.assertEqual(e["modes_status"], cat.BS_MODES_STATUS)
                self.assertEqual(e["independent_method"]["state"], "GAP")
            if e["model"] == "QH":
                self.assertEqual(e["prescription_domain"]["in_validated_domain"], e["params"]["EplM"] >= 1e8)
                self.assertTrue(all(e["bh_reference"]["match"].values()))


class AuditRejects(unittest.TestCase):
    def assertRejected(self, mutate, needle):
        problems = audited(mutate)
        self.assertTrue(any(needle in p for p in problems), problems)

    def test_state_without_evidence(self):
        self.assertRejected(lambda d: entry(d, "gs-compactness-0.44-E0").update(state="INCONCLUSIVE"), "stored evidence gives")
        self.assertRejected(lambda d: entry(d, "qh-EplM-1e+06-E0").update(state="ACCEPTED_V1"), "outside the validated")
        self.assertRejected(lambda d: entry(d, "bs-sigma_c-0.270963-E0").update(state="ACCEPTED_V1"), "stored evidence gives")

    def test_resolution_state_and_fraction(self):
        def m(d):
            entry(d, "wh-r0_over_M-2.02-E0")["observable_resolution"]["late"]["budget_fraction_of_norm"] = .2
        self.assertRejected(m, "budget fraction does not recompute")

        def n(d):
            r = entry(d, "wh-r0_over_M-2.02-E0")["observable_resolution"]["late"]
            r["refinement_budget_L2"] = r["norm_L2"]
            r["budget_fraction_of_norm"] = (r["refinement_budget_L2"]+r["non_subdominant_controls_L2"])/r["norm_L2"]
        self.assertRejected(n, "resolution state RESOLVED contradicts")

    def test_order_state(self):
        def m(d):
            entry(d, "gs-compactness-0.49-E0")["refinement"]["GS"]["late"]["p_observed"] = 4.0
        self.assertRejected(m, "contradicts p")

        def n(d):  # the GS order must not be graded against the BHGS order
            entry(d, "gs-compactness-0.49-E0")["declared_order"]["GS"] = 4
        self.assertRejected(n, "contradicts p")

    def test_bh_reference_flags(self):
        def m(d):
            entry(d, "gs-compactness-0.47-E0")["bh_reference"]["match"]["grid"] = True
        self.assertRejected(m, "flags do not recompute")

        def n(d):
            entry(d, "qh-EplM-1e+08-E0")["bh_reference"]["stage_bitwise_equal_to_block_BH"]["BH_fine"] = False
        self.assertRejected(n, "QH reference")

        def k(d):
            entry(d, "qh-EplM-1e+12-E0")["bh_reference"]["identity_max_abs_psi_minus_stage_minus_correction"][2] = 1e-17
        self.assertRejected(k, "QH reference")

    def test_modes(self):
        def m(d):
            e = entry(d, "bs-sigma_c-0.22-E0")
            e["modes"] = {"rows": [{"omega": [.3, -.2]}]}
        self.assertRejected(m, "BS modes not withdrawn")

        def n(d):
            entry(d, "wh-r0_over_M-2.005-E0")["modes"]["rows"][0]["bh_counterpart"] = "BH n=1"
        self.assertRejected(n, "overstated")

        def k(d):
            entry(d, "gs-compactness-0.4-E0")["modes"]["rows"][0]["sigma_num"] = None
        self.assertRejected(k, "without a finite positive sigma_num")

    def test_reflection(self):
        def m(d):
            a = entry(d, "wh-r0_over_M-2.0005-E0")["reflection"]["arrays"]["R_barrier"]
            a["variant_max_abs_difference"]["step_half"] = 2e-6
        self.assertRejected(m, "contradicts its variant numbers")

        def n(d):
            entry(d, "gs-compactness-0.49-E0")["reflection"]["arrays"]["R_interface"]["group_delay_claimable"] = False
        self.assertRejected(n, "group-delay claim")

    def test_transfer(self):
        def m(d):
            g = entry(d, "gs-compactness-0.44-E0")["transfer"]["gate"]["late"]
            g["fd_td_relative"] = 2*g["allowance_relative"]
        self.assertRejected(m, "state contradicts the discrepancy")

    def test_transfer_review_from_another_experiment(self):
        def m(d):  # the defect found in the inline review: numbers of the O120 summary under an E0 entry
            r = entry(d, "wh-r0_over_M-2.005-E0")["transfer"]["gate_review"]["all_windows"]["late"]
            r.update(fd_td_relative=3.176590309249019e-08, residual_budget_relative=6.437706350139913e-08,
                     ratio=0.4934351050634643)
        self.assertRejected(m, "not those of the entry's own gate")

    def test_identity_and_counts(self):
        def m(d):
            entry(d, "qh-EplM-1e+10-E0")["params"]["gamma"] = 2.
        self.assertRejected(m, "does not recompute")

        def n(d):
            entry(d, "qh-EplM-1e+10-E0")["identity"]["analysis_param_hash"] = entry(d, "qh-EplM-1e+08-E0")["identity"]["analysis_param_hash"]
        self.assertRejected(n, "collision")

        def k(d):
            d["counts"]["ACCEPTED_V1"] = 15
        self.assertRejected(k, "counts")


class Identities(unittest.TestCase):
    META = {"model_family": "QH", "model_scope": cat.MODEL_DEFINITIONS["QH"]["scope"],
            "solver_id": cat.MODEL_DEFINITIONS["QH"]["solver_id"], "baseline_provenance": {"declarations": []},
            **{k: f"setting {k}" for k in cat.QH_DEFINING_KEYS}}

    def test_stored_definition_must_match(self):
        cat.model_identity("QH", [self.META]*3)
        wrong = dict(self.META, model_scope="a quantum spacetime")
        with self.assertRaises(ValueError):
            cat.model_identity("QH", [self.META, wrong, self.META])
        with self.assertRaises(ValueError):
            cat.model_identity("WH", [self.META]*3)

    def test_gamma_changes_the_entry_identity_and_the_collision_guard_fires(self):
        ids = []
        for gamma in (1., 2.):
            q = cat.qh_prescription([self.META]*3, {"EplM": 1e10, "gamma": gamma})
            ident = {"model_definition": cat.model_identity("QH", [self.META]*3), "qh_prescription": q}
            ids.append((q, cat.analysis_identities("QH", {"EplM": 1e10, "gamma": gamma}, ident)))
        (q1, a1), (q2, a2) = ids
        self.assertEqual(q1["prescription_version_sha256"], q2["prescription_version_sha256"])
        self.assertNotEqual(q1["prescription_instance_sha256"], q2["prescription_instance_sha256"])
        self.assertEqual(a1["analysis_param_hash"], a2["analysis_param_hash"])  # request rule omits gamma
        self.assertNotEqual(a1["extended_analysis_identity_proposal"]["sha256"],
                            a2["extended_analysis_identity_proposal"]["sha256"])
        entries = [{"entry_id": f"qh-{g}", "identity": a} for g, (_, a) in zip((1, 2), ids)]
        with self.assertRaises(ValueError):
            cat.guard_collisions(entries)

    def test_bs_equilibrium_identity_is_the_request(self):
        meta = {k: 1. for k in cat.BS_REQUEST_KEYS+cat.BS_PROPERTY_KEYS}
        meta["literature"] = {"field_conversion": "sigma=sqrt(2)*phi_tilde"}
        a = cat.bs_equilibrium([meta]*3)
        b = cat.bs_equilibrium([meta, dict(meta, omega_s_over_mu=1.1), meta])
        self.assertEqual(a["equilibrium_id"], b["equilibrium_id"])
        self.assertTrue(a["properties_identical_at_all_levels"])
        self.assertFalse(b["properties_identical_at_all_levels"])
        with self.assertRaises(ValueError):
            cat.bs_equilibrium([meta, dict(meta, equilibrium_tol=2e-8), meta])


if __name__ == "__main__":
    unittest.main()
