"""Off-fiducial pilot plan, acceptance logic, storage and resume (no long solves)."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eco_library import offfiducial as off
from eco_library.boson_star import FIDUCIAL_SIGMA_C, SIGMA_C_DOMAIN
from eco_library.convergence import WINDOWS
from eco_library.quantum_horizon import reflection
from eco_library.schema import RunSpec, canonical_json
from eco_library.storage import RESUME_PROVENANCE_KEYS


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = off.build_plan()
        cls.jobs = cls.plan["jobs"]

    def test_plan_is_deterministic_and_identities_unique(self):
        self.assertEqual(canonical_json(self.plan), canonical_json(off.build_plan()))
        self.assertEqual(off.plan_sha256(self.plan), off.plan_sha256(json.loads(canonical_json(self.plan))))
        identities = [job["identity"] for job in self.jobs.values()]
        self.assertEqual(len(identities), len(set(identities)))
        # v0.3: +5 controls (QH time step, QH right and left/source domains, BHGS right and left domains).
        self.assertEqual(len(self.jobs), 53)
        roles = [job["role"] for job in self.jobs.values()]
        self.assertEqual(roles.count("production"), 12)
        self.assertEqual(roles.count("reference"), 6)
        self.assertEqual(roles.count("regression"), 1)

    def test_points_are_off_fiducial_and_inside_declared_domains(self):
        points = self.plan["points"]
        for name in ("GS", "WH", "BS", "QH"):
            self.assertNotEqual(points[name]["params"], points[name]["fiducial"])
        self.assertTrue(0 < points["GS"]["params"]["compactness"] < .5)
        self.assertGreater(points["WH"]["params"]["r0_over_M"], 2.)
        sigma_c = points["BS"]["params"]["sigma_c"]
        self.assertTrue(SIGMA_C_DOMAIN[0] <= sigma_c < FIDUCIAL_SIGMA_C)
        self.assertEqual(points["QH"]["params"], {"EplM": 1e10, "gamma": 1.})
        self.assertIn("not Damour-Solodukhin", points["WH"]["model"])
        self.assertIn("not polar", points["BS"]["model"])
        self.assertIn("not a spacetime", points["QH"]["model"])

    def test_common_experiment_is_shared_by_every_waveform_job(self):
        reference = RunSpec("BH").experiment.fingerprint
        for label, job in self.jobs.items():
            with self.subTest(label=label):
                if job["kind"] == "td":
                    spec = RunSpec.from_dict(job["spec"])
                    self.assertEqual(spec.experiment.fingerprint, reference)
                    self.assertEqual(spec.t_end, 520.)
                    self.assertAlmostEqual(spec.cfl*spec.sample_every, .7, places=15)
                elif job["kind"] in ("bs", "qh"):
                    s = job["request"]["settings"]
                    self.assertEqual((s["source_global"], s["observer_global"], s["sigma"], s["t_end"]),
                                     (55., 95., 4., 520.))
                    self.assertAlmostEqual(s["cfl"]*s["sample_every"], .7, places=15)
                elif job["kind"] == "qh_pde":
                    s = job["request"]["settings"]
                    self.assertEqual((s["source_global"], s["observer"], s["sigma"], s["surface"]),
                                     (55., 95., 4., -40.))

    def test_nested_levels_and_matched_references(self):
        spec = lambda label: RunSpec.from_dict(self.jobs[label]["spec"])
        gs = [spec(f"GS_{level}").dx for level in off.LEVEL_NAMES]
        bhgs = [spec(f"BHGS_{level}").dx for level in off.LEVEL_NAMES]
        np.testing.assert_allclose(np.array(gs[:-1])/gs[1:], 2., rtol=1e-12, atol=0)
        self.assertEqual(gs, bhgs)
        n0 = self.plan["gs_expected_interior_cells"]
        self.assertEqual(n0, [n0[0], 2*n0[0], 4*n0[0]])
        for name in ("GS_RIGHT_DOMAIN", "GS_HALF_TIMESTEP"):
            # v0.2: controls on the fine GS mesh declare the same nested cell count.
            self.assertEqual(self.jobs[name]["expected_interior_cells"], n0[2])
        for k, level in enumerate(off.LEVEL_NAMES):
            self.assertEqual(self.jobs[f"GS_{level}"]["expected_interior_cells"], n0[k])
            self.assertEqual(spec(f"BH_{level}").dx, off.LEVELS[k])
            self.assertEqual(spec(f"WH_{level}").dx, off.LEVELS[k])
            self.assertEqual(spec(f"BH_{level}").x_left, -250.)
            self.assertEqual(self.jobs[f"BS_{level}"]["request"]["settings"]["dx"], off.LEVELS[k])
            self.assertEqual(self.jobs[f"QH_{level}"]["request"]["settings"]["dx"], off.LEVELS[k])
        seq = self.plan["sequences"]
        self.assertEqual((seq["GS"]["bh_reference"], seq["WH"]["bh_reference"], seq["BS"]["bh_reference"]),
                         ("BHGS", "BH", "BH"))
        self.assertEqual((seq["GS"]["expected_order"], seq["WH"]["expected_order"],
                          seq["BS"]["expected_order"], seq["QH"]["expected_order"],
                          seq["BS_PROFILE"]["expected_order"], seq["QH_PDE"]["expected_order"]), (2, 4, 4, 4, 2, 2))

    def test_controls_change_only_their_declared_factor(self):
        expected = {
            "BH_RIGHT_DOMAIN": {"x_right"}, "BH_LEFT_DOMAIN": {"x_left"},
            "BH_HALF_TIMESTEP": {"cfl", "sample_every"}, "BHGS_HALF_TIMESTEP": {"cfl", "sample_every"},
            "GS_RIGHT_DOMAIN": {"x_right"}, "GS_HALF_TIMESTEP": {"cfl", "sample_every"},
            "WH_RIGHT_DOMAIN": {"x_right"}, "WH_LEFT_DOMAIN": {"x_left"},
            "WH_HALF_TIMESTEP": {"cfl", "sample_every"},
            "BS_RIGHT_DOMAIN": {"x_right"}, "BS_HALF_TIMESTEP": {"cfl", "sample_every"},
            "BS_BVP_TOL": {"equilibrium_tol"}, "BS_DEFAULT_PROFILE_TO_20001": {"background_samples"},
            "BS_DOMAIN_FIXED_SPACING": {"equilibrium_rmax", "background_samples"},
            "BHGS_RIGHT_DOMAIN": {"x_right"}, "BHGS_LEFT_DOMAIN": {"x_left"},
            "QH_HALF_TIMESTEP": {"cfl", "sample_every"}, "QH_RIGHT_DOMAIN": {"x_right"},
            "QH_LEFT_DOMAIN": {"x_left", "source_x_left"},
        }
        for name, control in self.plan["controls"].items():
            base, variant = self.jobs[control["base"]], self.jobs[control["variant"]]
            with self.subTest(control=name):
                name = name.removesuffix("_CORRECTION") if variant["kind"] == "qh" else name
                if variant["kind"] == "qh_control":
                    self.assertEqual(variant["request"]["base_request"], base["request"])
                    self.assertEqual(variant["request"]["base_request_id"], base["identity"])
                    self.assertEqual(variant["parents"], [control["base"]])
                    continue
                a = base["spec"] if base["kind"] == "td" else base["request"]["settings"]
                b = variant["spec"] if variant["kind"] == "td" else variant["request"]["settings"]
                changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                self.assertEqual(changed, expected[name])
        # The fixed-spacing domain control really preserves the mu-radial spacing.
        s20 = self.jobs["BS_PROFILE_20001"]["request"]["settings"]
        s24 = self.jobs["BS_DOMAIN_24001"]["request"]["settings"]
        self.assertAlmostEqual((s20["equilibrium_rmax"]-1e-5)/20000, (s24["equilibrium_rmax"]-1e-5)/24000, places=14)
        self.assertEqual(s24["background_samples"], 24001)

    def test_policy_is_inherited_and_never_a_bound(self):
        policy = self.plan["policy"]
        self.assertEqual((policy["order_band"], policy["richardson_safety_factor"],
                          policy["control_budget_fraction"], policy["common_grid_dt"]), (.5, 1.25, .1, .07))
        self.assertEqual(policy["acceptance_windows"], ["late", "full"])
        self.assertFalse(policy["estimates_are_error_bounds"])
        self.assertEqual(policy["windows_M"], WINDOWS)

    def test_derived_jobs_name_parents_and_causal_is_diagnostic(self):
        for h in off.PDE_LEVELS:
            job = self.jobs[f"QH_CAUSAL_{h:g}"]
            self.assertEqual(job["role"], "diagnostic")
            self.assertEqual(job["parents"], ["QH_fine", f"QH_PDE_{h:g}"])
            self.assertIn("not the production law", job["purpose"])
            self.assertEqual(job["request"]["pde_request_id"], self.jobs[f"QH_PDE_{h:g}"]["identity"])
        self.assertNotIn("independent_dx", self.jobs["QH_medium"]["request"]["settings"])


def _td_payload(failed=False):
    return {"kind": "td", "checks": {"finite": {"passed": not failed}}}


def _evidence(state="OBSERVED_ORDER_CONSISTENT", estimate=1e-6):
    row = {"state": state, "fine_error_estimate_L2": estimate if state == "OBSERVED_ORDER_CONSISTENT" else None,
           "fine_relative_error_estimate": 1e-4 if state == "OBSERVED_ORDER_CONSISTENT" else None}
    return {w: dict(row) for w in WINDOWS}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        plan = off.build_plan()
        point = plan["points"]["WH"]
        self.plan = {"points": {"WH": point}, "sequences": plan["sequences"], "controls": plan["controls"]}
        labels = plan["sequences"]["WH"]["labels"]+plan["sequences"]["BH"]["labels"]
        self.payloads = {label: _td_payload() for label in labels}
        self.sequences = {"WH": {"refinement": _evidence()}, "BH": {"refinement": _evidence()}}
        budget = {w: {"state": "CONDITIONAL_ESTIMATE", "relative_error_estimate": 1e-4} for w in WINDOWS}
        refinement = {w: {"state": "OBSERVED_ORDER_CONSISTENT"} for w in WINDOWS}
        self.residuals = {"WH": {"paired_budget": budget, "residual_refinement": refinement}}
        window = {"state": "BELOW_DECLARED_REFINEMENT_BUDGET", "relative_difference": 1e-9}
        self.controls = {c: {"state": "COMPLETE", "checks_pass": True,
                             "windows": {w: dict(window) for w in WINDOWS}} for c in point["controls"]}

    def decide(self, extra=None):
        return off.decide_point("WH", self.plan, self.payloads, self.sequences, self.residuals,
                                self.controls, extra or {})

    def test_accepted_only_with_all_evidence(self):
        result = self.decide()
        self.assertEqual(result["outcome"], "ACCEPTED_PILOT_SCOPE")
        self.assertTrue(all(v["state"] == "QUANTIFIED_TD_DOMINANT" for v in result["windows"].values()))
        self.assertIn("no total error bound", result["scope"])

    def test_non_subdominant_control_is_reported_as_separate_component(self):
        self.controls["WH_HALF_TIMESTEP"]["windows"]["late"] = {
            "state": "SYSTEMATIC_NOT_SUBDOMINANT", "relative_difference": 3e-5}
        result = self.decide()
        self.assertEqual(result["windows"]["late"]["state"], "MULTI_COMPONENT_MEASURED")
        self.assertEqual(result["windows"]["late"]["non_subdominant_controls_relative_difference"],
                         {"WH_HALF_TIMESTEP": 3e-5})
        self.assertEqual(result["outcome"], "ACCEPTED_PILOT_SCOPE")

    def test_unconfirmed_or_floor_order_is_not_accepted(self):
        for state in ("ORDER_NOT_CONFIRMED", "AGREEMENT_AT_FLOOR", "UNQUANTIFIED"):
            with self.subTest(state=state):
                self.sequences["WH"]["refinement"]["late"] = {"state": state, "fine_error_estimate_L2": None}
                result = self.decide()
                self.assertEqual(result["windows"]["late"]["state"], "UNQUANTIFIED")
                self.assertEqual(result["outcome"], "INCONCLUSIVE")

    def test_bh_reference_or_budget_missing_is_not_accepted(self):
        self.sequences["BH"]["refinement"]["full"] = {"state": "ORDER_NOT_CONFIRMED"}
        self.assertEqual(self.decide()["outcome"], "INCONCLUSIVE")
        self.setUp()
        self.residuals["WH"]["paired_budget"]["late"] = {"state": "UNQUANTIFIED"}
        self.assertEqual(self.decide()["outcome"], "INCONCLUSIVE")

    def test_hard_failure_rejects_and_missing_is_inconclusive(self):
        label = self.plan["sequences"]["BH"]["labels"][2]
        self.payloads[label] = _td_payload(failed=True)
        self.assertEqual(self.decide()["outcome"], "REJECTED")
        del self.payloads[label]
        self.assertEqual(self.decide()["outcome"], "INCONCLUSIVE")

    def test_incomplete_control_or_failed_gate_blocks_acceptance(self):
        self.controls["WH_LEFT_DOMAIN"]["state"] = "INCOMPLETE"
        self.assertEqual(self.decide()["outcome"], "INCONCLUSIVE")
        self.setUp()
        result = self.decide({"WH": {"passed": False, "reason": "fixture"}})
        self.assertEqual(result["outcome"], "INCONCLUSIVE")
        self.assertTrue(any("model gate" in r for r in result["reasons"]))


class HelperTests(unittest.TestCase):
    def test_turning_point_side(self):
        sc = list(off.BS_SEQUENCE)
        mass = [.50, .55, .59, .615, .630, .633, .632, .628]
        rows = [{"sigma_c": s, "state": "SOLVED", "mass_mu": m, "checks": {"x": True}} for s, m in zip(sc, mass)]
        self.assertEqual(off.turning_point_side(rows, .22)["state"], "LOW_CENTRAL_FIELD_SIDE")
        self.assertEqual(off.turning_point_side(rows, .28)["state"], "NOT_ESTABLISHED")
        early = deepcopy(rows)
        early[2]["mass_mu"] = .64  # sampled maximum three samples below the fiducial
        self.assertEqual(off.turning_point_side(early, .22)["state"], "NOT_ESTABLISHED")
        failed = deepcopy(rows)
        failed[0] = {"sigma_c": .18, "state": "FAILED"}
        self.assertEqual(off.turning_point_side(failed, .22)["state"], "UNQUANTIFIED")

    def test_table_comparison_brackets_input_rounding(self):
        rows = [{"state": "SOLVED", "checks": {"x": True}, "mass_mu": m, "omega_s_over_mu": w,
                 "radius_99_mu": r, "radial_spacing_mu": .01}
                for m, w, r in ((.53420, .77345, 4.52), (.53421, .773453, 4.53), (.53422, .77346, 4.54))]
        table = off.BS_TABLE_I["mini BS II"]
        out = off.table_comparison(rows, table)
        self.assertTrue(all(v["passed"] for v in out.values()))
        far = deepcopy(rows)
        for row in far:
            row["mass_mu"] += .001
        self.assertFalse(off.table_comparison(far, table)["mass_mu"]["passed"])
        far[1] = {"state": "FAILED"}
        self.assertFalse(off.table_comparison(far, table)["omega_s_over_mu"]["passed"])

    def test_pair_difference_and_control_states(self):
        t = np.arange(0., 520.01, .07)
        base = {"kind": "td", "t": t, "psi": np.sin(t), "checks": {"finite": {"passed": True}}, "x": np.arange(5.)}
        variant = {**base, "psi": np.sin(t)+1e-6, "x": np.arange(5.)}
        sequences = {"S": {"refinement": {w: {"fine_error_estimate_L2": 1e-3} for w in WINDOWS}}}
        control = {"base": "a", "variant": "b", "field": "psi", "estimate_sequence": "S"}
        out = off.control_evidence(control, {"a": base, "b": variant}, sequences)
        late = out["windows"]["late"]
        window = t[(t >= 190.) & (t <= 500.)]
        self.assertAlmostEqual(late["difference_L2"], 1e-6*math.sqrt(window[-1]-window[0]), delta=1e-12)
        self.assertEqual(late["state"], "BELOW_DECLARED_REFINEMENT_BUDGET")
        self.assertLess(late["fraction_of_conditional_refinement_estimate"], .1)
        self.assertEqual(out["max_overlap_grid_drift_M"], 0.)
        sequences["S"]["refinement"]["late"]["fine_error_estimate_L2"] = 1e-5
        self.assertEqual(off.control_evidence(control, {"a": base, "b": variant}, sequences)
                         ["windows"]["late"]["state"], "SYSTEMATIC_NOT_SUBDOMINANT")
        sequences["S"]["refinement"]["late"]["fine_error_estimate_L2"] = None
        self.assertEqual(off.control_evidence(control, {"a": base, "b": variant}, sequences)
                         ["windows"]["late"]["state"], "UNQUANTIFIED")
        bad = {**variant, "checks": {"finite": {"passed": False}}}
        self.assertTrue(all(v["state"] == "UNQUANTIFIED" for v in
                            off.control_evidence(control, {"a": base, "b": bad}, sequences)["windows"].values()))
        sequences["S"]["refinement"]["late"]["fine_error_estimate_L2"] = 1e-3
        short = {**variant, "t": t[t < 400.], "psi": np.sin(t[t < 400.])}
        out = off.control_evidence(control, {"a": base, "b": short}, sequences)
        self.assertEqual(out["windows"]["late"]["reason"], "RECORD_TOO_SHORT")
        self.assertEqual(off.control_evidence(control, {"a": base}, sequences)["state"], "INCOMPLETE")

    def test_scheduler_respects_parents_and_budget(self):
        jobs = {"a": {"parents": [], "estimated_seconds": 50.}, "b": {"parents": ["a"], "estimated_seconds": 5.},
                "c": {"parents": [], "estimated_seconds": 200.}}
        self.assertEqual(off.select_next(["c", "b", "a"], jobs, set(), [], 150.), "a")
        self.assertIsNone(off.select_next(["c", "b"], jobs, set(), [], 150.))
        self.assertEqual(off.select_next(["c", "b"], jobs, {"a"}, [], 150.), "b")
        self.assertIsNone(off.select_next(["b"], jobs, {"a"}, ["b"], 150.))

    def test_qh_hierarchy_identity_detects_a_changed_law(self):
        omega = np.r_[0., np.linspace(1e-4, 3., 500)]
        result = {"response": {"omega": omega, "R_horizon_global": reflection(omega, EplM=1e10, gamma=1.)}}
        checks = off.qh_model_checks({"EplM": 1e10, "gamma": 1.}, result)
        self.assertTrue(all(v["passed"] for v in checks.values()))
        self.assertLess(checks["qh_hierarchy_identity"]["max_abs"], 1e-12)
        tampered = deepcopy(result)
        tampered["response"]["R_horizon_global"] = reflection(omega, EplM=1e9, gamma=1.)
        checks = off.qh_model_checks({"EplM": 1e10, "gamma": 1.}, tampered)
        self.assertFalse(checks["qh_analytic_production_law"]["passed"])
        self.assertFalse(checks["qh_hierarchy_identity"]["passed"])

    def test_kernel_ratio_tolerance_is_floating_point_only(self):
        a = 3.510486252933441e-05
        self.assertTrue(off._same_ratio(a, a))
        self.assertTrue(off._same_ratio(a, 3.510486252933413e-05))  # observed v0.1 summation difference
        self.assertFalse(off._same_ratio(a, a*(1+1e-9)))
        self.assertEqual(off.POLICY["kernel_norm_ratio_relative_tolerance"], off.NORM_RATIO_RELATIVE_TOLERANCE)

    def test_runtime_regression_requires_match_and_only_the_known_warning(self):
        import h5py
        path = off.REPO.parent/"baseline"/"eco_outputs_final"/"ECO_waveforms.h5"
        with h5py.File(path, "r") as f:
            ts, ys = f["controlled_comparison"]["BH"]["time"][:], f["controlled_comparison"]["BH"]["waveform_real"][:]
        warned = {"kind": "td", "t": ts, "psi": ys+1e-15,
                  "checks": {"finite": {"passed": True}, "boundary_causal": {"passed": False}}}
        out = off.runtime_regression({"BH_RUNTIME_REPLAY": warned})
        self.assertEqual(out["state"], "PASS")
        self.assertFalse(out["bit_identical"])
        silent = {**warned, "checks": {"finite": {"passed": True}, "boundary_causal": {"passed": True}}}
        self.assertEqual(off.runtime_regression({"BH_RUNTIME_REPLAY": silent})["state"], "FAIL")
        drifted = {**warned, "psi": ys*(1+1e-8)}
        self.assertEqual(off.runtime_regression({"BH_RUNTIME_REPLAY": drifted})["state"], "FAIL")
        self.assertEqual(off.runtime_regression({})["state"], "UNAVAILABLE")

    def test_provenance_requires_plan_and_scripts(self):
        stored = {k: "v" for k in RESUME_PROVENANCE_KEYS}
        stored.update(plan_sha256="p", imported_scripts={"s": "h"})
        off.check_provenance(stored, dict(stored))
        for key, value in (("plan_sha256", "q"), ("imported_scripts", {"s": "other"})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                off.check_provenance(stored, {**stored, key: value})
        with self.assertRaisesRegex(ValueError, "Missing resume provenance"):
            off.check_provenance({k: v for k, v in stored.items() if k != "plan_sha256"}, stored)


class ExecutionTests(unittest.TestCase):
    """Real archive writing/readback with a synthetic solver; no physics claim."""

    @staticmethod
    def provenance(plan_hash="plan"):
        p = {k: "v" for k in RESUME_PROVENANCE_KEYS}
        p.update(source_sha256={"x.py": "1"}, plan_sha256=plan_hash, imported_scripts={"s": "h"})
        return p

    @staticmethod
    def fake_td(*args, **kwargs):
        t = np.arange(0., 10., .5)
        return {"t": t, "psi": np.cos(t), "x": np.arange(12.), "r": 2.+np.arange(12.), "V": np.zeros(12),
                "metadata": {"actual_dx": kwargs["dx"]}, "checks": {"finite": {"passed": True}}}

    def test_execute_publish_readback_reuse_and_refuse(self):
        plan = off.build_plan()
        job = plan["jobs"]["BH_coarse"]
        with tempfile.TemporaryDirectory() as temp, patch.object(off, "run_td", side_effect=self.fake_td):
            root = Path(temp)/"out"
            first = off.execute_job("BH_coarse", job, str(root), self.provenance(), plan["jobs"])
            self.assertEqual((first["execution_state"], first["reused"]), ("COMPLETE", False))
            path = off.output_path(root, job)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(first["file_sha256"], digest)
            second = off.execute_job("BH_coarse", job, str(root), self.provenance(), plan["jobs"])
            self.assertEqual((second["execution_state"], second["reused"]), ("COMPLETE", True))
            third = off.execute_job("BH_coarse", job, str(root), self.provenance("changed"), plan["jobs"])
            self.assertEqual(third["execution_state"], "FAILED")
            self.assertIn("plan_sha256", third["message"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            failures = list((root/"failures").glob("BH_coarse-*.json"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(json.loads(failures[0].read_text())["execution_state"], "FAILED")
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_solver_failure_is_recorded_without_output(self):
        plan = off.build_plan()
        job = plan["jobs"]["GS_coarse"]
        with tempfile.TemporaryDirectory() as temp, patch.object(
                off, "run_td", side_effect=ArithmeticError("fixture solver failure")):
            root = Path(temp)/"out"
            row = off.execute_job("GS_coarse", job, str(root), self.provenance(), plan["jobs"])
            self.assertEqual((row["execution_state"], row["exception"]), ("FAILED", "ArithmeticError"))
            self.assertFalse(off.output_path(root, job).exists())
            self.assertEqual(len(list((root/"failures").glob("*.json"))), 1)

    def test_background_json_checksum_is_enforced(self):
        plan = off.build_plan()
        job = plan["jobs"]["BS_BACKGROUND_SEQUENCE"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"out"
            off.persist(root, job, {"rows": [], "turning_point": {"state": "UNQUANTIFIED"}, "wall_seconds": 0.},
                        self.provenance())
            self.assertEqual(off.load_job(root, job, self.provenance())["identity"], job["identity"])
            path = off.output_path(root, job)
            data = json.loads(path.read_text())
            data["result"]["turning_point"]["state"] = "LOW_CENTRAL_FIELD_SIDE"
            path.write_text(canonical_json(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                off.load_job(root, job, self.provenance())


if __name__ == "__main__":
    unittest.main()
