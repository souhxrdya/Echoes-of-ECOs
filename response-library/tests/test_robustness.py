"""Phase 2 robustness module: supplement plan, alignment and classification helpers (no solves)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eco_library import offfiducial as off
from eco_library import robustness as rob
from eco_library.schema import Experiment, RunSpec


class SupplementPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = off.build_plan()
        cls.plan = rob.build_supplement(cls.base)

    def test_supplement_is_declared_at_e0_and_links_the_base_plan(self):
        self.assertEqual(self.plan["base_plan_sha256"], off.plan_sha256(self.base))
        self.assertEqual(len(self.plan["jobs"]), 8)
        with self.assertRaisesRegex(ValueError, "production experiment"):
            rob.build_supplement(off.build_plan(Experiment(sigma=3.)))

    def test_each_job_changes_only_its_declared_factors(self):
        expected = {"GS_HALF_TIMESTEP_RIGHT_DOMAIN": ("GS_fine", {"cfl", "sample_every", "x_right"}),
                    "WH_HALF_TIMESTEP_LEFT_DOMAIN": ("WH_fine", {"cfl", "sample_every", "x_left"}),
                    "BS_QUARTER_TIMESTEP": ("BS_fine", {"cfl", "sample_every"}),
                    "BS_HALF_TIMESTEP_RIGHT_DOMAIN": ("BS_fine", {"cfl", "sample_every", "x_right"}),
                    "QH_QUARTER_TIMESTEP": ("QH_medium", {"cfl", "sample_every"}),
                    "QH_PLANE_45": ("QH_medium", {"surface"}), "QH_PLANE_60": ("QH_medium", {"surface"}),
                    "QH_PLANE_50_HALF_TIMESTEP": ("QH_medium", {"surface", "cfl", "sample_every"})}
        for name, (base_label, changed) in expected.items():
            with self.subTest(job=name):
                job, base = self.plan["jobs"][name], self.base["jobs"][base_label]
                a = base["spec"] if base["kind"] == "td" else base["request"]["settings"]
                b = job["spec"] if job["kind"] == "td" else job["request"]["settings"]
                self.assertEqual({k for k in set(a) | set(b) if a.get(k) != b.get(k)}, changed)
                if job["kind"] == "td":
                    self.assertEqual(RunSpec.from_dict(job["spec"]).experiment, off.PRODUCTION_EXPERIMENT)
                else:
                    self.assertEqual(job["request"]["experiment"], self.base["experiment"])
                    off.validate_request_experiment(job)
                    cfl, every = b["cfl"], b["sample_every"]
                    self.assertAlmostEqual(cfl*every, .7, places=15)  # same output cadence

    def test_comparisons_name_existing_records(self):
        refs = []
        comps = self.plan["comparisons"]
        for spec in comps["time_step_convergence"].values():
            refs += spec["levels"]
        refs += [comps["plane_dependence"]["base"]]+list(comps["plane_dependence"]["planes"].values())
        for spec in comps["interactions"].values():
            refs += [spec[k] for k in ("base", "a", "b", "ab")]
        for where, label in refs:
            self.assertIn(label, (self.base if where == "base" else self.plan)["jobs"], (where, label))


class HelperTests(unittest.TestCase):
    def test_native_coincident_takes_coarse_samples_without_interpolation(self):
        t_c = .14*np.arange(100)
        t_m = .07*np.arange(199)
        t_f = .035*np.arange(397)
        f = lambda t: np.sin(t)
        t, ys = rob._native_coincident([{"t": t_c, "psi": f(t_c)}, {"t": t_m, "psi": f(t_m)}, {"t": t_f, "psi": f(t_f)}])
        np.testing.assert_array_equal(t, t_c)
        for y in ys:
            np.testing.assert_allclose(y, f(t_c), rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "native"):
            rob._native_coincident([{"t": t_c, "psi": f(t_c)}, {"t": t_m+.01, "psi": f(t_m)}])

    def test_interaction_of_additive_effects_is_zero(self):
        t = np.arange(0., 520.01, .07)
        y0 = np.sin(.4*t)
        a, b = 1e-6*np.cos(t), 2e-6*np.sin(.3*t)
        records = {"base": y0, "a": y0+a, "b": y0+b, "ab": y0+a+b}
        roots = {"r": None}

        def fake_record(ref, roots, plans):
            return {"t": t, "psi": records[ref[1]]}
        original = rob._record
        rob._record = fake_record
        try:
            rows = rob.interaction({"field": "psi", "base": ["r", "base"], "a": ["r", "a"], "b": ["r", "b"],
                                    "ab": ["r", "ab"]}, roots, {})
        finally:
            rob._record = original
        for row in rows.values():
            self.assertLess(row["interaction_fraction_of_largest_single"], 1e-9)

    def test_budget_includes_non_subdominant_controls_and_refuses_unquantified(self):
        report = {"sequences": {"S": {"refinement": {"late": {"state": "OBSERVED_ORDER_CONSISTENT",
                                                              "fine_error_estimate_L2": 1e-6}}}},
                  "controls": {"C1": {"estimate_sequence": "S", "windows": {"late": {
                                   "state": "SYSTEMATIC_NOT_SUBDOMINANT", "difference_L2": 3e-6}}},
                               "C2": {"estimate_sequence": "S", "windows": {"late": {
                                   "state": "BELOW_DECLARED_REFINEMENT_BUDGET", "difference_L2": 5e-8}}}}}
        total, names = rob._budget(report, ("S",), "late")
        self.assertAlmostEqual(total, 4e-6, places=18)
        self.assertEqual(names, ["C1"])
        report["sequences"]["S"]["refinement"]["late"] = {"state": "ORDER_NOT_CONFIRMED"}
        self.assertEqual(rob._budget(report, ("S",), "late"), (None, []))


if __name__ == "__main__":
    unittest.main()


class CatalogueBlockTests(unittest.TestCase):
    def test_blocks_rebuild_exactly_and_require_a_name(self):
        from eco_library import catalogue_base as cat
        for name in ("B1", "B2", "B3"):
            plan = off.build_plan(None, cat.BLOCKS[name], name)
            self.assertEqual(plan["plan_version"], off.BLOCK_PLAN_VERSION)
            self.assertEqual(off.canonical_json(off.rebuild_plan(json.loads(off.canonical_json(plan)))),
                             off.canonical_json(plan))
            self.assertIn(cat.BLOCKS[name]["BS"]["sigma_c"], plan["jobs"]["BS_BACKGROUND_SEQUENCE"]["request"]["sigma_c"])
            self.assertEqual(plan["jobs"]["BS_BACKGROUND_SEQUENCE"]["request"]["target_sigma_c"],
                             cat.BLOCKS[name]["BS"]["sigma_c"])
            for model in ("GS", "WH", "BS", "QH"):
                self.assertEqual(plan["points"][model]["params"], cat.BLOCKS[name][model])
        with self.assertRaisesRegex(ValueError, "catalogue block"):
            off.build_plan(None, cat.BLOCKS["B2"])
        self.assertEqual(off.plan_sha256(off.build_plan()), off.plan_sha256(off.build_plan(None, None, None)))

    def test_block_points_lie_in_declared_domains(self):
        from eco_library import catalogue_base as cat
        for name, block in cat.BLOCKS.items():
            for model, params in block.items():
                domain = cat.DOMAINS[model]
                value = params[domain["parameter"]]
                lo, hi = domain["interval"]
                self.assertTrue(lo <= value <= hi, (name, model, value))
                self.assertIn(value, domain["points"])


class ObservableResolutionTests(unittest.TestCase):
    def report(self, estimate=1e-6, norm=1e-3, control=0., control_state="SYSTEMATIC_NOT_SUBDOMINANT"):
        budget = {"state": "CONDITIONAL_ESTIMATE", "error_estimate_L2": estimate, "fine_residual_L2": norm}
        return {"sequences": {"WH": {"refinement": {w: {} for w in ("late", "full")}}},
                "residuals": {"WH": {"paired_budget": {"late": budget, "full": budget}}},
                "controls": {"C": {"estimate_sequence": "BH", "windows": {
                    "late": {"state": control_state, "difference_L2": control, "relative_difference": control},
                    "full": {"state": "BELOW_DECLARED_REFINEMENT_BUDGET", "difference_L2": 0.}}}}}

    def test_large_systematic_component_is_not_resolved_by_being_measured(self):
        from eco_library import catalogue_base as cat
        point = {"sequence": "WH", "bh_sequence": "BH"}
        rows = cat.observable_resolution(self.report(), "WH", point)
        self.assertEqual((rows["late"]["state"], rows["full"]["state"]), ("RESOLVED", "RESOLVED"))
        rows = cat.observable_resolution(self.report(control=2e-4), "WH", point)
        self.assertEqual(rows["late"]["state"], "UNRESOLVED")
        self.assertEqual(rows["full"]["state"], "RESOLVED")
        self.assertAlmostEqual(rows["late"]["budget_fraction_of_norm"], (1e-6+2e-4)/1e-3, places=12)
        rows = cat.observable_resolution(self.report(estimate=None), "WH", point)
        self.assertEqual(rows["late"]["state"], "UNQUANTIFIED")
