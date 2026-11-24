"""Execution status and scientific-evidence gates, without waveform solves."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from eco_library.run import build_pilot, gate_refinement, make_report, report_exit_code
from eco_library.convergence import WINDOWS
from eco_library.schema import RunSpec


class RunnerFailureTests(unittest.TestCase):
    @staticmethod
    def report():
        return {
            "unavailable_records": {},
            "execution": [{"record_id": "production", "state": "COMPLETE",
                           "checks_pass": True}],
            "sequences": {
                "BH": {"state": "NUMERICALLY_ASSESSED", "checks_pass": True,
                       "waveform_refinement": {
                           "late": {"state": "ORDER_NOT_CONFIRMED"}}}},
            "baseline_regression": {"BH": {"state": "PASS"}},
        }

    @staticmethod
    def evidence():
        return {
            "prompt": {
                "state": "OBSERVED_ORDER_CONSISTENT", "reason": "Conditional estimate",
                "fine_error_estimate_L2": .01, "fine_relative_error_estimate": .02,
                "p_observed": 3.9, "d_coarse_medium_L2": .15,
                "d_medium_fine_L2": .01, "fine_L2": .5,
                "actual_h": [.2, .1, .05], "is_error_bound": False,
            },
            "late": {
                "state": "AGREEMENT_AT_FLOOR", "reason": "Unidentifiable order",
                "fine_error_estimate_L2": None, "fine_relative_error_estimate": None,
                "p_observed": None, "d_coarse_medium_L2": 0.,
                "d_medium_fine_L2": 0., "is_error_bound": False,
            },
            "full": {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"},
        }

    def test_success_does_not_require_established_convergence(self):
        report = self.report()
        for state in ("ORDER_NOT_CONFIRMED", "AGREEMENT_AT_FLOOR", "UNQUANTIFIED"):
            with self.subTest(state=state):
                report["sequences"]["BH"]["waveform_refinement"]["late"]["state"] = state
                self.assertEqual(report_exit_code(report), 0)

    def test_missing_records_are_execution_failure(self):
        report = self.report()
        report["unavailable_records"] = {"missing": "record absent"}
        self.assertEqual(report_exit_code(report), 1)

    def test_execution_failure_is_not_hidden_by_existing_records(self):
        for state in ("FAILED", "WORKER_FAILED"):
            with self.subTest(state=state):
                report = self.report()
                report["execution"].append({"state": state, "reason": "fixture failure"})
                self.assertEqual(report_exit_code(report), 1)

    def test_regression_failure_has_distinct_exit_code(self):
        report = self.report()
        report["baseline_regression"]["GS"] = {"state": "FAIL"}
        self.assertEqual(report_exit_code(report), 2)

    def test_production_check_failure_has_distinct_exit_code(self):
        report = self.report()
        report["sequences"]["WH"] = {"state": "CHECKS_FAILED", "checks_pass": False}
        self.assertEqual(report_exit_code(report), 2)

    def test_execution_failure_takes_precedence(self):
        report = self.report()
        report["baseline_regression"]["BH"]["state"] = "FAIL"
        report["sequences"]["BH"]["checks_pass"] = False
        report["unavailable_records"] = {"missing": "record absent"}
        self.assertEqual(report_exit_code(report), 1)
        report["unavailable_records"] = {}
        report["execution"].append({"state": "FAILED"})
        self.assertEqual(report_exit_code(report), 1)

    def test_original_domain_replay_checks_are_not_production_checks(self):
        report = self.report()
        report["execution"].append({"record_id": "original-domain-replay", "state": "COMPLETE",
                                    "checks_pass": False, "failed_checks": ["boundary_margin"]})
        self.assertEqual(report_exit_code(report), 0)

    def test_report_only_does_not_require_execution_entries(self):
        report = self.report()
        report["execution"] = []
        self.assertEqual(report_exit_code(report), 0)

    def test_passing_gate_preserves_evidence(self):
        evidence = self.evidence()
        before = deepcopy(evidence)
        self.assertEqual(gate_refinement(evidence, True), before)
        self.assertEqual(evidence, before)

    def test_failed_gate_downgrades_every_window_and_clears_estimates(self):
        evidence = self.evidence()
        before = deepcopy(evidence)
        gated = gate_refinement(evidence, False)
        self.assertEqual(set(gated), set(evidence))
        for name, item in gated.items():
            with self.subTest(window=name):
                self.assertEqual(item["state"], "UNQUANTIFIED")
                self.assertEqual(item["reason"], "RECORD_CHECKS_FAILED")
                self.assertIsNone(item["fine_error_estimate_L2"])
                self.assertIsNone(item["fine_relative_error_estimate"])
                for diagnostic in ("p_observed", "d_coarse_medium_L2", "d_medium_fine_L2",
                                   "fine_L2", "actual_h", "is_error_bound"):
                    if diagnostic in before[name]:
                        self.assertEqual(item[diagnostic], before[name][diagnostic])
        self.assertEqual(evidence, before, "Gating must not overwrite the raw diagnostic input")

    def _report_with_failed_sequence(self, name):
        """Exercise real report assembly; mock I/O and numerical estimators only."""
        plan = build_pilot()
        records = {}
        for rid, value in plan["records"].items():
            spec = RunSpec.from_dict(value)
            records[rid] = {
                "spec": spec, "metadata": {"actual_dx": spec.dx},
                "checks": {"fixture_boundary": {"passed": True}},
                "t": np.arange(521.), "psi": np.zeros(521),
            }
        sequence = next(s for s in plan["sequences"] if s["name"] == name)
        failed_id = sequence["record_ids"][1]
        records[failed_id]["checks"]["fixture_boundary"]["passed"] = False
        raw_evidence = {w: deepcopy(self.evidence()["prompt"]) for w in WINDOWS}

        def fake_read(path, expected_spec=None, expected_provenance=None):
            self.assertEqual(path.stem, expected_spec.record_id)
            return records[expected_spec.record_id]

        def fake_common(rs, dt):
            return np.arange(521.), [r["psi"].copy() for r in rs]

        with patch("eco_library.run.read_record", side_effect=fake_read), \
                patch("eco_library.run.common_series", side_effect=fake_common), \
                patch("eco_library.run.assess", side_effect=lambda *a, **k: deepcopy(raw_evidence)), \
                patch("eco_library.run.baseline_replay", return_value={"BH": {"state": "PASS"}}):
            report = make_report(plan, "unused-output", {}, [], "unused-repo")
        self.assertEqual(report["unavailable_records"], {})
        self.assertEqual(report_exit_code(report), 2)
        self.assertEqual(report["sequences"][name]["failed_checks"][failed_id], ["fixture_boundary"])
        return report

    def _assert_gated(self, evidence):
        self.assertEqual(set(evidence), set(WINDOWS))
        for item in evidence.values():
            self.assertEqual(item["state"], "UNQUANTIFIED")
            self.assertEqual(item["reason"], "RECORD_CHECKS_FAILED")
            self.assertIsNone(item["fine_error_estimate_L2"])
            self.assertIsNone(item["fine_relative_error_estimate"])
            self.assertEqual(item["p_observed"], 3.9)
            self.assertEqual(item["d_medium_fine_L2"], .01)

    def _assert_budget_gated(self, budget):
        self.assertEqual(set(budget), set(WINDOWS))
        for item in budget.values():
            self.assertEqual(item["state"], "UNQUANTIFIED")
            self.assertEqual(item["reason"], "RECORD_CHECKS_FAILED")
            self.assertIsNone(item.get("error_estimate_L2"))

    def test_make_report_gates_failed_eco_and_its_residual(self):
        report = self._report_with_failed_sequence("GS")
        gs = report["sequences"]["GS"]
        self.assertFalse(gs["checks_pass"])
        self._assert_gated(gs["waveform_refinement"])
        self._assert_gated(gs["residual_refinement"])
        self._assert_budget_gated(gs["paired_waveform_budget"])
        for evidence in report["common_sample_grid_sensitivity"]["GS"].values():
            self._assert_gated(evidence)
        self.assertTrue(report["sequences"]["BH_GS_SPACINGS"]["checks_pass"])
        self.assertEqual(report["sequences"]["WH"]["residual_refinement"]["late"]["state"],
                         "OBSERVED_ORDER_CONSISTENT")

    def test_make_report_gates_residual_if_only_matched_bh_fails(self):
        report = self._report_with_failed_sequence("BH_GS_SPACINGS")
        gs = report["sequences"]["GS"]
        bh = report["sequences"]["BH_GS_SPACINGS"]
        self.assertTrue(gs["checks_pass"], "A failed reference does not invalidate the ECO-only waveform")
        self.assertFalse(bh["checks_pass"])
        self._assert_gated(bh["waveform_refinement"])
        self._assert_gated(gs["residual_refinement"])
        self._assert_budget_gated(gs["paired_waveform_budget"])
        self.assertEqual(gs["waveform_refinement"]["late"]["state"], "OBSERVED_ORDER_CONSISTENT")
        self.assertEqual(gs["waveform_refinement"]["late"]["fine_error_estimate_L2"], .01)
        for evidence in report["common_sample_grid_sensitivity"]["BH_GS_SPACINGS"].values():
            self._assert_gated(evidence)
        self.assertEqual(report["sequences"]["WH"]["residual_refinement"]["late"]["state"],
                         "OBSERVED_ORDER_CONSISTENT")


if __name__ == "__main__":
    unittest.main()
