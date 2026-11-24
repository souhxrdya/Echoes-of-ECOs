"""Regression tests for the 2026-11-23 audit of the off-fiducial pilot (findings A-E).

Written before the fixes; each test reproduces one audited defect.
A  experiment propagation and record-own structural checks;
B  fail-closed numerical evidence (complex/nonfinite/malformed input, estimate
   clearing, fixed-background gate, execution state versus acceptance);
C  explicit control coverage; D  format-specific citation numbering;
E  portable verification (integrity, resume, historical analysis, reproduction).
The adapter tests run real but short records (t_end of a few tens of M).
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from eco_library import boson_star
from eco_library import offfiducial as off
from eco_library import replay_policy
from eco_library.baseline_runtime import load_frozen_runtime
from eco_library.convergence import WINDOWS
from eco_library.schema import Experiment, RunSpec, canonical_json
from eco_library.storage import RESUME_PROVENANCE_KEYS, _check_resume_provenance

E_TEST = Experiment(sigma=5., source_global_rstar=60., observer_global_rstar=120.)
SHIFT = (60.-55.)+(120.-95.)


def _payload(t, psi, kind="td", **extra):
    return {"kind": kind, "t": t, "psi": psi, "checks": {"finite": {"passed": True}}, **extra}


def _smooth(t, h, scale=1e-2, eco=True):
    """Exact order-4 synthetic levels: y_h = f + scale h^4 g."""
    f = np.sin(.4*t)*np.exp(-((t-150.)/80.)**2)+1e-3*np.cos(.3*t)
    if eco:
        f = f+.01*np.sin(.5*t)
    return f+scale*h**4*np.cos(.37*t+.2)


# ------------------------------------------------------------------ finding A

class ExperimentPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = off.build_plan(experiment=E_TEST)
        cls.default = off.build_plan()

    def test_every_waveform_job_carries_the_requested_experiment(self):
        want = asdict(E_TEST)
        settings = {"source_global": 60., "observer_global": 120., "sigma": 5.}
        kinds = set()
        for label, job in self.plan["jobs"].items():
            kind = job["kind"]
            kinds.add(kind)
            with self.subTest(label=label):
                if kind == "td":
                    expected = off.FROZEN_REPLAY_EXPERIMENT if label == "BH_RUNTIME_REPLAY" else E_TEST
                    self.assertEqual(RunSpec.from_dict(job["spec"]).experiment, expected)
                elif kind in ("bs", "qh"):
                    self.assertEqual(job["request"]["experiment"], want)
                    s = job["request"]["settings"]
                    self.assertEqual({k: s[k] for k in settings}, settings)
                elif kind == "qh_control":
                    self.assertEqual(job["request"]["base_request"]["experiment"], want)
                elif kind == "qh_pde":
                    s = job["request"]["settings"]
                    self.assertEqual((s["source_global"], s["observer"], s["sigma"]), (60., 120., 5.))
                    self.assertEqual(job["request"]["experiment"], want)
        self.assertTrue({"td", "bs", "qh", "qh_control", "qh_pde"} <= kinds)

    def test_frozen_replay_experiment_is_separate_and_fixed(self):
        self.assertEqual(off.FROZEN_REPLAY_EXPERIMENT, Experiment())
        for plan in (self.plan, self.default):
            spec = RunSpec.from_dict(plan["jobs"]["BH_RUNTIME_REPLAY"]["spec"])
            self.assertEqual(spec.experiment, off.FROZEN_REPLAY_EXPERIMENT)
            self.assertEqual((spec.dx, spec.x_left, spec.t_end), (.1, -140., 520.))
        self.assertEqual(self.default["experiment"], asdict(off.PRODUCTION_EXPERIMENT))
        self.assertEqual(self.plan["experiment"], asdict(E_TEST))

    def test_identities_change_with_the_experiment_except_backgrounds_and_replay(self):
        for label, job in self.default["jobs"].items():
            other = self.plan["jobs"][label]
            with self.subTest(label=label):
                if job["kind"] in ("bs_sequence", "bs_benchmark") or label == "BH_RUNTIME_REPLAY":
                    self.assertEqual(job["identity"], other["identity"])
                else:
                    self.assertNotEqual(job["identity"], other["identity"])

    def test_windows_and_duration_follow_the_declared_propagation_shift(self):
        for name, seq in self.plan["sequences"].items():
            with self.subTest(sequence=name):
                self.assertEqual(seq["windows"], {w: [lo+SHIFT, hi+SHIFT] for w, (lo, hi) in WINDOWS.items()})
        self.assertEqual(RunSpec.from_dict(self.plan["jobs"]["WH_fine"]["spec"]).t_end, 520.+SHIFT)
        self.assertEqual(self.plan["jobs"]["QH_fine"]["request"]["settings"]["t_end"], 520.+SHIFT)
        self.assertEqual(self.default["sequences"]["WH"]["windows"], {w: list(v) for w, v in WINDOWS.items()})
        self.assertEqual(RunSpec.from_dict(self.default["jobs"]["WH_fine"]["spec"]).t_end, 520.)

    def test_wh_structural_check_reads_the_records_own_experiment(self):
        rt = load_frozen_runtime()
        rstar0 = float(rt["schwarzschild_rstar"](2.005))
        x = .2*np.arange(-50, 51)
        result = {"x": x, "V": np.exp(-x**2), "metadata": {
            "throat_Lambda": 2*(2.005-1)/2.005**2, "rstar0": rstar0,
            "source_native": 60.-rstar0, "observer_native": 120.-rstar0}}
        checks = off.td_model_checks(self.plan["jobs"]["WH_coarse"], result)
        self.assertTrue(checks["wh_coordinate_offset"]["passed"], checks["wh_coordinate_offset"])
        # The same metadata contradicts a record specified at the production experiment.
        checks = off.td_model_checks(self.default["jobs"]["WH_coarse"], result)
        self.assertFalse(checks["wh_coordinate_offset"]["passed"])


class AdapterExperimentTests(unittest.TestCase):
    """Short real adapter runs at a nondefault width, source and observer."""

    def assert_experiment_checks(self, result, n_audits=1):
        c = result["checks"]
        self.assertTrue(c["experiment_consistency"]["passed"], c["experiment_consistency"])
        audits = [k for k in c if k.startswith("initial_data_audit")]
        self.assertEqual(len(audits), n_audits, audits)
        for key in audits:
            self.assertTrue(c[key]["passed"], c[key])
            self.assertLessEqual(c[key]["relative_difference"], c[key]["tolerance"])
        m = result["metadata"]
        self.assertEqual((m["source_global"], m["observer_global"], m["sigma"]), (60., 120., 5.))

    def test_td_adapters_bh_gs_wh(self):
        rt = load_frozen_runtime()
        for model, params in (("BH", {}), ("GS", {"compactness": .44}), ("WH", {"r0_over_M": 2.005})):
            with self.subTest(model=model):
                spec = RunSpec(model, params, dx=.2, t_end=20., x_left=-250. if model == "BH" else None,
                               experiment=E_TEST)
                result = off.compute_job("t", {"kind": "td", "spec": spec.to_dict()}, None, None, {})
                self.assert_experiment_checks(result)
                m = result["metadata"]
                offset = float(rt["schwarzschild_rstar"](2.005)) if model == "WH" else 0.
                self.assertAlmostEqual(m["observer_native"], 120.-offset, places=12)
                self.assertAlmostEqual(m["x_extract"], 120.-offset, places=12)
                radii = rt["r_from_schwarzschild_rstar"](np.array([60., 120.]))
                self.assertAlmostEqual(m["source_areal_radius_over_M"], float(radii[0]), places=12)
                self.assertAlmostEqual(m["observer_areal_radius_over_M"], float(radii[1]), places=12)
                # Negative control: the same record is inconsistent with the frozen width.
                audit = off.initial_data_audit(result["x"], result["psi"][0], m["source_native"],
                                               m["observer_native"], 4.)
                self.assertFalse(audit["passed"])

    def test_boson_star_adapter(self):
        settings = {**off.NUMERICS, **off.experiment_settings(E_TEST), "dx": .2, "t_end": 20., **off.BS_SETTINGS}
        request = {"model": "BS", "params": {"sigma_c": .22}, "settings": settings, "experiment": asdict(E_TEST)}
        result = off.compute_job("t", {"kind": "bs", "request": request}, None, None, {})
        self.assert_experiment_checks(result)
        rt = load_frozen_runtime()
        radii = rt["r_from_schwarzschild_rstar"](np.array([60., 120.]))
        m = result["metadata"]
        self.assertAlmostEqual(m["observer_areal_radius_over_M"], float(radii[1]), delta=1e-9)
        self.assertEqual(m["extraction_points"]["observer"], m["observer_native"])
        tampered = deepcopy(request)
        tampered["settings"]["sigma"] = 4.
        checks = off.experiment_consistency(tampered["experiment"], tampered["settings"], result["metadata"])
        self.assertFalse(checks["passed"])

    def test_quantum_horizon_adapter(self):
        settings = {**off.NUMERICS, **off.experiment_settings(E_TEST), "dx": .2, "t_end": 20.,
                    **off.QH_SETTINGS, "source_t_end": 40.}
        request = {"model": "QH", "params": {"EplM": 1e10, "gamma": 1.}, "settings": settings,
                   "experiment": asdict(E_TEST)}
        result = off.compute_job("t", {"kind": "qh", "request": request}, None, None, {})
        # BH stage at the observer and the measured-source run at the monitor plane.
        self.assert_experiment_checks(result, n_audits=2)

    def test_inconsistent_request_is_refused_before_solving(self):
        settings = {**off.NUMERICS, **off.experiment_settings(E_TEST), "dx": .2, "t_end": 20., **off.BS_SETTINGS}
        request = {"model": "BS", "params": {"sigma_c": .22}, "settings": {**settings, "observer_global": 95.},
                   "experiment": asdict(E_TEST)}
        with patch.object(off, "run_boson_star", side_effect=AssertionError("solver must not run")):
            with self.assertRaisesRegex(ValueError, "experiment"):
                off.compute_job("t", {"kind": "bs", "request": request}, None, None, {})


# ------------------------------------------------------------------ finding B

class FailClosedEvidenceTests(unittest.TestCase):
    t = np.arange(0., 520.01, .07)

    def test_identical_grid_complex_variant_is_rejected_not_zero(self):
        a = _payload(self.t, np.sin(self.t))
        b = _payload(self.t, np.sin(self.t)+100j*np.sin(self.t))
        for x, y in ((a, b), (b, a)):
            with self.assertRaises(off.EvidenceInputError):
                off.pair_difference(x, y, "psi")

    def test_nonfinite_bool_string_object_and_shape_errors_are_rejected(self):
        good = np.sin(self.t)
        bad = {"nan": np.where(self.t > 300, np.nan, good), "inf": np.where(self.t > 300, np.inf, good),
               "bool": good > 0, "string": good.astype(str), "object": good.astype(object),
               "short": good[:-1], "2d": np.vstack([good, good])}
        for name, value in bad.items():
            for grid in ("identical", "different"):
                with self.subTest(case=name, grid=grid):
                    tb = self.t if grid == "identical" else self.t+.01
                    with self.assertRaises(off.EvidenceInputError):
                        off.pair_difference(_payload(self.t, good), _payload(tb, value), "psi")

    def test_malformed_or_nonmonotonic_grids_are_rejected(self):
        psi = np.sin(self.t)
        repeated = self.t.copy()
        repeated[100] = repeated[99]
        nan = self.t.copy()
        nan[5] = np.nan
        for name, grid in (("reversed", self.t[::-1].copy()), ("repeated", repeated), ("nan", nan),
                           ("complex", self.t+0j), ("bool", self.t > 1.)):
            with self.subTest(grid=name), self.assertRaises(off.EvidenceInputError):
                off.pair_difference(_payload(grid, psi), _payload(self.t, psi), "psi")

    def test_window_norms_reject_invalid_data_and_report_support(self):
        delta, ref = np.sin(self.t), np.cos(self.t)
        for bad in (np.where(self.t > 200, np.nan, delta), delta+1j, delta > 0):
            with self.assertRaises(off.EvidenceInputError):
                off._windows_l2(self.t, bad, ref)
        with self.assertRaises(off.EvidenceInputError):
            off._windows_l2(self.t, delta, ref[:-1])
        short = self.t[self.t < 300.]
        out = off._windows_l2(short, np.sin(short), np.cos(short))
        self.assertEqual((out["late"]["state"], out["late"]["reason"]), ("UNQUANTIFIED", "RECORD_TOO_SHORT"))
        sparse = np.array([0., 150., 250., 520.])
        out = off._windows_l2(sparse, np.ones(4), np.ones(4))
        self.assertEqual(out["ringdown"]["reason"], "TOO_FEW_SAMPLES")
        self.assertEqual(out["late"]["reason"], "TOO_FEW_SAMPLES")

    def test_control_evidence_records_invalid_input_instead_of_measuring(self):
        a = _payload(self.t, np.sin(self.t), x=np.arange(5.))
        b = _payload(self.t, np.sin(self.t)+1e-3j, x=np.arange(5.))
        sequences = {"S": {"refinement": {w: {"fine_error_estimate_L2": 1e-3} for w in WINDOWS}}}
        control = {"base": "a", "variant": "b", "field": "psi", "estimate_sequence": "S"}
        out = off.control_evidence(control, {"a": a, "b": b}, sequences)
        self.assertEqual(out["state"], "INVALID_INPUT")
        for row in out["windows"].values():
            self.assertEqual(row["state"], "UNQUANTIFIED")
            self.assertTrue(row["reason"].startswith("INVALID_INPUT"))
            self.assertNotIn("difference_L2", row)

    def test_complex_scattering_data_has_its_own_comparison(self):
        w = np.linspace(.01, 2., 400)
        R = .5*np.exp(3j*w)
        out = off.complex_difference({"omega": w, "value": R}, {"omega": w, "value": R*np.exp(1e-3j)})
        self.assertAlmostEqual(out["max_abs_phase_difference"], 1e-3, places=12)
        self.assertLess(out["max_abs_magnitude_difference"], 1e-15)
        self.assertAlmostEqual(out["relative_L2"], abs(np.exp(1e-3j)-1), places=12)
        for bad in ({"omega": w, "value": R.real}, {"omega": w[::-1], "value": R},
                    {"omega": w, "value": np.where(w > 1, np.nan, R)}, {"omega": w+1e-9, "value": R}):
            with self.assertRaises(off.EvidenceInputError):
                off.complex_difference({"omega": w, "value": R}, bad)


class EstimateClearingTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(0., 520.01, .07)
        self.payloads, labels = {}, {"E": [], "B": []}
        for k, h in enumerate((.2, .1, .05)):
            for name in ("E", "B"):
                label = f"{name}{k}"
                psi = _smooth(t, h, scale=1e-2 if name == "E" else 5e-3, eco=name == "E")
                self.payloads[label] = _payload(t, psi, metadata={"actual_dx": h})
                labels[name].append(label)
        self.plan = {"jobs": {}, "sequences": {
            "E": {"model": "WH", "labels": labels["E"], "field": "psi", "h": "actual_dx",
                  "expected_order": 4, "bh_reference": "B"},
            "B": {"model": "BH", "labels": labels["B"], "field": "psi", "h": "actual_dx",
                  "expected_order": 4, "bh_reference": None}}}

    def evaluate(self):
        seqs = {n: off.sequence_evidence(n, s, self.payloads, {}) for n, s in self.plan["sequences"].items()}
        return seqs, off.residual_evidence("E", self.plan["sequences"]["E"], seqs, self.payloads, self.plan)

    def test_passing_fixture_is_quantified(self):
        seqs, res = self.evaluate()
        self.assertEqual(seqs["E"]["refinement"]["late"]["state"], "OBSERVED_ORDER_CONSISTENT")
        self.assertIsNotNone(res["residual_refinement"]["late"]["fine_relative_error_estimate"])
        self.assertEqual(res["paired_budget"]["late"]["state"], "CONDITIONAL_ESTIMATE")

    def test_failed_parent_clears_every_unsupported_estimate(self):
        self.payloads["E1"]["checks"]["finite"]["passed"] = False
        seqs, res = self.evaluate()
        self.assertEqual(seqs["E"]["state"], "CHECKS_FAILED")
        for table in (seqs["E"]["refinement"], res["residual_refinement"]):
            for w, row in table.items():
                with self.subTest(window=w):
                    self.assertEqual(row["state"], "UNQUANTIFIED")
                    for key in off.ESTIMATE_KEYS:
                        self.assertIsNone(row.get(key), key)
        self.assertTrue(all(v["state"] == "UNQUANTIFIED" for v in res["paired_budget"].values()))


class FixedBackgroundGateTests(unittest.TestCase):
    t = np.arange(0., 520.01, .07)

    def payload(self, h, mass=.6, shift=0., dx=None, spacing=None):
        eq = {"r": np.linspace(0., 50., 11), "sigma": np.linspace(.22, 0., 11)+shift}
        meta = {"actual_dx": h if dx is None else dx, "raw_mass_mu_units": mass, "omega_s_over_mu": .8,
                "EKG_residual": 1e-8, "source_native": 50., "observer_native": 90., "cfl": .35,
                "background_reconstruction": {"radial_spacing_mu": spacing if spacing else .01}}
        return _payload(self.t, _smooth(self.t, h), kind="bs", metadata=meta, result={"equilibrium": eq})

    def test_varied_background_blocks_the_fixed_background_refinement_claim(self):
        seq = {"model": "BS", "labels": ["a", "b", "c"], "field": "psi", "h": "actual_dx",
               "expected_order": 4, "bh_reference": "BH"}
        same = {"a": self.payload(.2), "b": self.payload(.1), "c": self.payload(.05)}
        out = off.sequence_evidence("BS", seq, same, {})
        self.assertEqual((out["state"], out["refinement_kind"]), ("NUMERICALLY_ASSESSED", "TD_GRID_AT_FIXED_BACKGROUND"))
        self.assertTrue(out["fixed_background"]["identical"])
        for name, varied in (("equilibrium arrays", self.payload(.1, shift=1e-12)),
                             ("BVP mass", self.payload(.1, mass=.6000001))):
            with self.subTest(changed=name):
                out = off.sequence_evidence("BS", seq, {**same, "b": varied}, {})
                self.assertEqual(out["state"], "BACKGROUND_NOT_FIXED")
                self.assertFalse(out["fixed_background"]["identical"])
                for row in out["refinement"].values():
                    self.assertEqual(row["state"], "UNQUANTIFIED")
                    for key in off.ESTIMATE_KEYS:
                        self.assertIsNone(row.get(key))

    def test_profile_sampling_sequence_is_not_a_fixed_background_refinement(self):
        seq = {"model": "BS", "labels": ["a", "b", "c"], "field": "psi", "h": "radial_spacing_mu",
               "expected_order": 2, "bh_reference": None}
        payloads = {k: self.payload(h, dx=.1, spacing=s, shift=s) for k, h, s in
                    (("a", .2, .01), ("b", .1, .005), ("c", .05, .0025))}
        out = off.sequence_evidence("BS_PROFILE", seq, payloads, {})
        self.assertEqual((out["state"], out["refinement_kind"]), ("NUMERICALLY_ASSESSED", "BACKGROUND_PROFILE_SAMPLING"))
        payloads["b"] = self.payload(.1, dx=.05, spacing=.005)
        out = off.sequence_evidence("BS_PROFILE", seq, payloads, {})
        self.assertEqual(out["state"], "TD_GRID_NOT_FIXED")


class ExecutionVersusAcceptanceTests(unittest.TestCase):
    def test_completed_execution_with_failed_record_checks_is_reported_separately(self):
        plan = off.build_plan()
        job = plan["jobs"]["BH_coarse"]

        def fake_td(*args, **kwargs):
            t = np.arange(0., 10., .5)
            return {"t": t, "psi": np.cos(t), "x": np.arange(12.), "r": 2.+np.arange(12.), "V": np.zeros(12),
                    "metadata": {"actual_dx": kwargs["dx"]},
                    "checks": {"finite": {"passed": True}, "stability": {"passed": False}}}
        provenance = {k: "v" for k in RESUME_PROVENANCE_KEYS}
        provenance.update(source_sha256={"x.py": "1"}, plan_sha256="plan", imported_scripts={"s": "h"})
        with tempfile.TemporaryDirectory() as temp, patch.object(off, "run_td", side_effect=fake_td):
            row = off.execute_job("BH_coarse", job, str(Path(temp)/"out"), provenance, plan["jobs"])
        self.assertEqual(row["execution_state"], "COMPLETE")
        self.assertEqual(row["record_check_state"], "FAILED")
        self.assertIn("stability", row["failed_checks"])
        self.assertNotIn("state", row)


# ------------------------------------------------------------------ finding C

class ControlCoverageTests(unittest.TestCase):
    def test_new_plan_covers_every_applicable_factor(self):
        plan = off.build_plan()
        coverage = off.control_coverage(plan)
        for point, sequences in coverage.items():
            for seq, factors in sequences.items():
                for factor, row in factors.items():
                    with self.subTest(point=point, sequence=seq, factor=factor):
                        self.assertIn(row["status"], ("COVERED", "NOT_APPLICABLE"))
                        if row["status"] == "NOT_APPLICABLE":
                            self.assertTrue(row["reason"])
        self.assertEqual(coverage["QH"]["QH"]["time_step"]["controls"], ["QH_HALF_TIMESTEP"])
        self.assertEqual(coverage["QH"]["QH_CORRECTION"]["left_and_source_domain"]["controls"],
                         ["QH_LEFT_DOMAIN_CORRECTION"])
        self.assertEqual(coverage["GS"]["BHGS"]["right_domain"]["controls"], ["BHGS_RIGHT_DOMAIN"])
        self.assertEqual(coverage["GS"]["BHGS"]["left_domain"]["controls"], ["BHGS_LEFT_DOMAIN"])
        self.assertEqual(coverage["GS"]["GS"]["left_domain"]["status"], "NOT_APPLICABLE")
        self.assertEqual(coverage["BS"]["BS"]["left_domain"]["status"], "NOT_APPLICABLE")

    def test_qh_and_bhgs_controls_change_only_their_declared_factor(self):
        plan = off.build_plan()
        jobs = plan["jobs"]
        expected = {"QH_HALF_TIMESTEP": ({"cfl", "sample_every"}, "QH_medium"),
                    "QH_RIGHT_DOMAIN": ({"x_right"}, "QH_fine"),
                    "QH_LEFT_DOMAIN": ({"x_left", "source_x_left"}, "QH_fine"),
                    "BHGS_RIGHT_DOMAIN": ({"x_right"}, "BHGS_fine"), "BHGS_LEFT_DOMAIN": ({"x_left"}, "BHGS_fine")}
        for name, (changed, base) in expected.items():
            with self.subTest(control=name):
                control = plan["controls"][name]
                self.assertEqual(control["base"], base)
                a, b = jobs[control["base"]], jobs[control["variant"]]
                a = a["spec"] if a["kind"] == "td" else a["request"]["settings"]
                b = b["spec"] if b["kind"] == "td" else b["request"]["settings"]
                self.assertEqual({k for k in set(a) | set(b) if a.get(k) != b.get(k)}, changed)

    def test_v02_equivalent_plan_reports_exactly_the_audited_gaps(self):
        plan = off.build_plan()
        new = [c for c in plan["controls"] if c.startswith(("QH_HALF", "QH_RIGHT", "QH_LEFT", "BHGS_RIGHT", "BHGS_LEFT"))]
        for name in new:
            del plan["controls"][name]
        for point in plan["points"].values():
            point["controls"] = [c for c in point["controls"] if c not in new]
        for control in plan["controls"].values():  # v0.2 controls carried no factor key
            control.pop("factor_key")
        coverage = off.control_coverage(plan)
        missing = {(p, s, f) for p, seqs in coverage.items() for s, fs in seqs.items()
                   for f, row in fs.items() if row["status"] == "MISSING"}
        self.assertEqual(missing, {("QH", s, f) for s in ("QH", "QH_CORRECTION")
                                   for f in ("time_step", "right_domain", "left_and_source_domain")}
                         | {("GS", "BHGS", "right_domain"), ("GS", "BHGS", "left_domain")})

    def test_missing_coverage_blocks_acceptance(self):
        plan = off.build_plan()
        point = plan["points"]["WH"]
        labels = plan["sequences"]["WH"]["labels"]+plan["sequences"]["BH"]["labels"]
        payloads = {label: {"kind": "td", "checks": {"finite": {"passed": True}}} for label in labels}
        row = {"state": "OBSERVED_ORDER_CONSISTENT", "fine_error_estimate_L2": 1e-6, "fine_relative_error_estimate": 1e-4}
        sequences = {"WH": {"refinement": {w: dict(row) for w in WINDOWS}},
                     "BH": {"refinement": {w: dict(row) for w in WINDOWS}}}
        residuals = {"WH": {"paired_budget": {w: {"state": "CONDITIONAL_ESTIMATE", "relative_error_estimate": 1e-4}
                                              for w in WINDOWS},
                            "residual_refinement": {w: {"state": "OBSERVED_ORDER_CONSISTENT"} for w in WINDOWS}}}
        window = {"state": "BELOW_DECLARED_REFINEMENT_BUDGET", "relative_difference": 1e-9}
        controls = {c: {"state": "COMPLETE", "checks_pass": True, "windows": {w: dict(window) for w in WINDOWS}}
                    for c in point["controls"]}
        result = off.decide_point("WH", plan, payloads, sequences, residuals, controls, {})
        self.assertEqual(result["outcome"], "ACCEPTED_PILOT_SCOPE")
        del plan["controls"]["WH_LEFT_DOMAIN"]
        point["controls"].remove("WH_LEFT_DOMAIN")
        controls.pop("WH_LEFT_DOMAIN")
        result = off.decide_point("WH", plan, payloads, sequences, residuals, controls, {})
        self.assertEqual(result["outcome"], "INCONCLUSIVE")
        self.assertTrue(any("WH:left_domain" in r for r in result["reasons"]), result["reasons"])


# ------------------------------------------------------------------ finding D

class CitationTests(unittest.TestCase):
    def test_macedo_numbering_is_stated_per_format(self):
        lit = boson_star.LITERATURE
        numbers = lit["equation_numbers"]
        self.assertEqual(numbers["arXiv:1307.4812v3 PDF"], {
            "background": "6-11", "axial_master_with_source": "26", "axial_potential": "27",
            "mini_potential": "42", "axial_source_free": "52"})
        self.assertEqual(numbers["arXiv:1307.4812v3 HTML (LaTeXML)"], {
            "background": "6-11", "axial_master_with_source": "36", "axial_potential": "37",
            "mini_potential": "52", "axial_source_free": "62"})
        for path in ("eco_library/boson_star.py", "docs/METHOD.md"):
            with self.subTest(path=path):
                self.assertNotRegex((off.REPO/path).read_text(encoding="utf8"), r"v3 HTML numbering")


# ------------------------------------------------------------------ finding E

def _provenance(**changes):
    p = {k: "v" for k in RESUME_PROVENANCE_KEYS}
    p.update(source_sha256={"a.py": "1"}, plan_sha256="plan", imported_scripts={"s": "h"},
             python="3.11.9", platform="Windows-10-10.0.26300-SP0")
    p.update(changes)
    return p


def _fake_td(*args, **kwargs):
    t = np.arange(0., 10., .5)
    return {"t": t, "psi": np.cos(t), "x": np.arange(12.), "r": 2.+np.arange(12.), "V": np.zeros(12),
            "metadata": {"actual_dx": kwargs["dx"], "steps": 20}, "checks": {"finite": {"passed": True}}}


class PortableVerificationTests(unittest.TestCase):
    producer = _provenance()
    analyzer = _provenance(python="3.10.12", platform="Linux-x86_64", source_sha256={"a.py": "2"})

    def make_root(self, temp, labels=("BH_coarse", "BH_medium")):
        plan = off.build_plan()
        root = Path(temp)/"archive"
        with patch.object(off, "run_td", side_effect=_fake_td):
            for label in labels:
                job = plan["jobs"][label]
                off.persist(root, job, off.compute_job(label, job, root, self.producer, plan["jobs"]), self.producer)
        manifest = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob("*")) if p.is_file()}
        return plan, root, manifest

    def test_resume_refuses_but_historical_analysis_reads_and_keeps_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, _ = self.make_root(temp)
            job = plan["jobs"]["BH_coarse"]
            with self.assertRaisesRegex(ValueError, "Resume provenance mismatch"):
                off.load_job(root, job, self.analyzer)
            payload = off.load_job(root, job, None, mode="historical")
            self.assertEqual(payload["provenance"], self.producer)
            with self.assertRaisesRegex(ValueError, "mode"):
                off.load_job(root, job, self.analyzer, mode="anything")
        # The resume safeguard itself is unchanged: every key, in its declared order.
        with self.assertRaisesRegex(ValueError, "Resume provenance mismatch: source_sha256"):
            _check_resume_provenance(self.producer, self.analyzer)
        with self.assertRaisesRegex(ValueError, "Resume provenance mismatch: python"):
            _check_resume_provenance(self.producer, {**self.producer, "python": "3.10.12"})

    def test_historical_verification_is_read_only_and_separates_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, manifest = self.make_root(temp)
            before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
            out = off.verify_archive(root, plan, manifest=manifest, analyzer_provenance=self.analyzer,
                                     labels=["BH_coarse", "BH_medium"])
            self.assertEqual({p: p.stat().st_mtime_ns for p in root.rglob("*")}, before)
            self.assertEqual(out["integrity"]["state"], "PASS")
            self.assertEqual({v["archive_state"] for v in out["readback"].values()}, {"VERIFIED"})
            self.assertEqual(out["analyzer_provenance"], self.analyzer)
            producers = list(out["producer_provenance"].values())
            self.assertEqual(len(producers), 1)
            self.assertEqual(producers[0]["provenance"], self.producer)
            self.assertEqual(sorted(producers[0]["labels"]), ["BH_coarse", "BH_medium"])
            self.assertEqual(out["resume_compatible"], False)

    def test_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, manifest = self.make_root(temp)
            job = plan["jobs"]["BH_coarse"]
            path = off.output_path(root, job)
            with h5py.File(path, "r+") as f:  # change a value but not its stored checksum
                data = f["waveform"]["psi"]
                data[3] = data[3]+1e-3
            with self.assertRaisesRegex(ValueError, "checksum"):
                off.load_job(root, job, None, mode="historical")
            out = off.verify_archive(root, plan, manifest=manifest, analyzer_provenance=self.analyzer,
                                     labels=["BH_coarse", "BH_medium"])
            self.assertEqual(out["integrity"]["state"], "FAIL")
            self.assertIn(str(path.relative_to(root)).replace("\\", "/"), out["integrity"]["mismatched"])
            self.assertEqual(out["readback"]["BH_coarse"]["archive_state"], "FAILED")
            self.assertEqual(out["readback"]["BH_medium"]["archive_state"], "VERIFIED")

    def test_spoofed_provenance_with_recomputed_checksums_fails_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, manifest = self.make_root(temp)
            job = plan["jobs"]["BH_medium"]
            path = off.output_path(root, job)
            with h5py.File(path, "r+") as f:  # internally consistent rewrite of the producer
                f.attrs["provenance_json"] = canonical_json(self.analyzer)
            # Resume readback is now fooled; the immutable manifest is not.
            off.load_job(root, job, self.analyzer)
            out = off.verify_archive(root, plan, manifest=manifest, analyzer_provenance=self.analyzer,
                                     labels=["BH_medium"])
            self.assertEqual(out["integrity"]["state"], "FAIL")

    def test_missing_and_extra_files_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, manifest = self.make_root(temp)
            extra = dict(manifest)
            extra["records/absent.h5"] = "0"*64
            out = off.verify_archive(root, plan, manifest=extra, analyzer_provenance=self.analyzer, labels=[])
            self.assertEqual(out["integrity"]["missing"], ["records/absent.h5"])
            self.assertEqual(out["integrity"]["state"], "FAIL")

    def test_reproduction_is_classified_against_a_derived_roundoff_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, root, _ = self.make_root(temp, labels=("BH_coarse",))
            with patch.object(off, "run_td", side_effect=_fake_td):
                same = off.reproduce_job(root, plan, "BH_coarse")
            self.assertEqual(same["state"], "BITWISE_IDENTICAL")

            def nudged(*args, **kwargs):
                r = _fake_td(*args, **kwargs)
                r["psi"] = r["psi"]*(1+2**-52)
                return r

            def drifted(*args, **kwargs):
                r = _fake_td(*args, **kwargs)
                r["psi"] = r["psi"]+1e-9
                return r
            with patch.object(off, "run_td", side_effect=nudged):
                self.assertEqual(off.reproduce_job(root, plan, "BH_coarse")["state"], "WITHIN_ROUNDOFF_BOUND")
            with patch.object(off, "run_td", side_effect=drifted):
                out = off.reproduce_job(root, plan, "BH_coarse")
            self.assertEqual(out["state"], "EXCEEDS_ROUNDOFF_BOUND")
            self.assertEqual(out["roundoff_bound"], replay_policy.roundoff_bound(20, 1.))


class ReplayPolicyTests(unittest.TestCase):
    def test_reference_environment_and_bound_are_explicit(self):
        env = replay_policy.REFERENCE_ENVIRONMENTS["frozen-controlled-records"]
        self.assertEqual({k: env[k] for k in ("system", "python", "numpy", "scipy")},
                         {"system": "Windows", "python": "3.11.9", "numpy": "1.26.4", "scipy": "1.15.3"})
        self.assertTrue(env["basis"])
        matched, differences = replay_policy.matches_reference("frozen-controlled-records", {
            **{k: env[k] for k in replay_policy.ENVIRONMENT_KEYS}})
        self.assertEqual((matched, differences), (True, {}))
        matched, differences = replay_policy.matches_reference("frozen-controlled-records", {
            **{k: env[k] for k in replay_policy.ENVIRONMENT_KEYS}, "system": "Linux"})
        self.assertFalse(matched)
        self.assertEqual(set(differences), {"system"})
        self.assertEqual(replay_policy.roundoff_bound(1000, 2.), replay_policy.OPS_PER_STEP*2.**-53*1000*2.)
        for bad in ((0, 1.), (10, 0.), (10, float("nan")), (True, 1.)):
            with self.assertRaises(ValueError):
                replay_policy.roundoff_bound(*bad)


if __name__ == "__main__":
    unittest.main()
