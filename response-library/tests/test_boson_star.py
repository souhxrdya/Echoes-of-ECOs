import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from eco_library import replay_policy

from eco_library.boson_star import (
    FIDUCIAL_SIGMA_C, _parameterize_profile, load_boson_runtime, run_boson_star, solve_equilibrium,
)


class BosonRuntimeTests(unittest.TestCase):
    def test_sample_count_rejected_before_loading(self):
        for value in (True, 5000., 500.5, 499, 200001, float("nan"), None):
            with self.subTest(value=value), patch("eco_library.boson_star.load_boson_runtime") as loader:
                with self.assertRaisesRegex(ValueError, "background_samples"):
                    solve_equilibrium({}, background_samples=value)
                with self.assertRaisesRegex(ValueError, "background_samples"):
                    run_boson_star({}, dx=.1, background_samples=value)
                loader.assert_not_called()

    def test_sampling_transform_has_explicit_provenance_and_no_equation_change(self):
        default = load_boson_runtime()
        finer = load_boson_runtime(background_samples=10001)
        self.assertFalse(default.provenance["numerical_transform"]["applied"])
        self.assertTrue(finer.provenance["numerical_transform"]["applied"])
        modified = []
        for old, new in zip(default.provenance["declarations"], finer.provenance["declarations"]):
            self.assertEqual(old["source_sha256"], new["source_sha256"])
            if "original_ast_sha256" in old:
                self.assertEqual(old["original_ast_sha256"], old["executed_ast_sha256"])
                self.assertEqual(old["original_ast_sha256"], new["original_ast_sha256"])
                if old["executed_ast_sha256"] != new["executed_ast_sha256"]:
                    modified.append(old["name"])
        self.assertEqual(modified, ["solve_mini_boson_star"])
        fixture = ast.parse("def f():\n    untouched=1\n    rr=np.linspace(1e-5,rmax,5000)\n    return untouched\n").body[0]
        _parameterize_profile(fixture, 10001)
        self.assertEqual(fixture.body[1].value.args[2].value, 10001)
        self.assertEqual(fixture.body[0].value.value, 1)
        with self.assertRaisesRegex(ValueError, "assignment changed"):
            _parameterize_profile(fixture, 20001)

    def test_only_declarations_and_no_saved_profile(self):
        runtime = load_boson_runtime()
        for name in ("bs_bg", "bs", "ARTIFACT_DIR", "pd", "plt", "comparison_waveforms"):
            self.assertNotIn(name, runtime.namespace)
        selected = {d["name"] for d in runtime.provenance["declarations"]}
        self.assertIn("solve_mini_boson_star", selected)
        self.assertIn("boson_native_tortoise_at_radius", selected)

    def test_bad_hash_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.ipynb"
            path.write_text("untrusted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_boson_runtime(path)

    def test_nonfinite_and_unsupported_inputs_precede_solve(self):
        for field in ("dx", "t_end", "cfl", "source_global", "observer_global", "sigma", "x_right",
                      "equilibrium_rmax", "equilibrium_tol"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value), patch("eco_library.boson_star.load_boson_runtime") as loader:
                    with self.assertRaises(ValueError):
                        run_boson_star({}, **{"dx": .2, field: value})
                    loader.assert_not_called()
        for params in ({"sigma_c": .01}, {"sigma_c": .5}, {"phi_tilde_c": .1916}, {"lambda": 1},
                       {"sigma_c": True}, {"sigma_c": float("nan")}):
            with self.subTest(params=params), patch("eco_library.boson_star.load_boson_runtime") as loader:
                with self.assertRaises(ValueError):
                    solve_equilibrium(params)
                loader.assert_not_called()


class BosonPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = run_boson_star({"sigma_c": FIDUCIAL_SIGMA_C}, dx=.1, t_end=520.)

    def test_published_background_and_mass_units(self):
        meta = self.record["metadata"]
        self.assertAlmostEqual(meta["phi_tilde_c"], .1916, places=12)
        self.assertLess(abs(meta["omega_s_over_mu"]/.853087-1), 2e-4)
        self.assertLess(abs(meta["raw_mass_mu_units"]/.63300-1), 2e-4)
        # Frozen R99 uses a first-above-threshold radial sample, not interpolation.
        self.assertLess(abs(meta["radius_99_mu_units"]/7.86149-1), 2e-3)
        eq = self.record["equilibrium"]
        np.testing.assert_array_equal(eq["r_over_M"], eq["r"]/meta["raw_mass_mu_units"])
        np.testing.assert_array_equal(eq["axial_V_M2"], eq["axial_V_mu2"]*meta["raw_mass_mu_units"]**2)
        self.assertEqual(meta["radial_stability"], "NOT_COMPUTED")

    def test_controlled_native_mapping_not_naive_shift(self):
        meta = self.record["metadata"]
        self.assertNotEqual(meta["source_native"], 55.)
        self.assertNotEqual(meta["observer_native"], 95.)
        self.assertEqual(meta["x_extract"], meta["observer_native"])
        self.assertTrue(self.record["checks"]["geometry"]["passed"])
        self.assertIsNone(meta["native_to_global_offset_right_exterior"])
        json.dumps(meta, allow_nan=False)

    def test_actual_initial_gaussian_and_characteristics(self):
        r = self.record
        expected = np.exp(-.5*((r["x"]-r["metadata"]["source_native"])/4.)**2)
        expected[0] = 0.
        np.testing.assert_array_equal(expected, r["initial_data"]["psi"])
        self.assertEqual(r["initial_data"]["Pi"][0], 0.)
        self.assertLess(np.max(abs(r["initial_data"]["u_minus"])), 1e-35)
        self.assertFalse(np.array_equal(r["initial_data"]["u_plus"], expected))
        self.assertEqual(r["metadata"]["F_in"], 1.)

    def test_tail_approximation_not_zero_error(self):
        meta = self.record["metadata"]
        self.assertEqual(meta["finite_tail"]["state"], "UNQUANTIFIED")
        self.assertFalse(meta["finite_tail"]["is_error_bound"])
        self.assertEqual(meta["background_reconstruction"]["samples"], 5000)
        self.assertEqual(meta["surface_status"], "NO_FINITE_PHYSICAL_SURFACE")
        self.assertTrue(all(c["passed"] for c in self.record["checks"].values()))

    def _stored(self):
        path = Path(__file__).resolve().parents[2]/"baseline"/"eco_outputs_final"/"ECO_waveforms.h5"
        # test file is tests/library/, so the repository root is parents[2].
        with h5py.File(path, "r") as f:
            stored = f["controlled_comparison"]["BS"]
            return stored["time"][:], stored["waveform_real"][:]

    def test_independent_solve_exact_full_frozen_controlled_record(self):
        """Bitwise replay, asserted in the environment where it was established (replay_policy)."""
        matched, differences = replay_policy.matches_reference("frozen-controlled-records")
        if not matched:
            self.skipTest("bitwise replay is defined for the reference environment only; this runtime differs in "
                          f"{differences}; test_portable_full_frozen_controlled_record applies here")
        t, psi = self._stored()
        self.assertEqual(len(self.record["t"]), len(t))
        np.testing.assert_array_equal(self.record["t"], t)
        np.testing.assert_array_equal(self.record["psi"], psi)

    def test_portable_full_frozen_controlled_record(self):
        """Every runtime: identical sample times and psi within the derived roundoff bound."""
        t, psi = self._stored()
        self.assertEqual(len(self.record["t"]), len(t))
        np.testing.assert_array_equal(self.record["t"], t)
        bound = replay_policy.roundoff_bound(self.record["metadata"]["steps"], 1.)
        self.assertLessEqual(float(np.max(np.abs(self.record["psi"]-psi))), bound)

    def test_background_tolerance_control_preserves_scope(self):
        tight = solve_equilibrium({"sigma_c": FIDUCIAL_SIGMA_C}, equilibrium_tol=2e-8)
        meta, base = tight["metadata"], self.record["metadata"]
        self.assertLess(meta["EKG_residual"], 4e-8)
        self.assertLess(abs(meta["raw_mass_mu_units"]/base["raw_mass_mu_units"]-1), 1e-7)
        self.assertLess(abs(meta["omega_s_over_mu"]/base["omega_s_over_mu"]-1), 1e-7)
        self.assertEqual(meta["finite_tail"]["state"], "UNQUANTIFIED")
        self.assertEqual(meta["background_reconstruction"]["state"], "UNQUANTIFIED")

    def test_nested_dense_profiles_sample_the_same_bvp_solution(self):
        coarse = solve_equilibrium({}, background_samples=5001)
        fine = solve_equilibrium({}, background_samples=10001)
        for name in ("r", "sigma", "dsigma", "a", "alpha", "mass_mu"):
            np.testing.assert_array_equal(coarse["equilibrium"][name], fine["equilibrium"][name][::2])
        self.assertEqual(coarse["metadata"]["raw_mass_mu_units"], fine["metadata"]["raw_mass_mu_units"])
        self.assertEqual(coarse["metadata"]["omega_s_over_mu"], fine["metadata"]["omega_s_over_mu"])
        self.assertEqual(coarse["metadata"]["EKG_residual"], fine["metadata"]["EKG_residual"])
        self.assertEqual(fine["metadata"]["background_samples"], 10001)
        self.assertEqual(fine["metadata"]["background_reconstruction"]["state"], "UNQUANTIFIED")
        # Quadrature is recomputed on the finer samples, not an interpolation
        # of the original coarse tortoise map.
        self.assertGreater(np.max(abs(coarse["equilibrium"]["x_mu"]-fine["equilibrium"]["x_mu"][::2])), 1e-8)

    def test_failed_bvp_is_not_silently_replaced(self):
        runtime = load_boson_runtime()
        runtime.namespace["solve_mini_boson_star"] = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("fixture BVP failure"))
        with patch("eco_library.boson_star.load_boson_runtime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "fixture BVP failure"):
                solve_equilibrium({"sigma_c": FIDUCIAL_SIGMA_C})


if __name__ == "__main__":
    unittest.main()
