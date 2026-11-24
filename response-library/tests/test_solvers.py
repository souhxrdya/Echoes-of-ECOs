from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eco_library.baseline_runtime import NOTEBOOK_SHA256, load_frozen_runtime
from eco_library.solvers import refinement_spacings, run_td


class FrozenRuntimeTests(unittest.TestCase):
    def test_hash_and_no_notebook_side_effect_namespace(self):
        runtime = load_frozen_runtime()
        self.assertEqual(runtime.provenance["notebook_sha256"], NOTEBOOK_SHA256)
        for name in ("ARTIFACT_DIR", "bh", "gs", "wh", "_BH_CACHE", "_GS_CACHE", "plt"):
            self.assertNotIn(name, runtime.namespace)

    def test_untrusted_notebook_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.ipynb"
            path.write_text("not notebook JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_frozen_runtime(path)

    def test_namespaces_are_independent(self):
        first, second = load_frozen_runtime(), load_frozen_runtime()
        self.assertIsNot(first.namespace, second.namespace)
        self.assertIsNot(first["Config"], second["Config"])


class SolverTests(unittest.TestCase):
    def test_nested_gs_resolution(self):
        spacings = refinement_spacings("GS", {"compactness": .47})
        runtime = load_frozen_runtime()
        cells = []
        actual = []
        for h in spacings:
            x, _, _, meta = runtime["gravastar_grid"](.47, cfg=runtime["Config"](), dx=h)
            cells.append(meta["shell_index"])
            actual.append(np.mean(np.diff(x)))
        self.assertEqual(cells, [23, 46, 92])
        np.testing.assert_allclose(np.array(actual[:-1]) / actual[1:], 2., rtol=1e-12)

    def test_unsupported_model_not_silent_approximation(self):
        with self.assertRaises(NotImplementedError):
            run_td("BS", {}, dx=.2)

    def test_parameter_and_geometry_guards(self):
        with self.assertRaises(ValueError):
            run_td("GS", {"compactness": .5}, dx=.2)
        with self.assertRaises(ValueError):
            run_td("WH", {"r0": 2.01}, dx=.2)
        with self.assertRaises(ValueError):
            run_td("BH", {}, dx=.2, observer_global=500.)

    def test_nonfinite_input_rejected_before_runtime_or_allocation(self):
        for field in ("dx", "t_end", "cfl", "source_global", "observer_global", "sigma", "x_left", "x_right"):
            for value in (float("nan"), float("inf"), -float("inf")):
                kwargs = {"dx": .2, field: value}
                with self.subTest(field=field, value=value):
                    with patch("eco_library.solvers.load_frozen_runtime") as loader:
                        with self.assertRaises(ValueError):
                            run_td("BH", {}, **kwargs)
                        loader.assert_not_called()

    def test_nonfinite_parameters_rejected_before_runtime(self):
        for model, key in (("GS", "compactness"), ("WH", "r0_over_M")):
            with patch("eco_library.solvers.load_frozen_runtime") as loader:
                with self.assertRaises(ValueError):
                    run_td(model, {key: float("nan")}, dx=.2)
                loader.assert_not_called()

    def test_short_bh_run_explicit_observer_and_no_cache(self):
        numpy_had_trapezoid = hasattr(np, "trapezoid")
        record = run_td("BH", {}, dx=.4, t_end=3., observer_global=90.)
        self.assertEqual(record["metadata"]["x_extract"], 90.)
        self.assertEqual(record["metadata"]["observer_global"], 90.)
        self.assertEqual(record["metadata"]["numerics_evidence"], "UNQUANTIFIED")
        self.assertTrue(record["checks"]["finite"]["passed"])
        self.assertTrue(record["checks"]["boundary_causal"]["passed"])
        self.assertEqual(record["metadata"]["x_left"], -250.)
        self.assertEqual(hasattr(np, "trapezoid"), numpy_had_trapezoid)

    def test_explicit_frozen_domain_reports_causal_warning(self):
        record = run_td("BH", {}, dx=1., t_end=520., x_left=-140.)
        self.assertFalse(record["checks"]["boundary_causal"]["passed"])
        self.assertAlmostEqual(record["checks"]["boundary_causal"]["left_return_M"], 398.)
        self.assertLess(record["metadata"]["boundary_margin_M"], 0.)

    def test_wormhole_physical_coordinates_and_throat(self):
        record = run_td("WH", {"r0_over_M": 2.0005}, dx=.4, t_end=3.)
        meta = record["metadata"]
        self.assertAlmostEqual(meta["source_native"] + meta["rstar0"], 55.)
        self.assertAlmostEqual(meta["observer_native"] + meta["rstar0"], 95.)
        self.assertTrue(np.any(record["x"] == 0.))
        self.assertTrue(record["checks"]["interface"]["passed"])
        self.assertNotEqual(meta["observer_native"], meta["observer_global"])

    def test_gravastar_uses_regular_center(self):
        record = run_td("GS", {"compactness": .47}, dx=.4, t_end=3.)
        self.assertIsNone(record["checks"]["boundary_causal"]["left_return_M"])
        self.assertTrue(record["checks"]["interface"]["passed"])
        self.assertNotEqual(record["metadata"]["requested_dx"], record["metadata"]["actual_dx"])


if __name__ == "__main__":
    unittest.main()
