"""Synthetic controls for the QH sensitivity audit; not physics regressions."""
import unittest
import numpy as np

from scripts.qh_spectral_controls import nested_grid, compare, plan


class SpectralControlTests(unittest.TestCase):
    def test_nested_grid_preserves_old_nodes_and_endpoints(self):
        grid = np.r_[np.geomspace(1e-4, .01, 9), np.linspace(.02, 1.5, 50)]
        for factor in (1, 2, 4):
            other = nested_grid(grid, factor)
            np.testing.assert_array_equal(other[::factor], grid)
            self.assertEqual(len(other), factor*(len(grid)-1)+1)
            self.assertTrue(np.all(np.diff(other) > 0))
        for factor in (True, 0, 1.5):
            with self.assertRaises(ValueError):
                nested_grid(grid, factor)

    def test_integral_difference_no_normalization_or_alignment(self):
        t = np.arange(521.)
        reference = {"t": t, "psi": np.ones_like(t), "response": {"correction": np.zeros_like(t)}}
        changed = {"t": t, "psi": np.full_like(t, 2.), "response": {"correction": np.ones_like(t)}}
        out = compare(reference, changed)
        self.assertEqual(out["total"]["late"]["relative_difference"], 1.)
        self.assertAlmostEqual(out["correction"]["late"]["difference_L2"], np.sqrt(310.))
        self.assertIsNone(out["correction"]["late"]["relative_difference"])
        self.assertFalse(out["total"]["late"]["is_error_bound"])

    def test_comparison_rejects_grid_mismatch(self):
        record = {"t": np.arange(521.), "psi": np.zeros(521), "response": {"correction": np.zeros(521)}}
        with self.assertRaises(ValueError):
            compare(record, {**record, "t": record["t"]+.01})

    def test_invalid_fields_and_partial_support_fail_closed(self):
        t = np.arange(231.)
        record = {"t": t, "psi": np.ones_like(t), "response": {"correction": np.ones_like(t)}}
        self.assertEqual(compare(record, record)["total"]["late"]["state"], "UNQUANTIFIED")
        with self.assertRaises(ValueError):
            compare(record, {**record, "psi": record["psi"]+1j})
        for grid in (np.arange(3)+1j, np.arange(3).astype(str)):
            with self.assertRaises(ValueError):
                nested_grid(grid, 2)

    def test_plan_separates_controls(self):
        jobs = plan()
        self.assertEqual(len(jobs), 13)
        self.assertEqual(jobs["plane50"], {"kind": "fresh", "surface": -50.})
        self.assertEqual(jobs["source750"], {"kind": "fresh", "source_t_end": 750.})
        self.assertEqual(jobs["radius400"], {"kind": "scattering", "scattering_x_right": 400.})


if __name__ == "__main__":
    unittest.main()
