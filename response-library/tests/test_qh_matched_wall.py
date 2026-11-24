import copy
import unittest

import numpy as np

from scripts.qh_compare_matched_wall import validate_pair, paired_metrics


class MatchedWallTests(unittest.TestCase):
    def fixture(self):
        impulse = np.arange(4.)
        common = {"provenance": {"reference_archive_sha256": "same"}}
        pde = {**common, "request": {"kind": "fine_independent_PDE"}, "result": {
            "checks": {"finite": {"passed": True}},
            "independent": {"dt": .1, "surface": -40., "params": {"EplM": 1e8}},
            "wall_kernel": {"nfft": 32768, "causal_impulse_used": impulse}}}
        fd = {**common, "request": {"kind": "causal_wall_FD_control"}, "result": {
            "checks": {"finite": {"passed": True}},
            "wall": {"dt": .1, "surface": -40., "params": {"EplM": 1e8},
                     "nfft": 32768, "impulse": impulse.copy()}}}
        return pde, fd

    def test_exact_wall_and_reference_binding(self):
        pde, fd = self.fixture()
        validate_pair(pde, fd)
        for mode in ("coefficient", "period", "source", "failed_check"):
            other = copy.deepcopy(fd)
            if mode == "coefficient":
                other["result"]["wall"]["impulse"][0] += 1e-15
            elif mode == "period":
                other["result"]["wall"]["nfft"] *= 2
            elif mode == "source":
                other["provenance"]["reference_archive_sha256"] = "different"
            else:
                other["result"]["checks"]["finite"]["passed"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_pair(pde, other)

    def test_raw_decomposition_and_reference_norm(self):
        t = np.arange(521.)
        record = lambda value: {"t": t, "psi": np.full_like(t, value)}
        arrays, result, identity = paired_metrics(record(1.), record(1.25), record(2.))
        self.assertLess(identity, 1e-15)
        relative = result["late"]["relative_L2"]
        self.assertAlmostEqual(relative["analytic_FD_minus_PDE"], 1.)
        self.assertAlmostEqual(relative["analytic_FD_minus_causal_FD"], .75)
        self.assertAlmostEqual(relative["causal_FD_minus_PDE"], .25)
        np.testing.assert_array_equal(arrays["matched_residual"], .25*np.ones_like(arrays["t"]))
        self.assertFalse(result["late"]["is_total_error_bound"])

    def test_missing_support_has_no_numerical_discrepancy(self):
        record = {"t": np.arange(101.), "psi": np.ones(101)}
        _, result, _ = paired_metrics(record, record, record)
        self.assertEqual(result["late"], {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"})


if __name__ == "__main__":
    unittest.main()
