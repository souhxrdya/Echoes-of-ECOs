import unittest
import numpy as np

from scripts.qh_fine_independent import plan, comparison, retained_kernel
from eco_library.quantum_horizon import independent_td


class FineIndependentTests(unittest.TestCase):
    def test_fixed_period_and_safe_step_budget(self):
        p = plan()
        for h in p["levels"]:
            nt = int(np.ceil(p["t_end"]/h))+1
            nker = 1 << (max(32*nt, 32768)-1).bit_length()
            self.assertEqual(nker*h, p["kernel_period_M"])
            self.assertLess(nt, 100000)
        self.assertEqual(p["expected_order"], 2)

    def test_kernel_prefix_reconstructs_exact_pde_prescription(self):
        direct = independent_td(dx=.2, t_end=20.)
        saved = retained_kernel(direct)
        np.testing.assert_array_equal(saved["causal_impulse_used"], saved["full_periodic_impulse"][:len(direct["time"])])
        np.testing.assert_allclose(np.fft.rfft(saved["full_periodic_impulse"]),
                                   saved["physical_local_reflectivity"].conj(), atol=2e-15, rtol=1e-12)
        self.assertFalse(saved["negative_lag_norm_is_error_bound"])

    def test_partial_support_unquantified_and_raw_norm_definition(self):
        t = np.arange(521.)
        a, b = {"t": t, "psi": np.ones_like(t)}, {"t": t, "psi": np.full_like(t, 2.)}
        out = comparison(a, b)
        self.assertEqual(out["late"]["relative_L2"], 1.)
        self.assertEqual(out["late"]["sample_norm_relative_L2"], 1.)
        self.assertFalse(out["late"]["is_error_bound"])
        short = {"t": t[:230], "psi": t[:230]}
        self.assertEqual(comparison(short, short)["late"]["state"], "UNQUANTIFIED")
        with self.assertRaises(ValueError):
            comparison(a, {"t": t, "psi": t+1j})


if __name__ == "__main__":
    unittest.main()
