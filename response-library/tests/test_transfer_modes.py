"""Transfer products and complex modes: identities, conventions, archives and anchored roots."""
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from eco_library import modes, transfer


class TransferIdentityTests(unittest.TestCase):
    grid = np.linspace(.05, 1.5, 120)

    def test_gs_flux_unitarity_and_monitor_plane_identity(self):
        products, planes, identities, conditioning = transfer.real_axis_products("GS", {"compactness": .44}, 95., self.grid)
        self.assertLess(identities["barrier_flux_max_abs_omega_0.05_to_1.5"], 1e-6)
        self.assertLess(identities["shell_unitarity_max_abs"], 1e-13)
        moved, _, _, _ = transfer.real_axis_products("GS", {"compactness": .44}, 95., self.grid, monitor=-80.)
        expected = products["K"]*np.exp(1j*self.grid*(-80.+65.))
        self.assertLess(float(np.max(np.abs(moved["K"]-expected)/(1+np.abs(expected)))), 1e-6)
        self.assertEqual(planes["monitor_rstar"], -65.)
        self.assertGreater(conditioning["min_abs_one_minus_loop"], 0.)

    def test_wh_throat_identities_and_parity_split(self):
        products, _, identities, _ = transfer.real_axis_products("WH", {"r0_over_M": 2.005}, 95., self.grid)
        self.assertLess(identities["throat_flux_max_abs"], 1e-13)
        self.assertLess(identities["odd_parity_eigenvalue_max_abs"], 1e-13)
        np.testing.assert_allclose(products["K"], .5*(products["K_even"]+products["K_odd"]), rtol=1e-9, atol=1e-12)

    def test_models_without_a_sharp_interface_are_refused(self):
        for model in ("BS", "QH", "BH"):
            with self.assertRaises(ValueError):
                transfer.seam_of(model, {})

    def test_archive_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = transfer.write_archive(Path(temp)/"a.h5", {"model": "GS"}, {"K": np.arange(4)+1j})
            self.assertEqual(transfer.read_archive(path)["header"], {"model": "GS"})
            with self.assertRaises(ValueError):
                transfer.write_archive(path, {"model": "GS"}, {"K": np.arange(4)+1j})
            with h5py.File(path, "r+") as f:
                f["arrays"]["K"][1] = 7.
            with self.assertRaisesRegex(ValueError, "checksum"):
                transfer.read_archive(path)


class ModeTests(unittest.TestCase):
    def test_gs_fiducial_seed_is_resolved_afresh(self):
        seed = modes.frozen_seeds("GS")[0]["omega"]
        w, res, ok = modes.solve(lambda z: modes.gs_residual(z, .47), seed*(1+1e-4), modes.TOLERANCE["GS"])
        self.assertTrue(ok)
        self.assertLess(abs(w-seed), 1e-9)

    def test_wh_epjc_table_6_prony_values_are_close_to_poles(self):
        for row in modes.wh_literature_check():
            self.assertEqual(row["parity"], "even")
            self.assertLess(row["relative_difference"], 1e-3)

    def test_variation_states_never_turn_agreement_into_zero_uncertainty(self):
        def residual(z, **kw):
            return z-(.3-.1j)
        out = modes.variation(residual, .3-.1j, 1e-10, ({"a": 1}, {"a": 2}))
        self.assertEqual((out["evidence"], out["sigma_num"]), ("UNQUANTIFIED", None))

        def drifting(z, a=0):
            return z-(.3-.1j)-1e-6*a
        out = modes.variation(drifting, .3-.1j, 1e-10, ({"a": 1}, {"a": 2}))
        self.assertEqual(out["evidence"], "VERIFIED_BY_VARIATION")
        self.assertAlmostEqual(out["sigma_num"], 2e-6, places=12)

        def failing(z, a=0):
            if a == 2:
                raise ValueError("no solution")
            return z-(.3-.1j)
        self.assertEqual(modes.variation(failing, .3-.1j, 1e-10, ({"a": 1}, {"a": 2}))["evidence"], "NOT_CONVERGED")

    def test_region_and_tolerance_are_declared(self):
        self.assertEqual(modes.REGION, {"re_min": .05, "re_max": 1., "im_min": -.6, "im_max": 0.})
        self.assertFalse(modes.in_region(.3+.01j))
        self.assertTrue(modes.in_region(.3-.01j))


if __name__ == "__main__":
    unittest.main()
