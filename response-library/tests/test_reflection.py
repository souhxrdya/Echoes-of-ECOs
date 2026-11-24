"""Reflection coefficients 0.2: asymptotic outgoing start, frozen-range guard, WH formula, evidence and delays."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from eco_library import reflection as rf
from eco_library.transfer import runtime, seam_of


class SeriesTests(unittest.TestCase):
    def test_third_coefficient_vanishes_for_l2_so_the_error_uses_two_terms(self):
        z = np.array([.3+0j])
        a1 = -6/(2j*z)
        a2 = (-4*a1 + 6)/(4j*z)
        a3 = ((2*3-6)*a2 + (6-2*1*3)*a1)/(6j*z)
        self.assertEqual(abs(a3[0]), 0.)
        S, dS, err, k, unconverged = rf.outgoing_series(np.array([.01, .3]), 1285.)
        self.assertFalse(unconverged.any())
        self.assertTrue(np.all(k > 3))

    def test_series_is_an_outgoing_solution_between_two_radii(self):
        """psi = e^{i z r*} S(r) at r = 300 carried by DOP853 to r = 150 equals the series there."""
        z = .2
        def rstar(r):
            return r + 2*np.log(r/2-1)
        def state(r):
            S, dS, *_ = rf.outgoing_series(np.array([z]), r)
            return np.exp(1j*z*rstar(r))*np.array([S[0], 1j*z/(1-2/r)*S[0]+dS[0]])
        def rhs(r, y):
            f, fp = 1-2/r, 2/r**2
            V = f*(6/r**2-6/r**3)
            return [y[1], -(fp/f)*y[1]-(z*z-V)/(f*f)*y[0]]
        sol = solve_ivp(rhs, (300., 150.), state(300.), method="DOP853", rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(sol.y[:, -1], state(150.), rtol=1e-9)


class FrozenRangeTests(unittest.TestCase):
    def test_frozen_inverse_saturates_and_starts_beyond_it_are_refused(self):
        inv = runtime()["r_from_schwarzschild_rstar"]
        self.assertEqual(float(inv(1500.)), float(inv(3000.)))          # the documented clip
        self.assertAlmostEqual(float(inv(1300.)) + 2*np.log(float(inv(1300.))/2-1), 1300., places=8)
        with self.assertRaises(ValueError):
            rf.barrier(np.array([.02]), -1.7, radii={(0., .05): 1500.})


class CoefficientTests(unittest.TestCase):
    def test_wh_interface_formula_equals_the_frozen_wh_seam_transfer(self):
        rt = runtime()
        p = {"r0_over_M": 2.005}
        seam = seam_of("WH", p)
        w = np.array([.15, .3, .45])+0j
        sc = rt["make_seam_scattering"](w, seam, 95.-seam, -65.)
        _, _, lam = rf.throat_coefficients(w, 2.005)
        frozen = rt["wh_seam_transfer"](sc, sc, lam)["R_eff"]
        np.testing.assert_allclose(rf.interface("WH", p, w.real, sc.Rbar), frozen, rtol=0, atol=1e-15)

    def test_barrier_is_independent_of_start_radius_and_step_and_matches_the_independent_solve(self):
        seam = seam_of("GS", {"compactness": .44})
        w = np.array([.02, .1, .45])
        R0 = rf.barrier(w, seam)[0]
        R1 = rf.barrier(w, seam, radii=rf.VARIANT_RADII)[0]
        Ri = rf.independent_barrier(w, seam)
        self.assertLess(np.max(np.abs(R1-R0)), 1e-10)
        self.assertLess(np.max(np.abs(Ri-R0)), 1e-10)

    def test_plane_wave_start_of_export_01_is_not_converged(self):
        rt = runtime()
        seam = seam_of("GS", {"compactness": .44})
        w = np.array([.4425])
        old = rt["make_seam_scattering"](w+0j, seam, 95.-seam, -65., step=.025).Rbar
        self.assertGreater(abs(old[0]-rf.barrier(w, seam)[0][0]), 1e-6)


class EvidenceAndDelayTests(unittest.TestCase):
    def test_variation_rule(self):
        R = np.exp(1j*np.linspace(0, 1, 9))
        self.assertEqual(rf.evidence(R, {"a": R+1e-9, "b": R-1e-9}, 1e-9)[0], "VERIFIED_BY_VARIATION")
        self.assertEqual(rf.evidence(R, {"a": R+1e-9, "b": R-1e-9}, 2e-6)[0], "NOT_CONVERGED")
        self.assertEqual(rf.evidence(R, {"a": R+1e-5, "b": R}, None)[0], "NOT_CONVERGED")
        self.assertEqual(rf.evidence(R, {"a": R.copy(), "b": R.copy()}, None)[0], "UNQUANTIFIED")
        self.assertEqual(rf.evidence(R, {"a": R+1e-9}, None)[0], "UNQUANTIFIED")

    def test_local_delay_of_a_pure_delay_and_states(self):
        w = rf.GRID
        a = 17.
        f = lambda x: .5*np.exp(2j*x*a)
        tau = rf._local_tau(f(w), f(w+rf.DELTA), f(w-rf.DELTA), rf.DELTA)
        np.testing.assert_allclose(tau, 2*a, rtol=1e-8)
        R = np.array([1e-4, .5, .5, .5])
        t = np.array([10., 10., 10., np.nan])
        s = np.array([0., 1e-6, 1e-4, 0.])
        self.assertEqual(list(rf.delay_states(R, t, s)), ["PHASE_ILL_CONDITIONED", "RESOLVED", "UNRESOLVED", "UNDEFINED"])

    def test_archive_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = rf.write(Path(tmp)/"x.h5", {"omega": rf.GRID[:4], "R": np.ones(4, complex)}, {"entry_id": "synthetic"})
            meta, arrays = rf.read(p)
            self.assertEqual(meta["entry_id"], "synthetic")
            with self.assertRaises(ValueError):
                rf.write(p, {"omega": rf.GRID[:4]}, {})
            with h5py.File(p, "r+") as h:
                h["arrays/R"][0] = 2.
            with self.assertRaises(ValueError):
                rf.read(p)


if __name__ == "__main__":
    unittest.main()
