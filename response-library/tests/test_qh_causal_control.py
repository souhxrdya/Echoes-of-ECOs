import unittest

import numpy as np

from scripts.qh_causal_control import physical_wall_response, wall_kernel, response_with_wall, plan


class CausalWallTests(unittest.TestCase):
    def test_direct_dtft_sign_and_zero_lag_weight(self):
        impulse = np.array([.3, -.1, .7, .02])
        w = np.linspace(0., 2., 31)
        h = .05
        explicit = np.sum(impulse[None, :]*np.exp(1j*w[:, None]*h*np.arange(4)), axis=1)
        np.testing.assert_allclose(physical_wall_response(impulse, h, w), explicit, rtol=2e-15, atol=5e-16)
        np.testing.assert_array_equal(physical_wall_response(np.array([.08]), h, w), .08*np.ones_like(w))

    def test_delay_and_numpy_convolution(self):
        n, h, lag = 256, .1, 17
        impulse = np.zeros(lag+1)
        impulse[-1] = .4
        omega = 2*np.pi*np.fft.rfftfreq(n, h)
        local = physical_wall_response(impulse, h, omega)
        np.testing.assert_allclose(local, .4*np.exp(1j*omega*lag*h), atol=5e-15)
        source = np.r_[np.sin(np.arange(40.)/5), np.zeros(n-40)]
        actual = np.fft.irfft(np.conj(local)*np.fft.rfft(source), n)
        np.testing.assert_allclose(actual, np.convolve(source, impulse)[:n], atol=3e-15)

    def test_kernel_matches_independent_prescription(self):
        from eco_library.quantum_horizon import reflection
        wall = wall_kernel(.1, t_end=20.)
        w = 2*np.pi*np.fft.rfftfreq(wall["nfft"], .1)
        full = np.fft.irfft(np.conj(reflection(w, surface=-40.)), wall["nfft"])
        np.testing.assert_array_equal(wall["impulse"], full[:201])
        self.assertEqual(wall["period_M"], .1*wall["nfft"])
        self.assertGreater(wall["negative_lag_L2_fraction"], 0.)

    def test_zero_wall_transfer_and_plane_guard(self):
        t = .1*np.arange(1024)
        source = {"time": t, "field": np.exp(-.5*((t-20)/4)**2),
                  "derivative": -(t-20)/16*np.exp(-.5*((t-20)/4)**2), "surface": -40.}
        scattering = {"omega": np.linspace(.001, 2., 100), "R_barrier": np.full(100, -.5+0j),
                      "T_detector": np.ones(100, complex), "surface": -40., "observer": 95.}
        wall = {"impulse": np.zeros(1), "dt": .1, "surface": -40., "params": {"EplM": 1e8, "gamma": 1.}}
        response = response_with_wall(source, t, scattering, wall)
        np.testing.assert_array_equal(response["correction"], np.zeros_like(t))
        with self.assertRaisesRegex(ValueError, "planes"):
            response_with_wall(source, t, scattering, {**wall, "surface": -50.})

    def test_bad_inputs_rejected(self):
        for impulse in (np.array([1j]), np.array(['1']), np.array([np.nan])):
            with self.assertRaises(ValueError):
                physical_wall_response(impulse, .1, np.array([.1]))
        for omega in (np.array([-1.]), np.array([100.]), np.array([1j])):
            with self.assertRaises(ValueError):
                physical_wall_response(np.ones(1), .1, omega)
        for h in (True, 0., -1., np.nan, .00001):
            with self.assertRaises(ValueError):
                wall_kernel(h)

    def test_plan_fixed_physical_period(self):
        p = plan()
        periods = []
        for h in p["levels"]:
            nt = int(np.ceil(p["t_end"]/h))+1
            periods.append(h*(1 << (max(32*nt, 32768)-1).bit_length()))
        self.assertEqual(len(set(periods)), 1)


if __name__ == "__main__":
    unittest.main()
