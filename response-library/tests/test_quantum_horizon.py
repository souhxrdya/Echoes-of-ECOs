"""QH physics identities, source/plane conventions, and full frozen replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from eco_library import replay_policy

from eco_library.quantum_horizon import (
    _runtime, reflection, measured_ingoing_spectrum, reconstruct,
    independent_td, run_quantum_horizon,
)


class QuantumUnitTests(unittest.TestCase):
    def test_only_pinned_declarations(self):
        rt = _runtime()
        for name in ("qh", "bh", "qh_source_records", "plt", "ARTIFACT_DIR", "pd"):
            self.assertNotIn(name, rt.namespace)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"untrusted.ipynb"
            path.write_text("{}", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _runtime(path)

    def test_flux_and_hermitian_reflection(self):
        w = np.linspace(.001, 1., 200)
        r = reflection(w)
        np.testing.assert_allclose(abs(r)**2, np.exp(-8*np.pi*w), atol=4e-16, rtol=1e-13)
        np.testing.assert_array_equal(reflection(-w), r.conj())
        self.assertEqual(reflection(0.), 1.)

    def test_hierarchy_phase_not_absorption(self):
        w = np.linspace(.01, .8, 1000)
        r1, r2 = reflection(w, EplM=1e8), reflection(w, EplM=1e10)
        np.testing.assert_allclose(abs(r1), abs(r2), atol=3e-16, rtol=1e-13)
        np.testing.assert_allclose(r2, r1*np.exp(8j*w*np.log(100)), atol=5e-15, rtol=1e-12)

    def test_reference_plane_rephasing(self):
        w = np.linspace(.01, 1., 100)
        a, b = reflection(w, surface=-40), reflection(w, surface=-50)
        np.testing.assert_allclose(b, a*np.exp(-20j*w), atol=3e-16, rtol=1e-12)

    def test_measured_characteristic_sign(self):
        t = np.arange(1024)*.1
        w = 2*np.pi*5/(len(t)*.1)
        field, dx = np.cos(w*t), -w*np.sin(w*t)
        parts = measured_ingoing_spectrum(t, field, dx, len(t))
        self.assertLess(abs(parts["outgoing_fft"][5]), 2e-12)
        self.assertLess(abs(parts["ingoing_fft"][5]-512), 3e-12)
        self.assertEqual(parts["ingoing_fft"][0], 0)
        self.assertEqual(parts["dc_status"], "DIRECTION_UNDEFINED_SET_TO_ZERO")

    def test_reconstruction_rejects_nonreal_jost_grid_before_cast(self):
        t = np.arange(32)*.1
        source = {"time": t, "field": np.cos(t), "derivative": -np.sin(t), "surface": -40.}
        grid = np.linspace(.01, 1., 32)
        sc = {"surface": -40., "observer": 95., "R_barrier": np.full(32, -.5+0j),
              "T_detector": np.ones(32, complex)}
        for invalid in (grid+1j, grid.astype(str), grid.astype(object), np.ones(32, bool)):
            with self.subTest(dtype=invalid.dtype), self.assertRaisesRegex(ValueError, "real numeric"):
                reconstruct(source, t, {**sc, "omega": invalid})

    def test_invalid_before_load_or_allocation(self):
        for name in ("dx", "t_end", "cfl", "surface", "source_global", "observer_global", "sigma",
                     "x_left", "x_right", "source_t_end", "source_x_left", "scattering_step",
                     "scattering_x_right", "independent_dx"):
            for value in (float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value), patch("eco_library.quantum_horizon._runtime") as loader:
                    with self.assertRaises(ValueError):
                        run_quantum_horizon(**{name: value})
                    loader.assert_not_called()
        for kwargs in ({"dx": 1e-10}, {"nfft": True}, {"nfft": 128}, {"omega_grid": [1+1j]*8},
                       {"omega_grid": np.linspace(.1, 100, 20)}, {"sample_every": True}):
            with self.subTest(kwargs=kwargs), patch("eco_library.quantum_horizon._runtime") as loader:
                with self.assertRaises(ValueError):
                    run_quantum_horizon(**kwargs)
                loader.assert_not_called()

    def test_zero_wall_independent_pde(self):
        r = independent_td(dx=.2, t_end=20., constant_reflection=0.)
        np.testing.assert_array_equal(r["correction"], np.zeros_like(r["time"]))
        self.assertEqual(r["negative_lag_kernel_L2_fraction"], 0.)


class QuantumReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is deliberately the original domain for adapter fidelity only.
        cls.record = run_quantum_horizon(x_left=-140., independent_dx=.1)

    def require_reference_environment(self):
        matched, differences = replay_policy.matches_reference("frozen-controlled-records")
        if not matched:
            self.skipTest("bitwise replay is defined for the reference environment only; this runtime differs in "
                          f"{differences}; the portable replay tests apply here")

    @staticmethod
    def stored_waveform():
        path = Path(__file__).resolve().parents[2]/"baseline"/"eco_outputs_final"/"ECO_waveforms.h5"
        with h5py.File(path, "r") as f:
            stored = f["controlled_comparison"]["QH"]
            return stored["time"][:], stored["waveform_real"][:]

    @staticmethod
    def stored_source():
        path = Path(__file__).resolve().parents[2]/"baseline"/"eco_outputs_final"/"qh_measured_source.npz"
        with np.load(path) as saved:
            return {k: saved[k] for k in ("source_t", "monitor_field", "monitor_derivative", "K")}

    def test_full_frozen_controlled_replay(self):
        self.require_reference_environment()
        t, psi = self.stored_waveform()
        np.testing.assert_array_equal(self.record["t"], t)
        np.testing.assert_array_equal(self.record["psi"], psi)

    def test_source_and_transfer_replay(self):
        self.require_reference_environment()
        saved = self.stored_source()
        np.testing.assert_array_equal(self.record["source"]["time"], saved["source_t"])
        np.testing.assert_array_equal(self.record["source"]["field"], saved["monitor_field"])
        np.testing.assert_array_equal(self.record["source"]["derivative"], saved["monitor_derivative"])
        np.testing.assert_array_equal(self.record["response"]["K"], saved["K"])

    def test_portable_full_frozen_controlled_replay(self):
        """Every runtime: identical sample times; psi (BH stage plus reconstruction from the measured
        source) within the roundoff bound of both evolutions' step counts."""
        t, psi = self.stored_waveform()
        np.testing.assert_array_equal(self.record["t"], t)
        steps = self.record["metadata"]["steps"]+self.record["source"]["metadata"]["steps"]
        self.assertLessEqual(float(np.max(np.abs(self.record["psi"]-psi))), replay_policy.roundoff_bound(steps, 1.))

    def test_portable_source_and_transfer_replay(self):
        """Every runtime: identical source sample times; the monitor field within the source run's bound,
        its spatial derivative within that bound times 1/dx (one division by the grid spacing), and K
        within the bound of the Jost integration's step count."""
        saved = self.stored_source()
        meta = self.record["metadata"]
        np.testing.assert_array_equal(self.record["source"]["time"], saved["source_t"])
        steps = self.record["source"]["metadata"]["steps"]
        bound = replay_policy.roundoff_bound(steps, 1.)
        self.assertLessEqual(float(np.max(np.abs(self.record["source"]["field"]-saved["monitor_field"]))), bound)
        self.assertLessEqual(float(np.max(np.abs(self.record["source"]["derivative"]-saved["monitor_derivative"]))),
                             bound/meta["actual_dx"])
        jost = int(round((meta["scattering_x_right"]-self.record["source"]["surface"])/meta["scattering_step"]))
        self.assertLessEqual(float(np.max(np.abs(self.record["response"]["K"]-saved["K"]))),
                             replay_policy.roundoff_bound(jost, 1.))

    def test_plane_source_and_status_metadata(self):
        r = self.record
        self.assertFalse(r["checks"]["boundary_causal"]["passed"])
        self.assertTrue(all(v["passed"] for k, v in r["checks"].items() if k != "boundary_causal"))
        self.assertEqual(r["metadata"]["numerics_evidence"], "UNQUANTIFIED")
        self.assertEqual(r["metadata"]["reflection_planes"]["R_barrier"], 0.)
        self.assertEqual(r["metadata"]["reflection_planes"]["R_horizon_local"], -40.)
        self.assertIn("conj(K_physical)", r["metadata"]["fft_action"])
        self.assertAlmostEqual(r["metadata"]["source_omega_c"], .41221024651953303, places=11)

    def test_independent_pde_and_padding(self):
        r = self.record
        self.assertAlmostEqual(r["independent"]["raw_late_relative_L2"], .0052760383451430784, places=11)
        self.assertLess(r["independent"]["negative_lag_kernel_L2_fraction"], .001)
        other = reconstruct(r["source"], r["t"], r["scattering"], nfft=2*r["response"]["nfft"])
        m = (r["t"] >= 190) & (r["t"] <= 500)
        ratio = np.linalg.norm((other["correction"]-r["response"]["correction"])[m])/np.linalg.norm(r["response"]["correction"][m])
        self.assertLess(ratio, 5e-5)
        self.assertLess(other["global_local_identity_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
