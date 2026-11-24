"""Result files: mapping rules, identities, evidence labels and file integrity."""
import json
import math
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from eco_library import interface_export as ie
from eco_library.catalogue_base import MATCH_RULES, reference_match, sequence_match
from eco_library.schema import Experiment

REPO = Path(__file__).resolve().parents[1]
E0_METADATA = {"source_global": 55., "observer_global": 95., "source_areal_radius_over_M": 48.69885506594945,
               "observer_areal_radius_over_M": 87.48950707288445}


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.e0 = asdict(Experiment())

    def test_request_example_identities(self):
        # Values printed in the analysis request, sections 6.2-6.3.
        self.assertEqual(Experiment().fingerprint, ie.REQUEST_VECTORS["library_experiment_fingerprint_E0"])
        for model in ("GS", "WH"):
            v = ie.REQUEST_VECTORS[model]
            self.assertEqual(ie.library_point_id(model, v["params"]), v["point_id"])
            self.assertTrue(ie.analysis_param_hash(model, v["params"])[0].startswith(v["param_hash_prefix"]))

    def test_experiment_mapping_follows_the_table(self):
        inputs = ie.analysis_experiment_inputs(self.e0, E0_METADATA)
        self.assertEqual((inputs["time_convention"], inputs["units"], inputs["pulse_shape"], inputs["pulse_direction"]),
                         ("exp(-i omega t); Im(omega) < 0 damped", "G=c=1; M_ADM=1", "gaussian", "left-going"))
        self.assertEqual((inputs["x_source_global"], inputs["x_observer_global"], inputs["F_in"]), (55., 95., 1.))
        self.assertAlmostEqual(ie.tortoise(inputs["source_areal_radius"]), 55., delta=1e-9)

    def test_anything_outside_the_table_is_refused_not_mapped(self):
        changes = ({"time_convention": "exp(+i*omega*t)"}, {"units": "G=c=1"}, {"amplitude": 2.},
                   {"initial_direction": "right"}, {"pulse_shape": "sech"}, {"ell": 3},
                   {"physical_fourier_convention": "integral psi(t)*exp(-i*omega*t) dt"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ie.analysis_experiment_inputs({**self.e0, **change}, E0_METADATA)
        with self.assertRaises(ValueError):
            ie.analysis_experiment_inputs(self.e0, {**E0_METADATA, "observer_areal_radius_over_M": 87.49})
        with self.assertRaises(ValueError):
            ie.analysis_experiment_inputs(self.e0, {**E0_METADATA, "source_global": 55.5})

    def test_qh_hierarchy_is_named_not_transformed_and_gamma_is_reported_unmapped(self):
        digest, named, unmapped = ie.analysis_param_hash("QH", {"EplM": 1e10, "gamma": 1.})
        self.assertEqual(named, {"EplM_hierarchy": 1e10})
        self.assertEqual(unmapped, {"gamma": 1.})
        _, other, _ = ie.analysis_param_hash("QH", {"EplM": 1e10, "gamma": 2.})
        self.assertEqual(named, other)  # the collision the unmapped field documents
        with self.assertRaises(ValueError):
            ie.analysis_param_hash("GS", {"a_over_M": 2.})


class MatchTests(unittest.TestCase):
    META = {"solver_id": "s", "x_left": -250., "x_right": 400., "actual_dx": .05, "Nx": 13001, "dt": .0175,
            "observer_global": 95., "observer_native": 95.}

    def test_identical_records_match_on_every_item(self):
        match, detail = reference_match(self.META, dict(self.META))
        self.assertEqual(set(match), set(MATCH_RULES))
        self.assertTrue(all(match.values()))
        self.assertEqual(detail["dt_relative_difference"], 0.)

    def test_nominal_agreement_is_not_a_match(self):
        bh = {**self.META, "dt": np.nextafter(.0175, 1.), "actual_dx": np.nextafter(.05, 1.)}
        match, _ = reference_match(self.META, bh)
        self.assertFalse(match["dt"] or match["grid"] or match["extraction"])
        self.assertTrue(match["solver"] and match["domain"])
        self.assertFalse(reference_match(self.META, {**self.META, "solver_id": "other"})[0]["solver"])

    def test_sequence_match_needs_every_level(self):
        other = {**self.META, "dt": .0176}
        match, levels = sequence_match([self.META]*3, [self.META, self.META, other])
        self.assertFalse(match["dt"])
        self.assertEqual(len(levels), 3)
        with self.assertRaises(ValueError):
            sequence_match([self.META]*3, [self.META]*2)


class EvidenceLabelTests(unittest.TestCase):
    def test_variation_states(self):
        R = np.exp(1j*np.linspace(0., 1., 5))
        self.assertEqual(ie._variation_state(R, {"a": R+1e-9, "b": R-1e-9})[0], "VERIFIED_BY_VARIATION")
        self.assertEqual(ie._variation_state(R, {"a": R+1e-9, "b": R+1e-4})[0], "NOT_CONVERGED")
        self.assertEqual(ie._variation_state(R, {"a": R.copy(), "b": R.copy()})[0], "UNQUANTIFIED")
        self.assertEqual(ie._variation_state(R, {"a": R+1e-9})[0], "UNQUANTIFIED")
        bad = R.copy()
        bad[2] = np.nan
        self.assertEqual(ie._variation_state(R, {"a": R+1e-9, "b": bad})[0], "UNQUANTIFIED")

    def test_independent_gs_interface_agrees_with_closed_form_and_frozen_ode(self):
        from scipy.integrate import solve_ivp
        from eco_library.frozen_declarations import load_declarations
        from eco_library.transfer import runtime
        w = np.array([.05, .3, .45, .9, 1.4])
        closed = runtime()["gs_shell_reflectivity"](w, .44)
        rt = load_declarations({13: ("gravastar_regular_data",)}, {"solve_ivp": solve_ivp}, label="export_test")
        a, f = 1/.44, 1-2*.44
        frozen = []
        for z in w:
            q, pr = rt["gravastar_regular_data"](a, z)
            frozen.append(np.linalg.solve([[1., -q], [1j*z, -f*pr]], [-1., 1j*z])[0])
        notebook = ie.gs_interface_by_radial_ode(w, .44, **ie.GS_INTERFACE_VARIANTS["radial_ode_notebook_settings"])
        tight = ie.gs_interface_by_radial_ode(w, .44, **ie.GS_INTERFACE_VARIANTS["radial_ode_tight"])
        self.assertLess(np.max(np.abs(notebook-np.asarray(frozen))), 1e-12)  # re-implementation = frozen ODE
        self.assertLess(np.max(np.abs(tight-closed)), 1e-9)
        self.assertLess(np.max(np.abs(notebook-closed)), 1e-8)


class FileIntegrityTests(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        header = {"version": ie.VERSION, "entry_id": "synthetic-test-only"}
        arrays = {"common": {"t": .07*np.arange(4), "eco": np.ones((3, 4))}}
        with tempfile.TemporaryDirectory() as tmp:
            path = ie.write_export(Path(tmp)/"x.h5", header, arrays)
            got, data = ie.read_export(path)
            self.assertEqual(got, header)
            np.testing.assert_array_equal(data["common"]["eco"], arrays["common"]["eco"])
            with self.assertRaises(ValueError):
                ie.write_export(path, header, arrays)  # never overwritten
            with h5py.File(path, "r+") as h:
                h["arrays/common/eco"][0, 0] = 2.
            with self.assertRaises(ValueError):
                ie.read_export(path)
            path2 = ie.write_export(Path(tmp)/"y.h5", header, arrays)
            with h5py.File(path2, "r+") as h:
                h.attrs["header"] = json.dumps({**header, "entry_id": "changed"})
            with self.assertRaises(ValueError):
                ie.read_export(path2)


if __name__ == "__main__":
    unittest.main()
