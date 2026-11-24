import unittest

import numpy as np

from eco_library.run import build_pilot, _paired_budget
from eco_library.schema import RunSpec
from eco_library.convergence import WINDOWS, assess, common_series


class PlanTests(unittest.TestCase):
    def test_matched_references(self):
        p = build_pilot()
        self.assertEqual(len(p["records"]), 13)
        specs = {k: RunSpec.from_dict(v) for k, v in p["records"].items()}
        for seq in p["sequences"]:
            hs = np.array([specs[rid].dx for rid in seq["record_ids"]])
            np.testing.assert_allclose(hs[:-1]/hs[1:], 2)
            if seq["bh_reference_ids"]:
                rhs = [specs[rid].dx for rid in seq["bh_reference_ids"]]
                np.testing.assert_array_equal(hs, rhs)
                for rid in seq["bh_reference_ids"]:
                    self.assertEqual(specs[rid].x_left, -250)
        self.assertEqual(specs[p["baseline_replay_id"]].x_left, -140)

    def test_budget_never_omits_bh(self):
        t = np.arange(521.)
        a = {n: {"fine_error_estimate_L2": .01} for n in WINDOWS}
        b = {n: {"fine_error_estimate_L2": None} for n in WINDOWS}
        out = _paired_budget(a, b, t, np.ones_like(t))
        self.assertTrue(all(v["state"] == "UNQUANTIFIED" for v in out.values()))
        b = {n: {"fine_error_estimate_L2": .02} for n in WINDOWS}
        out = _paired_budget(a, b, t, np.ones_like(t))
        self.assertTrue(all(v["error_estimate_L2"] == .03 for v in out.values()))
        self.assertTrue(all(v["is_error_bound"] is False for v in out.values()))

    def test_robustness_controls(self):
        p = build_pilot(with_controls=True)
        self.assertEqual(len(p["records"]), 24)
        self.assertEqual(len(p["controls"]), 11)
        for c in p["controls"]:
            b, v = [RunSpec.from_dict(p["records"][c[k]]) for k in ("base_id", "variant_id")]
            self.assertEqual(b.dx, v.dx)
            self.assertEqual(b.experiment.fingerprint, v.experiment.fingerprint)
            if c["kind"] == "HALF_TIMESTEP":
                self.assertEqual(v.cfl, b.cfl/2)
                self.assertEqual(v.cfl*v.sample_every, b.cfl*b.sample_every)

    def test_invalid_refinement_inputs(self):
        t = np.arange(10.)
        ys = [np.ones(10)]*3
        for tt, yy in ((t[::-1], ys), (t, [np.full(10, np.nan)]*3), (t, [np.ones(9)]*3)):
            with self.assertRaises(ValueError):
                assess(tt, yy, [.2, .1, .05], expected_order=2)
        with self.assertRaises(ValueError):
            common_series([], dt=.1)
        with self.assertRaises(ValueError):
            common_series([{"t": t, "psi": ys[0]}], dt=0)


if __name__ == "__main__":
    unittest.main()
