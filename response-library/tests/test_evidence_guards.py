"""Adversarial inputs must never become false zero-error evidence."""
import unittest

import numpy as np

from eco_library.adapter_pilot import difference
from eco_library.convergence import assess, common_series


class EvidenceGuards(unittest.TestCase):
    def test_complex_divergence_cannot_be_discarded(self):
        t = np.linspace(0, 1, 101)
        hs = [.2, .1, .05]
        ys = [np.sin(t) + 1j*np.cos(t)/h for h in hs]
        with self.assertRaisesRegex(ValueError, "real numeric"):
            assess(t, ys, hs, expected_order=2, windows={"w": (0, 1)})
        with self.assertRaisesRegex(ValueError, "real numeric"):
            common_series([{"t": t, "psi": ys[0]}])

    def test_strings_bools_objects_and_complex_times_are_not_numeric_evidence(self):
        t = np.linspace(0, 1, 101)
        invalid = [t.astype(str), t.astype(object), np.ones(len(t), dtype=bool),
                   t.astype(complex)]
        for values in invalid:
            for key in ("t", "psi"):
                with self.subTest(dtype=values.dtype, key=key), self.assertRaises(ValueError):
                    common_series([{**{"t": t, "psi": t}, key: values}])
            with self.subTest(dtype=values.dtype), self.assertRaises(ValueError):
                assess(t, [values]*3, [.2, .1, .05], expected_order=2)
        for kwargs in ({"expected_order": True}, {"expected_order": "2"},
                       {"expected_order": 2, "order_tolerance": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                assess(t, [t]*3, [.2, .1, .05], **kwargs)

    def test_partial_window_is_unquantified_not_truncated(self):
        t = np.arange(231.)
        a, b = {"t": t, "psi": np.sin(t/10)}, {"t": t, "psi": np.sin(t/10)+.01}
        out = difference(a, b)
        self.assertEqual(out["prompt"]["state"], "MEASURED_SENSITIVITY")
        self.assertGreater(out["prompt"]["difference_L2"], 0.)
        for name in ("late", "full"):
            self.assertEqual(out[name]["state"], "UNQUANTIFIED")
            self.assertEqual(out[name]["reason"], "RECORD_TOO_SHORT")
            self.assertIsNone(out[name]["difference_L2"])
            self.assertIsNone(out[name]["relative_difference"])

    def test_no_window_samples_never_yields_zero_error(self):
        t = np.arange(10.)
        record = {"t": t, "psi": np.ones_like(t)}
        for value in difference(record, record).values():
            self.assertEqual(value["state"], "UNQUANTIFIED")
            self.assertIsNone(value["difference_L2"])

    def test_invalid_native_grids_and_too_few_common_samples(self):
        for record in ({"t": [0., 1., 1.], "psi": [0., 1., 2.]},
                       {"t": [0., 1., 2.], "psi": [0., 1.]},
                       {"t": [0., .01], "psi": [0., 1.]}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                common_series([record])
        with self.assertRaises(ValueError):
            common_series([{"t": np.arange(10.), "psi": np.ones(10)}], dt=True)

    def test_full_support_retains_exact_zero_as_sensitivity_not_bound(self):
        t = np.arange(521.)
        record = {"t": t, "psi": np.cos(t/10)}
        for value in difference(record, record).values():
            self.assertEqual(value["state"], "MEASURED_SENSITIVITY")
            self.assertEqual(value["difference_L2"], 0.)
            self.assertIn("not a total error", value["meaning"])


if __name__ == "__main__":
    unittest.main()
