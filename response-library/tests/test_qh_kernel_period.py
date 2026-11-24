import unittest
from unittest.mock import patch

import numpy as np

from scripts.qh_causal_control import wall_kernel
from scripts.qh_kernel_period_controls import period_wall


class KernelPeriodTests(unittest.TestCase):
    def test_factor_guard_precedes_allocation(self):
        for value in (True, 0, 3, 1., None):
            with patch("scripts.qh_kernel_period_controls.wall_kernel") as build:
                with self.assertRaises(ValueError):
                    period_wall(value)
                build.assert_not_called()

    def test_default_period_preserves_exact_kernel(self):
        reference, control = wall_kernel(.025), period_wall(1)
        np.testing.assert_array_equal(control["impulse"], reference["impulse"])
        self.assertEqual(reference["full_kernel_sha256"], control["full_kernel_sha256"])
        self.assertEqual(reference["period_M"], control["period_M"])


if __name__ == "__main__":
    unittest.main()
