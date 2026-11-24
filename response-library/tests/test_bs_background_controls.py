import unittest

from scripts.bs_background_controls import plan


class BackgroundControlPlan(unittest.TestCase):
    def test_nested_refinement_and_matched_domain_spacing(self):
        p = plan()
        self.assertEqual(len(p["jobs"]), 6)
        settings = {k: v["settings"] for k, v in p["jobs"].items()}
        h = lambda key: (settings[key]["equilibrium_rmax"]-1e-5)/(settings[key]["background_samples"]-1)
        self.assertEqual(h("profile5001")/h("profile10001"), 2.)
        self.assertEqual(h("profile10001")/h("profile20001"), 2.)
        self.assertAlmostEqual(h("domain6001"), h("profile5001"), places=16)
        self.assertAlmostEqual(h("domain24001"), h("profile20001"), places=16)
        self.assertEqual(settings["baseline5000"]["background_samples"], 5000)
        self.assertTrue(all(v["dx"] == .1 for v in settings.values()))


if __name__ == "__main__":
    unittest.main()
