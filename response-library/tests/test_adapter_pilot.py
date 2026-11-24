import unittest

from eco_library.adapter_pilot import plan
from eco_library.adapter_archive import request_id


class AdapterPlanTests(unittest.TestCase):
    def test_bounded_distinct_jobs(self):
        p = plan()
        self.assertEqual(len(p["jobs"]), 10)
        self.assertEqual(len({request_id(r) for r in p["jobs"].values()}), 10)
        for model in ("BS", "QH"):
            requests = [p["jobs"][model+"_"+level] for level in ("coarse", "medium", "fine")]
            self.assertEqual([r["settings"]["dx"] for r in requests], [.2, .1, .05])
            self.assertTrue(all(r["params"] == requests[0]["params"] for r in requests))
        self.assertEqual(p["jobs"]["QH_frozen_replay"]["settings"]["x_left"], -140.)
        self.assertEqual(p["jobs"]["QH_medium"]["settings"]["x_left"], -250.)
        self.assertEqual(p["jobs"]["QH_medium"]["settings"]["independent_dx"], .05)
