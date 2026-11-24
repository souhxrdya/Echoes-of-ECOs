from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from eco_library.adapter_archive import write_archive, read_archive


class AdapterArchiveTests(unittest.TestCase):
    def fixture(self):
        return {"t": np.arange(4.), "psi": np.ones(4), "x": np.arange(10.),
                "r": np.arange(10.), "V": np.zeros(10), "metadata": {"missing": None},
                "checks": {"finite": {"passed": True}},
                "scattering": {"R": np.array([1j, 2j]), "plane": -40., "band": [.1, .9]},
                "nested": {"source": {"field": np.arange(8.)}}}

    def test_roundtrip_overwrite_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/"record.h5"
            req = {"model": "QH", "params": {"EplM": 1e8}}
            write_archive(path, req, self.fixture(), {}, root)
            out = read_archive(path, expected_request=req)["result"]
            np.testing.assert_array_equal(out["scattering"]["R"], self.fixture()["scattering"]["R"])
            self.assertIsNone(out["metadata"]["missing"])
            with self.assertRaises(FileExistsError):
                write_archive(path, req, self.fixture(), {}, root)
            with h5py.File(path, "r+") as h:
                h["result/scattering/R"][0] = 4j
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_archive(path)

    def test_metadata_tampering_and_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            req = {"model": "BS"}
            with self.assertRaises(ValueError):
                write_archive(root/"baseline"/"x.h5", req, self.fixture(), {}, root)
            path = root/"x.h5"
            write_archive(path, req, self.fixture(), {}, root)
            with h5py.File(path, "r+") as h:
                h["result/scattering/plane"].attrs["value"] = "0.0"
            with self.assertRaisesRegex(ValueError, "metadata checksum"):
                read_archive(path)

    def test_invalid_payload_not_published(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.fixture()
            result["scattering"]["R"][0] = np.nan
            with self.assertRaises(ValueError):
                write_archive(root/"bad.h5", {"model": "QH"}, result, {}, root)
            self.assertEqual(list(root.iterdir()), [])

    def test_empty_groups_and_zero_dimensional_arrays_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.fixture()
            result.update(empty={}, scalar_array=np.array(2j))
            path = root/"record.h5"
            write_archive(path, {"model": "BS"}, result, {}, root)
            actual = read_archive(path)["result"]
            self.assertIsInstance(actual["scalar_array"], np.ndarray)
            self.assertEqual(np.shape(actual["scalar_array"]), ())
            self.assertEqual(actual["empty"], {})
            copied = root/"copy.h5"
            write_archive(copied, {"model": "BS"}, actual, {}, root)
            self.assertEqual(read_archive(copied)["result"]["scalar_array"], np.array(2j))
            with h5py.File(path, "r+") as f:
                del f["result/empty"]
            with self.assertRaisesRegex(ValueError, "membership"):
                read_archive(path)
