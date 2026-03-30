import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.MO_DataGenerator import MO_DataGenerator


class TestMODataGenerator(unittest.TestCase):
    def test_load_or_generate_is_deterministic_per_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rel_a, dist_a = MO_DataGenerator.load_or_generate_data(5, instance_name="AB20", data_dir=tmp_dir)
            rel_b, dist_b = MO_DataGenerator.load_or_generate_data(5, instance_name="AB20", data_dir=tmp_dir)

            self.assertTrue(np.array_equal(rel_a, rel_b))
            self.assertTrue(np.array_equal(dist_a, dist_b))
            self.assertTrue((Path(tmp_dir) / "AB20_MO_matrices.pkl").exists())


if __name__ == "__main__":
    unittest.main()
