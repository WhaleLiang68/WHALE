import pickle
import unittest

import numpy as np

from src.utils import config
from src.utils.DataExtractor import DataProcessingEnv


class Du62RealFlowMatrixLoadingTests(unittest.TestCase):
    def test_du62_upper_triangular_flow_matrix_is_completed(self):
        with open(config.FILE_PATH, "rb") as file:
            problems, flow_matrices, sizes, layout_widths, layout_lengths = pickle.load(file)

        raw = np.asarray(flow_matrices["Du62"])
        self.assertTrue(np.allclose(np.tril(raw, -1), 0), "Du62 raw flow matrix should be upper triangular in the dataset")

        env = DataProcessingEnv(instance="Du62")
        expected = raw + raw.T - np.diag(np.diag(raw))

        np.testing.assert_array_equal(env.F, expected)
        self.assertTrue(np.allclose(env.F, env.F.T), "Loaded Du62 flow matrix should be symmetric after completion")
        self.assertEqual(env.F.shape, (62, 62))


if __name__ == "__main__":
    unittest.main()
