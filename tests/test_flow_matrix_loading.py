import unittest
from unittest.mock import mock_open, patch

import numpy as np

from src.utils.DataExtractor import DataProcessingEnv


class FlowMatrixLoadingTests(unittest.TestCase):
    def _build_payload(self, flow_matrix):
        return (
            {"toy": flow_matrix.shape[0]},
            {"toy": flow_matrix},
            {"toy": object()},
            {"toy": 10.0},
            {"toy": 20.0},
        )

    def _build_env(self, flow_matrix):
        payload = self._build_payload(flow_matrix)
        with patch("builtins.open", mock_open(read_data=b"mock")), \
             patch("src.utils.DataExtractor.pickle.load", return_value=payload), \
             patch(
                 "src.utils.DataExtractor.FBSUtil.getAreaData",
                 return_value=(np.ones(flow_matrix.shape[0]), np.ones(flow_matrix.shape[0])),
             ), \
             patch("src.utils.DataExtractor.FBSUtil.get_instance_aspect_limit", return_value=1.0):
            return DataProcessingEnv(instance="toy")

    def test_upper_triangular_flow_matrix_is_completed(self):
        raw = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
        env = self._build_env(raw)
        expected = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(env.F, expected)

    def test_full_flow_matrix_is_kept_as_is(self):
        raw = np.array(
            [
                [0.0, 4.0, 2.0],
                [1.0, 0.0, 3.0],
                [7.0, 5.0, 0.0],
            ]
        )
        env = self._build_env(raw)
        np.testing.assert_array_equal(env.F, raw)

    def test_ab20_variants_are_forced_to_use_paper_csv_matrix(self):
        raw = np.zeros((20, 20), dtype=float)
        payload = (
            {"AB20-ar50": 20},
            {"AB20-ar50": raw},
            {"AB20-ar50": object()},
            {"AB20-ar50": 10.0},
            {"AB20-ar50": 20.0},
        )
        expected = np.full((20, 20), 7.0, dtype=float)
        with patch("builtins.open", mock_open(read_data=b"mock")), \
             patch("src.utils.DataExtractor.pickle.load", return_value=payload), \
             patch(
                 "src.utils.DataExtractor.FBSUtil.getAreaData",
                 return_value=(np.ones(20), np.ones(20)),
             ), \
             patch("src.utils.DataExtractor.FBSUtil.get_instance_aspect_limit", return_value=1.0), \
             patch("src.utils.DataExtractor.FlowMatrixUtil.load_ab20_1963_matrix", return_value=(expected, None)):
            env = DataProcessingEnv(instance="AB20-ar50")

        np.testing.assert_array_equal(env.F, expected)


if __name__ == "__main__":
    unittest.main()
