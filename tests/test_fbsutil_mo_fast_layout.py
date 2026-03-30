import unittest

import numpy as np

from src.utils import FBSUtil
from src.utils.FBSModel import FBSModel


class TestFBSUtilFastLayout(unittest.TestCase):
    def test_get_coordinates_fast_matches_original(self):
        cases = [
            (
                [3, 1, 2, 4],
                [0, 1, 0, 1],
                np.array([6.0, 8.0, 10.0, 4.0]),
                5.0,
            ),
            (
                [5, 2, 4, 1, 3],
                [0, 0, 1, 0, 1],
                np.array([4.0, 7.0, 3.0, 9.0, 5.0]),
                7.0,
            ),
            (
                [2, 6, 1, 4, 3, 5],
                [0, 1, 0, 0, 1, 1],
                np.array([5.0, 11.0, 4.0, 8.0, 6.0, 7.0]),
                9.0,
            ),
        ]

        for permutation, bay, area, height in cases:
            with self.subTest(permutation=permutation, bay=bay):
                model = FBSModel(permutation=permutation, bay=bay)
                expected = FBSUtil.getCoordinates_mao(model, area, height)
                actual = FBSUtil.getCoordinates_mao_fast(model, area, height)
                for expected_values, actual_values in zip(expected, actual):
                    self.assertTrue(np.allclose(expected_values, actual_values, atol=1e-9, rtol=1e-9))

    def test_evaluate_layout_fast_matches_original(self):
        area = np.array([4.0, 7.0, 3.0, 9.0, 5.0])
        flow = np.array(
            [
                [0.0, 2.0, 1.0, 0.5, 1.5],
                [2.0, 0.0, 1.2, 0.0, 0.8],
                [1.0, 1.2, 0.0, 1.1, 0.4],
                [0.5, 0.0, 1.1, 0.0, 2.5],
                [1.5, 0.8, 0.4, 2.5, 0.0],
            ],
            dtype=float,
        )
        aspect_limits = np.array([2.0, 2.5, 1.8, 2.2, 3.0], dtype=float)
        model = FBSModel(permutation=[5, 2, 4, 1, 3], bay=[0, 0, 1, 0, 1])

        for distance_metric in ("manhattan", "euclidean"):
            with self.subTest(distance_metric=distance_metric):
                expected = FBSUtil.evaluate_layout(
                    model,
                    area,
                    7.0,
                    flow,
                    aspect_limits,
                    v_worst=123.0,
                    k_penalty=2,
                    distance_metric=distance_metric,
                )
                actual = FBSUtil.evaluate_layout_fast(
                    model,
                    area,
                    7.0,
                    flow,
                    aspect_limits,
                    v_worst=123.0,
                    k_penalty=2,
                    distance_metric=distance_metric,
                )
                array_keys = [
                    "fac_x",
                    "fac_y",
                    "fac_b",
                    "fac_h",
                    "fac_aspect_ratio",
                    "D",
                    "TM",
                    "infeasible_mask",
                    "lower_bounds",
                    "upper_bounds",
                    "aspect_limits",
                ]
                scalar_keys = ["mhc", "cost", "d_inf", "is_feasible", "v_worst"]
                for key in array_keys:
                    self.assertTrue(np.allclose(expected[key], actual[key], atol=1e-9, rtol=1e-9), key)
                for key in scalar_keys:
                    if isinstance(expected[key], (bool, np.bool_)):
                        self.assertEqual(expected[key], actual[key], key)
                    else:
                        self.assertAlmostEqual(expected[key], actual[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
