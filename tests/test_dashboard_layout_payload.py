import unittest
from unittest.mock import patch

import numpy as np

from src.dashboard import server


class TestDashboardLayoutPayload(unittest.TestCase):
    def test_single_objective_solution_builds_layout_payload(self):
        solution = '[[14, 39, 9, 46], [33, 47, 26, 15], [7, 49, 44, 29, 5, 51, 48], [54, 25, 38, 43, 27, 52, 16, 50, 11, 40, 37, 31], [53, 57, 59, 60, 32, 58], [62, 35, 3, 2, 61, 12, 56, 21, 20, 18, 30, 8, 41, 19], [10, 23, 1, 28, 24, 22, 36, 42, 13], [17, 45, 55, 34, 4, 6]]'
        payload = server.build_layout_payload_from_solution('Du62', solution)
        self.assertEqual(payload['instance'], 'Du62')
        self.assertEqual(payload['facilityCount'], 62)
        self.assertTrue(payload['rectangles'])
        self.assertIn('aspectLimit', payload['rectangles'][0])

    def test_ab20_layout_uses_forced_paper_flow_matrix(self):
        expected = np.full((20, 20), 3.0, dtype=float)
        solution = str([[idx + 1] for idx in range(20)])
        captured = {}

        def fake_evaluate_layout(*args, **kwargs):
            captured["flow_matrix"] = np.asarray(args[3], dtype=float).copy()
            n = int(captured["flow_matrix"].shape[0])
            zeros = np.zeros(n, dtype=float)
            return {
                "fac_x": zeros.copy(),
                "fac_y": zeros.copy(),
                "fac_b": np.ones(n, dtype=float),
                "fac_h": np.ones(n, dtype=float),
                "fac_aspect_ratio": np.ones(n, dtype=float),
                "aspect_limits": np.ones(n, dtype=float),
                "TM": captured["flow_matrix"].copy(),
                "mhc": 0.0,
                "cost": 0.0,
                "d_inf": 0,
                "is_feasible": True,
            }

        with patch("src.dashboard.server.FlowMatrixUtil.load_ab20_1963_matrix", return_value=(expected, None)), \
             patch("src.dashboard.server.FBSUtil.getAreaData", return_value=(np.ones(20), np.ones(20))), \
             patch("src.dashboard.server.FBSUtil.evaluate_layout", side_effect=fake_evaluate_layout):
            payload = server.build_layout_payload_from_solution("AB20-ar50", solution)

        np.testing.assert_array_equal(captured["flow_matrix"], expected)
        self.assertEqual(payload["instance"], "AB20-ar50")


if __name__ == '__main__':
    unittest.main()
