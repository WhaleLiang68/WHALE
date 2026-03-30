import unittest

from src.dashboard import server


class TestDashboardLayoutPayload(unittest.TestCase):
    def test_single_objective_solution_builds_layout_payload(self):
        solution = '[[14, 39, 9, 46], [33, 47, 26, 15], [7, 49, 44, 29, 5, 51, 48], [54, 25, 38, 43, 27, 52, 16, 50, 11, 40, 37, 31], [53, 57, 59, 60, 32, 58], [62, 35, 3, 2, 61, 12, 56, 21, 20, 18, 30, 8, 41, 19], [10, 23, 1, 28, 24, 22, 36, 42, 13], [17, 45, 55, 34, 4, 6]]'
        payload = server.build_layout_payload_from_solution('Du62', solution)
        self.assertEqual(payload['instance'], 'Du62')
        self.assertEqual(payload['facilityCount'], 62)
        self.assertTrue(payload['rectangles'])
        self.assertIn('aspectLimit', payload['rectangles'][0])


if __name__ == '__main__':
    unittest.main()
