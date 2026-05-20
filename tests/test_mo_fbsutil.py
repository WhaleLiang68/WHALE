import unittest
from types import SimpleNamespace

import numpy as np

from src.utils.MO_FBSUtil import MO_FBSUtil


class TestMOFBSUtil(unittest.TestCase):
    def test_loop_and_vectorized_adjacency_match(self):
        samples = [
            (
                np.array([0.0, 2.5, 5.5]),
                np.array([0.0, 0.0, 0.0]),
                np.array([2.0, 3.0, 2.0]),
                np.array([1.0, 1.0, 1.0]),
            ),
            (
                np.array([0.0, 0.0, 3.0, 3.0]),
                np.array([0.0, 2.0, 0.0, 2.0]),
                np.array([2.0, 2.0, 2.0, 2.0]),
                np.array([2.0, 2.0, 2.0, 2.0]),
            ),
            (
                np.array([1.0, 4.5, 8.0, 8.0]),
                np.array([1.0, 1.0, 1.0, 4.0]),
                np.array([2.0, 3.0, 2.0, 2.0]),
                np.array([2.0, 2.0, 2.0, 3.0]),
            ),
        ]

        for fac_x, fac_y, fac_b, fac_h in samples:
            n = len(fac_x)
            loop_matrix = MO_FBSUtil._get_adjacency_length_loop(fac_x, fac_y, fac_b, fac_h, n)
            vec_matrix = MO_FBSUtil._get_adjacency_length_vectorized(fac_x, fac_y, fac_b, fac_h, n)
            self.assertTrue(np.allclose(loop_matrix, vec_matrix, atol=1e-9, rtol=1e-9))

    def test_calculate_objectives_respects_aspect_limits(self):
        fac_x = np.array([0.0, 2.5])
        fac_y = np.array([0.0, 0.0])
        fac_b = np.array([2.0, 3.0])
        fac_h = np.array([1.0, 1.0])
        rel_matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
        dist_req_matrix = np.array([[0.0, 3.0], [3.0, 0.0]])
        aspect_limits = np.array([2.0, 2.0])

        objectives = MO_FBSUtil.calculate_objectives(
            fac_x,
            fac_y,
            fac_b,
            fac_h,
            mhc=10.0,
            n=2,
            rel_matrix=rel_matrix,
            dist_req_matrix=dist_req_matrix,
            aspect_limits=aspect_limits,
        )

        self.assertAlmostEqual(objectives[0], 10.0)
        self.assertAlmostEqual(objectives[1], 5.0)
        self.assertAlmostEqual(objectives[2], 7.5)
        self.assertAlmostEqual(objectives[3], 0.0)

    def test_calculate_ar_scores_uses_paper_triangular_satisfaction(self):
        aspect_ratios, scores = MO_FBSUtil.calculate_ar_scores(
            fac_b=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            fac_h=np.ones(5),
            aspect_limits=np.full(5, 2.5),
        )

        np.testing.assert_allclose(aspect_ratios, np.array([1.0, 1.5, 2.0, 2.5, 3.0]))
        np.testing.assert_allclose(scores, np.array([0.0, 1.0, 0.5, 0.0, 0.0]), atol=1e-12)

    def test_compare_solution_quality_prefers_feasible_then_pareto(self):
        feasible = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([10.0, -4.0, -4.0, -0.9]),
            fitness=0.2,
        )
        infeasible = SimpleNamespace(
            current_is_feasible=False,
            current_d_inf=1,
            constraint_violation=1.5,
            mo_objectives_min=np.array([8.0, -5.0, -5.0, -0.95]),
            fitness=1_000_100.0,
        )
        dominated = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([12.0, -3.0, -3.0, -0.7]),
            fitness=0.4,
        )

        self.assertEqual(MO_FBSUtil.compare_solution_quality(feasible, infeasible), -1)
        self.assertEqual(MO_FBSUtil.compare_solution_quality(feasible, dominated), -1)
        self.assertEqual(MO_FBSUtil.compare_solution_quality(dominated, feasible), 1)

    def test_update_archive_and_representative(self):
        candidates = []
        candidate_a = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([10.0, -5.0, -4.0, -0.8]),
            MHC=10.0,
            fitness=0.3,
        )
        candidate_b = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([9.0, -4.0, -5.0, -0.85]),
            MHC=9.0,
            fitness=0.25,
        )
        candidate_c = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([12.0, -3.0, -3.0, -0.7]),
            MHC=12.0,
            fitness=0.6,
        )

        candidates, inserted_a, _ = MO_FBSUtil.update_pareto_archive(candidates, candidate_a, max_size=8)
        candidates, inserted_b, _ = MO_FBSUtil.update_pareto_archive(candidates, candidate_b, max_size=8)
        candidates, inserted_c, _ = MO_FBSUtil.update_pareto_archive(candidates, candidate_c, max_size=8)

        self.assertTrue(inserted_a)
        self.assertTrue(inserted_b)
        self.assertFalse(inserted_c)
        self.assertEqual(len(candidates), 2)

        representative, score, index = MO_FBSUtil.select_representative_solution(candidates)
        self.assertIsNotNone(representative)
        self.assertTrue(np.isfinite(score))
        self.assertIn(index, {0, 1})

    def test_select_nfcs_subset_limits_archive_size(self):
        candidates = [
            SimpleNamespace(
                current_is_feasible=True,
                current_d_inf=0,
                constraint_violation=0.0,
                mo_objectives_min=np.array([10.0 + i, -5.0 + 0.2 * i, -4.0 - 0.1 * i, -0.8 - 0.01 * i]),
                MHC=10.0 + i,
            )
            for i in range(6)
        ]
        subset = MO_FBSUtil.select_nfcs_subset(candidates, max_size=3)
        self.assertEqual(len(subset), 3)

    def test_archive_hypervolume_uses_union_volume(self):
        candidates = [
            SimpleNamespace(mo_objectives_min=np.array([0.2, 0.3, 0.2, 0.3], dtype=float)),
            SimpleNamespace(mo_objectives_min=np.array([0.3, 0.2, 0.3, 0.2], dtype=float)),
        ]

        hv = MO_FBSUtil.archive_hypervolume(
            candidates,
            ideal=np.zeros(4, dtype=float),
            nadir=np.ones(4, dtype=float),
            reference_margin=0.1,
        )

        expected = (0.2 * 0.1 * 0.2 * 0.1) + (0.1 * 0.2 * 0.1 * 0.2) - (0.1 * 0.1 * 0.1 * 0.1)
        self.assertAlmostEqual(hv, expected, places=12)

    def test_update_archive_can_require_candidate_retained_after_trim(self):
        archive = [
            SimpleNamespace(
                current_is_feasible=True,
                current_d_inf=0,
                constraint_violation=0.0,
                mo_objectives_min=np.array([0.0, 1.0, 1.0, 1.0], dtype=float),
            ),
            SimpleNamespace(
                current_is_feasible=True,
                current_d_inf=0,
                constraint_violation=0.0,
                mo_objectives_min=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            ),
        ]
        middle_candidate = SimpleNamespace(
            current_is_feasible=True,
            current_d_inf=0,
            constraint_violation=0.0,
            mo_objectives_min=np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
        )

        updated_default, inserted_default, _ = MO_FBSUtil.update_pareto_archive(
            archive,
            middle_candidate,
            max_size=2,
        )
        updated_strict, inserted_strict, _ = MO_FBSUtil.update_pareto_archive(
            archive,
            middle_candidate,
            max_size=2,
            require_candidate_retained=True,
        )

        self.assertTrue(inserted_default)
        self.assertEqual(len(updated_default), 2)
        self.assertFalse(any(np.allclose(item.mo_objectives_min, middle_candidate.mo_objectives_min) for item in updated_default))
        self.assertFalse(inserted_strict)
        self.assertEqual(len(updated_strict), 2)
        self.assertTrue(all(np.allclose(left.mo_objectives_min, right.mo_objectives_min) for left, right in zip(updated_strict, archive)))


if __name__ == "__main__":
    unittest.main()
