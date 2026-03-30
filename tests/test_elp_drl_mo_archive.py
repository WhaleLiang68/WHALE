import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.algorithms.ELP_DRL_MO import ELP


class TestELPDRLMOArchive(unittest.TestCase):
    @staticmethod
    def _make(mhc, cr, dr, ar):
        return SimpleNamespace(
            current_is_feasible=True, current_d_inf=0, constraint_violation=0.0,
            mo_objectives_raw=np.array([mhc, cr, dr, ar], dtype=float),
            mo_objectives_min=np.array([mhc, -cr, -dr, -ar], dtype=float),
            MHC=float(mhc), CR=float(cr), DR=float(dr), AR=float(ar), fitness=math.inf,
        )

    def test_light_clone_preserves_static_fields_and_layout_encoding(self):
        from src.utils.FBSModel import FBSModel

        solver = ELP.__new__(ELP)
        solution = SimpleNamespace(
            fbs_model=FBSModel([1, 2, 3], [0, 1, 1]),
            areas=np.array([1.0, 2.0, 3.0]),
            H=6.0,
            F=np.ones((3, 3), dtype=float),
            aspect_limits=np.array([2.0, 2.0, 2.0]),
            actions={10: 'segment_insert'},
            some_static='ok',
        )

        clone = solver._light_clone_solution(solution)
        self.assertIsNot(clone, solution)
        self.assertIsNot(clone.fbs_model, solution.fbs_model)
        self.assertEqual(clone.fbs_model.permutation, solution.fbs_model.permutation)
        self.assertEqual(clone.fbs_model.bay, solution.fbs_model.bay)
        self.assertIs(clone.areas, solution.areas)
        self.assertEqual(clone.some_static, 'ok')

    def test_archive_keeps_nondominated_feasible_solutions(self):
        solver = ELP.__new__(ELP)
        solver.mo_weights = np.full(4, 0.25, dtype=float)
        solver.archive_limit = 8
        solver.pareto_archive = []
        solver.representative_solution = None
        solver.representative_decision_score = math.inf
        solver.representative_archive_index = None
        solver.mo_ideal = None
        solver.mo_nadir = None
        solver.mo_worst_feasible_mhc = None
        solver._last_transition_meta = {}
        solver.feasible_solution_count = 0
        solver.gbest_update_count = 0
        solver.best_history = []
        solver.best_feasible_cost = math.inf
        solver.best_energy = math.inf
        solver.worst_feasible_cost = None
        solver.best_feasible_solution = None
        solver.gbest = None
        solver.true_gbest = None
        solver.s = None

        self.assertTrue(solver._observe_feasible_state(self._make(10.0, 5.0, 4.0, 0.80)))
        self.assertTrue(solver._observe_feasible_state(self._make(9.0, 4.0, 5.0, 0.85)))
        self.assertFalse(solver._observe_feasible_state(self._make(12.0, 3.0, 3.0, 0.70)))

        self.assertEqual(len(solver.pareto_archive), 2)
        self.assertIsNotNone(solver.representative_solution)
        self.assertIn(float(solver.representative_solution.MHC), {9.0, 10.0})


if __name__ == "__main__":
    unittest.main()
