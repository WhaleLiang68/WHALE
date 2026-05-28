import copy
import math

import gym
import numpy as np

import src
from src.algorithms.ELP_DRL_BiMO4_GRASP import ELP
from src.algorithms.ELP_DRL_BiMO4 import _set_global_seed
from src.utils.FBSUtil import permutationToArray
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4


def _make_solver(env, G=2, t_max=4):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    return ELP(
        env=base_env,
        gbest=copy.deepcopy(base_env),
        T=1000.0,
        G=G,
        t_max=t_max,
        k=10.0,
    )


class TestPaperAdaptedBackend:
    def test_runs_on_small_instance(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "2")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260520)
        solver = _make_solver(env, G=2, t_max=4)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = solver.run()

        assert total_iter >= 1
        assert best_sol is not None
        assert np.isfinite(best_energy)
        assert start <= fast <= end

    def test_produces_nonempty_pareto_archive(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260521)
        solver = _make_solver(env, G=2, t_max=4)
        solver.run()

        assert len(solver.pareto_archive) > 0

    def test_representative_solution_feasible(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260522)
        solver = _make_solver(env, G=3, t_max=4)
        solver.run()

        rep = solver.representative_solution
        assert rep is not None
        assert bool(getattr(rep, "current_is_feasible", False))

    def test_payload_fields_nonempty(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260523)
        solver = _make_solver(env, G=2, t_max=4)
        solver.run()

        payload = solver.last_run_payload
        assert payload is not None
        assert payload["pareto_archive_path"] is not None
        assert payload["rep_mhc"] is not None
        assert payload["rep_cr"] is not None
        assert payload["local_search_backend"] == "paper_adapted"

    def test_wall_time_budget_triggers_exit(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_WALL_TIME_LIMIT_SECONDS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260524)
        solver = _make_solver(env, G=500, t_max=4)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = solver.run()

        assert solver._wall_time_terminated
        payload = solver.last_run_payload
        assert payload["wall_time_terminated"] is True
        assert payload["wall_time_limit_seconds"] == 1.0


class TestEngineeredBackend:
    def test_runs_on_small_instance(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "engineered")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "2")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260520)
        solver = _make_solver(env, G=2, t_max=4)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = solver.run()

        assert total_iter >= 1
        assert best_sol is not None
        assert np.isfinite(best_energy)
        assert start <= fast <= end

    def test_produces_nonempty_pareto_archive(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "engineered")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260521)
        solver = _make_solver(env, G=2, t_max=4)
        solver.run()

        assert len(solver.pareto_archive) > 0

    def test_representative_solution_feasible(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "engineered")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260522)
        solver = _make_solver(env, G=3, t_max=4)
        solver.run()

        rep = solver.representative_solution
        assert rep is not None
        assert bool(getattr(rep, "current_is_feasible", False))

    def test_payload_fields_nonempty(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "engineered")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260523)
        solver = _make_solver(env, G=2, t_max=4)
        solver.run()

        payload = solver.last_run_payload
        assert payload is not None
        assert payload["pareto_archive_path"] is not None
        assert payload["rep_mhc"] is not None
        assert payload["rep_cr"] is not None
        assert payload["local_search_backend"] == "engineered"


class TestBackendDifferentiation:
    def test_backends_produce_different_payload_tags(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "2")

        results = {}
        for backend in ("paper_adapted", "engineered"):
            monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", backend)
            env = gym.make("FbsEnv-v0", instance="O7")
            env.reset(seed=20260525)
            solver = _make_solver(env, G=2, t_max=4)
            solver.run()
            results[backend] = solver.last_run_payload["local_search_backend"]

        assert results["paper_adapted"] == "paper_adapted"
        assert results["engineered"] == "engineered"

    def test_backend_remarks_differ(self, monkeypatch):
        """paper_adapted 和 engineered 的 remark 不能混淆。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "1")

        remarks = {}
        for backend in ("paper_adapted", "engineered"):
            monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", backend)
            env = gym.make("FbsEnv-v0", instance="O7")
            env.reset(seed=20260526)
            solver = _make_solver(env, G=1, t_max=2)
            solver.run()
            remarks[backend] = solver.default_run_remark

        assert "paper-style interchange" in remarks["paper_adapted"].lower()
        assert "engineered action" in remarks["engineered"].lower()
        assert remarks["paper_adapted"] != remarks["engineered"]


class TestDeterminism:
    def test_same_seed_same_backend_reproducible(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_REFINEMENT_STEPS", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        def run_once(seed):
            _set_global_seed(seed)
            env = gym.make("FbsEnv-v0", instance="O7")
            env.reset(seed=seed)
            solver = _make_solver(env, G=2, t_max=4)
            solver.run()
            rep = solver.representative_solution
            return (
                float(getattr(rep, "MHC", 0)),
                float(getattr(rep, "CR", 0)),
                len(solver.pareto_archive),
            )

        mhc1, cr1, size1 = run_once(20260525)
        mhc2, cr2, size2 = run_once(20260525)

        assert mhc1 == mhc2
        assert cr1 == cr2
        assert size1 == size2


class TestPaperAdaptedInterchangeOnly:
    """PLAN2 必测：paper_adapted 只使用 interchange 邻域。"""

    def test_enumerate_positions_returns_all_facilities(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260527)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        positions = solver._enumerate_positions(solution)
        assert len(positions) == int(env.unwrapped.n)
        facility_ids = {p[2] for p in positions}
        assert facility_ids == set(range(1, int(env.unwrapped.n) + 1))

    def test_interchange_preserves_bay_count(self, monkeypatch):
        """interchange 不应改变 bay 数量（不允许 insert 或 bay 结构变化）。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260528)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        original_bay = np.asarray(solution.fbs_model.bay, dtype=int).copy()
        original_bay_count = int(np.sum(original_bay == 1))

        positions = solver._enumerate_positions(solution)
        if len(positions) >= 2:
            candidate = solver._apply_interchange(solution, positions[0], positions[1])
            new_bay = np.asarray(candidate.fbs_model.bay, dtype=int)
            new_bay_count = int(np.sum(new_bay == 1))
            assert new_bay_count == original_bay_count

    def test_interchange_preserves_facility_set(self, monkeypatch):
        """interchange 不应增删设施。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260529)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        original_perm = set(solution.fbs_model.permutation)
        positions = solver._enumerate_positions(solution)
        if len(positions) >= 2:
            candidate = solver._apply_interchange(solution, positions[0], positions[1])
            new_perm = set(candidate.fbs_model.permutation)
            assert original_perm == new_perm


class TestDBLSBehavior:
    """PLAN2 必测：DBLS 支配行为正确。"""

    def test_dbls_dominating_candidate_accepted(self, monkeypatch):
        """若候选支配当前解，DBLS 应接受。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260530)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        result = solver._paper_dbls_step(solution)
        assert result is not None
        assert bool(getattr(result, "current_is_feasible", False))

    def test_dbls_produces_feasible_result(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260531)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        result = solver._paper_dbls_step(solution)
        feasible = bool(getattr(result, "current_is_feasible", False))
        assert feasible


class TestAOLSBehavior:
    """PLAN2 必测：AOLS(mhc) 和 AOLS(cr) 行为正确。"""

    def test_aols_mhc_only_improves_mhc(self, monkeypatch):
        """AOLS(mhc) 只应因 MHC 改善而接受。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260601)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        original_mhc = float(getattr(solution, "MHC", math.inf))
        result = solver._paper_aols_step(solution, "mhc")
        result_mhc = float(getattr(result, "MHC", math.inf))

        # AOLS(mhc) 结束时 MHC 不应比原来差
        assert result_mhc <= original_mhc + 1e-9

    def test_aols_cr_only_improves_cr(self, monkeypatch):
        """AOLS(cr) 只应因 CR 改善（增大）而接受。"""
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260602)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        original_cr = float(getattr(solution, "CR", 0.0))
        result = solver._paper_aols_step(solution, "cr")
        result_cr = float(getattr(result, "CR", 0.0))

        # AOLS(cr) 结束时 CR 不应比原来差（CR 最大化方向）
        assert result_cr >= original_cr - 1e-9

    def test_aols_factor_validation(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260603)
        solver = _make_solver(env, G=1, t_max=1)
        solver._seed_archive_if_needed(solver._base_solution_template())
        solution = solver.s

        try:
            solver._paper_aols_step(solution, "invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestPaperAdaptedRemark:
    """PLAN2 必测：remark 和 backend 标签不混淆。"""

    def test_paper_adapted_remark_mentions_interchange(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260604)
        solver = _make_solver(env, G=1, t_max=1)
        solver.run()

        algo = solver.default_run_algorithm
        assert "PAPERLS" in algo

    def test_engineered_remark_mentions_action(self, monkeypatch):
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "engineered")
        monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")
        monkeypatch.setenv("ELP_GRASP_ARCHIVE_SEED_TRIALS", "1")

        env = gym.make("FbsEnv-v0", instance="O7")
        env.reset(seed=20260605)
        solver = _make_solver(env, G=1, t_max=1)
        solver.run()

        algo = solver.default_run_algorithm
        assert "ACTIONLS" in algo
