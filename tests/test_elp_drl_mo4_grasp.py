import copy

import gym
import numpy as np

import src
from src.algorithms.ELP_DRL_MO4_GRASP import ELP


def test_grasp_solver_runs_on_mo4_env(monkeypatch):
    monkeypatch.setenv("ELP_GRASP_MAX_FACILITY_CANDIDATES", "4")
    monkeypatch.setenv("ELP_GRASP_REFINE_STEPS", "2")
    monkeypatch.setenv("ELP_GRASP_LOCAL_SEARCH_PASSES", "1")

    env = gym.make("FbsEnv-v0", instance="O7")
    env.reset(seed=20260520)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    solver = ELP(
        env=base_env,
        gbest=copy.deepcopy(base_env),
        T=1000.0,
        G=2,
        t_max=4,
        k=10.0,
    )

    total_iter, is_valid, best_sol, best_energy, start, end, fast = solver.run()

    assert total_iter == 2
    assert is_valid is True
    assert best_sol is not None
    assert np.isfinite(best_energy)
    assert len(solver.pareto_archive) > 0
    assert solver.last_run_payload["pareto_archive_path"] is not None
    assert solver.last_run_payload["rep_mhc"] is not None
    assert solver.last_run_payload["rep_cr"] is not None
    assert start <= fast <= end
