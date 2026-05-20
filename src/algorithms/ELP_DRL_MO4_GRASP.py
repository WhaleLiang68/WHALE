import copy
import os

import gym
import numpy as np
from loguru import logger

import src
from src.algorithms.ELP_DRL_MO4_Paper import (
    ELP as _PaperELP,
    _get_initial_solution_energy,
    _save_experiment_row,
    _set_global_seed,
)
from src.utils.BO_MREFLPBenchmark import BO_MREFLPBenchmark
from src.utils.FBSModel import FBSModel
from src.utils.MO_FBSUtil_GRASP import MO_FBSUtil


class ELP(_PaperELP):
    """沿用 MO4 搜索骨架，但目标与环境切换到论文 BO-MREFLP 口径。"""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.true_cr_matrix = np.asarray(getattr(base_env, "CR_matrix"), dtype=float)
        self.default_run_algorithm = "ELP_DRL_MO4_GRASP"
        self.default_run_remark = "MO4 search skeleton under BO-MREFLP true MHC+CR objectives"
        super().__init__(
            env=env,
            gbest=gbest,
            T=T,
            G=G,
            t_max=t_max,
            k=k,
            archive_limit=archive_limit,
            objective_weights=objective_weights if objective_weights is not None else [0.5, 0.5, 0.0, 0.0],
        )
        self.paper_preference_matrix = None
        self.paper_preference_payload = {"source": "true_cr_matrix"}
        self.paper_area_utilization = 1.0
        self.paper_instance_profile = copy.deepcopy(getattr(base_env, "paper_instance_profile", {}))
        self.archive_reference_front_enabled = False

    def _run_impl(self):
        previous_algorithm = os.environ.get("ELP_EXP_ALGORITHM")
        previous_remark = os.environ.get("ELP_EXP_REMARK")
        try:
            os.environ.setdefault("ELP_EXP_ALGORITHM", self.default_run_algorithm)
            os.environ.setdefault("ELP_EXP_REMARK", self.default_run_remark)
            return super()._run_impl()
        finally:
            if previous_algorithm is None:
                os.environ.pop("ELP_EXP_ALGORITHM", None)
            else:
                os.environ["ELP_EXP_ALGORITHM"] = previous_algorithm
            if previous_remark is None:
                os.environ.pop("ELP_EXP_REMARK", None)
            else:
                os.environ["ELP_EXP_REMARK"] = previous_remark

    def _archive_objective_points(self):
        rows = []
        for candidate in list(getattr(self, "pareto_archive", []) or []):
            objectives_raw = np.asarray(getattr(candidate, "mo_objectives_raw", []), dtype=float).reshape(-1)
            if objectives_raw.size >= 2 and np.all(np.isfinite(objectives_raw[:2])):
                rows.append([float(objectives_raw[0]), float(objectives_raw[1])])
        return np.asarray(rows, dtype=float)

    def _compute_reference_front_metrics(self):
        candidate_points = self._archive_objective_points()
        if candidate_points.size == 0:
            return {
                "archive_hypervolume": 0.0,
                "archive_spacing": 0.0,
                "archive_igd": 0.0,
                "reference_front_path": None,
                "reference_front_size": 0,
                "reference_front_archive_count": None,
                "archive_hypervolume_mode": "bo_mreflp_benchmark_fixed",
                "archive_hypervolume_reference_point": [1.1, 1.1],
                "paper_solution_count": 0,
                "paper_pareto_count": 0,
                "paper_solution_set_mode": "benchmark_nd_archive",
                "paper_solution_set_detail": "fixed_reference_front",
                "paper_pr": None,
                "paper_sp": None,
                "paper_sp_raw": None,
                "paper_sp_norm": None,
                "paper_ops": None,
                "paper_ops_components": [None, None],
                "paper_best_mhc": None,
                "paper_mean_mhc": None,
                "paper_best_f3": None,
                "paper_mean_f3": None,
                "paper_final_window_ratio": None,
                "paper_final_window_start_progress": None,
                "paper_final_window_empty": True,
                "paper_history_solution_count": 0,
                "paper_history_pareto_count": 0,
                "benchmark_gd": 0.0,
                "benchmark_igd_plus": 0.0,
                "benchmark_coverage_ref_to_s": 0.0,
                "benchmark_epsilon_multiplicative": 0.0,
                "benchmark_normalization_path": None,
            }

        metrics = BO_MREFLPBenchmark.evaluate_points(self.instance_name, candidate_points)
        return {
            "archive_hypervolume": metrics["hv"],
            "archive_spacing": metrics["spread_delta"],
            "archive_igd": metrics["igd"],
            "reference_front_path": metrics["reference_front_path"],
            "reference_front_size": metrics["reference_front_size"],
            "reference_front_archive_count": None,
            "archive_hypervolume_mode": "bo_mreflp_benchmark_fixed",
            "archive_hypervolume_reference_point": metrics["hv_ref_point"],
            "paper_solution_count": metrics["candidate_point_count"],
            "paper_pareto_count": metrics["candidate_nd_size"],
            "paper_solution_set_mode": "benchmark_nd_archive",
            "paper_solution_set_detail": "fixed_reference_front",
            "paper_pr": None,
            "paper_sp": None,
            "paper_sp_raw": None,
            "paper_sp_norm": None,
            "paper_ops": None,
            "paper_ops_components": [None, None],
            "paper_best_mhc": float(np.min(candidate_points[:, 0])),
            "paper_mean_mhc": float(np.mean(candidate_points[:, 0])),
            "paper_best_f3": float(np.min(candidate_points[:, 1])),
            "paper_mean_f3": float(np.mean(candidate_points[:, 1])),
            "paper_final_window_ratio": None,
            "paper_final_window_start_progress": None,
            "paper_final_window_empty": False,
            "paper_history_solution_count": metrics["candidate_point_count"],
            "paper_history_pareto_count": metrics["candidate_nd_size"],
            "benchmark_gd": metrics["gd"],
            "benchmark_igd_plus": metrics["igd_plus"],
            "benchmark_coverage_ref_to_s": metrics["coverage_ref_to_s"],
            "benchmark_epsilon_multiplicative": metrics["epsilon_multiplicative"],
            "benchmark_normalization_path": metrics["normalization_path"],
        }

    def _score_candidate_encoding(self, permutation, bay, solution):
        candidate_model = FBSModel(
            np.asarray(permutation, dtype=int).tolist(),
            np.asarray(bay, dtype=int).tolist(),
        )
        metrics = solution.evaluate_fbs_model(candidate_model)
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            cr_matrix=self.true_cr_matrix,
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        search_energy = MO_FBSUtil.search_energy(
            objectives_min,
            is_feasible=True,
            d_inf=0,
            total_violation=0.0,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        return (
            float(search_energy),
            0,
            0.0,
            float(metrics["mhc"]),
            np.asarray(metrics["permutation"], dtype=int),
            np.asarray(bay, dtype=int),
        )

    def _sync_solution_metrics(self, solution, metrics):
        solution.fac_x = metrics["fac_x"]
        solution.fac_y = metrics["fac_y"]
        solution.fac_b = metrics["fac_b"]
        solution.fac_h = metrics["fac_h"]
        solution.fac_aspect_ratio = metrics["fac_aspect_ratio"]
        solution.lower_bounds = metrics["lower_bounds"]
        solution.upper_bounds = metrics["upper_bounds"]
        solution.aspect_limits = metrics["aspect_limits"]
        solution.infeasible_mask = metrics["infeasible_mask"]
        solution.D = metrics["D"]
        solution.TM = metrics["TM"]
        solution.MHC = float(metrics["mhc"])
        solution.F3 = float(metrics["cr"])
        solution.CR = float(metrics["cr"])
        solution.DR = float(metrics["dr"])
        solution.AR = float(metrics["ar"])
        solution.raw_cost = float(metrics["cost"])
        solution.mo_objectives_raw = np.asarray(metrics["mo_objectives_raw"], dtype=float)
        solution.mo_objectives_min = np.asarray(metrics["mo_objectives_min"], dtype=float)
        solution.constraint_violation = 0.0
        solution.current_d_inf = 0
        solution.current_is_feasible = True
        solution.feasible_solution_count = getattr(self, "feasible_solution_count", 0)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", np.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.best_fitness = getattr(self, "best_feasible_cost", np.inf)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        self._refresh_solution_search_metrics(solution)
        solution.state = solution.constructState()

    def _evaluate_solution(self, solution):
        metrics = solution.evaluate_fbs_model(solution.fbs_model)
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            cr_matrix=self.true_cr_matrix,
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        metrics.update(
            {
                "f3": float(objectives_raw[1]),
                "cr": float(objectives_raw[1]),
                "dr": float(objectives_raw[2]),
                "ar": float(objectives_raw[3]),
                "mo_objectives_raw": np.asarray(objectives_raw, dtype=float),
                "mo_objectives_min": objectives_min,
                "constraint_violation": 0.0,
            }
        )
        self._sync_solution_metrics(solution, metrics)
        return metrics


def _format_summary_metrics(solver, best_energy):
    payload = getattr(solver, "last_run_payload", {}) or {}
    rep_mhc = payload.get("rep_mhc")
    rep_cr = payload.get("rep_cr")
    pareto_size = payload.get("pareto_size")
    hv = payload.get("archive_hypervolume")
    igd = payload.get("archive_igd")
    gd = payload.get("benchmark_gd")
    igd_plus = payload.get("benchmark_igd_plus")
    spread = payload.get("archive_spacing")
    coverage = payload.get("benchmark_coverage_ref_to_s")
    epsilon = payload.get("benchmark_epsilon_multiplicative")

    def _fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    return (
        f"representative decision score: {float(best_energy):.6f} | "
        f"rep_mhc: {_fmt(rep_mhc)} | rep_cr: {_fmt(rep_cr)} | "
        f"pareto_size: {pareto_size if pareto_size is not None else 'NA'} | "
        f"HV: {_fmt(hv)} | GD: {_fmt(gd)} | IGD: {_fmt(igd)} | "
        f"IGD+: {_fmt(igd_plus)} | Delta: {_fmt(spread)} | "
        f"C(Ref,S): {_fmt(coverage)} | EPS*: {_fmt(epsilon)}"
    )


if __name__ == "__main__":
    def _parse_env_int(name, default):
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw.strip())
        except Exception:
            return int(default)

    def _parse_env_float(name, default):
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw.strip())
        except Exception:
            return float(default)

    def _parse_env_flag(name, default):
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

    exp_instance = os.getenv("ELP_EXP_INSTANCE", "previous8")
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2", "pso", "mopso"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}_GRASP" if baseline_enabled else "ELP_DRL_MO4_GRASP"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    exp_remark = os.getenv(
        "ELP_EXP_REMARK",
        "MO4 search skeleton under BO-MREFLP true MHC+CR objectives"
        if not baseline_enabled
        else "MO baseline on BO-MREFLP true MHC+CR objectives",
    )
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 1)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)
    G = _parse_env_int("ELP_G", 500)
    t_max = _parse_env_int("ELP_T_MAX", 200)
    T_initial = _parse_env_float("ELP_T_INITIAL", 1000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260519)
    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
        run_seed = int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Experiment seed: {run_seed} | strict_determinism: {strict_determinism}")
        env = gym.make("FbsPaperEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info(
            f"GRASP paper instance | instance={base_env.instance} | rows={base_env.m} | cols={base_env.c} | n={base_env.n}"
        )
        logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
        solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        if baseline_enabled:
            return solver, solver.run_moea_baseline(
                algorithm_name=baseline_algo,
                population_size=baseline_population,
                generations=baseline_generations,
                sequence_length=baseline_sequence_length,
                seed=run_seed,
            )
        return solver, solver.run()

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            elp_solver, result_tuple = _run_once(i)
            total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
            logger.info(f"Experiment {i + 1} complete | {_format_summary_metrics(elp_solver, best_energy)}")
            if not baseline_enabled:
                for telemetry_line in elp_solver.format_action_telemetry():
                    logger.info(f"Telemetry | {telemetry_line}")
            _save_experiment_row(
                exp_instance,
                exp_algorithm,
                exp_remark,
                total_iter,
                is_valid,
                best_sol,
                best_energy,
                start,
                end,
                fast,
                elp_solver,
            )
    else:
        elp_solver, result_tuple = _run_once(0)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
        print(f"Single run complete | {_format_summary_metrics(elp_solver, best_energy)}")
        _save_experiment_row(
            exp_instance,
            exp_algorithm,
            exp_remark,
            total_iter,
            is_valid,
            best_sol,
            best_energy,
            start,
            end,
            fast,
            elp_solver,
        )
