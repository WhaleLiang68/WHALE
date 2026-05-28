import copy
import datetime
import math
import os

import gym
import numpy as np
from loguru import logger

import src
import src.utils.FBSUtil as FBSUtil
from src.algorithms.ELP_DRL_MO4 import (
    ELP as _MO4ELP,
    _get_initial_solution_energy,
    _save_experiment_row,
    _set_global_seed,
)
from src.utils.FBSModel import FBSModel
from src.utils.MO_ReferenceFrontUtil import OBJECTIVE_DEFINITION_VERSION


class ELP(_MO4ELP):
    """真正的 GRASP 基线：MHC/CR 构造，MO4 局部搜索与评测。"""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        super().__init__(
            env=env,
            gbest=gbest,
            T=T,
            G=G,
            t_max=t_max,
            k=k,
            archive_limit=archive_limit,
            objective_weights=objective_weights,
        )
        self.default_run_algorithm = "ELP_DRL_MO4_GRASP"
        self.default_run_remark = "GRASP constructive baseline with MO4 archive evaluation"
        self.grasp_alpha = self._parse_env_float("ELP_GRASP_ALPHA", 0.45)
        self.grasp_constructive_mix = self._normalize_constructive_mix(
            [
                self._parse_env_float("ELP_GRASP_PC1", 0.30),
                self._parse_env_float("ELP_GRASP_PC2", 0.20),
                self._parse_env_float("ELP_GRASP_PC3", 0.30),
                self._parse_env_float("ELP_GRASP_PC4", 0.20),
            ]
        )
        self.grasp_max_facility_candidates = int(
            max(2, self._parse_env_int("ELP_GRASP_MAX_FACILITY_CANDIDATES", 12))
        )
        self.grasp_refine_steps = int(max(1, self._parse_env_int("ELP_GRASP_REFINE_STEPS", self.t_max)))
        self.grasp_local_search_passes = int(max(1, self._parse_env_int("ELP_GRASP_LOCAL_SEARCH_PASSES", 2)))
        self.grasp_archive_seed_trials = int(max(1, self._parse_env_int("ELP_GRASP_ARCHIVE_SEED_TRIALS", 4)))

    @staticmethod
    def _parse_env_int(name, default):
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw.strip())
        except Exception:
            return int(default)

    @staticmethod
    def _parse_env_float(name, default):
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw.strip())
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_constructive_mix(values):
        mix = np.asarray(values, dtype=float).reshape(-1)
        if mix.size != 4:
            raise ValueError("GRASP 构造器配比必须正好包含 4 项。")
        mix = np.clip(mix, 0.0, None)
        if float(np.sum(mix)) <= 0.0:
            mix = np.asarray([0.30, 0.20, 0.30, 0.20], dtype=float)
        return (mix / np.sum(mix)).tolist()

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    def _base_solution_template(self):
        return copy.deepcopy(self.env)

    def _determine_target_bays(self, template_solution):
        bay_array = np.asarray(getattr(template_solution.fbs_model, "bay", []), dtype=int).reshape(-1)
        current_bays = int(np.sum(bay_array == 1)) if bay_array.size > 0 else 0
        estimated_bays = FBSUtil.select_B(
            np.asarray(template_solution.areas, dtype=float),
            int(template_solution.n),
            float(template_solution.fac_limit_aspect),
            float(template_solution.W),
        )
        if estimated_bays is None:
            estimated_bays = current_bays
        if estimated_bays is None or int(estimated_bays) <= 0:
            estimated_bays = 2
        return int(max(1, min(int(template_solution.n), int(estimated_bays))))

    def _structure_to_encoding(self, structure):
        normalized = [list(map(int, bay)) for bay in structure if len(bay) > 0]
        if not normalized:
            raise ValueError("区带结构不能为空。")
        permutation, bay = FBSUtil.arrayToPermutation(normalized)
        bay = np.asarray(bay, dtype=int).reshape(-1)
        bay[-1] = 1
        return np.asarray(permutation, dtype=int).reshape(-1), bay

    def _solution_from_structure(self, template_solution, structure):
        permutation, bay = self._structure_to_encoding(structure)
        candidate = self._light_clone_solution(template_solution)
        candidate.fbs_model = FBSModel(
            permutation=np.asarray(permutation, dtype=int).tolist(),
            bay=np.asarray(bay, dtype=int).tolist(),
        )
        self._evaluate_solution(candidate)
        return candidate

    def _facility_proxy_scores(self, remaining, placed, objective_name):
        remaining_idx = np.asarray(list(remaining), dtype=int) - 1
        if remaining_idx.size == 0:
            return np.asarray([], dtype=float)
        if not placed:
            return np.random.random(size=remaining_idx.size)

        placed_idx = np.asarray(list(placed), dtype=int) - 1
        if objective_name == "mhc":
            matrix = np.asarray(getattr(self.env, "F", []), dtype=float)
            proxy = np.sum(np.abs(matrix[np.ix_(remaining_idx, placed_idx)]), axis=1)
            return proxy.astype(float)

        rel_matrix = np.asarray(getattr(self, "rel_matrix", []), dtype=float)
        dist_req_matrix = np.asarray(getattr(self, "dist_req_matrix", []), dtype=float)
        if rel_matrix.size == 0 or dist_req_matrix.size == 0:
            return np.random.random(size=remaining_idx.size)
        relational_strength = np.sum(np.abs(rel_matrix[np.ix_(remaining_idx, placed_idx)]), axis=1)
        distance_pressure = np.sum(np.abs(dist_req_matrix[np.ix_(remaining_idx, placed_idx)]), axis=1)
        return (0.7 * relational_strength + 0.3 * distance_pressure).astype(float)

    def _select_facility_subset(self, remaining, placed, objective_name):
        remaining = list(map(int, remaining))
        if len(remaining) <= self.grasp_max_facility_candidates:
            return remaining

        proxy_scores = self._facility_proxy_scores(remaining, placed, objective_name)
        ranked_indices = np.argsort(-proxy_scores)
        top_k = ranked_indices[: self.grasp_max_facility_candidates]
        subset = [remaining[int(idx)] for idx in top_k.tolist()]
        np.random.shuffle(subset)
        return subset

    def _partial_structure_coordinates(self, structure):
        coordinates = {}
        for row_idx, bay in enumerate(structure):
            for col_idx, facility in enumerate(bay):
                coordinates[int(facility)] = (int(row_idx), int(col_idx))
        return coordinates

    def _partial_primary_score(self, structure, objective_name):
        coordinates = self._partial_structure_coordinates(structure)
        facilities = sorted(coordinates.keys())
        if len(facilities) <= 1:
            return 0.0

        if objective_name == "mhc":
            matrix = np.asarray(getattr(self.env, "F", []), dtype=float)
            total_score = 0.0
            for left_idx, left_facility in enumerate(facilities[:-1]):
                left_row, left_col = coordinates[left_facility]
                for right_facility in facilities[left_idx + 1 :]:
                    right_row, right_col = coordinates[right_facility]
                    manhattan = abs(left_row - right_row) + abs(left_col - right_col)
                    total_score += float(matrix[left_facility - 1, right_facility - 1]) * float(manhattan)
            return float(total_score)

        rel_matrix = np.asarray(getattr(self, "rel_matrix", []), dtype=float)
        if rel_matrix.size == 0:
            return 0.0

        total_score = 0.0
        for left_idx, left_facility in enumerate(facilities[:-1]):
            left_row, left_col = coordinates[left_facility]
            for right_facility in facilities[left_idx + 1 :]:
                right_row, right_col = coordinates[right_facility]
                manhattan = abs(left_row - right_row) + abs(left_col - right_col)
                adjacency = 1.0 if manhattan == 1 else 0.0
                total_score += float(rel_matrix[left_facility - 1, right_facility - 1]) * adjacency
        return float(total_score)

    def _candidate_order_key(self, primary_value, objective_name):
        if objective_name == "mhc":
            return float(primary_value)
        return -float(primary_value)

    def _enumerate_constructive_candidates(self, template_solution, structure, remaining, objective_name):
        _ = template_solution
        placed = [facility for bay in structure for facility in bay]
        sampled_facilities = self._select_facility_subset(remaining, placed, objective_name)
        rows = len(structure)
        candidates = []

        for facility in sampled_facilities:
            for row_idx in range(rows):
                row_length = len(structure[row_idx])
                for position_idx in range(row_length + 1):
                    candidate_structure = [list(current_bay) for current_bay in structure]
                    candidate_structure[row_idx].insert(position_idx, int(facility))
                    primary_value = self._partial_primary_score(candidate_structure, objective_name)
                    candidates.append(
                        {
                            "facility": int(facility),
                            "row_idx": int(row_idx),
                            "position_idx": int(position_idx),
                            "primary": float(primary_value),
                            "order_key": self._candidate_order_key(primary_value, objective_name),
                        }
                    )
        candidates.sort(key=lambda item: item["order_key"])
        return candidates

    def _build_rcl(self, candidates, objective_name, variant_name):
        if not candidates:
            return []

        active_pool = sorted(candidates, key=lambda item: item["order_key"])

        if variant_name in {"C1_GR", "C3_GR"}:
            if objective_name == "mhc":
                best_value = float(active_pool[0]["primary"])
                worst_value = float(active_pool[-1]["primary"])
                threshold = best_value * self.grasp_alpha + worst_value * (1.0 - self.grasp_alpha)
                rcl = [item for item in active_pool if float(item["primary"]) <= threshold]
            else:
                best_value = float(active_pool[0]["primary"])
                worst_value = float(active_pool[-1]["primary"])
                threshold = best_value * self.grasp_alpha + worst_value * (1.0 - self.grasp_alpha)
                rcl = [item for item in active_pool if float(item["primary"]) >= threshold]
            return rcl if rcl else active_pool[:1]

        subset_size = int(max(1, math.ceil(len(active_pool) * float(self.grasp_alpha))))
        chosen_indices = np.random.choice(len(active_pool), size=subset_size, replace=False)
        sampled = [active_pool[int(idx)] for idx in np.asarray(chosen_indices, dtype=int).tolist()]
        sampled.sort(key=lambda item: item["order_key"])
        return sampled

    def _select_constructive_choice(self, candidates, objective_name, variant_name):
        rcl = self._build_rcl(candidates, objective_name, variant_name)
        if not rcl:
            raise RuntimeError("GRASP 构造阶段生成了空 RCL。")
        if variant_name in {"C1_GR", "C3_GR"}:
            chosen = rcl[int(np.random.randint(0, len(rcl)))]
        else:
            chosen = min(rcl, key=lambda item: item["order_key"])
        return chosen

    def _construct_solution(self, template_solution, variant_name):
        target_bays = self._determine_target_bays(template_solution)
        facilities = list(range(1, int(template_solution.n) + 1))
        np.random.shuffle(facilities)
        structure = [[] for _ in range(target_bays)]

        # 先在每个区带中放一个随机设施，保留论文“随机初始布局”的核心思想。
        for bay_idx in range(target_bays):
            structure[bay_idx].append(int(facilities.pop()))

        objective_name = "mhc" if variant_name in {"C1_GR", "C2_RG"} else "cr"
        while facilities:
            candidates = self._enumerate_constructive_candidates(
                template_solution=template_solution,
                structure=structure,
                remaining=facilities,
                objective_name=objective_name,
            )
            chosen = self._select_constructive_choice(candidates, objective_name, variant_name)
            structure[int(chosen["row_idx"])].insert(int(chosen["position_idx"]), int(chosen["facility"]))
            facilities.remove(int(chosen["facility"]))

        return self._solution_from_structure(template_solution, structure)

    def _sample_recipe_refinement(self, solution):
        current = self._light_clone_solution(solution)
        current_score = float(getattr(current, "fitness", math.inf))

        for _ in range(self.grasp_refine_steps):
            best_candidate = None
            best_score = current_score
            for action_idx in self.valid_actions:
                candidate = self.generate_candidate_by_action(current, action_idx)
                candidate_score = float(getattr(candidate, "fitness", math.inf))
                if not bool(getattr(candidate, "current_is_feasible", False)):
                    continue
                if candidate_score + 1e-12 < best_score:
                    best_candidate = candidate
                    best_score = candidate_score
            if best_candidate is None:
                break
            current = best_candidate
            current_score = best_score
            self._observe_feasible_state(current)

        return current

    def _refine_solution(self, solution):
        current = self._light_clone_solution(solution)
        for _ in range(self.grasp_local_search_passes):
            current = self._greedy_local_search(current)
            current = self._sample_recipe_refinement(current)
        return current

    def _sample_constructive_variant(self):
        variants = np.asarray(["C1_GR", "C2_RG", "C3_GR", "C4_RG"], dtype=object)
        chosen_index = int(np.random.choice(np.arange(variants.size), p=np.asarray(self.grasp_constructive_mix, dtype=float)))
        return str(variants[chosen_index])

    def _seed_archive_if_needed(self, template_solution):
        if self.pareto_archive:
            return
        for _ in range(self.grasp_archive_seed_trials):
            variant_name = self._sample_constructive_variant()
            candidate = self._construct_solution(template_solution, variant_name)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)
                self.s = self._light_clone_solution(candidate)
                break

    def _finalize_payload(self, start_time, end_time, fast_time):
        self._refresh_archive_state()
        archive_path = self._save_pareto_archive(start_time, algorithm_name=self.default_run_algorithm)
        reference_metrics = self._compute_reference_front_metrics()

        best_solution = self.representative_solution
        best_energy = float(self.representative_decision_score) if self.representative_solution is not None else math.inf
        if best_solution is None and self.best_feasible_solution is not None:
            best_solution = copy.deepcopy(self.best_feasible_solution)
            best_energy = float(getattr(best_solution, "decision_score", getattr(best_solution, "fitness", math.inf)))
        if best_solution is None:
            best_solution = copy.deepcopy(self.s)
            best_energy = float(getattr(best_solution, "decision_score", getattr(best_solution, "fitness", math.inf)))

        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else self._safe_float(getattr(best_solution, "MHC", None)),
            "rep_cr": None if best_solution is None else self._safe_float(getattr(best_solution, "CR", None)),
            "rep_dr": None if best_solution is None else self._safe_float(getattr(best_solution, "DR", None)),
            "rep_ar": None if best_solution is None else self._safe_float(getattr(best_solution, "AR", None)),
            "decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "stable_decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "archive_hypervolume": reference_metrics.get("archive_hypervolume"),
            "archive_spacing": reference_metrics.get("archive_spacing"),
            "archive_igd": reference_metrics.get("archive_igd"),
            "reference_front_path": reference_metrics.get("reference_front_path"),
            "reference_front_size": reference_metrics.get("reference_front_size"),
            "reference_front_archive_count": reference_metrics.get("reference_front_archive_count"),
            "archive_hypervolume_mode": reference_metrics.get("archive_hypervolume_mode"),
            "archive_hypervolume_reference_point": reference_metrics.get("archive_hypervolume_reference_point"),
            "objective_definition_version": OBJECTIVE_DEFINITION_VERSION,
            "grasp_alpha": float(self.grasp_alpha),
            "grasp_constructive_mix": list(self.grasp_constructive_mix),
            "grasp_refine_steps": int(self.grasp_refine_steps),
            "runtime_seconds": float((end_time - start_time).total_seconds()),
            "best_result_seconds": None if fast_time is None else float((fast_time - start_time).total_seconds()),
        }
        return best_solution, best_energy

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        best_seen_score = math.inf
        self._reset_baseline_archive_state()

        template_solution = self._base_solution_template()
        self._evaluate_solution(template_solution)
        self.s = self._light_clone_solution(template_solution)
        self._seed_archive_if_needed(template_solution)

        for iteration_idx in range(int(max(1, self.G))):
            variant_name = self._sample_constructive_variant()
            candidate = self._construct_solution(template_solution, variant_name)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)
            candidate = self._refine_solution(candidate)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)
            self.s = self._light_clone_solution(candidate)

            current_rep_score = float(getattr(self, "representative_decision_score", math.inf))
            if np.isfinite(current_rep_score) and current_rep_score + 1e-12 < best_seen_score:
                best_seen_score = current_rep_score
                fast_time = datetime.datetime.now()

            logger.info(
                "GRASP iteration {} / {} | variant={} | archive_size={} | rep_score={}",
                iteration_idx + 1,
                int(self.G),
                variant_name,
                len(self.pareto_archive),
                "NA" if not np.isfinite(current_rep_score) else f"{current_rep_score:.6f}",
            )

        end_time = datetime.datetime.now()
        best_solution, best_energy = self._finalize_payload(start_time, end_time, fast_time)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        total_iter = int(max(1, self.G))
        return total_iter, is_valid, best_solution, best_energy, start_time, end_time, fast_time


def _format_summary_metrics(solver, best_energy):
    payload = getattr(solver, "last_run_payload", {}) or {}
    hv = payload.get("archive_hypervolume")
    igd = payload.get("archive_igd")
    spacing = payload.get("archive_spacing")
    rep_mhc = payload.get("rep_mhc")
    rep_cr = payload.get("rep_cr")

    def _fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    return (
        f"representative decision score: {float(best_energy):.6f} | "
        f"rep_mhc: {_fmt(rep_mhc)} | rep_cr: {_fmt(rep_cr)} | "
        f"HV: {_fmt(hv)} | IGD: {_fmt(igd)} | Spacing: {_fmt(spacing)}"
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

    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_MO4_GRASP")
    exp_remark = os.getenv(
        "ELP_EXP_REMARK",
        "True GRASP constructive baseline under MO4 evaluation pipeline",
    )
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 1)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)
    G = _parse_env_int("ELP_G", 80)
    t_max = _parse_env_int("ELP_T_MAX", 60)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260520)

    def _run_once(run_index):
        run_seed = int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Experiment seed: {run_seed} | strict_determinism: {strict_determinism}")
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        except Exception:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
        solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        return solver, solver.run()

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(f"Experiment {i + 1} complete | {_format_summary_metrics(elp_solver, best_energy)}")
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
            except Exception as exc:
                logger.exception(f"Experiment {i + 1} failed: {exc}")
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
