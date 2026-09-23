import copy
import json
import math
import os

import gym
import numpy as np
from loguru import logger

from src.algorithms.ELP_DRL_BiMO4 import (
    ELP as BiMO4ELP,
    _format_summary_metrics,
    _get_initial_solution_energy,
    _parse_env_flag,
    _parse_env_float,
    _parse_env_int,
    _parse_env_int_list,
    _preflight_required_files,
    _save_experiment_row,
    _set_global_seed,
)
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4


class ELPParetoH(BiMO4ELP):
    """使用 Pareto 最大退化和二维访问历史的 BiMO4 对照版本。"""

    ACCEPTANCE_VERSION = "pareto_max_deterioration_grid_history_v1"

    def __init__(self, *args, **kwargs):
        self.pareto_h_beta = float(_parse_env_float("ELP_BIMO_PARETO_H_BETA", 0.35))
        self.pareto_h_tau_start = float(_parse_env_float("ELP_BIMO_PARETO_H_TAU_START", 0.20))
        self.pareto_h_tau_end = float(_parse_env_float("ELP_BIMO_PARETO_H_TAU_END", 0.02))
        self.pareto_h_tau_power = float(_parse_env_float("ELP_BIMO_PARETO_H_TAU_POWER", 1.25))
        self.pareto_h_grid_bins = int(_parse_env_int("ELP_BIMO_PARETO_H_GRID_BINS", 20))
        self._validate_pareto_h_parameters()
        self._reset_pareto_h_state()
        super().__init__(*args, **kwargs)

    def _validate_pareto_h_parameters(self):
        if not np.isfinite(self.pareto_h_beta) or self.pareto_h_beta < 0.0:
            raise ValueError("ELP_BIMO_PARETO_H_BETA 必须是非负有限数。")
        if not np.isfinite(self.pareto_h_tau_start) or self.pareto_h_tau_start <= 0.0:
            raise ValueError("ELP_BIMO_PARETO_H_TAU_START 必须是正有限数。")
        if not np.isfinite(self.pareto_h_tau_end) or self.pareto_h_tau_end <= 0.0:
            raise ValueError("ELP_BIMO_PARETO_H_TAU_END 必须是正有限数。")
        if self.pareto_h_tau_end > self.pareto_h_tau_start:
            raise ValueError("ELP_BIMO_PARETO_H_TAU_END 不能大于 ELP_BIMO_PARETO_H_TAU_START。")
        if not np.isfinite(self.pareto_h_tau_power) or self.pareto_h_tau_power <= 0.0:
            raise ValueError("ELP_BIMO_PARETO_H_TAU_POWER 必须是正有限数。")
        if self.pareto_h_grid_bins < 2:
            raise ValueError("ELP_BIMO_PARETO_H_GRID_BINS 必须至少为 2。")

    def _reset_pareto_h_state(self):
        self.pareto_h_histogram = {}
        self.pareto_h_reference_ideal = None
        self.pareto_h_reference_span = None
        self.pareto_h_reference_source_size = 0
        self.pareto_h_nondominated_decisions = 0
        self.pareto_h_nondominated_accepts = 0

    def _reset_mo_logging_state(self):
        super()._reset_mo_logging_state()
        self._reset_pareto_h_state()

    def _reset_baseline_archive_state(self):
        super()._reset_baseline_archive_state()
        self._reset_pareto_h_state()

    @staticmethod
    def _objective_vector(solution):
        values = getattr(solution, "mo_objectives_min", None)
        if values is None:
            return None
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size < 2 or not np.all(np.isfinite(vector[:2])):
            return None
        return vector[:2].copy()

    def _try_initialize_pareto_h_reference(self, *solutions):
        if self.pareto_h_reference_ideal is not None:
            return True

        vectors = []
        for archived in getattr(self, "pareto_archive", []) or []:
            if not bool(getattr(archived, "current_is_feasible", False)):
                continue
            vector = self._objective_vector(archived)
            if vector is not None:
                vectors.append(vector)
        for solution in solutions:
            if solution is None or not bool(getattr(solution, "current_is_feasible", False)):
                continue
            vector = self._objective_vector(solution)
            if vector is not None:
                vectors.append(vector)

        if len(vectors) < 2:
            return False
        matrix = np.asarray(vectors, dtype=float)
        ideal = np.min(matrix, axis=0)
        nadir = np.max(matrix, axis=0)
        raw_span = nadir - ideal
        magnitude = np.maximum(np.maximum(np.abs(ideal), np.abs(nadir)), 1.0)
        # 任一目标尚无真实跨度时不冻结网格，避免 CR=0 的启动档案制造畸小尺度。
        if np.any(raw_span <= magnitude * 1e-9):
            return False
        span = raw_span
        if not np.all(np.isfinite(ideal)) or not np.all(np.isfinite(span)) or np.any(span <= 0.0):
            raise ValueError("无法从可行双目标解建立 Pareto-H 固定归一化参考框架。")

        self.pareto_h_reference_ideal = ideal
        self.pareto_h_reference_span = span
        self.pareto_h_reference_source_size = int(matrix.shape[0])
        return True

    def _pareto_h_grid_key(self, solution):
        vector = self._objective_vector(solution)
        if vector is None or self.pareto_h_reference_ideal is None:
            return None
        normalized = (vector - self.pareto_h_reference_ideal) / self.pareto_h_reference_span
        coordinates = np.floor(normalized * float(self.pareto_h_grid_bins)).astype(np.int64)
        return tuple(int(value) for value in coordinates.tolist())

    def _pareto_h_visit_count(self, solution):
        key = self._pareto_h_grid_key(solution)
        if key is None:
            return 0
        return int(self.pareto_h_histogram.get(key, 0) or 0)

    def _update_histogram(self, energy):
        # 保留原直方图供状态编码和旧遥测使用，但接受公式只读取二维目标网格。
        super()._update_histogram(energy)
        current = getattr(self, "s", None)
        if current is None or not bool(getattr(current, "current_is_feasible", False)):
            return
        self._try_initialize_pareto_h_reference(current)
        key = self._pareto_h_grid_key(current)
        if key is not None:
            self.pareto_h_histogram[key] = int(self.pareto_h_histogram.get(key, 0) or 0) + 1

    def _pareto_h_tau(self):
        progress = float(np.clip(getattr(self, "current_progress_ratio", 0.0) or 0.0, 0.0, 1.0))
        phase = progress ** self.pareto_h_tau_power
        ratio = self.pareto_h_tau_end / self.pareto_h_tau_start
        return float(self.pareto_h_tau_start * (ratio ** phase))

    def _pareto_h_objective_deterioration(self, current_solution, candidate_solution):
        current = self._objective_vector(current_solution)
        candidate = self._objective_vector(candidate_solution)
        if current is None or candidate is None:
            raise ValueError("可行解缺少有限的 [MHC, -CR] 双目标向量。")
        if self.pareto_h_reference_span is None:
            raise RuntimeError("Pareto-H 固定归一化参考框架尚未建立。")
        normalized_delta = (candidate - current) / self.pareto_h_reference_span
        return float(np.max(np.maximum(normalized_delta, 0.0)))

    def _pareto_h_warmup_deterioration(self, current_solution, candidate_solution):
        current = self._objective_vector(current_solution)
        candidate = self._objective_vector(candidate_solution)
        if current is None or candidate is None:
            raise ValueError("可行解缺少有限的 [MHC, -CR] 双目标向量。")
        pair_scale = np.maximum(np.maximum(np.abs(current), np.abs(candidate)), 1.0)
        normalized_delta = (candidate - current) / pair_scale
        return float(np.max(np.maximum(normalized_delta, 0.0)))

    def _pareto_h_probability(self, deterioration, current_visits, candidate_visits):
        tau = self._pareto_h_tau()
        current_history = math.log1p(max(int(current_visits), 0))
        candidate_history = math.log1p(max(int(candidate_visits), 0))
        exponent = (
            -max(float(deterioration), 0.0) / max(tau, 1e-12)
            + self.pareto_h_beta * (current_history - candidate_history)
        )
        exponent = float(np.clip(exponent, -700.0, 0.0))
        return float(math.exp(exponent)), exponent, tau, current_history, candidate_history

    def _archive_change_preview(self, candidate_solution):
        _preview, archive_would_change, _removed = MO_FBSUtil_BiMO4.update_pareto_archive(
            self.pareto_archive,
            candidate_solution,
            max_size=self.archive_limit,
            clone_fn=lambda item: item,
            quality_gate_when_full=self.archive_quality_gate_when_full,
            quality_hv_tol=self.archive_quality_hv_tol,
            quality_spacing_tol=self.archive_quality_spacing_tol,
            spacing_guard_when_full=bool(getattr(self, "archive_spacing_guard_when_full", False)),
            spacing_guard_rel_tol=float(getattr(self, "archive_spacing_guard_rel_tol", 0.0) or 0.0),
            spacing_guard_hv_gain_rel=float(getattr(self, "archive_spacing_guard_hv_gain_rel", 0.0) or 0.0),
            require_candidate_retained=bool(getattr(self, "archive_require_candidate_retained", False)),
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        return bool(archive_would_change)

    def _constraint_tie_probability(self, current_solution, candidate_solution):
        current_proxy = float(getattr(current_solution, "proxy_energy", current_solution.fitness))
        candidate_proxy = float(getattr(candidate_solution, "proxy_energy", candidate_solution.fitness))
        scale = max(abs(current_proxy), abs(candidate_proxy), 1.0)
        deterioration = max((candidate_proxy - current_proxy) / scale, 0.0)
        probability, exponent, tau, _, _ = self._pareto_h_probability(deterioration, 0, 0)
        return probability, deterioration, exponent, tau

    def _augment_bimo_transition_meta(self, current_solution, candidate_solution, meta):
        quality_features = self._candidate_archive_quality_features(candidate_solution)
        meta.update(
            {
                "current_mhc": float(getattr(current_solution, "MHC", math.inf)),
                "candidate_mhc": float(getattr(candidate_solution, "MHC", math.inf)),
                "current_cr": float(getattr(current_solution, "CR", 0.0)),
                "candidate_cr": float(getattr(candidate_solution, "CR", 0.0)),
                "mhc_relative_gain": self._relative_gain(
                    getattr(current_solution, "MHC", math.inf),
                    getattr(candidate_solution, "MHC", math.inf),
                    maximize=False,
                ),
                "cr_relative_gain": self._relative_gain(
                    getattr(current_solution, "CR", 0.0),
                    getattr(candidate_solution, "CR", 0.0),
                    maximize=True,
                ),
                "archive_sparse_distance": float(quality_features["sparse_distance"]),
                "archive_extreme_gain": float(quality_features["extreme_gain"]),
                "archive_hv_gain_proxy": float(quality_features["hv_gain_proxy"]),
            }
        )
        cr_info = getattr(candidate_solution, "bimo_cr_boundary_repartition_info", None)
        if isinstance(cr_info, dict):
            meta.update(
                {
                    "cr_boundary_repartition_used": True,
                    "cr_boundary_repartition_operation": str(cr_info.get("operation", "")),
                    "cr_boundary_repartition_relation_score": float(cr_info.get("relation_score", 0.0) or 0.0),
                    "cr_boundary_repartition_cr_gain": float(cr_info.get("cr_gain", 0.0) or 0.0),
                    "cr_boundary_repartition_mhc_loss": float(cr_info.get("mhc_loss", 0.0) or 0.0),
                    "cr_boundary_repartition_archive_would_change": bool(cr_info.get("archive_would_change", False)),
                    "cr_boundary_repartition_bay_pair_idx": int(cr_info.get("bay_pair_idx", -1)),
                }
            )
        return meta

    def _accept_candidate_with_context(self, current_solution, candidate_solution):
        comparison = MO_FBSUtil_BiMO4.compare_solution_quality(candidate_solution, current_solution)
        current_tilde = self._tilde_energy(current_solution.fitness)
        candidate_tilde = self._tilde_energy(candidate_solution.fitness)
        archive_would_change = self._archive_change_preview(candidate_solution)

        probability = 0.0
        deterioration = 0.0
        exponent = 0.0
        tau = self._pareto_h_tau()
        current_visits = 0
        candidate_visits = 0
        current_history = 0.0
        candidate_history = 0.0
        mode = "pareto_dominated_reject"

        if comparison < 0:
            probability = 1.0
            accept = True
            mode = "pareto_dominates_accept"
        elif comparison > 0:
            accept = False
        elif bool(getattr(current_solution, "current_is_feasible", False)) and bool(
            getattr(candidate_solution, "current_is_feasible", False)
        ):
            reference_ready = self._try_initialize_pareto_h_reference(current_solution, candidate_solution)
            if reference_ready:
                deterioration = self._pareto_h_objective_deterioration(current_solution, candidate_solution)
                current_visits = self._pareto_h_visit_count(current_solution)
                candidate_visits = self._pareto_h_visit_count(candidate_solution)
                mode = "pareto_nondominated_grid_history"
            else:
                # 预热阶段仍按双目标最大正退化接受，但在网格尺度可靠前不引入伪历史偏置。
                deterioration = self._pareto_h_warmup_deterioration(current_solution, candidate_solution)
                mode = "pareto_nondominated_reference_warmup"
            probability, exponent, tau, current_history, candidate_history = self._pareto_h_probability(
                deterioration,
                current_visits,
                candidate_visits,
            )
            accept = bool(np.random.rand() < probability)
            self.pareto_h_nondominated_decisions += 1
            if accept:
                self.pareto_h_nondominated_accepts += 1
        else:
            probability, deterioration, exponent, tau = self._constraint_tie_probability(
                current_solution,
                candidate_solution,
            )
            accept = bool(np.random.rand() < probability)
            mode = "constraint_tie_relative_proxy"

        meta = {
            "comparison": int(comparison),
            "archive_would_change": bool(archive_would_change),
            "raw_probability": float(probability),
            "probability_cap": 1.0,
            "late_nonarchive_gate_applied": False,
            "late_nonarchive_gate_scale": 1.0,
            "late_nonarchive_gate_cap": 1.0,
            "late_nonarchive_gate_phase": 0.0,
            "late_nonarchive_stagnation_windows": float(self._nonarchive_stagnation_windows()),
            "probability_after_gate": float(probability),
            "current_proxy": float(getattr(current_solution, "proxy_energy", current_solution.fitness)),
            "candidate_proxy": float(getattr(candidate_solution, "proxy_energy", candidate_solution.fitness)),
            "current_violation": float(getattr(current_solution, "constraint_violation", 0.0) or 0.0),
            "candidate_violation": float(getattr(candidate_solution, "constraint_violation", 0.0) or 0.0),
            "current_d_inf": int(getattr(current_solution, "current_d_inf", 0) or 0),
            "candidate_d_inf": int(getattr(candidate_solution, "current_d_inf", 0) or 0),
            "accepted": bool(accept),
            "acceptance_mode": mode,
            "pareto_h_objective_deterioration": float(deterioration),
            "pareto_h_tau": float(tau),
            "pareto_h_beta": float(self.pareto_h_beta),
            "pareto_h_exponent": float(exponent),
            "pareto_h_current_visits": int(current_visits),
            "pareto_h_candidate_visits": int(candidate_visits),
            "pareto_h_current_history": float(current_history),
            "pareto_h_candidate_history": float(candidate_history),
        }
        self._last_transition_meta = self._augment_bimo_transition_meta(
            current_solution,
            candidate_solution,
            meta,
        )
        return bool(accept), float(probability), float(current_tilde), float(candidate_tilde)

    def _pareto_h_summary_fields(self):
        decisions = int(self.pareto_h_nondominated_decisions)
        return {
            "acceptanceFormulaVersion": self.ACCEPTANCE_VERSION,
            "paretoHBeta": float(self.pareto_h_beta),
            "paretoHTauStart": float(self.pareto_h_tau_start),
            "paretoHTauEnd": float(self.pareto_h_tau_end),
            "paretoHTauPower": float(self.pareto_h_tau_power),
            "paretoHGridBins": int(self.pareto_h_grid_bins),
            "paretoHReferenceReady": self.pareto_h_reference_ideal is not None,
            "paretoHReferenceIdeal": (
                None if self.pareto_h_reference_ideal is None else self.pareto_h_reference_ideal.tolist()
            ),
            "paretoHReferenceSpan": (
                None if self.pareto_h_reference_span is None else self.pareto_h_reference_span.tolist()
            ),
            "paretoHReferenceSourceSize": int(self.pareto_h_reference_source_size),
            "paretoHVisitedCells": int(len(self.pareto_h_histogram)),
            "paretoHNondominatedDecisions": decisions,
            "paretoHNondominatedAccepts": int(self.pareto_h_nondominated_accepts),
            "paretoHNondominatedAcceptRate": (
                None if decisions <= 0 else float(self.pareto_h_nondominated_accepts) / float(decisions)
            ),
        }

    def _patch_run_summary_file(self, archive_update_count):
        super()._patch_run_summary_file(archive_update_count)
        if not isinstance(getattr(self, "mo_run_summary", None), dict):
            return
        run_summary_path = self._resolved_result_path(self.mo_run_summary.get("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.update(self._pareto_h_summary_fields())
            run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _finalize_archive_update_reporting(self):
        fields = self._pareto_h_summary_fields()
        if isinstance(getattr(self, "mo_run_summary", None), dict):
            self.mo_run_summary.update(fields)
        if isinstance(getattr(self, "last_run_payload", None), dict):
            self.last_run_payload.update(
                {
                    "acceptance_formula_version": fields["acceptanceFormulaVersion"],
                    "pareto_h_beta": fields["paretoHBeta"],
                    "pareto_h_tau_start": fields["paretoHTauStart"],
                    "pareto_h_tau_end": fields["paretoHTauEnd"],
                    "pareto_h_grid_bins": fields["paretoHGridBins"],
                    "pareto_h_visited_cells": fields["paretoHVisitedCells"],
                    "pareto_h_nondominated_accept_rate": fields["paretoHNondominatedAcceptRate"],
                }
            )
        super()._finalize_archive_update_reporting()


if __name__ == "__main__":
    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    _preflight_required_files(exp_instance)
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_BiMO4_PARETO_H")
    exp_remark = os.getenv(
        "ELP_EXP_REMARK",
        "BiMO4 with Pareto max-deterioration and 2D objective-history acceptance",
    )
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 30)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)

    G = _parse_env_int("ELP_G", 1000)
    t_max = _parse_env_int("ELP_T_MAX", 300)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260427)
    fixed_seeds = _parse_env_int_list("ELP_FIXED_SEEDS")
    if fixed_seeds:
        exp_number = len(fixed_seeds)

    def _run_once(run_index):
        run_seed = int(fixed_seeds[run_index]) if fixed_seeds else int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info("Experiment seed: {} | strict_determinism: {}", run_seed, strict_determinism)
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        except Exception:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info("Initial solution energy: {}", _get_initial_solution_energy(base_env))
        solver = ELPParetoH(
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
            logger.info("Starting experiment {} for {}", i + 1, exp_algorithm)
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(
                    "Experiment {} complete | {}",
                    i + 1,
                    _format_summary_metrics(elp_solver, best_energy),
                )
                for telemetry_line in elp_solver.format_action_telemetry():
                    logger.info("Telemetry | {}", telemetry_line)
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
                logger.exception("Experiment {} failed: {}", i + 1, exc)
                raise
    else:
        elp_solver, result_tuple = _run_once(0)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
        print(f"Single run complete | {_format_summary_metrics(elp_solver, best_energy)}")
        for telemetry_line in elp_solver.format_action_telemetry():
            print(telemetry_line)
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
