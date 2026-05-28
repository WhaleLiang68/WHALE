import copy
import datetime
import json
import math
import os
import time
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from loguru import logger

import src.algorithms.ELP_DRL_MO4 as mo4_module
import src.utils.FBSUtil as FBSUtil
import src.utils.config as config
from src.algorithms.ELP_DRL_MO4 import ELP as MO4ELP
from src.algorithms.ELP_DRL_MO4 import MOEAD
from src.algorithms.ELP_DRL_MO4 import NSGA2
from src.algorithms.ELP_DRL_MO4 import SPEA2
from src.algorithms.ELP_DRL_MO4 import SPEA2Survival
from src.algorithms.ELP_DRL_MO4 import Problem
from src.algorithms.ELP_DRL_MO4 import StandardDQNAgent
from src.algorithms.ELP_DRL_MO4 import StandardQLearningAgent
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceMutation
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceSampling
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceUniformCrossover
from src.algorithms.ELP_DRL_MO4 import _PYMOO_IMPORT_ERROR
from src.algorithms.ELP_DRL_MO4 import _get_initial_solution_energy
from src.algorithms.ELP_DRL_MO4 import _save_experiment_row
from src.algorithms.ELP_DRL_MO4 import _set_global_seed
from src.algorithms.ELP_DRL_MO4 import get_reference_directions
from src.algorithms.ELP_DRL_MO4 import get_termination
from src.algorithms.ELP_DRL_MO4 import minimize
from src.utils.CR_MatrixStore import CRMatrixStore
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4

np.bool8 = np.bool_


class _ActionSequenceBiMOProblem(Problem):
    def __init__(self, solver, base_solution, sequence_length, use_constraints=True):
        self.solver = solver
        self.base_solution = solver._light_clone_solution(base_solution)
        self.action_count = int(len(solver.valid_actions))
        self.use_constraints = bool(use_constraints)
        super().__init__(
            n_var=int(max(1, sequence_length)),
            n_obj=2,
            n_ieq_constr=2 if self.use_constraints else 0,
            xl=np.zeros(int(max(1, sequence_length)), dtype=int),
            xu=np.full(int(max(1, sequence_length)), self.action_count - 1, dtype=int),
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        sequences = np.asarray(X, dtype=int)
        sample_count = int(sequences.shape[0])
        objectives = np.zeros((sample_count, 2), dtype=float)
        constraints = np.zeros((sample_count, 2), dtype=float)
        for idx in range(sample_count):
            candidate = self.solver._evaluate_action_sequence(self.base_solution, sequences[idx])
            objective_vector = np.asarray(candidate.mo_objectives_min, dtype=float)[:2]
            is_feasible = bool(getattr(candidate, "current_is_feasible", False))
            d_inf = int(getattr(candidate, "current_d_inf", 0) or 0)
            violation = max(float(getattr(candidate, "constraint_violation", 0.0) or 0.0), 0.0)
            if self.use_constraints:
                objectives[idx, :] = objective_vector
                constraints[idx, 0] = 0.0 if is_feasible else 1.0
                constraints[idx, 1] = violation
            else:
                penalty = 0.0 if is_feasible else (1_000_000.0 + 10_000.0 * max(d_inf, 0) + violation)
                objectives[idx, :] = objective_vector + float(penalty)
        out["F"] = objectives
        if self.use_constraints:
            out["G"] = constraints


class ELP(MO4ELP):
    OBJECTIVE_DEFINITION_VERSION = MO_FBSUtil_BiMO4.OBJECTIVE_DEFINITION_VERSION

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        weights = np.asarray(objective_weights if objective_weights is not None else [0.5, 0.5], dtype=float).reshape(-1)
        if weights.size < 2:
            weights = np.pad(weights, (0, 2 - weights.size), constant_values=0.5)
        weights = np.clip(weights[:2], 0.0, None)
        if not np.any(weights > 0):
            weights = np.asarray([0.5, 0.5], dtype=float)
        weights = weights / np.sum(weights)
        self.rl_context_dim = 0
        self.mo_adaptive_weights_enabled = self._parse_env_flag("ELP_BIMO_ADAPTIVE_WEIGHTS_ENABLE", True)
        self.mo_adaptive_weight_blend = float(
            min(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_BLEND", 0.15), 0.0), 1.0)
        )
        legacy_floor = float(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_FLOOR", 0.20))
        self.mo_adaptive_weight_min_component = float(
            min(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_MIN_COMPONENT", legacy_floor), 1e-6), 0.49)
        )
        self.mo_adaptive_weight_floor = self.mo_adaptive_weight_min_component
        self.mo_adaptive_weight_refresh_interval_steps = int(
            max(1, _parse_env_int("ELP_BIMO_ADAPTIVE_WEIGHT_REFRESH_INTERVAL_STEPS", 250))
        )
        self.mo_adaptive_weight_deadband = float(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_DEADBAND", 0.08), 0.0))
        self.mo_base_weights = self._normalize_bi_weights(weights.copy(), floor_value=self.mo_adaptive_weight_min_component)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_last_weight_update_step = -10**9
        self.mo_running_min = np.asarray([math.inf, math.inf], dtype=float)
        self.mo_running_max = np.asarray([-math.inf, -math.inf], dtype=float)
        super().__init__(
            env=env,
            gbest=gbest,
            T=T,
            G=G,
            t_max=t_max,
            k=k,
            archive_limit=archive_limit,
            objective_weights=weights,
        )
        facility_count = int(getattr(self.env, "n", len(getattr(self.env, "areas", [])) or 0))
        self.rel_matrix, self.cr_matrix_payload, self.cr_matrix_path = CRMatrixStore.load_matrix(
            instance_name=self.instance_name,
            expected_facility_count=facility_count,
        )
        self.dist_req_matrix = None
        # 双目标版本不复用旧的 4 目标参考前沿定义，避免指标口径错配。
        self.archive_reference_front_enabled = False
        self.archive_reference_front_payload = None
        self.archive_reference_vectors = []
        self.archive_update_count = 0
        self._reset_running_objective_bounds()

    def _run_impl(self):
        previous_algorithm = os.getenv("ELP_EXP_ALGORITHM")
        previous_definition_version = mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION
        previous_mo_util = mo4_module.MO_FBSUtil
        if not previous_algorithm:
            os.environ["ELP_EXP_ALGORITHM"] = "ELP_DRL_BiMO4"
        mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION = self.OBJECTIVE_DEFINITION_VERSION
        mo4_module.MO_FBSUtil = MO_FBSUtil_BiMO4
        try:
            result = super()._run_impl()
            self._finalize_archive_update_reporting()
            report_snapshot = self._report_representative_snapshot()
            if report_snapshot and result is not None and len(result) >= 7:
                patched_result = list(result)
                patched_result[1] = bool(getattr(report_snapshot["solution"], "current_is_feasible", False))
                patched_result[2] = copy.deepcopy(report_snapshot["solution"])
                if report_snapshot["score"] is not None:
                    patched_result[3] = float(report_snapshot["score"])
                return tuple(patched_result)
            return result
        finally:
            mo4_module.MO_FBSUtil = previous_mo_util
            mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION = previous_definition_version
            if previous_algorithm is None:
                os.environ.pop("ELP_EXP_ALGORITHM", None)

    def _reset_mo_logging_state(self):
        super()._reset_mo_logging_state()
        self.archive_update_count = 0
        self._reset_running_objective_bounds()
        base_weights = np.asarray(getattr(self, "mo_base_weights", [0.5, 0.5]), dtype=float)
        floor_value = float(getattr(self, "mo_adaptive_weight_min_component", 0.20) or 0.20)
        self.mo_base_weights = self._normalize_bi_weights(base_weights, floor_value=floor_value)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_update_step = -10**9

    def _reset_baseline_archive_state(self):
        super()._reset_baseline_archive_state()
        self._reset_running_objective_bounds()
        base_weights = np.asarray(getattr(self, "mo_base_weights", [0.5, 0.5]), dtype=float)
        floor_value = float(getattr(self, "mo_adaptive_weight_min_component", 0.20) or 0.20)
        self.mo_base_weights = self._normalize_bi_weights(base_weights, floor_value=floor_value)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_update_step = -10**9

    def _reset_running_objective_bounds(self):
        self.mo_running_min = np.asarray([math.inf, math.inf], dtype=float)
        self.mo_running_max = np.asarray([-math.inf, -math.inf], dtype=float)

    def _update_running_objective_bounds(self, objectives_min):
        vector = np.asarray(objectives_min, dtype=float).reshape(-1)[:2]
        if vector.size < 2 or not np.all(np.isfinite(vector)):
            return
        if not hasattr(self, "mo_running_min") or not hasattr(self, "mo_running_max"):
            self._reset_running_objective_bounds()
        self.mo_running_min = np.minimum(self.mo_running_min, vector)
        self.mo_running_max = np.maximum(self.mo_running_max, vector)

    @staticmethod
    def _repo_root_path():
        return Path(__file__).resolve().parents[2]

    def _resolved_result_path(self, relative_path):
        if not relative_path:
            return None
        return self._repo_root_path() / str(relative_path).replace("/", os.sep)

    def _effective_archive_update_count(self):
        counter_value = int(getattr(self, "archive_update_count", 0) or 0)
        trace_bucket = getattr(self, "_mo_total_counters", {}) or {}
        bucket_value = int(trace_bucket.get("archiveChanges", 0) or 0)
        return max(counter_value, bucket_value)

    def _patch_run_summary_file(self, archive_update_count):
        run_summary_path = self._resolved_result_path(getattr(self.mo_run_summary, "get", lambda *_: None)("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.pop("gbestUpdateCount", None)
            payload["archiveUpdateCount"] = int(archive_update_count)
            run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_action_stats_file(self, archive_update_count):
        action_stats_path = self._resolved_result_path(getattr(self.mo_run_summary, "get", lambda *_: None)("actionStatsPath"))
        if action_stats_path is None or not action_stats_path.exists():
            return
        payload = json.loads(action_stats_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            meta_section = payload.get("meta")
            if isinstance(meta_section, dict):
                meta_section.pop("gbestUpdateCount", None)
                meta_section["archiveUpdateCount"] = int(archive_update_count)
            overall_section = payload.get("overall")
            if isinstance(overall_section, dict):
                overall_section.pop("gbestUpdateCount", None)
                overall_section["archiveUpdateCount"] = int(archive_update_count)
                overall_section["archiveChanges"] = int(archive_update_count)
            action_stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_summary_csv(self, archive_update_count):
        summary_csv_path = Path(config.RESULT_PATH) / "mo_runs_summary" / f"{self.instance_name}-ELP_DRL_BiMO4.csv"
        if not summary_csv_path.exists() or not isinstance(self.mo_run_summary, dict):
            return
        frame = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
        if "runId" not in frame.columns:
            return
        run_id = str(self.mo_run_summary.get("runId") or "")
        row_mask = frame["runId"].astype(str) == run_id
        if not row_mask.any():
            return
        if "gbestUpdateCount" in frame.columns:
            frame = frame.drop(columns=["gbestUpdateCount"])
        if "archiveUpdateCount" not in frame.columns:
            frame["archiveUpdateCount"] = pd.NA
        frame.loc[row_mask, "archiveUpdateCount"] = int(archive_update_count)
        frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _solution_array_payload(solution):
        solution_array = getattr(getattr(solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        return solution_array

    @staticmethod
    def _normalize_bi_weights(weights, floor_value=0.20):
        vector = np.asarray(weights, dtype=float).reshape(-1)
        if vector.size < 2:
            vector = np.pad(vector, (0, 2 - vector.size), constant_values=0.5)
        vector = np.clip(vector[:2], float(floor_value), None)
        total = float(np.sum(vector))
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray([0.5, 0.5], dtype=float)
        return vector / total

    def _compute_adaptive_weight_target(self):
        if not bool(getattr(self, "mo_adaptive_weights_enabled", False)):
            return self.mo_base_weights.copy()
        if len(self.pareto_archive) < 2:
            return self.mo_base_weights.copy()
        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if normalized.shape[0] < 2:
            return self.mo_base_weights.copy()
        rows = []
        for idx in range(normalized.shape[0]):
            delta = normalized - normalized[idx]
            norms = np.linalg.norm(delta, axis=1)
            norms[idx] = np.inf
            nearest = float(np.min(norms))
            rows.append((nearest, idx))
        rows.sort(key=lambda item: item[0], reverse=True)
        sparse_vector = np.asarray(normalized[int(rows[0][1])], dtype=float)
        target = 1.0 - sparse_vector
        target = np.maximum(target, float(getattr(self, "mo_adaptive_weight_min_component", 0.20)))
        return self._normalize_bi_weights(
            target,
            floor_value=getattr(self, "mo_adaptive_weight_min_component", 0.20),
        )

    def _refresh_dynamic_weights(self):
        target = self._compute_adaptive_weight_target()
        self.mo_last_weight_target = np.asarray(target, dtype=float)
        if not bool(getattr(self, "mo_adaptive_weights_enabled", False)):
            self.mo_weights = self.mo_base_weights.copy()
            return self.mo_weights
        current_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
        last_update_step = int(getattr(self, "mo_last_weight_update_step", -10**9) or -10**9)
        refresh_interval = int(max(1, int(getattr(self, "mo_adaptive_weight_refresh_interval_steps", 250) or 250)))
        if int(getattr(self, "mo_weight_update_count", 0) or 0) > 0 and (current_step - last_update_step) < refresh_interval:
            return self.mo_weights
        current = np.asarray(getattr(self, "mo_weights", self.mo_base_weights), dtype=float)
        deadband = float(max(getattr(self, "mo_adaptive_weight_deadband", 0.08) or 0.08, 0.0))
        if float(np.sum(np.abs(target - current))) < deadband:
            return self.mo_weights
        blend = float(getattr(self, "mo_adaptive_weight_blend", 0.15) or 0.15)
        updated = (1.0 - blend) * current + blend * target
        self.mo_weights = self._normalize_bi_weights(
            updated,
            floor_value=getattr(self, "mo_adaptive_weight_min_component", 0.20),
        )
        self.mo_weight_update_count = int(getattr(self, "mo_weight_update_count", 0) or 0) + 1
        self.mo_last_weight_update_step = current_step
        return self.mo_weights

    def _agent_state_context(self, solution):
        return None

    def _report_representative_snapshot(self):
        report_solution, report_score, report_index = MO_FBSUtil_BiMO4.select_knee_solution(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        if report_solution is None:
            report_solution = self.representative_solution
            report_score = float(getattr(self, "representative_decision_score", math.inf))
            report_index = self.representative_archive_index
        if report_solution is None:
            return None
        return {
            "solution": copy.deepcopy(report_solution),
            "score": self._safe_float(report_score),
            "index": None if report_index is None else int(report_index),
            "mhc": self._safe_float(getattr(report_solution, "MHC", None)),
            "cr": self._safe_float(getattr(report_solution, "CR", None)),
            "solution_array": self._solution_array_payload(report_solution),
        }

    def _patch_archive_json_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        archive_path = self._resolved_result_path(self.mo_run_summary.get("paretoArchivePath"))
        if archive_path is None or not archive_path.exists():
            return
        payload = json.loads(archive_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["representativeArchiveIndex"] = None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
        payload["representativeDecisionScore"] = report_snapshot["score"]
        payload["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
        payload["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
        payload["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
        payload["dqnContextMode"] = "disabled"
        payload["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_run_summary_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        run_summary_path = self._resolved_result_path(self.mo_run_summary.get("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["searchRepresentativeArchiveIndex"] = None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
        payload["searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
        payload["searchRepMhc"] = self._safe_float(getattr(self.representative_solution, "MHC", None)) if self.representative_solution is not None else None
        payload["searchRepCr"] = self._safe_float(getattr(self.representative_solution, "CR", None)) if self.representative_solution is not None else None
        payload["representativeArchiveIndex"] = None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
        payload["representativeDecisionScore"] = report_snapshot["score"]
        payload["decisionScore"] = report_snapshot["score"]
        payload["repMhc"] = report_snapshot["mhc"]
        payload["repCr"] = report_snapshot["cr"]
        payload["representativeSolution"] = report_snapshot["solution_array"]
        payload["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
        payload["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
        payload["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
        payload["adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
        payload["adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
        payload["adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
        payload["dqnContextMode"] = "disabled"
        payload["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_summary_csv_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        summary_csv_path = Path(config.RESULT_PATH) / "mo_runs_summary" / f"{self.instance_name}-ELP_DRL_BiMO4.csv"
        if not summary_csv_path.exists():
            return
        frame = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
        if "runId" not in frame.columns:
            return
        row_mask = frame["runId"].astype(str) == str(self.mo_run_summary.get("runId") or "")
        if not row_mask.any():
            return
        for column in (
            "searchRepresentativeArchiveIndex",
            "searchRepresentativeDecisionScore",
            "searchRepMhc",
            "searchRepCr",
            "adaptiveWeightsEnabled",
            "adaptiveWeightBlend",
            "adaptiveWeightUpdates",
            "objectiveWeights",
            "baseObjectiveWeights",
            "adaptiveWeightTarget",
            "dqnContextMode",
            "dqnContextDim",
        ):
            if column not in frame.columns:
                frame[column] = pd.NA
        frame.loc[row_mask, "searchRepresentativeArchiveIndex"] = (
            None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
        )
        frame.loc[row_mask, "searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
        frame.loc[row_mask, "searchRepMhc"] = self._safe_float(getattr(self.representative_solution, "MHC", None)) if self.representative_solution is not None else None
        frame.loc[row_mask, "searchRepCr"] = self._safe_float(getattr(self.representative_solution, "CR", None)) if self.representative_solution is not None else None
        if "representativeArchiveIndex" in frame.columns:
            frame.loc[row_mask, "representativeArchiveIndex"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
            )
        if "representativeDecisionScore" in frame.columns:
            frame.loc[row_mask, "representativeDecisionScore"] = report_snapshot["score"]
        if "decisionScore" in frame.columns:
            frame.loc[row_mask, "decisionScore"] = report_snapshot["score"]
        if "repMhc" in frame.columns:
            frame.loc[row_mask, "repMhc"] = report_snapshot["mhc"]
        if "repCr" in frame.columns:
            frame.loc[row_mask, "repCr"] = report_snapshot["cr"]
        frame.loc[row_mask, "adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
        frame.loc[row_mask, "adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
        frame.loc[row_mask, "adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
        frame.loc[row_mask, "objectiveWeights"] = json.dumps(np.asarray(self.mo_weights, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "baseObjectiveWeights"] = json.dumps(np.asarray(self.mo_base_weights, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "adaptiveWeightTarget"] = json.dumps(np.asarray(self.mo_last_weight_target, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "dqnContextMode"] = "disabled"
        frame.loc[row_mask, "dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    def _apply_report_representative_reporting(self):
        report_snapshot = self._report_representative_snapshot()
        if not report_snapshot:
            return None
        if isinstance(self.mo_run_summary, dict):
            self.mo_run_summary["searchRepresentativeArchiveIndex"] = (
                None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
            )
            self.mo_run_summary["searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
            self.mo_run_summary["searchRepMhc"] = (
                self._safe_float(getattr(self.representative_solution, "MHC", None))
                if self.representative_solution is not None
                else None
            )
            self.mo_run_summary["searchRepCr"] = (
                self._safe_float(getattr(self.representative_solution, "CR", None))
                if self.representative_solution is not None
                else None
            )
            self.mo_run_summary["representativeArchiveIndex"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
            )
            self.mo_run_summary["representativeDecisionScore"] = report_snapshot["score"]
            self.mo_run_summary["decisionScore"] = report_snapshot["score"]
            self.mo_run_summary["repMhc"] = report_snapshot["mhc"]
            self.mo_run_summary["repCr"] = report_snapshot["cr"]
            self.mo_run_summary["representativeSolution"] = report_snapshot["solution_array"]
            self.mo_run_summary["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
            self.mo_run_summary["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
            self.mo_run_summary["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
            self.mo_run_summary["adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
            self.mo_run_summary["adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
            self.mo_run_summary["adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
            self.mo_run_summary["dqnContextMode"] = "disabled"
            self.mo_run_summary["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        if isinstance(self.last_run_payload, dict):
            self.last_run_payload["search_representative_archive_index"] = (
                None if self.representative_archive_index is None else int(self.representative_archive_index)
            )
            self.last_run_payload["search_representative_decision_score"] = self._safe_float(self.representative_decision_score)
            self.last_run_payload["search_rep_mhc"] = (
                self._safe_float(getattr(self.representative_solution, "MHC", None))
                if self.representative_solution is not None
                else None
            )
            self.last_run_payload["search_rep_cr"] = (
                self._safe_float(getattr(self.representative_solution, "CR", None))
                if self.representative_solution is not None
                else None
            )
            self.last_run_payload["rep_mhc"] = report_snapshot["mhc"]
            self.last_run_payload["rep_cr"] = report_snapshot["cr"]
            self.last_run_payload["decision_score"] = report_snapshot["score"]
            self.last_run_payload["report_representative_archive_index"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"])
            )
            self.last_run_payload["objective_weights"] = np.asarray(self.mo_weights, dtype=float).tolist()
            self.last_run_payload["base_objective_weights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
            self.last_run_payload["adaptive_weight_target"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
            self.last_run_payload["adaptive_weights_enabled"] = bool(self.mo_adaptive_weights_enabled)
            self.last_run_payload["adaptive_weight_updates"] = int(self.mo_weight_update_count)
            self.last_run_payload["dqn_context_mode"] = "disabled"
            self.last_run_payload["dqn_context_dim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        self._patch_archive_json_with_report_representative(report_snapshot)
        self._patch_run_summary_with_report_representative(report_snapshot)
        self._patch_summary_csv_with_report_representative(report_snapshot)
        return report_snapshot

    def _finalize_archive_update_reporting(self):
        archive_update_count = self._effective_archive_update_count()
        if isinstance(self.mo_run_summary, dict):
            self.mo_run_summary.pop("gbestUpdateCount", None)
            self.mo_run_summary["archiveUpdateCount"] = int(archive_update_count)
        if isinstance(self.last_run_payload, dict):
            self.last_run_payload.pop("gbest_update_count", None)
            self.last_run_payload.pop("gbestUpdateCount", None)
            self.last_run_payload["archive_update_count"] = int(archive_update_count)
        self._patch_run_summary_file(archive_update_count)
        self._patch_action_stats_file(archive_update_count)
        self._patch_summary_csv(archive_update_count)
        self._apply_report_representative_reporting()

    def _compute_reference_front_metrics(self):
        archive_hypervolume = self._safe_float(
            MO_FBSUtil_BiMO4.archive_hypervolume(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )
        archive_spacing = self._safe_float(
            MO_FBSUtil_BiMO4.archive_spacing(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )
        return {
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": None,
            "reference_front_path": None,
            "reference_front_size": None,
            "reference_front_archive_count": None,
            "archive_hypervolume_mode": "self_reference_2d",
            "archive_hypervolume_reference_point": None,
        }

    def _score_candidate_encoding(self, permutation, bay, solution):
        metrics = FBSUtil._evaluate_candidate_encoding_fast(
            permutation,
            bay,
            solution.areas,
            solution.H,
            solution.F,
            solution.aspect_limits,
            v_worst=self.mo_worst_feasible_mhc,
            k_penalty=self.k_penalty,
            distance_metric="manhattan",
        )
        objectives_raw = MO_FBSUtil_BiMO4.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
        )
        objectives_min = MO_FBSUtil_BiMO4.to_minimization(objectives_raw)
        self._update_running_objective_bounds(objectives_min)
        constraint_violation = MO_FBSUtil_BiMO4.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        search_energy = MO_FBSUtil_BiMO4.search_energy(
            objectives_min,
            is_feasible=bool(metrics["is_feasible"]),
            d_inf=int(metrics["d_inf"]),
            total_violation=constraint_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        return (
            float(search_energy),
            int(metrics["d_inf"]),
            float(constraint_violation),
            float(metrics["mhc"]),
            np.asarray(permutation, dtype=int),
            np.asarray(bay, dtype=int),
        )

    def _refresh_solution_search_metrics(self, solution):
        objectives_min = getattr(solution, "mo_objectives_min", None)
        if objectives_min is None:
            return float(getattr(solution, "fitness", math.inf))

        self._update_running_objective_bounds(objectives_min)
        proxy_energy = MO_FBSUtil_BiMO4.surrogate_energy(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        decision_score = MO_FBSUtil_BiMO4.decision_score(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        total_violation = float(getattr(solution, "constraint_violation", 0.0) or 0.0)
        search_energy = MO_FBSUtil_BiMO4.search_energy(
            objectives_min,
            is_feasible=bool(getattr(solution, "current_is_feasible", False)),
            d_inf=int(getattr(solution, "current_d_inf", 0) or 0),
            total_violation=total_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )

        solution.proxy_energy = float(proxy_energy)
        solution.decision_score = float(decision_score)
        solution.fitness = float(search_energy)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", math.inf)
        solution.best_fitness = getattr(self, "best_feasible_cost", math.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        return float(search_energy)

    def _refresh_archive_state(self):
        feasible_archive = [candidate for candidate in self.pareto_archive if getattr(candidate, "current_is_feasible", False)]
        self.pareto_archive = feasible_archive
        self.mo_ideal, self.mo_nadir = MO_FBSUtil_BiMO4.compute_ideal_nadir(self.pareto_archive)
        self._refresh_dynamic_weights()

        for candidate in self.pareto_archive:
            self._refresh_solution_search_metrics(candidate)

        representative, decision_score, archive_index = MO_FBSUtil_BiMO4.select_representative_solution(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        self.representative_solution = copy.deepcopy(representative) if representative is not None else None
        self.representative_decision_score = float(decision_score)
        self.representative_archive_index = archive_index
        if self.representative_solution is not None:
            self._refresh_solution_search_metrics(self.representative_solution)

    def _build_elite_anchor_pool(self):
        anchors = []
        max_anchor_count = int(max(1, self.mo_elite_anchor_count))

        primary_candidates = [self.best_feasible_solution]
        if bool(getattr(self, "mo_elite_include_representative", True)):
            primary_candidates.append(self.representative_solution)
        for candidate in self._sorted_elite_candidates(primary_candidates):
            self._append_unique_elite_anchor(anchors, candidate)
            break

        extreme_candidates = []
        if self.pareto_archive:
            if bool(getattr(self, "mo_elite_include_extremes", True)):
                objective_count = 2
                for objective_idx in range(objective_count):
                    extreme_candidate = min(
                        self.pareto_archive,
                        key=lambda item: float(
                            np.asarray(getattr(item, "mo_objectives_min", [math.inf] * objective_count), dtype=float)[
                                objective_idx
                            ]
                        ),
                    )
                    extreme_candidates.append(extreme_candidate)
                if extreme_candidates and len(anchors) < max_anchor_count:
                    offset = int(
                        (
                            int(getattr(self, "_trace_global_step", 0) or 0)
                            + int(getattr(self, "elite_trigger_count", 0) or 0)
                        )
                        % len(extreme_candidates)
                    )
                    rotated_extremes = extreme_candidates[offset:] + extreme_candidates[:offset]
                    for extreme_candidate in rotated_extremes:
                        before_count = len(anchors)
                        self._append_unique_elite_anchor(anchors, extreme_candidate)
                        if len(anchors) > before_count:
                            break
            if bool(getattr(self, "mo_elite_include_sparse", True)):
                for sparse_candidate in self._elite_sparse_archive_candidates(self.mo_elite_sparse_anchor_count):
                    self._append_unique_elite_anchor(anchors, sparse_candidate)
                    if len(anchors) >= max_anchor_count:
                        break

        if len(anchors) < max_anchor_count:
            fill_candidates = []
            fill_candidates.extend(primary_candidates)
            fill_candidates.extend(extreme_candidates)
            fill_candidates.extend(self._elite_sparse_archive_candidates(max_anchor_count))
            for candidate in self._sorted_elite_candidates(fill_candidates):
                self._append_unique_elite_anchor(anchors, candidate)
                if len(anchors) >= max_anchor_count:
                    break

        return anchors[:max_anchor_count]

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
        solution.CR = float(metrics["cr"])
        solution.DR = 0.0
        solution.AR = 0.0
        solution.raw_cost = float(metrics["cost"])
        solution.mo_objectives_raw = np.asarray(metrics["mo_objectives_raw"], dtype=float)
        solution.mo_objectives_min = np.asarray(metrics["mo_objectives_min"], dtype=float)
        solution.constraint_violation = float(metrics["constraint_violation"])
        solution.current_d_inf = int(metrics["d_inf"])
        solution.current_is_feasible = bool(metrics["is_feasible"])
        solution.feasible_solution_count = getattr(self, "feasible_solution_count", 0)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", math.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.best_fitness = getattr(self, "best_feasible_cost", math.inf)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        self._refresh_solution_search_metrics(solution)
        solution.state = solution.constructState()

    def _evaluate_solution(self, solution):
        metrics = FBSUtil.evaluate_layout_fast(
            solution.fbs_model,
            solution.areas,
            solution.H,
            solution.F,
            solution.aspect_limits,
            v_worst=self.mo_worst_feasible_mhc,
            k_penalty=self.k_penalty,
            distance_metric="manhattan",
        )
        objectives_raw = MO_FBSUtil_BiMO4.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
        )
        constraint_violation = MO_FBSUtil_BiMO4.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        objectives_min = MO_FBSUtil_BiMO4.to_minimization(objectives_raw)
        self._update_running_objective_bounds(objectives_min)
        metrics.update(
            {
                "cr": float(objectives_raw[1]),
                "dr": 0.0,
                "ar": 0.0,
                "mo_objectives_raw": np.asarray(objectives_raw, dtype=float),
                "mo_objectives_min": objectives_min,
                "constraint_violation": float(constraint_violation),
            }
        )
        self._sync_solution_metrics(solution, metrics)
        return metrics

    def _observe_feasible_state(self, solution):
        if not getattr(solution, "current_is_feasible", False):
            self._last_archive_observation = {"archive_changed": False, "rep_changed": False}
            return False

        self.feasible_solution_count += 1
        self.mo_worst_feasible_mhc = (
            float(solution.MHC)
            if self.mo_worst_feasible_mhc is None
            else max(float(self.mo_worst_feasible_mhc), float(solution.MHC))
        )

        previous_rep_vector = None
        previous_rep_score = float(self.representative_decision_score)
        if self.representative_solution is not None and getattr(self.representative_solution, "mo_objectives_min", None) is not None:
            previous_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)

        updated_archive, inserted, removed = MO_FBSUtil_BiMO4.update_pareto_archive(
            self.pareto_archive,
            solution,
            max_size=self.archive_limit,
            clone_fn=copy.deepcopy,
        )
        archive_changed = bool(inserted)
        if archive_changed:
            self.pareto_archive = updated_archive
        elif not self.pareto_archive and getattr(solution, "current_is_feasible", False):
            self.pareto_archive = [copy.deepcopy(solution)]
            archive_changed = True
        if archive_changed:
            self.archive_update_count = int(getattr(self, "archive_update_count", 0) or 0) + 1

        self._refresh_archive_state()
        self._refresh_solution_search_metrics(solution)

        if self.representative_solution is not None:
            self.best_feasible_solution = copy.deepcopy(self.representative_solution)
            self.gbest = copy.deepcopy(self.representative_solution)
            self.true_gbest = copy.deepcopy(self.representative_solution)
            self.best_feasible_cost = float(self.representative_decision_score)
            self.best_energy = float(self.representative_decision_score)
        elif np.isfinite(getattr(solution, "decision_score", math.inf)):
            self.best_feasible_solution = copy.deepcopy(solution)
            self.gbest = copy.deepcopy(solution)
            self.true_gbest = copy.deepcopy(solution)
            self.best_feasible_cost = float(solution.decision_score)
            self.best_energy = float(solution.decision_score)

        current_rep_vector = None
        if self.representative_solution is not None and getattr(self.representative_solution, "mo_objectives_min", None) is not None:
            current_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)
        rep_changed = False
        if previous_rep_vector is None and current_rep_vector is not None:
            rep_changed = True
        elif previous_rep_vector is not None and current_rep_vector is None:
            rep_changed = True
        elif previous_rep_vector is not None and current_rep_vector is not None:
            rep_changed = not np.allclose(previous_rep_vector, current_rep_vector, atol=1e-9, rtol=1e-7)
        rep_score_changed = abs(float(self.representative_decision_score) - previous_rep_score) > 1e-12
        rep_changed = bool(rep_changed or rep_score_changed)

        self._last_archive_observation = {
            "archive_changed": bool(archive_changed),
            "rep_changed": bool(rep_changed),
            "removed_count": int(removed),
        }
        return bool(archive_changed)

    def _archive_item_payload(self, solution, index):
        solution_array = getattr(getattr(solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        return {
            "index": int(index),
            "decisionScore": float(getattr(solution, "decision_score", math.inf)),
            "searchEnergy": float(getattr(solution, "fitness", math.inf)),
            "mhc": float(getattr(solution, "MHC", math.inf)),
            "cr": float(getattr(solution, "CR", 0.0)),
            "dInf": int(getattr(solution, "current_d_inf", 0) or 0),
            "constraintViolation": float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            "isFeasible": bool(getattr(solution, "current_is_feasible", False)),
            "moObjectivesRaw": np.asarray(getattr(solution, "mo_objectives_raw", []), dtype=float).tolist(),
            "moObjectivesMin": np.asarray(getattr(solution, "mo_objectives_min", []), dtype=float).tolist(),
            "solution": solution_array,
        }

    def _save_pareto_archive(self, start_time, algorithm_name=None):
        if not self.pareto_archive:
            self.pareto_archive_path = None
            return None

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        archive_dir = os.path.join(os.path.abspath(config.RESULT_PATH), "pareto_archives")
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = (
            start_time.strftime("%Y%m%d_%H%M%S_%f")
            if start_time is not None
            else datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        algo_tag = self._normalize_algorithm_tag(algorithm_name)
        archive_path = os.path.join(archive_dir, f"{self.instance_name}-{algo_tag}-{timestamp}.json")
        payload = {
            "instance": self.instance_name,
            "algorithm": algo_tag,
            "objectiveDefinitionVersion": self.OBJECTIVE_DEFINITION_VERSION,
            "objectiveNames": ["MHC", "CR"],
            "objectiveDirections": ["min", "max"],
            "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
            "archiveSize": len(self.pareto_archive),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "representativeDecisionScore": None if not np.isfinite(self.representative_decision_score) else float(self.representative_decision_score),
            "items": [self._archive_item_payload(solution, index + 1) for index, solution in enumerate(self.pareto_archive)],
        }
        with open(archive_path, "w", encoding="utf-8") as file_obj:
            import json
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        self.pareto_archive_path = os.path.relpath(archive_path, repo_root).replace("\\", "/")
        return self.pareto_archive_path

    def _ensure_pymoo_available(self):
        if NSGA2 is not None and MOEAD is not None and SPEA2 is not None:
            return
        raise ImportError(
            "缺少 pymoo 依赖，无法运行 NSGA-II/MOEA-D/SPEA2 双目标基线。请先安装 `pymoo>=0.6.1`。"
        ) from _PYMOO_IMPORT_ERROR

    def run_moea_baseline(
        self,
        algorithm_name,
        population_size=64,
        generations=80,
        sequence_length=None,
        seed=None,
    ):
        algo_key = str(algorithm_name or "").strip().lower().replace("-", "").replace("/", "")
        previous_mo_util = mo4_module.MO_FBSUtil
        mo4_module.MO_FBSUtil = MO_FBSUtil_BiMO4
        try:
            self._ensure_pymoo_available()
            if algo_key not in {"nsga2", "moead", "spea2"}:
                raise ValueError(f"Unsupported baseline algorithm for BiMO: {algorithm_name}")

            population_size = int(max(8, population_size))
            generations = int(max(1, generations))
            sequence_length = int(max(1, sequence_length if sequence_length is not None else self.t_max))
            run_seed = None if seed is None else int(seed)
            wall_time_limit_seconds = float(max(getattr(self, "wall_time_limit_seconds", 0.0) or 0.0, 0.0))

            self._reset_baseline_archive_state()
            start_time = datetime.datetime.now()
            fast_time = start_time

            base_solution = self._light_clone_solution(self.s)
            self._evaluate_solution(base_solution)
            self._observe_feasible_state(base_solution)

            problem = _ActionSequenceBiMOProblem(
                solver=self,
                base_solution=base_solution,
                sequence_length=sequence_length,
                use_constraints=(algo_key != "moead"),
            )
            sampling = _ActionSequenceSampling()
            crossover = _ActionSequenceUniformCrossover(swap_prob=0.5)
            mutation = _ActionSequenceMutation(mutation_prob=1.0 / float(max(1, sequence_length)))

            if algo_key == "nsga2":
                algorithm = NSGA2(
                    pop_size=population_size,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    eliminate_duplicates=True,
                )
                effective_population = population_size
            elif algo_key == "spea2":
                algorithm = SPEA2(
                    pop_size=population_size,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    survival=SPEA2Survival(normalize=False),
                    eliminate_duplicates=True,
                )
                effective_population = population_size
            else:
                n_partitions = self._compute_moead_partitions(population_size, objective_count=2)
                ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
                algorithm = MOEAD(
                    ref_dirs=ref_dirs,
                    n_neighbors=min(20, max(2, len(ref_dirs) - 1)),
                    prob_neighbor_mating=0.7,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                )
                effective_population = int(len(ref_dirs))

            termination = get_termination("n_gen", generations)
            termination_mode = "n_gen"
            if wall_time_limit_seconds > 0.0:
                from pymoo.core.termination import TerminateIfAny

                termination = TerminateIfAny(
                    get_termination("n_gen", generations),
                    get_termination("time", wall_time_limit_seconds),
                )
                termination_mode = "n_gen_or_time"

            optimize_start = time.perf_counter()
            result = minimize(
                problem,
                algorithm,
                termination=termination,
                seed=run_seed,
                save_history=False,
                verbose=False,
            )
            optimize_runtime_seconds = float(max(time.perf_counter() - optimize_start, 0.0))
            actual_generations = int(
                max(
                    1,
                    int(
                        getattr(
                            getattr(result, "algorithm", None),
                            "n_gen",
                            getattr(algorithm, "n_gen", generations),
                        )
                        or generations
                    ),
                )
            )
            wall_time_terminated = bool(
                wall_time_limit_seconds > 0.0
                and optimize_runtime_seconds + 1e-9 >= max(wall_time_limit_seconds - 1.0, wall_time_limit_seconds * 0.98)
            )

            best_observed = self.best_feasible_cost
            for sequence in self._collect_result_sequences(result):
                candidate = self._evaluate_action_sequence(base_solution, sequence)
                changed = self._observe_feasible_state(candidate)
                if changed and np.isfinite(self.best_feasible_cost) and self.best_feasible_cost < best_observed:
                    best_observed = float(self.best_feasible_cost)
                    fast_time = datetime.datetime.now()

            self._refresh_archive_state()
            end_time = datetime.datetime.now()
            iteration_count = int(effective_population * actual_generations)

            best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else self._light_clone_solution(base_solution)
            if self.representative_solution is not None:
                best_solution = copy.deepcopy(self.representative_solution)
            is_valid = bool(getattr(best_solution, "current_is_feasible", False))
            best_energy = float(getattr(best_solution, "decision_score", math.inf))
            if not np.isfinite(best_energy):
                best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)

            stable_decision_score = self._safe_float(getattr(best_solution, "proxy_energy", None))
            archive_algo_name = f"MO_BASELINE_{algo_key.upper()}"
            archive_path = self._save_pareto_archive(start_time, algorithm_name=archive_algo_name)
            reference_metrics = self._compute_reference_front_metrics()
            self.last_run_payload = {
                "pareto_archive_path": archive_path,
                "pareto_size": len(self.pareto_archive),
                "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
                "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
                "rep_dr": None,
                "rep_ar": None,
                "decision_score": self._safe_float(best_energy),
                "stable_decision_score": stable_decision_score,
                "archive_hypervolume": reference_metrics["archive_hypervolume"],
                "archive_spacing": reference_metrics["archive_spacing"],
                "archive_igd": None,
                "reference_front_path": None,
                "reference_front_size": None,
                "reference_front_archive_count": None,
                "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
                "archive_hypervolume_reference_point": None,
                "mo_run_id": None,
                "mo_bundle_dir": None,
                "mo_trace_path": None,
                "mo_events_path": None,
                "mo_action_stats_path": None,
                "mo_run_summary_path": None,
                "wall_time_terminated": bool(wall_time_terminated),
                "wall_time_limit_seconds": self._safe_float(wall_time_limit_seconds),
                "runtime_seconds": self._safe_float(optimize_runtime_seconds),
                "baseline_algorithm": algo_key.upper(),
                "baseline_population": int(effective_population),
                "baseline_generations": int(actual_generations),
                "baseline_generations_requested": int(generations),
                "baseline_sequence_length": int(sequence_length),
                "baseline_seed": run_seed,
                "baseline_termination_mode": termination_mode,
            }
            return iteration_count, is_valid, best_solution, best_energy, start_time, end_time, fast_time
        finally:
            mo4_module.MO_FBSUtil = previous_mo_util


def _format_summary_metrics(solver, best_energy):
    payload = getattr(solver, "last_run_payload", {}) or {}
    hv = payload.get("archive_hypervolume")
    spacing = payload.get("archive_spacing")

    def _fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    return (
        f"representative decision score: {float(best_energy):.6f} | "
        f"HV: {_fmt(hv)} | Spacing: {_fmt(spacing)}"
    )


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


def _parse_env_int_list(name):
    raw = os.getenv(name)
    if raw is None:
        return []
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception:
            continue
    return values


if __name__ == "__main__":
    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}" if baseline_enabled else "ELP_DRL_BiMO4"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    default_remark = "WarmStart(GA)+ELP+Bi-objective Pareto archive (MHC min, CR max)"
    exp_remark = os.getenv("ELP_EXP_REMARK", default_remark)
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

    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
        if fixed_seeds:
            run_seed = int(fixed_seeds[run_index])
        else:
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
            try:
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
            except Exception as exc:
                logger.exception(f"Experiment {i + 1} failed: {exc}")
    else:
        elp_solver, result_tuple = _run_once(0)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
        print(f"Single run complete | {_format_summary_metrics(elp_solver, best_energy)}")
        if not baseline_enabled:
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
