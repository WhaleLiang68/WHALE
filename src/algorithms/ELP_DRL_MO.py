import copy
import datetime
import json
import math
import os
from pathlib import Path

import gym
import numpy as np
from loguru import logger

import src
import src.utils.ExperimentsUtil as ExperimentsUtil
import src.utils.FBSUtil as FBSUtil
import src.utils.MO_ExperimentsUtil as MO_ExperimentsUtil
import src.utils.config as config
from src.utils.FBSModel import FBSModel
from src.algorithms.ELP_DRL_Standard import ELP as StandardELP
from src.algorithms.ELP_DRL_Standard import StandardDQNAgent
from src.algorithms.ELP_DRL_Standard import StandardQLearningAgent
from src.algorithms.ELP_DRL_Standard import _get_initial_solution_energy
from src.utils.MO_DataGenerator import MO_DataGenerator
from src.utils.MO_FBSUtil import MO_FBSUtil

np.bool8 = np.bool_


class ELP(StandardELP):
    """Multi-objective ELP variant that keeps a Pareto archive and representative solution."""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.instance_name = str(getattr(base_env, "instance", "UNKNOWN") or "UNKNOWN")
        self.mo_weights = np.asarray(
            objective_weights if objective_weights is not None else [0.25, 0.25, 0.25, 0.25],
            dtype=float,
        ).reshape(-1)
        if self.mo_weights.size < 4:
            self.mo_weights = np.pad(self.mo_weights, (0, 4 - self.mo_weights.size), constant_values=0.25)
        self.mo_weights = self.mo_weights[:4]
        if not np.any(self.mo_weights > 0):
            self.mo_weights = np.full(4, 0.25, dtype=float)
        self.mo_weights = self.mo_weights / np.sum(self.mo_weights)

        self.archive_limit = int(max(4, archive_limit))
        self.fast_segment_insert_budget = 8
        self.fast_segment_insert_segment_lengths = (2, 3)
        self.pareto_archive = []
        self.representative_solution = None
        self.representative_decision_score = math.inf
        self.representative_archive_index = None
        self.mo_ideal = None
        self.mo_nadir = None
        self.pareto_archive_path = None
        self.last_run_payload = None
        self.mo_run_summary = None
        self.mo_worst_feasible_mhc = None
        self._last_transition_meta = {}
        self.mo_trace_interval = int(max(1, int(os.getenv("ELP_MO_TRACE_INTERVAL", "1000"))))
        self.agent_mode = None

        facility_count = int(getattr(base_env, "n", len(getattr(base_env, "areas", [])) or 0))
        self.rel_matrix, self.dist_req_matrix = MO_DataGenerator.load_or_generate_data(
            facility_count,
            instance_name=self.instance_name,
        )

        super().__init__(env=env, gbest=gbest, T=T, G=G, t_max=t_max, k=k)
        self._reset_mo_logging_state()
        self.temperature_floor_samples = 16
        self.action_labels[10] = "fast_segment_insert"
        if 10 in self.action_telemetry:
            self.action_telemetry[10]["name"] = "fast_segment_insert"

    @staticmethod
    def _copy_bay_structure(bay_structure):
        return [
            [
                int(facility)
                for facility in (
                    current_bay.tolist() if isinstance(current_bay, np.ndarray) else current_bay
                )
            ]
            for current_bay in bay_structure
        ]

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _make_trace_bucket():
        return {
            "steps": 0,
            "accepted": 0,
            "rejected": 0,
            "candidateDominatesAccepts": 0,
            "candidateDominatedRejects": 0,
            "nondominatedAccepts": 0,
            "nondominatedRejects": 0,
            "archiveChanges": 0,
            "representativeChanges": 0,
            "rewardSum": 0.0,
            "rewardCount": 0,
            "lossSum": 0.0,
            "lossCount": 0,
            "acceptProbabilitySum": 0.0,
            "acceptProbabilityCount": 0,
            "localSearchTriggers": 0,
            "localSearchImprovements": 0,
            "diversifyAttempts": 0,
            "diversifyAccepts": 0,
            "reheatTriggers": 0,
            "actionSelected": {},
            "actionAccepted": {},
            "actionImproved": {},
        }

    def _reset_mo_logging_state(self):
        self.mo_recorder = None
        self.mo_run_summary = None
        self._run_start_time = None
        self._trace_global_step = 0
        self._trace_episode_index = -1
        self._trace_step_index = -1
        self._current_action_context = {}
        self._last_archive_observation = {"archive_changed": False, "rep_changed": False}
        self._last_loss_value = None
        self._mo_total_counters = self._make_trace_bucket()
        self._mo_window_counters = self._make_trace_bucket()

    @staticmethod
    def _bump_named_counter(counter_map, key, amount=1):
        key = str(key)
        counter_map[key] = int(counter_map.get(key, 0)) + int(amount)

    def _action_counter_key(self, action_idx):
        if action_idx is None:
            return "unknown"
        action_idx = int(action_idx)
        return f"{action_idx}:{self.action_labels.get(action_idx, str(action_idx))}"

    def _set_action_context(self, phase, action_idx=None, action_name=None):
        if action_name is None and action_idx is not None:
            action_name = self.action_labels.get(int(action_idx), str(action_idx))
        self._current_action_context = {
            "phase": str(phase),
            "action_idx": None if action_idx is None else int(action_idx),
            "action_name": str(action_name or phase),
        }

    def _clear_action_context(self):
        self._current_action_context = {}

    def _record_action_selection(self, action_idx, previous_cost, next_cost, global_best=False, phase="main"):
        delta = super()._record_action_selection(action_idx, previous_cost, next_cost, global_best=global_best, phase=phase)
        self._set_action_context(phase=phase, action_idx=action_idx)
        action_key = self._action_counter_key(action_idx)
        self._bump_named_counter(self._mo_total_counters["actionSelected"], action_key)
        self._bump_named_counter(self._mo_window_counters["actionSelected"], action_key)
        return delta

    def _record_action_acceptance(self, action_idx, previous_cost, next_cost, improved=False, phase="main"):
        super()._record_action_acceptance(action_idx, previous_cost, next_cost, improved=improved, phase=phase)
        action_key = self._action_counter_key(action_idx)
        self._bump_named_counter(self._mo_total_counters["actionAccepted"], action_key)
        self._bump_named_counter(self._mo_window_counters["actionAccepted"], action_key)
        if improved:
            self._bump_named_counter(self._mo_total_counters["actionImproved"], action_key)
            self._bump_named_counter(self._mo_window_counters["actionImproved"], action_key)

    def _record_mo_event(self, event_type, **fields):
        if getattr(self, "mo_recorder", None) is None:
            return
        elapsed_seconds = None
        if getattr(self, "_run_start_time", None) is not None:
            elapsed_seconds = (datetime.datetime.now() - self._run_start_time).total_seconds()
        context = dict(getattr(self, "_current_action_context", {}) or {})
        payload = {
            "globalStep": int(getattr(self, "_trace_global_step", 0) or 0),
            "episode": None if getattr(self, "_trace_episode_index", -1) < 0 else int(self._trace_episode_index) + 1,
            "stepInEpisode": None if getattr(self, "_trace_step_index", -1) < 0 else int(self._trace_step_index) + 1,
            "elapsedSeconds": self._safe_float(elapsed_seconds),
            "phase": context.get("phase"),
            "actionIdx": context.get("action_idx"),
            "actionName": context.get("action_name"),
            "temperature": self._safe_float(getattr(self, "T", None)),
            "temperatureMin": self._safe_float(getattr(self, "T_min", None)),
            "archiveSize": len(getattr(self, "pareto_archive", []) or []),
            "representativeDecisionScore": self._safe_float(getattr(self, "representative_decision_score", None)),
        }
        payload.update(fields)
        self.mo_recorder.record_event(event_type, payload)

    def _ensure_mo_logging_state(self):
        if not hasattr(self, "_mo_total_counters") or not hasattr(self, "_mo_window_counters"):
            self._reset_mo_logging_state()

    def _increment_archive_counters(self, archive_changed=False, rep_changed=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["archiveChanges"] += int(bool(archive_changed))
            bucket["representativeChanges"] += int(bool(rep_changed))

    def _note_local_search(self, improved=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["localSearchTriggers"] += 1
            if improved:
                bucket["localSearchImprovements"] += 1

    def _note_diversification(self, accepted=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["diversifyAttempts"] += 1
            if accepted:
                bucket["diversifyAccepts"] += 1

    def _note_reheat(self):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["reheatTriggers"] += 1

    def _update_transition_counters(self, accept, probability, reward, loss_value):
        self._ensure_mo_logging_state()
        comparison = int((getattr(self, "_last_transition_meta", {}) or {}).get("comparison", 0))
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["steps"] += 1
            bucket["accepted"] += int(bool(accept))
            bucket["rejected"] += int(not accept)
            bucket["acceptProbabilitySum"] += float(probability)
            bucket["acceptProbabilityCount"] += 1
            bucket["rewardSum"] += float(reward)
            bucket["rewardCount"] += 1
            if loss_value is not None and np.isfinite(loss_value):
                bucket["lossSum"] += float(loss_value)
                bucket["lossCount"] += 1
            if comparison < 0 and accept:
                bucket["candidateDominatesAccepts"] += 1
            elif comparison > 0 and not accept:
                bucket["candidateDominatedRejects"] += 1
            elif comparison == 0 and accept:
                bucket["nondominatedAccepts"] += 1
            elif comparison == 0 and not accept:
                bucket["nondominatedRejects"] += 1

    def _build_bucket_snapshot(self, bucket):
        reward_count = int(bucket.get("rewardCount", 0) or 0)
        loss_count = int(bucket.get("lossCount", 0) or 0)
        prob_count = int(bucket.get("acceptProbabilityCount", 0) or 0)
        return {
            "steps": int(bucket.get("steps", 0) or 0),
            "accepted": int(bucket.get("accepted", 0) or 0),
            "rejected": int(bucket.get("rejected", 0) or 0),
            "candidateDominatesAccepts": int(bucket.get("candidateDominatesAccepts", 0) or 0),
            "candidateDominatedRejects": int(bucket.get("candidateDominatedRejects", 0) or 0),
            "nondominatedAccepts": int(bucket.get("nondominatedAccepts", 0) or 0),
            "nondominatedRejects": int(bucket.get("nondominatedRejects", 0) or 0),
            "archiveChanges": int(bucket.get("archiveChanges", 0) or 0),
            "representativeChanges": int(bucket.get("representativeChanges", 0) or 0),
            "localSearchTriggers": int(bucket.get("localSearchTriggers", 0) or 0),
            "localSearchImprovements": int(bucket.get("localSearchImprovements", 0) or 0),
            "diversifyAttempts": int(bucket.get("diversifyAttempts", 0) or 0),
            "diversifyAccepts": int(bucket.get("diversifyAccepts", 0) or 0),
            "reheatTriggers": int(bucket.get("reheatTriggers", 0) or 0),
            "meanReward": None if reward_count == 0 else float(bucket["rewardSum"] / reward_count),
            "meanLoss": None if loss_count == 0 else float(bucket["lossSum"] / loss_count),
            "meanAcceptProbability": None if prob_count == 0 else float(bucket["acceptProbabilitySum"] / prob_count),
            "actionSelected": dict(bucket.get("actionSelected", {}) or {}),
            "actionAccepted": dict(bucket.get("actionAccepted", {}) or {}),
            "actionImproved": dict(bucket.get("actionImproved", {}) or {}),
        }

    def _solution_trace_fields(self, solution, prefix):
        if solution is None:
            return {
                f"{prefix}Feasible": None,
                f"{prefix}SearchEnergy": None,
                f"{prefix}DecisionScore": None,
                f"{prefix}ProxyEnergy": None,
                f"{prefix}Mhc": None,
                f"{prefix}Cr": None,
                f"{prefix}Dr": None,
                f"{prefix}Ar": None,
                f"{prefix}DInf": None,
                f"{prefix}ConstraintViolation": None,
            }
        return {
            f"{prefix}Feasible": bool(getattr(solution, "current_is_feasible", False)),
            f"{prefix}SearchEnergy": self._safe_float(getattr(solution, "fitness", None)),
            f"{prefix}DecisionScore": self._safe_float(getattr(solution, "decision_score", None)),
            f"{prefix}ProxyEnergy": self._safe_float(getattr(solution, "proxy_energy", None)),
            f"{prefix}Mhc": self._safe_float(getattr(solution, "MHC", None)),
            f"{prefix}Cr": self._safe_float(getattr(solution, "CR", None)),
            f"{prefix}Dr": self._safe_float(getattr(solution, "DR", None)),
            f"{prefix}Ar": self._safe_float(getattr(solution, "AR", None)),
            f"{prefix}DInf": int(getattr(solution, "current_d_inf", 0) or 0),
            f"{prefix}ConstraintViolation": self._safe_float(getattr(solution, "constraint_violation", None)),
        }

    def _record_trace_snapshot(self, global_step, total_steps, episode_idx, step_idx, agent):
        if getattr(self, "mo_recorder", None) is None:
            return
        if not self.mo_recorder.should_record_trace(global_step, total_steps=total_steps):
            return
        elapsed_seconds = None
        if getattr(self, "_run_start_time", None) is not None:
            elapsed_seconds = (datetime.datetime.now() - self._run_start_time).total_seconds()
        payload = {
            "globalStep": int(global_step),
            "episode": int(episode_idx) + 1,
            "stepInEpisode": int(step_idx) + 1,
            "progressRatio": float(global_step / max(1, total_steps)),
            "elapsedSeconds": self._safe_float(elapsed_seconds),
            "temperature": self._safe_float(getattr(self, "T", None)),
            "temperatureMin": self._safe_float(getattr(self, "T_min", None)),
            "acceptRateWindow": self._safe_float(getattr(self, "accept_rate_window", None)),
            "noImproveSteps": int(getattr(self, "no_improve_steps", 0) or 0),
            "feasibleSolutionCount": int(getattr(self, "feasible_solution_count", 0) or 0),
            "gbestUpdateCount": int(getattr(self, "gbest_update_count", 0) or 0),
            "diversificationCount": int(getattr(self, "diversification_count", 0) or 0),
            "reheatTriggerCount": int(getattr(self, "reheat_trigger_count", 0) or 0),
            "eliteTriggerCount": int(getattr(self, "elite_trigger_count", 0) or 0),
            "eliteImprovementCount": int(getattr(self, "elite_improvement_count", 0) or 0),
            "archiveSize": len(getattr(self, "pareto_archive", []) or []),
            "epsilon": self._safe_float(getattr(agent, "epsilon", None)),
            "latestLoss": self._safe_float(getattr(self, "_last_loss_value", None)),
            "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
            "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
            "window": self._build_bucket_snapshot(self._mo_window_counters),
            "cumulative": self._build_bucket_snapshot(self._mo_total_counters),
        }
        payload.update(self._solution_trace_fields(getattr(self, "s", None), "current"))
        payload.update(self._solution_trace_fields(getattr(self, "representative_solution", None), "representative"))
        self.mo_recorder.record_trace(payload)
        self._mo_window_counters = self._make_trace_bucket()

    def _build_action_stats_payload(self, agent, total_iter):
        return {
            "meta": {
                "runId": None if getattr(self, "mo_recorder", None) is None else self.mo_recorder.run_id,
                "instance": self.instance_name,
                "algorithm": "ELP_DRL_MO",
                "agentMode": getattr(self, "agent_mode", None),
                "iterations": int(total_iter),
            },
            "overall": {
                **self._build_bucket_snapshot(self._mo_total_counters),
                "feasibleSolutionCount": int(getattr(self, "feasible_solution_count", 0) or 0),
                "gbestUpdateCount": int(getattr(self, "gbest_update_count", 0) or 0),
                "archiveSize": len(getattr(self, "pareto_archive", []) or []),
                "reheatTriggerCount": int(getattr(self, "reheat_trigger_count", 0) or 0),
                "diversificationCount": int(getattr(self, "diversification_count", 0) or 0),
                "eliteTriggerCount": int(getattr(self, "elite_trigger_count", 0) or 0),
                "eliteImprovementCount": int(getattr(self, "elite_improvement_count", 0) or 0),
                "epsilonEnd": self._safe_float(getattr(agent, "epsilon", None)),
                "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
                "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
                "representativeDecisionScore": self._safe_float(getattr(self, "representative_decision_score", None)),
            },
            "actions": self.get_action_telemetry(),
        }

    def _light_clone_solution(self, solution):
        clone = copy.copy(solution)
        for key, value in solution.__dict__.items():
            setattr(clone, key, value)
        clone.fbs_model = FBSModel(solution.fbs_model.permutation, solution.fbs_model.bay)
        return clone

    def _prepare_light_clone_with_encoding(self, solution, permutation, bay):
        clone = self._light_clone_solution(solution)
        clone.fbs_model.permutation = np.asarray(permutation, dtype=int).tolist()
        clone.fbs_model.bay = np.asarray(bay, dtype=int).tolist()
        return clone

    def _restore_solution_snapshot(self, solution, snapshot):
        for key, value in snapshot.__dict__.items():
            setattr(solution, key, value)
        solution.fbs_model = FBSModel(snapshot.fbs_model.permutation, snapshot.fbs_model.bay)
        return solution

    def _candidate_from_light_clone(self, solution, action_idx):
        candidate = self._light_clone_solution(solution)
        recipe = self.action_recipes[action_idx]
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

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
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
            dist_req_matrix=self.dist_req_matrix,
            aspect_limits=metrics["aspect_limits"],
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        constraint_violation = MO_FBSUtil.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        search_energy = MO_FBSUtil.search_energy(
            objectives_min,
            is_feasible=bool(metrics["is_feasible"]),
            d_inf=int(metrics["d_inf"]),
            total_violation=constraint_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        return (
            float(search_energy),
            int(metrics["d_inf"]),
            float(constraint_violation),
            float(metrics["mhc"]),
            np.asarray(permutation, dtype=int),
            np.asarray(bay, dtype=int),
        )

    def _apply_fast_segment_insert(self, solution):
        permutation = np.asarray(solution.fbs_model.permutation, dtype=int).copy()
        bay = np.asarray(solution.fbs_model.bay, dtype=int).copy()
        bay_structure = self._copy_bay_structure(FBSUtil.permutationToArray(permutation, bay))
        if not bay_structure:
            return False

        candidate_pool = []
        seen = set()
        max_candidates = int(max(1, self.fast_segment_insert_budget))
        max_attempts = max_candidates * 6
        eligible_bays = [
            bay_idx
            for bay_idx, current_bay in enumerate(bay_structure)
            if any(len(current_bay) > segment_length for segment_length in self.fast_segment_insert_segment_lengths)
        ]
        if not eligible_bays:
            return False

        attempts = 0
        while len(candidate_pool) < max_candidates and attempts < max_attempts:
            attempts += 1
            bay_idx = int(np.random.choice(eligible_bays))
            current_bay = bay_structure[bay_idx]
            feasible_lengths = [
                segment_length
                for segment_length in self.fast_segment_insert_segment_lengths
                if len(current_bay) > segment_length
            ]
            if not feasible_lengths:
                continue

            segment_length = int(np.random.choice(feasible_lengths))
            start_idx = int(np.random.randint(0, len(current_bay) - segment_length + 1))
            segment = current_bay[start_idx : start_idx + segment_length]
            remaining = current_bay[:start_idx] + current_bay[start_idx + segment_length :]
            insert_choices = [insert_idx for insert_idx in range(len(remaining) + 1) if insert_idx != start_idx]
            if not insert_choices:
                continue

            insert_idx = int(np.random.choice(insert_choices))
            candidate_bay = remaining[:insert_idx] + segment + remaining[insert_idx:]
            if candidate_bay == current_bay:
                continue

            candidate_structure = self._copy_bay_structure(bay_structure)
            candidate_structure[bay_idx] = candidate_bay
            candidate_perm, candidate_bay_flags = FBSUtil.arrayToPermutation(candidate_structure)
            candidate_key = (
                tuple(np.asarray(candidate_perm, dtype=int).tolist()),
                tuple(np.asarray(candidate_bay_flags, dtype=int).tolist()),
            )
            if candidate_key in seen:
                continue

            seen.add(candidate_key)
            candidate_pool.append(self._score_candidate_encoding(candidate_perm, candidate_bay_flags, solution))

        if not candidate_pool:
            return False

        candidate_pool.sort(key=lambda item: item[:4])
        best_perm = candidate_pool[0][4]
        best_bay = candidate_pool[0][5]
        solution.fbs_model.permutation = best_perm.tolist()
        solution.fbs_model.bay = best_bay.tolist()
        return True

    def _apply_recipe(self, solution, recipe):
        layout_dirty = False
        for primitive_action in recipe:
            if primitive_action == 3 and layout_dirty:
                self._evaluate_solution(solution)
                layout_dirty = False
            if primitive_action == 10:
                layout_dirty = self._apply_fast_segment_insert(solution) or layout_dirty
                continue
            solution._apply_action(solution.actions[primitive_action])
            layout_dirty = True

    def generate_candidate_by_action(self, solution, action_idx):
        return self._candidate_from_light_clone(solution, action_idx)

    def _generate_candidate_by_recipe(self, solution, recipe):
        candidate = self._light_clone_solution(solution)
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

    def _greedy_local_search(self, solution):
        current_cost = float(getattr(solution, "fitness", np.inf))
        accepted_snapshot = self._light_clone_solution(solution)
        improved = True

        while improved:
            improved = False

            bay_structure = FBSUtil.permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )
            n_bays = len(bay_structure)

            for i in range(n_bays - 1):
                bay_structure[i], bay_structure[i + 1] = bay_structure[i + 1], bay_structure[i]
                new_perm, new_bay = FBSUtil.arrayToPermutation(bay_structure)
                solution.fbs_model.permutation = np.asarray(new_perm, dtype=int).tolist()
                solution.fbs_model.bay = np.asarray(new_bay, dtype=int).tolist()
                self._evaluate_solution(solution)

                if (
                    solution.current_is_feasible
                    and np.isfinite(solution.fitness)
                    and solution.fitness < current_cost
                ):
                    current_cost = float(solution.fitness)
                    self._observe_feasible_state(solution)
                    accepted_snapshot = self._light_clone_solution(solution)
                    improved = True
                else:
                    bay_structure[i], bay_structure[i + 1] = bay_structure[i + 1], bay_structure[i]
                    self._restore_solution_snapshot(solution, accepted_snapshot)

            bay_structure = FBSUtil.permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )

            for bay_idx, bay in enumerate(bay_structure):
                bay = list(bay)
                n_fac = len(bay)
                if n_fac < 2:
                    continue

                for j in range(n_fac - 1):
                    new_bay_structure = self._copy_bay_structure(bay_structure)
                    new_bay_structure[bay_idx][j], new_bay_structure[bay_idx][j + 1] = (
                        new_bay_structure[bay_idx][j + 1],
                        new_bay_structure[bay_idx][j],
                    )
                    new_perm, new_bay_arr = FBSUtil.arrayToPermutation(new_bay_structure)
                    solution.fbs_model.permutation = np.asarray(new_perm, dtype=int).tolist()
                    solution.fbs_model.bay = np.asarray(new_bay_arr, dtype=int).tolist()
                    self._evaluate_solution(solution)

                    if (
                        solution.current_is_feasible
                        and np.isfinite(solution.fitness)
                        and solution.fitness < current_cost
                    ):
                        current_cost = float(solution.fitness)
                        self._observe_feasible_state(solution)
                        accepted_snapshot = self._light_clone_solution(solution)
                        bay_structure = FBSUtil.permutationToArray(
                            solution.fbs_model.permutation,
                            solution.fbs_model.bay,
                        )
                        bay = list(bay_structure[bay_idx]) if bay_idx < len(bay_structure) else []
                        n_fac = len(bay)
                        improved = True
                    else:
                        self._restore_solution_snapshot(solution, accepted_snapshot)

        return solution

    def _refresh_solution_search_metrics(self, solution):
        objectives_min = getattr(solution, "mo_objectives_min", None)
        if objectives_min is None:
            return float(getattr(solution, "fitness", math.inf))

        proxy_energy = MO_FBSUtil.surrogate_energy(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        decision_score = MO_FBSUtil.decision_score(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        total_violation = float(getattr(solution, "constraint_violation", 0.0) or 0.0)
        search_energy = MO_FBSUtil.search_energy(
            objectives_min,
            is_feasible=bool(getattr(solution, "current_is_feasible", False)),
            d_inf=int(getattr(solution, "current_d_inf", 0) or 0),
            total_violation=total_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
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
        self.mo_ideal, self.mo_nadir = MO_FBSUtil.compute_ideal_nadir(self.pareto_archive)

        for candidate in self.pareto_archive:
            self._refresh_solution_search_metrics(candidate)

        representative, decision_score, archive_index = MO_FBSUtil.select_representative_solution(
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
        solution.DR = float(metrics["dr"])
        solution.AR = float(metrics["ar"])
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
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
            dist_req_matrix=self.dist_req_matrix,
            aspect_limits=metrics["aspect_limits"],
        )
        constraint_violation = MO_FBSUtil.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        metrics.update(
            {
                "cr": float(objectives_raw[1]),
                "dr": float(objectives_raw[2]),
                "ar": float(objectives_raw[3]),
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

        updated_archive, inserted, removed = MO_FBSUtil.update_pareto_archive(
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

        feasible_cost = float(getattr(solution, "fitness", math.inf))
        self.worst_feasible_cost = (
            feasible_cost if self.worst_feasible_cost is None else max(float(self.worst_feasible_cost), feasible_cost)
        )

        rep_changed = False
        if self.representative_solution is not None:
            current_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)
            if previous_rep_vector is None:
                rep_changed = True
            elif not np.allclose(current_rep_vector, previous_rep_vector, atol=1e-9, rtol=1e-7):
                rep_changed = True
            elif float(self.representative_decision_score) < previous_rep_score - 1e-12:
                rep_changed = True
            if rep_changed:
                self.gbest_update_count += 1
                self.best_history.append(float(self.representative_decision_score))

        self._last_archive_observation = {
            "archive_changed": bool(archive_changed),
            "rep_changed": bool(rep_changed),
            "inserted": bool(inserted),
            "removedCount": int(removed or 0),
        }
        if archive_changed or rep_changed:
            self._increment_archive_counters(archive_changed=archive_changed, rep_changed=rep_changed)
        if archive_changed:
            self._record_mo_event(
                "archive_update",
                inserted=bool(inserted),
                removedCount=int(removed or 0),
                candidateDecisionScore=self._safe_float(getattr(solution, "decision_score", None)),
                candidateMhc=self._safe_float(getattr(solution, "MHC", None)),
                candidateCr=self._safe_float(getattr(solution, "CR", None)),
                candidateDr=self._safe_float(getattr(solution, "DR", None)),
                candidateAr=self._safe_float(getattr(solution, "AR", None)),
            )
        if rep_changed and self.representative_solution is not None:
            self._record_mo_event(
                "representative_update",
                representativeArchiveIndex=None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
                representativeDecisionScore=self._safe_float(self.representative_decision_score),
                representativeMhc=self._safe_float(getattr(self.representative_solution, "MHC", None)),
                representativeCr=self._safe_float(getattr(self.representative_solution, "CR", None)),
                representativeDr=self._safe_float(getattr(self.representative_solution, "DR", None)),
                representativeAr=self._safe_float(getattr(self.representative_solution, "AR", None)),
            )

        for env_obj in (solution, getattr(self, "s", None), self.gbest, self.true_gbest):
            if env_obj is None:
                continue
            env_obj.feasible_solution_count = self.feasible_solution_count
            env_obj.best_feasible_cost = self.best_feasible_cost
            env_obj.worst_feasible_cost = self.worst_feasible_cost
            env_obj.best_fitness = self.best_feasible_cost
            env_obj.current_v_worst = self.worst_feasible_cost
            if getattr(env_obj, "mo_objectives_min", None) is not None:
                self._refresh_solution_search_metrics(env_obj)
        return bool(archive_changed or rep_changed)

    def _layout_score(self, solution):
        proxy_score = float(getattr(solution, "decision_score", math.inf))
        return (
            0 if getattr(solution, "current_is_feasible", False) else 1,
            int(getattr(solution, "current_d_inf", 0)),
            float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            proxy_score,
        )

    def _sample_temperature_floor(self, n_samples=None, target_accept=None):
        n_samples = int(n_samples or self.temperature_floor_samples)
        target_accept = float(target_accept or self.temperature_floor_target_accept)
        fallback = max(self.T_initial * 0.05, 1.0)
        cap = max(self.T_initial * self.temperature_floor_cap_ratio, fallback)

        if self.best_feasible_solution is None or not np.isfinite(getattr(self.best_feasible_solution, "fitness", np.inf)):
            return float(fallback)
        if not 0.0 < target_accept < 1.0:
            target_accept = self.temperature_floor_target_accept

        base_solution = self._light_clone_solution(self.best_feasible_solution)
        allowed_actions = [
            table_idx
            for table_idx, action_idx in enumerate(self.valid_actions)
            if action_idx in self.temperature_floor_action_ids
        ]
        if not allowed_actions:
            allowed_actions = self._get_allowed_action_indices(base_solution)
        if not allowed_actions:
            return float(fallback)

        sample_budget = max(1, min(int(n_samples), len(allowed_actions)))
        sampled_action_table_indices = np.random.permutation(allowed_actions)[:sample_budget]

        current_tilde = self._tilde_energy(base_solution.fitness)
        deltas = []
        for action_table_idx in sampled_action_table_indices:
            action_idx = self.valid_actions[int(action_table_idx)]
            recipe = self.action_recipes[action_idx]
            candidate = self._generate_candidate_by_recipe(base_solution, recipe)
            candidate_tilde = self._tilde_energy(candidate.fitness)
            delta = candidate_tilde - current_tilde
            if np.isfinite(delta) and delta > 0:
                deltas.append(float(delta))

        if not deltas:
            return float(fallback)

        delta_q = float(np.percentile(deltas, self.temperature_floor_quantile))
        sampled_floor = -delta_q / math.log(target_accept)
        if not np.isfinite(sampled_floor):
            return float(fallback)
        return float(min(cap, max(sampled_floor, fallback)))

    def _proxy_gap_band(self, solution):
        if self.representative_solution is None:
            return 3
        current_score = float(getattr(solution, "decision_score", math.inf))
        best_score = float(getattr(self, "representative_decision_score", math.inf))
        if not np.isfinite(current_score) or not np.isfinite(best_score):
            return 3
        gap_ratio = max(current_score - best_score, 0.0) / max(abs(best_score), 1e-8)
        if gap_ratio <= 0.01:
            return 0
        if gap_ratio <= 0.05:
            return 1
        if gap_ratio <= 0.15:
            return 2
        return 3

    def _archive_size_band(self):
        archive_size = len(self.pareto_archive)
        if archive_size <= 1:
            return 0
        if archive_size <= 4:
            return 1
        if archive_size <= 8:
            return 2
        return 3

    def _archive_crowding_band(self, solution):
        if not getattr(solution, "current_is_feasible", False) or len(self.pareto_archive) < 2:
            return 3
        current_vector = MO_FBSUtil.normalize_objective_vector(
            getattr(solution, "mo_objectives_min", None),
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if current_vector is None:
            return 3

        distances = []
        for candidate in self.pareto_archive:
            candidate_vector = MO_FBSUtil.normalize_objective_vector(
                getattr(candidate, "mo_objectives_min", None),
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
            )
            if candidate_vector is None:
                continue
            distance = float(np.linalg.norm(current_vector - candidate_vector, ord=2))
            if distance > 1e-12:
                distances.append(distance)
        if not distances:
            return 0
        nearest = min(distances)
        if nearest <= 0.05:
            return 0
        if nearest <= 0.15:
            return 1
        if nearest <= 0.30:
            return 2
        return 3

    def state_encoder(self, solution):
        d_band = self._d_inf_band(solution.current_d_inf)
        t_band = self._temperature_band()
        proxy_band = self._proxy_gap_band(solution)
        archive_band = self._archive_size_band()
        stagnation_band = self._stagnation_band()
        hist_band = self._histogram_band(solution.fitness)
        crowding_band = self._archive_crowding_band(solution)
        state = d_band
        for band in (t_band, proxy_band, archive_band, stagnation_band, hist_band, crowding_band):
            state = state * 4 + band
        return state

    def _accept_candidate_with_context(self, current_solution, candidate_solution):
        comparison = MO_FBSUtil.compare_solution_quality(candidate_solution, current_solution)
        current_tilde = self._tilde_energy(current_solution.fitness)
        candidate_tilde = self._tilde_energy(candidate_solution.fitness)
        archive_preview, archive_would_change, _ = MO_FBSUtil.update_pareto_archive(
            self.pareto_archive,
            candidate_solution,
            max_size=self.archive_limit,
            clone_fn=lambda item: item,
        )
        _ = archive_preview

        if comparison < 0:
            accept = True
            probability = 1.0
        elif comparison > 0:
            accept = False
            probability = 0.0
        else:
            exponent = (current_tilde - candidate_tilde) / max(self.T, 1e-12)
            exponent = max(min(exponent, 700.0), -700.0)
            probability = math.exp(exponent)
            accept = bool(np.random.rand() < probability)

        self._last_transition_meta = {
            "comparison": comparison,
            "archive_would_change": bool(archive_would_change),
            "current_proxy": float(getattr(current_solution, "proxy_energy", current_solution.fitness)),
            "candidate_proxy": float(getattr(candidate_solution, "proxy_energy", candidate_solution.fitness)),
            "current_violation": float(getattr(current_solution, "constraint_violation", 0.0) or 0.0),
            "candidate_violation": float(getattr(candidate_solution, "constraint_violation", 0.0) or 0.0),
            "current_d_inf": int(getattr(current_solution, "current_d_inf", 0) or 0),
            "candidate_d_inf": int(getattr(candidate_solution, "current_d_inf", 0) or 0),
            "accepted": bool(accept),
        }
        return bool(accept), float(probability), float(current_tilde), float(candidate_tilde)

    def _accept_candidate(self, current_cost, candidate_cost):
        candidate = getattr(self, "_pending_candidate", None)
        current = getattr(self, "s", None)
        if candidate is None or current is None:
            return super()._accept_candidate(current_cost, candidate_cost)
        return self._accept_candidate_with_context(current, candidate)

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        meta = dict(getattr(self, "_last_transition_meta", {}) or {})
        reward = 0.0
        comparison = int(meta.get("comparison", 0))
        if comparison < 0:
            reward += 1.2
        elif comparison > 0:
            reward -= 0.8
        else:
            proxy_delta = float(meta.get("current_proxy", previous_cost) - meta.get("candidate_proxy", next_cost))
            reward += float(np.clip(proxy_delta, -1.0, 1.0))

        if accept and meta.get("archive_would_change", False):
            reward += 1.0
        elif (not accept) and meta.get("archive_would_change", False):
            reward -= 0.15

        reward += 0.30 * (float(previous_d_inf) - float(next_d_inf))
        reward += 0.10 * float(
            np.clip(meta.get("current_violation", 0.0) - meta.get("candidate_violation", 0.0), -5.0, 5.0)
        )
        if previous_d_inf > 0 and next_d_inf == 0:
            reward += 0.50
        if accept and np.isfinite(next_cost) and np.isfinite(previous_best_feasible) and next_cost < previous_best_feasible:
            reward += 0.30
        if not accept:
            reward -= 0.15
        return float(np.clip(reward, -3.0, 3.0))

    def _current_best_energy_for_logging(self):
        if np.isfinite(self.representative_decision_score):
            return float(self.representative_decision_score)
        return super()._current_best_energy_for_logging()

    def _attempt_diversification(self, global_step, force=False):
        if (not force) and (global_step - self.last_diversify_step < max(10, self.t_max // 4)):
            return
        previous_context = dict(getattr(self, "_current_action_context", {}) or {})
        self._set_action_context("diversify", action_name="diversify_recipe")
        candidate = copy.deepcopy(self.s)
        self._apply_recipe(candidate, self.diversify_recipe)
        self._evaluate_solution(candidate)
        self._pending_candidate = candidate
        accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
        self._pending_candidate = None
        self.prob_history.append(prob)
        if accept:
            previous_best = self.best_feasible_cost
            previous_cost = self.s.fitness
            previous_d_inf = self.s.current_d_inf
            self.s = candidate
            self.current_energy = self.s.fitness
            archive_improved = self._observe_feasible_state(self.s)
            improved = bool(archive_improved)
            reward = self._compute_transition_reward(
                previous_cost,
                self.s.fitness,
                previous_d_inf,
                self.s.current_d_inf,
                previous_best,
                accept=True,
            )
            if improved:
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
        else:
            improved = False
            self.no_improve_steps += 1
            reward = -0.2
        self._note_diversification(accepted=accept)
        self._record_mo_event(
            "diversification",
            forced=bool(force),
            accepted=bool(accept),
            improved=bool(improved),
            probability=self._safe_float(prob),
            reward=self._safe_float(reward),
        )
        if accept:
            self._update_histogram(self.s.fitness)
        else:
            self._update_histogram(self.current_energy)
        self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
        self.energy_history.append(self.s.fitness)
        self.last_diversify_step = global_step
        self.diversification_count += 1
        self.T = max(self.T, self.T_min)
        self._current_action_context = previous_context
        return reward

    def _attempt_reheating(self, global_step, total_steps, fast_time):
        if not self.reheat_enabled:
            return fast_time, False
        if self.reheat_episode_count >= self.reheat_max_per_episode:
            return fast_time, False
        progress_ratio = float(global_step) / float(max(1, total_steps))
        if progress_ratio > self.reheat_progress_cap_ratio:
            return fast_time, False
        if self.no_improve_steps < self.reheat_no_improve_threshold:
            return fast_time, False
        if self.accept_rate_window > self.reheat_accept_rate_threshold:
            return fast_time, False
        if self.T >= self.reheat_temp_gate_ratio * self.T_initial:
            return fast_time, False
        if global_step - self.last_reheat_step < self.reheat_cooldown_steps:
            return fast_time, False

        previous_temperature = float(self.T)
        target_low = self.reheat_target_low_ratio * self.T_initial
        target_high = self.reheat_target_high_ratio * self.T_initial
        self.T = min(target_high, max(self.T, target_low))

        no_improve_before = self.no_improve_steps
        self.last_reheat_step = global_step
        self.reheat_episode_count += 1
        self.reheat_trigger_count += 1
        self.no_improve_steps = 0
        self._note_reheat()
        self._record_mo_event(
            "reheat_triggered",
            previousTemperature=self._safe_float(previous_temperature),
            newTemperature=self._safe_float(self.T),
            noImproveBefore=int(no_improve_before),
            acceptRateWindow=self._safe_float(self.accept_rate_window),
            progressRatio=float(progress_ratio),
        )

        self._attempt_diversification(global_step, force=True)
        fast_time, _ = self._elite_intensification(fast_time)
        if self.enable_reheat_logging:
            logger.info(
                f"Reheat triggered | step: {global_step} | T: {previous_temperature:.6f} -> {self.T:.6f} | "
                f"accept_rate_window: {self.accept_rate_window:.3f} | no_improve: {no_improve_before}"
            )
        return fast_time, True


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
            "dr": float(getattr(solution, "DR", 0.0)),
            "ar": float(getattr(solution, "AR", 0.0)),
            "dInf": int(getattr(solution, "current_d_inf", 0) or 0),
            "constraintViolation": float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            "isFeasible": bool(getattr(solution, "current_is_feasible", False)),
            "moObjectivesRaw": np.asarray(getattr(solution, "mo_objectives_raw", []), dtype=float).tolist(),
            "moObjectivesMin": np.asarray(getattr(solution, "mo_objectives_min", []), dtype=float).tolist(),
            "solution": solution_array,
        }

    def _save_pareto_archive(self, start_time):
        if not self.pareto_archive:
            self.pareto_archive_path = None
            return None

        repo_root = Path(__file__).resolve().parents[2]
        archive_dir = Path(config.RESULT_PATH) / "pareto_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = start_time.strftime("%Y%m%d_%H%M%S_%f") if start_time is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.instance_name}-ELP_DRL_MO-{timestamp}.json"
        archive_path = archive_dir / filename

        payload = {
            "instance": self.instance_name,
            "algorithm": "ELP_DRL_MO",
            "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
            "archiveSize": len(self.pareto_archive),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "representativeDecisionScore": None if not np.isfinite(self.representative_decision_score) else float(self.representative_decision_score),
            "items": [self._archive_item_payload(solution, index + 1) for index, solution in enumerate(self.pareto_archive)],
        }
        archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.pareto_archive_path = archive_path.resolve().relative_to(repo_root).as_posix()
        return self.pareto_archive_path

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        self._reset_mo_logging_state()
        self._run_start_time = start_time

        agent_mode = os.getenv("ELP_RL_AGENT", "dqn").strip().lower()
        self.agent_mode = agent_mode
        run_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_MO")
        run_remark = os.getenv("ELP_EXP_REMARK", "")
        self.mo_recorder = MO_ExperimentsUtil.MOExperimentRecorder(
            instance=self.instance_name,
            algorithm=run_algorithm,
            start_time=start_time,
            trace_interval=self.mo_trace_interval,
            remark=run_remark,
            result_root=config.RESULT_PATH,
        )

        if agent_mode == "qlearning":
            agent = StandardQLearningAgent(s_dim=16384, a_dim=len(self.valid_actions))
        else:
            agent = StandardDQNAgent(
                s_dim=16384,
                a_dim=len(self.valid_actions),
                epsilon=0.50,
                epsilon_min=self.dqn_epsilon_min,
                epsilon_decay=self.dqn_epsilon_decay,
                gamma=0.95,
                lr=3e-4,
                batch_size=64,
                replay_capacity=50000,
                warmup_steps=self.dqn_warmup_steps,
                update_every=4,
                target_update_every=self.dqn_target_update_every,
                idqn_k=2,
                embedding_dim=32,
                hidden_dim=64,
                grad_clip=10.0,
            )
        if not self._bootstrap_until_first_feasible():
            logger.warning("Failed to seed a feasible archive before ELP search.")
        if np.isfinite(self.best_feasible_cost):
            self.best_energy = self.best_feasible_cost
            self.current_energy = self.s.fitness

        global_step = 0
        total_steps = max(1, self.G * self.t_max)
        next_progress_marker_idx = 0
        for episode in range(self.G):
            if self.worst_feasible_cost is None and not self._bootstrap_until_first_feasible(max_attempts=max(200, 2 * self.t_max)):
                logger.warning(f"Episode {episode}: feasible archive still unavailable.")
                continue
            episode_best_before = self.best_feasible_cost
            self._prepare_episode_start(episode)
            for step_idx in range(self.t_max):
                self._trace_global_step = global_step
                self._trace_episode_index = episode
                self._trace_step_index = step_idx
                self.current_progress_ratio = float(global_step) / float(max(1, total_steps))
                current_state_idx = self.state_encoder(self.s)
                allowed_actions = self._get_allowed_action_indices(self.s)
                action_table_idx = agent.select_action(current_state_idx, allowed_actions=allowed_actions)
                real_action_idx = self.valid_actions[action_table_idx]
                previous_cost = self.s.fitness
                previous_d_inf = self.s.current_d_inf
                previous_best_feasible = self.best_feasible_cost

                candidate = self.generate_candidate_by_action(self.s, real_action_idx)
                self._record_action_selection(
                    real_action_idx,
                    previous_cost,
                    candidate.fitness,
                    phase="main",
                )
                self._pending_candidate = candidate
                accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
                self._pending_candidate = None
                self.prob_history.append(prob)
                self._record_acceptance(accept)
                improved = False

                if accept:
                    self.s = candidate
                    self.current_energy = self.s.fitness
                    archive_improved = self._observe_feasible_state(self.s)
                    if archive_improved:
                        self._record_action_global_best(real_action_idx, phase="main")
                    accepted_improved = bool(
                        np.isfinite(previous_cost)
                        and np.isfinite(self.s.fitness)
                        and self.s.fitness < previous_cost
                    )
                    self._record_action_acceptance(
                        real_action_idx,
                        previous_cost,
                        self.s.fitness,
                        improved=accepted_improved,
                        phase="main",
                    )
                    improved = bool(archive_improved)
                    relative_improvement = 0.0
                    if np.isfinite(previous_cost) and abs(previous_cost) > 1e-12 and np.isfinite(self.s.fitness):
                        relative_improvement = max((previous_cost - self.s.fitness) / abs(previous_cost), 0.0)
                    trigger_local_search = improved or (relative_improvement >= 0.005)
                    self._update_histogram(self.s.fitness)
                    if improved:
                        self.no_improve_steps = 0
                        fast_time = datetime.datetime.now()
                    else:
                        self.no_improve_steps += 1

                    if trigger_local_search:
                        local_search_best_before = self.best_feasible_cost
                        local_search_cost_before = self.s.fitness
                        previous_context = dict(getattr(self, "_current_action_context", {}) or {})
                        self._set_action_context("local_search", action_name="greedy_local_search")
                        self.s = self._greedy_local_search(self.s)
                        self.current_energy = self.s.fitness
                        local_search_improved = bool(
                            np.isfinite(self.best_feasible_cost)
                            and self.best_feasible_cost < local_search_best_before
                        )
                        self._note_local_search(improved=local_search_improved)
                        self._record_mo_event(
                            "local_search",
                            improved=local_search_improved,
                            beforeSearchEnergy=self._safe_float(local_search_cost_before),
                            afterSearchEnergy=self._safe_float(self.s.fitness),
                        )
                        self._current_action_context = previous_context
                        if self.s.current_is_feasible and self.s.fitness < previous_best_feasible:
                            improved = True

                else:
                    self._update_histogram(self.s.fitness)
                    self.no_improve_steps += 1

                if improved:
                    self.no_improve_steps = 0
                    fast_time = datetime.datetime.now()
                    fast_time, _ = self._elite_intensification(fast_time)

                gbest_recomputed = bool(
                    np.isfinite(self.best_feasible_cost)
                    and self.best_feasible_cost < previous_best_feasible
                )
                if gbest_recomputed:
                    self.T_min = self._sample_temperature_floor()
                    self._record_mo_event(
                        "temperature_floor_update",
                        bestDecisionScore=self._safe_float(self.best_feasible_cost),
                        temperatureMin=self._safe_float(self.T_min),
                    )

                next_state_idx = self.state_encoder(self.s)
                allowed_next_actions = self._get_allowed_action_indices(self.s)
                reward = self._compute_transition_reward(
                    previous_cost,
                    self.s.fitness,
                    previous_d_inf,
                    self.s.current_d_inf,
                    previous_best_feasible,
                    accept,
                )
                done_flag = step_idx == self.t_max - 1
                loss_value = agent.update_Q(
                    current_state_idx,
                    action_table_idx,
                    reward,
                    next_state_idx,
                    done=done_flag,
                    allowed_next_actions=allowed_next_actions,
                )
                self._last_loss_value = loss_value
                self._update_transition_counters(accept, prob, reward, loss_value)
                self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
                self.energy_history.append(self.s.fitness)
                global_step += 1
                self._trace_global_step = global_step
                while (
                    next_progress_marker_idx < len(self.progress_markers)
                    and (global_step / total_steps) >= self.progress_markers[next_progress_marker_idx]
                ):
                    self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
                    next_progress_marker_idx += 1

                self._record_trace_snapshot(global_step, total_steps, episode, step_idx, agent)
                fast_time, reheated = self._attempt_reheating(global_step, total_steps, fast_time)
                if (not reheated) and self.no_improve_steps >= self.diversify_trigger_no_improve:
                    self._attempt_diversification(global_step)
                self._clear_action_context()

            finalize_episode = getattr(agent, "finalize_episode", None)
            if callable(finalize_episode):
                finalize_episode()

            self.T = max(self.T * self.cooling_per_step, self.T_min)

            if np.isfinite(self.best_feasible_cost) and self.best_feasible_cost < episode_best_before:
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            agent.decay_epsilon()

        while next_progress_marker_idx < len(self.progress_markers):
            self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
            next_progress_marker_idx += 1

        end_time = datetime.datetime.now()
        self._record_trace_snapshot(global_step, total_steps, max(self.G - 1, 0), max(self.t_max - 1, 0), agent)
        self._refresh_archive_state()

        best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else copy.deepcopy(self.s)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)
        if self.representative_solution is not None:
            best_solution = copy.deepcopy(self.representative_solution)
            best_energy = float(self.representative_decision_score)
            is_valid = bool(getattr(best_solution, "current_is_feasible", False))

        archive_path = self._save_pareto_archive(start_time)
        solution_array = getattr(getattr(best_solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        best_result_seconds = None if fast_time is None else (fast_time - start_time).total_seconds()
        self._record_mo_event(
            "run_completed",
            totalIterations=int(global_step),
            runtimeSeconds=self._safe_float((end_time - start_time).total_seconds()),
            bestResultSeconds=self._safe_float(best_result_seconds),
            archivePath=archive_path,
        )
        action_stats = self._build_action_stats_payload(agent, global_step)
        run_summary = {
            "startTime": start_time.isoformat(),
            "fastTime": None if fast_time is None else fast_time.isoformat(),
            "endTime": end_time.isoformat(),
            "runtimeSeconds": (end_time - start_time).total_seconds(),
            "bestResultSeconds": best_result_seconds,
            "iterations": int(global_step),
            "isValid": bool(is_valid),
            "gbestUpdateCount": int(self.gbest_update_count),
            "feasibleSolutionCount": int(self.feasible_solution_count),
            "archiveSize": int(len(self.pareto_archive)),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "decisionScore": self._safe_float(best_energy),
            "repMhc": None if best_solution is None else self._safe_float(getattr(best_solution, "MHC", None)),
            "repCr": None if best_solution is None else self._safe_float(getattr(best_solution, "CR", None)),
            "repDr": None if best_solution is None else self._safe_float(getattr(best_solution, "DR", None)),
            "repAr": None if best_solution is None else self._safe_float(getattr(best_solution, "AR", None)),
            "paretoArchivePath": archive_path,
            "agentMode": agent_mode,
            "epsilonEnd": self._safe_float(getattr(agent, "epsilon", None)),
            "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
            "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
            "G": int(self.G),
            "tMax": int(self.t_max),
            "archiveLimit": int(self.archive_limit),
            "traceInterval": int(self.mo_trace_interval),
            "objectiveWeights": np.asarray(self.mo_weights, dtype=float).tolist(),
            "representativeSolution": solution_array,
        }
        self.mo_run_summary = self.mo_recorder.finalize(run_summary, action_stats)
        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
            "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
            "rep_dr": None if best_solution is None else float(getattr(best_solution, "DR", 0.0)),
            "rep_ar": None if best_solution is None else float(getattr(best_solution, "AR", 0.0)),
            "decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "mo_run_id": self.mo_run_summary.get("runId"),
            "mo_bundle_dir": self.mo_run_summary.get("bundleDir"),
            "mo_trace_path": self.mo_run_summary.get("tracePath"),
            "mo_events_path": self.mo_run_summary.get("eventsPath"),
            "mo_action_stats_path": self.mo_run_summary.get("actionStatsPath"),
            "mo_run_summary_path": self.mo_run_summary.get("runSummaryPath"),
        }
        return global_step, is_valid, best_solution, best_energy, start_time, end_time, fast_time


def _save_experiment_row(exp_instance, exp_algorithm, exp_remark, total_iter, is_valid, best_sol, best_energy, start, end, fast, solver):
    extra_fields = dict(getattr(solver, "last_run_payload", {}) or {})
    MO_ExperimentsUtil.save_legacy_mo_experiment_result(
        exp_instance=exp_instance,
        exp_algorithm=exp_algorithm,
        exp_iterations=total_iter,
        exp_solution=best_sol.fbs_model.array_2d,
        exp_fitness=best_energy,
        exp_start_time=start,
        exp_fast_time=fast,
        exp_end_time=end,
        exp_is_valid_aspect_ratio=is_valid,
        exp_remark=exp_remark,
        exp_gbest_updates=solver.gbest_update_count,
        exp_extra_fields=extra_fields,
    )


if __name__ == "__main__":
    exp_instance = "Du62"
    exp_algorithm = "ELP_DRL_MO"
    exp_remark = "WarmStart(GA)+ELP+Pareto archive with representative solution"
    exp_number = 30
    is_exp = True

    G = 1000
    t_max = 300
    T_initial = 10000.0
    k_hist = 10.0

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                env.reset()
                base_env = env.unwrapped if hasattr(env, "unwrapped") else env
                initial_gbest = copy.deepcopy(base_env)
                logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
                elp_solver = ELP(
                    env=base_env,
                    gbest=initial_gbest,
                    T=T_initial,
                    G=G,
                    t_max=t_max,
                    k=k_hist,
                )
                total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
                logger.info(f"Experiment {i + 1} complete | representative decision score: {best_energy}")
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
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
        elp_solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
        print(f"Single run complete | representative decision score: {best_energy}")
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
