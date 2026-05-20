import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchControllerPostStepResult:
    """单步搜索控制链执行结果。"""

    fast_time: object
    reheated: bool
    mid_structural_shot: bool
    archive_switched: bool
    final_elite_pushed: bool


class DQNSearchController:
    """DQN 主链路搜索控制器。"""

    def current_search_phase(self, runtime, solution=None):
        if solution is not None and getattr(solution, "current_d_inf", 0) > 0:
            return "infeasible"
        progress = min(max(float(runtime.current_progress_ratio), 0.0), 1.0)
        if progress < 0.25:
            return "early"
        if progress < 0.75:
            return "mid"
        return "late"

    def get_local_search_policy(self, runtime, solution, action_idx=None):
        phase = self.current_search_phase(runtime, solution)
        policy = {
            "phase": phase,
            "action_ids": set(runtime.local_search_trigger_action_ids),
            "trigger_no_improve": int(runtime.local_search_trigger_no_improve),
            "gap_ratio": float(runtime.local_search_trigger_gap_ratio),
            "guided_action_ids": set(runtime.topk_guided_trigger_action_ids),
            "guided_no_improve": int(runtime.topk_guided_trigger_no_improve),
            "guided_gap_ratio": float(runtime.topk_guided_trigger_gap_ratio),
            "guided_topk": int(runtime.topk_guided_topk),
            "guided_max_iters": int(runtime.topk_guided_max_iters),
        }

        if phase == "early":
            policy["action_ids"] = {11, 14, 15}
            policy["trigger_no_improve"] = max(
                policy["trigger_no_improve"],
                max(40, runtime.local_search_trigger_no_improve * 2),
            )
            policy["gap_ratio"] = min(max(policy["gap_ratio"], 1e-4), 0.02)
            policy["guided_action_ids"] = {11, 14, 15}
            policy["guided_no_improve"] = max(
                policy["guided_no_improve"],
                max(80, runtime.topk_guided_trigger_no_improve * 2),
            )
            policy["guided_gap_ratio"] = min(max(policy["guided_gap_ratio"], 1e-4), 0.008)
            policy["guided_topk"] = max(5, min(policy["guided_topk"], 6))
            policy["guided_max_iters"] = 1
        elif phase == "late":
            policy["action_ids"] = set(policy["action_ids"]) | {0, 2}
            policy["trigger_no_improve"] = max(10, runtime.local_search_trigger_no_improve // 2)
            policy["gap_ratio"] = max(policy["gap_ratio"], 0.05)
            policy["guided_action_ids"] = set(policy["guided_action_ids"]) | {0, 2}
            policy["guided_no_improve"] = max(20, runtime.topk_guided_trigger_no_improve // 2)
            policy["guided_gap_ratio"] = max(policy["guided_gap_ratio"], 0.025)
            policy["guided_topk"] = min(
                runtime.topk_guided_target_position_cap,
                max(policy["guided_topk"], runtime.topk_guided_topk + 2),
            )
            policy["guided_max_iters"] = min(max(policy["guided_max_iters"], runtime.topk_guided_max_iters + 1), 3)

        if action_idx is not None and int(action_idx) == 10 and phase != "early":
            policy["guided_topk"] = min(
                runtime.topk_guided_target_position_cap,
                max(policy["guided_topk"], runtime.topk_guided_topk + 2),
            )
        return policy

    def should_trigger_local_search(self, runtime, solution, action_idx, accepted_improved, previous_d_inf):
        if not getattr(solution, "current_is_feasible", False):
            return False
        if not np.isfinite(getattr(solution, "fitness", np.inf)):
            return False
        policy = self.get_local_search_policy(runtime, solution, action_idx=action_idx)
        if accepted_improved:
            return True
        if int(getattr(solution, "current_d_inf", 0)) < int(previous_d_inf):
            return True
        if runtime._relative_gap_to_best(solution) <= float(policy["gap_ratio"]):
            return True
        return (
            int(action_idx) in policy["action_ids"]
            and runtime.no_improve_steps >= policy["trigger_no_improve"]
        )

    def should_run_topk_guided_search(self, runtime, solution, action_idx=None):
        policy = self.get_local_search_policy(runtime, solution, action_idx=action_idx)
        guided_topk = int(policy["guided_topk"])
        guided_max_iters = int(policy["guided_max_iters"])
        if not runtime.topk_guided_enabled:
            return False, guided_topk, guided_max_iters
        if not getattr(solution, "current_is_feasible", False):
            return False, guided_topk, guided_max_iters
        if not np.isfinite(getattr(solution, "fitness", np.inf)):
            return False, guided_topk, guided_max_iters
        if runtime._relative_gap_to_best(solution) <= float(policy["guided_gap_ratio"]):
            return True, guided_topk, guided_max_iters
        if action_idx is None:
            return False, guided_topk, guided_max_iters
        return (
            int(action_idx) in policy["guided_action_ids"]
            and runtime.no_improve_steps >= policy["guided_no_improve"]
        ), guided_topk, guided_max_iters

    def accept_candidate(self, runtime, current_cost, candidate_cost):
        if np.isfinite(candidate_cost) and not np.isfinite(current_cost):
            return True, 1.0, float("inf"), float(candidate_cost)
        if np.isfinite(current_cost) and not np.isfinite(candidate_cost):
            return False, 0.0, float(current_cost), float("inf")
        if not np.isfinite(current_cost) and not np.isfinite(candidate_cost):
            return False, 0.0, float("inf"), float("inf")

        if candidate_cost < current_cost:
            return True, 1.0, current_cost, candidate_cost

        current_tilde = current_cost + runtime.k_hist * runtime._get_H_value(current_cost)
        candidate_tilde = candidate_cost + runtime.k_hist * runtime._get_H_value(candidate_cost)
        exponent = (current_tilde - candidate_tilde) / runtime._effective_acceptance_temperature()
        exponent = max(min(exponent, 700.0), -700.0)
        prob = math.exp(exponent)
        return bool(np.random.rand() < prob), prob, current_tilde, candidate_tilde

    def run_post_step_controls(self, runtime, global_step, total_steps, fast_time):
        fast_time, reheated = runtime._attempt_reheating(global_step, total_steps, fast_time)
        mid_structural_shot = False
        if not reheated:
            fast_time, mid_structural_shot = runtime._attempt_mid_structural_shot(global_step, total_steps, fast_time)
        archive_switched = False
        if (not reheated) and (not mid_structural_shot):
            archive_switched = runtime._attempt_archive_switch(global_step)
        final_elite_pushed = False
        if (not reheated) and (not mid_structural_shot) and (not archive_switched):
            fast_time, final_elite_pushed = runtime._attempt_final_elite_push(global_step, total_steps, fast_time)
        if (
            (not reheated)
            and (not mid_structural_shot)
            and (not archive_switched)
            and (not final_elite_pushed)
            and runtime.no_improve_steps >= runtime.diversify_trigger_no_improve
        ):
            runtime._attempt_diversification(global_step)
        return SearchControllerPostStepResult(
            fast_time=fast_time,
            reheated=bool(reheated),
            mid_structural_shot=bool(mid_structural_shot),
            archive_switched=bool(archive_switched),
            final_elite_pushed=bool(final_elite_pushed),
        )
