class DQNOperatorDispatcher:
    """DQN 主链路动作分发器。"""

    def current_phase_action_ids(self, runtime, solution):
        if getattr(solution, "current_d_inf", 0) > 0:
            return list(runtime.infeasible_phase_action_ids["repair"])
        progress = min(max(float(runtime.current_progress_ratio), 0.0), 1.0)
        if progress < 0.35:
            return list(runtime.phase_action_ids["early"])
        if progress < 0.75:
            return list(runtime.phase_action_ids["mid"])
        return list(runtime.phase_action_ids["late"])

    def get_allowed_action_indices(self, runtime, solution):
        current_action_ids = runtime._current_action_ids(solution)
        phase_actions = set(self.current_phase_action_ids(runtime, solution))
        allowed = [
            local_idx
            for local_idx, action_idx in enumerate(current_action_ids)
            if action_idx in phase_actions
        ]
        return allowed if allowed else list(range(len(current_action_ids)))

    def generate_candidate_by_action_fallback(self, runtime, solution, action_idx, phase="main"):
        action_idx = int(action_idx)
        if phase == "main" and action_idx == 10 and runtime.segment_insert_light_enabled:
            return runtime._generate_segment_insert_light_candidate(solution)
        return runtime._generate_candidate_by_recipe(solution, runtime.action_recipes[action_idx])

    def generate_candidate_by_action(self, runtime, solution, action_idx, phase="main"):
        action_idx = int(action_idx)
        if (
            phase == "main"
            and runtime.two_stage_heavy_actions_enabled
            and action_idx in runtime.two_stage_heavy_action_ids
        ):
            candidate = runtime._generate_two_stage_heavy_candidate(
                solution,
                action_idx,
                enable_local_proxy=False,
            )
            if candidate is not None:
                return candidate
        return self.generate_candidate_by_action_fallback(runtime, solution, action_idx, phase=phase)
