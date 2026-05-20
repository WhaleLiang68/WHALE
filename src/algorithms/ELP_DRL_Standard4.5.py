import copy
import datetime
import math
import os
import random
from collections import deque
from src.utils.FBSUtil import permutationToArray, arrayToPermutation
import gym
import numpy as np
from loguru import logger

try:
    import torch
except Exception:
    torch = None

import src
import src.utils.ExperimentsUtil as ExperimentsUtil
import src.utils.FBSUtil as FBSUtil
import src.utils.config as runtime_config
from src.algorithms.DQN import (
    DQNOperatorDispatcher,
    DQNProgramRunner,
    DQNRewardEngine,
    DQNSearchController,
    DQNStateEncoder,
    DQNTwoStageLearnedEvaluator,
    DQNTransitionEngine,
)

np.bool8 = np.bool_


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except Exception:
        return int(default)


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except Exception:
        return float(default)


def _env_int_list(name):
    raw = os.getenv(name)
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception:
            continue
    return values


def _set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


class ELP:
    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        base_gbest = gbest.unwrapped if hasattr(gbest, "unwrapped") else gbest
        self.s = copy.deepcopy(base_gbest)
        self.T = float(T)
        self.T_initial = float(T)
        self.G = int(G)
        self.t_max = int(t_max)
        self.k_hist = float(k)
        self.k_penalty = 1
        self.use_fast_evaluate = _env_flag("ELP_USE_FAST_EVALUATE", True)
        self.cooling_per_episode = min(
            1.0,
            max(1e-8, float(_env_float("ELP_COOLING_PER_EPISODE", 0.998))),
        )
        default_cooling_per_step = self.cooling_per_episode ** (1.0 / float(max(1, self.t_max)))
        self.cooling_per_step = min(
            1.0,
            max(1e-8, float(_env_float("ELP_COOLING_PER_STEP", default_cooling_per_step))),
        )
        self.temperature_floor_samples = 100
        self.temperature_floor_target_accept = 0.05
        self.temperature_floor_quantile = 25
        self.temperature_floor_cap_ratio = max(
            0.05,
            _env_float("ELP_TEMPERATURE_FLOOR_CAP_RATIO", 0.15),
        )
        self.temperature_floor_lower_ratio = max(
            1e-6,
            _env_float("ELP_TEMPERATURE_FLOOR_LOWER_RATIO", 0.05),
        )
        self.T_min = max(self.T_initial * self.temperature_floor_lower_ratio, 1.0)
        runtime_defaults = copy.deepcopy(getattr(runtime_config, "ELP_RUNTIME_DEFAULTS", {}))
        self.runtime_config_name = str(
            getattr(runtime_config, "ELP_RUNTIME_CONFIG_NAME", "default_runtime")
        )

        def _runtime_default(key, fallback):
            return copy.deepcopy(runtime_defaults.get(key, fallback))

        self.action_recipes = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [],
            5: [5],
            6: [6],
            7: [7],
            8: [8],
            9: [9],
            10: [10],
            11: [11],
            12: [12],
            13: [13],
            14: [14],
            15: [15],
        }
        self.temperature_floor_action_ids = [11, 14, 10, 2, 0]
        self.valid_actions = [0, 1, 2, 3, 9, 10, 11, 14, 15]
        self.phase_action_ids = {
            "early": [11, 14, 15, 9, 2, 1, 0, 10, 3],
            "mid": [10, 11, 14, 9, 2, 0, 1, 15, 3],
            "late": [10, 0, 2, 1, 11, 3],
            "infeasible": [3, 11, 14, 15, 9, 2, 1, 0, 10],
        }
        self.logger = logger
        # 将 DQN 主链路拆分到独立模块，主文件只保留调度与算子实现。
        self.dqn_operator_dispatcher = DQNOperatorDispatcher()
        self.dqn_search_controller = DQNSearchController()
        self.dqn_state_encoder = DQNStateEncoder()
        self.dqn_reward_engine = DQNRewardEngine()
        self.dqn_two_stage_evaluator = DQNTwoStageLearnedEvaluator()
        self.dqn_transition_engine = DQNTransitionEngine()
        self.dqn_program_runner = DQNProgramRunner()
        self.mid_structural_shot_enabled = _env_flag("ELP_MID_STRUCTURAL_SHOT_ENABLED", True)
        self.mid_structural_shot_start_ratio = min(
            max(float(_runtime_default("mid_structural_shot_start_ratio", 0.42)), 0.0),
            1.0,
        )
        self.mid_structural_shot_end_ratio = min(
            max(
                float(_runtime_default("mid_structural_shot_end_ratio", 0.76)),
                self.mid_structural_shot_start_ratio,
            ),
            1.0,
        )
        self.mid_structural_shot_no_improve = max(
            60,
            int(_runtime_default("mid_structural_shot_no_improve", 140)),
        )
        self.mid_structural_shot_cooldown_steps = max(
            80,
            int(_runtime_default("mid_structural_shot_cooldown_steps", max(700, self.t_max * 5))),
        )
        self.mid_structural_shot_max_count = max(
            1,
            int(_runtime_default("mid_structural_shot_max_count", 2)),
        )
        self.mid_structural_shot_action_ids = tuple(
            int(action_idx)
            for action_idx in _runtime_default("mid_structural_shot_action_ids", [11, 14, 15])
        )
        self.mid_structural_shot_trials_per_action = max(
            1,
            int(_runtime_default("mid_structural_shot_trials_per_action", 2)),
        )
        self.mid_structural_shot_raw_multiplier = max(
            1.0,
            float(_runtime_default("mid_structural_shot_raw_multiplier", 1.65)),
        )
        self.mid_structural_shot_eval_bonus = max(
            0,
            int(_runtime_default("mid_structural_shot_eval_bonus", 3)),
        )
        self.mid_structural_shot_diversity_floor = min(
            max(float(_runtime_default("mid_structural_shot_diversity_floor", 0.16)), 0.0),
            1.0,
        )
        self.mid_structural_shot_archive_slack_ratio = max(
            0.0,
            float(_runtime_default("mid_structural_shot_archive_slack_ratio", 0.06)),
        )
        self.mid_structural_shot_target_temp_ratio = min(
            max(float(_runtime_default("mid_structural_shot_target_temp_ratio", 0.46)), 0.05),
            1.0,
        )
        self.mid_structural_shot_post_switch_cooldown_steps = max(
            40,
            int(_runtime_default("mid_structural_shot_post_switch_cooldown_steps", max(160, self.t_max))),
        )
        self.local_search_on_any_feasible_accept = _env_flag(
            "ELP_LOCAL_SEARCH_ON_ANY_FEASIBLE_ACCEPT",
            True,
        )
        self.episode_chain_restart_enabled = _env_flag(
            "ELP_EPISODE_CHAIN_RESTART_ENABLED",
            False,
        )
        self.problem_size = len(getattr(self.s.fbs_model, "permutation", []))
        if self.problem_size >= 80:
            default_light_trials = 8
            default_light_segment_lengths = (2, 3)
        elif self.problem_size >= 60:
            default_light_trials = 10
            default_light_segment_lengths = (2, 3)
        elif self.problem_size >= 40:
            default_light_trials = 12
            default_light_segment_lengths = (2, 3, 4)
        else:
            default_light_trials = 14
            default_light_segment_lengths = (2, 3, 4)
        self.segment_insert_light_enabled = _env_flag("ELP_SEGMENT_INSERT_LIGHT_ENABLED", True)
        self.segment_insert_light_trials = max(
            4,
            _env_int("ELP_SEGMENT_INSERT_LIGHT_TRIALS", default_light_trials),
        )
        segment_lengths_override = _env_int_list("ELP_SEGMENT_INSERT_LIGHT_SEGMENT_LENGTHS")
        if segment_lengths_override:
            cleaned_segment_lengths = tuple(
                sorted({int(length) for length in segment_lengths_override if int(length) >= 2})
            )
            self.segment_insert_light_segment_lengths = (
                cleaned_segment_lengths if cleaned_segment_lengths else default_light_segment_lengths
            )
        else:
            self.segment_insert_light_segment_lengths = default_light_segment_lengths
        if self.problem_size >= 60:
            default_topk_guided_topk = 10
            default_topk_guided_max_iters = 2
        elif self.problem_size >= 40:
            default_topk_guided_topk = 12
            default_topk_guided_max_iters = 3
        else:
            default_topk_guided_topk = 20
            default_topk_guided_max_iters = 4
        self.action_labels = {
            0: "facility_swap",
            1: "bay_flip",
            2: "bay_swap",
            3: "repair",
            4: "idle",
            5: "facility_insert",
            6: "bay_shuffle",
            7: "facility_shuffle",
            8: "ga_action",
            9: "flow_guided_swap",
            10: "segment_insert",
            11: "cross_bay_relocate",
            12: "bay_split_by_flow",
            13: "bay_merge_by_flow",
            14: "adjacent_bay_repartition_by_flow",
            15: "adjacent_bay_block_repartition_by_flow",
        }
        self.bootstrap_recipes = [
            [3],
            [3, 3],
            [2, 3],
            [0, 3],
            [2, 0, 3],
            [6, 3],
            [9, 3],
            [10, 3],
            [11, 3],
            [6, 9, 3],
        ]
        self.elite_actions = [10, 11, 14, 0]
        self.elite_action_trials = {10: 4, 11: 4, 14: 3, 0: 1}
        self.elite_max_rounds = 4
        self.elite_late_progress_ratio = min(
            max(float(_runtime_default("elite_late_progress_ratio", 0.88)), 0.0),
            1.0,
        )
        elite_final_progress_ratio_default = _runtime_default("elite_final_progress_ratio", 0.95)
        self.elite_final_progress_ratio = min(
            max(
                float(elite_final_progress_ratio_default),
                self.elite_late_progress_ratio,
            ),
            1.0,
        )
        self.elite_late_no_improve = max(
            60,
            int(_runtime_default("elite_late_no_improve", 140)),
        )
        self.elite_final_no_improve = max(
            self.elite_late_no_improve,
            int(_runtime_default("elite_final_no_improve", 220)),
        )
        self.elite_late_round_multiplier = max(
            1.0,
            float(_runtime_default("elite_late_round_multiplier", 1.5)),
        )
        self.elite_final_round_multiplier = max(
            self.elite_late_round_multiplier,
            float(_runtime_default("elite_final_round_multiplier", 2.0)),
        )
        self.elite_late_trial_multiplier = max(
            1.0,
            float(_runtime_default("elite_late_trial_multiplier", 1.35)),
        )
        self.elite_final_trial_multiplier = max(
            self.elite_late_trial_multiplier,
            float(_runtime_default("elite_final_trial_multiplier", 1.85)),
        )
        self.elite_late_seed_count = max(
            1,
            int(_runtime_default("elite_late_seed_count", 2)),
        )
        self.elite_final_seed_count = max(
            1,
            int(_runtime_default("elite_final_seed_count", 1)),
        )
        self.elite_chain_takeover_gain_ratio = max(
            0.0,
            float(_runtime_default("elite_chain_takeover_gain_ratio", 0.0)),
        )
        self.elite_late_chain_takeover_gain_ratio = max(
            0.0,
            float(_runtime_default("elite_late_chain_takeover_gain_ratio", 8e-4)),
        )
        self.elite_final_chain_takeover_gain_ratio = max(
            0.0,
            float(_runtime_default("elite_final_chain_takeover_gain_ratio", 1.5e-3)),
        )
        self.light_restart_recipe = [0, 3]
        self.diversify_recipe = [15, 6, 9, 3]
        self.bin_width = 50.0
        self.bin_width_recent_window = max(200, _env_int("ELP_BIN_WIDTH_RECENT_WINDOW", 4000))
        self.bin_width_target_bins = max(16, _env_int("ELP_BIN_WIDTH_TARGET_BINS", 64))
        self.bin_width_lower_ratio = max(1e-8, _env_float("ELP_BIN_WIDTH_LOWER_RATIO", 2e-5))
        self.bin_width_upper_ratio = max(
            self.bin_width_lower_ratio * 2.0,
            _env_float("ELP_BIN_WIDTH_UPPER_RATIO", 2e-3),
        )
        self.bin_width_fallback_ratio = max(
            self.bin_width_lower_ratio,
            _env_float("ELP_BIN_WIDTH_FALLBACK_RATIO", 2e-4),
        )
        self.bin_width_min_abs = max(1.0, _env_float("ELP_BIN_WIDTH_MIN_ABS", 10.0))
        self.energy_histogram = {}
        self.energy_history = []
        self.modified_energy_history = []
        self.prob_history = []
        self.best_history = []
        self.gbest_plot_path = None
        self.gbest_update_count = 0
        self.no_improve_steps = 0
        self.episodes_without_improvement = 0
        self.last_diversify_step = -max(1, self.t_max)
        self.diversification_count = 0
        self.feasible_solution_count = 0
        self.best_feasible_cost = np.inf
        self.worst_feasible_cost = None
        self.best_feasible_solution = None
        self.action_telemetry = {
            action_idx: self._new_action_telemetry_entry(action_idx)
            for action_idx in sorted(set(self.valid_actions) | set(self.elite_actions))
        }
        self.elite_trigger_count = 0
        self.elite_improvement_count = 0
        self.elite_total_gain = 0.0
        self.enable_reheat_logging = _env_flag("ELP_ENABLE_REHEAT_LOG", False)

        self.temperature_floor_target_accept = float(
            _runtime_default("temperature_floor_target_accept", self.temperature_floor_target_accept)
        )
        self.temperature_floor_quantile = int(
            _runtime_default("temperature_floor_quantile", self.temperature_floor_quantile)
        )

        self.dqn_epsilon_start = float(_runtime_default("dqn_epsilon_start", 0.50))
        self.dqn_epsilon_min = float(_runtime_default("dqn_epsilon_min", 0.08))
        self.dqn_epsilon_decay = float(_runtime_default("dqn_epsilon_decay", 0.9994))
        self.dqn_gamma = float(_runtime_default("dqn_gamma", 0.95))
        self.dqn_lr = float(_runtime_default("dqn_lr", 3e-4))
        self.dqn_batch_size = int(_runtime_default("dqn_batch_size", 64))
        self.dqn_replay_capacity = int(_runtime_default("dqn_replay_capacity", 50000))
        self.dqn_warmup_steps = int(_runtime_default("dqn_warmup_steps", 3000))
        self.dqn_update_every = int(_runtime_default("dqn_update_every", 4))
        self.dqn_target_update_every = int(_runtime_default("dqn_target_update_every", 500))
        self.dqn_idqn_k = int(_runtime_default("dqn_idqn_k", 2))
        self.dqn_embedding_dim = int(_runtime_default("dqn_embedding_dim", 32))
        self.dqn_hidden_dim = int(_runtime_default("dqn_hidden_dim", 64))
        self.dqn_grad_clip = float(_runtime_default("dqn_grad_clip", 10.0))
        self.dqn_step_epsilon_schedule = bool(
            _runtime_default("dqn_step_epsilon_schedule", False)
        )
        self.dqn_epsilon_schedule_warmup_steps = int(
            _runtime_default("dqn_epsilon_schedule_warmup_steps", 0)
        )
        self.dqn_epsilon_schedule_decay_ratio = float(
            _runtime_default("dqn_epsilon_schedule_decay_ratio", 1.0)
        )
        self.diversify_trigger_no_improve = int(
            _runtime_default("diversify_trigger_no_improve", 90)
        )
        self.reheat_enabled = bool(_runtime_default("reheat_enabled", True))
        self.elite_actions = [
            int(action_idx)
            for action_idx in _runtime_default("elite_actions", [10, 11, 14, 0])
        ]
        self.elite_action_trials = {int(action_idx): 1 for action_idx in self.elite_actions}
        configured_elite_trials = _runtime_default("elite_action_trials", {})
        if isinstance(configured_elite_trials, dict):
            for action_idx in self.elite_actions:
                configured_value = configured_elite_trials.get(
                    action_idx,
                    configured_elite_trials.get(str(action_idx), self.elite_action_trials[action_idx]),
                )
                self.elite_action_trials[int(action_idx)] = int(max(1, configured_value))
        self.elite_max_rounds = int(_runtime_default("elite_max_rounds", 4))

        self.accept_window_size = int(_runtime_default("accept_window_size", 120))
        self.accept_window = deque(maxlen=self.accept_window_size)
        self.accept_rate_window = 1.0
        accept_window_reset_default = _runtime_default("accept_window_reset_each_episode", None)
        if accept_window_reset_default is None:
            accept_window_reset_default = bool(self.episode_chain_restart_enabled)
        no_improve_reset_default = _runtime_default("no_improve_reset_on_episode_restart", None)
        if no_improve_reset_default is None:
            no_improve_reset_default = bool(self.episode_chain_restart_enabled)
        reheat_reset_default = _runtime_default("reheat_reset_each_episode", None)
        if reheat_reset_default is None:
            reheat_reset_default = bool(self.episode_chain_restart_enabled)
        self.accept_window_reset_each_episode = bool(accept_window_reset_default)
        self.no_improve_reset_on_episode_restart = bool(no_improve_reset_default)
        self.reheat_reset_each_episode = bool(reheat_reset_default)
        self.reheat_accept_rate_threshold = float(
            _runtime_default("reheat_accept_rate_threshold", 0.04)
        )
        self.reheat_no_improve_threshold = int(
            _runtime_default("reheat_no_improve_threshold", 180)
        )
        self.reheat_temp_gate_ratio = float(_runtime_default("reheat_temp_gate_ratio", 0.65))
        self.reheat_target_low_ratio = float(_runtime_default("reheat_target_low_ratio", 0.68))
        self.reheat_target_high_ratio = float(_runtime_default("reheat_target_high_ratio", 0.74))
        self.reheat_cooldown_steps = max(60, self.t_max // 5)
        self.reheat_progress_cap_ratio = float(
            _runtime_default("reheat_progress_cap_ratio", 0.55)
        )
        self.reheat_max_per_episode = int(_runtime_default("reheat_max_per_episode", 1))
        self.reheat_episode_count = 0
        self.last_reheat_step = -self.reheat_cooldown_steps
        self.reheat_trigger_count = 0

        self.acceptance_phase_enabled = bool(
            _runtime_default("acceptance_phase_enabled", False)
        )
        self.acceptance_phase_mode = str(
            _runtime_default("acceptance_phase_mode", "two")
        )
        self.acceptance_phase_split_ratio = float(
            _runtime_default("acceptance_phase_split_ratio", 0.60)
        )
        self.acceptance_phase_first_split_ratio = float(
            _runtime_default("acceptance_phase_first_split_ratio", 0.25)
        )
        self.acceptance_phase_second_split_ratio = float(
            _runtime_default("acceptance_phase_second_split_ratio", 0.60)
        )
        self.acceptance_early_temp_multiplier = float(
            _runtime_default("acceptance_early_temp_multiplier", 1.0)
        )
        self.acceptance_mid_temp_multiplier = float(
            _runtime_default("acceptance_mid_temp_multiplier", 1.0)
        )
        self.acceptance_late_temp_multiplier_start = float(
            _runtime_default("acceptance_late_temp_multiplier_start", 1.0)
        )
        self.acceptance_late_temp_multiplier_end = float(
            _runtime_default("acceptance_late_temp_multiplier_end", 1.0)
        )
        self.current_progress_ratio = 0.0

        self.reward_profile = str(_runtime_default("reward_profile", "baseline")).strip().lower()
        self.reward_train_immediate_weight = max(
            0.0,
            float(_runtime_default("reward_train_immediate_weight", 0.80)),
        )
        self.reward_train_final_weight = max(
            0.0,
            float(_runtime_default("reward_train_final_weight", 0.20)),
        )
        reward_train_weight_sum = (
            self.reward_train_immediate_weight + self.reward_train_final_weight
        )
        # 训练奖励默认以 immediate 为主，降低 local search 对信用分配的污染。
        if reward_train_weight_sum <= 1e-12:
            self.reward_train_immediate_weight = 1.0
            self.reward_train_final_weight = 0.0
        else:
            self.reward_train_immediate_weight /= reward_train_weight_sum
            self.reward_train_final_weight /= reward_train_weight_sum
        # s2奖励会基于相对改进幅度做EMA缩放。
        self.reward_rel_delta_window = max(
            50,
            int(_runtime_default("reward_rel_delta_window", 400)),
        )
        self.reward_rel_delta_scale_min = max(
            1e-6,
            float(_runtime_default("reward_rel_delta_scale_min", 5e-4)),
        )
        self.reward_rel_delta_scale_max = max(
            self.reward_rel_delta_scale_min * 2.0,
            float(_runtime_default("reward_rel_delta_scale_max", 1e-2)),
        )
        self.reward_rel_delta_scale_default = min(
            max(
                float(_runtime_default("reward_rel_delta_scale_default", 2e-3)),
                self.reward_rel_delta_scale_min,
            ),
            self.reward_rel_delta_scale_max,
        )
        self.reward_rel_delta_scale_ema_beta = min(
            max(float(_runtime_default("reward_rel_delta_scale_ema_beta", 0.92)), 0.0),
            0.9999,
        )
        self.reward_rel_delta_history = deque(maxlen=self.reward_rel_delta_window)
        self.reward_rel_delta_scale_ema = self.reward_rel_delta_scale_default
        self.reward_cost_scale_window = max(
            50,
            int(_runtime_default("reward_cost_scale_window", 200)),
        )
        self.reward_cost_scale_floor_ratio = max(
            1e-6,
            float(_runtime_default("reward_cost_scale_floor_ratio", 5e-4)),
        )
        self.reward_cost_scale_cap_ratio = max(
            self.reward_cost_scale_floor_ratio * 10.0,
            float(_runtime_default("reward_cost_scale_cap_ratio", 2e-2)),
        )
        self.reward_cost_scale_min_abs = max(
            1.0,
            float(_runtime_default("reward_cost_scale_min_abs", 5.0)),
        )
        self.recent_feasible_costs = deque(maxlen=self.reward_cost_scale_window)

        self.dqn_epsilon_start = _env_float("ELP_DQN_EPSILON_START", self.dqn_epsilon_start)
        self.dqn_epsilon_min = _env_float("ELP_DQN_EPSILON_MIN", self.dqn_epsilon_min)
        self.dqn_epsilon_decay = _env_float("ELP_DQN_EPSILON_DECAY", self.dqn_epsilon_decay)
        self.dqn_gamma = _env_float("ELP_DQN_GAMMA", self.dqn_gamma)
        self.dqn_lr = _env_float("ELP_DQN_LR", self.dqn_lr)
        self.dqn_batch_size = max(1, _env_int("ELP_DQN_BATCH_SIZE", self.dqn_batch_size))
        self.dqn_replay_capacity = max(1, _env_int("ELP_DQN_REPLAY_CAPACITY", self.dqn_replay_capacity))
        self.dqn_warmup_steps = max(0, _env_int("ELP_DQN_WARMUP_STEPS", self.dqn_warmup_steps))
        self.dqn_update_every = max(1, _env_int("ELP_DQN_UPDATE_EVERY", self.dqn_update_every))
        self.dqn_target_update_every = max(1, _env_int("ELP_DQN_TARGET_UPDATE_EVERY", self.dqn_target_update_every))
        self.dqn_idqn_k = max(1, _env_int("ELP_DQN_IDQN_K", self.dqn_idqn_k))
        self.dqn_embedding_dim = max(1, _env_int("ELP_DQN_EMBEDDING_DIM", self.dqn_embedding_dim))
        self.dqn_hidden_dim = max(1, _env_int("ELP_DQN_HIDDEN_DIM", self.dqn_hidden_dim))
        self.dqn_grad_clip = max(0.0, _env_float("ELP_DQN_GRAD_CLIP", self.dqn_grad_clip))
        self.dqn_step_epsilon_schedule = _env_flag("ELP_DQN_STEP_EPSILON_SCHEDULE", self.dqn_step_epsilon_schedule)
        self.dqn_epsilon_schedule_warmup_steps = max(
            0,
            _env_int("ELP_DQN_EPSILON_SCHEDULE_WARMUP_STEPS", self.dqn_epsilon_schedule_warmup_steps),
        )
        self.dqn_epsilon_schedule_decay_ratio = max(
            0.0,
            _env_float("ELP_DQN_EPSILON_SCHEDULE_DECAY_RATIO", self.dqn_epsilon_schedule_decay_ratio),
        )
        self.temperature_floor_target_accept = _env_float(
            "ELP_TEMPERATURE_FLOOR_TARGET_ACCEPT",
            self.temperature_floor_target_accept,
        )
        self.temperature_floor_quantile = _env_int(
            "ELP_TEMPERATURE_FLOOR_QUANTILE",
            self.temperature_floor_quantile,
        )
        self.diversify_trigger_no_improve = max(
            1,
            _env_int("ELP_DIVERSIFY_TRIGGER_NO_IMPROVE", self.diversify_trigger_no_improve),
        )
        self.reheat_enabled = _env_flag("ELP_REHEAT_ENABLED", self.reheat_enabled)
        self.elite_max_rounds = max(1, _env_int("ELP_ELITE_MAX_ROUNDS", self.elite_max_rounds))
        elite_actions_override = _env_int_list("ELP_ELITE_ACTIONS")
        if elite_actions_override:
            self.elite_actions = [int(action_idx) for action_idx in elite_actions_override]
            self.elite_action_trials = {
                int(action_idx): int(self.elite_action_trials.get(int(action_idx), 1))
                for action_idx in self.elite_actions
            }
        self.accept_window_size = max(1, _env_int("ELP_ACCEPT_WINDOW_SIZE", self.accept_window_size))
        self.accept_window = deque(self.accept_window, maxlen=self.accept_window_size)
        self.accept_window_reset_each_episode = _env_flag(
            "ELP_ACCEPT_WINDOW_RESET_EACH_EPISODE",
            self.accept_window_reset_each_episode,
        )
        self.no_improve_reset_on_episode_restart = _env_flag(
            "ELP_NO_IMPROVE_RESET_ON_EPISODE_RESTART",
            self.no_improve_reset_on_episode_restart,
        )
        self.reheat_reset_each_episode = _env_flag(
            "ELP_REHEAT_RESET_EACH_EPISODE",
            self.reheat_reset_each_episode,
        )
        self.reheat_accept_rate_threshold = _env_float(
            "ELP_REHEAT_ACCEPT_RATE_THRESHOLD",
            self.reheat_accept_rate_threshold,
        )
        self.reheat_no_improve_threshold = max(
            1,
            _env_int("ELP_REHEAT_NO_IMPROVE_THRESHOLD", self.reheat_no_improve_threshold),
        )
        self.reheat_temp_gate_ratio = _env_float("ELP_REHEAT_TEMP_GATE_RATIO", self.reheat_temp_gate_ratio)
        self.reheat_target_low_ratio = _env_float("ELP_REHEAT_TARGET_LOW_RATIO", self.reheat_target_low_ratio)
        self.reheat_target_high_ratio = _env_float("ELP_REHEAT_TARGET_HIGH_RATIO", self.reheat_target_high_ratio)
        self.reheat_progress_cap_ratio = _env_float(
            "ELP_REHEAT_PROGRESS_CAP_RATIO",
            self.reheat_progress_cap_ratio,
        )
        self.reheat_max_per_episode = max(
            1,
            _env_int("ELP_REHEAT_MAX_PER_EPISODE", self.reheat_max_per_episode),
        )
        self.acceptance_phase_enabled = _env_flag(
            "ELP_ACCEPTANCE_PHASE_ENABLED",
            self.acceptance_phase_enabled,
        )
        self.reward_profile = os.getenv("ELP_REWARD_PROFILE", self.reward_profile).strip().lower()

        # Top-K????????????1?
        self.topk_guided_enabled = _env_flag("ELP_TOPK_GUIDED_ENABLED", True)
        self.topk_guided_topk = max(
            5,
            _env_int("ELP_TOPK_GUIDED_TOPK", default_topk_guided_topk),
        )
        self.topk_guided_max_iters = max(
            1,
            _env_int("ELP_TOPK_GUIDED_MAX_ITERS", default_topk_guided_max_iters),
        )
        self.topk_guided_window_radius = max(
            1,
            _env_int("ELP_TOPK_GUIDED_WINDOW_RADIUS", 2 if self.problem_size >= 60 else 3),
        )
        self.topk_guided_target_position_cap = max(
            8,
            _env_int("ELP_TOPK_GUIDED_TARGET_POSITION_CAP", 18 if self.problem_size >= 60 else 24),
        )
        self.final_elite_push_enabled = _env_flag("ELP_FINAL_ELITE_PUSH_ENABLED", True)
        self.final_elite_push_progress_ratio = min(
            max(float(_runtime_default("final_elite_push_progress_ratio", 0.86)), 0.0),
            1.0,
        )
        self.final_elite_push_no_improve = max(
            60,
            int(_runtime_default("final_elite_push_no_improve", 180)),
        )
        self.final_elite_push_accept_rate_threshold = min(
            max(float(_runtime_default("final_elite_push_accept_rate_threshold", 0.03)), 0.0),
            1.0,
        )
        self.final_elite_push_cooldown_steps = max(
            40,
            int(_runtime_default("final_elite_push_cooldown_steps", max(800, self.t_max * 6))),
        )
        self.final_elite_push_max_count = max(
            1,
            int(_runtime_default("final_elite_push_max_count", 4)),
        )
        self.final_elite_push_post_switch_cooldown_steps = max(
            40,
            int(_runtime_default("final_elite_push_post_switch_cooldown_steps", max(200, self.t_max * 2))),
        )
        self.final_elite_push_guided_local_search = bool(
            _runtime_default("final_elite_push_guided_local_search", True)
        )
        final_elite_push_guided_topk_default = _runtime_default("final_elite_push_guided_topk", None)
        if final_elite_push_guided_topk_default is None:
            final_elite_push_guided_topk_default = self.topk_guided_topk + 3
        self.final_elite_push_guided_topk = max(
            self.topk_guided_topk,
            int(final_elite_push_guided_topk_default),
        )
        final_elite_push_guided_max_iters_default = _runtime_default(
            "final_elite_push_guided_max_iters",
            None,
        )
        if final_elite_push_guided_max_iters_default is None:
            final_elite_push_guided_max_iters_default = min(self.topk_guided_max_iters + 2, 5)
        self.final_elite_push_guided_max_iters = max(
            self.topk_guided_max_iters,
            int(final_elite_push_guided_max_iters_default),
        )
        self.mid_structural_shot_guided_local_search = bool(
            _runtime_default("mid_structural_shot_guided_local_search", True)
        )
        mid_structural_guided_topk_default = _runtime_default("mid_structural_shot_guided_topk", None)
        if mid_structural_guided_topk_default is None:
            mid_structural_guided_topk_default = self.topk_guided_topk + 2
        self.mid_structural_shot_guided_topk = max(
            self.topk_guided_topk,
            int(mid_structural_guided_topk_default),
        )
        mid_structural_guided_max_iters_default = _runtime_default(
            "mid_structural_shot_guided_max_iters",
            None,
        )
        if mid_structural_guided_max_iters_default is None:
            mid_structural_guided_max_iters_default = min(self.topk_guided_max_iters + 1, 4)
        self.mid_structural_shot_guided_max_iters = max(
            self.topk_guided_max_iters,
            int(mid_structural_guided_max_iters_default),
        )
        self.two_stage_heavy_actions_enabled = _env_flag(
            "ELP_TWO_STAGE_HEAVY_ACTIONS_ENABLED",
            True,
        )
        self.two_stage_heavy_action_ids = set(
            _env_int_list("ELP_TWO_STAGE_HEAVY_ACTION_IDS") or [9, 10, 11, 14, 15]
        )
        self.two_stage_random_survivor = bool(_runtime_default("two_stage_random_survivor", True))
        configured_two_stage_proposal_counts = _runtime_default(
            "two_stage_proposal_counts",
            {9: 14, 10: 18, 11: 20, 14: 18, 15: 18},
        )
        self.two_stage_proposal_counts = {
            action_idx: max(
                8,
                int(
                    configured_two_stage_proposal_counts.get(
                        action_idx,
                        configured_two_stage_proposal_counts.get(str(action_idx), default_value),
                    )
                ),
            )
            for action_idx, default_value in {9: 14, 10: 18, 11: 20, 14: 18, 15: 18}.items()
        }
        configured_two_stage_eval_counts = _runtime_default(
            "two_stage_eval_counts",
            {9: 4, 10: 5, 11: 6, 14: 6, 15: 6},
        )
        self.two_stage_eval_counts = {
            action_idx: max(
                2,
                int(
                    configured_two_stage_eval_counts.get(
                        action_idx,
                        configured_two_stage_eval_counts.get(str(action_idx), default_value),
                    )
                ),
            )
            for action_idx, default_value in {9: 4, 10: 5, 11: 6, 14: 6, 15: 6}.items()
        }
        self.two_stage_pair_pool_cap = max(
            4,
            int(_runtime_default("two_stage_pair_pool_cap", 10)),
        )
        two_stage_candidate_window_radius_default = _runtime_default(
            "two_stage_candidate_window_radius",
            None,
        )
        if two_stage_candidate_window_radius_default is None:
            two_stage_candidate_window_radius_default = 2 if self.problem_size >= 60 else 3
        self.two_stage_candidate_window_radius = max(
            1,
            int(two_stage_candidate_window_radius_default),
        )
        self.two_stage_proxy_weights = {
            str(weight_key): max(0.0, float(weight_value))
            for weight_key, weight_value in _runtime_default(
                "two_stage_proxy_weights",
                {
                    "order_penalty": 1.00,
                    "adjacent_cross_penalty": 0.90,
                    "global_cross_penalty": 0.40,
                    "area_balance_penalty": 0.60,
                    "geometry_penalty": 1.40,
                },
            ).items()
        }
        self.two_stage_action14_proxy_weights = {
            str(weight_key): max(0.0, float(weight_value))
            for weight_key, weight_value in _runtime_default(
                "two_stage_action14_proxy_weights",
                {
                    "cut_gain": 3.20,
                    "pair_area_gain": 1.60,
                    "geometry_gain": 1.80,
                },
            ).items()
        }
        self.two_stage_action11_proxy_weights = {
            str(weight_key): max(0.0, float(weight_value))
            for weight_key, weight_value in _runtime_default(
                "two_stage_action11_proxy_weights",
                {
                    "target_affinity_gain": 3.00,
                    "source_damage_penalty": 2.10,
                    "area_gain": 1.40,
                    "geometry_gain": 1.60,
                    "boundary_bonus": 0.45,
                },
            ).items()
        }
        self.two_stage_action15_proxy_weights = {
            str(weight_key): max(0.0, float(weight_value))
            for weight_key, weight_value in _runtime_default(
                "two_stage_action15_proxy_weights",
                {
                    "cut_gain": 2.80,
                    "block_affinity_gain": 2.40,
                    "pair_area_gain": 1.30,
                    "geometry_gain": 1.50,
                    "cohesion_bonus": 0.80,
                    "exchange_bonus": 0.25,
                },
            ).items()
        }
        self.two_stage_local_proxy_enabled = bool(
            _runtime_default("two_stage_local_proxy_enabled", True)
        )
        self.two_stage_local_proxy_action_ids = set(
            int(action_idx)
            for action_idx in _runtime_default("two_stage_local_proxy_action_ids", [11, 14, 15])
        )
        self.two_stage_local_proxy_weights = {
            str(weight_key): max(0.0, float(weight_value))
            for weight_key, weight_value in _runtime_default(
                "two_stage_local_proxy_weights",
                {
                    "order_penalty": 0.95,
                    "adjacent_cross_penalty": 1.20,
                    "global_cross_penalty": 0.60,
                    "area_balance_penalty": 0.95,
                    "geometry_penalty": 1.15,
                },
            ).items()
        }
        configured_two_stage_local_proxy_mix = _runtime_default(
            "two_stage_local_proxy_mix",
            {11: 0.25, 14: 0.35, 15: 0.30},
        )
        self.two_stage_local_proxy_mix = {
            int(action_idx): max(0.0, float(weight_value))
            for action_idx, weight_value in configured_two_stage_local_proxy_mix.items()
        }
        self.two_stage_local_proxy_phase_mix = {
            str(phase_name): max(0.0, float(weight_value))
            for phase_name, weight_value in _runtime_default(
                "two_stage_local_proxy_phase_mix",
                {"off": 1.00, "ramp": 0.70, "late": 0.08},
            ).items()
        }
        self.two_stage_learned_evaluator_enabled = _env_flag(
            "ELP_TWO_STAGE_LEARNED_EVALUATOR_ENABLED",
            bool(_runtime_default("two_stage_learned_evaluator_enabled", False)),
        )
        self.two_stage_learned_evaluator_collect_data = _env_flag(
            "ELP_TWO_STAGE_LEARNED_EVALUATOR_COLLECT_DATA",
            bool(_runtime_default("two_stage_learned_evaluator_collect_data", False)),
        )
        self.two_stage_learned_evaluator_collect_full_labels = _env_flag(
            "ELP_TWO_STAGE_LEARNED_EVALUATOR_COLLECT_FULL_LABELS",
            bool(_runtime_default("two_stage_learned_evaluator_collect_full_labels", False)),
        )
        self.two_stage_learned_evaluator_model_path = str(
            os.getenv(
                "ELP_TWO_STAGE_LEARNED_EVALUATOR_MODEL_PATH",
                str(_runtime_default("two_stage_learned_evaluator_model_path", "")),
            )
        ).strip()
        self.two_stage_learned_evaluator_dataset_path = str(
            os.getenv(
                "ELP_TWO_STAGE_LEARNED_EVALUATOR_DATASET_PATH",
                str(_runtime_default("two_stage_learned_evaluator_dataset_path", "")),
            )
        ).strip()
        self.two_stage_learned_evaluator_hidden_dim = max(
            16,
            int(_runtime_default("two_stage_learned_evaluator_hidden_dim", 64)),
        )
        self.two_stage_learned_evaluator_message_steps = max(
            1,
            int(_runtime_default("two_stage_learned_evaluator_message_steps", 2)),
        )
        self.two_stage_learned_evaluator_dropout = min(
            max(float(_runtime_default("two_stage_learned_evaluator_dropout", 0.05)), 0.0),
            0.50,
        )
        self.two_stage_learned_evaluator_edge_topk = max(
            2,
            int(_runtime_default("two_stage_learned_evaluator_edge_topk", 12)),
        )
        self.dqn_two_stage_evaluator.configure(self)
        self.local_search_trigger_action_ids = set(
            _env_int_list("ELP_LOCAL_SEARCH_TRIGGER_ACTION_IDS") or [10, 11, 14, 15]
        )
        self.local_search_trigger_no_improve = max(
            0,
            _env_int("ELP_LOCAL_SEARCH_TRIGGER_NO_IMPROVE", max(30, self.t_max // 6)),
        )
        self.local_search_trigger_gap_ratio = max(
            0.0,
            _env_float("ELP_LOCAL_SEARCH_TRIGGER_GAP_RATIO", 0.03),
        )
        self.topk_guided_trigger_action_ids = set(
            _env_int_list("ELP_TOPK_TRIGGER_ACTION_IDS") or [10, 11, 14, 15]
        )
        self.topk_guided_trigger_no_improve = max(
            0,
            _env_int("ELP_TOPK_TRIGGER_NO_IMPROVE", max(60, self.t_max // 4)),
        )
        self.topk_guided_trigger_gap_ratio = max(
            0.0,
            _env_float("ELP_TOPK_TRIGGER_GAP_RATIO", 0.015),
        )

        # ???????? + ???????2?
        self.high_flow_warmstart_enabled = _env_flag("ELP_HIGH_FLOW_WARMSTART_ENABLED", True)
        self.high_flow_warmstart_restarts = max(1, _env_int("ELP_HIGH_FLOW_WARMSTART_RESTARTS", 6))
        self.high_flow_warmstart_stagnation_episodes = max(
            2,
            int(_runtime_default("high_flow_warmstart_stagnation_episodes", 4)),
        )
        self.high_flow_warmstart_use_topk = bool(
            _runtime_default("high_flow_warmstart_use_topk", True)
        )
        high_flow_warmstart_topk_default = _runtime_default("high_flow_warmstart_topk", None)
        if high_flow_warmstart_topk_default is None:
            high_flow_warmstart_topk_default = self.topk_guided_topk
        self.high_flow_warmstart_topk = max(
            5,
            int(high_flow_warmstart_topk_default),
        )
        high_flow_warmstart_max_iters_default = _runtime_default("high_flow_warmstart_max_iters", None)
        if high_flow_warmstart_max_iters_default is None:
            high_flow_warmstart_max_iters_default = self.topk_guided_max_iters
        self.high_flow_warmstart_max_iters = max(
            1,
            int(high_flow_warmstart_max_iters_default),
        )
        self.high_flow_area_balance_weight = max(
            0.0,
            float(_runtime_default("high_flow_area_balance_weight", 0.30)),
        )
        self.high_flow_geometry_pressure_weight = max(
            0.0,
            float(_runtime_default("high_flow_geometry_pressure_weight", 0.20)),
        )
        high_flow_episode0_restarts_default = _runtime_default("high_flow_warmstart_episode0_restarts", None)
        if high_flow_episode0_restarts_default is None:
            high_flow_episode0_restarts_default = min(3, self.high_flow_warmstart_restarts)
        self.high_flow_warmstart_episode0_restarts = max(
            1,
            int(high_flow_episode0_restarts_default),
        )
        self.high_flow_warmstart_episode0_refine_topk_count = max(
            0,
            int(_runtime_default("high_flow_warmstart_episode0_refine_topk_count", 1)),
        )
        high_flow_episode0_topk_default = _runtime_default("high_flow_warmstart_episode0_topk", None)
        if high_flow_episode0_topk_default is None:
            high_flow_episode0_topk_default = max(5, self.high_flow_warmstart_topk - 2)
        self.high_flow_warmstart_episode0_topk = max(
            5,
            int(high_flow_episode0_topk_default),
        )
        high_flow_episode0_max_iters_default = _runtime_default(
            "high_flow_warmstart_episode0_max_iters",
            None,
        )
        if high_flow_episode0_max_iters_default is None:
            high_flow_episode0_max_iters_default = max(1, self.high_flow_warmstart_max_iters - 1)
        self.high_flow_warmstart_episode0_max_iters = max(
            1,
            int(high_flow_episode0_max_iters_default),
        )
        self.elite_archive_enabled = _env_flag("ELP_ELITE_ARCHIVE_ENABLED", True)
        self.elite_archive_feasible_capacity = max(
            1,
            _env_int("ELP_ELITE_ARCHIVE_FEASIBLE_CAPACITY", 3),
        )
        self.elite_archive_frontier_capacity = max(
            0,
            _env_int("ELP_ELITE_ARCHIVE_FRONTIER_CAPACITY", 2),
        )
        self.elite_archive_min_diversity = min(
            max(_env_float("ELP_ELITE_ARCHIVE_MIN_DIVERSITY", 0.20), 0.0),
            1.0,
        )
        self.elite_archive_near_duplicate_ratio = min(
            max(_env_float("ELP_ELITE_ARCHIVE_NEAR_DUPLICATE_RATIO", 0.50), 0.05),
            1.0,
        )
        self.elite_archive_feasible_slack_ratio = max(
            0.0,
            _env_float("ELP_ELITE_ARCHIVE_FEASIBLE_SLACK_RATIO", 0.03),
        )
        self.elite_archive_frontier_max_d_inf = max(
            1,
            _env_int("ELP_ELITE_ARCHIVE_FRONTIER_MAX_D_INF", 2),
        )
        self.elite_archive_seed_count = max(
            1,
            _env_int("ELP_ELITE_ARCHIVE_SEED_COUNT", 2),
        )
        self.elite_archive_multi_seed_progress_gate_ratio = min(
            max(_env_float("ELP_ELITE_ARCHIVE_MULTI_SEED_PROGRESS_GATE_RATIO", 0.82), 0.0),
            1.0,
        )
        self.elite_archive_multi_seed_no_improve = max(
            80,
            _env_int(
                "ELP_ELITE_ARCHIVE_MULTI_SEED_NO_IMPROVE",
                max(180, self.diversify_trigger_no_improve + 40),
            ),
        )
        self.archive_switch_enabled = _env_flag("ELP_ARCHIVE_SWITCH_ENABLED", True)
        self.archive_switch_no_improve = max(
            40,
            _env_int(
                "ELP_ARCHIVE_SWITCH_NO_IMPROVE",
                max(120, self.diversify_trigger_no_improve),
            ),
        )
        self.archive_switch_accept_rate_threshold = min(
            max(_env_float("ELP_ARCHIVE_SWITCH_ACCEPT_RATE_THRESHOLD", 0.08), 0.0),
            1.0,
        )
        self.archive_switch_cooldown_steps = max(
            40,
            int(_runtime_default("archive_switch_cooldown_steps", max(200, self.t_max * 4))),
        )
        self.archive_switch_progress_gate_ratio = min(
            max(float(_runtime_default("archive_switch_progress_gate_ratio", 0.68)), 0.0),
            1.0,
        )
        self.archive_switch_full_progress_ratio = min(
            max(
                float(
                    _runtime_default(
                        "archive_switch_full_progress_ratio",
                        min(0.90, self.archive_switch_progress_gate_ratio + 0.14),
                    )
                ),
                self.archive_switch_progress_gate_ratio,
            ),
            1.0,
        )
        self.archive_switch_target_temp_ratio = max(
            0.05,
            float(_runtime_default("archive_switch_target_temp_ratio", 0.40)),
        )
        self.archive_switch_diversity_weight = max(
            0.0,
            float(_runtime_default("archive_switch_diversity_weight", 1.80)),
        )
        self.archive_switch_quality_weight = max(
            0.0,
            float(_runtime_default("archive_switch_quality_weight", 1.00)),
        )
        self.archive_switch_staleness_weight = max(
            0.0,
            float(_runtime_default("archive_switch_staleness_weight", 0.40)),
        )
        self.archive_switch_recent_use_penalty = max(
            0.0,
            float(_runtime_default("archive_switch_recent_use_penalty", 0.35)),
        )
        self.archive_switch_feasible_slack_ratio = max(
            0.0,
            float(_runtime_default("archive_switch_feasible_slack_ratio", 0.03)),
        )
        self.archive_switch_min_score = max(
            0.0,
            float(_runtime_default("archive_switch_min_score", 0.80)),
        )
        self.archive_switch_ramp_no_improve_multiplier = max(
            1.0,
            float(_runtime_default("archive_switch_ramp_no_improve_multiplier", 1.35)),
        )
        self.archive_switch_ramp_cooldown_multiplier = max(
            1.0,
            float(_runtime_default("archive_switch_ramp_cooldown_multiplier", 1.50)),
        )
        self.archive_switch_ramp_min_score_bonus = float(
            _runtime_default("archive_switch_ramp_min_score_bonus", 0.25)
        )
        self.archive_switch_late_no_improve_multiplier = max(
            0.1,
            float(_runtime_default("archive_switch_late_no_improve_multiplier", 1.00)),
        )
        self.archive_switch_late_cooldown_multiplier = max(
            0.1,
            float(_runtime_default("archive_switch_late_cooldown_multiplier", 1.35)),
        )
        self.archive_switch_late_min_score_bonus = float(
            _runtime_default("archive_switch_late_min_score_bonus", 0.10)
        )
        self.mid_structural_shot_late_archive_lock_steps = max(
            80,
            int(_runtime_default("mid_structural_shot_late_archive_lock_steps", max(800, self.t_max * 5))),
        )
        self.mid_structural_shot_late_archive_lock_gap_ratio = max(
            0.0,
            float(_runtime_default("mid_structural_shot_late_archive_lock_gap_ratio", 0.02)),
        )

        self.current_global_step = 0
        self.mid_structural_shot_count = 0
        self.mid_structural_shot_success_count = 0
        self.last_mid_structural_shot_step = -self.mid_structural_shot_cooldown_steps
        self.last_mid_structural_shot_success_step = -10**9
        self.mid_structural_shot_blocked_after_fail = False
        self.mid_structural_shot_fail_archive_switch_count = -1
        self.mid_structural_shot_fail_gbest_update_count = -1
        self.final_elite_push_count = 0
        self.final_elite_push_success_count = 0
        self.last_final_elite_push_step = -self.final_elite_push_cooldown_steps
        self.final_elite_push_blocked_after_fail = False
        self.final_elite_push_fail_archive_switch_count = -1
        self.final_elite_push_fail_gbest_update_count = -1
        self.archive_entry_counter = 0
        self.elite_archive_feasible = []
        self.elite_archive_frontier = []
        self.archive_switch_count = 0
        self.last_archive_switch_step = -self.archive_switch_cooldown_steps
        self.two_stage_telemetry = {
            action_idx: {
                "calls": 0,
                "raw_candidates": 0,
                "stage1_survivors": 0,
                "stage2_evaluated": 0,
                "fallbacks": 0,
                "raw_template_counts": {},
                "survivor_template_counts": {},
                "stage2_template_counts": {},
                "selected_template_counts": {},
                "phase_calls": {},
                "phase_selected_counts": {},
                "mode_calls": {},
                "mode_selected_counts": {},
                "local_proxy_mix_sum": 0.0,
                "local_proxy_mix_count": 0,
            }
            for action_idx in sorted(self.two_stage_heavy_action_ids)
        }

        logger.info(
            f"Runtime config applied | config: {self.runtime_config_name} | "
            f"accept_phase: {self.acceptance_phase_enabled} | "
            f"reward: {self.reward_profile} | "
            f"train_reward_mix: {self.reward_train_immediate_weight:.2f}/{self.reward_train_final_weight:.2f} | "
            f"two_stage_eval: {'learned' if self.two_stage_learned_evaluator_enabled else 'proxy'}"
        )

        self.progress_markers = [step / 10.0 for step in range(1, 11)]
        self.gbest = copy.deepcopy(base_gbest)
        self.true_gbest = copy.deepcopy(base_gbest)
        self.current_energy = np.inf
        self.best_energy = np.inf
        self._evaluate_solution(self.s)
        self._observe_feasible_state(self.s)
        self.current_energy = self.s.fitness
        if np.isfinite(self.current_energy):
            self._update_histogram(self.current_energy)
            self.energy_history.append(self.current_energy)
            self.modified_energy_history.append(self._tilde_energy(self.current_energy))

    def _sync_solution_metrics(self, solution, metrics):
        solution.fac_x = metrics["fac_x"]
        solution.fac_y = metrics["fac_y"]
        solution.fac_b = metrics["fac_b"]
        solution.fac_h = metrics["fac_h"]
        solution.fac_aspect_ratio = metrics["fac_aspect_ratio"]
        solution.lower_bounds = metrics["lower_bounds"]
        solution.upper_bounds = metrics["upper_bounds"]
        solution.infeasible_mask = metrics["infeasible_mask"]
        solution.D = metrics["D"]
        solution.TM = metrics["TM"]
        solution.MHC = metrics["mhc"]
        solution.fitness = metrics["cost"]
        solution.current_d_inf = metrics["d_inf"]
        solution.current_is_feasible = metrics["is_feasible"]
        solution.current_v_worst = self.worst_feasible_cost
        solution.feasible_solution_count = self.feasible_solution_count
        solution.best_feasible_cost = self.best_feasible_cost
        solution.worst_feasible_cost = self.worst_feasible_cost
        solution.best_fitness = self.best_feasible_cost
        solution.state = solution.constructState()

    def _phase_prefix(self, phase):
        return "elite_" if phase == "elite" else ""

    def _new_action_telemetry_entry(self, action_idx):
        return {
            "name": self.action_labels[action_idx],
            "selected": 0,
            "explore_selected": 0,
            "exploit_selected": 0,
            "accepted": 0,
            "candidate_improved": 0,
            "improved": 0,
            "final_improved": 0,
            "global_best_hits": 0,
            "local_search_triggered": 0,
            "local_search_improved": 0,
            "delta_sum": 0.0,
            "accepted_delta_sum": 0.0,
            "final_delta_sum": 0.0,
            "local_search_delta_sum": 0.0,
            "immediate_reward_sum": 0.0,
            "final_reward_sum": 0.0,
            "reward_uplift_sum": 0.0,
            "epsilon_sum": 0.0,
            "allowed_count_sum": 0.0,
            "elite_selected": 0,
            "elite_explore_selected": 0,
            "elite_exploit_selected": 0,
            "elite_accepted": 0,
            "elite_candidate_improved": 0,
            "elite_improved": 0,
            "elite_final_improved": 0,
            "elite_global_best_hits": 0,
            "elite_local_search_triggered": 0,
            "elite_local_search_improved": 0,
            "elite_delta_sum": 0.0,
            "elite_accepted_delta_sum": 0.0,
            "elite_final_delta_sum": 0.0,
            "elite_local_search_delta_sum": 0.0,
            "elite_immediate_reward_sum": 0.0,
            "elite_final_reward_sum": 0.0,
            "elite_reward_uplift_sum": 0.0,
            "elite_epsilon_sum": 0.0,
            "elite_allowed_count_sum": 0.0,
        }

    def _record_action_selection(
        self,
        action_idx,
        previous_cost,
        next_cost,
        phase="main",
        global_best=False,
        selection_mode=None,
        epsilon_value=None,
        allowed_count=None,
    ):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}selected"] += 1
        if selection_mode == "explore":
            stats[f"{prefix}explore_selected"] += 1
        elif selection_mode == "exploit":
            stats[f"{prefix}exploit_selected"] += 1
        if epsilon_value is not None and np.isfinite(epsilon_value):
            stats[f"{prefix}epsilon_sum"] += float(epsilon_value)
        if allowed_count is not None:
            stats[f"{prefix}allowed_count_sum"] += float(max(0, allowed_count))
        delta = 0.0
        if np.isfinite(previous_cost) and np.isfinite(next_cost):
            delta = float(previous_cost - next_cost)
            stats[f"{prefix}delta_sum"] += delta
            if delta > 1e-9:
                stats[f"{prefix}candidate_improved"] += 1
        if global_best:
            stats[f"{prefix}global_best_hits"] += 1
        return delta

    def _record_action_acceptance(self, action_idx, previous_cost, next_cost, improved=False, phase="main"):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}accepted"] += 1
        if improved:
            stats[f"{prefix}improved"] += 1
        if np.isfinite(previous_cost) and np.isfinite(next_cost):
            stats[f"{prefix}accepted_delta_sum"] += float(previous_cost - next_cost)

    def _record_action_final_outcome(
        self,
        action_idx,
        previous_cost,
        immediate_cost,
        final_cost,
        improved=False,
        local_search_triggered=False,
        local_search_improved=False,
        immediate_reward=None,
        final_reward=None,
        phase="main",
    ):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        if improved:
            stats[f"{prefix}final_improved"] += 1
        if local_search_triggered:
            stats[f"{prefix}local_search_triggered"] += 1
        if local_search_improved:
            stats[f"{prefix}local_search_improved"] += 1
        if np.isfinite(immediate_cost) and np.isfinite(final_cost):
            stats[f"{prefix}local_search_delta_sum"] += float(immediate_cost - final_cost)
        if np.isfinite(previous_cost) and np.isfinite(final_cost):
            stats[f"{prefix}final_delta_sum"] += float(previous_cost - final_cost)
        if immediate_reward is not None and np.isfinite(immediate_reward):
            stats[f"{prefix}immediate_reward_sum"] += float(immediate_reward)
        if final_reward is not None and np.isfinite(final_reward):
            stats[f"{prefix}final_reward_sum"] += float(final_reward)
        if (
            immediate_reward is not None
            and final_reward is not None
            and np.isfinite(immediate_reward)
            and np.isfinite(final_reward)
        ):
            stats[f"{prefix}reward_uplift_sum"] += float(final_reward - immediate_reward)

    def _record_action_global_best(self, action_idx, phase="main"):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}global_best_hits"] += 1

    def get_action_telemetry(self):
        summary = {}
        for action_idx, stats in self.action_telemetry.items():
            item = dict(stats)
            for prefix in ("", "elite_"):
                selected = stats[f"{prefix}selected"]
                accepted = stats[f"{prefix}accepted"]
                improved = stats[f"{prefix}improved"]
                final_improved = stats[f"{prefix}final_improved"]
                item[f"{prefix}accept_rate"] = 0.0 if selected == 0 else accepted / selected
                item[f"{prefix}candidate_improve_rate"] = 0.0 if selected == 0 else stats[f"{prefix}candidate_improved"] / selected
                item[f"{prefix}improve_rate"] = 0.0 if accepted == 0 else improved / accepted
                item[f"{prefix}final_improve_rate"] = 0.0 if selected == 0 else final_improved / selected
                item[f"{prefix}avg_delta"] = 0.0 if selected == 0 else stats[f"{prefix}delta_sum"] / selected
                item[f"{prefix}avg_accepted_delta"] = 0.0 if accepted == 0 else stats[f"{prefix}accepted_delta_sum"] / accepted
                item[f"{prefix}avg_final_delta"] = 0.0 if selected == 0 else stats[f"{prefix}final_delta_sum"] / selected
                item[f"{prefix}avg_local_search_delta"] = 0.0 if selected == 0 else stats[f"{prefix}local_search_delta_sum"] / selected
                item[f"{prefix}avg_immediate_reward"] = 0.0 if selected == 0 else stats[f"{prefix}immediate_reward_sum"] / selected
                item[f"{prefix}avg_final_reward"] = 0.0 if selected == 0 else stats[f"{prefix}final_reward_sum"] / selected
                item[f"{prefix}avg_reward_uplift"] = 0.0 if selected == 0 else stats[f"{prefix}reward_uplift_sum"] / selected
                item[f"{prefix}explore_rate"] = 0.0 if selected == 0 else stats[f"{prefix}explore_selected"] / selected
                item[f"{prefix}exploit_rate"] = 0.0 if selected == 0 else stats[f"{prefix}exploit_selected"] / selected
                item[f"{prefix}avg_epsilon"] = 0.0 if selected == 0 else stats[f"{prefix}epsilon_sum"] / selected
                item[f"{prefix}avg_allowed_count"] = 0.0 if selected == 0 else stats[f"{prefix}allowed_count_sum"] / selected
            summary[action_idx] = item
        return summary

    def format_action_telemetry(self):
        telemetry = self.get_action_telemetry()
        ordered = sorted(
            telemetry.items(),
            key=lambda item: (
                item[1]["global_best_hits"] + item[1]["elite_global_best_hits"],
                item[1]["avg_accepted_delta"] + item[1]["elite_avg_accepted_delta"],
                item[1]["accepted"] + item[1]["elite_accepted"],
            ),
            reverse=True,
        )
        lines = [
            (
                "Elite summary | triggers={triggers} | improvements={improvements} | total_gain={gain:.2f}"
            ).format(
                triggers=self.elite_trigger_count,
                improvements=self.elite_improvement_count,
                gain=self.elite_total_gain,
            )
        ]
        if getattr(self, "last_rl_agent_mode", None):
            lines.append(
                "RL runtime | agent={agent} | state_dim={state_dim}".format(
                    agent=self.last_rl_agent_mode,
                    state_dim=int(getattr(self, "last_state_dim", 0)),
                )
            )
        if getattr(self, "last_rl_agent_mode", None) == "dqn":
            lines.append(
                (
                    "DQN config | eps_start={eps_start:.3f} eps_min={eps_min:.3f} eps_decay={eps_decay:.6f} "
                    "warmup={warmup} batch={batch} replay={replay} update={update} target={target} "
                    "step_sched={step_sched} sched_warmup={sched_warmup} sched_decay_ratio={sched_decay:.3f}"
                ).format(
                    eps_start=float(self.dqn_epsilon_start),
                    eps_min=float(self.dqn_epsilon_min),
                    eps_decay=float(self.dqn_epsilon_decay),
                    warmup=int(self.dqn_warmup_steps),
                    batch=int(self.dqn_batch_size),
                    replay=int(self.dqn_replay_capacity),
                    update=int(self.dqn_update_every),
                    target=int(self.dqn_target_update_every),
                    step_sched=bool(self.dqn_step_epsilon_schedule),
                    sched_warmup=int(self.dqn_epsilon_schedule_warmup_steps),
                    sched_decay=float(self.dqn_epsilon_schedule_decay_ratio),
                )
            )
        for action_idx, stats in ordered:
            lines.append(
                (
                    "Action {idx} [{name}] | main sel={selected} exr={explore_rate:.1%}/{exploit_rate:.1%} "
                    "acc={accepted} ({accept_rate:.1%}) cand_imp={candidate_improved} final_imp={final_improved} "
                    "ls={local_search_triggered}/{local_search_improved} gbest={gbest} "
                    "avg_cand={avg_cand_delta:.2f} avg_final={avg_final_delta:.2f} avg_ls={avg_local_search_delta:.2f} "
                    "avg_r={avg_immediate_reward:.3f}->{avg_final_reward:.3f} ({avg_reward_uplift:+.3f}) "
                    "avg_eps={avg_epsilon:.3f} avg_allowed={avg_allowed:.2f} | "
                    "elite sel={elite_selected} acc={elite_accepted} ({elite_accept_rate:.1%}) "
                    "cand_imp={elite_candidate_improved} final_imp={elite_final_improved} "
                    "ls={elite_local_search_triggered}/{elite_local_search_improved} gbest={elite_gbest} "
                    "avg_cand={elite_avg_cand_delta:.2f} avg_final={elite_avg_final_delta:.2f} "
                    "avg_r={elite_avg_immediate_reward:.3f}->{elite_avg_final_reward:.3f} ({elite_avg_reward_uplift:+.3f})"
                ).format(
                    idx=action_idx,
                    name=stats["name"],
                    selected=stats["selected"],
                    explore_rate=stats["explore_rate"],
                    exploit_rate=stats["exploit_rate"],
                    accepted=stats["accepted"],
                    accept_rate=stats["accept_rate"],
                    candidate_improved=stats["candidate_improved"],
                    final_improved=stats["final_improved"],
                    local_search_triggered=stats["local_search_triggered"],
                    local_search_improved=stats["local_search_improved"],
                    gbest=stats["global_best_hits"],
                    avg_cand_delta=stats["avg_delta"],
                    avg_final_delta=stats["avg_final_delta"],
                    avg_local_search_delta=stats["avg_local_search_delta"],
                    avg_immediate_reward=stats["avg_immediate_reward"],
                    avg_final_reward=stats["avg_final_reward"],
                    avg_reward_uplift=stats["avg_reward_uplift"],
                    avg_epsilon=stats["avg_epsilon"],
                    avg_allowed=stats["avg_allowed_count"],
                    elite_selected=stats["elite_selected"],
                    elite_accepted=stats["elite_accepted"],
                    elite_accept_rate=stats["elite_accept_rate"],
                    elite_candidate_improved=stats["elite_candidate_improved"],
                    elite_final_improved=stats["elite_final_improved"],
                    elite_local_search_triggered=stats["elite_local_search_triggered"],
                    elite_local_search_improved=stats["elite_local_search_improved"],
                    elite_gbest=stats["elite_global_best_hits"],
                    elite_avg_cand_delta=stats["elite_avg_delta"],
                    elite_avg_final_delta=stats["elite_avg_final_delta"],
                    elite_avg_immediate_reward=stats["elite_avg_immediate_reward"],
                    elite_avg_final_reward=stats["elite_avg_final_reward"],
                    elite_avg_reward_uplift=stats["elite_avg_reward_uplift"],
                )
            )
        if self.two_stage_heavy_actions_enabled and self.two_stage_telemetry:
            lines.append(
                "Two-stage summary | enabled=True | actions={actions}".format(
                    actions="/".join(str(int(action_idx)) for action_idx in sorted(self.two_stage_telemetry)),
                )
            )
            for action_idx in sorted(self.two_stage_telemetry):
                stats = self.two_stage_telemetry[action_idx]
                calls = max(1, int(stats["calls"]))
                selected_templates = self._format_two_stage_template_counter(
                    stats.get("selected_template_counts", {}),
                    topn=3,
                )
                survivor_templates = self._format_two_stage_template_counter(
                    stats.get("survivor_template_counts", {}),
                    topn=3,
                )
                phase_calls = self._format_two_stage_template_counter(
                    stats.get("phase_calls", {}),
                    topn=3,
                )
                phase_selected = self._format_two_stage_template_counter(
                    stats.get("phase_selected_counts", {}),
                    topn=3,
                )
                mode_calls = self._format_two_stage_template_counter(
                    stats.get("mode_calls", {}),
                    topn=4,
                )
                mode_selected = self._format_two_stage_template_counter(
                    stats.get("mode_selected_counts", {}),
                    topn=4,
                )
                local_proxy_mix_count = max(1, int(stats.get("local_proxy_mix_count", 0)))
                avg_local_proxy_mix = float(stats.get("local_proxy_mix_sum", 0.0)) / float(local_proxy_mix_count)
                lines.append(
                    (
                        "Two-stage {idx} [{name}] | calls={calls} avg_raw={avg_raw:.2f} "
                        "avg_survivors={avg_survivors:.2f} avg_stage2={avg_stage2:.2f} "
                        "fallbacks={fallbacks} | avg_local_mix={avg_local_mix:.3f} "
                        "| phases={phase_calls} | selected_phases={phase_selected} "
                        "| modes={mode_calls} | selected_modes={mode_selected} "
                        "| top_survivors={top_survivors} | top_selected={top_selected}"
                    ).format(
                        idx=int(action_idx),
                        name=self.action_labels.get(int(action_idx), f"action_{int(action_idx)}"),
                        calls=int(stats["calls"]),
                        avg_raw=float(stats["raw_candidates"]) / float(calls),
                        avg_survivors=float(stats["stage1_survivors"]) / float(calls),
                        avg_stage2=float(stats["stage2_evaluated"]) / float(calls),
                        fallbacks=int(stats["fallbacks"]),
                        avg_local_mix=avg_local_proxy_mix,
                        phase_calls=phase_calls,
                        phase_selected=phase_selected,
                        mode_calls=mode_calls,
                        mode_selected=mode_selected,
                        top_survivors=survivor_templates,
                        top_selected=selected_templates,
                )
            )
        return lines

    def build_action_telemetry_rows(
        self,
        exp_instance,
        exp_algorithm,
        exp_remark,
        run_seed,
        run_index,
        is_valid,
        best_energy,
        start_time,
        end_time,
        fast_time,
    ):
        telemetry = self.get_action_telemetry()
        rows = []
        run_duration = None
        fast_duration = None
        if start_time is not None and end_time is not None:
            run_duration = (end_time - start_time).total_seconds()
        if start_time is not None and fast_time is not None:
            fast_duration = (fast_time - start_time).total_seconds()
        common = {
            "instance": exp_instance,
            "algorithm": exp_algorithm,
            "remark": exp_remark,
            "run_seed": run_seed,
            "run_index": run_index,
            "agent_mode": getattr(self, "last_rl_agent_mode", ""),
            "state_dim": int(getattr(self, "last_state_dim", 0)),
            "runtime_config": getattr(self, "runtime_config_name", ""),
            "best_energy": float(best_energy),
            "is_valid": bool(is_valid),
            "gbest_updates": int(self.gbest_update_count),
            "elite_triggers": int(self.elite_trigger_count),
            "elite_improvements": int(self.elite_improvement_count),
            "elite_total_gain": float(self.elite_total_gain),
            "start_time": start_time,
            "end_time": end_time,
            "fast_time": fast_time,
            "run_duration_sec": run_duration,
            "fast_best_sec": fast_duration,
        }
        if getattr(self, "last_rl_agent_mode", None) == "dqn":
            common.update(
                {
                    "dqn_epsilon_start": float(self.dqn_epsilon_start),
                    "dqn_epsilon_min": float(self.dqn_epsilon_min),
                    "dqn_epsilon_decay": float(self.dqn_epsilon_decay),
                    "dqn_gamma": float(self.dqn_gamma),
                    "dqn_lr": float(self.dqn_lr),
                    "dqn_batch_size": int(self.dqn_batch_size),
                    "dqn_replay_capacity": int(self.dqn_replay_capacity),
                    "dqn_warmup_steps": int(self.dqn_warmup_steps),
                    "dqn_update_every": int(self.dqn_update_every),
                    "dqn_target_update_every": int(self.dqn_target_update_every),
                    "dqn_idqn_k": int(self.dqn_idqn_k),
                    "dqn_embedding_dim": int(self.dqn_embedding_dim),
                    "dqn_hidden_dim": int(self.dqn_hidden_dim),
                    "dqn_grad_clip": float(self.dqn_grad_clip),
                    "dqn_step_epsilon_schedule": bool(self.dqn_step_epsilon_schedule),
                    "dqn_epsilon_schedule_warmup_steps": int(self.dqn_epsilon_schedule_warmup_steps),
                    "dqn_epsilon_schedule_decay_ratio": float(self.dqn_epsilon_schedule_decay_ratio),
                    "reward_train_immediate_weight": float(self.reward_train_immediate_weight),
                    "reward_train_final_weight": float(self.reward_train_final_weight),
                }
            )
        for action_idx in sorted(telemetry):
            row = dict(common)
            row["action_idx"] = int(action_idx)
            row.update(telemetry[action_idx])
            rows.append(row)
        return rows

    def _evaluate_solution(self, solution):
        evaluator = FBSUtil.evaluate_layout_fast if self.use_fast_evaluate else FBSUtil.evaluate_layout
        metrics = evaluator(
            solution.fbs_model,
            solution.areas,
            solution.H,
            solution.F,
            solution.aspect_limits,
            v_worst=self.worst_feasible_cost,
            k_penalty=self.k_penalty,
            distance_metric="manhattan",
        )
        self._sync_solution_metrics(solution, metrics)
        return metrics

    def _observe_feasible_state(self, solution):
        if not getattr(solution, "current_is_feasible", False):
            return False
        cost = float(solution.fitness)
        self.feasible_solution_count += 1
        self.recent_feasible_costs.append(cost)
        if self.worst_feasible_cost is None:
            self.worst_feasible_cost = cost
        else:
            self.worst_feasible_cost = max(float(self.worst_feasible_cost), cost)

        improved = False
        if cost < self.best_feasible_cost:
            self.best_feasible_cost = cost
            self.best_feasible_solution = copy.deepcopy(solution)
            self.gbest = copy.deepcopy(solution)
            self.true_gbest = copy.deepcopy(solution)
            self.best_energy = cost
            self.gbest_update_count += 1
            self.best_history.append(cost)
            improved = True

        for env_obj in (solution, self.s, self.gbest, self.true_gbest):
            if env_obj is None:
                continue
            env_obj.feasible_solution_count = self.feasible_solution_count
            env_obj.best_feasible_cost = self.best_feasible_cost
            env_obj.worst_feasible_cost = self.worst_feasible_cost
            env_obj.best_fitness = self.best_feasible_cost
            env_obj.current_v_worst = self.worst_feasible_cost
        self._observe_archive_candidate(solution)
        return improved

    def _constraint_violation(self, solution):
        short_side = np.minimum(solution.fac_b, solution.fac_h)
        long_side = np.maximum(solution.fac_b, solution.fac_h)
        short_violation = np.maximum(solution.lower_bounds - short_side, 0.0)
        long_violation = np.maximum(long_side - solution.upper_bounds, 0.0)
        return float(np.sum(short_violation + long_violation))

    def _layout_score(self, solution):
        mhc = float(solution.MHC) if np.isfinite(solution.MHC) else float("inf")
        return (
            0 if solution.current_is_feasible else 1,
            int(solution.current_d_inf),
            self._constraint_violation(solution),
            mhc,
        )

    def _archive_all_entries(self):
        return list(self.elite_archive_feasible) + list(self.elite_archive_frontier)

    def _build_layout_reference(self, permutation, bay):
        perm = np.asarray(permutation, dtype=int).reshape(-1)
        bay_flags = np.asarray(bay, dtype=int).reshape(-1)
        if perm.size == 0 or perm.size != bay_flags.size:
            return None
        bay_flags = bay_flags.copy()
        bay_flags[-1] = 1
        positions = np.empty(perm.size, dtype=int)
        positions[np.asarray(perm, dtype=int) - 1] = np.arange(perm.size, dtype=int)
        facility_to_bay = np.empty(perm.size, dtype=int)
        start_idx = 0
        bay_idx = 0
        for end_idx, marker in enumerate(bay_flags.tolist()):
            if int(marker) != 1:
                continue
            current_bay = perm[start_idx : end_idx + 1]
            if current_bay.size > 0:
                facility_to_bay[np.asarray(current_bay, dtype=int) - 1] = int(bay_idx)
            start_idx = end_idx + 1
            bay_idx += 1
        boundary_indices = np.where(bay_flags == 1)[0].astype(int)
        return {
            "permutation": perm.astype(int),
            "bay": bay_flags.astype(int),
            "positions": positions.astype(int),
            "facility_to_bay": facility_to_bay.astype(int),
            "boundary_indices": boundary_indices,
        }

    def _build_layout_reference_from_solution(self, solution):
        return self._build_layout_reference(
            getattr(solution.fbs_model, "permutation", []),
            getattr(solution.fbs_model, "bay", []),
        )

    def _layout_distance(self, left_reference, right_reference):
        if left_reference is None or right_reference is None:
            return 1.0
        left_positions = np.asarray(left_reference["positions"], dtype=int).reshape(-1)
        right_positions = np.asarray(right_reference["positions"], dtype=int).reshape(-1)
        if left_positions.size == 0 or left_positions.size != right_positions.size:
            return 1.0
        n = int(left_positions.size)
        if n <= 1:
            return 0.0
        left_bay_ids = np.asarray(left_reference["facility_to_bay"], dtype=int).reshape(-1)
        right_bay_ids = np.asarray(right_reference["facility_to_bay"], dtype=int).reshape(-1)
        left_same_bay = left_bay_ids[:, None] == left_bay_ids[None, :]
        right_same_bay = right_bay_ids[:, None] == right_bay_ids[None, :]
        upper_idx = np.triu_indices(n, k=1)
        same_bay_diff = float(
            np.mean(left_same_bay[upper_idx] != right_same_bay[upper_idx])
        ) if upper_idx[0].size > 0 else 0.0
        position_diff = float(
            np.mean(np.abs(left_positions - right_positions))
        ) / float(max(1, n - 1))
        left_boundaries = set(int(value) for value in np.asarray(left_reference["boundary_indices"], dtype=int).tolist())
        right_boundaries = set(int(value) for value in np.asarray(right_reference["boundary_indices"], dtype=int).tolist())
        boundary_union = max(len(left_boundaries | right_boundaries), 1)
        boundary_diff = float(len(left_boundaries ^ right_boundaries)) / float(boundary_union)
        return float(
            min(
                max(
                    0.45 * same_bay_diff
                    + 0.35 * position_diff
                    + 0.20 * boundary_diff,
                    0.0,
                ),
                1.0,
            )
        )

    def _make_archive_entry(self, solution, lane):
        layout_reference = self._build_layout_reference_from_solution(solution)
        if layout_reference is None:
            return None
        self.archive_entry_counter += 1
        is_feasible = bool(getattr(solution, "current_is_feasible", False))
        fitness = float(getattr(solution, "fitness", np.inf))
        mhc = float(getattr(solution, "MHC", np.inf))
        d_inf = int(getattr(solution, "current_d_inf", 10**9))
        violation = float(self._constraint_violation(solution))
        entry = {
            "id": int(self.archive_entry_counter),
            "lane": str(lane),
            "solution": copy.deepcopy(solution),
            "layout_reference": layout_reference,
            "is_feasible": is_feasible,
            "fitness": fitness,
            "mhc": mhc,
            "d_inf": d_inf,
            "violation": violation,
            "created_step": int(self.current_global_step),
            "last_used_step": -10**9,
            "use_count": 0,
        }
        if str(lane) == "feasible":
            entry["lane_quality_key"] = (
                float(fitness),
                float(mhc),
                float(violation),
            )
        else:
            entry["lane_quality_key"] = (
                int(d_inf),
                float(violation),
                float(fitness),
                float(mhc),
            )
        return entry

    @staticmethod
    def _archive_entry_better(candidate_entry, target_entry):
        return candidate_entry["lane_quality_key"] < target_entry["lane_quality_key"]

    def _archive_worst_entry_index(self, archive_entries):
        if not archive_entries:
            return None
        return max(
            range(len(archive_entries)),
            key=lambda idx: archive_entries[idx]["lane_quality_key"],
        )

    def _archive_nearest_entry(self, candidate_entry, archive_entries):
        if not archive_entries:
            return None, 1.0
        best_idx = None
        best_distance = float("inf")
        for idx, archive_entry in enumerate(archive_entries):
            distance = self._layout_distance(
                candidate_entry["layout_reference"],
                archive_entry["layout_reference"],
            )
            if distance < best_distance:
                best_idx = int(idx)
                best_distance = float(distance)
        return best_idx, float(best_distance)

    def _upsert_feasible_archive_entry(self, solution):
        if (not self.elite_archive_enabled) or self.elite_archive_feasible_capacity <= 0:
            return False
        candidate_entry = self._make_archive_entry(solution, lane="feasible")
        if candidate_entry is None:
            return False
        feasible_entries = self.elite_archive_feasible
        all_entries = self._archive_all_entries()
        near_duplicate_threshold = (
            float(self.elite_archive_min_diversity) * float(self.elite_archive_near_duplicate_ratio)
        )
        nearest_same_idx, nearest_same_distance = self._archive_nearest_entry(candidate_entry, feasible_entries)
        _, nearest_all_distance = self._archive_nearest_entry(candidate_entry, all_entries)

        if nearest_all_distance < near_duplicate_threshold:
            if (
                nearest_same_idx is not None
                and self._archive_entry_better(candidate_entry, feasible_entries[nearest_same_idx])
            ):
                feasible_entries[nearest_same_idx] = candidate_entry
                return True
            return False

        if len(feasible_entries) < int(self.elite_archive_feasible_capacity):
            feasible_entries.append(candidate_entry)
            return True

        worst_idx = self._archive_worst_entry_index(feasible_entries)
        if worst_idx is None:
            feasible_entries.append(candidate_entry)
            return True

        if (
            nearest_same_idx is not None
            and nearest_same_distance < float(self.elite_archive_min_diversity)
            and self._archive_entry_better(candidate_entry, feasible_entries[nearest_same_idx])
        ):
            feasible_entries[nearest_same_idx] = candidate_entry
            return True

        worst_entry = feasible_entries[worst_idx]
        if self._archive_entry_better(candidate_entry, worst_entry):
            feasible_entries[worst_idx] = candidate_entry
            return True

        if nearest_all_distance >= float(self.elite_archive_min_diversity):
            worst_cost = float(worst_entry["fitness"])
            slack_cost = worst_cost * (1.0 + float(self.elite_archive_feasible_slack_ratio))
            if np.isfinite(candidate_entry["fitness"]) and candidate_entry["fitness"] <= slack_cost:
                feasible_entries[worst_idx] = candidate_entry
                return True
        return False

    def _should_consider_frontier_archive(self, solution):
        if not self.elite_archive_enabled or self.elite_archive_frontier_capacity <= 0:
            return False
        if bool(getattr(solution, "current_is_feasible", False)):
            return False
        if not np.isfinite(getattr(solution, "fitness", np.inf)):
            return False
        d_inf = int(getattr(solution, "current_d_inf", 10**9))
        return 0 < d_inf <= int(self.elite_archive_frontier_max_d_inf)

    def _upsert_frontier_archive_entry(self, solution):
        if not self._should_consider_frontier_archive(solution):
            return False
        candidate_entry = self._make_archive_entry(solution, lane="frontier")
        if candidate_entry is None:
            return False
        frontier_entries = self.elite_archive_frontier
        all_entries = self._archive_all_entries()
        near_duplicate_threshold = (
            float(self.elite_archive_min_diversity) * float(self.elite_archive_near_duplicate_ratio)
        )
        nearest_same_idx, nearest_same_distance = self._archive_nearest_entry(candidate_entry, frontier_entries)
        _, nearest_all_distance = self._archive_nearest_entry(candidate_entry, all_entries)

        if nearest_all_distance < near_duplicate_threshold:
            if (
                nearest_same_idx is not None
                and self._archive_entry_better(candidate_entry, frontier_entries[nearest_same_idx])
            ):
                frontier_entries[nearest_same_idx] = candidate_entry
                return True
            return False

        if len(frontier_entries) < int(self.elite_archive_frontier_capacity):
            frontier_entries.append(candidate_entry)
            return True

        worst_idx = self._archive_worst_entry_index(frontier_entries)
        if worst_idx is None:
            frontier_entries.append(candidate_entry)
            return True

        if (
            nearest_same_idx is not None
            and nearest_same_distance < float(self.elite_archive_min_diversity)
            and self._archive_entry_better(candidate_entry, frontier_entries[nearest_same_idx])
        ):
            frontier_entries[nearest_same_idx] = candidate_entry
            return True

        worst_entry = frontier_entries[worst_idx]
        if self._archive_entry_better(candidate_entry, worst_entry):
            frontier_entries[worst_idx] = candidate_entry
            return True

        if nearest_all_distance >= float(self.elite_archive_min_diversity):
            if (
                int(candidate_entry["d_inf"]) <= int(worst_entry["d_inf"])
                and float(candidate_entry["violation"]) <= float(worst_entry["violation"]) * 1.10
            ):
                frontier_entries[worst_idx] = candidate_entry
                return True
        return False

    def _observe_archive_candidate(self, solution):
        if not self.elite_archive_enabled:
            return False
        if bool(getattr(solution, "current_is_feasible", False)):
            return self._upsert_feasible_archive_entry(solution)
        return self._upsert_frontier_archive_entry(solution)

    def _mark_archive_entry_used(self, archive_entry, global_step):
        if archive_entry is None:
            return
        archive_entry["last_used_step"] = int(global_step)
        archive_entry["use_count"] = int(archive_entry.get("use_count", 0)) + 1

    def _archive_switch_phase(self):
        progress_ratio = float(self.current_progress_ratio)
        if progress_ratio < float(self.archive_switch_progress_gate_ratio):
            return "off"
        if progress_ratio < float(self.archive_switch_full_progress_ratio):
            return "ramp"
        return "late"

    def _effective_archive_switch_controls(self):
        phase = self._archive_switch_phase()
        base_no_improve = max(int(self.archive_switch_no_improve), 1)
        base_cooldown = max(int(self.archive_switch_cooldown_steps), 1)
        base_min_score = float(self.archive_switch_min_score)
        target_temp_ratio = float(self.archive_switch_target_temp_ratio)

        if phase == "ramp":
            return {
                "phase": phase,
                "no_improve": max(
                    1,
                    int(math.ceil(base_no_improve * float(self.archive_switch_ramp_no_improve_multiplier))),
                ),
                "cooldown": max(
                    1,
                    int(math.ceil(base_cooldown * float(self.archive_switch_ramp_cooldown_multiplier))),
                ),
                "min_score": max(
                    0.0,
                    base_min_score + float(self.archive_switch_ramp_min_score_bonus),
                ),
                "target_temp_ratio": min(target_temp_ratio, 0.32),
            }

        if phase == "late":
            return {
                "phase": phase,
                "no_improve": max(
                    1,
                    int(math.floor(base_no_improve * float(self.archive_switch_late_no_improve_multiplier))),
                ),
                "cooldown": max(
                    1,
                    int(math.floor(base_cooldown * float(self.archive_switch_late_cooldown_multiplier))),
                ),
                "min_score": max(
                    0.0,
                    base_min_score + float(self.archive_switch_late_min_score_bonus),
                ),
                "target_temp_ratio": target_temp_ratio,
            }

        return {
            "phase": phase,
            "no_improve": base_no_improve,
            "cooldown": base_cooldown,
            "min_score": base_min_score,
            "target_temp_ratio": target_temp_ratio,
        }

    def _elite_phase(self):
        progress_ratio = float(self.current_progress_ratio)
        if (
            progress_ratio >= float(self.elite_final_progress_ratio)
            or self.no_improve_steps >= int(self.elite_final_no_improve)
        ):
            return "final"
        if (
            progress_ratio >= float(self.elite_late_progress_ratio)
            or self.no_improve_steps >= int(self.elite_late_no_improve)
        ):
            return "late"
        return "normal"

    def _effective_elite_controls(self):
        phase = self._elite_phase()
        base_rounds = max(int(self.elite_max_rounds), 1)
        base_seed_limit = max(1, min(int(self.elite_archive_seed_count), max(1, int(self.elite_archive_feasible_capacity))))
        trial_map = {int(action_idx): int(max(1, trial_count)) for action_idx, trial_count in self.elite_action_trials.items()}

        if phase == "late":
            trial_multiplier = float(self.elite_late_trial_multiplier)
            total_rounds = max(1, int(math.ceil(base_rounds * float(self.elite_late_round_multiplier))))
            seed_limit = max(1, min(int(self.elite_late_seed_count), max(1, int(self.elite_archive_feasible_capacity))))
        elif phase == "final":
            trial_multiplier = float(self.elite_final_trial_multiplier)
            total_rounds = max(1, int(math.ceil(base_rounds * float(self.elite_final_round_multiplier))))
            seed_limit = max(1, min(int(self.elite_final_seed_count), max(1, int(self.elite_archive_feasible_capacity))))
        else:
            trial_multiplier = 1.0
            total_rounds = base_rounds
            seed_limit = max(1, min(1, base_seed_limit))

        for action_idx in list(trial_map.keys()):
            scaled_trials = int(math.ceil(float(trial_map[action_idx]) * trial_multiplier))
            if phase == "late" and action_idx in {10, 11, 14}:
                scaled_trials += 1
            if phase == "final":
                if action_idx in {10, 11}:
                    scaled_trials += 2
                elif action_idx == 14:
                    scaled_trials += 1
                elif action_idx == 0:
                    scaled_trials += 1
            trial_map[action_idx] = max(1, scaled_trials)

        return {
            "phase": phase,
            "seed_limit": max(1, seed_limit),
            "total_rounds": total_rounds,
            "trial_map": trial_map,
        }

    def _should_takeover_after_elite(self, elite_result, elite_phase):
        if elite_result is None:
            return False
        elite_cost = float(getattr(elite_result, "fitness", np.inf))
        if not np.isfinite(elite_cost):
            return False
        current_solution = getattr(self, "s", None)
        if current_solution is None or not bool(getattr(current_solution, "current_is_feasible", False)):
            return True
        current_cost = float(getattr(current_solution, "fitness", np.inf))
        if not np.isfinite(current_cost):
            return True
        if elite_cost + 1e-9 >= current_cost:
            return False
        gain_ratio = max(current_cost - elite_cost, 0.0) / max(abs(current_cost), 1.0)
        if elite_phase == "final":
            required_ratio = float(self.elite_final_chain_takeover_gain_ratio)
        elif elite_phase == "late":
            required_ratio = float(self.elite_late_chain_takeover_gain_ratio)
        else:
            required_ratio = float(self.elite_chain_takeover_gain_ratio)
        return bool(gain_ratio + 1e-12 >= required_ratio)

    def _select_elite_seed_entries(self, seed_limit=None):
        if self.elite_archive_enabled and self.elite_archive_feasible:
            ordered_entries = sorted(
                self.elite_archive_feasible,
                key=lambda item: item["lane_quality_key"],
            )
            selected_entries = [ordered_entries[0]]
            max_seed_limit = max(1, int(seed_limit or self.elite_archive_seed_count))
            allow_multi_seed = bool(
                len(ordered_entries) > 1
                and max_seed_limit > 1
                and (
                    float(self.current_progress_ratio) >= float(self.elite_archive_multi_seed_progress_gate_ratio)
                    or self.no_improve_steps >= int(self.elite_archive_multi_seed_no_improve)
                )
            )
            if allow_multi_seed:
                current_reference = self._build_layout_reference_from_solution(self.s)
                remaining_entries = [entry for entry in ordered_entries[1:] if entry["id"] != ordered_entries[0]["id"]]
                while remaining_entries and len(selected_entries) < max_seed_limit:
                    selected_references = [entry["layout_reference"] for entry in selected_entries]
                    next_entry = max(
                        remaining_entries,
                        key=lambda entry: (
                            min(
                                self._layout_distance(entry["layout_reference"], selected_reference)
                                for selected_reference in selected_references
                            ),
                            self._layout_distance(entry["layout_reference"], current_reference),
                            -float(entry["fitness"]),
                        ),
                    )
                    selected_entries.append(next_entry)
                    remaining_entries = [entry for entry in remaining_entries if entry["id"] != next_entry["id"]]
            return selected_entries[:max_seed_limit]
        if self.best_feasible_solution is None:
            return []
        fallback_entry = self._make_archive_entry(self.best_feasible_solution, lane="feasible")
        return [fallback_entry] if fallback_entry is not None else []

    def _archive_switch_quality_score(self, archive_entry):
        if archive_entry is None:
            return 0.0
        if bool(archive_entry.get("is_feasible", False)):
            current_cost = float(getattr(self.s, "fitness", np.inf))
            archive_cost = float(archive_entry.get("fitness", np.inf))
            if np.isfinite(current_cost) and np.isfinite(archive_cost):
                gap_ratio = max(current_cost - archive_cost, 0.0) / max(abs(current_cost), 1.0)
            else:
                gap_ratio = 0.0
            best_gap_ratio = 0.0
            if np.isfinite(self.best_feasible_cost) and np.isfinite(archive_cost):
                best_gap_ratio = max(float(self.best_feasible_cost) - archive_cost, 0.0) / max(abs(float(self.best_feasible_cost)), 1.0)
            return float(1.0 + gap_ratio + 0.5 * best_gap_ratio)
        return float(
            0.8 / (1.0 + max(int(archive_entry.get("d_inf", 10**9)), 0))
            + 0.6 / (1.0 + max(float(archive_entry.get("violation", np.inf)), 0.0))
        )

    def _current_archive_switch_quality_score(self):
        if bool(getattr(self.s, "current_is_feasible", False)):
            return 1.0
        current_d_inf = max(int(getattr(self.s, "current_d_inf", 10**9)), 0)
        current_violation = max(float(self._constraint_violation(self.s)), 0.0)
        return float(
            0.8 / (1.0 + current_d_inf)
            + 0.6 / (1.0 + current_violation)
        )

    def _select_mid_structural_shot_seed_solutions(self):
        seeds = []
        current_solution = copy.deepcopy(self.s)
        self._evaluate_solution(current_solution)
        if current_solution.current_is_feasible and np.isfinite(current_solution.fitness):
            seeds.append(current_solution)

        current_reference = self._build_layout_reference_from_solution(current_solution)
        if current_reference is None:
            return seeds

        archive_candidates = []
        if self.elite_archive_enabled and self.elite_archive_feasible:
            max_allowed_cost = float("inf")
            if np.isfinite(self.best_feasible_cost):
                max_allowed_cost = float(self.best_feasible_cost) * (
                    1.0 + float(self.mid_structural_shot_archive_slack_ratio)
                )
            for archive_entry in self.elite_archive_feasible:
                archive_cost = float(archive_entry.get("fitness", np.inf))
                if np.isfinite(max_allowed_cost) and np.isfinite(archive_cost) and archive_cost > max_allowed_cost:
                    continue
                diversity = self._layout_distance(current_reference, archive_entry["layout_reference"])
                archive_candidates.append((float(diversity), archive_entry))

        if archive_candidates:
            archive_candidates.sort(
                key=lambda item: (
                    item[0],
                    -float(item[1].get("fitness", np.inf)),
                ),
                reverse=True,
            )
            best_diversity, best_entry = archive_candidates[0]
            if float(best_diversity) >= float(self.mid_structural_shot_diversity_floor):
                seeds.append(copy.deepcopy(best_entry["solution"]))
        elif self.best_feasible_solution is not None:
            best_solution = copy.deepcopy(self.best_feasible_solution)
            self._evaluate_solution(best_solution)
            if best_solution.current_is_feasible and np.isfinite(best_solution.fitness):
                diversity = self._layout_distance(
                    current_reference,
                    self._build_layout_reference_from_solution(best_solution),
                )
                if float(diversity) >= float(self.mid_structural_shot_diversity_floor):
                    seeds.append(best_solution)

        return seeds

    def _attempt_mid_structural_shot(self, global_step, total_steps, fast_time):
        if not self.mid_structural_shot_enabled:
            return fast_time, False
        if self.mid_structural_shot_count >= int(self.mid_structural_shot_max_count):
            return fast_time, False
        if self.mid_structural_shot_blocked_after_fail:
            if (
                self.archive_switch_count <= int(self.mid_structural_shot_fail_archive_switch_count)
                and self.gbest_update_count <= int(self.mid_structural_shot_fail_gbest_update_count)
            ):
                return fast_time, False
            self.mid_structural_shot_blocked_after_fail = False
        if not getattr(self.s, "current_is_feasible", False):
            return fast_time, False
        if not np.isfinite(getattr(self.s, "fitness", np.inf)):
            return fast_time, False
        progress_ratio = float(global_step) / float(max(1, total_steps))
        if progress_ratio < float(self.mid_structural_shot_start_ratio):
            return fast_time, False
        if progress_ratio > float(self.mid_structural_shot_end_ratio):
            return fast_time, False
        if self.no_improve_steps < int(self.mid_structural_shot_no_improve):
            return fast_time, False
        if global_step - self.last_mid_structural_shot_step < int(self.mid_structural_shot_cooldown_steps):
            return fast_time, False
        if global_step - self.last_archive_switch_step < int(self.mid_structural_shot_post_switch_cooldown_steps):
            return fast_time, False

        self.last_mid_structural_shot_step = int(global_step)
        self.mid_structural_shot_count += 1

        current_reference = self._build_layout_reference_from_solution(self.s)
        current_rank = self._candidate_rank_key(self.s)
        current_cost = float(getattr(self.s, "fitness", np.inf))
        raw_multiplier = float(self.mid_structural_shot_raw_multiplier)
        eval_bonus = int(self.mid_structural_shot_eval_bonus)

        best_candidate = None
        best_key = None
        best_diversity = 0.0
        for seed_solution in self._select_mid_structural_shot_seed_solutions():
            if not getattr(seed_solution, "current_is_feasible", False):
                continue
            if not np.isfinite(getattr(seed_solution, "fitness", np.inf)):
                continue
            for action_idx in self.mid_structural_shot_action_ids:
                base_raw, base_eval = self._effective_two_stage_budgets(action_idx)
                raw_budget = max(base_raw, int(math.ceil(float(base_raw) * raw_multiplier)))
                eval_budget = max(base_eval, int(base_eval + eval_bonus))
                for _ in range(int(self.mid_structural_shot_trials_per_action)):
                    candidate = self._generate_two_stage_heavy_candidate(
                        seed_solution,
                        action_idx,
                        raw_budget_override=raw_budget,
                        eval_budget_override=eval_budget,
                        phase_tag="mid_shot",
                        enable_local_proxy=True,
                    )
                    if candidate is None or not np.isfinite(getattr(candidate, "fitness", np.inf)):
                        continue
                    candidate_reference = self._build_layout_reference_from_solution(candidate)
                    diversity = self._layout_distance(current_reference, candidate_reference)
                    candidate_key = self._candidate_rank_key(candidate) + (-float(diversity),)
                    if best_candidate is None or candidate_key < best_key:
                        best_candidate = candidate
                        best_key = candidate_key
                        best_diversity = float(diversity)

        if best_candidate is None:
            self.mid_structural_shot_blocked_after_fail = True
            self.mid_structural_shot_fail_archive_switch_count = int(self.archive_switch_count)
            self.mid_structural_shot_fail_gbest_update_count = int(self.gbest_update_count)
            return fast_time, False

        if (
            self.mid_structural_shot_guided_local_search
            and getattr(best_candidate, "current_is_feasible", False)
            and np.isfinite(getattr(best_candidate, "fitness", np.inf))
        ):
            refined_candidate = self._greedy_local_search(
                copy.deepcopy(best_candidate),
                enable_guided=True,
                guided_topk=self.mid_structural_shot_guided_topk,
                guided_max_iters=self.mid_structural_shot_guided_max_iters,
            )
            if (
                getattr(refined_candidate, "current_is_feasible", False)
                and np.isfinite(getattr(refined_candidate, "fitness", np.inf))
                and self._candidate_rank_key(refined_candidate) <= self._candidate_rank_key(best_candidate)
            ):
                best_candidate = refined_candidate

        adopt = False
        if self._candidate_rank_key(best_candidate) < current_rank:
            adopt = True
        else:
            accept, _, _, _ = self._accept_candidate(current_cost, float(best_candidate.fitness))
            if accept and best_diversity >= float(self.mid_structural_shot_diversity_floor):
                adopt = True

        if not adopt:
            self.mid_structural_shot_blocked_after_fail = True
            self.mid_structural_shot_fail_archive_switch_count = int(self.archive_switch_count)
            self.mid_structural_shot_fail_gbest_update_count = int(self.gbest_update_count)
            return fast_time, False

        best_before = float(self.best_feasible_cost)
        self.s = copy.deepcopy(best_candidate)
        self.current_energy = self.s.fitness
        if self.s.current_is_feasible:
            self._observe_feasible_state(self.s)
        else:
            self._observe_archive_candidate(self.s)
        self.no_improve_steps = 0
        self.mid_structural_shot_success_count += 1
        self.last_mid_structural_shot_success_step = int(global_step)
        self.mid_structural_shot_blocked_after_fail = False
        self.T = max(self.T, float(self.mid_structural_shot_target_temp_ratio) * self.T_initial, self.T_min)
        self._update_histogram(self.current_energy)
        self.energy_history.append(self.current_energy)
        self.modified_energy_history.append(self._tilde_energy(self.current_energy))
        if np.isfinite(self.best_feasible_cost) and self.best_feasible_cost + 1e-9 < best_before:
            fast_time = datetime.datetime.now()
        return fast_time, True

    def _attempt_archive_switch(self, global_step):
        if not self.archive_switch_enabled or not self.elite_archive_enabled:
            return False
        switch_controls = self._effective_archive_switch_controls()
        if switch_controls["phase"] == "off":
            return False
        if global_step - self.last_archive_switch_step < int(switch_controls["cooldown"]):
            return False
        if self.no_improve_steps < int(switch_controls["no_improve"]):
            return False
        if self.accept_rate_window > float(self.archive_switch_accept_rate_threshold):
            return False
        archive_entries = self._archive_all_entries()
        if not archive_entries:
            return False

        current_reference = self._build_layout_reference_from_solution(self.s)
        current_is_feasible = bool(getattr(self.s, "current_is_feasible", False))
        current_cost = float(getattr(self.s, "fitness", np.inf))
        if (
            switch_controls["phase"] == "late"
            and current_is_feasible
            and global_step - int(self.last_mid_structural_shot_success_step) <= int(self.mid_structural_shot_late_archive_lock_steps)
            and self._relative_gap_to_best(self.s) <= float(self.mid_structural_shot_late_archive_lock_gap_ratio)
        ):
            return False
        current_quality_score = self._current_archive_switch_quality_score()
        candidate_pool = []
        for archive_entry in archive_entries:
            if (
                current_is_feasible
                and bool(archive_entry.get("is_feasible", False))
                and np.isfinite(current_cost)
            ):
                archive_cost = float(archive_entry.get("fitness", np.inf))
                max_allowed_cost = current_cost * (1.0 + float(self.archive_switch_feasible_slack_ratio))
                if np.isfinite(archive_cost) and archive_cost > max_allowed_cost:
                    continue
            diversity = self._layout_distance(
                current_reference,
                archive_entry["layout_reference"],
            )
            if diversity < float(self.elite_archive_min_diversity) * 0.60:
                continue
            quality_score = self._archive_switch_quality_score(archive_entry)
            relative_quality_gain = quality_score - current_quality_score
            if (
                current_is_feasible
                and bool(archive_entry.get("is_feasible", False))
                and np.isfinite(current_cost)
            ):
                archive_cost = float(archive_entry.get("fitness", np.inf))
                if np.isfinite(archive_cost) and archive_cost > current_cost:
                    if (
                        self.no_improve_steps < int(switch_controls["no_improve"]) * 2
                        or diversity < float(self.elite_archive_min_diversity) * 1.25
                    ):
                        continue
            staleness = min(
                max(int(global_step) - int(archive_entry.get("last_used_step", -10**9)), 0)
                / float(max(1, int(self.archive_switch_cooldown_steps))),
                2.0,
            )
            recent_use_penalty = max(
                0.0,
                1.0 - staleness,
            )
            total_score = (
                float(self.archive_switch_diversity_weight) * diversity
                + float(self.archive_switch_quality_weight) * relative_quality_gain
                + float(self.archive_switch_staleness_weight) * staleness
                - float(self.archive_switch_recent_use_penalty) * recent_use_penalty
            )
            if total_score < float(switch_controls["min_score"]):
                continue
            candidate_pool.append((float(total_score), archive_entry))

        if not candidate_pool:
            return False

        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        selected_entry = candidate_pool[0][1]
        self._mark_archive_entry_used(selected_entry, global_step)
        self.s = copy.deepcopy(selected_entry["solution"])
        self._evaluate_solution(self.s)
        self.current_energy = self.s.fitness
        self.no_improve_steps = 0
        self.last_archive_switch_step = int(global_step)
        self.archive_switch_count += 1
        self.T = max(self.T, float(switch_controls["target_temp_ratio"]) * self.T_initial, self.T_min)
        self._update_histogram(self.s.fitness)
        self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
        self.energy_history.append(self.s.fitness)
        return True

    def _attempt_final_elite_push(self, global_step, total_steps, fast_time):
        if not self.final_elite_push_enabled:
            return fast_time, False
        if self.final_elite_push_count >= int(self.final_elite_push_max_count):
            return fast_time, False
        if self.final_elite_push_blocked_after_fail:
            if (
                self.archive_switch_count <= int(self.final_elite_push_fail_archive_switch_count)
                and self.gbest_update_count <= int(self.final_elite_push_fail_gbest_update_count)
            ):
                return fast_time, False
            self.final_elite_push_blocked_after_fail = False
        if self.best_feasible_solution is None or not np.isfinite(self.best_feasible_cost):
            return fast_time, False
        progress_ratio = float(global_step) / float(max(1, total_steps))
        if progress_ratio < float(self.final_elite_push_progress_ratio):
            return fast_time, False
        if self.no_improve_steps < int(self.final_elite_push_no_improve):
            return fast_time, False
        if self.accept_rate_window > float(self.final_elite_push_accept_rate_threshold):
            return fast_time, False
        if global_step - self.last_final_elite_push_step < int(self.final_elite_push_cooldown_steps):
            return fast_time, False
        if global_step - self.last_archive_switch_step < int(self.final_elite_push_post_switch_cooldown_steps):
            return fast_time, False

        best_before = float(self.best_feasible_cost)
        self.last_final_elite_push_step = int(global_step)
        self.final_elite_push_count += 1

        fast_time, _ = self._elite_intensification(fast_time)
        improved = bool(
            np.isfinite(self.best_feasible_cost)
            and self.best_feasible_cost + 1e-9 < best_before
        )

        recent_mid_shot_success = (
            global_step - int(self.last_mid_structural_shot_success_step)
            <= int(self.mid_structural_shot_late_archive_lock_steps)
        )
        if (
            getattr(self.s, "current_is_feasible", False)
            and np.isfinite(getattr(self.s, "fitness", np.inf))
            and (recent_mid_shot_success or self._relative_gap_to_best(self.s) <= float(self.mid_structural_shot_late_archive_lock_gap_ratio))
        ):
            current_candidate = copy.deepcopy(self.s)
            current_before = float(current_candidate.fitness)
            current_candidate = self._greedy_local_search(
                current_candidate,
                enable_guided=True,
                guided_topk=max(int(self.final_elite_push_guided_topk), int(self.mid_structural_shot_guided_topk)),
                guided_max_iters=max(int(self.final_elite_push_guided_max_iters), int(self.mid_structural_shot_guided_max_iters)),
            )
            if (
                current_candidate.current_is_feasible
                and np.isfinite(current_candidate.fitness)
                and current_candidate.fitness + 1e-9 < current_before
            ):
                self._observe_feasible_state(current_candidate)
                self.s = copy.deepcopy(current_candidate)
                self.current_energy = self.s.fitness
                self.no_improve_steps = 0
                self._update_histogram(self.current_energy)
                self.energy_history.append(self.current_energy)
                self.modified_energy_history.append(self._tilde_energy(self.current_energy))
                fast_time = datetime.datetime.now()
                improved = True

        if self.final_elite_push_guided_local_search and self.best_feasible_solution is not None:
            candidate = copy.deepcopy(self.best_feasible_solution)
            self._evaluate_solution(candidate)
            if candidate.current_is_feasible and np.isfinite(candidate.fitness):
                candidate_before = float(candidate.fitness)
                candidate = self._greedy_local_search(
                    candidate,
                    enable_guided=True,
                    guided_topk=self.final_elite_push_guided_topk,
                    guided_max_iters=self.final_elite_push_guided_max_iters,
                )
                if (
                    candidate.current_is_feasible
                    and np.isfinite(candidate.fitness)
                    and candidate.fitness + 1e-9 < candidate_before
                ):
                    self._observe_feasible_state(candidate)
                    fast_time = datetime.datetime.now()
                    improved = True
                    if self._should_takeover_after_elite(candidate, "final"):
                        self.s = copy.deepcopy(candidate)
                        self.current_energy = self.s.fitness
                        self.no_improve_steps = 0
                        self._update_histogram(self.current_energy)
                        self.energy_history.append(self.current_energy)
                        self.modified_energy_history.append(self._tilde_energy(self.current_energy))

        if improved:
            self.no_improve_steps = 0
            self.final_elite_push_success_count += 1
            self.final_elite_push_blocked_after_fail = False
        else:
            self.final_elite_push_blocked_after_fail = True
            self.final_elite_push_fail_archive_switch_count = int(self.archive_switch_count)
            self.final_elite_push_fail_gbest_update_count = int(self.gbest_update_count)
        return fast_time, improved

    def _get_bin_index(self, energy):
        if not np.isfinite(energy):
            return -1
        return int(energy / self.bin_width)

    def _get_H_value(self, energy):
        idx = self._get_bin_index(energy)
        return self.energy_histogram.get(idx, 0)

    def _update_histogram(self, energy):
        idx = self._get_bin_index(energy)
        self.energy_histogram[idx] = self.energy_histogram.get(idx, 0) + 1

    def _tilde_energy(self, raw_cost):
        return raw_cost + self.k_hist * self._get_H_value(raw_cost)

    def _get_histogram_reference_energy(self):
        if self.best_feasible_solution is not None:
            best_energy = float(getattr(self.best_feasible_solution, "fitness", np.inf))
            if np.isfinite(best_energy):
                return best_energy
        if np.isfinite(self.best_feasible_cost):
            return float(self.best_feasible_cost)
        return np.inf

    def _get_adaptive_bin_width(self, best_energy):
        if not np.isfinite(best_energy):
            return float(self.bin_width)
        energy_scale = max(abs(float(best_energy)), 1.0)
        min_width = max(float(self.bin_width_min_abs), energy_scale * float(self.bin_width_lower_ratio))
        max_width = max(min_width, energy_scale * float(self.bin_width_upper_ratio))
        fallback_width = max(min_width, energy_scale * float(self.bin_width_fallback_ratio))

        finite_history = [float(value) for value in self.energy_history if np.isfinite(value)]
        if finite_history:
            recent_window = finite_history[-int(self.bin_width_recent_window):]
            if len(recent_window) >= 32:
                q10, q90 = np.percentile(np.asarray(recent_window, dtype=float), [10, 90])
                spread = max(float(q90) - float(q10), 0.0)
                if spread > 0:
                    candidate_width = spread / float(max(1, int(self.bin_width_target_bins)))
                    if np.isfinite(candidate_width) and candidate_width > 0:
                        return float(min(max(candidate_width, min_width), max_width))

        return float(min(max(fallback_width, min_width), max_width))

    def _refresh_bin_width_from_best(self):
        reference_energy = self._get_histogram_reference_energy()
        new_bin_width = self._get_adaptive_bin_width(reference_energy)
        if abs(new_bin_width - float(self.bin_width)) <= 1e-12:
            return False

        self.bin_width = new_bin_width
        rebuilt_histogram = {}
        for energy in self.energy_history:
            if not np.isfinite(energy):
                continue
            idx = self._get_bin_index(float(energy))
            rebuilt_histogram[idx] = rebuilt_histogram.get(idx, 0) + 1
        self.energy_histogram = rebuilt_histogram
        return True

    def _sample_temperature_floor(self, n_samples=None, target_accept=None):
        n_samples = int(n_samples or self.temperature_floor_samples)
        target_accept = float(target_accept or self.temperature_floor_target_accept)
        fallback = max(self.T_initial * self.temperature_floor_lower_ratio, 1.0)
        cap = max(self.T_initial * self.temperature_floor_cap_ratio, fallback)

        if self.best_feasible_solution is None or not np.isfinite(getattr(self.best_feasible_solution, 'fitness', np.inf)):
            return float(fallback)

        if not 0.0 < target_accept < 1.0:
            target_accept = self.temperature_floor_target_accept

        base_solution = copy.deepcopy(self.best_feasible_solution)
        allowed_actions = [
            table_idx
            for table_idx, action_idx in enumerate(self.valid_actions)
            if action_idx in self.temperature_floor_action_ids
        ]
        if not allowed_actions:
            allowed_actions = self._get_allowed_action_indices(base_solution)
        if not allowed_actions:
            return float(fallback)

        current_tilde = self._tilde_energy(base_solution.fitness)
        local_v_worst = max(float(base_solution.fitness), 1.0)
        deltas = []

        for _ in range(max(1, n_samples)):
            action_table_idx = int(np.random.choice(allowed_actions))
            action_idx = self.valid_actions[action_table_idx]
            candidate = copy.deepcopy(base_solution)
            recipe = self.action_recipes[action_idx]
            self._apply_recipe(candidate, recipe)
            local_metrics = FBSUtil.evaluate_layout(
                candidate.fbs_model,
                candidate.areas,
                candidate.H,
                candidate.F,
                candidate.aspect_limits,
                v_worst=local_v_worst,
                k_penalty=self.k_penalty,
                distance_metric="manhattan",
            )
            candidate_cost = float(local_metrics["cost"])
            candidate_tilde = self._tilde_energy(candidate_cost)
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

    def _refresh_temperature_floor(self, allow_raise=False):
        sampled_floor = self._sample_temperature_floor()
        if not np.isfinite(sampled_floor):
            return float(self.T_min)

        fallback = max(self.T_initial * self.temperature_floor_lower_ratio, 1.0)
        sampled_floor = max(float(sampled_floor), fallback)
        current_floor = float(self.T_min) if np.isfinite(self.T_min) else sampled_floor
        if allow_raise:
            self.T_min = float(sampled_floor)
        else:
            self.T_min = float(max(fallback, min(current_floor, sampled_floor)))
        return float(self.T_min)

    def _apply_recipe(self, solution, recipe):
        layout_dirty = False
        for primitive_action in recipe:
            if primitive_action == 3 and layout_dirty:
                self._evaluate_solution(solution)
                layout_dirty = False
            solution._apply_action(solution.actions[primitive_action])
            layout_dirty = True

    def _generate_segment_insert_light_candidate(self, solution):
        bay_structure = permutationToArray(solution.fbs_model.permutation, solution.fbs_model.bay)
        bay_candidates = [idx for idx, current_bay in enumerate(bay_structure) if len(current_bay) >= 3]
        if not bay_candidates:
            return self._generate_candidate_by_recipe(solution, self.action_recipes[10])

        best_candidate = None
        best_cost = float("inf")
        fallback_candidate = None
        trial_count = max(1, int(self.segment_insert_light_trials))

        for _ in range(trial_count):
            bay_idx = int(np.random.choice(np.asarray(bay_candidates, dtype=int)))
            current_bay = list(bay_structure[bay_idx])
            segment_lengths = [
                int(seg_len)
                for seg_len in self.segment_insert_light_segment_lengths
                if len(current_bay) > int(seg_len)
            ]
            if not segment_lengths:
                continue

            segment_len = int(np.random.choice(np.asarray(segment_lengths, dtype=int)))
            start_idx = int(np.random.randint(0, len(current_bay) - segment_len + 1))
            segment = current_bay[start_idx : start_idx + segment_len]
            remaining = current_bay[:start_idx] + current_bay[start_idx + segment_len :]
            if len(remaining) == 0:
                continue

            insert_positions = [
                int(pos)
                for pos in range(len(remaining) + 1)
                if pos != start_idx
            ]
            if not insert_positions:
                continue
            insert_idx = int(np.random.choice(np.asarray(insert_positions, dtype=int)))
            candidate_bay = remaining[:insert_idx] + segment + remaining[insert_idx:]
            if candidate_bay == current_bay:
                continue

            candidate_structure = [list(item) for item in bay_structure]
            candidate_structure[bay_idx] = candidate_bay
            candidate_perm, candidate_bay_flags = arrayToPermutation(candidate_structure)

            candidate = copy.deepcopy(solution)
            candidate.fbs_model.permutation = np.asarray(candidate_perm, dtype=int).tolist()
            candidate.fbs_model.bay = np.asarray(candidate_bay_flags, dtype=int).tolist()
            self._evaluate_solution(candidate)

            if fallback_candidate is None:
                fallback_candidate = candidate
            if np.isfinite(candidate.fitness) and candidate.fitness < best_cost:
                best_candidate = candidate
                best_cost = float(candidate.fitness)

        if best_candidate is not None:
            return best_candidate
        if fallback_candidate is not None:
            return fallback_candidate
        return self._generate_candidate_by_recipe(solution, self.action_recipes[10])

    def _generate_candidate_by_action_fallback(self, solution, action_idx, phase="main"):
        return self.dqn_operator_dispatcher.generate_candidate_by_action_fallback(
            self,
            solution,
            action_idx,
            phase=phase,
        )

    def generate_candidate_by_action(self, solution, action_idx, phase="main"):
        return self.dqn_operator_dispatcher.generate_candidate_by_action(
            self,
            solution,
            action_idx,
            phase=phase,
        )

    def _generate_candidate_by_recipe(self, solution, recipe):
        candidate = copy.deepcopy(solution)
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

    def _relative_gap_to_best(self, solution):
        if (
            self.best_feasible_solution is None
            or not np.isfinite(self.best_feasible_cost)
            or not np.isfinite(getattr(solution, "fitness", np.inf))
        ):
            return float("inf")
        return max(float(solution.fitness) - float(self.best_feasible_cost), 0.0) / max(
            abs(float(self.best_feasible_cost)),
            1.0,
        )

    def _current_search_phase(self, solution=None):
        return self.dqn_search_controller.current_search_phase(self, solution)

    def _effective_two_stage_local_proxy_mix(self, action_idx):
        if not self.two_stage_local_proxy_enabled:
            return 0.0, "disabled"
        base_mix = float(self.two_stage_local_proxy_mix.get(int(action_idx), 0.0))
        if base_mix <= 0.0:
            return 0.0, "disabled"
        phase = self._archive_switch_phase()
        phase_multiplier = float(self.two_stage_local_proxy_phase_mix.get(str(phase), 0.0))
        return max(0.0, base_mix * phase_multiplier), str(phase)

    def _get_local_search_policy(self, solution, action_idx=None):
        return self.dqn_search_controller.get_local_search_policy(self, solution, action_idx=action_idx)

    def _should_trigger_local_search(self, solution, action_idx, accepted_improved, previous_d_inf):
        return self.dqn_search_controller.should_trigger_local_search(
            self,
            solution,
            action_idx,
            accepted_improved,
            previous_d_inf,
        )

    def _should_run_topk_guided_search(self, solution, action_idx=None):
        return self.dqn_search_controller.should_run_topk_guided_search(
            self,
            solution,
            action_idx=action_idx,
        )

    def _get_reward_cost_scale(self, previous_cost, next_cost, previous_best_feasible):
        reference_cost = previous_best_feasible
        if not np.isfinite(reference_cost):
            reference_cost = previous_cost if np.isfinite(previous_cost) else next_cost
        reference_cost = max(abs(float(reference_cost)) if np.isfinite(reference_cost) else 0.0, 1.0)

        floor_value = max(
            float(self.reward_cost_scale_min_abs),
            reference_cost * float(self.reward_cost_scale_floor_ratio),
        )
        cap_value = max(
            floor_value * 2.0,
            reference_cost * float(self.reward_cost_scale_cap_ratio),
        )
        candidates = []
        if np.isfinite(previous_cost) and np.isfinite(next_cost):
            candidates.append(abs(float(previous_cost) - float(next_cost)))
        if np.isfinite(previous_best_feasible):
            if np.isfinite(previous_cost):
                candidates.append(abs(float(previous_cost) - float(previous_best_feasible)))
            if np.isfinite(next_cost):
                candidates.append(abs(float(next_cost) - float(previous_best_feasible)))
        if len(self.recent_feasible_costs) >= 8:
            recent = np.asarray(self.recent_feasible_costs, dtype=float)
            q75, q25 = np.percentile(recent, [75, 25])
            iqr = float(q75 - q25)
            if np.isfinite(iqr) and iqr > 0.0:
                candidates.append(iqr)
            recent_diffs = np.abs(np.diff(recent))
            if recent_diffs.size > 0:
                recent_med = float(np.median(recent_diffs))
                if np.isfinite(recent_med) and recent_med > 0.0:
                    candidates.append(recent_med)

        finite_candidates = [float(value) for value in candidates if np.isfinite(value) and value > 0.0]
        if finite_candidates:
            scale = float(np.median(np.asarray(finite_candidates, dtype=float)))
        else:
            scale = floor_value
        return float(min(max(scale, floor_value), cap_value))

    @staticmethod
    def _normalize_state_feature(values):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.full(arr.shape, 0.5, dtype=np.float32)
        v_min = float(np.min(finite))
        v_max = float(np.max(finite))
        if abs(v_max - v_min) <= 1e-12:
            return np.full(arr.shape, 0.5, dtype=np.float32)
        return ((arr - v_min) / (v_max - v_min)).astype(np.float32)

    @staticmethod
    def _fit_state_feature_length(values, target_size, fill_value=0.0):
        target_size = int(max(0, target_size))
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if target_size <= 0:
            return np.asarray([], dtype=np.float32)
        if arr.size == target_size:
            return arr
        if arr.size == 0:
            return np.full(target_size, float(fill_value), dtype=np.float32)
        if arr.size > target_size:
            return arr[:target_size]
        return np.pad(
            arr,
            (0, target_size - arr.size),
            mode="constant",
            constant_values=float(fill_value),
        ).astype(np.float32)

    def _build_facility_structural_state(self, solution, n):
        return self.dqn_state_encoder.build_facility_structural_state(self, solution, n)

    def _build_global_state_context(self, solution, tm, permutation, bay_ranges, aspect_violation, infeasible_mask):
        return self.dqn_state_encoder.build_global_state_context(
            self,
            solution,
            tm,
            permutation,
            bay_ranges,
            aspect_violation,
            infeasible_mask,
        )

    def state_encoder(self, solution):
        return self.dqn_state_encoder.encode(self, solution)

    def _current_phase_action_ids(self, solution):
        return self.dqn_operator_dispatcher.current_phase_action_ids(self, solution)

    def _get_allowed_action_indices(self, solution):
        return self.dqn_operator_dispatcher.get_allowed_action_indices(self, solution)

    def _effective_two_stage_budgets(self, action_idx):
        raw_budget = int(self.two_stage_proposal_counts.get(action_idx, 12))
        eval_budget = int(self.two_stage_eval_counts.get(action_idx, 4))
        return int(raw_budget), int(eval_budget)

    def _acceptance_temp_multiplier(self):
        if not self.acceptance_phase_enabled:
            return 1.0
        progress = min(max(float(self.current_progress_ratio), 0.0), 1.0)
        if self.acceptance_phase_mode == "three":
            first_split = min(max(float(self.acceptance_phase_first_split_ratio), 0.0), 1.0)
            second_split = min(max(float(self.acceptance_phase_second_split_ratio), first_split), 1.0)
            if progress <= first_split:
                return max(float(self.acceptance_early_temp_multiplier), 1e-6)
            if progress <= second_split:
                return max(float(self.acceptance_mid_temp_multiplier), 1e-6)
            tail_ratio = (progress - second_split) / max(1e-8, 1.0 - second_split)
            start = float(self.acceptance_late_temp_multiplier_start)
            end = float(self.acceptance_late_temp_multiplier_end)
            return max(start + (end - start) * tail_ratio, 1e-6)
        split = min(max(float(self.acceptance_phase_split_ratio), 0.0), 1.0)
        if progress <= split:
            return max(float(self.acceptance_early_temp_multiplier), 1e-6)
        tail_ratio = (progress - split) / max(1e-8, 1.0 - split)
        start = float(self.acceptance_late_temp_multiplier_start)
        end = float(self.acceptance_late_temp_multiplier_end)
        return max(start + (end - start) * tail_ratio, 1e-6)

    def _effective_acceptance_temperature(self):
        return max(self.T * self._acceptance_temp_multiplier(), 1e-12)

    def _update_reward_rel_delta_scale(self, previous_cost, next_cost):
        if not (np.isfinite(previous_cost) and np.isfinite(next_cost)):
            return float(self.reward_rel_delta_scale_ema)
        base = max(abs(float(previous_cost)), 1.0)
        step_ratio = abs(float(next_cost) - float(previous_cost)) / base
        if not np.isfinite(step_ratio) or step_ratio <= 0.0:
            return float(self.reward_rel_delta_scale_ema)

        self.reward_rel_delta_history.append(float(step_ratio))
        if len(self.reward_rel_delta_history) >= 8:
            median_ratio = float(np.median(np.asarray(self.reward_rel_delta_history, dtype=float)))
        else:
            median_ratio = float(step_ratio)

        target_scale = min(
            max(median_ratio, float(self.reward_rel_delta_scale_min)),
            float(self.reward_rel_delta_scale_max),
        )
        beta = float(self.reward_rel_delta_scale_ema_beta)
        self.reward_rel_delta_scale_ema = (
            beta * float(self.reward_rel_delta_scale_ema)
            + (1.0 - beta) * target_scale
        )
        self.reward_rel_delta_scale_ema = min(
            max(float(self.reward_rel_delta_scale_ema), float(self.reward_rel_delta_scale_min)),
            float(self.reward_rel_delta_scale_max),
        )
        return float(self.reward_rel_delta_scale_ema)

    def _get_reward_rel_delta_scale(self, previous_cost, next_cost):
        scale = self._update_reward_rel_delta_scale(previous_cost, next_cost)
        if not np.isfinite(scale) or scale <= 0.0:
            return float(self.reward_rel_delta_scale_default)
        return float(scale)

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        return self.dqn_reward_engine.compute_transition_reward(
            self,
            previous_cost,
            next_cost,
            previous_d_inf,
            next_d_inf,
            previous_best_feasible,
            accept,
        )

    def _compute_reward_components(
        self,
        previous_cost,
        immediate_cost,
        final_cost,
        previous_d_inf,
        immediate_d_inf,
        final_d_inf,
        previous_best_feasible,
        accept,
    ):
        return self.dqn_reward_engine.compute_reward_components(
            self,
            previous_cost,
            immediate_cost,
            final_cost,
            previous_d_inf,
            immediate_d_inf,
            final_d_inf,
            previous_best_feasible,
            accept,
        )

    def _compose_training_reward(self, reward_components):
        return self.dqn_reward_engine.compose_training_reward(self, reward_components)

    def _accept_candidate(self, current_cost, candidate_cost):
        return self.dqn_search_controller.accept_candidate(self, current_cost, candidate_cost)

    def _bootstrap_feasible_archive(self, max_attempts=None):
        if self.worst_feasible_cost is not None:
            return True
        max_attempts = max_attempts or max(400, 6 * self.t_max)
        attempts = 0
        best_candidate = copy.deepcopy(self.s)
        best_score = self._layout_score(best_candidate)
        restart_interval = max(1, len(self.bootstrap_recipes))

        while attempts < max_attempts:
            base_solution = best_candidate
            for recipe in self.bootstrap_recipes:
                if attempts >= max_attempts:
                    break
                candidate = self._generate_candidate_by_recipe(base_solution, recipe)
                attempts += 1
                candidate_score = self._layout_score(candidate)
                if candidate_score < best_score:
                    best_candidate = copy.deepcopy(candidate)
                    best_score = candidate_score
                    self.s = copy.deepcopy(candidate)
                    self.current_energy = self.s.fitness
                    self.no_improve_steps = 0
                else:
                    self.no_improve_steps += 1
                if candidate.current_is_feasible:
                    self.s = copy.deepcopy(candidate)
                    self.current_energy = self.s.fitness
                    self._observe_feasible_state(self.s)
                    self._update_histogram(self.current_energy)
                    return True

            if attempts >= max_attempts or self.worst_feasible_cost is not None:
                break

            if attempts % restart_interval == 0:
                restart_solution = copy.deepcopy(self.env)
                restart_solution.reset()
                attempts += 1
                restart_score = self._layout_score(restart_solution)
                if restart_score < best_score or not np.isfinite(self.s.MHC):
                    best_candidate = copy.deepcopy(restart_solution)
                    best_score = restart_score
                    self.s = copy.deepcopy(restart_solution)
                    self.current_energy = self.s.fitness
                if restart_solution.current_is_feasible:
                    self.s = copy.deepcopy(restart_solution)
                    self.current_energy = self.s.fitness
                    self._observe_feasible_state(self.s)
                    self._update_histogram(self.current_energy)
                    return True

        self.s = copy.deepcopy(best_candidate)
        self.current_energy = self.s.fitness
        return self.worst_feasible_cost is not None

    def _activate_main_search_from_feasible(self):
        if self.best_feasible_solution is None:
            return False
        self.s = copy.deepcopy(self.best_feasible_solution)
        self._evaluate_solution(self.s)
        self.current_energy = self.s.fitness
        self.T = self.T_initial
        self.no_improve_steps = 0
        self.episodes_without_improvement = 0
        self.last_diversify_step = -max(1, self.t_max)
        self.last_reheat_step = -self.reheat_cooldown_steps
        self.reheat_episode_count = 0
        self.accept_window.clear()
        self.accept_rate_window = 1.0
        self.bin_width = self._get_adaptive_bin_width(self._get_histogram_reference_energy())
        self.energy_histogram = {}
        self.energy_history = []
        self.modified_energy_history = []
        self.prob_history = []
        if np.isfinite(self.current_energy):
            self._update_histogram(self.current_energy)
            self.energy_history.append(self.current_energy)
            self.modified_energy_history.append(self._tilde_energy(self.current_energy))
        self._refresh_temperature_floor(allow_raise=True)
        self.T = max(self.T, self.T_min)
        # logger.info(
        #     f"T_min updated | reason: activate_main_search | best energy: {float(self.best_feasible_cost):.6f} | T_min: {float(self.T_min):.6f} | temperature: {float(self.T):.6f}"
        # )
        return True

    def _bootstrap_until_first_feasible(self, max_attempts=None):
        if self.worst_feasible_cost is not None:
            return self._activate_main_search_from_feasible()
        np_state = np.random.get_state()
        py_state = random.getstate()
        success = self._bootstrap_feasible_archive(max_attempts=max_attempts)
        if success:
            np.random.set_state(np_state)
            random.setstate(py_state)
            return self._activate_main_search_from_feasible()
        return False

    def _restart_from_best_feasible(self):
        if self.best_feasible_solution is None:
            return False
        self.s = copy.deepcopy(self.best_feasible_solution)
        self._evaluate_solution(self.s)
        self.current_energy = self.s.fitness
        if self.no_improve_reset_on_episode_restart:
            self.no_improve_steps = 0
        return True

    @staticmethod
    def _to_int_list(values):
        array_values = np.asarray(values, dtype=int).reshape(-1)
        return [int(value) for value in array_values.tolist()]

    def _bay_sizes_from_bay(self, bay):
        sizes = []
        count = 0
        for marker in self._to_int_list(bay):
            count += 1
            if marker == 1:
                sizes.append(count)
                count = 0
        return sizes

    @staticmethod
    def _make_bay_by_sizes(sizes):
        total = int(sum(int(size) for size in sizes))
        if total <= 0:
            return []
        bay = [0] * total
        cursor = 0
        for size in sizes:
            cursor += int(size)
            if 0 < cursor <= total:
                bay[cursor - 1] = 1
        bay[-1] = 1
        return bay

    @staticmethod
    def _is_valid_bay(bay):
        bay_arr = np.asarray(bay, dtype=int).reshape(-1)
        return bay_arr.size > 0 and int(bay_arr[-1]) == 1 and bool(np.any(bay_arr == 1))

    def _propose_boundary_moves(self, bay):
        bay_arr = np.asarray(bay, dtype=int).reshape(-1).copy()
        if bay_arr.size <= 1:
            return []
        bay_arr[-1] = 1
        boundary_indices = np.where(bay_arr == 1)[0].tolist()
        candidates = []
        seen = set()
        # ???????????bay???????
        for idx in boundary_indices[:-1]:
            for shift in (-1, 1):
                new_idx = int(idx + shift)
                if new_idx < 0 or new_idx >= bay_arr.size - 1:
                    continue
                candidate = bay_arr.copy()
                candidate[idx] = 0
                candidate[new_idx] = 1
                candidate[-1] = 1
                candidate_list = [int(v) for v in candidate.tolist()]
                key = tuple(candidate_list)
                if key in seen or not self._is_valid_bay(candidate_list):
                    continue
                seen.add(key)
                candidates.append(candidate_list)
        return candidates

    def _build_candidate_from_layout(self, base_solution, permutation, bay):
        perm_list = self._to_int_list(permutation)
        bay_list = self._to_int_list(bay)
        if len(perm_list) == 0 or len(perm_list) != len(bay_list):
            return None
        bay_list[-1] = 1
        candidate = copy.deepcopy(base_solution)
        candidate.fbs_model.permutation = perm_list
        candidate.fbs_model.bay = bay_list
        self._evaluate_solution(candidate)
        if not candidate.current_is_feasible or not np.isfinite(candidate.fitness):
            return None
        return candidate

    def _build_evaluated_candidate_from_layout(self, base_solution, permutation, bay):
        perm_list = self._to_int_list(permutation)
        bay_list = self._to_int_list(bay)
        if len(perm_list) == 0 or len(perm_list) != len(bay_list):
            return None
        bay_list[-1] = 1
        candidate = copy.deepcopy(base_solution)
        candidate.fbs_model.permutation = perm_list
        candidate.fbs_model.bay = bay_list
        self._evaluate_solution(candidate)
        return candidate

    def _candidate_rank_key(self, candidate):
        feasible_rank = 0 if getattr(candidate, "current_is_feasible", False) else 1
        d_inf = int(getattr(candidate, "current_d_inf", 10**9))
        cost = float(getattr(candidate, "fitness", np.inf))
        if not np.isfinite(cost):
            cost = float("inf")
        mhc = float(getattr(candidate, "MHC", np.inf))
        if not np.isfinite(mhc):
            mhc = float("inf")
        violation = float("inf")
        try:
            violation = float(self._constraint_violation(candidate))
        except Exception:
            violation = float("inf")
        return feasible_rank, d_inf, cost, violation, mhc

    @staticmethod
    def _two_stage_template_name(meta):
        if isinstance(meta, dict):
            template = str(meta.get("template", "")).strip()
            if template:
                return template
        return "default"

    @staticmethod
    def _bump_two_stage_template_counter(counter, template_name):
        if not isinstance(counter, dict):
            return
        template_name = str(template_name).strip() or "default"
        counter[template_name] = int(counter.get(template_name, 0)) + 1

    def _record_two_stage_template_counter(self, stats, field, meta):
        if not isinstance(stats, dict):
            return
        counter = stats.get(field)
        if not isinstance(counter, dict):
            counter = {}
            stats[field] = counter
        self._bump_two_stage_template_counter(counter, self._two_stage_template_name(meta))

    @staticmethod
    def _increment_named_counter(stats, field, name):
        if not isinstance(stats, dict):
            return
        counter = stats.get(field)
        if not isinstance(counter, dict):
            counter = {}
            stats[field] = counter
        counter[str(name)] = int(counter.get(str(name), 0)) + 1

    @staticmethod
    def _format_two_stage_template_counter(counter, topn=3):
        if not isinstance(counter, dict) or not counter:
            return "none"
        ordered = sorted(
            counter.items(),
            key=lambda item: (int(item[1]), str(item[0])),
            reverse=True,
        )[: max(1, int(topn))]
        return ",".join(f"{name}:{count}" for name, count in ordered)

    def _append_two_stage_proposal(self, proposal_pool, seen_keys, permutation, bay, raw_budget, meta=None):
        perm_list = self._to_int_list(permutation)
        bay_list = self._to_int_list(bay)
        if len(perm_list) == 0 or len(perm_list) != len(bay_list):
            return False
        bay_list[-1] = 1
        key = (tuple(perm_list), tuple(bay_list))
        if key in seen_keys or len(proposal_pool) >= int(raw_budget):
            return False
        seen_keys.add(key)
        proposal_pool.append((perm_list, bay_list, dict(meta) if isinstance(meta, dict) else {}))
        return True

    def _append_two_stage_structure_candidate(self, proposal_pool, seen_keys, bay_structure, raw_budget, meta=None):
        if len(proposal_pool) >= int(raw_budget):
            return False
        normalized_structure = [list(current_bay) for current_bay in bay_structure if len(current_bay) > 0]
        if len(normalized_structure) == 0:
            return False
        candidate_perm, candidate_bay = arrayToPermutation(normalized_structure)
        return self._append_two_stage_proposal(
            proposal_pool,
            seen_keys,
            candidate_perm,
            candidate_bay,
            raw_budget,
            meta=meta,
        )

    def _build_two_stage_pair_rows(self, solution, sym_flow):
        pair_rows = self._build_topk_pair_rows(solution)
        if pair_rows:
            return pair_rows
        if sym_flow.ndim != 2 or sym_flow.shape[0] != sym_flow.shape[1]:
            return []
        pair_scores = np.triu(sym_flow, 1)
        pair_indices = np.argwhere(pair_scores > 0)
        rows = [
            (int(i), int(j), float(pair_scores[i, j]))
            for i, j in pair_indices
        ]
        rows.sort(key=lambda item: item[2], reverse=True)
        return rows

    def _prepare_two_stage_context(self, solution):
        perm = self._to_int_list(solution.fbs_model.permutation)
        bay = self._to_int_list(solution.fbs_model.bay)
        if len(perm) == 0 or len(perm) != len(bay):
            return None
        bay[-1] = 1
        bay_structure = [list(current_bay) for current_bay in permutationToArray(perm, bay)]
        if len(bay_structure) == 0:
            return None

        n = len(perm)
        area_array = np.asarray(getattr(solution, "areas", []), dtype=float).reshape(-1)
        if area_array.size != n:
            area_array = np.ones(n, dtype=float)
        flow_matrix = np.asarray(getattr(solution, "F", []), dtype=float)
        if flow_matrix.ndim != 2 or flow_matrix.shape[0] != n or flow_matrix.shape[1] != n:
            flow_matrix = np.zeros((n, n), dtype=float)
        sym_flow = flow_matrix + flow_matrix.T

        aspect_limits = np.asarray(getattr(solution, "aspect_limits", []), dtype=float).reshape(-1)
        if aspect_limits.size != n:
            aspect_limits = np.full(n, np.inf, dtype=float)
        finite_aspect_limits = aspect_limits[np.isfinite(aspect_limits) & (aspect_limits > 0)]
        default_aspect_limit = float(np.max(finite_aspect_limits)) if finite_aspect_limits.size > 0 else 4.0

        position_map = {int(facility): int(idx) for idx, facility in enumerate(perm)}
        facility_to_bay = {}
        bay_areas = []
        for bay_idx, current_bay in enumerate(bay_structure):
            bay_area = 0.0
            for facility in current_bay:
                facility_to_bay[int(facility)] = int(bay_idx)
                facility_idx = int(facility) - 1
                if 0 <= facility_idx < area_array.size:
                    bay_area += float(area_array[facility_idx])
            bay_areas.append(float(bay_area))

        flow_scale = float(np.sum(np.triu(sym_flow, 1)))
        total_area = float(np.sum(area_array))
        return {
            "permutation": perm,
            "bay": bay,
            "bay_structure": bay_structure,
            "positions": position_map,
            "facility_to_bay": facility_to_bay,
            "areas": area_array,
            "flow": flow_matrix,
            "sym_flow": sym_flow,
            "aspect_limits": aspect_limits,
            "default_aspect_limit": default_aspect_limit,
            "bay_areas": bay_areas,
            "target_bay_area": total_area / float(max(1, len(bay_structure))),
            "avg_bay_size": float(n) / float(max(1, len(bay_structure))),
            "flow_scale": max(flow_scale, 1.0),
            "pair_rows": self._build_two_stage_pair_rows(solution, sym_flow),
            "H": max(float(getattr(solution, "H", 1.0)), 1e-8),
        }

    @staticmethod
    def _preserve_order_unique(values):
        seen = set()
        ordered = []
        for value in values:
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _score_adjacent_pair_cross_flow(self, sym_flow, left_bay, right_bay):
        if len(left_bay) == 0 or len(right_bay) == 0:
            return 0.0
        cross_flow = 0.0
        for left_facility in left_bay:
            left_idx = int(left_facility) - 1
            if left_idx < 0 or left_idx >= sym_flow.shape[0]:
                continue
            right_indices = np.asarray(right_bay, dtype=int) - 1
            right_indices = right_indices[(right_indices >= 0) & (right_indices < sym_flow.shape[0])]
            if right_indices.size == 0:
                continue
            cross_flow += float(np.sum(sym_flow[left_idx, right_indices]))
        return float(cross_flow)

    @staticmethod
    def _compute_bay_area_proxy(current_bay, area_array):
        bay_area = 0.0
        for facility in current_bay:
            facility_idx = int(facility) - 1
            if 0 <= facility_idx < area_array.size:
                bay_area += float(area_array[facility_idx])
        return float(bay_area)

    def _compute_bay_geometry_pressure_proxy(
        self,
        current_bay,
        bay_area,
        area_array,
        aspect_limits,
        default_aspect_limit,
        H_value,
        avg_bay_size,
    ):
        if len(current_bay) == 0 or bay_area <= 0.0:
            return 1.0
        min_width_sum = 0.0
        max_area_share = 0.0
        for facility in current_bay:
            facility_idx = int(facility) - 1
            if facility_idx < 0 or facility_idx >= area_array.size:
                continue
            area_value = max(float(area_array[facility_idx]), 1e-12)
            aspect_limit = default_aspect_limit
            if 0 <= facility_idx < aspect_limits.size:
                limit_value = float(aspect_limits[facility_idx])
                if np.isfinite(limit_value) and limit_value > 0.0:
                    aspect_limit = limit_value
            min_width_sum += math.sqrt(area_value / max(aspect_limit, 1e-8))
            max_area_share = max(max_area_share, area_value / max(float(bay_area), 1e-8))
        bay_width_proxy = float(bay_area) / max(H_value, 1e-8)
        pressure = max(0.0, min_width_sum / max(bay_width_proxy, 1e-8) - 1.0)
        pressure += 0.50 * max(0.0, max_area_share - 0.70)
        pressure += 0.15 * abs(len(current_bay) - avg_bay_size) / max(avg_bay_size, 1.0)
        return float(pressure)

    @staticmethod
    def _compute_facility_support_to_bay(facility, current_bay, sym_flow):
        facility_idx = int(facility) - 1
        if facility_idx < 0 or facility_idx >= sym_flow.shape[0] or len(current_bay) == 0:
            return 0.0
        bay_indices = np.asarray(current_bay, dtype=int) - 1
        bay_indices = bay_indices[(bay_indices >= 0) & (bay_indices < sym_flow.shape[0])]
        if bay_indices.size == 0:
            return 0.0
        return float(np.sum(sym_flow[facility_idx, bay_indices]))

    @staticmethod
    def _compute_block_support_to_bay(block, current_bay, sym_flow):
        if len(block) == 0 or len(current_bay) == 0:
            return 0.0
        block_indices = np.asarray(block, dtype=int) - 1
        bay_indices = np.asarray(current_bay, dtype=int) - 1
        block_indices = block_indices[(block_indices >= 0) & (block_indices < sym_flow.shape[0])]
        bay_indices = bay_indices[(bay_indices >= 0) & (bay_indices < sym_flow.shape[0])]
        if block_indices.size == 0 or bay_indices.size == 0:
            return 0.0
        return float(np.sum(sym_flow[np.ix_(block_indices, bay_indices)]))

    @staticmethod
    def _compute_block_internal_flow(block, sym_flow):
        if len(block) <= 1:
            return 0.0
        block_indices = np.asarray(block, dtype=int) - 1
        block_indices = block_indices[(block_indices >= 0) & (block_indices < sym_flow.shape[0])]
        if block_indices.size <= 1:
            return 0.0
        sub_matrix = sym_flow[np.ix_(block_indices, block_indices)]
        return float(np.sum(np.triu(sub_matrix, 1)))

    def _get_two_stage_proxy_weights(self, action_idx):
        weights = dict(self.two_stage_proxy_weights)
        action_idx = int(action_idx)
        if action_idx == 9:
            weights["order_penalty"] *= 1.15
        elif action_idx == 10:
            weights["order_penalty"] *= 1.20
            weights["adjacent_cross_penalty"] *= 0.85
        elif action_idx == 11:
            weights["global_cross_penalty"] *= 1.25
            weights["geometry_penalty"] *= 1.10
        elif action_idx == 14:
            weights["adjacent_cross_penalty"] *= 1.25
            weights["area_balance_penalty"] *= 1.10
        elif action_idx == 15:
            weights["adjacent_cross_penalty"] *= 1.15
            weights["order_penalty"] *= 1.10
        return weights

    def _score_action14_proxy_bonus(self, context, candidate_structure, meta):
        if not isinstance(meta, dict):
            return 0.0
        pair_idx = int(meta.get("pair_idx", -1))
        if pair_idx < 0:
            return 0.0
        if pair_idx + 1 >= len(candidate_structure) or pair_idx + 1 >= len(context["bay_structure"]):
            return 0.0

        base_left = list(context["bay_structure"][pair_idx])
        base_right = list(context["bay_structure"][pair_idx + 1])
        candidate_left = list(candidate_structure[pair_idx])
        candidate_right = list(candidate_structure[pair_idx + 1])
        if len(candidate_left) == 0 or len(candidate_right) == 0:
            return -float("inf")

        sym_flow = context["sym_flow"]
        area_array = context["areas"]
        aspect_limits = context["aspect_limits"]
        default_aspect_limit = float(context["default_aspect_limit"])
        H_value = max(float(context["H"]), 1e-8)
        avg_bay_size = max(float(context["avg_bay_size"]), 1.0)
        flow_scale = max(float(context["flow_scale"]), 1.0)

        base_cut_flow = self._score_adjacent_pair_cross_flow(sym_flow, base_left, base_right)
        candidate_cut_flow = self._score_adjacent_pair_cross_flow(sym_flow, candidate_left, candidate_right)
        cut_gain = (float(base_cut_flow) - float(candidate_cut_flow)) / flow_scale

        base_left_area = self._compute_bay_area_proxy(base_left, area_array)
        base_right_area = self._compute_bay_area_proxy(base_right, area_array)
        candidate_left_area = self._compute_bay_area_proxy(candidate_left, area_array)
        candidate_right_area = self._compute_bay_area_proxy(candidate_right, area_array)
        base_pair_area = max(base_left_area + base_right_area, 1e-8)
        candidate_pair_area = max(candidate_left_area + candidate_right_area, 1e-8)
        base_pair_balance_penalty = abs(base_left_area - base_right_area) / base_pair_area
        candidate_pair_balance_penalty = abs(candidate_left_area - candidate_right_area) / candidate_pair_area
        pair_area_gain = float(base_pair_balance_penalty - candidate_pair_balance_penalty)

        base_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                base_left,
                base_left_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                base_right,
                base_right_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        candidate_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                candidate_left,
                candidate_left_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                candidate_right,
                candidate_right_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        geometry_gain = float(base_geometry_pressure - candidate_geometry_pressure)

        weights = self.two_stage_action14_proxy_weights
        return (
            float(weights["cut_gain"]) * cut_gain
            + float(weights["pair_area_gain"]) * pair_area_gain
            + float(weights["geometry_gain"]) * geometry_gain
        )

    def _score_action11_proxy_bonus(self, context, candidate_structure, meta):
        if not isinstance(meta, dict):
            return 0.0
        facility = int(meta.get("facility", -1))
        source_bay_idx = int(meta.get("source_bay_idx", -1))
        target_bay_idx = int(meta.get("target_bay_idx", -1))
        source_pos = int(meta.get("source_pos", -1))
        insert_idx = int(meta.get("insert_idx", -1))
        if facility <= 0 or source_bay_idx < 0 or target_bay_idx < 0:
            return 0.0
        if source_bay_idx >= len(context["bay_structure"]) or target_bay_idx >= len(context["bay_structure"]):
            return 0.0
        if source_bay_idx >= len(candidate_structure) or target_bay_idx >= len(candidate_structure):
            return 0.0

        base_source = list(context["bay_structure"][source_bay_idx])
        base_target = list(context["bay_structure"][target_bay_idx])
        candidate_source = list(candidate_structure[source_bay_idx])
        candidate_target = list(candidate_structure[target_bay_idx])
        if len(candidate_source) == 0 or len(candidate_target) == 0:
            return -float("inf")

        sym_flow = context["sym_flow"]
        flow_scale = max(float(context["flow_scale"]), 1.0)
        area_array = context["areas"]
        aspect_limits = context["aspect_limits"]
        default_aspect_limit = float(context["default_aspect_limit"])
        H_value = max(float(context["H"]), 1e-8)
        avg_bay_size = max(float(context["avg_bay_size"]), 1.0)
        target_bay_area = max(float(context["target_bay_area"]), 1e-8)

        base_source_without_facility = [node for node in base_source if int(node) != int(facility)]
        source_support = self._compute_facility_support_to_bay(facility, base_source_without_facility, sym_flow)
        target_support = self._compute_facility_support_to_bay(facility, base_target, sym_flow)
        total_support = max(
            float(np.sum(sym_flow[int(facility) - 1, :])) if 0 <= int(facility) - 1 < sym_flow.shape[0] else 0.0,
            1.0,
        )
        target_affinity_gain = target_support / total_support
        source_damage_penalty = source_support / total_support

        base_source_area = self._compute_bay_area_proxy(base_source, area_array)
        base_target_area = self._compute_bay_area_proxy(base_target, area_array)
        candidate_source_area = self._compute_bay_area_proxy(candidate_source, area_array)
        candidate_target_area = self._compute_bay_area_proxy(candidate_target, area_array)
        base_area_penalty = (
            abs(base_source_area - target_bay_area) + abs(base_target_area - target_bay_area)
        ) / target_bay_area
        candidate_area_penalty = (
            abs(candidate_source_area - target_bay_area) + abs(candidate_target_area - target_bay_area)
        ) / target_bay_area
        area_gain = float(base_area_penalty - candidate_area_penalty)

        base_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                base_source,
                base_source_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                base_target,
                base_target_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        candidate_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                candidate_source,
                candidate_source_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                candidate_target,
                candidate_target_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        geometry_gain = float(base_geometry_pressure - candidate_geometry_pressure)

        boundary_bonus = 0.0
        if len(base_source) > 0 and source_pos in {0, len(base_source) - 1}:
            boundary_bonus += 0.6
        if len(base_target) > 0 and insert_idx in {0, len(base_target)}:
            boundary_bonus += 0.2
        else:
            boundary_bonus += 0.1

        weights = self.two_stage_action11_proxy_weights
        return (
            float(weights["target_affinity_gain"]) * target_affinity_gain
            - float(weights["source_damage_penalty"]) * source_damage_penalty
            + float(weights["area_gain"]) * area_gain
            + float(weights["geometry_gain"]) * geometry_gain
            + float(weights["boundary_bonus"]) * boundary_bonus
        )

    def _score_action15_proxy_bonus(self, context, candidate_structure, meta):
        if not isinstance(meta, dict):
            return 0.0
        pair_idx = int(meta.get("pair_idx", -1))
        move_type = str(meta.get("move_type", "")).strip().lower()
        left_block_size = int(meta.get("left_block_size", 0))
        right_block_size = int(meta.get("right_block_size", 0))
        if pair_idx < 0 or pair_idx + 1 >= len(context["bay_structure"]):
            return 0.0
        if pair_idx + 1 >= len(candidate_structure):
            return 0.0

        base_left = list(context["bay_structure"][pair_idx])
        base_right = list(context["bay_structure"][pair_idx + 1])
        candidate_left = list(candidate_structure[pair_idx])
        candidate_right = list(candidate_structure[pair_idx + 1])
        if len(candidate_left) == 0 or len(candidate_right) == 0:
            return -float("inf")

        sym_flow = context["sym_flow"]
        flow_scale = max(float(context["flow_scale"]), 1.0)
        area_array = context["areas"]
        aspect_limits = context["aspect_limits"]
        default_aspect_limit = float(context["default_aspect_limit"])
        H_value = max(float(context["H"]), 1e-8)
        avg_bay_size = max(float(context["avg_bay_size"]), 1.0)

        base_cut_flow = self._score_adjacent_pair_cross_flow(sym_flow, base_left, base_right)
        candidate_cut_flow = self._score_adjacent_pair_cross_flow(sym_flow, candidate_left, candidate_right)
        cut_gain = (float(base_cut_flow) - float(candidate_cut_flow)) / flow_scale

        base_left_area = self._compute_bay_area_proxy(base_left, area_array)
        base_right_area = self._compute_bay_area_proxy(base_right, area_array)
        candidate_left_area = self._compute_bay_area_proxy(candidate_left, area_array)
        candidate_right_area = self._compute_bay_area_proxy(candidate_right, area_array)
        base_pair_area = max(base_left_area + base_right_area, 1e-8)
        candidate_pair_area = max(candidate_left_area + candidate_right_area, 1e-8)
        base_pair_balance_penalty = abs(base_left_area - base_right_area) / base_pair_area
        candidate_pair_balance_penalty = abs(candidate_left_area - candidate_right_area) / candidate_pair_area
        pair_area_gain = float(base_pair_balance_penalty - candidate_pair_balance_penalty)

        base_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                base_left,
                base_left_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                base_right,
                base_right_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        candidate_geometry_pressure = (
            self._compute_bay_geometry_pressure_proxy(
                candidate_left,
                candidate_left_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
            + self._compute_bay_geometry_pressure_proxy(
                candidate_right,
                candidate_right_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        ) / 2.0
        geometry_gain = float(base_geometry_pressure - candidate_geometry_pressure)

        block_affinity_gain = 0.0
        cohesion_bonus = 0.0
        exchange_bonus = 0.0
        if move_type == "left_to_right" and left_block_size > 0 and left_block_size <= len(base_left):
            moved_left_block = list(base_left[-left_block_size:])
            remaining_left = list(base_left[:-left_block_size])
            block_affinity_gain = (
                self._compute_block_support_to_bay(moved_left_block, base_right, sym_flow)
                - self._compute_block_support_to_bay(moved_left_block, remaining_left, sym_flow)
            ) / flow_scale
            cohesion_bonus = self._compute_block_internal_flow(moved_left_block, sym_flow) / flow_scale
        elif move_type == "right_to_left" and right_block_size > 0 and right_block_size <= len(base_right):
            moved_right_block = list(base_right[:right_block_size])
            remaining_right = list(base_right[right_block_size:])
            block_affinity_gain = (
                self._compute_block_support_to_bay(moved_right_block, base_left, sym_flow)
                - self._compute_block_support_to_bay(moved_right_block, remaining_right, sym_flow)
            ) / flow_scale
            cohesion_bonus = self._compute_block_internal_flow(moved_right_block, sym_flow) / flow_scale
        elif (
            move_type == "exchange"
            and left_block_size > 0
            and right_block_size > 0
            and left_block_size <= len(base_left)
            and right_block_size <= len(base_right)
        ):
            moved_left_block = list(base_left[-left_block_size:])
            moved_right_block = list(base_right[:right_block_size])
            remaining_left = list(base_left[:-left_block_size])
            remaining_right = list(base_right[right_block_size:])
            block_affinity_gain = (
                self._compute_block_support_to_bay(moved_left_block, base_right, sym_flow)
                - self._compute_block_support_to_bay(moved_left_block, remaining_left, sym_flow)
                + self._compute_block_support_to_bay(moved_right_block, base_left, sym_flow)
                - self._compute_block_support_to_bay(moved_right_block, remaining_right, sym_flow)
            ) / flow_scale
            cohesion_bonus = (
                self._compute_block_internal_flow(moved_left_block, sym_flow)
                + self._compute_block_internal_flow(moved_right_block, sym_flow)
            ) / flow_scale
            exchange_bonus = 1.0

        weights = self.two_stage_action15_proxy_weights
        return (
            float(weights["cut_gain"]) * cut_gain
            + float(weights["block_affinity_gain"]) * block_affinity_gain
            + float(weights["pair_area_gain"]) * pair_area_gain
            + float(weights["geometry_gain"]) * geometry_gain
            + float(weights["cohesion_bonus"]) * cohesion_bonus
            + float(weights["exchange_bonus"]) * exchange_bonus
        )

    def _get_two_stage_local_proxy_indices(self, context, candidate_structure, action_idx, meta):
        if (
            not self.two_stage_local_proxy_enabled
            or int(action_idx) not in self.two_stage_local_proxy_action_ids
            or not isinstance(meta, dict)
        ):
            return []
        total_bays = int(len(candidate_structure))
        if total_bays <= 0:
            return []
        affected_indices = set()
        action_idx = int(action_idx)
        if action_idx == 11:
            source_bay_idx = int(meta.get("source_bay_idx", -1))
            target_bay_idx = int(meta.get("target_bay_idx", -1))
            for bay_idx in (
                source_bay_idx - 1,
                source_bay_idx,
                source_bay_idx + 1,
                target_bay_idx - 1,
                target_bay_idx,
                target_bay_idx + 1,
            ):
                if 0 <= int(bay_idx) < total_bays:
                    affected_indices.add(int(bay_idx))
        elif action_idx in {14, 15}:
            pair_idx = int(meta.get("pair_idx", -1))
            for bay_idx in (pair_idx - 1, pair_idx, pair_idx + 1, pair_idx + 2):
                if 0 <= int(bay_idx) < total_bays:
                    affected_indices.add(int(bay_idx))
        return sorted(int(bay_idx) for bay_idx in affected_indices)

    def _score_two_stage_local_proxy(self, context, candidate_structure, action_idx, meta):
        affected_indices = self._get_two_stage_local_proxy_indices(
            context,
            candidate_structure,
            action_idx,
            meta,
        )
        if not affected_indices:
            return 0.0

        sym_flow = context["sym_flow"]
        area_array = context["areas"]
        aspect_limits = context["aspect_limits"]
        default_aspect_limit = float(context["default_aspect_limit"])
        H_value = max(float(context["H"]), 1e-8)
        avg_bay_size = max(float(context["avg_bay_size"]), 1.0)
        target_bay_area = max(float(context["target_bay_area"]), 1e-8)

        facility_to_bay = {}
        position_map = {}
        bay_areas = []
        ordered_facilities = []
        for bay_idx, current_bay in enumerate(candidate_structure):
            for position, facility in enumerate(current_bay):
                facility_to_bay[int(facility)] = int(bay_idx)
                position_map[int(facility)] = int(position)

        for bay_idx in affected_indices:
            current_bay = list(candidate_structure[bay_idx])
            if len(current_bay) == 0:
                return -float("inf")
            ordered_facilities.extend(int(facility) for facility in current_bay)
            bay_areas.append(self._compute_bay_area_proxy(current_bay, area_array))

        local_flow_mass = 0.0
        order_penalty = 0.0
        adjacent_cross_penalty = 0.0
        global_cross_penalty = 0.0
        for left_pos in range(len(ordered_facilities)):
            facility_i = int(ordered_facilities[left_pos])
            facility_i_idx = facility_i - 1
            bay_i = int(facility_to_bay.get(facility_i, -1))
            pos_i = int(position_map.get(facility_i, 0))
            if facility_i_idx < 0 or facility_i_idx >= sym_flow.shape[0] or bay_i < 0:
                continue
            for right_pos in range(left_pos + 1, len(ordered_facilities)):
                facility_j = int(ordered_facilities[right_pos])
                facility_j_idx = facility_j - 1
                bay_j = int(facility_to_bay.get(facility_j, -1))
                pos_j = int(position_map.get(facility_j, 0))
                if facility_j_idx < 0 or facility_j_idx >= sym_flow.shape[0] or bay_j < 0:
                    continue
                weight = float(sym_flow[facility_i_idx, facility_j_idx])
                if weight <= 0.0:
                    continue
                local_flow_mass += weight
                bay_gap = abs(int(bay_i) - int(bay_j))
                if bay_gap == 0:
                    order_penalty += weight * abs(int(pos_i) - int(pos_j))
                elif bay_gap == 1:
                    adjacent_cross_penalty += weight
                else:
                    global_cross_penalty += weight * float(min(bay_gap, 3))

        local_flow_scale = max(float(local_flow_mass), 1.0)
        local_facility_count = max(1, len(ordered_facilities))
        order_penalty = order_penalty / float(local_flow_scale * local_facility_count)
        adjacent_cross_penalty = adjacent_cross_penalty / local_flow_scale
        global_cross_penalty = global_cross_penalty / local_flow_scale

        area_balance_penalty = (
            sum(abs(float(bay_area) - target_bay_area) for bay_area in bay_areas)
            / float(target_bay_area * max(1, len(bay_areas)))
        )
        geometry_penalty = 0.0
        for bay_idx in affected_indices:
            current_bay = list(candidate_structure[bay_idx])
            bay_area = self._compute_bay_area_proxy(current_bay, area_array)
            geometry_penalty += self._compute_bay_geometry_pressure_proxy(
                current_bay,
                bay_area,
                area_array,
                aspect_limits,
                default_aspect_limit,
                H_value,
                avg_bay_size,
            )
        geometry_penalty = geometry_penalty / float(max(1, len(affected_indices)))

        weights = self.two_stage_local_proxy_weights
        local_penalty = (
            float(weights["order_penalty"]) * order_penalty
            + float(weights["adjacent_cross_penalty"]) * adjacent_cross_penalty
            + float(weights["global_cross_penalty"]) * global_cross_penalty
            + float(weights["area_balance_penalty"]) * area_balance_penalty
            + float(weights["geometry_penalty"]) * geometry_penalty
        )
        return -float(local_penalty)

    def _score_two_stage_proxy(self, context, permutation, bay, action_idx, meta=None, local_proxy_mix=0.0):
        perm = self._to_int_list(permutation)
        bay_flags = self._to_int_list(bay)
        if len(perm) == 0 or len(perm) != len(bay_flags):
            return -float("inf")
        bay_flags[-1] = 1
        bay_structure = [list(current_bay) for current_bay in permutationToArray(perm, bay_flags)]
        if len(bay_structure) == 0:
            return -float("inf")

        facility_to_bay = {}
        position_map = {int(facility): int(pos) for pos, facility in enumerate(perm)}
        bay_areas = []
        area_array = context["areas"]
        sym_flow = context["sym_flow"]
        aspect_limits = context["aspect_limits"]
        default_aspect_limit = float(context["default_aspect_limit"])
        H_value = max(float(context["H"]), 1e-8)

        for bay_idx, current_bay in enumerate(bay_structure):
            if len(current_bay) == 0:
                return -float("inf")
            bay_area = 0.0
            for facility in current_bay:
                facility = int(facility)
                facility_to_bay[facility] = int(bay_idx)
                facility_idx = facility - 1
                if 0 <= facility_idx < area_array.size:
                    bay_area += float(area_array[facility_idx])
            bay_areas.append(float(bay_area))

        order_penalty = 0.0
        adjacent_cross_penalty = 0.0
        global_cross_penalty = 0.0
        n = len(perm)
        for facility_i in range(1, n + 1):
            bay_i = facility_to_bay.get(facility_i)
            pos_i = position_map.get(facility_i)
            if bay_i is None or pos_i is None:
                continue
            for facility_j in range(facility_i + 1, n + 1):
                bay_j = facility_to_bay.get(facility_j)
                pos_j = position_map.get(facility_j)
                if bay_j is None or pos_j is None:
                    continue
                weight = float(sym_flow[facility_i - 1, facility_j - 1])
                if weight <= 0.0:
                    continue
                bay_gap = abs(int(bay_i) - int(bay_j))
                if bay_gap == 0:
                    order_penalty += weight * abs(int(pos_i) - int(pos_j))
                elif bay_gap == 1:
                    adjacent_cross_penalty += weight
                else:
                    global_cross_penalty += weight * float(min(bay_gap, 3))

        flow_scale = max(float(context["flow_scale"]), 1.0)
        order_penalty = order_penalty / float(flow_scale * max(1, n))
        adjacent_cross_penalty = adjacent_cross_penalty / flow_scale
        global_cross_penalty = global_cross_penalty / flow_scale

        target_bay_area = max(float(context["target_bay_area"]), 1e-8)
        area_balance_penalty = (
            sum(abs(float(bay_area) - target_bay_area) for bay_area in bay_areas)
            / float(target_bay_area * max(1, len(bay_areas)))
        )

        avg_bay_size = max(float(context["avg_bay_size"]), 1.0)
        geometry_penalty = 0.0
        for current_bay, bay_area in zip(bay_structure, bay_areas):
            if bay_area <= 0.0:
                geometry_penalty += 1.0
                continue
            min_width_sum = 0.0
            max_area_share = 0.0
            for facility in current_bay:
                facility_idx = int(facility) - 1
                if facility_idx < 0 or facility_idx >= area_array.size:
                    continue
                area_value = max(float(area_array[facility_idx]), 1e-12)
                aspect_limit = default_aspect_limit
                if 0 <= facility_idx < aspect_limits.size:
                    limit_value = float(aspect_limits[facility_idx])
                    if np.isfinite(limit_value) and limit_value > 0.0:
                        aspect_limit = limit_value
                min_width_sum += math.sqrt(area_value / max(aspect_limit, 1e-8))
                max_area_share = max(max_area_share, area_value / max(float(bay_area), 1e-8))
            bay_width_proxy = float(bay_area) / H_value
            geometry_penalty += max(0.0, min_width_sum / max(bay_width_proxy, 1e-8) - 1.0)
            geometry_penalty += 0.50 * max(0.0, max_area_share - 0.70)
            geometry_penalty += 0.15 * abs(len(current_bay) - avg_bay_size) / avg_bay_size
        geometry_penalty = geometry_penalty / float(max(1, len(bay_structure)))

        weights = self._get_two_stage_proxy_weights(action_idx)
        total_penalty = (
            float(weights["order_penalty"]) * order_penalty
            + float(weights["adjacent_cross_penalty"]) * adjacent_cross_penalty
            + float(weights["global_cross_penalty"]) * global_cross_penalty
            + float(weights["area_balance_penalty"]) * area_balance_penalty
            + float(weights["geometry_penalty"]) * geometry_penalty
        )
        proxy_score = -float(total_penalty)
        if int(action_idx) == 11:
            action11_bonus = self._score_action11_proxy_bonus(context, bay_structure, meta)
            if not np.isfinite(action11_bonus):
                return -float("inf")
            proxy_score += float(action11_bonus)
        if int(action_idx) == 14:
            action14_bonus = self._score_action14_proxy_bonus(context, bay_structure, meta)
            if not np.isfinite(action14_bonus):
                return -float("inf")
            proxy_score += float(action14_bonus)
        if int(action_idx) == 15:
            action15_bonus = self._score_action15_proxy_bonus(context, bay_structure, meta)
            if not np.isfinite(action15_bonus):
                return -float("inf")
            proxy_score += float(action15_bonus)
        if float(local_proxy_mix) > 0.0 and int(action_idx) in self.two_stage_local_proxy_action_ids:
            local_proxy_score = self._score_two_stage_local_proxy(
                context,
                bay_structure,
                action_idx,
                meta,
            )
            if not np.isfinite(local_proxy_score):
                return -float("inf")
            proxy_score += float(local_proxy_mix) * float(local_proxy_score)
        return float(proxy_score)

    def _propose_two_stage_flow_guided_swap(self, context, raw_budget):
        proposals = []
        seen_keys = set()
        perm = list(context["permutation"])
        bay = list(context["bay"])
        position_map = context["positions"]
        pair_rows = context["pair_rows"][: max(1, min(self.two_stage_pair_pool_cap, len(context["pair_rows"])))]
        if not pair_rows:
            return proposals

        for left_idx, right_idx, _pair_score in pair_rows:
            facility_a = int(left_idx + 1)
            facility_b = int(right_idx + 1)
            if facility_a not in position_map or facility_b not in position_map:
                continue
            pos_a = int(position_map[facility_a])
            pos_b = int(position_map[facility_b])
            endpoint_positions = [pos_a, pos_b]
            guided_positions = self._collect_guided_target_positions(perm, bay, endpoint_positions)
            near_a = [
                int(pos)
                for pos in guided_positions
                if abs(int(pos) - pos_a) <= self.two_stage_candidate_window_radius + 1
            ]
            near_b = [
                int(pos)
                for pos in guided_positions
                if abs(int(pos) - pos_b) <= self.two_stage_candidate_window_radius + 1
            ]
            random_positions = [int(pos) for pos in guided_positions if int(pos) not in {pos_a, pos_b}]
            np.random.shuffle(random_positions)
            candidate_positions = self._preserve_order_unique(
                [pos_b]
                + near_b
                + [pos_a]
                + near_a
                + guided_positions[:4]
                + random_positions[:2]
            )
            for target_pos in candidate_positions:
                if target_pos != pos_a:
                    candidate_perm = perm.copy()
                    candidate_perm[pos_a], candidate_perm[target_pos] = (
                        candidate_perm[target_pos],
                        candidate_perm[pos_a],
                    )
                    self._append_two_stage_proposal(
                        proposals,
                        seen_keys,
                        candidate_perm,
                        bay,
                        raw_budget,
                    )
                    if len(proposals) >= int(raw_budget):
                        return proposals
                if target_pos != pos_b:
                    candidate_perm = perm.copy()
                    candidate_perm[pos_b], candidate_perm[target_pos] = (
                        candidate_perm[target_pos],
                        candidate_perm[pos_b],
                    )
                    self._append_two_stage_proposal(
                        proposals,
                        seen_keys,
                        candidate_perm,
                        bay,
                        raw_budget,
                    )
                    if len(proposals) >= int(raw_budget):
                        return proposals
        return proposals

    def _propose_two_stage_segment_insert(self, context, raw_budget):
        proposals = []
        seen_keys = set()
        bay_structure = [list(current_bay) for current_bay in context["bay_structure"]]
        sym_flow = context["sym_flow"]
        candidate_bays = []
        for bay_idx, current_bay in enumerate(bay_structure):
            if len(current_bay) < 3:
                continue
            internal_flow = 0.0
            for left_pos in range(len(current_bay)):
                left_facility = int(current_bay[left_pos]) - 1
                if left_facility < 0 or left_facility >= sym_flow.shape[0]:
                    continue
                right_indices = np.asarray(current_bay[left_pos + 1 :], dtype=int) - 1
                right_indices = right_indices[
                    (right_indices >= 0) & (right_indices < sym_flow.shape[0])
                ]
                if right_indices.size == 0:
                    continue
                internal_flow += float(np.sum(sym_flow[left_facility, right_indices]))
            candidate_bays.append((float(internal_flow), int(bay_idx)))
        candidate_bays.sort(key=lambda item: item[0], reverse=True)

        for _score, bay_idx in candidate_bays[: max(1, min(4, len(candidate_bays)))]:
            current_bay = list(bay_structure[bay_idx])
            segment_lengths = [
                int(segment_len)
                for segment_len in self.segment_insert_light_segment_lengths
                if len(current_bay) > int(segment_len)
            ]
            if not segment_lengths:
                continue

            facility_strengths = []
            for pos, facility in enumerate(current_bay):
                facility_idx = int(facility) - 1
                other_indices = np.asarray(
                    [other - 1 for other in current_bay if int(other) != int(facility)],
                    dtype=int,
                )
                if facility_idx < 0 or facility_idx >= sym_flow.shape[0] or other_indices.size == 0:
                    local_strength = 0.0
                else:
                    other_indices = other_indices[
                        (other_indices >= 0) & (other_indices < sym_flow.shape[0])
                    ]
                    local_strength = float(np.sum(sym_flow[facility_idx, other_indices]))
                facility_strengths.append((float(local_strength), int(pos)))
            facility_strengths.sort(key=lambda item: item[0], reverse=True)
            anchor_positions = [int(pos) for _strength, pos in facility_strengths[: min(3, len(facility_strengths))]]
            anchor_positions = self._preserve_order_unique(
                anchor_positions + [0, max(0, len(current_bay) - 1)]
            )
            extra_anchor_positions = [int(pos) for pos in range(len(current_bay)) if int(pos) not in set(anchor_positions)]
            np.random.shuffle(extra_anchor_positions)
            anchor_positions.extend(extra_anchor_positions[:1])

            for segment_len in segment_lengths:
                start_candidates = []
                for anchor_pos in anchor_positions:
                    start_min = max(0, int(anchor_pos) - int(segment_len) + 1)
                    start_max = min(int(anchor_pos), len(current_bay) - int(segment_len))
                    for start_idx in range(start_min, start_max + 1):
                        start_candidates.append(int(start_idx))
                start_candidates.extend([0, max(0, len(current_bay) - int(segment_len))])
                start_candidates = self._preserve_order_unique(start_candidates)

                for start_idx in start_candidates:
                    segment = current_bay[start_idx : start_idx + int(segment_len)]
                    remaining = current_bay[:start_idx] + current_bay[start_idx + int(segment_len) :]
                    if len(remaining) == 0:
                        continue
                    insert_positions = [0, len(remaining), max(0, start_idx - 1), min(len(remaining), start_idx + 1)]
                    for anchor_pos in anchor_positions:
                        if start_idx <= int(anchor_pos) < start_idx + int(segment_len):
                            continue
                        removed_before = sum(
                            1
                            for removed_pos in range(start_idx, start_idx + int(segment_len))
                            if removed_pos < int(anchor_pos)
                        )
                        adjusted_anchor = max(
                            0,
                            min(len(remaining) - 1, int(anchor_pos) - removed_before),
                        )
                        insert_positions.extend([adjusted_anchor, min(len(remaining), adjusted_anchor + 1)])
                    random_insert_positions = list(range(len(remaining) + 1))
                    np.random.shuffle(random_insert_positions)
                    insert_positions.extend(random_insert_positions[:2])
                    for insert_idx in self._preserve_order_unique(insert_positions):
                        candidate_bay = remaining[:insert_idx] + segment + remaining[insert_idx:]
                        if candidate_bay == current_bay:
                            continue
                        candidate_structure = [list(item) for item in bay_structure]
                        candidate_structure[bay_idx] = candidate_bay
                        self._append_two_stage_structure_candidate(
                            proposals,
                            seen_keys,
                            candidate_structure,
                            raw_budget,
                        )
                        if len(proposals) >= int(raw_budget):
                            return proposals
        return proposals

    def _propose_two_stage_cross_bay_relocate(self, context, raw_budget):
        proposals = []
        seen_keys = set()
        bay_structure = [list(current_bay) for current_bay in context["bay_structure"]]
        sym_flow = context["sym_flow"]
        area_array = context["areas"]
        facility_scores = []
        for bay_idx, current_bay in enumerate(bay_structure):
            current_set = {int(facility) for facility in current_bay}
            for pos, facility in enumerate(current_bay):
                facility_idx = int(facility) - 1
                if facility_idx < 0 or facility_idx >= sym_flow.shape[0]:
                    continue
                total_flow = float(np.sum(sym_flow[facility_idx, :]))
                same_indices = np.asarray([node - 1 for node in current_set if node != int(facility)], dtype=int)
                same_flow = 0.0
                if same_indices.size > 0:
                    same_indices = same_indices[
                        (same_indices >= 0) & (same_indices < sym_flow.shape[0])
                    ]
                    same_flow = float(np.sum(sym_flow[facility_idx, same_indices]))
                cross_flow = max(total_flow - same_flow, 0.0)
                boundary_bonus = 0.0
                if pos == 0 or pos == len(current_bay) - 1:
                    boundary_bonus = 0.10 * max(total_flow, 1.0)
                facility_area = float(area_array[facility_idx]) if facility_idx < area_array.size else 0.0
                facility_scores.append(
                    (
                        float(cross_flow + boundary_bonus + 0.02 * facility_area),
                        int(facility),
                        int(bay_idx),
                    )
                )
        facility_scores.sort(key=lambda item: item[0], reverse=True)

        for _score, facility, source_bay_idx in facility_scores[: max(1, min(6, len(facility_scores)))]:
            source_bay = list(bay_structure[source_bay_idx])
            if len(source_bay) <= 1:
                continue
            source_pos = int(source_bay.index(int(facility))) if int(facility) in source_bay else -1
            facility_idx = int(facility) - 1
            facility_area = float(area_array[facility_idx]) if 0 <= facility_idx < area_array.size else 0.0
            target_scores = []
            for target_bay_idx, target_bay in enumerate(bay_structure):
                if target_bay_idx == source_bay_idx or len(target_bay) == 0:
                    continue
                target_indices = np.asarray(target_bay, dtype=int) - 1
                target_indices = target_indices[
                    (target_indices >= 0) & (target_indices < sym_flow.shape[0])
                ]
                if target_indices.size == 0:
                    target_flow = 0.0
                else:
                    target_flow = float(np.sum(sym_flow[facility_idx, target_indices]))
                projected_area = float(context["bay_areas"][target_bay_idx]) + facility_area
                area_penalty = abs(projected_area - float(context["target_bay_area"])) / max(
                    float(context["target_bay_area"]),
                    1e-8,
                )
                adjacency_bonus = 0.0
                if abs(int(target_bay_idx) - int(source_bay_idx)) == 1:
                    adjacency_bonus = 0.10 * max(target_flow, 1.0)
                target_scores.append(
                    (
                        float(target_flow + adjacency_bonus - 0.15 * area_penalty),
                        int(target_bay_idx),
                    )
                )
            target_scores.sort(key=lambda item: item[0], reverse=True)
            prioritized_targets = [target_bay_idx for _score, target_bay_idx in target_scores[: min(3, len(target_scores))]]
            prioritized_targets.extend(
                [
                    idx
                    for idx in (source_bay_idx - 1, source_bay_idx + 1)
                    if 0 <= idx < len(bay_structure) and idx != source_bay_idx
                ]
            )
            for target_bay_idx in self._preserve_order_unique(prioritized_targets):
                target_bay = list(bay_structure[target_bay_idx])
                if len(target_bay) == 0:
                    continue
                anchor_strengths = []
                for pos, anchor_facility in enumerate(target_bay):
                    anchor_idx = int(anchor_facility) - 1
                    if anchor_idx < 0 or anchor_idx >= sym_flow.shape[0]:
                        anchor_flow = 0.0
                    else:
                        anchor_flow = float(sym_flow[facility_idx, anchor_idx])
                    anchor_strengths.append((anchor_flow, int(pos)))
                anchor_strengths.sort(key=lambda item: item[0], reverse=True)
                insert_positions = [0, len(target_bay)]
                for _anchor_flow, anchor_pos in anchor_strengths[: min(2, len(anchor_strengths))]:
                    insert_positions.extend([int(anchor_pos), int(anchor_pos) + 1])
                    for delta in range(1, self.two_stage_candidate_window_radius + 1):
                        insert_positions.extend(
                            [
                                max(0, int(anchor_pos) - delta),
                                min(len(target_bay), int(anchor_pos) + delta),
                            ]
                        )
                random_insert_positions = list(range(len(target_bay) + 1))
                np.random.shuffle(random_insert_positions)
                insert_positions.extend(random_insert_positions[:2])

                for insert_idx in self._preserve_order_unique(insert_positions):
                    reduced_structure = []
                    target_new_idx = None
                    for bay_idx, current_bay in enumerate(bay_structure):
                        if bay_idx == source_bay_idx:
                            updated_bay = [node for node in current_bay if int(node) != int(facility)]
                        else:
                            updated_bay = list(current_bay)
                        if len(updated_bay) == 0:
                            continue
                        if bay_idx == target_bay_idx:
                            target_new_idx = len(reduced_structure)
                        reduced_structure.append(updated_bay)
                    if target_new_idx is None:
                        continue
                    target_updated = list(reduced_structure[target_new_idx])
                    insert_idx = max(0, min(len(target_updated), int(insert_idx)))
                    target_updated = target_updated[:insert_idx] + [int(facility)] + target_updated[insert_idx:]
                    reduced_structure[target_new_idx] = target_updated
                    facility_template = "boundary_escape" if source_pos in {0, len(source_bay) - 1} else "cross_core"
                    target_template = "adjacent_bridge" if abs(int(target_bay_idx) - int(source_bay_idx)) == 1 else "flow_affinity"
                    candidate_meta = {
                        "facility": int(facility),
                        "source_bay_idx": int(source_bay_idx),
                        "target_bay_idx": int(target_bay_idx),
                        "source_pos": int(source_pos),
                        "insert_idx": int(insert_idx),
                        "template": f"{facility_template}/{target_template}",
                    }
                    self._append_two_stage_structure_candidate(
                        proposals,
                        seen_keys,
                        reduced_structure,
                        raw_budget,
                        meta=candidate_meta,
                    )
                    if len(proposals) >= int(raw_budget):
                        return proposals
        return proposals

    def _select_two_stage_adjacent_pairs(self, context):
        bay_structure = context["bay_structure"]
        sym_flow = context["sym_flow"]
        pair_candidates = []
        for bay_idx in range(len(bay_structure) - 1):
            left_bay = bay_structure[bay_idx]
            right_bay = bay_structure[bay_idx + 1]
            if len(left_bay) == 0 or len(right_bay) == 0:
                continue
            cross_flow = self._score_adjacent_pair_cross_flow(sym_flow, left_bay, right_bay)
            pair_candidates.append((float(cross_flow), int(bay_idx)))
        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        return [bay_idx for _score, bay_idx in pair_candidates[: max(1, min(4, len(pair_candidates)))]]

    def _evaluate_two_stage_split_cut_flow(self, sym_flow, merged_sequence, split_idx):
        left_indices = np.asarray(merged_sequence[:split_idx], dtype=int) - 1
        right_indices = np.asarray(merged_sequence[split_idx:], dtype=int) - 1
        if left_indices.size == 0 or right_indices.size == 0:
            return float("inf")
        left_indices = left_indices[(left_indices >= 0) & (left_indices < sym_flow.shape[0])]
        right_indices = right_indices[(right_indices >= 0) & (right_indices < sym_flow.shape[0])]
        if left_indices.size == 0 or right_indices.size == 0:
            return float("inf")
        return float(np.sum(sym_flow[np.ix_(left_indices, right_indices)]))

    def _propose_two_stage_adjacent_bay_repartition(self, context, raw_budget):
        proposals = []
        seen_keys = set()
        bay_structure = [list(current_bay) for current_bay in context["bay_structure"]]
        sym_flow = context["sym_flow"]
        areas = context["areas"]
        for pair_idx in self._select_two_stage_adjacent_pairs(context):
            left_bay = list(bay_structure[pair_idx])
            right_bay = list(bay_structure[pair_idx + 1])
            merged_sequence = left_bay + right_bay
            if len(merged_sequence) < 2:
                continue
            current_split = len(left_bay)
            split_candidates = [current_split]
            for delta in range(1, self.two_stage_candidate_window_radius + 2):
                split_candidates.extend([current_split - delta, current_split + delta])

            prefix_areas = []
            cumulative_area = 0.0
            for facility in merged_sequence:
                facility_idx = int(facility) - 1
                if 0 <= facility_idx < areas.size:
                    cumulative_area += float(areas[facility_idx])
                prefix_areas.append(float(cumulative_area))
            total_pair_area = prefix_areas[-1] if prefix_areas else 0.0
            best_area_split = None
            if total_pair_area > 0.0:
                best_area_split = min(
                    range(1, len(merged_sequence)),
                    key=lambda split_idx: abs(prefix_areas[split_idx - 1] - total_pair_area / 2.0),
                )
                split_candidates.append(int(best_area_split))

            cut_scores = [
                (
                    self._evaluate_two_stage_split_cut_flow(sym_flow, merged_sequence, split_idx),
                    int(split_idx),
                )
                for split_idx in range(1, len(merged_sequence))
            ]
            cut_scores.sort(key=lambda item: item[0])
            top_cut_splits = [split_idx for _score, split_idx in cut_scores[:2]]
            split_candidates.extend(top_cut_splits)
            random_split_candidates = list(range(1, len(merged_sequence)))
            np.random.shuffle(random_split_candidates)
            split_candidates.extend(random_split_candidates[:2])

            for split_idx in self._preserve_order_unique(
                [split for split in split_candidates if 0 < int(split) < len(merged_sequence)]
            ):
                candidate_structure = [list(item) for item in bay_structure]
                candidate_structure[pair_idx] = merged_sequence[:split_idx]
                candidate_structure[pair_idx + 1] = merged_sequence[split_idx:]
                if best_area_split is not None and int(split_idx) == int(best_area_split):
                    template_name = "balanced_cut"
                elif int(split_idx) in set(int(v) for v in top_cut_splits):
                    template_name = "cut_min"
                elif abs(int(split_idx) - int(current_split)) <= 1:
                    template_name = "boundary_shift"
                else:
                    template_name = "repartition"
                candidate_meta = {
                    "pair_idx": int(pair_idx),
                    "split_idx": int(split_idx),
                    "current_split": int(current_split),
                    "template": f"cross_pair/{template_name}",
                }
                self._append_two_stage_structure_candidate(
                    proposals,
                    seen_keys,
                    candidate_structure,
                    raw_budget,
                    meta=candidate_meta,
                )
                if len(proposals) >= int(raw_budget):
                    return proposals
        return proposals

    def _propose_two_stage_adjacent_bay_block_repartition(self, context, raw_budget):
        proposals = []
        seen_keys = set()
        bay_structure = [list(current_bay) for current_bay in context["bay_structure"]]
        for pair_idx in self._select_two_stage_adjacent_pairs(context):
            left_bay = list(bay_structure[pair_idx])
            right_bay = list(bay_structure[pair_idx + 1])
            if len(left_bay) + len(right_bay) < 2:
                continue
            left_block_sizes = list(range(1, min(3, len(left_bay)) + 1))
            right_block_sizes = list(range(1, min(3, len(right_bay)) + 1))
            random.shuffle(left_block_sizes)
            random.shuffle(right_block_sizes)

            for left_block_size in left_block_sizes:
                left_prefix = left_bay[:-left_block_size]
                left_suffix = left_bay[-left_block_size:]
                if len(left_prefix) > 0:
                    candidate_structure = [list(item) for item in bay_structure]
                    candidate_structure[pair_idx] = left_prefix
                    candidate_structure[pair_idx + 1] = left_suffix + right_bay
                    candidate_meta = {
                        "pair_idx": int(pair_idx),
                        "move_type": "left_to_right",
                        "left_block_size": int(left_block_size),
                        "right_block_size": 0,
                        "template": "cross_pair/left_bridge",
                    }
                    self._append_two_stage_structure_candidate(
                        proposals,
                        seen_keys,
                        candidate_structure,
                        raw_budget,
                        meta=candidate_meta,
                    )
                    if len(proposals) >= int(raw_budget):
                        return proposals
                for right_block_size in right_block_sizes:
                    right_prefix = right_bay[:right_block_size]
                    right_suffix = right_bay[right_block_size:]
                    if len(right_suffix) > 0:
                        candidate_structure = [list(item) for item in bay_structure]
                        candidate_structure[pair_idx] = left_bay + right_prefix
                        candidate_structure[pair_idx + 1] = right_suffix
                        candidate_meta = {
                            "pair_idx": int(pair_idx),
                            "move_type": "right_to_left",
                            "left_block_size": 0,
                            "right_block_size": int(right_block_size),
                            "template": "cross_pair/right_bridge",
                        }
                        self._append_two_stage_structure_candidate(
                            proposals,
                            seen_keys,
                            candidate_structure,
                            raw_budget,
                            meta=candidate_meta,
                        )
                        if len(proposals) >= int(raw_budget):
                            return proposals
                    new_left = left_prefix + right_prefix
                    new_right = left_suffix + right_suffix
                    if len(new_left) == 0 or len(new_right) == 0:
                        continue
                    candidate_structure = [list(item) for item in bay_structure]
                    candidate_structure[pair_idx] = new_left
                    candidate_structure[pair_idx + 1] = new_right
                    candidate_meta = {
                        "pair_idx": int(pair_idx),
                        "move_type": "exchange",
                        "left_block_size": int(left_block_size),
                        "right_block_size": int(right_block_size),
                        "template": "cross_pair/balanced_exchange",
                    }
                    self._append_two_stage_structure_candidate(
                        proposals,
                        seen_keys,
                        candidate_structure,
                        raw_budget,
                        meta=candidate_meta,
                    )
                    if len(proposals) >= int(raw_budget):
                        return proposals
        return proposals

    def _propose_two_stage_candidates(self, context, action_idx, raw_budget):
        action_idx = int(action_idx)
        if action_idx == 9:
            return self._propose_two_stage_flow_guided_swap(context, raw_budget)
        if action_idx == 10:
            return self._propose_two_stage_segment_insert(context, raw_budget)
        if action_idx == 11:
            return self._propose_two_stage_cross_bay_relocate(context, raw_budget)
        if action_idx == 14:
            return self._propose_two_stage_adjacent_bay_repartition(context, raw_budget)
        if action_idx == 15:
            return self._propose_two_stage_adjacent_bay_block_repartition(context, raw_budget)
        return []

    def _select_two_stage_survivors(self, scored_candidates, eval_budget):
        if not scored_candidates:
            return []
        ordered = sorted(scored_candidates, key=lambda item: item[0], reverse=True)
        eval_budget = max(1, int(eval_budget))
        if len(ordered) <= eval_budget:
            return ordered
        survivor_count = eval_budget
        survivors = []
        if self.two_stage_random_survivor and eval_budget >= 2:
            survivor_count = max(1, eval_budget - 1)
        survivors.extend(ordered[:survivor_count])
        if self.two_stage_random_survivor and len(ordered) > survivor_count:
            random_idx = int(np.random.randint(survivor_count, len(ordered)))
            survivors.append(ordered[random_idx])
        return survivors[:eval_budget]

    def _generate_two_stage_heavy_candidate(
        self,
        solution,
        action_idx,
        raw_budget_override=None,
        eval_budget_override=None,
        phase_tag="main",
        enable_local_proxy=False,
    ):
        action_idx = int(action_idx)
        fallback_phase = "elite" if str(phase_tag) == "elite" else "main"
        telemetry = self.two_stage_telemetry.setdefault(
            action_idx,
            {
                "calls": 0,
                "raw_candidates": 0,
                "stage1_survivors": 0,
                "stage2_evaluated": 0,
                "fallbacks": 0,
            },
        )
        telemetry["calls"] += 1
        self._increment_named_counter(telemetry, "mode_calls", str(phase_tag))
        if enable_local_proxy:
            local_proxy_mix, local_proxy_phase = self._effective_two_stage_local_proxy_mix(action_idx)
        else:
            local_proxy_mix, local_proxy_phase = 0.0, "disabled"
        self._increment_named_counter(telemetry, "phase_calls", local_proxy_phase)
        telemetry["local_proxy_mix_sum"] = float(telemetry.get("local_proxy_mix_sum", 0.0)) + float(local_proxy_mix)
        telemetry["local_proxy_mix_count"] = int(telemetry.get("local_proxy_mix_count", 0)) + 1

        context = self._prepare_two_stage_context(solution)
        if context is None:
            telemetry["fallbacks"] += 1
            return self._generate_candidate_by_action_fallback(solution, action_idx, phase=fallback_phase)

        if raw_budget_override is None or eval_budget_override is None:
            raw_budget, eval_budget = self._effective_two_stage_budgets(action_idx)
        else:
            raw_budget = int(max(1, raw_budget_override))
            eval_budget = int(max(1, eval_budget_override))
        proposals = self._propose_two_stage_candidates(context, action_idx, raw_budget)
        telemetry["raw_candidates"] += int(len(proposals))
        for _permutation, _bay, meta in proposals:
            self._record_two_stage_template_counter(telemetry, "raw_template_counts", meta)
        if not proposals:
            telemetry["fallbacks"] += 1
            return self._generate_candidate_by_action_fallback(solution, action_idx, phase=fallback_phase)

        def _safe_record_float(value, fallback=1e18):
            try:
                value = float(value)
            except Exception:
                return float(fallback)
            if not np.isfinite(value):
                return float(fallback)
            return float(value)

        def _evaluate_entry(entry):
            cache_key = (tuple(entry["permutation"]), tuple(entry["bay"]))
            if cache_key not in evaluation_cache:
                evaluation_cache[cache_key] = self._build_evaluated_candidate_from_layout(
                    solution,
                    entry["permutation"],
                    entry["bay"],
                )
            return evaluation_cache.get(cache_key)

        group_id = (
            f"{self.runtime_config_name}:{int(getattr(self, 'current_global_step', 0))}:"
            f"{str(phase_tag)}:{int(action_idx)}:{int(telemetry['calls'])}"
        )
        scored_candidates = []
        evaluation_cache = {}
        for proposal_idx, (permutation, bay, meta) in enumerate(proposals):
            proxy_score = self._score_two_stage_proxy(
                context,
                permutation,
                bay,
                action_idx,
                meta=meta,
                local_proxy_mix=local_proxy_mix,
            )
            if not np.isfinite(proxy_score):
                continue
            learned_score = None
            learned_record = None
            if self.two_stage_learned_evaluator_enabled:
                learned_score, learned_record = self.dqn_two_stage_evaluator.score_candidate(
                    self,
                    context,
                    permutation,
                    bay,
                    action_idx,
                    meta=meta,
                )
            stage1_score = (
                float(learned_score)
                if learned_score is not None and np.isfinite(learned_score)
                else float(proxy_score)
            )
            scored_candidates.append(
                {
                    "proposal_idx": int(proposal_idx),
                    "permutation": self._to_int_list(permutation),
                    "bay": self._to_int_list(bay),
                    "meta": meta,
                    "proxy_score": float(proxy_score),
                    "learned_score": (
                        float(learned_score)
                        if learned_score is not None and np.isfinite(learned_score)
                        else None
                    ),
                    "stage1_score": float(stage1_score),
                    "learned_record": learned_record,
                    "selected_survivor": False,
                    "selected_best": False,
                }
            )
        if not scored_candidates:
            telemetry["fallbacks"] += 1
            return self._generate_candidate_by_action_fallback(solution, action_idx, phase=fallback_phase)

        survivors = self._select_two_stage_survivors(
            [(entry["stage1_score"], entry) for entry in scored_candidates],
            eval_budget,
        )
        telemetry["stage1_survivors"] += int(len(survivors))
        for _stage1_score, survivor_entry in survivors:
            survivor_entry["selected_survivor"] = True
            self._record_two_stage_template_counter(
                telemetry,
                "survivor_template_counts",
                survivor_entry["meta"],
            )
        best_candidate = None
        best_rank = None
        best_meta = None
        best_entry = None
        labeled_candidates = []
        if (
            self.dqn_two_stage_evaluator.collect_data
            and self.dqn_two_stage_evaluator.collect_full_labels
        ):
            for candidate_entry in scored_candidates:
                candidate = _evaluate_entry(candidate_entry)
                if candidate is None:
                    continue
                candidate_entry["candidate_rank_key"] = self._candidate_rank_key(candidate)
                labeled_candidates.append(candidate_entry)

        for _stage1_score, survivor_entry in survivors:
            candidate = _evaluate_entry(survivor_entry)
            if candidate is None:
                continue
            telemetry["stage2_evaluated"] += 1
            self._record_two_stage_template_counter(
                telemetry,
                "stage2_template_counts",
                survivor_entry["meta"],
            )
            candidate_rank = self._candidate_rank_key(candidate)
            survivor_entry["candidate_rank_key"] = candidate_rank
            if not self.dqn_two_stage_evaluator.collect_full_labels:
                labeled_candidates.append(survivor_entry)
            if best_candidate is None or candidate_rank < best_rank:
                best_candidate = candidate
                best_rank = candidate_rank
                best_meta = survivor_entry["meta"]
                best_entry = survivor_entry

        if self.dqn_two_stage_evaluator.collect_data and labeled_candidates:
            unique_labeled_candidates = {}
            for candidate_entry in labeled_candidates:
                unique_labeled_candidates[int(candidate_entry["proposal_idx"])] = candidate_entry
            ordered_labeled_candidates = sorted(
                unique_labeled_candidates.values(),
                key=lambda item: item.get("candidate_rank_key", (1, 10**9, float("inf"), float("inf"), float("inf"))),
            )
            dataset_records = []
            for rank_index, candidate_entry in enumerate(ordered_labeled_candidates):
                learned_record = candidate_entry.get("learned_record")
                if learned_record is None:
                    learned_record = self.dqn_two_stage_evaluator.build_graph_record(
                        self,
                        context,
                        candidate_entry["permutation"],
                        candidate_entry["bay"],
                        action_idx,
                        meta=candidate_entry["meta"],
                    )
                if learned_record is None:
                    continue
                feasible_rank, d_inf, cost, violation, mhc = candidate_entry["candidate_rank_key"]
                candidate_entry["selected_best"] = (
                    best_entry is not None
                    and int(candidate_entry["proposal_idx"]) == int(best_entry["proposal_idx"])
                )
                dataset_records.append(
                    {
                        "group_id": group_id,
                        "runtime_config": self.runtime_config_name,
                        "global_step": int(getattr(self, "current_global_step", 0)),
                        "phase_tag": str(phase_tag),
                        "label_scope": (
                            "full"
                            if self.dqn_two_stage_evaluator.collect_full_labels
                            else "survivor"
                        ),
                        "action_idx": int(action_idx),
                        "proposal_idx": int(candidate_entry["proposal_idx"]),
                        "template": str(learned_record.get("template", "default")),
                        "proxy_score": _safe_record_float(candidate_entry.get("proxy_score", 0.0)),
                        "learned_score": (
                            _safe_record_float(candidate_entry.get("learned_score"))
                            if candidate_entry.get("learned_score") is not None
                            else None
                        ),
                        "stage1_score": _safe_record_float(candidate_entry.get("stage1_score", 0.0)),
                        "selected_survivor": bool(candidate_entry.get("selected_survivor", False)),
                        "selected_best": bool(candidate_entry.get("selected_best", False)),
                        "true_rank_index": int(rank_index),
                        "feasible_rank": int(feasible_rank),
                        "d_inf": int(d_inf),
                        "cost": _safe_record_float(cost),
                        "violation": _safe_record_float(violation),
                        "mhc": _safe_record_float(mhc),
                        "node_features": learned_record["node_features"],
                        "global_features": learned_record["global_features"],
                        "edge_index": learned_record["edge_index"],
                        "edge_weight": learned_record["edge_weight"],
                    }
                )
            self.dqn_two_stage_evaluator.append_records(dataset_records)

        if best_candidate is not None:
            self._record_two_stage_template_counter(telemetry, "selected_template_counts", best_meta)
            self._increment_named_counter(telemetry, "phase_selected_counts", local_proxy_phase)
            self._increment_named_counter(telemetry, "mode_selected_counts", str(phase_tag))
            return best_candidate
        telemetry["fallbacks"] += 1
        return self._generate_candidate_by_action_fallback(solution, action_idx, phase=fallback_phase)

    def _build_topk_pair_rows(self, solution):
        tm = np.asarray(solution.TM, dtype=float)
        if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
            return []
        rows = []
        n = int(tm.shape[0])
        for i in range(n):
            for j in range(i + 1, n):
                pair_mhc = float(tm[i, j] + tm[j, i])
                if pair_mhc > 0:
                    rows.append((i, j, pair_mhc))
        rows.sort(key=lambda item: item[2], reverse=True)
        return rows

    def _bay_position_ranges(self, bay):
        bay_arr = np.asarray(bay, dtype=int).reshape(-1).copy()
        if bay_arr.size == 0:
            return []
        bay_arr[-1] = 1
        end_positions = np.where(bay_arr == 1)[0].tolist()
        start_positions = [0] + [int(pos) + 1 for pos in end_positions[:-1]]
        return [
            (int(start), int(end))
            for start, end in zip(start_positions, end_positions)
            if int(start) <= int(end)
        ]

    def _collect_guided_target_positions(self, perm, bay, endpoint_positions):
        n = len(perm)
        endpoint_positions = sorted({int(pos) for pos in endpoint_positions if 0 <= int(pos) < n})
        if not endpoint_positions:
            return list(range(n))

        bay_ranges = self._bay_position_ranges(bay)
        boundary_positions = set()
        candidate_positions = set(endpoint_positions)
        for start, end in bay_ranges:
            boundary_positions.update({int(start), int(end)})
            if start + 1 <= end:
                boundary_positions.add(int(start + 1))
            if end - 1 >= start:
                boundary_positions.add(int(end - 1))

        candidate_positions.update(boundary_positions)
        for pos in endpoint_positions:
            for delta in range(1, self.topk_guided_window_radius + 1):
                if pos - delta >= 0:
                    candidate_positions.add(int(pos - delta))
                if pos + delta < n:
                    candidate_positions.add(int(pos + delta))
            for bay_idx, (start, end) in enumerate(bay_ranges):
                if start <= pos <= end:
                    candidate_positions.update({int(start), int(end)})
                    if bay_idx > 0:
                        prev_start, prev_end = bay_ranges[bay_idx - 1]
                        candidate_positions.update({int(prev_start), int(prev_end)})
                    if bay_idx + 1 < len(bay_ranges):
                        next_start, next_end = bay_ranges[bay_idx + 1]
                        candidate_positions.update({int(next_start), int(next_end)})
                    break

        ordered_positions = sorted(
            candidate_positions,
            key=lambda pos: (
                0 if pos in endpoint_positions else 1 if pos in boundary_positions else 2,
                min(abs(int(pos) - int(endpoint)) for endpoint in endpoint_positions),
                int(pos),
            ),
        )
        return ordered_positions[: min(len(ordered_positions), self.topk_guided_target_position_cap)]

    def _collect_guided_insert_positions(self, perm, bay, endpoint_positions):
        target_positions = self._collect_guided_target_positions(perm, bay, endpoint_positions)
        max_insert_idx = max(0, len(perm) - 1)
        insert_positions = {0, max_insert_idx}
        for pos in target_positions:
            insert_positions.add(int(min(max_insert_idx, pos)))
            insert_positions.add(int(min(max_insert_idx, pos + 1)))
        ordered_positions = sorted(
            insert_positions,
            key=lambda pos: (
                min(abs(int(pos) - int(endpoint)) for endpoint in endpoint_positions) if endpoint_positions else 0,
                int(pos),
            ),
        )
        return ordered_positions

    def _topk_guided_enhanced_local_search(self, solution, topk=None, max_iters=None):
        current = copy.deepcopy(solution)
        self._evaluate_solution(current)
        if not current.current_is_feasible or not np.isfinite(current.fitness):
            return current

        topk = max(1, int(self.topk_guided_topk if topk is None else topk))
        max_iters = max(1, int(self.topk_guided_max_iters if max_iters is None else max_iters))

        for _ in range(max_iters):
            pair_rows = self._build_topk_pair_rows(current)
            if not pair_rows:
                break

            top_rows = pair_rows[: max(1, min(topk, len(pair_rows)))]
            endpoint_ids = sorted({int(row[0]) for row in top_rows} | {int(row[1]) for row in top_rows})
            endpoint_facilities = [int(node_id + 1) for node_id in endpoint_ids]

            perm = self._to_int_list(current.fbs_model.permutation)
            bay = self._to_int_list(current.fbs_model.bay)
            position_map = {int(facility): int(index) for index, facility in enumerate(perm)}
            endpoint_positions = sorted(
                {
                    int(position_map[facility])
                    for facility in endpoint_facilities
                    if facility in position_map
                }
            )
            if not endpoint_positions:
                break

            n = len(perm)
            guided_target_positions = self._collect_guided_target_positions(perm, bay, endpoint_positions)
            guided_insert_positions = self._collect_guided_insert_positions(perm, bay, endpoint_positions)
            best_candidate = None
            best_candidate_cost = float(current.fitness)

            # 1) ????????????
            for p1 in endpoint_positions:
                for p2 in guided_target_positions:
                    if p1 == p2:
                        continue
                    candidate_perm = perm.copy()
                    candidate_perm[p1], candidate_perm[p2] = candidate_perm[p2], candidate_perm[p1]
                    candidate = self._build_candidate_from_layout(current, candidate_perm, bay)
                    if candidate is None:
                        continue
                    if candidate.fitness + 1e-9 < best_candidate_cost:
                        best_candidate = candidate
                        best_candidate_cost = float(candidate.fitness)

            # 2) ???????????
            for src in endpoint_positions:
                for dst in guided_insert_positions:
                    if src == dst:
                        continue
                    candidate_perm = perm.copy()
                    facility = candidate_perm.pop(src)
                    candidate_perm.insert(dst, facility)
                    candidate = self._build_candidate_from_layout(current, candidate_perm, bay)
                    if candidate is None:
                        continue
                    if candidate.fitness + 1e-9 < best_candidate_cost:
                        best_candidate = candidate
                        best_candidate_cost = float(candidate.fitness)

            # 3) ???????
            base_perm = perm if best_candidate is None else self._to_int_list(best_candidate.fbs_model.permutation)
            for bay_candidate in self._propose_boundary_moves(bay):
                candidate = self._build_candidate_from_layout(current, base_perm, bay_candidate)
                if candidate is None:
                    continue
                if candidate.fitness + 1e-9 < best_candidate_cost:
                    best_candidate = candidate
                    best_candidate_cost = float(candidate.fitness)

            if best_candidate is None:
                break

            current = best_candidate
            self._observe_feasible_state(current)

        return current

    def _estimate_group_geometry_pressure(self, members, group_area_sum, area_array, aspect_limits, H):
        if len(members) == 0 or not np.isfinite(group_area_sum) or group_area_sum <= 0 or not np.isfinite(H) or H <= 0:
            return 0.0
        width = max(float(group_area_sum) / float(H), 1e-8)
        indices = np.asarray(members, dtype=int).reshape(-1)
        member_areas = np.asarray(area_array, dtype=float).reshape(-1)[indices]
        lengths = member_areas / width
        min_side = np.minimum(lengths, width)
        aspect_ratio = np.divide(
            np.maximum(lengths, width),
            np.maximum(min_side, 1e-8),
        )
        aspect_limits_arr = np.asarray(aspect_limits, dtype=float).reshape(-1)
        if aspect_limits_arr.size == np.asarray(area_array, dtype=float).reshape(-1).size:
            limits = aspect_limits_arr[indices]
        else:
            limits = np.full(aspect_ratio.shape, np.inf, dtype=float)
        valid_limits = np.where(
            np.isfinite(limits) & (limits > 0),
            limits,
            np.full(aspect_ratio.shape, np.inf, dtype=float),
        )
        pressure = np.maximum(aspect_ratio - valid_limits, 0.0) / np.maximum(valid_limits, 1.0)
        if pressure.size == 0:
            return 0.0
        return float(np.mean(pressure))

    def _score_high_flow_group(self, node, members, group_area_sum, sym_flow, area_array, target_area, aspect_limits, H):
        if len(members) == 0:
            flow_score = 0.0
        else:
            flow_score = float(np.mean(sym_flow[int(node), np.asarray(members, dtype=int)]))
        projected_members = list(members) + [int(node)]
        projected_area = float(group_area_sum) + float(area_array[int(node)])
        area_penalty = abs(projected_area - float(target_area)) / max(float(target_area), 1e-8)
        geometry_penalty = self._estimate_group_geometry_pressure(
            projected_members,
            projected_area,
            area_array,
            aspect_limits,
            H,
        )
        return float(
            flow_score
            - self.high_flow_area_balance_weight * area_penalty
            - self.high_flow_geometry_pressure_weight * geometry_penalty
        )

    def _score_high_flow_group_final(self, members, sym_flow, group_area_sum, target_area, area_array, aspect_limits, H):
        if len(members) <= 1:
            flow_score = 0.0
        else:
            sub_matrix = sym_flow[np.ix_(members, members)]
            flow_score = float(np.sum(sub_matrix))
        area_penalty = abs(float(group_area_sum) - float(target_area)) / max(float(target_area), 1e-8)
        geometry_penalty = self._estimate_group_geometry_pressure(
            members,
            group_area_sum,
            area_array,
            aspect_limits,
            H,
        )
        return float(
            flow_score
            - self.high_flow_area_balance_weight * area_penalty
            - self.high_flow_geometry_pressure_weight * geometry_penalty
        )

    def _build_high_flow_grouped_perm(self, flow, bay_sizes, area=None, aspect_limits=None, H=None):
        flow_matrix = np.asarray(flow, dtype=float)
        if flow_matrix.ndim != 2 or flow_matrix.shape[0] != flow_matrix.shape[1]:
            return []
        n = int(flow_matrix.shape[0])
        bay_sizes = [int(size) for size in bay_sizes if int(size) > 0]
        if not bay_sizes or sum(bay_sizes) != n:
            return []

        group_count = len(bay_sizes)
        sym_flow = flow_matrix + flow_matrix.T
        node_strength = np.sum(sym_flow, axis=1)
        seed_order = np.argsort(-node_strength)
        area_array = np.asarray(area, dtype=float).reshape(-1)
        if area_array.size != n:
            area_array = np.ones(n, dtype=float)
        aspect_limits_arr = np.asarray(aspect_limits, dtype=float).reshape(-1)
        if aspect_limits_arr.size != n:
            aspect_limits_arr = np.full(n, np.inf, dtype=float)
        H_value = max(float(H) if H is not None else 0.0, 1e-8)
        target_area = float(np.sum(area_array)) / float(max(1, group_count))

        groups = [[] for _ in range(group_count)]
        capacities = bay_sizes.copy()
        group_area_sums = [0.0 for _ in range(group_count)]
        used_nodes = set()
        seed_pointer = 0

        for group_idx in range(group_count):
            while seed_pointer < n and int(seed_order[seed_pointer]) in used_nodes:
                seed_pointer += 1
            if seed_pointer >= n or capacities[group_idx] <= 0:
                continue
            seed_node = int(seed_order[seed_pointer])
            groups[group_idx].append(seed_node)
            used_nodes.add(seed_node)
            capacities[group_idx] -= 1
            group_area_sums[group_idx] += float(area_array[seed_node])

        remaining_nodes = [node for node in range(n) if node not in used_nodes]
        for node in remaining_nodes:
            best_group = None
            best_score = -float("inf")
            for group_idx in range(group_count):
                if capacities[group_idx] <= 0:
                    continue
                score = self._score_high_flow_group(
                    node,
                    groups[group_idx],
                    group_area_sums[group_idx],
                    sym_flow,
                    area_array,
                    target_area,
                    aspect_limits_arr,
                    H_value,
                )
                if score > best_score:
                    best_score = score
                    best_group = group_idx
            if best_group is None:
                continue
            groups[best_group].append(int(node))
            capacities[best_group] -= 1
            group_area_sums[best_group] += float(area_array[int(node)])
            used_nodes.add(int(node))

        leftover_nodes = [node for node in range(n) if node not in used_nodes]
        for group_idx in range(group_count):
            while capacities[group_idx] > 0 and leftover_nodes:
                node = int(leftover_nodes.pop(0))
                groups[group_idx].append(node)
                capacities[group_idx] -= 1
                group_area_sums[group_idx] += float(area_array[node])

        group_scores = []
        for group_idx, members in enumerate(groups):
            score = self._score_high_flow_group_final(
                members,
                sym_flow,
                group_area_sums[group_idx],
                target_area,
                area_array,
                aspect_limits_arr,
                H_value,
            )
            group_scores.append((group_idx, score))
        group_scores.sort(key=lambda item: item[1], reverse=True)

        permutation = []
        for group_idx, _score in group_scores:
            members = list(groups[group_idx])
            members.sort(key=lambda node: float(np.sum(sym_flow[node, members])) if members else 0.0, reverse=True)
            permutation.extend([int(node + 1) for node in members])

        return permutation

    def _build_high_flow_grouped_perm_random(self, flow, bay_sizes, area=None, aspect_limits=None, H=None):
        flow_matrix = np.asarray(flow, dtype=float)
        if flow_matrix.ndim != 2 or flow_matrix.shape[0] != flow_matrix.shape[1]:
            return []
        n = int(flow_matrix.shape[0])
        bay_sizes = [int(size) for size in bay_sizes if int(size) > 0]
        if not bay_sizes or sum(bay_sizes) != n:
            return []

        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        group_count = len(bay_sizes)
        sym_flow = flow_matrix + flow_matrix.T
        node_strength = np.sum(sym_flow, axis=1)
        ranked_nodes = np.argsort(-node_strength).tolist()
        top_pool = ranked_nodes[: max(group_count * 3, group_count)]
        area_array = np.asarray(area, dtype=float).reshape(-1)
        if area_array.size != n:
            area_array = np.ones(n, dtype=float)
        aspect_limits_arr = np.asarray(aspect_limits, dtype=float).reshape(-1)
        if aspect_limits_arr.size != n:
            aspect_limits_arr = np.full(n, np.inf, dtype=float)
        H_value = max(float(H) if H is not None else 0.0, 1e-8)
        target_area = float(np.sum(area_array)) / float(max(1, group_count))

        groups = [[] for _ in range(group_count)]
        capacities = bay_sizes.copy()
        group_area_sums = [0.0 for _ in range(group_count)]
        used_nodes = set()

        replace_seed = len(top_pool) < group_count
        seed_nodes = rng.choice(np.asarray(top_pool, dtype=int), size=group_count, replace=replace_seed).tolist()
        for group_idx, seed_node in enumerate(seed_nodes):
            if capacities[group_idx] <= 0:
                continue
            seed_node = int(seed_node)
            if seed_node in used_nodes:
                continue
            groups[group_idx].append(seed_node)
            used_nodes.add(seed_node)
            capacities[group_idx] -= 1
            group_area_sums[group_idx] += float(area_array[seed_node])

        remaining_nodes = [node for node in range(n) if node not in used_nodes]
        rng.shuffle(remaining_nodes)
        for node in remaining_nodes:
            scores = []
            valid_groups = []
            for group_idx in range(group_count):
                if capacities[group_idx] <= 0:
                    continue
                score = self._score_high_flow_group(
                    node,
                    groups[group_idx],
                    group_area_sums[group_idx],
                    sym_flow,
                    area_array,
                    target_area,
                    aspect_limits_arr,
                    H_value,
                )
                scores.append(score)
                valid_groups.append(group_idx)
            if not valid_groups:
                continue
            scores_arr = np.asarray(scores, dtype=float)
            temperature = max(1e-6, float(np.std(scores_arr)) + 1e-6)
            probs = np.exp((scores_arr - np.max(scores_arr)) / temperature)
            probs_sum = float(np.sum(probs))
            if (not np.isfinite(probs_sum)) or probs_sum <= 0:
                chosen_group = int(valid_groups[int(np.argmax(scores_arr))])
            else:
                probs = probs / probs_sum
                chosen_group = int(rng.choice(np.asarray(valid_groups, dtype=int), p=probs))
            groups[chosen_group].append(int(node))
            capacities[chosen_group] -= 1
            group_area_sums[chosen_group] += float(area_array[int(node)])
            used_nodes.add(int(node))

        leftover_nodes = [node for node in range(n) if node not in used_nodes]
        for group_idx in range(group_count):
            while capacities[group_idx] > 0 and leftover_nodes:
                node = int(leftover_nodes.pop(0))
                groups[group_idx].append(node)
                capacities[group_idx] -= 1
                group_area_sums[group_idx] += float(area_array[node])

        group_scores = []
        for group_idx, members in enumerate(groups):
            score = self._score_high_flow_group_final(
                members,
                sym_flow,
                group_area_sums[group_idx],
                target_area,
                area_array,
                aspect_limits_arr,
                H_value,
            )
            group_scores.append((group_idx, score))
        rng.shuffle(group_scores)
        group_scores.sort(key=lambda item: item[1], reverse=True)

        permutation = []
        for group_idx, _score in group_scores:
            members = list(groups[group_idx])
            rng.shuffle(members)
            members.sort(key=lambda node: float(np.sum(sym_flow[node, members])) if members else 0.0, reverse=True)
            permutation.extend([int(node + 1) for node in members])

        return permutation

    def _high_flow_group_warmstart(self, base_solution, stage_tag="init"):
        base = copy.deepcopy(base_solution)
        self._evaluate_solution(base)
        if not base.current_is_feasible or not np.isfinite(base.fitness):
            return base

        bay_sizes = self._bay_sizes_from_bay(base.fbs_model.bay)
        if not bay_sizes:
            return base
        bay_template = self._make_bay_by_sizes(bay_sizes)
        flow_matrix = np.asarray(base.F, dtype=float)
        if flow_matrix.ndim != 2 or flow_matrix.shape[0] != flow_matrix.shape[1]:
            return base

        best = base
        best_cost = float(best.fitness)
        stage_is_episode0 = str(stage_tag).strip().lower() == "episode0"
        restarts = max(1, int(self.high_flow_warmstart_restarts))
        if stage_is_episode0:
            restarts = min(restarts, int(self.high_flow_warmstart_episode0_restarts))
        raw_candidates = []
        for restart_idx in range(restarts):
            if restart_idx == 0:
                grouped_perm = self._build_high_flow_grouped_perm(
                    flow_matrix,
                    bay_sizes,
                    area=base.areas,
                    aspect_limits=base.aspect_limits,
                    H=base.H,
                )
            else:
                grouped_perm = self._build_high_flow_grouped_perm_random(
                    flow_matrix,
                    bay_sizes,
                    area=base.areas,
                    aspect_limits=base.aspect_limits,
                    H=base.H,
                )
            if not grouped_perm:
                continue
            candidate = self._build_candidate_from_layout(base, grouped_perm, bay_template)
            if candidate is None:
                continue
            if stage_is_episode0:
                raw_candidates.append(candidate)
            elif self.high_flow_warmstart_use_topk and self.topk_guided_enabled:
                candidate = self._topk_guided_enhanced_local_search(
                    candidate,
                    topk=self.high_flow_warmstart_topk,
                    max_iters=self.high_flow_warmstart_max_iters,
                )
            if (
                candidate.current_is_feasible
                and np.isfinite(candidate.fitness)
                and candidate.fitness + 1e-9 < best_cost
            ):
                best = candidate
                best_cost = float(candidate.fitness)

        if stage_is_episode0 and raw_candidates:
            feasible_candidates = [
                candidate
                for candidate in raw_candidates
                if candidate.current_is_feasible and np.isfinite(candidate.fitness)
            ]
            feasible_candidates.sort(key=lambda candidate: float(candidate.fitness))
            if feasible_candidates:
                best = feasible_candidates[0]
                best_cost = float(best.fitness)
                if self.high_flow_warmstart_use_topk and self.topk_guided_enabled:
                    refine_count = min(
                        len(feasible_candidates),
                        int(self.high_flow_warmstart_episode0_refine_topk_count),
                    )
                    for candidate in feasible_candidates[:refine_count]:
                        refined = self._topk_guided_enhanced_local_search(
                            copy.deepcopy(candidate),
                            topk=self.high_flow_warmstart_episode0_topk,
                            max_iters=self.high_flow_warmstart_episode0_max_iters,
                        )
                        if (
                            refined.current_is_feasible
                            and np.isfinite(refined.fitness)
                            and refined.fitness + 1e-9 < best_cost
                        ):
                            best = refined
                            best_cost = float(refined.fitness)

        if best_cost + 1e-9 < float(base.fitness):
            logger.info(
                f"High-flow warmstart improved | stage: {stage_tag} | "
                f"{float(base.fitness):.6f} -> {best_cost:.6f} | restarts: {restarts}"
            )
        return best

    def _prepare_episode_start(self, episode_idx):
        self._refresh_bin_width_from_best()
        if self.reheat_reset_each_episode:
            self.reheat_episode_count = 0
        if self.accept_window_reset_each_episode:
            self.accept_window.clear()
            self.accept_rate_window = 1.0
        if episode_idx == 0:
            if (
                self.high_flow_warmstart_enabled
                and self.s.current_is_feasible
                and np.isfinite(self.s.fitness)
            ):
                candidate = self._high_flow_group_warmstart(self.s, stage_tag="episode0")
                if (
                    candidate.current_is_feasible
                    and np.isfinite(candidate.fitness)
                    and candidate.fitness + 1e-9 < self.s.fitness
                ):
                    self.s = candidate
                    self.current_energy = self.s.fitness
                    self.no_improve_steps = 0
                    self._observe_feasible_state(self.s)
            return
        if self.episode_chain_restart_enabled:
            if not self._restart_from_best_feasible():
                return
        if (
            self.high_flow_warmstart_enabled
            and self.episodes_without_improvement >= self.high_flow_warmstart_stagnation_episodes
            and self.s.current_is_feasible
            and np.isfinite(self.s.fitness)
        ):
            candidate = self._high_flow_group_warmstart(self.s, stage_tag="stagnation")
            if (
                candidate.current_is_feasible
                and np.isfinite(candidate.fitness)
                and candidate.fitness + 1e-9 < self.s.fitness
            ):
                self.s = candidate
                self.current_energy = self.s.fitness
                self.no_improve_steps = 0
                self._observe_feasible_state(self.s)
        recipe = None
        if self.episodes_without_improvement >= 5:
            recipe = self.diversify_recipe
        elif self.episodes_without_improvement >= 2:
            recipe = self.light_restart_recipe
        if recipe is None:
            return
        candidate = copy.deepcopy(self.s)
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        if candidate.current_is_feasible:
            self.s = candidate
            self.current_energy = self.s.fitness
            self._observe_feasible_state(self.s)

    def _elite_intensify_seed(self, base_solution, fast_time, max_rounds, action_trials_override=None):
        base = copy.deepcopy(base_solution)
        self._evaluate_solution(base)
        if not base.current_is_feasible or not np.isfinite(base.fitness):
            return base, fast_time, False

        improved_any = False
        for _ in range(max_rounds):
            best_candidate = None
            best_action_idx = None
            best_candidate_cost = float(base.fitness)

            for action_idx in self.elite_actions:
                if action_trials_override is None:
                    trial_count = self.elite_action_trials.get(action_idx, 1)
                else:
                    trial_count = action_trials_override.get(action_idx, self.elite_action_trials.get(action_idx, 1))
                for _trial in range(trial_count):
                    candidate = self.generate_candidate_by_action(base, action_idx, phase="elite")
                    self._record_action_selection(
                        action_idx,
                        base.fitness,
                        candidate.fitness,
                        phase="elite",
                    )
                    if (
                        candidate.current_is_feasible
                        and np.isfinite(candidate.fitness)
                        and candidate.fitness < best_candidate_cost
                    ):
                        best_candidate = candidate
                        best_action_idx = action_idx
                        best_candidate_cost = float(candidate.fitness)

            if best_candidate is None or not (best_candidate_cost < float(base.fitness)):
                break

            improved_any = True
            self._record_action_acceptance(
                best_action_idx,
                base.fitness,
                best_candidate_cost,
                improved=True,
                phase="elite",
            )
            base = best_candidate
            archive_improved = self._observe_feasible_state(base)
            if archive_improved:
                self._record_action_global_best(best_action_idx, phase="elite")
                fast_time = datetime.datetime.now()

        return base, fast_time, improved_any

    def _elite_intensification(self, fast_time):
        elite_controls = self._effective_elite_controls()
        seed_entries = self._select_elite_seed_entries(seed_limit=elite_controls["seed_limit"])
        if not seed_entries:
            return fast_time, False

        self.elite_trigger_count += 1
        improved_any = False
        best_result = None
        best_result_initial_cost = float("inf")
        rounds_per_seed = max(
            1,
            int(math.ceil(float(elite_controls["total_rounds"]) / float(max(1, len(seed_entries))))),
        )

        for seed_entry in seed_entries:
            seed_solution = copy.deepcopy(seed_entry["solution"])
            self._evaluate_solution(seed_solution)
            if not seed_solution.current_is_feasible or not np.isfinite(seed_solution.fitness):
                continue
            self._mark_archive_entry_used(seed_entry, self.current_global_step)
            seed_initial_cost = float(seed_solution.fitness)
            seed_result, fast_time, seed_improved = self._elite_intensify_seed(
                seed_solution,
                fast_time,
                rounds_per_seed,
                action_trials_override=elite_controls["trial_map"],
            )
            if not seed_improved:
                continue
            improved_any = True
            if (
                best_result is None
                or float(seed_result.fitness) + 1e-9 < float(best_result.fitness)
            ):
                best_result = seed_result
                best_result_initial_cost = seed_initial_cost

        if improved_any and best_result is not None:
            final_gain = max(best_result_initial_cost - float(best_result.fitness), 0.0)
            self.elite_improvement_count += 1
            self.elite_total_gain += final_gain
            if self._should_takeover_after_elite(best_result, elite_controls["phase"]):
                self.s = copy.deepcopy(best_result)
                self.current_energy = self.s.fitness
                self.no_improve_steps = 0
                self._update_histogram(self.current_energy)
                self.energy_history.append(self.current_energy)
                self.modified_energy_history.append(self._tilde_energy(self.current_energy))
        return fast_time, improved_any

    @staticmethod
    def _clone_solution_snapshot_value(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        return copy.deepcopy(value)

    def _capture_solution_snapshot(self, solution):
        snapshot = {
            "permutation": self._to_int_list(solution.fbs_model.permutation),
            "bay": self._to_int_list(solution.fbs_model.bay),
        }
        for field_name in (
            "fac_x",
            "fac_y",
            "fac_b",
            "fac_h",
            "fac_aspect_ratio",
            "lower_bounds",
            "upper_bounds",
            "infeasible_mask",
            "D",
            "TM",
            "MHC",
            "fitness",
            "current_d_inf",
            "current_is_feasible",
            "current_v_worst",
            "feasible_solution_count",
            "best_feasible_cost",
            "worst_feasible_cost",
            "best_fitness",
            "state",
        ):
            snapshot[field_name] = self._clone_solution_snapshot_value(getattr(solution, field_name, None))
        return snapshot

    def _restore_solution_snapshot(self, solution, snapshot):
        solution.fbs_model.permutation = list(snapshot["permutation"])
        solution.fbs_model.bay = list(snapshot["bay"])
        for field_name, value in snapshot.items():
            if field_name in {"permutation", "bay"}:
                continue
            setattr(solution, field_name, self._clone_solution_snapshot_value(value))

    #
    # def _greedy_local_search(self, solution):
    #     """
    def _greedy_local_search(self, solution, enable_guided=False, guided_topk=None, guided_max_iters=None):
        """
        Two-stage greedy local search:
        Stage 1 swaps adjacent bays and accepts only improving feasible moves.
        Stage 2 swaps adjacent facilities within each bay and keeps only improvements.
        """
        if enable_guided and self.topk_guided_enabled:
            guided_solution = self._topk_guided_enhanced_local_search(
                solution,
                topk=self.topk_guided_topk if guided_topk is None else guided_topk,
                max_iters=self.topk_guided_max_iters if guided_max_iters is None else guided_max_iters,
            )
            if (
                guided_solution.current_is_feasible
                and np.isfinite(guided_solution.fitness)
                and guided_solution.fitness <= solution.fitness
            ):
                solution = guided_solution

        best_snapshot = self._capture_solution_snapshot(solution)

        def _commit_current_state():
            nonlocal best_snapshot
            self._observe_feasible_state(solution)
            best_snapshot = self._capture_solution_snapshot(solution)

        stage1_improved = True
        while stage1_improved:
            stage1_improved = False
            bay_structure = permutationToArray(
                best_snapshot["permutation"],
                best_snapshot["bay"],
            )
            n_bays = len(bay_structure)
            for bay_idx in range(n_bays - 1):
                new_structure = [list(current_bay) for current_bay in bay_structure]
                new_structure[bay_idx], new_structure[bay_idx + 1] = (
                    new_structure[bay_idx + 1],
                    new_structure[bay_idx],
                )
                new_perm, new_bay = arrayToPermutation(new_structure)
                solution.fbs_model.permutation = self._to_int_list(new_perm)
                solution.fbs_model.bay = self._to_int_list(new_bay)
                self._evaluate_solution(solution)
                if (
                    solution.current_is_feasible
                    and np.isfinite(solution.fitness)
                    and solution.fitness + 1e-9 < float(best_snapshot["fitness"])
                ):
                    _commit_current_state()
                    stage1_improved = True
                    break
                self._restore_solution_snapshot(solution, best_snapshot)

        stage2_improved = True
        while stage2_improved:
            stage2_improved = False
            bay_structure = permutationToArray(
                best_snapshot["permutation"],
                best_snapshot["bay"],
            )
            for bay_idx, current_bay in enumerate(bay_structure):
                current_bay = list(current_bay)
                n_fac = len(current_bay)
                if n_fac < 2:
                    continue
                for facility_idx in range(n_fac - 1):
                    new_structure = [list(item) for item in bay_structure]
                    new_structure[bay_idx][facility_idx], new_structure[bay_idx][facility_idx + 1] = (
                        new_structure[bay_idx][facility_idx + 1],
                        new_structure[bay_idx][facility_idx],
                    )
                    new_perm, new_bay = arrayToPermutation(new_structure)
                    solution.fbs_model.permutation = self._to_int_list(new_perm)
                    solution.fbs_model.bay = self._to_int_list(new_bay)
                    self._evaluate_solution(solution)
                    if (
                        solution.current_is_feasible
                        and np.isfinite(solution.fitness)
                        and solution.fitness + 1e-9 < float(best_snapshot["fitness"])
                    ):
                        _commit_current_state()
                        stage2_improved = True
                        break
                    self._restore_solution_snapshot(solution, best_snapshot)
                if stage2_improved:
                    break

        self._restore_solution_snapshot(solution, best_snapshot)
        return solution

    def _record_acceptance(self, accepted):
        self.accept_window.append(1.0 if accepted else 0.0)
        if self.accept_window:
            self.accept_rate_window = float(sum(self.accept_window)) / float(len(self.accept_window))
        else:
            self.accept_rate_window = 1.0

    def _attempt_diversification(self, global_step, force=False):
        if (not force) and (global_step - self.last_diversify_step < max(10, self.t_max // 4)):
            return
        candidate = copy.deepcopy(self.s)
        self._apply_recipe(candidate, self.diversify_recipe)
        self._evaluate_solution(candidate)
        accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
        self.prob_history.append(prob)
        if accept:
            previous_best = self.best_feasible_cost
            previous_cost = self.s.fitness
            previous_d_inf = self.s.current_d_inf
            self.s = candidate
            self.current_energy = self.s.fitness
            if self.s.current_is_feasible:
                archive_improved = self._observe_feasible_state(self.s)
            else:
                self._observe_archive_candidate(self.s)
                archive_improved = False
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
            self.no_improve_steps += 1
            reward = -0.2
        if accept:
            self._update_histogram(self.s.fitness)
        else:
            self._update_histogram(self.current_energy)
        self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
        self.energy_history.append(self.s.fitness)
        self.last_diversify_step = global_step
        self.diversification_count += 1
        self.T = max(self.T, self.T_min)
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

        self._attempt_diversification(global_step, force=True)
        fast_time, _ = self._elite_intensification(fast_time)
        if self.enable_reheat_logging:
            logger.info(
                f"Reheat triggered | step: {global_step} | T: {previous_temperature:.6f} -> {self.T:.6f} | "
                f"accept_rate_window: {self.accept_rate_window:.3f} | no_improve: {no_improve_before}"
            )
        return fast_time, True

    def _current_best_energy_for_logging(self):
        if np.isfinite(self.best_feasible_cost):
            return float(self.best_feasible_cost)
        if np.isfinite(self.best_energy):
            return float(self.best_energy)
        if np.isfinite(self.current_energy):
            return float(self.current_energy)
        return float('inf')

    def _log_training_progress(self, progress_ratio, start_time):
        elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
        progress_percent = int(round(progress_ratio * 100))
        archive_phase = self._archive_switch_phase() if self.elite_archive_enabled else "off"
        logger.info(
            f'Training progress | {progress_percent}% | best energy: {self._current_best_energy_for_logging():.6f} | '
            f'temperature: {self.T:.6f} | temp_floor: {self.T_min:.6f} | reheats: {self.reheat_trigger_count} | '
            f'archive: F{len(self.elite_archive_feasible)}/I{len(self.elite_archive_frontier)} ({archive_phase}) | '
            f'mid_structural_shots: {self.mid_structural_shot_count}/{self.mid_structural_shot_success_count} | '
            f'archive_switches: {self.archive_switch_count} | final_elite_pushes: {self.final_elite_push_count}/{self.final_elite_push_success_count} | '
            f'accept_rate_window: {self.accept_rate_window:.3f} | '
            f'elapsed: {elapsed_seconds:.1f}s'
        )

    def run(self):
        return self._run_impl()

    def _run_impl(self):
        return self.dqn_program_runner.run_training_loop(self)

def main():
    return DQNProgramRunner().run_script(
        elp_cls=ELP,
        runtime_config=runtime_config,
        logger=logger,
        gym=gym,
        experiments_util=ExperimentsUtil,
        fbs_util=FBSUtil,
        env_flag=_env_flag,
        env_int=_env_int,
        env_float=_env_float,
        env_int_list=_env_int_list,
        set_global_seed=_set_global_seed,
        os_module=os,
        copy_module=copy,
    )


if __name__ == "__main__":
    main()
