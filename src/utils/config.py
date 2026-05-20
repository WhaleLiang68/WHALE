# -*- coding: utf-8 -*-
"""
工业厂房布局优化算法配置文件
"""
import numpy as np
from pathlib import Path
import os

from sympy.utilities.codegen import Result

# ========================
# 基础路径配置
# ========================
script_dir = os.path.dirname(os.path.abspath(__file__))
print("[os] 脚本所在目录:", script_dir)
# 向上回退两级目录，移除 ua-flp-LSA\tests
base_dir = os.path.dirname(os.path.dirname(script_dir)) # 项目根目录
print("[os] 根目录:", base_dir)

# ========================
# 问题实例路径
# ========================
file_relative_path = os.path.join("data", "maoyan_cont_instances.pkl")
FILE_PATH = os.path.join(base_dir, file_relative_path)
print("实例文件所在目录:", FILE_PATH)

# ========================
# 结果保存路径
# ========================
result_relative_path = os.path.join("files", "expresults")
RESULT_PATH = os.path.join(base_dir, result_relative_path)
print("结果文件所在目录:", RESULT_PATH)
# ========================
# QLearning结果保存路径
# ========================
QLearning_result_relative_path = os.path.join("files", "QLearningResult")
QLearning_RESULT_PATH = os.path.join(base_dir, QLearning_result_relative_path)
print("QLearning结果文件所在目录:", QLearning_RESULT_PATH)
# ========================
# 问题实例参数
# ========================
FACILITY_CONFIG = {

}

# ========================
# 算法参数
# ========================
ALGORITHM = {

}

# ========================
# 实验验证配置
# ========================
VALIDATION = {

}

# ========================
# ELP/DQN 运行默认参数
# 说明：
# 1. 这些值作为主算法的默认配置来源，后续优先在这里修改。
# 2. 主算法仍然允许通过环境变量覆盖这些默认值。
# 3. 当前默认值按现有主线的有效参数整理。
# ========================
ELP_RUNTIME_CONFIG_NAME = "default_runtime"

ELP_RUNTIME_DEFAULTS = {
    "temperature_floor_target_accept": 0.03,
    "temperature_floor_quantile": 20,
    "dqn_epsilon_start": 0.50,
    "dqn_epsilon_min": 0.08,
    "dqn_epsilon_decay": 0.9994,
    "dqn_gamma": 0.95,
    "dqn_lr": 3e-4,
    "dqn_batch_size": 64,
    "dqn_replay_capacity": 50000,
    "dqn_warmup_steps": 1200,
    "dqn_update_every": 4,
    "dqn_target_update_every": 500,
    "dqn_idqn_k": 2,
    "dqn_embedding_dim": 32,
    "dqn_hidden_dim": 128,
    "dqn_grad_clip": 10.0,
    "dqn_step_epsilon_schedule": False,
    "dqn_epsilon_schedule_warmup_steps": 0,
    "dqn_epsilon_schedule_decay_ratio": 1.0,
    "dqn_guidance_enabled": False,
    "diversify_trigger_no_improve": 90,
    "reheat_enabled": True,
    "elite_actions": [10, 11, 14, 0],
    "elite_action_trials": {
        10: 4,
        11: 4,
        14: 3,
        0: 1,
    },
    "elite_max_rounds": 4,
    "accept_window_size": 120,
    "accept_window_reset_each_episode": None,
    "no_improve_reset_on_episode_restart": None,
    "reheat_reset_each_episode": None,
    "reheat_accept_rate_threshold": 0.04,
    "reheat_no_improve_threshold": 180,
    "reheat_temp_gate_ratio": 0.65,
    "reheat_target_low_ratio": 0.68,
    "reheat_target_high_ratio": 0.74,
    "reheat_progress_cap_ratio": 0.55,
    "reheat_max_per_episode": 1,
    "acceptance_phase_enabled": False,
    "acceptance_phase_mode": "two",
    "acceptance_phase_split_ratio": 0.60,
    "acceptance_phase_first_split_ratio": 0.25,
    "acceptance_phase_second_split_ratio": 0.60,
    "acceptance_early_temp_multiplier": 1.0,
    "acceptance_mid_temp_multiplier": 1.0,
    "acceptance_late_temp_multiplier_start": 1.0,
    "acceptance_late_temp_multiplier_end": 1.0,
    "reward_profile": "baseline",
    "reward_train_immediate_weight": 0.80,
    "reward_train_final_weight": 0.20,
    "action_guidance_progress_ratio": 0.60,
    "action_guidance_window": 400,
    "action_guidance_min_selected": 40,
    "action_guidance_accept_threshold": 0.015,
    "action_guidance_weight_multiplier": 0.2,
    "action_guidance_q_penalty": -0.03,
    "action_base_explore_weights": {
        0: 1.00,
        1: 0.85,
        2: 0.95,
        3: 1.00,
        9: 1.25,
        10: 1.20,
        11: 1.40,
        14: 1.35,
        15: 1.10,
    },
    "reward_rel_delta_window": 400,
    "reward_rel_delta_scale_min": 5e-4,
    "reward_rel_delta_scale_max": 1e-2,
    "reward_rel_delta_scale_default": 2e-3,
    "reward_rel_delta_scale_ema_beta": 0.92,
    "reward_cost_scale_window": 200,
    "reward_cost_scale_floor_ratio": 5e-4,
    "reward_cost_scale_cap_ratio": 2e-2,
    "reward_cost_scale_min_abs": 5.0,
    "mid_structural_shot_start_ratio": 0.42,
    "mid_structural_shot_end_ratio": 0.76,
    "mid_structural_shot_no_improve": 140,
    "mid_structural_shot_cooldown_steps": 700,
    "mid_structural_shot_max_count": 2,
    "mid_structural_shot_action_ids": [11, 14, 15],
    "mid_structural_shot_trials_per_action": 2,
    "mid_structural_shot_raw_multiplier": 1.65,
    "mid_structural_shot_eval_bonus": 3,
    "mid_structural_shot_diversity_floor": 0.16,
    "mid_structural_shot_archive_slack_ratio": 0.06,
    "mid_structural_shot_target_temp_ratio": 0.46,
    "mid_structural_shot_post_switch_cooldown_steps": 160,
    "mid_structural_shot_guided_local_search": True,
    "mid_structural_shot_guided_topk": None,
    "mid_structural_shot_guided_max_iters": None,
    "mid_structural_shot_late_archive_lock_steps": 800,
    "mid_structural_shot_late_archive_lock_gap_ratio": 0.02,
    "elite_late_progress_ratio": 0.88,
    "elite_final_progress_ratio": 0.95,
    "elite_late_no_improve": 140,
    "elite_final_no_improve": 220,
    "elite_late_round_multiplier": 1.5,
    "elite_final_round_multiplier": 2.0,
    "elite_late_trial_multiplier": 1.35,
    "elite_final_trial_multiplier": 1.85,
    "elite_late_seed_count": 2,
    "elite_final_seed_count": 1,
    "elite_chain_takeover_gain_ratio": 0.0,
    "elite_late_chain_takeover_gain_ratio": 8e-4,
    "elite_final_chain_takeover_gain_ratio": 1.5e-3,
    "final_elite_push_progress_ratio": 0.86,
    "final_elite_push_no_improve": 180,
    "final_elite_push_accept_rate_threshold": 0.03,
    "final_elite_push_cooldown_steps": 800,
    "final_elite_push_max_count": 4,
    "final_elite_push_post_switch_cooldown_steps": 200,
    "final_elite_push_guided_local_search": True,
    "final_elite_push_guided_topk": None,
    "final_elite_push_guided_max_iters": None,
    "two_stage_random_survivor": True,
    "two_stage_proposal_counts": {
        9: 14,
        10: 18,
        11: 20,
        14: 18,
        15: 18,
    },
    "two_stage_eval_counts": {
        9: 4,
        10: 5,
        11: 6,
        14: 6,
        15: 6,
    },
    "two_stage_pair_pool_cap": 10,
    "two_stage_candidate_window_radius": None,
    "two_stage_proxy_weights": {
        "order_penalty": 1.00,
        "adjacent_cross_penalty": 0.90,
        "global_cross_penalty": 0.40,
        "area_balance_penalty": 0.60,
        "geometry_penalty": 1.40,
    },
    "two_stage_action14_proxy_weights": {
        "cut_gain": 3.20,
        "pair_area_gain": 1.60,
        "geometry_gain": 1.80,
    },
    "two_stage_action11_proxy_weights": {
        "target_affinity_gain": 3.00,
        "source_damage_penalty": 2.10,
        "area_gain": 1.40,
        "geometry_gain": 1.60,
        "boundary_bonus": 0.45,
    },
    "two_stage_action15_proxy_weights": {
        "cut_gain": 2.80,
        "block_affinity_gain": 2.40,
        "pair_area_gain": 1.30,
        "geometry_gain": 1.50,
        "cohesion_bonus": 0.80,
        "exchange_bonus": 0.25,
    },
    "two_stage_local_proxy_enabled": True,
    "two_stage_local_proxy_action_ids": [11, 14, 15],
    "two_stage_local_proxy_weights": {
        "order_penalty": 0.95,
        "adjacent_cross_penalty": 1.20,
        "global_cross_penalty": 0.60,
        "area_balance_penalty": 0.95,
        "geometry_penalty": 1.15,
    },
    "two_stage_local_proxy_mix": {
        11: 0.25,
        14: 0.35,
        15: 0.30,
    },
    "two_stage_local_proxy_phase_mix": {
        "off": 1.00,
        "ramp": 0.70,
        "late": 0.08,
    },
    "two_stage_learned_evaluator_enabled": False,
    "two_stage_learned_evaluator_collect_data": False,
    "two_stage_learned_evaluator_collect_full_labels": False,
    "two_stage_learned_evaluator_model_path": r"D:\whale\DRL_FBS\two_stage_graph_evaluator.pt",
    "two_stage_learned_evaluator_dataset_path": r"D:\whale\DRL_FBS\two_stage_graph_evaluator_dataset_merged.jsonl",
    "two_stage_learned_evaluator_hidden_dim": 64,
    "two_stage_learned_evaluator_message_steps": 2,
    "two_stage_learned_evaluator_dropout": 0.05,
    "two_stage_learned_evaluator_edge_topk": 12,
    "two_stage_action14_exact_eval_enabled": True,
    "two_stage_action14_reorder_block_cap": 2,
    "two_stage_action14_reorder_variants_per_pair": 1,
    "two_stage_action14_exact_probe_topk": 4,
    "two_stage_action14_exact_select_by_post_ls_probe": False,
    "two_stage_learned_survivor_diversity_enabled": True,
    "two_stage_learned_survivor_diversity_weight": 0.22,
    "two_stage_learned_survivor_diversity_template_bonus": 0.10,
    "archive_switch_cooldown_steps": 400,
    "archive_switch_progress_gate_ratio": 0.68,
    "archive_switch_full_progress_ratio": 0.82,
    "archive_switch_target_temp_ratio": 0.40,
    "archive_switch_diversity_weight": 1.80,
    "archive_switch_quality_weight": 1.00,
    "archive_switch_staleness_weight": 0.40,
    "archive_switch_recent_use_penalty": 0.35,
    "archive_switch_feasible_slack_ratio": 0.03,
    "archive_switch_min_score": 0.80,
    "archive_switch_ramp_no_improve_multiplier": 1.35,
    "archive_switch_ramp_cooldown_multiplier": 1.50,
    "archive_switch_ramp_min_score_bonus": 0.25,
    "archive_switch_late_no_improve_multiplier": 1.00,
    "archive_switch_late_cooldown_multiplier": 1.35,
    "archive_switch_late_min_score_bonus": 0.10,
    "high_flow_warmstart_stagnation_episodes": 4,
    "high_flow_warmstart_use_topk": True,
    "high_flow_warmstart_topk": None,
    "high_flow_warmstart_max_iters": None,
    "high_flow_area_balance_weight": 0.30,
    "high_flow_geometry_pressure_weight": 0.20,
    "high_flow_warmstart_episode0_restarts": None,
    "high_flow_warmstart_episode0_refine_topk_count": 1,
    "high_flow_warmstart_episode0_topk": None,
    "high_flow_warmstart_episode0_max_iters": None,
}

def validate_config():
    """配置参数校验"""
    assert FACILITY_CONFIG["n"] == len(FACILITY_CONFIG["area"]), \
        "设施数量与面积数组长度不一致"
    assert 1.0 <= FACILITY_CONFIG["beta"] <= 5.0, \
        "长宽比限制应在1-5之间"
    assert 0 <= ALGORITHM["dynamic_programming"]["balance_weight"] <= 1, \
        "权重参数应在0-1之间"

if __name__ == "__main__":
    validate_config()
    print("配置校验通过!")
