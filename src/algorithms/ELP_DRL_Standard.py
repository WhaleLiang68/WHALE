import copy
import datetime
import math
import os
import random
from collections import deque
from src.utils.FBSUtil import permutationToArray, arrayToPermutation
import gym
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

import src
import src.utils.ExperimentsUtil as ExperimentsUtil
import src.utils.FBSUtil as FBSUtil
from src.utils.PopulationOptimizer import PopulationOptimizer

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
    strict_determinism = _env_flag("ELP_STRICT_DETERMINISM", True)
    # 统一控制 Python / NumPy / Torch 的随机源，避免同 seed 下结果漂移。
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(bool(strict_determinism), warn_only=True)
        except Exception:
            pass
        try:
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(strict_determinism)
                torch.backends.cudnn.benchmark = not bool(strict_determinism)
        except Exception:
            pass
    return bool(strict_determinism)


class StandardQLearningAgent:
    def __init__(
        self,
        s_dim,
        a_dim,
        epsilon=0.50,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        alpha=0.1,
        gamma=0.95,
    ):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.Q = np.zeros((s_dim, a_dim), dtype=float)

    def select_action(self, s, deterministic=False, allowed_actions=None):
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.a_dim))
        allowed_actions = np.asarray(allowed_actions, dtype=int).reshape(-1)
        if (not deterministic) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(allowed_actions))
        q_row = np.take(self.Q[s], allowed_actions)
        max_q = np.max(q_row)
        best_mask = np.isclose(q_row, max_q).reshape(-1)
        best_actions = allowed_actions[best_mask]
        return int(np.random.choice(best_actions))

    def update_Q(self, s, a, reward, s_next, done=False, allowed_next_actions=None):
        td_target = reward
        if not done:
            if allowed_next_actions is None or len(allowed_next_actions) == 0:
                td_target += self.gamma * np.max(self.Q[s_next])
            else:
                next_actions = np.asarray(allowed_next_actions, dtype=int).reshape(-1)
                td_target += self.gamma * np.max(np.take(self.Q[s_next], next_actions))
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_epsilon_schedule(self, global_step):
        return self.epsilon

    def set_action_guidance(self, explore_weights=None, q_bias=None):
        return None


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = deque(maxlen=self.capacity)

    def add(self, transition):
        self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, int(batch_size))
        states, actions, rewards, next_states, dones, next_masks, gamma_ns = zip(*batch)
        return (
            np.asarray(states, dtype=np.int64),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.int64),
            np.asarray(dones, dtype=np.float32),
            np.asarray(next_masks, dtype=np.bool_),
            np.asarray(gamma_ns, dtype=np.float32),
        )

    def __len__(self):
        return len(self.storage)


_QNetworkBase = nn.Module if nn is not None else object


class _QNetwork(_QNetworkBase):
    def __init__(self, state_dim, action_dim, embedding_dim=32, hidden_dim=64):
        if nn is None:
            raise ImportError('PyTorch is not available. Set ELP_RL_AGENT=qlearning or install torch.')
        super().__init__()
        self.state_dim = int(state_dim)
        self.embedding = nn.Embedding(self.state_dim, int(embedding_dim))
        self.layers = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, state_idx):
        idx = state_idx.long().clamp(0, self.state_dim - 1)
        features = self.embedding(idx)
        return self.layers(features)


class StandardDQNAgent:
    """Double-DQN with iDQN-style 2-step target (K=2)."""

    def __init__(
        self,
        s_dim,
        a_dim,
        epsilon=0.50,
        epsilon_min=0.08,
        epsilon_decay=0.9992,
        gamma=0.95,
        lr=3e-4,
        batch_size=64,
        replay_capacity=50000,
        warmup_steps=2000,
        update_every=4,
        target_update_every=400,
        idqn_k=2,
        embedding_dim=32,
        hidden_dim=64,
        grad_clip=10.0,
        step_epsilon_schedule=False,
        epsilon_warmup_steps=0,
        epsilon_decay_steps=0,
    ):
        if torch is None:
            raise ImportError('PyTorch is required for StandardDQNAgent.')

        self.s_dim = int(s_dim)
        self.a_dim = int(a_dim)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_start = float(epsilon)
        self.step_epsilon_schedule = bool(step_epsilon_schedule)
        self.epsilon_warmup_steps = int(max(0, epsilon_warmup_steps))
        self.epsilon_decay_steps = int(max(self.epsilon_warmup_steps, epsilon_decay_steps))
        self.gamma = float(gamma)

        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.update_every = int(max(1, update_every))
        self.target_update_every = int(max(1, target_update_every))
        self.idqn_k = int(max(1, idqn_k))
        self.grad_clip = float(max(0.0, grad_clip))

        self.device = torch.device('cpu')
        self.online_net = _QNetwork(
            self.s_dim,
            self.a_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net = _QNetwork(
            self.s_dim,
            self.a_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()

        self.action_explore_weights = np.ones(self.a_dim, dtype=np.float32)
        self.action_q_bias = np.zeros(self.a_dim, dtype=np.float32)

        self.replay = _ReplayBuffer(replay_capacity)
        self.n_step_buffer = deque(maxlen=self.idqn_k)
        self.total_steps = 0
        self.optimize_steps = 0

    def _build_action_mask(self, allowed_actions):
        mask = np.zeros(self.a_dim, dtype=np.bool_)
        if allowed_actions is None or len(allowed_actions) == 0:
            mask[:] = True
            return mask
        action_indices = np.asarray(allowed_actions, dtype=int).reshape(-1)
        action_indices = action_indices[(action_indices >= 0) & (action_indices < self.a_dim)]
        if action_indices.size == 0:
            mask[:] = True
            return mask
        mask[action_indices] = True
        return mask

    @staticmethod
    def _masked_argmax(q_values, action_mask):
        floor_value = torch.finfo(q_values.dtype).min
        masked_q = q_values.masked_fill(~action_mask, floor_value)
        has_valid = action_mask.any(dim=1)
        greedy = masked_q.argmax(dim=1)
        fallback = q_values.argmax(dim=1)
        return torch.where(has_valid, greedy, fallback)

    def select_action(self, s, deterministic=False, allowed_actions=None):
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.a_dim))
        allowed_actions = np.asarray(allowed_actions, dtype=int).reshape(-1)

        if (not deterministic) and (np.random.rand() < self.epsilon):
            weights = np.take(self.action_explore_weights, allowed_actions)
            weights = np.maximum(weights, 0.0)
            total_weight = float(np.sum(weights))
            if total_weight <= 0.0:
                return int(np.random.choice(allowed_actions))
            probs = weights / total_weight
            return int(np.random.choice(allowed_actions, p=probs))

        with torch.no_grad():
            state_tensor = torch.tensor([int(s)], dtype=torch.long, device=self.device)
            q_values = self.online_net(state_tensor).squeeze(0).detach().cpu().numpy()

        q_values = q_values + self.action_q_bias
        candidate_q = np.take(q_values, allowed_actions)
        best_q = np.max(candidate_q)
        best_mask = np.isclose(candidate_q, best_q).reshape(-1)
        best_actions = allowed_actions[best_mask]
        return int(np.random.choice(best_actions))

    def set_action_guidance(self, explore_weights=None, q_bias=None):
        if explore_weights is not None:
            weights = np.asarray(explore_weights, dtype=np.float32).reshape(-1)
            if weights.size == self.a_dim:
                self.action_explore_weights = np.maximum(weights, 0.0)
        if q_bias is not None:
            bias = np.asarray(q_bias, dtype=np.float32).reshape(-1)
            if bias.size == self.a_dim:
                self.action_q_bias = bias

    def update_epsilon_schedule(self, global_step):
        if not self.step_epsilon_schedule:
            return self.epsilon
        step = int(max(0, global_step))
        if step <= self.epsilon_warmup_steps:
            self.epsilon = self.epsilon_start
            return self.epsilon
        if self.epsilon_decay_steps <= self.epsilon_warmup_steps:
            self.epsilon = self.epsilon_min
            return self.epsilon
        ratio = (step - self.epsilon_warmup_steps) / float(max(1, self.epsilon_decay_steps - self.epsilon_warmup_steps))
        ratio = min(max(ratio, 0.0), 1.0)
        self.epsilon = self.epsilon_start + ratio * (self.epsilon_min - self.epsilon_start)
        return self.epsilon

    def _build_n_step_transition(self):
        state0, action0, _, _, _, _ = self.n_step_buffer[0]
        total_reward = 0.0
        discount = 1.0
        next_state = int(self.n_step_buffer[0][3])
        done_flag = False
        next_mask = self.n_step_buffer[0][5]

        horizon = min(self.idqn_k, len(self.n_step_buffer))
        for idx in range(horizon):
            _, _, reward_i, next_state_i, done_i, next_mask_i = self.n_step_buffer[idx]
            total_reward += discount * float(reward_i)
            discount *= self.gamma
            next_state = int(next_state_i)
            done_flag = bool(done_i)
            next_mask = next_mask_i
            if done_flag:
                break

        gamma_n = 0.0 if done_flag else float(discount)
        return (
            int(state0),
            int(action0),
            float(total_reward),
            int(next_state),
            bool(done_flag),
            np.asarray(next_mask, dtype=np.bool_),
            float(gamma_n),
        )

    def _append_transition(self, s, a, reward, s_next, done, allowed_next_actions):
        transition = (
            int(s),
            int(a),
            float(reward),
            int(s_next),
            bool(done),
            self._build_action_mask(allowed_next_actions),
        )
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.idqn_k:
            self.replay.add(self._build_n_step_transition())
            self.n_step_buffer.popleft()

        if done:
            while self.n_step_buffer:
                self.replay.add(self._build_n_step_transition())
                self.n_step_buffer.popleft()

    def _optimize(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        if self.total_steps % self.update_every != 0:
            return None

        states, actions, rewards, next_states, dones, next_masks, gamma_ns = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.long, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.long, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        gamma_ns_t = torch.tensor(gamma_ns, dtype=torch.float32, device=self.device)
        next_masks_t = torch.tensor(next_masks, dtype=torch.bool, device=self.device)

        q_current = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            q_online_next = self.online_net(next_states_t)
            next_actions = self._masked_argmax(q_online_next, next_masks_t).unsqueeze(1)
            q_target_next = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * gamma_ns_t * q_target_next

        loss = self.loss_fn(q_current, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.optimize_steps += 1
        if self.optimize_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def update_Q(self, s, a, reward, s_next, done=False, allowed_next_actions=None):
        self.total_steps += 1
        self._append_transition(s, a, reward, s_next, done, allowed_next_actions)
        return self._optimize()

    def decay_epsilon(self):
        if self.step_epsilon_schedule:
            return self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def finalize_episode(self):
        # Flush residual n-step transitions at episode boundary to avoid cross-episode leakage.
        while self.n_step_buffer:
            state0, action0, _, _, _, _ = self.n_step_buffer[0]
            total_reward = 0.0
            discount = 1.0
            next_state = int(self.n_step_buffer[0][3])
            next_mask = self.n_step_buffer[0][5]
            for idx in range(len(self.n_step_buffer)):
                _, _, reward_i, next_state_i, _done_i, next_mask_i = self.n_step_buffer[idx]
                total_reward += discount * float(reward_i)
                discount *= self.gamma
                next_state = int(next_state_i)
                next_mask = next_mask_i
            self.replay.add((
                int(state0),
                int(action0),
                float(total_reward),
                int(next_state),
                True,
                np.asarray(next_mask, dtype=np.bool_),
                0.0,
            ))
            self.n_step_buffer.popleft()


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
        self.cooling_per_step = 0.998
        self.temperature_floor_samples = 100
        self.temperature_floor_target_accept = 0.05
        self.temperature_floor_quantile = 25
        self.temperature_floor_cap_ratio = 0.50
        self.T_min = max(self.T_initial * 0.05, 1.0)
        self.action_recipes = {
            0: [0],
            2: [2],
            3: [3],
            6: [6],
            9: [9],
            10: [10],
            11: [11],
        }
        self.temperature_floor_action_ids = [10, 2, 0, 11]
        self.valid_actions = [0, 2, 3, 6, 9, 10, 11]
        # self.valid_actions = [0, 2, 3]
        self.action_labels = {
            0: "facility_swap",
            2: "bay_swap",
            3: "repair",
            6: "bay_shuffle",
            9: "flow_guided_swap",
            10: "segment_insert",
            11: "cross_bay_relocate",
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
        self.elite_actions = [10, 2, 11, 0]
        self.elite_action_trials = {action_idx: 1 for action_idx in self.elite_actions}
        self.elite_max_rounds = 4
        self.light_restart_recipe = [0, 3]
        self.diversify_recipe = [15, 6, 9, 3]
        self.bin_width = 50.0
        self.bin_width_schedule = [
            (8000.0, 20.0),
            (6500.0, 10.0),
            (5700.0, 5.0),
            (-np.inf, 1.0),
        ]
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
            action_idx: {
                "name": self.action_labels[action_idx],
                "selected": 0,
                "accepted": 0,
                "improved": 0,
                "global_best_hits": 0,
                "delta_sum": 0.0,
                "accepted_delta_sum": 0.0,
                "elite_selected": 0,
                "elite_accepted": 0,
                "elite_improved": 0,
                "elite_global_best_hits": 0,
                "elite_delta_sum": 0.0,
                "elite_accepted_delta_sum": 0.0,
            }
            for action_idx in self.valid_actions
        }
        self.elite_trigger_count = 0
        self.elite_improvement_count = 0
        self.elite_total_gain = 0.0
        self.enable_reheat_logging = _env_flag("ELP_ENABLE_REHEAT_LOG", False)
        self.tune_profile = os.getenv("ELP_TUNE_PROFILE", "ED1").strip().upper()
        if self.tune_profile == "ABB":
            logger.warning("ELP_TUNE_PROFILE=ABB has been retired, fallback to ABA")
            self.tune_profile = "ABA"
        supported_profiles = {"B0", "E1", "D1", "ED1", "ABA", "ABA_S1", "ABA_S2"}
        if self.tune_profile not in supported_profiles:
            logger.warning(f"Unknown ELP_TUNE_PROFILE={self.tune_profile}, fallback to ED1")
            self.tune_profile = "ED1"

        self.profile_ab = self.tune_profile in {"ABA", "ABA_S1", "ABA_S2"}
        self.profile_elp = self.tune_profile in {"E1", "ED1", "ABA", "ABA_S1", "ABA_S2"}
        self.profile_dqn = self.tune_profile in {"D1", "ED1", "ABA", "ABA_S1", "ABA_S2"}

        self.dqn_epsilon_min = 0.08
        self.dqn_epsilon_decay = 0.9992
        self.dqn_warmup_steps = 2000
        self.dqn_target_update_every = 400
        self.dqn_step_epsilon_schedule = False
        self.dqn_epsilon_schedule_warmup_steps = 0
        self.dqn_epsilon_schedule_decay_ratio = 1.0
        self.dqn_guidance_enabled = False

        self.diversify_trigger_no_improve = 120
        self.reheat_enabled = False
        self.elite_actions = [10, 11, 0]
        self.elite_action_trials = {action_idx: 1 for action_idx in self.elite_actions}
        self.elite_max_rounds = 4
        if self.profile_elp:
            self.elite_action_trials[10] = 2
            self.elite_action_trials[11] = 2
            self.diversify_trigger_no_improve = 90
            self.temperature_floor_target_accept = 0.03
            self.temperature_floor_quantile = 20
            self.reheat_enabled = True

        self.accept_window_size = 120
        self.accept_window = deque(maxlen=self.accept_window_size)
        self.accept_rate_window = 1.0
        self.accept_window_reset_each_episode = True
        self.no_improve_reset_on_episode_restart = True
        self.reheat_reset_each_episode = True
        self.reheat_accept_rate_threshold = 0.04
        self.reheat_no_improve_threshold = 180
        self.reheat_temp_gate_ratio = 0.65
        self.reheat_target_low_ratio = 0.68
        self.reheat_target_high_ratio = 0.74
        self.reheat_cooldown_steps = max(60, self.t_max // 5)
        self.reheat_progress_cap_ratio = 0.55
        self.reheat_max_per_episode = 1
        self.reheat_episode_count = 0
        self.last_reheat_step = -self.reheat_cooldown_steps
        self.reheat_trigger_count = 0

        self.acceptance_phase_enabled = False
        self.acceptance_phase_mode = "two"
        self.acceptance_phase_split_ratio = 0.60
        self.acceptance_phase_first_split_ratio = 0.25
        self.acceptance_phase_second_split_ratio = 0.60
        self.acceptance_early_temp_multiplier = 1.0
        self.acceptance_mid_temp_multiplier = 1.0
        self.acceptance_late_temp_multiplier_start = 1.0
        self.acceptance_late_temp_multiplier_end = 1.0
        self.current_progress_ratio = 0.0

        self.reward_profile = "baseline"
        self.action_guidance_progress_ratio = 0.60
        self.action_guidance_window = 400
        self.action_guidance_min_selected = 40
        self.action_guidance_accept_threshold = 0.015
        self.action_guidance_weight_multiplier = 0.2
        self.action_guidance_q_penalty = -0.03
        self.action_base_explore_weights = {
            0: 1.0,
            2: 1.0,
            3: 1.0,
            6: 1.0,
            9: 1.0,
            10: 1.0,
            11: 1.0,
        }

        if self.profile_dqn:
            self.dqn_epsilon_min = 0.08
            self.dqn_epsilon_decay = 0.9994
            self.dqn_warmup_steps = 3000
            self.dqn_target_update_every = 500

        if self.tune_profile == "ABA":
            self.dqn_epsilon_min = 0.06
            self.dqn_epsilon_decay = 0.985
            self.dqn_warmup_steps = 3000
            self.dqn_target_update_every = 400

            self.elite_action_trials[10] = 3
            self.elite_action_trials[11] = 3
            self.elite_action_trials[0] = 2
            self.elite_max_rounds = 5
            self.diversify_trigger_no_improve = 80
            self.reheat_accept_rate_threshold = 0.03
            self.reheat_no_improve_threshold = 220
            self.reheat_progress_cap_ratio = 0.60
            self.reheat_target_low_ratio = 0.66
            self.reheat_target_high_ratio = 0.72

            self.acceptance_phase_enabled = True
            self.acceptance_phase_mode = "two"
            self.acceptance_phase_split_ratio = 0.60
            self.acceptance_early_temp_multiplier = 1.25
            self.acceptance_late_temp_multiplier_start = 0.78
            self.acceptance_late_temp_multiplier_end = 0.50

        elif self.tune_profile == "ABA_S1":
            self.dqn_epsilon_min = 0.06
            self.dqn_epsilon_decay = 0.985
            self.dqn_warmup_steps = 3000
            self.dqn_target_update_every = 400

            self.elite_action_trials[10] = 4
            self.elite_action_trials[11] = 5
            self.elite_action_trials[0] = 2
            self.elite_max_rounds = 7
            self.diversify_trigger_no_improve = 140
            self.reheat_accept_rate_threshold = 0.02
            self.reheat_no_improve_threshold = 420
            self.reheat_temp_gate_ratio = 0.58
            self.reheat_progress_cap_ratio = 0.45
            self.reheat_target_low_ratio = 0.60
            self.reheat_target_high_ratio = 0.66
            self.reheat_max_per_episode = 8
            self.accept_window_reset_each_episode = False
            self.no_improve_reset_on_episode_restart = False
            self.reheat_reset_each_episode = False

            self.acceptance_phase_enabled = True
            self.acceptance_phase_mode = "three"
            self.acceptance_phase_first_split_ratio = 0.25
            self.acceptance_phase_second_split_ratio = 0.60
            self.acceptance_early_temp_multiplier = 1.15
            self.acceptance_mid_temp_multiplier = 1.00
            self.acceptance_late_temp_multiplier_start = 0.72
            self.acceptance_late_temp_multiplier_end = 0.35

        elif self.tune_profile == "ABA_S2":
            self.dqn_epsilon_min = 0.04
            self.dqn_epsilon_decay = 0.985
            self.dqn_warmup_steps = 3000
            self.dqn_target_update_every = 400
            self.dqn_step_epsilon_schedule = True
            self.dqn_epsilon_schedule_warmup_steps = 12000
            self.dqn_epsilon_schedule_decay_ratio = 0.65
            self.dqn_guidance_enabled = True

            self.elite_action_trials[10] = 4
            self.elite_action_trials[11] = 5
            self.elite_action_trials[0] = 2
            self.elite_max_rounds = 7
            self.diversify_trigger_no_improve = 140
            self.reheat_accept_rate_threshold = 0.02
            self.reheat_no_improve_threshold = 420
            self.reheat_temp_gate_ratio = 0.58
            self.reheat_progress_cap_ratio = 0.45
            self.reheat_target_low_ratio = 0.60
            self.reheat_target_high_ratio = 0.66
            self.reheat_max_per_episode = 8
            self.accept_window_reset_each_episode = False
            self.no_improve_reset_on_episode_restart = False
            self.reheat_reset_each_episode = False

            self.acceptance_phase_enabled = True
            self.acceptance_phase_mode = "three"
            self.acceptance_phase_first_split_ratio = 0.25
            self.acceptance_phase_second_split_ratio = 0.60
            self.acceptance_early_temp_multiplier = 1.15
            self.acceptance_mid_temp_multiplier = 1.00
            self.acceptance_late_temp_multiplier_start = 0.72
            self.acceptance_late_temp_multiplier_end = 0.35

            self.reward_profile = "s2"
            self.action_base_explore_weights = {
                0: 1.20,
                2: 0.65,
                3: 1.0,
                6: 0.15,
                9: 0.35,
                10: 1.80,
                11: 1.60,
            }

        self.action_recent_events = deque(maxlen=self.action_guidance_window)
        self.action_recent_selected = {action_idx: 0 for action_idx in self.valid_actions}
        self.action_recent_accepted = {action_idx: 0 for action_idx in self.valid_actions}
        self.action_recent_gbest = {action_idx: 0 for action_idx in self.valid_actions}

        logger.info(
            f"Tune profile applied | profile: {self.tune_profile} | "
            f"elp_enhanced: {self.profile_elp} | dqn_enhanced: {self.profile_dqn} | "
            f"accept_phase: {self.acceptance_phase_enabled}"
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

    def _record_action_selection(self, action_idx, previous_cost, next_cost, phase="main", global_best=False):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}selected"] += 1
        delta = 0.0
        if np.isfinite(previous_cost) and np.isfinite(next_cost):
            delta = float(previous_cost - next_cost)
            stats[f"{prefix}delta_sum"] += delta
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

    def _record_action_global_best(self, action_idx, phase="main"):
        stats = self.action_telemetry[action_idx]
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}global_best_hits"] += 1

    def _record_recent_action_outcome(self, action_idx, accepted=False, global_best=False):
        if not self.dqn_guidance_enabled:
            return
        if len(self.action_recent_events) >= self.action_guidance_window:
            old_action_idx, old_accepted, old_gbest = self.action_recent_events.popleft()
            self.action_recent_selected[old_action_idx] = max(0, self.action_recent_selected[old_action_idx] - 1)
            if old_accepted:
                self.action_recent_accepted[old_action_idx] = max(0, self.action_recent_accepted[old_action_idx] - 1)
            if old_gbest:
                self.action_recent_gbest[old_action_idx] = max(0, self.action_recent_gbest[old_action_idx] - 1)
        self.action_recent_events.append((action_idx, bool(accepted), bool(global_best)))
        self.action_recent_selected[action_idx] += 1
        if accepted:
            self.action_recent_accepted[action_idx] += 1
        if global_best:
            self.action_recent_gbest[action_idx] += 1

    def _recent_action_accept_rate(self, action_idx):
        selected = self.action_recent_selected.get(action_idx, 0)
        if selected <= 0:
            return 0.0
        return float(self.action_recent_accepted.get(action_idx, 0)) / float(selected)

    def _build_action_guidance(self):
        explore_weights = np.ones(len(self.valid_actions), dtype=np.float32)
        q_bias = np.zeros(len(self.valid_actions), dtype=np.float32)
        if not self.dqn_guidance_enabled:
            return explore_weights, q_bias
        for table_idx, action_idx in enumerate(self.valid_actions):
            explore_weights[table_idx] = float(self.action_base_explore_weights.get(action_idx, 1.0))
        if self.current_progress_ratio < self.action_guidance_progress_ratio:
            return explore_weights, q_bias
        for table_idx, action_idx in enumerate(self.valid_actions):
            if action_idx == 3:
                continue
            selected = self.action_recent_selected.get(action_idx, 0)
            gbest_hits = self.action_recent_gbest.get(action_idx, 0)
            accept_rate = self._recent_action_accept_rate(action_idx)
            if selected >= self.action_guidance_min_selected and gbest_hits == 0 and accept_rate < self.action_guidance_accept_threshold:
                explore_weights[table_idx] *= float(self.action_guidance_weight_multiplier)
                q_bias[table_idx] += float(self.action_guidance_q_penalty)
        return explore_weights, q_bias

    def get_action_telemetry(self):
        summary = {}
        for action_idx, stats in self.action_telemetry.items():
            item = dict(stats)
            for prefix in ("", "elite_"):
                selected = stats[f"{prefix}selected"]
                accepted = stats[f"{prefix}accepted"]
                improved = stats[f"{prefix}improved"]
                item[f"{prefix}accept_rate"] = 0.0 if selected == 0 else accepted / selected
                item[f"{prefix}improve_rate"] = 0.0 if accepted == 0 else improved / accepted
                item[f"{prefix}avg_delta"] = 0.0 if selected == 0 else stats[f"{prefix}delta_sum"] / selected
                item[f"{prefix}avg_accepted_delta"] = 0.0 if accepted == 0 else stats[f"{prefix}accepted_delta_sum"] / accepted
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
        for action_idx, stats in ordered:
            lines.append(
                (
                    "Action {idx} [{name}] | main sel={selected} acc={accepted} ({accept_rate:.1%}) "
                    "imp={improved} gbest={gbest} avg_acc_delta={avg_acc_delta:.2f} | "
                    "elite sel={elite_selected} acc={elite_accepted} ({elite_accept_rate:.1%}) "
                    "imp={elite_improved} gbest={elite_gbest} avg_acc_delta={elite_avg_acc_delta:.2f}"
                ).format(
                    idx=action_idx,
                    name=stats["name"],
                    selected=stats["selected"],
                    accepted=stats["accepted"],
                    accept_rate=stats["accept_rate"],
                    improved=stats["improved"],
                    gbest=stats["global_best_hits"],
                    avg_acc_delta=stats["avg_accepted_delta"],
                    elite_selected=stats["elite_selected"],
                    elite_accepted=stats["elite_accepted"],
                    elite_accept_rate=stats["elite_accept_rate"],
                    elite_improved=stats["elite_improved"],
                    elite_gbest=stats["elite_global_best_hits"],
                    elite_avg_acc_delta=stats["elite_avg_accepted_delta"],
                )
            )
        return lines

    def _evaluate_solution(self, solution):
        metrics = FBSUtil.evaluate_layout(
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
        for threshold, width in self.bin_width_schedule:
            if best_energy > threshold:
                return float(width)
        return float(self.bin_width_schedule[-1][1])

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
        fallback = max(self.T_initial * 0.05, 1.0)
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

    def _apply_recipe(self, solution, recipe):
        layout_dirty = False
        for primitive_action in recipe:
            if primitive_action == 3 and layout_dirty:
                self._evaluate_solution(solution)
                layout_dirty = False
            solution._apply_action(solution.actions[primitive_action])
            layout_dirty = True

    def generate_candidate_by_action(self, solution, action_idx):
        candidate = copy.deepcopy(solution)
        recipe = self.action_recipes[action_idx]
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

    def _generate_candidate_by_recipe(self, solution, recipe):
        candidate = copy.deepcopy(solution)
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

    def _d_inf_band(self, d_inf):
        if d_inf <= 0:
            return 0
        if d_inf == 1:
            return 1
        if d_inf == 2:
            return 2
        return 3

    def _temperature_band(self):
        ratio = self.T / max(self.T_initial, 1e-8)
        if ratio >= 0.75:
            return 0
        if ratio >= 0.50:
            return 1
        if ratio >= 0.25:
            return 2
        return 3

    def _histogram_visit_band(self, visits):
        if visits <= 0:
            return 0
        if visits == 1:
            return 1
        if visits == 2:
            return 2
        return 3

    def _histogram_band(self, energy):
        return self._histogram_visit_band(self._get_H_value(energy))

    def _histogram_context_bands(self, energy):
        idx = self._get_bin_index(energy)
        if idx < 0:
            return 0, 0, 0
        return tuple(
            self._histogram_visit_band(self.energy_histogram.get(idx + offset, 0))
            for offset in (-1, 0, 1)
        )

    def _stagnation_band(self):
        if self.no_improve_steps < 10:
            return 0
        if self.no_improve_steps < 40:
            return 1
        if self.no_improve_steps < 100:
            return 2
        return 3

    def _relative_cost_gap_band(self, solution):
        if (
            self.best_feasible_solution is None
            or not np.isfinite(self.best_feasible_cost)
            or not np.isfinite(solution.fitness)
        ):
            return 3
        gap_ratio = max(float(solution.fitness) - float(self.best_feasible_cost), 0.0) / max(
            abs(float(self.best_feasible_cost)), 1.0
        )
        if gap_ratio <= 0.01:
            return 0
        if gap_ratio <= 0.05:
            return 1
        if gap_ratio <= 0.15:
            return 2
        return 3

    def state_encoder(self, solution):
        d_band = self._d_inf_band(solution.current_d_inf)
        t_band = self._temperature_band()
        gap_band = self._relative_cost_gap_band(solution)
        s_band = self._stagnation_band()
        h_left_band, h_center_band, h_right_band = self._histogram_context_bands(solution.fitness)
        state = d_band
        for band in (t_band, gap_band, s_band, h_left_band, h_center_band, h_right_band):
            state = state * 4 + band
        return state

    def _get_allowed_action_indices(self, solution):
        allow_repair = getattr(solution, 'current_d_inf', 0) > 0
        allowed = [
            table_idx
            for table_idx, action_idx in enumerate(self.valid_actions)
            if action_idx != 3 or allow_repair
        ]
        return allowed if allowed else list(range(len(self.valid_actions)))

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

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        reward = 0.0
        scale = max(float(self.worst_feasible_cost or 0.0), 1.0)
        if self.reward_profile == "s2":
            if np.isfinite(previous_cost) and np.isfinite(next_cost):
                rel_delta = (previous_cost - next_cost) / max(abs(float(previous_cost)), 1.0)
                reward += 3.0 * math.tanh(rel_delta / 0.002)
                if accept and next_cost > previous_cost:
                    worsening_penalty = 0.35 + min((next_cost - previous_cost) / max(abs(float(previous_cost)), 1.0) / 0.002, 1.0)
                    if previous_d_inf == 0 and next_d_inf == 0:
                        worsening_penalty += 0.15
                    reward -= worsening_penalty
            reward += 0.25 * (previous_d_inf - next_d_inf)
            if previous_d_inf > 0 and next_d_inf == 0:
                reward += 0.5
            if np.isfinite(next_cost) and next_cost < previous_best_feasible:
                reward += 1.25
            if not accept:
                reward -= 0.05
            return float(np.clip(reward, -3.0, 3.0))

        if np.isfinite(previous_cost) and np.isfinite(next_cost):
            reward += (previous_cost - next_cost) / scale
            if accept and next_cost > previous_cost:
                worsening_penalty = 0.25 + min((next_cost - previous_cost) / scale, 1.0)
                if previous_d_inf == 0 and next_d_inf == 0:
                    worsening_penalty += 0.15
                reward -= worsening_penalty
        reward += 0.25 * (previous_d_inf - next_d_inf)
        if previous_d_inf > 0 and next_d_inf == 0:
            reward += 0.5
        if np.isfinite(next_cost) and next_cost < previous_best_feasible:
            reward += 1.0
        if not accept:
            reward -= 0.2
        return float(np.clip(reward, -2.5, 2.5))

    # def _accept_candidate(self, current_cost, candidate_cost):
    #     if np.isfinite(candidate_cost) and not np.isfinite(current_cost):
    #         return True, 1.0, float("inf"), float(candidate_cost)
    #     if np.isfinite(current_cost) and not np.isfinite(candidate_cost):
    #         return False, 0.0, float(current_cost), float("inf")
    #     if not np.isfinite(current_cost) and not np.isfinite(candidate_cost):
    #         return False, 0.0, float("inf"), float("inf")
    #     current_tilde = current_cost + self.k_hist * self._get_H_value(current_cost)
    #     candidate_tilde = candidate_cost + self.k_hist * self._get_H_value(candidate_cost)
    #     if candidate_tilde < current_tilde:
    #         return True, 1.0, current_tilde, candidate_tilde
    #     exponent = (current_tilde - candidate_tilde) / max(self.T, 1e-12)
    #     exponent = max(min(exponent, 700.0), -700.0)
    #     prob = math.exp(exponent)
    #     return bool(np.random.rand() < prob), prob, current_tilde, candidate_tilde
    def _accept_candidate(self, current_cost, candidate_cost):
        # Boundary handling (unchanged)
        if np.isfinite(candidate_cost) and not np.isfinite(current_cost):
            return True, 1.0, float("inf"), float(candidate_cost)
        if np.isfinite(current_cost) and not np.isfinite(candidate_cost):
            return False, 0.0, float(current_cost), float("inf")
        if not np.isfinite(current_cost) and not np.isfinite(candidate_cost):
            return False, 0.0, float("inf"), float("inf")

        # Rule 1: always accept when the true cost improves
        if candidate_cost < current_cost:
            return True, 1.0, current_cost, candidate_cost

        # Rule 2: otherwise use Metropolis acceptance on tilde energy
        current_tilde = current_cost + self.k_hist * self._get_H_value(current_cost)
        candidate_tilde = candidate_cost + self.k_hist * self._get_H_value(candidate_cost)
        exponent = (current_tilde - candidate_tilde) / self._effective_acceptance_temperature()
        exponent = max(min(exponent, 700.0), -700.0)
        prob = math.exp(exponent)
        return bool(np.random.rand() < prob), prob, current_tilde, candidate_tilde

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
        self.T_min = self._sample_temperature_floor()
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

    def _prepare_episode_start(self, episode_idx):
        self._refresh_bin_width_from_best()
        if self.reheat_reset_each_episode:
            self.reheat_episode_count = 0
        if self.accept_window_reset_each_episode:
            self.accept_window.clear()
            self.accept_rate_window = 1.0
        if episode_idx == 0:
            return
        if not self._restart_from_best_feasible():
            return
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
    def _elite_intensification(self, fast_time):
        if self.best_feasible_solution is None:
            return fast_time, False
        base = copy.deepcopy(self.best_feasible_solution)
        self._evaluate_solution(base)
        if not base.current_is_feasible or not np.isfinite(base.fitness):
            return fast_time, False

        self.elite_trigger_count += 1
        improved_any = False
        initial_cost = float(base.fitness)

        for _ in range(self.elite_max_rounds):
            best_candidate = None
            best_action_idx = None
            best_candidate_cost = float(base.fitness)

            for action_idx in self.elite_actions:
                trial_count = self.elite_action_trials.get(action_idx, 1)
                for _trial in range(trial_count):
                    candidate = self.generate_candidate_by_action(base, action_idx)
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
            self.s = copy.deepcopy(base)
            self.current_energy = self.s.fitness
            self.no_improve_steps = 0
            self._update_histogram(self.current_energy)
            self.energy_history.append(self.current_energy)
            self.modified_energy_history.append(self._tilde_energy(self.current_energy))

        if improved_any:
            final_gain = max(initial_cost - float(base.fitness), 0.0)
            self.elite_improvement_count += 1
            self.elite_total_gain += final_gain
            self.s = copy.deepcopy(base)
            self.current_energy = self.s.fitness
        return fast_time, improved_any

    #
    # def _greedy_local_search(self, solution):
    #     """
    #     Two-stage greedy local search (fast in-place version).
    #     It mutates permutation/bay in place and reverts when no improvement is found.
    #     This avoids deepcopy and keeps the local search much faster.
    #     """
    #     def _as_python_list(sequence):
    #         if isinstance(sequence, np.ndarray):
    #             return sequence.tolist()
    #         return list(sequence)
    #
    #     perm = _as_python_list(solution.fbs_model.permutation)
    #     bay = _as_python_list(solution.fbs_model.bay)
    #     current_cost = solution.fitness
    #
    #     # Stage 1: swap adjacent bays
    #     stage1_improved = True
    #     while stage1_improved:
    #         stage1_improved = False
    #         bay_structure = permutationToArray(perm, bay)
    #         n_bays = len(bay_structure)
    #
    #         for i in range(n_bays - 1):
    #             # Swap bay i and i+1 in place
    #             bay_structure[i], bay_structure[i + 1] = (
    #                 bay_structure[i + 1], bay_structure[i]
    #             )
    #             new_perm, new_bay = arrayToPermutation(bay_structure)
    #
    #             # Write back in place and evaluate
    #             new_perm_list = _as_python_list(new_perm)
    #             new_bay_list = _as_python_list(new_bay)
    #             solution.fbs_model.permutation = new_perm_list
    #             solution.fbs_model.bay = new_bay_list
    #             self._evaluate_solution(solution)
    #
    #             if (
    #                     solution.current_is_feasible
    #                     and np.isfinite(solution.fitness)
    #                     and solution.fitness < current_cost
    #             ):
    #                 # Accept and keep the new bay layout
    #                 current_cost = solution.fitness
    #                 self._observe_feasible_state(solution)
    #                 perm = new_perm_list
    #                 bay = new_bay_list
    #                 stage1_improved = True
    #                 # bay_structure already stores the swapped state
    #             else:
    #                 # Revert bay_structure; solution will be overwritten next loop
    #                 bay_structure[i], bay_structure[i + 1] = (
    #                     bay_structure[i + 1], bay_structure[i]
    #                 )
    #                 solution.fbs_model.permutation = perm
    #                 solution.fbs_model.bay = bay
    #                 solution.fitness = current_cost
    #
    #     # Stage 2: swap adjacent facilities inside each bay
    #     stage2_improved = True
    #     while stage2_improved:
    #         stage2_improved = False
    #         bay_structure = permutationToArray(perm, bay)
    #
    #         for b_idx, b in enumerate(bay_structure):
    #             b = list(b)
    #             n_fac = len(b)
    #             if n_fac < 2:
    #                 continue
    #
    #             for j in range(n_fac - 1):
    #                 # Swap adjacent facilities inside the current bay
    #                 bay_structure[b_idx][j], bay_structure[b_idx][j + 1] = (
    #                     bay_structure[b_idx][j + 1], bay_structure[b_idx][j]
    #                 )
    #                 new_perm, new_bay = arrayToPermutation(bay_structure)
    #
    #                 new_perm_list = _as_python_list(new_perm)
    #                 new_bay_list = _as_python_list(new_bay)
    #                 solution.fbs_model.permutation = new_perm_list
    #                 solution.fbs_model.bay = new_bay_list
    #                 self._evaluate_solution(solution)
    #
    #                 if (
    #                         solution.current_is_feasible
    #                         and np.isfinite(solution.fitness)
    #                         and solution.fitness < current_cost
    #                 ):
    #                     current_cost = solution.fitness
    #                     self._observe_feasible_state(solution)
    #                     perm = new_perm_list
    #                     bay = new_bay_list
    #                     stage2_improved = True
    #                 else:
    #                     # Revert bay_structure and solution
    #                     bay_structure[b_idx][j], bay_structure[b_idx][j + 1] = (
    #                         bay_structure[b_idx][j + 1], bay_structure[b_idx][j]
    #                     )
    #                     solution.fbs_model.permutation = perm
    #                     solution.fbs_model.bay = bay
    #                     solution.fitness = current_cost
    #
    #     return solution
    def _greedy_local_search(self, solution):
        """
        Two-stage greedy local search:
        Stage 1 swaps adjacent bays and accepts only improving feasible moves.
        Stage 2 swaps adjacent facilities within each bay and keeps only improvements.
        """
        improved = True
        while improved:
            improved = False

            bay_structure = permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )
            n_bays = len(bay_structure)

            for i in range(n_bays - 1):
                new_structure = bay_structure[:]
                new_structure[i], new_structure[i + 1] = (
                    new_structure[i + 1],
                    new_structure[i],
                )
                new_perm, new_bay = arrayToPermutation(new_structure)

                candidate = copy.deepcopy(solution)
                candidate.fbs_model.permutation = new_perm
                candidate.fbs_model.bay = new_bay
                self._evaluate_solution(candidate)

                if (
                    candidate.current_is_feasible
                    and np.isfinite(candidate.fitness)
                    and candidate.fitness < solution.fitness
                ):
                    solution = candidate
                    self._observe_feasible_state(solution)
                    bay_structure = permutationToArray(
                        solution.fbs_model.permutation,
                        solution.fbs_model.bay,
                    )
                    n_bays = len(bay_structure)
                    improved = True

            bay_structure = permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )

            for bay_idx, bay in enumerate(bay_structure):
                bay = list(bay)
                n_fac = len(bay)
                if n_fac < 2:
                    continue

                for j in range(n_fac - 1):
                    new_bay_structure = [list(b) for b in bay_structure]
                    new_bay_structure[bay_idx][j], new_bay_structure[bay_idx][j + 1] = (
                        new_bay_structure[bay_idx][j + 1],
                        new_bay_structure[bay_idx][j],
                    )
                    new_perm, new_bay_arr = arrayToPermutation(
                        [np.array(b) for b in new_bay_structure]
                    )

                    candidate = copy.deepcopy(solution)
                    candidate.fbs_model.permutation = new_perm
                    candidate.fbs_model.bay = new_bay_arr
                    self._evaluate_solution(candidate)

                    if (
                        candidate.current_is_feasible
                        and np.isfinite(candidate.fitness)
                        and candidate.fitness < solution.fitness
                    ):
                        solution = candidate
                        self._observe_feasible_state(solution)
                        bay_structure = permutationToArray(
                            solution.fbs_model.permutation,
                            solution.fbs_model.bay,
                        )
                        bay = list(bay_structure[bay_idx]) if bay_idx < len(bay_structure) else []
                        n_fac = len(bay)
                        improved = True

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

    def _current_best_metric_label(self):
        return 'best energy'

    def _extra_training_progress_text(self):
        return ''

    def _log_training_progress(self, progress_ratio, start_time):
        elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
        progress_percent = int(round(progress_ratio * 100))
        extra_text = self._extra_training_progress_text()
        logger.info(
            f'Training progress | {progress_percent}% | {self._current_best_metric_label()}: {self._current_best_energy_for_logging():.6f} | '
            f'temperature: {self.T:.6f} | reheats: {self.reheat_trigger_count} | accept_rate_window: {self.accept_rate_window:.3f}{extra_text} | elapsed: {elapsed_seconds:.1f}s'
        )

    def run(self):
        return self._run_impl()

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        global_step = 0
        total_steps = max(1, self.G * self.t_max)
        agent_mode = os.getenv('ELP_RL_AGENT', 'dqn').strip().lower()
        if agent_mode == 'qlearning':
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
                step_epsilon_schedule=self.dqn_step_epsilon_schedule,
                epsilon_warmup_steps=self.dqn_epsilon_schedule_warmup_steps,
                epsilon_decay_steps=int(total_steps * self.dqn_epsilon_schedule_decay_ratio),
            )
        if not self._bootstrap_until_first_feasible():
            logger.warning("Failed to seed a feasible archive before ELP search.")
        if np.isfinite(self.best_feasible_cost):
            self.best_energy = self.best_feasible_cost
            self.current_energy = self.s.fitness

        next_progress_marker_idx = 0
        for episode in range(self.G):
            if self.worst_feasible_cost is None and not self._bootstrap_until_first_feasible(max_attempts=max(200, 2 * self.t_max)):
                logger.warning(f"Episode {episode}: feasible archive still unavailable.")
                continue
            episode_best_before = self.best_feasible_cost
            self._prepare_episode_start(episode)
            for step_idx in range(self.t_max):
                self.current_progress_ratio = float(global_step) / float(max(1, total_steps))
                update_epsilon_schedule = getattr(agent, 'update_epsilon_schedule', None)
                if callable(update_epsilon_schedule):
                    update_epsilon_schedule(global_step)
                set_action_guidance = getattr(agent, 'set_action_guidance', None)
                if callable(set_action_guidance):
                    explore_weights, q_bias = self._build_action_guidance()
                    set_action_guidance(explore_weights=explore_weights, q_bias=q_bias)
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
                accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
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
                        self.s = self._greedy_local_search(self.s)
                        self.current_energy = self.s.fitness
                        # 补充：局部搜索后重新检查是否刷新全局最优
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
                    # logger.info(
                    #     f"T_min updated | reason: new_gbest | best energy: {float(self.best_feasible_cost):.6f} | T_min: {float(self.T_min):.6f} | temperature: {float(self.T):.6f}"
                    # )

                self._record_recent_action_outcome(real_action_idx, accepted=accept, global_best=gbest_recomputed)

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
                done_flag = (step_idx == self.t_max - 1)
                agent.update_Q(
                    current_state_idx,
                    action_table_idx,
                    reward,
                    next_state_idx,
                    done=done_flag,
                    allowed_next_actions=allowed_next_actions,
                )
                self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
                self.energy_history.append(self.s.fitness)
                # self.T *= self.cooling_per_step
                global_step += 1
                while (
                    next_progress_marker_idx < len(self.progress_markers)
                    and (global_step / total_steps) >= self.progress_markers[next_progress_marker_idx]
                ):
                    self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
                    next_progress_marker_idx += 1

                fast_time, reheated = self._attempt_reheating(global_step, total_steps, fast_time)
                if (not reheated) and self.no_improve_steps >= self.diversify_trigger_no_improve:
                    self._attempt_diversification(global_step)

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
        best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else copy.deepcopy(self.s)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)
        return (
            self.G * self.t_max,
            is_valid,
            best_solution,
            best_energy,
            start_time,
            end_time,
            fast_time,
        )



def _get_initial_solution_energy(env_obj):
    fitness = float(getattr(env_obj, "fitness", np.inf))
    if np.isfinite(fitness):
        return fitness

    evaluator = getattr(env_obj, "_evaluate_current_layout", None)
    if callable(evaluator):
        evaluator(snapshot_best=True)
        fitness = float(getattr(env_obj, "fitness", np.inf))
        if np.isfinite(fitness):
            return fitness

    metrics = FBSUtil.evaluate_layout(
        env_obj.fbs_model,
        env_obj.areas,
        env_obj.H,
        env_obj.F,
        env_obj.aspect_limits,
        v_worst=getattr(env_obj, "current_v_worst", None),
        k_penalty=getattr(env_obj, "k_penalty", 1),
        distance_metric=getattr(env_obj, "distance_metric", "manhattan"),
    )
    fitness = float(metrics["cost"])
    if np.isfinite(fitness):
        return fitness
    return float(metrics["mhc"])

if __name__ == "__main__":
    profile_tag = os.getenv("ELP_TUNE_PROFILE", "ED1").strip().upper()
    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", f"ELP_RL_Standard_{profile_tag}")
    exp_remark_default = "WarmStart(GA)+ELP+full-Q-learning,移除了 facility_insert (5) 和 facility_shuffle (7)"
    exp_remark = os.getenv("ELP_EXP_REMARK", exp_remark_default)
    exp_number = _env_int("ELP_EXP_NUMBER", 50)
    is_exp = _env_flag("ELP_IS_EXP", True)

    G = _env_int("ELP_G", 500)
    t_max = _env_int("ELP_T_MAX", 200)
    T_initial = _env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _env_float("ELP_K_HIST", 10.0)

    fixed_seeds = _env_int_list("ELP_FIXED_SEEDS")
    base_seed = _env_int("ELP_BASE_SEED", 20260328)
    if fixed_seeds:
        exp_number = len(fixed_seeds)

    def _seed_for_run(run_index):
        if fixed_seeds:
            return int(fixed_seeds[run_index])
        return int(base_seed + run_index)

    if is_exp:
        for i in range(exp_number):
            run_seed = _seed_for_run(i)
            strict_determinism = _set_global_seed(run_seed)
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            logger.info(f"Experiment seed: {run_seed} | strict_determinism: {strict_determinism}")
            try:
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
                elp_solver = ELP(
                    env=base_env,
                    gbest=initial_gbest,
                    T=T_initial,
                    G=G,
                    t_max=t_max,
                    k=k_hist,
                )
                total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
                logger.info(f"Experiment {i + 1} complete | best energy: {best_energy}")
                for telemetry_line in elp_solver.format_action_telemetry():
                    logger.info(f"Telemetry | {telemetry_line}")
                ExperimentsUtil.save_experiment_result(
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
                    exp_gbest_updates=elp_solver.gbest_update_count,
                )
            except Exception as exc:
                logger.exception(f"Experiment {i + 1} failed: {exc}")
    else:
        run_seed = _seed_for_run(0)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Single-run seed: {run_seed} | strict_determinism: {strict_determinism}")
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
        elp_solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
        print(f"Single run complete | best energy: {best_energy}")
        for telemetry_line in elp_solver.format_action_telemetry():
            print(f"Telemetry | {telemetry_line}")




