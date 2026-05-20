import copy
import datetime
import math
import os
import random
from collections import deque, defaultdict
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
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


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
        self.s_dim = int(s_dim)
        self.a_dim = int(a_dim)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.Q = defaultdict(lambda: np.zeros(self.a_dim, dtype=float))

    @staticmethod
    def _state_key(state):
        array_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if array_state.size == 0:
            return b""
        return array_state.tobytes()

    def select_action(self, s, deterministic=False, allowed_actions=None):
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.a_dim))
        allowed_actions = np.asarray(allowed_actions, dtype=int).reshape(-1)
        if (not deterministic) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(allowed_actions))
        state_key = self._state_key(s)
        q_row = np.take(self.Q[state_key], allowed_actions)
        max_q = np.max(q_row)
        best_mask = np.isclose(q_row, max_q).reshape(-1)
        best_actions = allowed_actions[best_mask]
        return int(np.random.choice(best_actions))

    def update_Q(self, s, a, reward, s_next, done=False, allowed_next_actions=None):
        state_key = self._state_key(s)
        next_state_key = self._state_key(s_next)
        td_target = float(reward)
        if not done:
            if allowed_next_actions is None or len(allowed_next_actions) == 0:
                td_target += self.gamma * np.max(self.Q[next_state_key])
            else:
                next_actions = np.asarray(allowed_next_actions, dtype=int).reshape(-1)
                td_target += self.gamma * np.max(np.take(self.Q[next_state_key], next_actions))
        self.Q[state_key][int(a)] += self.alpha * (td_target - self.Q[state_key][int(a)])

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
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
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
        width = int(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, int(action_dim)),
        )

    def forward(self, state_vec):
        x = state_vec.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.layers(x)


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

    def _format_state(self, state):
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size == self.s_dim:
            return arr
        if arr.size <= 0:
            return np.zeros(self.s_dim, dtype=np.float32)
        if arr.size > self.s_dim:
            return arr[: self.s_dim]
        return np.pad(arr, (0, self.s_dim - arr.size), mode='constant', constant_values=0.0).astype(np.float32)

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

        state_vec = self._format_state(s).reshape(1, -1)
        with torch.no_grad():
            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
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
        next_state = np.asarray(self.n_step_buffer[0][3], dtype=np.float32).copy()
        done_flag = False
        next_mask = self.n_step_buffer[0][5]

        horizon = min(self.idqn_k, len(self.n_step_buffer))
        for idx in range(horizon):
            _, _, reward_i, next_state_i, done_i, next_mask_i = self.n_step_buffer[idx]
            total_reward += discount * float(reward_i)
            discount *= self.gamma
            next_state = np.asarray(next_state_i, dtype=np.float32).copy()
            done_flag = bool(done_i)
            next_mask = next_mask_i
            if done_flag:
                break

        gamma_n = 0.0 if done_flag else float(discount)
        return (
            np.asarray(state0, dtype=np.float32).copy(),
            int(action0),
            float(total_reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done_flag),
            np.asarray(next_mask, dtype=np.bool_),
            float(gamma_n),
        )

    def _append_transition(self, s, a, reward, s_next, done, allowed_next_actions):
        transition = (
            self._format_state(s).copy(),
            int(a),
            float(reward),
            self._format_state(s_next).copy(),
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

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
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
            next_state = np.asarray(self.n_step_buffer[0][3], dtype=np.float32).copy()
            next_mask = self.n_step_buffer[0][5]
            for idx in range(len(self.n_step_buffer)):
                _, _, reward_i, next_state_i, _done_i, next_mask_i = self.n_step_buffer[idx]
                total_reward += discount * float(reward_i)
                discount *= self.gamma
                next_state = np.asarray(next_state_i, dtype=np.float32).copy()
                next_mask = next_mask_i
            self.replay.add((
                np.asarray(state0, dtype=np.float32).copy(),
                int(action0),
                float(total_reward),
                np.asarray(next_state, dtype=np.float32).copy(),
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
        self.temperature_floor_action_ids = [2, 0, 1]
        self.valid_actions = [0, 1, 2, 3, 4]
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
        self.elite_actions = [10, 11, 0, 2]
        self.elite_action_trials = {10: 4, 11: 4, 0: 1, 2: 1}
        self.elite_max_rounds = 4
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
            for action_idx in sorted(set(self.valid_actions) | set(self.elite_actions))
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
            self.elite_action_trials[10] = 4
            self.elite_action_trials[11] = 4
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
        # s2???????????????EMA???????????????
        self.reward_rel_delta_window = max(50, _env_int("ELP_REWARD_REL_DELTA_WINDOW", 400))
        self.reward_rel_delta_scale_min = max(
            1e-6,
            _env_float("ELP_REWARD_REL_DELTA_SCALE_MIN", 5e-4),
        )
        self.reward_rel_delta_scale_max = max(
            self.reward_rel_delta_scale_min * 2.0,
            _env_float("ELP_REWARD_REL_DELTA_SCALE_MAX", 1e-2),
        )
        self.reward_rel_delta_scale_default = min(
            max(
                _env_float("ELP_REWARD_REL_DELTA_SCALE_DEFAULT", 2e-3),
                self.reward_rel_delta_scale_min,
            ),
            self.reward_rel_delta_scale_max,
        )
        self.reward_rel_delta_scale_ema_beta = min(
            max(_env_float("ELP_REWARD_REL_DELTA_SCALE_EMA_BETA", 0.92), 0.0),
            0.9999,
        )
        self.reward_rel_delta_history = deque(maxlen=self.reward_rel_delta_window)
        self.reward_rel_delta_scale_ema = self.reward_rel_delta_scale_default
        self.action_guidance_progress_ratio = 0.60
        self.action_guidance_window = 400
        self.action_guidance_min_selected = 40
        self.action_guidance_accept_threshold = 0.015
        self.action_guidance_weight_multiplier = 0.2
        self.action_guidance_q_penalty = -0.03
        self.action_base_explore_weights = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 0.25,
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

            self.elite_action_trials[10] = 4
            self.elite_action_trials[11] = 4
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
                1: 0.85,
                2: 0.75,
                3: 1.0,
                4: 0.35,
            }

        # Top-K????????????1?
        self.topk_guided_enabled = _env_flag("ELP_TOPK_GUIDED_ENABLED", True)
        self.topk_guided_topk = max(5, _env_int("ELP_TOPK_GUIDED_TOPK", 30))
        self.topk_guided_max_iters = max(1, _env_int("ELP_TOPK_GUIDED_MAX_ITERS", 6))

        # ???????? + ???????2?
        self.high_flow_warmstart_enabled = _env_flag("ELP_HIGH_FLOW_WARMSTART_ENABLED", True)
        self.high_flow_warmstart_restarts = max(1, _env_int("ELP_HIGH_FLOW_WARMSTART_RESTARTS", 6))
        self.high_flow_warmstart_stagnation_episodes = max(
            2,
            _env_int("ELP_HIGH_FLOW_WARMSTART_STAGNATION_EPISODES", 4),
        )
        self.high_flow_warmstart_use_topk = _env_flag("ELP_HIGH_FLOW_WARMSTART_USE_TOPK", True)
        self.high_flow_warmstart_topk = max(
            5,
            _env_int("ELP_HIGH_FLOW_WARMSTART_TOPK", self.topk_guided_topk),
        )
        self.high_flow_warmstart_max_iters = max(
            1,
            _env_int("ELP_HIGH_FLOW_WARMSTART_MAX_ITERS", self.topk_guided_max_iters),
        )

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

    def generate_candidate_by_action(self, solution, action_idx, phase="main"):
        action_idx = int(action_idx)
        if phase == "main" and action_idx == 10 and self.segment_insert_light_enabled:
            return self._generate_segment_insert_light_candidate(solution)
        return self._generate_candidate_by_recipe(solution, self.action_recipes[action_idx])

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

    def state_encoder(self, solution):
        """
        ?? gym-flp-fbs ?1D?????????8??????? n*8?
        ?????
        [??ID???, x/W, y/H, b/W, h/H, ??????, ??????, ??????]
        """
        permutation = np.asarray(solution.fbs_model.permutation, dtype=np.float32).reshape(-1)
        n = int(permutation.size)
        if n <= 0:
            return np.zeros(8, dtype=np.float32)

        tm = np.asarray(solution.TM, dtype=np.float32)
        if tm.ndim == 2 and tm.shape[0] == tm.shape[1]:
            sources = np.sum(tm, axis=1)
            sinks = np.sum(tm, axis=0)
        else:
            sources = np.zeros(n, dtype=np.float32)
            sinks = np.zeros(n, dtype=np.float32)

        norm_permutation = self._normalize_state_feature(permutation)
        norm_sources = self._normalize_state_feature(sources)
        norm_sinks = self._normalize_state_feature(sinks)

        width = max(float(getattr(solution, "W", 1.0)), 1e-8)
        height = max(float(getattr(solution, "H", 1.0)), 1e-8)

        fac_x = self._fit_state_feature_length(getattr(solution, "fac_x", []), n, fill_value=0.0) / width
        fac_y = self._fit_state_feature_length(getattr(solution, "fac_y", []), n, fill_value=0.0) / height
        fac_b = self._fit_state_feature_length(getattr(solution, "fac_b", []), n, fill_value=0.0) / width
        fac_h = self._fit_state_feature_length(getattr(solution, "fac_h", []), n, fill_value=0.0) / height

        aspect_ratio = self._fit_state_feature_length(getattr(solution, "fac_aspect_ratio", []), n, fill_value=1.0)
        aspect_limits = np.asarray(getattr(solution, "aspect_limits", []), dtype=np.float32).reshape(-1)
        finite_limits = aspect_limits[np.isfinite(aspect_limits)]
        if finite_limits.size > 0:
            aspect_limit_value = float(np.max(finite_limits))
        else:
            finite_ratio = aspect_ratio[np.isfinite(aspect_ratio)]
            aspect_limit_value = float(np.max(finite_ratio)) if finite_ratio.size > 0 else 1.0
        aspect_limit_value = max(aspect_limit_value, 1e-8)
        aspect_norm = aspect_ratio / aspect_limit_value

        state_components = [
            self._fit_state_feature_length(norm_permutation, n, fill_value=0.5),
            fac_x,
            fac_y,
            fac_b,
            fac_h,
            self._fit_state_feature_length(norm_sources, n, fill_value=0.5),
            self._fit_state_feature_length(norm_sinks, n, fill_value=0.5),
            aspect_norm.astype(np.float32),
        ]
        state_vector = np.concatenate(state_components)
        return state_vector.astype(np.float32)


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
        reward = 0.0
        scale = max(float(self.worst_feasible_cost or 0.0), 1.0)
        if self.reward_profile == "s2":
            if np.isfinite(previous_cost) and np.isfinite(next_cost):
                base = max(abs(float(previous_cost)), 1.0)
                adaptive_scale = max(
                    self._get_reward_rel_delta_scale(previous_cost, next_cost),
                    1e-12,
                )
                rel_delta = (previous_cost - next_cost) / base
                reward += 3.0 * math.tanh(rel_delta / adaptive_scale)
                if accept and next_cost > previous_cost:
                    worsening_penalty = 0.35 + min(
                        (next_cost - previous_cost) / base / adaptive_scale,
                        1.0,
                    )
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
            best_candidate = None
            best_candidate_cost = float(current.fitness)

            # 1) ????????????
            for p1 in endpoint_positions:
                for p2 in range(n):
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
                for dst in range(n):
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

    def _build_high_flow_grouped_perm(self, flow, bay_sizes):
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

        groups = [[] for _ in range(group_count)]
        capacities = bay_sizes.copy()
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

        remaining_nodes = [node for node in range(n) if node not in used_nodes]
        for node in remaining_nodes:
            best_group = None
            best_score = -float("inf")
            for group_idx in range(group_count):
                if capacities[group_idx] <= 0:
                    continue
                if len(groups[group_idx]) == 0:
                    score = 0.0
                else:
                    score = float(np.mean([sym_flow[node, member] for member in groups[group_idx]]))
                if score > best_score:
                    best_score = score
                    best_group = group_idx
            if best_group is None:
                continue
            groups[best_group].append(int(node))
            capacities[best_group] -= 1
            used_nodes.add(int(node))

        leftover_nodes = [node for node in range(n) if node not in used_nodes]
        for group_idx in range(group_count):
            while capacities[group_idx] > 0 and leftover_nodes:
                node = int(leftover_nodes.pop(0))
                groups[group_idx].append(node)
                capacities[group_idx] -= 1

        group_scores = []
        for group_idx, members in enumerate(groups):
            if len(members) <= 1:
                score = 0.0
            else:
                sub_matrix = sym_flow[np.ix_(members, members)]
                score = float(np.sum(sub_matrix))
            group_scores.append((group_idx, score))
        group_scores.sort(key=lambda item: item[1], reverse=True)

        permutation = []
        for group_idx, _score in group_scores:
            members = list(groups[group_idx])
            members.sort(key=lambda node: float(np.sum(sym_flow[node, members])) if members else 0.0, reverse=True)
            permutation.extend([int(node + 1) for node in members])

        return permutation

    def _build_high_flow_grouped_perm_random(self, flow, bay_sizes):
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

        groups = [[] for _ in range(group_count)]
        capacities = bay_sizes.copy()
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

        remaining_nodes = [node for node in range(n) if node not in used_nodes]
        rng.shuffle(remaining_nodes)
        for node in remaining_nodes:
            scores = []
            valid_groups = []
            for group_idx in range(group_count):
                if capacities[group_idx] <= 0:
                    continue
                members = groups[group_idx]
                if len(members) == 0:
                    score = 0.0
                else:
                    score = float(np.mean([sym_flow[node, member] for member in members]))
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
            used_nodes.add(int(node))

        leftover_nodes = [node for node in range(n) if node not in used_nodes]
        for group_idx in range(group_count):
            while capacities[group_idx] > 0 and leftover_nodes:
                node = int(leftover_nodes.pop(0))
                groups[group_idx].append(node)
                capacities[group_idx] -= 1

        group_scores = []
        for group_idx, members in enumerate(groups):
            if len(members) <= 1:
                score = 0.0
            else:
                sub_matrix = sym_flow[np.ix_(members, members)]
                score = float(np.sum(sub_matrix))
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
        restarts = max(1, int(self.high_flow_warmstart_restarts))
        for restart_idx in range(restarts):
            if restart_idx == 0:
                grouped_perm = self._build_high_flow_grouped_perm(flow_matrix, bay_sizes)
            else:
                grouped_perm = self._build_high_flow_grouped_perm_random(flow_matrix, bay_sizes)
            if not grouped_perm:
                continue
            candidate = self._build_candidate_from_layout(base, grouped_perm, bay_template)
            if candidate is None:
                continue
            if self.high_flow_warmstart_use_topk and self.topk_guided_enabled:
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
        if self.topk_guided_enabled:
            guided_solution = self._topk_guided_enhanced_local_search(
                solution,
                topk=self.topk_guided_topk,
                max_iters=self.topk_guided_max_iters,
            )
            if (
                guided_solution.current_is_feasible
                and np.isfinite(guided_solution.fitness)
                and guided_solution.fitness <= solution.fitness
            ):
                solution = guided_solution

        # ?????????????????????deepcopy???
        candidate = copy.deepcopy(solution)

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

                candidate.fbs_model.permutation = new_perm
                candidate.fbs_model.bay = new_bay
                self._evaluate_solution(candidate)

                if (
                    candidate.current_is_feasible
                    and np.isfinite(candidate.fitness)
                    and candidate.fitness < solution.fitness
                ):
                    solution = copy.deepcopy(candidate)
                    self._observe_feasible_state(solution)
                    candidate = copy.deepcopy(solution)
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

                    candidate.fbs_model.permutation = new_perm
                    candidate.fbs_model.bay = new_bay_arr
                    self._evaluate_solution(candidate)

                    if (
                        candidate.current_is_feasible
                        and np.isfinite(candidate.fitness)
                        and candidate.fitness < solution.fitness
                    ):
                        solution = copy.deepcopy(candidate)
                        self._observe_feasible_state(solution)
                        candidate = copy.deepcopy(solution)
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

    def _log_training_progress(self, progress_ratio, start_time):
        elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
        progress_percent = int(round(progress_ratio * 100))
        logger.info(
            f'Training progress | {progress_percent}% | best energy: {self._current_best_energy_for_logging():.6f} | '
            f'temperature: {self.T:.6f} | reheats: {self.reheat_trigger_count} | accept_rate_window: {self.accept_rate_window:.3f} | elapsed: {elapsed_seconds:.1f}s'
        )

    def run(self):
        return self._run_impl()

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        global_step = 0
        total_steps = max(1, self.G * self.t_max)
        agent_mode = os.getenv('ELP_RL_AGENT', 'dqn').strip().lower()
        initial_state_vector = np.asarray(self.state_encoder(self.s), dtype=np.float32).reshape(-1)
        state_dim = int(initial_state_vector.size) if initial_state_vector.size > 0 else 8
        if agent_mode == 'qlearning':
            agent = StandardQLearningAgent(s_dim=state_dim, a_dim=len(self.valid_actions))
        else:
            agent = StandardDQNAgent(
                s_dim=state_dim,
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
                trigger_local_search = False

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
                    trigger_local_search = improved
                    self._update_histogram(self.s.fitness)
                    if improved:
                        self.no_improve_steps = 0
                        fast_time = datetime.datetime.now()
                    else:
                        self.no_improve_steps += 1

                else:
                    self._update_histogram(self.s.fitness)
                    self.no_improve_steps += 1

                # ??????????????TD??????????????
                rl_next_state_idx = self.state_encoder(self.s)
                rl_next_d_inf = self.s.current_d_inf
                rl_next_cost = self.s.fitness
                rl_allowed_next_actions = self._get_allowed_action_indices(self.s)
                reward = self._compute_transition_reward(
                    previous_cost,
                    rl_next_cost,
                    previous_d_inf,
                    rl_next_d_inf,
                    previous_best_feasible,
                    accept,
                )
                done_flag = (step_idx == self.t_max - 1)
                agent.update_Q(
                    current_state_idx,
                    action_table_idx,
                    reward,
                    rl_next_state_idx,
                    done=done_flag,
                    allowed_next_actions=rl_allowed_next_actions,
                )

                if trigger_local_search:
                    self.s = self._greedy_local_search(self.s)
                    self.current_energy = self.s.fitness
                    # ????????????????????
                    if self.s.current_is_feasible and self.s.fitness < previous_best_feasible:
                        improved = True

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
    exp_remark_default = "ELP+DQN(main:0/1/2/3/4-idle; elite:10/11-highfreq)"
    exp_remark = os.getenv("ELP_EXP_REMARK", exp_remark_default)
    exp_number = _env_int("ELP_EXP_NUMBER", 50)
    is_exp = _env_flag("ELP_IS_EXP", True)
    print_telemetry = _env_flag("ELP_PRINT_TELEMETRY", False)
    save_experiment_result = _env_flag("ELP_SAVE_EXPERIMENT_RESULT", True)

    G = _env_int("ELP_G", 500)
    t_max = _env_int("ELP_T_MAX", 200)
    T_initial = _env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _env_float("ELP_K_HIST", 20.0)

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
            _set_global_seed(run_seed)
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            logger.info(f"Experiment seed: {run_seed}")
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
                if print_telemetry:
                    for telemetry_line in elp_solver.format_action_telemetry():
                        logger.info(f"Telemetry | {telemetry_line}")
                if save_experiment_result:
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
        _set_global_seed(run_seed)
        logger.info(f"Single-run seed: {run_seed}")
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
        if print_telemetry:
            for telemetry_line in elp_solver.format_action_telemetry():
                print(f"Telemetry | {telemetry_line}")




