import hashlib
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GraphConv, global_mean_pool
except Exception:
    Batch = None
    Data = None
    GraphConv = None
    global_mean_pool = None


class StandardQLearningAgent:
    """标准 Q-learning 智能体。"""

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
        self.last_action_info = {
            "selection_mode": "unknown",
            "epsilon": float(self.epsilon),
            "allowed_count": 0,
        }

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
            action_idx = int(np.random.choice(allowed_actions))
            self.last_action_info = {
                "selection_mode": "explore",
                "epsilon": float(self.epsilon),
                "allowed_count": int(len(allowed_actions)),
            }
            return action_idx
        state_key = self._state_key(s)
        q_row = np.take(self.Q[state_key], allowed_actions)
        max_q = np.max(q_row)
        best_mask = np.isclose(q_row, max_q).reshape(-1)
        best_actions = allowed_actions[best_mask]
        action_idx = int(np.random.choice(best_actions))
        self.last_action_info = {
            "selection_mode": "exploit",
            "epsilon": float(self.epsilon),
            "allowed_count": int(len(allowed_actions)),
        }
        return action_idx

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

    def get_last_action_info(self):
        return dict(self.last_action_info)


class _ReplayBuffer:
    """简单经验回放缓存。"""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = deque(maxlen=self.capacity)

    def add(self, transition):
        self.storage.append(transition)

    def __len__(self):
        return len(self.storage)

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


_QNetworkBase = nn.Module if nn is not None else object


class _QNetwork(_QNetworkBase):
    """当前主链路使用的 MLP Q 网络。"""

    def __init__(self, state_dim, action_dim, embedding_dim=32, hidden_dim=64):
        if nn is None:
            raise ImportError("PyTorch is not available. Set ELP_RL_AGENT=qlearning or install torch.")
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
            raise ImportError("PyTorch is required for StandardDQNAgent.")

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

        self.device = torch.device("cpu")
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
        self.last_action_info = {
            "selection_mode": "unknown",
            "epsilon": float(self.epsilon),
            "allowed_count": 0,
        }

    def _format_state(self, state):
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size == self.s_dim:
            return arr
        if arr.size <= 0:
            return np.zeros(self.s_dim, dtype=np.float32)
        if arr.size > self.s_dim:
            return arr[: self.s_dim]
        return np.pad(arr, (0, self.s_dim - arr.size), mode="constant", constant_values=0.0).astype(np.float32)

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

        is_explore = bool((not deterministic) and (np.random.rand() < self.epsilon))
        if is_explore:
            weights = np.take(self.action_explore_weights, allowed_actions)
            weights = np.maximum(weights, 0.0)
            total_weight = float(np.sum(weights))
            if total_weight <= 0.0:
                action_idx = int(np.random.choice(allowed_actions))
            else:
                probs = weights / total_weight
                action_idx = int(np.random.choice(allowed_actions, p=probs))
            self.last_action_info = {
                "selection_mode": "explore",
                "epsilon": float(self.epsilon),
                "allowed_count": int(len(allowed_actions)),
            }
            return action_idx

        state_vec = self._format_state(s).reshape(1, -1)
        with torch.no_grad():
            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
            q_values = self.online_net(state_tensor).squeeze(0).detach().cpu().numpy()

        q_values = q_values + self.action_q_bias
        candidate_q = np.take(q_values, allowed_actions)
        best_q = np.max(candidate_q)
        best_mask = np.isclose(candidate_q, best_q).reshape(-1)
        best_actions = allowed_actions[best_mask]
        action_idx = int(np.random.choice(best_actions))
        self.last_action_info = {
            "selection_mode": "exploit",
            "epsilon": float(self.epsilon),
            "allowed_count": int(len(allowed_actions)),
        }
        return action_idx

    def set_action_guidance(self, explore_weights=None, q_bias=None):
        if explore_weights is not None:
            weights = np.asarray(explore_weights, dtype=np.float32).reshape(-1)
            if weights.size == self.a_dim:
                self.action_explore_weights = np.maximum(weights, 0.0)
        if q_bias is not None:
            bias = np.asarray(q_bias, dtype=np.float32).reshape(-1)
            if bias.size == self.a_dim:
                self.action_q_bias = bias

    def get_last_action_info(self):
        return dict(self.last_action_info)

    def update_epsilon_schedule(self, global_step):
        if not self.step_epsilon_schedule:
            return self.epsilon
        global_step = int(max(0, global_step))
        if global_step <= self.epsilon_warmup_steps:
            self.epsilon = self.epsilon_start
            return self.epsilon
        if self.epsilon_decay_steps <= self.epsilon_warmup_steps:
            self.epsilon = self.epsilon_min
            return self.epsilon
        ratio = (global_step - self.epsilon_warmup_steps) / float(
            max(1, self.epsilon_decay_steps - self.epsilon_warmup_steps)
        )
        ratio = min(max(ratio, 0.0), 1.0)
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - ratio * (self.epsilon_start - self.epsilon_min),
        )
        return self.epsilon

    def _append_transition(self, s, a, reward, s_next, done, allowed_next_actions):
        state = self._format_state(s)
        next_state = self._format_state(s_next)
        next_mask = self._build_action_mask(allowed_next_actions)
        transition = (
            state,
            int(a),
            float(reward),
            next_state,
            float(done),
            next_mask,
            float(self.gamma),
        )
        self.replay.add(transition)

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
        self.n_step_buffer.clear()


class DQNStateEncoder:
    """DQN 主链路状态编码器。"""

    def build_facility_structural_state(self, runtime, solution, n):
        permutation = runtime._to_int_list(getattr(solution.fbs_model, "permutation", []))
        bay = runtime._to_int_list(getattr(solution.fbs_model, "bay", []))
        position_norm = np.full(n, 0.5, dtype=np.float32)
        bay_index_norm = np.zeros(n, dtype=np.float32)
        in_bay_rank_norm = np.zeros(n, dtype=np.float32)
        bay_size_norm = np.zeros(n, dtype=np.float32)

        def resolve_facility_index(facility_id, fallback_index):
            try:
                facility_int = int(facility_id)
            except Exception:
                facility_int = int(fallback_index) + 1
            if 1 <= facility_int <= n:
                return facility_int - 1
            if 0 <= facility_int < n:
                return facility_int
            if 0 <= int(fallback_index) < n:
                return int(fallback_index)
            return None

        denom_position = float(max(n - 1, 1))
        for position, facility_id in enumerate(permutation[:n]):
            facility_index = resolve_facility_index(facility_id, position)
            if facility_index is not None:
                position_norm[facility_index] = float(position) / denom_position

        bay_ranges = runtime._bay_position_ranges(bay)
        bay_count = max(1, len(bay_ranges))
        bay_index_denom = float(max(bay_count - 1, 1))
        for bay_idx, (start, end) in enumerate(bay_ranges):
            bay_size = max(1, int(end) - int(start) + 1)
            rank_denom = float(max(bay_size - 1, 1))
            bay_size_ratio = float(bay_size) / float(max(n, 1))
            for local_rank, position in enumerate(range(int(start), int(end) + 1)):
                if position >= len(permutation):
                    break
                facility_index = resolve_facility_index(permutation[position], position)
                if facility_index is None:
                    continue
                bay_index_norm[facility_index] = 0.0 if bay_count <= 1 else float(bay_idx) / bay_index_denom
                in_bay_rank_norm[facility_index] = 0.0 if bay_size <= 1 else float(local_rank) / rank_denom
                bay_size_norm[facility_index] = bay_size_ratio

        return (
            position_norm,
            bay_index_norm,
            in_bay_rank_norm,
            bay_size_norm,
            permutation,
            bay_ranges,
        )

    def build_global_state_context(self, runtime, solution, tm, permutation, bay_ranges, aspect_violation, infeasible_mask):
        n = max(1, len(permutation))
        progress_ratio = min(max(float(getattr(runtime, "current_progress_ratio", 0.0)), 0.0), 1.0)
        temp_ratio = float(getattr(runtime, "T", 0.0)) / max(float(getattr(runtime, "T_initial", 1.0)), 1e-8)
        temp_ratio = min(max(temp_ratio, 0.0), 2.0) / 2.0
        feasible_flag = 1.0 if getattr(solution, "current_is_feasible", False) else 0.0
        current_d_inf = max(0.0, float(getattr(solution, "current_d_inf", 0)))
        d_inf_ratio = min(current_d_inf / float(n), 1.0)

        no_improve_steps = max(0.0, float(getattr(runtime, "no_improve_steps", 0)))
        no_improve_ratio = 0.0
        no_improve_scale = max(10.0, float(getattr(runtime, "t_max", 1)))
        if no_improve_scale > 1.0:
            no_improve_ratio = min(
                math.log1p(no_improve_steps) / math.log1p(no_improve_scale),
                1.0,
            )

        current_cost = float(getattr(solution, "fitness", np.inf))
        best_feasible_cost = float(getattr(runtime, "best_feasible_cost", np.inf))
        if np.isfinite(current_cost) and np.isfinite(best_feasible_cost):
            gap_ratio = max(current_cost - best_feasible_cost, 0.0) / max(abs(best_feasible_cost), 1.0)
            gap_ratio = min(gap_ratio, 1.0)
        else:
            gap_ratio = 1.0

        accept_rate_window = min(max(float(getattr(runtime, "accept_rate_window", 0.0)), 0.0), 1.0)
        violation_ratio = float(
            np.mean(np.maximum(aspect_violation, infeasible_mask.astype(np.float32)))
        ) if aspect_violation.size > 0 else 0.0

        bay_sizes = np.asarray(
            [int(end) - int(start) + 1 for start, end in bay_ranges],
            dtype=np.float32,
        )
        if bay_sizes.size > 0:
            bay_count_ratio = float(bay_sizes.size) / float(max(n, 1))
            mean_bay_size_ratio = float(np.mean(bay_sizes)) / float(max(n, 1))
            max_bay_size_ratio = float(np.max(bay_sizes)) / float(max(n, 1))
            bay_size_std_ratio = float(np.std(bay_sizes)) / float(max(n, 1))
        else:
            bay_count_ratio = 0.0
            mean_bay_size_ratio = 0.0
            max_bay_size_ratio = 0.0
            bay_size_std_ratio = 0.0

        cross_bay_flow_ratio = 0.0
        adjacent_bay_flow_ratio = 0.0
        if (
            tm.ndim == 2
            and tm.shape[0] == tm.shape[1]
            and int(tm.shape[0]) >= len(permutation)
            and len(permutation) > 1
            and bay_ranges
        ):
            perm_zero_based = []
            for position, facility_id in enumerate(permutation):
                try:
                    facility_int = int(facility_id)
                except Exception:
                    facility_int = int(position) + 1
                if 1 <= facility_int <= int(tm.shape[0]):
                    perm_zero_based.append(facility_int - 1)
                else:
                    perm_zero_based = []
                    break
            if len(perm_zero_based) == len(permutation):
                tm_perm = np.asarray(tm[np.ix_(perm_zero_based, perm_zero_based)], dtype=np.float32)
                sym_flow = tm_perm + tm_perm.T
                upper_mask = np.triu(np.ones(sym_flow.shape, dtype=bool), 1)
                total_pair_flow = float(np.sum(sym_flow[upper_mask]))
                if total_pair_flow > 0.0:
                    bay_ids_by_position = np.zeros(len(permutation), dtype=np.int32)
                    for bay_idx, (start, end) in enumerate(bay_ranges):
                        bay_ids_by_position[int(start): int(end) + 1] = int(bay_idx)
                    cross_mask = bay_ids_by_position[:, None] != bay_ids_by_position[None, :]
                    adjacent_mask = np.abs(
                        bay_ids_by_position[:, None] - bay_ids_by_position[None, :]
                    ) == 1
                    cross_bay_flow_ratio = float(np.sum(sym_flow[np.logical_and(upper_mask, cross_mask)])) / total_pair_flow
                    adjacent_bay_flow_ratio = float(
                        np.sum(sym_flow[np.logical_and(upper_mask, adjacent_mask)])
                    ) / total_pair_flow

        action_space_size_getter = getattr(runtime, "_current_action_space_size", None)
        if callable(action_space_size_getter):
            action_space_size = max(1, int(action_space_size_getter(solution)))
        else:
            action_space_size = max(1, len(getattr(runtime, "valid_actions", [])))
        allowed_action_ratio = float(len(runtime._get_allowed_action_indices(solution))) / float(
            action_space_size
        )
        phase_name = runtime._current_search_phase(solution)
        phase_one_hot = np.asarray(
            [
                1.0 if phase_name == "early" else 0.0,
                1.0 if phase_name == "mid" else 0.0,
                1.0 if phase_name == "late" else 0.0,
                1.0 if phase_name == "infeasible" else 0.0,
            ],
            dtype=np.float32,
        )

        context = np.asarray(
            [
                progress_ratio,
                temp_ratio,
                accept_rate_window,
                feasible_flag,
                d_inf_ratio,
                no_improve_ratio,
                gap_ratio,
                violation_ratio,
                bay_count_ratio,
                mean_bay_size_ratio,
                max_bay_size_ratio,
                bay_size_std_ratio,
                cross_bay_flow_ratio,
                adjacent_bay_flow_ratio,
                allowed_action_ratio,
            ],
            dtype=np.float32,
        )
        extra_state_context = np.asarray([], dtype=np.float32)
        extra_state_builder = getattr(runtime, "_build_extra_state_context", None)
        if callable(extra_state_builder):
            extra_state_context = np.asarray(
                extra_state_builder(
                    solution,
                    tm,
                    permutation,
                    bay_ranges,
                    aspect_violation,
                    infeasible_mask,
                ),
                dtype=np.float32,
            ).reshape(-1)
        return np.concatenate([context, phase_one_hot, extra_state_context]).astype(np.float32)

    def encode(self, runtime, solution):
        """
        面向 DQN 的状态编码：
        1. 设施级特征统一使用“设施视角”对齐，避免 permutation 与几何字段错位。
        2. 显式加入 bay 结构、可行性与搜索上下文，减少隐藏状态对动作收益的干扰。
        """
        permutation = np.asarray(solution.fbs_model.permutation, dtype=np.float32).reshape(-1)
        n = int(permutation.size)
        if n <= 0:
            return np.zeros(19, dtype=np.float32)

        tm = np.asarray(solution.TM, dtype=np.float32)
        if tm.ndim == 2 and tm.shape[0] == tm.shape[1]:
            total_flow = np.sum(tm, axis=1) + np.sum(tm, axis=0)
        else:
            total_flow = np.zeros(n, dtype=np.float32)

        width = max(float(getattr(solution, "W", 1.0)), 1e-8)
        height = max(float(getattr(solution, "H", 1.0)), 1e-8)

        fac_x = runtime._fit_state_feature_length(getattr(solution, "fac_x", []), n, fill_value=0.0) / width
        fac_y = runtime._fit_state_feature_length(getattr(solution, "fac_y", []), n, fill_value=0.0) / height
        fac_b = runtime._fit_state_feature_length(getattr(solution, "fac_b", []), n, fill_value=0.0) / width
        fac_h = runtime._fit_state_feature_length(getattr(solution, "fac_h", []), n, fill_value=0.0) / height

        total_flow_norm = runtime._normalize_state_feature(total_flow)

        (
            position_norm,
            bay_index_norm,
            in_bay_rank_norm,
            bay_size_norm,
            permutation_list,
            bay_ranges,
        ) = self.build_facility_structural_state(runtime, solution, n)

        aspect_ratio = runtime._fit_state_feature_length(
            getattr(solution, "fac_aspect_ratio", []),
            n,
            fill_value=1.0,
        )
        aspect_limits = runtime._fit_state_feature_length(
            getattr(solution, "aspect_limits", []),
            n,
            fill_value=1.0,
        )
        safe_aspect_limits = np.where(
            np.isfinite(aspect_limits) & (aspect_limits > 1e-8),
            aspect_limits,
            1.0,
        ).astype(np.float32)
        safe_aspect_ratio = np.nan_to_num(
            aspect_ratio,
            nan=1.0,
            posinf=np.max(safe_aspect_limits),
            neginf=1.0,
        ).astype(np.float32)
        aspect_violation = np.tanh(
            np.maximum(safe_aspect_ratio / safe_aspect_limits - 1.0, 0.0)
        ).astype(np.float32)
        infeasible_mask = np.clip(
            np.nan_to_num(
                runtime._fit_state_feature_length(
                    getattr(solution, "infeasible_mask", []),
                    n,
                    fill_value=0.0,
                ),
                nan=0.0,
            ),
            0.0,
            1.0,
        ).astype(np.float32)

        global_context = self.build_global_state_context(
            runtime,
            solution,
            tm,
            permutation_list,
            bay_ranges,
            aspect_violation,
            infeasible_mask,
        )

        state_components = [
            position_norm,
            fac_x,
            fac_y,
            fac_b,
            fac_h,
            runtime._fit_state_feature_length(total_flow_norm, n, fill_value=0.5),
            bay_index_norm,
            in_bay_rank_norm,
            bay_size_norm,
            aspect_violation,
            infeasible_mask,
            global_context,
        ]
        state_vector = np.concatenate(state_components)
        return state_vector.astype(np.float32)


class DQNRewardEngine:
    """DQN 主链路奖励引擎。"""

    def compute_transition_reward(
        self,
        runtime,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        reward = 0.0
        scale = runtime._get_reward_cost_scale(previous_cost, next_cost, previous_best_feasible)
        if runtime.reward_profile == "s2":
            if np.isfinite(previous_cost) and np.isfinite(next_cost):
                base = max(abs(float(previous_cost)), 1.0)
                adaptive_scale = max(
                    runtime._get_reward_rel_delta_scale(previous_cost, next_cost),
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

    def compute_reward_components(
        self,
        runtime,
        previous_cost,
        immediate_cost,
        final_cost,
        previous_d_inf,
        immediate_d_inf,
        final_d_inf,
        previous_best_feasible,
        accept,
    ):
        immediate_reward = self.compute_transition_reward(
            runtime,
            previous_cost,
            immediate_cost,
            previous_d_inf,
            immediate_d_inf,
            previous_best_feasible,
            accept,
        )
        final_reward = self.compute_transition_reward(
            runtime,
            previous_cost,
            final_cost,
            previous_d_inf,
            final_d_inf,
            previous_best_feasible,
            accept,
        )
        return {
            "immediate_reward": float(immediate_reward),
            "final_reward": float(final_reward),
            "reward_uplift": float(final_reward - immediate_reward),
        }

    def compose_training_reward(self, runtime, reward_components):
        immediate_reward = float(reward_components["immediate_reward"])
        final_reward = float(reward_components["final_reward"])
        return float(
            runtime.reward_train_immediate_weight * immediate_reward
            + runtime.reward_train_final_weight * final_reward
        )


_TwoStageGraphRankerBase = nn.Module if nn is not None else object


class TwoStageGraphRanker(_TwoStageGraphRankerBase):
    """two-stage 第一阶段候选排序图网络。"""

    def __init__(
        self,
        node_dim,
        global_dim,
        hidden_dim=64,
        message_steps=2,
        dropout=0.05,
    ):
        if nn is None or GraphConv is None or global_mean_pool is None:
            raise ImportError("torch-geometric is required for TwoStageGraphRanker.")
        super().__init__()
        self.node_dim = int(node_dim)
        self.global_dim = int(global_dim)
        hidden_dim = int(max(16, hidden_dim))
        message_steps = int(max(1, message_steps))
        dropout = float(min(max(dropout, 0.0), 0.50))

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.graph_convs = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim, aggr="add") for _ in range(message_steps)]
        )
        self.graph_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(message_steps)]
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_batch):
        x = graph_batch.x.float()
        edge_index = graph_batch.edge_index.long()
        edge_weight = getattr(graph_batch, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        batch = getattr(graph_batch, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        hidden = self.node_encoder(x)
        for conv_layer, norm_layer in zip(self.graph_convs, self.graph_norms):
            updated = conv_layer(hidden, edge_index, edge_weight=edge_weight)
            hidden = torch.relu(norm_layer(hidden + updated))

        graph_embedding = global_mean_pool(hidden, batch)
        global_features = getattr(graph_batch, "global_features", None)
        if global_features is None:
            global_features = torch.zeros(
                (graph_embedding.size(0), self.global_dim),
                dtype=graph_embedding.dtype,
                device=graph_embedding.device,
            )
        elif global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        global_embedding = self.global_encoder(global_features.float())
        return self.head(torch.cat([graph_embedding, global_embedding], dim=-1)).squeeze(-1)


class DQNTwoStageLearnedEvaluator:
    """two-stage learned evaluator 适配器。"""

    heavy_action_order = (9, 10, 11, 14, 15)
    node_feature_names = (
        "area_norm",
        "flow_norm",
        "position_norm",
        "bay_index_norm",
        "in_bay_rank_norm",
        "bay_size_norm",
        "bay_area_ratio",
        "boundary_left",
        "boundary_right",
        "aspect_limit_norm",
    )
    global_feature_names = (
        "action_9",
        "action_10",
        "action_11",
        "action_14",
        "action_15",
        "bay_count_ratio",
        "avg_bay_size_ratio",
        "target_bay_area_ratio",
        "height_ratio",
        "flow_density",
    )

    def __init__(self):
        self.enabled = False
        self.collect_data = False
        self.collect_full_labels = False
        self.model_path = ""
        self.dataset_path = ""
        self.hidden_dim = 64
        self.message_steps = 2
        self.dropout = 0.05
        self.edge_topk = 12
        self.device = torch.device("cpu") if torch is not None else None
        self.model = None
        self.model_loaded = False
        self.model_missing_logged = False
        self.dataset_error_logged = False
        self.dataset_write_logged = False
        self.node_dim = len(self.node_feature_names)
        self.global_dim = len(self.global_feature_names)
        self.edge_cache = {}

    def configure(self, runtime):
        self.enabled = bool(getattr(runtime, "two_stage_learned_evaluator_enabled", False))
        self.collect_data = bool(getattr(runtime, "two_stage_learned_evaluator_collect_data", False))
        self.collect_full_labels = bool(
            getattr(runtime, "two_stage_learned_evaluator_collect_full_labels", False)
        )
        self.model_path = str(getattr(runtime, "two_stage_learned_evaluator_model_path", "")).strip()
        self.dataset_path = str(getattr(runtime, "two_stage_learned_evaluator_dataset_path", "")).strip()
        self.hidden_dim = int(max(16, getattr(runtime, "two_stage_learned_evaluator_hidden_dim", 64)))
        self.message_steps = int(max(1, getattr(runtime, "two_stage_learned_evaluator_message_steps", 2)))
        self.dropout = float(
            min(max(getattr(runtime, "two_stage_learned_evaluator_dropout", 0.05), 0.0), 0.50)
        )
        self.edge_topk = int(max(2, getattr(runtime, "two_stage_learned_evaluator_edge_topk", 12)))
        self.device = torch.device("cpu") if torch is not None else None
        self.model = None
        self.model_loaded = False
        self.model_missing_logged = False
        self.dataset_error_logged = False
        self.dataset_write_logged = False
        self.edge_cache = {}
        self._ensure_parent_dir(self.dataset_path)
        if self.collect_data and self.dataset_path:
            try:
                Path(self.dataset_path).expanduser().resolve().touch(exist_ok=True)
            except Exception as exc:
                runtime.logger.exception(
                    "two-stage evaluator 数据集文件初始化失败 | "
                    f"path: {self.dataset_path} | error: {exc}"
                )

    @staticmethod
    def _ensure_parent_dir(file_path):
        if not file_path:
            return
        Path(file_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_facility_index(facility_id, fallback_index, n):
        try:
            facility_int = int(facility_id)
        except Exception:
            facility_int = int(fallback_index) + 1
        if 1 <= facility_int <= n:
            return facility_int - 1
        if 0 <= facility_int < n:
            return facility_int
        if 0 <= int(fallback_index) < n:
            return int(fallback_index)
        return None

    @staticmethod
    def _safe_float_list(values):
        if values is None:
            return []
        return [float(value) for value in values]

    def _action_one_hot(self, action_idx):
        action_idx = int(action_idx)
        return [
            1.0 if action_idx == heavy_action else 0.0
            for heavy_action in self.heavy_action_order
        ]

    @staticmethod
    def _edge_cache_key(sym_flow, edge_topk):
        sym_flow = np.asarray(sym_flow, dtype=np.float32)
        flow_digest = hashlib.blake2b(sym_flow.tobytes(), digest_size=16).hexdigest()
        return (int(sym_flow.shape[0]), int(sym_flow.shape[1]), int(edge_topk), flow_digest)

    def _build_edge_lists(self, sym_flow):
        sym_flow = np.asarray(sym_flow, dtype=np.float32)
        n = int(sym_flow.shape[0])
        edge_sources = []
        edge_targets = []
        edge_weights = []
        max_edge_weight = float(np.max(sym_flow)) if sym_flow.size > 0 else 0.0
        max_edge_weight = max(max_edge_weight, 1.0)

        for node_idx in range(n):
            row = sym_flow[node_idx].copy()
            row[node_idx] = 0.0
            if self.edge_topk > 0 and row.size > self.edge_topk:
                candidate_indices = np.argpartition(-row, self.edge_topk - 1)[: self.edge_topk]
            else:
                candidate_indices = np.arange(row.size)
            for target_idx in candidate_indices:
                weight = float(row[int(target_idx)])
                if weight <= 0.0:
                    continue
                edge_sources.append(int(node_idx))
                edge_targets.append(int(target_idx))
                edge_weights.append(weight / max_edge_weight)

        for node_idx in range(n):
            edge_sources.append(int(node_idx))
            edge_targets.append(int(node_idx))
            edge_weights.append(1.0)

        if not edge_weights:
            edge_sources = list(range(n))
            edge_targets = list(range(n))
            edge_weights = [1.0] * n

        return [edge_sources, edge_targets], edge_weights

    def _get_edge_cache_entry(self, sym_flow):
        cache_key = self._edge_cache_key(sym_flow, self.edge_topk)
        cached_entry = self.edge_cache.get(cache_key)
        if cached_entry is not None:
            return cached_entry

        edge_index, edge_weight = self._build_edge_lists(sym_flow)
        cached_entry = {
            "edge_index_list": edge_index,
            "edge_weight_list": self._safe_float_list(edge_weight),
            "edge_index_tensor": None,
            "edge_weight_tensor": None,
        }
        if torch is not None:
            cached_entry["edge_index_tensor"] = torch.tensor(edge_index, dtype=torch.long)
            cached_entry["edge_weight_tensor"] = torch.tensor(edge_weight, dtype=torch.float32)
        self.edge_cache[cache_key] = cached_entry
        return cached_entry

    def _prepare_context_cache(self, runtime, context):
        cache_key = (
            int(self.edge_topk),
            int(self.node_dim),
            int(self.global_dim),
        )
        cached_context = context.get("_two_stage_learned_cache")
        if cached_context is not None and cached_context.get("cache_key") == cache_key:
            return cached_context

        area_array = np.asarray(context["areas"], dtype=np.float32).reshape(-1)
        sym_flow = np.asarray(context["sym_flow"], dtype=np.float32)
        n = int(area_array.size)
        aspect_limits = np.asarray(context["aspect_limits"], dtype=np.float32).reshape(-1)
        default_aspect_limit = max(float(context["default_aspect_limit"]), 1e-8)
        total_area = max(float(np.sum(area_array)), 1e-8)
        total_flow = (
            np.sum(sym_flow, axis=1).astype(np.float32)
            if sym_flow.ndim == 2
            else np.zeros(n, dtype=np.float32)
        )
        flow_scale = max(float(context["flow_scale"]), 1.0)
        safe_aspect_limits = np.where(
            np.isfinite(aspect_limits) & (aspect_limits > 1e-8),
            aspect_limits,
            default_aspect_limit,
        ).astype(np.float32)
        total_area_sqrt = math.sqrt(total_area)
        edge_cache_entry = self._get_edge_cache_entry(sym_flow)

        permutation_to_array_fn = getattr(runtime, "permutationToArray", None)
        if permutation_to_array_fn is None:
            from src.utils.FBSUtil import permutationToArray as _permutation_to_array

            permutation_to_array_fn = _permutation_to_array

        cached_context = {
            "cache_key": cache_key,
            "n": n,
            "area_array": area_array,
            "sym_flow": sym_flow,
            "area_norm": np.clip(area_array / total_area, 0.0, 1.0).astype(np.float32),
            "total_flow_norm": np.clip(total_flow / flow_scale, 0.0, 4.0).astype(np.float32) / 4.0,
            "aspect_limit_norm": np.clip(safe_aspect_limits / default_aspect_limit, 0.0, 4.0).astype(np.float32) / 4.0,
            "default_aspect_limit": default_aspect_limit,
            "total_area": total_area,
            "total_area_sqrt": total_area_sqrt,
            "target_bay_area": max(float(context["target_bay_area"]), 1e-8),
            "avg_bay_size": float(context["avg_bay_size"]),
            "flow_density": min(flow_scale / float(max(1, n * n)), 10.0) / 10.0,
            "height_ratio": min(float(context["H"]) / max(total_area_sqrt, 1e-8), 4.0) / 4.0,
            "edge_cache_entry": edge_cache_entry,
            "permutation_to_array_fn": permutation_to_array_fn,
        }
        context["_two_stage_learned_cache"] = cached_context
        return cached_context

    def _build_graph_components(
        self,
        runtime,
        prepared_context,
        permutation,
        bay,
        action_idx,
        meta=None,
        include_record=False,
    ):
        perm = runtime._to_int_list(permutation)
        bay_flags = runtime._to_int_list(bay)
        n = int(prepared_context["n"])
        if len(perm) == 0 or len(perm) != len(bay_flags) or len(perm) != n:
            return None, None

        bay_flags[-1] = 1
        permutation_to_array_fn = prepared_context["permutation_to_array_fn"]
        bay_structure = [list(current_bay) for current_bay in permutation_to_array_fn(perm, bay_flags)]
        if len(bay_structure) == 0:
            return None, None

        position_norm = np.zeros(n, dtype=np.float32)
        bay_index_norm = np.zeros(n, dtype=np.float32)
        in_bay_rank_norm = np.zeros(n, dtype=np.float32)
        bay_size_norm = np.zeros(n, dtype=np.float32)
        bay_area_ratio = np.zeros(n, dtype=np.float32)
        boundary_left = np.zeros(n, dtype=np.float32)
        boundary_right = np.zeros(n, dtype=np.float32)
        bay_count = max(1, len(bay_structure))
        bay_index_denom = float(max(1, bay_count - 1))
        position_denom = float(max(1, n - 1))
        target_bay_area = prepared_context["target_bay_area"]
        area_array = prepared_context["area_array"]

        for position, facility_id in enumerate(perm):
            facility_index = self._resolve_facility_index(facility_id, position, n)
            if facility_index is not None:
                position_norm[facility_index] = float(position) / position_denom

        for bay_idx, current_bay in enumerate(bay_structure):
            bay_size = max(1, len(current_bay))
            bay_area = 0.0
            for facility in current_bay:
                facility_idx = int(facility) - 1
                if 0 <= facility_idx < area_array.size:
                    bay_area += float(area_array[facility_idx])
            bay_area_feature = min(float(bay_area) / target_bay_area, 3.0) / 3.0
            for local_rank, facility_id in enumerate(current_bay):
                facility_index = self._resolve_facility_index(facility_id, local_rank, n)
                if facility_index is None:
                    continue
                bay_index_norm[facility_index] = 0.0 if bay_count <= 1 else float(bay_idx) / bay_index_denom
                in_bay_rank_norm[facility_index] = (
                    0.0 if bay_size <= 1 else float(local_rank) / float(max(1, bay_size - 1))
                )
                bay_size_norm[facility_index] = float(bay_size) / float(max(1, n))
                bay_area_ratio[facility_index] = bay_area_feature
                if local_rank == 0:
                    boundary_left[facility_index] = 1.0
                if local_rank == bay_size - 1:
                    boundary_right[facility_index] = 1.0

        node_features = np.stack(
            [
                prepared_context["area_norm"],
                prepared_context["total_flow_norm"],
                position_norm,
                bay_index_norm,
                in_bay_rank_norm,
                bay_size_norm,
                bay_area_ratio,
                boundary_left,
                boundary_right,
                prepared_context["aspect_limit_norm"],
            ],
            axis=1,
        ).astype(np.float32)
        global_features = np.asarray(
            self._action_one_hot(action_idx)
            + [
                float(bay_count) / float(max(1, n)),
                prepared_context["avg_bay_size"] / float(max(1, n)),
                target_bay_area / prepared_context["total_area"],
                prepared_context["height_ratio"],
                prepared_context["flow_density"],
            ],
            dtype=np.float32,
        )

        edge_cache_entry = prepared_context["edge_cache_entry"]
        graph_data = None
        if Data is not None and torch is not None:
            graph_data = Data(
                x=torch.from_numpy(node_features),
                edge_index=edge_cache_entry["edge_index_tensor"],
                edge_weight=edge_cache_entry["edge_weight_tensor"],
                global_features=torch.from_numpy(global_features.reshape(1, -1)),
            )

        record = None
        if include_record:
            record = {
                "node_features": node_features.tolist(),
                "global_features": global_features.tolist(),
                "edge_index": edge_cache_entry["edge_index_list"],
                "edge_weight": edge_cache_entry["edge_weight_list"],
                "template": runtime._two_stage_template_name(meta) if hasattr(runtime, "_two_stage_template_name") else "default",
            }
        return graph_data, record

    def build_graph_record(self, runtime, context, permutation, bay, action_idx, meta=None):
        prepared_context = self._prepare_context_cache(runtime, context)
        _graph_data, record = self._build_graph_components(
            runtime,
            prepared_context,
            permutation,
            bay,
            action_idx,
            meta=meta,
            include_record=True,
        )
        return record

    def build_data_from_record(self, record):
        if Data is None or torch is None:
            raise ImportError("torch-geometric is required for learned evaluator data conversion.")
        node_features = torch.tensor(record["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(record["edge_index"], dtype=torch.long)
        edge_weight = torch.tensor(record["edge_weight"], dtype=torch.float32)
        global_features = torch.tensor(record["global_features"], dtype=torch.float32)
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            global_features=global_features.unsqueeze(0),
        )

    def _build_model(self):
        return TwoStageGraphRanker(
            node_dim=self.node_dim,
            global_dim=self.global_dim,
            hidden_dim=self.hidden_dim,
            message_steps=self.message_steps,
            dropout=self.dropout,
        )

    def _load_model_if_needed(self, runtime):
        if not self.enabled:
            return False
        if self.model_loaded and self.model is not None:
            return True
        if torch is None or Data is None or GraphConv is None:
            if not self.model_missing_logged:
                runtime.logger.warning("Two-stage learned evaluator disabled: torch-geometric 不可用。")
                self.model_missing_logged = True
            return False
        if not self.model_path:
            return False
        model_file = Path(self.model_path)
        if not model_file.exists():
            return False
        checkpoint = torch.load(model_file, map_location="cpu")
        checkpoint_hidden_dim = int(checkpoint.get("hidden_dim", self.hidden_dim))
        checkpoint_message_steps = int(checkpoint.get("message_steps", self.message_steps))
        checkpoint_dropout = float(checkpoint.get("dropout", self.dropout))
        checkpoint_node_dim = int(checkpoint.get("node_dim", self.node_dim))
        checkpoint_global_dim = int(checkpoint.get("global_dim", self.global_dim))
        self.model = TwoStageGraphRanker(
            node_dim=checkpoint_node_dim,
            global_dim=checkpoint_global_dim,
            hidden_dim=checkpoint_hidden_dim,
            message_steps=checkpoint_message_steps,
            dropout=checkpoint_dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_loaded = True
        return True

    def score_candidate(self, runtime, context, permutation, bay, action_idx, meta=None):
        if not self._load_model_if_needed(runtime):
            return None, None
        prepared_context = self._prepare_context_cache(runtime, context)
        graph_data, record = self._build_graph_components(
            runtime,
            prepared_context,
            permutation,
            bay,
            action_idx,
            meta=meta,
            include_record=self.collect_data,
        )
        if graph_data is None:
            return None, None
        with torch.no_grad():
            batch = Batch.from_data_list([graph_data])
            score = float(self.model(batch).reshape(-1)[0].detach().cpu().item())
        return score, record

    def score_candidates_batch(self, runtime, context, proposals, action_idx):
        if not self._load_model_if_needed(runtime):
            runtime.logger.warning(
                "Two-stage learned evaluator batch 不可用，已返回空结果 | "
                f"reason: model_not_loaded | action: {int(action_idx)} | proposals: {len(proposals)}"
            )
            return {}
        prepared_context = self._prepare_context_cache(runtime, context)
        include_record = bool(self.collect_data)
        graph_data_list = []
        scored_indices = []
        record_map = {}

        for proposal_idx, (permutation, bay, meta) in enumerate(proposals):
            graph_data, record = self._build_graph_components(
                runtime,
                prepared_context,
                permutation,
                bay,
                action_idx,
                meta=meta,
                include_record=include_record,
            )
            if graph_data is None:
                continue
            graph_data_list.append(graph_data)
            scored_indices.append(int(proposal_idx))
            if record is not None:
                record_map[int(proposal_idx)] = record

        if not graph_data_list:
            runtime.logger.warning(
                "Two-stage learned evaluator batch 构图为空，已返回空结果 | "
                f"action: {int(action_idx)} | proposals: {len(proposals)}"
            )
            return {}

        with torch.no_grad():
            batch = Batch.from_data_list(graph_data_list)
            scores = self.model(batch).reshape(-1).detach().cpu().tolist()

        finite_count = 0
        for score in scores:
            try:
                if np.isfinite(float(score)):
                    finite_count += 1
            except Exception:
                pass
        if finite_count != len(scores):
            runtime.logger.warning(
                "Two-stage learned evaluator batch 输出包含无效值 | "
                f"action: {int(action_idx)} | proposals: {len(proposals)} | "
                f"scored: {len(scores)} | finite: {finite_count}"
            )

        return {
            proposal_idx: (
                float(score),
                record_map.get(int(proposal_idx)),
            )
            for proposal_idx, score in zip(scored_indices, scores)
        }

    def append_records(self, records, logger=None):
        if not self.collect_data or not self.dataset_path or not records:
            return
        self._ensure_parent_dir(self.dataset_path)
        try:
            with Path(self.dataset_path).open("a", encoding="utf-8") as output_file:
                for record in records:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 写入成功后重置错误标记，便于后续再次捕获新的间歇性问题。
            self.dataset_error_logged = False
            if logger is not None and not self.dataset_write_logged:
                logger.info(
                    "two-stage evaluator 数据集开始写入 | "
                    f"path: {self.dataset_path} | records: {len(records)}"
                )
                self.dataset_write_logged = True
        except Exception as exc:
            # 数据采集失败不影响主算法运行，但需要打印清晰日志便于排查。
            if not self.dataset_error_logged:
                error_message = (
                    "two-stage evaluator 数据集写入失败，后续同类错误将暂不重复打印 | "
                    f"path: {self.dataset_path} | records: {len(records)} | error: {exc}"
                )
                try:
                    if logger is not None:
                        logger.exception(error_message)
                    else:
                        print(error_message)
                except Exception:
                    print(error_message)
                self.dataset_error_logged = True
            return
