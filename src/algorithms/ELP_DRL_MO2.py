import copy
import datetime
import json
import math
import os
import random
from collections import deque

import gym
import numpy as np

from loguru import logger

from src.algorithms.ELP_DRL_MO import ELP as BaseMOELP
from src.algorithms.ELP_DRL_MO import _save_experiment_row
from src.algorithms.ELP_DRL_Standard import _get_initial_solution_energy, _set_global_seed
import src.utils.MO_ExperimentsUtil as MO_ExperimentsUtil
import src.utils.config as config
from src.utils.MO_FBSUtil import MO_FBSUtil

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


_QNetBase = nn.Module if nn is not None else object


class _ConditionedQNetwork(_QNetBase):
    def __init__(self, state_dim, pref_dim, action_dim, hidden_dim=256):
        if nn is None:
            raise ImportError("PyTorch is required for _ConditionedQNetwork")
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(state_dim + pref_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, state_tensor, pref_tensor):
        x = torch.cat([state_tensor, pref_tensor], dim=1)
        return self.model(x)


class ConditionedDQNAgent:
    def __init__(
        self,
        state_dim,
        pref_dim,
        a_dim,
        gamma=0.98,
        lr=2e-4,
        epsilon=0.35,
        epsilon_min=0.03,
        epsilon_decay=0.9996,
        batch_size=128,
        replay_capacity=100000,
        target_update_every=500,
        warmup_steps=4000,
        grad_clip=10.0,
        hidden_dim=256,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for ConditionedDQNAgent")

        self.state_dim = int(state_dim)
        self.pref_dim = int(pref_dim)
        self.a_dim = int(a_dim)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.warmup_steps = int(warmup_steps)
        self.grad_clip = float(grad_clip)

        self.device = torch.device("cpu")
        self.online_net = _ConditionedQNetwork(self.state_dim, self.pref_dim, self.a_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = _ConditionedQNetwork(self.state_dim, self.pref_dim, self.a_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=int(replay_capacity))

        self.total_steps = 0
        self.optimize_steps = 0

    def _mask_from_allowed(self, allowed_actions):
        mask = np.zeros(self.a_dim, dtype=bool)
        if allowed_actions is None:
            mask[:] = True
        else:
            for idx in allowed_actions:
                if 0 <= int(idx) < self.a_dim:
                    mask[int(idx)] = True
        if not np.any(mask):
            mask[:] = True
        return mask

    def select_action(self, state_vec, pref_vec, allowed_actions=None):
        self.total_steps += 1
        mask = self._mask_from_allowed(allowed_actions)
        valid_indices = np.where(mask)[0]

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_indices))

        with torch.no_grad():
            s = torch.tensor(np.asarray(state_vec, dtype=np.float32)[None, :], dtype=torch.float32, device=self.device)
            p = torch.tensor(np.asarray(pref_vec, dtype=np.float32)[None, :], dtype=torch.float32, device=self.device)
            q = self.online_net(s, p).squeeze(0)
            floor = torch.finfo(q.dtype).min
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
            q = torch.where(mask_t, q, floor)
            return int(torch.argmax(q).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def remember(self, s, p, a, r, s2, p2, done, allowed_next):
        self.replay.append(
            (
                np.asarray(s, dtype=np.float32),
                np.asarray(p, dtype=np.float32),
                int(a),
                float(r),
                np.asarray(s2, dtype=np.float32),
                np.asarray(p2, dtype=np.float32),
                float(done),
                self._mask_from_allowed(allowed_next).astype(np.bool_),
            )
        )

    def update(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None

        batch = random.sample(self.replay, self.batch_size)
        states, prefs, actions, rewards, next_states, next_prefs, dones, next_masks = map(list, zip(*batch))

        states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        prefs_t = torch.tensor(np.asarray(prefs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.asarray(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        next_prefs_t = torch.tensor(np.asarray(next_prefs), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(np.asarray(dones), dtype=torch.float32, device=self.device)
        next_masks_t = torch.tensor(np.asarray(next_masks), dtype=torch.bool, device=self.device)

        q_values = self.online_net(states_t, prefs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            q_next_online = self.online_net(next_states_t, next_prefs_t)
            q_next_target = self.target_net(next_states_t, next_prefs_t)
            floor = torch.finfo(q_next_online.dtype).min
            q_next_online_masked = torch.where(next_masks_t, q_next_online, floor)
            next_actions = torch.argmax(q_next_online_masked, dim=1, keepdim=True)
            q_next_target_masked = torch.where(next_masks_t, q_next_target, floor)
            q_next = q_next_target_masked.gather(1, next_actions).squeeze(1)
            target = rewards_t + (1.0 - dones_t) * self.gamma * q_next

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.optimize_steps += 1
        if self.optimize_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


class ELP(BaseMOELP):
    """ELP_DRL_MO2: 连续状态 + 条件化DQN + 精简干预机制。"""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        super().__init__(env, gbest, T, G=G, t_max=t_max, k=k, archive_limit=archive_limit, objective_weights=objective_weights)

        self.mo2_pref_bank = self._build_preference_bank()
        self.mo2_state_dim = 24
        self.mo2_pref_dim = 4
        self.mo2_spacing_reward_weight = float(os.getenv("ELP_MO2_SPACING_REWARD_WEIGHT", "0.70"))
        self.mo2_density_reward_weight = float(os.getenv("ELP_MO2_DENSITY_REWARD_WEIGHT", "0.18"))
        self.mo2_hv_reward_weight = float(os.getenv("ELP_MO2_HV_REWARD_WEIGHT", "0.12"))
        self.mo2_marginal_hv_reward_weight = float(os.getenv("ELP_MO2_MARGINAL_HV_REWARD_WEIGHT", "0.36"))
        self.mo2_archive_change_bonus = float(os.getenv("ELP_MO2_ARCHIVE_CHANGE_BONUS", "0.24"))
        self.mo2_weak_archive_penalty = float(os.getenv("ELP_MO2_WEAK_ARCHIVE_PENALTY", "0.14"))
        self.mo2_useful_hv_rel_threshold = float(os.getenv("ELP_MO2_USEFUL_HV_REL_THRESHOLD", "0.010"))
        self.mo2_archive_quality_guard_enabled = self._parse_env_flag("ELP_MO2_ARCHIVE_QUALITY_GUARD", True)
        self.mo2_quality_rep_slack = float(os.getenv("ELP_MO2_QUALITY_REP_SLACK", "0.08"))
        self.mo2_quality_median_slack = float(os.getenv("ELP_MO2_QUALITY_MEDIAN_SLACK", "0.05"))
        self.mo2_quality_override_score_bonus = float(os.getenv("ELP_MO2_QUALITY_OVERRIDE_SCORE_BONUS", "0.03"))
        self.mo2_quality_override_hv_rel = float(os.getenv("ELP_MO2_QUALITY_OVERRIDE_HV_REL", "0.06"))
        self.mo2_quality_override_spacing_tol = float(os.getenv("ELP_MO2_QUALITY_OVERRIDE_SPACING_TOL", "0.04"))
        self.mo2_quality_penalty_weight = float(os.getenv("ELP_MO2_QUALITY_PENALTY_WEIGHT", "0.28"))
        self.mo2_rep_reward_weight = float(os.getenv("ELP_MO2_REP_REWARD_WEIGHT", "0.12"))
        self.mo2_topk_reward_weight = float(os.getenv("ELP_MO2_TOPK_REWARD_WEIGHT", "0.08"))
        self.mo2_topk_score_k = int(os.getenv("ELP_MO2_TOPK_SCORE_K", "8"))
        self.mo2_core_quality_start_progress = float(os.getenv("ELP_MO2_CORE_QUALITY_START_PROGRESS", "0.62"))
        self.mo2_core_quality_only_useful = self._parse_env_flag("ELP_MO2_CORE_QUALITY_ONLY_USEFUL", True)
        self.mo2_reference_margin_ratio = float(os.getenv("ELP_MO2_REFERENCE_MARGIN_RATIO", "0.25"))
        self.mo2_reference_min_span_ratio = float(os.getenv("ELP_MO2_REFERENCE_MIN_SPAN_RATIO", "0.40"))
        self.mo2_reference_reuse = self._parse_env_flag("ELP_MO2_REFERENCE_REUSE", True)
        self.mo2_reference_rebuild = self._parse_env_flag("ELP_MO2_REFERENCE_REBUILD", False)
        self.mo2_reference_min_archive_size = int(os.getenv("ELP_MO2_REFERENCE_MIN_ARCHIVE_SIZE", str(max(12, archive_limit // 4))))
        self.archive_spacing_guard_when_full = self._parse_env_flag("ELP_MO2_ARCHIVE_SPACING_GUARD", False)
        self.archive_spacing_guard_rel_tol = float(os.getenv("ELP_MO2_ARCHIVE_SPACING_GUARD_REL_TOL", "0.03"))
        self.archive_spacing_guard_hv_gain_rel = float(os.getenv("ELP_MO2_ARCHIVE_SPACING_GUARD_HV_GAIN_REL", "0.12"))
        self.archive_require_candidate_retained = True
        self._last_mo2_archive_feedback = {}
        self._mo2_archive_snapshot = self._build_empty_archive_snapshot()
        self.mo2_reference_ideal = None
        self.mo2_reference_nadir = None
        self.mo2_reference_path = self._mo2_reference_frame_path()
        self.mo2_reference_source = "pending"

        # 关闭与RL并行缝合的强干预，保留主链路可解释性
        self.reheat_enabled = False
        self.local_search_disable_after_progress = 0.0
        self.diversify_trigger_no_improve = 10**9

    def _build_preference_bank(self):
        bank = [
            np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
            np.array([0.55, 0.15, 0.15, 0.15], dtype=np.float32),
            np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),
            np.array([0.15, 0.15, 0.55, 0.15], dtype=np.float32),
            np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),
            np.array([0.40, 0.40, 0.10, 0.10], dtype=np.float32),
            np.array([0.40, 0.10, 0.40, 0.10], dtype=np.float32),
            np.array([0.10, 0.40, 0.10, 0.40], dtype=np.float32),
        ]
        return [w / np.sum(w) for w in bank]

    def _sample_episode_preference(self, episode):
        # 前中后期都保留锚点偏好，避免后期塌缩到单一主权重。
        progress = float(episode) / float(max(1, self.G - 1))
        base = np.asarray(self.mo_weights, dtype=np.float32)
        anchor = self.mo2_pref_bank[int(episode) % len(self.mo2_pref_bank)].copy()
        jitter = np.random.dirichlet(np.ones(4, dtype=np.float32)).astype(np.float32)
        if progress < 0.50:
            mixed = 0.70 * anchor + 0.20 * base + 0.10 * jitter
        elif progress < 0.85:
            mixed = 0.45 * anchor + 0.45 * base + 0.10 * jitter
        else:
            mixed = 0.30 * anchor + 0.60 * base + 0.10 * jitter
        return (mixed / np.sum(mixed)).astype(np.float32)

    def _safe_obj_vector(self, solution):
        raw = getattr(solution, "mo_objectives_min", None)
        if raw is None:
            return np.full(4, np.inf, dtype=np.float32)
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            arr = np.pad(arr, (0, 4 - arr.size), constant_values=np.inf)
        return arr[:4]

    def _normalize_obj_delta(self, prev_obj, next_obj):
        if self.mo_ideal is not None and self.mo_nadir is not None:
            denom = np.maximum(np.asarray(self.mo_nadir, dtype=np.float32) - np.asarray(self.mo_ideal, dtype=np.float32), 1.0)
        else:
            denom = np.maximum(np.abs(prev_obj), np.ones_like(prev_obj))
        return np.clip((prev_obj - next_obj) / denom, -2.0, 2.0)

    @staticmethod
    def _build_empty_archive_snapshot():
        return {
            "count": 0,
            "hv": 0.0,
            "spacing": 0.0,
            "mean_nn": 0.0,
            "normalized": np.empty((0, 4), dtype=float),
            "ideal": None,
            "nadir": None,
        }

    def _mo2_reference_frame_path(self):
        safe_instance = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(self.instance_name))
        return os.path.join(config.RESULT_PATH, "reference_frames", f"{safe_instance}_mo2_fixed_reference.json")

    def _collect_reference_candidates(self):
        candidates = []
        for candidate in list(getattr(self, "pareto_archive", []) or []):
            if getattr(candidate, "current_is_feasible", False) and getattr(candidate, "mo_objectives_min", None) is not None:
                candidates.append(candidate)
        for candidate in [
            getattr(self, "s", None),
            getattr(self, "gbest", None),
            getattr(self, "true_gbest", None),
            getattr(self, "best_feasible_solution", None),
        ]:
            if candidate is None:
                continue
            if getattr(candidate, "current_is_feasible", False) and getattr(candidate, "mo_objectives_min", None) is not None:
                candidates.append(candidate)
        return candidates

    def _build_reference_frame_from_candidates(self, candidates):
        ideal, nadir = MO_FBSUtil.compute_ideal_nadir(candidates)
        if ideal is None or nadir is None:
            return None, None
        ideal = np.asarray(ideal, dtype=float)
        nadir = np.asarray(nadir, dtype=float)
        scale = np.maximum(np.maximum(np.abs(ideal), np.abs(nadir)), 1.0)
        min_span = np.maximum(scale * float(self.mo2_reference_min_span_ratio), 1.0)
        span = np.maximum(nadir - ideal, min_span)
        margin = span * float(self.mo2_reference_margin_ratio)
        fixed_ideal = ideal - margin
        fixed_nadir = ideal + span + margin
        fixed_nadir = np.maximum(fixed_nadir, fixed_ideal + min_span)
        return fixed_ideal.astype(float), fixed_nadir.astype(float)

    def _load_reference_frame_from_disk(self):
        if not self.mo2_reference_reuse or self.mo2_reference_rebuild:
            return False
        if not self.mo2_reference_path or not os.path.exists(self.mo2_reference_path):
            return False
        try:
            with open(self.mo2_reference_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            ideal = np.asarray(payload.get("ideal", []), dtype=float)
            nadir = np.asarray(payload.get("nadir", []), dtype=float)
            archive_size = int(payload.get("archiveSizeAtBuild", 0) or 0)
            if ideal.size != 4 or nadir.size != 4:
                return False
            if archive_size < int(self.mo2_reference_min_archive_size):
                return False
            self.mo2_reference_ideal = ideal
            self.mo2_reference_nadir = np.maximum(nadir, ideal + 1e-8)
            self.mo2_reference_source = "cache"
            return True
        except Exception as exc:
            logger.warning(f"Failed to load MO2 fixed reference frame: {exc}")
            return False

    def _save_reference_frame_to_disk(self, archive_size_override=None, source_override=None, extra_payload=None):
        if self.mo2_reference_ideal is None or self.mo2_reference_nadir is None or not self.mo2_reference_path:
            return
        archive_size = archive_size_override
        if archive_size is None:
            archive_size = len([candidate for candidate in self._collect_reference_candidates() if getattr(candidate, "current_is_feasible", False)])
        if archive_size < int(self.mo2_reference_min_archive_size):
            return
        try:
            os.makedirs(os.path.dirname(self.mo2_reference_path), exist_ok=True)
            payload = {
                "instance": self.instance_name,
                "generatedAt": datetime.datetime.now().isoformat(),
                "source": self.mo2_reference_source if source_override is None else source_override,
                "archiveSizeAtBuild": int(archive_size),
                "ideal": np.asarray(self.mo2_reference_ideal, dtype=float).tolist(),
                "nadir": np.asarray(self.mo2_reference_nadir, dtype=float).tolist(),
                "marginRatio": float(self.mo2_reference_margin_ratio),
                "minSpanRatio": float(self.mo2_reference_min_span_ratio),
            }
            if extra_payload:
                payload.update(dict(extra_payload))
            with open(self.mo2_reference_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save MO2 fixed reference frame: {exc}")

    def _ensure_mo2_reference_frame(self, force_rebuild=False):
        if (
            not force_rebuild
            and self.mo2_reference_ideal is not None
            and self.mo2_reference_nadir is not None
        ):
            return True
        if not force_rebuild and self._load_reference_frame_from_disk():
            return True

        candidates = self._collect_reference_candidates()
        ideal, nadir = self._build_reference_frame_from_candidates(candidates)
        if ideal is None or nadir is None:
            return False
        self.mo2_reference_ideal = ideal
        self.mo2_reference_nadir = nadir
        self.mo2_reference_source = "bootstrap_archive"
        self._save_reference_frame_to_disk()
        return True

    def _reference_frame_arrays(self):
        if self._ensure_mo2_reference_frame(force_rebuild=False):
            return (
                np.asarray(self.mo2_reference_ideal, dtype=float),
                np.asarray(self.mo2_reference_nadir, dtype=float),
            )
        return None, None

    def calibrate_reference_candidates(self, calibration_episodes, calibration_steps):
        calibration_episodes = max(1, int(calibration_episodes))
        calibration_steps = max(1, int(calibration_steps))
        quality_guard_enabled = bool(self.mo2_archive_quality_guard_enabled)
        self.mo2_reference_ideal = None
        self.mo2_reference_nadir = None
        self.mo2_reference_source = "calibration_live"
        self._mo2_archive_snapshot = self._build_empty_archive_snapshot()
        self.mo2_archive_quality_guard_enabled = False

        try:
            if not self._bootstrap_until_first_feasible(max_attempts=max(400, 4 * calibration_steps)):
                logger.warning("MO2 reference calibration failed to bootstrap a feasible archive.")
                return []

            total_steps = max(1, calibration_episodes * calibration_steps)
            global_step = 0
            for episode in range(calibration_episodes):
                self._prepare_episode_start(episode)
                for _ in range(calibration_steps):
                    self.current_progress_ratio = float(global_step) / float(total_steps)
                    allowed_actions = self._get_allowed_action_indices(self.s)
                    if not allowed_actions:
                        break
                    action_table_idx = int(np.random.choice(allowed_actions))
                    real_action_idx = self.valid_actions[action_table_idx]
                    candidate = self.generate_candidate_by_action(self.s, real_action_idx)
                    self._pending_candidate = candidate
                    accept, _, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
                    self._pending_candidate = None
                    if accept:
                        self.s = candidate
                        self.current_energy = self.s.fitness
                        self._observe_feasible_state(self.s)
                    global_step += 1
                self.T = max(self.T * self.cooling_per_step, self.T_min)

            return [copy.deepcopy(candidate) for candidate in self._collect_reference_candidates()]
        finally:
            self.mo2_archive_quality_guard_enabled = quality_guard_enabled

    def build_and_save_reference_from_candidates(self, candidates, source, extra_payload=None):
        candidates = [
            candidate
            for candidate in list(candidates or [])
            if getattr(candidate, "current_is_feasible", False) and getattr(candidate, "mo_objectives_min", None) is not None
        ]
        if not candidates:
            return False
        ideal, nadir = self._build_reference_frame_from_candidates(candidates)
        if ideal is None or nadir is None:
            return False
        self.mo2_reference_ideal = ideal
        self.mo2_reference_nadir = nadir
        self.mo2_reference_source = str(source)
        self._save_reference_frame_to_disk(
            archive_size_override=len(candidates),
            source_override=str(source),
            extra_payload=extra_payload,
        )
        return True

    def _refresh_archive_state(self):
        feasible_archive = [candidate for candidate in self.pareto_archive if getattr(candidate, "current_is_feasible", False)]
        self.pareto_archive = feasible_archive

        fixed_ideal, fixed_nadir = self._reference_frame_arrays()
        if fixed_ideal is None or fixed_nadir is None:
            fixed_ideal, fixed_nadir = MO_FBSUtil.compute_ideal_nadir(self.pareto_archive)
        self.mo_ideal = None if fixed_ideal is None else np.asarray(fixed_ideal, dtype=float)
        self.mo_nadir = None if fixed_nadir is None else np.asarray(fixed_nadir, dtype=float)

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
        self._mo2_archive_snapshot = self._archive_diversity_snapshot(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )

    def _archive_diversity_snapshot(self, archive=None, ideal=None, nadir=None):
        archive = list(self.pareto_archive if archive is None else archive)
        normalized, ideal_arr, nadir_arr = MO_FBSUtil._normalized_archive_matrix(archive, ideal=ideal, nadir=nadir)
        count = int(normalized.shape[0])
        if count == 0:
            return self._build_empty_archive_snapshot()

        hv = float(MO_FBSUtil.archive_hypervolume(archive, ideal=ideal_arr, nadir=nadir_arr))
        if count <= 1:
            return {
                "count": count,
                "hv": hv,
                "spacing": 0.0,
                "mean_nn": 0.0,
                "normalized": normalized,
                "ideal": ideal_arr,
                "nadir": nadir_arr,
            }

        nearest = np.full(count, np.inf, dtype=float)
        for idx in range(count):
            delta = normalized - normalized[idx]
            norms = np.linalg.norm(delta, axis=1)
            norms[idx] = np.inf
            nearest[idx] = float(np.min(norms))
        finite = nearest[np.isfinite(nearest)]
        mean_nn = float(np.mean(finite)) if finite.size else 0.0
        spacing = 0.0
        if finite.size and mean_nn > 1e-12:
            spacing = float(np.sqrt(np.mean((finite - mean_nn) ** 2)))
        return {
            "count": count,
            "hv": hv,
            "spacing": spacing,
            "mean_nn": mean_nn,
            "normalized": normalized,
            "ideal": ideal_arr,
            "nadir": nadir_arr,
        }

    def _refresh_mo2_archive_snapshot(self):
        self._mo2_archive_snapshot = self._archive_diversity_snapshot(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        return self._mo2_archive_snapshot

    def _solution_archive_profile(self, solution, snapshot=None):
        snapshot = self._mo2_archive_snapshot if snapshot is None else snapshot
        if snapshot.get("count", 0) <= 1:
            return {"nearest": 0.0, "ratio": 1.0, "uniformity": 0.0}
        objectives = getattr(solution, "mo_objectives_min", None)
        if objectives is None:
            return {"nearest": 0.0, "ratio": 1.0, "uniformity": 0.0}
        normalized = MO_FBSUtil.normalize_objective_vector(
            objectives,
            ideal=snapshot.get("ideal"),
            nadir=snapshot.get("nadir"),
        )
        if normalized is None:
            return {"nearest": 0.0, "ratio": 1.0, "uniformity": 0.0}

        archive_normalized = np.asarray(snapshot.get("normalized"), dtype=float)
        if archive_normalized.size == 0:
            return {"nearest": 0.0, "ratio": 1.0, "uniformity": 0.0}

        norms = np.linalg.norm(archive_normalized - normalized[None, :], axis=1)
        norms = norms[norms > 1e-9]
        nearest = float(np.min(norms)) if norms.size else 0.0
        mean_nn = float(snapshot.get("mean_nn", 0.0) or 0.0)
        ratio = nearest / max(mean_nn, 1e-8) if mean_nn > 0.0 else 1.0
        # 比值接近 1 代表候选点与当前前沿的局部密度更协调。
        uniformity = float(np.clip(1.0 - abs(math.log(max(ratio, 1e-8), 2.0)), -1.0, 1.0))
        return {"nearest": nearest, "ratio": ratio, "uniformity": uniformity}

    def _score_solution_under_reference(self, solution):
        if solution is None or getattr(solution, "mo_objectives_min", None) is None:
            return math.inf
        return float(
            MO_FBSUtil.decision_score(
                solution.mo_objectives_min,
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
                weights=self.mo_weights,
            )
        )

    def _archive_score_array(self, archive):
        scores = []
        for candidate in list(archive or []):
            if not getattr(candidate, "current_is_feasible", False):
                continue
            score = self._score_solution_under_reference(candidate)
            if np.isfinite(score):
                scores.append(float(score))
        if not scores:
            return np.empty(0, dtype=float)
        return np.asarray(scores, dtype=float)

    def _archive_topk_mean_score(self, archive, k=None):
        scores = self._archive_score_array(archive)
        if scores.size == 0:
            return math.inf
        k = int(self.mo2_topk_score_k if k is None else k)
        k = max(1, min(k, int(scores.size)))
        ranked = np.sort(scores)
        return float(np.mean(ranked[:k]))

    def _evaluate_archive_candidate_quality(self, solution, before_archive, tentative_archive, before_snapshot, tentative_snapshot):
        feedback = {
            "candidate_score": math.inf,
            "rep_score_before": math.inf,
            "topk_score_before": math.inf,
            "median_score_before": math.inf,
            "quality_limit": math.inf,
            "quality_override_limit": math.inf,
            "quality_excess": 0.0,
            "weak_tail_excess": 0.0,
            "quality_override_used": False,
            "quality_rejected": False,
            "quality_guard_active": bool(self.mo2_archive_quality_guard_enabled),
            "quality_pass": True,
            "quality_scale": max(float(self.mo2_quality_median_slack), 1e-6),
        }
        if not bool(self.mo2_archive_quality_guard_enabled):
            return feedback
        if self.mo_ideal is None or self.mo_nadir is None:
            return feedback

        before_scores = self._archive_score_array(before_archive)
        candidate_score = self._score_solution_under_reference(solution)
        feedback["candidate_score"] = float(candidate_score)
        if before_scores.size == 0 or not np.isfinite(candidate_score):
            return feedback

        rep_score_before = float(np.min(before_scores))
        topk_score_before = self._archive_topk_mean_score(before_archive)
        median_score_before = float(np.median(before_scores))
        progress = float(getattr(self, "current_progress_ratio", 0.0) or 0.0)
        rep_slack = float(self.mo2_quality_rep_slack) * (1.15 - 0.55 * progress)
        median_slack = float(self.mo2_quality_median_slack) * (1.10 - 0.30 * progress)
        rep_slack = max(rep_slack, 1e-6)
        median_slack = max(median_slack, 1e-6)
        quality_limit = min(rep_score_before + rep_slack, median_score_before + median_slack)

        before_hv = float(before_snapshot.get("hv", 0.0) or 0.0)
        after_hv = float(tentative_snapshot.get("hv", 0.0) or 0.0)
        before_spacing = float(before_snapshot.get("spacing", 0.0) or 0.0)
        after_spacing = float(tentative_snapshot.get("spacing", 0.0) or 0.0)
        if before_hv <= 1e-12:
            hv_rel_gain = 1.0 if after_hv > before_hv + 1e-12 else 0.0
        else:
            hv_rel_gain = (after_hv - before_hv) / before_hv
        spacing_rel_worse = 0.0
        if before_snapshot.get("count", 0) >= 2 and before_spacing > 1e-12:
            spacing_rel_worse = max(after_spacing - before_spacing, 0.0) / before_spacing

        quality_override_limit = quality_limit + float(self.mo2_quality_override_score_bonus)
        quality_pass = candidate_score <= quality_limit + 1e-12
        quality_override_used = False
        if not quality_pass:
            quality_override_used = bool(
                candidate_score <= quality_override_limit + 1e-12
                and hv_rel_gain >= float(self.mo2_quality_override_hv_rel)
                and spacing_rel_worse <= float(self.mo2_quality_override_spacing_tol)
            )
            quality_pass = quality_override_used

        feedback.update(
            {
                "rep_score_before": rep_score_before,
                "topk_score_before": float(topk_score_before),
                "median_score_before": median_score_before,
                "quality_limit": float(quality_limit),
                "quality_override_limit": float(quality_override_limit),
                "quality_excess": float(max(candidate_score - quality_limit, 0.0)),
                "weak_tail_excess": float(max(candidate_score - median_score_before, 0.0)),
                "quality_override_used": bool(quality_override_used),
                "quality_rejected": bool(not quality_pass),
                "quality_pass": bool(quality_pass),
                "quality_scale": float(max(rep_slack, median_slack)),
            }
        )
        return feedback

    def _observe_feasible_state(self, solution):
        if not getattr(solution, "current_is_feasible", False):
            self._last_mo2_archive_feedback = {}
            self._last_archive_observation = {"archive_changed": False, "rep_changed": False}
            return False

        before_snapshot = dict(getattr(self, "_mo2_archive_snapshot", {}) or {})
        if before_snapshot.get("count", 0) != len(self.pareto_archive):
            before_snapshot = self._archive_diversity_snapshot(
                self.pareto_archive,
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
            )
        before_archive = list(self.pareto_archive)
        before_hv = float(before_snapshot.get("hv", 0.0) or 0.0)
        before_spacing = float(before_snapshot.get("spacing", 0.0) or 0.0)

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

        tentative_archive, inserted, removed = MO_FBSUtil.update_pareto_archive(
            self.pareto_archive,
            solution,
            max_size=self.archive_limit,
            clone_fn=copy.deepcopy,
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
        tentative_inserted = bool(inserted)
        tentative_snapshot = before_snapshot
        candidate_retained = False
        if tentative_inserted:
            tentative_snapshot = self._archive_diversity_snapshot(
                tentative_archive,
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
            )
            candidate_retained = any(
                MO_FBSUtil._duplicate_objectives(solution, existing, atol=1e-9) for existing in tentative_archive
            )

        quality_feedback = self._evaluate_archive_candidate_quality(
            solution,
            before_archive,
            tentative_archive if tentative_inserted else before_archive,
            before_snapshot,
            tentative_snapshot,
        )
        rep_score_before = float(quality_feedback.get("rep_score_before", math.inf))
        topk_score_before = float(quality_feedback.get("topk_score_before", math.inf))
        quality_rejected = bool(tentative_inserted and candidate_retained and quality_feedback.get("quality_rejected", False))
        if quality_rejected:
            tentative_archive = before_archive
            tentative_inserted = False
            candidate_retained = False
            removed = 0
            tentative_snapshot = before_snapshot

        archive_changed = bool(tentative_inserted)
        if archive_changed:
            self.pareto_archive = tentative_archive
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
            "inserted": bool(tentative_inserted),
            "removedCount": int(removed or 0),
            "qualityRejected": bool(quality_rejected),
        }
        if archive_changed or rep_changed:
            observed_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
            if int(getattr(self, "_trace_step_index", -1) or -1) >= 0:
                observed_step += 1
            self._last_effective_archive_change_step = max(
                int(getattr(self, "_last_effective_archive_change_step", 0) or 0),
                int(observed_step),
            )
        if archive_changed or rep_changed:
            self._increment_archive_counters(archive_changed=archive_changed, rep_changed=rep_changed)
        if archive_changed:
            self._record_mo_event(
                "archive_update",
                inserted=bool(tentative_inserted),
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

        after_snapshot = dict(getattr(self, "_mo2_archive_snapshot", {}) or before_snapshot)
        after_hv = float(after_snapshot.get("hv", 0.0) or 0.0)
        after_spacing = float(after_snapshot.get("spacing", 0.0) or 0.0)
        local_profile = self._solution_archive_profile(solution, snapshot=after_snapshot)
        archive_observation = dict(getattr(self, "_last_archive_observation", {}) or {})
        rep_score_after = float(self.representative_decision_score if np.isfinite(self.representative_decision_score) else math.inf)
        topk_score_after = self._archive_topk_mean_score(self.pareto_archive)

        candidate_retained = any(
            MO_FBSUtil._duplicate_objectives(solution, existing, atol=1e-9) for existing in self.pareto_archive
        )
        marginal_hv_gain = 0.0
        marginal_hv_rel = 0.0
        if candidate_retained:
            kept_without_candidate = [
                existing
                for existing in self.pareto_archive
                if not MO_FBSUtil._duplicate_objectives(solution, existing, atol=1e-9)
            ]
            hv_without_candidate = float(
                MO_FBSUtil.archive_hypervolume(kept_without_candidate, ideal=self.mo_ideal, nadir=self.mo_nadir)
            )
            marginal_hv_gain = max(after_hv - hv_without_candidate, 0.0)
            marginal_hv_rel = marginal_hv_gain / max(after_hv, 1e-12) if after_hv > 1e-12 else 0.0

        if before_hv <= 1e-12:
            hv_rel_gain = 1.0 if after_hv > before_hv + 1e-12 else 0.0
        else:
            hv_rel_gain = (after_hv - before_hv) / before_hv

        spacing_rel_gain = 0.0
        if before_snapshot.get("count", 0) >= 2 and before_spacing > 1e-12:
            spacing_rel_gain = (before_spacing - after_spacing) / before_spacing
        rep_rel_gain = 0.0
        if np.isfinite(rep_score_before) and rep_score_before > 1e-12 and np.isfinite(rep_score_after):
            rep_rel_gain = (rep_score_before - rep_score_after) / rep_score_before
        topk_rel_gain = 0.0
        if np.isfinite(topk_score_before) and topk_score_before > 1e-12 and np.isfinite(topk_score_after):
            topk_rel_gain = (topk_score_before - topk_score_after) / topk_score_before

        useful_archive_update = bool(
            archive_changed
            and candidate_retained
            and (
                marginal_hv_rel >= float(self.mo2_useful_hv_rel_threshold)
                or spacing_rel_gain > 1e-6
                or rep_rel_gain > 1e-6
                or topk_rel_gain > 1e-6
                or int(archive_observation.get("removedCount", 0) or 0) > 0
            )
        )
        self._last_mo2_archive_feedback = {
            "archive_changed": bool(archive_changed),
            "candidate_retained": bool(candidate_retained),
            "useful_archive_update": useful_archive_update,
            "hv_rel_gain": float(hv_rel_gain),
            "marginal_hv_gain": float(marginal_hv_gain),
            "marginal_hv_rel": float(marginal_hv_rel),
            "spacing_rel_gain": float(spacing_rel_gain),
            "rep_score_before": float(rep_score_before),
            "rep_score_after": float(rep_score_after),
            "rep_rel_gain": float(rep_rel_gain),
            "topk_score_before": float(topk_score_before),
            "topk_score_after": float(topk_score_after),
            "topk_rel_gain": float(topk_rel_gain),
            "before_spacing": float(before_spacing),
            "after_spacing": float(after_spacing),
            "removed_count": int(archive_observation.get("removedCount", 0) or 0),
            "candidate_nearest": float(local_profile["nearest"]),
            "candidate_nn_ratio": float(local_profile["ratio"]),
            "candidate_uniformity": float(local_profile["uniformity"]),
        }
        self._last_mo2_archive_feedback.update(quality_feedback)
        self._last_mo2_archive_feedback["quality_rejected"] = bool(quality_rejected)
        return bool(archive_changed or rep_changed)

    def state_encoder_vector(self, solution, preference):
        pref = np.asarray(preference, dtype=np.float32)
        archive_snapshot = self._mo2_archive_snapshot
        local_profile = self._solution_archive_profile(solution, snapshot=archive_snapshot)

        t_ratio = float(self.T / max(self.T_initial, 1e-8))
        progress = float(getattr(self, "current_progress_ratio", 0.0) or 0.0)
        d_inf = float(getattr(solution, "current_d_inf", 0) or 0)
        d_inf_n = np.tanh(d_inf / 5.0)

        cost = float(getattr(solution, "fitness", np.inf))
        best = float(getattr(self, "best_feasible_cost", np.inf))
        gap = 0.0 if (not np.isfinite(cost) or not np.isfinite(best)) else max(cost - best, 0.0) / max(abs(best), 1.0)
        gap_n = np.tanh(gap)

        archive_size = float(len(getattr(self, "pareto_archive", []) or []))
        archive_size_n = np.tanh(archive_size / 32.0)

        stagnation = float(getattr(self, "no_improve_steps", 0) or 0)
        stagnation_n = np.tanh(stagnation / 100.0)

        violation = float(getattr(solution, "constraint_violation", 0.0) or 0.0)
        violation_n = np.tanh(violation / 10.0)

        obj = self._safe_obj_vector(solution)
        finite_obj = np.where(np.isfinite(obj), obj, 1e9)
        obj_scale = np.maximum(np.abs(finite_obj), 1.0)
        obj_n = np.clip(np.log1p(obj_scale) / 20.0, 0.0, 1.0)

        ideal = getattr(self, "mo_ideal", None)
        nadir = getattr(self, "mo_nadir", None)
        spread = np.ones(4, dtype=np.float32)
        obj_pos = np.zeros(4, dtype=np.float32)
        if ideal is not None and nadir is not None:
            ideal_arr = np.asarray(ideal, dtype=np.float32)
            spread = np.maximum(np.asarray(nadir, dtype=np.float32) - ideal_arr, 1e-8)
            obj_pos = np.clip((finite_obj - ideal_arr) / spread, 0.0, 3.0)
        else:
            obj_pos = np.clip(np.log1p(np.abs(finite_obj)) / 20.0, 0.0, 3.0)

        archive_hv_n = np.tanh(float(archive_snapshot.get("hv", 0.0) or 0.0))
        archive_spacing_n = np.tanh(8.0 * float(archive_snapshot.get("spacing", 0.0) or 0.0))
        local_nn_n = np.tanh(6.0 * float(local_profile["nearest"]))
        local_ratio_n = np.tanh(float(local_profile["ratio"]) - 1.0)

        feature = np.array(
            [
                t_ratio,
                progress,
                d_inf_n,
                gap_n,
                archive_size_n,
                stagnation_n,
                violation_n,
                float(getattr(solution, "current_is_feasible", False)),
                obj_n[0],
                obj_n[1],
                obj_n[2],
                obj_n[3],
                obj_pos[0],
                obj_pos[1],
                obj_pos[2],
                obj_pos[3],
                float(getattr(self, "accept_rate_window", 0.0) or 0.0),
                float(getattr(self, "k_hist", 0.0) or 0.0) / 10.0,
                float(getattr(self, "gbest_update_count", 0) or 0) / max(1.0, float(self.G * self.t_max)),
                float(getattr(self, "feasible_solution_count", 0) or 0) / max(1.0, float(self.G * self.t_max)),
                archive_hv_n,
                archive_spacing_n,
                local_nn_n,
                local_ratio_n,
            ],
            dtype=np.float32,
        )

        # 固定24维：20维基础状态 + 4维前沿密度上下文。
        if feature.size < self.mo2_state_dim:
            feature = np.pad(feature, (0, self.mo2_state_dim - feature.size), constant_values=0.0)
        return feature[: self.mo2_state_dim], pref[:4]

    def _compute_conditioned_reward(self, previous_solution, next_solution, accept, preference, archive_changed):
        prev_obj = self._safe_obj_vector(previous_solution)
        next_obj = self._safe_obj_vector(next_solution)
        obj_delta = self._normalize_obj_delta(prev_obj, next_obj)

        pref = np.asarray(preference, dtype=np.float32)
        weighted_gain = float(np.dot(pref, obj_delta))

        d_inf_prev = float(getattr(previous_solution, "current_d_inf", 0) or 0)
        d_inf_next = float(getattr(next_solution, "current_d_inf", 0) or 0)
        feas_bonus = 0.0
        if d_inf_prev > 0 and d_inf_next == 0:
            feas_bonus += 0.6
        feas_bonus += 0.1 * np.clip((d_inf_prev - d_inf_next), -5.0, 5.0)

        archive_feedback = dict(getattr(self, "_last_mo2_archive_feedback", {}) or {})
        progress = float(getattr(self, "current_progress_ratio", 0.0) or 0.0)
        spacing_term = 0.0
        density_term = 0.0
        hv_term = 0.0
        contribution_term = 0.0
        quality_penalty_term = 0.0
        rep_term = 0.0
        topk_term = 0.0
        archive_bonus = -0.05
        if accept and getattr(next_solution, "current_is_feasible", False):
            candidate_retained = bool(archive_feedback.get("candidate_retained", False))
            useful_archive_update = bool(archive_feedback.get("useful_archive_update", False))
            spacing_gain = float(archive_feedback.get("spacing_rel_gain", 0.0) or 0.0)
            hv_gain = float(archive_feedback.get("hv_rel_gain", 0.0) or 0.0)
            marginal_hv_rel = float(archive_feedback.get("marginal_hv_rel", 0.0) or 0.0)
            rep_rel_gain = float(archive_feedback.get("rep_rel_gain", 0.0) or 0.0)
            topk_rel_gain = float(archive_feedback.get("topk_rel_gain", 0.0) or 0.0)
            density_uniformity = float(archive_feedback.get("candidate_uniformity", 0.0) or 0.0)
            quality_excess = float(archive_feedback.get("quality_excess", 0.0) or 0.0)
            quality_scale = max(float(archive_feedback.get("quality_scale", 0.0) or 0.0), 1e-6)
            quality_rejected = bool(archive_feedback.get("quality_rejected", False))
            quality_penalty_term = -float(self.mo2_quality_penalty_weight) * float(np.tanh(quality_excess / quality_scale))
            if quality_rejected:
                quality_penalty_term -= 0.6 * float(self.mo2_quality_penalty_weight)
            if candidate_retained:
                # 后期逐步提高均匀性权重，避免训练后段继续向单一区域挤压。
                spacing_weight = self.mo2_spacing_reward_weight * (0.60 + 0.80 * progress)
                density_weight = self.mo2_density_reward_weight * (0.50 + 0.90 * progress)
                hv_weight = self.mo2_hv_reward_weight * (1.10 - 0.35 * progress)
                spacing_term = spacing_weight * float(np.tanh(4.0 * spacing_gain))
                density_term = density_weight * density_uniformity
                hv_term = hv_weight * float(np.tanh(3.0 * hv_gain))
                contribution_term = self.mo2_marginal_hv_reward_weight * float(np.tanh(10.0 * marginal_hv_rel))
                core_quality_active = progress >= float(self.mo2_core_quality_start_progress)
                if bool(self.mo2_core_quality_only_useful):
                    core_quality_active = core_quality_active and useful_archive_update
                if core_quality_active:
                    # 代表解/前沿核心质量只在后期轻量奖励，避免前期探索被过早压制。
                    phase = (progress - float(self.mo2_core_quality_start_progress)) / max(
                        1.0 - float(self.mo2_core_quality_start_progress), 1e-6
                    )
                    phase = float(np.clip(phase, 0.0, 1.0))
                    rep_gain_pos = max(rep_rel_gain, 0.0)
                    topk_gain_pos = max(topk_rel_gain, 0.0)
                    rep_term = self.mo2_rep_reward_weight * phase * float(np.tanh(4.0 * rep_gain_pos))
                    topk_term = self.mo2_topk_reward_weight * phase * float(np.tanh(4.0 * topk_gain_pos))
            if archive_changed and useful_archive_update:
                archive_bonus = self.mo2_archive_change_bonus
            elif archive_changed and candidate_retained:
                archive_bonus = 0.35 * self.mo2_archive_change_bonus
            elif archive_changed:
                archive_bonus = -float(self.mo2_weak_archive_penalty)

        reject_penalty = -0.08 if not accept else 0.0

        reward = 1.45 * weighted_gain + feas_bonus + archive_bonus + reject_penalty + spacing_term + density_term + hv_term + contribution_term + rep_term + topk_term + quality_penalty_term
        return float(np.clip(reward, -3.5, 3.5))

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        self._reset_mo_logging_state()
        self._run_start_time = start_time

        run_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_MO2")
        run_remark = os.getenv("ELP_EXP_REMARK", "")
        self.mo_recorder = MO_ExperimentsUtil.MOExperimentRecorder(
            instance=self.instance_name,
            algorithm=run_algorithm,
            start_time=start_time,
            trace_interval=self.mo_trace_interval,
            remark=run_remark,
            result_root=config.RESULT_PATH,
        )

        agent = ConditionedDQNAgent(
            state_dim=self.mo2_state_dim,
            pref_dim=self.mo2_pref_dim,
            a_dim=len(self.valid_actions),
            epsilon=0.35,
            epsilon_min=max(0.01, self.dqn_epsilon_min * 0.5),
            epsilon_decay=self.dqn_epsilon_decay,
            gamma=0.98,
            lr=2e-4,
            batch_size=128,
            replay_capacity=120000,
            warmup_steps=max(3000, self.dqn_warmup_steps),
            target_update_every=max(300, self.dqn_target_update_every),
            hidden_dim=256,
        )

        if not self._bootstrap_until_first_feasible():
            logger.warning("Failed to seed a feasible archive before ELP_DRL_MO2 search.")
        if self.mo2_reference_rebuild:
            logger.warning("ELP_MO2_REFERENCE_REBUILD=1 会基于当前运行早期档案重建参考系，只建议用于显式校准模式。")
        self._ensure_mo2_reference_frame(force_rebuild=bool(self.mo2_reference_rebuild))
        self._refresh_archive_state()

        global_step = 0
        total_steps = max(1, self.G * self.t_max)
        next_progress_marker_idx = 0

        for episode in range(self.G):
            if self.worst_feasible_cost is None and not self._bootstrap_until_first_feasible(max_attempts=max(200, 2 * self.t_max)):
                logger.warning(f"Episode {episode}: feasible archive still unavailable.")
                continue

            episode_pref = self._sample_episode_preference(episode)
            episode_best_before = self.best_feasible_cost
            self._prepare_episode_start(episode)

            for step_idx in range(self.t_max):
                self._trace_global_step = global_step
                self._trace_episode_index = episode
                self._trace_step_index = step_idx
                self.current_progress_ratio = float(global_step) / float(max(1, total_steps))

                state_vec, pref_vec = self.state_encoder_vector(self.s, episode_pref)
                allowed_actions = self._get_allowed_action_indices(self.s)
                action_table_idx = agent.select_action(state_vec, pref_vec, allowed_actions=allowed_actions)
                real_action_idx = self.valid_actions[action_table_idx]

                previous_solution = copy.deepcopy(self.s)
                previous_cost = self.s.fitness

                candidate = self.generate_candidate_by_action(self.s, real_action_idx)
                self._record_action_selection(real_action_idx, previous_cost, candidate.fitness, phase="main")

                self._pending_candidate = candidate
                accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
                self._pending_candidate = None
                self.prob_history.append(prob)
                self._record_acceptance(accept)

                archive_changed = False
                if accept:
                    self.s = candidate
                    self.current_energy = self.s.fitness
                    archive_changed = bool(self._observe_feasible_state(self.s))
                    accepted_improved = bool(np.isfinite(previous_cost) and np.isfinite(self.s.fitness) and self.s.fitness < previous_cost)
                    self._record_action_acceptance(real_action_idx, previous_cost, self.s.fitness, improved=accepted_improved, phase="main")
                    if archive_changed:
                        self._record_action_global_best(real_action_idx, phase="main")
                        self.no_improve_steps = 0
                        fast_time = datetime.datetime.now()
                    else:
                        self.no_improve_steps += 1
                else:
                    self.no_improve_steps += 1

                self._refresh_bin_width_from_best()

                next_state_vec, next_pref_vec = self.state_encoder_vector(self.s, episode_pref)
                reward = self._compute_conditioned_reward(previous_solution, self.s, accept, episode_pref, archive_changed)
                done_flag = step_idx == self.t_max - 1

                allowed_next_actions = self._get_allowed_action_indices(self.s)
                agent.remember(state_vec, pref_vec, action_table_idx, reward, next_state_vec, next_pref_vec, done_flag, allowed_next_actions)
                loss_value = agent.update()

                self._last_loss_value = loss_value
                self._update_transition_counters(accept, prob, reward, loss_value)
                self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
                self.energy_history.append(self.s.fitness)

                global_step += 1
                self._trace_global_step = global_step

                while next_progress_marker_idx < len(self.progress_markers) and (global_step / total_steps) >= self.progress_markers[next_progress_marker_idx]:
                    self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
                    next_progress_marker_idx += 1

                self._record_trace_snapshot(global_step, total_steps, episode, step_idx, agent)
                self._clear_action_context()

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
        self._save_reference_frame_to_disk()

        best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else copy.deepcopy(self.s)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)
        if self.representative_solution is not None:
            best_solution = copy.deepcopy(self.representative_solution)
            best_energy = float(self.representative_decision_score)
            is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        representative_stable_score = None
        if best_solution is not None:
            representative_stable_score = self._safe_float(getattr(best_solution, "proxy_energy", None))
            if representative_stable_score is None and getattr(best_solution, "mo_objectives_min", None) is not None:
                representative_stable_score = self._safe_float(
                    MO_FBSUtil.surrogate_energy(
                        best_solution.mo_objectives_min,
                        ideal=self.mo_ideal,
                        nadir=self.mo_nadir,
                        weights=self.mo_weights,
                    )
                )

        archive_hypervolume = self._safe_float(
            MO_FBSUtil.archive_hypervolume(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )
        archive_spacing = self._safe_float(
            MO_FBSUtil.archive_spacing(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )

        archive_path = self._save_pareto_archive(start_time, algorithm_name=run_algorithm)
        reference_metrics = self._compute_reference_front_metrics()
        best_result_seconds = None if fast_time is None else (fast_time - start_time).total_seconds()

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
            "decisionScore": self._safe_float(best_energy),
            "stableDecisionScore": representative_stable_score,
            "archiveHypervolume": archive_hypervolume,
            "archiveSpacing": archive_spacing,
            "archiveIgd": reference_metrics["archive_igd"],
            "referenceFrontPath": reference_metrics["reference_front_path"],
            "referenceFrontSize": reference_metrics["reference_front_size"],
            "referenceArchiveCount": reference_metrics["reference_front_archive_count"],
            "repMhc": None if best_solution is None else self._safe_float(getattr(best_solution, "MHC", None)),
            "repCr": None if best_solution is None else self._safe_float(getattr(best_solution, "CR", None)),
            "repDr": None if best_solution is None else self._safe_float(getattr(best_solution, "DR", None)),
            "repAr": None if best_solution is None else self._safe_float(getattr(best_solution, "AR", None)),
            "paretoArchivePath": archive_path,
            "agentMode": "conditioned_dqn",
            "epsilonEnd": self._safe_float(getattr(agent, "epsilon", None)),
            "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0),
            "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0),
            "G": int(self.G),
            "tMax": int(self.t_max),
            "archiveLimit": int(self.archive_limit),
            "traceInterval": int(self.mo_trace_interval),
            "objectiveWeights": np.asarray(self.mo_weights, dtype=float).tolist(),
            "mo2PreferenceBankSize": int(len(self.mo2_pref_bank)),
            "mo2ReferencePath": self.mo2_reference_path,
            "mo2ReferenceSource": self.mo2_reference_source,
        }
        action_stats = self._build_action_stats_payload(agent, global_step)
        self.mo_run_summary = self.mo_recorder.finalize(run_summary, action_stats)

        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
            "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
            "rep_dr": None if best_solution is None else float(getattr(best_solution, "DR", 0.0)),
            "rep_ar": None if best_solution is None else float(getattr(best_solution, "AR", 0.0)),
            "decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "stable_decision_score": representative_stable_score,
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": reference_metrics["archive_igd"],
            "reference_front_path": reference_metrics["reference_front_path"],
            "reference_front_size": reference_metrics["reference_front_size"],
            "reference_front_archive_count": reference_metrics["reference_front_archive_count"],
            "mo2_reference_path": self.mo2_reference_path,
            "mo2_reference_source": self.mo2_reference_source,
            "mo_run_id": self.mo_run_summary.get("runId"),
            "mo_bundle_dir": self.mo_run_summary.get("bundleDir"),
            "mo_trace_path": self.mo_run_summary.get("tracePath"),
            "mo_events_path": self.mo_run_summary.get("eventsPath"),
            "mo_action_stats_path": self.mo_run_summary.get("actionStatsPath"),
            "mo_run_summary_path": self.mo_run_summary.get("runSummaryPath"),
        }
        return global_step, is_valid, best_solution, best_energy, start_time, end_time, fast_time


if __name__ == "__main__":
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

    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}" if baseline_enabled else "ELP_DRL_MO2"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    default_remark = (
        "MO baseline on action-sequence encoding"
        if baseline_enabled
        else "Conditioned-DQN+ELP+Pareto archive with representative solution"
    )
    exp_remark = os.getenv("ELP_EXP_REMARK", default_remark)
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 30)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)
    calibrate_reference = _parse_env_flag("ELP_MO2_CALIBRATE_REFERENCE", False)

    G = _parse_env_int("ELP_G", 1000)
    t_max = _parse_env_int("ELP_T_MAX", 300)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260427)
    calibration_runs = _parse_env_int("ELP_MO2_CALIBRATION_RUNS", 8)
    calibration_G = _parse_env_int("ELP_MO2_CALIBRATION_G", 20)
    calibration_t_max = _parse_env_int("ELP_MO2_CALIBRATION_TMAX", 40)

    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
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

    def _calibrate_reference_once(run_index):
        run_seed = int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Calibration seed: {run_seed} | strict_determinism: {strict_determinism}")
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        except Exception:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=max(1, calibration_G),
            t_max=max(1, calibration_t_max),
            k=k_hist,
        )
        solver.mo2_reference_reuse = False
        solver.mo2_reference_rebuild = False
        candidates = solver.calibrate_reference_candidates(calibration_G, calibration_t_max)
        return solver, candidates

    def _run_reference_calibration():
        logger.info(
            f"Starting MO2 reference calibration | instance={exp_instance} | runs={calibration_runs} | G={calibration_G} | t_max={calibration_t_max}"
        )
        aggregated_candidates = []
        owner_solver = None
        per_run_sizes = []
        for run_index in range(calibration_runs):
            solver, candidates = _calibrate_reference_once(run_index)
            owner_solver = solver
            aggregated_candidates.extend(candidates)
            per_run_sizes.append(len(candidates))
            logger.info(f"Calibration run {run_index + 1}/{calibration_runs} collected {len(candidates)} feasible reference candidates")
        if owner_solver is None or not aggregated_candidates:
            raise RuntimeError("MO2 reference calibration produced no feasible candidates.")
        extra_payload = {
            "calibrationRuns": int(calibration_runs),
            "calibrationEpisodesPerRun": int(calibration_G),
            "calibrationStepsPerEpisode": int(calibration_t_max),
            "calibrationCandidateCountPerRun": per_run_sizes,
            "calibrationTotalCandidates": int(len(aggregated_candidates)),
        }
        ok = owner_solver.build_and_save_reference_from_candidates(
            aggregated_candidates,
            source="calibration_runs",
            extra_payload=extra_payload,
        )
        if not ok:
            raise RuntimeError("MO2 reference calibration failed to build a fixed reference frame.")
        logger.info(
            f"MO2 reference calibration complete | path={owner_solver.mo2_reference_path} | total_candidates={len(aggregated_candidates)}"
        )
        print(owner_solver.mo2_reference_path)

    if calibrate_reference:
        _run_reference_calibration()
    elif is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(f"Experiment {i + 1} complete | representative decision score: {best_energy}")
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
        print(f"Single run complete | representative decision score: {best_energy}")
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
