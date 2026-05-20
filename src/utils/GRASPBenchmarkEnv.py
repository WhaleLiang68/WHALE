from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, Optional

import gym
import numpy as np
from gym import spaces

from src.utils.FBSModel import FBSModel
from src.utils.GRASPInstanceLoader import GRASPInstanceLoader


class GRASPBenchmarkEnv(gym.Env):
    """论文 BO-MREFLP 固定网格环境。"""

    metadata = {"render_modes": []}

    def __init__(self, instance=None, seed=None, options=None):
        super().__init__()
        payload = GRASPInstanceLoader.load_instance(str(instance or "A-10-10"))
        self.uuid = uuid.uuid4()
        self.instance = payload["name"]
        self.n = int(payload["n"])
        self.m = int(payload["rows"])
        self.c = int(payload["cols"])
        self.F = np.asarray(payload["mhc_matrix"], dtype=float)
        self.CR_matrix = np.asarray(payload["cr_matrix"], dtype=float)
        self.WMHC = self.F.copy()
        self.WCR = self.CR_matrix.copy()
        self.H = float(self.m)
        self.W = float(self.c)
        self.areas = np.ones(self.n, dtype=float)
        self.aspect_limits = np.ones(self.n, dtype=float)
        self.fac_limit_aspect = 1.0
        self.paper_instance_profile = {
            "source": "grasp_paper_benchmark",
            "instance": self.instance,
            "rows": self.m,
            "cols": self.c,
        }
        self.actions = {
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
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.n * 3,), dtype=np.float64)
        self._fixed_bay = self._build_fixed_bay()
        self._clear_runtime_tracking()
        if seed is not None:
            np.random.seed(int(seed))
        if options is not None and "seed" in options:
            np.random.seed(int(options["seed"]))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _build_fixed_bay(self) -> np.ndarray:
        bay = np.zeros(self.n, dtype=int)
        for row_idx in range(self.m):
            bay[(row_idx + 1) * self.c - 1] = 1
        return bay

    def _clear_runtime_tracking(self):
        self.fbs_model = None
        self.fac_x = np.zeros(self.n, dtype=float)
        self.fac_y = np.zeros(self.n, dtype=float)
        self.fac_b = np.ones(self.n, dtype=float)
        self.fac_h = np.ones(self.n, dtype=float)
        self.fac_aspect_ratio = np.ones(self.n, dtype=float)
        self.lower_bounds = np.ones(self.n, dtype=float)
        self.upper_bounds = np.ones(self.n, dtype=float)
        self.infeasible_mask = np.zeros(self.n, dtype=bool)
        self.D = np.zeros((self.n, self.n), dtype=float)
        self.TM = np.zeros((self.n, self.n), dtype=float)
        self.MHC = np.inf
        self.CR = np.inf
        self.fitness = np.inf
        self.best_fitness = np.inf
        self.best_MHC = np.inf
        self.previous_fitness = np.inf
        self.previous_MHC = np.inf
        self.previous_CR = np.inf
        self.previous_d_inf = 0
        self.current_d_inf = 0
        self.current_is_feasible = True
        self.current_v_worst = None
        self.feasible_solution_count = 0
        self.best_feasible_cost = np.inf
        self.worst_feasible_cost = None
        self.best_feasible_solution = None
        self.state = np.zeros(self.n * 3, dtype=float)

    def _make_initial_model(self) -> FBSModel:
        permutation = np.arange(1, self.n + 1, dtype=int)
        np.random.shuffle(permutation)
        return FBSModel(permutation.tolist(), self._fixed_bay.copy().tolist(), genes=permutation.tolist())

    def _normalize_permutation(self, permutation) -> np.ndarray:
        values = np.asarray(permutation, dtype=int).reshape(-1).tolist()
        expected = list(range(1, self.n + 1))
        seen = set()
        deduped = []
        for value in values:
            if 1 <= int(value) <= self.n and int(value) not in seen:
                deduped.append(int(value))
                seen.add(int(value))
        for value in expected:
            if value not in seen:
                deduped.append(value)
        return np.asarray(deduped[: self.n], dtype=int)

    def _coords_from_permutation(self, permutation: np.ndarray):
        fac_x = np.zeros(self.n, dtype=float)
        fac_y = np.zeros(self.n, dtype=float)
        for idx, facility in enumerate(permutation.tolist()):
            row = idx // self.c
            col = idx % self.c
            fac_x[facility - 1] = float(col)
            fac_y[facility - 1] = float(row)
        return fac_x, fac_y

    def evaluate_fbs_model(self, fbs_model: FBSModel) -> Dict[str, Any]:
        permutation = self._normalize_permutation(fbs_model.permutation)
        fac_x, fac_y = self._coords_from_permutation(permutation)
        distance_matrix = np.abs(fac_x[:, None] - fac_x[None, :]) + np.abs(fac_y[:, None] - fac_y[None, :])
        mhc = float(np.sum(np.triu(self.WMHC * distance_matrix, k=1)))
        cr = float(np.sum(np.triu(self.WCR * distance_matrix, k=1)))
        tm = self.WMHC * distance_matrix
        return {
            "fac_x": fac_x,
            "fac_y": fac_y,
            "fac_b": np.ones(self.n, dtype=float),
            "fac_h": np.ones(self.n, dtype=float),
            "fac_aspect_ratio": np.ones(self.n, dtype=float),
            "lower_bounds": np.ones(self.n, dtype=float),
            "upper_bounds": np.ones(self.n, dtype=float),
            "aspect_limits": np.ones(self.n, dtype=float),
            "infeasible_mask": np.zeros(self.n, dtype=bool),
            "D": distance_matrix,
            "TM": tm,
            "mhc": mhc,
            "cr": cr,
            "cost": mhc + cr,
            "d_inf": 0,
            "is_feasible": True,
            "v_worst": None,
            "constraint_violation": 0.0,
            "permutation": permutation,
        }

    def _sync_metrics(self, metrics: Dict[str, Any]):
        self.fac_x = metrics["fac_x"]
        self.fac_y = metrics["fac_y"]
        self.fac_b = metrics["fac_b"]
        self.fac_h = metrics["fac_h"]
        self.fac_aspect_ratio = metrics["fac_aspect_ratio"]
        self.lower_bounds = metrics["lower_bounds"]
        self.upper_bounds = metrics["upper_bounds"]
        self.infeasible_mask = metrics["infeasible_mask"]
        self.D = metrics["D"]
        self.TM = metrics["TM"]
        self.MHC = float(metrics["mhc"])
        self.CR = float(metrics["cr"])
        self.fitness = float(metrics["cost"])
        self.current_d_inf = 0
        self.current_is_feasible = True
        self.current_v_worst = None

    def _evaluate_current_layout(self, snapshot_best: bool = True) -> Dict[str, Any]:
        metrics = self.evaluate_fbs_model(self.fbs_model)
        self._sync_metrics(metrics)
        self.feasible_solution_count += 1
        self.best_MHC = min(self.best_MHC, self.MHC)
        if self.fitness < self.best_feasible_cost:
            self.best_feasible_cost = float(self.fitness)
            self.best_fitness = float(self.fitness)
            if snapshot_best:
                self.best_feasible_solution = copy.deepcopy(self.fbs_model)
        self.worst_feasible_cost = float(self.fitness) if self.worst_feasible_cost is None else max(
            float(self.worst_feasible_cost),
            float(self.fitness),
        )
        return metrics

    def reset(self, seed=None, options=None, fbs_model: FBSModel = None):
        if seed is not None:
            np.random.seed(int(seed))
        if options is not None and "fbs_model" in options:
            fbs_model = options["fbs_model"]
        self._clear_runtime_tracking()
        self.fbs_model = copy.deepcopy(fbs_model) if fbs_model is not None else self._make_initial_model()
        self.fbs_model.permutation = self._normalize_permutation(self.fbs_model.permutation).tolist()
        self.fbs_model.bay = self._fixed_bay.copy().tolist()
        self._evaluate_current_layout(snapshot_best=True)
        self.previous_fitness = self.fitness
        self.previous_MHC = self.MHC
        self.previous_CR = self.CR
        self.state = self.constructState()
        info = {
            "fitness": self.fitness,
            "mhc": self.MHC,
            "cr": self.CR,
            "d_inf": 0,
            "is_feasible": True,
        }
        return self.state, info

    def _swap_two(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy()
        i, j = np.random.choice(self.n, size=2, replace=False)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def _swap_rows(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy()
        r1, r2 = np.random.choice(self.m, size=2, replace=False)
        s1, e1 = r1 * self.c, (r1 + 1) * self.c
        s2, e2 = r2 * self.c, (r2 + 1) * self.c
        row1 = new_perm[s1:e1].copy()
        row2 = new_perm[s2:e2].copy()
        new_perm[s1:e1] = row2
        new_perm[s2:e2] = row1
        return new_perm

    def _shuffle_row(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy()
        row = int(np.random.randint(0, self.m))
        start = row * self.c
        end = (row + 1) * self.c
        block = new_perm[start:end].copy()
        np.random.shuffle(block)
        new_perm[start:end] = block
        return new_perm

    def _insert_facility(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy().tolist()
        src, dst = np.random.choice(self.n, size=2, replace=False)
        value = new_perm.pop(int(src))
        new_perm.insert(int(dst), value)
        return np.asarray(new_perm, dtype=int)

    def _segment_insert(self, permutation: np.ndarray, segment_len: int = 2) -> np.ndarray:
        if self.n <= segment_len:
            return permutation.copy()
        new_perm = permutation.copy().tolist()
        start = int(np.random.randint(0, self.n - segment_len + 1))
        segment = new_perm[start : start + segment_len]
        del new_perm[start : start + segment_len]
        insert_at = int(np.random.randint(0, len(new_perm) + 1))
        new_perm[insert_at:insert_at] = segment
        return np.asarray(new_perm, dtype=int)

    def _flow_guided_swap(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy()
        combined = np.maximum(self.WMHC, 0.0)
        pair_index = np.unravel_index(int(np.argmax(np.triu(combined, k=1))), combined.shape)
        u, v = int(pair_index[0] + 1), int(pair_index[1] + 1)
        pos_u = int(np.where(new_perm == u)[0][0])
        pos_v = int(np.where(new_perm == v)[0][0])
        if abs(pos_u - pos_v) <= 1:
            return self._swap_two(permutation)
        target_pos = pos_v - 1 if pos_u < pos_v else pos_v + 1
        target_pos = int(np.clip(target_pos, 0, self.n - 1))
        new_perm[pos_u], new_perm[target_pos] = new_perm[target_pos], new_perm[pos_u]
        return new_perm

    def _cross_row_relocate(self, permutation: np.ndarray) -> np.ndarray:
        new_perm = permutation.copy().tolist()
        row_from, row_to = np.random.choice(self.m, size=2, replace=False)
        src = row_from * self.c + int(np.random.randint(0, self.c))
        dst = row_to * self.c + int(np.random.randint(0, self.c))
        value = new_perm.pop(int(src))
        new_perm.insert(int(dst), value)
        return np.asarray(new_perm[: self.n], dtype=int)

    def _apply_action(self, action_name: str):
        permutation = self._normalize_permutation(self.fbs_model.permutation)
        if action_name == "facility_swap":
            new_perm = self._swap_two(permutation)
        elif action_name == "bay_flip":
            new_perm = self._shuffle_row(permutation)
        elif action_name == "bay_swap":
            new_perm = self._swap_rows(permutation)
        elif action_name == "repair":
            new_perm = permutation.copy()
        elif action_name == "facility_insert":
            new_perm = self._insert_facility(permutation)
        elif action_name == "bay_shuffle":
            new_perm = self._shuffle_row(permutation)
        elif action_name == "facility_shuffle":
            new_perm = permutation.copy()
            np.random.shuffle(new_perm)
        elif action_name == "flow_guided_swap":
            new_perm = self._flow_guided_swap(permutation)
        elif action_name == "segment_insert":
            new_perm = self._segment_insert(permutation, segment_len=2)
        elif action_name == "cross_bay_relocate":
            new_perm = self._cross_row_relocate(permutation)
        elif action_name == "bay_split_by_flow":
            new_perm = self._segment_insert(permutation, segment_len=3 if self.n >= 3 else 2)
        elif action_name == "bay_merge_by_flow":
            new_perm = self._swap_rows(permutation)
        elif action_name == "adjacent_bay_repartition_by_flow":
            new_perm = self._cross_row_relocate(permutation)
        elif action_name == "adjacent_bay_block_repartition_by_flow":
            new_perm = self._segment_insert(permutation, segment_len=min(3, self.n))
        elif action_name == "ga_action":
            new_perm = permutation.copy()
            np.random.shuffle(new_perm)
        elif action_name == "idle":
            return
        else:
            raise ValueError(f"Invalid action: {action_name}")

        self.fbs_model.permutation = self._normalize_permutation(new_perm).tolist()
        self.fbs_model.bay = self._fixed_bay.copy().tolist()

    def calculate_reward(self, previous_mhc, previous_cr):
        mhc_gain = (float(previous_mhc) - float(self.MHC)) / max(abs(float(previous_mhc)), 1.0)
        cr_gain = (float(previous_cr) - float(self.CR)) / max(abs(float(previous_cr)), 1.0)
        reward = 0.5 * mhc_gain + 0.5 * cr_gain
        return float(np.clip(reward, -2.5, 2.5))

    def _build_info(self, reward: float, action_name: str) -> Dict[str, Any]:
        return {
            "TimeLimit.truncated": False,
            "current_fitness": self.fitness,
            "previous_fitness": self.previous_fitness,
            "reward": reward,
            "facility_count": self.n,
            "action_taken": action_name,
            "layout_dimensions": (self.H, self.W),
            "instance": self.instance,
            "mhc": self.MHC,
            "cr": self.CR,
            "d_inf": 0,
            "is_feasible": True,
            "v_worst": None,
            "best_feasible_cost": self.best_feasible_cost,
            "worst_feasible_cost": self.worst_feasible_cost,
        }

    def step(self, action):
        action_name = self.actions[int(action)]
        previous_fitness = self.fitness
        previous_mhc = self.MHC
        previous_cr = self.CR
        self._apply_action(action_name)
        self._evaluate_current_layout(snapshot_best=True)
        self.state = self.constructState()
        reward = self.calculate_reward(previous_mhc, previous_cr)
        self.previous_fitness = previous_fitness
        self.previous_MHC = previous_mhc
        self.previous_CR = previous_cr
        info = self._build_info(reward=reward, action_name=action_name)
        return self.state, reward, False, False, info

    def step2(self, action):
        return self.step(action)

    def constructState(self):
        state = np.zeros((self.n, 3), dtype=np.float64)
        permutation = np.asarray(self.fbs_model.permutation, dtype=float)
        tm = np.asarray(self.TM, dtype=float)
        sources = np.sum(tm, axis=1)
        sinks = np.sum(tm, axis=0)

        def scale_to_255(values):
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                return values
            value_min = np.min(values)
            value_max = np.max(values)
            if np.isclose(value_min, value_max):
                return np.zeros_like(values, dtype=np.float64)
            return ((values - value_min) / (value_max - value_min) * 255.0).astype(np.float64)

        state[:, 0] = scale_to_255(permutation)
        state[:, 1] = scale_to_255(sources)
        state[:, 2] = scale_to_255(sinks)
        return state.flatten()
