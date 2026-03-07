import numpy as np
import copy
from dataclasses import dataclass

@dataclass
class Solution:
    model: object # FBSModel
    objectives: list # [MHC, CR, DR, AR]
    energy: float

class ParetoArchive:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.solutions = [] # 存储 Solution 对象

    def _dominates(self, obj_a, obj_b):
        """
        判断解 A 是否支配解 B
        目标方向: [Min, Max, Max, Max]
        """
        # 转换为统一的最小化问题: [Min, -Max, -Max, -Max]
        a_norm = [obj_a[0], -obj_a[1], -obj_a[2], -obj_a[3]]
        b_norm = [obj_b[0], -obj_b[1], -obj_b[2], -obj_b[3]]

        better_in_any = False
        for a, b in zip(a_norm, b_norm):
            if a > b: return False # A在某方面比B差
            if a < b: better_in_any = True

        return better_in_any

    def update(self, new_solution):
        """更新档案，返回 (是否加入, 奖励值)"""

        # 1. 重复检测 (非常重要！防止档案被同质化)
        for sol in self.solutions:
            # 如果目标值完全一样，视为重复解，不加入
            if np.allclose(sol.objectives, new_solution.objectives, rtol=1e-5):
                return False, 0

        to_remove = []
        is_dominated = False

        # 2. 支配检查
        for sol in self.solutions:
            if self._dominates(sol.objectives, new_solution.objectives):
                is_dominated = True
                break
            if self._dominates(new_solution.objectives, sol.objectives):
                to_remove.append(sol)

        if is_dominated:
            return False, -1

            # 移除被支配解
        for sol in to_remove:
            self.solutions.remove(sol)

        self.solutions.append(new_solution)

        # 3. NFCS 剪枝
        if len(self.solutions) > self.capacity:
            self._nfcs_pruning()

        # 奖励机制：移除别人得大分，单纯加入得小分
        return True, (10 if len(to_remove) > 0 else 2)

    def _nfcs_pruning(self):
        """
        基于最近最远候选解法 (NFCS) 进行剪枝
        保留边界解 + 距离当前集合最远的解
        """
        # 1. 归一化目标值 (公式 30)
        objs = np.array([s.objectives for s in self.solutions])
        # 转换 Max 目标为 Min 方便处理: [Min, 1/Max, 1/Max, 1/Max] 或者直接 Min-Max 归一化
        # 这里使用简单的 Min-Max 归一化
        min_vals = np.min(objs, axis=0)
        max_vals = np.max(objs, axis=0)
        range_vals = max_vals - min_vals + 1e-6
        norm_objs = (objs - min_vals) / range_vals

        keep_indices = []

        # 2. 极值优先保留 (每个目标的最好解)
        # 目标方向: [Min, Max, Max, Max]
        best_indices = [
            np.argmin(objs[:, 0]), # MHC Min
            np.argmax(objs[:, 1]), # CR Max
            np.argmax(objs[:, 2]), # DR Max
            np.argmax(objs[:, 3])  # AR Max
        ]
        keep_indices.extend(best_indices)
        keep_indices = list(set(keep_indices)) # 去重

        # 3. 迭代选择最大-最小距离的解
        while len(keep_indices) < self.capacity:
            candidates = [i for i in range(len(self.solutions)) if i not in keep_indices]
            if not candidates: break

            max_min_dist = -1
            best_candidate = -1

            for c_idx in candidates:
                # 计算 candidate 到 current_set 的最小距离
                current_set_norm = norm_objs[keep_indices]
                c_norm = norm_objs[c_idx]

                # 欧氏距离
                dists = np.sqrt(np.sum((current_set_norm - c_norm)**2, axis=1))
                min_dist = np.min(dists)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = c_idx

            if best_candidate != -1:
                keep_indices.append(best_candidate)

        # 更新档案
        self.solutions = [self.solutions[i] for i in keep_indices]

    def calculate_diversity_reward(self, new_solution):
        """
        公式 33: 多样性奖励 R_div
        R_div = tanh(min_distance)
        """
        if not self.solutions: return 1.0

        # 简单计算归一化距离
        # 注意：这里需要实时归一化，为简化计算，使用最近一次的统计值
        objs = np.array([s.objectives for s in self.solutions])
        new_obj = np.array(new_solution.objectives)

        min_vals = np.min(objs, axis=0)
        max_vals = np.max(objs, axis=0)
        # 动态扩展范围以包含新解
        min_vals = np.minimum(min_vals, new_obj)
        max_vals = np.maximum(max_vals, new_obj)

        range_vals = max_vals - min_vals + 1e-6

        norm_archive = (objs - min_vals) / range_vals
        norm_new = (new_obj - min_vals) / range_vals

        dists = np.sqrt(np.sum((norm_archive - norm_new)**2, axis=1))
        min_dist = np.min(dists)

        return np.tanh(min_dist)