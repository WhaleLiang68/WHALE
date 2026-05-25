import copy
import math
from typing import Sequence

import numpy as np


class MO_FBSUtil_BiMO4:
    """双目标 ELP 工具：MHC 最小化，CR 最大化。"""

    OBJECTIVE_DEFINITION_VERSION = "ELP_DRL_BiMO4_paper_v1"
    TCHEBYCHEFF_LINEAR_EPS = 0.05

    @staticmethod
    def _as_float_vector(values, *, minimum_size=None, fill_value=0.0):
        vector = np.asarray(values, dtype=float).reshape(-1)
        if minimum_size is not None and vector.size < int(minimum_size):
            padding = np.full(int(minimum_size) - vector.size, float(fill_value), dtype=float)
            vector = np.concatenate([vector, padding])
        return vector

    @staticmethod
    def calculate_overlap_length(min1, max1, min2, max2):
        return float(max(0.0, min(max1, max2) - max(min1, min2)))

    @staticmethod
    def get_boundary_adjacency_matrix(fac_x, fac_y, fac_b, fac_h, n, tolerance=1e-3):
        fac_x = np.asarray(fac_x, dtype=float).reshape(int(n))
        fac_y = np.asarray(fac_y, dtype=float).reshape(int(n))
        fac_b = np.asarray(fac_b, dtype=float).reshape(int(n))
        fac_h = np.asarray(fac_h, dtype=float).reshape(int(n))

        xi_min = fac_x - fac_b / 2.0
        xi_max = fac_x + fac_b / 2.0
        yi_min = fac_y - fac_h / 2.0
        yi_max = fac_y + fac_h / 2.0

        overlap_x = np.maximum(
            0.0,
            np.minimum(xi_max[:, None], xi_max[None, :]) - np.maximum(xi_min[:, None], xi_min[None, :]),
        )
        overlap_y = np.maximum(
            0.0,
            np.minimum(yi_max[:, None], yi_max[None, :]) - np.maximum(yi_min[:, None], yi_min[None, :]),
        )
        gap_x = np.abs(fac_x[:, None] - fac_x[None, :]) - (fac_b[:, None] + fac_b[None, :]) / 2.0
        gap_y = np.abs(fac_y[:, None] - fac_y[None, :]) - (fac_h[:, None] + fac_h[None, :]) / 2.0

        # 论文口径：只要共享边界即计分，不再按接触边长度放大。
        touch_vertical = (np.abs(gap_x) <= tolerance) & (overlap_y > tolerance)
        touch_horizontal = (np.abs(gap_y) <= tolerance) & (overlap_x > tolerance)
        adjacency = np.logical_or(touch_vertical, touch_horizontal).astype(float)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    @staticmethod
    def calculate_total_constraint_violation(fac_b, fac_h, lower_bounds, upper_bounds):
        widths = MO_FBSUtil_BiMO4._as_float_vector(fac_b)
        heights = MO_FBSUtil_BiMO4._as_float_vector(fac_h)
        short_side = np.minimum(widths, heights)
        long_side = np.maximum(widths, heights)
        lower = MO_FBSUtil_BiMO4._as_float_vector(lower_bounds, minimum_size=short_side.size, fill_value=0.0)
        upper = MO_FBSUtil_BiMO4._as_float_vector(upper_bounds, minimum_size=long_side.size, fill_value=np.inf)
        short_violation = np.maximum(lower - short_side, 0.0)
        long_violation = np.maximum(long_side - upper, 0.0)
        return float(np.sum(short_violation + long_violation))

    @staticmethod
    def calculate_objectives(
        fac_x,
        fac_y,
        fac_b,
        fac_h,
        mhc,
        n,
        rel_matrix=None,
    ):
        f1_mhc = float(mhc)
        f2_cr = 0.0
        if rel_matrix is not None:
            adjacency_matrix = MO_FBSUtil_BiMO4.get_boundary_adjacency_matrix(fac_x, fac_y, fac_b, fac_h, n)
            relation_matrix = np.asarray(rel_matrix, dtype=float)
            f2_cr = float(np.sum(np.triu(relation_matrix * adjacency_matrix, k=1)))
        return np.asarray([f1_mhc, f2_cr], dtype=float)

    @staticmethod
    def to_minimization(objectives_raw):
        objectives = MO_FBSUtil_BiMO4._as_float_vector(objectives_raw, minimum_size=2, fill_value=0.0)
        return np.asarray([objectives[0], -objectives[1]], dtype=float)

    @staticmethod
    def from_minimization(objectives_min):
        objectives = MO_FBSUtil_BiMO4._as_float_vector(objectives_min, minimum_size=2, fill_value=0.0)
        return np.asarray([objectives[0], -objectives[1]], dtype=float)

    @staticmethod
    def _normalize_weights(weights, size):
        weights = np.asarray([0.5, 0.5] if weights is None else weights, dtype=float).reshape(-1)
        if weights.size < int(size):
            weights = np.pad(weights, (0, int(size) - weights.size), constant_values=0.0)
        weights = np.clip(weights[: int(size)], 0.0, None)
        if not np.any(weights > 0):
            weights = np.full(int(size), 1.0 / float(max(1, int(size))), dtype=float)
        return weights / np.sum(weights)

    @staticmethod
    def _tchebycheff_score(normalized, weights, linear_eps=None):
        normalized = MO_FBSUtil_BiMO4._as_float_vector(normalized, minimum_size=2, fill_value=0.0)[:2]
        weights = MO_FBSUtil_BiMO4._normalize_weights(weights, normalized.size)
        weighted = normalized * weights
        linear_eps = (
            MO_FBSUtil_BiMO4.TCHEBYCHEFF_LINEAR_EPS
            if linear_eps is None
            else float(max(linear_eps, 0.0))
        )
        return float(np.max(weighted) + linear_eps * np.sum(weighted))

    @staticmethod
    def normalize_with_running_bounds(objectives_min, running_min=None, running_max=None):
        vector = MO_FBSUtil_BiMO4._as_float_vector(objectives_min, minimum_size=2, fill_value=0.0)[:2]
        if running_min is None or running_max is None:
            return None
        running_min = MO_FBSUtil_BiMO4._as_float_vector(running_min, minimum_size=2, fill_value=0.0)[:2]
        running_max = MO_FBSUtil_BiMO4._as_float_vector(running_max, minimum_size=2, fill_value=0.0)[:2]
        if not np.all(np.isfinite(running_min)) or not np.all(np.isfinite(running_max)):
            return None
        span = np.maximum(running_max - running_min, 1e-12)
        normalized = (vector - running_min) / span
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def normalize_objective_vector(objectives_min, ideal=None, nadir=None):
        vector = MO_FBSUtil_BiMO4._as_float_vector(objectives_min, minimum_size=2, fill_value=0.0)[:2]
        if ideal is None or nadir is None:
            return None
        ideal = MO_FBSUtil_BiMO4._as_float_vector(ideal, minimum_size=2, fill_value=0.0)[:2]
        nadir = MO_FBSUtil_BiMO4._as_float_vector(nadir, minimum_size=2, fill_value=0.0)[:2]
        # 当档案只有一个点或某一维无跨度时，不能把该维强行归一成 0，
        # 否则代表解与搜索能量会退化为恒定值，破坏接受准则与多目标排序。
        if np.allclose(ideal, nadir, atol=1e-12, rtol=1e-9):
            return None
        span = np.maximum(nadir - ideal, 1e-12)
        normalized = (vector - ideal) / span
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, 0.0, None)

    @staticmethod
    def decision_score(objectives_min, ideal=None, nadir=None, weights=None, running_min=None, running_max=None):
        normalized = MO_FBSUtil_BiMO4.normalize_objective_vector(objectives_min, ideal=ideal, nadir=nadir)
        if normalized is None:
            normalized = MO_FBSUtil_BiMO4.normalize_with_running_bounds(
                objectives_min,
                running_min=running_min,
                running_max=running_max,
            )
        if normalized is None:
            raw_vector = MO_FBSUtil_BiMO4._as_float_vector(objectives_min, minimum_size=2, fill_value=0.0)[:2]
            normalized = np.clip(raw_vector, 0.0, None)
        return MO_FBSUtil_BiMO4._tchebycheff_score(normalized, weights)

    @staticmethod
    def surrogate_energy(objectives_min, ideal=None, nadir=None, weights=None, running_min=None, running_max=None):
        normalized = MO_FBSUtil_BiMO4.normalize_objective_vector(objectives_min, ideal=ideal, nadir=nadir)
        if normalized is None:
            normalized = MO_FBSUtil_BiMO4.normalize_with_running_bounds(
                objectives_min,
                running_min=running_min,
                running_max=running_max,
            )
        if normalized is None:
            raw_vector = MO_FBSUtil_BiMO4._as_float_vector(objectives_min, minimum_size=2, fill_value=0.0)[:2]
            normalized = np.clip(raw_vector, 0.0, None)
        return MO_FBSUtil_BiMO4._tchebycheff_score(normalized, weights)

    @staticmethod
    def search_energy(
        objectives_min,
        *,
        is_feasible,
        d_inf,
        total_violation,
        ideal=None,
        nadir=None,
        weights=None,
        running_min=None,
        running_max=None,
    ):
        base_energy = MO_FBSUtil_BiMO4.surrogate_energy(
            objectives_min,
            ideal=ideal,
            nadir=nadir,
            weights=weights,
            running_min=running_min,
            running_max=running_max,
        )
        if is_feasible:
            return float(base_energy)
        return float(1_000_000.0 + 10_000.0 * max(int(d_inf), 0) + max(float(total_violation), 0.0) + base_energy)

    @staticmethod
    def pareto_dominates(left_objectives, right_objectives, atol=1e-9):
        left = MO_FBSUtil_BiMO4._as_float_vector(left_objectives, minimum_size=2, fill_value=0.0)[:2]
        right = MO_FBSUtil_BiMO4._as_float_vector(right_objectives, minimum_size=2, fill_value=0.0)[:2]
        return bool(np.all(left <= right + atol) and np.any(left < right - atol))

    @staticmethod
    def _value(entity, name, default=None):
        if isinstance(entity, dict):
            return entity.get(name, default)
        return getattr(entity, name, default)

    @staticmethod
    def constraint_signature(entity):
        is_feasible = bool(MO_FBSUtil_BiMO4._value(entity, "current_is_feasible", False))
        d_inf = int(MO_FBSUtil_BiMO4._value(entity, "current_d_inf", 0) or 0)
        total_violation = MO_FBSUtil_BiMO4._value(entity, "constraint_violation", None)
        if total_violation is None:
            total_violation = MO_FBSUtil_BiMO4.calculate_total_constraint_violation(
                MO_FBSUtil_BiMO4._value(entity, "fac_b", []),
                MO_FBSUtil_BiMO4._value(entity, "fac_h", []),
                MO_FBSUtil_BiMO4._value(entity, "lower_bounds", []),
                MO_FBSUtil_BiMO4._value(entity, "upper_bounds", []),
            )
        return is_feasible, d_inf, float(total_violation)

    @staticmethod
    def compare_solution_quality(left, right, atol=1e-9):
        left_feasible, left_d_inf, left_violation = MO_FBSUtil_BiMO4.constraint_signature(left)
        right_feasible, right_d_inf, right_violation = MO_FBSUtil_BiMO4.constraint_signature(right)

        if left_feasible and not right_feasible:
            return -1
        if right_feasible and not left_feasible:
            return 1

        if not left_feasible and not right_feasible:
            left_key = (left_d_inf, left_violation)
            right_key = (right_d_inf, right_violation)
            if left_key < right_key:
                return -1
            if right_key < left_key:
                return 1
            return 0

        left_objectives = MO_FBSUtil_BiMO4._value(left, "mo_objectives_min", None)
        right_objectives = MO_FBSUtil_BiMO4._value(right, "mo_objectives_min", None)
        if left_objectives is None or right_objectives is None:
            left_energy = float(MO_FBSUtil_BiMO4._value(left, "fitness", math.inf))
            right_energy = float(MO_FBSUtil_BiMO4._value(right, "fitness", math.inf))
            if left_energy < right_energy - atol:
                return -1
            if right_energy < left_energy - atol:
                return 1
            return 0

        left_dominates = MO_FBSUtil_BiMO4.pareto_dominates(left_objectives, right_objectives, atol=atol)
        right_dominates = MO_FBSUtil_BiMO4.pareto_dominates(right_objectives, left_objectives, atol=atol)
        if left_dominates and not right_dominates:
            return -1
        if right_dominates and not left_dominates:
            return 1
        return 0

    @staticmethod
    def compute_ideal_nadir(candidates: Sequence):
        objective_matrix = []
        for candidate in candidates:
            objectives = MO_FBSUtil_BiMO4._value(candidate, "mo_objectives_min", None)
            if objectives is None:
                continue
            objective_matrix.append(MO_FBSUtil_BiMO4._as_float_vector(objectives, minimum_size=2, fill_value=0.0)[:2])
        if not objective_matrix:
            return None, None
        matrix = np.asarray(objective_matrix, dtype=float)
        return np.min(matrix, axis=0), np.max(matrix, axis=0)

    @staticmethod
    def _normalized_archive_matrix(candidates: Sequence, ideal=None, nadir=None):
        vectors = []
        for candidate in candidates:
            objectives = MO_FBSUtil_BiMO4._value(candidate, "mo_objectives_min", None)
            if objectives is None:
                continue
            vectors.append(MO_FBSUtil_BiMO4._as_float_vector(objectives, minimum_size=2, fill_value=0.0)[:2])
        if not vectors:
            return np.empty((0, 2), dtype=float), ideal, nadir

        matrix = np.asarray(vectors, dtype=float)
        if ideal is None or nadir is None:
            ideal = np.min(matrix, axis=0)
            nadir = np.max(matrix, axis=0)
        span = np.maximum(np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float), 1e-12)
        normalized = (matrix - np.asarray(ideal, dtype=float)) / span
        normalized[~np.isfinite(normalized)] = 0.0
        normalized = np.clip(normalized, 0.0, None)
        return normalized, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float)

    @staticmethod
    def archive_hypervolume(candidates: Sequence, ideal=None, nadir=None, reference_margin=0.1):
        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(candidates, ideal=ideal, nadir=nadir)
        if normalized.size == 0:
            return 0.0
        reference = np.max(normalized, axis=0) + max(float(reference_margin), 1e-9)
        points = normalized[np.argsort(normalized[:, 0])]
        hv = 0.0
        best_y = float(reference[1])
        for point in points:
            x = float(point[0])
            y = float(point[1])
            if y < best_y - 1e-12:
                hv += max(float(reference[0]) - x, 0.0) * max(best_y - y, 0.0)
                best_y = y
        return float(hv)

    @staticmethod
    def archive_spacing(candidates: Sequence, ideal=None, nadir=None):
        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(candidates, ideal=ideal, nadir=nadir)
        count = int(normalized.shape[0])
        if count <= 1:
            return 0.0
        distances = []
        for idx in range(count):
            delta = normalized - normalized[idx]
            norms = np.linalg.norm(delta, axis=1)
            norms[idx] = np.inf
            nearest = float(np.min(norms))
            if np.isfinite(nearest):
                distances.append(nearest)
        if not distances:
            return 0.0
        distances = np.asarray(distances, dtype=float)
        mean_distance = float(np.mean(distances))
        if mean_distance <= 1e-12:
            return 0.0
        return float(np.sqrt(np.mean((distances - mean_distance) ** 2)))

    @staticmethod
    def _duplicate_objectives(left, right, atol=1e-9):
        left_objectives = MO_FBSUtil_BiMO4._value(left, "mo_objectives_min", None)
        right_objectives = MO_FBSUtil_BiMO4._value(right, "mo_objectives_min", None)
        if left_objectives is None or right_objectives is None:
            return False
        return bool(np.allclose(left_objectives, right_objectives, atol=atol, rtol=1e-7))

    @staticmethod
    def select_nfcs_subset(candidates: Sequence, max_size: int, ideal=None, nadir=None):
        candidates = list(candidates)
        if len(candidates) <= int(max_size):
            return candidates

        normalized_vectors, ideal, nadir = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            candidates,
            ideal=ideal,
            nadir=nadir,
        )
        selected_indices = set()
        for column in range(normalized_vectors.shape[1]):
            selected_indices.add(int(np.argmin(normalized_vectors[:, column])))
            if len(selected_indices) >= int(max_size):
                break
        if not selected_indices:
            selected_indices.add(0)

        while len(selected_indices) < int(max_size):
            remaining_indices = [idx for idx in range(len(candidates)) if idx not in selected_indices]
            if not remaining_indices:
                break
            best_idx = None
            best_distance = -math.inf
            best_score = math.inf
            for idx in remaining_indices:
                current_vector = normalized_vectors[idx]
                min_distance = min(
                    float(np.linalg.norm(current_vector - normalized_vectors[selected_idx], ord=2))
                    for selected_idx in selected_indices
                )
                score = MO_FBSUtil_BiMO4.decision_score(
                    MO_FBSUtil_BiMO4._value(candidates[idx], "mo_objectives_min", None),
                    ideal=ideal,
                    nadir=nadir,
                )
                if min_distance > best_distance + 1e-12 or (
                    abs(min_distance - best_distance) <= 1e-12 and score < best_score
                ):
                    best_idx = idx
                    best_distance = min_distance
                    best_score = score
            if best_idx is None:
                break
            selected_indices.add(int(best_idx))

        return [candidates[idx] for idx in sorted(selected_indices)]

    @staticmethod
    def _compute_knee_distances(candidates: Sequence, ideal=None, nadir=None):
        """计算二维 Pareto 档案中每个点相对两端极值连线的膝点距离。"""
        normalized, ideal, nadir = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            candidates,
            ideal=ideal,
            nadir=nadir,
        )
        count = int(normalized.shape[0])
        if count == 0:
            return np.asarray([], dtype=float), ideal, nadir
        if count <= 2:
            return np.zeros(count, dtype=float), ideal, nadir

        mhc_extreme_idx = int(np.argmin(normalized[:, 0]))
        cr_extreme_idx = int(np.argmin(normalized[:, 1]))
        left = normalized[mhc_extreme_idx]
        right = normalized[cr_extreme_idx]
        baseline = right - left
        baseline_norm = float(np.linalg.norm(baseline))
        if baseline_norm <= 1e-12:
            return np.zeros(count, dtype=float), ideal, nadir

        distances = np.zeros(count, dtype=float)
        for idx in range(count):
            vector = normalized[idx] - left
            cross_value = abs(float(baseline[0] * vector[1] - baseline[1] * vector[0]))
            distances[idx] = cross_value / baseline_norm
        distances[mhc_extreme_idx] = 0.0
        distances[cr_extreme_idx] = 0.0
        return distances, ideal, nadir

    @staticmethod
    def select_representative_solution(candidates: Sequence, ideal=None, nadir=None, weights=None):
        feasible_candidates = [candidate for candidate in candidates if MO_FBSUtil_BiMO4.constraint_signature(candidate)[0]]
        if not feasible_candidates:
            return None, math.inf, None
        if ideal is None or nadir is None:
            ideal, nadir = MO_FBSUtil_BiMO4.compute_ideal_nadir(feasible_candidates)

        best_candidate = None
        best_index = None
        best_key = None
        for idx, candidate in enumerate(feasible_candidates):
            objectives_min = MO_FBSUtil_BiMO4._value(candidate, "mo_objectives_min", None)
            score = MO_FBSUtil_BiMO4.decision_score(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            surrogate = MO_FBSUtil_BiMO4.surrogate_energy(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            mhc = float(MO_FBSUtil_BiMO4._value(candidate, "MHC", math.inf))
            key = (score, surrogate, mhc, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
                best_index = idx
        return best_candidate, float(best_key[0]), int(best_index)

    @staticmethod
    def select_knee_solution(candidates: Sequence, ideal=None, nadir=None, weights=None):
        feasible_candidates = [candidate for candidate in candidates if MO_FBSUtil_BiMO4.constraint_signature(candidate)[0]]
        if not feasible_candidates:
            return None, math.inf, None
        knee_distances, ideal, nadir = MO_FBSUtil_BiMO4._compute_knee_distances(
            feasible_candidates,
            ideal=ideal,
            nadir=nadir,
        )

        best_candidate = None
        best_index = None
        best_key = None
        best_score = math.inf
        for idx, candidate in enumerate(feasible_candidates):
            objectives_min = MO_FBSUtil_BiMO4._value(candidate, "mo_objectives_min", None)
            score = MO_FBSUtil_BiMO4.decision_score(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            surrogate = MO_FBSUtil_BiMO4.surrogate_energy(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            mhc = float(MO_FBSUtil_BiMO4._value(candidate, "MHC", math.inf))
            knee_distance = float(knee_distances[idx]) if idx < len(knee_distances) else 0.0
            key = (-knee_distance, score, surrogate, mhc, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
                best_index = idx
                best_score = float(score)
        return best_candidate, float(best_score), int(best_index)

    @staticmethod
    def update_pareto_archive(
        candidates: Sequence,
        candidate,
        max_size=None,
        clone_fn=None,
        atol=1e-9,
        **_,
    ):
        archive = [item for item in candidates if MO_FBSUtil_BiMO4.constraint_signature(item)[0]]
        if candidate is None or not MO_FBSUtil_BiMO4.constraint_signature(candidate)[0]:
            return archive, False, 0

        clone = copy.deepcopy if clone_fn is None else clone_fn
        kept = []
        removed = 0
        for existing in archive:
            comparison = MO_FBSUtil_BiMO4.compare_solution_quality(candidate, existing, atol=atol)
            if comparison > 0:
                return archive, False, 0
            if comparison < 0:
                removed += 1
                continue
            if MO_FBSUtil_BiMO4._duplicate_objectives(candidate, existing, atol=atol):
                return archive, False, 0
            kept.append(existing)

        kept.append(clone(candidate))
        if max_size is not None and len(kept) > int(max_size):
            trim_ideal, trim_nadir = MO_FBSUtil_BiMO4.compute_ideal_nadir(kept)
            kept = MO_FBSUtil_BiMO4.select_nfcs_subset(kept, int(max_size), ideal=trim_ideal, nadir=trim_nadir)
        return kept, True, removed
