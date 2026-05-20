import copy
import math
from typing import Iterable, Sequence

import numpy as np


class MO_FBSUtil:
    """Utilities for multi-objective evaluation and Pareto archive management."""

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
    def _get_adjacency_length_loop(fac_x, fac_y, fac_b, fac_h, n):
        adjacency_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                xi_min, xi_max = fac_x[i] - fac_b[i] / 2.0, fac_x[i] + fac_b[i] / 2.0
                yi_min, yi_max = fac_y[i] - fac_h[i] / 2.0, fac_y[i] + fac_h[i] / 2.0
                xj_min, xj_max = fac_x[j] - fac_b[j] / 2.0, fac_x[j] + fac_b[j] / 2.0
                yj_min, yj_max = fac_y[j] - fac_h[j] / 2.0, fac_y[j] + fac_h[j] / 2.0

                dist_x = max(0.0, abs(fac_x[i] - fac_x[j]) - (fac_b[i] + fac_b[j]) / 2.0)
                dist_y = max(0.0, abs(fac_y[i] - fac_y[j]) - (fac_h[i] + fac_h[j]) / 2.0)
                contact_y = MO_FBSUtil.calculate_overlap_length(yi_min, yi_max, yj_min, yj_max)
                contact_x = MO_FBSUtil.calculate_overlap_length(xi_min, xi_max, xj_min, xj_max)

                total_contact = 0.0
                tolerance = 1e-3
                if dist_x < tolerance:
                    total_contact += contact_y
                if dist_y < tolerance:
                    total_contact += contact_x

                adjacency_matrix[i, j] = total_contact
                adjacency_matrix[j, i] = total_contact
        return adjacency_matrix

    @staticmethod
    def _get_adjacency_length_vectorized(fac_x, fac_y, fac_b, fac_h, n):
        fac_x = np.asarray(fac_x, dtype=float).reshape(int(n))
        fac_y = np.asarray(fac_y, dtype=float).reshape(int(n))
        fac_b = np.asarray(fac_b, dtype=float).reshape(int(n))
        fac_h = np.asarray(fac_h, dtype=float).reshape(int(n))

        xi_min = fac_x - fac_b / 2.0
        xi_max = fac_x + fac_b / 2.0
        yi_min = fac_y - fac_h / 2.0
        yi_max = fac_y + fac_h / 2.0

        xj_min = xi_min[None, :]
        xj_max = xi_max[None, :]
        yj_min = yi_min[None, :]
        yj_max = yi_max[None, :]

        xi_min = xi_min[:, None]
        xi_max = xi_max[:, None]
        yi_min = yi_min[:, None]
        yi_max = yi_max[:, None]

        overlap_x = np.maximum(0.0, np.minimum(xi_max, xj_max) - np.maximum(xi_min, xj_min))
        overlap_y = np.maximum(0.0, np.minimum(yi_max, yj_max) - np.maximum(yi_min, yj_min))

        dist_x = np.maximum(0.0, np.abs(fac_x[:, None] - fac_x[None, :]) - (fac_b[:, None] + fac_b[None, :]) / 2.0)
        dist_y = np.maximum(0.0, np.abs(fac_y[:, None] - fac_y[None, :]) - (fac_h[:, None] + fac_h[None, :]) / 2.0)

        tolerance = 1e-3
        adjacency_matrix = np.where(dist_x < tolerance, overlap_y, 0.0) + np.where(dist_y < tolerance, overlap_x, 0.0)
        np.fill_diagonal(adjacency_matrix, 0.0)
        return adjacency_matrix

    @staticmethod
    def get_adjacency_length(fac_x, fac_y, fac_b, fac_h, n):
        return MO_FBSUtil._get_adjacency_length_vectorized(fac_x, fac_y, fac_b, fac_h, n)

    @staticmethod
    def calculate_total_constraint_violation(fac_b, fac_h, lower_bounds, upper_bounds):
        widths = MO_FBSUtil._as_float_vector(fac_b)
        heights = MO_FBSUtil._as_float_vector(fac_h)
        short_side = np.minimum(widths, heights)
        long_side = np.maximum(widths, heights)
        lower = MO_FBSUtil._as_float_vector(lower_bounds, minimum_size=short_side.size, fill_value=0.0)
        upper = MO_FBSUtil._as_float_vector(upper_bounds, minimum_size=long_side.size, fill_value=np.inf)
        short_violation = np.maximum(lower - short_side, 0.0)
        long_violation = np.maximum(long_side - upper, 0.0)
        return float(np.sum(short_violation + long_violation))

    @staticmethod
    def calculate_ar_scores(fac_b, fac_h, aspect_limits=None, optimal_aspect_ratio=1.5):
        widths = MO_FBSUtil._as_float_vector(fac_b)
        heights = MO_FBSUtil._as_float_vector(fac_h, minimum_size=widths.size, fill_value=1.0)
        minimum_side = np.minimum(widths, heights)
        aspect_ratios = np.divide(
            np.maximum(widths, heights),
            minimum_side,
            out=np.full_like(widths, np.inf, dtype=float),
            where=minimum_side > 0,
        )

        if aspect_limits is None:
            aspect_limits = np.full(aspect_ratios.size, 2.5, dtype=float)
        else:
            aspect_limits = MO_FBSUtil._as_float_vector(
                aspect_limits,
                minimum_size=aspect_ratios.size,
                fill_value=2.5,
            )
            if aspect_limits.size > aspect_ratios.size:
                aspect_limits = aspect_limits[: aspect_ratios.size]

        aspect_limits = np.clip(aspect_limits, 1.0 + 1e-12, None)
        lower_limits = np.ones_like(aspect_limits, dtype=float)
        optimal = np.full_like(aspect_limits, float(optimal_aspect_ratio), dtype=float)
        narrow_limit_mask = aspect_limits <= optimal
        if np.any(narrow_limit_mask):
            # 若数据集给出的上界低于论文默认最优值，使用可行区间中点保持满意度函数可定义。
            optimal[narrow_limit_mask] = 0.5 * (lower_limits[narrow_limit_mask] + aspect_limits[narrow_limit_mask])
        optimal = np.clip(optimal, lower_limits + 1e-12, aspect_limits - 1e-12)

        scores = np.zeros_like(aspect_ratios, dtype=float)
        finite_mask = np.isfinite(aspect_ratios)
        left_mask = finite_mask & (aspect_ratios >= lower_limits) & (aspect_ratios <= optimal)
        scores[left_mask] = np.divide(
            aspect_ratios[left_mask] - lower_limits[left_mask],
            optimal[left_mask] - lower_limits[left_mask],
            out=np.zeros_like(aspect_ratios[left_mask], dtype=float),
            where=(optimal[left_mask] - lower_limits[left_mask]) > 1e-12,
        )
        right_mask = finite_mask & (aspect_ratios > optimal) & (aspect_ratios <= aspect_limits)
        scores[right_mask] = np.divide(
            aspect_limits[right_mask] - aspect_ratios[right_mask],
            aspect_limits[right_mask] - optimal[right_mask],
            out=np.zeros_like(aspect_ratios[right_mask], dtype=float),
            where=(aspect_limits[right_mask] - optimal[right_mask]) > 1e-12,
        )
        scores[~np.isfinite(scores)] = 0.0
        return aspect_ratios, np.clip(scores, 0.0, 1.0)

    @staticmethod
    def calculate_objectives(
        fac_x,
        fac_y,
        fac_b,
        fac_h,
        mhc,
        n,
        rel_matrix=None,
        dist_req_matrix=None,
        aspect_limits=None,
        optimal_aspect_ratio=1.5,
    ):
        f1_mhc = float(mhc)

        f2_cr = 0.0
        if rel_matrix is not None:
            adjacency_matrix = MO_FBSUtil.get_adjacency_length(fac_x, fac_y, fac_b, fac_h, n)
            f2_cr = float(np.sum(np.triu(np.asarray(rel_matrix, dtype=float) * adjacency_matrix, k=1)))

        f3_dr = 0.0
        if dist_req_matrix is not None:
            delta_x = np.asarray(fac_x, dtype=float)[:, None] - np.asarray(fac_x, dtype=float)[None, :]
            delta_y = np.asarray(fac_y, dtype=float)[:, None] - np.asarray(fac_y, dtype=float)[None, :]
            distance_matrix = np.sqrt(delta_x**2 + delta_y**2)
            f3_dr = float(np.sum(np.triu(np.asarray(dist_req_matrix, dtype=float) * distance_matrix, k=1)))

        _aspect_ratios, ar_scores = MO_FBSUtil.calculate_ar_scores(
            fac_b,
            fac_h,
            aspect_limits=aspect_limits,
            optimal_aspect_ratio=optimal_aspect_ratio,
        )
        f4_ar = float(np.mean(ar_scores)) if ar_scores.size else 0.0

        return [f1_mhc, f2_cr, f3_dr, f4_ar]

    @staticmethod
    def to_minimization(objectives_raw):
        objectives = MO_FBSUtil._as_float_vector(objectives_raw, minimum_size=4, fill_value=0.0)
        return np.asarray([objectives[0], -objectives[1], -objectives[2], -objectives[3]], dtype=float)

    @staticmethod
    def from_minimization(objectives_min):
        objectives = MO_FBSUtil._as_float_vector(objectives_min, minimum_size=4, fill_value=0.0)
        return np.asarray([objectives[0], -objectives[1], -objectives[2], -objectives[3]], dtype=float)

    @staticmethod
    def aggregated_energy(objectives, weights):
        mhc, cr, dr, ar = MO_FBSUtil._as_float_vector(objectives, minimum_size=4, fill_value=0.0)[:4]
        weights = MO_FBSUtil._as_float_vector(weights, minimum_size=4, fill_value=0.25)[:4]
        if not np.any(weights > 0):
            weights = np.full(4, 0.25, dtype=float)
        weights = weights / np.sum(weights)

        epsilon = 1e-6
        term1 = weights[0] * max(mhc, 0.0)
        term2 = weights[1] * (1.0 / max(cr, epsilon))
        term3 = weights[2] * (1.0 / max(dr, epsilon))
        term4 = weights[3] * (1.0 / max(ar, epsilon))
        return float(term1 + term2 + term3 + term4)

    @staticmethod
    def normalize_objective_vector(objectives_min, ideal=None, nadir=None):
        vector = MO_FBSUtil._as_float_vector(objectives_min, minimum_size=4, fill_value=0.0)[:4]
        if ideal is None or nadir is None:
            return None
        ideal = MO_FBSUtil._as_float_vector(ideal, minimum_size=vector.size, fill_value=0.0)[: vector.size]
        nadir = MO_FBSUtil._as_float_vector(nadir, minimum_size=vector.size, fill_value=0.0)[: vector.size]
        span = np.maximum(nadir - ideal, 1e-12)
        normalized = (vector - ideal) / span
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, 0.0, None)

    @staticmethod
    def decision_score(objectives_min, ideal=None, nadir=None, weights=None):
        normalized = MO_FBSUtil.normalize_objective_vector(objectives_min, ideal=ideal, nadir=nadir)
        if normalized is None:
            raw_objectives = MO_FBSUtil.from_minimization(objectives_min)
            fallback_weights = np.full(4, 0.25, dtype=float) if weights is None else weights
            return MO_FBSUtil.aggregated_energy(raw_objectives, fallback_weights)

        weights = np.full(normalized.size, 1.0 / normalized.size, dtype=float) if weights is None else MO_FBSUtil._as_float_vector(weights, minimum_size=normalized.size, fill_value=1.0)
        weights = np.clip(weights[: normalized.size], 0.0, None)
        if not np.any(weights > 0):
            weights = np.full(normalized.size, 1.0 / normalized.size, dtype=float)
        weights = weights / np.sum(weights)
        return float(np.dot(normalized, weights))

    @staticmethod
    def surrogate_energy(objectives_min, ideal=None, nadir=None, weights=None):
        normalized = MO_FBSUtil.normalize_objective_vector(objectives_min, ideal=ideal, nadir=nadir)
        if normalized is None:
            raw_objectives = MO_FBSUtil.from_minimization(objectives_min)
            fallback_weights = np.full(4, 0.25, dtype=float) if weights is None else weights
            return MO_FBSUtil.aggregated_energy(raw_objectives, fallback_weights)

        weights = np.full(normalized.size, 1.0 / normalized.size, dtype=float) if weights is None else MO_FBSUtil._as_float_vector(weights, minimum_size=normalized.size, fill_value=1.0)
        weights = np.clip(weights[: normalized.size], 0.0, None)
        if not np.any(weights > 0):
            weights = np.full(normalized.size, 1.0 / normalized.size, dtype=float)
        weights = weights / np.sum(weights)
        return float(np.linalg.norm(normalized * weights, ord=2))

    @staticmethod
    def search_energy(objectives_min, *, is_feasible, d_inf, total_violation, ideal=None, nadir=None, weights=None):
        base_energy = MO_FBSUtil.surrogate_energy(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
        if is_feasible:
            return float(base_energy)
        return float(1_000_000.0 + 10_000.0 * max(int(d_inf), 0) + max(float(total_violation), 0.0) + base_energy)

    @staticmethod
    def pareto_dominates(left_objectives, right_objectives, atol=1e-9):
        left = MO_FBSUtil._as_float_vector(left_objectives, minimum_size=4, fill_value=0.0)[:4]
        right = MO_FBSUtil._as_float_vector(right_objectives, minimum_size=4, fill_value=0.0)[:4]
        return bool(np.all(left <= right + atol) and np.any(left < right - atol))

    @staticmethod
    def _value(entity, name, default=None):
        if isinstance(entity, dict):
            return entity.get(name, default)
        return getattr(entity, name, default)

    @staticmethod
    def constraint_signature(entity):
        is_feasible = bool(MO_FBSUtil._value(entity, "current_is_feasible", False))
        d_inf = int(MO_FBSUtil._value(entity, "current_d_inf", 0) or 0)
        total_violation = MO_FBSUtil._value(entity, "constraint_violation", None)
        if total_violation is None:
            total_violation = MO_FBSUtil.calculate_total_constraint_violation(
                MO_FBSUtil._value(entity, "fac_b", []),
                MO_FBSUtil._value(entity, "fac_h", []),
                MO_FBSUtil._value(entity, "lower_bounds", []),
                MO_FBSUtil._value(entity, "upper_bounds", []),
            )
        return is_feasible, d_inf, float(total_violation)

    @staticmethod
    def compare_solution_quality(left, right, atol=1e-9):
        left_feasible, left_d_inf, left_violation = MO_FBSUtil.constraint_signature(left)
        right_feasible, right_d_inf, right_violation = MO_FBSUtil.constraint_signature(right)

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

        left_objectives = MO_FBSUtil._value(left, "mo_objectives_min", None)
        right_objectives = MO_FBSUtil._value(right, "mo_objectives_min", None)
        if left_objectives is None or right_objectives is None:
            left_energy = float(MO_FBSUtil._value(left, "fitness", math.inf))
            right_energy = float(MO_FBSUtil._value(right, "fitness", math.inf))
            if left_energy < right_energy - atol:
                return -1
            if right_energy < left_energy - atol:
                return 1
            return 0

        left_dominates = MO_FBSUtil.pareto_dominates(left_objectives, right_objectives, atol=atol)
        right_dominates = MO_FBSUtil.pareto_dominates(right_objectives, left_objectives, atol=atol)
        if left_dominates and not right_dominates:
            return -1
        if right_dominates and not left_dominates:
            return 1
        return 0

    @staticmethod
    def compute_ideal_nadir(candidates: Sequence):
        objective_matrix = []
        for candidate in candidates:
            objectives = MO_FBSUtil._value(candidate, "mo_objectives_min", None)
            if objectives is None:
                continue
            objective_matrix.append(MO_FBSUtil._as_float_vector(objectives, minimum_size=4, fill_value=0.0)[:4])
        if not objective_matrix:
            return None, None
        matrix = np.asarray(objective_matrix, dtype=float)
        return np.min(matrix, axis=0), np.max(matrix, axis=0)

    @staticmethod
    def _normalized_archive_matrix(candidates: Sequence, ideal=None, nadir=None):
        vectors = []
        for candidate in candidates:
            objectives = MO_FBSUtil._value(candidate, "mo_objectives_min", None)
            if objectives is None:
                continue
            vectors.append(MO_FBSUtil._as_float_vector(objectives, minimum_size=4, fill_value=0.0)[:4])
        if not vectors:
            return np.empty((0, 4), dtype=float), ideal, nadir

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
    def _prepare_hv_boxes(lower_points, upper):
        lower_points = np.asarray(lower_points, dtype=float)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower_points.size == 0 or upper.size == 0:
            return np.empty((0, upper.size), dtype=float), upper
        if lower_points.ndim == 1:
            lower_points = lower_points.reshape(1, -1)
        valid = np.all(np.isfinite(lower_points), axis=1)
        valid &= np.all(lower_points < upper[None, :], axis=1)
        if not np.any(valid):
            return np.empty((0, upper.size), dtype=float), upper
        clipped = np.clip(lower_points[valid], 0.0, upper[None, :])
        return clipped, upper

    @staticmethod
    def _union_hypervolume(lower_points, upper):
        lower_points, upper = MO_FBSUtil._prepare_hv_boxes(lower_points, upper)
        if lower_points.size == 0:
            return 0.0

        dim = int(upper.size)
        if dim == 1:
            return float(max(upper[0] - float(np.min(lower_points[:, 0])), 0.0))

        # 递归切片计算并集体积，避免重叠区域被重复计数。
        boundaries = np.unique(lower_points[:, 0])
        boundaries = boundaries[boundaries < upper[0]]
        if boundaries.size == 0:
            return 0.0
        boundaries = np.concatenate([boundaries, [upper[0]]])

        volume = 0.0
        for idx in range(boundaries.size - 1):
            left = float(boundaries[idx])
            right = float(boundaries[idx + 1])
            width = right - left
            if width <= 1e-15:
                continue
            active = lower_points[:, 0] <= left + 1e-15
            if not np.any(active):
                continue
            slice_volume = MO_FBSUtil._union_hypervolume(lower_points[active, 1:], upper[1:])
            volume += width * slice_volume
        return float(volume)

    @staticmethod
    def archive_hypervolume(candidates: Sequence, ideal=None, nadir=None, reference_margin=0.1):
        normalized, _, _ = MO_FBSUtil._normalized_archive_matrix(candidates, ideal=ideal, nadir=nadir)
        if normalized.size == 0:
            return 0.0
        margin = max(float(reference_margin), 1e-9)
        reference = np.max(normalized, axis=0) + margin
        return MO_FBSUtil._union_hypervolume(normalized, reference)

    @staticmethod
    def archive_spacing(candidates: Sequence, ideal=None, nadir=None):
        normalized, _, _ = MO_FBSUtil._normalized_archive_matrix(candidates, ideal=ideal, nadir=nadir)
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
    def archive_igd(candidates: Sequence, reference_vectors=None, ideal=None, nadir=None):
        if not reference_vectors:
            return None
        normalized_candidates, ideal, nadir = MO_FBSUtil._normalized_archive_matrix(
            candidates,
            ideal=ideal,
            nadir=nadir,
        )
        if normalized_candidates.size == 0:
            return None

        normalized_reference = []
        for vector in reference_vectors:
            normalized = MO_FBSUtil.normalize_objective_vector(vector, ideal=ideal, nadir=nadir)
            if normalized is None:
                continue
            normalized_reference.append(normalized)
        if not normalized_reference:
            return None
        normalized_reference = np.asarray(normalized_reference, dtype=float)

        total = 0.0
        for reference_vector in normalized_reference:
            delta = normalized_candidates - reference_vector
            distance = float(np.min(np.linalg.norm(delta, axis=1)))
            if not np.isfinite(distance):
                return None
            total += distance
        return float(total / float(len(normalized_reference)))

    @staticmethod
    def _subset_hypervolume_from_normalized(normalized_vectors, reference_margin=0.1):
        normalized_vectors = np.asarray(normalized_vectors, dtype=float)
        if normalized_vectors.size == 0:
            return 0.0
        reference = np.max(normalized_vectors, axis=0) + max(float(reference_margin), 1e-9)
        return MO_FBSUtil._union_hypervolume(normalized_vectors, reference)

    @staticmethod
    def _subset_spacing_from_normalized(normalized_vectors):
        normalized_vectors = np.asarray(normalized_vectors, dtype=float)
        count = int(normalized_vectors.shape[0])
        if count <= 1:
            return 0.0
        distances = []
        for idx in range(count):
            delta = normalized_vectors - normalized_vectors[idx]
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
        left_objectives = MO_FBSUtil._value(left, "mo_objectives_min", None)
        right_objectives = MO_FBSUtil._value(right, "mo_objectives_min", None)
        if left_objectives is None or right_objectives is None:
            return False
        return bool(np.allclose(left_objectives, right_objectives, atol=atol, rtol=1e-7))

    @staticmethod
    def select_nfcs_subset(
        candidates: Sequence,
        max_size: int,
        ideal=None,
        nadir=None,
        reference_vectors=None,
        hv_weight=1.0,
        igd_weight=0.35,
        spacing_weight=0.05,
    ):
        candidates = list(candidates)
        if len(candidates) <= int(max_size):
            return candidates

        if ideal is None or nadir is None:
            ideal, nadir = MO_FBSUtil.compute_ideal_nadir(candidates)

        normalized_vectors = []
        for candidate in candidates:
            normalized = MO_FBSUtil.normalize_objective_vector(
                MO_FBSUtil._value(candidate, "mo_objectives_min", None),
                ideal=ideal,
                nadir=nadir,
            )
            if normalized is None:
                normalized = np.zeros(4, dtype=float)
            normalized_vectors.append(normalized)
        normalized_vectors = np.asarray(normalized_vectors, dtype=float)

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
                score = MO_FBSUtil.decision_score(
                    MO_FBSUtil._value(candidates[idx], "mo_objectives_min", None),
                    ideal=ideal,
                    nadir=nadir,
                )
                # 在线阶段回退为轻量 farthest-first，避免在主循环中反复试算精确集合指标。
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
    def select_representative_solution(candidates: Sequence, ideal=None, nadir=None, weights=None):
        feasible_candidates = [candidate for candidate in candidates if MO_FBSUtil.constraint_signature(candidate)[0]]
        if not feasible_candidates:
            return None, math.inf, None

        if ideal is None or nadir is None:
            ideal, nadir = MO_FBSUtil.compute_ideal_nadir(feasible_candidates)

        best_candidate = None
        best_index = None
        best_key = None
        for idx, candidate in enumerate(feasible_candidates):
            objectives_min = MO_FBSUtil._value(candidate, "mo_objectives_min", None)
            score = MO_FBSUtil.decision_score(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            surrogate = MO_FBSUtil.surrogate_energy(objectives_min, ideal=ideal, nadir=nadir, weights=weights)
            mhc = float(MO_FBSUtil._value(candidate, "MHC", math.inf))
            key = (score, surrogate, mhc, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
                best_index = idx
        return best_candidate, float(best_key[0]), int(best_index)

    @staticmethod
    def update_pareto_archive(
        candidates: Sequence,
        candidate,
        max_size=None,
        clone_fn=None,
        atol=1e-9,
        quality_gate_when_full=False,
        quality_hv_tol=1e-10,
        quality_spacing_tol=1e-8,
        spacing_guard_when_full=False,
        spacing_guard_rel_tol=0.0,
        spacing_guard_hv_gain_rel=0.0,
        require_candidate_retained=False,
        ideal=None,
        nadir=None,
        reference_vectors=None,
        quality_igd_tol=1e-8,
    ):
        archive = [item for item in candidates if MO_FBSUtil.constraint_signature(item)[0]]
        if candidate is None or not MO_FBSUtil.constraint_signature(candidate)[0]:
            return archive, False, 0

        clone = copy.deepcopy if clone_fn is None else clone_fn
        full_before_insert = max_size is not None and len(archive) >= int(max_size)
        kept = []
        removed = 0
        for existing in archive:
            comparison = MO_FBSUtil.compare_solution_quality(candidate, existing, atol=atol)
            if comparison > 0:
                return archive, False, 0
            if comparison < 0:
                removed += 1
                continue
            if MO_FBSUtil._duplicate_objectives(candidate, existing, atol=atol):
                return archive, False, 0
            kept.append(existing)

        kept.append(clone(candidate))
        if max_size is not None and len(kept) > int(max_size):
            trim_ideal, trim_nadir = MO_FBSUtil.compute_ideal_nadir(kept)
            kept = MO_FBSUtil.select_nfcs_subset(
                kept,
                int(max_size),
                ideal=trim_ideal,
                nadir=trim_nadir,
                reference_vectors=reference_vectors,
            )

        candidate_retained = any(MO_FBSUtil._duplicate_objectives(candidate, existing, atol=atol) for existing in kept)
        if bool(require_candidate_retained) and not candidate_retained:
            # 候选解如果在截断后都无法保留，就不应让它扰动已有档案。
            return archive, False, 0

        if bool(quality_gate_when_full) and full_before_insert:
            quality_ideal = ideal
            quality_nadir = nadir
            if quality_ideal is None or quality_nadir is None:
                quality_ideal, quality_nadir = MO_FBSUtil.compute_ideal_nadir(list(archive) + list(kept))
            before_hv = MO_FBSUtil.archive_hypervolume(archive, ideal=quality_ideal, nadir=quality_nadir)
            after_hv = MO_FBSUtil.archive_hypervolume(kept, ideal=quality_ideal, nadir=quality_nadir)
            before_spacing = MO_FBSUtil.archive_spacing(archive, ideal=quality_ideal, nadir=quality_nadir)
            after_spacing = MO_FBSUtil.archive_spacing(kept, ideal=quality_ideal, nadir=quality_nadir)
            before_igd = MO_FBSUtil.archive_igd(archive, reference_vectors=reference_vectors, ideal=quality_ideal, nadir=quality_nadir)
            after_igd = MO_FBSUtil.archive_igd(kept, reference_vectors=reference_vectors, ideal=quality_ideal, nadir=quality_nadir)

            if bool(spacing_guard_when_full) and before_spacing > 1e-12:
                spacing_rel_worse = (after_spacing - before_spacing) / before_spacing
                hv_rel_gain = 0.0
                if before_hv <= 1e-12:
                    hv_rel_gain = math.inf if after_hv > before_hv + float(quality_hv_tol) else 0.0
                else:
                    hv_rel_gain = (after_hv - before_hv) / before_hv
                if (
                    spacing_rel_worse > float(spacing_guard_rel_tol)
                    and hv_rel_gain < float(spacing_guard_hv_gain_rel)
                ):
                    return archive, False, 0

            hv_improved = after_hv > before_hv + float(quality_hv_tol)
            hv_not_worse = after_hv >= before_hv - float(quality_hv_tol)
            igd_improved = False
            igd_not_worse = True
            if before_igd is not None and after_igd is not None:
                igd_improved = after_igd + float(quality_igd_tol) < before_igd
                igd_not_worse = after_igd <= before_igd + float(quality_igd_tol)
            spacing_improved = after_spacing + float(quality_spacing_tol) < before_spacing
            # 质量门控按 HV > IGD > Spacing 的优先级判定。
            if not (hv_improved or (hv_not_worse and igd_improved) or (hv_not_worse and igd_not_worse and spacing_improved)):
                return archive, False, 0
        return kept, True, removed
