import numpy as np


class MO_PaperMetricsUtil:
    """按论文【21】定义计算 PR、SP、OPS。"""

    @staticmethod
    def _as_objective_matrix(values):
        matrix = np.asarray(values, dtype=float)
        if matrix.size == 0:
            return np.empty((0, 2), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < 2:
            raise ValueError("论文指标至少需要两个目标值。")
        matrix = matrix[:, :2]
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        return matrix[finite_mask]

    @staticmethod
    @staticmethod
    def to_minimization(raw_objectives):
        matrix = MO_PaperMetricsUtil._as_objective_matrix(raw_objectives)
        if matrix.size == 0:
            return matrix
        return np.column_stack([matrix[:, 0], -matrix[:, 1]])

    @staticmethod
    def nondominated_mask(raw_objectives, atol=1e-9):
        matrix = MO_PaperMetricsUtil._as_objective_matrix(raw_objectives)
        if matrix.size == 0:
            return np.zeros(0, dtype=bool), matrix

        objectives_min = MO_PaperMetricsUtil.to_minimization(matrix)
        mask = np.ones(objectives_min.shape[0], dtype=bool)
        for idx in range(objectives_min.shape[0]):
            if not mask[idx]:
                continue
            current = objectives_min[idx]
            dominates_current = np.all(objectives_min <= current + atol, axis=1) & np.any(
                objectives_min < current - atol,
                axis=1,
            )
            dominates_current[idx] = False
            if np.any(dominates_current):
                mask[idx] = False
        return mask, matrix

    @staticmethod
    def pareto_front(raw_objectives):
        mask, matrix = MO_PaperMetricsUtil.nondominated_mask(raw_objectives)
        return matrix[mask]

    @staticmethod
    def calculate_pr(all_objectives_raw):
        all_solutions = MO_PaperMetricsUtil._as_objective_matrix(all_objectives_raw)
        if all_solutions.size == 0:
            return None
        pareto = MO_PaperMetricsUtil.pareto_front(all_solutions)
        return float(len(pareto) / len(all_solutions))

    @staticmethod
    def calculate_sp(pareto_objectives_raw):
        pareto = MO_PaperMetricsUtil._as_objective_matrix(pareto_objectives_raw)
        count = int(pareto.shape[0])
        if count <= 1:
            return 0.0

        nearest_distances = []
        for idx in range(count):
            delta = np.abs(pareto - pareto[idx])
            distances = np.sum(delta, axis=1)
            distances[idx] = np.inf
            nearest = float(np.min(distances))
            if np.isfinite(nearest):
                nearest_distances.append(nearest)
        if not nearest_distances:
            return 0.0

        nearest_distances = np.asarray(nearest_distances, dtype=float)
        mean_distance = float(np.mean(nearest_distances))
        return float(np.sqrt(np.sum((nearest_distances - mean_distance) ** 2) / max(count - 1, 1)))

    @staticmethod
    def _normalize_by_reference(values, reference_values):
        matrix = MO_PaperMetricsUtil._as_objective_matrix(values)
        reference = MO_PaperMetricsUtil._as_objective_matrix(reference_values)
        if matrix.size == 0 or reference.size == 0:
            return matrix

        minima = np.min(reference, axis=0)
        spans = np.max(reference, axis=0) - minima
        normalized = np.zeros_like(matrix, dtype=float)
        valid_dims = spans > 1e-12
        if np.any(valid_dims):
            normalized[:, valid_dims] = (matrix[:, valid_dims] - minima[valid_dims]) / spans[valid_dims]
        return normalized

    @staticmethod
    def calculate_sp_normalized(all_objectives_raw, pareto_objectives_raw):
        normalized_pareto = MO_PaperMetricsUtil._normalize_by_reference(
            pareto_objectives_raw,
            all_objectives_raw,
        )
        return MO_PaperMetricsUtil.calculate_sp(normalized_pareto)

    @staticmethod
    def calculate_ops(all_objectives_raw, pareto_objectives_raw):
        all_solutions = MO_PaperMetricsUtil._as_objective_matrix(all_objectives_raw)
        pareto = MO_PaperMetricsUtil._as_objective_matrix(pareto_objectives_raw)
        if all_solutions.size == 0 or pareto.size == 0:
            return None, [None, None]

        all_range = np.max(all_solutions, axis=0) - np.min(all_solutions, axis=0)
        pareto_range = np.max(pareto, axis=0) - np.min(pareto, axis=0)
        components = []
        for idx in range(2):
            denominator = float(all_range[idx])
            if denominator <= 1e-12:
                components.append(0.0)
            else:
                components.append(float(pareto_range[idx] / denominator))
        return float(np.prod(components)), components

    @staticmethod
    def calculate_summary(all_objectives_raw, pareto_objectives_raw=None):
        all_solutions = MO_PaperMetricsUtil._as_objective_matrix(all_objectives_raw)
        if pareto_objectives_raw is None:
            pareto = MO_PaperMetricsUtil.pareto_front(all_solutions)
        else:
            pareto = MO_PaperMetricsUtil._as_objective_matrix(pareto_objectives_raw)
        if all_solutions.size == 0:
            return {
                "paper_solution_count": 0,
                "paper_pareto_count": 0,
                "paper_pr": None,
                "paper_sp": None,
                "paper_sp_raw": None,
                "paper_sp_norm": None,
                "paper_ops": None,
                "paper_ops_components": [None, None],
                "paper_best_mhc": None,
                "paper_mean_mhc": None,
                "paper_best_f3": None,
                "paper_mean_f3": None,
            }

        ops, components = MO_PaperMetricsUtil.calculate_ops(all_solutions, pareto)
        paper_sp_raw = MO_PaperMetricsUtil.calculate_sp(pareto)
        return {
            "paper_solution_count": int(len(all_solutions)),
            "paper_pareto_count": int(len(pareto)),
            "paper_pr": MO_PaperMetricsUtil.calculate_pr(all_solutions),
            "paper_sp": paper_sp_raw,
            "paper_sp_raw": paper_sp_raw,
            "paper_sp_norm": MO_PaperMetricsUtil.calculate_sp_normalized(all_solutions, pareto),
            "paper_ops": ops,
            "paper_ops_components": components,
            "paper_best_mhc": float(np.min(all_solutions[:, 0])),
            "paper_mean_mhc": float(np.mean(all_solutions[:, 0])),
            "paper_best_f3": float(np.max(all_solutions[:, 1])),
            "paper_mean_f3": float(np.mean(all_solutions[:, 1])),
        }
