import hashlib
import pickle
from pathlib import Path

import numpy as np


class MO_PaperPreferenceUtil:
    """论文口径下的静态非物流偏好矩阵工具。"""

    VERSION = "paper_proxy_preference_v1"
    DEFAULT_DENSITY = 0.08
    DEFAULT_MAX_WEIGHT = 5

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).resolve().parent / "data"

    @staticmethod
    def _stable_seed(instance_name: str) -> int:
        token = f"{instance_name}:{MO_PaperPreferenceUtil.VERSION}".encode("utf-8")
        digest = hashlib.sha256(token).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)

    @staticmethod
    def _pair_indices(n: int):
        return [(i, j) for i in range(int(n)) for j in range(i + 1, int(n))]

    @staticmethod
    def generate_matrix(n, seed=None, density=None, max_weight=None):
        facility_count = int(n)
        if facility_count <= 1:
            return np.zeros((facility_count, facility_count), dtype=int)

        density = MO_PaperPreferenceUtil.DEFAULT_DENSITY if density is None else float(density)
        density = float(np.clip(density, 0.0, 1.0))
        max_weight = MO_PaperPreferenceUtil.DEFAULT_MAX_WEIGHT if max_weight is None else int(max_weight)
        max_weight = int(max(1, max_weight))

        rng = np.random.default_rng(seed)
        pairs = MO_PaperPreferenceUtil._pair_indices(facility_count)
        pair_count = len(pairs)
        edge_count = int(round(pair_count * density))
        edge_count = min(pair_count, max(facility_count, edge_count))

        matrix = np.zeros((facility_count, facility_count), dtype=int)
        if edge_count <= 0:
            return matrix

        chosen = rng.choice(pair_count, size=edge_count, replace=False)
        weights = rng.integers(1, max_weight + 1, size=edge_count)
        for pair_idx, weight in zip(chosen, weights):
            i, j = pairs[int(pair_idx)]
            matrix[i, j] = int(weight)
            matrix[j, i] = int(weight)
        return matrix

    @staticmethod
    def load_or_generate_matrix(n, instance_name="Du62", data_dir=None, density=None, max_weight=None):
        facility_count = int(n)
        instance_key = str(instance_name or "UNKNOWN")
        target_dir = MO_PaperPreferenceUtil.default_data_dir() if data_dir is None else Path(data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{instance_key}_paper_preference_matrix.pkl"

        if filepath.exists():
            with filepath.open("rb") as handle:
                payload = pickle.load(handle)
            matrix = np.asarray(payload["preference_matrix"], dtype=int)
            if matrix.shape == (facility_count, facility_count):
                return matrix, payload

        seed = MO_PaperPreferenceUtil._stable_seed(instance_key)
        matrix = MO_PaperPreferenceUtil.generate_matrix(
            facility_count,
            seed=seed,
            density=density,
            max_weight=max_weight,
        )
        payload = {
            "version": MO_PaperPreferenceUtil.VERSION,
            "instance_name": instance_key,
            "facility_count": facility_count,
            "seed": int(seed),
            "density": MO_PaperPreferenceUtil.DEFAULT_DENSITY if density is None else float(density),
            "max_weight": MO_PaperPreferenceUtil.DEFAULT_MAX_WEIGHT if max_weight is None else int(max_weight),
            "preference_matrix": matrix,
        }
        with filepath.open("wb") as handle:
            pickle.dump(payload, handle)
        return matrix, payload

    @staticmethod
    def score_layout(fac_x, fac_y, preference_matrix):
        weights = np.asarray(preference_matrix, dtype=float)
        if weights.size == 0 or not np.any(weights > 0):
            return 0.0

        fac_x = np.asarray(fac_x, dtype=float).reshape(-1)
        fac_y = np.asarray(fac_y, dtype=float).reshape(-1)
        distance_matrix = np.abs(fac_x[:, None] - fac_x[None, :]) + np.abs(fac_y[:, None] - fac_y[None, :])
        upper_weights = np.triu(weights, k=1)
        total_weight = float(np.sum(upper_weights))
        if total_weight <= 0.0:
            return 0.0

        # 与论文中的用户评分保持同量纲，越接近的偏好设施对得分越高。
        proximity = np.divide(
            upper_weights,
            1.0 + distance_matrix,
            out=np.zeros_like(upper_weights, dtype=float),
            where=upper_weights > 0.0,
        )
        return float(100.0 * np.sum(proximity) / total_weight)
