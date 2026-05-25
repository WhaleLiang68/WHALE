import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


class CRMatrixStore:
    """实例级 CR 矩阵存取工具。"""

    LEVEL_MAPPING = {
        "A": 6,
        "E": 5,
        "I": 4,
        "O": 3,
        "U": 2,
        "X": 1,
    }
    GENERATION_RULE = (
        "symmetric_uniform_random_levels_without_ratio_control;"
        "off_diagonal in {1,2,3,4,5,6}; diagonal=0"
    )

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "cr_matrices"

    @staticmethod
    def build_path(instance_name: str, data_dir=None) -> Path:
        instance_key = str(instance_name or "").strip()
        if not instance_key:
            raise ValueError("instance_name 不能为空。")
        target_dir = CRMatrixStore.default_data_dir() if data_dir is None else Path(data_dir)
        return target_dir / f"{instance_key}_CR.pkl"

    @staticmethod
    def generate_matrix(facility_count: int):
        n = int(facility_count)
        if n <= 0:
            raise ValueError("facility_count 必须为正整数。")
        rng = np.random.default_rng()
        matrix = np.zeros((n, n), dtype=int)
        upper_indices = np.triu_indices(n, k=1)
        # 论文口径：关系矩阵本身不出现 0，0 只在“不相邻时 CR 记 0”的评分阶段出现。
        values = rng.integers(1, 7, size=len(upper_indices[0]), endpoint=False)
        matrix[upper_indices] = values
        matrix[(upper_indices[1], upper_indices[0])] = values
        return matrix

    @staticmethod
    def save_matrix(instance_name: str, matrix, data_dir=None, overwrite=False):
        path = CRMatrixStore.build_path(instance_name=instance_name, data_dir=data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not bool(overwrite):
            raise FileExistsError(f"CR 矩阵文件已存在，拒绝覆盖: {path}")

        matrix = np.asarray(matrix, dtype=int)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("CR 矩阵必须是方阵。")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("CR 矩阵必须对称。")
        if not np.all(np.diag(matrix) == 0):
            raise ValueError("CR 矩阵主对角线必须为 0。")
        upper_values = matrix[np.triu_indices(matrix.shape[0], k=1)]
        if upper_values.size and (np.min(upper_values) < 1 or np.max(upper_values) > 6):
            raise ValueError("CR 矩阵非对角元素必须落在 {1,2,3,4,5,6}。")

        payload = {
            "instance_name": str(instance_name),
            "facility_count": int(matrix.shape[0]),
            "generation_rule": CRMatrixStore.GENERATION_RULE,
            "level_mapping": dict(CRMatrixStore.LEVEL_MAPPING),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cr_matrix": matrix,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    @staticmethod
    def load_matrix(instance_name: str, expected_facility_count=None, data_dir=None):
        path = CRMatrixStore.build_path(instance_name=instance_name, data_dir=data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"未找到实例 {instance_name} 的 CR 矩阵文件: {path}。"
                "请先运行 scripts/generate_cr_matrices.py 生成。"
            )
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        matrix = np.asarray(payload["cr_matrix"], dtype=int)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"CR 矩阵文件格式非法: {path}")
        if expected_facility_count is not None and matrix.shape[0] != int(expected_facility_count):
            raise ValueError(
                f"CR 矩阵尺寸与实例设施数不一致: matrix={matrix.shape[0]}, expected={int(expected_facility_count)}"
            )
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"CR 矩阵不是对称矩阵: {path}")
        if not np.all(np.diag(matrix) == 0):
            raise ValueError(f"CR 矩阵主对角线不为 0: {path}")
        return matrix, payload, path
