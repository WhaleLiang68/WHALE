from __future__ import annotations

from pathlib import Path

import numpy as np

import src.utils.config as config


class FlowMatrixUtil:
    """统一管理实例物流量矩阵的加载与覆盖口径。"""

    _AB20_1963_CSV = "AB20(1963).csv"

    @staticmethod
    def data_dir() -> Path:
        return Path(config.FILE_PATH).resolve().parent

    @staticmethod
    def normalize_instance_name(instance_name: str | None) -> str:
        return str(instance_name or "").strip()

    @classmethod
    def uses_ab20_paper_flow(cls, instance_name: str | None) -> bool:
        normalized = cls.normalize_instance_name(instance_name).upper()
        return normalized == "AB20" or normalized.startswith("AB20-")

    @classmethod
    def load_ab20_1963_matrix(cls) -> tuple[np.ndarray, Path]:
        csv_path = cls.data_dir() / cls._AB20_1963_CSV
        with csv_path.open("r", encoding="utf-8-sig") as handle:
            matrix = np.loadtxt(handle, delimiter=",", dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"AB20 覆盖物流量矩阵必须为方阵，当前形状为 {matrix.shape}")
        return matrix, csv_path

    @classmethod
    def get_raw_flow_matrix(cls, flow_matrices, instance_name: str | None) -> np.ndarray:
        normalized = cls.normalize_instance_name(instance_name)
        if cls.uses_ab20_paper_flow(normalized):
            matrix, _ = cls.load_ab20_1963_matrix()
            return np.asarray(matrix, dtype=float)
        if normalized not in flow_matrices:
            raise KeyError(f"未找到实例 {normalized} 的物流量矩阵。")
        return np.asarray(flow_matrices[normalized], dtype=float)

    @staticmethod
    def symmetrize_if_upper_triangular(raw_flow) -> np.ndarray:
        matrix = np.asarray(raw_flow, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"物流量矩阵必须为方阵，当前形状为 {matrix.shape}")
        if np.allclose(np.tril(matrix, -1), 0.0):
            return matrix + matrix.T - np.diag(np.diag(matrix))
        return matrix.copy()
