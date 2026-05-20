from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GRASPInstanceSpec:
    name: str
    rows: int
    cols: int
    facilities: int


_INSTANCE_SPECS = {
    "previous6": (2, 3, 6),
    "previous8": (2, 4, 8),
    "A-10-10": (2, 5, 10),
    "A-10-20": (2, 5, 10),
    "A-10-30": (2, 5, 10),
    "A-10-40": (2, 5, 10),
    "A-10-50": (2, 5, 10),
    "A-10-60": (2, 5, 10),
    "A-10-70": (2, 5, 10),
    "A-10-80": (2, 5, 10),
    "A-10-90": (2, 5, 10),
    "O-10": (2, 5, 10),
    "Y-10": (2, 5, 10),
    "A-12-10": (2, 6, 12),
    "A-12-20": (2, 6, 12),
    "A-12-30": (2, 6, 12),
    "A-12-40": (2, 6, 12),
    "A-12-50": (2, 6, 12),
    "A-12-60": (2, 6, 12),
    "A-12-70": (2, 6, 12),
    "A-12-80": (2, 6, 12),
    "A-12-90": (2, 6, 12),
    "S-12": (2, 6, 12),
    "Y-12": (2, 6, 12),
    "previous12": (3, 4, 12),
    "N-15": (3, 5, 15),
    "O-15": (3, 5, 15),
    "S-15": (3, 5, 15),
    "Y-15": (3, 5, 15),
    "previous15": (3, 5, 15),
    "A-20-10": (4, 5, 20),
    "A-20-20": (4, 5, 20),
    "A-20-30": (4, 5, 20),
    "A-20-40": (4, 5, 20),
    "A-20-50": (4, 5, 20),
    "A-20-60": (4, 5, 20),
    "A-20-70": (4, 5, 20),
    "A-20-80": (4, 5, 20),
    "A-20-90": (4, 5, 20),
    "N-20": (4, 5, 20),
    "O-20": (4, 5, 20),
    "S-20": (4, 5, 20),
    "Y-20": (4, 5, 20),
    "A-25-10": (5, 5, 25),
    "A-25-20": (5, 5, 25),
    "A-25-30": (5, 5, 25),
    "A-25-40": (5, 5, 25),
    "A-25-50": (5, 5, 25),
    "A-25-60": (5, 5, 25),
    "A-25-70": (5, 5, 25),
    "A-25-80": (5, 5, 25),
    "A-25-90": (5, 5, 25),
    "S-25": (5, 5, 25),
    "Y-25": (5, 5, 25),
    "Y-30": (5, 6, 30),
    "Y-35": (5, 7, 35),
    "Y-40": (5, 8, 40),
    "Y-45": (5, 9, 45),
    "Y-50": (5, 10, 50),
    "Y-60": (6, 10, 60),
}


class GRASPInstanceLoader:
    """加载论文 GRASP 双目标离散设施布局算例。"""

    @staticmethod
    def default_root() -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "GRASP_Instances"

    @staticmethod
    def normalize_instance_name(instance_name: str) -> str:
        name = str(instance_name).strip()
        if name.endswith("_t"):
            name = name[:-2]
        return name

    @classmethod
    def get_spec(cls, instance_name: str) -> GRASPInstanceSpec:
        key = cls.normalize_instance_name(instance_name)
        if key not in _INSTANCE_SPECS:
            valid = ", ".join(sorted(_INSTANCE_SPECS))
            raise KeyError(f"未知 GRASP 论文实例: {instance_name}. 可选实例: {valid}")
        rows, cols, facilities = _INSTANCE_SPECS[key]
        return GRASPInstanceSpec(name=key, rows=rows, cols=cols, facilities=facilities)

    @staticmethod
    def _read_matrix_file(path: Path) -> tuple[int, np.ndarray]:
        with path.open("r", encoding="utf-8") as handle:
            raw_lines = [line.strip() for line in handle]
        lines = [line for line in raw_lines if line]
        if len(lines) < 2:
            raise ValueError(f"实例文件内容不足: {path}")

        facility_count = int(lines[0])
        matrix_rows = []
        for line in lines[2:]:
            matrix_rows.append([float(token) for token in line.split()])
        matrix = np.asarray(matrix_rows, dtype=float)
        if matrix.shape != (facility_count, facility_count):
            raise ValueError(
                f"实例矩阵形状错误: {path} | 期望 {(facility_count, facility_count)} | 实际 {matrix.shape}"
            )
        return facility_count, matrix

    @staticmethod
    def _symmetrize_upper(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"矩阵必须为方阵，当前形状为 {matrix.shape}")
        diagonal = np.diag(np.diag(matrix))
        if np.allclose(np.tril(matrix, -1), 0.0):
            return matrix + matrix.T - diagonal
        if np.allclose(np.triu(matrix, 1), 0.0):
            return matrix + matrix.T - diagonal
        return matrix

    @classmethod
    def load_instance(cls, instance_name: str, root: Path | None = None) -> dict:
        spec = cls.get_spec(instance_name)
        base_root = cls.default_root() if root is None else Path(root)
        mhc_path = base_root / "instancesMHC" / f"{spec.name}_t.txt"
        if not mhc_path.exists():
            mhc_path = base_root / "instancesMHC" / f"{spec.name}.txt"
        cr_path = base_root / "instancesCR" / f"{spec.name}_t_CR.txt"
        if not cr_path.exists():
            cr_path = base_root / "instancesCR" / f"{spec.name}_CR.txt"

        if not mhc_path.exists():
            raise FileNotFoundError(f"未找到 MHC 实例文件: {mhc_path}")
        if not cr_path.exists():
            raise FileNotFoundError(f"未找到 CR 实例文件: {cr_path}")

        mhc_n, mhc_upper = cls._read_matrix_file(mhc_path)
        cr_n, cr_upper = cls._read_matrix_file(cr_path)
        if mhc_n != cr_n or mhc_n != spec.facilities:
            raise ValueError(
                f"实例规模不一致: {spec.name} | spec={spec.facilities} | mhc={mhc_n} | cr={cr_n}"
            )

        mhc_matrix = cls._symmetrize_upper(mhc_upper)
        cr_matrix = cls._symmetrize_upper(cr_upper)
        return {
            "name": spec.name,
            "rows": int(spec.rows),
            "cols": int(spec.cols),
            "n": int(spec.facilities),
            "mhc_matrix": mhc_matrix,
            "cr_matrix": cr_matrix,
            "mhc_path": mhc_path,
            "cr_path": cr_path,
        }
