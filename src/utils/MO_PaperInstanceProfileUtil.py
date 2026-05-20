import copy
from pathlib import Path

import numpy as np

import src.utils.config as config


class MO_PaperInstanceProfileUtil:
    """论文对比线专用实例口径覆盖，避免污染默认实验环境。"""

    FLOW_PROFILE_VERSION = "paper_flow_profile_v1"
    CONSTRAINT_PROFILE_VERSION = "paper_constraint_profile_v1"
    _RAW_FLOW_INSTANCES = {"O7", "O9"}
    _AB20_1963_CSV = "AB20(1963).csv"
    _SC_CONSTRAINT_NOTES = {
        "SC30": "论文正文未单列 SC30 的 AR 数值；当前论文线按常用基准口径使用最大长宽比上限 5.0。",
        "SC35": "论文正文未单列 SC35 的 AR 数值；当前论文线按常用基准口径使用最大长宽比上限 4.0。",
    }

    @staticmethod
    def _base_env(env):
        return env.unwrapped if hasattr(env, "unwrapped") else env

    @staticmethod
    def _data_dir():
        return Path(config.FILE_PATH).resolve().parent

    @classmethod
    def _load_ab20_1963_matrix(cls):
        csv_path = cls._data_dir() / cls._AB20_1963_CSV
        # 文件带 UTF-8 BOM，显式声明编码，避免 Windows 按系统编码误读。
        with csv_path.open("r", encoding="utf-8-sig") as handle:
            matrix = np.loadtxt(handle, delimiter=",", dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"AB20 论文流量矩阵必须为方阵，当前形状为 {matrix.shape}")
        return matrix, csv_path

    @classmethod
    def build_metadata(cls, env):
        base_env = cls._base_env(env)
        instance = str(getattr(base_env, "instance", "UNKNOWN") or "UNKNOWN")
        return {
            "paperFlowProfileVersion": cls.FLOW_PROFILE_VERSION,
            "paperConstraintProfileVersion": cls.CONSTRAINT_PROFILE_VERSION,
            "paperFlowOverrideApplied": False,
            "paperFlowSource": "default_env",
            "paperFlowSourcePath": None,
            "paperConstraintMode": "max_aspect_ratio_only",
            "paperAspectRatioLimit": float(getattr(base_env, "fac_limit_aspect", np.nan)),
            "paperConstraintProfileNote": cls._SC_CONSTRAINT_NOTES.get(instance),
        }

    @classmethod
    def apply_to_env(cls, env):
        base_env = cls._base_env(env)
        instance = str(getattr(base_env, "instance", "UNKNOWN") or "UNKNOWN")
        metadata = cls.build_metadata(base_env)

        if instance in cls._RAW_FLOW_INSTANCES:
            raw_flow = np.asarray(base_env.FlowMatrices[instance], dtype=float)
            base_env.F = raw_flow.copy()
            metadata.update(
                {
                    "paperFlowOverrideApplied": True,
                    "paperFlowSource": "raw_pickle_matrix",
                }
            )
        elif instance == "AB20-ar3":
            paper_flow, csv_path = cls._load_ab20_1963_matrix()
            expected_shape = np.asarray(base_env.F).shape
            if paper_flow.shape != expected_shape:
                raise ValueError(
                    f"AB20 论文流量矩阵形状 {paper_flow.shape} 与环境形状 {expected_shape} 不一致"
                )
            base_env.F = paper_flow.copy()
            metadata.update(
                {
                    "paperFlowOverrideApplied": True,
                    "paperFlowSource": "csv",
                    "paperFlowSourcePath": csv_path.resolve().as_posix(),
                }
            )

        # 环境快照会被复制多次，元数据也必须随实例一起保留。
        base_env.paper_instance_profile = copy.deepcopy(metadata)
        return metadata
