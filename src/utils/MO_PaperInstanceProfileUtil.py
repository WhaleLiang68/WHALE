import copy

import numpy as np

from src.utils.FlowMatrixUtil import FlowMatrixUtil


class MO_PaperInstanceProfileUtil:
    """论文对比线专用实例口径覆盖，避免污染默认实验环境。"""

    FLOW_PROFILE_VERSION = "paper_flow_profile_v1"
    CONSTRAINT_PROFILE_VERSION = "paper_constraint_profile_v1"
    _RAW_FLOW_INSTANCES = {"O7", "O9"}
    _SC_CONSTRAINT_NOTES = {
        "SC30": "论文正文未单列 SC30 的 AR 数值；当前论文线按常用基准口径使用最大长宽比上限 5.0。",
        "SC35": "论文正文未单列 SC35 的 AR 数值；当前论文线按常用基准口径使用最大长宽比上限 4.0。",
    }

    @staticmethod
    def _base_env(env):
        return env.unwrapped if hasattr(env, "unwrapped") else env

    @classmethod
    def _load_ab20_1963_matrix(cls):
        return FlowMatrixUtil.load_ab20_1963_matrix()

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
        elif FlowMatrixUtil.uses_ab20_paper_flow(instance):
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
