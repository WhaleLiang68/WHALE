import numpy as np

from src.utils.MO_FBSUtil_MO4 import MO_FBSUtil as _MO4_FBSUtil
from src.utils.MO_PaperPreferenceUtil import MO_PaperPreferenceUtil


class MO_FBSUtil(_MO4_FBSUtil):
    """论文口径专用目标函数，保留 MO4 兼容壳但只激活两个真实目标。"""

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
        preference_matrix=None,
        area_utilization=1.0,
    ):
        _ = (fac_b, fac_h, n, rel_matrix, dist_req_matrix, aspect_limits, optimal_aspect_ratio)
        f1_mhc = float(mhc)
        f3_proxy = MO_PaperPreferenceUtil.score_layout(fac_x, fac_y, preference_matrix)

        # 第 3、4 维仅用于兼容 MO4 现有结构；它们保持常量且权重为 0，不参与有效优化。
        return [f1_mhc, float(f3_proxy), 0.0, float(area_utilization)]

    @staticmethod
    def aggregated_energy(objectives, weights):
        mhc, f3_proxy, _neutral, _area_utilization = MO_FBSUtil._as_float_vector(
            objectives,
            minimum_size=4,
            fill_value=0.0,
        )[:4]
        weights = MO_FBSUtil._as_float_vector(weights, minimum_size=4, fill_value=0.0)[:4]
        if not np.any(weights > 0):
            weights = np.asarray([0.5, 0.5, 0.0, 0.0], dtype=float)
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.sum(weights)

        epsilon = 1e-6
        return float(weights[0] * max(mhc, 0.0) + weights[1] * (1.0 / max(f3_proxy, epsilon)))
