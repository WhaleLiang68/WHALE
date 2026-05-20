import numpy as np

from src.utils.MO_FBSUtil_MO4 import MO_FBSUtil as _BaseMOFBSUtil


class MO_FBSUtil(_BaseMOFBSUtil):
    """论文 BO-MREFLP 口径：MHC 和 CR 均为最小化目标。"""

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
        cr_matrix=None,
    ):
        _ = (fac_b, fac_h, n, rel_matrix, dist_req_matrix, aspect_limits, optimal_aspect_ratio)
        f1_mhc = float(mhc)
        f2_cr = 0.0
        if cr_matrix is not None:
            distance_matrix = (
                np.abs(np.asarray(fac_x, dtype=float)[:, None] - np.asarray(fac_x, dtype=float)[None, :])
                + np.abs(np.asarray(fac_y, dtype=float)[:, None] - np.asarray(fac_y, dtype=float)[None, :])
            )
            f2_cr = float(np.sum(np.triu(np.asarray(cr_matrix, dtype=float) * distance_matrix, k=1)))
        return [f1_mhc, f2_cr, 0.0, 0.0]

    @staticmethod
    def to_minimization(objectives_raw):
        objectives = _BaseMOFBSUtil._as_float_vector(objectives_raw, minimum_size=4, fill_value=0.0)
        return np.asarray([objectives[0], objectives[1], objectives[2], objectives[3]], dtype=float)

    @staticmethod
    def from_minimization(objectives_min):
        objectives = _BaseMOFBSUtil._as_float_vector(objectives_min, minimum_size=4, fill_value=0.0)
        return np.asarray([objectives[0], objectives[1], objectives[2], objectives[3]], dtype=float)

    @staticmethod
    def aggregated_energy(objectives, weights):
        mhc, cr, _neutral1, _neutral2 = _BaseMOFBSUtil._as_float_vector(
            objectives,
            minimum_size=4,
            fill_value=0.0,
        )[:4]
        weights = _BaseMOFBSUtil._as_float_vector(weights, minimum_size=4, fill_value=0.0)[:4]
        if not np.any(weights > 0):
            weights = np.asarray([0.5, 0.5, 0.0, 0.0], dtype=float)
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.sum(weights)
        return float(weights[0] * mhc + weights[1] * cr)
