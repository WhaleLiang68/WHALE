import numpy as np

from src.utils.MO_FBSUtil_GRASP import MO_FBSUtil


def test_grasp_true_cr_is_minimization_objective():
    fac_x = np.array([0.0, 1.0])
    fac_y = np.array([0.0, 0.0])
    objectives = MO_FBSUtil.calculate_objectives(
        fac_x=fac_x,
        fac_y=fac_y,
        fac_b=np.ones(2),
        fac_h=np.ones(2),
        mhc=7.0,
        n=2,
        cr_matrix=np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float),
    )
    objectives_min = MO_FBSUtil.to_minimization(objectives)

    assert objectives[0] == 7.0
    assert objectives[1] == -1.0
    assert np.allclose(objectives_min[:2], np.array([7.0, -1.0]))
