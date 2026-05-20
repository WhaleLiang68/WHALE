import numpy as np

from src.utils.MO_PaperPreferenceUtil import MO_PaperPreferenceUtil


def test_generate_matrix_is_symmetric_and_repeatable():
    left = MO_PaperPreferenceUtil.generate_matrix(8, seed=123, density=0.25, max_weight=5)
    right = MO_PaperPreferenceUtil.generate_matrix(8, seed=123, density=0.25, max_weight=5)

    assert np.array_equal(left, right)
    assert np.array_equal(left, left.T)
    assert np.all(np.diag(left) == 0)
    assert np.max(left) <= 5


def test_score_layout_rewards_closer_preferred_pairs():
    preference = np.zeros((3, 3), dtype=int)
    preference[0, 1] = preference[1, 0] = 5

    close_score = MO_PaperPreferenceUtil.score_layout([0, 1, 10], [0, 0, 0], preference)
    far_score = MO_PaperPreferenceUtil.score_layout([0, 10, 1], [0, 0, 0], preference)

    assert close_score > far_score
