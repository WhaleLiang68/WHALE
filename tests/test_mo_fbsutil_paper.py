import pytest

from src.utils.MO_FBSUtil_Paper import MO_FBSUtil


def test_paper_objectives_keep_only_two_effective_dimensions():
    objectives = MO_FBSUtil.calculate_objectives(
        fac_x=[0.0, 1.0],
        fac_y=[0.0, 0.0],
        fac_b=[1.0, 1.0],
        fac_h=[1.0, 1.0],
        mhc=12.5,
        n=2,
        preference_matrix=[[0, 5], [5, 0]],
        area_utilization=0.9,
    )

    assert objectives[0] == pytest.approx(12.5)
    assert objectives[1] > 0.0
    assert objectives[2] == pytest.approx(0.0)
    assert objectives[3] == pytest.approx(0.9)


def test_paper_aggregated_energy_ignores_neutral_dimensions():
    energy = MO_FBSUtil.aggregated_energy([10.0, 2.0, 0.0, 0.9], [0.5, 0.5, 0.0, 0.0])

    assert energy == pytest.approx(5.25)
