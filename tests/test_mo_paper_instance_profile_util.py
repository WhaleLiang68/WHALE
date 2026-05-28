from types import SimpleNamespace

import numpy as np

from src.utils.MO_PaperInstanceProfileUtil import MO_PaperInstanceProfileUtil


def test_o7_uses_raw_pickle_flow_without_symmetric_completion():
    raw = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )
    env = SimpleNamespace(instance="O7", FlowMatrices={"O7": raw}, F=raw + raw.T)

    metadata = MO_PaperInstanceProfileUtil.apply_to_env(env)

    np.testing.assert_array_equal(env.F, raw)
    assert metadata["paperFlowOverrideApplied"] is True
    assert metadata["paperFlowSource"] == "raw_pickle_matrix"


def test_ab20_uses_paper_csv_matrix():
    original = np.zeros((20, 20), dtype=float)
    env = SimpleNamespace(instance="AB20-ar3", FlowMatrices={"AB20-ar3": original}, F=original.copy())

    metadata = MO_PaperInstanceProfileUtil.apply_to_env(env)

    expected, _ = MO_PaperInstanceProfileUtil._load_ab20_1963_matrix()
    np.testing.assert_array_equal(env.F, expected)
    assert metadata["paperFlowOverrideApplied"] is True
    assert metadata["paperFlowSource"] == "csv"
    assert metadata["paperFlowSourcePath"].endswith("/data/AB20(1963).csv")


def test_any_ab20_variant_uses_paper_csv_matrix():
    original = np.zeros((20, 20), dtype=float)
    env = SimpleNamespace(instance="AB20-ar50", FlowMatrices={"AB20-ar50": original}, F=original.copy())

    metadata = MO_PaperInstanceProfileUtil.apply_to_env(env)

    expected, _ = MO_PaperInstanceProfileUtil._load_ab20_1963_matrix()
    np.testing.assert_array_equal(env.F, expected)
    assert metadata["paperFlowOverrideApplied"] is True
    assert metadata["paperFlowSource"] == "csv"


def test_sc_constraints_are_marked_as_partial_profile():
    env = SimpleNamespace(
        instance="SC30",
        FlowMatrices={"SC30": np.zeros((2, 2))},
        F=np.zeros((2, 2)),
        fac_limit_aspect=5.0,
    )

    metadata = MO_PaperInstanceProfileUtil.apply_to_env(env)

    assert metadata["paperFlowOverrideApplied"] is False
    assert metadata["paperConstraintMode"] == "max_aspect_ratio_only"
    assert metadata["paperAspectRatioLimit"] == 5.0
    assert "常用基准口径" in metadata["paperConstraintProfileNote"]
