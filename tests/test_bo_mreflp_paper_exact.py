import numpy as np

from src.utils.BO_MREFLPPaperExact import BO_MREFLPPaperExact


def test_pareto_aux_front_matches_java_tolerance_behavior():
    points = np.asarray(
        [
            [10.0, 80.0],
            [10.00005, 80.00005],
            [9.0, 85.0],
            [8.0, 90.0],
        ],
        dtype=float,
    )

    front = BO_MREFLPPaperExact.pareto_aux_front(points)

    assert front.shape == (3, 2)
    assert any(np.array_equal(row, np.asarray([10.0, 80.0], dtype=float)) for row in front)
    assert not any(np.array_equal(row, np.asarray([10.00005, 80.00005], dtype=float)) for row in front)


def test_archive_audit_passes_real_a1010_archive():
    archive_path = (
        r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\expresults\pareto_archives"
        r"\A-10-10-ELP_DRL_MO4_GRASP-20260519_212908_323799.json"
    )

    metrics = BO_MREFLPPaperExact.evaluate_archive(
        instance_name="A-10-10",
        archive_path=archive_path,
        save_report=False,
    )

    assert metrics["paper_exact_all_legal"] is True
    assert metrics["paper_exact_all_objectives_match"] is True
    assert metrics["paper_exact_candidate_point_count"] > 0
    assert metrics["paper_exact_reference_front_size"] > 0
