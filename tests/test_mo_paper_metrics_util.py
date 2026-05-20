import pytest

from src.utils.MO_PaperMetricsUtil import MO_PaperMetricsUtil


def test_paper_metrics_match_formula_shapes():
    all_points = [
        [10.0, 1.0],
        [9.0, 2.0],
        [8.0, 1.0],
        [7.0, 0.5],
    ]
    pareto = MO_PaperMetricsUtil.pareto_front(all_points)
    summary = MO_PaperMetricsUtil.calculate_summary(all_points, pareto)

    assert summary["paper_solution_count"] == 4
    assert summary["paper_pareto_count"] == 3
    assert summary["paper_pr"] == pytest.approx(0.75)
    assert summary["paper_sp"] >= 0.0
    assert summary["paper_sp_raw"] == pytest.approx(summary["paper_sp"])
    assert summary["paper_sp_norm"] >= 0.0
    assert summary["paper_ops"] == pytest.approx((2.0 / 3.0) * 1.0)


def test_paper_metrics_handle_singleton_front():
    summary = MO_PaperMetricsUtil.calculate_summary([[5.0, 2.0]])

    assert summary["paper_pr"] == pytest.approx(1.0)
    assert summary["paper_sp"] == pytest.approx(0.0)
    assert summary["paper_sp_raw"] == pytest.approx(0.0)
    assert summary["paper_sp_norm"] == pytest.approx(0.0)
    assert summary["paper_ops"] == pytest.approx(0.0)


def test_normalized_spacing_removes_objective_scale_bias():
    all_points = [
        [1_000_000.0, 1.0],
        [1_100_000.0, 2.0],
        [1_300_000.0, 3.0],
    ]
    summary = MO_PaperMetricsUtil.calculate_summary(all_points)

    assert summary["paper_sp_raw"] > 10_000.0
    assert 0.0 <= summary["paper_sp_norm"] <= 1.0


def test_pr_counts_duplicate_objective_points_as_distinct_solutions():
    all_points = [
        [10.0, 1.0],
        [10.0, 1.0],
        [11.0, 0.5],
    ]
    summary = MO_PaperMetricsUtil.calculate_summary(all_points)

    assert summary["paper_solution_count"] == 3
    assert summary["paper_pareto_count"] == 2
    assert summary["paper_pr"] == pytest.approx(2.0 / 3.0)
