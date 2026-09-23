from types import SimpleNamespace

import numpy as np
import pytest

from src.algorithms.ELP_DRL_BiMO4_ParetoH import ELPParetoH


def _solver():
    solver = object.__new__(ELPParetoH)
    solver.pareto_h_beta = 0.35
    solver.pareto_h_tau_start = 0.20
    solver.pareto_h_tau_end = 0.02
    solver.pareto_h_tau_power = 1.25
    solver.pareto_h_grid_bins = 20
    solver.current_progress_ratio = 0.0
    solver.pareto_archive = []
    solver._reset_pareto_h_state()
    return solver


def _solution(mhc, neg_cr):
    return SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([mhc, neg_cr], dtype=float),
    )


def test_history_bias_prefers_less_visited_objective_cell():
    solver = _solver()

    explore_probability, *_ = solver._pareto_h_probability(
        deterioration=0.10,
        current_visits=20,
        candidate_visits=0,
    )
    crowded_probability, *_ = solver._pareto_h_probability(
        deterioration=0.10,
        current_visits=0,
        candidate_visits=20,
    )

    assert explore_probability > crowded_probability
    assert 0.0 < crowded_probability < 1.0
    assert 0.0 < explore_probability <= 1.0


def test_tau_cools_monotonically_from_start_to_end():
    solver = _solver()
    solver.current_progress_ratio = 0.0
    tau_start = solver._pareto_h_tau()
    solver.current_progress_ratio = 0.5
    tau_middle = solver._pareto_h_tau()
    solver.current_progress_ratio = 1.0
    tau_end = solver._pareto_h_tau()

    assert tau_start == pytest.approx(0.20)
    assert tau_start > tau_middle > tau_end
    assert tau_end == pytest.approx(0.02)


def test_max_deterioration_only_counts_worsened_objectives():
    solver = _solver()
    solver.pareto_h_reference_ideal = np.asarray([900.0, -60.0])
    solver.pareto_h_reference_span = np.asarray([100.0, 10.0])

    current = _solution(1000.0, -50.0)
    candidate = _solution(1020.0, -55.0)

    # MHC 变差 20/100，CR 对应的最小化目标改善 5/10，因此只取 0.2。
    assert solver._pareto_h_objective_deterioration(current, candidate) == pytest.approx(0.20)


def test_reference_frame_is_frozen_and_grid_is_not_clipped():
    solver = _solver()
    first = _solution(100.0, -10.0)
    second = _solution(200.0, -20.0)

    assert solver._try_initialize_pareto_h_reference(first, second)
    original_ideal = solver.pareto_h_reference_ideal.copy()
    original_span = solver.pareto_h_reference_span.copy()

    outside = _solution(400.0, -40.0)
    assert solver._try_initialize_pareto_h_reference(outside)
    assert np.array_equal(solver.pareto_h_reference_ideal, original_ideal)
    assert np.array_equal(solver.pareto_h_reference_span, original_span)
    assert solver._pareto_h_grid_key(outside) == (60, -40)


def test_reference_waits_until_both_objectives_have_real_span():
    solver = _solver()
    first = _solution(100.0, 0.0)
    second = _solution(200.0, 0.0)

    assert not solver._try_initialize_pareto_h_reference(first, second)
    assert solver.pareto_h_reference_ideal is None
    assert solver._pareto_h_warmup_deterioration(first, second) == pytest.approx(0.5)
