from src.algorithms.ELP_DRL_MO4_Paper import ELP


def _build_solver_stub():
    solver = object.__new__(ELP)
    solver.paper_final_window_ratio = 0.10
    solver.paper_final_window_start_progress = 0.90
    solver.paper_solution_set_mode = "terminal_window"
    solver._paper_collect_terminal_population = False
    solver.current_progress_ratio = 0.0
    solver._paper_total_steps = 0
    solver._trace_global_step = 0
    return solver


def test_terminal_window_mode_uses_progress_ratio():
    solver = _build_solver_stub()

    solver.current_progress_ratio = 0.89
    assert solver._is_in_paper_final_window() is False

    solver.current_progress_ratio = 0.90
    assert solver._is_in_paper_final_window() is True


def test_final_population_mode_uses_explicit_collection_flag():
    solver = _build_solver_stub()
    solver.paper_solution_set_mode = "final_population"
    solver.current_progress_ratio = 1.0

    assert solver._is_in_paper_final_window() is False

    solver._paper_collect_terminal_population = True
    assert solver._is_in_paper_final_window() is True
