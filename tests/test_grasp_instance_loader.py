import numpy as np

from src.utils.GRASPInstanceLoader import GRASPInstanceLoader


def test_a_10_10_loader_returns_expected_shapes():
    payload = GRASPInstanceLoader.load_instance("A-10-10")
    assert payload["rows"] == 2
    assert payload["cols"] == 5
    assert payload["n"] == 10
    assert payload["mhc_matrix"].shape == (10, 10)
    assert payload["cr_matrix"].shape == (10, 10)
    assert np.allclose(payload["mhc_matrix"], payload["mhc_matrix"].T)
    assert np.allclose(payload["cr_matrix"], payload["cr_matrix"].T)
