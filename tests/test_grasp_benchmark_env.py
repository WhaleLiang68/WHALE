import numpy as np
import pytest

from src.utils.FBSModel import FBSModel
from src.utils.GRASPBenchmarkEnv import GRASPBenchmarkEnv


def test_grasp_env_evaluates_identity_layout():
    env = GRASPBenchmarkEnv(instance="A-10-10")
    model = FBSModel(permutation=list(range(1, 11)), bay=env._build_fixed_bay().tolist())
    metrics = env.evaluate_fbs_model(model)

    assert metrics["mhc"] >= 0.0
    assert np.isfinite(metrics["cr"])
    assert metrics["D"].shape == (10, 10)
    assert np.allclose(metrics["D"], metrics["D"].T)
    assert metrics["is_feasible"] is True


def test_grasp_env_rejects_invalid_permutation():
    env = GRASPBenchmarkEnv(instance="A-10-10")
    model = FBSModel(permutation=[1, 1, 3, 4, 5, 6, 7, 8, 9, 10], bay=env._build_fixed_bay().tolist())

    with pytest.raises(ValueError, match="非法排列"):
        env.evaluate_fbs_model(model)
