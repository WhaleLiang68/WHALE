import json

import numpy as np
import pytest

from src.utils.BO_MREFLPBenchmark import BO_MREFLPBenchmark


def _write_points(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{left} {right}" for left, right in rows), encoding="utf-8")


def test_build_benchmark_uses_all_points_for_ideal_and_nadir(tmp_path):
    results_root = tmp_path / "GRASP_Results"
    instance_name = "A-10-10"

    _write_points(results_root / "0" / "0" / f"{instance_name}.txt", [(1, 5), (3, 1)])
    _write_points(results_root / "0" / "1" / f"{instance_name}.txt", [(5, 10)])
    _write_points(results_root / "0" / "NSBBO" / f"{instance_name}.txt", [(4, 0.5)])
    _write_points(results_root / "0" / "NSGA-II" / f"{instance_name}.txt", [(2, 4)])

    payload = BO_MREFLPBenchmark.build_benchmark_package(
        instance_name=instance_name,
        results_root=results_root,
        benchmark_root=tmp_path / "benchmark",
        include_algorithms=["GRASP1", "GRASP2", "NSBBO", "NSGA-II"],
    )

    normalization = payload["normalization_payload"]
    reference = payload["reference_payload"]

    assert normalization["ideal"] == pytest.approx([1.0, 0.5])
    assert normalization["nadir"] == pytest.approx([5.0, 10.0])
    assert normalization["hv_ref_point"] == pytest.approx([1.1, 1.1])
    assert np.asarray(reference["reference_front"], dtype=float) == pytest.approx(
        np.asarray([[1.0, 5.0], [2.0, 4.0], [3.0, 1.0], [4.0, 0.5]], dtype=float)
    )


def test_evaluate_points_loads_saved_benchmark_package(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "reference_front").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "normalization").mkdir(parents=True, exist_ok=True)

    instance_name = "A-10-10"
    reference_payload = {
        "schema_version": BO_MREFLPBenchmark.REFERENCE_SCHEMA_VERSION,
        "instance": instance_name,
        "reference_front": [[0.0, 1.0], [1.0, 0.0]],
        "reference_front_size": 2,
    }
    normalization_payload = {
        "schema_version": BO_MREFLPBenchmark.NORMALIZATION_SCHEMA_VERSION,
        "instance": instance_name,
        "ideal": [0.0, 0.0],
        "nadir": [1.0, 1.0],
        "hv_ref_point": [1.1, 1.1],
    }
    (benchmark_root / "reference_front" / f"{instance_name}.json").write_text(
        json.dumps(reference_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (benchmark_root / "normalization" / f"{instance_name}.json").write_text(
        json.dumps(normalization_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = BO_MREFLPBenchmark.evaluate_points(
        instance_name=instance_name,
        points=[[0.0, 1.0], [1.0, 0.0]],
        benchmark_root=benchmark_root,
    )

    assert metrics["reference_front_size"] == 2
    assert metrics["gd"] == pytest.approx(0.0)
    assert metrics["igd"] == pytest.approx(0.0)
    assert metrics["igd_plus"] == pytest.approx(0.0)
    assert metrics["epsilon_multiplicative"] == pytest.approx(1.0)
    assert metrics["coverage_ref_to_s"] == pytest.approx(1.0)
