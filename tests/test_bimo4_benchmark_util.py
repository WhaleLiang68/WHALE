import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.BiMO4BenchmarkUtil import build_reference_fronts
from src.utils.BiMO4BenchmarkUtil import compute_unified_metrics
from src.utils.BiMO4BenchmarkUtil import filter_nondominated
from src.utils.BiMO4BenchmarkUtil import parse_benchmark_remark


class TestBiMO4BenchmarkUtil(unittest.TestCase):
    def test_parse_benchmark_remark(self):
        payload = parse_benchmark_remark(
            "benchmark_id=bimo4_compare_v1; budget_seconds=1800; seed=20260526; phase=main"
        )
        self.assertEqual(payload["benchmark_id"], "bimo4_compare_v1")
        self.assertEqual(payload["budget_seconds"], "1800")
        self.assertEqual(payload["seed"], "20260526")
        self.assertEqual(payload["phase"], "main")

    def test_filter_nondominated_2d(self):
        points = np.asarray(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [3.5, 3.5],
            ],
            dtype=float,
        )
        filtered = filter_nondominated(points)
        self.assertEqual(filtered.shape[0], 3)
        self.assertFalse(np.any(np.all(np.isclose(filtered, [3.5, 3.5]), axis=1)))

    def test_build_reference_fronts_and_unified_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            archive_dir = repo_root / "files" / "expresults" / "pareto_archives"
            archive_dir.mkdir(parents=True, exist_ok=True)

            archive_a = archive_dir / "Du62-ELP_DRL_BiMO4-a.json"
            archive_a.write_text(
                json.dumps(
                    {
                        "instance": "Du62",
                        "algorithm": "ELP_DRL_BiMO4",
                        "items": [
                            {"isFeasible": True, "moObjectivesMin": [1.0, -4.0], "mhc": 1.0, "cr": 4.0},
                            {"isFeasible": True, "moObjectivesMin": [2.0, -5.0], "mhc": 2.0, "cr": 5.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            archive_b = archive_dir / "Du62-MO_BASELINE_NSGA2-b.json"
            archive_b.write_text(
                json.dumps(
                    {
                        "instance": "Du62",
                        "algorithm": "MO_BASELINE_NSGA2",
                        "items": [
                            {"isFeasible": True, "moObjectivesMin": [1.5, -4.5], "mhc": 1.5, "cr": 4.5},
                            {"isFeasible": True, "moObjectivesMin": [3.0, -3.0], "mhc": 3.0, "cr": 3.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            run_frame = pd.DataFrame(
                [
                    {
                        "instance": "Du62",
                        "budget_seconds": 1800,
                        "algorithm": "ELP_DRL_BiMO4",
                        "seed": 1,
                        "phase": "main",
                        "benchmark_id": "bench",
                        "pareto_archive_path": archive_a.relative_to(repo_root).as_posix(),
                        "rep_mhc": 1.0,
                        "rep_cr": 4.0,
                    },
                    {
                        "instance": "Du62",
                        "budget_seconds": 1800,
                        "algorithm": "MO_BASELINE_NSGA2",
                        "seed": 1,
                        "phase": "main",
                        "benchmark_id": "bench",
                        "pareto_archive_path": archive_b.relative_to(repo_root).as_posix(),
                        "rep_mhc": 1.5,
                        "rep_cr": 4.5,
                    },
                ]
            )

            reference_fronts = build_reference_fronts(
                run_frame,
                repo_root=repo_root,
                output_dir=repo_root / "reference_fronts",
                benchmark_id="bench",
            )
            payload = reference_fronts[("Du62", 1800)]
            self.assertEqual(payload["pointCount"], 3)

            unified = compute_unified_metrics(run_frame, repo_root=repo_root, reference_fronts=reference_fronts)
            self.assertEqual(len(unified), 2)
            self.assertIn("hv_ref_front", unified.columns)
            self.assertTrue((pd.to_numeric(unified["coverage_ref_to_s"], errors="coerce") >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
