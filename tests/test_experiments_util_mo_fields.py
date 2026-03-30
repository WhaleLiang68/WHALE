import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.utils.config as config
from src.utils.ExperimentsUtil import save_experiment_result


class TestExperimentsUtilMOFields(unittest.TestCase):
    def test_save_experiment_result_writes_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            start = datetime.datetime(2026, 3, 26, 10, 0, 0)
            fast = start + datetime.timedelta(seconds=3)
            end = start + datetime.timedelta(seconds=7)
            with patch.object(config, "RESULT_PATH", tmp_dir):
                save_experiment_result(
                    exp_instance="AB20",
                    exp_algorithm="ELP_DRL_MO",
                    exp_iterations=10,
                    exp_solution=[[1, 2], [3, 4]],
                    exp_fitness=0.42,
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_is_valid_aspect_ratio=True,
                    exp_remark="mo-test",
                    exp_gbest_updates=2,
                    exp_extra_fields={
                        "pareto_size": 3,
                        "pareto_archive_path": "files/expresults/pareto_archives/sample.json",
                        "rep_mhc": 12.5,
                        "rep_cr": 8.0,
                        "rep_dr": 6.0,
                        "rep_ar": 0.9,
                        "decision_score": 0.42,
                    },
                )

            output_path = Path(tmp_dir) / "AB20-ELP_DRL_MO.csv"
            self.assertTrue(output_path.exists())
            frame = pd.read_csv(output_path, encoding="utf-8-sig")
            self.assertIn("pareto_size", frame.columns)
            self.assertIn("pareto_archive_path", frame.columns)
            self.assertEqual(int(frame.loc[0, "pareto_size"]), 3)
            self.assertAlmostEqual(float(frame.loc[0, "rep_mhc"]), 12.5)
            self.assertEqual(frame.loc[0, "pareto_archive_path"], "files/expresults/pareto_archives/sample.json")


if __name__ == "__main__":
    unittest.main()
