import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.utils.config as config
from src.utils.MO_ExperimentsUtil import repair_legacy_mo_result_csv
from src.utils.MO_ExperimentsUtil import save_legacy_mo_experiment_result


class TestMOLegacyCsvUtil(unittest.TestCase):
    def test_repair_legacy_csv_realigns_pareto_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "Du62-ELP_DRL_MO.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "??", "??", "??", "????", "?", "????", "????", "????", "????",
                    "???????", "???????????", "???????", "gbest????", "??",
                    "pareto_size", "pareto_archive_path", "rep_mhc", "rep_cr", "rep_dr", "rep_ar", "decision_score",
                ])
                writer.writerow([
                    "Du62", "ELP_DRL_MO", "2026-03-28", "300000", "[]", "0.42", "s", "f", "e",
                    "10", "3", "True", "7", "remark", "files/expresults/pareto_archives/a.json", "64",
                    "12.5", "8.1", "7.2", "1.0", "0.42", "run-1", "bundle", "trace", "events", "actions", "summary",
                ])

            frame = repair_legacy_mo_result_csv(csv_path)
            self.assertIn("pareto_size", frame.columns)
            self.assertIn("mo_run_id", frame.columns)
            self.assertEqual(int(frame.loc[0, "pareto_size"]), 64)
            self.assertEqual(frame.loc[0, "pareto_archive_path"], "files/expresults/pareto_archives/a.json")
            self.assertEqual(frame.loc[0, "mo_run_id"], "run-1")

    def test_save_legacy_mo_experiment_result_appends_canonical_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(config, "RESULT_PATH", tmp_dir):
                save_legacy_mo_experiment_result(
                    exp_instance="Du62",
                    exp_algorithm="ELP_DRL_MO",
                    exp_iterations=10,
                    exp_solution=[[1, 2]],
                    exp_fitness=0.25,
                    exp_start_time=pd.Timestamp("2026-03-30T10:00:00"),
                    exp_fast_time=pd.Timestamp("2026-03-30T10:00:05"),
                    exp_end_time=pd.Timestamp("2026-03-30T10:00:10"),
                    exp_is_valid_aspect_ratio=True,
                    exp_remark="test",
                    exp_gbest_updates=3,
                    exp_extra_fields={
                        "pareto_archive_path": "files/expresults/pareto_archives/b.json",
                        "pareto_size": 5,
                        "rep_mhc": 100.0,
                        "rep_cr": 50.0,
                        "rep_dr": 25.0,
                        "rep_ar": 1.0,
                        "decision_score": 0.25,
                        "mo_run_id": "run-2",
                    },
                )

            frame = pd.read_csv(Path(tmp_dir) / "Du62-ELP_DRL_MO.csv", encoding="utf-8-sig")
            self.assertEqual(int(frame.loc[0, "pareto_size"]), 5)
            self.assertEqual(frame.loc[0, "pareto_archive_path"], "files/expresults/pareto_archives/b.json")
            self.assertEqual(frame.loc[0, "mo_run_id"], "run-2")


if __name__ == "__main__":
    unittest.main()
