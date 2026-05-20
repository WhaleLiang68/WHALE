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
            self.assertIn("archive_igd", frame.columns)
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
                        "archive_igd": 0.03,
                        "reference_front_path": "files/expresults/reference_fronts/Du62_global_reference_front.json",
                        "mo_run_id": "run-2",
                    },
                )

            frame = pd.read_csv(Path(tmp_dir) / "Du62-ELP_DRL_MO.csv", encoding="utf-8-sig")
            self.assertEqual(int(frame.loc[0, "pareto_size"]), 5)
            self.assertEqual(frame.loc[0, "pareto_archive_path"], "files/expresults/pareto_archives/b.json")
            self.assertEqual(frame.loc[0, "mo_run_id"], "run-2")
            self.assertAlmostEqual(float(frame.loc[0, "archive_igd"]), 0.03)
            self.assertEqual(
                frame.loc[0, "reference_front_path"],
                "files/expresults/reference_fronts/Du62_global_reference_front.json",
            )

    def test_repair_canonical_legacy_csv_preserves_custom_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "Du62-ELP_DRL_MO4_PAPER.csv"
            frame = pd.DataFrame(
                [
                    {
                        "实例": "Du62",
                        "算法": "ELP_DRL_MO4_PAPER",
                        "日期": "2026-05-16",
                        "迭代次数": 6400,
                        "解": "[]",
                        "适应度值": 0.5,
                        "开始时间": "s",
                        "最快时间": "f",
                        "结束时间": "e",
                        "运行时间（秒）": 10,
                        "最快最佳结果时间（秒）": 3,
                        "宽高比是否满足": True,
                        "gbest更新次数": 7,
                        "备注": "paper",
                        "paper_pr": 0.125,
                        "paper_sp": 1.25,
                    }
                ]
            )
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

            repaired = repair_legacy_mo_result_csv(csv_path)

            self.assertIn("paper_pr", repaired.columns)
            self.assertIn("paper_sp", repaired.columns)
            self.assertAlmostEqual(float(repaired.loc[0, "paper_pr"]), 0.125)
            self.assertAlmostEqual(float(repaired.loc[0, "paper_sp"]), 1.25)

    def test_save_legacy_mo_experiment_result_preserves_custom_columns_across_appends(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(config, "RESULT_PATH", tmp_dir):
                for run_idx, paper_pr in enumerate([0.1, 0.2], start=1):
                    save_legacy_mo_experiment_result(
                        exp_instance="Du62",
                        exp_algorithm="ELP_DRL_MO4_PAPER",
                        exp_iterations=run_idx,
                        exp_solution=[[run_idx]],
                        exp_fitness=0.25,
                        exp_start_time=pd.Timestamp(f"2026-05-16T10:00:0{run_idx}"),
                        exp_fast_time=pd.Timestamp(f"2026-05-16T10:00:0{run_idx}"),
                        exp_end_time=pd.Timestamp(f"2026-05-16T10:00:1{run_idx}"),
                        exp_is_valid_aspect_ratio=True,
                        exp_remark="paper",
                        exp_gbest_updates=run_idx,
                        exp_extra_fields={
                            "paper_pr": paper_pr,
                            "paper_sp": float(run_idx),
                        },
                    )

            frame = pd.read_csv(Path(tmp_dir) / "Du62-ELP_DRL_MO4_PAPER.csv", encoding="utf-8-sig")
            self.assertEqual(list(frame["paper_pr"]), [0.1, 0.2])
            self.assertEqual(list(frame["paper_sp"]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
