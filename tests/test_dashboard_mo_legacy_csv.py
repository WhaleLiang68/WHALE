import unittest

from src.dashboard import server


class DashboardMoLegacyCsvTests(unittest.TestCase):
    def test_real_multi_objective_csv_last_row_is_parsed(self):
        csv_path = server.REPO_ROOT / "files" / "expresults" / "Du62-ELP_DRL_MO.csv"
        payload = server.load_results(csv_path)
        row = payload["rows"][-1]

        self.assertEqual(row["runIndex"], 14)
        self.assertEqual(row["instance"], "Du62")
        self.assertEqual(row["algorithm"], "ELP_DRL_MO")
        self.assertEqual(row["date"], "2026-03-28")
        self.assertAlmostEqual(row["decisionScore"], 0.11551480985738258)
        self.assertAlmostEqual(row["runtimeSeconds"], 65706.757452)
        self.assertEqual(row["gbestUpdates"], 231)
        self.assertEqual(row["paretoSize"], 64)
        self.assertEqual(row["paretoArchivePath"], "files/expresults/pareto_archives/Du62-ELP_DRL_MO-20260327_215305_026163.json")
        self.assertAlmostEqual(row["repMhc"], 8714638.484496523)
        self.assertAlmostEqual(row["repCr"], 7160.882812227182)
        self.assertAlmostEqual(row["repDr"], 386593.5917597896)
        self.assertAlmostEqual(row["repAr"], 1.0)


if __name__ == "__main__":
    unittest.main()
