import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from src.dashboard import server


class TestDashboardMOResults(unittest.TestCase):
    def test_load_results_reads_chinese_headers_and_mo_fields(self):
        repo_tmp_root = Path('files') / '_test_tmp'
        repo_tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=repo_tmp_root)
        try:
            csv_path = Path(tmp_dir) / 'Du62-ELP_DRL_MO.csv'
            with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        '实例', '算法', '日期', '迭代次数', '解', '适应度值', '开始时间', '最快时间', '结束时间',
                        '运行时间（秒）', '最快最佳结果时间（秒）', '宽高比是否满足', 'gbest更新次数', '备注',
                        'pareto_size', 'pareto_archive_path', 'rep_mhc', 'rep_cr', 'rep_dr', 'rep_ar', 'decision_score',
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    '实例': 'Du62',
                    '算法': 'ELP_DRL_MO',
                    '日期': '2026-03-27',
                    '迭代次数': '300000',
                    '解': '[[1, 2], [3, 4]]',
                    '适应度值': '0.42',
                    '开始时间': '2026-03-27T10:00:00',
                    '最快时间': '2026-03-27T10:05:00',
                    '结束时间': '2026-03-27T10:10:00',
                    '运行时间（秒）': '600',
                    '最快最佳结果时间（秒）': '300',
                    '宽高比是否满足': 'True',
                    'gbest更新次数': '7',
                    '备注': 'mo-test',
                    'pareto_size': '3',
                    'pareto_archive_path': 'files/expresults/pareto_archives/sample.json',
                    'rep_mhc': '12.5',
                    'rep_cr': '8.1',
                    'rep_dr': '7.2',
                    'rep_ar': '0.95',
                    'decision_score': '0.42',
                })

            payload = server.load_results(csv_path)
            row = payload['rows'][0]
            self.assertEqual(row['instance'], 'Du62')
            self.assertEqual(row['algorithm'], 'ELP_DRL_MO')
            self.assertEqual(row['paretoArchivePath'], 'files/expresults/pareto_archives/sample.json')
            self.assertEqual(row['paretoSize'], 3)
            self.assertAlmostEqual(row['repMhc'], 12.5)
            self.assertAlmostEqual(row['repCr'], 8.1)
            self.assertAlmostEqual(row['repDr'], 7.2)
            self.assertAlmostEqual(row['repAr'], 0.95)
            self.assertAlmostEqual(row['decisionScore'], 0.42)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
