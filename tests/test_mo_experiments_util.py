import datetime
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.MO_ExperimentsUtil import MOExperimentRecorder
from src.utils.MO_ExperimentsUtil import export_mo_analysis_tables
from src.utils.MO_ExperimentsUtil import load_mo_action_stats_frame
from src.utils.MO_ExperimentsUtil import load_mo_event_frame
from src.utils.MO_ExperimentsUtil import load_mo_run_summary_frame
from src.utils.MO_ExperimentsUtil import load_mo_trace_frame


class TestMOExperimentsUtil(unittest.TestCase):
    def test_recorder_writes_bundle_and_exports_flat_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_root = Path(tmp_dir) / 'files' / 'expresults'
            start_time = datetime.datetime(2026, 3, 29, 9, 0, 0)
            recorder = MOExperimentRecorder(
                instance='Du62',
                algorithm='ELP_DRL_MO',
                start_time=start_time,
                trace_interval=1000,
                remark='unit-test',
                result_root=result_root,
            )

            recorder.record_trace({'globalStep': 1000, 'archiveSize': 3, 'decisionScore': 0.2})
            recorder.record_event('representative_update', {'globalStep': 800, 'decisionScore': 0.21})
            action_stats = {
                'meta': {'algorithm': 'ELP_DRL_MO'},
                'overall': {'steps': 1000, 'accepted': 120},
                'actions': {
                    '1': {'name': 'swap', 'selected': 50, 'accepted': 10},
                    '10': {'name': 'fast_segment_insert', 'selected': 20, 'accepted': 5},
                },
            }
            run_summary = {
                'startTime': start_time.isoformat(),
                'endTime': (start_time + datetime.timedelta(seconds=5)).isoformat(),
                'runtimeSeconds': 5.0,
                'iterations': 1000,
                'decisionScore': 0.2,
                'archiveSize': 3,
                'repMhc': 10.0,
                'repCr': 4.0,
                'repDr': 2.0,
                'repAr': 1.0,
            }
            finalized = recorder.finalize(run_summary, action_stats)

            self.assertIn('runId', finalized)
            self.assertTrue((result_root.parents[1] / finalized['tracePath']).exists())
            self.assertTrue((result_root.parents[1] / finalized['eventsPath']).exists())
            self.assertTrue((result_root.parents[1] / finalized['actionStatsPath']).exists())
            self.assertTrue((result_root.parents[1] / finalized['runSummaryPath']).exists())

            summary = load_mo_run_summary_frame(result_root=result_root, instance='Du62', algorithm='ELP_DRL_MO')
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary.loc[0, 'decisionScore']), 0.2)

            trace = load_mo_trace_frame(result_root.parents[1] / finalized['tracePath'])
            events = load_mo_event_frame(result_root.parents[1] / finalized['eventsPath'])
            action_frame = load_mo_action_stats_frame(result_root.parents[1] / finalized['actionStatsPath'])
            self.assertEqual(len(trace), 1)
            self.assertEqual(len(events), 1)
            self.assertEqual(set(action_frame['actionIdx']), {1, 10})

            outputs = export_mo_analysis_tables(result_root=result_root, instance='Du62', algorithm='ELP_DRL_MO')
            self.assertTrue(Path(outputs['summary']).exists())
            self.assertTrue(Path(outputs['trace']).exists())
            self.assertTrue(Path(outputs['events']).exists())
            self.assertTrue(Path(outputs['action_stats']).exists())
            exported_trace = pd.read_csv(outputs['trace'], encoding='utf-8-sig')
            self.assertEqual(len(exported_trace), 1)

    def test_recorder_summary_csv_can_expand_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_root = Path(tmp_dir) / 'files' / 'expresults'
            start_time = datetime.datetime(2026, 5, 10, 9, 0, 0)

            recorder_a = MOExperimentRecorder(
                instance='AB20-ar3',
                algorithm='ELP_DRL_MO',
                start_time=start_time,
                result_root=result_root,
            )
            recorder_a.finalize(
                {
                    'startTime': start_time.isoformat(),
                    'endTime': start_time.isoformat(),
                    'runtimeSeconds': 1.0,
                    'decisionScore': 0.2,
                },
                {'meta': {}, 'overall': {}, 'actions': {}},
            )

            recorder_b = MOExperimentRecorder(
                instance='AB20-ar3',
                algorithm='ELP_DRL_MO',
                start_time=start_time + datetime.timedelta(seconds=5),
                result_root=result_root,
            )
            recorder_b.finalize(
                {
                    'startTime': (start_time + datetime.timedelta(seconds=5)).isoformat(),
                    'endTime': (start_time + datetime.timedelta(seconds=6)).isoformat(),
                    'runtimeSeconds': 1.0,
                    'decisionScore': 0.1,
                    'archiveIgd': 0.03,
                    'referenceFrontPath': 'files/expresults/reference_fronts/AB20-ar3_global_reference_front.json',
                },
                {'meta': {}, 'overall': {}, 'actions': {}},
            )

            summary = load_mo_run_summary_frame(result_root=result_root, instance='AB20-ar3', algorithm='ELP_DRL_MO')
            self.assertEqual(len(summary), 2)
            self.assertIn('archiveIgd', summary.columns)
            self.assertAlmostEqual(float(summary.iloc[1]['archiveIgd']), 0.03)


if __name__ == '__main__':
    unittest.main()
