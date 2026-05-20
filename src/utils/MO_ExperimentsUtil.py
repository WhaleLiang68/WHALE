import csv
import json
import math
import re
from pathlib import Path

import pandas as pd

import src.utils.config as config


BENCHMARK_COLUMN_DISPLAY_MAP = {
    'archive_hypervolume': 'HV',
    'benchmark_gd': 'GD',
    'archive_igd': 'IGD',
    'benchmark_igd_plus': 'IGD+',
    'archive_spacing': 'Spread(Δ)',
    'benchmark_coverage_ref_to_s': 'C(Ref, S)',
    'benchmark_epsilon_multiplicative': 'multiplicative ε',
}


def _sanitize_filename(name):
    return re.sub(r'[\/*?:"<>|]', '', str(name or 'UNKNOWN'))


def _normalize_value(value):
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _dedupe_columns(columns):
    seen = {}
    deduped = []
    for column in columns:
        base = str(column or 'unnamed')
        count = seen.get(base, 0)
        seen[base] = count + 1
        deduped.append(base if count == 0 else f'{base}__{count + 1}')
    return deduped


def _display_column_name(column_name):
    return BENCHMARK_COLUMN_DISPLAY_MAP.get(str(column_name), str(column_name))


def _migrate_benchmark_display_columns(frame):
    if frame is None or frame.empty and len(frame.columns) == 0:
        return frame
    for legacy_name, display_name in BENCHMARK_COLUMN_DISPLAY_MAP.items():
        if legacy_name not in frame.columns:
            continue
        if display_name in frame.columns:
            frame[display_name] = frame[display_name].where(frame[display_name].notna(), frame[legacy_name])
            frame = frame.drop(columns=[legacy_name])
        else:
            frame = frame.rename(columns={legacy_name: display_name})
    return frame


def _read_csv_preserve_malformed_rows(csv_path):
    """保留历史坏行的额外字段，修复变长 CSV 后再追加新结果。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame()
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return pd.DataFrame()

    header = _dedupe_columns(rows[0])
    max_len = max(len(row) for row in rows)
    extra_count = max(0, max_len - len(header))
    columns = header + [f'legacy_extra_{idx + 1:03d}' for idx in range(extra_count)]

    records = []
    for raw_row in rows[1:]:
        if not raw_row or not any(str(cell).strip() for cell in raw_row):
            continue
        padded = list(raw_row) + [None] * max(0, len(columns) - len(raw_row))
        records.append({column: padded[idx] if idx < len(padded) else None for idx, column in enumerate(columns)})
    return pd.DataFrame(records, columns=columns)


def _append_row_csv(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_frame = pd.DataFrame([_normalize_value(row)])
    row_frame = _migrate_benchmark_display_columns(row_frame)
    if not csv_path.exists():
        row_frame.to_csv(
            csv_path,
            index=False,
            mode='w',
            header=True,
            encoding='utf-8-sig',
        )
        return

    # 追加字段时需要重写整张表，避免把新列直接拼到旧 CSV 末尾导致列错位。
    try:
        frame = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        frame = _read_csv_preserve_malformed_rows(csv_path)
    frame = _migrate_benchmark_display_columns(frame)
    columns = list(frame.columns)
    for column in row_frame.columns:
        if column not in columns:
            columns.append(column)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
        if column not in row_frame.columns:
            row_frame[column] = None
    frame = frame[columns]
    row_frame = row_frame[columns]
    if frame.empty:
        frame = row_frame.copy()
    else:
        frame.loc[len(frame)] = row_frame.iloc[0]
    frame.to_csv(csv_path, index=False, encoding='utf-8-sig')


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(_normalize_value(row), ensure_ascii=False))
            handle.write('\n')


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class MOExperimentRecorder:
    def __init__(
        self,
        instance,
        algorithm,
        start_time,
        trace_interval=1000,
        remark='',
        result_root=None,
    ):
        self.instance = str(instance)
        self.algorithm = str(algorithm)
        self.start_time = start_time
        self.remark = str(remark or '')
        self.trace_interval = int(max(1, trace_interval))
        self.result_root = Path(result_root or config.RESULT_PATH)
        self.repo_root = self.result_root.resolve().parents[1]

        timestamp = (
            start_time.strftime('%Y%m%d_%H%M%S_%f')
            if start_time is not None
            else pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')
        )
        self.run_id = f"{_sanitize_filename(self.instance)}-{_sanitize_filename(self.algorithm)}-{timestamp}"
        self.bundle_dir = self.result_root / 'mo_runs' / self.run_id
        self.summary_csv_path = (
            self.result_root
            / 'mo_runs_summary'
            / f"{_sanitize_filename(self.instance)}-{_sanitize_filename(self.algorithm)}.csv"
        )

        self.trace_records = []
        self.event_records = []

    def should_record_trace(self, global_step, total_steps=None):
        global_step = int(max(0, global_step))
        if global_step == 0:
            return False
        if global_step % self.trace_interval == 0:
            return True
        if total_steps is not None and int(global_step) >= int(total_steps):
            return True
        return False

    def record_trace(self, payload):
        self.trace_records.append(_normalize_value(payload))

    def record_event(self, event_type, payload):
        record = {'eventType': str(event_type)}
        if payload:
            record.update(_normalize_value(payload))
        self.event_records.append(record)

    def finalize(self, run_summary, action_stats):
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        trace_path = self.bundle_dir / 'trace.jsonl'
        events_path = self.bundle_dir / 'events.jsonl'
        action_stats_path = self.bundle_dir / 'action_stats.json'
        run_summary_path = self.bundle_dir / 'run_summary.json'

        _write_jsonl(trace_path, self.trace_records)
        _write_jsonl(events_path, self.event_records)
        action_stats_path.write_text(
            json.dumps(_normalize_value(action_stats), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        run_summary = dict(_normalize_value(run_summary) or {})
        run_summary.update(
            {
                'runId': self.run_id,
                'instance': self.instance,
                'algorithm': self.algorithm,
                'remark': self.remark,
                'traceInterval': self.trace_interval,
                'traceCount': len(self.trace_records),
                'eventCount': len(self.event_records),
                'bundleDir': self.bundle_dir.resolve().relative_to(self.repo_root).as_posix(),
                'tracePath': trace_path.resolve().relative_to(self.repo_root).as_posix(),
                'eventsPath': events_path.resolve().relative_to(self.repo_root).as_posix(),
                'actionStatsPath': action_stats_path.resolve().relative_to(self.repo_root).as_posix(),
                'runSummaryPath': run_summary_path.resolve().relative_to(self.repo_root).as_posix(),
            }
        )

        run_summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        _append_row_csv(self.summary_csv_path, run_summary)
        return run_summary


def load_mo_run_summary_frame(result_root=None, instance=None, algorithm=None):
    result_root = Path(result_root or config.RESULT_PATH)
    summary_dir = result_root / 'mo_runs_summary'
    frames = []
    for csv_path in sorted(summary_dir.glob('*.csv')):
        try:
            frame = pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception:
            frame = _read_csv_preserve_malformed_rows(csv_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    if instance is not None and 'instance' in summary.columns:
        summary = summary[summary['instance'].astype(str) == str(instance)]
    if algorithm is not None and 'algorithm' in summary.columns:
        summary = summary[summary['algorithm'].astype(str) == str(algorithm)]
    if 'startTime' in summary.columns:
        summary = summary.sort_values('startTime').reset_index(drop=True)
    return summary


def load_mo_trace_frame(trace_path):
    return pd.DataFrame(_read_jsonl(trace_path))


def load_mo_event_frame(events_path):
    return pd.DataFrame(_read_jsonl(events_path))


def load_mo_action_stats_frame(action_stats_path):
    payload = json.loads(Path(action_stats_path).read_text(encoding='utf-8'))
    meta = dict(payload.get('meta', {}) or {})
    overall = dict(payload.get('overall', {}) or {})
    rows = []
    for action_idx, stats in (payload.get('actions', {}) or {}).items():
        row = dict(meta)
        row.update(overall)
        row['actionIdx'] = int(action_idx)
        row.update(dict(stats or {}))
        rows.append(_normalize_value(row))
    return pd.DataFrame(rows)


def export_mo_analysis_tables(
    result_root=None,
    instance=None,
    algorithm=None,
    run_id=None,
    export_dir=None,
):
    result_root = Path(result_root or config.RESULT_PATH)
    export_dir = Path(export_dir or (result_root / 'mo_analysis'))
    export_dir.mkdir(parents=True, exist_ok=True)

    repo_root = result_root.resolve().parents[1]
    summary = load_mo_run_summary_frame(result_root=result_root, instance=instance, algorithm=algorithm)
    if run_id is not None and not summary.empty and 'runId' in summary.columns:
        summary = summary[summary['runId'].astype(str) == str(run_id)]
    summary = summary.reset_index(drop=True)

    trace_frames = []
    event_frames = []
    action_frames = []

    for _, row in summary.iterrows():
        row_dict = row.to_dict()
        trace_path = row_dict.get('tracePath')
        events_path = row_dict.get('eventsPath')
        action_stats_path = row_dict.get('actionStatsPath')
        join_cols = {
            'runId': row_dict.get('runId'),
            'instance': row_dict.get('instance'),
            'algorithm': row_dict.get('algorithm'),
            'startTime': row_dict.get('startTime'),
        }

        if trace_path:
            trace_frame = load_mo_trace_frame(repo_root / trace_path if not Path(trace_path).is_absolute() else trace_path)
            if not trace_frame.empty:
                for key, value in join_cols.items():
                    trace_frame[key] = value
                trace_frames.append(trace_frame)

        if events_path:
            event_frame = load_mo_event_frame(repo_root / events_path if not Path(events_path).is_absolute() else events_path)
            if not event_frame.empty:
                for key, value in join_cols.items():
                    event_frame[key] = value
                event_frames.append(event_frame)

        if action_stats_path:
            action_frame = load_mo_action_stats_frame(repo_root / action_stats_path if not Path(action_stats_path).is_absolute() else action_stats_path)
            if not action_frame.empty:
                action_frames.append(action_frame)

    summary_path = export_dir / 'mo_run_summary.csv'
    trace_path = export_dir / 'mo_trace.csv'
    events_path = export_dir / 'mo_events.csv'
    action_path = export_dir / 'mo_action_stats.csv'

    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    pd.concat(trace_frames, ignore_index=True).to_csv(trace_path, index=False, encoding='utf-8-sig') if trace_frames else pd.DataFrame().to_csv(trace_path, index=False, encoding='utf-8-sig')
    pd.concat(event_frames, ignore_index=True).to_csv(events_path, index=False, encoding='utf-8-sig') if event_frames else pd.DataFrame().to_csv(events_path, index=False, encoding='utf-8-sig')
    pd.concat(action_frames, ignore_index=True).to_csv(action_path, index=False, encoding='utf-8-sig') if action_frames else pd.DataFrame().to_csv(action_path, index=False, encoding='utf-8-sig')

    return {
        'summary': summary_path,
        'trace': trace_path,
        'events': events_path,
        'action_stats': action_path,
    }

LEGACY_RESULT_BASE_COLUMNS = [
    "实例",
    "算法",
    "日期",
    "迭代次数",
    "解",
    "适应度值",
    "开始时间",
    "最快时间",
    "结束时间",
    "运行时间（秒）",
    "最快最佳结果时间（秒）",
    "宽高比是否满足",
    "gbest更新次数",
    "备注",
]

LEGACY_RESULT_BASELINE_EXTRA_COLUMNS = [
    'baseline_algorithm',
    'baseline_population',
    'baseline_generations',
    'baseline_sequence_length',
    'baseline_seed',
    'baseline_pso_inertia',
    'baseline_pso_c1',
    'baseline_pso_c2',
    'baseline_pso_vmax_ratio',
    'baseline_pso_mutation_prob',
    'baseline_pso_guide_limit',
]

LEGACY_RESULT_MO_EXTRA_COLUMNS = [
    'pareto_size',
    'pareto_archive_path',
    'rep_mhc',
    'rep_cr',
    'rep_dr',
    'rep_ar',
    'decision_score',
    'stable_decision_score',
    'HV',
    'Spread(Δ)',
    'IGD',
    'reference_front_path',
    'reference_front_size',
    'reference_front_archive_count',
    'archive_hypervolume_mode',
    'archive_hypervolume_reference_point',
    'GD',
    'IGD+',
    'C(Ref, S)',
    'multiplicative ε',
    'benchmark_normalization_path',
    'wall_time_terminated',
    'wall_time_limit_seconds',
    'mo_run_id',
    'mo_bundle_dir',
    'mo_trace_path',
    'mo_events_path',
    'mo_action_stats_path',
    'mo_run_summary_path',
] + LEGACY_RESULT_BASELINE_EXTRA_COLUMNS

LEGACY_RESULT_COLUMNS = LEGACY_RESULT_BASE_COLUMNS + LEGACY_RESULT_MO_EXTRA_COLUMNS


def _legacy_result_csv_path(exp_instance, exp_algorithm, result_root=None):
    result_root = Path(result_root or config.RESULT_PATH)
    instance_clean = _sanitize_filename(exp_instance)
    algorithm_clean = _sanitize_filename(exp_algorithm)
    base_instance_clean = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', instance_clean)
    return result_root / f'{base_instance_clean}-{algorithm_clean}.csv'


def _seconds_between(start_time, end_time):
    if start_time is None or end_time is None:
        return None
    try:
        return (end_time - start_time).total_seconds()
    except Exception:
        return None


def _looks_like_archive_path(value):
    text = str(value or '').strip()
    if not text:
        return False
    return text.endswith('.json') or 'pareto_archives/' in text or 'pareto_archives\\' in text


def _map_legacy_mo_extras(extra_values):
    values = list(extra_values or [])
    if not values:
        return {}

    size_first_full_order_v2 = [
        'pareto_size',
        'pareto_archive_path',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'stable_decision_score',
        'archive_hypervolume',
        'archive_spacing',
        'archive_igd',
        'reference_front_path',
        'reference_front_size',
        'reference_front_archive_count',
        'archive_hypervolume_mode',
        'archive_hypervolume_reference_point',
        'wall_time_terminated',
        'wall_time_limit_seconds',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    size_first_full_order_v3 = size_first_full_order_v2 + LEGACY_RESULT_BASELINE_EXTRA_COLUMNS
    size_first_full_order = [
        'pareto_size',
        'pareto_archive_path',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'stable_decision_score',
        'archive_hypervolume',
        'archive_spacing',
        'archive_igd',
        'reference_front_path',
        'reference_front_size',
        'reference_front_archive_count',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    size_first_without_quality_order = [
        'pareto_size',
        'pareto_archive_path',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    size_first_legacy_order = [
        'pareto_size',
        'pareto_archive_path',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
    ]
    path_first_full_order_v2 = [
        'pareto_archive_path',
        'pareto_size',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'stable_decision_score',
        'archive_hypervolume',
        'archive_spacing',
        'archive_igd',
        'reference_front_path',
        'reference_front_size',
        'reference_front_archive_count',
        'archive_hypervolume_mode',
        'archive_hypervolume_reference_point',
        'wall_time_terminated',
        'wall_time_limit_seconds',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    path_first_full_order_v3 = path_first_full_order_v2 + LEGACY_RESULT_BASELINE_EXTRA_COLUMNS
    path_first_full_order = [
        'pareto_archive_path',
        'pareto_size',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'stable_decision_score',
        'archive_hypervolume',
        'archive_spacing',
        'archive_igd',
        'reference_front_path',
        'reference_front_size',
        'reference_front_archive_count',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    path_first_without_quality_order = [
        'pareto_archive_path',
        'pareto_size',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
        'mo_run_id',
        'mo_bundle_dir',
        'mo_trace_path',
        'mo_events_path',
        'mo_action_stats_path',
        'mo_run_summary_path',
    ]
    path_first_legacy_order = [
        'pareto_archive_path',
        'pareto_size',
        'rep_mhc',
        'rep_cr',
        'rep_dr',
        'rep_ar',
        'decision_score',
    ]

    if _looks_like_archive_path(values[0]):
        if len(values) >= len(path_first_full_order_v3):
            order = path_first_full_order_v3
        elif len(values) >= len(path_first_full_order_v2):
            order = path_first_full_order_v2
        elif len(values) >= len(path_first_full_order):
            order = path_first_full_order
        elif len(values) >= len(path_first_without_quality_order):
            order = path_first_without_quality_order
        else:
            order = path_first_legacy_order
    else:
        if len(values) >= len(size_first_full_order_v3):
            order = size_first_full_order_v3
        elif len(values) >= len(size_first_full_order_v2):
            order = size_first_full_order_v2
        elif len(values) >= len(size_first_full_order):
            order = size_first_full_order
        elif len(values) >= len(size_first_without_quality_order):
            order = size_first_without_quality_order
        else:
            order = size_first_legacy_order
    return {
        key: _normalize_value(value)
        for key, value in zip(order, values)
    }


def _normalize_legacy_result_row(raw_row):
    raw = list(raw_row or [])
    if not raw:
        return None
    base_values = (raw + [''] * len(LEGACY_RESULT_BASE_COLUMNS))[: len(LEGACY_RESULT_BASE_COLUMNS)]
    row = {
        column: _normalize_value(value) if value != '' else None
        for column, value in zip(LEGACY_RESULT_BASE_COLUMNS, base_values)
    }
    row.update(_map_legacy_mo_extras(raw[len(LEGACY_RESULT_BASE_COLUMNS):]))
    return row


def repair_legacy_mo_result_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=LEGACY_RESULT_COLUMNS)

    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return pd.DataFrame(columns=LEGACY_RESULT_COLUMNS)

    header = _dedupe_columns(rows[0])
    if header[: len(LEGACY_RESULT_BASE_COLUMNS)] == LEGACY_RESULT_BASE_COLUMNS:
        max_len = max(len(row) for row in rows)
        extra_count = max(0, max_len - len(header))
        columns = header + [f'legacy_extra_{idx + 1:03d}' for idx in range(extra_count)]
        records = []
        for raw_row in rows[1:]:
            if not raw_row or not any(str(cell).strip() for cell in raw_row):
                continue
            padded = list(raw_row) + [None] * max(0, len(columns) - len(raw_row))
            records.append(
                {
                    column: _normalize_value(padded[idx]) if idx < len(padded) and padded[idx] != '' else None
                    for idx, column in enumerate(columns)
                }
            )
        frame = pd.DataFrame(records, columns=columns)
        frame = _migrate_benchmark_display_columns(frame)
        for column in LEGACY_RESULT_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        custom_columns = [column for column in frame.columns if column not in LEGACY_RESULT_COLUMNS]
        frame = frame[LEGACY_RESULT_COLUMNS + custom_columns]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return frame

    normalized_rows = []
    for raw_row in rows[1:]:
        if not raw_row or not any(str(cell).strip() for cell in raw_row):
            continue
        normalized = _normalize_legacy_result_row(raw_row)
        if normalized is not None:
            normalized_rows.append(normalized)

    discovered_columns = []
    for row in normalized_rows:
        for key in row.keys():
            if key not in LEGACY_RESULT_COLUMNS and key not in discovered_columns:
                discovered_columns.append(key)

    all_columns = LEGACY_RESULT_COLUMNS + discovered_columns
    frame = pd.DataFrame([{column: row.get(column) for column in all_columns} for row in normalized_rows], columns=all_columns)
    frame = _migrate_benchmark_display_columns(frame)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return frame


def save_legacy_mo_experiment_result(
    exp_instance,
    exp_algorithm,
    exp_iterations,
    exp_solution,
    exp_fitness,
    exp_start_time,
    exp_fast_time,
    exp_end_time,
    exp_is_valid_aspect_ratio=None,
    exp_remark='',
    exp_gbest_updates=None,
    exp_extra_fields=None,
    result_root=None,
):
    csv_path = _legacy_result_csv_path(exp_instance, exp_algorithm, result_root=result_root)
    existing_frame = repair_legacy_mo_result_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=LEGACY_RESULT_COLUMNS)

    row = {
        "实例": exp_instance,
        "算法": exp_algorithm,
        "日期": exp_start_time.date() if exp_start_time is not None else None,
        "迭代次数": exp_iterations,
        "解": _normalize_value(exp_solution),
        "适应度值": _normalize_value(exp_fitness),
        "开始时间": exp_start_time,
        "最快时间": exp_fast_time,
        "结束时间": exp_end_time,
        "运行时间（秒）": _seconds_between(exp_start_time, exp_end_time),
        "最快最佳结果时间（秒）": _seconds_between(exp_start_time, exp_fast_time),
        "宽高比是否满足": exp_is_valid_aspect_ratio,
        "gbest更新次数": exp_gbest_updates,
        "备注": exp_remark,
    }

    for key, value in dict(exp_extra_fields or {}).items():
        row[_display_column_name(key)] = _normalize_value(value)

    columns = list(existing_frame.columns) if not existing_frame.empty else list(LEGACY_RESULT_COLUMNS)
    for column in LEGACY_RESULT_COLUMNS:
        if column not in columns:
            columns.append(column)
    for key in row.keys():
        if key not in columns:
            columns.append(key)

    if existing_frame.empty and not csv_path.exists():
        frame = pd.DataFrame(columns=columns)
    else:
        frame = existing_frame.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns]

    row_frame = pd.DataFrame([{column: row.get(column) for column in columns}], columns=columns)
    if frame.empty:
        frame = row_frame.copy()
    else:
        frame.loc[len(frame)] = row_frame.iloc[0]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return frame
