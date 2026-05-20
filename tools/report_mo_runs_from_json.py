import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "files" / "expresults" / "mo_runs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "files" / "expresults" / "mo_analysis_reports"


def to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(q)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    ratio = rank - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def iter_run_summary_files(runs_root):
    if not runs_root.exists():
        return
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists() and summary_path.is_file():
            yield run_dir, summary_path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_run_record(run_dir, summary_path, payload):
    start_time = str(payload.get("startTime") or "")
    end_time = str(payload.get("endTime") or "")
    start_dt = parse_iso_datetime(start_time)

    runtime_seconds = to_float(payload.get("runtimeSeconds"))
    best_result_seconds = to_float(payload.get("bestResultSeconds"))
    iterations = to_int(payload.get("iterations"))
    gbest_updates = to_int(payload.get("gbestUpdateCount"))
    feasible_count = to_int(payload.get("feasibleSolutionCount"))
    archive_size = to_int(payload.get("archiveSize"))
    archive_limit = to_int(payload.get("archiveLimit"))

    decision_score = to_float(payload.get("decisionScore"))
    stable_decision_score = to_float(payload.get("stableDecisionScore"))
    if stable_decision_score is not None:
        effective_score = stable_decision_score
        score_source = "stableDecisionScore"
    else:
        effective_score = decision_score
        score_source = "decisionScore"

    runtime_hours = None if runtime_seconds is None else runtime_seconds / 3600.0
    iter_per_second = None
    if runtime_seconds and runtime_seconds > 0 and iterations is not None:
        iter_per_second = iterations / runtime_seconds

    feasible_per_second = None
    if runtime_seconds and runtime_seconds > 0 and feasible_count is not None:
        feasible_per_second = feasible_count / runtime_seconds

    gbest_per_hour = None
    if runtime_seconds and runtime_seconds > 0 and gbest_updates is not None:
        gbest_per_hour = gbest_updates / (runtime_seconds / 3600.0)

    archive_fill_ratio = None
    if archive_limit and archive_limit > 0 and archive_size is not None:
        archive_fill_ratio = archive_size / archive_limit

    best_result_ratio = None
    if runtime_seconds and runtime_seconds > 0 and best_result_seconds is not None:
        best_result_ratio = best_result_seconds / runtime_seconds

    record = {
        "runId": str(payload.get("runId") or run_dir.name),
        "instance": str(payload.get("instance") or ""),
        "algorithm": str(payload.get("algorithm") or ""),
        "startTime": start_time,
        "endTime": end_time,
        "runtimeSeconds": runtime_seconds,
        "runtimeHours": runtime_hours,
        "bestResultSeconds": best_result_seconds,
        "bestResultRatio": best_result_ratio,
        "iterations": iterations,
        "iterPerSecond": iter_per_second,
        "gbestUpdateCount": gbest_updates,
        "gbestPerHour": gbest_per_hour,
        "feasibleSolutionCount": feasible_count,
        "feasiblePerSecond": feasible_per_second,
        "archiveSize": archive_size,
        "archiveLimit": archive_limit,
        "archiveFillRatio": archive_fill_ratio,
        "representativeArchiveIndex": to_int(payload.get("representativeArchiveIndex")),
        "decisionScore": decision_score,
        "stableDecisionScore": stable_decision_score,
        "effectiveScore": effective_score,
        "scoreSource": score_source,
        "archiveHypervolume": to_float(payload.get("archiveHypervolume")),
        "archiveSpacing": to_float(payload.get("archiveSpacing")),
        "repMhc": to_float(payload.get("repMhc")),
        "repCr": to_float(payload.get("repCr")),
        "repDr": to_float(payload.get("repDr")),
        "repAr": to_float(payload.get("repAr")),
        "isValid": payload.get("isValid"),
        "agentMode": str(payload.get("agentMode") or ""),
        "G": to_int(payload.get("G")),
        "tMax": to_int(payload.get("tMax")),
        "traceInterval": to_int(payload.get("traceInterval")),
        "paretoArchivePath": str(payload.get("paretoArchivePath") or ""),
        "runDir": run_dir.as_posix(),
        "runSummaryPath": summary_path.as_posix(),
        "_startDt": start_dt,
    }
    return record


def select_records(records, instance=None, algorithm=None):
    selected = []
    for row in records:
        if instance and str(row.get("instance") or "") != str(instance):
            continue
        if algorithm and str(row.get("algorithm") or "") != str(algorithm):
            continue
        selected.append(row)
    return selected


def ranking_sort_key(row):
    effective_score = row.get("effectiveScore")
    hypervolume = row.get("archiveHypervolume")
    spacing = row.get("archiveSpacing")
    runtime_hours = row.get("runtimeHours")
    return (
        math.inf if effective_score is None else effective_score,
        math.inf if hypervolume is None else -hypervolume,
        math.inf if spacing is None else spacing,
        math.inf if runtime_hours is None else runtime_hours,
        str(row.get("runId") or ""),
    )


def trend_sort_key(row):
    start_dt = row.get("_startDt")
    return (
        start_dt is None,
        datetime.max if start_dt is None else start_dt,
        str(row.get("runId") or ""),
    )


def compute_ranking(records):
    ranked = sorted(records, key=ranking_sort_key)
    rows = []
    for index, row in enumerate(ranked, start=1):
        item = dict(row)
        item["overallRank"] = index
        rows.append(item)
    return rows


def compute_trend(records):
    ordered = sorted(records, key=trend_sort_key)
    rows = []
    best_score_so_far = math.inf
    best_hv_so_far = -math.inf
    previous_score = None
    previous_hv = None

    for index, row in enumerate(ordered, start=1):
        current_score = row.get("effectiveScore")
        current_hv = row.get("archiveHypervolume")

        score_delta_vs_prev = None
        if previous_score is not None and current_score is not None:
            score_delta_vs_prev = current_score - previous_score

        hv_delta_vs_prev = None
        if previous_hv is not None and current_hv is not None:
            hv_delta_vs_prev = current_hv - previous_hv

        best_before = None if math.isinf(best_score_so_far) else best_score_so_far
        score_delta_vs_best_before = None
        if best_before is not None and current_score is not None:
            score_delta_vs_best_before = current_score - best_before

        new_best_score = False
        if current_score is not None and current_score < best_score_so_far:
            best_score_so_far = current_score
            new_best_score = True

        new_best_hv = False
        best_hv_before = None if best_hv_so_far < -1e100 else best_hv_so_far
        if current_hv is not None and current_hv > best_hv_so_far:
            best_hv_so_far = current_hv
            new_best_hv = True

        item = dict(row)
        item["sequence"] = index
        item["scoreDeltaVsPrev"] = score_delta_vs_prev
        item["scoreDeltaVsBestBefore"] = score_delta_vs_best_before
        item["bestScoreSoFar"] = None if math.isinf(best_score_so_far) else best_score_so_far
        item["newBestScore"] = new_best_score
        item["hypervolumeDeltaVsPrev"] = hv_delta_vs_prev
        item["bestHypervolumeSoFar"] = None if best_hv_so_far < -1e100 else best_hv_so_far
        item["newBestHypervolume"] = new_best_hv
        item["bestHypervolumeBefore"] = best_hv_before
        rows.append(item)

        previous_score = current_score
        previous_hv = current_hv
    return rows


def build_anomaly_rows(records):
    runtime_hours = [v for v in (row.get("runtimeHours") for row in records) if v is not None]
    effective_scores = [v for v in (row.get("effectiveScore") for row in records) if v is not None]
    hypervolumes = [v for v in (row.get("archiveHypervolume") for row in records) if v is not None]
    spacings = [v for v in (row.get("archiveSpacing") for row in records) if v is not None]

    runtime_p90 = percentile(runtime_hours, 0.90)
    score_p75 = percentile(effective_scores, 0.75)
    hv_p25 = percentile(hypervolumes, 0.25)
    spacing_p75 = percentile(spacings, 0.75)

    anomaly_rows = []
    for row in sorted(records, key=trend_sort_key):
        flags = []

        if row.get("stableDecisionScore") is None:
            flags.append("缺少 stableDecisionScore")
        if row.get("archiveFillRatio") is not None and row.get("archiveFillRatio") < 0.95:
            flags.append("Pareto 档案未填满")
        if row.get("bestResultRatio") is not None and row.get("bestResultRatio") > 0.98:
            flags.append("最好结果出现偏晚")
        if runtime_p90 is not None and row.get("runtimeHours") is not None and row.get("runtimeHours") > runtime_p90:
            flags.append("运行时间偏长")
        if score_p75 is not None and row.get("effectiveScore") is not None and row.get("effectiveScore") > score_p75:
            flags.append("有效分数偏高")
        if hv_p25 is not None and row.get("archiveHypervolume") is not None and row.get("archiveHypervolume") < hv_p25:
            flags.append("超体积偏低")
        if spacing_p75 is not None and row.get("archiveSpacing") is not None and row.get("archiveSpacing") > spacing_p75:
            flags.append("Spacing 偏大")

        if not flags:
            continue

        anomaly_rows.append(
            {
                "runId": row.get("runId"),
                "instance": row.get("instance"),
                "algorithm": row.get("algorithm"),
                "startTime": row.get("startTime"),
                "runtimeHours": row.get("runtimeHours"),
                "effectiveScore": row.get("effectiveScore"),
                "scoreSource": row.get("scoreSource"),
                "archiveSize": row.get("archiveSize"),
                "archiveLimit": row.get("archiveLimit"),
                "archiveHypervolume": row.get("archiveHypervolume"),
                "archiveSpacing": row.get("archiveSpacing"),
                "bestResultRatio": row.get("bestResultRatio"),
                "issueCount": len(flags),
                "issues": "；".join(flags),
                "runSummaryPath": row.get("runSummaryPath"),
            }
        )
    return anomaly_rows


def serialize_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key)) for key in fieldnames})


def infer_output_stem(instance, algorithm):
    left = instance if instance else "ALL_INSTANCE"
    right = algorithm if algorithm else "ALL_ALGORITHM"
    return f"{left}-{right}-MO-RUN-REPORT"


def print_brief_summary(records, ranking_rows, anomaly_rows):
    print(f"Loaded runs: {len(records)}")
    if not ranking_rows:
        return
    best = ranking_rows[0]
    print(
        "Best by effective score:",
        best.get("runId"),
        f"effective={best.get('effectiveScore')}",
        f"source={best.get('scoreSource')}",
        f"hv={best.get('archiveHypervolume')}",
        f"spacing={best.get('archiveSpacing')}",
    )
    print(f"Anomaly runs: {len(anomaly_rows)}")


def main():
    parser = argparse.ArgumentParser(description="从 mo_runs/*/run_summary.json 生成多目标对比报表。")
    parser.add_argument("--instance", default=None, help="仅保留指定实例（如 Du62）")
    parser.add_argument("--algorithm", default=None, help="仅保留指定算法（如 ELP_DRL_MO）")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="mo_runs 根目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="报表输出目录")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    raw_records = []
    for run_dir, summary_path in iter_run_summary_files(runs_root):
        try:
            payload = load_json(summary_path)
            raw_records.append(normalize_run_record(run_dir, summary_path, payload))
        except Exception as exc:
            print(f"[WARN] 跳过 {summary_path}: {exc}")

    selected_records = select_records(raw_records, instance=args.instance, algorithm=args.algorithm)
    if not selected_records:
        print("No run_summary.json matched the requested filters.")
        return

    ranking_rows = compute_ranking(selected_records)
    trend_rows = compute_trend(selected_records)
    anomaly_rows = build_anomaly_rows(selected_records)

    stem = infer_output_stem(args.instance, args.algorithm)
    ranking_path = output_dir / f"{stem}-ranking.csv"
    trend_path = output_dir / f"{stem}-trend.csv"
    anomaly_path = output_dir / f"{stem}-anomalies.csv"

    ranking_fields = [
        "overallRank",
        "runId",
        "instance",
        "algorithm",
        "startTime",
        "runtimeHours",
        "iterations",
        "effectiveScore",
        "scoreSource",
        "decisionScore",
        "stableDecisionScore",
        "archiveSize",
        "archiveLimit",
        "archiveFillRatio",
        "archiveHypervolume",
        "archiveSpacing",
        "repMhc",
        "repCr",
        "repDr",
        "repAr",
        "gbestUpdateCount",
        "feasibleSolutionCount",
        "bestResultSeconds",
        "bestResultRatio",
        "iterPerSecond",
        "feasiblePerSecond",
        "gbestPerHour",
        "agentMode",
        "runSummaryPath",
    ]
    trend_fields = [
        "sequence",
        "runId",
        "instance",
        "algorithm",
        "startTime",
        "effectiveScore",
        "scoreSource",
        "scoreDeltaVsPrev",
        "scoreDeltaVsBestBefore",
        "bestScoreSoFar",
        "newBestScore",
        "archiveHypervolume",
        "hypervolumeDeltaVsPrev",
        "bestHypervolumeBefore",
        "bestHypervolumeSoFar",
        "newBestHypervolume",
        "archiveSpacing",
        "runtimeHours",
        "bestResultRatio",
        "archiveSize",
        "archiveLimit",
        "runSummaryPath",
    ]
    anomaly_fields = [
        "runId",
        "instance",
        "algorithm",
        "startTime",
        "runtimeHours",
        "effectiveScore",
        "scoreSource",
        "archiveSize",
        "archiveLimit",
        "archiveHypervolume",
        "archiveSpacing",
        "bestResultRatio",
        "issueCount",
        "issues",
        "runSummaryPath",
    ]

    write_csv(ranking_path, ranking_rows, ranking_fields)
    write_csv(trend_path, trend_rows, trend_fields)
    write_csv(anomaly_path, anomaly_rows, anomaly_fields)

    print_brief_summary(selected_records, ranking_rows, anomaly_rows)
    print("Outputs:")
    print(f"- ranking:   {ranking_path.as_posix()}")
    print(f"- trend:     {trend_path.as_posix()}")
    print(f"- anomalies: {anomaly_path.as_posix()}")


if __name__ == "__main__":
    main()
