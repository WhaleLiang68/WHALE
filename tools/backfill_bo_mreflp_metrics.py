import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.BO_MREFLPBenchmark import BO_MREFLPBenchmark

DISPLAY_COLUMNS = {
    "archive_hypervolume": "HV",
    "benchmark_gd": "GD",
    "archive_igd": "IGD",
    "benchmark_igd_plus": "IGD+",
    "archive_spacing": "Spread(Δ)",
    "benchmark_coverage_ref_to_s": "C(Ref, S)",
    "benchmark_epsilon_multiplicative": "multiplicative ε",
}


def _safe_literal_list(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    return parsed


def _load_points_from_row(row):
    archive_path = row.get("pareto_archive_path")
    if isinstance(archive_path, str) and archive_path.strip():
        candidate_path = Path(archive_path)
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
        if candidate_path.exists():
            return BO_MREFLPBenchmark.load_points_from_json(candidate_path)

    rep_mhc = row.get("rep_mhc")
    rep_cr = row.get("rep_cr")
    if pd.notna(rep_mhc) and pd.notna(rep_cr):
        return [[float(rep_mhc), float(rep_cr)]]
    return None


def _ensure_columns(frame):
    required_columns = [
        "HV",
        "Spread(Δ)",
        "IGD",
        "reference_front_path",
        "reference_front_size",
        "archive_hypervolume_mode",
        "archive_hypervolume_reference_point",
        "GD",
        "IGD+",
        "C(Ref, S)",
        "multiplicative ε",
        "benchmark_normalization_path",
    ]
    for legacy_name, display_name in DISPLAY_COLUMNS.items():
        if legacy_name in frame.columns and display_name not in frame.columns:
            frame = frame.rename(columns={legacy_name: display_name})
        elif legacy_name in frame.columns and display_name in frame.columns:
            frame[display_name] = frame[display_name].where(frame[display_name].notna(), frame[legacy_name])
            frame = frame.drop(columns=[legacy_name])
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = None
    object_columns = [
        "reference_front_path",
        "archive_hypervolume_mode",
        "archive_hypervolume_reference_point",
        "benchmark_normalization_path",
    ]
    for column in object_columns:
        frame[column] = frame[column].astype(object)
    return frame


def backfill(csv_path):
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    frame = _ensure_columns(frame)

    for idx, row in frame.iterrows():
        instance_name = row.get("instance") or row.get("实例")
        points = _load_points_from_row(row)
        if not instance_name or points is None:
            continue

        metrics = BO_MREFLPBenchmark.evaluate_points(str(instance_name), points)
        frame.at[idx, "HV"] = metrics["hv"]
        frame.at[idx, "Spread(Δ)"] = metrics["spread_delta"]
        frame.at[idx, "IGD"] = metrics["igd"]
        frame.at[idx, "reference_front_path"] = metrics["reference_front_path"]
        frame.at[idx, "reference_front_size"] = metrics["reference_front_size"]
        frame.at[idx, "archive_hypervolume_mode"] = "bo_mreflp_benchmark_fixed"
        frame.at[idx, "archive_hypervolume_reference_point"] = str(metrics["hv_ref_point"])
        frame.at[idx, "GD"] = metrics["gd"]
        frame.at[idx, "IGD+"] = metrics["igd_plus"]
        frame.at[idx, "C(Ref, S)"] = metrics["coverage_ref_to_s"]
        frame.at[idx, "multiplicative ε"] = metrics["epsilon_multiplicative"]
        frame.at[idx, "benchmark_normalization_path"] = metrics["normalization_path"]

    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return frame


def main():
    parser = argparse.ArgumentParser(description="回填 BO-MREFLP 固定 benchmark 指标到实验 CSV。")
    parser.add_argument("--csv", required=True, help="实验结果 CSV 路径。")
    args = parser.parse_args()
    frame = backfill(args.csv)
    print(f"updated_rows={len(frame)} csv={Path(args.csv).resolve().as_posix()}")


if __name__ == "__main__":
    main()
