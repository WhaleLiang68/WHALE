import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception as exc:  # pragma: no cover - 依赖缺失时给出明确错误
    wilcoxon = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.BiMO4BenchmarkUtil import benchmark_root
import src.utils.config as config


RESULT_ROOT = Path(config.RESULT_PATH)
TARGET_ALGORITHM = "ELP_DRL_BiMO4"
MINIMIZE_METRICS = {
    "rep_mhc": True,
    "rep_cr": False,
    "hv_ref_front": False,
    "igd_ref_front": True,
    "spread_delta_ref_front": True,
    "coverage_ref_to_s": True,
    "coverage_s_to_ref": False,
    "runtime_seconds": True,
}
PLOT_METRICS = [
    "hv_ref_front",
    "igd_ref_front",
    "spread_delta_ref_front",
    "coverage_ref_to_s",
]


def parse_args():
    parser = argparse.ArgumentParser(description="统计分析 BiMO4 双目标对比实验结果。")
    parser.add_argument("--benchmark-id", default="bimo4_compare_v1", help="实验批次 ID。")
    parser.add_argument("--target-algorithm", default=TARGET_ALGORITHM, help="默认与其它算法比较的目标算法。")
    return parser.parse_args()


def require_scipy():
    if wilcoxon is None:
        raise RuntimeError(
            "当前环境缺少 scipy，无法执行 Wilcoxon 显著性检验。"
        ) from SCIPY_IMPORT_ERROR


def metric_orientation(metric_name):
    if metric_name not in MINIMIZE_METRICS:
        raise KeyError(f"未知指标: {metric_name}")
    return bool(MINIMIZE_METRICS[metric_name])


def build_paired_test_rows(unified_frame, target_algorithm):
    require_scipy()
    rows = []
    metrics = [metric for metric in MINIMIZE_METRICS.keys() if metric in unified_frame.columns]
    for (instance, budget_seconds), instance_budget_group in unified_frame.groupby(["instance", "budget_seconds"]):
        target_group = instance_budget_group[instance_budget_group["algorithm"].astype(str) == str(target_algorithm)]
        if target_group.empty:
            continue
        for baseline_algorithm in sorted(
            algorithm
            for algorithm in instance_budget_group["algorithm"].astype(str).unique().tolist()
            if algorithm != str(target_algorithm)
        ):
            baseline_group = instance_budget_group[instance_budget_group["algorithm"].astype(str) == baseline_algorithm]
            merged = target_group.merge(
                baseline_group,
                on=["instance", "budget_seconds", "seed"],
                suffixes=("_target", "_baseline"),
            )
            if merged.empty:
                continue
            for metric in metrics:
                target_values = pd.to_numeric(merged[f"{metric}_target"], errors="coerce")
                baseline_values = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
                paired = pd.DataFrame({"target": target_values, "baseline": baseline_values}).dropna()
                if paired.empty:
                    continue
                if metric_orientation(metric):
                    diffs = paired["target"].astype(float) - paired["baseline"].astype(float)
                    wins = int((diffs < -1e-12).sum())
                    ties = int((np.abs(diffs) <= 1e-12).sum())
                    losses = int((diffs > 1e-12).sum())
                else:
                    diffs = paired["target"].astype(float) - paired["baseline"].astype(float)
                    wins = int((diffs > 1e-12).sum())
                    ties = int((np.abs(diffs) <= 1e-12).sum())
                    losses = int((diffs < -1e-12).sum())
                nonzero = diffs[np.abs(diffs) > 1e-12]
                if nonzero.empty:
                    statistic = 0.0
                    pvalue = 1.0
                else:
                    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", correction=False)
                    statistic = float(result.statistic)
                    pvalue = float(result.pvalue)
                decisive = wins + losses
                rows.append(
                    {
                        "instance": str(instance),
                        "budget_seconds": int(budget_seconds),
                        "target_algorithm": str(target_algorithm),
                        "baseline_algorithm": str(baseline_algorithm),
                        "metric": metric,
                        "paired_count": int(len(paired)),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "win_rate": None if decisive == 0 else float(wins / decisive),
                        "win_balance": None if decisive == 0 else float((wins - losses) / decisive),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_pvalue": pvalue,
                        "target_mean": float(paired["target"].mean()),
                        "baseline_mean": float(paired["baseline"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_boxplots(unified_frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for budget_seconds, budget_group in unified_frame.groupby("budget_seconds"):
        for metric in PLOT_METRICS:
            if metric not in budget_group.columns:
                continue
            instances = sorted(budget_group["instance"].astype(str).unique().tolist())
            figure, axes = plt.subplots(1, len(instances), figsize=(6 * max(1, len(instances)), 5), squeeze=False)
            axes = axes.ravel()
            for idx, instance in enumerate(instances):
                axis = axes[idx]
                group = budget_group[budget_group["instance"].astype(str) == instance]
                algorithms = sorted(group["algorithm"].astype(str).unique().tolist())
                series_list = [
                    pd.to_numeric(
                        group[group["algorithm"].astype(str) == algorithm][metric],
                        errors="coerce",
                    ).dropna()
                    for algorithm in algorithms
                ]
                axis.boxplot(series_list, labels=algorithms, showfliers=False)
                axis.set_title(f"{instance} | {metric}")
                axis.tick_params(axis="x", rotation=45)
                axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(output_dir / f"boxplot-budget{int(budget_seconds)}-{metric}.png", dpi=180)
            plt.close(figure)


def plot_pareto_fronts(unified_frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for (instance, budget_seconds), group in unified_frame.groupby(["instance", "budget_seconds"]):
        figure, axis = plt.subplots(figsize=(6, 5))
        for algorithm, algorithm_group in group.groupby("algorithm"):
            ranked = algorithm_group.sort_values("hv_ref_front", ascending=False, na_position="last")
            best_row = ranked.iloc[0]
            archive_path = best_row.get("archive_path_abs")
            if not archive_path or not Path(archive_path).exists():
                continue
            payload = json.loads(Path(archive_path).read_text(encoding="utf-8"))
            points = []
            for item in list(payload.get("items") or []):
                if not bool(item.get("isFeasible", item.get("current_is_feasible", False))):
                    continue
                mhc = item.get("mhc", item.get("MHC"))
                cr = item.get("cr", item.get("CR"))
                if mhc is None or cr is None:
                    vector = item.get("moObjectivesMin") or item.get("mo_objectives_min")
                    if isinstance(vector, (list, tuple)) and len(vector) >= 2:
                        mhc = float(vector[0])
                        cr = float(-float(vector[1]))
                if mhc is None or cr is None:
                    continue
                points.append((float(mhc), float(cr)))
            if not points:
                continue
            points = np.asarray(points, dtype=float)
            order = np.argsort(points[:, 0])
            axis.plot(points[order, 0], points[order, 1], marker="o", linewidth=1.2, label=str(algorithm))
        axis.set_title(f"{instance} | budget={int(budget_seconds)}s")
        axis.set_xlabel("MHC")
        axis.set_ylabel("CR")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"pareto-{instance}-budget{int(budget_seconds)}.png", dpi=180)
        plt.close(figure)


def plot_budget_sensitivity(summary_frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["hv_ref_front_mean", "igd_ref_front_mean", "spread_delta_ref_front_mean", "coverage_ref_to_s_mean"]:
        if metric not in summary_frame.columns:
            continue
        instances = sorted(summary_frame["instance"].astype(str).unique().tolist())
        figure, axes = plt.subplots(1, len(instances), figsize=(6 * max(1, len(instances)), 5), squeeze=False)
        axes = axes.ravel()
        for idx, instance in enumerate(instances):
            axis = axes[idx]
            group = summary_frame[summary_frame["instance"].astype(str) == instance]
            for algorithm, algorithm_group in group.groupby("algorithm"):
                axis.plot(
                    algorithm_group["budget_seconds"].astype(int).tolist(),
                    pd.to_numeric(algorithm_group[metric], errors="coerce").tolist(),
                    marker="o",
                    label=str(algorithm),
                )
            axis.set_title(f"{instance} | {metric}")
            axis.set_xlabel("Budget (s)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"budget-sensitivity-{metric}.png", dpi=180)
        plt.close(figure)


def main():
    args = parse_args()
    root = benchmark_root(RESULT_ROOT, args.benchmark_id)
    unified_path = root / "unified_metrics.csv"
    summary_path = root / "summary_stats.csv"
    if not unified_path.exists() or not summary_path.exists():
        raise FileNotFoundError("请先运行 summarize_bimo4_comparison.py 生成统一指标和汇总表。")

    unified_frame = pd.read_csv(unified_path, encoding="utf-8-sig")
    summary_frame = pd.read_csv(summary_path, encoding="utf-8-sig")
    stats_dir = root / "analysis"
    stats_dir.mkdir(parents=True, exist_ok=True)

    paired_frame = build_paired_test_rows(unified_frame, target_algorithm=args.target_algorithm)
    paired_path = stats_dir / "paired_wilcoxon.csv"
    paired_frame.to_csv(paired_path, index=False, encoding="utf-8-sig")

    plot_metric_boxplots(unified_frame, stats_dir / "plots")
    plot_pareto_fronts(unified_frame, stats_dir / "plots")
    plot_budget_sensitivity(summary_frame, stats_dir / "plots")

    print(f"统计检验: {paired_path.resolve().as_posix()}")
    print(f"图表目录: {(stats_dir / 'plots').resolve().as_posix()}")


if __name__ == "__main__":
    main()
