import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.BiMO4BenchmarkUtil import benchmark_root
from src.utils.BiMO4BenchmarkUtil import build_reference_fronts
from src.utils.BiMO4BenchmarkUtil import compute_unified_metrics
from src.utils.BiMO4BenchmarkUtil import load_benchmark_runs
from src.utils.BiMO4BenchmarkUtil import summarize_unified_metrics
import src.utils.config as config


RESULT_ROOT = Path(config.RESULT_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 BiMO4 双目标对比实验结果并统一重算指标。")
    parser.add_argument("--benchmark-id", default="bimo4_compare_v1", help="实验批次 ID。")
    parser.add_argument("--instances", nargs="+", default=["Du62", "SC35", "AB20-ar3"], help="实例列表。")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "ELP_DRL_BiMO4",
            "MO_BASELINE_NSGA2",
            "MO_BASELINE_MOEAD",
            "MO_BASELINE_SPEA2",
            "ELP_DRL_BiMO4_GRASP_PAPERLS",
            "ELP_DRL_BiMO4_GRASP_ACTIONLS",
        ],
        help="算法名列表，应与实验落盘算法名一致。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = benchmark_root(RESULT_ROOT, args.benchmark_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_benchmark_runs(
        args.benchmark_id,
        instances=args.instances,
        algorithms=args.algorithms,
        result_root=RESULT_ROOT,
    )
    raw_path = output_dir / "raw_runs.csv"
    raw_frame.to_csv(raw_path, index=False, encoding="utf-8-sig")

    if raw_frame.empty:
        print(f"未找到 benchmark_id={args.benchmark_id} 的实验结果。")
        return

    reference_dir = output_dir / "reference_fronts"
    reference_fronts = build_reference_fronts(
        raw_frame,
        repo_root=REPO_ROOT,
        output_dir=reference_dir,
        benchmark_id=args.benchmark_id,
    )
    unified_frame = compute_unified_metrics(
        raw_frame,
        repo_root=REPO_ROOT,
        reference_fronts=reference_fronts,
    )
    unified_path = output_dir / "unified_metrics.csv"
    unified_frame.to_csv(unified_path, index=False, encoding="utf-8-sig")

    summary_frame = summarize_unified_metrics(unified_frame)
    summary_path = output_dir / "summary_stats.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"原始结果: {raw_path.resolve().as_posix()}")
    print(f"统一指标: {unified_path.resolve().as_posix()}")
    print(f"汇总统计: {summary_path.resolve().as_posix()}")
    print(f"参考前沿目录: {reference_dir.resolve().as_posix()}")


if __name__ == "__main__":
    main()
