import argparse
import json
from pathlib import Path

from src.utils.BO_MREFLPBenchmark import BO_MREFLPBenchmark


def _discover_instances(results_root, include_algorithms):
    instances = set()
    results_root = Path(results_root)
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        for algorithm_name in include_algorithms:
            for alias in BO_MREFLPBenchmark.ALGORITHM_DIR_ALIASES[algorithm_name]:
                algorithm_dir = run_dir / alias
                if not algorithm_dir.exists():
                    continue
                for filepath in algorithm_dir.glob("*.txt"):
                    if filepath.name.endswith("_time.txt"):
                        continue
                    instances.add(filepath.stem)
    return sorted(instances)


def main():
    parser = argparse.ArgumentParser(description="构建 BO-MREFLP 固定 benchmark package。")
    parser.add_argument("--instance", help="实例名，例如 A-10-10。默认构建全部实例。")
    parser.add_argument("--output-key", help="输出文件名键，默认等于实例名。")
    parser.add_argument(
        "--results-root",
        default=str(BO_MREFLPBenchmark.DEFAULT_RESULTS_ROOT),
        help="GRASP_Results 根目录。",
    )
    parser.add_argument(
        "--benchmark-root",
        default=str(BO_MREFLPBenchmark.DEFAULT_BENCHMARK_ROOT),
        help="benchmark 输出根目录。",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=["NSGA-II", "NSBBO", "GRASP1", "GRASP2", "GRASP3", "GRASP4"],
        help="参与构建公共参考前沿的算法集合。",
    )
    args = parser.parse_args()

    if args.instance:
        instances = [args.instance]
    else:
        instances = _discover_instances(args.results_root, args.algorithms)

    if not instances:
        raise ValueError("未发现任何可构建 benchmark 的实例。")

    summaries = []
    for instance_name in instances:
        payload = BO_MREFLPBenchmark.build_benchmark_package(
            instance_name=instance_name,
            results_root=args.results_root,
            benchmark_root=args.benchmark_root,
            output_key=args.output_key if args.instance else None,
            include_algorithms=args.algorithms,
        )
        summaries.append(
            {
                "instance": instance_name,
                "reference_front_path": payload["reference_front_path"],
                "normalization_path": payload["normalization_path"],
                "reference_front_size": payload["reference_payload"]["reference_front_size"],
                "source_point_count": payload["reference_payload"]["source_point_count"],
            }
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
