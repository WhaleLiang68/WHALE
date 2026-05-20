import argparse
import json

from src.utils.BO_MREFLPBenchmark import BO_MREFLPBenchmark


def main():
    parser = argparse.ArgumentParser(description="使用固定 benchmark package 评估 BO-MREFLP 候选集。")
    parser.add_argument("--instance", required=True, help="实例名，例如 A-10-10。")
    parser.add_argument("--input", required=True, help="待评估候选集文件，支持 .txt / .json。")
    parser.add_argument("--output-key", help="benchmark 文件名键，默认等于实例名。")
    parser.add_argument(
        "--benchmark-root",
        default=str(BO_MREFLPBenchmark.DEFAULT_BENCHMARK_ROOT),
        help="benchmark 根目录。",
    )
    parser.add_argument(
        "--keep-input-order",
        action="store_true",
        help="默认会先做 ND 过滤；传入该参数后保留输入原集合。",
    )
    parser.add_argument(
        "--include-points",
        action="store_true",
        help="输出候选集和公共参考前沿点集明细。",
    )
    args = parser.parse_args()

    metrics = BO_MREFLPBenchmark.evaluate_file(
        instance_name=args.instance,
        path=args.input,
        benchmark_root=args.benchmark_root,
        output_key=args.output_key,
        filter_nondominated=not args.keep_input_order,
        include_point_sets=args.include_points,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
