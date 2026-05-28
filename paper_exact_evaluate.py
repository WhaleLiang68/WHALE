import argparse
import json

from src.utils.BO_MREFLPPaperExact import BO_MREFLPPaperExact


def main():
    parser = argparse.ArgumentParser(description="按原论文 Java 口径评估 BO-MREFLP archive。")
    parser.add_argument("--instance", required=True, help="实例名，例如 A-10-10。")
    parser.add_argument("--archive", required=True, help="待评估的 archive JSON 路径。")
    parser.add_argument(
        "--results-root",
        default=str(BO_MREFLPPaperExact.DEFAULT_RESULTS_ROOT),
        help="论文基线结果目录，默认使用 data/GRASP_Results。",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=list(BO_MREFLPPaperExact.DEFAULT_ALGORITHMS),
        help="参与动态 Ref 构造的论文算法集合。",
    )
    parser.add_argument(
        "--exclude-candidate-from-ref",
        action="store_true",
        help="默认会把当前 archive 一并并入动态 Ref；传入该参数后仅使用论文基线结果构造 Ref。",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="默认会输出 audit JSON；传入该参数后只打印指标。",
    )
    args = parser.parse_args()

    metrics = BO_MREFLPPaperExact.evaluate_archive(
        instance_name=args.instance,
        archive_path=args.archive,
        results_root=args.results_root,
        include_algorithms=args.algorithms,
        include_candidate=not args.exclude_candidate_from_ref,
        save_report=not args.skip_report,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
