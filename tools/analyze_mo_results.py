import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.MO_ExperimentsUtil import export_mo_analysis_tables
from src.utils.MO_ExperimentsUtil import load_mo_run_summary_frame


def main():
    parser = argparse.ArgumentParser(description="Export MO run bundles into flat CSV tables.")
    parser.add_argument("--instance", default=None, help="Filter by instance name")
    parser.add_argument("--algorithm", default=None, help="Filter by algorithm name")
    parser.add_argument("--run-id", default=None, help="Filter by a specific run id")
    parser.add_argument("--result-root", default=None, help="Override result root directory")
    parser.add_argument("--export-dir", default=None, help="Directory for flattened CSV outputs")
    args = parser.parse_args()

    summary = load_mo_run_summary_frame(
        result_root=args.result_root,
        instance=args.instance,
        algorithm=args.algorithm,
    )
    if args.run_id and not summary.empty and "runId" in summary.columns:
        summary = summary[summary["runId"].astype(str) == str(args.run_id)].reset_index(drop=True)

    if summary.empty:
        print("No MO runs matched the requested filters.")
        return

    outputs = export_mo_analysis_tables(
        result_root=args.result_root,
        instance=args.instance,
        algorithm=args.algorithm,
        run_id=args.run_id,
        export_dir=args.export_dir,
    )

    display_cols = [
        col for col in [
            "runId", "instance", "algorithm", "startTime", "runtimeSeconds",
            "decisionScore", "archiveSize", "repMhc", "repCr", "repDr", "repAr",
            "tracePath", "eventsPath", "actionStatsPath",
        ]
        if col in summary.columns
    ]
    print(summary[display_cols].to_string(index=False))
    print("\nExported tables:")
    for name, path in outputs.items():
        print(f"- {name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
