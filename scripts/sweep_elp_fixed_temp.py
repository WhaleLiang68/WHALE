"""固定温度扫描 ELP.py 的 k 与 bin_width 参数。

该脚本不修改 ELP.py。为避免 ELP.py 每个外层循环结束时衰减 T 和 k，
这里把一次测试组织为 G=1、t_max=总步数，使搜索过程保持固定温度。
所有明细结果默认追加到同一个 CSV，并用 sweep_tag/remark 区分实验批次。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import random
import sys
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym  # noqa: E402

from src.algorithms.ELP import ELP  # noqa: E402


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"参数列表为空: {raw!r}")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"seed 列表为空: {raw!r}")
    return values


def parse_bin_width_list(raw: str) -> list[str]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        lower_item = item.lower()
        if lower_item == "adaptive" or lower_item.startswith("adaptive-"):
            values.append(lower_item)
        else:
            values.append(str(float(item)))
    if not values:
        raise ValueError(f"bin_width 列表为空: {raw!r}")
    return values


def parse_bin_width_spec(bin_width_spec: str) -> tuple[str, float, bool, str]:
    """解析固定箱宽或自适应箱宽策略。"""
    raw_spec = str(bin_width_spec).strip()
    lower_spec = raw_spec.lower()
    if lower_spec == "adaptive":
        return raw_spec, 20.0, True, "spread"
    if lower_spec.startswith("adaptive-"):
        strategy = lower_spec.split("-", 1)[1].strip()
        if strategy not in {"spread", "scale", "hybrid"}:
            raise ValueError(f"未知自适应分箱规格: {bin_width_spec!r}")
        return lower_spec, 20.0, True, strategy
    return raw_spec, float(raw_spec), False, "spread"


def percentile_summary(values, quantiles=(0, 50, 90, 99, 100)) -> dict[str, float]:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        return {f"q{q}": float("nan") for q in quantiles}
    result = np.percentile(finite_values, quantiles)
    return {f"q{q}": float(value) for q, value in zip(quantiles, result)}


def safe_mean(values) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def safe_std(values) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.std(finite_values, ddof=0))


def safe_min(values) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.min(finite_values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def reset_env(env, seed: int):
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    except Exception:
        env.reset()


def disable_plots(solver: ELP) -> None:
    solver._plot_histogram = MethodType(lambda self: None, solver)
    solver._plot_energy_curve = MethodType(lambda self: None, solver)
    solver._plot_modified_energy_curve = MethodType(lambda self: None, solver)
    solver._plot_prob_curve = MethodType(lambda self: None, solver)
    solver._plot_gbest_trend = MethodType(lambda self: None, solver)


def cap_greedy_search(solver: ELP, greedy_steps: int) -> None:
    original_greedy = solver._greedy_search_step

    def controlled_greedy(self, max_steps=500):
        if greedy_steps <= 0:
            return False
        return original_greedy(max_steps=min(int(max_steps), int(greedy_steps)))

    solver._greedy_search_step = MethodType(controlled_greedy, solver)


def run_one(
    instance: str,
    seed: int,
    steps: int,
    temperature: float,
    k_value: float,
    bin_width_spec: str,
    greedy_steps: int,
    adaptive_refresh_interval: int,
):
    set_seed(seed)
    env = gym.make("FbsEnv-v0", instance=instance)
    reset_env(env, seed)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    initial_gbest = copy.deepcopy(base_env)
    normalized_bin_width_spec, initial_bin_width, use_adaptive_bin_width, adaptive_strategy = parse_bin_width_spec(
        bin_width_spec
    )
    bin_width_mode = "adaptive" if use_adaptive_bin_width else "fixed"
    solver = ELP(
        env=base_env,
        gbest=initial_gbest,
        T=float(temperature),
        Q_matrix=np.zeros((1, 6)),
        G=1,
        t_max=int(steps),
        k=float(k_value),
        bin_width=float(initial_bin_width),
        adaptive_bin_width=use_adaptive_bin_width,
        adaptive_bin_width_strategy=adaptive_strategy,
        bin_width_refresh_interval=int(adaptive_refresh_interval),
    )
    disable_plots(solver)
    cap_greedy_search(solver, greedy_steps)

    start_wall = datetime.now()
    # ELP.py 会打印 prob/T；扫描时压制这些中间输出，终端只保留汇总。
    with contextlib.redirect_stdout(io.StringIO()):
        total_iter, is_valid, best_sol, best_energy, start_time, end_time, fast_time = solver.run()
    end_wall = datetime.now()

    if hasattr(env, "close"):
        env.close()

    hist_counts = np.asarray(list(solver.energy_histogram.values()), dtype=float)
    h_stats = percentile_summary(hist_counts, quantiles=(0, 50, 90, 99, 100))
    k_h_stats = percentile_summary(hist_counts * float(k_value), quantiles=(0, 50, 90, 99, 100))
    prob_stats = percentile_summary(solver.prob_history, quantiles=(0, 10, 50, 90, 100))
    energy_stats = percentile_summary(solver.energy_history, quantiles=(0, 10, 50, 90, 100))
    modified_stats = percentile_summary(solver.modified_energy_history, quantiles=(0, 10, 50, 90, 100))

    finite_probs = np.asarray([float(v) for v in solver.prob_history if np.isfinite(v)], dtype=float)
    finite_energy = np.asarray([float(v) for v in solver.energy_history if np.isfinite(v)], dtype=float)
    nonzero_delta = np.abs(np.diff(finite_energy))
    nonzero_delta = nonzero_delta[nonzero_delta > 1e-9]
    delta_stats = percentile_summary(nonzero_delta, quantiles=(50, 75, 90, 95, 99))
    bin_width_history = np.asarray(
        [float(value) for value in getattr(solver, "bin_width_history", []) if np.isfinite(value)],
        dtype=float,
    )

    return {
        "instance": instance,
        "seed": int(seed),
        "steps": int(steps),
        "temperature": float(temperature),
        "k": float(k_value),
        "bin_width": str(normalized_bin_width_spec),
        "bin_width_mode": bin_width_mode,
        "adaptive_bin_width_strategy": adaptive_strategy,
        "initial_bin_width": float(initial_bin_width),
        "final_bin_width": float(getattr(solver, "bin_width", initial_bin_width)),
        "mean_bin_width": float(np.mean(bin_width_history)) if bin_width_history.size else float(initial_bin_width),
        "greedy_steps": int(greedy_steps),
        "best_energy": float(best_energy),
        "is_valid": bool(is_valid),
        "gbest_updates": int(getattr(solver, "gbest_update_count", 0)),
        "total_iter": int(total_iter),
        "runtime_sec": (end_wall - start_wall).total_seconds(),
        "fast_best_sec": (fast_time - start_time).total_seconds(),
        "prob_mean": float(np.mean(finite_probs)) if finite_probs.size else float("nan"),
        "prob_q10": prob_stats["q10"],
        "prob_q50": prob_stats["q50"],
        "prob_q90": prob_stats["q90"],
        "hist_bins": int(len(solver.energy_histogram)),
        "H_q50": h_stats["q50"],
        "H_q90": h_stats["q90"],
        "H_q99": h_stats["q99"],
        "H_max": h_stats["q100"],
        "kH_q50": k_h_stats["q50"],
        "kH_q90": k_h_stats["q90"],
        "kH_q99": k_h_stats["q99"],
        "kH_max": k_h_stats["q100"],
        "energy_q10": energy_stats["q10"],
        "energy_q50": energy_stats["q50"],
        "energy_q90": energy_stats["q90"],
        "modified_q10": modified_stats["q10"],
        "modified_q50": modified_stats["q50"],
        "modified_q90": modified_stats["q90"],
        "nonzero_abs_delta_q50": delta_stats["q50"],
        "nonzero_abs_delta_q75": delta_stats["q75"],
        "nonzero_abs_delta_q90": delta_stats["q90"],
        "nonzero_abs_delta_q95": delta_stats["q95"],
        "nonzero_abs_delta_q99": delta_stats["q99"],
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open("r", newline="", encoding="utf-8-sig") as input_file:
            reader = csv.DictReader(input_file)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fieldnames and all(fieldname in existing_fieldnames for fieldname in fieldnames):
            fieldnames = existing_fieldnames
        elif existing_fieldnames and existing_fieldnames != fieldnames:
            merged_fieldnames = list(existing_fieldnames)
            for fieldname in fieldnames:
                if fieldname not in merged_fieldnames:
                    merged_fieldnames.append(fieldname)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            with temp_path.open("w", newline="", encoding="utf-8-sig") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
            temp_path.replace(path)
            return
    with path.open("a", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["k"], row["bin_width"], row.get("bin_width_mode", "fixed"))
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (k_value, bin_width, bin_width_mode), group_rows in groups.items():
        best_values = np.asarray([row["best_energy"] for row in group_rows], dtype=float)
        runtime_values = np.asarray([row["runtime_sec"] for row in group_rows], dtype=float)
        valid_values = np.asarray([1.0 if row["is_valid"] else 0.0 for row in group_rows], dtype=float)
        summary_rows.append(
            {
                "k": float(k_value),
                "bin_width": str(bin_width),
                "bin_width_mode": str(bin_width_mode),
                "mean_final_bin_width": safe_mean([row.get("final_bin_width", float("nan")) for row in group_rows]),
                "mean_bin_width": safe_mean([row.get("mean_bin_width", float("nan")) for row in group_rows]),
                "runs": len(group_rows),
                "mean_best_energy": safe_mean(best_values),
                "min_best_energy": safe_min(best_values),
                "std_best_energy": safe_std(best_values),
                "mean_runtime_sec": float(np.mean(runtime_values)),
                "valid_rate": float(np.mean(valid_values)),
                "mean_prob": safe_mean([row["prob_mean"] for row in group_rows]),
                "mean_H_max": float(np.mean([row["H_max"] for row in group_rows])),
                "mean_kH_max": float(np.mean([row["kH_max"] for row in group_rows])),
                "mean_nonzero_abs_delta_q50": safe_mean(
                    [row["nonzero_abs_delta_q50"] for row in group_rows]
                ),
            }
        )
    summary_rows.sort(key=lambda item: (item["mean_best_energy"], item["mean_runtime_sec"]))
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="固定温度扫描 ELP.py 的 k/bin_width 参数。")
    parser.add_argument("--instance", default="Du62")
    parser.add_argument("--steps", type=int, default=1000, help="每组参数每个 seed 的总搜索步数。")
    parser.add_argument("--temperature", type=float, default=5000.0)
    parser.add_argument("--k-values", default="0,20,50,100,200")
    parser.add_argument("--bin-widths", default="500,1000,2500,5000,10000", help="逗号分隔；可包含 adaptive。")
    parser.add_argument("--seeds", default="20260510")
    parser.add_argument("--greedy-steps", type=int, default=0, help="ELP.py 内部贪婪增强步数；0 表示关闭。")
    parser.add_argument("--adaptive-refresh-interval", type=int, default=200, help="自适应 bin_width 的刷新步数间隔。")
    parser.add_argument(
        "--output-file",
        default=str(ROOT / "files" / "elp_param_sweeps" / "elp_fixed_temp_sweep_all.csv"),
        help="所有扫描明细追加写入的总 CSV。",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help="可选：把本次扫描汇总追加写入该 CSV；默认不写，避免分散结果。",
    )
    parser.add_argument("--tag", default=None, help="输出文件标签；默认使用时间戳。")
    parser.add_argument("--remark", default="", help="写入总 CSV 的备注，用于区分本次扫描目的。")
    args = parser.parse_args()

    k_values = parse_float_list(args.k_values)
    bin_widths = parse_bin_width_list(args.bin_widths)
    seeds = parse_int_list(args.seeds)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file) if str(args.summary_file).strip() else None
    sweep_started_at = datetime.now().isoformat(timespec="seconds")

    rows = []
    total = len(k_values) * len(bin_widths) * len(seeds)
    current = 0
    for k_value in k_values:
        for bin_width in bin_widths:
            for seed in seeds:
                current += 1
                print(
                    f"[{current}/{total}] instance={args.instance} seed={seed} "
                    f"T={args.temperature} k={k_value} bin_width={bin_width} steps={args.steps}"
                )
                row = run_one(
                    instance=args.instance,
                    seed=seed,
                    steps=args.steps,
                    temperature=args.temperature,
                    k_value=k_value,
                    bin_width_spec=bin_width,
                    greedy_steps=args.greedy_steps,
                    adaptive_refresh_interval=args.adaptive_refresh_interval,
                )
                row = {
                    "sweep_tag": tag,
                    "remark": str(args.remark),
                    "created_at": sweep_started_at,
                    **row,
                }
                rows.append(row)
                append_row(output_file, row)
                print(
                    f"    best={row['best_energy']:.6f} valid={row['is_valid']} "
                    f"prob_mean={row['prob_mean']:.4f} Hmax={row['H_max']:.0f} "
                    f"kHmax={row['kH_max']:.1f} final_bw={row['final_bin_width']:.3f} "
                    f"runtime={row['runtime_sec']:.1f}s"
                )

    summary_rows = summarize(rows)
    if summary_file is not None and summary_rows:
        for summary_row in summary_rows:
            append_row(
                summary_file,
                {
                    "sweep_tag": tag,
                    "remark": str(args.remark),
                    "created_at": sweep_started_at,
                    "instance": args.instance,
                    "steps": int(args.steps),
                    "temperature": float(args.temperature),
                    "greedy_steps": int(args.greedy_steps),
                    **summary_row,
                },
            )
    if summary_rows:
        best = summary_rows[0]
        print(
            "BEST_BY_MEAN "
            f"k={best['k']} bin_width={best['bin_width']} "
            f"mean_best={best['mean_best_energy']:.6f} "
            f"mean_runtime={best['mean_runtime_sec']:.1f}s "
            f"mean_prob={best['mean_prob']:.4f}"
        )
    print(f"OUTPUT_CSV={output_file}")
    if summary_file is not None:
        print(f"SUMMARY_CSV={summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
