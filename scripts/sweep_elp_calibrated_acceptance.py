"""按接受概率公式联合校准 ELP.py 的 T、k 与 bin_width。

核心思想：
    p = exp(-(delta_cost + k * delta_H) / T)
    lambda = k / T

脚本先对每个 bin_width 采样正劣化 delta_cost，并按目标接受率 p0 反推 T：
    T = -Q(delta_cost_positive) / log(p0)

随后按 lambda 生成 k=lambda*T，再调用固定温度 ELP 扫描逻辑验证。
所有验证明细追加到同一个总 CSV。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gym  # noqa: E402

from src.algorithms.ELP import ELP  # noqa: E402
from sweep_elp_fixed_temp import (  # noqa: E402
    append_row,
    cap_greedy_search,
    disable_plots,
    parse_float_list,
    parse_int_list,
    reset_env,
    run_one,
    set_seed,
)


def finite_percentile(values, quantile: float) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.percentile(finite_values, float(quantile)))


def calibrate_temperature_from_delta(delta_positive, target_accept: float, quantile: float, fallback: float) -> float:
    if not 0.0 < float(target_accept) < 1.0:
        raise ValueError(f"target_accept 必须在 (0, 1) 内: {target_accept}")
    delta_q = finite_percentile(delta_positive, quantile)
    if not np.isfinite(delta_q) or delta_q <= 0:
        return float(fallback)
    temperature = -float(delta_q) / math.log(float(target_accept))
    if not np.isfinite(temperature) or temperature <= 0:
        return float(fallback)
    return float(temperature)


def make_probe_solver(instance: str, seed: int, bin_width: float, warmup_steps: int, warmup_temperature: float):
    set_seed(seed)
    env = gym.make("FbsEnv-v0", instance=instance)
    reset_env(env, seed)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    solver = ELP(
        env=base_env,
        gbest=copy.deepcopy(base_env),
        T=float(warmup_temperature),
        Q_matrix=np.zeros((1, 6)),
        G=1,
        t_max=max(1, int(warmup_steps)),
        k=0.0,
        bin_width=float(bin_width),
    )
    disable_plots(solver)
    cap_greedy_search(solver, 0)
    if warmup_steps > 0:
        # 只用于把采样链带到较合理区域，避免初始不可行状态支配 delta 分布。
        with contextlib.redirect_stdout(io.StringIO()):
            solver.run()
    return env, solver


def sample_delta_distribution(
    instance: str,
    seed: int,
    bin_width: float,
    warmup_steps: int,
    sample_steps: int,
    warmup_temperature: float,
    probe_accept_prob: float,
    feasible_only: bool,
):
    env, solver = make_probe_solver(
        instance=instance,
        seed=seed,
        bin_width=bin_width,
        warmup_steps=warmup_steps,
        warmup_temperature=warmup_temperature,
    )
    delta_positive = []
    delta_h_values = []
    raw_delta_values = []
    accepted_moves = 0

    for _ in range(max(1, int(sample_steps))):
        current_energy = solver._calculate_search_energy(solver.s)
        candidate = solver._generate_new_solution(solver.s)
        candidate_energy = solver._calculate_search_energy(candidate)
        current_objective = solver._calculate_energy(solver.s)
        candidate_objective = solver._calculate_energy(candidate)
        if np.isfinite(current_energy) and np.isfinite(candidate_energy):
            current_h = solver._get_H_value(current_energy)
            candidate_h = solver._get_H_value(candidate_energy)
            if np.isfinite(current_objective) and np.isfinite(candidate_objective):
                raw_delta = float(candidate_objective) - float(current_objective)
            elif feasible_only:
                raw_delta = None
            else:
                raw_delta = float(candidate_energy) - float(current_energy)
            delta_h = float(candidate_h) - float(current_h)
            delta_h_values.append(delta_h)
            if raw_delta is not None:
                raw_delta_values.append(raw_delta)
                if raw_delta > 0:
                    delta_positive.append(raw_delta)

            search_delta = float(candidate_energy) - float(current_energy)
            if search_delta < 0 or np.random.rand() < float(probe_accept_prob):
                solver.s = candidate
                accepted_moves += 1

        if np.isfinite(current_energy):
            solver._update_histogram(current_energy)
            solver.search_energy_history.append(float(current_energy))

    if hasattr(env, "close"):
        env.close()

    return {
        "delta_positive": delta_positive,
        "raw_delta": raw_delta_values,
        "delta_h": delta_h_values,
        "accepted_moves": accepted_moves,
        "final_bin_width": float(getattr(solver, "bin_width", bin_width)),
        "hist_bins": int(len(getattr(solver, "energy_histogram", {}))),
    }


def calibrate_for_bin_width(
    instance: str,
    bin_width: float,
    calibration_seeds: list[int],
    warmup_steps: int,
    sample_steps: int,
    warmup_temperature: float,
    probe_accept_prob: float,
    feasible_only: bool,
):
    all_positive = []
    all_raw_delta = []
    all_delta_h = []
    accepted_moves = 0
    hist_bins = []
    for seed in calibration_seeds:
        stats = sample_delta_distribution(
            instance=instance,
            seed=seed,
            bin_width=bin_width,
            warmup_steps=warmup_steps,
            sample_steps=sample_steps,
            warmup_temperature=warmup_temperature,
            probe_accept_prob=probe_accept_prob,
            feasible_only=feasible_only,
        )
        all_positive.extend(stats["delta_positive"])
        all_raw_delta.extend(stats["raw_delta"])
        all_delta_h.extend(stats["delta_h"])
        accepted_moves += int(stats["accepted_moves"])
        hist_bins.append(float(stats["hist_bins"]))

    return {
        "delta_positive_values": all_positive,
        "bin_width": float(bin_width),
        "positive_count": int(len(all_positive)),
        "raw_count": int(len(all_raw_delta)),
        "delta_positive_q50": finite_percentile(all_positive, 50),
        "delta_positive_q75": finite_percentile(all_positive, 75),
        "delta_positive_q90": finite_percentile(all_positive, 90),
        "delta_h_q10": finite_percentile(all_delta_h, 10),
        "delta_h_q50": finite_percentile(all_delta_h, 50),
        "delta_h_q90": finite_percentile(all_delta_h, 90),
        "mean_hist_bins": float(np.mean(hist_bins)) if hist_bins else float("nan"),
        "accepted_moves": int(accepted_moves),
    }


def write_calibration_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="按接受概率公式校准 T/k/bin_width 并验证。")
    parser.add_argument("--instance", default="Du62")
    parser.add_argument("--bin-widths", default="5,10,20,50,100")
    parser.add_argument("--target-accepts", default="0.03,0.05,0.10,0.20")
    parser.add_argument("--lambdas", default="0,0.001,0.002,0.005,0.01")
    parser.add_argument("--seeds", default="20260510,20260511,20260512")
    parser.add_argument("--calibration-seeds", default="")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--sample-steps", type=int, default=3000)
    parser.add_argument("--validation-steps", type=int, default=10000)
    parser.add_argument("--warmup-temperature", type=float, default=1000.0)
    parser.add_argument("--delta-quantile", type=float, default=75.0)
    parser.add_argument("--fallback-temperature", type=float, default=1000.0)
    parser.add_argument("--probe-accept-prob", type=float, default=0.05)
    parser.add_argument(
        "--include-infeasible-calibration",
        action="store_true",
        help="校准 T 时允许使用不可行代理能量差；默认只用可行 fitness 差。",
    )
    parser.add_argument("--greedy-steps", type=int, default=0)
    parser.add_argument(
        "--output-file",
        default=str(ROOT / "files" / "elp_param_sweeps" / "elp_fixed_temp_sweep_all.csv"),
    )
    parser.add_argument(
        "--calibration-output",
        default="",
        help="校准统计 CSV；默认写入 files/elp_param_sweeps 下的 tag 文件。",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--remark", default="")
    args = parser.parse_args()

    bin_widths = parse_float_list(args.bin_widths)
    target_accepts = parse_float_list(args.target_accepts)
    lambdas = parse_float_list(args.lambdas)
    seeds = parse_int_list(args.seeds)
    calibration_seeds = parse_int_list(args.calibration_seeds) if str(args.calibration_seeds).strip() else list(seeds)
    tag = args.tag or f"calibrated_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    remark = str(args.remark).strip() or "按接受概率公式校准T和k"
    output_file = Path(args.output_file)
    calibration_output = (
        Path(args.calibration_output)
        if str(args.calibration_output).strip()
        else ROOT / "files" / "elp_param_sweeps" / f"{tag}_calibration.csv"
    )
    created_at = datetime.now().isoformat(timespec="seconds")

    calibration_rows = []
    validation_total = len(bin_widths) * len(target_accepts) * len(lambdas) * len(seeds)
    validation_index = 0

    for bin_width in bin_widths:
        print(
            f"CALIBRATE bin_width={bin_width:g} warmup={args.warmup_steps} "
            f"sample={args.sample_steps} seeds={','.join(str(v) for v in calibration_seeds)}"
        )
        calibration_stats = calibrate_for_bin_width(
            instance=args.instance,
            bin_width=bin_width,
            calibration_seeds=calibration_seeds,
            warmup_steps=args.warmup_steps,
            sample_steps=args.sample_steps,
            warmup_temperature=args.warmup_temperature,
            probe_accept_prob=args.probe_accept_prob,
            feasible_only=not bool(args.include_infeasible_calibration),
        )
        public_calibration_stats = {
            key: value
            for key, value in calibration_stats.items()
            if key != "delta_positive_values"
        }

        for target_accept in target_accepts:
            calibrated_temperature = calibrate_temperature_from_delta(
                delta_positive=calibration_stats["delta_positive_values"],
                target_accept=target_accept,
                quantile=args.delta_quantile,
                fallback=args.fallback_temperature,
            )
            for lambda_value in lambdas:
                calibrated_k = float(lambda_value) * float(calibrated_temperature)
                calib_row = {
                    "sweep_tag": tag,
                    "remark": remark,
                    "created_at": created_at,
                    "instance": args.instance,
                    "bin_width": float(bin_width),
                    "target_accept": float(target_accept),
                    "lambda": float(lambda_value),
                    "calibrated_temperature": float(calibrated_temperature),
                    "calibrated_k": float(calibrated_k),
                    "delta_quantile": float(args.delta_quantile),
                    **public_calibration_stats,
                }
                calibration_rows.append(calib_row)

                for seed in seeds:
                    validation_index += 1
                    print(
                        f"[{validation_index}/{validation_total}] bw={bin_width:g} "
                        f"p0={target_accept:g} lambda={lambda_value:g} "
                        f"T={calibrated_temperature:.6f} k={calibrated_k:.6f} seed={seed}"
                    )
                    row = run_one(
                        instance=args.instance,
                        seed=seed,
                        steps=args.validation_steps,
                        temperature=calibrated_temperature,
                        k_value=calibrated_k,
                        bin_width_spec=str(float(bin_width)),
                        greedy_steps=args.greedy_steps,
                        adaptive_refresh_interval=200,
                    )
                    row = {
                        "sweep_tag": tag,
                        "remark": remark,
                        "created_at": created_at,
                        "calibration_method": "T=-Qpos/log(p0), k=lambda*T",
                        "target_accept": float(target_accept),
                        "lambda": float(lambda_value),
                        "calibrated_temperature": float(calibrated_temperature),
                        "calibrated_k": float(calibrated_k),
                        "delta_quantile": float(args.delta_quantile),
                        "calibration_positive_count": int(calibration_stats["positive_count"]),
                        "calibration_delta_positive_q75": float(calibration_stats["delta_positive_q75"]),
                        "calibration_delta_h_q50": float(calibration_stats["delta_h_q50"]),
                        **row,
                    }
                    append_row(output_file, row)
                    print(
                        f"    best={row['best_energy']:.6f} prob={row['prob_mean']:.4f} "
                        f"runtime={row['runtime_sec']:.1f}s"
                    )

    write_calibration_rows(calibration_output, calibration_rows)
    print(f"OUTPUT_CSV={output_file}")
    print(f"CALIBRATION_CSV={calibration_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
