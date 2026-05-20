"""用逐轮加预算方式联合寻找 ELP.py 的 T、k、bin_width。

这不是再次按接受概率公式直接反推参数。上一版校准容易被大跳变
delta 支配，导致 T 被推到百万级。本脚本采用更稳妥的实验寻优：

1. 在已知有效区间内构造 T/k/bin_width 候选网格。
2. 第一轮用少量 seed 和较小步数粗筛。
3. 后续轮次只保留前 N 个候选，并增加 seed/步数。
4. 全部结果追加到统一 CSV，按 sweep_tag 和 optimization_stage 区分。

脚本支持断点续跑：同一 tag/stage/steps/seed/T/k/bin_width 已存在的结果会跳过。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_elp_fixed_temp import append_row, parse_float_list, parse_int_list, run_one  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    temperature: float
    k: float
    bin_width: float


@dataclass(frozen=True)
class StageConfig:
    index: int
    steps: int
    seeds: tuple[int, ...]
    advance_top_n: int


def parse_stage_seeds(raw: str) -> list[tuple[int, ...]]:
    stages = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        stages.append(tuple(parse_int_list(part)))
    if not stages:
        raise ValueError("stage seeds 不能为空")
    return stages


def parse_int_stage_values(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"阶段参数为空: {raw!r}")
    return values


def finite_mean(values) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def finite_std(values) -> float:
    finite_values = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.std(finite_values, ddof=0))


def float_key(value: object) -> str:
    return f"{float(value):.12g}"


def run_key(tag: str, stage_index: int, steps: int, seed: int, candidate: Candidate) -> tuple[str, ...]:
    return (
        str(tag),
        str(stage_index),
        str(int(steps)),
        str(int(seed)),
        float_key(candidate.temperature),
        float_key(candidate.k),
        float_key(candidate.bin_width),
    )


def row_to_run_key(row: dict) -> tuple[str, ...] | None:
    try:
        return (
            str(row.get("sweep_tag", "")),
            str(int(float(row.get("optimization_stage", "")))),
            str(int(float(row.get("steps", "")))),
            str(int(float(row.get("seed", "")))),
            float_key(row.get("temperature")),
            float_key(row.get("k")),
            float_key(row.get("bin_width")),
        )
    except Exception:
        return None


def load_existing_rows(path: Path, tag: str) -> dict[tuple[str, ...], dict]:
    existing = {}
    if not path.exists() or path.stat().st_size == 0:
        return existing
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if row.get("sweep_tag") != tag:
                continue
            key = row_to_run_key(row)
            if key is not None:
                # 若曾经重复写入，保留最后一条，避免汇总重复计数。
                existing[key] = row
    return existing


def candidate_key(candidate: Candidate) -> tuple[str, str, str]:
    return (float_key(candidate.temperature), float_key(candidate.k), float_key(candidate.bin_width))


def make_candidates(
    temperatures: list[float],
    k_values: list[float],
    bin_widths: list[float],
    candidate_limit: int,
    candidate_seed: int,
) -> list[Candidate]:
    candidates = [
        Candidate(float(temperature), float(k_value), float(bin_width))
        for temperature, k_value, bin_width in itertools.product(temperatures, k_values, bin_widths)
    ]
    # 先按数值排序，保证完整网格时结果可复现。
    candidates.sort(key=lambda item: (item.temperature, item.k, item.bin_width))
    if candidate_limit <= 0 or candidate_limit >= len(candidates):
        return candidates

    # 限制候选数量时，使用固定随机种子抽样，并强制保留已知有效中心点。
    rng = np.random.default_rng(int(candidate_seed))
    center = Candidate(1000.0, 2.0, 20.0)
    selected = []
    if center in candidates:
        selected.append(center)
    remaining = [candidate for candidate in candidates if candidate not in selected]
    sample_size = max(0, int(candidate_limit) - len(selected))
    if sample_size > 0:
        indices = rng.choice(len(remaining), size=sample_size, replace=False)
        selected.extend(remaining[int(index)] for index in indices)
    selected.sort(key=lambda item: (item.temperature, item.k, item.bin_width))
    return selected


def build_stages(args) -> list[StageConfig]:
    steps = parse_int_stage_values(args.stage_steps)
    top_ns = parse_int_stage_values(args.stage_top_ns)
    seed_stages = parse_stage_seeds(args.stage_seeds)
    if not (len(steps) == len(top_ns) == len(seed_stages)):
        raise ValueError(
            "stage_steps、stage_top_ns、stage_seeds 的阶段数量必须一致；"
            f"当前分别为 {len(steps)}, {len(top_ns)}, {len(seed_stages)}"
        )
    return [
        StageConfig(index=index + 1, steps=int(step), seeds=tuple(seed_stages[index]), advance_top_n=int(top_ns[index]))
        for index, step in enumerate(steps)
    ]


def summarize_stage_rows(
    rows_by_key: dict[tuple[str, ...], dict],
    tag: str,
    stage: StageConfig,
    candidates: list[Candidate],
    score_alpha: float,
) -> list[dict]:
    candidate_lookup = {candidate_key(candidate): candidate for candidate in candidates}
    grouped: dict[tuple[str, str, str], list[dict]] = {key: [] for key in candidate_lookup}
    for key, row in rows_by_key.items():
        if key[0] != tag or key[1] != str(stage.index) or key[2] != str(stage.steps):
            continue
        c_key = (key[4], key[5], key[6])
        if c_key in grouped:
            grouped[c_key].append(row)

    summary_rows = []
    for c_key, rows in grouped.items():
        candidate = candidate_lookup[c_key]
        best_values = []
        runtime_values = []
        prob_values = []
        valid_values = []
        for row in rows:
            try:
                best_values.append(float(row["best_energy"]))
                runtime_values.append(float(row.get("runtime_sec", "nan")))
                prob_values.append(float(row.get("prob_mean", "nan")))
                valid_values.append(1.0 if str(row.get("is_valid", "")).lower() == "true" else 0.0)
            except Exception:
                continue
        runs = len(best_values)
        mean_best = finite_mean(best_values)
        std_best = finite_std(best_values)
        score = mean_best + float(score_alpha) * std_best if np.isfinite(mean_best) else float("inf")
        summary_rows.append(
            {
                "sweep_tag": tag,
                "optimization_stage": int(stage.index),
                "steps": int(stage.steps),
                "expected_runs": int(len(stage.seeds)),
                "runs": int(runs),
                "temperature": float(candidate.temperature),
                "k": float(candidate.k),
                "bin_width": float(candidate.bin_width),
                "mean_best_energy": float(mean_best),
                "min_best_energy": float(np.min(best_values)) if best_values else float("nan"),
                "max_best_energy": float(np.max(best_values)) if best_values else float("nan"),
                "std_best_energy": float(std_best),
                "selection_score": float(score),
                "mean_runtime_sec": finite_mean(runtime_values),
                "mean_prob": finite_mean(prob_values),
                "valid_rate": finite_mean(valid_values),
                "completed": bool(runs == len(stage.seeds)),
            }
        )
    summary_rows.sort(
        key=lambda item: (
            not item["completed"],
            item["selection_score"],
            item["mean_best_energy"],
            item["std_best_energy"],
            item["mean_runtime_sec"],
        )
    )
    return summary_rows


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="逐轮联合寻找 ELP.py 的 T/k/bin_width。")
    parser.add_argument("--instance", default="Du62")
    parser.add_argument("--temperatures", default="500,700,900,1000,1100,1300,1600")
    parser.add_argument("--k-values", default="0,1,2,3,4,6,8")
    parser.add_argument("--bin-widths", default="5,10,20,30,50")
    parser.add_argument(
        "--stage-steps",
        default="5000,10000,20000",
        help="各阶段总步数，逗号分隔。",
    )
    parser.add_argument(
        "--stage-seeds",
        default=(
            "20260510,20260511;"
            "20260510,20260511,20260512,20260513,20260514;"
            "20260510,20260511,20260512,20260513,20260514,20260515,20260516,20260517,20260518,20260519"
        ),
        help="各阶段 seed；阶段之间用分号分隔，阶段内部用逗号分隔。",
    )
    parser.add_argument(
        "--stage-top-ns",
        default="40,8,0",
        help="各阶段完成后保留前 N 个进入下一阶段；最后一阶段填 0。",
    )
    parser.add_argument("--candidate-limit", type=int, default=0, help="第一阶段候选上限；0 表示完整网格。")
    parser.add_argument("--candidate-seed", type=int, default=20260513)
    parser.add_argument("--score-alpha", type=float, default=0.25, help="选择分数 = 均值 + alpha * 标准差。")
    parser.add_argument("--greedy-steps", type=int, default=0)
    parser.add_argument("--adaptive-refresh-interval", type=int, default=200)
    parser.add_argument(
        "--output-file",
        default=str(ROOT / "files" / "elp_param_sweeps" / "elp_fixed_temp_sweep_all.csv"),
    )
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--remark", default="")
    args = parser.parse_args()

    temperatures = parse_float_list(args.temperatures)
    k_values = parse_float_list(args.k_values)
    bin_widths = parse_float_list(args.bin_widths)
    stages = build_stages(args)
    tag = args.tag or f"stage_successive_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    remark = str(args.remark).strip() or "逐轮联合寻找 T/k/bin_width"
    output_file = Path(args.output_file)
    summary_file = (
        Path(args.summary_file)
        if str(args.summary_file).strip()
        else ROOT / "files" / "elp_param_sweeps" / f"{tag}_summary.csv"
    )
    created_at = datetime.now().isoformat(timespec="seconds")

    candidates = make_candidates(
        temperatures=temperatures,
        k_values=k_values,
        bin_widths=bin_widths,
        candidate_limit=int(args.candidate_limit),
        candidate_seed=int(args.candidate_seed),
    )
    all_summary_rows = []

    print(f"TAG={tag}")
    print(f"OUTPUT_CSV={output_file}")
    print(f"SUMMARY_CSV={summary_file}")
    print(f"INITIAL_CANDIDATES={len(candidates)}")

    for stage in stages:
        existing = load_existing_rows(output_file, tag)
        total_runs = len(candidates) * len(stage.seeds)
        current_run = 0
        print(
            f"STAGE {stage.index}: steps={stage.steps} seeds={','.join(str(v) for v in stage.seeds)} "
            f"candidates={len(candidates)} total_runs={total_runs}"
        )

        for candidate in candidates:
            for seed in stage.seeds:
                current_run += 1
                key = run_key(tag, stage.index, stage.steps, seed, candidate)
                if key in existing:
                    print(
                        f"[{current_run}/{total_runs}] SKIP stage={stage.index} seed={seed} "
                        f"T={candidate.temperature:g} k={candidate.k:g} bw={candidate.bin_width:g}"
                    )
                    continue

                print(
                    f"[{current_run}/{total_runs}] RUN stage={stage.index} seed={seed} "
                    f"T={candidate.temperature:g} k={candidate.k:g} bw={candidate.bin_width:g} "
                    f"steps={stage.steps}"
                )
                row = run_one(
                    instance=args.instance,
                    seed=seed,
                    steps=stage.steps,
                    temperature=candidate.temperature,
                    k_value=candidate.k,
                    bin_width_spec=str(float(candidate.bin_width)),
                    greedy_steps=int(args.greedy_steps),
                    adaptive_refresh_interval=int(args.adaptive_refresh_interval),
                )
                row = {
                    "sweep_tag": tag,
                    "remark": remark,
                    "created_at": created_at,
                    "optimization_stage": int(stage.index),
                    "stage_advance_top_n": int(stage.advance_top_n),
                    "selection_score_alpha": float(args.score_alpha),
                    **row,
                }
                append_row(output_file, row)
                existing[key] = row
                print(
                    f"    best={row['best_energy']:.6f} prob={row['prob_mean']:.4f} "
                    f"runtime={row['runtime_sec']:.1f}s"
                )

        existing = load_existing_rows(output_file, tag)
        stage_summary = summarize_stage_rows(
            rows_by_key=existing,
            tag=tag,
            stage=stage,
            candidates=candidates,
            score_alpha=float(args.score_alpha),
        )
        all_summary_rows.extend(stage_summary)
        write_summary(summary_file, all_summary_rows)

        print(f"STAGE {stage.index} TOP")
        for rank, row in enumerate(stage_summary[: min(10, len(stage_summary))], start=1):
            print(
                f"  {rank:02d} T={row['temperature']:g} k={row['k']:g} bw={row['bin_width']:g} "
                f"mean={row['mean_best_energy']:.6f} std={row['std_best_energy']:.6f} "
                f"score={row['selection_score']:.6f} runs={row['runs']}/{row['expected_runs']} "
                f"prob={row['mean_prob']:.4f}"
            )

        if stage.advance_top_n > 0:
            completed_rows = [row for row in stage_summary if row["completed"]]
            next_rows = completed_rows[: int(stage.advance_top_n)]
            if not next_rows:
                raise RuntimeError(f"第 {stage.index} 阶段没有完整候选，无法进入下一阶段。")
            candidates = [
                Candidate(
                    temperature=float(row["temperature"]),
                    k=float(row["k"]),
                    bin_width=float(row["bin_width"]),
                )
                for row in next_rows
            ]

    print("FINAL_TOP")
    final_stage = stages[-1]
    final_rows = [row for row in all_summary_rows if int(row["optimization_stage"]) == final_stage.index]
    final_rows.sort(
        key=lambda item: (
            not item["completed"],
            item["selection_score"],
            item["mean_best_energy"],
            item["std_best_energy"],
        )
    )
    for rank, row in enumerate(final_rows[: min(10, len(final_rows))], start=1):
        print(
            f"  {rank:02d} T={row['temperature']:g} k={row['k']:g} bw={row['bin_width']:g} "
            f"mean={row['mean_best_energy']:.6f} min={row['min_best_energy']:.6f} "
            f"max={row['max_best_energy']:.6f} std={row['std_best_energy']:.6f} "
            f"score={row['selection_score']:.6f} runs={row['runs']}/{row['expected_runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
