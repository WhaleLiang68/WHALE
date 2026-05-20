"""使用 Optuna/TPE 自动寻找 ELP.py 的 T、k、bin_width。

适用场景：
    ELP 参数寻优是有噪声、单次运行昂贵的黑箱优化问题。参数只有 3 个，
    不适合用深度学习拟合，适合用 TPE/贝叶斯优化一类小样本方法。

设计原则：
    1. Optuna 只负责提出候选参数，不替代最终验证。
    2. 每个 trial 按阶段增加预算，差参数提前剪枝。
    3. 评分采用 mean(best_energy) + alpha * std(best_energy)，避免偶然 seed 误导。
    4. 结果仍写入统一 CSV，便于和前面的实验一起分析。
    5. 支持断点续跑：同一 tag/stage/seed/T/k/bin_width 已有结果会复用。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import optuna
except ModuleNotFoundError as exc:  # pragma: no cover - 用于给用户明确环境错误
    raise SystemExit(
        "缺少依赖 optuna。请先在 tensorflow 环境中安装：\n"
        "C:\\Users\\17122\\AppData\\Local\\conda\\conda\\envs\\tensorflow\\python.exe -m pip install optuna"
    ) from exc

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_elp_fixed_temp import append_row, parse_float_list, parse_int_list, run_one  # noqa: E402


@dataclass(frozen=True)
class StageConfig:
    index: int
    steps: int
    seeds: tuple[int, ...]


def parse_stage_seeds(raw: str) -> list[tuple[int, ...]]:
    result = []
    for stage_part in str(raw).split(";"):
        stage_part = stage_part.strip()
        if stage_part:
            result.append(tuple(parse_int_list(stage_part)))
    if not result:
        raise ValueError("stage-seeds 不能为空")
    return result


def parse_int_list_arg(raw: str) -> list[int]:
    result = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError(f"整数列表为空: {raw!r}")
    return result


def build_stages(stage_steps: str, stage_seeds: str) -> list[StageConfig]:
    steps = parse_int_list_arg(stage_steps)
    seeds = parse_stage_seeds(stage_seeds)
    if len(steps) != len(seeds):
        raise ValueError(f"stage-steps 与 stage-seeds 阶段数量不一致: {len(steps)} vs {len(seeds)}")
    return [
        StageConfig(index=index + 1, steps=int(step), seeds=tuple(seeds[index]))
        for index, step in enumerate(steps)
    ]


def float_key(value: object) -> str:
    return f"{float(value):.12g}"


def run_key(tag: str, stage_index: int, steps: int, seed: int, temperature: float, k: float, bin_width: float):
    return (
        str(tag),
        str(int(stage_index)),
        str(int(steps)),
        str(int(seed)),
        float_key(temperature),
        float_key(k),
        float_key(bin_width),
    )


def row_to_run_key(row: dict) -> tuple[str, ...] | None:
    try:
        return run_key(
            tag=str(row.get("sweep_tag", "")),
            stage_index=int(float(row.get("optimization_stage", ""))),
            steps=int(float(row.get("steps", ""))),
            seed=int(float(row.get("seed", ""))),
            temperature=float(row.get("temperature")),
            k=float(row.get("k")),
            bin_width=float(row.get("bin_width")),
        )
    except Exception:
        return None


def load_existing_rows(output_file: Path, tag: str) -> dict[tuple[str, ...], dict]:
    existing: dict[tuple[str, ...], dict] = {}
    if not output_file.exists() or output_file.stat().st_size == 0:
        return existing
    with output_file.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if row.get("sweep_tag") != tag:
                continue
            key = row_to_run_key(row)
            if key is not None:
                # 若历史中重复写入同一运行，只保留最后一条。
                existing[key] = row
    return existing


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def mean_with_invalid_penalty(rows: list[dict], invalid_penalty: float) -> tuple[float, float, float, float]:
    values = []
    valid_flags = []
    for row in rows:
        value = finite_float(row.get("best_energy"), default=float("nan"))
        is_valid = str(row.get("is_valid", "")).lower() == "true"
        if math.isfinite(value) and is_valid:
            values.append(value)
            valid_flags.append(1.0)
        else:
            values.append(float(invalid_penalty))
            valid_flags.append(0.0)
    if not values:
        return float(invalid_penalty), 0.0, float(invalid_penalty), 0.0
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=0)), float(np.min(array)), float(np.mean(valid_flags))


def stage_score(rows: list[dict], score_alpha: float, invalid_penalty: float) -> dict:
    mean_best, std_best, min_best, valid_rate = mean_with_invalid_penalty(rows, invalid_penalty)
    runtimes = [finite_float(row.get("runtime_sec"), default=math.nan) for row in rows]
    probs = [finite_float(row.get("prob_mean"), default=math.nan) for row in rows]
    finite_runtimes = [value for value in runtimes if math.isfinite(value)]
    finite_probs = [value for value in probs if math.isfinite(value)]
    return {
        "mean_best_energy": mean_best,
        "std_best_energy": std_best,
        "min_best_energy": min_best,
        "selection_score": mean_best + float(score_alpha) * std_best,
        "valid_rate": valid_rate,
        "mean_runtime_sec": float(np.mean(finite_runtimes)) if finite_runtimes else float("nan"),
        "mean_prob": float(np.mean(finite_probs)) if finite_probs else float("nan"),
    }


def parse_param_triples(raw: str) -> list[dict]:
    triples = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.replace("/", ":").split(":")]
        if len(pieces) != 3:
            raise ValueError(f"参数三元组格式应为 T:k:bin_width，收到: {part!r}")
        triples.append(
            {
                "temperature": float(pieces[0]),
                "k": float(pieces[1]),
                "bin_width": float(pieces[2]),
            }
        )
    return triples


def normalize_candidate(candidate: dict, bin_width_choices: list[float], k_step: float) -> dict:
    bin_width = float(candidate["bin_width"])
    if bin_width not in bin_width_choices:
        closest = min(bin_width_choices, key=lambda value: abs(value - bin_width))
        bin_width = float(closest)
    k_value = float(candidate["k"])
    if k_step > 0:
        k_value = round(k_value / float(k_step)) * float(k_step)
    return {
        "temperature": float(candidate["temperature"]),
        "k": float(k_value),
        "bin_width": float(bin_width),
    }


def warm_start_from_csv(
    output_file: Path,
    instance: str,
    top_n: int,
    temperature_low: float,
    temperature_high: float,
    k_low: float,
    k_high: float,
    bin_width_choices: list[float],
    k_step: float,
) -> list[dict]:
    if top_n <= 0 or not output_file.exists() or output_file.stat().st_size == 0:
        return []
    grouped: dict[tuple[str, str, str], list[float]] = {}
    with output_file.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if row.get("instance") != instance:
                continue
            temperature = finite_float(row.get("temperature"))
            k_value = finite_float(row.get("k"))
            bin_width = finite_float(row.get("bin_width"))
            best_energy = finite_float(row.get("best_energy"))
            if not all(math.isfinite(value) for value in (temperature, k_value, bin_width, best_energy)):
                continue
            if not (temperature_low <= temperature <= temperature_high and k_low <= k_value <= k_high):
                continue
            if bin_width not in bin_width_choices:
                continue
            key = (float_key(temperature), float_key(k_value), float_key(bin_width))
            grouped.setdefault(key, []).append(best_energy)

    summary = []
    for key, values in grouped.items():
        summary.append(
            {
                "temperature": float(key[0]),
                "k": float(key[1]),
                "bin_width": float(key[2]),
                "mean_best": float(np.mean(np.asarray(values, dtype=float))),
                "runs": len(values),
            }
        )
    summary.sort(key=lambda row: (row["mean_best"], -row["runs"]))
    return [
        normalize_candidate(row, bin_width_choices=bin_width_choices, k_step=k_step)
        for row in summary[: int(top_n)]
    ]


def unique_param_dicts(candidates: Iterable[dict]) -> list[dict]:
    seen = set()
    result = []
    for candidate in candidates:
        key = (float_key(candidate["temperature"]), float_key(candidate["k"]), float_key(candidate["bin_width"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def write_study_summary(study: optuna.Study, summary_file: Path, tag: str) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        row = {
            "sweep_tag": tag,
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value if trial.value is not None else "",
            "temperature": trial.params.get("temperature", ""),
            "k": trial.params.get("k", ""),
            "bin_width": trial.params.get("bin_width", ""),
            "stage1_score": trial.intermediate_values.get(1, ""),
            "stage2_score": trial.intermediate_values.get(2, ""),
            "stage3_score": trial.intermediate_values.get(3, ""),
            "mean_best_energy": trial.user_attrs.get("mean_best_energy", ""),
            "std_best_energy": trial.user_attrs.get("std_best_energy", ""),
            "min_best_energy": trial.user_attrs.get("min_best_energy", ""),
            "valid_rate": trial.user_attrs.get("valid_rate", ""),
            "mean_prob": trial.user_attrs.get("mean_prob", ""),
            "mean_runtime_sec": trial.user_attrs.get("mean_runtime_sec", ""),
        }
        rows.append(row)
    fieldnames = [
        "sweep_tag",
        "trial_number",
        "state",
        "value",
        "temperature",
        "k",
        "bin_width",
        "stage1_score",
        "stage2_score",
        "stage3_score",
        "mean_best_energy",
        "std_best_energy",
        "min_best_energy",
        "valid_rate",
        "mean_prob",
        "mean_runtime_sec",
    ]
    with summary_file.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary_points(summary_file: Path) -> list[dict]:
    """读取 Optuna 摘要，优先使用最终阶段分数作为可视化目标。"""
    if not summary_file.exists() or summary_file.stat().st_size == 0:
        return []
    points = []
    with summary_file.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            temperature = finite_float(row.get("temperature"))
            k_value = finite_float(row.get("k"))
            bin_width = finite_float(row.get("bin_width"))
            score = finite_float(row.get("stage3_score"))
            score_source = "stage3_score"
            if not math.isfinite(score):
                score = finite_float(row.get("value"))
                score_source = "value"
            if not all(math.isfinite(value) for value in (temperature, k_value, bin_width, score)):
                continue
            points.append(
                {
                    "trial_number": int(finite_float(row.get("trial_number"), default=-1)),
                    "temperature": float(temperature),
                    "k": float(k_value),
                    "bin_width": float(bin_width),
                    "score": float(score),
                    "score_source": score_source,
                    "state": row.get("state", ""),
                }
            )
    return points


def save_optuna_builtin_plots(study: optuna.Study, artifacts_dir: Path) -> list[str]:
    """保存 Optuna 内置图；这些图只使用 COMPLETE trial。"""
    messages = []
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) < 2:
        return [f"SKIP optuna builtin plots: COMPLETE trial 数量不足 ({len(complete_trials)})"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as optuna_plot

    plot_jobs = [
        ("optuna_optimization_history.png", lambda: optuna_plot.plot_optimization_history(study)),
        ("optuna_param_importance_complete_trials.png", lambda: optuna_plot.plot_param_importances(study)),
        (
            "optuna_contour_temperature_k_complete_trials.png",
            lambda: optuna_plot.plot_contour(study, params=["temperature", "k"]),
        ),
    ]
    for filename, plotter in plot_jobs:
        output_path = artifacts_dir / filename
        try:
            ax = plotter()
            figure = ax.figure
            figure.tight_layout()
            figure.savefig(output_path, dpi=180)
            plt.close(figure)
            messages.append(f"OK {output_path}")
        except Exception as exc:
            messages.append(f"FAIL {filename}: {type(exc).__name__}: {exc}")
            plt.close("all")
    return messages


def save_surrogate_importance(points: list[dict], artifacts_dir: Path) -> list[str]:
    """基于摘要中所有可用 trial 训练轻量代理模型并输出参数重要性。"""
    messages = []
    if len(points) < 5:
        return [f"SKIP surrogate importance: 可用点数量不足 ({len(points)})"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor

    feature_names = ["temperature", "k", "bin_width"]
    x = np.asarray([[point[name] for name in feature_names] for point in points], dtype=float)
    y = np.asarray([point["score"] for point in points], dtype=float)
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=20260514,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)

    csv_path = artifacts_dir / "surrogate_param_importance_all_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["parameter", "importance"])
        writer.writeheader()
        for name, importance in pairs:
            writer.writerow({"parameter": name, "importance": float(importance)})
    messages.append(f"OK {csv_path}")

    figure, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar([name for name, _ in pairs], [importance for _, importance in pairs], color="#3b6ea8")
    ax.set_title("Parameter Importance From All Available Trials")
    ax.set_ylabel("Random forest importance")
    ax.set_xlabel("Parameter")
    figure.tight_layout()
    png_path = artifacts_dir / "surrogate_param_importance_all_trials.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    messages.append(f"OK {png_path}")
    return messages


def save_surrogate_contours(points: list[dict], artifacts_dir: Path) -> list[str]:
    """基于摘要点保存二维等高线图，目标值越低越好。"""
    messages = []
    if len(points) < 5:
        return [f"SKIP surrogate contours: 可用点数量不足 ({len(points)})"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("temperature", "k", "surrogate_contour_temperature_k_all_trials.png"),
        ("temperature", "bin_width", "surrogate_contour_temperature_bin_width_all_trials.png"),
        ("k", "bin_width", "surrogate_contour_k_bin_width_all_trials.png"),
    ]
    for x_name, y_name, filename in plot_specs:
        # 同一二维坐标下可能有第三个参数不同的多条 trial；等高线取该坐标下的最好分数。
        coord_best: dict[tuple[float, float], float] = {}
        for point in points:
            coord = (float(point[x_name]), float(point[y_name]))
            score = float(point["score"])
            coord_best[coord] = min(coord_best.get(coord, float("inf")), score)
        x = np.asarray([coord[0] for coord in coord_best], dtype=float)
        y = np.asarray([coord[1] for coord in coord_best], dtype=float)
        z = np.asarray([score for score in coord_best.values()], dtype=float)
        if len(set(np.round(x, 12))) < 2 or len(set(np.round(y, 12))) < 2:
            messages.append(f"SKIP {filename}: {x_name}/{y_name} 唯一值不足")
            continue
        figure, ax = plt.subplots(figsize=(7.0, 5.2))
        try:
            contour = ax.tricontourf(x, y, z, levels=14, cmap="viridis_r")
            figure.colorbar(contour, ax=ax, label="selection score (lower is better)")
        except Exception as exc:
            plt.close(figure)
            messages.append(f"FAIL {filename}: {type(exc).__name__}: {exc}")
            continue
        scatter = ax.scatter(x, y, c=z, cmap="viridis_r", edgecolor="black", linewidth=0.4, s=38)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"Surrogate Contour: {x_name} vs {y_name}")
        figure.colorbar(scatter, ax=ax, label="observed score")
        figure.tight_layout()
        output_path = artifacts_dir / filename
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        messages.append(f"OK {output_path}")
    return messages


def write_visualization_artifacts(study: optuna.Study, summary_file: Path, artifacts_dir: Path) -> None:
    """生成参数重要性和等高线图，并写入 manifest 说明使用的数据来源。"""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    messages = []
    messages.extend(save_optuna_builtin_plots(study, artifacts_dir))
    points = load_summary_points(summary_file)
    messages.append(f"SUMMARY_POINTS={len(points)}")
    messages.extend(save_surrogate_importance(points, artifacts_dir))
    messages.extend(save_surrogate_contours(points, artifacts_dir))
    manifest_path = artifacts_dir / "artifact_manifest.txt"
    manifest_path.write_text("\n".join(messages) + "\n", encoding="utf-8")
    print(f"ARTIFACTS_DIR={artifacts_dir}")
    for message in messages:
        print(f"  {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 Optuna/TPE 自动寻找 ELP.py 的 T/k/bin_width。")
    parser.add_argument("--instance", default="Du62")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=0, help="总超时秒数；0 表示不限制。")
    parser.add_argument("--temperature-low", type=float, default=400.0)
    parser.add_argument("--temperature-high", type=float, default=2000.0)
    parser.add_argument("--temperature-log", action="store_true", help="对 T 使用 log 采样；默认线性采样。")
    parser.add_argument("--k-low", type=float, default=0.0)
    parser.add_argument("--k-high", type=float, default=12.0)
    parser.add_argument("--k-step", type=float, default=0.5)
    parser.add_argument("--bin-widths", default="1,2,5,10,20,30,50,80,100")
    parser.add_argument("--stage-steps", default="5000,10000,20000")
    parser.add_argument(
        "--stage-seeds",
        default=(
            "20260510,20260511;"
            "20260510,20260511,20260512,20260513,20260514;"
            "20260510,20260511,20260512,20260513,20260514,20260515,20260516,20260517,20260518,20260519"
        ),
    )
    parser.add_argument("--score-alpha", type=float, default=0.25)
    parser.add_argument("--invalid-penalty", type=float, default=1e18)
    parser.add_argument("--greedy-steps", type=int, default=0)
    parser.add_argument("--adaptive-refresh-interval", type=int, default=200)
    parser.add_argument("--sampler-seed", type=int, default=20260513)
    parser.add_argument("--n-startup-trials", type=int, default=12)
    parser.add_argument("--warm-start-top-n", type=int, default=8)
    parser.add_argument(
        "--enqueue-params",
        default=(
            "1000:2:20,1000:4:20,1000:0:20,1000:20:20,"
            "800:2:20,1200:2:20,1000:2:10,1000:2:30"
        ),
        help="手动 warm-start 参数，格式 T:k:bin_width，多个用逗号分隔。",
    )
    parser.add_argument(
        "--output-file",
        default=str(ROOT / "files" / "elp_param_sweeps" / "elp_fixed_temp_sweep_all.csv"),
    )
    parser.add_argument("--storage-file", default="")
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--no-visualizations", action="store_true", help="只写 CSV 摘要，不生成可视化图。")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--study-name", default="")
    parser.add_argument("--remark", default="")
    args = parser.parse_args()

    tag = args.tag or f"stage_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_name = str(args.study_name).strip() or tag
    remark = str(args.remark).strip() or "Optuna/TPE 自动寻找 T/k/bin_width"
    output_file = Path(args.output_file)
    storage_file = (
        Path(args.storage_file)
        if str(args.storage_file).strip()
        else ROOT / "files" / "elp_param_sweeps" / f"{tag}.db"
    )
    summary_file = (
        Path(args.summary_file)
        if str(args.summary_file).strip()
        else ROOT / "files" / "elp_param_sweeps" / f"{tag}_optuna_summary.csv"
    )
    artifacts_dir = (
        Path(args.artifacts_dir)
        if str(args.artifacts_dir).strip()
        else ROOT / "files" / "elp_param_sweeps" / f"{tag}_artifacts"
    )
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_file.as_posix()}"

    stages = build_stages(args.stage_steps, args.stage_seeds)
    bin_width_choices = [float(value) for value in parse_float_list(args.bin_widths)]
    created_at = datetime.now().isoformat(timespec="seconds")

    sampler = optuna.samplers.TPESampler(
        seed=int(args.sampler_seed),
        n_startup_trials=int(args.n_startup_trials),
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    manual_warm_start = [
        normalize_candidate(candidate, bin_width_choices=bin_width_choices, k_step=float(args.k_step))
        for candidate in parse_param_triples(args.enqueue_params)
    ]
    csv_warm_start = warm_start_from_csv(
        output_file=output_file,
        instance=args.instance,
        top_n=int(args.warm_start_top_n),
        temperature_low=float(args.temperature_low),
        temperature_high=float(args.temperature_high),
        k_low=float(args.k_low),
        k_high=float(args.k_high),
        bin_width_choices=bin_width_choices,
        k_step=float(args.k_step),
    )
    if int(args.n_trials) > 0:
        for candidate in unique_param_dicts([*manual_warm_start, *csv_warm_start]):
            study.enqueue_trial(candidate, skip_if_exists=True)

    print(f"TAG={tag}")
    print(f"STUDY={study_name}")
    print(f"STORAGE={storage_file}")
    print(f"OUTPUT_CSV={output_file}")
    print(f"SUMMARY_CSV={summary_file}")
    print(f"STAGES={[(stage.steps, stage.seeds) for stage in stages]}")
    print(f"BIN_WIDTHS={bin_width_choices}")

    def objective(trial: optuna.Trial) -> float:
        temperature = trial.suggest_float(
            "temperature",
            float(args.temperature_low),
            float(args.temperature_high),
            log=bool(args.temperature_log),
        )
        if float(args.k_step) > 0:
            k_value = trial.suggest_float(
                "k",
                float(args.k_low),
                float(args.k_high),
                step=float(args.k_step),
            )
        else:
            k_value = trial.suggest_float("k", float(args.k_low), float(args.k_high))
        bin_width = float(trial.suggest_categorical("bin_width", bin_width_choices))

        last_score = float("inf")
        for stage in stages:
            existing = load_existing_rows(output_file, tag)
            rows = []
            for seed in stage.seeds:
                key = run_key(
                    tag=tag,
                    stage_index=stage.index,
                    steps=stage.steps,
                    seed=seed,
                    temperature=temperature,
                    k=k_value,
                    bin_width=bin_width,
                )
                if key in existing:
                    row = existing[key]
                    rows.append(row)
                    print(
                        f"SKIP trial={trial.number} stage={stage.index} seed={seed} "
                        f"T={temperature:.6g} k={k_value:.6g} bw={bin_width:.6g}"
                    )
                    continue

                print(
                    f"RUN trial={trial.number} stage={stage.index} seed={seed} "
                    f"T={temperature:.6g} k={k_value:.6g} bw={bin_width:.6g} steps={stage.steps}"
                )
                row = run_one(
                    instance=args.instance,
                    seed=int(seed),
                    steps=int(stage.steps),
                    temperature=float(temperature),
                    k_value=float(k_value),
                    bin_width_spec=str(float(bin_width)),
                    greedy_steps=int(args.greedy_steps),
                    adaptive_refresh_interval=int(args.adaptive_refresh_interval),
                )
                row = {
                    "sweep_tag": tag,
                    "remark": remark,
                    "created_at": created_at,
                    "optimizer": "optuna_tpe_successive_halving",
                    "study_name": study_name,
                    "trial_number": int(trial.number),
                    "optimization_stage": int(stage.index),
                    "stage_expected_runs": int(len(stage.seeds)),
                    "selection_score_alpha": float(args.score_alpha),
                    **row,
                }
                append_row(output_file, row)
                rows.append(row)
                print(
                    f"    best={float(row['best_energy']):.6f} valid={row['is_valid']} "
                    f"prob={float(row['prob_mean']):.4f} runtime={float(row['runtime_sec']):.1f}s"
                )

            metrics = stage_score(rows, score_alpha=float(args.score_alpha), invalid_penalty=float(args.invalid_penalty))
            last_score = float(metrics["selection_score"])
            trial.report(last_score, step=int(stage.index))
            for key, value in metrics.items():
                trial.set_user_attr(key, float(value))
            trial.set_user_attr("last_completed_stage", int(stage.index))

            print(
                f"STAGE_SCORE trial={trial.number} stage={stage.index} "
                f"score={last_score:.6f} mean={metrics['mean_best_energy']:.6f} "
                f"std={metrics['std_best_energy']:.6f} valid_rate={metrics['valid_rate']:.2f} "
                f"prob={metrics['mean_prob']:.4f}"
            )
            is_final_stage = stage.index == stages[-1].index
            if (not is_final_stage) and trial.should_prune():
                raise optuna.TrialPruned(f"stage={stage.index} score={last_score:.6f}")
        return last_score

    timeout = int(args.timeout) if int(args.timeout) > 0 else None
    if int(args.n_trials) > 0:
        study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout, gc_after_trial=True)
    write_study_summary(study, summary_file=summary_file, tag=tag)
    if not bool(args.no_visualizations):
        write_visualization_artifacts(study, summary_file=summary_file, artifacts_dir=artifacts_dir)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print("BEST_TRIAL")
        print(f"  number={study.best_trial.number}")
        print(f"  value={study.best_value:.6f}")
        print(f"  params={study.best_params}")
    else:
        print("BEST_TRIAL unavailable: 没有 COMPLETE trial")
    print("TOP_TRIALS")
    completed_trials.sort(key=lambda trial: trial.value if trial.value is not None else float("inf"))
    for rank, trial in enumerate(completed_trials[:10], start=1):
        print(
            f"  {rank:02d} trial={trial.number} value={trial.value:.6f} "
            f"T={trial.params.get('temperature')} k={trial.params.get('k')} bw={trial.params.get('bin_width')} "
            f"stage={trial.user_attrs.get('last_completed_stage')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
