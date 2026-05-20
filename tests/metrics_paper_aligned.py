import os
import json
import csv
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ==========================================
# 配置区
# ==========================================
CSV_REGISTRY_PATH = r'C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\expresults\A-10-10-ELP_DRL_MO4_GRASP.csv'
OUTPUT_METRICS_FILE = r'C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\expresults\A-10-10-ELP_DRL_MO4_metrics.csv'


# ==========================================
# 1. 基础工具函数
# ==========================================
def extract_json_path_from_line(line, csv_filepath):
    """从 CSV 行中提取 JSON 路径（支持相对/绝对路径 + 常见回退目录）"""
    json_path = None
    if '.json' in line:
        for part in line.split(','):
            part = part.strip().strip('"').strip("'")
            if '.json' in part:
                json_path = part
                break

    if json_path and not os.path.isabs(json_path):
        csv_dir = os.path.dirname(os.path.abspath(csv_filepath))
        guess_path = os.path.normpath(os.path.join(csv_dir, json_path))
        if os.path.exists(guess_path):
            return guess_path

        filename = os.path.basename(json_path)
        fallback_path = os.path.normpath(os.path.join(csv_dir, 'pareto_archives', filename))
        if os.path.exists(fallback_path):
            return fallback_path

    return json_path


def load_json_results(filepath):
    """精准提取 mhc 和 cr；兼容常见 JSON 结构；返回 shape=(k,2)"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    # 常见包装层
    if isinstance(data, dict):
        for key in ['items', 'results', 'data', 'pareto_front', 'solutions']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    results = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'mhc' in item and 'cr' in item:
                    results.append([float(item['mhc']), float(item['cr'])])
            elif isinstance(item, list) and len(item) >= 2:
                # 兜底：取最后两列
                results.append([float(item[-2]), float(item[-1])])

    return np.asarray(results, dtype=float)


def nd_filter(points: np.ndarray) -> np.ndarray:
    """非支配过滤（最小化两目标）：返回 points 的第一前沿"""
    if points is None or points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points 必须是 shape=(n,2)，但得到 {points.shape}")

    fronts = NonDominatedSorting().do(points, only_non_dominated_front=True)
    # pymoo: only_non_dominated_front=True 时直接返回 indices
    return points[fronts]


def calculate_spread(results_norm: np.ndarray, ref_set_norm: np.ndarray) -> float:
    """Spread (Δ) 分布度（归一化空间）"""
    if results_norm is None or len(results_norm) < 2:
        return 0.0

    ideal_point = np.min(ref_set_norm, axis=0)
    nadir_point = np.max(ref_set_norm, axis=0)

    d_ext = (
        np.min(np.linalg.norm(results_norm - ideal_point, axis=1))
        + np.min(np.linalg.norm(results_norm - nadir_point, axis=1))
    )

    sorted_idx = np.argsort(results_norm[:, 0])
    sorted_results = results_norm[sorted_idx]
    distances = np.linalg.norm(sorted_results[1:] - sorted_results[:-1], axis=1)

    d_mean = np.mean(distances) if len(distances) else 0.0
    numerator = d_ext + np.sum(np.abs(distances - d_mean))
    denominator = d_ext + len(distances) * d_mean
    return float(numerator / denominator) if denominator != 0 else 0.0


def calculate_coverage(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """
    Coverage C(A, B): 集合 B 中被集合 A 弱支配的比例。
    论文表格中的 C 值为 C(Ref, Algorithm)：
      - set_a = Ref
      - set_b = Algorithm 的非支配集
    越小越好。
    """
    if set_b is None or len(set_b) == 0:
        return 0.0

    dominated_count = 0
    for b in set_b:
        # 弱支配：a_i <= b_i (所有目标都是最小化)
        if np.any(np.all(set_a <= b, axis=1)):
            dominated_count += 1
    return dominated_count / len(set_b)


def calculate_epsilon_multiplicative(ref_set: np.ndarray, approx_set: np.ndarray) -> float:
    """
    Multiplicative Epsilon 指标（对齐论文 Eq.(8) 的定义）：
      ε(Ref, S) = inf{ ε : ∀ s∈S, ∃ r∈Ref 使得 r <= ε * s }（逐目标）
    最小化问题下：对每个 s，所需 ε = min_r max_i (r_i / s_i)；最后取 max_s。

    注意：这里在【归一化空间】计算；为避免除零，对 s_i 进行一个极小值截断。
    """
    if ref_set is None or len(ref_set) == 0 or approx_set is None or len(approx_set) == 0:
        return 0.0

    eps_floor = 1e-12
    S = np.maximum(approx_set, eps_floor)

    # 对每个 s：计算所有 r 的 ratio，然后取 min_r 的 max_ratio
    # ratio(r,s) = max_i (r_i / s_i)
    per_s = []
    for s in S:
        ratios = np.max(ref_set / s, axis=1)  # shape=(|Ref|,)
        per_s.append(np.min(ratios))
    return float(np.max(per_s))


# ==========================================
# 2. 核心执行逻辑（严格对齐论文“参考集 Ref + 归一化 + 指标”口径）
# ==========================================
def run_paper_exact_evaluation():
    print(">>> 步骤 1: 扫描所有数据（每行一个算法输出 JSON）...")
    run_tasks = []

    with open(CSV_REGISTRY_PATH, 'r', encoding='utf-8-sig') as f:
        for row_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            j_path = extract_json_path_from_line(line, CSV_REGISTRY_PATH)
            if j_path and os.path.exists(j_path):
                run_tasks.append((row_idx, j_path))

    if not run_tasks:
        raise ValueError("CSV 中未解析到任何存在的 JSON 路径。请检查 CSV_REGISTRY_PATH。")

    # 论文口径：Ref = Update( S_alg1 ∪ S_alg2 ∪ ... )，其中每个 S_alg 是该算法的非支配集
    # 在你的 CSV 里，每一行就是一个“算法/一次运行”的集合；我们先对每行做 ND 过滤，再求全局 Ref。
    all_sets_raw = []
    filtered_tasks = []
    for row_idx, j_path in run_tasks:
        try:
            arr = load_json_results(j_path)
            arr_nd = nd_filter(arr)  # 防御：确保每行自身是 ND
            if arr_nd.size > 0:
                all_sets_raw.append(arr_nd)
                filtered_tasks.append((row_idx, j_path, arr_nd))
        except Exception:
            # 跳过坏行
            continue

    if not all_sets_raw:
        raise ValueError("未能加载到任何有效数据！请检查 CSV 和 JSON。")

    union_all = np.vstack(all_sets_raw)

    # 全局参考集 Ref：对 union_all 再做一次 ND 过滤
    reference_set_raw = nd_filter(union_all)

    # 归一化边界：使用 Ref 的 ideal/nadir（更贴近“以参考集为基准”的实验口径）
    ideal_point = np.min(reference_set_raw, axis=0)
    nadir_point = np.max(reference_set_raw, axis=0)
    denom = nadir_point - ideal_point
    denom[denom == 0] = 1e-12

    print(f"    Ref 大小: {len(reference_set_raw)}")
    print(f"    Ideal (from Ref): {ideal_point}")
    print(f"    Nadir (from Ref): {nadir_point}")

    def normalize(data: np.ndarray) -> np.ndarray:
        """归一化到 [0,1]（基于 Ref 的 ideal/nadir）"""
        return (data - ideal_point) / denom

    reference_set_norm = normalize(reference_set_raw)

    # 指标初始化（在归一化空间内）
    ref_point_hv = np.array([1.1, 1.1], dtype=float)
    ind_hv = HV(ref_point=ref_point_hv)
    ind_gd = GD(reference_set_norm)
    ind_igd = IGD(reference_set_norm)
    ind_igd_plus = IGDPlus(reference_set_norm)

    print("\n>>> 步骤 2: 逐行评估并写出指标（与论文表格字段对齐）...")
    file_exists = os.path.exists(OUTPUT_METRICS_FILE)

    with open(OUTPUT_METRICS_FILE, 'a', newline='', encoding='utf-8-sig') as out_f:
        writer = csv.writer(out_f)
        if not file_exists:
            writer.writerow(['Row_Index', 'Size', 'C (vs Ref)', 'HV', 'Epsilon', 'GD', 'IGD', 'IGD+', 'Spread', 'JSON_File'])

        for row_idx, j_path, my_results_raw in filtered_tasks:
            try:
                # 论文对比通常使用该算法的 ND 集（这里已经 nd_filter 过）
                my_results_norm = normalize(my_results_raw)

                sz = len(my_results_raw)

                c_val = calculate_coverage(reference_set_norm, my_results_norm)
                eps_val = calculate_epsilon_multiplicative(reference_set_norm, my_results_norm)
                hv_val = float(ind_hv(my_results_norm))
                gd_val = float(ind_gd(my_results_norm))
                igd_val = float(ind_igd(my_results_norm))
                igdp_val = float(ind_igd_plus(my_results_norm))
                spread_val = calculate_spread(my_results_norm, reference_set_norm)

                writer.writerow([
                    row_idx, sz,
                    f"{c_val:.4f}", f"{hv_val:.4f}", f"{eps_val:.4f}",
                    f"{gd_val:.4f}", f"{igd_val:.4f}", f"{igdp_val:.4f}",
                    f"{spread_val:.4f}", j_path
                ])
                out_f.flush()

                print(f"[完成] 行 {row_idx} | Size:{sz} | C:{c_val:.2f} | HV:{hv_val:.4f} | eps:{eps_val:.4f} | IGD:{igd_val:.4f} | Δ:{spread_val:.4f}")

            except Exception as e:
                print(f"[错误] 第 {row_idx} 行处理失败: {e}")

    print(f"\n>>> 任务结束！指标已生成至: {OUTPUT_METRICS_FILE}")


if __name__ == '__main__':
    run_paper_exact_evaluation()
