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
    """从 CSV 行中提取 JSON 绝对路径"""
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
        if os.path.exists(guess_path): return guess_path

        filename = os.path.basename(json_path)
        fallback_path = os.path.normpath(os.path.join(csv_dir, 'pareto_archives', filename))
        if os.path.exists(fallback_path): return fallback_path

    return json_path


def load_json_results(filepath):
    """【修复版】精准提取 mhc 和 cr，过滤无关干扰项"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ['items', 'results', 'data', 'pareto_front', 'solutions']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    results = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # 严格提取 mhc 和 cr
                if 'mhc' in item and 'cr' in item:
                    results.append([float(item['mhc']), float(item['cr'])])
            elif isinstance(item, list) and len(item) >= 2:
                results.append([float(item[-2]), float(item[-1])])

    arr = np.array(results)
    return arr


def calculate_spread(results_norm, ref_set_norm):
    """Spread (Δ) 分布度计算 (在归一化空间内)"""
    if len(results_norm) < 2: return 0.0
    ideal_point = np.min(ref_set_norm, axis=0)
    nadir_point = np.max(ref_set_norm, axis=0)

    d_ext = sum([np.min(np.linalg.norm(results_norm - ideal_point, axis=1)),
                 np.min(np.linalg.norm(results_norm - nadir_point, axis=1))])

    sorted_idx = np.argsort(results_norm[:, 0])
    sorted_results = results_norm[sorted_idx]
    distances = np.linalg.norm(sorted_results[1:] - sorted_results[:-1], axis=1)

    d_mean = np.mean(distances)
    numerator = d_ext + np.sum(np.abs(distances - d_mean))
    denominator = d_ext + len(distances) * d_mean
    return numerator / denominator if denominator != 0 else 0.0


def calculate_coverage(set_a, set_b):
    """
    Coverage C(A, B): 集合 B 中被集合 A 弱支配的比例。
    论文表格中的 C 值为 C(Ref, Algorithm)，即算法求出的解有多少被参考集支配。
    值越小越好 (0.0 代表你的解全是无敌的)。
    """
    if len(set_b) == 0: return 0.0
    dominated_count = 0
    for b in set_b:
        for a in set_a:
            if np.all(a <= b):
                dominated_count += 1
                break
    return dominated_count / len(set_b)


def calculate_epsilon(set_a, set_b):
    """Additive Epsilon (ε) (在归一化空间内)"""
    if len(set_a) == 0 or len(set_b) == 0: return 0.0
    eps_values = []
    for b in set_b:
        eps_values.append(np.min(np.max(set_a - b, axis=1)))
    return np.max(eps_values)


# ==========================================
# 2. 核心执行逻辑
# ==========================================
def run_paper_exact_evaluation():
    print(">>> 步骤 1: 扫描所有数据，构建全局参考集并获取归一化边界...")
    all_runs_data = []
    run_tasks = []

    with open(CSV_REGISTRY_PATH, 'r', encoding='utf-8-sig') as f:
        for row_idx, line in enumerate(f, start=1):
            if not line.strip(): continue
            j_path = extract_json_path_from_line(line, CSV_REGISTRY_PATH)
            if j_path and os.path.exists(j_path):
                run_tasks.append((row_idx, j_path))

    for _, j_path in run_tasks:
        try:
            arr = load_json_results(j_path)
            if arr.size > 0: all_runs_data.append(arr)
        except:
            continue

    if not all_runs_data:
        raise ValueError("未能加载到任何有效数据！请检查 CSV 和 JSON。")

    # 合并所有数据
    combined_all = np.vstack(all_runs_data)

    # 获取全局极值用于归一化 (jMetal 标准做法)
    ideal_point = np.min(combined_all, axis=0)
    nadir_point = np.max(combined_all, axis=0)
    denominator = nadir_point - ideal_point
    denominator[denominator == 0] = 1e-9  # 防止除零

    print(f"    全局极小值(Ideal): {ideal_point}")
    print(f"    全局极大值(Nadir): {nadir_point}")

    def normalize(data):
        """归一化函数：将数据缩放到 [0, 1]"""
        return (data - ideal_point) / denominator

    # 提取全局非支配解集 (参考集 Ref)，并对其进行归一化
    fronts = NonDominatedSorting().do(combined_all)
    reference_set_raw = combined_all[fronts[0]]
    reference_set_norm = normalize(reference_set_raw)
    print(f"    全局参考集(Ref)构建完成，包含 {len(reference_set_norm)} 个非支配解。")

    # 初始化基于归一化空间的指标计算器
    ref_point_hv = np.array([1.1, 1.1])  # 归一化后的标准 HV 参考点
    ind_hv = HV(ref_point=ref_point_hv)
    ind_gd = GD(reference_set_norm)
    ind_igd = IGD(reference_set_norm)
    ind_igd_plus = IGDPlus(reference_set_norm)

    print("\n>>> 步骤 2: 开始逐行精确评估 (使用归一化数据)...")
    file_exists = os.path.exists(OUTPUT_METRICS_FILE)

    with open(OUTPUT_METRICS_FILE, 'a', newline='', encoding='utf-8-sig') as out_f:
        writer = csv.writer(out_f)
        if not file_exists:
            writer.writerow(
                ['Row_Index', 'Size', 'C (vs Ref)', 'HV', 'Epsilon', 'GD', 'IGD', 'IGD+', 'Spread', 'JSON_File'])

        for row_idx, j_path in run_tasks:
            try:
                my_results_raw = load_json_results(j_path)
                if my_results_raw.size == 0: continue

                # 【核心】：将当前运行的数据归一化
                my_results_norm = normalize(my_results_raw)

                # 计算所有指标
                sz = len(my_results_raw)
                # C(Ref, MyResults): 你的解中有多少比例被参考集支配 (越小越好)
                c_val = calculate_coverage(reference_set_norm, my_results_norm)
                eps_val = calculate_epsilon(reference_set_norm, my_results_norm)
                hv_val = ind_hv(my_results_norm)
                gd_val = ind_gd(my_results_norm)
                igd_val = ind_igd(my_results_norm)
                igdp_val = ind_igd_plus(my_results_norm)
                spread_val = calculate_spread(my_results_norm, reference_set_norm)

                # 写入并打印 (保留 4 位小数，与论文更贴近)
                writer.writerow([row_idx, sz, f"{c_val:.4f}", f"{hv_val:.4f}", f"{eps_val:.4f}",
                                 f"{gd_val:.4f}", f"{igd_val:.4f}", f"{igdp_val:.4f}", f"{spread_val:.4f}", j_path])
                out_f.flush()

                print(
                    f"[完成] 行 {row_idx} | Size:{sz} | C:{c_val:.2f} | HV:{hv_val:.4f} | IGD:{igd_val:.4f} | Spread:{spread_val:.4f}")

            except Exception as e:
                print(f"[错误] 第 {row_idx} 行处理失败: {e}")

    print(f"\n>>> 任务结束！指标已严格按照论文标准生成至: {OUTPUT_METRICS_FILE}")


if __name__ == '__main__':
    run_paper_exact_evaluation()