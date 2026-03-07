import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import math  # <--- 重写所需

# --- 1. 配置 (请根据需要修改) ---

# 【【【 确保此 CSV 文件与脚本在同一目录 】】】
csv_file_path = r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\expresults\AB20-ar7-GA_ELP_QLearning-test.csv"

# 【【【 确保此 .pkl 文件与脚本在同一目录 】】】
# (这是您项目中 config.FILE_PATH 指向的文件)
pickle_file_path = r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\data\maoyan_cont_instances.pkl"

# 【【【 输入 CSV 中对应的实例名称 】】】
# 这必须与您加载 gym 环境时使用的名称一致 (例如: "AB20-ar3")
exp_instance = "AB20-ar7"


# --- 2. 重写 FBSModel.py ---
class FBSModel:
    def __init__(self, permutation=None, bay=None, genes=None):
        self.permutation = permutation if permutation is not None else []
        self.bay = bay if bay is not None else []
        self.genes = genes if genes is not None else []

    @property
    def array_2d(self):
        try:
            bay_copy = self.bay.copy()
            perm_copy = self.permutation.copy()
            bay_copy[-1] = 1
            array = []
            start = 0
            for i, val in enumerate(bay_copy):
                if val == 1:
                    array.append(perm_copy[start: i + 1])
                    start = i + 1
            return [np.array(sub_array) for sub_array in array]
        except Exception:
            return []


# --- 3. 重写 FBSUtil.py 中的必要函数 (无 GYM 依赖) ---

# --- 修正后的 getAreaData (替换原函数) ---
def getAreaData(df):
    # 1. 获取所有包含 "Area" 的列
    # 排除 "Aspect" 以防万一
    all_area_cols = [c for c in df.columns if "Area" in c and "Aspect" not in c]

    print(f"DEBUG: 原始 Area 列名 (前10个): {all_area_cols[:10]}")

    # 2. 【核心修复】分离“带数字的列”和“不带数字的列”
    numbered_cols = []
    other_cols = []

    for col in all_area_cols:
        match = re.search(r'(\d+)', col)
        if match:
            # 存储为元组 (数字值, 列名) 以便排序
            numbered_cols.append((int(match.group(1)), col))
        else:
            other_cols.append(col)

    # 3. 对带数字的列进行数值排序 (1, 2, 3... 10)
    numbered_cols.sort(key=lambda x: x[0])

    # 提取排序后的列名
    sorted_area_cols = [x[1] for x in numbered_cols]

    # 如果有其他列，看情况是否需要（通常 Area1-Area20 才是我们需要的）
    # 如果 sorted_area_cols 的数量正好等于设施数量(20)，那就只用这些
    # 这里我们优先使用排序后的编号列
    final_cols = sorted_area_cols

    print(f"DEBUG: 排序后的 Area 列名 (前10个): {final_cols[:10]}")

    # 检查数量是否合理 (AB20 应该有 20 个)
    if len(final_cols) < 20:
        print(f"警告: 识别到的编号 Area 列只有 {len(final_cols)} 个，可能遗漏！尝试加入非编号列...")
        final_cols.extend(other_cols)

    areas = df[final_cols].to_numpy().flatten()

    # --- 处理 Aspect ---
    aspect_cols = [c for c in df.columns if "Aspect" in c]
    try:
        # 同样尝试排序
        aspect_cols.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 999)
    except:
        pass

    aspects = df[aspect_cols].to_numpy().flatten()
    # 取第一个有效值作为限制
    aspect_limit = aspects[0] if len(aspects) > 0 else 99

    return areas, aspect_limit


def permutationToArray(permutation, bay):
    #
    bay_copy = np.array(bay)
    bay_copy[-1] = 1
    array = []
    start = 0
    for i, val in enumerate(bay_copy):
        if val == 1:
            array.append(permutation[start: i + 1])
            start = i + 1
    return array


def arrayToPermutation(array):
    #
    permutation = []
    bay = []
    for sub_array in array:
        permutation.extend(sub_array)
        bay.extend([0] * (len(sub_array) - 1) + [1])
    permutation = np.array(permutation)
    bay = np.array(bay)
    return permutation, bay


def getCoordinates_mao(fbs_model, area, H):
    #
    permutation = fbs_model.permutation
    bay = fbs_model.bay
    bays = permutationToArray(permutation, bay)

    n = len(permutation)
    lengths = np.zeros(n)  # 对应 fac_h (高度)
    widths = np.zeros(n)  # 对应 fac_b (宽度)
    fac_x = np.zeros(n)
    fac_y = np.zeros(n)
    x = 0
    start = 0
    for b in bays:
        b_np = np.array(b)
        indices = b_np - 1
        bay_areas = area[indices]

        bay_width = np.sum(bay_areas) / H
        widths[start: start + len(bay_areas)] = bay_width

        if bay_width == 0:
            bay_lengths = np.zeros(len(bay_areas))
        else:
            bay_lengths = bay_areas / bay_width

        lengths[start: start + len(bay_areas)] = bay_lengths

        fac_x[start: start + len(bay_areas)] = bay_width * 0.5 + x
        x += bay_width

        y = np.cumsum(bay_lengths) - bay_lengths * 0.5
        fac_y[start: start + len(bay_areas)] = y
        start += len(bay_areas)

    order = np.argsort(permutation)
    fac_x = fac_x[order]
    fac_y = fac_y[order]
    lengths = lengths[order]  # 高度
    widths = widths[order]  # 宽度
    return fac_x, fac_y, lengths, widths  # 返回 (x, y, 高度, 宽度)


def getManhattanDistances(x, y):
    #
    x = np.asarray(x)
    y = np.asarray(y)
    x_diff = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    y_diff = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
    return x_diff + y_diff


def getMHC(D, F, fbs_model):
    #
    return np.sum(D * F)


def getFitness(mhc, fac_b, fac_h, fac_limit_aspect=None, k=3):
    #
    fac_b = np.array(fac_b)
    fac_h = np.array(fac_h)
    MHC = mhc

    if fac_limit_aspect is None:
        non_feasible = (fac_b < 1) | (fac_h < 1)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
            aspect_ratio[np.isnan(aspect_ratio)] = 0
            aspect_ratio[np.isinf(aspect_ratio)] = 999
        non_feasible = (aspect_ratio < 1) | (aspect_ratio > fac_limit_aspect)

    non_feasible_counter = np.sum(non_feasible)
    fitness = MHC + (MHC - 4720.35) * (non_feasible_counter ** k)
    return fitness


def StatusUpdatingDevice(fbs_model, a, H, F, fac_limit_aspect_ratio):
    #

    # --- 【【【 BUG 修复于此 】】】 ---
    # getCoordinates_mao 返回 (x, y, 高度, 宽度)
    # 我们必须按此顺序分配
    fac_x, fac_y, fac_h, fac_b = getCoordinates_mao(fbs_model, a, H)  # <-- 已修正

    with np.errstate(divide='ignore', invalid='ignore'):
        fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        fac_aspect_ratio[np.isnan(fac_aspect_ratio)] = 0
        fac_aspect_ratio[np.isinf(fac_aspect_ratio)] = 999

    D = getManhattanDistances(fac_x, fac_y)
    TM = D * F
    mhc = getMHC(D, F, fbs_model)
    fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect_ratio)
    # 在 StatusUpdatingDevice 函数中添加打印
    print(f"DEBUG: F shape: {F.shape}, F sum: {np.sum(F)}")
    print(f"DEBUG: Areas sum: {np.sum(a)}, First 5 areas: {a[:5]}")
    print(f"DEBUG: Coordinates X sum: {np.sum(fac_x)}, Y sum: {np.sum(fac_y)}")
    print(f"DEBUG: Distance Matrix D sum: {np.sum(D)}")
    print(f"DEBUG: Calculated MHC: {mhc}")

    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, TM, mhc, fitness)


# --- 4. 重写 DataExtractor.py 中的 render 和 constructState 逻辑 ---

def constructStateColors(permutation, TM, n):
    # (部分)
    sources = np.sum(TM, axis=1)
    sinks = np.sum(TM, axis=0)
    permutation_arr = np.array(permutation)

    def safe_normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val: return np.ones_like(arr) * 255
        return ((arr - min_val) / (max_val - min_val)) * 255

    R_perm = safe_normalize(permutation_arr)
    G_perm = safe_normalize(sources[permutation_arr - 1])
    B_perm = safe_normalize(sinks[permutation_arr - 1])

    state_colors = {}
    for i, label in enumerate(permutation_arr):
        state_colors[label] = (R_perm[i] / 255, G_perm[i] / 255, B_perm[i] / 255)
    return state_colors


def render_layout(fbs_model, W, H, fac_x, fac_y, fac_b, fac_h,
                  fac_aspect_ratio, fac_limit_aspect, state_colors,
                  MHC, fitness):
    #

    print("正在渲染布局图...")
    fig, ax = plt.subplots()
    ax.set_title("Facility layout (Standalone Visualizer - Corrected)")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    plt.grid(False)
    plt.gca().set_aspect("equal", adjustable="box")

    for i, facility_label in enumerate(fbs_model.permutation):
        facility_idx = facility_label - 1

        # 现在 fac_b 是 宽度, fac_h 是 高度 (已修正)
        x_from = fac_x[facility_idx] - fac_b[facility_idx] / 2
        x_to = fac_x[facility_idx] + fac_b[facility_idx] / 2
        y_from = fac_y[facility_idx] - fac_h[facility_idx] / 2
        y_to = fac_y[facility_idx] + fac_h[facility_idx] / 2

        line_color = "red" if fac_aspect_ratio[facility_idx] > fac_limit_aspect else "green"

        R_val, G_val, B_val = state_colors.get(facility_label, (1.0, 1.0, 1.0))
        face_color = (R_val, G_val, B_val, 0.7)

        rect = patches.Rectangle(
            (x_from, y_from),
            width=x_to - x_from,  # <-- 正确的宽度
            height=y_to - y_from,  # <-- 正确的高度
            edgecolor=line_color,
            facecolor=face_color,
            linewidth=1
        )
        ax.add_patch(rect)

        ax.text(
            x_from + (x_to - x_from) / 2,
            y_from + (y_to - y_from) / 2,
            f"{int(facility_label)}",
            ha="center",
            va="center",
            color="white" if np.mean(face_color[:3]) < 0.5 else "black"
        )

    plt.figtext(0.5, 0.93, f"MHC: {MHC:.2f}", ha="center", fontsize=12)
    plt.figtext(0.5, 0.96, f"Fitness: {fitness:.2f}", ha="center", fontsize=12)

    plt.show()
    print("\n渲染完成。")


# --- 【【【 新增函数：步骤 4.5 】】】 ---
def check_facility_overlap(fbs_model, fac_x, fac_y, fac_b, fac_h):
    """
    检查所有设施对之间是否存在重叠。
    """
    print("\n--- 检查设施间重叠 ---")

    facilities_indices = np.arange(len(fbs_model.permutation))
    n = len(facilities_indices)
    overlap_found = False

    for i in range(n):
        for j in range(i + 1, n):
            idx_i = i
            idx_j = j

            f1_left = fac_x[idx_i] - fac_b[idx_i] / 2
            f1_right = fac_x[idx_i] + fac_b[idx_i] / 2
            f1_bottom = fac_y[idx_i] - fac_h[idx_i] / 2
            f1_top = fac_y[idx_i] + fac_h[idx_i] / 2

            f2_left = fac_x[idx_j] - fac_b[idx_j] / 2
            f2_right = fac_x[idx_j] + fac_b[idx_j] / 2
            f2_bottom = fac_y[idx_j] - fac_h[idx_j] / 2
            f2_top = fac_y[idx_j] + fac_h[idx_j] / 2

            x_overlap = (f1_left < f2_right) and (f1_right > f2_left)
            y_overlap = (f1_bottom < f2_top) and (f1_top > f2_bottom)

            if x_overlap and y_overlap:
                overlap_x_amount = min(f1_right, f2_right) - max(f1_left, f2_left)
                overlap_y_amount = min(f1_top, f2_top) - max(f1_bottom, f2_bottom)

                tolerance = 1e-6

                if overlap_x_amount > tolerance and overlap_y_amount > tolerance:
                    label_i = fbs_model.permutation[i]
                    label_j = fbs_model.permutation[j]

                    print(f"  [!!] 重叠发现: 设施 {label_i} 和 设施 {label_j} 重叠。")
                    overlap_found = True

    if not overlap_found:
        print("  [OK] 所有设施均未重叠。")

    print("--- 重叠检查结束 ---")
    return overlap_found


import numpy as np

# === 1. 硬编码数据源 (AB20 标准数据) ===
# 设施 1-20 的面积 (Sum = 600)
AREAS = np.array([27, 18, 27, 18, 18, 18, 9, 9, 9, 24, 23, 14, 27, 27, 108, 108, 27, 27, 27, 35])

# 你的 CSV 解 (Bay 结构)
# 注意：转换为 0-based 索引 (减1)
BAYS_0_BASED = [
    [5, 3, 1, 6, 7, 19, 17],  # [6, 4, 2, 7, 8, 20, 18] -> 减1
    [18, 4],  # [19, 5]
    [0, 2, 9, 8, 13, 14],  # [1, 3, 10, 9, 14, 15]
    [12, 11],  # [13, 12]
    [16],  # [17]
    [15],  # [16]
    [10]  # [11]
]

# 扁平化排列
PERMUTATION = [item for sublist in BAYS_0_BASED for item in sublist]


# AB20 对称流量矩阵 F (Sum = 754.16)
# 这里我们根据你提供的信息，假设 F 计算是正确的，不需要硬编码整个矩阵
# 我们只模拟几何计算部分

# === 2. 核心计算函数 ===
def calculate_coords_and_distance(h_val, use_euclidean=False):
    n = 20
    widths = np.zeros(n)
    lengths = np.zeros(n)
    fac_x = np.zeros(n)
    fac_y = np.zeros(n)

    current_x = 0
    start_idx = 0

    # 按照 Bay 结构计算坐标
    for bay in BAYS_0_BASED:
        bay_indices = np.array(bay)
        bay_areas = AREAS[bay_indices]

        # 核心逻辑：Bay 宽度 = Bay 总面积 / 厂房高度 H
        bay_width = np.sum(bay_areas) / h_val

        # 设施高度 = 设施面积 / Bay 宽度
        bay_lengths = bay_areas / bay_width

        # 记录尺寸
        widths[start_idx: start_idx + len(bay)] = bay_width
        lengths[start_idx: start_idx + len(bay)] = bay_lengths

        # 计算重心坐标
        # x = current_x + width / 2
        fac_x[start_idx: start_idx + len(bay)] = current_x + bay_width * 0.5

        # y = cumsum(lengths) - length / 2
        y_coords = np.cumsum(bay_lengths) - bay_lengths * 0.5
        fac_y[start_idx: start_idx + len(bay)] = y_coords

        current_x += bay_width
        start_idx += len(bay)

    # 恢复原始顺序 (虽然计算 D 不需要，但为了严谨)
    # 这一步其实对于 Distance 矩阵计算是不必要的，因为 D[i][j] 是对 ID 为 i 和 j 的设施算的
    # 但我们需要把 fac_x, fac_y 映射回 ID 0..19 的顺序

    # PERMUTATION[k] 是第 k 个放置的设施 ID
    # fac_x[k] 是第 k 个位置的坐标
    # 所以：coords[ID] = fac_x[k] where PERMUTATION[k] == ID

    final_x = np.zeros(n)
    final_y = np.zeros(n)
    final_x[PERMUTATION] = fac_x
    final_y[PERMUTATION] = fac_y

    # 计算距离矩阵 D
    if use_euclidean:
        x_diff = (final_x[:, np.newaxis] - final_x[np.newaxis, :]) ** 2
        y_diff = (final_y[:, np.newaxis] - final_y[np.newaxis, :]) ** 2
        D = np.sqrt(x_diff + y_diff)
    else:
        x_diff = np.abs(final_x[:, np.newaxis] - final_x[np.newaxis, :])
        y_diff = np.abs(final_y[:, np.newaxis] - final_y[np.newaxis, :])
        D = x_diff + y_diff

    return D


# === 3. 模拟求解 ===
# 由于我没有你的完整 F 矩阵，我无法算出确切的 MHC。
# 但你可以把这段代码复制进你的项目，利用你现有的 F 矩阵来运行。

def run_investigation(F_matrix):
    target = 4724.8197
    print(f"目标 MHC: {target}")

    best_h = -1
    min_diff = float('inf')
    best_mode = ""

    # 扫描 H (10 - 50)
    for h in np.arange(10, 50, 0.1):
        # 1. 曼哈顿模式
        D_man = calculate_coords_and_distance(h, use_euclidean=False)
        mhc_man = np.sum(D_man * F_matrix)

        if abs(mhc_man - target) < min_diff:
            min_diff = abs(mhc_man - target)
            best_h = h
            best_mode = "Manhattan"

        # 2. 欧氏模式
        D_euc = calculate_coords_and_distance(h, use_euclidean=True)
        mhc_euc = np.sum(D_euc * F_matrix)

        if abs(mhc_euc - target) < min_diff:
            min_diff = abs(mhc_euc - target)
            best_h = h
            best_mode = "Euclidean"

    print(f"\n>>> 调查结果:")
    print(f"最接近的参数: H = {best_h:.1f}")
    print(f"使用的距离公式: {best_mode}")
    print(f"最小误差: {min_diff:.4f}")

    # 特别检查正方形 (H=24.49)
    D_sq = calculate_coords_and_distance(24.4949, use_euclidean=False)
    mhc_sq = np.sum(D_sq * F_matrix)
    print(f"\n[参考] 正方形布局 (H=24.5, Manhattan) MHC: {mhc_sq:.4f}")


# === 这里的 main 只是为了让你把上面的函数拷走 ===
# 请在你的可视化代码中，加载 F 矩阵后，调用 run_investigation(F)
# --- 【【【 新增函数结束 】】】 ---

# --- 5. 主执行函数 ---

def main():
    print(f"正在使用 CSV 文件: {csv_file_path}")
    print(f"正在使用 Pickle 文件: {pickle_file_path}")
    print(f"使用实例名称: {exp_instance}")

    if not os.path.exists(csv_file_path):
        print(f"错误: 找不到 CSV 文件。路径: {csv_file_path}")
        return
    if not os.path.exists(pickle_file_path):
        print(f"错误: 找不到 .pkl 数据文件。路径: {pickle_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        df['适应度值'] = pd.to_numeric(df['适应度值'], errors='coerce')

        # --- 【修改逻辑】筛选满足宽高比约束的解 ---
        # 假设 CSV 中有一列名为 "宽高比是否满足"
        target_col = '宽高比是否满足'
        best_row = None

        if target_col in df.columns:
            # 兼容布尔值和字符串 (例如 "TRUE", "True", "true")
            # 筛选该列为 True 的行
            feasible_df = df[df[target_col].astype(str).str.upper() == 'TRUE']

            if not feasible_df.empty:
                print(f"\n找到 {len(feasible_df)} 个满足 '{target_col}' 为 TRUE 的解。")
                print("正在从中选择适应度值最优（最小）的解...")
                best_row = feasible_df.loc[feasible_df['适应度值'].idxmin()]
            else:
                print(f"\n警告: 数据中没有找到 '{target_col}' 为 TRUE 的解。")
                print("将回退到选择全局适应度最优的解（可能不满足约束）。")
                best_row = df.loc[df['适应度值'].idxmin()]
        else:
            print(f"\n提示: CSV 中未找到列 '{target_col}'。")
            print("将直接选择全局适应度最优的解。")
            best_row = df.loc[df['适应度值'].idxmin()]
        # --- 【修改结束】 ---

    except Exception as e:
        print(f"读取或处理 CSV 文件时出错: {e}")
        return

    print(f"\n已找到目标解 (行索引 {best_row.name}):")
    print(f"  适应度值: {best_row['适应度值']}")
    # 尝试打印约束满足情况，如果列存在
    if '宽高比是否满足' in best_row:
        print(f"  宽高比满足: {best_row['宽高比是否满足']}")

    bay_list_str = best_row['解']

    try:
        matches = re.findall(r'array\(\[([\d,\s.]*)\]', bay_list_str)
        bay_list = []
        for m in matches:
            m_clean = m.strip()
            if not m_clean:
                bay_list.append([])
            else:
                bay_list.append([int(float(x.strip())) for x in m_clean.split(',') if x.strip()])

        if not bay_list and bay_list_str != "[]":
            print("警告: Regex 未能解析 'array()'。尝试后备方案 ast.literal_eval...")
            import ast
            bay_list = ast.literal_eval(bay_list_str)

    except Exception as e:
        print(f"解析 '解' 字符串时出错: {e}")
        return

    permutation_np, bay_np = arrayToPermutation(bay_list)
    fbs_model = FBSModel(
        permutation=permutation_np.astype(int).tolist(),
        bay=bay_np.astype(int).tolist()
    )
    print("FBSModel 重建成功。")

    try:
        with open(pickle_file_path, "rb") as file:
            (problems, FlowMatrices, sizes, LayoutWidths, LayoutLengths) = pickle.load(file)

        F = FlowMatrices[exp_instance]
        # F = FlowMatrices[exp_instance]
        F = F + F.T
        run_investigation(F)
        areas, fac_limit_aspect = getAreaData(sizes[exp_instance])
        H = LayoutWidths[exp_instance]
        W = LayoutLengths[exp_instance]

        n = problems[exp_instance]
        print("环境参数加载成功。")
    except Exception as e:
        print(f"加载 .pkl 文件或提取实例数据时出错: {e}")
        return

    try:
        (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, TM, MHC, fitness) = StatusUpdatingDevice(
            fbs_model, areas, H, F, fac_limit_aspect
        )
        state_colors = constructStateColors(fbs_model.permutation, TM, n)
        print("坐标和属性计算完成。")
    except Exception as e:
        print(f"计算布局属性时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 【【【 新增代码块：步骤 5.6a 】】】 ---
    # 检查单个设施约束 (D_inf)
    print(f"\n--- 检查单个设施约束 (约束: 1 <= AR <= {fac_limit_aspect:.2f}) ---")

    # D_inf: 不可行的设施数量
    infeasible_count = 0

    # 遍历所有设施 (按 1..n 顺序检查，更清晰)
    all_labels = sorted(fbs_model.permutation)

    for facility_label in all_labels:
        facility_idx = facility_label - 1  # 转换为 0-based 索引

        # 获取该设施的计算宽高比
        ar = fac_aspect_ratio[facility_idx]

        # 检查约束
        is_feasible = (ar >= 1) and (ar <= fac_limit_aspect)

        if not is_feasible:
            infeasible_count += 1
            # 使用 :<2 来对齐个位数和十位数
            print(f"  [!!] 设施 {facility_label:<2}: 不可行. 宽高比 = {ar:.2f}")
        else:
            print(f"  [OK] 设施 {facility_label:<2}: 可行.   宽高比 = {ar:.2f}")

    print(f"--- 总结: {infeasible_count} 个设施不可行 (D_inf = {infeasible_count}) ---")  #
    # --- 【【【 新增代码块结束 】】】 ---


    overlap_found = check_facility_overlap(fbs_model, fac_x, fac_y, fac_b, fac_h)

    try:
        render_layout(fbs_model, W, H, fac_x, fac_y, fac_b, fac_h,
                      fac_aspect_ratio, fac_limit_aspect, state_colors,
                      MHC, fitness)
    except Exception as e:
        print(f"渲染 Matplotlib 图像时出错: {e}")

    # --- 距离公式验证脚本 ---
    print("\n" + "=" * 30)
    print(">>> 正在进行【距离公式】最终验证...")

    # 1. 计算坐标 (确认 H=20)
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(fbs_model, areas, H=20)

    # 2. 计算曼哈顿距离 MHC
    D_man = getManhattanDistances(fac_x, fac_y)
    MHC_man = getMHC(D_man, F, fbs_model)

    # 3. 计算欧氏距离 MHC
    # (临时定义欧氏距离函数，防止未导入)
    def calc_euclidean(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x_diff = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
        y_diff = (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
        return np.sqrt(x_diff + y_diff)

    D_euc = calc_euclidean(fac_x, fac_y)
    MHC_euc = getMHC(D_euc, F, fbs_model)

    target = 4724.8197

    print(f"1. 曼哈顿 MHC (当前代码): {MHC_man:.4f}")
    print(f"2. 欧氏   MHC (嫌疑对象): {MHC_euc:.4f}")
    print(f"3. CSV 记录的目标值:      {target:.4f}")

    if abs(MHC_euc - target) < 10:
        print(f"\n>>> 【破案了】！CSV 记录确实是基于【欧氏距离】计算的。")
        print(">>> 原因：在跑 ar7 实验时，代码可能临时切换到了欧氏距离。")
        print(">>> 建议：修改可视化代码，使用欧氏距离来复现该结果。")
    elif abs(MHC_man - target) < 10:
        print(f"\n>>> 曼哈顿距离匹配成功（但这与你之前的反馈矛盾）。")
    else:
        print(f"\n>>> 两者都不匹配。请提供 CSV 中【解】这一列的完整字符串，以便我人工复核。")
    print("=" * 30 + "\n")

    # --- 曼哈顿距离强制扫描脚本 ---
    print("\n" + "=" * 30)
    print(">>> 正在进行【曼哈顿距离】全域扫描 (H=10~60)...")

    target_mhc = 4724.8197
    best_h = -1
    min_diff = float('inf')
    min_mhc_calc = float('inf')

    # 扫描范围扩大，步长加密
    for test_h in np.arange(10, 60, 0.05):
        # 1. 计算坐标
        tx, ty, _, _ = getCoordinates_mao(fbs_model, areas, test_h)
        # 2. 强制使用曼哈顿
        tD = getManhattanDistances(tx, ty)
        tMHC = getMHC(tD, F, fbs_model)

        diff = abs(tMHC - target_mhc)

        # 记录全局最小值
        if tMHC < min_mhc_calc:
            min_mhc_calc = tMHC
            best_h_at_min = test_h

        if diff < min_diff:
            min_diff = diff
            best_h = test_h

        # 如果误差极小，打印出来
        if diff < 5.0:
            print(f"  [发现] H={test_h:.2f} -> MHC={tMHC:.2f} (误差: {diff:.2f})")

    print("-" * 20)
    print(f"扫描结果:")
    print(f"1. 曼哈顿距离能达到的【最小 MHC】是: {min_mhc_calc:.2f} (在 H={best_h_at_min:.2f} 时)")
    print(f"2. 目标 MHC 是: {target_mhc:.2f}")
    print(f"3. 最小差距: {min_diff:.2f}")

    if min_diff > 100:
        print("\n>>> 结论: 【数学上的不可能】")
        print("    在这个排列下，无论 H 是多少，曼哈顿距离都无法低至 4724。")
        print("    这证明 CSV 数据绝对不是用标准曼哈顿距离生成的。")
        print("    (请重新考虑欧氏距离的可能性，或者 F 矩阵被缩放的可能性)")
    else:
        print(f"\n>>> 结论: 找到了！请将 H 设置为 {best_h:.2f} 即可复现。")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
