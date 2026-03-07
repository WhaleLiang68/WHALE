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