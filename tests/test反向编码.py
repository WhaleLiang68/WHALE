import numpy as np


# =============================================================================
# 依赖的辅助函数 (Mocked / Simulated)
# (我根据您的代码库逻辑，模拟了这些函数以便此脚本能独立运行)
# =============================================================================

def permutationToArray(permutation, bay):
    """
    模拟: 将 (perm, bay) 结构转换为区带列表 (bay_list)。
    [此函数是 encode_genes_from_solution 的依赖项]
    """
    print(f"  [Debug] 模拟调用 permutationToArray...")
    bay_list = []
    current_zone = []
    for i, facility_label in enumerate(permutation):
        current_zone.append(facility_label)
        if bay[i] == 1:
            bay_list.append(current_zone)
            current_zone = []

    # 确保即使最后一个 bay[i] 不是 1 也能正确处理
    if current_zone:
        bay_list.append(current_zone)

    return bay_list


class ZGeneCoding:
    """
    模拟: ZGeneCoding 类，仅包含测试所需的 decode_genes 方法。
    """

    @staticmethod
    def decode_genes(genes, permutation):
        """
        模拟: ZGeneCoding.decode_genes 的核心逻辑。
        它必须通过排序基因值来重构 bay_list。
        [此函数用于验证 encode_genes_from_solution 的可逆性]
        """
        print(f"  [Debug] 模拟调用 ZGeneCoding.decode_genes...")
        n = len(permutation)
        # (设施标签, 基因值)
        gene_info = []

        # 1. 根据 permutation 顺序获取基因值
        for i, facility_label in enumerate(permutation):
            gene_info.append((facility_label, genes[i]))

        # 2. 【解码核心】根据基因值(genes[i])对设施进行排序
        gene_info.sort(key=lambda x: x[1])

        # 3. 重构 bay_list
        bay_list = []
        current_zone = []
        current_zone_id = -1

        for facility_label, gene_val in gene_info:
            zone_id = int(np.floor(gene_val))

            if zone_id != current_zone_id:
                if current_zone:
                    bay_list.append(current_zone)
                current_zone = [facility_label]  # 开始一个新区带
                current_zone_id = zone_id
            else:
                current_zone.append(facility_label)  # 添加到当前区带

        # 添加最后一个区带
        if current_zone:
            bay_list.append(current_zone)

        # 4. (可选) 重构 bay 数组
        bay = np.zeros(n, dtype=int)

        # 创建一个 {设施标签: 在新排序中的索引} 的映射
        flat_perm_decoded = [label for zone in bay_list for label in zone]
        perm_map = {label: i for i, label in enumerate(flat_perm_decoded)}

        idx = 0
        for zone in bay_list:
            idx += len(zone)
            if idx > 0 and idx <= n:
                # 找到该区带最后一个设施
                last_facility_in_zone = zone[-1]
                # 找到它在原始 perm 中的索引
                original_index = permutation.index(last_facility_in_zone)
                # bay[original_index] = 1 # <-- 这是一个更复杂的逻辑
                pass  # 简化的 mock 中,我们主要关心 bay_list

        return bay_list, bay


# =============================================================================
# 您要测试的函数 (来自您的提示)
# =============================================================================
def encode_genes_from_solution(permutation, bay):
    """
    【【【 新版本：增加了随机性 】】】
    执行 ZGeneCoding 的逆向操作。
    读取一个 (permutation, bay) 结构，并生成一个
    保证可以解码回该结构的、全新的、且包含随机性的 genes 数组。
    """
    # 1. 将 (perm, bay) 转换为二维列表
    bay_list = permutationToArray(permutation, bay)

    n = len(permutation)
    genes = np.zeros(n)

    # 映射表 {设施标签: 基因索引}
    # 这是关键：设施标签 -> 它在 *输入* perm 数组中的索引
    perm_map = {label: i for i, label in enumerate(permutation)}

    zone_id = 1
    for zone in bay_list:
        # 'zone' 是一个有序列表, 例如 [11, 15, 12]

        # --- 【【【 新增随机性 】】】 ---

        # 1. 获取当前区带的设施数量
        zone_size = len(zone)

        # 2. 生成 'zone_size' 个 (0.01, 0.99) 之间的随机小数
        random_keys = np.random.uniform(low=0.01, high=0.99, size=zone_size)

        # 3. 【关键】对小数进行排序，以确保它们是升序的
        sorted_random_keys = np.sort(random_keys)

        # --- 【【【 修改结束 】】】 ---

        # 4. 将这些升序的随机小数分配给区带中的设施
        for i, facility_label in enumerate(zone):
            # 5. 找到该设施在原始 permutation 数组中的索引
            gene_index = perm_map[facility_label]

            # 6. 创建新的 gene
            genes[gene_index] = float(zone_id) + sorted_random_keys[i]

        zone_id += 1  # 增加下一个区带的 ID

    return genes


# =============================================================================
# 测试执行
# =============================================================================
def run_test():
    print("--- 启动测试: encode_genes_from_solution ---")

    # 1. 定义输入数据
    # 假设 (perm, bay) 结构如下
    # perm = [3, 1, 4, 5, 2]
    # bay = [0, 1, 0, 0, 1]
    # 这对应 bay_list = [[3, 1], [4, 5, 2]]
    perm_input = [3, 1, 4, 5, 2]
    bay_input = [0, 1, 0, 0, 1]

    # 这个是“黄金标准”，是我们期望解码后能得到的结果
    bay_list_expected = [[3, 1], [4, 5, 2]]

    print(f"输入 Permutation: {perm_input}")
    print(f"输入 Bay: {bay_input}")
    print(f"预期 Bay List (解码后): {bay_list_expected}")
    print("-" * 30)

    # --- 测试 1: 随机性 ---
    print("[测试 1: 随机性]")
    print("  Running encode_genes_from_solution (Run 1)...")
    genes_1 = encode_genes_from_solution(perm_input, bay_input)
    print(f"  Output Genes (Run 1): {np.round(genes_1, 3)}")

    print("  Running encode_genes_from_solution (Run 2)...")
    genes_2 = encode_genes_from_solution(perm_input, bay_input)
    print(f"  Output Genes (Run 2): {np.round(genes_2, 3)}")

    if np.array_equal(genes_1, genes_2):
        print("  >>> [测试失败] 随机性: 两次运行的基因数组相同。")
    else:
        print("  >>> [测试通过] 随机性: 两次运行的基因数组不同。")
    print("-" * 30)

    # --- 测试 2: 逆向解码 (关键测试) ---
    print("[测试 2: 逆向解码 (可逆性)]")

    # 解码 Run 1
    print("  Decoding Run 1...")
    bay_list_decoded_1, _ = ZGeneCoding.decode_genes(genes_1, perm_input)
    print(f"  Decoded Bay List (Run 1): {bay_list_decoded_1}")

    # 解码 Run 2
    print("  Decoding Run 2...")
    bay_list_decoded_2, _ = ZGeneCoding.decode_genes(genes_2, perm_input)
    print(f"  Decoded Bay List (Run 2): {bay_list_decoded_2}")

    # 验证
    # 注意：Python 中 [1, 2] == [1, 2] 为 True
    is_match_1 = (bay_list_decoded_1 == bay_list_expected)
    is_match_2 = (bay_list_decoded_2 == bay_list_expected)

    if is_match_1 and is_match_2:
        print("  >>> [测试通过] 逆向解码: 两个随机基因组都成功解码回了原始结构。")
    else:
        print("  >>> [测试失败] 逆向解码: 解码结果与预期不符。")
        if not is_match_1: print(f"     Run 1 失败!")
        if not is_match_2: print(f"     Run 2 失败!")

    print("\n--- 测试完毕 ---")


if __name__ == "__main__":
    run_test()