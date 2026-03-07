import pickle
import numpy as np
import pandas as pd
import re

# 请修改为你的 pickle 路径
pickle_file_path = r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\data\maoyan_cont_instances.pkl"


def get_clean_areas(df):
    # 模拟你代码中的读取逻辑
    cols = df.filter(regex=re.compile("Area", re.IGNORECASE)).columns
    # 这里我们只取数值，不深究排序，先看总数和内容
    return df[cols].to_numpy().flatten()


try:
    with open(pickle_file_path, "rb") as file:
        (problems, FlowMatrices, sizes, LayoutWidths, LayoutLengths) = pickle.load(file)

    instance_ok = "AB20-ar3"
    instance_bad = "AB20-ar7"

    print(f"=== 对比 {instance_ok} (正常) vs {instance_bad} (异常) ===")

    # 1. 检查 H 和 W
    h_ok = LayoutWidths[instance_ok]
    w_ok = LayoutLengths[instance_ok]
    h_bad = LayoutWidths[instance_bad]
    w_bad = LayoutLengths[instance_bad]

    print(f"\n1. 布局尺寸 (H, W):")
    print(f"   {instance_ok}: H={h_ok}, W={w_ok}")
    print(f"   {instance_bad}: H={h_bad}, W={w_bad}")

    if h_ok != h_bad or w_ok != w_bad:
        print("   [!!!] 警告: H 或 W 不一致！这直接解释了为什么 MHC 不同！")
    else:
        print("   [OK] 尺寸一致。")

    # 2. 检查 Areas (面积)
    areas_ok = get_clean_areas(sizes[instance_ok])
    areas_bad = get_clean_areas(sizes[instance_bad])

    print(f"\n2. 设施面积 (前 10 个):")
    print(f"   {instance_ok}: {areas_ok[:10]}")
    print(f"   {instance_bad}: {areas_bad[:10]}")

    if not np.array_equal(areas_ok, areas_bad):
        print("   [!!!] 致命错误: 两个实例的面积数据不一致！")
        # 进一步检查是顺序乱了还是数值变了
        if np.isclose(np.sum(areas_ok), np.sum(areas_bad)):
            print("         -> 总面积相同，说明是【顺序乱了】。Pickle 数据里的列排序有问题。")
            print("         -> 请强制使用 AB20-ar3 的面积数据来计算 AB20-ar7！")
        else:
            print("         -> 总面积不同，说明是完全不同的数据集。")
    else:
        print("   [OK] 面积数据完全一致。")

    # 3. 检查 F 矩阵
    f_ok = FlowMatrices[instance_ok]
    f_bad = FlowMatrices[instance_bad]
    print(f"\n3. 流量矩阵 F Sum:")
    print(f"   {instance_ok}: {np.sum(f_ok):.2f}")
    print(f"   {instance_bad}: {np.sum(f_bad):.2f}")

    if not np.isclose(np.sum(f_ok), np.sum(f_bad)):
        print("   [!!!] 警告: F 矩阵不一致！")

except Exception as e:
    print(f"读取出错: {e}")