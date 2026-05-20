import pickle
import numpy as np
import os
import csv
import pandas
# 获取脚本所在目录（方法1）
script_dir = os.path.dirname(os.path.abspath(__file__))
print("[os] 脚本所在目录:", script_dir)
# 向上回退两级目录，移除 ua-flp-LSA\tests
base_dir = os.path.dirname(os.path.dirname(script_dir))
relative_path = "ua-flp-LSA2\data\maoyan_cont_instances.pkl"
file_path = os.path.join(base_dir, relative_path)
print("[os] 文件所在目录:", file_path)

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")

if isinstance(data, dict):
    print("字典键:", data.keys())
    # 打印第一个实例的键（如果是嵌套字典）
    if data:
        sample_key = next(iter(data))
        print(f"实例 '{sample_key}' 的字段:", data[sample_key].keys())
elif isinstance(data, list):
    print("列表长度:", len(data))
    if data:
        print("第一个元素类型:", type(data[0]))
        print("第二个元素类型:", type(data[1]))
        print("第三个元素类型:", type(data[2]))
        print("第四个元素类型:", type(data[3]))
        print("第五个元素类型:", type(data[4]))
elif isinstance(data, np.ndarray):
    print("数组形状:", data.shape)
else:
    print("文件内容:", data)
print("文件内容:", data[0])
# 检查所有字典的键是否相同
keys_set = {tuple(d.keys()) for d in data}  # 提取所有字典的键名集合
print("所有字典的键是否一致:", "是" if len(keys_set) == 1 else "否")
all_keys = set()
for item in data:
    all_keys.update(item.keys())

print("所有字典的键:", all_keys)

from collections import defaultdict

key_counter = defaultdict(int)
for item in data:
    for key in item.keys():
        key_counter[key] += 1

print("键出现次数:", dict(key_counter))

def analyze_value_types(data):
    type_dict = {}
    for item in data:
        for key, value in item.items():
            if key not in type_dict:
                type_dict[key] = set()
            type_dict[key].add(type(value).__name__)
    return type_dict

print("键值类型分布:", analyze_value_types(data))
import pandas as pd

# 打印第一个字典中某个复杂键的值
sample_key = "SC35"

# sample_value = pd.DataFrame(data[1][sample_key])
sample_value = data[4][sample_key]
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", None)  # 自动调整列宽
pd.set_option("display.max_colwidth", None)  # 显示完整列内容（不截断）
# 处理单值情况：如果不是列表/元组，则转换为列表
if not isinstance(sample_value, (list, tuple)):
    sample_value = [sample_value]  # 确保内容可迭代

# 定义CSV文件名
csv_filename = "sample_value.csv"

# 写入CSV文件
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入单行（若需多行可改为循环）
    writer.writerow(sample_value)

print(f"数据已保存至 {csv_filename}")


def save_3d_to_csv(data, filename="output.csv"):
    # 将输入数据转换为NumPy数组（如果还不是）
    arr = np.array(data)

    # 检查形状是否符合预期
    print(f"原始数据形状: {arr.shape}")

    if arr.ndim == 3 and arr.shape[0] == 1:
        # 去除第一维（大小为1的维度）
        arr_2d = arr[0]  # 或 arr.squeeze(0)
        print(f"降维后形状: {arr_2d.shape}")

        # 创建DataFrame并保存
        df = pd.DataFrame(arr_2d)
        df.to_csv(filename, index=False)
        print(f"数据已保存至 {filename} (20行, 2列)")
    else:
        print("数据维度不符合预期，请检查数据结构")

# save_3d_to_csv(sample_value, "3d_to_2d.csv")

print(f"键 '{sample_key}' 的值类型:", type(sample_value))
pd.set_option('display.max_rows', None)
if isinstance(sample_value, pd.DataFrame):
    print("DataFrame 结构:")
    print(sample_value.head())
elif isinstance(sample_value, np.ndarray):
    print("ndarray 形状:", sample_value.shape)
    print("值内容:", sample_value)
else:
    print("值内容:", sample_value)


# import matplotlib.pyplot as plt
#
# matrix = data[1]['AB20-ar3']  # 假设取第一个字典的矩阵
# plt.imshow(matrix, cmap='viridis')
# plt.colorbar()
# plt.title("AB20-ar3 Matrix Visualization")
# plt.show()

target_key = sample_key # 可替换为其他键名

# 提取所有字典中该键的值（若存在）
extracted_data = []
for item in data:
    if target_key in item:
        extracted_data.append(item[target_key])
    else:
        extracted_data.append(None)  # 标记缺失值

print(f"共提取 {len(extracted_data)} 个值，其中非空值数量: {sum(x is not None for x in extracted_data)}")
from collections import defaultdict

type_counter = defaultdict(int)
for value in extracted_data:
    if value is not None:
        type_counter[type(value).__name__] += 1

print("数据类型分布:", dict(type_counter))
# 示例输出: {'DataFrame': 3, 'ndarray': 1, 'float': 1}

import matplotlib.pyplot as plt
import numpy as np

# 假设 matrix 是你的 20x20 矩阵（示例数据，替换为你的实际数据）
matrix = sample_value

# 设置打印选项，显示完整数组（不省略）
np.set_printoptions(threshold=np.inf, linewidth=200)

# 打印矩阵标题
# print(f"矩阵 '{sample_key}' 的形状: {matrix.shape}\n")

# 逐行打印矩阵内容（格式对齐）
# for row in matrix:
#     # 格式化每行数据（保留两位小数，对齐数值）
#     formatted_row = [f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x) for x in row]
#     print("  ".join(formatted_row))

