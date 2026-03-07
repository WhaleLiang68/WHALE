# -*- coding: utf-8 -*-
"""
工业厂房布局优化算法配置文件
"""
import numpy as np
from pathlib import Path
import os

from sympy.utilities.codegen import Result

# ========================
# 基础路径配置
# ========================
script_dir = os.path.dirname(os.path.abspath(__file__))
print("[os] 脚本所在目录:", script_dir)
# 向上回退两级目录，移除 ua-flp-LSA\tests
base_dir = os.path.dirname(os.path.dirname(script_dir)) # 项目根目录
print("[os] 根目录:", base_dir)

# ========================
# 问题实例路径
# ========================
file_relative_path = "data\maoyan_cont_instances.pkl"
FILE_PATH = os.path.join(base_dir, file_relative_path)
print("实例文件所在目录:", FILE_PATH)

# ========================
# 结果保存路径
# ========================
result_relative_path = "files\expresults"
RESULT_PATH = os.path.join(base_dir, result_relative_path)
print("结果文件所在目录:", RESULT_PATH)
# ========================
# QLearning结果保存路径
# ========================
QLearning_result_relative_path = "files\QLearningResult"
QLearning_RESULT_PATH = os.path.join(base_dir, QLearning_result_relative_path)
print("QLearning结果文件所在目录:", QLearning_RESULT_PATH)
# ========================
# 问题实例参数
# ========================
FACILITY_CONFIG = {

}

# ========================
# 算法参数
# ========================
ALGORITHM = {

}

# ========================
# 实验验证配置
# ========================
VALIDATION = {

}

def validate_config():
    """配置参数校验"""
    assert FACILITY_CONFIG["n"] == len(FACILITY_CONFIG["area"]), \
        "设施数量与面积数组长度不一致"
    assert 1.0 <= FACILITY_CONFIG["beta"] <= 5.0, \
        "长宽比限制应在1-5之间"
    assert 0 <= ALGORITHM["dynamic_programming"]["balance_weight"] <= 1, \
        "权重参数应在0-1之间"

if __name__ == "__main__":
    validate_config()
    print("配置校验通过!")