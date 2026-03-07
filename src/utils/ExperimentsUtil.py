import pandas as pd
import os
import re
import src.utils.config as config

def save_experiment_result(
        exp_instance, exp_algorithm, exp_iterations, exp_solution, exp_fitness,
        exp_start_time, exp_fast_time, exp_end_time,
        exp_is_valid_aspect_ratio=None,  # <--- 1. 添加新参数
        exp_remark="",
        exp_gbest_updates=None
):
    """
    保存实验结果
    """

    # 清理文件名中的非法字符
    def sanitize_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "", name)

    exp_instance_clean = sanitize_filename(exp_instance)
    exp_algorithm_clean = sanitize_filename(exp_algorithm)

    # 若实例名中带有 '_YYYY-MM-DD' 这样的日期后缀，则仅用前半部分作为文件名
    base_instance_clean = re.sub(r'_\d{4}-\d{2}-\d{2}$', "", exp_instance_clean)

    # 生成保存目录和文件名
    # save_dir = "/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA/files/expresults"
    save_dir=config.RESULT_PATH
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    filename = f"{base_instance_clean}-{exp_algorithm_clean}.csv"
    filepath = os.path.join(save_dir, filename)

    # 构建 DataFrame
    exp_date = exp_start_time.date() if exp_start_time is not None else None

    exp_result = pd.DataFrame({
        "实例": [exp_instance],
        "算法": [exp_algorithm],
        "日期": [exp_date],
        "迭代次数": [exp_iterations],
        "解": [exp_solution],
        "适应度值": [exp_fitness],
        "开始时间": [exp_start_time],
        "最快时间": [exp_fast_time],
        "结束时间": [exp_end_time],
        "运行时间（秒）": [(exp_end_time - exp_start_time).total_seconds()],
        "最快最佳结果时间（秒）": [(exp_fast_time - exp_start_time).total_seconds()],
        "宽高比是否满足": [exp_is_valid_aspect_ratio],  # <--- 2. 添加新列
        "gbest更新次数": [exp_gbest_updates],
        "备注": [exp_remark],
    })

    # 保存文件（追加模式）
    try:
        exp_result.to_csv(
            filepath,
            index=False,
            mode="a",
            header=not os.path.exists(filepath),
            encoding='utf_8_sig'
        )
    except Exception as e:
        print(f"保存失败！错误信息: {e}")
        return None

    return exp_result