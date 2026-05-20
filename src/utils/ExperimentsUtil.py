import os
import re
import csv
import datetime

import pandas as pd

import src.utils.config as config

TELEMETRY_PATH = os.path.join(config.base_dir, "files", "telemetry")


def _normalize_for_csv(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalize_for_csv(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_for_csv(item) for key, item in value.items()}
    return value


def save_experiment_result(
    exp_instance,
    exp_algorithm,
    exp_iterations,
    exp_solution,
    exp_fitness,
    exp_start_time,
    exp_fast_time,
    exp_end_time,
    exp_is_valid_aspect_ratio=None,
    exp_remark="",
    exp_gbest_updates=None,
    exp_extra_fields=None,
):
    def sanitize_filename(name):
        return re.sub(r'[\/*?:"<>|]', "", name)

    exp_instance_clean = sanitize_filename(exp_instance)
    exp_algorithm_clean = sanitize_filename(exp_algorithm)
    base_instance_clean = re.sub(r'_\d{4}-\d{2}-\d{2}$', "", exp_instance_clean)

    save_dir = config.RESULT_PATH
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{base_instance_clean}-{exp_algorithm_clean}.csv"
    filepath = os.path.join(save_dir, filename)

    exp_date = exp_start_time.date() if exp_start_time is not None else None
    exp_solution = _normalize_for_csv(exp_solution)

    exp_result_payload = {
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
        "宽高比是否满足": [exp_is_valid_aspect_ratio],
        "gbest更新次数": [exp_gbest_updates],
        "备注": [exp_remark],
    }

    if exp_extra_fields:
        for key, value in exp_extra_fields.items():
            exp_result_payload[key] = [_normalize_for_csv(value)]

    exp_result = pd.DataFrame(exp_result_payload)

    try:
        exp_result.to_csv(
            filepath,
            index=False,
            mode="a",
            header=not os.path.exists(filepath),
            encoding="utf_8_sig",
        )
    except Exception as exc:
        print(f"?????????: {exc}")
        return None

    return exp_result


def save_action_telemetry_rows(rows, filename="action_telemetry_summary.csv"):
    if not rows:
        return None

    os.makedirs(TELEMETRY_PATH, exist_ok=True)
    filepath = os.path.join(TELEMETRY_PATH, filename)

    try:
        telemetry_df = pd.DataFrame([_normalize_for_csv(row) for row in rows])
        expected_columns = [str(column) for column in telemetry_df.columns]
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf_8_sig", newline="") as file_obj:
                reader = csv.reader(file_obj)
                existing_header = next(reader, [])
            if existing_header != expected_columns:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filepath = os.path.join(
                    TELEMETRY_PATH,
                    f"{os.path.splitext(filename)[0]}_schema_backup_{timestamp}.csv",
                )
                os.replace(filepath, backup_filepath)
        telemetry_df.to_csv(
            filepath,
            index=False,
            mode="a",
            header=not os.path.exists(filepath),
            encoding="utf_8_sig",
        )
    except Exception as exc:
        print(f"动作Telemetry保存失败: {exc}")
        return None

    return telemetry_df
