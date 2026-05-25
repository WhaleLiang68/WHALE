import argparse
import sys
from pathlib import Path

import gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.CR_MatrixStore import CRMatrixStore


def _resolve_facility_count(instance_name: str, explicit_count=None):
    if explicit_count is not None:
        count = int(explicit_count)
        if count <= 0:
            raise ValueError("--n 必须为正整数。")
        return count

    env = gym.make("FbsEnv-v0", instance=instance_name)
    try:
        try:
            env.reset()
        except TypeError:
            env.reset(seed=None)
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        facility_count = int(getattr(base_env, "n", len(getattr(base_env, "areas", [])) or 0))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    if facility_count <= 0:
        raise ValueError(f"无法自动推断实例 {instance_name} 的设施数，请显式传入 --n。")
    return facility_count


def main():
    parser = argparse.ArgumentParser(
        description="为指定实例生成并固化 CR（非物流关系强度）矩阵。默认若目标文件已存在则报错退出。"
    )
    parser.add_argument("--instance", required=True, help="实例名，例如 Du62")
    parser.add_argument("--n", type=int, default=None, help="设施数量；不传则尝试从环境自动推断")
    parser.add_argument("--data-dir", default=None, help="输出目录，默认 data/cr_matrices")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的 CR 矩阵文件")
    args = parser.parse_args()

    instance_name = str(args.instance).strip()
    if not instance_name:
        raise ValueError("--instance 不能为空。")

    facility_count = _resolve_facility_count(instance_name=instance_name, explicit_count=args.n)
    matrix = CRMatrixStore.generate_matrix(facility_count)
    output_path = CRMatrixStore.save_matrix(
        instance_name=instance_name,
        matrix=matrix,
        data_dir=args.data_dir,
        overwrite=bool(args.force),
    )
    print(f"实例: {instance_name}")
    print(f"设施数: {facility_count}")
    print(f"输出文件: {Path(output_path).resolve()}")
    print(f"生成规则: {CRMatrixStore.GENERATION_RULE}")
    print(f"等级映射: {CRMatrixStore.LEVEL_MAPPING}")


if __name__ == "__main__":
    main()
