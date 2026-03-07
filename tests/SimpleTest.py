import gym
import src
import numpy as np
import src.utils.FBSUtil as FBSUtil  # 确保导入 FBSUtil
from src.utils.PopulationOptimizer import PopulationOptimizer
from src.utils.FBSModel import FBSModel


def run_simple_test():
    print("--- 开始 SimpleTest (种群优化器测试) ---")

    # 1. 初始化环境
    instance_name = "AB20-ar3"
    print(f"正在加载实例: {instance_name}")

    try:
        env = gym.make("FbsEnv-v0", instance=instance_name)
        env.reset()
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    # 输出原始物流量矩阵
    print("\n" + "=" * 30)
    print("原始物流量矩阵 (Flow Matrix Sample):")
    print("=" * 30)
    if hasattr(env, 'F'):
        F_matrix = np.array(env.F)
        print(f"Matrix Shape: {F_matrix.shape}")
        # 仅打印前5行示例，以免刷屏
        print(F_matrix[:5, :5])
        print("... (展示部分)")
    else:
        print("未在环境中找到物流量矩阵 (env.F)")
    print("=" * 30 + "\n")

    # 2. 初始化优化器
    pop_size = 100
    max_generations = 1000

    print(f"初始化优化器 (Pop Size: {pop_size}, Max Gens: {max_generations})...")
    optimizer = PopulationOptimizer(
        env=env,
        pop_size=pop_size,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=max_generations,
        k_coefficient=0.5
    )

    # 3. 执行优化
    print("开始优化...")
    best_model = optimizer.optimize()

    if best_model:
        print("\n" + "=" * 30)
        print("优化完成! 最优结果如下:")
        print("=" * 30)
        print(f"最优适应度 (Fitness/Manhattan Cost): {optimizer.best_fitness:.2f}")
        print(f"最优排序 (Permutation): {best_model.permutation}")
        print(f"最优区带 (Bay): {best_model.bay}")

        # --- 计算并输出坐标 ---
        print("\n" + "-" * 30)
        print("1. 设施坐标详情 (中心点 x, y):")
        print("-" * 30)

        coords_map = {}

        try:
            fac_x, fac_y, fac_b, fac_h = FBSUtil.getCoordinates_mao(
                best_model,
                env.areas,
                env.H
            )

            print(f"{'ID':<5} | {'X':<10} | {'Y':<10} | {'Width':<8} | {'Height':<8}")
            print("-" * 55)

            for i, fac_id in enumerate(best_model.permutation):
                fac_id = i + 1  # 假设 ID 从 1 开始

                cx = fac_x[i]
                cy = fac_y[i]
                cb = fac_b[i]
                ch = fac_h[i]

                # 存入 map 供后续计算成本使用
                coords_map[fac_id] = (cx, cy, cb, ch)

                print(f"{fac_id:<5} | {cx:<10.4f} | {cy:<10.4f} | {cb:<8.4f} | {ch:<8.4f}")

            num_facilities = len(env.areas)
            for fac_id in range(1, num_facilities + 1):
                if fac_id in coords_map:
                    cx, cy, cb, ch = coords_map[fac_id]
                    print(f"{fac_id:<5} | {cx:<10.4f} | {cy:<10.4f} | {cb:<8.4f} | {ch:<8.4f}")
                else:
                    print(f"{fac_id:<5} | N/A")

        except Exception as e:
            print(f"计算坐标时出错: {e}")

        # --- [核心修改] 同时计算曼哈顿和欧几里得距离 ---
        print("\n" + "-" * 30)
        print("2. 物流搬运成本核算 (Distance & Cost Verification):")
        print("   Manhattan (Man) = |x1-x2| + |y1-y2| (通常用于 Fitness)")
        print("   Euclidean (Euc) = sqrt((x1-x2)^2 + (y1-y2)^2)")
        print("-" * 30)

        if hasattr(env, 'F'):
            calc_mhc_man = 0.0
            calc_mhc_euc = 0.0

            # 表头：分别显示 Man 和 Euc 的距离与成本
            print(
                f"{'Link':<8} | {'Flow':<6} | {'Dist(Man)':<10} | {'Cost(Man)':<10} | {'Dist(Euc)':<10} | {'Cost(Euc)':<10}")
            print("-" * 75)

            n_facilities = len(env.areas)

            for i in range(n_facilities):
                for j in range(i + 1, n_facilities):
                    id_i = i + 1
                    id_j = j + 1

                    flow = F_matrix[i][j] + F_matrix[j][i]

                    if flow > 0:
                        if id_i in coords_map and id_j in coords_map:
                            xi, yi, _, _ = coords_map[id_i]
                            xj, yj, _, _ = coords_map[id_j]

                            # 1. 曼哈顿距离 (Manhattan)
                            dist_man = abs(xi - xj) + abs(yi - yj)
                            cost_man = flow * dist_man
                            calc_mhc_man += cost_man

                            # 2. 欧几里得距离 (Euclidean)
                            dist_euc = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                            cost_euc = flow * dist_euc
                            calc_mhc_euc += cost_euc

                            print(
                                f"{id_i:<2}-{id_j:<2}   | {flow:<6.0f} | {dist_man:<10.2f} | {cost_man:<10.2f} | {dist_euc:<10.2f} | {cost_euc:<10.2f}")

            print("-" * 75)
            print(f"统计结果:")
            print(f"  [Manhattan] 总成本: {calc_mhc_man:.2f}")
            print(f"  [Euclidean] 总成本: {calc_mhc_euc:.2f}")
            print(f"  [Algorithm] Fitness: {optimizer.best_fitness:.2f}")

            print("\n对比结论:")
            if abs(calc_mhc_man - optimizer.best_fitness) < 1e-3:
                print("  ✅ 算法 Fitness 与曼哈顿距离计算结果一致。")
            else:
                print(
                    f"  ⚠️ 算法 Fitness 与曼哈顿距离存在偏差 (Diff: {abs(calc_mhc_man - optimizer.best_fitness):.2f})")
                print("     (可能包含惩罚项或其他约束计算)")

            print(f"  ℹ️ 欧几里得成本通常比曼哈顿低: {calc_mhc_euc < calc_mhc_man}")

        else:
            print("无法计算: 环境中未找到流量矩阵 env.F")

        print("=" * 30 + "\n")

        # 可视化
        env.reset(fbs_model=best_model)
        env.render()

    else:
        print("优化失败，未找到最优解。")


if __name__ == "__main__":
    run_simple_test()