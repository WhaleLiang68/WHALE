import gym
import src  # 确保 gym 环境已注册
import numpy as np
import copy
from loguru import logger  # GA_ELP4.py 依赖了 logger


# 确保 gym 环境 'FbsEnv-v0' 已经注册
# (通常在 src.__init__.py 或类似文件中完成)

def run_action_5_via_step():
    """
    测试封装在 env.step(5) 中的种群优化动作。
    """
    INSTANCE_NAME = "AB20-ar3"  #

    logger.info(f"--- 正在测试 动作五 (env.step(4)) ---")
    logger.info(f"初始化环境: FbsEnv-v0, 实例: {INSTANCE_NAME}")

    try:
        # 1. 初始化环境 (与 GA_ELP4.py 中相同)
        env = gym.make("FbsEnv-v0", instance=INSTANCE_NAME)

        # 2. 重置环境并获取初始状态
        initial_state = env.reset()

        # 假设 env 在 reset 后具有 'fitness' 属性
        # (基于 GA_ELP4.py 中 _calculate_energy 对 solution.fitness 的引用)
        initial_fitness = env.fitness
        initial_permutation = copy.deepcopy(env.fbs_model.permutation)

        logger.info(f"状态 (Action 5 之前):")
        logger.info(f"  初始适应度 (Fitness): {initial_fitness:.2f}")
        logger.info(f"  初始基因 (genes): {env.fbs_model.genes}")
        logger.info(f"  初始布局 (Permutation): {initial_permutation}")

    except Exception as e:
        logger.error(f"环境初始化失败: {e}")
        logger.error("请确保 'FbsEnv-v0' 已正确注册，并且实例文件 '{INSTANCE_NAME}' 存在。")
        return

    logger.info(f"--- 正在调用 env.step(4)... ---")

    try:
        # 3. 执行 动作五
        # 在 GA_ELP4.py 中，动作是通过 new_s.step(op) 调用的
        # 我们在此直接对 env 调用 step(5)
        state,reward,_,_,info=env.step(4)
        logger.info(f"--- 动作五 执行完毕 ---")

        # 4. 打印输出结果
        # 动作五 (PopulationOptimizer) 应该已经更新了 env 的内部状态
        final_fitness = env.fitness  #
        final_permutation = env.fbs_model.permutation
        final_bay = env.fbs_model.bay

        logger.info(f"状态 (Action 5 之后):")
        # logger.info(f"状态:{state}")
        logger.info(f"  最终适应度 (env.fitness): {final_fitness:.2f}")
        logger.info(f"  返回的奖励 (Reward): {reward}")
        logger.info(f"  返回的信息 (Info): {info}")
        logger.info(f"  最终基因 (genes): {env.fbs_model.genes}")
        logger.info(f"  最终布局 (Permutation): {final_permutation,final_bay}")

        logger.info(f"\n--- 总结 ---")
        fitness_change = final_fitness - initial_fitness
        logger.info(f"适应度变化: {initial_fitness:.2f} -> {final_fitness:.2f} (变化量: {fitness_change:.2f})")

        if final_fitness < initial_fitness:
            logger.success("结果: 动作五 找到了一个更优的解。")
        elif final_fitness == initial_fitness:
            logger.warning("结果: 动作五 未找到更优的解 (或返回了相同的解)。")
        else:
            logger.warning("结果: 警告：动作五 返回了一个更差的解。")

        # 可选：渲染最终布局
        logger.info("正在渲染最终布局...")
        env.render()


        state,reward,_,_,info=env.step(3)
        logger.info(f"--- 动作四 执行完毕 ---")

        # 4. 打印输出结果
        # 动作五 (PopulationOptimizer) 应该已经更新了 env 的内部状态
        final_fitness = env.fitness  #
        final_permutation = env.fbs_model.permutation
        final_bay = env.fbs_model.bay

        logger.info(f"状态 (Action 4 之后):")
        # logger.info(f"状态:{state}")
        logger.info(f"  最终适应度 (env.fitness): {final_fitness:.2f}")
        logger.info(f"  返回的奖励 (Reward): {reward}")
        logger.info(f"  返回的信息 (Info): {info}")
        logger.info(f"  最终基因 (genes): {env.fbs_model.genes}")
        logger.info(f"  最终布局 (Permutation): {final_permutation,final_bay}")

        logger.info(f"\n--- 总结 ---")
        fitness_change = final_fitness - initial_fitness
        logger.info(f"适应度变化: {initial_fitness:.2f} -> {final_fitness:.2f} (变化量: {fitness_change:.2f})")

        if final_fitness < initial_fitness:
            logger.success("结果: 动作四 找到了一个更优的解。")
        elif final_fitness == initial_fitness:
            logger.warning("结果: 动作四 未找到更优的解 (或返回了相同的解)。")
        else:
            logger.warning("结果: 警告：动作四 返回了一个更差的解。")

        # 可选：渲染最终布局
        logger.info("正在渲染最终布局...")
        env.render()

        state,reward,_,_,info=env.step(2)
        logger.info(f"--- 动作三 执行完毕 ---")

        # 4. 打印输出结果
        # 动作五 (PopulationOptimizer) 应该已经更新了 env 的内部状态
        final_fitness = env.fitness  #
        final_permutation = env.fbs_model.permutation
        final_bay = env.fbs_model.bay

        logger.info(f"状态 (Action 3 之后):")
        # logger.info(f"状态:{state}")
        logger.info(f"  最终适应度 (env.fitness): {final_fitness:.2f}")
        logger.info(f"  返回的奖励 (Reward): {reward}")
        logger.info(f"  返回的信息 (Info): {info}")
        logger.info(f"  最终基因 (genes): {env.fbs_model.genes}")
        logger.info(f"  最终布局 (Permutation): {final_permutation,final_bay}")

        logger.info(f"\n--- 总结 ---")
        fitness_change = final_fitness - initial_fitness
        logger.info(f"适应度变化: {initial_fitness:.2f} -> {final_fitness:.2f} (变化量: {fitness_change:.2f})")

        if final_fitness < initial_fitness:
            logger.success("结果: 动作三 找到了一个更优的解。")
        elif final_fitness == initial_fitness:
            logger.warning("结果: 动作三 未找到更优的解 (或返回了相同的解)。")
        else:
            logger.warning("结果: 警告：动作三 返回了一个更差的解。")

        # 可选：渲染最终布局
        logger.info("正在渲染最终布局...")
        env.render()

        state, reward, _, _, info = env.step(4)
        logger.info(f"--- 动作五 执行完毕 ---")

        # 4. 打印输出结果
        # 动作五 (PopulationOptimizer) 应该已经更新了 env 的内部状态
        final_fitness = env.fitness  #
        final_permutation = env.fbs_model.permutation
        final_bay = env.fbs_model.bay

        logger.info(f"状态 (Action 5 之后):")
        # logger.info(f"状态:{state}")
        logger.info(f"  最终适应度 (env.fitness): {final_fitness:.2f}")
        logger.info(f"  返回的奖励 (Reward): {reward}")
        logger.info(f"  返回的信息 (Info): {info}")
        logger.info(f"  最终基因 (genes): {env.fbs_model.genes}")
        logger.info(f"  最终布局 (Permutation): {final_permutation, final_bay}")

        logger.info(f"\n--- 总结 ---")
        fitness_change = final_fitness - initial_fitness
        logger.info(f"适应度变化: {initial_fitness:.2f} -> {final_fitness:.2f} (变化量: {fitness_change:.2f})")

        if final_fitness < initial_fitness:
            logger.success("结果: 动作五 找到了一个更优的解。")
        elif final_fitness == initial_fitness:
            logger.warning("结果: 动作五 未找到更优的解 (或返回了相同的解)。")
        else:
            logger.warning("结果: 警告：动作五 返回了一个更差的解。")

        # 可选：渲染最终布局
        logger.info("正在渲染最终布局...")
        env.render()



    except Exception as e:
        logger.error(f"执行 env.step(2) 时出错: {e}")


# --- 运行测试 ---
if __name__ == "__main__":
    # 确保 FbsEnv-v0, src, GA_ELP4.py 及其依赖项
    # 位于您的 PYTHONPATH 中，以便导入
    run_action_5_via_step()