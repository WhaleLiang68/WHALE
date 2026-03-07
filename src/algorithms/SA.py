import numpy as np
import math
import copy
import datetime
import src
import gym
import numpy as np
import os
import logging
import src.utils.FBSUtil as FBSUtil
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from src.utils.FBSModel import FBSModel
import src.utils.ExperimentsUtil as ExperimentsUtil
from src.algorithms.RL.Q_Learning import QLearningAgent, evaluate_policy

class SA:
    def __init__(self, env, G=100, M=50, t_initial=10000.0, alpha=0.95, num_operators=5):
        """
        参数:
        env: 自定义环境对象，需实现 reset(), step(), fitness 属性
        G: 外层循环次数
        M: 内层循环次数
        t_initial: 初始温度
        alpha: 温度衰减系数
        num_operators: 操作符数量（如交换、插入等邻域操作）
        """
        self.env = env
        self.G = G
        self.M = M
        self.t = t_initial
        self.alpha = alpha
        self.num_operators = num_operators

        # # 初始化 Q-learning 代理
        # self.q_agent = QLearningAgent(
        #     s_dim=1,  # 假设单状态
        #     a_dim=num_operators
        #     # lr=0.1,
        #     # gamma=0.9,
        #     # exp_noise=0.1
        # )

        # 记录最优解
        self.gBest = None
        self.best_fitness = np.inf

    def run(self):
        """执行主优化流程"""
        start_time = datetime.datetime.now()
        fast_time = start_time


        # 初始化解
        self.env.reset()
        # s = copy.deepcopy(self.env.fbs_model)
        s = copy.deepcopy(self.env)
        current_fitness = self.env.fitness

        for g in range(self.G):
            # agent = QLearningAgent(s_dim=1, a_dim=5)
            for m in range(self.M):
                # 1. 选择 Q 值最大的操作符（伪代码第3行）
                # agent = QLearningAgent(s_dim=1, a_dim=5)
                # op = agent.sequential_evaluate_actions(env, s=0)
                # print("选择的动作：",op)
                # op = self.q_agent.select_action(s=0, deterministic=True)
                op = np.random.randint(0, 4)  # 随机选择动作
                # next_state, reward, done, info = env.step(op)  # 执行动作

                # 2. 生成新解（伪代码第4行）
                s_prime = self._apply_operator(s, op)
                # s_prime = self.env
                delta_f = s_prime.fitness - current_fitness

                # 3. Metropolis 准则接受判断（伪代码第6-7行）
                if delta_f < 0 or np.random.rand() < math.exp(-delta_f / self.t):
                    s = s_prime
                    current_fitness = s.fitness

                    # 4. 贪婪两阶段局部搜索（伪代码第8行）
                    # s = self.greedy_two_stage_search(s)

                    # 5. 更新 Q 表（伪代码第5行，Equation 10）
                    # agent.update_Q(s=0, a=op, f_ns=current_fitness)
                    # reward = -delta_f  # 假设奖励为适应度改进量
                    # self.q_agent.train(s=0, a=op, r=reward, s_next=0, dw=False)

                    # # 6. 更新全局最优解（伪代码第10行）
                    # if current_fitness < self.best_fitness:
                    #     self.gBest = copy.deepcopy(s.fbs_model)
                    #     self.best_fitness = current_fitness
                    #     fast_time = datetime.datetime.now()
                    self.gBest = copy.deepcopy(s.fbs_model)
                    self.best_fitness = current_fitness
                    fast_time = datetime.datetime.now()

            # 7. 降温（伪代码第11行）
            self.t *= self.alpha

        end_time = datetime.datetime.now()
        return (
            self.G * self.M,
            self.gBest,
            self.best_fitness,
            start_time,
            end_time,
            fast_time
        )

    # ------------------------ 关键子方法 ------------------------
    def _apply_operator(self, s, action):
        """应用操作符生成新解（需根据实际需求实现）"""
        new_s = copy.deepcopy(s)
        next_state, _, _, _ = new_s.step(action)
        return new_s

    def greedy_two_stage_search(self, s):
        """基于动作的两阶段贪婪搜索"""
        # 阶段1：使用 bay_shuffle 进行区带内局部优化
        improved = True
        max_attempts = 50  # 防止无限循环

        while improved and max_attempts > 0:
            improved = False

            # 保存当前状态
            original_state = copy.deepcopy(s)

            # 执行 bay_shuffle 动作（需确保 step2 支持 bay_id 参数）
            _, reward, _, info = s.step2(action=5)

            # 检查是否改进
            if info["current_fitness"] < original_state.fitness:
                improved = True
                break  # 找到改进后跳出循环
            else:
                # 恢复原状态
                s = original_state

            max_attempts -= 1

        # 阶段2：使用 facility_shuffle 进行全局优化
        improved = True
        max_attempts = 20

        while improved and max_attempts > 0:
            improved = False

            # 保存当前状态
            original_state = copy.deepcopy(s)

            # 执行 facility_shuffle 动作
            _, reward, _, info = s.step2(action=6)

            if info["current_fitness"] < original_state.fitness:
                improved = True
            else:
                s = original_state

            max_attempts -= 1

        return s

    def _calculate_fitness(self, solution):
        """计算解的适应度（需根据实际需求实现）"""
        # 示例：假设 solution 有 mhc 属性
        return solution.mhc

if __name__ == "__main__":
    # 实验参数
    exp_instance = "AB20-ar3"
    exp_algorithm = "SA"
    exp_remark = ""
    exp_number = 30
    is_exp = True
    # 获取当前日期并格式化为字符串（年-月-日）
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 算法参数
    max_iterations = 10000
    initial_temp = 10000.0
    alpha = 0.995
    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i+1}次实验")
            try:
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                sa_solver = SA(env, t_initial=initial_temp, alpha=alpha)
                iteration,best_solution, best_fitness,exp_start_time,exp_end_time,exp_fast_time = sa_solver.run()
                print("ok")
                print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
                # 保存结果
                ExperimentsUtil.save_experiment_result(
                    exp_instance=f"{exp_instance}_{current_date}",
                    # exp_instance=exp_instance,
                    exp_algorithm=exp_algorithm,
                    exp_iterations=iteration,
                    exp_solution=best_solution.array_2d,
                    exp_fitness=best_fitness,
                    exp_start_time=exp_start_time,
                    exp_fast_time=exp_fast_time,
                    exp_end_time=exp_end_time,
                    exp_remark=exp_remark
                )
                env.reset()
                print(env.state)
                print(f"Solution: {env.fbs_model.permutation, env.fbs_model.bay}, Fitness: {env.fitness}")
                # env.render()
            except Exception as e:
                logger.error(f"实验 {i + 1} 失败！错误信息: {e}")

    else:
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        # 注意：原代码这里有错误，缺少了实例化SA的步骤，已修正
        sa_solver = SA(env, t_initial=initial_temp, alpha=alpha)
        iteration, best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = sa_solver.run()

        print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution)

        # 保存单次实验结果，同样添加日期
        ExperimentsUtil.save_experiment_result(
            exp_instance=f"{exp_instance}_{current_date}",  # 文件名中加入日期
            exp_algorithm=exp_algorithm,
            exp_iterations=iteration,
            exp_solution=best_solution.array_2d,
            exp_fitness=best_fitness,
            exp_start_time=exp_start_time,
            exp_fast_time=exp_fast_time,
            exp_end_time=exp_end_time,
            exp_remark=exp_remark
        )

        env.render()