import numpy as np
import math
import copy
import datetime
import src
import gym
import os
import logging
import src.utils.FBSUtil as FBSUtil
from loguru import logger
from src.utils.FBSModel import FBSModel
import src.utils.ExperimentsUtil as ExperimentsUtil
from src.algorithms.RL.Q_Learning import QLearningAgent, evaluate_policy
from src.utils.PopulationOptimizer import PopulationOptimizer


class Q_LearningELP:
    def __init__(self, env, T, Q_matrix, G=100, t_max=50, k=0.1,num_operators=5,
                 pop_size=50, pop_generations=100):
        """
        ELP算法初始化（遵循图片中算法输入参数定义）
        参数:
        env: 自定义环境对象，需实现 reset(), step(), fitness 属性
        gbest: 初始合法解（对应算法输入s=gbest）
        T: 温度参数（算法输入T）
        Q_matrix: Q值矩阵（算法输入Q值矩阵）
        G: 最大迭代步数（外层循环上限，算法输入G）
        t_max: 内循环步数（内层循环上限，算法输入t_max）
        k: 系数（算法输入系数k）
        """
        self.env = env
        self.T = T  # 初始温度
        self.Q_matrix = copy.deepcopy(Q_matrix)  # Q值矩阵（用于后续扩展，当前保留结构）
        self.G = G  # 外层循环最大迭代次数
        self.t_max = t_max  # 内层循环最大步数
        self.k = k  # 能量计算系数k

        # 初始化种群优化器
        self.pop_optimizer = PopulationOptimizer(
            env=env,
            pop_size=pop_size,
            max_generations=pop_generations,
            k_coefficient=0.5
        )

        # 初始化 Q-learning 代理
        self.q_agent = QLearningAgent(
            s_dim=1,  # 假设单状态
            a_dim=num_operators
            # lr=0.1,
            # gamma=0.9,
            # exp_noise=0.1
        )

        # 初始化当前解和能量（E(s)用适应度表示，与原代码fitness逻辑一致）
        # 记录最优解
        self.gBest = None
        self.best_energy = np.inf
        self.current_energy = np.inf

    def _calculate_energy(self, solution):
        """计算解的能量E(s)（图片中算法核心指标，映射原代码fitness）
        注：ELP算法中能量越低表示解越优，与原代码fitness优化目标一致
        """
        # 若solution是环境对象，取其fitness；若为FBSModel，直接取对应适应度属性
        if hasattr(solution, 'fitness'):
            return solution.fitness
        elif hasattr(solution, 'mhc'):  # 兼容原代码中FBSModel的mhc属性
            return solution.mhc
        else:
            raise ValueError("解对象缺少能量计算所需的属性（fitness或mhc）")

    def _calculate_H(self, current_E, t):
        """计算H(E(s),t)函数（图片中能量修正项，此处实现基于迭代步的衰减函数）
        设计逻辑：随内循环步数t增加，H值递减，符合"迭代后期减少能量扰动"的直觉
        """
        # H = 初始扰动强度 * (1 - t/self.t_max)，确保t∈[1,t_max]时H非负
        initial_disturbance = 10.0  # 初始扰动强度，可根据问题调整
        return initial_disturbance * (1 - t / self.t_max)

    def _generate_new_solution(self, s, action):
        """基于当前解s产生新解s'（对应图片中"基于s产生新的解s'"步骤）
        沿用原SA算法的邻域操作逻辑，确保解的合法性
        """
        new_s = copy.deepcopy(s)
        # 随机选择邻域操作符（0-4，与原代码op选择范围一致）
        next_state, _, _, _ = new_s.step(action)
        # 执行操作生成新解（通过环境step方法保证解的合法性）
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

    def run(self):
        """执行ELP算法主流程（严格遵循图片中算法逻辑）"""
        start_time = datetime.datetime.now()
        fast_time = start_time  # 记录首次找到最优解的时间
        g = 1  # 外层循环计数器（算法中g从1开始）
        # 初始化解
        # 使用种群优化生成初始解 - 修改此处以适应新的返回值格式
        logger.info("开始种群优化生成初始解...")
        # 修正：只接收最优解对象，然后单独计算其适应度
        pop_best_solution = self.pop_optimizer.optimize()  # 现在只返回FBSModel对象

        # 单独计算种群优化结果的适应度
        self.env.reset(fbs_model=pop_best_solution)
        pop_best_fitness = self.env.fitness  # 从环境中获取适应度值
        logger.info(f"种群优化完成，初始最优适应度: {pop_best_fitness}")

        # 初始化当前状态和全局最优
        s = copy.deepcopy(self.env)
        self.current_energy = pop_best_fitness # 在个实验里能量就等于适应度
        self.gBest = copy.deepcopy(pop_best_solution)
        self.best_energy = pop_best_fitness
        fast_time = datetime.datetime.now()

        while g < self.G:  # 外层循环：g < G（图片中算法外层循环条件）
            t = 1  # 内层循环计数器（算法中t从1开始）
            agent = QLearningAgent(s_dim=1, a_dim=5)
            while t < self.t_max:  # 内层循环：t < t_max（图片中算法内层循环条件）
                op = agent.sequential_evaluate_actions(env, s=0)
                # 1. 基于当前解s产生新解s'（图片中步骤1）
                s_prime = self._generate_new_solution(s,op)
                # 2. 计算当前解和新解的原始能量（图片中E(s)和E(s')）
                E_s = self.current_energy
                E_s_prime = self._calculate_energy(s_prime)
                # 3. 计算修正后的能量（图片中E'(s) = E(s) + k*H(E(s),t)）
                H = self._calculate_H(E_s, t)
                E_prime_s = E_s + self.k * H
                E_prime_s_prime = E_s_prime + self.k * H
                # 4. 接受准则（图片中"若I()<E)rp() tcn"修正为标准Metropolis准则，基于能量差）
                delta_E_prime = E_prime_s_prime - E_prime_s  # 修正能量差
                r = np.random.rand()  # 产生[0,1)随机数r（图片中步骤）
                # 定义玻尔兹曼常数（单位：J/K）
                k_boltzmann = 1.380649e-23
                if delta_E_prime < 0 or r < math.exp(-delta_E_prime / (self.T * k_boltzmann)):
                    # 接受新解：更新当前解和当前能量
                    self.s = copy.deepcopy(s_prime)
                    self.current_energy = E_s_prime
                    # 5. 更新全局最优解gbest（图片中"更新gbest"步骤）
                    if self.current_energy < self.best_energy:
                        self.gbest = copy.deepcopy(self.s)
                        self.best_energy = self.current_energy
                        fast_time = datetime.datetime.now()  # 更新首次找到最优解的时间

                    s = self.greedy_two_stage_search(s)

                    # 5. 更新 Q 表（伪代码第5行，Equation 10）
                    agent.update_Q(s=0, a=op, f_ns=self.current_energy)

                # 6. 更新H(E(s),t)（图片中步骤，此处H随t动态计算，无需额外存储）
                # 7. 内循环步数+1（图片中"令t=t+1"）
                t += 1

            # 8. 外层循环步数+1（图片中"令g=g+1"）
            g += 1
            # 可选：温度衰减（保留原SA的温度逻辑，增强算法收敛性）
            self.T *= 0.995  # 衰减系数可调整，与原代码alpha保持一致

        end_time = datetime.datetime.now()
        # 返回结果格式与原SA代码兼容，便于后续实验分析
        total_iterations = self.G * self.t_max  # 总迭代次数（外层*内层）
        return (
            total_iterations,
            self.gbest,
            self.best_energy,
            start_time,
            end_time,
            fast_time
        )


if __name__ == "__main__":
    # 实验参数（与原SA代码保持一致，确保实验可对比）
    exp_instance = "AB20-ar7"
    exp_algorithm = "LELP_with_PopOpt"  # 算法名称改为ELP
    exp_remark = "加上了种群优化算法和qlearning"
    exp_number = 30
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # ELP算法参数（根据图片定义配置）
    G = 100  # 外层循环最大迭代步数G
    t_max = 50  # 内循环步数t_max
    T_initial = 10000.0  # 初始温度T
    k = 0.1  # 系数k（可根据实验调整）
    Q_matrix = np.zeros((1, 5))  # Q值矩阵（1个状态，5个操作符，初始为0）

    if is_exp:
        # 多轮实验（30次）
        for i in range(exp_number):
            logger.info(f"第{i + 1}次实验（ELP算法）")
            try:
                # 1. 初始化环境
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                env.reset()  # 重置环境获取初始合法解
                initial_gbest = copy.deepcopy(env)  # 初始gbest为环境初始解

                # 2. 实例化ELP算法（传入图片要求的所有输入参数）
                elp_solver = Q_LearningELP(
                    env=env,
                    T=T_initial,
                    Q_matrix=Q_matrix,
                    G=G,
                    t_max=t_max,
                    k=k
                )

                # 3. 运行ELP算法
                total_iter, best_sol, best_energy, start, end, fast = elp_solver.run()

                # 4. 输出结果
                logger.info(f"第{i + 1}次实验完成 | 最优能量: {best_energy}")
                print(f"Best Solution: {best_sol.fbs_model.array_2d}, Best Energy: {best_energy}")

                # 5. 保存实验结果（与原SA代码保存格式一致）
                ExperimentsUtil.save_experiment_result(
                    exp_instance=f"{exp_instance}_{current_date}",
                    exp_algorithm=exp_algorithm,
                    exp_iterations=total_iter,
                    exp_solution=best_sol.fbs_model.array_2d,
                    exp_fitness=best_energy,  # 能量对应原fitness，字段名保持兼容
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_remark=exp_remark
                )

                # 重置环境，准备下一轮实验
                env.reset()
                print(f"重置后环境状态: {env.state}")
                print(f"重置后解: {env.fbs_model.permutation, env.fbs_model.bay}, 能量: {env.fitness}")

            except Exception as e:
                logger.error(f"第{i + 1}次实验失败！错误信息: {str(e)}")
    else:
        # 单次实验
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        env.reset()
        initial_gbest = copy.deepcopy(env)

        # 实例化ELP算法
        elp_solver = Q_LearningELP(
            env=env,
            T=T_initial,
            Q_matrix=Q_matrix,
            G=G,
            t_max=t_max,
            k=k
        )

        # 运行并输出结果
        total_iter, best_sol, best_energy, start, end, fast = elp_solver.run()
        print(f"单次实验完成 | 总迭代次数: {total_iter}")
        print(f"Best Solution: {best_sol.fbs_model.array_2d}, Best Energy: {best_energy}")

        # 保存单次实验结果
        ExperimentsUtil.save_experiment_result(
            exp_instance=f"{exp_instance}_{current_date}",
            exp_algorithm=exp_algorithm,
            exp_iterations=total_iter,
            exp_solution=best_sol.fbs_model.array_2d,
            exp_fitness=best_energy,
            exp_start_time=start,
            exp_fast_time=fast,
            exp_end_time=end,
            exp_remark=exp_remark
        )

        # 渲染环境（原SA代码功能）
        env.reset(fbs_model=best_sol.fbs_model)
        env.render()