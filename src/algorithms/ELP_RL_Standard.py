import numpy as np
np.bool8 = np.bool_
import math
import copy
import datetime
import src
import gym
import os
import logging
import matplotlib.pyplot as plt
import src.utils.FBSUtil as FBSUtil
from loguru import logger
from src.utils.FBSModel import FBSModel
import src.utils.ExperimentsUtil as ExperimentsUtil
from src.utils.PopulationOptimizer import PopulationOptimizer
# 注意：这里我们定义了自己的 StandardQLearningAgent，不再依赖原版 Q_Learning.py 的平均值逻辑
# from src.algorithms.RL.Q_Learning import QLearningAgent 

# =============================================================================
# 新增：标准 Q-Learning Agent (带学习率 alpha)
# 替代原有的基于平均值的 Agent，以符合强化学习的标准定义
# =============================================================================
class StandardQLearningAgent:
    def __init__(self, s_dim, a_dim, epsilon=0.5, alpha=0.1, gamma=0.9, initial_fitness=1.0):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子 (本场景单步反馈，影响较小，但在标准公式中存在)
        
        # 初始化 Q 表
        # 初始 Q 值设为 1/initial_fitness，给一个合理的基准值
        self.e = 1e-8
        # self.Q = np.full((s_dim, a_dim), 1.0 / (initial_fitness + self.e))
        # 使用 0 初始化，因为我们将使用相对奖励，让 Agent 自己去学正负
        self.Q = np.zeros((s_dim, a_dim))

    def select_action(self, s, deterministic=False):
        # 探索：以 epsilon 概率随机选择
        if not deterministic and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.a_dim)
            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return action
        # 利用：选择 Q 值最大的动作
        else:
            return np.argmax(self.Q[s, :])

    def update_Q(self, s, a, reward):
        """
        标准 Q-Learning 更新公式:
        Q(s, a) = Q(s, a) + alpha * (reward - Q(s, a))
        注：因为是单步任务(Bandit-like context)，这里暂不加 gamma * max(Q(s'))
        如果视作连续序列，可加上 + self.gamma * np.max(self.Q[s_next, :])
        """
        old_q = self.Q[s, a]
        new_q = old_q + self.alpha * (reward - old_q)
        self.Q[s, a] = new_q

# =============================================================================
# 主 ELP 类
# =============================================================================
class ELP:
    def __init__(self, env, gbest, T, G=100, t_max=50, k=0.1):
        """ Q_matrix,
        ELP算法初始化
        """
        self.env = env
        self.gbest = copy.deepcopy(gbest)
        self.T = T
        self.T_initial = T
        # Q_matrix 参数这里保留接口，实际使用 Agent 内部的 Q 表
        self.G = G
        self.t_max = t_max
        self.k = k

        # 初始化当前解和能量
        self.s = copy.deepcopy(gbest)
        self.current_energy = self._calculate_energy(self.s)
        self.best_energy = self.current_energy

        # 记录数据
        self.energy_history = []
        self.modified_energy_history = []
        self.prob_history = []

        # 直方图相关
        self.energy_histogram = {}
        self.bin_width = 5.0

        self.best_history = [self.best_energy]
        self.gbest_plot_path = None
        self.gbest_update_count = 0

        self.true_gbest = copy.deepcopy(gbest)
        
        # 状态感知相关参数
        self.no_improve_steps = 0  # 连续未改进次数

    def _calculate_energy(self, solution):
        if hasattr(solution, 'fitness'):
            return solution.fitness
        elif hasattr(solution, 'mhc'):
            return solution.mhc
        else:
            raise ValueError("解对象缺少能量计算所需的属性")

    def _get_bin_index(self, energy):
        return int(energy / self.bin_width)

    def _update_histogram(self, energy):
        idx = self._get_bin_index(energy)
        if idx not in self.energy_histogram:
            self.energy_histogram[idx] = 0
        self.energy_histogram[idx] += 1

    def _get_H_value(self, energy):
        idx = self._get_bin_index(energy)
        return self.energy_histogram.get(idx, 0)

    def _generate_new_solution(self, s, op):
        new_s = copy.deepcopy(s)
        new_s.step(op)
        return new_s

    def _check_aspect_ratio_constraint(self, solution):
        try:
            fac_x, fac_y, fac_b, fac_h, _, _, _, _, _ = FBSUtil.StatusUpdatingDevice(
                solution.fbs_model, self.env.areas, self.env.H, self.env.F, self.env.fac_limit_aspect
            )
            # 重新计算宽高比
            with np.errstate(divide='ignore', invalid='ignore'):
                fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
                fac_aspect_ratio[np.isnan(fac_aspect_ratio)] = 0
                
            if np.any(fac_aspect_ratio > self.env.fac_limit_aspect):
                return False
            return True
        except Exception as e:
            logger.error(f"宽高比约束检查出错: {e}")
            return False

    # def _greedy_search_step(self, max_steps=50):
    #     # 简化版贪婪搜索
    #     self.env.reset(fbs_model=self.gbest.fbs_model)
    #     current_best_fitness = self.best_energy
    #     improved = False
        
    #     for _ in range(max_steps):
    #         backup_model = copy.deepcopy(self.env.fbs_model)
    #         # 只尝试微调动作 (Swap)
    #         op = 0 # facility_swap
    #         _, _, _, _, info = self.env.step(op)
    #         new_fitness = info['current_fitness']

    #         if new_fitness < current_best_fitness and self._check_aspect_ratio_constraint(self.env):
    #             current_best_fitness = new_fitness
    #             self.gbest = copy.deepcopy(self.env)
    #             self.best_energy = current_best_fitness
    #             improved = True
    #         else:
    #             self.env.reset(fbs_model=backup_model)
        
    #     if improved:
    #         self.s = copy.deepcopy(self.gbest)
    #         self.current_energy = self.best_energy
    def _greedy_search_step(self, max_steps=100):
        """
        【增强版】贪婪搜索：
        不仅仅尝试一次，而是只要有改进就继续尝试，直到无法改进或达到步数限制。
        这对于在最后阶段“挤”出那 100-200 的 fitness 至关重要。
        """
        self.env.reset(fbs_model=self.gbest.fbs_model)
        current_best = self.best_energy
        
        no_imp_count = 0
        max_no_imp = 20 # 连续20次没改进就退出
        
        for _ in range(max_steps):
            backup_model = copy.deepcopy(self.env.fbs_model)
            
            # 随机选择微调动作：主要是 Swap(0) 和 Insert(5)，偶尔尝试 Repair(3)
            op = np.random.choice([0, 1, 5, 3], p=[0.4, 0.3, 0.2, 0.1])
            
            _, _, _, _, info = self.env.step(op)
            new_fitness = info['current_fitness']

            # 贪婪接受：只要变好就接受
            if new_fitness < current_best and self._check_aspect_ratio_constraint(self.env):
                current_best = new_fitness
                self.gbest = copy.deepcopy(self.env)
                self.best_energy = current_best
                no_imp_count = 0 # 重置计数器
                # logger.debug(f"Greedy Improved: {current_best}")
            else:
                self.env.reset(fbs_model=backup_model) # 回滚
                no_imp_count += 1
            
            if no_imp_count >= max_no_imp:
                break
        
        # 同步回 self.s
        if current_best < self.current_energy:
            self.s = copy.deepcopy(self.gbest)
            self.current_energy = self.best_energy

    # -------------------------------------------------------------------------
    # 新增：状态定义函数 (离散化)
    # -------------------------------------------------------------------------
    def _get_state_index(self, current_g, total_G):
        """
        状态定义 (组合状态):
        维度1: 停滞程度 (0:顺利, 1:轻微, 2:中度, 3:严重)
        维度2: 搜索阶段 (0:前期, 1:中期, 2:后期)
        总状态数 = 3 *4 = 12
        """
        # 1. 停滞状态
        """
        根据连续未改进步数 (no_improve_steps) 定义状态。
        状态 0: 进展顺利 (0-10步未改进)
        状态 1: 轻微停滞 (10-50步未改进)
        状态 2: 中度停滞 (50-150步未改进)
        状态 3: 严重停滞 (>150步未改进) - 需要 Shuffle
        """
        if self.no_improve_steps < 10:
            stag_idx = 0
        elif self.no_improve_steps < 50:
            stag_idx = 1
        elif self.no_improve_steps < 150:
            stag_idx = 2
        else:
            stag_idx = 3

            
        # 2. 阶段状态
        progress = current_g / total_G
        if progress < 0.3:
            phase_idx = 0 # 前期
        elif progress < 0.7:
            phase_idx = 1 # 中期
        else:
            phase_idx = 2 # 后期
            
        # 组合状态索引 (0-11)
        return phase_idx * 4 + stag_idx

    # 绘图函数保留 (篇幅原因省略具体实现，保持原样即可)
    def _plot_histogram(self):
        pass # 请保持原有的绘图代码
    def _plot_energy_curve(self):
        pass # 请保持原有的绘图代码
    def _plot_prob_curve(self):
        pass # 请保持原有的绘图代码

    def run(self):
        """执行ELP算法主流程"""
        start_time = datetime.datetime.now()
        fast_time = start_time
        g = 0 
        
        # 初始化 RL Agent
        # 状态空间 s_dim=4 (对应上面的4种停滞状态)
        # 动作空间 a_dim=8 (0-7号动作)
        state_dim = 12
        valid_actions = [0, 1, 2, 3, 5]
        action_dim = len(valid_actions)
        # action_dim = 8
        agent = StandardQLearningAgent(s_dim=state_dim, a_dim=action_dim, 
                                       epsilon=0.8, alpha=0.1, 
                                       initial_fitness=self.current_energy)

        # 拒绝惩罚系数 (beta)
        penalty_beta = 10.0

        while g < self.G:
            t = 0
            while t < self.t_max:
                
                # 1. 获取当前状态 (State Perception)
                current_state_idx = self._get_state_index(g, G)

                # 2. RL 选择动作
                op = agent.select_action(current_state_idx)
                real_action_op = valid_actions[op] # 映射
                
                # 3. 生成新解
                # s_prime = self._generate_new_solution(self.s, op)
                s_prime = self._generate_new_solution(self.s, real_action_op)

                # 4. 计算能量
                E_s = self.current_energy
                E_s_prime = self._calculate_energy(s_prime)
                
                # 5. ELP 修正能量计算
                penalty_factor = self.k
                H_s = self._get_H_value(E_s)
                H_s_prime = self._get_H_value(E_s_prime)
                
                E_prime_s = E_s + penalty_factor * H_s
                E_prime_s_prime = E_s_prime + penalty_factor * H_s_prime
                
                # 6. Metropolis 接受准则
                delta_E_prime = E_prime_s_prime - E_prime_s
                k_boltzmann = 1.0
                accept = False
                prob = 0.0

                if delta_E_prime < 0:
                    prob = 1.0
                    accept = True
                else:
                    exponent = -delta_E_prime / (self.T * k_boltzmann + 1e-10)
                    prob = math.exp(exponent)
                    if np.random.rand() < prob:
                        accept = True
                
                self.prob_history.append(prob)
                reward = 0.0
                scale_factor = 10000.0  # 放大系数
                # 7. 核心修正：基于接受/拒绝计算奖励并更新Q值
                if accept:
                    # --- 接受情况 ---
                    # self.no_improve_steps = 0 # 重置停滞计数

                    # 奖励计算: R = C / E (此处 C=1)
                    # 能量越低，E越小，1/E 越大，奖励越高
                    # reward = 1.0 / (E_s_prime + 1e-8)
                    # improvement = max(0, (E_s - E_s_prime))
                    # reward = (scale_factor / E_s_prime) + (improvement * 0.1)
                    improvement = E_s - E_s_prime

                    if improvement > 0:
                        self.no_improve_steps = 0  # 重置停滞计数
                        # reward = improvement * 1.0 # 正反馈
                        reward = (scale_factor / E_s_prime) + (improvement * 0.1)
                        # 【Jackpot 奖励】如果打破了历史最优
                        if E_s_prime < self.best_energy:
                            reward += 50.0 # 重赏！让它记住这个动作
                    else:
                        # 虽然接受了但变差了 (Metropolis特性) -> 轻微惩罚
                        reward = -1.0
                    
                    # 更新 Q 表
                    agent.update_Q(current_state_idx, op, reward)
                    
                    # 接受新解
                    self.s = s_prime
                    self.current_energy = E_s_prime
                    
                    # 更新全局最优
                    if self.current_energy < self.best_energy:
                        self.gbest = copy.deepcopy(self.s)
                        self.best_energy = self.current_energy
                        fast_time = datetime.datetime.now()
                        self._greedy_search_step(max_steps=200)
                        if self._check_aspect_ratio_constraint(self.gbest):
                            # self.best_energy = self.gbest.get_fitness()
                            self.true_gbest = copy.deepcopy(self.gbest)
                        self.best_history.append(self.best_energy)
                        self.gbest_update_count += 1
                else:
                    # --- 拒绝情况 ---[4
                    self.no_improve_steps += 1 # 增加停滞计数
                    
                    # 奖励计算: R = C / (E * beta)
                    # 施加惩罚，使奖励显著变小
                    # reward = 1.0 / ((E_s_prime * penalty_beta) + 1e-8)
                    reward = -2.0
                    
                    # 【关键修正】: 即使被拒绝，也要更新 Q 表，让 Agent 记住这个动作在当前状态下效果不好
                    agent.update_Q(current_state_idx, op, reward)

                # print(reward)
                # 8. 更新地形惩罚 H
                self._update_histogram(self.current_energy)
                H = self._get_H_value(self.current_energy)
                current_step_modified_energy = self.current_energy + penalty_factor * H
                self.modified_energy_history.append(current_step_modified_energy)
                self.energy_history.append(self.current_energy)
                
                t += 1

            g += 1
            # 温度衰减
            self.T *= 0.995
            # K值动态调整
            self.k = max(1.0 , self.k * (self.T/self.T_initial))
            # self.k = max(1.0 , self.k * (1 + 0.001 * g))

        end_time = datetime.datetime.now()
        
        is_valid = self._check_aspect_ratio_constraint(self.true_gbest)
        if not is_valid:
            logger.warning("最终解不满足宽高比约束")
        else:
            logger.info("最终解满足宽高比约束")

        return (
            self.G * self.t_max,
            is_valid,
            self.true_gbest,
            self._calculate_energy(self.true_gbest), # 返回 true_gbest 的能量
            start_time,
            end_time,
            fast_time
        )

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    exp_instance = "AB20-ar3"
    exp_algorithm = "ELP_RL_Standard"
    exp_remark = "WarmStart(GA)+EnhancedGreedy,修改了reward（全局最优奖励值）+repair GA"
    exp_number = 50
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 参数配置
    G = 2000
    t_max = 300
    T_initial = 10000.0
    k = 20
    # 注意：这里的 Q_matrix 仅作为占位符传入，实际由 Agent 内部管理
    # Q_matrix = np.zeros((1, 8)) 

    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i + 1}次实验（ELP-RL算法）")
            try:
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                env.reset()
                # initial_gbest = copy.deepcopy(env)
                logger.info("正在执行 GA 热启动...")
                # 实例化种群优化器
                pop_optimizer = PopulationOptimizer(
                    env=env,
                    pop_size=50,       # 种群规模
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    max_generations=200, # 跑50代足够产生一个不错的初始解 (~6000-7000)
                    k_coefficient=0.5
                )
                # 获取 GA 的最优解
                best_initial_model = pop_optimizer.optimize()
                logger.info(f"GA 热启动完成，初始 Fitness: {pop_optimizer.best_fitness:.2f}")
                
                # 用 GA 的解初始化环境
                env.reset(options={'fbs_model': best_initial_model})
                initial_gbest = copy.deepcopy(env)

                elp_solver = ELP(
                    env=env,
                    gbest=initial_gbest,
                    T=T_initial,
                    # Q_matrix=Q_matrix,
                    G=G,
                    t_max=t_max,
                    k=k
                )

                total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()

                logger.info(f"第{i + 1}次实验完成 | 最优能量: {best_energy}")
                
                # 简单的合法性检查
                if best_sol is None or not hasattr(best_sol, 'fbs_model'):
                    print(f"Error: Invalid best_sol")
                else:
                    print(f"Best Energy: {best_energy}")

                ExperimentsUtil.save_experiment_result(
                    exp_instance=f"{exp_instance}",
                    exp_algorithm=exp_algorithm,
                    exp_iterations=total_iter,
                    exp_solution=best_sol.fbs_model.array_2d,
                    exp_fitness=best_energy,
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_is_valid_aspect_ratio=is_valid,
                    exp_remark=exp_remark,
                    exp_gbest_updates=elp_solver.gbest_update_count
                )
                env.reset()

            except Exception as e:
                logger.error(f"第{i + 1}次实验失败！错误信息: {str(e)}")
                import traceback
                traceback.print_exc()
    else:
        # 单次运行测试
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        env.reset()
        # initial_gbest = copy.deepcopy(env)
        # logger.info("正在执行 GA 热启动...")
        # 实例化种群优化器
        pop_optimizer = PopulationOptimizer(
            env=env,
            pop_size=50,       # 种群规模
            crossover_rate=0.8,
            mutation_rate=0.1,
            max_generations=50, # 跑50代足够产生一个不错的初始解 (~6000-7000)
            k_coefficient=0.5
            )
        # 获取 GA 的最优解
        # best_initial_model = pop_optimizer.optimize()
        # logger.info(f"GA 热启动完成，初始 Fitness: {pop_optimizer.best_fitness:.2f}")
                
        # 用 GA 的解初始化环境
        # env.reset(options={'fbs_model': best_initial_model})
        initial_gbest = copy.deepcopy(env)

        elp_solver = ELP(
            env=env,
            gbest=initial_gbest,
            T=T_initial,
            # Q_matrix=Q_matrix,
            G=G,
            t_max=t_max,
            k=k
        )

        total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
        print(f"单次实验完成 | 最优能量: {best_energy}")