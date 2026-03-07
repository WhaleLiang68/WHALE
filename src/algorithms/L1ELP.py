import numpy as np
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
from src.algorithms.RL.Q_Learning import QLearningAgent


class ELP:
    def __init__(self, env, gbest, T, Q_matrix, G=100, t_max=50, k=0.1):
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
        self.gbest = copy.deepcopy(gbest)  # 初始全局最优解
        self.T = T  # 初始温度
        self.T_initial = T
        self.Q_matrix = copy.deepcopy(Q_matrix)  # Q值矩阵（用于后续扩展，当前保留结构）
        self.G = G  # 外层循环最大迭代次数
        self.t_max = t_max  # 内层循环最大步数
        self.k = k  # 能量计算系数k

        # 初始化当前解和能量（E(s)用适应度表示，与原代码fitness逻辑一致）
        self.s = copy.deepcopy(gbest)
        self.current_energy = self._calculate_energy(self.s)
        self.best_energy = self.current_energy  # 记录最优能量（对应原代码best_fitness）

        # ----------------------------------------
        self.energy_history = []
        self.modified_energy_history = [] # 记录修正后的能量历史
        self.prob_history = []

        # 初始化直方图 (使用字典)
        self.energy_histogram = {}
        # 设定直方图的“箱宽” (Bin Width)
        # 这个参数很重要：决定了多大范围内的能量被视为“同一个坑”
        # 对于你的问题(5000左右的量级)，设为 1.0 或 10.0 比较合适
        self.bin_width = 5.0
        # 记录 gbest 下降趋势
        self.best_history = [self.best_energy]
        self.gbest_plot_path = None
        self.gbest_update_count = 0

        self.action_5_prob = 0.05
        self.action_5_prob_min = 0.001  # 动作5的最小概率阈值
        self.action_5_decay = 0.9  # 动作5概率衰减系数（从0.05改为0.95，衰减更慢）
        self.true_gbest = copy.deepcopy(gbest)  # 初始化true_gbest

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

    def _get_bin_index(self, energy):
        """将连续的能量值转换为整数索引（分箱）"""
        return int(energy / self.bin_width)

    def _update_histogram(self, energy):
        """更新直方图：当前能量对应的计数 +1"""
        idx = self._get_bin_index(energy)
        if idx not in self.energy_histogram:
            self.energy_histogram[idx] = 0
        self.energy_histogram[idx] += 1

    def _get_H_value(self, energy):
        """获取当前能量对应的 H 值（访问次数）"""
        idx = self._get_bin_index(energy)
        return self.energy_histogram.get(idx, 0)

    def _generate_new_solution(self, s, op):
        """基于当前解s产生新解s'（对应图片中"基于s产生新的解s'"步骤）
        沿用原SA算法的邻域操作逻辑，确保解的合法性
        """
        new_s = copy.deepcopy(s)
        new_s.step(op)
        return new_s

    def _check_aspect_ratio_constraint(self, solution):
        """检查解的宽高比约束
        只检查宽高比约束，不检查其他约束条件
        """
        try:
            # 计算解的坐标和尺寸
            fac_x, fac_y, fac_b, fac_h = FBSUtil.getCoordinates_mao(
                solution.fbs_model, self.env.areas, self.env.H
            )
            
            # 检查宽高比约束
            fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
            if np.any(fac_aspect_ratio > self.env.fac_limit_aspect):
                logger.debug(f"宽高比检查失败: 最大宽高比 {np.max(fac_aspect_ratio):.2f} > 限制 {self.env.fac_limit_aspect}")
                return False
            
            logger.debug(f"宽高比检查通过: 最大宽高比 {np.max(fac_aspect_ratio):.2f} <= 限制 {self.env.fac_limit_aspect}")
            return True
            
        except Exception as e:
            logger.error(f"宽高比约束检查出错: {e}")
            return False

    def _greedy_search_step(self, max_steps=50):
        """
        利用 env.step 进行贪婪局部搜索。
        策略：执行动作 -> 变好保留 -> 变差回滚 (Reset)。
        """
        # 1. 确保环境当前状态与 gbest 同步
        #    注意：这里传入 gbest.fbs_model 确保从最优解开始
        self.env.reset(fbs_model=self.gbest.fbs_model)
        current_best_fitness = self.best_energy

        improved = False

        for _ in range(max_steps):
            # A. 备份当前状态 (用于回滚)
            #    必须深拷贝，因为 step 会原地修改 model
            backup_model = copy.deepcopy(self.env.fbs_model)

            # B. 执行动作 (只用 facility_swap，动作 0)
            #    这是最稳健的局部微调算子
            op = np.random.randint(0, 3)
            _, _, _, _, info = self.env.step(op)
            new_fitness = info['current_fitness']

            # C. 贪婪判断
            if new_fitness < current_best_fitness and self._check_aspect_ratio_constraint(self.env)==True:
                # --- 变好了：接受 ---
                current_best_fitness = new_fitness
                # 更新 ELP 类的全局最优记录
                self.gbest = copy.deepcopy(self.env)
                self.best_energy = current_best_fitness
                improved = True
                # logging.debug(f"贪婪搜索发现新高: {new_fitness}")
            else:
                # --- 变差了：回滚 (Revert) ---
                #    利用 reset 将环境恢复到备份的状态
                self.env.reset(fbs_model=backup_model)

        if improved:
            # 如果贪婪搜索找到了更好的解，
            # 顺便把当前 ELP 的搜索点 (self.s) 也拉过去，加速收敛
            self.s = copy.deepcopy(self.gbest)
            self.current_energy = self.best_energy

    def _plot_gbest_trend(self):
        """绘制并保存 gbest 下降趋势图"""
        if not self.best_history:
            return None
        plots_dir = os.path.join(os.getcwd(), "files", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(len(self.best_history)), self.best_history, marker="o")
        plt.title("ELP gbest trend")
        plt.xlabel("Improvement #")
        plt.ylabel("Best energy")
        plt.grid(alpha=0.3)
        filename = f"elp_gbest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.gbest_plot_path = os.path.join(plots_dir, filename)
        plt.tight_layout()
        plt.savefig(self.gbest_plot_path, dpi=150)
        plt.close()
        logger.info(f"gbest趋势图已保存: {self.gbest_plot_path}")
        return self.gbest_plot_path

    def _plot_histogram(self):
        """绘制并保存 H 值直方图 (能量分布频次)"""
        if not self.energy_histogram:
            logger.warning("直方图数据为空，无法绘制。")
            return None

        plots_dir = os.path.join(os.getcwd(), "files", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. 数据处理
        # 将字典按 key (bin index) 排序，保证 X 轴是连续的能量顺序
        sorted_indices = sorted(self.energy_histogram.keys())

        # 将 bin index 还原为近似能量值 (X轴)
        energies = [idx * self.bin_width for idx in sorted_indices]
        # 获取对应的频次 (Y轴)
        frequencies = [self.energy_histogram[idx] for idx in sorted_indices]

        # 2. 绘图
        plt.figure(figsize=(10, 6))
        # 使用 bar 图，宽度设为 bin_width 的 80% 以便视觉区分
        plt.bar(energies, frequencies, width=self.bin_width * 0.8, align='center', alpha=0.7, color='steelblue')

        plt.title(f"ELP Energy Landscape Histogram (H)\n(Bin Width: {self.bin_width})")
        plt.xlabel("Energy Level (Fitness)")
        plt.ylabel("Frequency (Visits)")
        plt.grid(axis='y', alpha=0.3)

        # 3. 保存文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"elp_histogram_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        logger.info(f"H直方图已保存: {filepath}")

        # H直方图数据保存
        # import csv
        # csv_filename = f"elp_histogram_data_{timestamp}.csv"
        # csv_path = os.path.join(plots_dir, csv_filename)
        # with open(csv_path, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Bin Index", "Approx Energy", "Frequency"])
        #     for idx in sorted_indices:
        #         writer.writerow([idx, idx * self.bin_width, self.energy_histogram[idx]])
        # logger.info(f"H直方图数据已保存: {csv_path}")

        return filepath

    def _plot_energy_curve(self):
        """绘制并保存完整的能量变化曲线 (Current Energy History)"""
        if not self.energy_history:
            logger.warning("能量历史数据为空，无法绘制曲线。")
            return

        plots_dir = os.path.join(os.getcwd(), "files", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))  # 设置宽一点，因为迭代次数很多

        # 绘制能量曲线
        # linewidth设置细一点，alpha设置透明度，以便看清密集的波动
        plt.plot(self.energy_history, linewidth=0.5, color='blue', alpha=0.6, label='Current Energy')

        # 可选：绘制一条红线表示最终的最优解，方便对比差距
        plt.axhline(y=self.best_energy, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best Found ({self.best_energy:.2f})')

        plt.title(f"ELP Optimization Trajectory\n(Total Iterations: {len(self.energy_history)})")
        plt.xlabel("Iteration Step")
        plt.ylabel("Energy (Fitness)")
        plt.legend()
        plt.grid(alpha=0.3)

        # 保存图片
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"elp_energy_curve_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        logger.info(f"能量变化曲线已保存: {filepath}")

    def _plot_modified_energy_curve(self):
        """绘制并保存 原始能量 vs 修正能量 对比曲线"""
        if not self.modified_energy_history:
            logger.warning("修正能量历史数据为空，无法绘制。")
            return

        plots_dir = os.path.join(os.getcwd(), "files", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # 1. 绘制原始能量 (Blue) - 如果你有记录的话
        if hasattr(self, 'energy_history') and self.energy_history:
            plt.plot(self.energy_history, linewidth=0.5, color='blue', alpha=0.5, label='Original Energy (Fitness)')

        # 2. 绘制修正能量 (Orange)
        plt.plot(self.modified_energy_history, linewidth=0.5, color='orange', alpha=0.8,
                 label='Modified Energy (E + k*H)')

        plt.title(f"ELP Energy Landscape Paving Process\n(Original vs Modified)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Energy Value")
        plt.legend()
        plt.grid(alpha=0.3)

        # 保存图片
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"elp_modified_energy_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        logger.info(f"修正能量变化图已保存: {filepath}")

    def _plot_prob_curve(self):
        """绘制并保存 接受概率 (Probability) 变化图"""
        if not self.prob_history:
            logger.warning("概率历史数据为空，无法绘制。")
            return

        plots_dir = os.path.join(os.getcwd(), "files", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # 使用散点图绘制，s=0.5 设置点的大小
        # 很多点会重叠在 y=0 和 y=1 附近
        plt.scatter(range(len(self.prob_history)), self.prob_history, s=0.5, color='green', alpha=0.5,
                    label='Acceptance Probability')

        plt.title(f"ELP Acceptance Probability History\n(Temperature Decay)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Probability P")
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0])  # 设置Y轴刻度
        plt.legend(markerscale=10)  # 图例里的点放大一点以便看见
        plt.grid(alpha=0.3)

        # 保存图片
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"elp_prob_curve_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        logger.info(f"概率变化图已保存: {filepath}")

    def run(self):
        """执行ELP算法主流程（严格遵循图片中算法逻辑）"""
        start_time = datetime.datetime.now()
        fast_time = start_time  # 记录首次找到最优解的时间
        g = 0  # 外层循环计数器（算法中g从1开始）

        agent = QLearningAgent(s_dim=1, a_dim=6, epsilon=0.8, initial_fitness=self.current_energy)

        while g < self.G:  # 外层循环：g < G（图片中算法外层循环条件）
            t = 0  # 内层循环计数器（算法中t从1开始）

            while t < self.t_max:  # 内层循环：t < t_max（图片中算法内层循环条件）
                # 1. 基于当前解s(X(1))产生新解s'(X(2))
                # if np.random.rand() < self.action_5_prob:
                #     # 1. 以 action_5_prob (例如 5%) 的概率选择 动作 5
                #     op = 7
                #     s_prime = self._generate_new_solution(self.s,op)
                #     self.action_5_prob = max(self.action_5_prob_min, self.action_5_prob * self.action_5_decay)
                # else:
                    # 2. 以 1 - action_5_prob (例如 95%) 的概率，
                    #    使用Q-Learning选择基本动作 (0, 1, 2, 3, 4，5，6)
                op = agent.select_action(self.s, 0)
                s_prime = self._generate_new_solution(self.s, op)

                # 2. 计算当前解和新解的原始能量
                E_s = self.current_energy # E(X(1))
                E_s_prime = self._calculate_energy(s_prime) # E(X(2))
                
                # # --- 修改代码---
                # # 为了使用浮点数作为字典键，我们对其进行舍入
                # # X(1) 对应的能量键
                # key_E_s = round(E_s, 2)
                # # X(2) 对应的能量键
                # key_E_s_prime = round(E_s_prime, 2)
                # # ----------------------------------------
                
                # 3. 计算修正后的能量
                penalty_factor = self.k
                H_s = self._get_H_value(E_s)
                H_s_prime = self._get_H_value(E_s_prime)
                # 3. 计算修正后的能量（图片中E'(s) = E(s) + k*H(E(s),t)）
                E_prime_s = E_s + penalty_factor * H_s
                E_prime_s_prime = E_s_prime + penalty_factor * H_s_prime
                # 4. 接受准则
                delta_E_prime = E_prime_s_prime - E_prime_s  # 修正能量差
                k_boltzmann = 1.0
                accept = False
                if delta_E_prime < 0:
                    prob = 1.0
                    accept = True
                else:
                    exponent = -delta_E_prime / (self.T * k_boltzmann + 1e-10)
                    prob = math.exp(exponent)
                    if np.random.rand() < prob:
                        accept = True
                self.prob_history.append(prob)
                
                # --- 修改代码 (根据 image_7d3076.png) ---
                # if ... : 对应 "当新产生的状态 X(2) 接受时"
                if accept:
                    # 接受新解：更新当前解和当前能量
                    # self.s = copy.deepcopy(s_prime)
                    # self.current_energy = E_s_prime
                    #
                    # # 更新 X(2) 的直方图函数值
                    # # 即 H(E(X(2))) = H(E(X(2))) + 1
                    # self.energy_histogram[key_E_s_prime] = self.energy_histogram.get(key_E_s_prime, 0) + 1
                    #
                    # # 5. 更新全局最优解gbest（图片中"更新gbest"步骤）
                    # if self.current_energy < self.best_energy:
                    #     self.gbest = copy.deepcopy(self.s)
                    #     self.best_energy = self.current_energy
                    #     fast_time = datetime.datetime.now()  # 更新首次找到最优解的时间
                    self.s = s_prime
                    self.current_energy = E_s_prime

                    agent.update_Q(0, op, E_s_prime)
                    # 5. 更新全局最优解gbest（图片中"更新gbest"步骤）
                    if self.current_energy < self.best_energy:
                        self.gbest = copy.deepcopy(self.s)
                        self.best_energy = self.current_energy
                        fast_time = datetime.datetime.now()  # 更新找到最优解的时间
                        self._greedy_search_step(max_steps=500)
                        if self._check_aspect_ratio_constraint(self.gbest) == True:
                            self.true_gbest = copy.deepcopy(self.gbest)
                        self.best_history.append(self.best_energy)
                        self.gbest_update_count += 1
                #
                # else:
                #     # 否则 (即拒绝新解)
                #     # 更新 X(1) 的直方图函数值
                #     # 即 H(E(X(1))) = H(E(X(1))) + 1
                #     self.energy_histogram[key_E_s] = self.energy_histogram.get(key_E_s, 0) + 1
                # ----------------------------------------
                # 6. 更新H(E(s),t)
                self._update_histogram(self.current_energy)
                H = self._get_H_value(self.current_energy)
                current_step_modified_energy = self.current_energy + penalty_factor * H
                self.modified_energy_history.append(current_step_modified_energy)

                # 7. 内循环步数+1（图片中"令t=t+1"）
                self.energy_history.append(self.current_energy)
                t += 1

            # 8. 外层循环步数+1（图片中"令g=g+1"）
            g += 1
            # 可选：温度衰减（保留原SA的温度逻辑，增强算法收敛性）
            self.T *= 0.995  # 衰减系数可调整，与原代码alpha保持一致
            self.k = max(1.0 , self.k * (self.T/self.T_initial))

        end_time = datetime.datetime.now()
        if self.prob_history:
            print(f"最后prob: {self.prob_history[-1]}")
        print(f"T: {self.T}")
        # 对最终结果进行合法性检查（仅检查宽高比约束）
        is_valid = self._check_aspect_ratio_constraint(self.gbest)
        if not is_valid:
            logger.warning("最终解不满足宽高比约束")
        else:
            logger.info("最终解满足宽高比约束")

        # 返回结果格式与原SA代码兼容，便于后续实验分析
        total_iterations = self.G * self.t_max  # 总迭代次数（外层*内层）
        # self._plot_gbest_trend()
        self._plot_histogram()
        self._plot_energy_curve()
        # self._plot_modified_energy_curve()
        self._plot_prob_curve()
        return (
            total_iterations,
            is_valid,  # <--- 添加此行
            self.gbest,
            self.best_energy,
            start_time,
            end_time,
            fast_time
        )


if __name__ == "__main__":
    # 实验参数（与原SA代码保持一致，确保实验可对比）
    exp_instance = "AB20-ar3"
    exp_algorithm = "ELP_QLearning1"  # 算法名称改为ELP
    exp_remark = "k=20*T/T_initial,bin_width=5,t_max = 300,G=2000,getfitness2，修改了bay_flip操作，g_best=5396.6，惩罚指数：1至5，操作1-6,无εQLearning"
    exp_number = 30
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # ELP算法参数（根据图片定义配置）
    G = 2000  # 外层循环最大迭代步数G
    t_max = 300  # 内循环步数t_max
    T_initial = 10000.0  # 初始温度T
    k = 20  # 系数k（可根据实验调整）
    Q_matrix = np.zeros((1, 7))  # Q值矩阵（1个状态，5个操作符，初始为0）

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
                elp_solver = ELP(
                    env=env,
                    gbest=initial_gbest,
                    T=T_initial,
                    Q_matrix=Q_matrix,
                    G=G,
                    t_max=t_max,
                    k=k
                )

                # 3. 运行ELP算法
                total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()

                # 4. 输出结果
                logger.info(f"第{i + 1}次实验完成 | 最优能量: {best_energy}")
                # 添加调试信息，检查 best_sol 的状态
                if best_sol is None:
                    logger.error(f"第{i + 1}次实验：best_sol 为 None")
                    print(f"Best Solution: None, Best Energy: {best_energy}")
                elif not hasattr(best_sol, 'fbs_model'):
                    logger.error(f"第{i + 1}次实验：best_sol 没有 fbs_model 属性")
                    print(f"Best Solution: [无fbs_model属性], Best Energy: {best_energy}")
                elif best_sol.fbs_model is None:
                    logger.error(f"第{i + 1}次实验：best_sol.fbs_model 为 None")
                    print(f"Best Solution: [fbs_model为None], Best Energy: {best_energy}")
                elif not hasattr(best_sol.fbs_model, 'permutation') or not hasattr(best_sol.fbs_model, 'bay'):
                    logger.error(f"第{i + 1}次实验：fbs_model 缺少 permutation 或 bay 属性")
                    print(f"Best Solution: [属性缺失], Best Energy: {best_energy}")
                elif len(best_sol.fbs_model.permutation) == 0 or len(best_sol.fbs_model.bay) == 0:
                    logger.error(f"第{i + 1}次实验：permutation 或 bay 为空")
                    logger.error(f"permutation: {best_sol.fbs_model.permutation}, bay: {best_sol.fbs_model.bay}")
                    print(f"Best Solution: [], Best Energy: {best_energy}")
                else:
                    print(f"Best Solution: {best_sol.fbs_model.array_2d}, Best Energy: {best_energy}")

                # 5. 保存实验结果（与原SA代码保存格式一致）
                ExperimentsUtil.save_experiment_result(
                    exp_instance=f"{exp_instance}",
                    exp_algorithm=exp_algorithm,
                    exp_iterations=total_iter,
                    exp_solution=best_sol.fbs_model.array_2d,
                    exp_fitness=best_energy,  # 能量对应原fitness，字段名保持兼容
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_is_valid_aspect_ratio=is_valid, # <--- 添加此行 (获取 is_valid)
                    exp_remark=exp_remark,
                    exp_gbest_updates=elp_solver.gbest_update_count
                )

                # 重置环境，准备下一轮实验
                env.reset()
                # print(f"重置后环境状态: {env.state}")
                print(f"重置后解: {env.fbs_model.permutation, env.fbs_model.bay}, 能量: {env.fitness}")

            except Exception as e:
                logger.error(f"第{i + 1}次实验失败！错误信息: {str(e)}")
    else:
        # 单次实验
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        env.reset()
        initial_gbest = copy.deepcopy(env)

        # 实例化ELP算法
        elp_solver = ELP(
            env=env,
            gbest=initial_gbest,
            T=T_initial,
            Q_matrix=Q_matrix,
            G=G,
            t_max=t_max,
            k=k
        )

        # 运行并输出结果
        total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
        print(f"单次实验完成 | 总迭代次数: {total_iter}")
        # 添加调试信息
        if best_sol is None or not hasattr(best_sol, 'fbs_model') or best_sol.fbs_model is None:
            print(f"Best Solution: [错误：best_sol或fbs_model无效], Best Energy: {best_energy}")
        elif len(best_sol.fbs_model.permutation) == 0 or len(best_sol.fbs_model.bay) == 0:
            print(f"Best Solution: [], Best Energy: {best_energy}")
            logger.error(f"permutation: {best_sol.fbs_model.permutation}, bay: {best_sol.fbs_model.bay}")
        else:
            print(f"Best Solution: {best_sol.fbs_model.array_2d}, Best Energy: {best_energy}")

        
        ExperimentsUtil.save_experiment_result(
            exp_instance=f"{exp_instance}_{current_date}",
            exp_algorithm=exp_algorithm,
            exp_iterations=total_iter,
            exp_solution=best_sol.fbs_model.array_2d,
            exp_fitness=best_energy,
            exp_start_time=start,
            exp_fast_time=fast,
            exp_end_time=end,
            exp_is_valid_aspect_ratio=is_valid, # <--- 添加此行
            exp_remark=exp_remark,
            exp_gbest_updates=elp_solver.gbest_update_count
        )
        # 渲染环境（原SA代码功能）
        env.reset(fbs_model=best_sol.fbs_model)
        env.render()