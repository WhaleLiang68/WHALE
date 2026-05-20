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
# from src.algorithms.RL.Q_Learning import QLearningAgent, evaluate_policy
np.bool8 = np.bool_

class ELP:
    def __init__(
        self,
        env,
        gbest,
        T,
        Q_matrix,
        G=100,
        t_max=50,
        k=10,
        bin_width=10.0,
        adaptive_bin_width=False,
        bin_width_recent_window=4000,
        bin_width_target_bins=64,
        bin_width_lower_ratio=2e-5,
        bin_width_upper_ratio=2e-3,
        bin_width_fallback_ratio=2e-4,
        bin_width_min_abs=10.0,
        bin_width_refresh_interval=200,
        adaptive_bin_width_strategy="spread",
        bin_width_scale_ratio=5e-6,
        bin_width_change_tolerance=0.05,
    ):
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
        self.gbest = gbest  # 初始全局最优解
        self.true_gbest = gbest
        self.T = T  # 初始温度
        self.T_initial = T
        self.Q_matrix = copy.deepcopy(Q_matrix)  # Q值矩阵（用于后续扩展，当前保留结构）
        self.G = G  # 外层循环最大迭代次数
        self.t_max = t_max  # 内层循环最大步数
        self.k = k  # 能量计算系数k

        # 初始化当前解和能量（E(s)用适应度表示，与原代码fitness逻辑一致）
        self.s = copy.deepcopy(gbest)
        self.current_energy = self._calculate_energy(self.s)
        self.current_search_energy = self._calculate_search_energy(self.s)
        self.best_energy = self.current_energy  # 记录最优能量（对应原代码best_fitness）

        self.true_best=self.s
        self.true_best_energy=self.best_energy

        self.energy_history = []
        self.modified_energy_history = [] # 记录修正后的能量历史
        self.prob_history = []

        # 初始化直方图 (使用字典)
        self.energy_histogram = {} 
        # 设定直方图的“箱宽” (Bin Width)
        # 这个参数很重要：决定了多大范围内的能量被视为“同一个坑”
        # 对于你的问题(5000左右的量级)，设为 1.0 或 10.0 比较合适
        self.bin_width = float(bin_width)
        self.adaptive_bin_width = bool(adaptive_bin_width)
        self.bin_width_recent_window = max(200, int(bin_width_recent_window))
        self.bin_width_target_bins = max(16, int(bin_width_target_bins))
        self.bin_width_lower_ratio = max(1e-8, float(bin_width_lower_ratio))
        self.bin_width_upper_ratio = max(
            self.bin_width_lower_ratio * 2.0,
            float(bin_width_upper_ratio),
        )
        self.bin_width_fallback_ratio = max(
            self.bin_width_lower_ratio,
            float(bin_width_fallback_ratio),
        )
        self.bin_width_min_abs = max(1.0, float(bin_width_min_abs))
        self.bin_width_refresh_interval = max(1, int(bin_width_refresh_interval))
        self.adaptive_bin_width_strategy = str(adaptive_bin_width_strategy).strip().lower() or "spread"
        if self.adaptive_bin_width_strategy not in {"spread", "scale", "hybrid"}:
            raise ValueError(
                f"未知的自适应分箱策略: {adaptive_bin_width_strategy!r}，"
                "可选值为 spread、scale、hybrid。"
            )
        self.bin_width_scale_ratio = max(1e-9, float(bin_width_scale_ratio))
        self.bin_width_change_tolerance = max(0.0, float(bin_width_change_tolerance))
        self.search_energy_history = []
        self.bin_width_history = [float(self.bin_width)]
        # 记录 gbest 下降趋势
        self.best_history = [self.best_energy]
        self.gbest_plot_path = None
        self.gbest_update_count = 0
        # 限制非有限能量告警次数，避免实验日志被刷屏
        self.non_finite_energy_warning_count = 0

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

    def _get_bin_index(self, energy):
        """将连续的能量值转换为整数索引（分箱）"""
        energy_value = float(energy)
        if not np.isfinite(energy_value):
            return None
        return int(energy_value / self.bin_width)

    def _calculate_search_energy(self, solution):
        """计算搜索阶段使用的有限代理能量，避免不可行解返回 inf 后搜索停滞。"""
        raw_energy = float(self._calculate_energy(solution))
        if np.isfinite(raw_energy):
            return raw_energy

        d_inf = getattr(solution, "current_d_inf", None)
        mhc = getattr(solution, "MHC", getattr(solution, "mhc", None))
        if d_inf is None or mhc is None:
            return np.inf

        mhc_value = float(mhc)
        if not np.isfinite(mhc_value):
            return np.inf

        penalty_anchor = getattr(solution, "best_feasible_cost", np.inf)
        if penalty_anchor is None or not np.isfinite(float(penalty_anchor)) or float(penalty_anchor) <= 0:
            # 尚未见到可行解时，用当前 MHC 的量级构造一个稳定的大惩罚项，
            # 让算法优先降低 d_inf，再在同一 d_inf 下比较 MHC。
            penalty_anchor = max(abs(mhc_value), 1.0) * 10.0

        return float(mhc_value) + float(max(int(d_inf), 1)) * float(penalty_anchor)

    def _warn_non_finite_energy(self, energy, context):
        """记录非有限能量告警，帮助定位不可行解阶段的问题。"""
        if self.non_finite_energy_warning_count < 10:
            logger.warning(
                f"{context}时检测到非有限能量 {energy}，本次跳过直方图分箱并按 H=0 处理。"
            )
        self.non_finite_energy_warning_count += 1

    def _update_histogram(self, energy):
        """更新直方图：当前能量对应的计数 +1"""
        idx = self._get_bin_index(energy)
        if idx is None:
            self._warn_non_finite_energy(energy, "更新直方图")
            return
        if idx not in self.energy_histogram:
            self.energy_histogram[idx] = 0
        self.energy_histogram[idx] += 1

    def _get_H_value(self, energy):
        """获取当前能量对应的 H 值（访问次数）"""
        idx = self._get_bin_index(energy)
        if idx is None:
            self._warn_non_finite_energy(energy, "读取H值")
            return 0
        return self.energy_histogram.get(idx, 0)

    def _get_histogram_reference_energy(self):
        """获取自适应分箱使用的能量量级参考。"""
        if np.isfinite(float(self.best_energy)):
            return float(self.best_energy)
        return np.inf

    def _get_bin_width_bounds(self, reference_energy):
        """根据当前目标值量级给出自适应箱宽的上下界。"""
        if not np.isfinite(reference_energy):
            return float(self.bin_width), float(self.bin_width), float(self.bin_width)

        energy_scale = max(abs(float(reference_energy)), 1.0)
        min_width = max(float(self.bin_width_min_abs), energy_scale * float(self.bin_width_lower_ratio))
        max_width = max(min_width, energy_scale * float(self.bin_width_upper_ratio))
        fallback_width = max(min_width, energy_scale * float(self.bin_width_fallback_ratio))
        return min_width, max_width, fallback_width

    def _clamp_bin_width(self, width, reference_energy):
        """把候选箱宽限制在当前实例尺度允许的范围内。"""
        min_width, max_width, _ = self._get_bin_width_bounds(reference_energy)
        if not np.isfinite(width) or float(width) <= 0:
            return float(min_width)
        return float(min(max(float(width), min_width), max_width))

    def _clamp_scale_bin_width(self, width, reference_energy):
        """按比例缩放策略的箱宽边界；允许低于 spread 策略的比例下界。"""
        if not np.isfinite(reference_energy):
            return float(self.bin_width)
        energy_scale = max(abs(float(reference_energy)), 1.0)
        min_width = max(1.0, float(self.bin_width_min_abs))
        max_width = max(min_width, energy_scale * float(self.bin_width_upper_ratio))
        if not np.isfinite(width) or float(width) <= 0:
            return float(min_width)
        return float(min(max(float(width), min_width), max_width))

    def _get_spread_adaptive_bin_width(self, reference_energy):
        """按近期能量分布跨度估计箱宽；这是旧版 adaptive 的逻辑。"""
        if not np.isfinite(reference_energy):
            return float(self.bin_width)

        min_width, max_width, fallback_width = self._get_bin_width_bounds(reference_energy)

        finite_history = [
            float(value)
            for value in self.search_energy_history[-int(self.bin_width_recent_window):]
            if np.isfinite(value)
        ]
        if len(finite_history) >= 32:
            q10, q90 = np.percentile(np.asarray(finite_history, dtype=float), [10, 90])
            spread = max(float(q90) - float(q10), 0.0)
            if spread > 0:
                candidate_width = spread / float(max(1, int(self.bin_width_target_bins)))
                if np.isfinite(candidate_width) and candidate_width > 0:
                    return float(min(max(candidate_width, min_width), max_width))

        return float(min(max(fallback_width, min_width), max_width))

    def _get_scale_adaptive_bin_width(self, reference_energy):
        """按最优目标值量级估计箱宽，避免不同规模算例需要手工固定 bin_width。"""
        if not np.isfinite(reference_energy):
            return float(self.bin_width)
        energy_scale = max(abs(float(reference_energy)), 1.0)
        candidate_width = energy_scale * float(self.bin_width_scale_ratio)
        return self._clamp_scale_bin_width(candidate_width, reference_energy)

    def _get_adaptive_bin_width(self, reference_energy):
        """根据配置策略估计能量分箱宽度。"""
        if self.adaptive_bin_width_strategy == "spread":
            return self._get_spread_adaptive_bin_width(reference_energy)
        if self.adaptive_bin_width_strategy == "scale":
            return self._get_scale_adaptive_bin_width(reference_energy)

        spread_width = self._get_spread_adaptive_bin_width(reference_energy)
        scale_width = self._get_scale_adaptive_bin_width(reference_energy)
        # 旧 spread 策略容易把 Du62 的箱宽放大到几百/几千，导致 H 过度粗粒度。
        # hybrid 取两者中更细的估计，保留跨规模缩放能力，同时限制过粗分箱。
        return self._clamp_scale_bin_width(min(spread_width, scale_width), reference_energy)

    def _refresh_adaptive_bin_width(self):
        """刷新自适应分箱，并用新的 bin_width 重建访问直方图。"""
        if not self.adaptive_bin_width:
            return False

        new_bin_width = self._get_adaptive_bin_width(self._get_histogram_reference_energy())
        if not np.isfinite(new_bin_width) or new_bin_width <= 0:
            return False
        relative_change = abs(float(new_bin_width) - float(self.bin_width)) / max(abs(float(self.bin_width)), 1e-12)
        if relative_change <= float(self.bin_width_change_tolerance):
            return False

        self.bin_width = float(new_bin_width)
        rebuilt_histogram = {}
        for energy in self.search_energy_history:
            if not np.isfinite(energy):
                continue
            idx = self._get_bin_index(float(energy))
            if idx is None:
                continue
            rebuilt_histogram[idx] = rebuilt_histogram.get(idx, 0) + 1
        self.energy_histogram = rebuilt_histogram
        self.bin_width_history.append(float(self.bin_width))
        return True

    def _calculate_H(self, current_E, t):
        """计算H(E(s),t)函数（图片中能量修正项，此处实现基于迭代步的衰减函数）
        设计逻辑：随内循环步数t增加，H值递减，符合"迭代后期减少能量扰动"的直觉
        """
        # H = 初始扰动强度 * (1 - t/self.t_max)，确保t∈[1,t_max]时H非负
        initial_disturbance = 300.0  # 初始扰动强度，可根据问题调整
        return initial_disturbance * (1 - t / self.t_max)

    def _generate_new_solution(self, s):
        """基于当前解s产生新解s'（对应图片中"基于s产生新的解s'"步骤）
        沿用原SA算法的邻域操作逻辑，确保解的合法性
        """
        new_s = copy.deepcopy(s)
        # 随机选择邻域操作符（0-4，与原代码op选择范围一致）
        op = np.random.randint(0, 6)
        # 执行操作生成新解（通过环境step方法保证解的合法性）
        new_s.step(op)
        # if np.random(0,1)<0.2:
        #     FBSUtil.greedy_local_search_action()
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
            op = np.random.randint(0, 5)
            _, _, _, _, info = self.env.step(op)
            new_fitness = info['current_fitness']
            
            # C. 贪婪判断
            if new_fitness < current_best_fitness and self._check_aspect_ratio_constraint(self.env)==True:
                # --- 变好了：接受 ---
                current_best_fitness = new_fitness
                # 更新 ELP 类的全局最优记录
                self.gbest = copy.deepcopy(self.env)
                self.true_gbest=copy.deepcopy(self.env)
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
            self.current_search_energy = self._calculate_search_energy(self.s)

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

        plt.figure(figsize=(12, 6)) # 设置宽一点，因为迭代次数很多
        
        # 绘制能量曲线
        # linewidth设置细一点，alpha设置透明度，以便看清密集的波动
        plt.plot(self.energy_history, linewidth=0.5, color='blue', alpha=0.6, label='Current Energy')
        
        # 可选：绘制一条红线表示最终的最优解，方便对比差距
        plt.axhline(y=self.best_energy, color='red', linestyle='--', linewidth=1.5, label=f'Best Found ({self.best_energy:.2f})')

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
        plt.plot(self.modified_energy_history, linewidth=0.5, color='orange', alpha=0.8, label='Modified Energy (E + k*H)')

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
        plt.scatter(range(len(self.prob_history)), self.prob_history, s=0.5, color='green', alpha=0.5, label='Acceptance Probability')

        plt.title(f"ELP Acceptance Probability History\n(Temperature Decay)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Probability P")
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0]) # 设置Y轴刻度
        plt.legend(markerscale=10) # 图例里的点放大一点以便看见
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
        # 初始化解
        self._refresh_adaptive_bin_width()

        while g < self.G:  # 外层循环：g < G（图片中算法外层循环条件）
            t = 0  # 内层循环计数器（算法中t从1开始）

            while t < self.t_max:  # 内层循环：t < t_max（图片中算法内层循环条件）
                # 1. 基于当前解s产生新解s'（图片中步骤1）
                # self.s.best_fitness=self.best_energy
                # print(f"self.s.best_fitness: {self.s.best_fitness}")
                # print(f"self.best_energy: {self.best_energy}")
                s_prime = self._generate_new_solution(self.s)
                # if s_prime.fbs_model.permutation == self.s.fbs_model.permutation and \
                #     s_prime.fbs_model.bay == self.s.fbs_model.bay:
                #     logger.warning("警告：产生了完全相同的新解（无效变异）！")
                #     continue
                # 2. 计算当前解和新解的原始能量（图片中E(s)和E(s')）
                # E_s = self.current_energy
                E_s = self._calculate_energy(self.s)
                self.current_energy = E_s
                search_E_s = self._calculate_search_energy(self.s)
                self.current_search_energy = search_E_s
                # penalty_factor = self.current_energy * 0.05
                penalty_factor = self.k
                E_s_prime = self._calculate_energy(s_prime)
                search_E_s_prime = self._calculate_search_energy(s_prime)
                H_s = self._get_H_value(search_E_s)
                H_s_prime = self._get_H_value(search_E_s_prime)
                # 3. 计算修正后的能量（图片中E'(s) = E(s) + k*H(E(s),t)）
                E_prime_s = search_E_s + penalty_factor * H_s
                E_prime_s_prime = search_E_s_prime + penalty_factor * H_s_prime
                # 4. 接受准则（图片中"若I()<E)rp() tcn"修正为标准Metropolis准则，基于能量差）
                delta_E_prime = E_prime_s_prime - E_prime_s  # 修正能量差
                # 定义玻尔兹曼常数（单位：J/K）
                # k_boltzmann = 1.380649e-23
                k_boltzmann = 1.0
                accept = False
                if delta_E_prime < 0:
                    prob = 1.0
                    accept = True
                else:
                    if np.isfinite(delta_E_prime):
                        exponent = -delta_E_prime / (self.T * k_boltzmann + 1e-10)
                        prob = math.exp(exponent)
                        if np.random.rand() < prob:
                            accept = True
                    else:
                        prob = 0.0
                self.prob_history.append(prob)
                if accept:
                    # 接受新解...
                # if delta_E_prime < 0 or np.random.rand() < math.exp(-delta_E_prime / (self.T * k_boltzmann + 1e-10)):
                    # if delta_E_prime >= 0:
                    #     print(f"delta_E_prime: {delta_E_prime}")
                    #     print(f"接受值: {math.exp(-delta_E_prime / (self.T * k_boltzmann + 1e-10))}")
                    # 接受新解：更新当前解和当前能量
                    self.s = s_prime
                    self.current_energy = E_s_prime
                    self.current_search_energy = search_E_s_prime
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
                    
                # self._update_histogram(self.current_energy)
                current_total_step = g * self.t_max + t
                total_steps = self.G * self.t_max

                if (
                    self.adaptive_bin_width
                    and current_total_step > 0
                    and current_total_step % self.bin_width_refresh_interval == 0
                ):
                    self._refresh_adaptive_bin_width()
    
                if current_total_step < 0.8 * total_steps:
                    self._update_histogram(self.current_search_energy)
                H=self._get_H_value(self.current_search_energy)
                current_step_modified_energy = self.current_search_energy + penalty_factor * H
                self.modified_energy_history.append(current_step_modified_energy)
                # 6. 更新H(E(s),t)（图片中步骤，此处H随t动态计算，无需额外存储）
                # 7. 内循环步数+1（图片中"令t=t+1"）
                self.search_energy_history.append(self.current_search_energy)
                self.energy_history.append(self.current_energy)
                t += 1


            # 8. 外层循环步数+1（图片中"令g=g+1"）
            g += 1
            # 可选：温度衰减（保留原SA的温度逻辑，增强算法收敛性）
            self.T *= 0.995  # 衰减系数可调整，与原代码alpha保持一致
            self.k = max(1.0 , self.k * (self.T/self.T_initial))

        end_time = datetime.datetime.now()
        print(f"prob: {prob}")
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
        self.true_best_energy=self._calculate_energy(self.true_gbest)
        return (
            total_iterations,
            is_valid,  # <--- 添加此行
            # self.gbest,
            self.true_gbest,
            # self.best_energy,
            self.true_best_energy,
            start_time,
            end_time,
            fast_time
        )


if __name__ == "__main__":
    # 实验参数
    exp_instance = "Du62"
    exp_algorithm = "ELP"  # 算法名称改为ELP
    exp_remark = "k=15*T/T_initial,bin_width=5,后期冻结直方图,t_max = 300,G=2000,getfitness2，修复了贪婪操作，修改repair，g_best=5396.6，惩罚指数：1至5,step05（facility_insert），最优贪婪0-4"
    exp_number = 50
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # ELP算法参数（根据图片定义配置）
    G = 2000 # 外层循环最大迭代步数G
    t_max = 300 # 内循环步数t_max
    T_initial = 10000.0  # 初始温度T
    k = 15  # 系数k（可根据实验调整）
    bin_width=5.0 # 直方图箱宽
    Q_matrix = np.zeros((1, 5))  # Q值矩阵（1个状态，5个操作符，初始为0）

    if is_exp:
        # 多轮实验（30次）
        for i in range(exp_number):
            logger.info(f"第{i + 1}次实验（ELP算法）")
            try:
                # 1. 初始化环境
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                env.reset()  # 重置环境获取初始合法解
                initial_gbest = env  # 初始gbest为环境初始解
                base_env=env
                # ========================================================
                if "AB20" in str(exp_instance):
                    try:
                        custom_tm = np.loadtxt(
                            r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\data\AB20(1963).csv",
                            delimiter=",", encoding="utf-8-sig")
                        if hasattr(base_env, 'F'):
                            base_env.F = custom_tm.copy()
                        if hasattr(base_env, 'TM'):
                            base_env.TM = custom_tm.copy()
                        logger.info(f"检测到实例名包含 AB20 ({exp_instance})，已成功从 AB20(1963).csv 覆盖自定义物流量！")
                    except Exception as e:
                        logger.error(f"加载自定义物流量失败: {e}")
                if "SC30" in str(exp_instance):
                    try:
                        custom_tm = np.loadtxt(
                            r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\data\SC30_Flow_Matrix.csv",
                            delimiter=",", encoding="utf-8-sig")
                        if hasattr(base_env, 'F'):
                            base_env.F = custom_tm.copy()
                        if hasattr(base_env, 'TM'):
                            base_env.TM = custom_tm.copy()
                        logger.info(
                            f"检测到实例名包含 SC30 ({exp_instance})，已成功从 SC30_Flow_Matrix.csv 覆盖自定义物流量！")
                    except Exception as e:
                        logger.error(f"加载自定义物流量失败: {e}")
                if "SC35" in str(exp_instance):
                    try:
                        custom_tm = np.loadtxt(
                            r"C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\data\SC35_Flow_Matrix.csv",
                            delimiter=",", encoding="utf-8-sig")
                        if hasattr(base_env, 'F'):
                            base_env.F = custom_tm.copy()
                        if hasattr(base_env, 'TM'):
                            base_env.TM = custom_tm.copy()
                        logger.info(
                            f"检测到实例名包含 SC35 ({exp_instance})，已成功从 SC35_Flow_Matrix.csv 覆盖自定义物流量！")
                    except Exception as e:
                        logger.error(f"加载自定义物流量失败: {e}")
                # ========================================================

                # 2. 实例化ELP算法（传入图片要求的所有输入参数）
                elp_solver = ELP(
                    env=env,
                    gbest=initial_gbest,
                    T=T_initial,
                    Q_matrix=Q_matrix,
                    G=G,
                    t_max=t_max,
                    k=k,
                    bin_width=bin_width
                )

                # 3. 运行ELP算法
                total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
                search_duration = (fast - start).total_seconds()

                # 4. 输出结果
                logger.info(f"第{i + 1}次实验完成 | 最优能量: {best_energy} | 寻优耗时: {search_duration:.4f}秒")
                print(f"Best Solution: {best_sol.fbs_model.array_2d}, Best Energy: {best_energy}")

                # 5. 保存实验结果
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
                logger.exception(f"第{i + 1}次实验失败！错误信息: {str(e)}")
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
