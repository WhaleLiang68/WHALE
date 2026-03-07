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
        self.Q_matrix = copy.deepcopy(Q_matrix)  # Q值矩阵（用于后续扩展，当前保留结构）
        self.G = G  # 外层循环最大迭代次数
        self.t_max = t_max  # 内层循环最大步数
        self.k = k  # 能量计算系数k

        # 初始化当前解和能量（E(s)用适应度表示，与原代码fitness逻辑一致）
        self.s = copy.deepcopy(gbest)
        self.current_energy = self._calculate_energy(self.s)
        self.best_energy = self.current_energy  # 记录最优能量（对应原代码best_fitness）
        
        # --- 新增代码 (根据 image_7d3076.png) ---
        # 初始化直方图函数 H(E(X))
        self.energy_histogram = {}
        # ----------------------------------------

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

    def _generate_new_solution(self, s):
        """基于当前解s产生新解s'（对应图片中"基于s产生新的解s'"步骤）
        沿用原SA算法的邻域操作逻辑，确保解的合法性
        """
        new_s = copy.deepcopy(s)
        # 随机选择邻域操作符（0-4，与原代码op选择范围一致）
        op = np.random.randint(0, 5)
        # 执行操作生成新解（通过环境step方法保证解的合法性）
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

    def run(self):
        """执行ELP算法主流程（严格遵循图片中算法逻辑）"""
        start_time = datetime.datetime.now()
        fast_time = start_time  # 记录首次找到最优解的时间
        g = 0  # 外层循环计数器（算法中g从1开始）
        # 初始化解

        while g < self.G:  # 外层循环：g < G（图片中算法外层循环条件）
            t = 0  # 内层循环计数器（算法中t从1开始）

            while t < self.t_max:  # 内层循环：t < t_max（图片中算法内层循环条件）
                # 1. 基于当前解s(X(1))产生新解s'(X(2))
                s_prime = self._generate_new_solution(self.s)
                # 2. 计算当前解和新解的原始能量
                E_s = self.current_energy # E(X(1))
                E_s_prime = self._calculate_energy(s_prime) # E(X(2))
                
                # --- 修改代码---
                # 为了使用浮点数作为字典键，我们对其进行舍入
                # X(1) 对应的能量键
                key_E_s = round(E_s, 2) 
                # X(2) 对应的能量键
                key_E_s_prime = round(E_s_prime, 2)
                # ----------------------------------------
                
                # 3. 计算修正后的能量
                H_val = self._calculate_H(E_s, t) # H(E(s),t)
                E_prime_s = E_s + self.k * H_val
                E_prime_s_prime = E_s_prime + self.k * H_val
                # 4. 接受准则
                delta_E_prime = E_prime_s_prime - E_prime_s  # 修正能量差
                r = np.random.rand()  # 产生[0,1)随机数r
                k_boltzmann = 1.380649e-23
                
                # --- 修改代码 (根据 image_7d3076.png) ---
                # if ... : 对应 "当新产生的状态 X(2) 接受时"
                if delta_E_prime < 0 or r < math.exp(-delta_E_prime / (self.T * k_boltzmann)):
                    # 接受新解：更新当前解和当前能量
                    self.s = copy.deepcopy(s_prime)
                    self.current_energy = E_s_prime
                    
                    # 更新 X(2) 的直方图函数值
                    # 即 H(E(X(2))) = H(E(X(2))) + 1
                    self.energy_histogram[key_E_s_prime] = self.energy_histogram.get(key_E_s_prime, 0) + 1

                    # 5. 更新全局最优解gbest（图片中"更新gbest"步骤）
                    if self.current_energy < self.best_energy:
                        self.gbest = copy.deepcopy(self.s)
                        self.best_energy = self.current_energy
                        fast_time = datetime.datetime.now()  # 更新首次找到最优解的时间
                
                else:
                    # 否则 (即拒绝新解)
                    # 更新 X(1) 的直方图函数值
                    # 即 H(E(X(1))) = H(E(X(1))) + 1
                    self.energy_histogram[key_E_s] = self.energy_histogram.get(key_E_s, 0) + 1
                # ----------------------------------------

                # 6. 更新H(E(s),t)（图片中步骤，此处H随t动态计算，无需额外存储）
                # 7. 内循环步数+1（图片中"令t=t+1"）
                t += 1

            # 8. 外层循环步数+1（图片中"令g=g+1"）
            g += 1
            # 可选：温度衰减（保留原SA的温度逻辑，增强算法收敛性）
            self.T *= 0.995  # 衰减系数可调整，与原代码alpha保持一致

        end_time = datetime.datetime.now()
        
        # (可选) 打印最终的直方图，查看能量分布
        logger.debug(f"最终能量直方图: {self.energy_histogram}")

        # 对最终结果进行合法性检查（仅检查宽高比约束）
        is_valid = self._check_aspect_ratio_constraint(self.gbest)
        if not is_valid:
            logger.warning("最终解不满足宽高比约束")
        else:
            logger.info("最终解满足宽高比约束")
        
        # 返回结果格式与原SA代码兼容，便于后续实验分析
        total_iterations = self.G * self.t_max  # 总迭代次数（外层*内层）
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
    exp_algorithm = "ELP"  # 算法名称改为ELP
    exp_remark = "增大迭代步数，修改版ELP+getfitness2+修改了bay_flip操作+g_best=5396.6"
    exp_number = 600
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # ELP算法参数（根据图片定义配置）
    G = 1000  # 外层循环最大迭代步数G
    t_max = 500  # 内循环步数t_max
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
                    exp_instance=f"{exp_instance}_{current_date}",
                    exp_algorithm=exp_algorithm,
                    exp_iterations=total_iter,
                    exp_solution=best_sol.fbs_model.array_2d,
                    exp_fitness=best_energy,  # 能量对应原fitness，字段名保持兼容
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_is_valid_aspect_ratio=is_valid, # <--- 添加此行 (获取 is_valid)
                    exp_remark=exp_remark
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
            exp_remark=exp_remark
        )
        # 渲染环境（原SA代码功能）
        env.reset(fbs_model=best_sol.fbs_model)
        env.render()