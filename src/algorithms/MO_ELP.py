import numpy as np
np.bool8 = np.bool_
import copy
import math
import datetime
from loguru import logger
import gym

import src.utils.FBSUtil as FBSUtil
from src.utils.MO_FBSUtil import MO_FBSUtil
from src.utils.ParetoArchive import ParetoArchive, Solution
from src.utils.FBSModel import FBSModel
from src.utils.MO_DataGenerator import MO_DataGenerator

class MO_ELP_Agent:
    """
    多目标 RL-ELP 代理
    对应 Mechanism 2: Multi-objective Task
    """
    def __init__(self, env, G=2000, t_max=300):
        self.env = env
        self.G = G
        self.t_max = t_max
        
        # --- 动作空间 (权重向量) ---
        # 对应开题报告中的 W_cost, W_shape, W_balance 等
        # 格式: [w_MHC, w_CR, w_DR, w_AR]
        self.actions = [
            [0.7, 0.1, 0.1, 0.1], # 0: 成本优先 (W_cost)
            [0.1, 0.1, 0.1, 0.7], # 1: 形状优先 (W_shape)
            [0.25, 0.25, 0.25, 0.25], # 2: 均衡权重 (W_balance)
            [0.1, 0.4, 0.4, 0.1], # 3: 关系优先 (W_rel) - 补充
            [0.4, 0.1, 0.1, 0.4]  # 4: 成本形状均衡 - 补充
        ]
        self.n_actions = len(self.actions)
        
        # --- Q-Learning 参数 ---
        self.Q = np.zeros((1, self.n_actions)) # 简化状态空间，假设只有一个全局状态或根据稀疏度分状态
        self.epsilon = 0.5
        self.alpha = 0.1
        self.gamma = 0.9
        
        # --- 档案 ---
        self.archive = ParetoArchive(capacity=50)
        
        # --- 辅助数据 (需要模拟或从文件读取) ---
        self.n = env.n
        # 模拟生成亲密关系矩阵和距离要求矩阵 (因为原pkl中没有)
        # 在实际使用中应从 data 读取
        # np.random.seed(42)
        # self.rel_matrix = np.random.randint(0, 5, (self.n, self.n)) 
        # self.dist_req_matrix = np.random.randint(0, 2, (self.n, self.n)) 
        # --- 加载多目标数据 ---
        # 自动识别算例名称 (假设 env 有 instance 属性，或者从 config 获取)
        instance_name = getattr(self.env, 'instance_name', 'UnknownInstance') 
        
        self.rel_matrix, self.dist_req_matrix = MO_DataGenerator.load_or_generate_data(
            self.n, 
            instance_name=instance_name
        )

    def get_state(self):
        """
        状态感知: 对应开题报告中的 St = {phi_sparse, phi_tag}
        简化实现：返回 0 (单一状态) 或基于档案稀疏度计算状态索引
        """
        return 0

    def select_action(self, state):
        """Epsilon-Greedy 策略选择权重向量"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def run(self):
        logger.info("启动多目标 RL-ELP 优化...")
        start_time = datetime.datetime.now()
        
        # 初始化
        self.env.reset()
        current_sol = copy.deepcopy(self.env.fbs_model)
        
        # 初始评估
        current_objs = self._evaluate(current_sol)
        current_energy = 999999 # 初始虚拟能量
        
        T = 5000.0 # 温度
        
        for g in range(self.G):
            # 1. RL 感知状态并选择权重
            state = self.get_state()
            action_idx = self.select_action(state)
            weights = self.actions[action_idx]
            
            # 计算当前权重下的能量
            current_energy = MO_FBSUtil.aggregated_energy(current_objs, weights)
            
            # 内循环 (ELP)
            for t in range(self.t_max):
                # 2. 生成候选解 (使用底层算子)
                # 随机选择底层算子 (swap, insert, etc.)
                op = np.random.randint(0, 5) 
                
                # 创建临时环境进行step (为了复用单目标的算子逻辑)
                # 注意：这里需要高效处理，避免频繁深拷贝环境
                # 使用 FBSUtil 直接操作 model
                candidate_sol = copy.deepcopy(current_sol)
                # 模拟 step (你需要确保 MO_FBSUtil 或 FBSUtil 有纯 Model 操作接口)
                # 这里假设复用现有的 step 逻辑，通过 hack 环境实现
                self.env.fbs_model = candidate_sol
                self.env.step(op) # 执行变异
                candidate_sol = self.env.fbs_model
                
                # 3. 计算候选解的多目标值
                candidate_objs = self._evaluate(candidate_sol)
                
                # 4. 计算聚合能量 (使用 RL 选择的 weights)
                candidate_energy = MO_FBSUtil.aggregated_energy(candidate_objs, weights)
                
                # 5. Metropolis 接受准则
                delta_E = candidate_energy - current_energy
                if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
                    # 接受 ELP 层面的新解
                    current_sol = copy.deepcopy(candidate_sol)
                    current_objs = candidate_objs
                    current_energy = candidate_energy
                    
                    # 6. Pareto 档案更新 (传递到档案层)
                    new_sol_obj = Solution(
                        model=copy.deepcopy(current_sol),
                        objectives=current_objs,
                        energy=current_energy
                    )
                    
                    is_added, r_dom = self.archive.update(new_sol_obj)
                    
                    # 7. 计算奖励并更新 Q 表 (RL层)
                    if is_added:
                        r_div = self.archive.calculate_diversity_reward(new_sol_obj)
                        # 公式 31: R = lambda1 * R_dom + lambda2 * R_div
                        reward = 1.0 * r_dom + 10.0 * r_div
                        
                        # Q-Learning 更新
                        old_q = self.Q[state, action_idx]
                        self.Q[state, action_idx] = old_q + self.alpha * (reward - old_q)
                        
                        # logger.info(f"Archive Update: Size={len(self.archive.solutions)}, Reward={reward:.2f}")

            # 衰减
            T *= 0.995
            self.epsilon *= 0.995
            
            if g % 10 == 0:
                print(f"Gen {g}/{self.G}: Archive Size {len(self.archive.solutions)}, T={T:.2f}")

        end_time = datetime.datetime.now()
        logger.info(f"优化完成。Pareto解数量: {len(self.archive.solutions)}")
        return self.archive.solutions

    def _evaluate(self, model):
        """辅助函数：计算给定 Model 的所有目标值"""
        # 复用 StatusUpdatingDevice 计算坐标和 MHC
        (fac_x, fac_y, fac_b, fac_h, _, _, _, mhc, _) = FBSUtil.StatusUpdatingDevice(
            model, self.env.areas, self.env.H, self.env.F, self.env.fac_limit_aspect
        )
        
        # 计算 CR, DR, AR
        objs = MO_FBSUtil.calculate_objectives(
            fac_x, fac_y, fac_b, fac_h, mhc, self.n,
            self.rel_matrix, self.dist_req_matrix
        )
        return objs

# 使用示例
if __name__ == "__main__":
    from src.utils.DataExtractor import DataProcessingEnv
    
    # 1. 加载环境
    env = gym.make("FbsEnv-v0", instance="AB20-ar3")
    
    # 2. 初始化多目标 Agent
    mo_agent = MO_ELP_Agent(env, G=2000, t_max=500) # 测试用参数
    
    # 3. 运行
    pareto_front = mo_agent.run()
    
    # 4. 输出结果
    print("\nPareto Front Solutions:")
    for i, sol in enumerate(pareto_front):
        print(f"Sol {i}: MHC={sol.objectives[0]:.0f}, CR={sol.objectives[1]:.1f}, AR={sol.objectives[3]:.2f}")