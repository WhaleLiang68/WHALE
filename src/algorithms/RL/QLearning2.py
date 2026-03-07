import os
import numpy as np
import gym
from src.utils import config

class QLearningAgent():
    def __init__(self, s_dim, a_dim, epsilon=0.5, initial_fitness=1.0):
        """
        s_dim: 状态空间维度 (通常为1，因为SA是单点搜索)
        a_dim: 动作空间维度 (对应算子数量，如 Facility Swap, Bay Swap, Bay Flip, Repair)
        initial_fitness: 初始解适应度值 f(s)
        """
        self.a_dim = a_dim
        self.e = 1e-8  # 防止除零的微小量
        
        # 【论文对应】初始化 Q 值
        # "At the start of each iteration, each operator is assigned an initial value equal to the inverse of the fitness value"
        self.Q = np.full((s_dim, a_dim), 1.0 / (initial_fitness + self.e)) 
        
        # 记录每个算子被选中的次数
        self.times = np.zeros((s_dim, a_dim), dtype=int) 

        # 探索参数
        self.epsilon = epsilon  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.995  # 衰减系数

    def select_action(self, s, is_training=True):
        """
        【论文对应】Section 4.2 & Fig. 2: epsilon-Greedy Controller
        根据 Q 表选择动作，而不是模拟执行。
        """
        # 1. 探索 (Exploration): 以 epsilon 概率随机选择
        # "To promote exploration... incorporated through the epsilon-greedy approach"
        if is_training and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.a_dim)
            
            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return action
        
        # 2. 开发 (Exploitation): 选择当前 Q 值最大的算子
        # "Select an operator OP with the maximum Q value"
        else:
            # 获取当前状态下所有动作的Q值
            q_values = self.Q[s, :]
            
            # 找到最大Q值的索引。如果有多个最大值，随机选一个（打破平局）
            # np.argmax 只返回第一个最大值，这里改用 where 以防 Q 值初始全一样时总是选第0个
            max_indices = np.where(q_values == np.max(q_values))[0]
            action = np.random.choice(max_indices)
            
            return action

    def update_Q(self, s, a, f_ns):
        """
        【论文对应】Eq. 10: Information Update
        Q_{t+1} = (Q_t * times + 1.0/f(ns)) / (times + 1)
        """
        current_times = self.times[s, a]
        
        # 计算 Reward (适应度的倒数)
        # "reward Q is assigned based on the degree of improvement... 1.0 / f(ns)"
        reward = 1.0 / (f_ns + self.e)
        
        # 公式分子：旧总分 + 新奖励
        numerator = self.Q[s, a] * current_times + reward
        
        # 公式分母：总次数 + 1
        denominator = current_times + 1

        # 更新 Q 表
        self.Q[s, a] = numerator / denominator
        
        # 更新次数
        self.times[s, a] += 1

    def save(self):
        """保存模型"""
        save_dir = config.QLearning_RESULT_PATH
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "Q_table.npy"), self.Q)
        np.save(os.path.join(save_dir, "times_table.npy"), self.times)

    def restore(self):
        """加载模型"""
        self.Q = np.load('model/Q_table.npy')
        self.times = np.load('model/times_table.npy')
        print('Q table and times table loaded.')

if __name__ == "__main__":
    # 测试代码
    instance = "AB20-ar3"
    try:
        env = gym.make("FbsEnv-v0", instance=instance)
    except:
        print("请确保 gym 环境已正确注册，此处仅为代码逻辑演示")
        exit()

    # 初始化环境
    state = env.reset() # 注意：gym新版本 reset 返回 (obs, info)
    if isinstance(state, tuple): state = state[0]
        
    initial_fitness = env.fitness if hasattr(env, 'fitness') else 1000.0
    print("初始适应度：", initial_fitness)

    # 初始化 Agent
    # s_dim=1 因为 LSA 通常只维护一个当前解，不区分复杂状态，可视作单状态(0)
    agent = QLearningAgent(s_dim=1, a_dim=5, epsilon=0.5, initial_fitness=initial_fitness)

    for episode in range(100): # 模拟迭代
        # 1. Controller 选择动作 (基于 Q 表)
        # 这里不需要传入 env，也不需要 simulate
        action = agent.select_action(s=0, is_training=True)
        
        # 2. 环境执行动作 (Env Step)
        # "Generate s' from s with OP"
        # 这里的 step 内部应当包含具体的 Swap/Flip 逻辑
        next_state, reward, done, truncated, info = env.step(action) 
        
        # 获取新解的适应度 (f_ns)
        # 注意：env.step 返回的 reward 通常是强化学习定义的奖励，
        # 但论文公式明确要求使用 f(ns) (Cost值) 来更新。
        # 假设 env.fitness 存储了当前的 Cost
        f_ns = env.fitness 

        # 3. 反馈更新 (Feedback & Update)
        # "Update Q with Equ. 10"
        agent.update_Q(s=0, a=action, f_ns=f_ns)

        print(f"Iter: {episode}, Action: {action}, Fitness: {f_ns:.2f}, Q-val: {agent.Q[0, action]:.6f}")

    print("\nFinal Q Table:\n", agent.Q)
    print("Times Table:\n", agent.times)