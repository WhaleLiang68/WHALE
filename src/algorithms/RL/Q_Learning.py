import os

import numpy as np
import gym
import copy

from src.utils import config
import src.utils.FBSUtil as FBSUtil


class QLearningAgent():

    # def __init__(self, s_dim, a_dim,initial_fitness=1.0):
    #
    #     """
    #     s_dim: 状态空间维度
    #     a_dim: 动作空间维度（操作符数量）
    #     initial_fitness: 初始解适应度值f(s)（默认取1.0防止除零）
    #     """
    #     self.a_dim = a_dim
    #     self.epsilon = 1e-8  # 防止f(s)=0导致除零错误
    #     self.Q = np.full((s_dim, a_dim), 1.0 / (initial_fitness + self.epsilon))  # 按公式初始化为1/f(s)
    #     self.times = np.zeros((s_dim, a_dim), dtype=int)  # 新增次数记录矩阵


    def __init__(self, s_dim, a_dim,epsilon=0.5,initial_fitness=1.0):

        """
        s_dim: 状态空间维度
        a_dim: 动作空间维度（操作符数量）
        initial_fitness: 初始解适应度值f(s)（默认取1.0防止除零）
        """
        self.a_dim = a_dim
        self.e = 1e-8  # 防止f(s)=0导致除零错误
        self.Q = np.full((s_dim, a_dim), 1.0 / (initial_fitness + self.e))  # 按公式初始化为1/f(s)
        self.times = np.zeros((s_dim, a_dim), dtype=int)  # 新增次数记录矩阵
        # self.times=0
        # self.q=1.0 / (initial_fitness + self.epsilon)
        self.epsilon = epsilon  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.995  # 衰减系数        self.epsilon = 0.3  # 初始探索率


    def sequential_evaluate_actions(self, env, s):
        """
        返回：最大Q值对应的动作
        """
        best_action = 0
        max_q = -np.inf
        q_values = []
        # 使用“模拟环境”思路：只在当前 fbs_model 上模拟动作，不真正修改 env
        for action in range(self.a_dim):
            f_ns = self._simulate_action_fitness(env, action)

            # 2. 按公式计算新Q值（不实际更新Q表）
            current_times = self.times[s, action]
            new_q = (self.Q[s, action] * current_times + (1.0 / (f_ns + self.e))) / (current_times + 1)
            q_values.append(new_q)

        # # 3. 恢复环境原始状态
        # env.set_state(original_state)

        # 4. 返回最大Q值对应的动作
        # print("最大Q值:",max(q_values))
        return np.argmax(q_values)

    def _simulate_action_fitness(self, env, action):
        """
        在不真正修改 env 的前提下，对给定动作进行一次“模拟”，返回得到的新解适应度 f_ns。
        只复制必要字段(fbs_model)，并用 FBSUtil 的动作与 StatusUpdatingDevice 计算 fitness。
        """
        # 环境必须提供 actions 映射和当前布局 fbs_model
        if not hasattr(env, "actions") or not hasattr(env, "fbs_model"):
            # 回退方案：退回到 deep copy + step，保证兼容性
            cloned_env = copy.deepcopy(env)
            cloned_env.step(action)
            return cloned_env.fitness

        action_name = env.actions[int(action)]

        # 只在当前 fbs_model 上做局部拷贝和操作
        sim_model = copy.deepcopy(env.fbs_model)

        # 将 permutation、bay 转为 numpy 数组以使用 FBSUtil 中的算子
        perm = np.array(sim_model.permutation)
        bay = np.array(sim_model.bay)

        # 基本邻域动作（与 DataProcessingEnv.step 中对应）
        if action_name == "facility_swap":
            perm, bay = FBSUtil.facility_swap(perm, bay)
        elif action_name == "bay_flip":
            perm, bay = FBSUtil.bay_flip(perm, bay)
        elif action_name == "bay_swap":
            perm, bay = FBSUtil.bay_swap(perm, bay)
        else:
            # 对于暂未显式支持的动作（如 repair、ga_action 等），
            # 先退回到 deep copy 方案，避免行为偏差。
            cloned_env = copy.deepcopy(env)
            cloned_env.step(action)
            return cloned_env.fitness

        # 写回模拟模型
        sim_model.permutation = perm.astype(int).tolist()
        sim_model.bay = bay.astype(int).tolist()

        # 使用 StatusUpdatingDevice 计算新布局的 mhc 和 fitness
        (
            _fac_x,
            _fac_y,
            _fac_h,
            _fac_b,
            _fac_aspect_ratio,
            _D,
            _TM,
            _mhc,
            fitness,
        ) = FBSUtil.StatusUpdatingDevice(
            sim_model,
            env.areas,
            env.H,
            env.F,
            env.fac_limit_aspect,
        )

        return fitness

    # def select_action(self, s, deterministic=True):
    #     """整合到动作选择逻辑"""
    #     if deterministic:
    #         return self.sequential_evaluate_actions(env, s)  # 需传入env对象
    #     else:
    #         # 保留原探索逻辑
    #         if np.random.rand() < 0.1:
    #             return np.random.randint(0, self.a_dim)
    #         else:
    #             return np.argmax(self.Q[s, :])
    def select_action(self, env,s, deterministic=False, is_training=True):
        if deterministic:
            return self.sequential_evaluate_actions(env, s)  # 注意：env应作为参数传入，避免全局变量
        if is_training and np.random.rand() < self.epsilon:
            # 训练时探索
            action = np.random.randint(0, self.a_dim)
            # 衰减ε
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # print("衰减率：",self.epsilon)
            return action
        else:
            # return np.argmax(self.Q[s, :])
            return self.sequential_evaluate_actions(env, s)

    def update_Q(self, s, a, f_ns):
        """
        严格按公式实现:
        Q_{t+1} = (Q_t * times + 1.0/f(ns)) / (times + 1)
        新增参数: f_ns (新解的适应度)
        """
        # print("原来的q值：",self.Q)
        # print("f_ns：", f_ns)
        current_times = self.times[s, a]
        # current_times = self.times
        epsilon = 1e-8  # 避免除零
        # print("current_times:",current_times)
        # 计算公式分子分母
        numerator = self.Q[s, a] * current_times + (1.0 / (f_ns + epsilon))
        # print("numerator:",numerator)
        # numerator = self.q * current_times + (1.0 / (f_ns + epsilon))
        denominator = current_times + 1

        # 更新Q值和次数
        self.Q[s, a] = numerator / denominator
        # self.q = numerator / denominator
        self.times[s, a] += 1
        # self.times += 1

    def save(self):
        """同时保存 Q 表和次数表"""
        save_dir = config.QLearning_RESULT_PATH
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        filename1 = f"Q_table.npy"
        filepath1 = os.path.join(save_dir, filename1)
        np.save(filepath1, self.Q)
        filename2 = f"times_table.npy"
        filepath2 = os.path.join(save_dir, filename2)
        np.save(filepath2, self.times)
        # print('Q table and times table saved.')

    def restore(self):
        """同时加载 Q 表和次数表"""
        self.Q = np.load('model/q_table.npy')
        self.times = np.load('model/times_table.npy')
        print('Q table and times table loaded.')


def evaluate_policy(env, agent):
    # s, info = env.reset()
    # done, ep_r, steps = False, 0, 0
    # while not done:
    #     a = agent.select_action(s, deterministic=True)
    #     s_next, r, dw, tr, info = env.step(a)
    #     done = (dw or tr)
    #     ep_r += r
    #     steps += 1
    #     s = s_next
    # return ep_r
    done = False
    if not done:
        # 1. 选择动作
        best_action = agent.select_action(env, s=0)
        print("选择的动作：", best_action)
        # action = agent.select_action(0, deterministic=True)

        # 2. 执行动作并获取新解适应度
        env.step(best_action)
        f_ns = env.fitness  # 假设环境提供新解适应度

        # 3. 按新公式更新 Q 值
        # agent.update_Q(s=0, a=best_action, f_ns=f_ns)
    return best_action

if __name__ == "__main__":
    # agent = QLearningAgent(s_dim=1, a_dim=5,epsilon=0.8,initial_fitness=1.0)
    instance = "AB20-ar3"
    env = gym.make("FbsEnv-v0", instance=instance)

    state = env.reset()
    print("初始适应度：",env.fitness)
    agent = QLearningAgent(s_dim=1, a_dim=5, epsilon=0.8, initial_fitness=env.fitness)
    for episode in range(2000):

        done = False
        if not done:
            # 1. 选择动作
            best_action = agent.select_action(env, s=0)
            print("选择的动作：", best_action)
            # action = agent.select_action(0, deterministic=True)

            # 2. 执行动作并获取新解适应度
            env.step(best_action)
            f_ns = env.fitness # 假设环境提供新解适应度
            print(f_ns)
            # 3. 按新公式更新 Q 值
            agent.update_Q(s=0, a=best_action, f_ns=f_ns)
            print(agent.Q)
            print(agent.times)