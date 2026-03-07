import os
import numpy as np
import gym
from src.utils import config

import os
import numpy as np
from src.utils import config


class QLearningAgent:
    def __init__(self, s_dim, a_dim, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        """
        标准 Q-Learning Agent

        Args:
            s_dim (int): 状态空间的大小 (例如 10 个阶段)
            a_dim (int): 动作空间的大小 (例如 4 个算子)
            learning_rate (float): 学习率 alpha (0.1 通常是不错的起点)
            gamma (float): 折扣因子 (0~1)，关注未来的程度
            epsilon (float): 探索率
        """
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 初始化 Q 表：二维数组 [状态数, 动作数]
        # 初始化为0，或者很小的值
        self.Q = np.zeros((s_dim, a_dim))

    def select_action(self, state, is_training=True):
        """
        标准的 epsilon-greedy 策略
        """
        if is_training and np.random.rand() < self.epsilon:
            # 探索：随机选
            action = np.random.randint(0, self.a_dim)

            # 衰减 epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return action
        else:
            # 开发：选 Q 值最大的
            # 使用类似于之前的逻辑，处理多个最大值的情况
            state_q = self.Q[state, :]
            max_q = np.max(state_q)
            # 找到所有最大值的索引，随机选一个打破平局
            max_indices = np.where(state_q == max_q)[0]
            action = np.random.choice(max_indices)
            return action

    def update_Q(self, s, a, r, s_next):
        """
        标准的 Q-Learning 更新公式 (Bellman Equation)
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # 1. 获取当前 Q 值 (Predict)
        q_predict = self.Q[s, a]

        # 2. 计算目标 Q 值 (Target)
        # 即使 s_next 是终止状态，在连续优化中我们也通常认为它有价值
        q_target = r + self.gamma * np.max(self.Q[s_next, :])

        # 3. 更新
        self.Q[s, a] += self.lr * (q_target - q_predict)

    def save(self):
        save_dir = config.QLearning_RESULT_PATH
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "Q_table_standard.npy"), self.Q)

    def restore(self):
        try:
            path = os.path.join(config.QLearning_RESULT_PATH, 'Q_table_standard.npy')
            self.Q = np.load(path)
            print('Standard Q table loaded.')
        except FileNotFoundError:
            print("No saved model found.")