import gym
import src.utils.FBSUtil as FBSUtil
from src.algorithms.RL.Q_Learning import QLearningAgent, evaluate_policy
from loguru import logger
import copy

agent = QLearningAgent(s_dim=1, a_dim=5)
instance = "BA12-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)

state = env.reset()
for episode in range(10):

    done = False
    if not done:
        # 1. 选择动作
        best_action = agent.sequential_evaluate_actions(env, s=0)
        print("选择的动作：",best_action)
        # action = agent.select_action(0, deterministic=True)

        # 2. 执行动作并获取新解适应度
        next_state, reward, done, info = env.step(best_action)
        f_ns = env.fitness  # 假设环境提供新解适应度

        # 3. 按新公式更新 Q 值
        agent.update_Q(s=0, a=best_action, f_ns=f_ns)
        print(agent.Q)
        print(agent.times)

evaluate_policy(env, agent)
print(agent.Q)
print(agent.times)