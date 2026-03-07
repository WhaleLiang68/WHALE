import copy
import datetime
import math
import warnings
import gym
import numpy as np
from loguru import logger
import src.utils.ExperimentsUtil as ExperimentsUtil
from src.algorithms.RL.Q_Learning import QLearningAgent
from src.utils.PopulationOptimizer import PopulationOptimizer

warnings.filterwarnings("ignore", category=UserWarning)

class Q_LearningSA:
    def __init__(self, env, G=100, M=50, t_initial=10000.0, alpha=0.95, num_operators=5,
                 pop_size=50, pop_generations=100):
        """
        参数:
        env: 自定义环境对象，需实现 reset(), step(), fitness 属性
        G: 外层循环次数
        M: 内层循环次数
        t_initial: 初始温度
        alpha: 温度衰减系数
        num_operators: 操作符数量（如交换、插入等邻域操作）
        """
        self.env = env
        self.G = G
        self.M = M
        self.t = t_initial
        self.alpha = alpha
        self.num_operators = num_operators
        self.best_Env=env
        self.num_operators = num_operators

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

        # 记录最优解
        self.gBest = None
        self.best_fitness = np.inf

    def run(self):
        """执行主优化流程（整合种群优化）"""
        start_time = datetime.datetime.now()
        fast_time = start_time

        # 使用种群优化生成初始解 - 修改此处以适应新的返回值格式
        logger.info("开始种群优化生成初始解...")
        # 修正：只接收最优解对象，然后单独计算其适应度
        pop_best_solution = self.pop_optimizer.optimize()  # 现在只返回FBSModel对象

        # 单独计算种群优化结果的适应度
        self.env.reset(fbs_model=pop_best_solution)
        pop_best_fitness = self.env.fitness  # 从环境中获取适应度值
        logger.info(f"种群优化完成，初始最优适应度: {pop_best_fitness}")

        s = copy.deepcopy(self.env)
        current_fitness = self.env.fitness

        for g in range(self.G):
            agent = QLearningAgent(s_dim=1, a_dim=5)
            for m in range(self.M):
                # 1. 选择 Q 值最大的操作符（伪代码第3行）
                # agent = QLearningAgent(s_dim=1, a_dim=5)
                op = agent.sequential_evaluate_actions(env, s=0)
                # print("选择的动作：",op)
                # op = self.q_agent.select_action(s=0, deterministic=True)

                # 2. 生成新解（伪代码第4行）
                s_prime = self._apply_operator(s, op)
                # s_prime = self.env
                delta_f = s_prime.fitness - current_fitness

                # 3. Metropolis 准则接受判断（伪代码第6-7行）
                if delta_f < 0 or np.random.rand() < math.exp(-delta_f / self.t):
                    s = s_prime
                    current_fitness = s.fitness

                    # 4. 贪婪两阶段局部搜索（伪代码第8行）
                    s = self.greedy_two_stage_search(s)

                    # 5. 更新 Q 表（伪代码第5行，Equation 10）
                    agent.update_Q(s=0, a=op, f_ns=current_fitness)
                    # reward = -delta_f  # 假设奖励为适应度改进量
                    # self.q_agent.train(s=0, a=op, r=reward, s_next=0, dw=False)

                    # # 6. 更新全局最优解（伪代码第10行）
                    # if current_fitness < self.best_fitness:
                    #     self.gBest = copy.deepcopy(s.fbs_model)
                    #     self.best_fitness = current_fitness
                    #     fast_time = datetime.datetime.now()
                    self.best_Env=s
                    self.gBest = copy.deepcopy(s.fbs_model)
                    self.best_fitness = current_fitness
                    fast_time = datetime.datetime.now()

            # 7. 降温（伪代码第11行）
            self.t *= self.alpha

        end_time = datetime.datetime.now()
        return (
            self.G * self.M,
            self.gBest,
            self.best_fitness,
            start_time,
            end_time,
            fast_time,
            self.best_Env
        )

    # ------------------------ 关键子方法 ------------------------
    def _apply_operator(self, s, action):
        """应用操作符生成新解（需根据实际需求实现）"""
        new_s = copy.deepcopy(s)
        next_state, _, _, _ = new_s.step(action)
        # new_s.reset(fbs_model = new_s.fbs_model)
        # 其他操作符...
        # new_s.fitness = self._calculate_fitness(new_s)  # 需实现适应度计算
        # print(s.fbs_model.permutation)
        # print(s.fbs_model.bay)
        # print(new_s.fbs_model.permutation)
        # print(new_s.fbs_model.bay)
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

    def _calculate_fitness(self, solution):
        """计算解的适应度（需根据实际需求实现）"""
        # 示例：假设 solution 有 mhc 属性
        return solution.mhc




if __name__ == "__main__":
    # 实验参数
    exp_instance = "AB20-ar7"
    exp_algorithm = "LSA_with_PopOpt"
    exp_remark = "LSA整合种群优化初始解"
    exp_number = 60
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 算法参数
    max_iterations = 10000
    initial_temp = 10000.0
    alpha = 0.995
    global_best_fitness = float('inf')
    global_best_env = None


    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i+1}次实验")
            try:
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                sa_solver = Q_LearningSA(env, t_initial=initial_temp, alpha=alpha)
                iteration,best_solution, best_fitness,exp_start_time,exp_end_time,exp_fast_time,best_Env = sa_solver.run()
                print("ok")
                print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
                # 更新全局最佳env
                if best_fitness < global_best_fitness:  # 最小化问题，若最大化则改为 >
                    global_best_fitness = best_fitness
                    global_best_env = {
                        'env': best_Env,  # 保存env对象
                        'instance': exp_instance,
                        'algorithm': exp_algorithm,
                        'solution': best_solution.array_2d,
                        'fitness': best_fitness,
                        'iteration': iteration,
                        'time_info': (exp_start_time, exp_fast_time, exp_end_time)
                    }
                # 保存结果
                ExperimentsUtil.save_experiment_result(
                    exp_instance=f"{exp_instance}_{current_date}",
                    exp_algorithm=exp_algorithm,
                    exp_iterations=iteration,
                    exp_solution=best_solution.array_2d,
                    exp_fitness=best_fitness,
                    exp_start_time=exp_start_time,
                    exp_fast_time=exp_fast_time,
                    exp_end_time=exp_end_time,
                    exp_remark=exp_remark
                )
                # env.reset()
                print(env.state)
                print(f"Solution: {env.fbs_model.permutation, env.fbs_model.bay}, Fitness: {env.fitness}")
                # env.render()
            except Exception as e:
                logger.error(f"实验 {i + 1} 失败！错误信息: {e}")
        best_env = global_best_env['env']
        best_env.render()
        # img_array=best_env.render(mode='rgb_array')
        # # 保存为图片文件
        # img = Image.fromarray(img_array)
        # img.save('environment_state.png')

        # best_env.save_render('global_best.png')

    else:
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        iteration,best_solution, best_fitness,exp_start_time,exp_end_time,exp_fast_time = Q_LearningSA(env, t_initial=initial_temp, alpha=alpha)
        # self.G * self.M,
        # self.gBest,
        # self.best_fitness,
        # start_time,
        # end_time,
        # fast_time
        print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution)
        env.render()