import copy
import datetime
import math
import warnings
import gym
import numpy as np
from loguru import logger
import src.utils.ExperimentsUtil as ExperimentsUtil
from src.utils.PopulationOptimizer import PopulationOptimizer

warnings.filterwarnings("ignore", category=UserWarning)

class SA:
    def __init__(self, env, G=100, M=50, t_initial=10000.0, alpha=0.95, num_operators=5,
                 pop_size=50, pop_generations=100):
        self.env = env
        self.G = G
        self.M = M
        self.t = t_initial
        self.alpha = alpha
        self.num_operators = num_operators

        # 初始化种群优化器
        self.pop_optimizer = PopulationOptimizer(
            env=env,
            pop_size=pop_size,
            max_generations=pop_generations,
            k_coefficient=0.5
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

        # 初始化当前状态和全局最优
        s = copy.deepcopy(self.env)
        current_fitness = pop_best_fitness
        self.gBest = copy.deepcopy(pop_best_solution)
        self.best_fitness = pop_best_fitness
        fast_time = datetime.datetime.now()

        # 后续模拟退火流程保持不变
        for g in range(self.G):
            for m in range(self.M):
                op = np.random.randint(0, self.num_operators)
                s_prime = self._apply_operator(s, op)
                delta_f = s_prime.fitness - current_fitness

                if delta_f < 0 or np.random.rand() < math.exp(-delta_f / self.t):
                    s = s_prime
                    current_fitness = s.fitness

                    if current_fitness < self.best_fitness:
                        self.gBest = copy.deepcopy(s.fbs_model)
                        self.best_fitness = current_fitness
                        fast_time = datetime.datetime.now()
                        logger.debug(f"更新最优解: {self.best_fitness}")

            self.t *= self.alpha
            logger.debug(f"第{g + 1}/{self.G}代，当前温度: {self.t:.2f}，最优适应度: {self.best_fitness}")

        end_time = datetime.datetime.now()
        return (
            self.G * self.M,
            self.gBest,
            self.best_fitness,
            start_time,
            end_time,
            fast_time
        )

    # 其他方法保持不变
    def _apply_operator(self, s, action):
        new_s = copy.deepcopy(s)
        next_state, _, _, _ = new_s.step(action)
        return new_s

    def greedy_two_stage_search(self, s):
        improved = True
        max_attempts = 50
        while improved and max_attempts > 0:
            improved = False
            original_state = copy.deepcopy(s)
            _, _, _, info = s.step2(action=5)
            if info["current_fitness"] < original_state.fitness:
                improved = True
                break
            else:
                s = original_state
            max_attempts -= 1

        improved = True
        max_attempts = 20
        while improved and max_attempts > 0:
            improved = False
            original_state = copy.deepcopy(s)
            _, _, _, info = s.step2(action=6)
            if info["current_fitness"] < original_state.fitness:
                improved = True
            else:
                s = original_state
            max_attempts -= 1
        return s

    def _calculate_fitness(self, solution):
        return solution.mhc


if __name__ == "__main__":
    # 主程序部分保持不变
    exp_instance = "AB20-ar7"
    exp_algorithm = "SA_with_PopOpt"
    exp_remark = "SA整合种群优化初始解"
    exp_number = 30
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    max_iterations = 10000
    initial_temp = 10000.0
    alpha = 0.995

    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i + 1}次实验")
            try:
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                sa_solver = SA(
                    env,
                    t_initial=initial_temp,
                    alpha=alpha,
                    pop_size=50,
                    pop_generations=10
                )
                results = sa_solver.run()
                iteration, best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = results
                logger.info(f"最佳适应度: {best_fitness}")
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
                env.reset()
            except Exception as e:
                logger.error(f"实验 {i + 1} 失败: {e}")
    else:
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        sa_solver = SA(
            env,
            t_initial=initial_temp,
            alpha=alpha,
            pop_size=50,
            pop_generations=10
        )
        results = sa_solver.run()
        iteration, best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = results
        print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution)
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
        env.render()
