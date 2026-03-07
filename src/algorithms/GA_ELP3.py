import numpy as np
import copy
import datetime
import gym
from loguru import logger
import src.utils.FBSUtil as FBSUtil
from src.utils.FBSModel import FBSModel
import src.utils.ExperimentsUtil as ExperimentsUtil
import math  # <-- ELP2 所需的导入

# =============================================================================
# ELP 算法 (从 ELP2.py 移植而来)
#
# =============================================================================
class ELP:
    def __init__(self, env, gbest_model, T, Q_matrix, G=100, t_max=50, k=0.1):
        """
        ELP算法初始化 (基于 ELP2.py)
        """
        self.env = env
        self.gbest_model = copy.deepcopy(gbest_model)
        self.s_model = copy.deepcopy(gbest_model)
        self.T = T
        self.Q_matrix = Q_matrix
        self.G = G
        self.t_max = t_max
        self.k = k
        self.current_energy = self._calculate_energy(self.s_model)
        self.best_energy = self.current_energy
        self.energy_histogram = {}

    def _calculate_energy(self, solution_model):
        """
        计算解(FBSModel)的能量E(s) (来自 ELP2.py)
        """
        try:
            _, info = self.env.reset(fbs_model=solution_model)
            if 'fitness' in info:
                return info['fitness']
            else:
                self.env.step(3) # 假设 3 是 "idle" 动作
                return self.env.fitness
        except Exception as e:
            logger.error(f"ELP: 计算能量时出错: {e}. 返回无穷大。")
            return float('inf')

    def _calculate_H(self, current_E, t):
        """计算H(E(s),t)函数 (来自 ELP2.py)"""
        initial_disturbance = 10.0
        return initial_disturbance * (1 - t / self.t_max)

    def _generate_new_solution_model(self, s_model):
        """
        基于当前解模型 s_model 产生新解模型 (来自 ELP2.py)
        """
        self.env.reset(fbs_model=s_model)
        op = np.random.randint(0, 5) # 假设有 0-4 共5个动作
        self.env.step(op)
        new_s_model = copy.deepcopy(self.env.fbs_model)
        return new_s_model

    def _check_aspect_ratio_constraint(self, solution_model):
        """检查解的宽高比约束 (来自 ELP2.py)"""
        try:
            fac_x, fac_y, fac_b, fac_h = FBSUtil.getCoordinates_mao(
                solution_model, self.env.areas, self.env.H
            )
            fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
            if np.any(fac_aspect_ratio > self.env.fac_limit_aspect):
                return False
            return True
        except Exception:
            return False

    def run(self):
        """执行ELP算法主流程 (来自 ELP2.py)"""
        start_time = datetime.datetime.now()
        fast_time = start_time
        g = 0
        while g < self.G:
            t = 0
            while t < self.t_max:
                s_model_prime = self._generate_new_solution_model(self.s_model)
                E_s = self.current_energy
                E_s_prime = self._calculate_energy(s_model_prime)
                key_E_s = round(E_s, 2)
                key_E_s_prime = round(E_s_prime, 2)
                H_val = self._calculate_H(E_s, t)
                E_prime_s = E_s + self.k * H_val
                E_prime_s_prime = E_s_prime + self.k * H_val
                delta_E_prime = E_prime_s_prime - E_prime_s
                r = np.random.rand()
                k_boltzmann = 1.380649e-23
                if delta_E_prime < 0 or (self.T > 0 and k_boltzmann > 0 and r < math.exp(-delta_E_prime / (self.T * k_boltzmann))):
                    self.s_model = copy.deepcopy(s_model_prime)
                    self.current_energy = E_s_prime
                    self.energy_histogram[key_E_s_prime] = self.energy_histogram.get(key_E_s_prime, 0) + 1
                    if self.current_energy < self.best_energy:
                        self.gbest_model = copy.deepcopy(self.s_model)
                        self.best_energy = self.current_energy
                        fast_time = datetime.datetime.now()
                else:
                    self.energy_histogram[key_E_s] = self.energy_histogram.get(key_E_s, 0) + 1
                t += 1
            g += 1
            self.T *= 0.995

        end_time = datetime.datetime.now()
        is_valid = self._check_aspect_ratio_constraint(self.gbest_model)
        total_iterations = self.G * self.t_max
        
        # 返回 ELP 的完整输出
        return (
            total_iterations,
            is_valid,
            self.gbest_model,
            self.best_energy,
            start_time,
            end_time,
            fast_time
        )
# =============================================================================
# (ELP 类结束)
# =============================================================================


# =============================================================================
# 种群优化器 (GA) - 已修改以集成 ELP
#
# =============================================================================
class PopulationOptimizer:
    def __init__(self, env, population_size, 
                 crossover_rate, mutation_rate, elite_rate,
                 elp_g, elp_t, elp_t_initial, elp_k):
        
        self.env = env
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        
        # --- 新增: ELP 算法的超参数 ---
        self.elp_g = elp_g                # ELP 的外循环 (G)
        self.elp_t = elp_t                # ELP 的内循环 (t_max)
        self.elp_t_initial = elp_t_initial  # ELP 的初始温度 (T)
        self.elp_k = elp_k                # ELP 的系数 (k)
        
        self.population = []
        self.fitnesses = []
        self.best_solution = None # 存储最优的 FBSModel
        self.best_fitness = float('inf')

    def _initialize_population(self):
        #
        self.population = []
        self.fitnesses = []
        for _ in range(self.population_size):
            _, info = self.env.reset()
            individual = copy.deepcopy(self.env.fbs_model)
            fitness = info['fitness']
            self.population.append(individual)
            self.fitnesses.append(fitness)
        
        # 初始化最优解
        self.best_fitness = np.min(self.fitnesses)
        self.best_solution = self.population[np.argmin(self.fitnesses)]

    def _selection(self, population, fitnesses, k=3):
        # 锦标赛选择
        selected_indices = np.random.choice(len(population), k, replace=False)
        selected_fitnesses = [fitnesses[i] for i in selected_indices]
        best_index_in_tournament = selected_indices[np.argmin(selected_fitnesses)]
        return population[best_index_in_tournament]

    def _crossover(self, parent1, parent2):
        # 顺序交叉
        offspring1, offspring2 = FBSUtil.FBSUtils.CrossoverActions.order_crossover(parent1, parent2)
        return offspring1, offspring2

    def _mutation(self, individual):
        # 设施交换变异
        perm, bay = FBSUtil.facility_swap(np.array(individual.permutation), np.array(individual.bay))
        individual.permutation = perm.tolist()
        individual.bay = bay.tolist()
        return individual

    def _evaluate(self, individual):
        # 评估个体适应度
        _, info = self.env.reset(fbs_model=individual)
        return info['fitness']

    def optimize(self, generations):
        """
        运行混合 GA-ELP 算法。
        (此方法已被重写)
        """
        start_time = datetime.datetime.now()
        fast_time = start_time
        
        self._initialize_population()
        
        logger.info(f"Gen 0 (Init), Best Fitness: {self.best_fitness}")

        for gen in range(generations):
            # 评估当前种群 (确保 fitnesses 是最新的)
            self.fitnesses = [self._evaluate(ind) for ind in self.population]

            # 1. 精英保留 (Elitism)
            sorted_indices = np.argsort(self.fitnesses)
            num_elites = int(self.population_size * self.elite_rate)
            elites = [self.population[i] for i in sorted_indices[:num_elites]]

            # --- 【【【 2. 局部搜索 (Memetic Step) 】】】 ---
            # 对精英个体应用 ELP 算法
            improved_elites = []
            for elite_model in elites:
                # 实例化 ELP 求解器
                elp_solver = ELP(
                    env=self.env,
                    gbest_model=elite_model,
                    T=self.elp_t_initial,
                    Q_matrix=np.zeros((1, 5)), # 必需参数 (来自 ELP2.py)
                    G=self.elp_g,
                    t_max=self.elp_t,
                    k=self.elp_k
                )
                
                # 运行 ELP 并获取优化后的解
                # ELP.run() 返回 (iters, is_valid, gbest_model, best_energy, ...)
                _, _, improved_model, improved_fitness, _, _, _ = elp_solver.run()
                
                improved_elites.append(improved_model)
            
            # 新种群首先由优化后的精英组成
            new_population = improved_elites
            
            # --- 3. GA 算子 (Global Search) ---
            #
            # 填充剩余的种群
            num_offspring = self.population_size - num_elites
            
            for _ in range(num_offspring // 2):
                # 锦标赛选择
                parent1 = self._selection(self.population, self.fitnesses, k=3)
                parent2 = self._selection(self.population, self.fitnesses, k=3)
                
                # 交叉
                if np.random.rand() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # 变异
                if np.random.rand() < self.mutation_rate:
                    offspring1 = self._mutation(offspring1)
                if np.random.rand() < self.mutation_rate:
                    offspring2 = self._mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # 更新种群
            self.population = new_population[:self.population_size] # 处理奇数种群
            
            # 评估新种群并更新全局最优
            self.fitnesses = [self._evaluate(ind) for ind in self.population]
            current_best_fitness = np.min(self.fitnesses)
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitnesses)]
                fast_time = datetime.datetime.now()
            
            logger.info(f"Gen {gen+1}/{generations}, Best Fitness: {self.best_fitness}")
        
        end_time = datetime.datetime.now()
        
        # 返回最优解 (FBSModel) 和统计信息
        return self.best_solution, self.best_fitness, start_time, end_time, fast_time


# =============================================================================
# 主程序 (用于测试混合算法)
# =============================================================================
if __name__ == "__main__":
    exp_instance = "AB20-ar3"
    exp_algorithm = "Hybrid_GA_ELP"
    exp_remark = "PopulationOptimizer + ELP2 local search"
    exp_number = 30 
    is_exp = True
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i + 1}次实验（{exp_algorithm}算法）")
            try:
                # 1. 初始化环境
                env = gym.make("FbsEnv-v0", instance=exp_instance)
                env.reset()

                # 2. 实例化混合优化器
                hybrid_solver = PopulationOptimizer(
                    env=env,
                    population_size=50,
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    elite_rate=0.1,         # 10% 的精英
                    elp_g=10,                 # ELP 外循环
                    elp_t=5,                  # ELP 内循环
                    elp_t_initial=1000.0,     # ELP 初始温度
                    elp_k=0.1                 # ELP 系数
                )

                # 3. 运行算法
                generations = 50 # GA 的总代数
                best_sol_model, best_fitness, start, end, fast = hybrid_solver.optimize(generations)

                # 4. 输出结果
                logger.info(f"第{i + 1}次实验完成 | 最优适应度: {best_fitness}")
                print(f"Best Solution: {best_sol_model.array_2d}, Best Fitness: {best_fitness}")

                # 5. 保存实验结果
                if best_sol_model:
                    solution_list_of_lists = [arr.tolist() for arr in best_sol_model.array_2d]
                    
                    # 检查最终解的宽高比
                    _, final_info = env.reset(fbs_model=best_sol_model)
                    is_valid = env.fac_aspect_ratio.max() <= env.fac_limit_aspect

                    ExperimentsUtil.save_experiment_result(
                        exp_instance=f"{exp_instance}_{current_date}",
                        exp_algorithm=exp_algorithm,
                        exp_iterations=generations * hybrid_solver.population_size, # 总评估次数
                        exp_solution=solution_list_of_lists,
                        exp_fitness=best_fitness,
                        exp_start_time=start,
                        exp_fast_time=fast,
                        exp_end_time=end,
                        exp_is_valid_aspect_ratio=is_valid,
                        exp_remark=exp_remark
                    )

            except Exception as e:
                logger.error(f"第{i + 1}次实验失败！错误信息: {e}")
                import traceback
                traceback.print_exc()
    else:
        logger.info("单次运行模式 (is_exp=False)。")
        # (此处可以添加单次运行的代码)