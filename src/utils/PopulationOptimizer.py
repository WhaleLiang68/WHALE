from typing import List, Tuple, Optional
import numpy as np
import src.utils.FBSUtil as FBSUtil
from src.utils.DataExtractor import DataProcessingEnv
from src.utils.FBSModel import FBSModel


class PopulationOptimizer:
    """种群优化器，用于在强化学习前生成优质初始解"""

    def __init__(self,
                 env,  # DataProcessingEnv实例
                 pop_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 max_generations: int = 100,
                 k_coefficient: float = 0.5):  # 新增k系数参数
        self.env = env
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.k_coefficient = k_coefficient
        self.population: List[FBSModel] = []
        self.best_fitness = float('inf')  # 适应度越小越优
        self.best_individual: Optional[FBSModel] = None

    def initialize_population(self) -> None:
        """初始化种群（生成多个初始布局方案）"""
        self.population = []
        for _ in range(self.pop_size):
            # ----------基因编码方式-------------------
            B = FBSUtil.select_B(self.env.areas, self.env.n, self.env.fac_limit_aspect, self.env.W)
            if B is None:
                # 处理没有可行B的情况，使用默认值
                B = 2
                print(f"没有找到可行的区带总数，使用默认值 B={B}")
            genes, permutation = FBSUtil.ZGeneCoding.generate_genes(self.env.n, B)
            bay_list, bay = FBSUtil.ZGeneCoding.decode_genes(genes, permutation)
            permutation, bay = FBSUtil.arrayToPermutation(bay_list)

            # ---------------------------------------
            # # 调用环境中的初始解生成方法
            # permutation, bay = FBSUtil.binary_solution_generator(
            #     self.env.areas,
            #     self.env.n,
            #     self.env.fac_limit_aspect,
            #     self.env.W
            # )
            bay[-1] = 1  # 确保bay的最后一个元素为1
            self.population.append(FBSModel(
                permutation.astype(int).tolist(),
                bay.astype(int).tolist(),
                genes=genes.tolist() if isinstance(genes, np.ndarray) else genes  # 确保是列表类型
            ))

    def evaluate_individual(self, individual: FBSModel) -> float:
        """评估个体适应度（复用环境的状态计算逻辑）"""
        # 临时计算当前个体的适应度
        _, _, _, _, _, _, _, _, fitness = FBSUtil.StatusUpdatingDevice(  
            individual,
            self.env.areas,
            self.env.H,
            self.env.F,
            self.env.fac_limit_aspect
        )
        return fitness

    def select_parents(self, selection_type: str = "roulette") -> List[FBSModel]:
        """选择算子（支持锦标赛选择和轮盘赌选择）"""
        if selection_type == "tournament":
            # 原锦标赛选择逻辑
            selected = []
            for _ in range(self.pop_size):
                indices = np.random.choice(len(self.population), 3, replace=False)
                candidates = [self.population[i] for i in indices]
                # candidates = np.random.choice(self.population, 3, replace=False)
                best_candidate = min(candidates, key=lambda x: self.evaluate_individual(x))
                selected.append(best_candidate)
            return selected
        elif selection_type == "roulette":
            # 轮盘赌选择（适应度越小权重越高，需做映射）
            fitnesses = np.asarray([self.evaluate_individual(ind) for ind in self.population], dtype=float)
            finite_mask = np.isfinite(fitnesses)
            if not np.any(finite_mask):
                indices = np.random.choice(len(self.population), self.pop_size, replace=True)
                return [self.population[i] for i in indices]
            max_fitness = np.max(fitnesses[finite_mask])
            weights = np.where(finite_mask, max_fitness - fitnesses + 1e-6, 1e-6)
            weight_sum = float(np.sum(weights))
            if (not np.isfinite(weight_sum)) or weight_sum <= 0:
                indices = np.random.choice(len(self.population), self.pop_size, replace=True)
                return [self.population[i] for i in indices]
            weights = weights / weight_sum
            indices = np.random.choice(len(self.population), self.pop_size, p=weights, replace=True)
            # 根据索引获取选中的个体
            selected = [self.population[i] for i in indices]
            # selected = np.random.choice(self.population, self.pop_size, p=weights)
            # return selected.tolist()
            return selected
        else:
            raise ValueError(f"不支持的选择类型: {selection_type}")

    def crossover(self, parent1: FBSModel, parent2: FBSModel) -> Tuple[FBSModel, FBSModel]:
        """实数编码交叉算子，随机选择单点交叉或多点交叉
           对选定设备位置进行算数交叉"""
        if (np.random.rand() > self.crossover_rate or
                parent1.genes is None or parent2.genes is None):
            return parent1, parent2

        # 将基因转换为numpy数组便于操作
        genes1 = np.array(parent1.genes, dtype=np.float64)
        genes2 = np.array(parent2.genes, dtype=np.float64)
        n = len(genes1)

        # 随机选择交叉类型（50%概率单点交叉，50%概率多点交叉）
        crossover_type = "single_point" if np.random.rand() < 0.5 else "multi_point"

        # 复制父代基因作为子代基础
        child1_genes = genes1.copy()
        child2_genes = genes2.copy()

        # 选择要交叉的设备位置
        if crossover_type == "single_point":
            # 单点交叉：随机选择一个设备位置进行算数交叉
            point = np.random.randint(n)  # 选择要交叉的设备索引
            alpha = np.random.rand()  # 随机权重因子

            # 算数交叉：child = alpha * parent1 + (1-alpha) * parent2
            child1_genes[point] = alpha * genes1[point] + (1 - alpha) * genes2[point]
            child2_genes[point] = alpha * genes2[point] + (1 - alpha) * genes1[point]

        else:  # multi_point
            # 多点交叉：随机选择多个设备位置进行算数交叉
            # 选择10%-30%的位置进行交叉
            num_points = max(1, int(np.random.uniform(0.1, 0.3) * n))
            points = np.random.choice(range(n), num_points, replace=False)

            # 为每个交叉点生成独立的权重因子
            alphas = np.random.rand(num_points)

            # 对每个选中的设备位置进行算数交叉
            for i, point in enumerate(points):
                alpha = alphas[i]
                child1_genes[point] = alpha * genes1[point] + (1 - alpha) * genes2[point]
                child2_genes[point] = alpha * genes2[point] + (1 - alpha) * genes1[point]

        # 解码逻辑保持不变
        def decode(child_genes, parent_perm):
            parent_perm_np = np.array(parent_perm)
            bay_list, bay = FBSUtil.ZGeneCoding.decode_genes(child_genes, parent_perm_np)
            permutation, _ = FBSUtil.arrayToPermutation(bay_list)
            return permutation, bay

        c1_perm, c1_bay = decode(child1_genes, parent1.permutation)
        c2_perm, c2_bay = decode(child2_genes, parent2.permutation)

        return (
            FBSModel(c1_perm.tolist(), c1_bay.tolist(), child1_genes.tolist()),
            FBSModel(c2_perm.tolist(), c2_bay.tolist(), child2_genes.tolist())
        )


    def mutate(self, individual: FBSModel) -> FBSModel:
        if np.random.rand() > self.mutation_rate or individual.genes is None:
            return individual

        # 从基因中推断区带总数B
        B = int(np.max(np.floor(individual.genes)))
        genes = np.array(individual.genes, dtype=np.float64)
        new_genes = genes.copy()

        # 随机选择一个基因进行变异
        mutate_index = np.random.randint(len(genes))
        z_i = genes[mutate_index]

        # 生成标准正态分布的随机数u
        u = np.random.normal(0, 1)
        ku = self.k_coefficient * u
        tanh_ku = np.tanh(ku)

        # 根据tanh(ku)的符号应用不同的变异公式
        if tanh_ku >= 0:
            z_i_prime = z_i + (B + 1 - z_i) * tanh_ku
        else:
            z_i_prime = z_i + (z_i - 1) * tanh_ku

        # 确保变异后的值在区间(1, B+1)内
        z_i_prime = max(1.0001, min(z_i_prime, B + 0.9999))
        new_genes[mutate_index] = z_i_prime

        # 重新解码得到新的布局信息
        permutation_np = np.array(individual.permutation)
        bay_list, bay = FBSUtil.ZGeneCoding.decode_genes(new_genes, permutation_np)
        permutation, _ = FBSUtil.arrayToPermutation(bay_list)

        return FBSModel(permutation.tolist(), bay.tolist(), new_genes.tolist())

    def evolve(self, selection_type: str = "roulette") -> None:
        """执行一代进化"""
        # 选择
        parents = self.select_parents(selection_type)
        # 交叉
        offspring = []
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < self.pop_size else parents[0]
            c1, c2 = self.crossover(p1, p2)
            offspring.extend([c1, c2])
        # 变异
        self.population = [self.mutate(ind) for ind in offspring[:self.pop_size]]

    def optimize(self) -> FBSModel:
        """执行种群优化并返回最优解"""
        self.initialize_population()

        # # 初始化种群后可视化第一个个体
        # if self.population:
        #     # 将个体传递给环境，用于生成可视化所需数据
        #     self.env.reset(fbs_model=self.population[0])
        #     # 调用环境的render方法进行可视化
        #     self.env.render()

        for gen in range(self.max_generations):
            # 评估当前种群
            current_fitnesses = [self.evaluate_individual(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitnesses)
            current_best = self.population[current_best_idx]
            current_best_fitness = current_fitnesses[current_best_idx]

            # 更新全局最优
            if self.best_individual is None or current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best
                # 确保bay的最后一个元素为1
                if self.best_individual.bay[-1] != 1:
                    self.best_individual.bay[-1] = 1

            # 打印进化信息
            if gen % 10 == 0:
                print(f"Generation {gen}, Best Fitness: {self.best_fitness:.2f}")

            # 进化
            self.evolve()

        return self.best_individual


# 使用示例：种群优化 -> 强化学习初始化
if __name__ == "__main__":
    # 1. 初始化环境

    env = DataProcessingEnv(
        instance="AB20-ar3",  # 替换为实际实例名
        seed=42
    )

    # 2. 执行种群优化
    optimizer = PopulationOptimizer(
        env=env,
        pop_size=50,
        max_generations=100,
        k_coefficient=0.5
    )
    best_initial_layout = optimizer.optimize()
    print(f"种群优化完成，最优初始适应度: {optimizer.best_fitness:.2f}")


    # 3. 用种群优化的最优解初始化强化学习环境
    env.reset(fbs_model=best_initial_layout)
    print("??????????????")
    env.render()

    # 4. 后续强化学习训练逻辑（示例）
    # for episode in range(1000):
    #     state = env.reset(fbs_model=best_initial_layout)  # 每次都用优化后的解初始化
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action = agent.choose_action(state)
    #         next_state, reward, done, info = env.step(action)
    #         agent.learn(state, action, reward, next_state)
    #         state = next_state
    #         total_reward += reward
    #     print(f"Episode {episode}, Total Reward: {total_reward}")



#
# if __name__ == "__main__":
#     # 实验参数
#     exp_instance = "AB20-ar7"
#     exp_algorithm = "PopulationOptimizer"
#     exp_remark = "+种群优化"
#     exp_number = 10
#     is_exp = True
#
#     # 算法参数（修正参数名称以匹配PopulationOptimizer构造函数）
#     max_iterations = 10000  # 保持该变量名，用于传入正确参数
#     pop_size = 50
#     crossover_rate = 0.8
#     mutation_rate = 0.1
#     global_best_fitness = float('inf')
#     global_best_env = None
#
#     if is_exp:
#         for i in range(exp_number):
#             logger.info(f"第{i + 1}次实验")
#             try:
#                 env = gym.make("FbsEnv-v0", instance=exp_instance)
#
#                 # 修正参数名称：将max_iter改为max_iterations（根据实际构造函数调整）
#                 optimizer = PopulationOptimizer(
#                     env=env,
#                     pop_size=pop_size,
#                     crossover_rate=crossover_rate,
#                     mutation_rate=mutation_rate
#                 )
#
#                 # 运行优化算法（保持其他部分不变）
#                 iteration, best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = optimizer.optimize()
#                 best_Env = env
#
#                 print("优化完成")
#                 print(f"最佳解: {best_solution.array_2d}, 最佳适应度: {best_fitness}")
#
#                 if best_fitness < global_best_fitness:
#                     global_best_fitness = best_fitness
#                     global_best_env = {
#                         'env': best_Env,
#                         'instance': exp_instance,
#                         'algorithm': exp_algorithm,
#                         'solution': best_solution.array_2d,
#                         'fitness': best_fitness,
#                         'iteration': iteration,
#                         'time_info': (exp_start_time, exp_fast_time, exp_end_time)
#                     }
#
#                 from datetime import timedelta  # 导入timedelta
#
#                 exp_duration = exp_end_time - exp_start_time  # 总耗时（秒）
#                 exp_fast_duration = exp_fast_time - exp_start_time  # 首次最优耗时（秒）
#
#                 # 转换为timedelta类型
#                 exp_start_timedelta = timedelta(seconds=0)  # 开始时间差为0
#                 exp_fast_timedelta = timedelta(seconds=exp_fast_duration)
#                 exp_end_timedelta = timedelta(seconds=exp_duration)
#
#                 # 修正3：传入转换后的timedelta参数
#                 ExperimentsUtil.save_experiment_result(
#                     exp_instance=exp_instance,
#                     exp_algorithm=exp_algorithm,
#                     exp_iterations=iteration,
#                     exp_solution=best_solution.array_2d,
#                     exp_fitness=best_fitness,
#                     exp_start_time=exp_start_timedelta,  # 改为timedelta
#                     exp_fast_time=exp_fast_timedelta,  # 改为timedelta
#                     exp_end_time=exp_end_timedelta,  # 改为timedelta
#                     exp_remark=exp_remark
#                 )
#
#                 # print(f"当前环境状态: {env.state}")
#                 # print(f"当前解: {env.fbs_model.permutation, env.fbs_model.bay}, 适应度: {env.fitness}")
#
#             except Exception as e:
#                 logger.error(f"实验 {i + 1} 失败！错误信息: {e}", exc_info=True)

        # if global_best_env:
        #     best_env = global_best_env['env']
        #     best_env.render()
