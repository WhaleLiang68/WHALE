import sys
import pickle
import uuid
import gym
from gym import spaces
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from loguru import logger
from typing import Optional, Dict, Any
import src.utils.config as config
from src.utils.FBSModel import FBSModel
import src.utils.FBSUtil as FBSUtil
# from src.utils.warnings_config import setup_warnings

# # 设置警告过滤器
# setup_warnings()

# 设置日志处理级别
logger.remove()
logger.add(
    sys.stderr,
    level="INFO"
)
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 继承 gym.Env 并整合 DataExtractor
class DataProcessingEnv(gym.Env):
    def __init__(self, instance=None, seed=None, options=None):
        super(DataProcessingEnv, self).__init__()  # 调用父类初始化

        # with open(config.FILE_PATH, "rb") as file:
        #     data = pickle.load(file)
        #     # 打印数据结构
        #     print("Loaded data types:", [type(d) for d in data])
        #     print("FlowMatrices keys:", data[1].keys() if isinstance(data[1], dict) else "Not a dict")
        #     (
        #         self.problems,
        #         self.FlowMatrices,
        #         self.sizes,
        #         self.LayoutWidths,
        #         self.LayoutLengths,
        #     ) = data

        # 从 .pkl 文件中加载预定义的设施布局问题数据
        with open(
                config.FILE_PATH,
                "rb",
        ) as file:
            (
                self.problems,
                self.FlowMatrices,
                self.sizes,
                self.LayoutWidths,
                self.LayoutLengths,
            ) = pickle.load(file)
        # 初始化实例，环境初始化过程中的基础操作
        self.instance = instance
        # 检查 instance 是否在数据中
        if self.instance not in self.problems:
            print(f"Error: Instance {self.instance} not found in problems data.")
        # 检查实例是否存在
        if instance not in self.FlowMatrices or instance not in self.problems:
            valid_instances = list(self.FlowMatrices.keys())
            raise ValueError(f"实例 '{instance}' 不存在。可用实例: {valid_instances}")
        self.uuid = uuid.uuid4()
        # 获取问题模型的设施数量
        # self.F = self.FlowMatrices[self.instance]  # 物流强度矩阵
        raw_F = self.FlowMatrices[self.instance]
        # 将上三角矩阵转换为对称矩阵：F = F + F.T
        # 注意：这里假设对角线为0（设施对自己没有物流）。如果对角线不为0，需要减去一次对角线。
        self.F = raw_F + raw_F.T
        # print("检查流矩阵对称性：")
        # print(self.F)
        self.n = self.problems[self.instance]  # 问题模型的设施数量
        self.areas,self.fac_limit_aspect = (
            FBSUtil.getAreaData(self.sizes[self.instance])
        )  # 面积，横纵比
        logger.debug(f"横纵比: {self.fac_limit_aspect}")
        logger.debug(f"面积: {self.areas}")
        self.H = self.LayoutWidths[self.instance]  # 厂房的长度
        self.W = self.LayoutLengths[self.instance]  # 厂房的宽度
        # self.fbs_model = None
        total_area = np.sum(self.areas)  # 设施的总面积
        self.actions = { # 一个字典，定义了动作编号与动作名称的映射
            0: "facility_swap",
            1: "bay_flip",
            2: "bay_swap",
            3: "repair",
            4: "idle",#什么也不做
            5: "facility_insert",
            6: "bay_shuffle",
            7: "facility_shuffle",
            8: "ga_action"
        }  # 动作空间
        self.action_space = spaces.Discrete(len(self.actions))  # 动作空间，创建一个离散空间，表示智能体可以选择的动作编号为 0 到 4
        # spaces.Box 表示连续的多维状态空间/low=0, high=255：状态中每个元素的最小值和最大值（类似 RGB 颜色的取值范围）
        """""
        shape = (self.n * 3,)：
        self.n是设施数量。
        每个设施用3个值表示（例如：坐标(x, y)和成本cost，或其他特征）。
        因此总维度为n * 3。
        """""
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.n * 3,), dtype=np.float64)  # 状态空间
        self.fitness = np.inf
        self.best_fitness = np.inf
        # 惩罚指数控制：从1逐步升至3
        self.penalty_k_min = 1.0
        self.penalty_k_max = 5.0
        self.penalty_k_growth = 1.001
        self.penalty_k = self.penalty_k_min

        # ------------------调试信息------------------
        logger.debug("-------------------init初始化信息------------------")
        logger.debug(f"实例: {self.instance}")
        logger.debug(f"设施数量: {self.n}")
        logger.debug(f"设施信息: {self.sizes[self.instance]}")
        logger.debug(f"设施面积: {self.areas}")
        logger.debug(f"设施横纵比: {self.fac_limit_aspect}")
        logger.debug(f"设施总长度H: {self.H}")
        logger.debug(f"设施总宽度W: {self.W}")
        logger.debug("--------------------------------------------------")

    def __getstate__(self):
       """
       为 deepcopy 和 pickle 提供指导，告诉它们哪些状态需要保存。
       我们返回包含所有可安全复制属性的字典。
       """
       # self.__dict__ 包含了类的所有实例属性
       return self.__dict__

    def __setstate__(self, state):
       """
       为 deepcopy 和 pickle 提供指导，告诉它们如何根据状态重建对象。
       """
       # 将保存的状态字典更新回实例的 __dict__ 中
       self.__dict__.update(state)

   # 将环境重置到初始状态，为新一轮训练或测试做准备
    def reset(self, seed=None, options=None, fbs_model: FBSModel = None):
        if seed is not None:
            np.random.seed(seed)  # 设置随机种子
        if options is not None and "fbs_model" in options:
            fbs_model = options["fbs_model"]
        self.penalty_k = self.penalty_k_min
        # if fbs_model is not None:
            # print(f"reset() 收到的fbs_model内存地址: {id(fbs_model)}")
            # print(f"reset() 收到传入的fbs_model: {fbs_model}")
        # else:
        #     print("reset() 未收到fbs_model（使用默认逻辑）")
        # 重置环境，生成初始解
        if fbs_model is None:
            # 如果fbs_model为None，则随机生成初始解
            # 初始解生成
            # ----------基因编码方式-------------------
            B=FBSUtil.select_B(self.areas, self.n, self.fac_limit_aspect, self.W)
            if B is None:
                # 处理没有可行B的情况，例如使用默认值
                B = 2
                print(f"没有找到可行的区带总数，使用默认值 B={B}")
            genes, permutation = FBSUtil.ZGeneCoding.generate_genes(self.n, B)
            # print(genes)
            bay_list, bay = FBSUtil.ZGeneCoding.decode_genes(genes, permutation)
            permutation, bay = FBSUtil.arrayToPermutation(bay_list)
            # ---------------------------------------
            # permutation, bay = FBSUtil.binary_solution_generator(self.areas, self.n, self.fac_limit_aspect, self.W)  # 采用k分初始解生成器
            # permutation,bay = FBSUtil.random_solution_generator(self.n) # 采用随机初始解生成器
            bay[-1] = 1  # bay的最后一个位置必须是1，表示最后一个设施是bay的结束
            self.fbs_model = FBSModel(
                permutation.astype(int).tolist(),
                bay.astype(int).tolist(),
                genes=genes.tolist() if isinstance(genes, np.ndarray) else genes
            )

        else:
            # 如果fbs_model不为None，则使用传入的fbs_model
            logger.debug(
                f"传入的fbs_model: permutation={self.fbs_model.permutation}, bay={self.fbs_model.bay}, genes={self.fbs_model.genes}")
            self.fbs_model = fbs_model
        (
            self.fac_x,
            self.fac_y,
            self.fac_h,
            self.fac_b,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.fitness,
        # ) = FBSUtil.StatusUpdatingDevice(self.fbs_model, self.areas, self.H,
        #                                  self.F, self.fac_limit_aspect) 
        ) = FBSUtil.StatusUpdatingDevice2(
            self.fbs_model,
            self.areas,
            self.H,
            self.F,
            self.fac_limit_aspect,
            g_best=self.best_fitness,
            penalty_k=self.penalty_k,
        )  
        self.penalty_k = min(
            self.penalty_k * self.penalty_k_growth, self.penalty_k_max
        )
        self.best_MHC = self.MHC                                   
        self.previous_fitness = self.fitness  # 初始化上一次的适应度值
        # 更新状态字典
        self.state = self.constructState()
        logger.debug("-------------------reset调试信息------------------")
        logger.debug(f"设施x坐标: {self.fac_x}")
        logger.debug(f"设施y坐标: {self.fac_y}")
        logger.debug(f"设施宽度: {self.fac_b}")
        logger.debug(f"设施高度: {self.fac_h}")
        logger.debug(f"设施横纵比: {self.fac_aspect_ratio}")
        logger.debug(f"设施距离矩阵: {self.D}")
        logger.debug(f"设施移动矩阵: {self.TM}")
        logger.debug(f"设施移动矩阵: {self.MHC}")
        logger.debug(f"设施适应度: {self.fitness}")
        logger.debug(f"状态: {self.state}")
        logger.debug("--------------------------------------------------")
        
        # 创建info字典
        info = {
            "fitness": self.fitness,
            "facility_count": self.n,
            "layout_dimensions": (self.H, self.W),
            "instance": self.instance
        }
        
        return self.state, info

    def calculate_reward_1(self):
        # 计算MHC改善程度
        # mhc_improvement = ((self.previous_MHC - self.MHC) /
        #                    self.previous_MHC if self.previous_MHC else 0)

        # 计算约束违反惩罚
        # aspect_ratio_penalty = sum(
        #     max(0, ar - self.fac_limit_aspect) +
        #     max(0, self.fac_limit_aspect - ar) for ar in self.fac_aspect_ratio)

        # 计算fitness改善程度
        fitness_improvement = ((self.previous_fitness - self.fitness) /
                               self.fitness if self.previous_fitness else 0)

        # # 综合奖励计算
        reward = (
            # 0.4 * mhc_improvement  # MHC改善权重
                1 * fitness_improvement  # 整体fitness改善权重
            # + 0.2 * aspect_ratio_penalty  # 约束违反惩罚权重
        )
        # reward = -self.fitness
        # 适应度和MHC的惩罚
        # reward = self.MHC - self.fitness
        return reward

    def calculate_reward_2(self):
        return -self.fitness

    def step(self, action):
        # 根据action执行相应的操作
        action_name = self.actions[int(action)]
        
        # 保存操作前的状态用于调试
        old_perm_len = len(self.fbs_model.permutation) if self.fbs_model.permutation else 0
        old_bay_len = len(self.fbs_model.bay) if self.fbs_model.bay else 0
        
        # if action_name == "facility_swap_single":
        # self.fbs_model.permutation, self.fbs_model.bay = (
        #     FBSUtil.facility_swap_single(self.fbs_model.permutation,
        #                                  self.fbs_model.bay))
        # elif action_name == "shuffle_single":
        # self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.shuffle_single(
        # self.fbs_model.permutation, self.fbs_model.Xbay)
        if action_name == "facility_swap":
            new_perm, new_bay = FBSUtil.facility_swap(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay))
            # 确保转换为列表并检查长度
            if len(new_perm) > 0 and len(new_bay) > 0:
                self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
                self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            else:
                logger.error(f"facility_swap 返回空数组: perm_len={len(new_perm)}, bay_len={len(new_bay)}")
        elif action_name == "bay_flip":
            new_perm, new_bay = FBSUtil.bay_flip(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay))
            if len(new_perm) > 0 and len(new_bay) > 0:
                self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
                self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            else:
                logger.error(f"bay_flip 返回空数组: perm_len={len(new_perm)}, bay_len={len(new_bay)}")
        elif action_name == "bay_swap":
            new_perm, new_bay = FBSUtil.bay_swap(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay))
            if len(new_perm) > 0 and len(new_bay) > 0:
                self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
                self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            else:
                logger.error(f"bay_swap 返回空数组: perm_len={len(new_perm)}, bay_len={len(new_bay)}")
        elif action_name == "bay_shuffle":
            new_perm, new_bay = FBSUtil.bay_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay)
            self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
            self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)

        elif action_name == "facility_shuffle":
            new_perm, new_bay = FBSUtil.facility_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay)
            self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
            self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            # elif action_name == "permutation_shuffle":
            #     self.fbs_model.permutation, self.fbs_model.bay = (
            #         FBSUtil.permutation_shuffle(self.fbs_model.permutation,
            #                                     self.fbs_model.bay))
        elif action_name == "repair":
            new_perm, new_bay = FBSUtil.repair(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay), 
                self.fac_b, self.fac_h, self.fac_limit_aspect)
            if len(new_perm) > 0 and len(new_bay) > 0:
                self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
                self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            else:
                logger.error(f"repair 返回空数组: perm_len={len(new_perm)}, bay_len={len(new_bay)}")

        elif action_name == "facility_insert":
            new_perm, new_bay = FBSUtil.facility_insert(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay))
            # 确保转换为列表并检查长度
            if len(new_perm) > 0 and len(new_bay) > 0:
                self.fbs_model.permutation = new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
                self.fbs_model.bay = new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
            else:
                logger.error(f"facility_swap 返回空数组: perm_len={len(new_perm)}, bay_len={len(new_bay)}")

        # --- 【【【 新增动作 5 的逻辑 】】】 ---
        elif action_name == "ga_action":
            # 'self' 就是 env_instance
            # 调用我们在 FBSUtil.py 中新增的函数
            # (注意: 您可能需要调整 pop_size 和 generations)
            new_perm, new_bay = FBSUtil.ga_population_action(
                self.fbs_model, 
                self, 
                pop_size=10, 
                generations=30
            )
            self.fbs_model.permutation = new_perm
            self.fbs_model.bay = new_bay
        # --- 【【【 新增结束 】】】 ---

        elif action_name == "idle":
            pass
        # elif action_name == "bay_shuffle":
        #     self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.bay_shuffle(
        #         self.fbs_model.permutation, self.fbs_model.bay)
        # elif action_name == "facility_shuffle":
        #     self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.facility_shuffle(
        #         self.fbs_model.permutation, self.fbs_model.bay)
        else:
            raise ValueError(f"Invalid action: {action_name}")

        self.previous_MHC = self.MHC  # 保存上一步的MHC
        self.previous_fitness = self.fitness  # 保存上一步的fitness
        # 更新最佳适应度
        # if self.MHC < self.best_MHC: 
        #     self.best_MHC = self.MHC     
        # 刷新状态
        (
            self.fac_x,
            self.fac_y,
            self.fac_h,
            self.fac_b,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.fitness,
        # ) = FBSUtil.StatusUpdatingDevice(self.fbs_model, self.areas, self.H,
        #                                  self.F, self.fac_limit_aspect)  

        ) = FBSUtil.StatusUpdatingDevice2(
            self.fbs_model,
            self.areas,
            self.H,
            self.F,
            self.fac_limit_aspect,
            g_best=self.best_fitness,
            penalty_k=self.penalty_k,
        )  
        self.penalty_k = min(
            self.penalty_k * self.penalty_k_growth, self.penalty_k_max
        )
                              
        # 更新状态字典
        self.state = self.constructState()
        # 计算奖励函数
        reward = self.calculate_reward_1()
        self.previous_fitness = self.fitness
        # 更新info字典，包含更多的调试信息
        info = {
            "TimeLimit.truncated": False,
            "current_fitness": self.fitness,  # 当前适应度值
            "previous_fitness": self.previous_fitness,  # 上一次的适应度值
            "reward": reward,  # 当前步骤的奖励
            "facility_count": self.n,  # 设施数量
            "action_taken": action_name,  # 执行的动作名称
            "layout_dimensions": (self.H, self.W),  # 布局的长和宽
        }
        return (
            self.state,
            reward,
            False,  # terminated
            False,  # truncated
            info,
        )

    def step2(self, action):
        # 根据action执行相应的操作
        action_name = self.actions[int(action)]
        if action_name == "bay_shuffle":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.bay_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay)
        elif action_name == "facility_shuffle":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.facility_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay)
        else:
            raise ValueError(f"Invalid action: {action_name}")

        self.previous_MHC = self.MHC  # 保存上一步的MHC
        self.previous_fitness = self.fitness  # 保存上一步的fitness

        # 刷新状态
        (
            self.fac_x,
            self.fac_y,
            self.fac_h,
            self.fac_b,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.fitness,
        ) = FBSUtil.StatusUpdatingDevice(self.fbs_model, self.areas, self.H,
                                         self.F, self.fac_limit_aspect)
        self.penalty_k = min(
            self.penalty_k * self.penalty_k_growth, self.penalty_k_max
        )
        # 更新状态字典
        self.state = self.constructState()
        # 计算奖励函数
        reward = self.calculate_reward_1()
        self.previous_fitness = self.fitness
        # 更新info字典，包含更多的调试信息
        info = {
            "TimeLimit.truncated": False,
            "current_fitness": self.fitness,  # 当前适应度值
            "previous_fitness": self.previous_fitness,  # 上一次的适应度值
            "reward": reward,  # 当前步骤的奖励
            "facility_count": self.n,  # 设施数量
            "action_taken": action_name,  # 执行的动作名称
            "layout_dimensions": (self.H, self.W),  # 布局的长和宽
        }
        return (
            self.state,
            reward,
            False,  # terminated
            False,  # truncated
            info,
        )

    def render(self):
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        ax.set_title("Facility layout")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        plt.grid(False)
        plt.gca().set_aspect("equal", adjustable="box")

        # 绘制设施矩形
        for i, facility_label in enumerate(self.fbs_model.permutation):
            facility_idx = facility_label - 1  # 设施索引从0开始
            x_from = self.fac_x[facility_idx] - self.fac_b[facility_idx] / 2
            x_to = self.fac_x[facility_idx] + self.fac_b[facility_idx] / 2
            y_from = self.fac_y[facility_idx] - self.fac_h[facility_idx] / 2
            y_to = self.fac_y[facility_idx] + self.fac_h[facility_idx] / 2

            # 边框颜色表示长宽比状态
            line_color = "red" if self.fac_aspect_ratio[facility_idx] > self.fac_limit_aspect else "green"

            # 填充颜色表示成本（RGB）
            state_reshaped = self.state.reshape(self.n, 3)
            R = state_reshaped[facility_idx, 0] / 255
            G = state_reshaped[facility_idx, 1] / 255
            B = state_reshaped[facility_idx, 2] / 255
            face_color = (R, G, B, 0.7)

            rect = patches.Rectangle(
                (x_from, y_from),
                width=x_to - x_from,
                height=y_to - y_from,
                edgecolor=line_color,
                facecolor=face_color,  # 填充颜色
                linewidth=1
            )
            ax.add_patch(rect)

            # 显示设施ID
            ax.text(
                x_from + (x_to - x_from) / 2,
                y_from + (y_to - y_from) / 2,
                f"{int(facility_label)}",
                ha="center",
                va="center",
                color="white" if np.mean(face_color[:3]) < 0.5 else "black"  # 自适应文字颜色
            )

        # 显示MHC和Fitness
        plt.figtext(0.5, 0.93, f"MHC: {self.MHC:.2f}", ha="center", fontsize=12)
        plt.figtext(0.5, 0.96,
                    # f"Fitness: {FBSUtil.getFitness(self.MHC, self.fac_b, self.fac_h, self.fac_limit_aspect):.2f}",
                    f"Fitness: {self.fitness:.2f}",
                    ha="center", fontsize=12)

        plt.show()

    def constructState(self):
        state = np.zeros((self.n, 3))
        permutation = self.fbs_model.permutation
        TM = self.TM
        sources = np.sum(TM, axis=1)
        sinks = np.sum(TM, axis=0)
        R = np.array(
            ((permutation - np.min(permutation)) / (np.max(permutation) - np.min(permutation)))
            * 255
        ).astype(np.float64)
        G = np.array(
            ((sources - np.min(sources)) / (np.max(sources) - np.min(sources)))
            * 255
        ).astype(np.float64)
        B = np.array(
            ((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)))
            * 255
        ).astype(np.float64)
        state[:, 0] = R
        state[:, 1] = G
        state[:, 2] = B
        return state.flatten()

