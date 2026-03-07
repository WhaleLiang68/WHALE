import numpy as np
import math

class MO_FBSUtil:
    """
    多目标优化专用工具类
    对应开题报告中的目标函数计算 (CR, DR, AR)
    """

    @staticmethod
    def calculate_overlap_length(min1, max1, min2, max2):
        """计算两个区间 [min1, max1] 和 [min2, max2] 的重叠长度"""
        overlap = max(0, min(max1, max2) - max(min1, min2))
        return overlap

    @staticmethod
    def get_adjacency_length(fac_x, fac_y, fac_b, fac_h, n):
        """
        计算设施间的接触周长 (邻接长度) lij
        对应目标2: Maximize Close Relationship (CR)
        """
        adjacency_matrix = np.zeros((n, n))

        # 这是一个 O(N^2) 的计算，对于 N=20~50 还可以接受
        for i in range(n):
            for j in range(i + 1, n):
                # 设施 i 的边界
                xi_min, xi_max = fac_x[i] - fac_b[i]/2, fac_x[i] + fac_b[i]/2
                yi_min, yi_max = fac_y[i] - fac_h[i]/2, fac_y[i] + fac_h[i]/2

                # 设施 j 的边界
                xj_min, xj_max = fac_x[j] - fac_b[j]/2, fac_x[j] + fac_b[j]/2
                yj_min, yj_max = fac_y[j] - fac_h[j]/2, fac_y[j] + fac_h[j]/2

                # 判断是否接触:
                # 两个矩形接触意味着：在某个维度上重叠长度 > 0，且在另一个维度上距离极小(≈0)

                # 1. 垂直边接触 (左右相邻)
                # X轴方向距离接近0，Y轴方向有重叠
                dist_x = max(0, abs(fac_x[i] - fac_x[j]) - (fac_b[i] + fac_b[j])/2)
                contact_y = MO_FBSUtil.calculate_overlap_length(yi_min, yi_max, yj_min, yj_max)

                # 2. 水平边接触 (上下相邻)
                # Y轴方向距离接近0，X轴方向有重叠
                dist_y = max(0, abs(fac_y[i] - fac_y[j]) - (fac_h[i] + fac_h[j])/2)
                contact_x = MO_FBSUtil.calculate_overlap_length(xi_min, xi_max, xj_min, xj_max)

                total_contact = 0
                tolerance = 1e-3 # 浮点数误差容忍度

                if dist_x < tolerance:
                    total_contact += contact_y
                if dist_y < tolerance:
                    total_contact += contact_x

                adjacency_matrix[i][j] = total_contact
                adjacency_matrix[j][i] = total_contact

        return adjacency_matrix

    @staticmethod
    def calculate_objectives(fac_x, fac_y, fac_b, fac_h, mhc, n,
                             rel_matrix=None, dist_req_matrix=None):
        """
        计算所有四个目标函数值
        Input:
            rel_matrix: 亲密关系等级矩阵 rij (需从数据加载)
            dist_req_matrix: 距离要求等级矩阵 sij (需从数据加载)
        Output:
            objs: [MHC, CR, DR, AR]
        """
        # 1. MHC (Min) - 已由外部传入
        f1_mhc = mhc

        # 2. CR (Max) - 亲密关系
        # 公式 10: Sum(rij * lij)
        f2_cr = 0
        if rel_matrix is not None:
            adj_matrix = MO_FBSUtil.get_adjacency_length(fac_x, fac_y, fac_b, fac_h, n)
            # 矩阵点乘求和 (只计算上三角)
            f2_cr = np.sum(np.triu(rel_matrix * adj_matrix))

        # 3. DR (Max) - 距离要求 (假设是要求保持距离，或者是加权距离)
        # 公式 11: Sum(sij * dij)
        # 如果是"Maximize Distance Requirement"，通常指对于特定设施对，距离越远越好
        f3_dr = 0
        if dist_req_matrix is not None:
            # 计算欧氏距离
            d_matrix = np.sqrt((fac_x[:, None] - fac_x[None, :])**2 +
                               (fac_y[:, None] - fac_y[None, :])**2)
            f3_dr = np.sum(np.triu(dist_req_matrix * d_matrix))

        # 4. AR (Max) - 纵横比满意度
        # 公式 12: Sum(ar_si)
        # 描述: 当纵横比在 [1, 2.5] 内满意度高
        f4_ar = 0
        aspect_ratios = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)

        # 定义满意度函数 (根据描述模拟)
        # 范围内为1，范围外指数衰减
        for ar in aspect_ratios:
            if 1.0 <= ar <= 2.5:
                score = 1.0
            else:
                # 简单的衰减函数，ar越大分数越低
                score = 1.0 / (1.0 + (ar - 2.5))
            f4_ar += score

        f4_ar = f4_ar / n # 归一化到 [0, 1] 区间便于处理

        return [f1_mhc, f2_cr, f3_dr, f4_ar]

    @staticmethod
    def aggregated_energy(objectives, weights):
        """
        公式 29: 聚合能量函数
        E(X,t) = w_MHC * f_MHC + w_CR * (1/f_CR) + ...
        注意：最大化目标需要取倒数转化为最小化能量
        """
        mhc, cr, dr, ar = objectives
        w_mhc, w_cr, w_dr, w_ar = weights

        # 防止除零
        epsilon = 1e-6

        term1 = w_mhc * mhc
        term2 = w_cr * (1.0 / (cr + epsilon))
        term3 = w_dr * (1.0 / (dr + epsilon))
        term4 = w_ar * (1.0 / (ar + epsilon))

        return term1 + term2 + term3 + term4