import numpy as np
import os
import pickle

class MO_DataGenerator:
    """
    多目标数据生成器
    生成符合论文要求的亲密关系矩阵(Rel)和距离要求矩阵(DistReq)
    """

    @staticmethod
    def generate_matrix(n, seed=None):
        """生成 n x n 的对称矩阵，值在 [0, 6] 之间"""
        if seed is not None:
            np.random.seed(seed)

        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                val = np.random.randint(0, 7)  # [0, 6]
                matrix[i][j] = val
                matrix[j][i] = val
        return matrix

    @staticmethod
    def load_or_generate_data(n, instance_name="AB20", data_dir="./data"):
        """加载或生成数据，确保实验一致性"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filepath = os.path.join(data_dir, f"{instance_name}_MO_matrices.pkl")

        if os.path.exists(filepath):
            # print(f"[MO] Loading data from {filepath}")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data['rel_matrix'], data['dist_req_matrix']
        else:
            print(f"[MO] Generating new data for {instance_name}")
            rel_matrix = MO_DataGenerator.generate_matrix(n, seed=42)
            dist_req_matrix = MO_DataGenerator.generate_matrix(n, seed=2024)

            data = {'rel_matrix': rel_matrix, 'dist_req_matrix': dist_req_matrix}
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return rel_matrix, dist_req_matrix
if __name__ == "__main__":
    # 测试生成
    n_facilities = 20
    rel, dist = MO_DataGenerator.load_or_generate_data(n_facilities, "AB20")
    print("Rel Matrix Sample:\n", rel[:5, :5])