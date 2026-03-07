import numpy as np

loaded_data = np.load(
    r'C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\QLearningResult\Q_table.npy',
    allow_pickle=True  # 允许加载包含对象的数组
)

# 查看内容
print(loaded_data)

loaded_array = np.load(r'C:\Users\17122\PycharmProjects\pythonProject\ua-flp-LSA2\files\QLearningResult\times_table.npy',
                       allow_pickle=True) # 替换为你的文件路径

# 查看内容
print(loaded_array)