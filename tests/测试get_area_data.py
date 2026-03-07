import src.utils.FBSUtil as FBSUtils
import pickle
import os

# 获取脚本所在目录（方法1）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 向上回退两级目录，移除 ua-flp-LSA\tests
base_dir = os.path.dirname(os.path.dirname(script_dir))
relative_path = "ua-flp-LSA2\data\maoyan_cont_instances.pkl"
file_path = os.path.join(base_dir, relative_path)

with open(file_path, "rb") as f:
    data = pickle.load(f)

sample_key = "AB20-ar3"
df = data[2][sample_key]
# 调用函数
areas, aspect = FBSUtils.getAreaData(df)
print("面积数组:", areas)  # 输出: [100. 200. 300.]
print("第一个长宽比:", aspect)  # 输出: 1.5