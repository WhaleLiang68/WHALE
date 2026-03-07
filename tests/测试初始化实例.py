# tests/测试初始化实例.py
import unittest
import numpy as np
import src.utils.config as config
from src.utils.DataExtractor import DataProcessingEnv

class TestDataProcessingEnv(unittest.TestCase):
    def setUp(self):
        # 使用存在的实例名称（例如 'BA12-maoyan'）
        self.valid_instance = "BA12-maoyan"
        self.env = DataProcessingEnv(instance=self.valid_instance)
        np.random.seed(42)

    def test_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.env.instance, self.valid_instance, "实例名称不匹配")
        self.assertIn(self.valid_instance, self.env.problems, "实例不存在于 problems 数据")
        self.assertEqual(self.env.n, self.env.problems[self.valid_instance], "设施数量不一致")
        self.assertEqual(len(self.env.areas), self.env.n, "面积数组长度错误")
        self.assertEqual(self.env.action_space.n, 5, "动作空间应为5个离散动作")
        self.assertEqual(self.env.observation_space.shape, (self.env.n * 3,), "状态空间形状错误")

    def test_reset(self):
        """测试环境重置"""
        state = self.env.reset()
        self.assertEqual(state.dtype, np.uint8)
        self.assertEqual(state.shape, (self.env.n * 3,))
        self.assertEqual(self.env.fbs_model.bay[-1], 1, "区带最后一个元素应为1")

    def test_step(self):
        """测试执行动作"""
        self.env.reset()
        action = 0  # "facility_swap"
        next_state, reward, done, info = self.env.step(action)
        self.assertIsInstance(next_state, np.ndarray, "下一状态应为数组")
        self.assertIsInstance(reward, float, "奖励应为浮点数")
        self.assertNotEqual(self.env.fitness, np.inf, "适应度未更新")

    def test_invalid_instance(self):
        """测试无效实例名称"""
        with self.assertRaises(ValueError):
            invalid_env = DataProcessingEnv(instance="invalid_instance_123")
            invalid_env.reset()

if __name__ == "__main__":
    unittest.main(verbosity=2)