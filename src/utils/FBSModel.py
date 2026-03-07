import numpy as np

class FBSModel:
    """
    存储设施布局问题的单个解（个体）。
    包含了排列、区带和基因编码。
    """
    def __init__(self, permutation=None, bay=None, genes=None):
        """
        初始化
        
        --- 修正: ---
        旧代码在 __init__ 中调用了 setter, 导致了循环依赖和 'AttributeError'
        
        新代码 (健壮性):
        我们直接初始化 *内部* 属性 (_permutation, _bay, _genes)，
        以绕过 setter 中的检查。
        在所有属性都初始化完毕后，我们再执行一次性的长度检查。
        """
        self._permutation = permutation.copy() if permutation is not None else []
        self._bay = bay.copy() if bay is not None else []
        self._genes = genes.copy() if genes is not None else []

        # 在所有内部属性都设置完毕后，再执行长度检查
        if len(self._permutation) != len(self._bay):
            raise ValueError(
                f"Permutation ({len(self._permutation)}) and bay ({len(self._bay)}) "
                "lengths must match upon initialization."
            )

    @property
    def permutation(self):
        return self._permutation

    @permutation.setter
    def permutation(self, value):
        if value is None:
            self._permutation = []
        # 检查 _bay 是否已存在 (防止在 __init__ 之外出错)
        elif hasattr(self, '_bay') and len(value) != len(self._bay):
            raise ValueError("Permutation and bay lengths must match.")
        else:
            self._permutation = value

    @property
    def bay(self):
        return self._bay

    @bay.setter
    def bay(self, value):
        if value is None:
            self._bay = []
        # 检查 _permutation 是否已存在
        elif hasattr(self, '_permutation') and len(value) != len(self._permutation):
            raise ValueError("Permutation and bay lengths must match.")
        else:
            self._bay = value

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value):
        self._genes = value.copy() if value is not None else []

    @property
    def array_2d(self):
        """
        辅助属性，用于获取二维数组格式的解
        (复现自 FBSUtil.permutationToArray)
        """
        try:
            if not self._permutation or not self._bay:
                return []
                
            bay_copy = self._bay.copy()
            perm_copy = self._permutation.copy()
            bay_copy[-1] = 1  # 确保最后一个是1
            array = []
            start = 0
            for i, val in enumerate(bay_copy):
                if val == 1:
                    array.append(np.array(perm_copy[start : i + 1]))
                    start = i + 1
            return array
        except Exception:
            return []

    def __str__(self):
        return f"FBSModel(Permutation: {self._permutation}, Bay: {self._bay})"