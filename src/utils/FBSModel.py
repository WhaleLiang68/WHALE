import numpy as np


class FBSModel:
    def __init__(self, permutation=None, bay=None, genes=None):
        self._permutation = self._normalize_sequence(permutation)
        self._bay = self._normalize_sequence(bay)
        self._genes = self._normalize_sequence(genes)

        if len(self._permutation) != len(self._bay):
            raise ValueError(
                f"Permutation ({len(self._permutation)}) and bay ({len(self._bay)}) "
                "lengths must match upon initialization."
            )

    @staticmethod
    def _normalize_sequence(value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        return list(value)

    @property
    def permutation(self):
        return self._permutation

    @permutation.setter
    def permutation(self, value):
        normalized = self._normalize_sequence(value)
        if hasattr(self, '_bay') and len(normalized) != len(self._bay):
            raise ValueError("Permutation and bay lengths must match.")
        self._permutation = normalized

    @property
    def bay(self):
        return self._bay

    @bay.setter
    def bay(self, value):
        normalized = self._normalize_sequence(value)
        if hasattr(self, '_permutation') and len(normalized) != len(self._permutation):
            raise ValueError("Permutation and bay lengths must match.")
        self._bay = normalized

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value):
        self._genes = self._normalize_sequence(value)

    @property
    def array_2d(self):
        perm_copy = self._normalize_sequence(self._permutation)
        bay_copy = self._normalize_sequence(self._bay)
        if len(perm_copy) == 0 or len(bay_copy) == 0:
            return []
        if len(perm_copy) != len(bay_copy):
            return []

        bay_copy[-1] = 1
        array = []
        start = 0
        for i, val in enumerate(bay_copy):
            if val == 1:
                array.append(perm_copy[start : i + 1])
                start = i + 1
        return array

    def __str__(self):
        return f"FBSModel(Permutation: {self._permutation}, Bay: {self._bay})"
