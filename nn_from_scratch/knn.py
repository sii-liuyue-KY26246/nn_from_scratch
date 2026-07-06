"""k-最近邻分类器。

本模块的核心目标不是学 k-NN 算法本身，
而是练习 NumPy 向量化编程——不用任何 Python 循环计算距离矩阵。

实现顺序建议: compute_distances_two_loops -> compute_distances_one_loop -> compute_distances_vectorized -> KNNClassifier
"""

import numpy as np


def compute_distances_two_loops(X_train, X_test):
    """用两重 Python 循环计算 L2 距离矩阵。（最慢，但最直观）

    参数:
        X_train: shape (N_train, D)
        X_test:  shape (N_test, D)

    返回:
        dists: shape (N_test, N_train)，dists[i, j] = ||X_test[i] - X_train[j]||_2
    """
    # TODO
    raise NotImplementedError


def compute_distances_one_loop(X_train, X_test):
    """用一重 Python 循环 + 向量化计算 L2 距离矩阵。（中等速度）

    提示:
        - 外层循环遍历 X_test 的每一行
        - 内层用 NumPy 向量化：一次计算一个测试样本到所有训练样本的距离
        - diff = X_test[i] - X_train  # shape (N_train, D)，利用了广播
        - dists[i] = np.sqrt(np.sum(diff**2, axis=1))
    """
    # TODO
    raise NotImplementedError


def compute_distances_vectorized(X_train, X_test):
    """完全向量化计算 L2 距离矩阵。（最快，零 Python 循环）

    利用展开公式: ||a - b||² = ||a||² + ||b||² - 2 * a · b

    参数:
        X_train: shape (N_train, D)
        X_test:  shape (N_test, D)

    返回:
        dists: shape (N_test, N_train)

    提示:
        - test_sq = np.sum(X_test**2, axis=1, keepdims=True)   # (N_test, 1)
        - train_sq = np.sum(X_train**2, axis=1, keepdims=True) # (N_train, 1)
        - cross = X_test @ X_train.T                           # (N_test, N_train)
        - dists = sqrt(test_sq + train_sq.T - 2 * cross)
        - 注意 np.clip(..., a_min=0) 防止浮点误差导致负数
    """
    # TODO
    raise NotImplementedError


class KNNClassifier:
    """k-最近邻分类器。

    用法:
        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """存储训练数据（k-NN 没有真正的"训练"过程）。"""
        # TODO
        raise NotImplementedError

    def predict(self, X_test):
        """预测 X_test 中每个样本的类别。

        提示:
            - 用 compute_distances_vectorized 计算距离
            - 对每个测试样本，找 k 个最近邻（np.argsort 或 np.argpartition）
            - 投票：最近邻中出现最多的类别（np.bincount 很好用）
        """
        # TODO
        raise NotImplementedError

    def score(self, X_test, y_test):
        """返回准确率。"""
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)
