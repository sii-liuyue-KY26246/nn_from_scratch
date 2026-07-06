"""数据预处理与加载工具。

包含归一化、编码、数据集切分、mini-batch 生成器等。
这些工具会在后续所有实验中反复使用。

实现顺序建议: normalize -> standardize -> one_hot_encode -> train_val_test_split -> DataLoader
"""

import numpy as np


def normalize(X, feature_range=(0, 1)):
    """Min-Max 归一化，将数据缩放到 [a, b] 范围。

    公式: X_norm = a + (X - X_min) / (X_max - X_min) * (b - a)

    参数:
        X: ndarray, shape (N, D)
        feature_range: tuple (a, b)，目标范围

    返回:
        X_norm: ndarray, shape (N, D)
        params: dict，包含 'min', 'max'，用于对测试集做同样的变换
    """
    # TODO
    raise NotImplementedError


def standardize(X, mean=None, std=None):
    """Z-score 标准化：X_std = (X - mean) / std

    参数:
        X:    ndarray, shape (N, D)
        mean: ndarray 或 None，如果为 None 则从 X 计算
        std:  ndarray 或 None，如果为 None 则从 X 计算

    返回:
        X_std: ndarray, shape (N, D)
        params: dict，包含 'mean', 'std'

    提示:
        - 沿 axis=0 计算 mean 和 std（每个特征独立）
        - std 加 1e-8 防止除零
        - 训练集计算 mean/std，测试集复用训练集的 mean/std
    """
    # TODO
    raise NotImplementedError


def one_hot_encode(y, num_classes=None):
    """将整数标签转为 one-hot 编码。

    参数:
        y:           ndarray, shape (N,)，整数标签
        num_classes: int 或 None，类别数。None 则自动推断

    返回:
        one_hot: ndarray, shape (N, num_classes)

    提示:
        - 创建全零矩阵，然后 one_hot[np.arange(N), y] = 1
    """
    # TODO
    raise NotImplementedError


def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, seed=None):
    """将数据集切分为训练、验证、测试三部分。

    参数:
        X:          ndarray, shape (N, ...)
        y:          ndarray, shape (N,)
        val_ratio:  float，验证集比例
        test_ratio: float，测试集比例
        seed:       int 或 None，随机种子

    返回:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)

    提示:
        - 先 shuffle 索引（用 seed 控制），再按比例切分
    """
    # TODO
    raise NotImplementedError


class DataLoader:
    """Mini-batch 数据加载器。

    支持 len()、for 循环迭代、shuffle。

    用法:
        loader = DataLoader(X, y, batch_size=32, shuffle=True)
        for x_batch, y_batch in loader:
            ...

    提示:
        - __len__ 返回 batch 数量（向上取整）
        - __iter__ 用 yield 产出每个 batch
        - shuffle 在每个 epoch 开始时打乱索引
        - 注意用 self.X 和 self.y，不要引用外部变量
    """

    def __init__(self, X, y, batch_size=32, shuffle=True):
        # TODO
        raise NotImplementedError

    def __len__(self):
        # TODO
        raise NotImplementedError

    def __iter__(self):
        # TODO
        raise NotImplementedError
