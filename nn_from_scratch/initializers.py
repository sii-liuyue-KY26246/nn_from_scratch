"""权重初始化方法。

不同的初始化方式对网络训练有巨大影响。
本模块实现几种经典初始化方法，并提供可视化工具验证效果。

实现顺序建议: zeros -> random_normal -> xavier -> he -> visualize_initializations
"""

import numpy as np


def zeros(shape):
    """全零初始化。

    几乎不应该用于权重（为什么？所有神经元输出相同，梯度相同，对称性无法打破）。
    但偏置通常用零初始化。
    """
    # TODO
    raise NotImplementedError


def random_normal(shape, scale=0.01):
    """小随机数初始化: W ~ N(0, scale²)

    参数:
        shape: tuple，权重形状
        scale: float，标准差
    """
    # TODO
    raise NotImplementedError


def xavier(shape):
    """Xavier / Glorot 初始化: W ~ N(0, 2/(fan_in + fan_out))

    适合 Sigmoid 和 Tanh 激活函数。

    参数:
        shape: tuple (fan_in, fan_out)

    提示:
        - fan_in = shape[0], fan_out = shape[1]
        - std = sqrt(2.0 / (fan_in + fan_out))
    """
    # TODO
    raise NotImplementedError


def he(shape):
    """He / Kaiming 初始化: W ~ N(0, 2/fan_in)

    适合 ReLU 激活函数。

    参数:
        shape: tuple (fan_in, fan_out)

    提示:
        - std = sqrt(2.0 / fan_in)
    """
    # TODO
    raise NotImplementedError


def visualize_initializations(in_features=784, out_features=256, n_layers=5):
    """可视化不同初始化方法在多层网络中的激活值分布。

    模拟一个 n_layers 层的全连接网络（只做前向传播，用 ReLU），
    在随机输入上跑一遍，记录每层输出的均值和标准差。

    参数:
        in_features:  输入维度
        out_features: 每层的输出维度（所有隐藏层相同）
        n_layers:     层数

    返回:
        results: dict，key 是初始化方法名，
                 value 是 list of (mean, std)，每层一个

    提示:
        - 对每种初始化方法（zeros, random_normal, xavier, he），
          依次做 n_layers 次 ReLU(x @ W)
        - 记录每层输出的 mean 和 std
        - 如果 std 趋近 0 → 激活值坍缩（梯度消失）
        - 如果 std 爆炸 → 激活值爆炸（梯度爆炸）
        - He 初始化 + ReLU 应该保持 std 大致稳定
    """
    # TODO
    raise NotImplementedError
