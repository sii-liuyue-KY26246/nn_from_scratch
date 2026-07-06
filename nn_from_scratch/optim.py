"""优化器模块。

实现 SGD 和 SGD+Momentum。
优化器接收模型的 parameters() 列表，在每次 step() 时更新参数。

实现顺序建议: SGD -> SGDMomentum
"""

import numpy as np


class SGD:
    """基础随机梯度下降。

    更新规则: param -= lr * grad

    用法:
        optimizer = SGD(model.parameters(), lr=0.01)
        # 训练循环中:
        loss, dout = compute_loss(...)
        model.backward(dout)
        optimizer.step()
        optimizer.zero_grad()
    """

    def __init__(self, parameters, lr=0.01):
        """
        参数:
            parameters: list of (param, grad) tuples，来自 model.parameters()
            lr:         学习率
        """
        # TODO
        raise NotImplementedError

    def step(self):
        """执行一步参数更新。

        提示:
            - 遍历 self.parameters，对每个 (param, grad) 做 param -= lr * grad
            - 注意要原地修改 param（用 -=），不能创建新数组
        """
        # TODO
        raise NotImplementedError

    def zero_grad(self):
        """将所有梯度清零。

        提示:
            - 遍历 self.parameters，对每个 (_, grad) 做 grad[...] = 0
            - 注意用 grad[...] = 0 而不是 grad = np.zeros_like(...)
              前者原地修改，后者创建新数组（引用断开）
        """
        # TODO
        raise NotImplementedError


class SGDMomentum:
    """带动量的 SGD。

    更新规则:
        v = momentum * v - lr * grad
        param += v

    动量的直觉：想象一个球在损失曲面上滚动。
    没有动量时，球在每个点只看当前的坡度。
    有动量时，球有"惯性"，会沿之前的方向继续滚动，
    这使得它能更快穿过狭长的峡谷，也能越过小的局部最小值。

    用法:
        optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)
    """

    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        提示:
            - 为每个参数维护一个速度变量 v，初始为零
            - 用 id(param) 做 key 存入字典
        """
        # TODO
        raise NotImplementedError

    def step(self):
        """
        提示:
            - v = momentum * v - lr * grad
            - param += v
        """
        # TODO
        raise NotImplementedError

    def zero_grad(self):
        # TODO
        raise NotImplementedError
