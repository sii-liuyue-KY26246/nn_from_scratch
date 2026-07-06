"""神经网络层模块。

实现全连接层（Affine）的前向和反向传播。
这是神经网络反向传播的"原子操作"——所有更复杂的网络都是由这些积木搭起来的。

实现顺序建议: Affine -> affine_relu_forward/backward -> Sequential
"""

import numpy as np
from .module import Module


class Affine(Module):
    """全连接层（仿射变换）: out = x @ W + b

    参数:
        in_features:  输入特征维度
        out_features: 输出特征维度

    前向:
        out = x @ W + b
        缓存 x（反向传播需要）

    反向:
        给定 dout (shape 同 out)，计算:
        - dx = dout @ W.T
        - dW = x.T @ dout
        - db = sum(dout, axis=0)
        将 dW, db 存储在 self.dW, self.db 中

    提示:
        - forward 的输入 x 可能是任意形状 (N, d1, d2, ...)，
          需要先 reshape 成 (N, D) 再做矩阵乘法，反向传播时再 reshape 回去
        - 这个 reshape 操作在 CNN 中很常见（flatten）
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重用小随机数初始化，偏置用零初始化
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        # 梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # 缓存
        self.cache = None

    def forward(self, x):
        """
        参数:
            x: shape (N, d1, d2, ...) 或 (N, D)

        返回:
            out: shape (N, out_features)
        """
        # TODO
        # 1. 保存原始形状，将 x reshape 成 (N, D)
        # 2. 计算 out = x_flat @ W + b
        # 3. 缓存 (x, x_flat) 或 (x_original_shape, x_flat)
        # 4. 返回 out
        raise NotImplementedError

    def backward(self, dout):
        """
        参数:
            dout: shape (N, out_features)，上游梯度

        返回:
            dx: shape 同 forward 的输入 x
        """
        # TODO
        # 1. 从 cache 取出数据
        # 2. self.dW = x_flat.T @ dout
        # 3. self.db = np.sum(dout, axis=0)
        # 4. dx_flat = dout @ self.W.T
        # 5. dx = dx_flat.reshape(原始形状)
        # 6. 返回 dx
        raise NotImplementedError

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def __repr__(self):
        return f"Affine({self.in_features}, {self.out_features})"


class Sequential(Module):
    """按顺序执行一系列层。

    用法:
        model = Sequential(
            Affine(784, 256),
            ReLU(),
            Affine(256, 10)
        )
        out = model(x)              # 前向
        model.backward(dout)        # 反向
        params = model.parameters() # 收集所有参数

    提示:
        - forward：依次调用每个层
        - backward：倒序调用每个层的 backward
        - parameters：收集所有层的 parameters
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        # TODO
        raise NotImplementedError

    def backward(self, dout):
        # TODO
        raise NotImplementedError

    def parameters(self):
        # TODO
        raise NotImplementedError

    def __repr__(self):
        lines = [f"  ({i}): {repr(layer)}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(lines) + "\n)"
