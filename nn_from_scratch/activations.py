"""激活函数模块。

实现常见激活函数的前向传播和反向传播。
每个激活函数都是一个继承自 Module 的类，无可学习参数。

实现顺序建议: Sigmoid -> ReLU -> LeakyReLU -> Tanh -> ELU
"""

import numpy as np
from .module import Module


class Sigmoid(Module):
    """Sigmoid 激活函数: σ(x) = 1 / (1 + exp(-x))

    前向: out = σ(x)
    反向: dx = dout * σ(x) * (1 - σ(x))

    提示:
        - 缓存 forward 的输出（而非输入），因为导数可以用输出本身表示
        - 注意数值稳定性：当 x 很大的负数时 exp(-x) 会溢出，
          可以利用 σ(-x) = 1 - σ(x) 的性质来处理
    """

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = 1 / (1+np.exp(-x))
        return self.cache
    def backward(self, dout):
        x = self.cache
        return dout * x * (1-x)
    def __repr__(self):
        return "Sigmoid()"


class ReLU(Module):
    """ReLU 激活函数: f(x) = max(0, x)

    前向: out = max(0, x)
    反向: dx = dout * (x > 0)

    提示:
        - 缓存 forward 的输入 x
        - backward 中，x > 0 的位置梯度通过，x <= 0 的位置梯度为 0
        - 注意 dout.copy()，不要修改原始的 dout
    """

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0,x)
    def backward(self, dout):
        return dout * (self.cache > 0)
    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    """Leaky ReLU: f(x) = x if x > 0, else alpha * x

    参数:
        alpha (float): 负半轴的斜率，默认 0.01

    提示:
        - np.where(condition, x_if_true, x_if_false) 是你的好帮手
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.where(x>0,x,self.alpha*x)
    def backward(self, dout):
        x = self.cache
        return dout * np.where(x>0,1,self.alpha)
    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class Tanh(Module):
    """Tanh 激活函数: f(x) = tanh(x)

    前向: out = tanh(x)
    反向: dx = dout * (1 - tanh²(x))

    提示:
        - 和 Sigmoid 一样，缓存输出而非输入
        - np.tanh 可以直接用
    """

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = np.tanh(x)
        return self.cache
    def backward(self, dout):
        x = self.cache
        return dout * (1- x**2)
    def __repr__(self):
        return "Tanh()"


class ELU(Module):
    """ELU 激活函数: f(x) = x if x > 0, else alpha * (exp(x) - 1)

    参数:
        alpha (float): 负半轴的饱和值，默认 1.0

    前向: out = x if x > 0, else alpha * (exp(x) - 1)
    反向: dx = dout * 1 if x > 0, else dout * (out + alpha)

    提示:
        - 需要同时缓存输入 x 和输出 out
        - 反向传播中，x <= 0 的部分梯度为 dout * (out + alpha)
          这可以从 d/dx [alpha*(exp(x)-1)] = alpha*exp(x) = out + alpha 推导
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.cache = None

    def forward(self, x):
        out = np.where(x>0,x, self.alpha * (np.exp(x)-1))
        self.cache = x, out
        return self.cache[1]
    def backward(self, dout):
        x = self.cache[0]
        out = self.cache[1]
        return dout * np.where(x>0,1,(out + self.alpha))
    def __repr__(self):
        return f"ELU(alpha={self.alpha})"
