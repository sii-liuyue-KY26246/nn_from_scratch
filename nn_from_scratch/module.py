"""nn_from_scratch 的基类模块。

所有层（激活函数、全连接层等）都继承自 Module。
Module 提供 __call__ -> forward 的快捷调用，以及 train/eval 模式切换。
"""

import numpy as np


class Module:
    """所有层和网络组件的基类。

    子类必须实现 forward() 方法，如果需要反向传播则同时实现 backward() 方法。

    用法:
        layer = SomeLayer(...)
        out = layer(x)          # 等价于 layer.forward(x)
        dx = layer.backward(dout)
    """

    def __init__(self):
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现 forward 方法")

    def backward(self, dout):
        raise NotImplementedError("子类必须实现 backward 方法")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """返回 (param_array, grad_array) 的列表。无参数的层返回空列表。"""
        return []

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __repr__(self):
        return f"{self.__class__.__name__}()"
