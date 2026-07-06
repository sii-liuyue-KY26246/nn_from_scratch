"""数值梯度检验工具。

这是你调试反向传播的核心武器。原理：用有限差分法近似梯度，
和你解析推导的梯度做对比。如果相对误差 < 1e-5，说明你的解析梯度是对的。

CS231n 反复强调：每次写完一个新的 backward，第一件事就是跑梯度检验。

实现顺序建议: eval_numerical_gradient -> eval_numerical_gradient_array -> gradient_check
"""

import numpy as np


def eval_numerical_gradient(f, x, h=1e-5):
    """计算标量函数 f 在 x 处的数值梯度。

    对 x 的每个元素分别做有限差分：
        df/dx_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2*h)

    参数:
        f: 函数，接收一个 ndarray，返回一个标量
        x: ndarray，求梯度的位置
        h: float，差分步长

    返回:
        grad: ndarray，shape 同 x，数值梯度

    提示:
        - 用 np.nditer 遍历 x 的每个元素（或用 flat 索引）
        - 注意要修改 x 的值、算 f、再改回来，不要创建副本（效率）
        - 也可以用 np.copy 先保存原值
    """
    # TODO
    raise NotImplementedError


def eval_numerical_gradient_array(f, x, dout, h=1e-5):
    """计算函数 f 在 x 处的数值梯度，其中 f 返回一个数组。

    利用链式法则：如果 out = f(x)，上游梯度为 dout，
    那么 df/dx_i = Σ_j dout_j * d(out_j)/d(x_i)

    实现方式：定义 g(x) = sum(f(x) * dout)，然后调用 eval_numerical_gradient(g, x)。

    参数:
        f:    函数，接收 ndarray，返回 ndarray
        x:    ndarray，求梯度的位置
        dout: ndarray，上游梯度，shape 同 f(x) 的输出
        h:    float，差分步长

    返回:
        grad: ndarray，shape 同 x

    提示:
        - 定义 scalar_f = lambda x: np.sum(f(x) * dout)
        - 然后调用 eval_numerical_gradient(scalar_f, x, h)
    """
    # TODO
    raise NotImplementedError


def gradient_check(f, x, analytic_grad, h=1e-5, threshold=1e-5, verbose=True):
    """对比数值梯度和解析梯度，报告相对误差。

    相对误差 = |num - ana| / max(|num|, |ana|, 1e-8)

    参数:
        f:              函数，接收 ndarray 返回标量
        x:              ndarray，求梯度的位置
        analytic_grad:  ndarray，你手算/代码算的解析梯度
        h:              float，差分步长
        threshold:      float，相对误差阈值
        verbose:        bool，是否打印详细信息

    返回:
        passed (bool): 是否通过检验（最大相对误差 < threshold）

    提示:
        - 先调用 eval_numerical_gradient 得到数值梯度
        - 逐元素算相对误差，取最大值
        - 分母加 1e-8 防止除零
    """
    # TODO
    raise NotImplementedError
