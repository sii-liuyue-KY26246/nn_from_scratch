"""损失函数模块。

每个损失函数是一个普通函数（不是类），接收预测值和标签，返回 (loss, grad)。
这种函数式设计是 CS231n 的风格，简洁且足够用。

实现顺序建议: mse_loss -> binary_cross_entropy_loss -> softmax_cross_entropy_loss -> svm_loss
"""

import numpy as np


def mse_loss(predictions, targets):
    """均方误差损失（Mean Squared Error）。

    用于回归任务。

    L = (1/N) * Σ (predictions - targets)²

    参数:
        predictions: shape (N, D) 或 (N,)，模型输出
        targets:     shape 同 predictions，真实值

    返回:
        loss (float): 标量损失值
        grad (ndarray): shape 同 predictions，dL/d(predictions)

    提示:
        - grad = (2/N) * (predictions - targets)
        - 如果输入是 (N,)，先想清楚 N 是什么
    """
    # TODO
    raise NotImplementedError


def binary_cross_entropy_loss(logits, targets):
    """二分类交叉熵损失。

    先对 logits 做 sigmoid 得到概率 p，再算交叉熵。

    L = -(1/N) * Σ [y * log(p) + (1-y) * log(1-p)]

    参数:
        logits:  shape (N,) 或 (N, 1)，未经 sigmoid 的原始输出
        targets: shape 同 logits，0 或 1 的标签

    返回:
        loss (float): 标量损失值
        grad (ndarray): shape 同 logits，dL/d(logits)

    提示:
        - 先算 sigmoid: p = 1 / (1 + exp(-logits))
        - 数值稳定性：用 np.clip(p, 1e-12, 1 - 1e-12) 防止 log(0)
        - 梯度推导的最终结果很简洁: grad = (1/N) * (p - targets)
    """
    # TODO
    raise NotImplementedError


def softmax_cross_entropy_loss(logits, targets):
    """Softmax + 交叉熵损失（多分类）。

    这是最常用的分类损失。先对 logits 做 softmax 得到概率分布，再算交叉熵。

    参数:
        logits:  shape (N, C)，N 个样本，C 个类别的原始分数
        targets: shape (N,)，每个样本的正确类别索引（整数，0 到 C-1）

    返回:
        loss (float): 标量损失值
        grad (ndarray): shape (N, C)，dL/d(logits)

    提示:
        - 数值稳定的 softmax：先减去每行最大值 logits - max(logits, axis=1, keepdims=True)
        - probs = exp(shifted) / sum(exp(shifted), axis=1, keepdims=True)
        - loss = -(1/N) * Σ log(probs[i, targets[i]])
        - 梯度非常简洁：grad = probs 的副本，然后 grad[i, targets[i]] -= 1，最后除以 N
    """
    # TODO
    raise NotImplementedError


def svm_loss(scores, targets):
    """多类 SVM / Hinge 损失。

    L_i = Σ_{j ≠ y_i} max(0, s_j - s_{y_i} + margin)

    参数:
        scores:  shape (N, C)，N 个样本，C 个类别的分数
        targets: shape (N,)，每个样本的正确类别索引

    返回:
        loss (float): 标量损失值
        grad (ndarray): shape (N, C)，dL/d(scores)

    提示:
        - margin 默认为 1.0
        - 梯度的计算：
          对于 j ≠ y_i 且 margin 被违反的项：grad[i, j] += 1
          对于正确类别：grad[i, y_i] -= (该样本违反 margin 的次数)
        - 最后除以 N
    """
    # TODO
    raise NotImplementedError
