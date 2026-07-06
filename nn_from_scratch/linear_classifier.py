"""线性分类器模块。

实现 SVM 和 Softmax 两种线性分类器，包含完整的训练循环。
这是 CS231n Assignment 1 的核心内容，也是你第一次完整写
"前向 → 算 loss → 反向求梯度 → 更新参数"的闭环。

实现顺序建议: SoftmaxClassifier -> SVMClassifier
（Softmax 你更熟悉，先实现它建立信心）
"""

import numpy as np
from .losses import softmax_cross_entropy_loss, svm_loss


class LinearClassifier:
    """线性分类器基类。

    scores = X @ W + b

    子类通过重写 _compute_loss_and_grad 来使用不同的损失函数。
    """

    def __init__(self, n_features, n_classes):
        """
        参数:
            n_features: 输入特征维度 D
            n_classes:  类别数 C
        """
        self.W = None  # shape (D, C)
        self.b = None  # shape (C,)
        self.n_features = n_features
        self.n_classes = n_classes
        self._init_weights()

    def _init_weights(self):
        """用小随机数初始化权重。

        提示:
            - W 用 np.random.randn * 0.01
            - b 用 np.zeros
        """
        # TODO
        raise NotImplementedError

    def _compute_loss_and_grad(self, scores, y):
        """子类重写此方法。返回 (loss, dscores)。"""
        raise NotImplementedError

    def fit(self, X, y, lr=1e-3, reg=1e-5, n_iters=1000, batch_size=256, verbose=False):
        """用 mini-batch SGD 训练分类器。

        参数:
            X:          shape (N, D)，训练数据
            y:          shape (N,)，训练标签
            lr:         学习率
            reg:        L2 正则化强度
            n_iters:    迭代次数
            batch_size: mini-batch 大小
            verbose:    是否每 100 次打印 loss

        返回:
            loss_history: list，每次迭代的 loss

        提示:
            1. 每次迭代随机采样一个 mini-batch
            2. 计算 scores = X_batch @ W + b
            3. 调用 _compute_loss_and_grad 得到 loss 和 dscores
            4. 从 dscores 推出 dW 和 db
               dW = X_batch.T @ dscores + reg * W  (加 L2 正则)
               db = np.sum(dscores, axis=0)
            5. 更新 W -= lr * dW, b -= lr * db
            6. loss 也要加上正则项: loss += 0.5 * reg * np.sum(W**2)
        """
        # TODO
        raise NotImplementedError

    def predict(self, X):
        """预测类别。

        返回:
            predictions: shape (N,)，每个样本的预测类别
        """
        # TODO
        raise NotImplementedError

    def score(self, X, y):
        """返回准确率。"""
        return np.mean(self.predict(X) == y)


class SoftmaxClassifier(LinearClassifier):
    """使用 Softmax + 交叉熵损失的线性分类器。"""

    def _compute_loss_and_grad(self, scores, y):
        # TODO: 调用 softmax_cross_entropy_loss
        raise NotImplementedError


class SVMClassifier(LinearClassifier):
    """使用 SVM / Hinge 损失的线性分类器。"""

    def _compute_loss_and_grad(self, scores, y):
        # TODO: 调用 svm_loss
        raise NotImplementedError
