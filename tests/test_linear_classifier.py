"""线性分类器测试。

运行: pytest tests/test_linear_classifier.py -v
"""
import numpy as np
from nn_from_scratch.linear_classifier import SoftmaxClassifier, SVMClassifier


class TestSoftmaxClassifier:
    def test_overfit_small_data(self):
        """在很小的数据集上应该能过拟合到 100% 准确率"""
        np.random.seed(42)
        N, D, C = 20, 10, 3
        X = np.random.randn(N, D)
        y = np.random.randint(0, C, N)
        clf = SoftmaxClassifier(D, C)
        clf.fit(X, y, lr=1.0, reg=0.0, n_iters=500, batch_size=N)
        assert clf.score(X, y) >= 0.95

    def test_loss_decreases(self):
        np.random.seed(0)
        N, D, C = 50, 5, 3
        X = np.random.randn(N, D)
        y = np.random.randint(0, C, N)
        clf = SoftmaxClassifier(D, C)
        history = clf.fit(X, y, lr=0.5, reg=0.0, n_iters=200, batch_size=N)
        assert history[-1] < history[0]

    def test_predict_shape(self):
        clf = SoftmaxClassifier(5, 3)
        X = np.random.randn(10, 5)
        preds = clf.predict(X)
        assert preds.shape == (10,)
        assert all(0 <= p < 3 for p in preds)


class TestSVMClassifier:
    def test_overfit_small_data(self):
        np.random.seed(42)
        N, D, C = 20, 10, 3
        X = np.random.randn(N, D)
        y = np.random.randint(0, C, N)
        clf = SVMClassifier(D, C)
        clf.fit(X, y, lr=0.5, reg=0.0, n_iters=500, batch_size=N)
        assert clf.score(X, y) >= 0.90

    def test_loss_decreases(self):
        np.random.seed(1)
        N, D, C = 50, 5, 3
        X = np.random.randn(N, D)
        y = np.random.randint(0, C, N)
        clf = SVMClassifier(D, C)
        history = clf.fit(X, y, lr=0.1, reg=0.0, n_iters=200, batch_size=N)
        assert history[-1] < history[0]
