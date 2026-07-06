"""k-NN 测试。

运行: pytest tests/test_knn.py -v
"""
import numpy as np
from nn_from_scratch.knn import (
    compute_distances_two_loops, compute_distances_one_loop,
    compute_distances_vectorized, KNNClassifier
)


class TestDistances:
    def setup_method(self):
        np.random.seed(0)
        self.X_train = np.random.randn(20, 5)
        self.X_test = np.random.randn(8, 5)

    def test_two_loops_shape(self):
        d = compute_distances_two_loops(self.X_train, self.X_test)
        assert d.shape == (8, 20)

    def test_three_methods_agree(self):
        d1 = compute_distances_two_loops(self.X_train, self.X_test)
        d2 = compute_distances_one_loop(self.X_train, self.X_test)
        d3 = compute_distances_vectorized(self.X_train, self.X_test)
        np.testing.assert_allclose(d1, d2, atol=1e-10)
        np.testing.assert_allclose(d1, d3, atol=1e-10)

    def test_self_distance_is_zero(self):
        d = compute_distances_vectorized(self.X_train, self.X_train)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-10)

    def test_non_negative(self):
        d = compute_distances_vectorized(self.X_train, self.X_test)
        assert np.all(d >= 0)


class TestKNNClassifier:
    def test_perfect_classification(self):
        """用训练集自身预测，k=1 时应该 100% 准确"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 3, 50)
        knn = KNNClassifier(k=1)
        knn.fit(X, y)
        assert knn.score(X, y) == 1.0

    def test_predict_shape(self):
        X_train = np.random.randn(30, 4)
        y_train = np.random.randint(0, 2, 30)
        X_test = np.random.randn(10, 4)
        knn = KNNClassifier(k=3)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert preds.shape == (10,)
