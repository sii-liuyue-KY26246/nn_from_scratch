"""损失函数测试。

运行: pytest tests/test_losses.py -v
"""
import numpy as np
from nn_from_scratch.losses import (
    mse_loss, binary_cross_entropy_loss, softmax_cross_entropy_loss, svm_loss
)
from nn_from_scratch.gradient_check import eval_numerical_gradient


class TestMSELoss:
    def test_zero_loss(self):
        x = np.array([1.0, 2.0, 3.0])
        loss, grad = mse_loss(x, x)
        assert abs(loss) < 1e-10
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 4.0])
        loss, grad = mse_loss(pred, target)
        # loss = (0 + 0 + 1) / 3 = 1/3
        np.testing.assert_allclose(loss, 1.0 / 3.0, atol=1e-7)

    def test_gradient_check(self):
        np.random.seed(0)
        pred = np.random.randn(5, 3)
        target = np.random.randn(5, 3)
        _, analytic = mse_loss(pred, target)
        f = lambda p: mse_loss(p, target)[0]
        numerical = eval_numerical_gradient(f, pred)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)


class TestBinaryCrossEntropyLoss:
    def test_gradient_check(self):
        np.random.seed(1)
        logits = np.random.randn(10)
        targets = np.random.randint(0, 2, size=10).astype(float)
        _, analytic = binary_cross_entropy_loss(logits, targets)
        f = lambda l: binary_cross_entropy_loss(l, targets)[0]
        numerical = eval_numerical_gradient(f, logits)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_perfect_prediction_low_loss(self):
        logits = np.array([10.0, -10.0, 10.0])  # 很确定的预测
        targets = np.array([1.0, 0.0, 1.0])
        loss, _ = binary_cross_entropy_loss(logits, targets)
        assert loss < 0.01


class TestSoftmaxCrossEntropyLoss:
    def test_uniform_scores(self):
        """所有类别分数相同时，loss 应约为 log(C)"""
        C = 10
        scores = np.zeros((5, C))
        targets = np.array([0, 1, 2, 3, 4])
        loss, _ = softmax_cross_entropy_loss(scores, targets)
        np.testing.assert_allclose(loss, np.log(C), atol=1e-5)

    def test_gradient_check(self):
        np.random.seed(2)
        scores = np.random.randn(8, 5)
        targets = np.random.randint(0, 5, size=8)
        _, analytic = softmax_cross_entropy_loss(scores, targets)
        f = lambda s: softmax_cross_entropy_loss(s, targets)[0]
        numerical = eval_numerical_gradient(f, scores)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_output_shape(self):
        scores = np.random.randn(4, 3)
        targets = np.array([0, 1, 2, 0])
        loss, grad = softmax_cross_entropy_loss(scores, targets)
        assert isinstance(loss, float) or loss.ndim == 0
        assert grad.shape == scores.shape


class TestSVMLoss:
    def test_gradient_check(self):
        np.random.seed(3)
        scores = np.random.randn(8, 5)
        targets = np.random.randint(0, 5, size=8)
        _, analytic = svm_loss(scores, targets)
        f = lambda s: svm_loss(s, targets)[0]
        numerical = eval_numerical_gradient(f, scores)
        np.testing.assert_allclose(analytic, numerical, atol=1e-4)

    def test_correct_class_high_score(self):
        """正确类别分数很高时，loss 应约为 0"""
        scores = np.zeros((3, 5))
        scores[0, 0] = 100.0
        scores[1, 1] = 100.0
        scores[2, 2] = 100.0
        targets = np.array([0, 1, 2])
        loss, _ = svm_loss(scores, targets)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)
